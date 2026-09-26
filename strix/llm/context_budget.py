"""Model-aware token budgets, resolved from LiteLLM model metadata with a
large configurable fallback for models LiteLLM doesn't map.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from strix.config import load_settings


logger = logging.getLogger(__name__)

# LiteLLM keys models without the routing prefix users type (``openai/``,
# ``litellm/``, ``ollama/`` ...). Strip a leading provider segment on lookup.
_STRIPPABLE_PREFIXES = (
    "openai/",
    "chatgpt/",
    "litellm/",
    "any-llm/",
    "ollama/",
    "ollama_chat/",
)

_DEFAULT_OUTPUT_TOKENS = 8_192
# Per-turn cap for unmapped models. Sized for reasoning models whose hidden
# thinking shares the output budget with the answer: Qwen3's model card
# recommends 32k output tokens, leaving room for a long trace plus a full
# tool-call payload.
_DEFAULT_TURN_OUTPUT_TOKENS = 32_768


def _lookup_key(model: str) -> str:
    for prefix in _STRIPPABLE_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _safe_get_model_info(model: str) -> dict[str, Any] | None:
    try:
        import litellm

        return dict(litellm.get_model_info(model))
    except Exception:  # noqa: BLE001 - unmapped models raise; caller falls back.
        return None


@lru_cache(maxsize=128)
def _model_info(model: str) -> dict[str, int]:
    lookup_key = _lookup_key(model)
    # Provider-qualified ChatGPT lookups may start a synchronous device-login
    # poll. LiteLLM keys the metadata by the underlying model slug.
    candidates = (lookup_key,) if model.startswith("chatgpt/") else (model, lookup_key)
    for candidate in candidates:
        info = _safe_get_model_info(candidate)
        if info is not None:
            return {
                "max_input_tokens": int(
                    info.get("max_input_tokens") or info.get("max_tokens") or 0
                ),
                "max_output_tokens": int(info.get("max_output_tokens") or 0),
            }
    logger.debug("No LiteLLM model info for %r; using configured fallbacks", model)
    return {"max_input_tokens": 0, "max_output_tokens": 0}


@lru_cache(maxsize=1)
def _tokenizer_overrides() -> dict[str, str]:
    """Parsed STRIX_CONTEXT_TOKENIZER_MAP: {lowercased substring: HF repo id}."""
    raw = load_settings().context.tokenizer_map
    overrides: dict[str, str] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        key, sep, repo_id = entry.partition("=")
        if not sep or not key.strip() or not repo_id.strip():
            logger.warning("Ignoring malformed STRIX_CONTEXT_TOKENIZER_MAP entry: %r", entry)
            continue
        overrides[key.strip().lower()] = repo_id.strip()
    return overrides


@lru_cache(maxsize=128)
def _tokenizer_repo_for(model: str) -> str | None:
    """HF tokenizer repo id configured for ``model``, or ``None`` if unconfigured."""
    overrides = _tokenizer_overrides()
    lookup_key = _lookup_key(model).lower()
    matches = [key for key in overrides if key in lookup_key]
    if not matches:
        return None
    return overrides[max(matches, key=len)]


@lru_cache(maxsize=8)
def _load_custom_tokenizer(repo_id: str) -> dict[str, Any] | None:
    """Load and cache an HF tokenizer for LiteLLM's ``custom_tokenizer`` param.

    Returns ``None`` (cached) on any failure so a bad/unreachable repo id is
    retried at most once per process, not on every ``count_tokens`` call.
    """
    try:
        return dict(litellm.create_pretrained_tokenizer(repo_id))
    except Exception:  # noqa: BLE001 - network/repo issues must not crash the scan.
        logger.warning("Failed to load tokenizer %r; falling back to byte-length", repo_id)
        return None


def context_window(model: str) -> int:
    """Input token capacity for ``model`` (configured fallback when unmapped)."""
    resolved = _model_info(model)["max_input_tokens"]
    return resolved or load_settings().context.fallback_context_tokens


def output_limit(model: str) -> int:
    """Max output tokens for ``model`` (a conservative default when unmapped)."""
    return _model_info(model)["max_output_tokens"] or _DEFAULT_OUTPUT_TOKENS


def turn_output_tokens(model: str) -> int:
    """Per-turn generation cap (``max_tokens``) for ``model``.

    ``STRIX_TURN_MAX_OUTPUT_TOKENS`` when set, else LiteLLM's output limit, else
    a reasoning-sized default. Clamped to half the context window so history
    plus the requested output always fit, which servers like SGLang enforce.
    """
    configured = load_settings().context.turn_max_output_tokens
    resolved = configured or _model_info(model)["max_output_tokens"] or _DEFAULT_TURN_OUTPUT_TOKENS
    return max(1, min(resolved, context_window(model) // 2))


def count_tokens(model: str, text: str) -> int:
    """Token count for ``text`` under ``model``.

    Uses a configured HuggingFace tokenizer (``STRIX_CONTEXT_TOKENIZER_MAP``)
    when ``model`` matches one, for accuracy on models LiteLLM doesn't
    recognize (it otherwise silently approximates with cl100k_base). Falls
    back to UTF-8 byte length (a guaranteed upper bound) when LiteLLM can't
    count, or when a configured tokenizer fails to load, so budget checks
    stay conservative rather than reverting to a known-inaccurate estimate.
    """
    if not text:
        return 0
    repo_id = _tokenizer_repo_for(model)
    if repo_id is not None:
        tokenizer = _load_custom_tokenizer(repo_id)
        if tokenizer is None:
            return len(text.encode("utf-8"))
        try:
            return int(
                litellm.token_counter(
                    model=_lookup_key(model), custom_tokenizer=tokenizer, text=text
                )
            )
        except Exception:  # noqa: BLE001 - tokenizer may reject this text.
            return len(text.encode("utf-8"))
    try:
        import litellm

        return int(litellm.token_counter(model=_lookup_key(model), text=text))
    except Exception:  # noqa: BLE001 - tokenizer may be unavailable for some models.
        return len(text.encode("utf-8"))
