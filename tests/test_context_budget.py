"""Tests for model-aware token budgets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from strix.config import load_settings
from strix.llm import context_budget


if TYPE_CHECKING:
    import pytest


def _clear_tokenizer_caches() -> None:
    context_budget._tokenizer_overrides.cache_clear()
    context_budget._tokenizer_repo_for.cache_clear()
    context_budget._load_custom_tokenizer.cache_clear()


def test_context_window_known_model() -> None:
    # gpt-4o is mapped by LiteLLM at 128k input tokens.
    assert context_budget.context_window("gpt-4o") == 128_000


def test_context_window_strips_provider_prefix() -> None:
    assert context_budget.context_window("openai/gpt-4o") == 128_000


def test_context_window_chatgpt_prefix_skips_provider_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_budget._model_info.cache_clear()
    calls: list[str] = []

    def _model_info(model: str) -> dict[str, int]:
        calls.append(model)
        return {"max_input_tokens": 1_050_000, "max_output_tokens": 128_000}

    monkeypatch.setattr("litellm.get_model_info", _model_info)
    try:
        assert context_budget.context_window("chatgpt/gpt-5.6-luna") == 1_050_000
        assert calls == ["gpt-5.6-luna"]
    finally:
        context_budget._model_info.cache_clear()


def test_context_window_unmapped_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    context_budget._model_info.cache_clear()

    def _raise(_model: str) -> dict[str, int]:
        raise ValueError("This model isn't mapped yet.")

    monkeypatch.setattr("litellm.get_model_info", _raise)
    expected = load_settings().context.fallback_context_tokens
    assert context_budget.context_window("totally-made-up-model") == expected
    context_budget._model_info.cache_clear()


def test_count_tokens_fallback_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**_kwargs: object) -> int:
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr("litellm.token_counter", _raise)
    # Falls back to UTF-8 byte length (upper bound on tokens).
    assert context_budget.count_tokens("weird-model", "x" * 400) == 400
    assert context_budget.count_tokens("weird-model", "😀" * 10) == 40


def test_count_tokens_empty_is_zero() -> None:
    assert context_budget.count_tokens("gpt-4o", "") == 0


def test_count_tokens_uses_custom_tokenizer_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_tokenizer_caches()
    sentinel_tokenizer = {"type": "huggingface_tokenizer", "tokenizer": object()}
    monkeypatch.setattr(
        context_budget,
        "load_settings",
        lambda: SimpleNamespace(context=SimpleNamespace(tokenizer_map="weird=some/repo")),
    )
    monkeypatch.setattr(
        "litellm.create_pretrained_tokenizer",
        lambda _repo_id: sentinel_tokenizer,
    )

    calls: list[dict[str, object]] = []

    def _token_counter(**kwargs: object) -> int:
        calls.append(kwargs)
        return 7

    monkeypatch.setattr("litellm.token_counter", _token_counter)
    try:
        assert context_budget.count_tokens("openai/weird-model", "hi") == 7
        assert calls == [
            {
                "model": "weird-model",
                "custom_tokenizer": sentinel_tokenizer,
                "text": "hi",
            }
        ]
    finally:
        _clear_tokenizer_caches()


def test_count_tokens_falls_back_to_byte_length_when_tokenizer_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_tokenizer_caches()
    monkeypatch.setattr(
        context_budget,
        "load_settings",
        lambda: SimpleNamespace(context=SimpleNamespace(tokenizer_map="weird=some/repo")),
    )

    def _raise(_repo_id: str) -> dict[str, object]:
        raise RuntimeError("network unreachable")

    monkeypatch.setattr("litellm.create_pretrained_tokenizer", _raise)

    def _unexpected_call(**_kwargs: object) -> int:
        raise AssertionError("token_counter should not be called when tokenizer load fails")

    monkeypatch.setattr("litellm.token_counter", _unexpected_call)
    try:
        assert context_budget.count_tokens("openai/weird-model", "x" * 400) == 400
    finally:
        _clear_tokenizer_caches()


def test_count_tokens_no_override_leaves_default_behavior_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_tokenizer_caches()
    monkeypatch.setattr(
        context_budget,
        "load_settings",
        lambda: SimpleNamespace(context=SimpleNamespace(tokenizer_map="")),
    )

    def _unexpected_call(_repo_id: str) -> dict[str, object]:
        raise AssertionError("create_pretrained_tokenizer should not be called")

    monkeypatch.setattr("litellm.create_pretrained_tokenizer", _unexpected_call)
    calls: list[dict[str, object]] = []

    def _token_counter(**kwargs: object) -> int:
        calls.append(kwargs)
        return 3

    monkeypatch.setattr("litellm.token_counter", _token_counter)
    try:
        assert context_budget.count_tokens("gpt-4o", "hi") == 3
        assert calls == [{"model": "gpt-4o", "custom_tokenizer": None, "text": "hi"}]
    finally:
        _clear_tokenizer_caches()


def test_tokenizer_repo_for_prefers_longest_match(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_tokenizer_caches()
    monkeypatch.setattr(
        context_budget,
        "load_settings",
        lambda: SimpleNamespace(
            context=SimpleNamespace(tokenizer_map="qwen=Org/generic-qwen,qwen3=Org/qwen3-specific")
        ),
    )
    try:
        assert context_budget._tokenizer_repo_for("openai/qwen3-27b") == "Org/qwen3-specific"
        assert context_budget._tokenizer_repo_for("openai/qwen2-7b") == "Org/generic-qwen"
    finally:
        _clear_tokenizer_caches()


def test_tokenizer_overrides_ignores_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_tokenizer_caches()
    monkeypatch.setattr(
        context_budget,
        "load_settings",
        lambda: SimpleNamespace(
            context=SimpleNamespace(tokenizer_map="not-a-valid-entry,qwen=Org/qwen-tokenizer")
        ),
    )
    try:
        assert context_budget._tokenizer_overrides() == {"qwen": "Org/qwen-tokenizer"}
    finally:
        _clear_tokenizer_caches()


def _patch_context(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    context = {"turn_max_output_tokens": None, "fallback_context_tokens": 200_000, **overrides}
    monkeypatch.setattr(
        context_budget, "load_settings", lambda: SimpleNamespace(context=SimpleNamespace(**context))
    )


def test_turn_output_tokens_known_model_uses_litellm_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_context(monkeypatch)
    # gpt-4o is mapped by LiteLLM at 16,384 output tokens.
    assert context_budget.turn_output_tokens("gpt-4o") == 16_384


def test_turn_output_tokens_unmapped_model_uses_reasoning_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_context(monkeypatch)
    assert (
        context_budget.turn_output_tokens("openai/totally-unknown-model-xyz")
        == context_budget._DEFAULT_TURN_OUTPUT_TOKENS
        == 32_768
    )


def test_turn_output_tokens_env_override_applies_to_all_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_context(monkeypatch, turn_max_output_tokens=12_000)
    assert context_budget.turn_output_tokens("gpt-4o") == 12_000
    assert context_budget.turn_output_tokens("openai/totally-unknown-model-xyz") == 12_000


def test_turn_output_tokens_clamped_to_half_the_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unmapped model on a 32k-context server must not request its whole window.
    _patch_context(monkeypatch, fallback_context_tokens=32_768)
    assert context_budget.turn_output_tokens("openai/totally-unknown-model-xyz") == 16_384


def test_count_tokens_cached_memoizes_by_content(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guard re-measures the whole history on every model call, so repeat
    # content must not be re-tokenized.
    calls: list[str] = []

    def _count(_model: str, text: str) -> int:
        calls.append(text)
        return len(text)

    monkeypatch.setattr(context_budget, "count_tokens", _count)
    context_budget._TOKEN_MEMO.clear()

    assert context_budget.count_tokens_cached("m", "hello") == 5
    assert context_budget.count_tokens_cached("m", "hello") == 5
    assert calls == ["hello"]

    assert context_budget.count_tokens_cached("m", "other") == 5
    assert calls == ["hello", "other"]
    # A different model must not read another model's count.
    assert context_budget.count_tokens_cached("n", "hello") == 5
    assert calls == ["hello", "other", "hello"]

    context_budget._TOKEN_MEMO.clear()


def test_count_tokens_cached_evicts_oldest_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_budget, "count_tokens", lambda _m, t: len(t))
    monkeypatch.setattr(context_budget, "_TOKEN_MEMO_MAXSIZE", 4)
    context_budget._TOKEN_MEMO.clear()

    for i in range(10):
        context_budget.count_tokens_cached("m", f"text-{i}")

    assert len(context_budget._TOKEN_MEMO) <= 4
    context_budget._TOKEN_MEMO.clear()


def test_count_tokens_cached_empty_is_zero() -> None:
    assert context_budget.count_tokens_cached("gpt-4o", "") == 0
