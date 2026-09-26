"""Tests for provider-agnostic conversation compaction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from litellm.exceptions import BadRequestError, ContextWindowExceededError, RateLimitError
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from strix.config import ContextSettings
from strix.llm import compaction


if TYPE_CHECKING:
    from agents.memory.session_settings import SessionSettings


class FakeSession:
    """Minimal in-memory Session for exercising compaction."""

    session_id = "fake"
    session_settings: SessionSettings | None = None

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    async def get_items(self, limit: int | None = None) -> list[Any]:
        return list(self._items) if limit is None else list(self._items[-limit:])

    async def add_items(self, items: list[Any]) -> None:
        self._items.extend(items)

    async def clear_session(self) -> None:
        self._items = []

    async def pop_item(self) -> Any:
        return self._items.pop() if self._items else None


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _call(call_id: str, name: str = "exec_command") -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _output(call_id: str, text: str = "done") -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def _turns(n: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i in range(n):
        items += [
            _user(f"task {i}"),
            _call(f"c{i}"),
            _output(f"c{i}", f"result {i}"),
            _assistant(f"ok {i}"),
        ]
    return items


def _has_orphan_tool_output(items: list[Any]) -> bool:
    call_ids = {i["call_id"] for i in items if compaction._is_tool_call(i)}
    return any(i["call_id"] not in call_ids for i in items if compaction._is_tool_output(i))


def test_is_context_overflow_uses_litellm_typed_error() -> None:
    overflow = ContextWindowExceededError(
        message="context length exceeded", model="m", llm_provider="openai"
    )
    assert compaction.is_context_overflow(overflow)
    assert not compaction.is_context_overflow(
        RateLimitError(message="slow down", model="m", llm_provider="openai")
    )
    assert not compaction.is_context_overflow(RuntimeError("maximum context length is 8192"))


def test_is_context_overflow_matches_untyped_openrouter_400() -> None:
    # OpenRouter overflows arrive as a plain BadRequestError, so match the message.
    openrouter = BadRequestError(
        message=(
            "litellm.BadRequestError: This endpoint's maximum context length is 16385 "
            "tokens. However, you requested about 75064 tokens. Please reduce the length "
            "of the messages."
        ),
        model="openrouter/openai/gpt-3.5-turbo",
        llm_provider="openrouter",
    )
    assert compaction.is_context_overflow(openrouter)


def test_is_context_overflow_ignores_rate_limit_shaped_bad_request() -> None:
    # A 400 that is really throttling must never be treated as an overflow.
    throttled = BadRequestError(
        message="Rate limit exceeded, please slow down",
        model="openrouter/openai/gpt-4o",
        llm_provider="openrouter",
    )
    assert not compaction.is_context_overflow(throttled)
    unrelated = BadRequestError(
        message="Invalid value for 'temperature'", model="m", llm_provider="openrouter"
    )
    assert not compaction.is_context_overflow(unrelated)


def test_select_split_never_orphans_tool_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compaction, "count_tokens", lambda _m, t: len(t))
    items = _turns(10)
    split = compaction._select_split("m", items, keep_tokens=25)
    recent = items[split:]
    assert recent  # something is kept
    assert not _has_orphan_tool_output(recent)


def test_select_split_handles_parallel_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compaction, "count_tokens", lambda _m, _t: 1)
    # Two parallel calls then their two outputs.
    items = [
        _user("start"),
        _call("a"),
        _call("b"),
        _output("a"),
        _output("b"),
        _assistant("done"),
    ]
    # keep_tokens picks a boundary that would land between the calls/outputs.
    split = compaction._select_split("m", items, keep_tokens=2)
    assert not _has_orphan_tool_output(items[split:])


def _patch_budget(monkeypatch: pytest.MonkeyPatch, *, keep_tokens: int, window: int) -> None:
    monkeypatch.setattr(compaction, "count_tokens", lambda _m, t: len(t))
    monkeypatch.setattr(compaction, "count_tokens_cached", lambda _m, t: len(t))
    monkeypatch.setattr(compaction, "context_window", lambda _m: window)
    monkeypatch.setattr(compaction, "output_limit", lambda _m: 0)
    monkeypatch.setattr(compaction, "turn_output_tokens", lambda _m: 0)
    context = ContextSettings()
    context.keep_tokens = keep_tokens
    context.compact_buffer_tokens = 0
    context.summary_max_tokens = 64
    context.auto_compact = True
    settings = SimpleNamespace(
        context=context,
        llm=SimpleNamespace(api_key=None, api_base=None, timeout=1, extra_headers=None),
    )
    monkeypatch.setattr(compaction, "load_settings", lambda: settings)


def _model_response(text: str) -> Any:
    chunk = ResponseOutputText(annotations=[], text=text, type="output_text")
    message = ResponseOutputMessage(
        id="msg", content=[chunk], role="assistant", status="completed", type="message"
    )
    return SimpleNamespace(output=[message])


def _patch_summary(
    monkeypatch: pytest.MonkeyPatch, text: str, captured: dict[str, Any] | None = None
) -> None:
    class FakeModel:
        async def get_response(self, **kwargs: Any) -> Any:
            if captured is not None:
                captured.update(kwargs)
            return _model_response(text)

    class FakeProvider:
        def get_model(self, model_name: str | None) -> Any:
            if captured is not None:
                captured["model"] = model_name
            return FakeModel()

    monkeypatch.setattr(compaction, "StrixProvider", FakeProvider)


@pytest.mark.asyncio
async def test_maybe_compact_noop_when_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_budget(monkeypatch, keep_tokens=50, window=1_000_000)
    session = FakeSession(_turns(10))
    before = await session.get_items()

    assert await compaction.maybe_compact(session, model="m") is False
    assert await session.get_items() == before


@pytest.mark.asyncio
async def test_maybe_compact_reserves_turn_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    # History that fits the window still compacts once it leaves no room for the
    # output each turn requests, or the provider would reject prompt + max_tokens.
    _patch_budget(monkeypatch, keep_tokens=30, window=1_000_000)
    monkeypatch.setattr(compaction, "turn_output_tokens", lambda _m: 1_000_000 - 50)
    _patch_summary(monkeypatch, "SUMMARY BODY")
    session = FakeSession(_turns(12))

    assert await compaction.maybe_compact(session, model="m") is True


@pytest.mark.asyncio
async def test_maybe_compact_rewrites_and_keeps_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_budget(monkeypatch, keep_tokens=30, window=4_000)
    _patch_summary(monkeypatch, "SUMMARY BODY")
    session = FakeSession(_turns(12))

    assert await compaction.maybe_compact(session, model="m", force=True) is True

    items = await session.get_items()
    assert items[0]["role"] == "user"
    assert items[0]["content"].startswith(compaction._CHECKPOINT_TAG)
    assert "SUMMARY BODY" in items[0]["content"]
    assert len(items) < len(_turns(12))
    assert not _has_orphan_tool_output(items)


@pytest.mark.asyncio
async def test_maybe_compact_updates_previous_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    # Window large enough to leave real room for the summary instructions.
    _patch_budget(monkeypatch, keep_tokens=30, window=4_000)
    captured: dict[str, Any] = {}
    _patch_summary(monkeypatch, "NEW", captured)

    prior = compaction._checkpoint_item("OLD SUMMARY TEXT")
    session = FakeSession([prior, *_turns(12)])

    assert await compaction.maybe_compact(session, model="m", force=True) is True
    assert "OLD SUMMARY TEXT" in captured["input"]


@pytest.mark.asyncio
async def test_summarize_routes_through_provider_with_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_budget(monkeypatch, keep_tokens=30, window=4_000)
    monkeypatch.setattr(
        compaction,
        "load_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(
                api_key=None, api_base=None, timeout=1, extra_headers={"X-Feature-Key": "svc"}
            )
        ),
    )
    captured: dict[str, Any] = {}
    _patch_summary(monkeypatch, "S", captured)

    assert await compaction._summarize("litellm/openai/some-model", "p", 64) == "S"
    assert captured["model"] == "litellm/openai/some-model"
    settings = captured["model_settings"]
    assert settings.extra_headers == {"X-Feature-Key": "svc"}
    assert settings.max_tokens == 64


def test_fit_to_tokens_truncates_oversized_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compaction, "count_tokens", lambda _m, t: len(t))
    text = "x" * 10_000

    fitted = compaction._fit_to_tokens("m", text, 500)

    assert len(fitted) <= 500
    assert compaction._HEAD_TRUNCATED_MARKER in fitted
    # Small text is returned untouched.
    assert compaction._fit_to_tokens("m", "short", 500) == "short"


def test_summary_output_tokens_capped_at_model_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    context = ContextSettings()
    monkeypatch.setattr(compaction, "load_settings", lambda: SimpleNamespace(context=context))

    monkeypatch.setattr(compaction, "output_limit", lambda _m: 1_000)
    context.summary_max_tokens = 4_096
    # Configured allowance above the model cap is clamped down to the cap.
    assert compaction._summary_output_tokens("m") == 1_000
    # Below the cap, the configured value is used unchanged.
    context.summary_max_tokens = 500
    assert compaction._summary_output_tokens("m") == 500


@pytest.mark.asyncio
async def test_maybe_compact_bounds_summary_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # A tiny window with a huge head must not send an oversized summary request.
    _patch_budget(monkeypatch, keep_tokens=30, window=4_000)
    captured: dict[str, Any] = {}
    _patch_summary(monkeypatch, "S", captured)
    big_turns = [{"role": "user", "content": "y" * 2_000} for _ in range(50)]
    session = FakeSession(big_turns)

    assert await compaction.maybe_compact(session, model="m") is True
    # count_tokens==len(chars); prompt must fit the model window.
    assert len(captured["input"]) <= 4_000


@pytest.mark.asyncio
async def test_summary_request_fits_when_room_is_below_old_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Head-input budget must shrink to the real room so the request fits.
    instructions = len(compaction._SUMMARY_INSTRUCTIONS)
    window = instructions + 64 + 256 + 300  # summary_max(64)+slack(256)+room(300)
    _patch_budget(monkeypatch, keep_tokens=30, window=window)
    captured: dict[str, Any] = {}
    _patch_summary(monkeypatch, "S", captured)
    session = FakeSession([{"role": "user", "content": "y" * 5_000} for _ in range(20)])

    assert await compaction.maybe_compact(session, model="m") is True
    assert len(captured["input"]) <= window


@pytest.mark.asyncio
async def test_maybe_compact_skips_when_summary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_budget(monkeypatch, keep_tokens=30, window=4_000)

    class BoomModel:
        async def get_response(self, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

    class BoomProvider:
        def get_model(self, _model_name: str | None) -> Any:
            return BoomModel()

    monkeypatch.setattr(compaction, "StrixProvider", BoomProvider)
    session = FakeSession(_turns(12))
    before = await session.get_items()

    assert await compaction.maybe_compact(session, model="m", force=True) is False
    assert await session.get_items() == before


@pytest.mark.asyncio
async def test_maybe_compact_skips_when_no_room_to_summarise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No room for any head -> no (doomed) summary is attempted.
    _patch_budget(monkeypatch, keep_tokens=30, window=200)
    captured: dict[str, Any] = {}
    _patch_summary(monkeypatch, "S", captured)
    session = FakeSession(_turns(12))
    before = await session.get_items()

    assert await compaction.maybe_compact(session, model="m", force=True) is False
    assert not captured
    assert await session.get_items() == before


def _patch_measure(monkeypatch: pytest.MonkeyPatch) -> None:
    """One char per token, so expected sizes are readable in the assertions."""
    monkeypatch.setattr(compaction, "count_tokens", lambda _m, t: len(t))
    monkeypatch.setattr(compaction, "count_tokens_cached", lambda _m, t: len(t))


def _patch_reserve(
    monkeypatch: pytest.MonkeyPatch,
    *,
    window: int,
    buffer: int,
    turn: int,
    keep: int = 8_000,
) -> None:
    monkeypatch.setattr(compaction, "context_window", lambda _m: window)
    monkeypatch.setattr(compaction, "turn_output_tokens", lambda _m: turn)
    context = ContextSettings()
    context.compact_buffer_tokens = buffer
    context.keep_tokens = keep
    monkeypatch.setattr(compaction, "load_settings", lambda: SimpleNamespace(context=context))


def _big_turns(n: int, size: int) -> list[dict[str, Any]]:
    """Turns whose tool output dwarfs everything else, as in a real scan."""
    items: list[dict[str, Any]] = []
    for i in range(n):
        items += [_call(f"c{i}"), _output(f"c{i}", "A" * size), _assistant(f"ok {i}")]
    return items


def test_reserve_tokens_adds_the_buffer_to_the_turn_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The deployment that overflowed: a 100k buffer must not be swallowed by
    # max() picking whichever single term happens to be larger.
    _patch_reserve(monkeypatch, window=262_144, buffer=100_000, turn=32_768)

    assert compaction.reserve_tokens("m") == 132_768
    assert compaction.budget_tokens("m") == 129_376


def test_reserve_tokens_needs_both_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neither term alone: the reserve must exceed each of them.
    _patch_reserve(monkeypatch, window=262_144, buffer=100_000, turn=32_768)
    reserve = compaction.reserve_tokens("m")

    assert reserve > 100_000
    assert reserve > 32_768


def test_reserve_tokens_clamps_the_buffer_on_a_small_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 100k buffer on a 32k model would otherwise reserve the whole window.
    _patch_reserve(monkeypatch, window=32_768, buffer=100_000, turn=16_384, keep=8_000)

    assert compaction.reserve_tokens("m") < 32_768
    assert compaction.budget_tokens("m") >= 8_000


def test_full_item_text_keeps_the_whole_tool_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # The summary serializer caps payloads at 2k chars. Measuring a request
    # with it is what let a 230k-token history read as comfortably in budget.
    _patch_measure(monkeypatch)
    item = _output("c0", "A" * 50_000)

    measured = compaction.measure_item_tokens("m", item)
    serialized = len(compaction._serialize_item(item))

    assert measured >= 50_000
    assert measured > serialized * 10


def test_full_item_text_counts_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reasoning round-trips on the wire but serializes to "" for the summary.
    _patch_measure(monkeypatch)
    item = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "deep thought"}]}

    assert compaction._serialize_item(item) == ""
    assert "deep thought" in compaction._full_item_text(item)
    assert compaction.measure_item_tokens("m", item) > 0


def test_full_item_text_never_measures_an_unknown_item_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_measure(monkeypatch)
    item = {"type": "local_shell_call", "action": {"command": "B" * 5_000}}

    assert compaction._serialize_item(item) == ""
    assert compaction.measure_item_tokens("m", item) > 5_000


@pytest.mark.asyncio
async def test_maybe_compact_sees_untruncated_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A history whose truncated rendering fits the window but whose real
    # payload does not: the exact shape of the reported overflow.
    _patch_budget(monkeypatch, keep_tokens=100, window=60_000)
    _patch_summary(monkeypatch, "SUMMARY BODY")
    items = [_user("opening"), *_big_turns(6, 20_000)]
    session = FakeSession(items)

    assert len(compaction._serialize_items(items)) < 60_000
    assert compaction.measure_items_tokens("m", items) > 60_000
    assert await compaction.maybe_compact(session, model="m") is True


def test_select_split_bounds_the_kept_tail_by_real_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measured through the truncating serializer, six 5k outputs read as ~12k
    # and all six would be kept; measured honestly, only the newest fit.
    _patch_measure(monkeypatch)
    items = [_user("opening"), *_big_turns(6, 5_000)]

    split = compaction._select_split("m", items, keep_tokens=12_000)

    assert compaction.measure_items_tokens("m", items[split:]) <= 12_000
    assert not _has_orphan_tool_output(items[split:])


def test_guard_split_snaps_forward_through_parallel_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _select_split snaps backwards, growing the tail -- fine when the head is
    # still summarised, but a hard cap must shrink instead.
    _patch_measure(monkeypatch)
    items = [
        _user("go"),
        _call("a"),
        _call("b"),
        _output("a", "A" * 2_000),
        _output("b", "B" * 2_000),
        _assistant("done"),
    ]
    budget = 2_500

    guard_tail = items[compaction._guard_split("m", items, budget) :]
    select_tail = items[compaction._select_split("m", items, budget) :]

    assert compaction.measure_items_tokens("m", guard_tail) <= budget
    assert not _has_orphan_tool_output(guard_tail)
    # The compaction split would have gone the other way and overshot.
    assert compaction.measure_items_tokens("m", select_tail) > budget


def test_guard_trim_drops_oldest_until_it_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_measure(monkeypatch)
    items = [_user("opening task"), *_big_turns(10, 1_000)]
    budget = 5_000

    trimmed = compaction.guard_trim_items("m", "", "", items, budget)

    assert compaction.measure_request_tokens("m", "", "", trimmed) <= budget
    assert not _has_orphan_tool_output(trimmed)
    assert trimmed[0] is items[0]
    assert compaction._ELISION_TAG in trimmed[1]["content"]
    assert trimmed[-1] is items[-1]


def test_guard_trim_preserves_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # The checkpoint holds every finding compacted away so far.
    _patch_measure(monkeypatch)
    checkpoint = compaction._checkpoint_item("EARLIER FINDINGS")
    items = [checkpoint, *_big_turns(10, 1_000)]

    trimmed = compaction.guard_trim_items("m", "", "", items, 4_000)

    assert trimmed[0] is checkpoint
    assert "EARLIER FINDINGS" in trimmed[0]["content"]
    assert compaction.measure_request_tokens("m", "", "", trimmed) <= 4_000


def test_guard_trim_never_rewrites_item_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    # These dicts may be the session's own, so only the list may change.
    _patch_measure(monkeypatch)
    items = [_user("opening"), *_big_turns(10, 1_000)]
    before = [dict(item) for item in items]

    trimmed = compaction.guard_trim_items("m", "", "", items, 5_000)

    assert items == before
    original_ids = {id(item) for item in items}
    added = [item for item in trimmed if id(item) not in original_ids]
    assert len(added) == 1
    assert compaction._ELISION_TAG in added[0]["content"]


def test_guard_trim_is_a_noop_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_measure(monkeypatch)
    items = [_user("opening"), *_big_turns(2, 100)]

    assert compaction.guard_trim_items("m", "", "", items, 100_000) == items


def test_guard_trim_counts_instructions_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    # The prepared system prompt and tool schemas share the same window.
    _patch_measure(monkeypatch)
    items = [_user("opening"), *_big_turns(10, 1_000)]
    budget = 6_000

    lean = compaction.guard_trim_items("m", "", "", items, budget)
    heavy = compaction.guard_trim_items("m", "S" * 3_000, "T" * 1_000, items, budget)

    assert len(heavy) < len(lean)
    assert compaction.measure_request_tokens("m", "S" * 3_000, "T" * 1_000, heavy) <= budget


@pytest.mark.asyncio
async def test_force_compacts_a_short_session_of_huge_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Overflow recovery must not be blocked by the item-count gate: five
    # enormous items are exactly the case that needs compacting.
    _patch_budget(monkeypatch, keep_tokens=100, window=10_000)
    _patch_summary(monkeypatch, "SUMMARY BODY")
    items = [
        _user("go"),
        _call("c0"),
        _output("c0", "A" * 9_000),
        _assistant("ok"),
        _user("next"),
    ]
    assert len(items) < compaction._MIN_ITEMS_TO_COMPACT
    session = FakeSession(items)

    assert await compaction.maybe_compact(session, model="m", force=True) is True
    assert await compaction.maybe_compact(FakeSession(list(items)), model="m") is False


@pytest.mark.asyncio
async def test_maybe_compact_trims_when_the_summary_leaves_it_over_budget(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Summarising the head does not guarantee the result fits.
    _patch_budget(monkeypatch, keep_tokens=5_000, window=4_000)
    _patch_summary(monkeypatch, "SUMMARY BODY")
    items = [_user("opening"), *_big_turns(8, 800)]
    session = FakeSession(items)

    with caplog.at_level("WARNING", logger="strix.llm.compaction"):
        assert await compaction.maybe_compact(session, model="m", force=True) is True

    final = await session.get_items()
    assert final[0]["content"].startswith(compaction._CHECKPOINT_TAG)
    assert "SUMMARY BODY" in final[0]["content"]
    assert compaction.measure_items_tokens("m", final) <= compaction.budget_tokens("m")
    assert not _has_orphan_tool_output(final)
    assert any("trimming oldest turns" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_maybe_compact_honours_the_observed_prompt_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The agent object carries only its own instructions; the SDK sends a much
    # larger prepared system prompt, which the hooks report back.
    _patch_budget(monkeypatch, keep_tokens=100, window=10_000)
    _patch_summary(monkeypatch, "SUMMARY BODY")
    items = [_user("opening"), *_big_turns(3, 2_000)]

    assert await compaction.maybe_compact(FakeSession(list(items)), model="m") is False
    assert (
        await compaction.maybe_compact(
            FakeSession(list(items)), model="m", min_overhead_tokens=6_000
        )
        is True
    )
