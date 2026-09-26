"""Tests for budget enforcement in ReportUsageHooks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strix.config import ContextSettings
from strix.core.hooks import (
    BudgetExceededError,
    BudgetPausedError,
    ReportUsageHooks,
    SubagentBudgetReservedError,
    recomputed_budget_flags,
)
from strix.llm import compaction


def _make_hooks(max_budget: float | None) -> ReportUsageHooks:
    return ReportUsageHooks(model="test-model", max_budget_usd=max_budget)


def _make_report_state(cost: float) -> MagicMock:
    state = MagicMock()
    state.get_total_llm_cost.return_value = cost
    state.record_sdk_usage = MagicMock()
    return state


def _make_context(agent_id: str = "test-agent", parent_id: str | None = None) -> MagicMock:
    ctx: MagicMock = MagicMock()
    ctx.context = {"agent_id": agent_id, "parent_id": parent_id}
    return ctx


def _make_warn_context(
    *,
    requests: int,
    parent_id: str | None = None,
    agent_id: str = "test-agent",
) -> MagicMock:
    ctx: MagicMock = MagicMock()
    ctx.context = {"agent_id": agent_id, "parent_id": parent_id}
    ctx.usage = MagicMock()
    ctx.usage.requests = requests
    return ctx


@pytest.mark.asyncio
async def test_no_budget_never_raises() -> None:
    hooks = _make_hooks(None)
    state = _make_report_state(9999.0)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(_make_context(), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_under_budget_does_not_raise() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(9.99)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(_make_context(), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_at_budget_raises() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(10.0)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetExceededError),
    ):
        await hooks.on_llm_end(_make_context(), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_over_budget_raises() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(10.01)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetExceededError),
    ):
        await hooks.on_llm_end(_make_context(), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_budget_check_uses_live_cost_accessor() -> None:
    # The check must read the live ledger, not the persisted run-record snapshot,
    # so it stays accurate even when a save fails after a usage record.
    hooks = _make_hooks(5.0)
    state = _make_report_state(6.0)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetExceededError),
    ):
        await hooks.on_llm_end(_make_context(), MagicMock(), MagicMock())
    state.get_total_llm_cost.assert_called_once()
    state.get_total_llm_usage.assert_not_called()


@pytest.mark.asyncio
async def test_error_message_includes_amounts() -> None:
    hooks = _make_hooks(5.0)
    state = _make_report_state(7.1234)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        with pytest.raises(BudgetExceededError, match=r"\$5\.00") as exc_info:
            await hooks.on_llm_end(_make_context(), MagicMock(), MagicMock())
        assert "7.1234" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subagent_stops_at_reserve() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(9.0)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(SubagentBudgetReservedError),
    ):
        await hooks.on_llm_end(_make_context(parent_id="root-1"), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_subagent_below_reserve_does_not_raise() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(8.99)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(_make_context(parent_id="root-1"), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_subagent_overshoot_to_full_budget_triggers_scan_wide_stop() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(10.5)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetExceededError),
    ):
        await hooks.on_llm_end(_make_context(parent_id="root-1"), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_root_keeps_running_inside_reserve() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(9.5)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(_make_context(parent_id=None), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_root_hard_stop_stays_at_full_budget() -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(10.0)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetExceededError),
    ):
        await hooks.on_llm_end(_make_context(parent_id=None), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_budget_warning_mentions_reserve() -> None:
    hooks = ReportUsageHooks(model="test-model", max_budget_usd=10.0)
    state = _make_report_state(7.5)
    root_items: list[Any] = []
    sub_items: list[Any] = []
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_start(
            _make_warn_context(requests=0, parent_id=None), MagicMock(), None, root_items
        )
        await hooks.on_llm_start(
            _make_warn_context(requests=0, parent_id="root-1"), MagicMock(), None, sub_items
        )
    assert "stopped at 90%" in root_items[0]["content"]
    assert "stopped at 90%" in sub_items[0]["content"]
    assert "root agent's final report" in sub_items[0]["content"]


@pytest.mark.asyncio
async def test_subagent_critical_budget_warning_reachable_before_reserve() -> None:
    hooks = ReportUsageHooks(model="test-model", max_budget_usd=10.0)
    state = _make_report_state(8.6)
    sub_items: list[Any] = []
    root_items: list[Any] = []
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_start(
            _make_warn_context(requests=0, parent_id="root-1"), MagicMock(), None, sub_items
        )
        await hooks.on_llm_start(
            _make_warn_context(requests=0, parent_id=None), MagicMock(), None, root_items
        )
    assert "[CRITICAL]" in sub_items[0]["content"]
    assert "[URGENT]" in root_items[0]["content"]


@pytest.mark.parametrize(
    ("parent_id", "cost", "expected"),
    [
        ("root-1", 0.0, None),
        ("root-1", 8.9999, None),
        ("root-1", 9.0, SubagentBudgetReservedError),
        ("root-1", 9.0001, SubagentBudgetReservedError),
        ("root-1", 9.5, SubagentBudgetReservedError),
        ("root-1", 9.9999, SubagentBudgetReservedError),
        ("root-1", 10.0, BudgetExceededError),
        ("root-1", 10.0001, BudgetExceededError),
        ("root-1", 25.0, BudgetExceededError),
        (None, 0.0, None),
        (None, 8.9999, None),
        (None, 9.0, None),
        (None, 9.5, None),
        (None, 9.9999, None),
        (None, 10.0, BudgetExceededError),
        (None, 10.0001, BudgetExceededError),
        (None, 25.0, BudgetExceededError),
    ],
)
@pytest.mark.asyncio
async def test_budget_enforcement_decision_table(
    parent_id: str | None, cost: float, expected: type[Exception] | None
) -> None:
    hooks = _make_hooks(10.0)
    state = _make_report_state(cost)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        if expected is None:
            await hooks.on_llm_end(_make_context(parent_id=parent_id), MagicMock(), MagicMock())
        else:
            with pytest.raises(expected):
                await hooks.on_llm_end(_make_context(parent_id=parent_id), MagicMock(), MagicMock())
    state.record_sdk_usage.assert_called_once()


@pytest.mark.asyncio
async def test_no_raise_when_report_state_none() -> None:
    hooks = _make_hooks(1.0)
    with patch("strix.core.hooks.get_global_report_state", return_value=None):
        # Should return early without raising, even with budget set
        await hooks.on_llm_end(_make_context(), MagicMock(), MagicMock())


@pytest.mark.parametrize("bad_budget", [0.0, -0.01, -5.0])
def test_non_positive_budget_rejected(bad_budget: float) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        ReportUsageHooks(model="test-model", max_budget_usd=bad_budget)


def test_budget_exceeded_error_is_runtime_error() -> None:
    err = BudgetExceededError("test")
    assert isinstance(err, RuntimeError)


def test_non_positive_max_turns_rejected() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ReportUsageHooks(model="test-model", max_turns=0)


@pytest.mark.asyncio
async def test_no_turn_warning_below_first_band() -> None:
    hooks = ReportUsageHooks(model="test-model", max_turns=100)
    items: list[Any] = []
    await hooks.on_llm_start(_make_warn_context(requests=68), MagicMock(), None, items)
    assert items == []


@pytest.mark.asyncio
async def test_turn_warning_notice_band() -> None:
    hooks = ReportUsageHooks(model="test-model", max_turns=100)
    items: list[Any] = []
    await hooks.on_llm_start(_make_warn_context(requests=69), MagicMock(), None, items)
    assert len(items) == 1
    content = items[0]["content"]
    assert "[NOTICE]" in content
    assert "finish_scan" in content


@pytest.mark.asyncio
async def test_turn_warning_escalates_and_names_subagent_tool() -> None:
    hooks = ReportUsageHooks(model="test-model", max_turns=100)
    items: list[Any] = []
    await hooks.on_llm_start(
        _make_warn_context(requests=95, parent_id="root-1"), MagicMock(), None, items
    )
    assert len(items) == 1
    content = items[0]["content"]
    assert "[CRITICAL]" in content
    assert "agent_finish" in content


@pytest.mark.asyncio
async def test_turn_warning_root_directive_distinct_from_subagent() -> None:
    hooks = ReportUsageHooks(model="test-model", max_turns=100)

    root_items: list[Any] = []
    await hooks.on_llm_start(
        _make_warn_context(requests=85, parent_id=None), MagicMock(), None, root_items
    )
    root = root_items[0]["content"]

    sub_items: list[Any] = []
    await hooks.on_llm_start(
        _make_warn_context(requests=85, parent_id="root-1"), MagicMock(), None, sub_items
    )
    sub = sub_items[0]["content"]

    assert root != sub
    assert "root agent" in root
    assert "finish_scan" in root
    assert "agent_finish" not in root
    assert "whole scan" in root
    assert "sub-agent" in sub
    assert "agent_finish" in sub
    assert "finish_scan" not in sub
    assert "confirmed" in sub


@pytest.mark.asyncio
async def test_budget_warning_root_directive_distinct_from_subagent() -> None:
    hooks = ReportUsageHooks(model="test-model", max_budget_usd=10.0)
    state = _make_report_state(8.6)

    root_items: list[Any] = []
    sub_items: list[Any] = []
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_start(
            _make_warn_context(requests=0, parent_id=None), MagicMock(), None, root_items
        )
        await hooks.on_llm_start(
            _make_warn_context(requests=0, parent_id="root-1"), MagicMock(), None, sub_items
        )

    root = root_items[0]["content"]
    sub = sub_items[0]["content"]
    assert "finish_scan" in root and "agent_finish" not in root
    assert "agent_finish" in sub and "finish_scan" not in sub
    assert "confirmed" in sub


@pytest.mark.parametrize("parent_id", [None, "root-1"])
@pytest.mark.asyncio
async def test_turn_warning_directive_escalates_per_stage(parent_id: str | None) -> None:
    hooks = ReportUsageHooks(model="test-model", max_turns=100)
    contents: dict[str, str] = {}
    for label, requests in (("notice", 69), ("urgent", 85), ("critical", 95)):
        items: list[Any] = []
        await hooks.on_llm_start(
            _make_warn_context(requests=requests, parent_id=parent_id), MagicMock(), None, items
        )
        contents[label] = items[0]["content"]

    assert len({contents["notice"], contents["urgent"], contents["critical"]}) == 3
    assert "[NOTICE]" in contents["notice"] and "begin planning" in contents["notice"]
    assert "[URGENT]" in contents["urgent"] and "prioritize" in contents["urgent"]
    assert "[CRITICAL]" in contents["critical"] and "STOP" in contents["critical"]


@pytest.mark.asyncio
async def test_no_turn_warning_when_max_turns_unset() -> None:
    hooks = ReportUsageHooks(model="test-model")
    items: list[Any] = []
    await hooks.on_llm_start(_make_warn_context(requests=999), MagicMock(), None, items)
    assert items == []


@pytest.mark.asyncio
async def test_no_budget_warning_below_first_band() -> None:
    hooks = ReportUsageHooks(model="test-model", max_budget_usd=10.0)
    state = _make_report_state(6.9)
    items: list[Any] = []
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_start(_make_warn_context(requests=0), MagicMock(), None, items)
    assert items == []


@pytest.mark.asyncio
async def test_budget_warning_broadcast_content() -> None:
    hooks = ReportUsageHooks(model="test-model", max_budget_usd=10.0)
    state = _make_report_state(9.6)
    items: list[Any] = []
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_start(_make_warn_context(requests=0), MagicMock(), None, items)
    assert len(items) == 1
    content = items[0]["content"]
    assert "[CRITICAL]" in content
    assert "shared across every agent" in content


@pytest.mark.asyncio
async def test_turn_and_budget_warnings_stack() -> None:
    hooks = ReportUsageHooks(model="test-model", max_budget_usd=10.0, max_turns=100)
    state = _make_report_state(8.6)
    items: list[Any] = []
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_start(_make_warn_context(requests=89), MagicMock(), None, items)
    assert len(items) == 2
    joined = " ".join(i["content"] for i in items)
    assert "Turn budget" in joined
    assert "cost budget" in joined


def _make_interactive_hooks(max_budget: float | None) -> ReportUsageHooks:
    return ReportUsageHooks(model="test-model", max_budget_usd=max_budget, interactive=True)


@pytest.mark.asyncio
async def test_interactive_at_budget_pauses_instead_of_stopping() -> None:
    hooks = _make_interactive_hooks(10.0)
    state = _make_report_state(10.0)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetPausedError),
    ):
        await hooks.on_llm_end(_make_context(parent_id=None), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_interactive_subagent_has_no_reserve() -> None:
    hooks = _make_interactive_hooks(10.0)
    state = _make_report_state(9.5)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(_make_context(parent_id="root-1"), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_interactive_subagent_pauses_at_full_budget() -> None:
    hooks = _make_interactive_hooks(10.0)
    state = _make_report_state(10.5)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetPausedError),
    ):
        await hooks.on_llm_end(_make_context(parent_id="root-1"), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_extend_budget_lifts_the_pause() -> None:
    hooks = _make_interactive_hooks(10.0)
    state = _make_report_state(10.5)
    hooks.extend_budget()
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(_make_context(parent_id=None), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_extend_budget_adds_original_amount_each_time() -> None:
    hooks = _make_interactive_hooks(10.0)
    hooks.extend_budget()
    hooks.extend_budget()
    state = _make_report_state(29.9)
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(_make_context(parent_id=None), MagicMock(), MagicMock())
    state = _make_report_state(30.0)
    with (
        patch("strix.core.hooks.get_global_report_state", return_value=state),
        pytest.raises(BudgetPausedError),
    ):
        await hooks.on_llm_end(_make_context(parent_id=None), MagicMock(), MagicMock())


@pytest.mark.asyncio
async def test_interactive_subagent_uses_root_warning_bands() -> None:
    hooks = _make_interactive_hooks(10.0)
    state = _make_report_state(7.4)
    items: list[Any] = []
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_start(
            _make_warn_context(requests=0, parent_id="root-1"), MagicMock(), None, items
        )
    assert len(items) == 1
    content = items[0]["content"]
    assert "[NOTICE]" in content
    assert "paused until the user chooses to continue" in content
    assert "reserve" not in content.lower()


@pytest.mark.parametrize(
    ("cost", "max_budget", "interactive", "expected"),
    [
        (0.0, None, False, (False, False)),
        (100.0, None, False, (False, False)),
        (5.0, 10.0, False, (False, False)),
        (9.0, 10.0, False, (False, True)),
        (10.0, 10.0, False, (True, True)),
        (10.0, 20.0, False, (False, False)),
        (10.0, 10.0, True, (False, False)),
    ],
)
def test_recomputed_budget_flags(
    cost: float,
    max_budget: float | None,
    interactive: bool,
    expected: tuple[bool, bool],
) -> None:
    assert recomputed_budget_flags(cost, max_budget, interactive=interactive) == expected


def _patch_guard_budget(
    monkeypatch: pytest.MonkeyPatch, *, window: int, turn: int, keep: int = 500
) -> None:
    """Pin the guard's window arithmetic and make one char cost one token."""
    monkeypatch.setattr(compaction, "count_tokens", lambda _m, t: len(t))
    monkeypatch.setattr(compaction, "count_tokens_cached", lambda _m, t: len(t))
    monkeypatch.setattr(compaction, "context_window", lambda _m: window)
    monkeypatch.setattr(compaction, "turn_output_tokens", lambda _m: turn)
    context = ContextSettings()
    context.compact_buffer_tokens = 1_000
    context.keep_tokens = keep
    monkeypatch.setattr(compaction, "load_settings", lambda: SimpleNamespace(context=context))


def _oversized_history(turns: int, size: int) -> list[Any]:
    items: list[Any] = [{"role": "user", "content": "opening task"}]
    for i in range(turns):
        items += [
            {"type": "function_call", "call_id": f"c{i}", "name": "exec", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": "A" * size},
        ]
    return items


@pytest.mark.asyncio
async def test_guard_trims_oversized_history_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK hands on_llm_start the very list it is about to send, so the
    # trim has to land on that object for the request to shrink.
    _patch_guard_budget(monkeypatch, window=8_000, turn=1_000)
    hooks = ReportUsageHooks(model="test-model")
    items = _oversized_history(10, 1_000)
    same_list = items
    before = len(items)

    await hooks.on_llm_start(_make_warn_context(requests=1), MagicMock(), "SYSTEM", items)

    assert items is same_list
    assert len(items) < before
    budget = compaction.budget_tokens("test-model")
    assert compaction.measure_request_tokens("test-model", "SYSTEM", "", items) <= budget


@pytest.mark.asyncio
async def test_guard_leaves_a_history_within_budget_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard_budget(monkeypatch, window=100_000, turn=1_000)
    hooks = ReportUsageHooks(model="test-model")
    items = _oversized_history(3, 100)
    before = list(items)

    await hooks.on_llm_start(_make_warn_context(requests=1), MagicMock(), "SYSTEM", items)

    assert items == before


@pytest.mark.asyncio
async def test_guard_runs_before_the_turn_warning_is_appended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The warning must survive the trim, and must not be measured as history.
    _patch_guard_budget(monkeypatch, window=8_000, turn=1_000)
    hooks = ReportUsageHooks(model="test-model", max_turns=100)
    items = _oversized_history(10, 1_000)
    before = len(items)

    await hooks.on_llm_start(_make_warn_context(requests=95), MagicMock(), "SYSTEM", items)

    assert len(items) < before
    assert "[CRITICAL]" in items[-1]["content"]


@pytest.mark.asyncio
async def test_guard_reports_the_prompt_overhead_it_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Session compaction cannot see the prepared system prompt; this is how it
    # finds out what that fixed cost really is.
    _patch_guard_budget(monkeypatch, window=100_000, turn=1_000)
    hooks = ReportUsageHooks(model="test-model")

    assert hooks.observed_prompt_overhead_tokens == 0

    await hooks.on_llm_start(_make_warn_context(requests=1), MagicMock(), "S" * 4_000, [])

    assert hooks.observed_prompt_overhead_tokens == 4_000


@pytest.mark.asyncio
async def test_guard_failure_never_blocks_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_model: str) -> int:
        raise RuntimeError("budget lookup exploded")

    monkeypatch.setattr(compaction, "budget_tokens", _boom)
    hooks = ReportUsageHooks(model="test-model", max_turns=100)
    items: list[Any] = []

    await hooks.on_llm_start(_make_warn_context(requests=95), MagicMock(), "SYSTEM", items)

    assert "[CRITICAL]" in items[0]["content"]
