"""Tests for the run-slot gate that bounds concurrently running agents.

Local LLM servers hold one prompt cache per slot, so an unbounded agent graph
evicts its own prefixes and re-prefills on every switch. ``STRIX_MAX_CONCURRENT_AGENTS``
caps how many sub-agents drive the model at once; root is exempt so it can never
queue behind the child it is waiting on.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agents.tool_context import ToolContext

from strix.config.settings import RuntimeSettings
from strix.core.agents import AgentCoordinator
from strix.core.slots import AgentSlots, SlotSnapshot
from strix.tools.agents_graph.tools import wait_for_agents


async def _drain() -> None:
    """Let every runnable task reach its next await."""
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_unlimited_by_default_lets_every_agent_run() -> None:
    slots = AgentSlots()

    assert not slots.enabled
    async with slots.hold("a"), slots.hold("b"), slots.hold("c"):
        assert slots.snapshot().active == 0  # nothing is tracked when the gate is off


@pytest.mark.asyncio
async def test_second_agent_queues_until_a_slot_frees() -> None:
    slots = AgentSlots(1)
    started: list[str] = []

    async def _work(agent_id: str, release: asyncio.Event) -> None:
        async with slots.hold(agent_id):
            started.append(agent_id)
            await release.wait()

    first_release = asyncio.Event()
    first = asyncio.create_task(_work("child-1", first_release))
    await _drain()

    second = asyncio.create_task(_work("child-2", asyncio.Event()))
    await _drain()

    assert started == ["child-1"]
    assert slots.snapshot() == SlotSnapshot(limit=1, active=1, queued=1)

    first_release.set()
    await first
    await _drain()

    assert started == ["child-1", "child-2"]

    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second


@pytest.mark.asyncio
async def test_root_is_exempt_and_never_queues() -> None:
    slots = AgentSlots(1)
    child_release = asyncio.Event()
    root_ran = asyncio.Event()

    async def _child() -> None:
        async with slots.hold("child"):
            await child_release.wait()

    async def _root() -> None:
        async with slots.hold("root", exempt=True):
            root_ran.set()

    child = asyncio.create_task(_child())
    await _drain()
    await asyncio.wait_for(_root(), timeout=1)

    assert root_ran.is_set()
    assert slots.snapshot().active == 1  # only the child holds a slot

    child_release.set()
    await child


@pytest.mark.asyncio
async def test_a_blocked_agent_hands_its_slot_to_a_queued_one() -> None:
    slots = AgentSlots(1)
    ran: list[str] = []

    async def _waiter() -> None:
        async with slots.hold("parent"):
            ran.append("parent-start")
            async with slots.released("parent"):
                await asyncio.sleep(0.05)
            ran.append("parent-resume")

    async def _queued() -> None:
        async with slots.hold("child"):
            ran.append("child")

    waiter = asyncio.create_task(_waiter())
    await _drain()
    child = asyncio.create_task(_queued())
    await _drain()

    await asyncio.wait_for(asyncio.gather(waiter, child), timeout=2)

    # The child ran inside the parent's blocked window, not after it.
    assert ran == ["parent-start", "child", "parent-resume"]
    assert slots.snapshot().active == 0


@pytest.mark.asyncio
async def test_releasing_an_agent_that_holds_nothing_is_a_no_op() -> None:
    slots = AgentSlots(1)

    async with slots.released("never-held"):
        pass

    assert slots.snapshot().active == 0
    async with slots.hold("child"):
        assert slots.snapshot().active == 1


@pytest.mark.asyncio
async def test_non_interactive_wait_for_agents_frees_the_slot() -> None:
    """The agent being waited on may be the one queued for this slot."""
    coordinator = AgentCoordinator()
    coordinator.configure_slots(1)
    await coordinator.register("parent", "Parent", parent_id="root")

    inner: dict[str, Any] = {
        "agent_id": "parent",
        "coordinator": coordinator,
        "interactive": False,
    }
    ctx = ToolContext(
        context=inner,
        tool_name="wait_for_agents",
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    free_during_wait: list[int] = []

    async def _observe() -> None:
        await asyncio.sleep(0.05)
        free_during_wait.append(coordinator.slots.snapshot().free)
        await coordinator.send("parent", {"type": "information", "content": "child done"})

    async with coordinator.slots.hold("parent"):
        assert coordinator.slots.snapshot().free == 0
        observer = asyncio.create_task(_observe())
        raw = await wait_for_agents.on_invoke_tool(
            ctx, json.dumps({"reason": "waiting", "timeout_seconds": 5})
        )
        await observer
        # Re-acquired on the way out, so the agent resumes turns under the cap.
        assert coordinator.slots.snapshot().free == 0

    assert json.loads(raw)["wait_outcome"] == "message_arrived"
    assert free_during_wait == [1]


def test_limit_reads_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RuntimeSettings().max_concurrent_agents == 0

    monkeypatch.setenv("STRIX_MAX_CONCURRENT_AGENTS", "2")

    assert RuntimeSettings().max_concurrent_agents == 2
