"""Concurrency gate bounding how many agents drive the model at once.

Strix spawns children as detached tasks, so an unbounded graph points every
agent at the same LLM endpoint simultaneously. That is fine for a hosted
provider and ruinous for a local one: each agent carries its own conversation
prefix, so N concurrent agents evict each other's prompt cache and the server
spends most of its time re-prefilling.

A slot is held for as long as an agent is *actively running turns*, not merely
alive. Everything that blocks on another agent - parking for a message,
``wait_for_agents`` - gives the slot back first, which is what keeps the gate
from deadlocking: a slot holder never waits on an agent stuck in the queue.

The root agent is deliberately exempt. It is the one agent that can stall
without going through a release point (polling the graph, sleeping in a shell
command, awaiting a reply it requested by message), and holding the last slot
there would strand the very child it is waiting for.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SlotSnapshot:
    """Point-in-time view of the gate, for reporting back to a spawning agent."""

    limit: int
    active: int
    queued: int

    @property
    def unlimited(self) -> bool:
        return self.limit <= 0

    @property
    def free(self) -> int:
        return max(0, self.limit - self.active)

    def describe(self) -> str | None:
        """One line an agent can act on, or None when the gate is off."""
        if self.unlimited:
            return None
        if self.free > 0:
            return f"Agent slots: {self.active}/{self.limit} busy — a slot is free for this child."
        return (
            f"Agent slots: {self.active}/{self.limit} busy, {self.queued} already queued — "
            "this child is registered but stays idle until a slot frees. Don't spawn more "
            "until the queue drains; wait for the running ones instead."
        )


class AgentSlots:
    """FIFO gate over concurrently running agents. ``limit <= 0`` disables it."""

    def __init__(self, limit: int = 0) -> None:
        self._limit = max(0, int(limit))
        self._sem = asyncio.Semaphore(self._limit) if self._limit else None
        # Re-entrancy depth per agent; an agent absent from the map holds nothing.
        self._depths: dict[str, int] = {}
        self._queued = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def enabled(self) -> bool:
        return self._sem is not None

    def snapshot(self) -> SlotSnapshot:
        return SlotSnapshot(limit=self._limit, active=len(self._depths), queued=self._queued)

    @contextlib.asynccontextmanager
    async def hold(self, agent_id: str, *, exempt: bool = False) -> AsyncIterator[None]:
        """Occupy a slot for the duration of the block, waiting for one if needed.

        ``exempt`` runs the block ungated - used for the root agent, which must
        never queue behind its own children.
        """
        if self._sem is None or exempt:
            yield
            return

        await self._acquire(agent_id)
        try:
            yield
        finally:
            self._release(agent_id)

    @contextlib.asynccontextmanager
    async def released(self, agent_id: str) -> AsyncIterator[None]:
        """Hand the slot back while the agent blocks on something other than the model.

        A no-op for an agent that holds nothing (exempt, or gate disabled). The
        slot is re-acquired on exit, which may itself queue.
        """
        if not self._release(agent_id):
            yield
            return
        try:
            yield
        finally:
            await self._acquire(agent_id)

    async def _acquire(self, agent_id: str) -> None:
        if self._sem is None:
            return
        depth = self._depths.get(agent_id, 0)
        if depth:
            self._depths[agent_id] = depth + 1
            return

        if self._sem.locked():
            logger.info(
                "agent %s waiting for a run slot (%d/%d busy, %d queued)",
                agent_id,
                len(self._depths),
                self._limit,
                self._queued,
            )
        self._queued += 1
        try:
            await self._sem.acquire()
        finally:
            self._queued -= 1
        self._depths[agent_id] = 1

    def _release(self, agent_id: str) -> bool:
        """Drop one level of the agent's hold; True if a slot went back to the pool."""
        if self._sem is None:
            return False
        depth = self._depths.get(agent_id, 0)
        if depth <= 0:
            return False
        if depth > 1:
            self._depths[agent_id] = depth - 1
            return False
        del self._depths[agent_id]
        self._sem.release()
        return True
