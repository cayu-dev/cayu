"""A failed approved tool round must not overwrite a requested interruption.

Sequence, with two ``CayuApp`` instances on one store (a worker and an API
process):

1. A session pauses for tool approval. The worker resolves it, and the approved
   round continues; the session is ``running``.
2. An operator interrupts the session through the other app. The store accepts
   it: status ``interrupting`` plus the durable ``pending_session_interrupt``
   and ``pending_interruption_cascade`` markers.
3. The round then fails before its approval is closed.

The continuation may pause the session again only when no interruption was
requested. It leaves ``interrupting`` alone and finishes the operator's request:
one ``session.interrupted`` event with the operator's reason, both markers
cleared, and the approval still pending.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.operator_interrupt_support import (
    OperatorInterrupt,
    assert_operator_interruption_finalized,
    session_paused_for_approval,
)
from tests.runtime.runtime_session_support import collect_tool_approval_events

from cayu.events import EventType
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore

_SESSION_ID = "interrupted-during-failing-approved-round"


class _TerminalEventFault:
    """Run a hook, then lose the approved tool's terminal event, once."""

    invocation_lifecycle_command_version = 1
    before_failure = None

    async def append_events(self, session_id, events):
        if self.before_failure is not None and any(
            event.type is EventType.TOOL_CALL_COMPLETED for event in events
        ):
            before_failure, self.before_failure = self.before_failure, None
            await before_failure()
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


class _MemoryStore(_TerminalEventFault, InMemorySessionStore):
    invocation_lifecycle_command_version = 1


class _SQLiteStore(_TerminalEventFault, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1


def _store(backend: str, tmp_path):
    return _MemoryStore() if backend == "memory" else _SQLiteStore(tmp_path / "sessions.db")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_approved_round_does_not_overwrite_requested_interruption(backend, tmp_path):
    async def scenario():
        store = _store(backend, tmp_path)
        worker, api, resolution = await session_paused_for_approval(store, session_id=_SESSION_ID)
        try:
            operator = OperatorInterrupt(api, store, session_id=_SESSION_ID)
            events_before = len(await store.load_events(_SESSION_ID))
            store.before_failure = operator.commit
            continuation_events = await collect_tool_approval_events(worker, resolution)
            await operator.settle()
            assert operator.tasks

            # The stream ends with the operator's interruption, not a second pause.
            assert continuation_events[-1].type is EventType.SESSION_INTERRUPTED
            assert continuation_events[-1].payload["interruption_type"] == "operator_requested"
            await assert_operator_interruption_finalized(
                store, worker, session_id=_SESSION_ID, events_before=events_before
            )
        finally:
            await worker.aclose()
            await api.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_approved_round_without_an_interruption_pauses_for_approval(backend, tmp_path):
    async def scenario():
        store = _store(backend, tmp_path)
        worker, api, resolution = await session_paused_for_approval(store, session_id=_SESSION_ID)
        try:

            async def no_interruption() -> None:
                return None

            store.before_failure = no_interruption
            continuation_events = await collect_tool_approval_events(worker, resolution)

            assert continuation_events[-1].type is EventType.SESSION_INTERRUPTED
            assert continuation_events[-1].payload["interruption_type"] == "tool_approval_required"
            checkpoint = await store.load_checkpoint(_SESSION_ID)
            assert checkpoint is not None and "pending_tool_approval" in checkpoint
        finally:
            await worker.aclose()
            await api.aclose()

    asyncio.run(scenario())
