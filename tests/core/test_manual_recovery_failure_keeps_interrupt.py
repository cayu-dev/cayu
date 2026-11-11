"""A failed manual tool-approval recovery must not overwrite a requested interruption.

Sequence, with two ``CayuApp`` instances on one store (a worker and an API
process):

1. A session pauses for tool approval, is approved, and loses its terminal tool
   event, so it is ``interrupted`` and needs manual recovery.
2. The worker starts ``recover_tool_approval``; the session is ``running``.
3. An operator interrupts the session through the other app. The store accepts
   it: status ``interrupting`` plus the durable ``pending_session_interrupt``
   and ``pending_interruption_cascade`` markers.
4. The recovery then fails before its result is persisted.

The recovery may only replace the status it set itself. It leaves
``interrupting`` alone and finishes the operator's request: one
``session.interrupted`` event with the operator's reason, both markers cleared,
and the approval still pending for a later recovery.
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

from cayu.approvals.tools import (
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.events import EventType
from cayu.runtime._environment_lifecycle import (
    EnvironmentFactoryOperation,
    EnvironmentFactoryResolutionResult,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.storage.sqlite import SQLiteSessionStore

_SESSION_ID = "interrupted-during-failing-recovery"


class _RecoveryFault:
    """Lose the approved tool's terminal event once, then fail one recovery read."""

    invocation_lifecycle_command_version = 1
    failed_terminal_once = False
    fail_recovery_read = False
    before_failure = None

    async def append_events(self, session_id, events):
        if not self.failed_terminal_once and any(
            event.type is EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)

    async def load_events(self, session_id, *args, **kwargs):
        if self.fail_recovery_read:
            current = await self.load(session_id)
            if current is not None and current.status is SessionStatus.RUNNING:
                self.fail_recovery_read = False
                await self.before_failure()
                raise RuntimeError("injected recovery read failure")
        return await super().load_events(session_id, *args, **kwargs)


class _MemoryStore(_RecoveryFault, InMemorySessionStore):
    invocation_lifecycle_command_version = 1


class _SQLiteStore(_RecoveryFault, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1


def _store(backend: str, tmp_path):
    return _MemoryStore() if backend == "memory" else _SQLiteStore(tmp_path / "sessions.db")


def _recovery(worker, resolution: ToolApprovalRequest):
    return worker.recover_tool_approval(
        ToolApprovalRecoveryRequest(
            session_id=_SESSION_ID,
            approval_id=resolution.approval_id,
            tool_round_id=resolution.tool_round_id,
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified result",
        )
    )


async def _session_needing_manual_recovery(store):
    """Return a worker, an API app and the approval whose tool lost its result."""

    worker, api, resolution = await session_paused_for_approval(store, session_id=_SESSION_ID)
    failed = await collect_tool_approval_events(worker, resolution)
    assert failed[-1].type is EventType.SESSION_INTERRUPTED
    assert store.failed_terminal_once
    return worker, api, resolution


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_manual_recovery_does_not_overwrite_requested_interruption(backend, tmp_path):
    async def scenario():
        store = _store(backend, tmp_path)
        worker, api, resolution = await _session_needing_manual_recovery(store)
        try:
            operator = OperatorInterrupt(api, store, session_id=_SESSION_ID)
            events_before = len(await store.load_events(_SESSION_ID))
            store.before_failure = operator.commit
            store.fail_recovery_read = True
            with pytest.raises(RuntimeError, match="injected recovery read failure"):
                async for _event in _recovery(worker, resolution):
                    pass
            await operator.settle()
            assert operator.tasks

            await assert_operator_interruption_finalized(
                store, worker, session_id=_SESSION_ID, events_before=events_before
            )

            # The verified outcome was not applied, so the same recovery still works.
            retried = [event async for event in _recovery(worker, resolution)]
            assert retried[-1].type is EventType.SESSION_COMPLETED
        finally:
            await worker.aclose()
            await api.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_recovery_reconnect_does_not_overwrite_requested_interruption(
    backend, tmp_path, monkeypatch
):
    """The same request, accepted while the recovery reconnects an environment that fails."""

    async def scenario():
        store = _store(backend, tmp_path)
        worker, api, resolution = await _session_needing_manual_recovery(store)
        try:
            operator = OperatorInterrupt(api, store, session_id=_SESSION_ID)
            lifecycle = worker._recovery_coordinator._environment_lifecycle
            resolve_factory = lifecycle.resolve_factory

            async def fail_reconnect_after_interrupt(**kwargs):
                if operator.tasks or kwargs["operation"] is not (
                    EnvironmentFactoryOperation.RECONNECT
                ):
                    return await resolve_factory(**kwargs)
                await operator.commit()
                return EnvironmentFactoryResolutionResult(
                    registered_environment=kwargs["registered_environment"],
                    events=[],
                    error=RuntimeError("environment factory unavailable"),
                )

            monkeypatch.setattr(lifecycle, "resolve_factory", fail_reconnect_after_interrupt)
            events_before = len(await store.load_events(_SESSION_ID))
            recovery_events = [event async for event in _recovery(worker, resolution)]
            await operator.settle()
            assert operator.tasks

            # The stream ends with the operator's interruption, not a second pause.
            assert recovery_events[-1].type is EventType.SESSION_INTERRUPTED
            assert recovery_events[-1].payload["interruption_type"] == "operator_requested"
            await assert_operator_interruption_finalized(
                store, worker, session_id=_SESSION_ID, events_before=events_before
            )

            retried = [event async for event in _recovery(worker, resolution)]
            assert retried[-1].type is EventType.SESSION_COMPLETED
        finally:
            await worker.aclose()
            await api.aclose()

    asyncio.run(scenario())
