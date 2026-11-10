"""Recovery-plan execution hands a pending group acknowledgement to its caller.

A recovery plan that claims a task continuation acknowledges the group
execution it opened. If that acknowledgement stays pending, the retry handle is
its only owner, so execution raises it instead of committing a failed receipt.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_recovery_plans import _app, _create_running_session

from cayu.events import EventType
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.recovery import (
    RecoveryExecutionRequest,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)
from cayu.tasks._execution_settlement import (
    TaskExecutionSettlement,
    TaskExecutionSettlementPending,
    pending_task_execution_settlements,
)
from cayu.tasks.memory import InMemoryTaskStore

_SESSION_ID = "sess_pending_acknowledgement"


def test_recovery_execution_raises_a_pending_acknowledgement_instead_of_a_receipt(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, _SESSION_ID)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=(_SESSION_ID,), inactive_for_seconds=0)
            )
        )
        handle = TaskExecutionSettlement(InMemoryTaskStore())

        async def acknowledgement_pending(**_kwargs):
            raise TaskExecutionSettlementPending(handle)

        monkeypatch.setattr(
            app._recovery_plan_coordinator, "_apply_decision", acknowledgement_pending
        )
        with pytest.raises(TaskExecutionSettlementPending) as raised:
            await app.execute_recovery(
                RecoveryExecutionRequest(plan=plan, execution_id="pending-acknowledgement")
            )
        assert raised.value.settlement is handle
        assert not [
            event
            for event in await store.load_events(_SESSION_ID)
            if event.type is EventType.RECOVERY_PLAN_ITEM_EXECUTED
        ]

    asyncio.run(run())


def test_every_items_pending_acknowledgement_reaches_the_caller(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = _app(store)
        session_ids = (f"{_SESSION_ID}_a", f"{_SESSION_ID}_b")
        for session_id in session_ids:
            await _create_running_session(store, app, session_id)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=session_ids, inactive_for_seconds=0)
            )
        )
        handles = []

        async def acknowledgement_pending(**_kwargs):
            handle = TaskExecutionSettlement(InMemoryTaskStore())
            handles.append(handle)
            raise TaskExecutionSettlementPending(handle)

        monkeypatch.setattr(
            app._recovery_plan_coordinator, "_apply_decision", acknowledgement_pending
        )
        with pytest.raises(BaseExceptionGroup) as raised:
            await app.execute_recovery(
                RecoveryExecutionRequest(plan=plan, execution_id="pending-acknowledgements")
            )
        assert len(handles) == len(session_ids)
        assert {id(item) for item in pending_task_execution_settlements(raised.value)} == {
            id(handle) for handle in handles
        }

    asyncio.run(run())


def test_a_cancelled_items_pending_acknowledgement_reaches_the_caller(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, _SESSION_ID)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=(_SESSION_ID,), inactive_for_seconds=0)
            )
        )
        handle = TaskExecutionSettlement(InMemoryTaskStore())

        async def cancelled_while_acknowledging(**_kwargs):
            # The item's own work is cancelled; its cancellation carries the handle.
            cancellation = asyncio.CancelledError()
            cancellation.__cause__ = TaskExecutionSettlementPending(handle)
            raise cancellation

        monkeypatch.setattr(
            app._recovery_plan_coordinator, "_apply_decision", cancelled_while_acknowledging
        )
        with pytest.raises(asyncio.CancelledError) as raised:
            await app.execute_recovery(
                RecoveryExecutionRequest(plan=plan, execution_id="cancelled-acknowledgement")
            )
        assert pending_task_execution_settlements(raised.value) == [handle]

    asyncio.run(run())
