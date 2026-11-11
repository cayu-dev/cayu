"""A background drain must not discard a recovery plan's descendant work."""

import asyncio
from datetime import UTC, datetime

from tests.core.test_recovery_plans import _app, _create_running_session

from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.recovery import (
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)
from cayu.sessions.requests import RunRequest


def test_drain_does_not_erase_an_in_progress_recovery_cascade(monkeypatch):
    async def scenario():
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, "repair-parent")
        await store.update_status("repair-parent", SessionStatus.INTERRUPTED)
        payload = {
            "interruption_type": "operator_requested",
            "reason": "stop descendants",
            "metadata": {},
        }
        await store.transform_checkpoint(
            "repair-parent",
            lambda _session, checkpoint: {
                **checkpoint,
                "pending_interruption_cascade": {
                    "attempt_id": "repair-attempt",
                    "interrupt_payload": payload,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            },
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="repair-child",
                parent_session_id="repair-parent",
                messages=[Message.text("user", "background work")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        plan = await app.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=("repair-parent",)))
        )
        child_started = asyncio.Event()
        release_child = asyncio.Event()
        original_interrupt = app._background_interruption_coordinator._interrupt_session

        async def delayed_interrupt(request):
            child_started.set()
            await release_child.wait()
            async for event in original_interrupt(request):
                yield event

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            delayed_interrupt,
        )
        repair = asyncio.create_task(
            app.execute_recovery(
                RecoveryExecutionRequest(plan=plan, execution_id="repair-execution")
            )
        )
        try:
            await asyncio.wait_for(child_started.wait(), 5)
            # If drain correctly waits, this lets the real child interrupt finish.
            # If it chooses not to wait for foreground recovery, it must at least
            # leave that recovery's shared workers and queue intact.
            asyncio.get_running_loop().call_later(0.05, release_child.set)
            assert await app.drain_background_interruptions(timeout_s=2)
            receipt = await asyncio.wait_for(repair, 5)
            child = await store.load("repair-child")
            checkpoint = await store.load_checkpoint("repair-parent")
            assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
            assert "pending_interruption_cascade" not in checkpoint
            assert child.status is SessionStatus.INTERRUPTED, (
                "drain cancelled the child worker but recovery still cleared the "
                "durable cascade marker and reported EXECUTED"
            )
        finally:
            release_child.set()
            if not repair.done():
                repair.cancel()
            await asyncio.gather(repair, return_exceptions=True)
            await app.drain_background_interruptions(timeout_s=2)

    asyncio.run(scenario())
