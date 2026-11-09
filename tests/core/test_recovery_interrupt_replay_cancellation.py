"""Joining an already-published interrupt must still redeliver caller cancellation."""

import asyncio

import pytest
from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingTerminalToolEventStore,
    _crashed_tool_round_app,
)
from tests.runtime.runtime_session_support import (
    collect_interrupt_events,
    collect_tool_round_recovery_events,
)
from tests.runtime.runtime_tool_support import PureRecordingTool

from cayu import AgentSpec, CayuApp, EventType
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.tools.rounds import ToolRoundRecoveryRequest


class _PausedRecoveryClaimStore(FailingTerminalToolEventStore):
    invocation_lifecycle_command_version = 1

    def __init__(self):
        super().__init__()
        self.pause_claim = False
        self.claim_entered = asyncio.Event()
        self.allow_claim = asyncio.Event()
        self.interrupt_committed = asyncio.Event()

    async def reserve_stalled_run_recovery(self, session_id, **kwargs):
        if self.pause_claim:
            self.pause_claim = False
            self.claim_entered.set()
            await self.allow_claim.wait()
        return await super().reserve_stalled_run_recovery(session_id, **kwargs)

    async def transition_status_and_checkpoint(self, session_id, **kwargs):
        result = await super().transition_status_and_checkpoint(session_id, **kwargs)
        if kwargs["to_status"] is SessionStatus.INTERRUPTING:
            self.interrupt_committed.set()
        return result


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_manual_recovery_interrupt_replay_preserves_caller_cancellation(cancel_caller):
    session_id = "cancel-interrupt-replay"
    store = _PausedRecoveryClaimStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    operator = CayuApp(session_store=store, enable_logging=False)
    operator.register_provider(FakeProvider([]), default=True)
    operator.register_agent(AgentSpec(name="assistant", model="fake-model"))
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    assert profile is not None
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="verified",
    )

    async def scenario():
        # Reconstruct a live epoch left behind by the crashed tool invocation.
        await rebind_test_invocation(
            store,
            session_id,
            target_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, current: (
                checkpoint_with_active_invocation_execution_profile(
                    current,
                    session_id=session.id,
                    interaction_id=profile.interaction_id,
                    run_epoch=session.run_epoch + 1,
                    profile=profile.profile,
                    expected=profile,
                )
            ),
        )
        await store.release_run_fence(session_id)
        store.pause_claim = True
        recovery = asyncio.create_task(collect_tool_round_recovery_events(app, request))
        interruption = None
        try:
            await asyncio.wait_for(store.claim_entered.wait(), 10)
            interruption = asyncio.create_task(
                collect_interrupt_events(
                    operator,
                    InterruptSessionRequest(session_id=session_id, reason="operator stop"),
                )
            )
            await asyncio.wait_for(store.interrupt_committed.wait(), 10)
            await operator._session_finalization.finalize_abandoned_session_by_id(session_id)
            events = await asyncio.wait_for(interruption, 10)
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert await operator.drain_background_interruptions(timeout_s=5)
            assert await app.drain_background_interruptions(timeout_s=5)
            if cancel_caller:
                recovery.cancel("caller stopped waiting for recovery")
                # Deliver cancellation while the durable claim is still shielded.
                await asyncio.sleep(0)
            store.allow_claim.set()
            if cancel_caller:
                with pytest.raises(
                    asyncio.CancelledError, match="caller stopped waiting for recovery"
                ):
                    await asyncio.wait_for(recovery, 10)
                assert recovery.cancelled()
            else:
                recovered = await asyncio.wait_for(recovery, 10)
                assert recovered[-1].id == events[-1].id
            assert tool.calls == [{}]
            assert not any(
                event.payload.get("manual_recovery")
                for event in await store.load_events(session_id)
            )
        finally:
            store.allow_claim.set()
            for task in (recovery, interruption):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (recovery, interruption) if task is not None),
                return_exceptions=True,
            )

    asyncio.run(scenario())
