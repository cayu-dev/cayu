"""Recovery claims contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.runtime.runtime_provider_support import FakeProvider, VersionedFakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingAfterPendingToolRoundCheckpointStore,
    FailingTerminalToolEventStore,
    _crashed_tool_round_app,
)
from tests.runtime.runtime_session_support import (
    _private_events_for_public_events,
    _private_record_for_public_event,
    _runtime_exception_graph,
    assert_only_model_step_publication_checkpoint,
    collect_events,
    collect_interrupt_events,
    collect_resume_events,
    collect_tool_round_recovery_events,
)
from tests.runtime.runtime_tool_support import PureRecordingTool, SideEffectTool

import cayu.runtime.execution_profiles as execution_profiles_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.context.structured_output import StructuredOutputSpec
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _recovery_coordinator as recovery_coordinator_module
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.runtime import _session_finalization as session_finalization_module
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, ResumeRequest, RunRequest
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.redaction import SecretRedactor


class BarrierSideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record execution.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self._entered = 0
        self._both_entered = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        self._entered += 1
        if self._entered >= 2:
            self._both_entered.set()
        # Hold both calls until each has started so a terminal-event crash leaves
        # BOTH started-but-unresolved (deterministic multi-call recovery shape).
        await self._both_entered.wait()
        return ToolResult(content="recorded")


class PureBarrierRecordingTool(BarrierSideEffectTool):
    """Pure fixed-result calls with a two-call synchronization barrier."""

    spec = BarrierSideEffectTool.spec.model_copy(update={"effect": ToolEffect.NONE})


class FailingAllTerminalToolEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failing = True

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if self.failing and any(
            event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            for event in events
        ):
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


class FailingPostPersistEventsLoadStore(FailingTerminalToolEventStore):
    invocation_lifecycle_command_version = 1

    """Crash the round like the base store, then fail the first event load that
    happens after a manual-recovery terminal event was persisted (the
    remaining-unknown recheck inside recover_tool_round)."""

    def __init__(self) -> None:
        super().__init__()
        self.arm_post_persist_load_failure = False

    async def load_events(self, session_id: str) -> list[Event]:
        events = await super().load_events(session_id)
        if self.arm_post_persist_load_failure and any(
            event.payload.get("manual_recovery") is True for event in events
        ):
            self.arm_post_persist_load_failure = False
            raise RuntimeError("events unavailable after manual recovery persisted")
        return events


class BlockingPostPersistEventsLoadStore(FailingTerminalToolEventStore):
    invocation_lifecycle_command_version = 1

    """Pause the first recovery after its manual terminal outcome is durable."""

    def __init__(self, *, ownership_clock=None) -> None:
        super().__init__(ownership_clock=ownership_clock)
        self.post_persist_load_started = asyncio.Event()
        self.allow_post_persist_load = asyncio.Event()
        self.blocked_post_persist_load = False

    async def load_events(self, session_id: str) -> list[Event]:
        events = await super().load_events(session_id)
        if not self.blocked_post_persist_load and any(
            event.payload.get("manual_recovery") is True for event in events
        ):
            self.blocked_post_persist_load = True
            self.post_persist_load_started.set()
            await self.allow_post_persist_load.wait()
        return events


def test_expired_recovery_claim_preserves_fatal_signal_and_existing_graph() -> None:
    fatal_leaf = GeneratorExit("manual recovery fence aborted")
    prior_failure = RuntimeError("fence commit acknowledgement failed")
    fatal_signal = BaseExceptionGroup(
        "manual recovery fence aborted",
        [fatal_leaf, prior_failure],
    )
    lease_failure = recovery_coordinator_module._IncompleteRecoveryClaimLost(
        "Incomplete-session recovery claim acknowledgement consumed its lease."
    )

    authoritative = recovery_coordinator_module._authoritative_expired_recovery_claim_failure(
        fatal_signal,
        lease_failure,
    )
    failure_graph = _runtime_exception_graph(authoritative)

    assert authoritative is fatal_signal
    assert sum(candidate is fatal_leaf for candidate in failure_graph) == 1
    assert sum(candidate is prior_failure for candidate in failure_graph) == 1
    assert sum(candidate is lease_failure for candidate in failure_graph) == 1


def test_cayu_app_recover_tool_round_failed_outcome_resumes():
    session_id = "sess_tool_round_manual_failed"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.FAILED,
                message="side effect confirmed absent downstream",
            ),
        )
    )

    assert tool.calls == [{}]
    recovered = next(
        event
        for event in recovery_events
        if event.type == EventType.TOOL_CALL_FAILED and event.payload.get("manual_recovery") is True
    )
    assert recovered.payload["result"]["is_error"] is True
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    transcript = asyncio.run(store.load_transcript(session_id))
    recovered_result = transcript[2].content[0]
    assert recovered_result.is_error is True
    assert recovered_result.content == "side effect confirmed absent downstream"


def test_cayu_app_recover_tool_round_rejects_invalid_targets():
    from cayu.runtime.tool_effects import ToolEffectConflict

    session_id = "sess_tool_round_manual_invalid"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    def recover(**overrides):
        request_kwargs = {
            "session_id": session_id,
            "round_id": round_id,
            "tool_call_id": "call_1",
            "outcome": ToolApprovalRecoveryOutcome.COMPLETED,
            "message": "verified",
        }
        request_kwargs.update(overrides)
        return asyncio.run(
            collect_tool_round_recovery_events(app, ToolRoundRecoveryRequest(**request_kwargs))
        )

    with pytest.raises(KeyError, match="Session not found"):
        recover(session_id="sess_missing")
    with pytest.raises(ValueError, match="does not match pending round"):
        recover(round_id="round_other")
    with pytest.raises(ValueError, match="not part of the paused round"):
        recover(tool_call_id="call_other")

    # Every rejection leaves the crashed status untouched and the checkpoint intact.
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.FAILED
    assert asyncio.run(store.load_checkpoint(session_id)) == checkpoint

    # Even the correct target cannot replace an already selected EXTERNAL
    # result with an operator declaration. Its existing evidence owns recovery.
    with pytest.raises(ToolEffectConflict, match="cannot replace durable external-effect"):
        recover()
    assert not any(
        event.payload.get("manual_recovery") is True
        for event in asyncio.run(store.load_events(session_id))
    )
    assert tool.calls == [{}]

    # A plain resume auto-repairs the round first; manual recovery then has no target.
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "continue")]),
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    with pytest.raises(RuntimeError, match="no pending tool round"):
        recover()
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_rejects_never_started_call():
    store = FailingAfterPendingToolRoundCheckpointStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_manual_not_started",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_manual_not_started"))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    assert tool.calls == []

    with pytest.raises(RuntimeError, match="requires a recorded tool.call.started"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id="sess_tool_round_manual_not_started",
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified",
                ),
            )
        )
    session = asyncio.run(store.load("sess_tool_round_manual_not_started"))
    assert session is not None and session.status == SessionStatus.FAILED
    assert tool.calls == []


def test_cayu_app_recover_tool_round_rejects_concurrent_recovery():
    session_id = "sess_tool_round_manual_concurrent"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    def recovery_request(message: str) -> ToolRoundRecoveryRequest:
        return ToolRoundRecoveryRequest(
            session_id=session_id,
            round_id=round_id,
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message=message,
        )

    async def run() -> None:
        first = app.recover_tool_round(recovery_request("verified externally"))
        # Drive the first recovery past its claim: the first yielded event proves
        # the status transition and the in-process registration happened.
        assert await anext(first) is not None

        second = app.recover_tool_round(recovery_request("duplicate attempt"))
        with pytest.raises(RuntimeError, match="active work in this process"):
            await anext(second)

        remaining = [event async for event in first]
        assert remaining[-1].type == EventType.SESSION_COMPLETED

    asyncio.run(run())
    # The duplicate attempt emitted nothing: exactly one terminal event exists for
    # the call, and the tool itself only ever ran once (before the crash).
    events = asyncio.run(store.load_events(session_id))
    manual_terminals = [
        event
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    ]
    assert len(manual_terminals) == 1
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_serializes_across_apps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "sess_tool_round_manual_cross_worker"
    current_time = {"value": datetime(2026, 7, 19, tzinfo=UTC)}

    def clock() -> datetime:
        return current_time["value"]

    monkeypatch.setattr(
        recovery_ownership_module,
        "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
        0.01,
    )
    store = BlockingPostPersistEventsLoadStore(ownership_clock=clock)
    app_a, store, original_tool, checkpoint = _crashed_tool_round_app(
        session_id,
        store=store,
        clock=clock,
        tool=PureRecordingTool(),
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    provider_b = VersionedFakeProvider([])
    tool_b = PureRecordingTool()
    app_b = CayuApp(session_store=store, enable_logging=False, clock=clock)
    app_b.register_provider(provider_b, default=True)
    app_b.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool_b],
    )

    async def run() -> None:
        first = asyncio.create_task(
            collect_tool_round_recovery_events(
                app_a,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified externally",
                ),
            )
        )
        try:
            await asyncio.wait_for(store.post_persist_load_started.wait(), timeout=10)
            claimed_checkpoint = await store.load_checkpoint(session_id)
            assert claimed_checkpoint is not None
            claim_marker = claimed_checkpoint["incomplete_session_recovery_claim"]
            assert claim_marker["operation"] == "manual_tool_round_recovery"
            assert claim_marker["tool_round_id"] == round_id
            assert claim_marker["tool_call_id"] == "call_1"
            current_time["value"] += timedelta(minutes=4)
            for _attempt in range(100):
                claimed_checkpoint = await store.load_checkpoint(session_id)
                assert claimed_checkpoint is not None
                claim_marker = claimed_checkpoint["incomplete_session_recovery_claim"]
                if claim_marker.get("renewed_at") == clock().isoformat():
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Manual recovery claim heartbeat did not renew its lease.")
            current_time["value"] += timedelta(minutes=2)
            with pytest.raises(
                RuntimeError,
                match="active incomplete-session recovery operation",
            ):
                await collect_tool_round_recovery_events(
                    app_b,
                    ToolRoundRecoveryRequest(
                        session_id=session_id,
                        round_id=round_id,
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="duplicate recovery",
                    ),
                )
        finally:
            store.allow_post_persist_load.set()

        first_events = await asyncio.wait_for(first, timeout=5)
        assert first_events[-1].type == EventType.SESSION_COMPLETED
        persisted = await store.load_events(session_id)
        manual_terminals = [
            event
            for event in persisted
            if event.type == EventType.TOOL_CALL_COMPLETED
            and event.payload.get("manual_recovery") is True
        ]
        assert len(manual_terminals) == 1
        assert original_tool.calls == [{}]
        assert tool_b.calls == []
        assert provider_b.requests == []
        assert_only_model_step_publication_checkpoint(await store.load_checkpoint(session_id))

    asyncio.run(run())


def test_cayu_app_recover_tool_round_claims_before_status_transition():
    class PausingFirstRecoveryTransitionStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.pause_recovery_transition = False
            self.recovery_transition_started = asyncio.Event()
            self.allow_recovery_transition = asyncio.Event()
            self.recovery_transition_count = 0

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            if self.pause_recovery_transition and to_status == SessionStatus.RUNNING:
                self.recovery_transition_count += 1
                if self.recovery_transition_count == 1:
                    self.recovery_transition_started.set()
                    await self.allow_recovery_transition.wait()
            return await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )

    session_id = "sess_tool_round_manual_claim_race"
    store = PausingFirstRecoveryTransitionStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]
    store.pause_recovery_transition = True

    def recovery_request(message: str) -> ToolRoundRecoveryRequest:
        return ToolRoundRecoveryRequest(
            session_id=session_id,
            round_id=round_id,
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message=message,
        )

    async def run() -> None:
        first = asyncio.create_task(
            collect_tool_round_recovery_events(
                app,
                recovery_request("verified externally"),
            )
        )
        await store.recovery_transition_started.wait()

        with pytest.raises(RuntimeError, match="active work in this process"):
            await collect_tool_round_recovery_events(
                app,
                recovery_request("duplicate attempt"),
            )
        assert store.recovery_transition_count == 1

        store.allow_recovery_transition.set()
        first_events = await first
        assert first_events[-1].type == EventType.SESSION_COMPLETED

    asyncio.run(run())
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.COMPLETED
    assert tool.calls == [{}]


@pytest.mark.parametrize(
    ("same_process_operator", "finalize_before_claim", "lose_fence_ack"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
    ids=[
        "cross-worker",
        "same-process",
        "terminal-before-claim",
        "interruption-fence-ack-lost",
    ],
)
def test_operator_interrupt_wins_race_with_manual_tool_round_recovery_claim(
    same_process_operator: bool,
    finalize_before_claim: bool,
    lose_fence_ack: bool,
):
    class PausingManualRecoveryClaimStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.pause_manual_claim = False
            self.manual_claim_started = asyncio.Event()
            self.allow_manual_claim = asyncio.Event()
            self.operator_interrupt_committed = asyncio.Event()
            self.hold_operator_interrupt_return = False
            self.allow_operator_interrupt_return = asyncio.Event()
            self.lose_fence_ack = False
            self.fence_ack_lost = False

        async def reserve_stalled_run_recovery(
            self, session_id, *, statuses, inactive_for_seconds, checkpoint_transform
        ):
            if self.pause_manual_claim:
                self.pause_manual_claim = False
                self.manual_claim_started.set()
                await self.allow_manual_claim.wait()
            return await super().reserve_stalled_run_recovery(
                session_id,
                statuses=statuses,
                inactive_for_seconds=inactive_for_seconds,
                checkpoint_transform=checkpoint_transform,
            )

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            if to_status == SessionStatus.INTERRUPTING:
                self.operator_interrupt_committed.set()
                if self.hold_operator_interrupt_return:
                    self.hold_operator_interrupt_return = False
                    await self.allow_operator_interrupt_return.wait()
            return session

        async def fence_run_and_transform_checkpoint(self, *args, **kwargs) -> Session:
            session = await super().fence_run_and_transform_checkpoint(*args, **kwargs)
            checkpoint = await self.load_checkpoint(session.id)
            marker = (
                None if checkpoint is None else checkpoint.get("incomplete_session_recovery_claim")
            )
            if (
                self.lose_fence_ack
                and not self.fence_ack_lost
                and type(marker) is dict
                and marker.get("operation") == "manual_tool_round_interruption_fence"
            ):
                self.fence_ack_lost = True
                raise RuntimeError("interrupted manual recovery fence acknowledgement lost")
            return session

    session_id = "sess_tool_round_manual_claim_operator_interrupt"
    store = PausingManualRecoveryClaimStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    operator_app = app
    if not same_process_operator:
        operator_app = CayuApp(session_store=store, enable_logging=False)
        operator_app.register_provider(FakeProvider([]), default=True)
        operator_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )
    active_profile = execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None

    async def run() -> None:
        # Model a live worker that crashed after its tool call became durable,
        # then pause recovery exactly before its atomic storage claim.
        current = await store.load(session_id)
        assert current is not None
        await rebind_test_invocation(
            store,
            session_id,
            target_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, current_checkpoint: (
                execution_profiles_module.checkpoint_with_active_invocation_execution_profile(
                    current_checkpoint,
                    session_id=session.id,
                    interaction_id=active_profile.interaction_id,
                    run_epoch=session.run_epoch + 1,
                    profile=active_profile.profile,
                    expected=active_profile,
                )
            ),
        )
        await store.release_run_fence(session_id)
        store.pause_manual_claim = True
        store.hold_operator_interrupt_return = lose_fence_ack
        recovery_task = asyncio.create_task(collect_tool_round_recovery_events(app, request))
        await asyncio.wait_for(store.manual_claim_started.wait(), timeout=15)

        interrupt_task = asyncio.create_task(
            collect_interrupt_events(
                operator_app,
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator won recovery claim race",
                    metadata={"source": "cross-worker-claim-regression"},
                ),
            )
        )
        await asyncio.wait_for(store.operator_interrupt_committed.wait(), timeout=15)
        if finalize_before_claim:
            await operator_app._session_finalization.finalize_abandoned_session_by_id(session_id)
            interruption_events = await asyncio.wait_for(interrupt_task, timeout=15)
            completed_stop = await store.load(session_id)
            assert recovery_task.done() is False
        store.lose_fence_ack = lose_fence_ack
        store.allow_manual_claim.set()

        if same_process_operator:
            interruption_events = await asyncio.wait_for(interrupt_task, timeout=15)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(recovery_task, timeout=15)
        elif lose_fence_ack:
            recovery_events = await asyncio.wait_for(recovery_task, timeout=15)
            store.allow_operator_interrupt_return.set()
            interruption_events = await asyncio.wait_for(interrupt_task, timeout=15)
            assert store.fence_ack_lost is True
            assert recovery_events[-1].id == interruption_events[-1].id
        else:
            recovery_events = await asyncio.wait_for(recovery_task, timeout=15)
            if not finalize_before_claim:
                interruption_events = await asyncio.wait_for(interrupt_task, timeout=15)
            assert recovery_events[-1].id == interruption_events[-1].id
        assert interruption_events[-1].type == EventType.SESSION_INTERRUPTED
        assert interruption_events[-1].payload["interruption_type"] == "operator_requested"
        assert interruption_events[-1].payload["reason"] == ("operator won recovery claim race")
        assert interruption_events[-1].payload["metadata"] == {
            "source": "cross-worker-claim-regression"
        }

        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        if finalize_before_claim:
            assert completed_stop is not None
            assert interrupted.run_epoch == completed_stop.run_epoch
        interrupted_checkpoint = await store.load_checkpoint(session_id)
        assert interrupted_checkpoint is not None
        assert interrupted_checkpoint["pending_tool_round"]["tool_round_id"] == request.round_id
        assert "incomplete_session_recovery_claim" not in interrupted_checkpoint
        # Both application instances observe the same terminal interruption and
        # may race to claim its durable cascade. Draining only the operator's
        # coordinator is insufficient when the recovery worker won that claim.
        assert await operator_app.drain_background_interruptions(timeout_s=1) is True
        if app is not operator_app:
            assert await app.drain_background_interruptions(timeout_s=1) is True
        drained_checkpoint = await store.load_checkpoint(session_id)
        assert drained_checkpoint is not None
        assert "pending_interruption_cascade" not in drained_checkpoint
        persisted = await store.load_events(session_id)
        terminal_interruptions = [
            event for event in persisted if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_interruptions) == 1
        assert not any(event.payload.get("manual_recovery") is True for event in persisted)
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False
        assert operator_app._session_control.has_active_tasks(session_id) is False

        # The operator stop prevents this attempt from applying the supplied
        # outcome; an explicit later retry can still recover the intact round.
        recovered = await collect_tool_round_recovery_events(app, request)
        assert recovered[-1].type == EventType.SESSION_COMPLETED
        assert_only_model_step_publication_checkpoint(await store.load_checkpoint(session_id))
        assert tool.calls == [{}]

    asyncio.run(run())


@pytest.mark.parametrize(
    "lose_fence_ack",
    [False, True],
    ids=["fence-confirmed", "fence-acknowledgement-lost"],
)
def test_manual_tool_round_recovery_finalizes_pending_operator_interrupt_before_claim(
    lose_fence_ack: bool,
):
    class PendingOperatorInterruptionStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lose_fence_ack = False
            self.fence_ack_lost = False

        async def fence_run_and_transform_checkpoint(self, *args, **kwargs) -> Session:
            fenced = await super().fence_run_and_transform_checkpoint(*args, **kwargs)
            if self.lose_fence_ack:
                self.lose_fence_ack = False
                self.fence_ack_lost = True
                raise RuntimeError("interrupted manual recovery fence acknowledgement lost")
            return fenced

    session_id = "sess_tool_round_manual_pending_operator_interrupt"
    store = PendingOperatorInterruptionStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )
    interruption_request_id = "pending-operator-interruption"
    interrupt_payload = {
        "reason": "operator interruption lost its terminal acknowledgement",
        "metadata": {"source": "pending-marker-regression"},
        "requested_by": None,
        "interruption_type": "operator_requested",
        "interruption_request_id": interruption_request_id,
    }
    active_profile = execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None

    async def run() -> None:
        # Model an operator interruption that durably changed status and wrote
        # its checkpoint marker, but crashed before its terminal event append.
        current = await store.load(session_id)
        assert current is not None
        await rebind_test_invocation(
            store,
            session_id,
            target_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, current_checkpoint: (
                execution_profiles_module.checkpoint_with_active_invocation_execution_profile(
                    current_checkpoint,
                    session_id=session.id,
                    interaction_id=active_profile.interaction_id,
                    run_epoch=session.run_epoch + 1,
                    profile=active_profile.profile,
                    expected=active_profile,
                )
            ),
        )
        await store.release_run_fence(session_id)
        await store.transition_status_and_checkpoint(
            session_id,
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.INTERRUPTING,
            checkpoint_transform=session_finalization_module._checkpoint_with_pending_session_interrupt(
                interrupt_payload,
            ),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        store.lose_fence_ack = lose_fence_ack
        interruption_events = await collect_tool_round_recovery_events(app, request)
        assert store.fence_ack_lost is lose_fence_ack
        assert interruption_events[-1].type == EventType.SESSION_INTERRUPTED
        assert interruption_events[-1].payload["interruption_type"] == "operator_requested"
        assert interruption_events[-1].payload["interruption_request_id"] == PRIVATE_EVENT_AUTHORITY
        assert interruption_events[-1].payload["reason"] == interrupt_payload["reason"]
        assert interruption_events[-1].payload["metadata"] == interrupt_payload["metadata"]
        assert await app.drain_background_interruptions(timeout_s=1) is True

        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        interrupted_checkpoint = await store.load_checkpoint(session_id)
        assert interrupted_checkpoint is not None
        assert interrupted_checkpoint["pending_tool_round"]["tool_round_id"] == request.round_id
        assert "pending_session_interrupt" not in interrupted_checkpoint
        assert "pending_interruption_cascade" not in interrupted_checkpoint
        assert "incomplete_session_recovery_claim" not in interrupted_checkpoint
        persisted = await store.load_events(session_id)
        operator_terminals = [
            event
            for event in persisted
            if event.type == EventType.SESSION_INTERRUPTED
            and event.payload.get("interruption_request_id") == interruption_request_id
        ]
        assert len(operator_terminals) == 1
        assert not any(event.payload.get("manual_recovery") is True for event in persisted)
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

        # Finalizing the operator stop leaves the intact round available for a
        # deliberate later retry without executing the original tool again.
        recovered = await collect_tool_round_recovery_events(app, request)
        assert recovered[-1].type == EventType.SESSION_COMPLETED
        assert_only_model_step_publication_checkpoint(await store.load_checkpoint(session_id))
        assert tool.calls == [{}]

    asyncio.run(run())


def test_manual_tool_round_recovery_rejects_pending_interruption_cascade_atomically():
    session_id = "sess_tool_round_manual_pending_interruption_cascade"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )
    cascade_marker = {
        "attempt_id": "pending-cascade-attempt",
        "interrupt_payload": {"interruption_type": "operator_requested"},
    }

    async def run() -> None:
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        def add_cascade(
            _session: Session,
            current: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = {} if current is None else dict(current)
            updated["pending_interruption_cascade"] = cascade_marker
            return updated

        await store.transform_checkpoint(session_id, add_cascade)
        before = await store.load(session_id)
        checkpoint_before = await store.load_checkpoint(session_id)
        events_before = await store.load_events(session_id)

        with pytest.raises(
            RuntimeError,
            match="incomplete background interruption cascade",
        ):
            await collect_tool_round_recovery_events(app, request)

        after = await store.load(session_id)
        assert before is not None and after is not None
        assert after.status == SessionStatus.INTERRUPTED
        assert after.run_epoch == before.run_epoch
        assert await store.load_checkpoint(session_id) == checkpoint_before
        assert await store.load_events(session_id) == events_before
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

        def remove_cascade(
            _session: Session,
            current: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = {} if current is None else dict(current)
            updated.pop("pending_interruption_cascade", None)
            return updated

        await store.transform_checkpoint(session_id, remove_cascade)
        recovered = await collect_tool_round_recovery_events(app, request)
        assert recovered[-1].type == EventType.SESSION_COMPLETED
        assert_only_model_step_publication_checkpoint(await store.load_checkpoint(session_id))
        assert tool.calls == [{}]

    asyncio.run(run())


def test_cayu_app_recover_tool_round_post_persist_failure_closes_to_interrupted():
    session_id = "sess_tool_round_manual_post_persist"
    store = FailingPostPersistEventsLoadStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    store.arm_post_persist_load_failure = True
    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="verified externally",
            ),
        )
    )

    # The failure hit AFTER the terminal event persisted: instead of rolling back
    # to the crashed status (which can be non-resumable), the session closes to
    # INTERRUPTED with the failure recorded on the terminal event.
    interrupted = recovery_events[-1]
    assert interrupted.type == EventType.SESSION_INTERRUPTED
    durable_interrupted = asyncio.run(
        _private_events_for_public_events(store, session_id, [interrupted])
    )[0]
    assert durable_interrupted.payload["tool_round_id"] == round_id
    assert durable_interrupted.payload["tool_call_id"] == "call_1"
    assert interrupted.payload["error_type"] == "RuntimeError"
    assert "events unavailable" in interrupted.payload["error"]
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    checkpoint_after = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint_after is not None and "pending_tool_round" in checkpoint_after

    # Re-targeting the same call is rejected (evidence is durable) ...
    with pytest.raises(RuntimeError, match="Resume the session"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="again",
                ),
            )
        )
    # ... and resume(), as the docstring prescribes, finishes the round from the
    # persisted operator outcome without re-running the tool.
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "continue")]),
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{}]
    transcript = asyncio.run(store.load_transcript(session_id))
    recovered_result = next(
        part
        for message in transcript
        for part in message.content
        if type(part).__name__ == "ToolResultPart"
    )
    assert recovered_result.content == "verified externally"
    assert recovered_result.is_error is False


@pytest.mark.parametrize(
    "stale_status",
    [SessionStatus.RUNNING, SessionStatus.INTERRUPTING],
)
def test_cayu_app_recover_tool_round_closes_stale_live_claim_failure_to_interrupted(
    stale_status: SessionStatus,
) -> None:
    session_id = f"sess_tool_round_manual_stale_live_claim_{stale_status}"
    app, store, original_tool, checkpoint = _crashed_tool_round_app(
        session_id, tool=PureRecordingTool()
    )
    del app
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    provider = VersionedFakeProvider(
        [
            ModelStreamEvent.text_delta("continued from durable recovery"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    recovery_app = CayuApp(session_store=store, enable_logging=False)
    recovery_app.register_provider(provider, default=True)
    recovery_tool = PureRecordingTool()
    recovery_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[recovery_tool],
    )

    async def run() -> None:
        now = datetime.now(UTC)
        stale_owner_ready = asyncio.Event()
        release_stale_owner = asyncio.Event()
        interaction_id = next(
            event.interaction_id
            for event in await store.load_events(session_id)
            if event.type == EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None

        def install_expired_claim(
            current_session: Session,
            current_checkpoint: dict | None,
        ) -> dict:
            current_profile = (
                execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
                    current_checkpoint
                )
            )
            assert current_profile is not None
            updated = execution_profiles_module.checkpoint_with_active_invocation_execution_profile(
                current_checkpoint,
                session_id=current_session.id,
                interaction_id=current_profile.interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=current_profile.profile,
                expected=current_profile,
            )
            updated["incomplete_session_recovery_claim"] = {
                "version": 1,
                "claim_id": "crashed-manual-recovery",
                "claimed_at": (now - timedelta(minutes=10)).isoformat(),
                "claim_expires_at": (now - timedelta(minutes=5)).isoformat(),
                "operation": "manual_tool_round_recovery",
                "tool_round_id": round_id,
                "tool_call_id": "call_1",
            }
            if stale_status == SessionStatus.INTERRUPTING:
                updated["pending_session_interrupt"] = {
                    "interruption_type": "operator_requested",
                    "reason": "operator stopped stale recovery",
                    "metadata": {"source": "stale-live-regression"},
                }
            return updated

        async def stale_owner() -> None:
            await rebind_test_invocation(
                store,
                session_id,
                target_status=SessionStatus.RUNNING,
                checkpoint_transform=install_expired_claim,
            )
            if stale_status == SessionStatus.INTERRUPTING:
                await store.transition_status(
                    session_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                    agent_name="assistant",
                    tool_name="side_effect",
                    payload={
                        "model_step_id": checkpoint["pending_tool_round"]["model_step_id"],
                        "model_attempt_id": checkpoint["pending_tool_round"]["model_attempt_id"],
                        "tool_round_id": round_id,
                        "tool_call_id": "call_1",
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session_id,
                            tool_round_id=round_id,
                            tool_call_id="call_1",
                        ),
                        "manual_recovery": True,
                        "arguments_state": "unavailable",
                        "result": ToolResult(content="verified externally").model_dump(),
                    },
                ),
            )
            stale_owner_ready.set()
            await release_stale_owner.wait()
            try:
                with pytest.raises(SessionRunFenced):
                    await store.update_metadata(session_id, {"stale_owner_write": True})
            finally:
                await store.release_run_fence(session_id)

        stale_owner_task = asyncio.create_task(stale_owner())
        await asyncio.wait_for(stale_owner_ready.wait(), timeout=10)
        try:
            recovery_events = await collect_tool_round_recovery_events(
                recovery_app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="must not replace durable evidence",
                ),
            )
        finally:
            release_stale_owner.set()
            await asyncio.wait_for(stale_owner_task, timeout=5)
        assert recovery_events[-1].type == EventType.SESSION_INTERRUPTED
        if stale_status == SessionStatus.INTERRUPTING:
            assert recovery_events[-1].payload["reason"] == "operator stopped stale recovery"
            assert recovery_events[-1].payload["metadata"] == {"source": "stale-live-regression"}
        else:
            assert recovery_events[-1].payload["manual_recovery_stale_live_failure"] is True
        interrupted = await store.load(session_id)
        assert interrupted is not None
        assert interrupted.status == SessionStatus.INTERRUPTED

        resume_events = await collect_resume_events(
            recovery_app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
        assert resume_events[-1].type == EventType.SESSION_COMPLETED
        manual_terminals = [
            event
            for event in await store.load_events(session_id)
            if event.type == EventType.TOOL_CALL_COMPLETED
            and event.payload.get("manual_recovery") is True
        ]
        assert len(manual_terminals) == 1
        assert original_tool.calls == [{}]
        assert recovery_tool.calls == []
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_cayu_app_recover_tool_round_rejects_structured_output_profile_drift():
    session_id = "sess_tool_round_manual_preflight"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified",
                    structured_output=StructuredOutputSpec(
                        json_schema={"type": "object", "properties": {}},
                        strategy="native",
                    ),
                ),
            )
        )
    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
    )
    # Profile admission fires before provider preflight or any status transition.
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.FAILED
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_rejects_secret_structured_output_preflight():
    secret = "tool-round-recovery-schema-secret-canary"
    session_id = "sess_tool_round_secret_preflight"
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        secret_redactor=SecretRedactor(secret),
    )

    with pytest.raises(ValueError, match="workload secret"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified",
                    structured_output=StructuredOutputSpec(
                        json_schema={"type": "string", "const": secret},
                    ),
                ),
            )
        )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.FAILED
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_multi_call_recovers_iteratively():
    session_id = "sess_tool_round_manual_multi"
    store = FailingAllTerminalToolEventStore()
    tool = PureBarrierRecordingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={"n": 1}),
                ModelStreamEvent.tool_call(id="call_2", name="side_effect", arguments={"n": 2}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]
    assert len(tool.calls) == 2
    started_ids = {
        event.payload["tool_call_id"]
        for event in asyncio.run(store.load_events(session_id))
        if event.type == EventType.TOOL_CALL_STARTED
    }
    assert started_ids == {"call_1", "call_2"}
    store.failing = False

    async def recover_remaining_calls() -> list[Event]:
        first_recovery = await collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="call_1 verified externally",
            ),
        )
        # One call per invocation: call_2 also started without a terminal event,
        # so the session stops INTERRUPTED naming it instead of closing the round.
        assert first_recovery[-1].type == EventType.SESSION_INTERRUPTED
        assert first_recovery[-1].payload["manual_recovery_required"] is True
        durable_first_terminal = (
            await _private_events_for_public_events(store, session_id, [first_recovery[-1]])
        )[0]
        assert durable_first_terminal.payload["tool_round_id"] == round_id
        assert durable_first_terminal.payload["tool_call_id"] == "call_2"
        assert first_recovery[-1].payload["tool_name"] == "side_effect"
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        checkpoint_after_first = await store.load_checkpoint(session_id)
        assert checkpoint_after_first is not None
        assert "pending_tool_round" in checkpoint_after_first
        assert invocation_scope._current_session_run_epoch(session_id) is None

        # The already-recovered call now has a terminal event — re-targeting rejects.
        with pytest.raises(RuntimeError, match="already has a terminal event"):
            await collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="again",
                ),
            )

        second_recovery = await collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_2",
                outcome=ToolApprovalRecoveryOutcome.FAILED,
                message="call_2 confirmed failed externally",
            ),
        )
        assert invocation_scope._current_session_run_epoch(session_id) is None
        return second_recovery

    second_recovery = asyncio.run(recover_remaining_calls())
    assert second_recovery[-1].type == EventType.SESSION_COMPLETED
    # Neither tool re-ran during either recovery.
    assert len(tool.calls) == 2
    assert_only_model_step_publication_checkpoint(asyncio.run(store.load_checkpoint(session_id)))

    transcript = asyncio.run(store.load_transcript(session_id))
    tool_message = transcript[2]
    results_by_id = {part.tool_call_id: part for part in tool_message.content}
    assert results_by_id["call_1"].content == "call_1 verified externally"
    assert results_by_id["call_1"].is_error is False
    assert results_by_id["call_2"].content == "call_2 confirmed failed externally"
    assert results_by_id["call_2"].is_error is True


def test_cayu_app_recovery_watcher_ignores_its_owned_interrupted_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery_ownership_module,
        "_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    session_id = "sess_tool_round_manual_owned_interrupted_transition"
    store = FailingAllTerminalToolEventStore()
    tool = PureBarrierRecordingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={"n": 1}),
                ModelStreamEvent.tool_call(id="call_2", name="side_effect", arguments={"n": 2}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="call_1 verified externally",
    )
    store.failing = False

    async def scenario() -> None:
        terminal_started = asyncio.Event()
        release_terminal = asyncio.Event()
        original_terminal_stream = app._terminal_event_publication.publish_recovered

        async def block_owned_terminal(request):
            if (
                request.event.type == EventType.SESSION_INTERRUPTED
                and request.event.payload.get("manual_recovery_required") is True
            ):
                terminal_started.set()
                await release_terminal.wait()
            async for event in original_terminal_stream(request):
                yield event

        monkeypatch.setattr(
            app._terminal_event_publication,
            "publish_recovered",
            block_owned_terminal,
        )
        recovery_task = asyncio.create_task(collect_tool_round_recovery_events(app, request))
        try:
            await asyncio.wait_for(terminal_started.wait(), timeout=10)
            await asyncio.sleep(0.05)
            assert recovery_task.done() is False
            interrupted = await store.load(session_id)
            assert interrupted is not None
            assert interrupted.status == SessionStatus.INTERRUPTED
            assert not any(
                event.type == EventType.SESSION_INTERRUPTED
                for event in await store.load_events(session_id)
            )

            release_terminal.set()
            recovery_events = await asyncio.wait_for(recovery_task, timeout=5)
        finally:
            release_terminal.set()
            if not recovery_task.done():
                recovery_task.cancel()
                await asyncio.gather(recovery_task, return_exceptions=True)

        assert recovery_events[-1].type == EventType.SESSION_INTERRUPTED
        assert recovery_events[-1].payload["interruption_type"] == "runtime_interrupted"
        assert recovery_events[-1].payload["manual_recovery_required"] is True
        assert recovery_events[-1].payload["tool_call_id"] == (
            f"{recovery_events[-1].id}:tool_call_id"
        )
        terminal_interruptions = [
            event
            for event in await store.load_events(session_id)
            if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_interruptions) == 1
        records = await store.query_events(EventQuery(session_id=session_id, limit=100))
        assert (
            _private_record_for_public_event(records, recovery_events[-1]).event.id
            == terminal_interruptions[0].id
        )
        assert len(tool.calls) == 2
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())
