"""Approval recovery contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingSecondTerminalToolEventStore,
    FailingTerminalToolEventStore,
)
from tests.runtime.runtime_session_support import (
    _AllowForkProfileAdoption,
    _approval_pause_app,
    _current_fork_profile_selection,
    _pending_tool_approval_from_public_event,
    _private_events_for_public_events,
    _private_record_for_public_event,
    _reviewer_actor,
    collect_events,
    collect_fork_events,
    collect_resume_events,
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
    collect_tool_round_recovery_events,
    transition_runtime_session_to_running,
)
from tests.runtime.runtime_tool_support import (
    EchoTool,
    ExpiringApprovalPolicy,
    PureRecordingTool,
    RequireApprovalPolicy,
    SideEffectApprovalPolicy,
    SideEffectTool,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.messages import Message, ToolResultPart
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import _tool_execution as tool_execution
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.forks import ForkSystemPromptPolicy
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.tools.base import ToolEffect, ToolResult
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.static import StaticVault


class _BlockingApprovalContinuationProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.continuation_started: asyncio.Event | None = None
        self.never_complete: asyncio.Event | None = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={"value": "secret"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if self.continuation_started is None or self.never_complete is None:
            raise AssertionError("Blocking approval continuation events were not initialized.")
        self.continuation_started.set()
        await self.never_complete.wait()
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


@pytest.mark.parametrize("scope", ["static", "dynamic", "unknown"])
def test_tool_approval_recovery_events_carry_resolved_by_and_expiry_stamp(
    scope: str,
):
    dynamic_scope = scope != "static"
    # Manual recovery for a pure call retains actor/expiry attribution. External
    # calls instead require the registered receipt reconciliation protocol.
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    recovery_message = (
        "approval-recovery-secret-canary" if dynamic_scope else "side effect completed externally"
    )
    store = FailingTerminalToolEventStore()
    tool = PureRecordingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, clock=lambda: clock["now"])
    app.register_provider(provider, default=True)
    if scope == "dynamic":
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"api_key": recovery_message}),
            ),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ExpiringApprovalPolicy(expires_in_seconds=60),
    )

    async def run():
        interrupt_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_actor",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event
            for event in interrupt_events
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        if scope == "unknown":
            checkpoint = await store.load_checkpoint("sess_recovery_actor")
            checkpoint["pending_tool_approval"]["secret_resolution_scope"] = "unknown"
            await store.checkpoint("sess_recovery_actor", checkpoint)

        # In-window approve crashes when the terminal tool event fails to
        # persist; the stream ends interrupted and recovery is required.
        crashed_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_recovery_actor",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed_events[-1].type == EventType.SESSION_INTERRUPTED

        # The operator reconciles the external outcome after the window closed.
        clock["now"] = datetime(2026, 7, 9, 12, 5, tzinfo=UTC)
        releases_before_recovery = store.release_calls
        recovery_events = await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_recovery_actor",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message=recovery_message,
                structured={"provided": recovery_message},
                reason=recovery_message,
                metadata={"provided": recovery_message},
                resolved_by=_reviewer_actor(),
            ),
        )
        return recovery_events, store.release_calls - releases_before_recovery

    recovery_events, release_calls = asyncio.run(run())

    resumed = next(event for event in recovery_events if event.type == EventType.SESSION_RESUMED)
    assert resumed.payload["resolved_by"]["subject"] == "reviewer@example.com"
    assert resumed.payload["expired"] is True
    assert resumed.payload["reason"] == (None if dynamic_scope else recovery_message)
    recovered = next(
        event
        for event in recovery_events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    )
    assert recovered.payload["resolved_by"]["subject"] == "reviewer@example.com"
    assert recovered.payload["resolved_by"]["source"] == "request"
    assert recovered.payload["expired"] is True
    assert recovered.payload["reason"] == (None if dynamic_scope else recovery_message)
    assert recovered.payload["metadata"] == (
        {} if dynamic_scope else {"provided": recovery_message}
    )
    expected_content = (
        "Externally verified tool output is unavailable because the invocation "
        "secret scope could not be reconstructed."
        if dynamic_scope
        else recovery_message
    )
    assert recovered.payload["result"]["content"] == expected_content
    if dynamic_scope:
        assert recovery_message not in repr(recovery_events)
        assert recovery_message not in repr(asyncio.run(store.load_events("sess_recovery_actor")))
        assert recovery_message not in repr(
            asyncio.run(store.load_transcript("sess_recovery_actor"))
        )
        assert recovery_message not in repr(provider.requests[-1].messages)
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in recovery_events)
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    assert release_calls == 1


@pytest.mark.parametrize(
    "approval_arrives_during_claim",
    [False, True],
    ids=["paired-at-preflight", "paired-during-claim"],
)
def test_ordinary_tool_round_recovery_rejects_approval_owned_round_without_mutation(
    approval_arrives_during_claim: bool,
) -> None:
    class BlockingRecoveryClaimStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_manual_recovery_claim = False
            self.manual_recovery_claim_started = asyncio.Event()
            self.allow_manual_recovery_claim = asyncio.Event()

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
            if self.block_manual_recovery_claim and to_status is SessionStatus.RUNNING:
                self.manual_recovery_claim_started.set()
                await self.allow_manual_recovery_claim.wait()
            return await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )

    async def run() -> None:
        race_kind = "claim" if approval_arrives_during_claim else "preflight"
        session_id = f"sess_ordinary_recovery_rejects_approval_{race_kind}"
        store = BlockingRecoveryClaimStore()
        tool = PureRecordingTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval_id"]
        crashed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed[-1].type is EventType.SESSION_INTERRUPTED
        paired_checkpoint = await store.load_checkpoint(session_id)
        assert paired_checkpoint is not None
        assert "pending_tool_approval" in paired_checkpoint
        assert "pending_tool_round" in paired_checkpoint

        recovery_request = ToolRoundRecoveryRequest(
            session_id=session_id,
            round_id=approval_event.payload["tool_round_id"],
            tool_call_id=approval_event.payload["tool_call_id"],
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="side effect completed externally",
        )
        recovery_task: asyncio.Task[list[Event]] | None = None
        if approval_arrives_during_claim:
            ordinary_checkpoint = dict(paired_checkpoint)
            ordinary_checkpoint.pop("pending_tool_approval")
            await store.checkpoint(session_id, ordinary_checkpoint)
            store.block_manual_recovery_claim = True
            recovery_task = asyncio.create_task(
                collect_tool_round_recovery_events(app, recovery_request)
            )
            await asyncio.wait_for(store.manual_recovery_claim_started.wait(), timeout=10)
            await store.checkpoint(session_id, paired_checkpoint)

        session_before = await store.load(session_id)
        checkpoint_before = await store.load_checkpoint(session_id)
        events_before = await store.load_events(session_id)
        transcript_before = await store.load_transcript(session_id)
        if recovery_task is None:
            with pytest.raises(RuntimeError, match="ToolApprovalRecoveryRequest"):
                await collect_tool_round_recovery_events(app, recovery_request)
        else:
            store.allow_manual_recovery_claim.set()
            with pytest.raises(
                RuntimeError,
                match="Invocation rebind source state changed after command preparation",
            ):
                await asyncio.wait_for(recovery_task, timeout=5)

        assert await store.load(session_id) == session_before
        assert await store.load_checkpoint(session_id) == checkpoint_before
        assert await store.load_events(session_id) == events_before
        assert await store.load_transcript(session_id) == transcript_before
        assert app._session_control.has_active_tasks(session_id) is False
        assert tool.calls == [{"value": "secret"}]

        recovered = await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect completed externally",
            ),
        )
        assert recovered[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(run())


def test_tool_approval_recovery_reconciles_ambiguous_append_acknowledgement() -> None:
    class AmbiguousRecoveryAppendStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_recovery_ack = False

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            manual_recovery = any(event.payload.get("manual_recovery") is True for event in events)
            await super().append_events(session_id, events)
            if manual_recovery and not self.failed_recovery_ack:
                self.failed_recovery_ack = True
                raise RuntimeError("approval recovery commit acknowledgement lost")

    async def scenario() -> None:
        session_id = "sess_approval_recovery_ambiguous_append"
        store = AmbiguousRecoveryAppendStore()
        tool = PureRecordingTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in paused if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        crashed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED

        recovery = await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect completed externally",
            ),
        )
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert recovery[-1].type == EventType.SESSION_INTERRUPTED
        assert recovery[-1].payload["manual_recovery_persisted"] is True
        persisted = await store.load_events(session_id)
        recovered = [
            event
            for event in persisted
            if event.payload.get("manual_recovery") is True
            and event.payload.get("tool_call_id") == "call_1"
        ]
        assert len(recovered) == 1

        resumed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "secret"}]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_mode",
    ["ordinary", "handled-prior-cancellation", "active-cancellation"],
)
def test_tool_approval_recovery_post_persist_fanout_failure_stays_resumable(
    failure_mode: str,
) -> None:
    async def scenario() -> None:
        grouped_cancellation = failure_mode != "ordinary"
        session_id = f"sess_approval_recovery_post_persist_failure_{failure_mode}"
        store = FailingTerminalToolEventStore()
        tool = PureRecordingTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in paused if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        crashed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED

        original_fan_out = app._event_writer.fan_out_persisted
        failed = False
        fan_out_failure: BaseException = (
            BaseExceptionGroup(
                "approval recovery fan-out cancelled and failed",
                [asyncio.CancelledError("fan-out cancelled"), RuntimeError("fan-out failed")],
            )
            if grouped_cancellation
            else RuntimeError("approval recovery fan-out unavailable")
        )

        async def fail_recovery_fan_out(events: list[Event]) -> list[Event]:
            nonlocal failed
            if not failed and any(event.payload.get("manual_recovery") is True for event in events):
                failed = True
                if failure_mode == "active-cancellation":
                    task = asyncio.current_task()
                    assert task is not None
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.sleep(0)
                raise fan_out_failure
            return await original_fan_out(events)

        app._event_writer.fan_out_persisted = fail_recovery_fan_out
        recovery_request = ToolApprovalRecoveryRequest(
            session_id=session_id,
            approval_id=approval_id,
            tool_round_id=approval_event.payload["tool_round_id"],
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="side effect completed externally",
        )
        if failure_mode == "handled-prior-cancellation":
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.sleep(0)
        recovery: list[Event] = []
        if grouped_cancellation:
            with pytest.raises(BaseExceptionGroup) as raised:
                await collect_tool_approval_recovery_events(app, recovery_request)
            assert raised.value is fan_out_failure
        else:
            recovery = await collect_tool_approval_recovery_events(app, recovery_request)
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        persisted = await store.load_events(session_id)
        terminal = [event for event in persisted if event.type == EventType.SESSION_INTERRUPTED][-1]
        if failure_mode == "handled-prior-cancellation":
            assert terminal.payload.get("abandoned") is not True
        elif failure_mode == "active-cancellation":
            assert terminal.payload["abandoned"] is True
        else:
            records = await store.query_events(EventQuery(session_id=session_id, limit=100))
            assert _private_record_for_public_event(records, recovery[-1]).event.id == terminal.id
            assert terminal.payload["manual_recovery_persisted"] is True
        assert (
            len(
                [
                    event
                    for event in persisted
                    if event.payload.get("manual_recovery") is True
                    and event.payload.get("tool_call_id") == "call_1"
                ]
            )
            == 1
        )

        resumed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "secret"}]

    asyncio.run(scenario())


def test_tool_approval_recovery_post_persist_cleanup_failure_is_not_suppressed() -> None:
    class FailingRecoveryReleaseStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_next_release = False

        async def release_session_invocation(self, command):
            self.release_calls += 1
            if self.fail_next_release:
                self.fail_next_release = False
                raise RuntimeError("approval recovery run fence release unavailable")
            return await InMemorySessionStore.release_session_invocation(self, command)

    async def scenario() -> None:
        session_id = "sess_approval_recovery_post_persist_cleanup_failure"
        store = FailingRecoveryReleaseStore()
        tool = PureRecordingTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in paused if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        crashed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED

        original_fan_out = app._event_writer.fan_out_persisted
        failed = False

        async def fail_recovery_fan_out(events: list[Event]) -> list[Event]:
            nonlocal failed
            if not failed and any(event.payload.get("manual_recovery") is True for event in events):
                failed = True
                raise RuntimeError("approval recovery fan-out unavailable")
            return await original_fan_out(events)

        app._event_writer.fan_out_persisted = fail_recovery_fan_out
        store.fail_next_release = True
        with pytest.raises(
            RuntimeError,
            match="approval recovery run fence release unavailable",
        ):
            await collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id=session_id,
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect completed externally",
                ),
            )

        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        persisted = await store.load_events(session_id)
        assert persisted[-1].type == EventType.SESSION_INTERRUPTED
        assert persisted[-1].payload["manual_recovery_persisted"] is True
        assert tool.calls == [{"value": "secret"}]

    asyncio.run(scenario())


def test_tool_approval_recovery_closes_continuation_before_aclose_returns():
    async def run() -> tuple[Event, SessionStatus, int, bool]:
        session_id = "sess_approval_recovery_stream_closed"
        store = FailingTerminalToolEventStore()
        app, _provider = _approval_pause_app(
            store=store,
            tool=PureRecordingTool(),
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        crashed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED

        releases_before = store.release_calls
        stream = app.recover_tool_approval(
            ToolApprovalRecoveryRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect completed externally",
            )
        )
        while True:
            boundary_event = await anext(stream)
            if boundary_event.type == EventType.MODEL_STARTED:
                break
        assert app._session_control.has_active_tasks(session_id) is True
        await stream.aclose()
        release_delta = store.release_calls - releases_before
        has_active_tasks = app._session_control.has_active_tasks(session_id)
        session = await store.load(session_id)
        assert session is not None
        return boundary_event, session.status, release_delta, has_active_tasks

    boundary_event, status, release_delta, has_active_tasks = asyncio.run(run())

    assert boundary_event.type == EventType.MODEL_STARTED
    assert status == SessionStatus.INTERRUPTED
    assert release_delta == 1
    assert has_active_tasks is False


def test_tool_approval_recovery_task_cancellation_finalizes_continuation():
    async def run() -> None:
        session_id = "sess_approval_recovery_task_cancelled"
        store = FailingTerminalToolEventStore()
        provider = _BlockingApprovalContinuationProvider()
        provider.continuation_started = asyncio.Event()
        provider.never_complete = asyncio.Event()
        app = CayuApp(session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[PureRecordingTool()],
            tool_policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        crashed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED

        releases_before = store.release_calls
        recovery_task = asyncio.create_task(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id=session_id,
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect completed externally",
                ),
            )
        )
        try:
            await asyncio.wait_for(provider.continuation_started.wait(), timeout=10)
        except BaseException:
            recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)
            raise
        assert recovery_task.cancelling() == 0
        recovery_task.cancel("cancel tool approval recovery")
        assert recovery_task.cancelling() == 1
        try:
            await recovery_task
        except asyncio.CancelledError as cancellation:
            assert cancellation.args == ("Provider operation cancelled",)
        else:
            pytest.fail("Tool approval recovery did not preserve task cancellation.")

        assert recovery_task.cancelled() is True
        assert recovery_task.cancelling() == 1
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_tool_approval" not in checkpoint
        events = await store.load_events(session_id)
        assert events[-1].type == EventType.SESSION_INTERRUPTED
        assert events[-1].payload["abandoned"] is True
        assert store.release_calls - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(run())


def test_tool_approval_recovery_releases_run_fence_once_when_setup_fails():
    class CountingRecoveryStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_recovery_event_load = False

        async def load_events(self, session_id: str) -> list[Event]:
            if self.fail_recovery_event_load:
                raise RuntimeError("recovery event read unavailable")
            return await super().load_events(session_id)

    async def run() -> int:
        store = CountingRecoveryStore()
        tool = PureRecordingTool()
        app, _provider = _approval_pause_app(
            store=store,
            tool=tool,
            policy=RequireApprovalPolicy(),
        )
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_release_once",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        crashed = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_recovery_release_once",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED

        releases_before_recovery = store.release_calls
        store.fail_recovery_event_load = True
        with pytest.raises(RuntimeError, match="recovery event read unavailable"):
            await collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_recovery_release_once",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="completed externally",
                ),
            )
        return store.release_calls - releases_before_recovery

    assert asyncio.run(run()) == 1


def test_cayu_app_approval_recovery_ignores_unrelated_terminal_tool_events():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_scoped_recovery",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]
    private_approval = asyncio.run(_pending_tool_approval_from_public_event(store, approval_event))
    asyncio.run(
        store.append_event(
            "sess_approval_scoped_recovery",
            Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="sess_approval_scoped_recovery",
                agent_name="assistant",
                tool_name="side_effect",
                payload={
                    "tool_call_id": "call_1",
                    "result": ToolResult(content="unrelated").model_dump(),
                },
            ),
        )
    )

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_scoped_recovery",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    private_resolution = asyncio.run(
        _private_events_for_public_events(store, "sess_approval_scoped_recovery", events)
    )
    assert private_resolution[2].payload["approval_id"] == private_approval.approval_id
    assert private_resolution[3].payload["approval_id"] == private_approval.approval_id
    assert tool.calls == [{"value": "secret"}]
    assert provider.requests[1].messages[-1].content[0].content == "recorded"


def test_cayu_app_recovers_staged_approval_result_without_reexecuting_tool():
    from cayu.runtime.tool_effects import ToolEffectConflict

    store = FailingTerminalToolEventStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered and continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_started_without_terminal",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]
    private_approval = asyncio.run(_pending_tool_approval_from_public_event(store, approval_event))

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_started_without_terminal",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"value": "secret"}]

    with pytest.raises(ToolEffectConflict, match="cannot replace durable external-effect"):
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_approval_started_without_terminal",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="operator replacement",
                    reason="operator claim cannot replace the selected result",
                ),
            )
        )
    assert tool.calls == [{"value": "secret"}]
    assert len(provider.requests) == 1

    recovery_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_started_without_terminal",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in recovery_events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    private_recovery_event = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_approval_started_without_terminal",
            [recovery_events[1]],
        )
    )[0]
    assert private_recovery_event.payload["approval_id"] == private_approval.approval_id
    assert private_recovery_event.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_approval_started_without_terminal",
        tool_round_id=private_recovery_event.payload["tool_round_id"],
        tool_call_id="call_1",
        approval_id=private_approval.approval_id,
    )
    assert "manual_recovery" not in recovery_events[1].payload
    assert recovery_events[1].payload["result"]["content"] == "recorded"
    assert tool.calls == [{"value": "secret"}]
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].content == "recorded"
    session = asyncio.run(store.load("sess_approval_started_without_terminal"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_tool_approval_recovery_can_target_non_gating_call_in_same_round():
    class PureEcho(EchoTool):
        spec = EchoTool.spec.model_copy(update={"effect": ToolEffect.NONE})

    store = FailingSecondTerminalToolEventStore()
    side_effect = PureRecordingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_gating",
                    name="side_effect",
                    arguments={"value": "approved"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_recovery",
                    name="echo",
                    arguments={"text": "outcome unknown"},
                ),
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
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[side_effect, PureEcho()],
        tool_policy=SideEffectApprovalPolicy(),
    )

    async def run() -> tuple[list[Event], list[Event], list[Event]]:
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_non_gating_approval_call",
                messages=[Message.text("user", "run both")],
            ),
        )
        approval_event = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        interrupted = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_recover_non_gating_approval_call",
                approval_id=approval_event.payload["approval_id"],
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        recovered = await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_recover_non_gating_approval_call",
                approval_id=approval_event.payload["approval_id"],
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id="call_recovery",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="echo completed before the crash",
            ),
        )
        return paused, interrupted, recovered

    _paused, interrupted, recovered = asyncio.run(run())
    private_interrupted = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_recover_non_gating_approval_call",
            interrupted,
        )
    )
    private_recovered = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_recover_non_gating_approval_call",
            recovered,
        )
    )

    assert interrupted[-1].type is EventType.SESSION_INTERRUPTED
    assert any(
        event.type is EventType.TOOL_CALL_STARTED
        and event.payload["tool_call_id"] == "call_recovery"
        for event in private_interrupted
    )
    assert not any(
        event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        and event.payload["tool_call_id"] == "call_recovery"
        for event in private_interrupted
    )
    recovered_call = next(
        event for event in private_recovered if event.payload.get("manual_recovery") is True
    )
    assert recovered_call.payload["tool_call_id"] == "call_recovery"
    assert recovered[-1].type is EventType.SESSION_COMPLETED
    assert side_effect.calls == [{"value": "approved"}]


def test_approval_continuation_redacts_metadata_before_next_tool_round_checkpoint():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret = "approval-continuation-metadata-secret"
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_first",
                    name="side_effect",
                    arguments={"value": "first"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_second",
                    name="side_effect",
                    arguments={"value": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run() -> tuple[list[Event], dict[str, Any] | None]:
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_continuation_metadata",
                messages=[Message.text("user", "run twice")],
            ),
        )
        approval_event = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        continued = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_continuation_metadata",
                approval_id=approval_event.payload["approval_id"],
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                metadata={"credential": secret},
            ),
        )
        checkpoint = await store.load_checkpoint("sess_approval_continuation_metadata")
        return continued, checkpoint

    events, checkpoint = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert checkpoint is not None
    assert checkpoint["pending_tool_round"]["request_metadata"] == {"credential": REDACTED_SECRET}
    assert secret not in json.dumps(
        [event.model_dump(mode="json") for event in events],
        sort_keys=True,
    )
    assert tool.calls == [{"value": "first"}]


def test_cayu_app_redacts_manual_tool_approval_recovery_result():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "manual-recovery-secret"
    store = FailingTerminalToolEventStore()
    tool = PureRecordingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered and continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_manual_recovery_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]

    _ = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_manual_recovery_redaction",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    recovery_events = asyncio.run(
        collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_manual_recovery_redaction",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message=f"side effect completed with {secret_value}",
                structured={"token": secret_value},
                artifacts=[{"metadata": {"token": secret_value}}],
                reason=f"operator confirmed {secret_value}",
                metadata={"token": secret_value},
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_manual_recovery_redaction"))

    tool_event = recovery_events[1]
    assert tool_event.type == EventType.TOOL_CALL_COMPLETED
    assert secret_value not in str(tool_event.payload)
    assert tool_event.payload["reason"] == f"operator confirmed {REDACTED_SECRET}"
    assert tool_event.payload["metadata"]["token"] == REDACTED_SECRET
    assert tool_event.payload["result"]["content"] == (
        f"side effect completed with {REDACTED_SECRET}"
    )
    assert tool_event.payload["result"]["structured"]["token"] == REDACTED_SECRET
    assert tool_event.payload["result"]["artifacts"][0]["metadata"]["token"] == REDACTED_SECRET

    tool_message = next(message for message in transcript if message.role == "tool")
    tool_part = tool_message.content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert secret_value not in str(tool_part.model_dump(mode="json"))
    assert tool_part.content == f"side effect completed with {REDACTED_SECRET}"
    assert tool_part.structured == {"token": REDACTED_SECRET}
    assert tool_part.artifacts[0]["metadata"]["token"] == REDACTED_SECRET
    assert provider.requests[1].messages[-1].role == "tool"
    assert secret_value not in str(provider.requests[1].messages[-1].model_dump(mode="json"))


def test_cayu_app_recovery_does_not_append_terminal_event_without_session_claim():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_claim_required",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]
    asyncio.run(
        transition_runtime_session_to_running(
            store,
            "sess_recovery_claim_required",
        )
    )

    with pytest.raises(ValueError, match="status transition not allowed"):
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_recovery_claim_required",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="confirmed externally",
                ),
            )
        )

    events = asyncio.run(store.load_events("sess_recovery_claim_required"))
    assert not any(event.payload.get("manual_recovery") is True for event in events)


def test_cayu_app_rejects_message_resume_with_pending_tool_approval():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_resume_blocked",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    with pytest.raises(RuntimeError, match="pending tool approval"):
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_tool_approval_resume_blocked",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )


def test_cayu_app_rejects_fork_with_pending_approval_checkpoint():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        execution_profile_policy=_AllowForkProfileAdoption(),
    )
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    tool = SideEffectTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupted_fork_source",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]
    source_checkpoint = asyncio.run(store.load_checkpoint("sess_interrupted_fork_source"))
    assert source_checkpoint is not None
    source_checkpoint["pending_tool_approval"]["task_id"] = "source_task"
    asyncio.run(store.checkpoint("sess_interrupted_fork_source", source_checkpoint))

    with pytest.raises(RuntimeError, match="awaiting tool approval cannot be forked"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_interrupted_fork_source",
                    session_id="sess_interrupted_fork_child",
                ),
            )
        )

    assert asyncio.run(store.load("sess_interrupted_fork_child")) is None
    source_checkpoint_after = asyncio.run(store.load_checkpoint("sess_interrupted_fork_source"))
    assert source_checkpoint_after == source_checkpoint

    asyncio.run(store.update_status("sess_interrupted_fork_source", SessionStatus.FAILED))
    with pytest.raises(RuntimeError, match="awaiting tool approval cannot be forked"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_interrupted_fork_source",
                    session_id="sess_failed_fork_without_checkpoint",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection("interrupted-fork-child"),
                ),
            )
        )
    assert asyncio.run(store.load("sess_failed_fork_without_checkpoint")) is None
    assert asyncio.run(store.load_checkpoint("sess_interrupted_fork_source")) == source_checkpoint

    asyncio.run(store.update_status("sess_interrupted_fork_source", SessionStatus.INTERRUPTED))
    with pytest.raises(ValueError, match="without checkpoint state"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_interrupted_fork_source",
                    session_id="sess_interrupted_fork_no_checkpoint",
                    copy_checkpoint=False,
                ),
            )
        )

    source_checkpoint = asyncio.run(store.load_checkpoint("sess_interrupted_fork_source"))
    assert source_checkpoint is not None
    source_checkpoint["pending_tool_approval"].pop("task_id", None)
    asyncio.run(store.checkpoint("sess_interrupted_fork_source", source_checkpoint))
    asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_interrupted_fork_source",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    assert tool.calls == [{"value": "secret"}]
