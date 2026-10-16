"""Supervision events produced by real runtime resolution entrances."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    versioned_test_provider_identity,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingTerminalToolEventStore,
    _crashed_tool_round_app,
)
from tests.runtime.runtime_session_support import (
    _approval_pause_app,
    _pending_tool_approval_from_public_event,
    _reviewer_actor,
    collect_events,
    collect_fork_events,
    collect_interrupt_events,
    collect_resume_events,
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
    collect_tool_round_recovery_events,
    collect_user_input_events,
)
from tests.runtime.runtime_supervision_support import (
    ARGUMENT_CANARY,
    approval_app,
    durable_timestamp,
    pause_for_approval,
    resolve_approval,
    supervision_page,
    two_call_provider,
)
from tests.runtime.runtime_tool_support import (
    ExpiringApprovalPolicy,
    PureRecordingTool,
    RequireApprovalPolicy,
    SideEffectTool,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
)
from cayu.approvals.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import (
    SupervisionEventKind,
    SupervisionEventOutcome,
    SupervisionEventQuery,
    SupervisionEventResultTooLarge,
    SupervisionMessageOrigin,
)
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_baseline_from_session_metadata,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.messaging import (
    EnqueueSessionMessageRequest,
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageDeliveryMode,
    SessionMessageQuery,
)
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import (
    ForkSessionRequest,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
)
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.tools.subagents import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)
from cayu.tools.user_input import UserInputTool

T0 = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


def test_approval_is_one_record_with_actor_reason_step_and_request_time():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=1)

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_approve")
        await collect_tool_approval_events(
            app,
            resolve_approval(
                "sess_sup_approve",
                pending,
                ToolApprovalDecision.APPROVE,
                reason="looks right",
                resolved_by=_reviewer_actor(),
            ),
        )
        return (
            pending,
            await supervision_page(store),
            await durable_timestamp(
                store, "sess_sup_approve", EventType.TOOL_CALL_APPROVAL_REQUESTED
            ),
            await durable_timestamp(store, "sess_sup_approve", EventType.TOOL_CALL_APPROVED),
        )

    pending, page, requested_at, approved_at = asyncio.run(run())

    assert len(page.events) == 1
    event = page.events[0]
    assert event.kind is SupervisionEventKind.APPROVE
    assert event.outcome is SupervisionEventOutcome.APPROVED
    assert event.session_id == "sess_sup_approve"
    assert event.approval_id == pending.approval_id
    assert event.tool_call_ids == (pending.tool_call_id,)
    assert event.tool_round_id == pending.tool_round_id
    assert event.model_step_id == pending.model_step_id
    assert event.model_attempt_id == pending.model_attempt_id
    assert event.tool_name == "side_effect"
    assert event.reason == "looks right"
    assert event.actor == ResolutionActor(
        subject="reviewer@example.com", tenant="acme", source=ResolutionActorSource.REQUEST
    )
    assert event.requested_at == requested_at
    assert event.resolved_at == approved_at
    assert requested_at < approved_at
    assert event.forked is False
    assert event.parent_session_id is None
    assert event.execution_profile_fingerprint is not None
    assert event.content is None
    assert page.has_more is False


def test_denying_a_two_call_round_is_one_record_listing_both_calls():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(two_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_deny")
        await collect_tool_approval_events(
            app,
            resolve_approval(
                "sess_sup_deny",
                pending,
                ToolApprovalDecision.DENY,
                reason="wrong account",
                resolved_by=_reviewer_actor(),
            ),
        )
        return await supervision_page(store)

    page = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.DENY]
    event = page.events[0]
    assert event.outcome is SupervisionEventOutcome.DENIED
    assert set(event.tool_call_ids) == {"call_1", "call_2"}
    assert event.reason == "wrong account"


def test_expiry_is_one_system_record_and_not_a_denial():
    clock = {"now": T0}
    store = InMemorySessionStore()
    app, _ = _approval_pause_app(
        store=store,
        tool=SideEffectTool(),
        policy=ExpiringApprovalPolicy(expires_in_seconds=60),
        clock=lambda: clock["now"],
    )

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_expire")
        clock["now"] = T0 + timedelta(minutes=2)
        await collect_tool_approval_events(
            app,
            resolve_approval(
                "sess_sup_expire",
                pending,
                ToolApprovalDecision.APPROVE,
                resolved_by=_reviewer_actor(),
            ),
        )
        return await supervision_page(store)

    page = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.EXPIRE]
    event = page.events[0]
    assert event.outcome is SupervisionEventOutcome.EXPIRED
    assert event.actor is not None
    assert event.actor.subject == "cayu:approval-expiry"
    assert event.actor.source is ResolutionActorSource.SYSTEM


def test_repeated_exact_resolution_is_counted_once():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=1)

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_retry")
        request = resolve_approval("sess_sup_retry", pending, ToolApprovalDecision.APPROVE)
        await collect_tool_approval_events(app, request)
        await collect_tool_approval_events(app, request)
        return await supervision_page(store)

    page = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.APPROVE]


def test_interrupt_that_closes_a_pending_approval_is_one_interrupt_record():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=1)

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_interrupt")
        await collect_interrupt_events(
            app,
            InterruptSessionRequest(
                session_id="sess_sup_interrupt",
                reason="stop this review",
                requested_by=_reviewer_actor(),
            ),
        )
        return pending, await supervision_page(store)

    pending, page = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.INTERRUPT]
    event = page.events[0]
    assert event.outcome is SupervisionEventOutcome.INTERRUPTED
    assert event.approval_id == pending.approval_id
    assert event.interruption_request_id is not None
    assert event.reason == "stop this review"
    assert event.actor is not None
    assert event.actor.subject == "reviewer@example.com"


def test_recovery_after_a_recorded_approval_is_a_second_touch():
    store = FailingTerminalToolEventStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[PureRecordingTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_recover")
        crashed = await collect_tool_approval_events(
            app, resolve_approval("sess_sup_recover", pending, ToolApprovalDecision.APPROVE)
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED
        await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_sup_recover",
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="done outside",
                reason="verified in the ledger",
                resolved_by=_reviewer_actor(),
            ),
        )
        return pending, await supervision_page(store)

    pending, page = asyncio.run(run())

    assert [event.kind for event in page.events] == [
        SupervisionEventKind.APPROVE,
        SupervisionEventKind.RECOVER,
    ]
    recovered = page.events[1]
    assert recovered.outcome is SupervisionEventOutcome.COMPLETED
    assert recovered.approval_id == pending.approval_id
    assert recovered.tool_call_ids == (pending.tool_call_id,)
    assert recovered.reason == "verified in the ledger"
    assert recovered.actor is not None
    assert recovered.actor.subject == "reviewer@example.com"


def test_content_is_absent_by_default_and_published_content_only_on_request():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1", name="side_effect", arguments={"value": ARGUMENT_CANARY}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_content")
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_content", pending, ToolApprovalDecision.APPROVE)
        )
        return await supervision_page(store), await supervision_page(store, include_content=True)

    default_page, content_page = asyncio.run(run())

    assert ARGUMENT_CANARY not in default_page.model_dump_json()
    assert default_page.events[0].content is None
    assert content_page.events[0].content is not None
    assert ARGUMENT_CANARY in content_page.model_dump_json()


def test_pages_resume_from_the_cursor_without_overlap_or_gaps():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=3)

    async def run():
        sessions = ("sess_sup_page_a", "sess_sup_page_b", "sess_sup_page_c")
        for session_id in sessions:
            pending = await pause_for_approval(app, store, session_id)
            await collect_tool_approval_events(
                app, resolve_approval(session_id, pending, ToolApprovalDecision.APPROVE)
            )
        first = await supervision_page(store, limit=2)
        second = await supervision_page(store, limit=2, cursor=first.next_cursor)
        third = await supervision_page(store, limit=2, cursor=second.next_cursor)
        return sessions, first, second, third

    sessions, first, second, third = asyncio.run(run())

    assert len(first.events) == 2
    assert first.has_more is True
    assert len(second.events) == 1
    assert third.events == ()
    assert third.has_more is False
    seen = [event.session_id for page in (first, second) for event in page.events]
    assert seen == list(sessions)
    sequences = [event.sequence for page in (first, second) for event in page.events]
    assert sequences == sorted(sequences)


def test_window_is_half_open_on_the_resolution_time():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=1)

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_window")
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_window", pending, ToolApprovalDecision.APPROVE)
        )
        approved_at = await durable_timestamp(
            store, "sess_sup_window", EventType.TOOL_CALL_APPROVED
        )
        ending_at = await store.query_supervision_events(
            SupervisionEventQuery(start_at=approved_at - timedelta(hours=1), end_at=approved_at)
        )
        starting_at = await store.query_supervision_events(
            SupervisionEventQuery(start_at=approved_at, end_at=approved_at + timedelta(hours=1))
        )
        return ending_at, starting_at

    ending_at, starting_at = asyncio.run(run())

    assert ending_at.events == ()
    assert [event.kind for event in starting_at.events] == [SupervisionEventKind.APPROVE]


def test_kind_and_actor_source_filters_select_records():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=2)

    async def run():
        approved = await pause_for_approval(app, store, "sess_sup_filter_a")
        await collect_tool_approval_events(
            app,
            resolve_approval(
                "sess_sup_filter_a",
                approved,
                ToolApprovalDecision.APPROVE,
                resolved_by=_reviewer_actor(),
            ),
        )
        denied = await pause_for_approval(app, store, "sess_sup_filter_b")
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_filter_b", denied, ToolApprovalDecision.DENY)
        )
        return (
            await supervision_page(store, kinds=(SupervisionEventKind.DENY,)),
            await supervision_page(store, actor_sources=(ResolutionActorSource.REQUEST,)),
        )

    by_kind, by_source = asyncio.run(run())

    assert [event.session_id for event in by_kind.events] == ["sess_sup_filter_b"]
    assert [event.session_id for event in by_source.events] == ["sess_sup_filter_a"]


def test_answer_is_one_record_with_the_question_time():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask", name="ask_user", arguments={"question": "Which period?"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("thanks"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])

    async def run():
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sup_answer",
                messages=[Message.text("user", "close the month")],
            ),
        )
        awaiting = next(e for e in paused if e.type == EventType.SESSION_AWAITING_USER_INPUT)
        await collect_user_input_events(
            app,
            UserInputResponse(
                session_id="sess_sup_answer",
                input_id=awaiting.payload["input_id"],
                answer="September",
                resolved_by=_reviewer_actor(),
            ),
        )
        return (
            await supervision_page(store),
            await supervision_page(store, include_content=True),
            await durable_timestamp(
                store, "sess_sup_answer", EventType.SESSION_AWAITING_USER_INPUT
            ),
        )

    page, content_page, asked_at = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.ANSWER]
    event = page.events[0]
    assert event.outcome is SupervisionEventOutcome.ANSWERED
    assert event.input_id is not None
    assert event.tool_call_ids == ("call_ask",)
    assert event.requested_at == asked_at
    assert event.actor is not None
    assert event.actor.subject == "reviewer@example.com"
    assert "September" not in page.model_dump_json()
    assert "September" in content_page.model_dump_json()


def test_tool_round_recovery_is_one_recover_record():
    session_id = "sess_sup_round_recovery"
    app, _store, _tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())

    async def run():
        await collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.FAILED,
                message="the export did not happen",
                resolved_by=_reviewer_actor(),
            ),
        )
        return await supervision_page(app.session_store)

    page = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.RECOVER]
    event = page.events[0]
    assert event.outcome is SupervisionEventOutcome.FAILED
    assert event.tool_call_ids == ("call_1",)
    assert event.approval_id is None
    assert event.input_id is None
    assert event.actor is not None
    assert event.actor.subject == "reviewer@example.com"


class _OneShotProvider(ModelProvider):
    name = "one-shot"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("steered")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_delivered_steering_message_is_one_steer_record():
    store = InMemorySessionStore()
    provider = _OneShotProvider()

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sup_steer",
                messages=[Message.text("user", "close the month")],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
            ),
            identity=profiled_session_identity(
                provider_name="one-shot", model="fake-model", provider=provider
            ),
        )
        accepting = CayuApp(session_store=store, enable_logging=False)
        await accepting.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_sup_steer",
                idempotency_key="steer-1",
                content="use the September ledger",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                requested_by=_reviewer_actor(),
            )
        )
        await store.transition_status(
            "sess_sup_steer",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.INTERRUPTED,
        )
        resuming = CayuApp(session_store=store, enable_logging=False)
        resuming.register_provider(provider)
        resuming.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect_resume_events(
            resuming,
            ResumeRequest(session_id="sess_sup_steer", messages=[Message.text("user", "resume")]),
        )
        return (
            await supervision_page(store),
            await durable_timestamp(store, "sess_sup_steer", EventType.SESSION_MESSAGE_QUEUED),
        )

    page, queued_at = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.STEER]
    event = page.events[0]
    assert event.outcome is SupervisionEventOutcome.DELIVERED
    assert event.queue_id is not None
    assert event.message_origin is SupervisionMessageOrigin.ACTOR
    assert event.requested_at == queued_at
    assert event.actor is not None
    assert event.actor.subject == "reviewer@example.com"
    assert "September ledger" not in page.model_dump_json()


def test_forked_sessions_are_marked_and_can_be_excluded():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=2)

    async def run():
        source = await pause_for_approval(app, store, "sess_sup_fork_source")
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_fork_source", source, ToolApprovalDecision.APPROVE)
        )
        await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_sup_fork_source",
                session_id="sess_sup_fork_child",
            ),
        )
        forked_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_sup_fork_child",
                messages=[Message.text("user", "try it again")],
            ),
        )
        requested = next(
            e for e in forked_events if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        child = await _pending_tool_approval_from_public_event(store, requested)
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_fork_child", child, ToolApprovalDecision.APPROVE)
        )
        return await supervision_page(store), await supervision_page(
            store, exclude_forked_sessions=True
        )

    every, unforked = asyncio.run(run())

    assert [(event.session_id, event.forked) for event in every.events] == [
        ("sess_sup_fork_source", False),
        ("sess_sup_fork_child", True),
    ]
    assert every.events[1].parent_session_id == "sess_sup_fork_source"
    assert [event.session_id for event in unforked.events] == ["sess_sup_fork_source"]


def test_fingerprint_filter_selects_exactly_its_version():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=1)

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_version")
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_version", pending, ToolApprovalDecision.APPROVE)
        )
        unfiltered = await supervision_page(store)
        fingerprint = unfiltered.events[0].execution_profile_fingerprint
        assert fingerprint is not None
        other = "0" * 64 if fingerprint != "0" * 64 else "1" * 64
        return (
            fingerprint,
            await supervision_page(store, execution_profile_fingerprint=fingerprint),
            await supervision_page(store, execution_profile_fingerprint=other),
        )

    fingerprint, same, other = asyncio.run(run())

    assert [event.execution_profile_fingerprint for event in same.events] == [fingerprint]
    assert other.events == ()


def test_usage_of_the_model_step_that_asked_is_joined():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 120, "output_tokens": 30},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_usage")
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_usage", pending, ToolApprovalDecision.APPROVE)
        )
        return await supervision_page(store)

    page = asyncio.run(run())

    usage = page.events[0].usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (120, 30)


class _BackgroundChildProvider(ModelProvider):
    """Parent starts one background reviewer, then both block until interrupted."""

    name = "fake"

    def __init__(self) -> None:
        self.child_started = asyncio.Event()
        self.parent_waiting = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        first_text = request.messages[0].content[0].text  # ty: ignore[unresolved-attribute]
        if first_text == "parent task" and len(request.messages) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_background",
                name="subagent",
                arguments={"agent": "reviewer", "task": "background review"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if first_text == "background review":
            self.child_started.set()
        else:
            self.parent_waiting.set()
        await asyncio.Event().wait()
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_cascaded_interrupt_counts_once_on_the_parent():
    store = InMemorySessionStore()
    provider = _BackgroundChildProvider()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="reviewer", mode=SubagentExecutionMode.BACKGROUND
                    )
                },
            ),
            SubagentResultTool(store),
        ],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    async def run():
        parent = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_sup_cascade_parent",
                    messages=[Message.text("user", "parent task")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_started.wait(), timeout=10)
        await asyncio.wait_for(provider.parent_waiting.wait(), timeout=10)
        await collect_interrupt_events(
            app,
            InterruptSessionRequest(
                session_id="sess_sup_cascade_parent",
                reason="stop everything",
                requested_by=_reviewer_actor(),
            ),
        )
        await asyncio.wait_for(parent, timeout=10)
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return await supervision_page(store, kinds=(SupervisionEventKind.INTERRUPT,))

    page = asyncio.run(run())

    assert [event.session_id for event in page.events] == ["sess_sup_cascade_parent"]
    assert page.events[0].actor is not None
    assert page.events[0].actor.subject == "reviewer@example.com"


def test_withdrawn_and_session_sourced_messages_keep_their_origin():
    store = InMemorySessionStore()
    identity = profiled_session_identity(
        provider_name="one-shot", model="fake-model", provider=_OneShotProvider()
    )

    async def run():
        for session_id in ("sess_sup_msg_target", "sess_sup_msg_source"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "task")],
                ),
                identity=identity,
            )
        source = await store.snapshot_session_message_source("sess_sup_msg_source")
        for key, conditions in (
            ("from-actor", SessionMessageConditions()),
            ("from-session", SessionMessageConditions(source=source)),
        ):
            await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="sess_sup_msg_target",
                    idempotency_key=key,
                    content="private steering",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    conditions=conditions,
                    requested_by=_reviewer_actor(),
                )
            )
        inspection = await store.inspect_session_messages(
            SessionMessageQuery(session_id="sess_sup_msg_target")
        )
        for record in inspection.records:
            await store.apply_session_message_action(
                SessionMessageActionRequest(
                    session_id=inspection.session_id,
                    session_instance_id=inspection.session_instance_id,
                    queue_id=record.queue_id,
                    expected_revision=record.revision,
                    idempotency_key="withdraw-" + record.queue_id,
                    action="withdraw",
                )
            )
        return await supervision_page(store, kinds=(SupervisionEventKind.STEER,))

    page = asyncio.run(run())

    assert [(event.outcome, event.message_origin) for event in page.events] == [
        (SupervisionEventOutcome.WITHDRAWN, SupervisionMessageOrigin.ACTOR),
        (SupervisionEventOutcome.WITHDRAWN, SupervisionMessageOrigin.SESSION),
    ]
    assert all(
        event.actor is not None and event.actor.subject == "reviewer@example.com"
        for event in page.events
    )
    assert "private steering" not in page.model_dump_json()


def test_unstamped_anchor_uses_the_profile_in_force():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=1)

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_in_force")
        await collect_interrupt_events(
            app,
            InterruptSessionRequest(session_id="sess_sup_in_force", reason="stop"),
        )
        page = await supervision_page(store, kinds=(SupervisionEventKind.INTERRUPT,))
        session = await store.load("sess_sup_in_force")
        return pending, page, session

    pending, page, session = asyncio.run(run())

    assert session is not None
    baseline = execution_profile_baseline_from_session_metadata(session.metadata)
    assert page.events[0].execution_profile_fingerprint == baseline.fingerprint
    assert pending.execution_profile_fingerprint == baseline.fingerprint


def test_page_over_its_byte_limit_fails_instead_of_truncating():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=1)

    async def run():
        pending = await pause_for_approval(app, store, "sess_sup_bytes")
        await collect_tool_approval_events(
            app, resolve_approval("sess_sup_bytes", pending, ToolApprovalDecision.APPROVE)
        )
        with pytest.raises(SupervisionEventResultTooLarge):
            await supervision_page(store, max_bytes=64)

    asyncio.run(run())


async def _collect(entrance, request) -> list:
    return [event async for event in entrance(request)]


def test_user_input_recovery_is_an_answer_and_a_recover():
    store = FailingTerminalToolEventStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask", name="ask_user", arguments={"question": "Proceed?"}
                    ),
                    ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), PureRecordingTool()],
    )

    async def run():
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sup_input_recovery",
                messages=[Message.text("user", "ask then act")],
            ),
        )
        awaiting = next(e for e in paused if e.type == EventType.SESSION_AWAITING_USER_INPUT)
        response = UserInputResponse(
            session_id="sess_sup_input_recovery",
            input_id=awaiting.payload["input_id"],
            answer="yes",
            resolved_by=_reviewer_actor(),
        )
        crashed = await collect_user_input_events(app, response)
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED
        await _collect(
            app.recover_user_input,
            UserInputRecoveryRequest(
                session_id="sess_sup_input_recovery",
                input_id=response.input_id,
                answer="yes",
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="recorded outside",
                resolved_by=_reviewer_actor(),
            ),
        )
        return await supervision_page(store)

    page = asyncio.run(run())

    assert [event.kind for event in page.events] == [
        SupervisionEventKind.ANSWER,
        SupervisionEventKind.RECOVER,
    ]
    recovered = page.events[1]
    assert recovered.input_id == page.events[0].input_id
    assert recovered.tool_call_ids == ("call_1",)
    assert recovered.requested_at is not None


class _BigOutputTool(Tool):
    spec = ToolSpec(
        name="big",
        description="Return a large result.",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:supervision-big-output", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx, args):
        return ToolResult(content="y" * 500_000)


class _FailSideEffectTerminalOnceStore(FailingTerminalToolEventStore):
    """Fail only the ``side_effect`` terminal, after the large results are durable."""

    invocation_lifecycle_command_version = 1

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if not self.failed_terminal_once and any(
            event.type == EventType.TOOL_CALL_COMPLETED and event.tool_name == "side_effect"
            for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await InMemorySessionStore.append_events(self, session_id, events)


def test_a_busy_session_does_not_block_other_touches():
    store = _FailSideEffectTerminalOnceStore()
    turns = [
        [
            ModelStreamEvent.tool_call(id=f"big_{index}", name="big", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        for index in range(9)
    ]
    turns.append(
        [
            ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    turns.append(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    busy = CayuApp(session_store=store)
    busy.register_provider(FakeProvider(turns), default=True)
    busy.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_BigOutputTool(), PureRecordingTool()],
    )
    later = approval_app(store, sessions=1)

    async def run():
        failed = await collect_events(
            busy,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sup_busy",
                messages=[Message.text("user", "work hard")],
            ),
        )
        assert failed[-1].type == EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint("sess_sup_busy")
        await collect_tool_round_recovery_events(
            busy,
            ToolRoundRecoveryRequest(
                session_id="sess_sup_busy",
                round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="done outside",
                resolved_by=_reviewer_actor(),
            ),
        )
        pending = await pause_for_approval(later, store, "sess_sup_after_busy")
        await collect_tool_approval_events(
            later, resolve_approval("sess_sup_after_busy", pending, ToolApprovalDecision.APPROVE)
        )
        whole = await supervision_page(store, include_content=True)
        walked = []
        page = await supervision_page(store, limit=1)
        walked.extend(page.events)
        while page.has_more:
            page = await supervision_page(store, limit=1, cursor=page.next_cursor)
            walked.extend(page.events)
        return whole, walked

    whole, walked = asyncio.run(run())

    expected = [
        ("sess_sup_busy", SupervisionEventKind.RECOVER),
        ("sess_sup_after_busy", SupervisionEventKind.APPROVE),
    ]
    assert [(event.session_id, event.kind) for event in whole.events] == expected
    assert [(event.session_id, event.kind) for event in walked] == expected


def test_interrupt_that_supersedes_a_question_keeps_its_linkage():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask", name="ask_user", arguments={"question": "Which period?"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])

    async def run():
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sup_supersede",
                messages=[Message.text("user", "close the month")],
            ),
        )
        awaiting = next(e for e in paused if e.type == EventType.SESSION_AWAITING_USER_INPUT)
        await collect_interrupt_events(
            app,
            InterruptSessionRequest(
                session_id="sess_sup_supersede",
                reason="answered by phone",
                requested_by=_reviewer_actor(),
            ),
        )
        return awaiting, await supervision_page(store, kinds=(SupervisionEventKind.INTERRUPT,))

    awaiting, page = asyncio.run(run())

    event = page.events[0]
    assert event.input_id is not None
    assert event.tool_call_ids == ("call_ask",)
    assert event.model_step_id is not None
    assert event.tool_round_id is not None
    assert awaiting.type == EventType.SESSION_AWAITING_USER_INPUT


def test_message_without_a_recorded_actor_has_unknown_origin():
    store = InMemorySessionStore()
    identity = profiled_session_identity(
        provider_name="one-shot", model="fake-model", provider=_OneShotProvider()
    )

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sup_anonymous_msg",
                messages=[Message.text("user", "task")],
            ),
            identity=identity,
        )
        await store.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_sup_anonymous_msg",
                idempotency_key="anonymous",
                content="private steering",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        inspection = await store.inspect_session_messages(
            SessionMessageQuery(session_id="sess_sup_anonymous_msg")
        )
        record = inspection.records[0]
        await store.apply_session_message_action(
            SessionMessageActionRequest(
                session_id=inspection.session_id,
                session_instance_id=inspection.session_instance_id,
                queue_id=record.queue_id,
                expected_revision=record.revision,
                idempotency_key="withdraw-anonymous",
                action="withdraw",
            )
        )
        return await supervision_page(store, kinds=(SupervisionEventKind.STEER,))

    page = asyncio.run(run())

    assert page.events[0].message_origin is SupervisionMessageOrigin.UNKNOWN
    assert page.events[0].actor is None
