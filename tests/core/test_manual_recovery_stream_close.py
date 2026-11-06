"""Closing a manual-recovery stream at its first event settles the claimed session."""

import asyncio

from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailingTerminalToolEventStore
from tests.runtime.runtime_session_support import (
    _approval_pause_app,
    collect_events,
    collect_tool_approval_events,
    collect_user_input_events,
)
from tests.runtime.runtime_tool_support import PureRecordingTool, RequireApprovalPolicy

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.approvals.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools.user_input import UserInputTool


def test_closing_manual_recovery_at_resume_releases_claim():
    asyncio.run(_scenario())


async def _scenario():
    store = FailingTerminalToolEventStore()
    app, provider = _approval_pause_app(
        store=store,
        tool=PureRecordingTool(),
        policy=RequireApprovalPolicy(),
    )
    initial = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="closed-recovery",
            messages=[Message.text("user", "Use the tool")],
        ),
    )
    approval = next(e for e in initial if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    resolution = ToolApprovalRequest(
        session_id="closed-recovery",
        approval_id=approval.payload["approval_id"],
        tool_round_id=approval.payload["tool_round_id"],
        tool_call_id=approval.payload["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
    )
    failed = await collect_tool_approval_events(app, resolution)
    assert failed[-1].type is EventType.SESSION_INTERRUPTED
    assert store.failed_terminal_once
    releases = store.release_calls
    stream = app.recover_tool_approval(
        ToolApprovalRecoveryRequest(
            session_id="closed-recovery",
            approval_id=resolution.approval_id,
            tool_round_id=resolution.tool_round_id,
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified result",
        )
    )
    try:
        while True:
            event = await anext(stream)
            if event.type is EventType.INTERACTION_RESUMED:
                break
    finally:
        await stream.aclose()
    session = await store.load("closed-recovery")
    assert session.status is SessionStatus.INTERRUPTED
    assert store.release_calls > releases
    assert len(provider.requests) == 1


def test_closing_user_input_recovery_at_resume_releases_claim():
    asyncio.run(_user_input_scenario())


async def _user_input_scenario():
    store = FailingTerminalToolEventStore()
    app = CayuApp(session_store=store, enable_logging=False)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_ask", name="ask_user", arguments={"question": "Q?"}
                ),
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
        ]
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), PureRecordingTool()],
    )
    paused = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="closed-input-recovery",
            messages=[Message.text("user", "ask first")],
        ),
    )
    awaiting = next(e for e in paused if e.type is EventType.SESSION_AWAITING_USER_INPUT)
    response = UserInputResponse(
        session_id="closed-input-recovery", input_id=awaiting.payload["input_id"], answer="Sep"
    )
    crashed = await collect_user_input_events(app, response)
    assert crashed[-1].type is EventType.SESSION_INTERRUPTED
    releases = store.release_calls
    stream = app.recover_user_input(
        UserInputRecoveryRequest(
            session_id=response.session_id,
            input_id=response.input_id,
            answer=response.answer,
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="recorded outside",
        )
    )
    try:
        while True:
            event = await anext(stream)
            if event.type is EventType.INTERACTION_RESUMED:
                break
    finally:
        await stream.aclose()
    session = await store.load("closed-input-recovery")
    assert session.status is SessionStatus.INTERRUPTED
    assert store.release_calls > releases
    assert len(provider.requests) == 1
