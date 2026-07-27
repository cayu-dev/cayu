from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    Message,
    ModelStreamEvent,
    PendingToolApproval,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
)
from cayu.core.tools import Tool, ToolContext
from cayu.evals import ScriptedModelProvider
from cayu.runtime import InMemorySessionStore, PendingActionQuery, RunRequest, SessionStatus


class _RecordingTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record a call that must remain behind approval.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        effect="external",
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content="recorded")


class _ExpiringApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            approval_expires_in_seconds=60,
        )


@pytest.mark.parametrize(
    ("conflicting_event_type", "expected_error"),
    [
        (EventType.TOOL_CALL_APPROVED, "contradictory tool-call descriptor"),
        (EventType.TOOL_CALL_STARTED, "outside the pending tool round"),
    ],
)
def test_conflicting_approval_descriptor_cannot_execute_external_tool(
    conflicting_event_type: EventType,
    expected_error: str,
) -> None:
    clock = {"now": datetime(2026, 7, 27, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = _RecordingTool()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-1",
                    name=tool.spec.name,
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("resolved"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, clock=lambda: clock["now"])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=_ExpiringApprovalPolicy(),
    )

    async def scenario() -> tuple[list[Event], list[Event]]:
        paused_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-approval-descriptor-conflict",
                    messages=[Message.text("user", "run the side effect")],
                )
            )
        ]
        request_event = next(
            event for event in paused_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = PendingToolApproval.from_event(request_event)
        conflicting_payload: dict[str, object] = {
            "model_step_id": pending.model_step_id,
            "model_attempt_id": pending.model_attempt_id,
            "tool_round_id": pending.tool_round_id,
            "approval_id": pending.approval_id,
            "tool_call_id": "unknown-call",
        }
        if conflicting_event_type == EventType.TOOL_CALL_STARTED:
            conflicting_payload["arguments"] = {"value": "secret"}
        await store.append_event(
            "session-approval-descriptor-conflict",
            Event(
                type=conflicting_event_type,
                session_id="session-approval-descriptor-conflict",
                agent_name=pending.agent_name,
                environment_name=pending.environment_name,
                tool_name="different_tool",
                payload=conflicting_payload,
            ),
        )

        clock["now"] = datetime(2026, 7, 27, 12, 2, tzinfo=UTC)
        resolution_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="session-approval-descriptor-conflict",
                    approval_id=pending.approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        return paused_events, resolution_events

    paused_events, resolution_events = asyncio.run(scenario())

    assert paused_events[-1].type == EventType.SESSION_INTERRUPTED
    assert resolution_events[-1].type == EventType.SESSION_INTERRUPTED
    assert expected_error in resolution_events[-1].payload["error"]
    assert resolution_events[-1].payload["tool_evidence_conflict"] is True
    assert not any(
        event.type
        in {
            EventType.TOOL_CALL_APPROVAL_EXPIRED,
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
        }
        for event in resolution_events
    )
    assert tool.calls == []
    session = asyncio.run(store.load("session-approval-descriptor-conflict"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    page = asyncio.run(
        store.query_pending_actions(
            PendingActionQuery(session_id="session-approval-descriptor-conflict")
        )
    )
    assert page.actions == []
    assert [issue.code for issue in page.issues] == ["source_invalid"]
