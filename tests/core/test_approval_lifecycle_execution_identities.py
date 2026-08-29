from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import rebind_test_invocation

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelStreamEvent,
    PendingToolApproval,
    PendingToolApprovalEventView,
    ResolutionActor,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
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
from cayu.runtime import (
    EventSink,
    InMemorySessionStore,
    PendingActionQuery,
    RunRequest,
    SessionStatus,
)
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
)


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
        self.contexts: list[ToolContext] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        self.contexts.append(ctx)
        return ToolResult(content="recorded")


class _RequireApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
            metadata={"scope": "human"},
        )


class _ExpiringApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            approval_expires_in_seconds=60,
        )


class _FailingAfterPendingToolRoundCheckpointStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed_pending_tool_round_once = False

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        checkpoint = await super().load_checkpoint(session_id)
        if (
            not self.failed_pending_tool_round_once
            and checkpoint is not None
            and "pending_tool_round" in checkpoint
        ):
            self.failed_pending_tool_round_once = True
            raise RuntimeError("pending tool round checkpoint persisted before crash")
        return checkpoint


class _FailingTerminalToolEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed_terminal_once = False

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if not self.failed_terminal_once and any(
            event.type is EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


def test_approval_and_tool_evidence_reference_the_admitted_execution_profile() -> None:
    async def scenario() -> None:
        session_id = "session-attributed-approval"
        store = InMemorySessionStore()
        tool = _RecordingTool()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-1",
                        name=tool.spec.name,
                        arguments={"value": "record"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[tool],
            tool_policy=_RequireApprovalPolicy(),
        )

        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run the side effect")],
                )
            )
        ]
        checkpoint = await store.load_checkpoint(session_id)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active is not None
        fingerprint = active.profile.fingerprint
        request_event = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        assert request_event.payload["execution_profile_fingerprint"] == fingerprint
        assert request_event.payload["approval"]["execution_profile_fingerprint"] == fingerprint
        durable_request_event = next(
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = PendingToolApprovalEventView.from_event(durable_request_event)
        assert approval.execution_profile_fingerprint == fingerprint

        resumed = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        attributed_types = {
            EventType.SESSION_RESUMED,
            EventType.TOOL_CALL_APPROVED,
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_COMPLETED,
        }
        attributed = [event for event in resumed if event.type in attributed_types]
        assert {event.type for event in attributed} == attributed_types, [
            (event.type, event.payload) for event in resumed
        ]
        assert {event.payload.get("execution_profile_fingerprint") for event in attributed} == {
            fingerprint
        }
        assert tool.calls == [{"value": "record"}]

    asyncio.run(scenario())


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
            event
            for event in await store.load_events("session-approval-descriptor-conflict")
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = PendingToolApprovalEventView.from_event(request_event)
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
                    tool_round_id=pending.tool_round_id,
                    tool_call_id=pending.tool_call_id,
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


def test_approval_request_drift_is_rejected_while_a_mixed_round_can_still_execute() -> None:
    class SimulatedProcessLoss(BaseException):
        pass

    class StopAfterAmbiguousBlock(EventSink):
        def __init__(self) -> None:
            self.failed = False

        async def emit(self, event: Event) -> None:
            if (
                event.type is EventType.TOOL_CALL_BLOCKED
                and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
                and not self.failed
            ):
                self.failed = True
                raise SimulatedProcessLoss()

    async def scenario() -> None:
        session_id = "sess_mixed_policy_resolution_request_drift"
        store = _FailingAfterPendingToolRoundCheckpointStore()
        sink = StopAfterAmbiguousBlock()
        tool = _RecordingTool()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ambiguous",
                        name=tool.spec.name,
                        arguments={"value": "must remain blocked"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_allowed",
                        name=tool.spec.name,
                        arguments={"value": "may execute after acknowledgement"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("resolved"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[tool],
            tool_policy=_RequireApprovalPolicy(),
        )
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use both tools")],
                )
            )
        ]
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        legacy_round = dict(checkpoint["pending_tool_round"])
        legacy_calls = [dict(call) for call in legacy_round["tool_calls"]]
        for legacy_call in legacy_calls:
            legacy_call.pop("policy_evidence")
        legacy_calls[0].update(policy_decision=None, reason=None, metadata={})
        legacy_calls[1].update(
            policy_decision=ToolPolicyDecision.ALLOW.value,
            reason="durable allow",
            metadata={},
        )
        legacy_round["tool_calls"] = legacy_calls
        legacy_round.pop("policy_state")
        legacy_round.pop("policy_context_version")
        await store.checkpoint(
            session_id,
            {**checkpoint, "pending_tool_round": legacy_round},
        )
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        await rebind_test_invocation(store, session_id)

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        approval = PendingToolApproval.model_validate(checkpoint["pending_tool_approval"])
        assert approval.tool_call_id == "call_ambiguous"

        request = ToolApprovalRequest(
            session_id=session_id,
            approval_id=approval.approval_id,
            tool_round_id=approval.tool_round_id,
            tool_call_id=approval.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
            metadata={"condition": "original"},
            resolved_by=ResolutionActor(subject="operator-1"),
        )
        with pytest.raises(SimulatedProcessLoss):
            _ = [event async for event in app.resolve_tool_approval(request)]
        assert sink.failed is True
        assert tool.calls == []
        blocked_events = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_BLOCKED
            and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
        ]
        assert len(blocked_events) == 1

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert recovered.actions == (IncompleteSessionRecoveryAction.PENDING_APPROVAL,)

        conflicting = [
            event
            async for event in app.resolve_tool_approval(
                request.model_copy(
                    update={
                        "metadata": {"condition": "changed"},
                        "resolved_by": ResolutionActor(subject="operator-2"),
                    }
                )
            )
        ]
        assert [event.type for event in conflicting] == [
            EventType.INTERACTION_RESUMED,
            EventType.SESSION_INTERRUPTED,
        ]
        assert "different resolution request" in conflicting[-1].payload["error"]
        assert tool.calls == []

        completed = [event async for event in app.resolve_tool_approval(request)]
        assert completed[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "may execute after acknowledgement"}]
        assert tool.contexts[0].metadata["condition"] == "original"
        blocked_events = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_BLOCKED
            and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
        ]
        assert len(blocked_events) == 1

    asyncio.run(scenario())


def test_expired_pre_digest_approval_grant_retry_fails_closed_without_coercion() -> None:
    # Expiry must not contradict a prior grant, but a grant recorded before
    # request digests existed also cannot authorize execution after upgrade.
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = _RecordingTool()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
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

    async def scenario() -> list[Event]:
        session_id = "sess_expiry_retry"
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use the tool")],
                )
            )
        ]
        approval_event = next(
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = PendingToolApprovalEventView.from_event(approval_event)

        # A prior in-window resolve crashed after recording the grant.
        await store.append_event(
            session_id,
            Event(
                type=EventType.TOOL_CALL_APPROVED,
                session_id=session_id,
                agent_name=approval.agent_name,
                environment_name=approval.environment_name,
                tool_name=approval.tool_name,
                payload={
                    "model_step_id": approval.model_step_id,
                    "model_attempt_id": approval.model_attempt_id,
                    "tool_round_id": approval.tool_round_id,
                    "approval_id": approval.approval_id,
                    "tool_call_id": approval.tool_call_id,
                },
            ),
        )

        clock["now"] = datetime(2026, 7, 9, 12, 5, tzinfo=UTC)
        return [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

    events = asyncio.run(scenario())

    assert not any(event.type is EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert not any(event.type is EventType.TOOL_CALL_APPROVAL_DENIED for event in events)
    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert (
        "prior durable resolution activity has no exact resolution request identity"
        in events[-1].payload["error"]
    )
    assert tool.calls == []


def test_tool_approval_recovery_does_not_authorize_unstarted_sibling() -> None:
    async def scenario() -> None:
        session_id = "sess_approval_recovery_pending_sibling"
        store = _FailingTerminalToolEventStore()
        tool = _RecordingTool()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_first",
                        name=tool.spec.name,
                        arguments={"value": "first"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_second",
                        name=tool.spec.name,
                        arguments={"value": "second"},
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
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[tool],
            tool_policy=_RequireApprovalPolicy(),
        )
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run both")],
                )
            )
        ]
        approval_event = next(
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = PendingToolApprovalEventView.from_event(approval_event)
        request = ToolApprovalRequest(
            session_id=session_id,
            approval_id=approval.approval_id,
            tool_round_id=approval.tool_round_id,
            tool_call_id=approval.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
            reason="approved under original conditions",
            metadata={"condition": "original"},
            resolved_by=ResolutionActor(subject="operator-1"),
        )

        interrupted = [event async for event in app.resolve_tool_approval(request)]
        assert interrupted[-1].type is EventType.SESSION_INTERRUPTED
        assert tool.calls == [{"value": "first"}]

        recovered = [
            event
            async for event in app.recover_tool_approval(
                ToolApprovalRecoveryRequest(
                    session_id=session_id,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.tool_round_id,
                    tool_call_id="call_first",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="first call completed externally",
                    metadata={"condition": "recovery-only"},
                    resolved_by=ResolutionActor(subject="recovery-operator"),
                )
            )
        ]
        assert recovered[-1].type is EventType.SESSION_INTERRUPTED
        assert "cannot authorize pending sibling execution" in recovered[-1].payload["error"]
        assert tool.calls == [{"value": "first"}]

        completed = [event async for event in app.resolve_tool_approval(request)]
        assert completed[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "first"}, {"value": "second"}]
        assert tool.contexts[-1].metadata["condition"] == "original"

    asyncio.run(scenario())
