"""Tool recovery contracts."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.runtime.runtime_provider_support import FakeProvider, NativeStructuredOutputFakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingAfterPendingToolRoundCheckpointStore,
    FailingOrdinaryToolResultCloseStore,
    FailingSecondTerminalToolEventStore,
    FailingTerminalToolEventStore,
    _crashed_tool_round_app,
)
from tests.runtime.runtime_session_support import (
    _INTERACTION_LIFECYCLE_EVENT_TYPES,
    _private_events_for_public_events,
    _private_record_for_public_event,
    _tool_round_identity,
    assert_only_model_step_publication_checkpoint,
    checkpoint_without_active_invocation_profile,
    collect_events,
    collect_interrupt_events,
    collect_resume_events,
    collect_tool_approval_events,
    collect_tool_round_recovery_events,
)
from tests.runtime.runtime_tool_support import PureRecordingTool, SideEffectTool

import cayu.runtime.execution_profiles as execution_profiles_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    PendingToolApprovalEventView,
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRequest,
)
from cayu.budgets.run_limits import RunLimits
from cayu.context.structured_output import StructuredOutputSpec
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runners.base import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, ExecResult, Runner
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions._terminal_evidence import _checkpoint_with_session_run_operation
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import EventRecord, Session, SessionStatus
from cayu.sessions.recovery import (
    RecoveryBlockerCode,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)
from cayu.sessions.requests import InterruptSessionRequest, ResumeRequest, RunRequest
from cayu.sessions.transcript_queries import TranscriptQuery
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.commands import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    ExecCommandTool,
)
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.static import StaticVault


class RecordingNativeStructuredOutputProvider(NativeStructuredOutputFakeProvider):
    def __init__(self, events: list[ModelStreamEvent] | list[list[ModelStreamEvent]]) -> None:
        super().__init__(events)
        self.preflight_schemas: list[dict[str, Any]] = []

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        self.preflight_schemas.append(json_schema)


def test_cayu_app_recovers_pending_tool_round_from_recorded_terminal_event():
    store = FailingOrdinaryToolResultCloseStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_recorded",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_recorded"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == [{}]

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_recorded",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == [{}]
    receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            "sess_tool_round_recover_recorded",
            f"tool-round:{checkpoint['pending_tool_round']['tool_round_id']}",
        )
    )
    assert receipt is not None
    assert receipt.intent["tool_call_ids"] == ["call_1"]
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert_only_model_step_publication_checkpoint(
        asyncio.run(store.load_checkpoint("sess_tool_round_recover_recorded"))
    )

    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_recorded"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert transcript[2].content[0].tool_call_id == "call_1"
    assert transcript[2].content[0].content == "recorded"
    assert provider.requests[1].messages[-3].role == "assistant"
    assert provider.requests[1].messages[-2].role == "tool"
    assert provider.requests[1].messages[-1].content[0].text == "continue"

    lifecycle = asyncio.run(
        store.query_events(
            EventQuery(
                session_id="sess_tool_round_recover_recorded",
                event_types=tuple(_INTERACTION_LIFECYCLE_EVENT_TYPES),
            )
        )
    )
    interaction_ids = {record.event.interaction_id for record in lifecycle}
    assert len(interaction_ids) == 1
    assert None not in interaction_ids
    lifecycle_types = [record.event.type for record in lifecycle]
    assert lifecycle_types.count(EventType.INTERACTION_STARTED) == 1
    assert EventType.INTERACTION_PAUSED in lifecycle_types
    assert EventType.INTERACTION_RESUMED in lifecycle_types
    assert lifecycle_types[-1] is EventType.INTERACTION_COMPLETED

    attributed_transcript = asyncio.run(
        store.query_transcript(
            TranscriptQuery(session_id="sess_tool_round_recover_recorded", limit=10)
        )
    )
    assert {record.interaction_id for record in attributed_transcript.records} == interaction_ids


def test_cayu_app_recovers_legacy_failed_policy_outcome_without_reexecution():
    class LegacyCommandDenialTool(Tool):
        spec = ToolSpec(
            name="exec_command",
            description="Model a command denial stored before canonical blocked events.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(
                content="Command denied by policy.",
                structured={
                    "error": "command_denied",
                    "decision": "deny",
                    "reason": "Command denied by policy.",
                },
                is_error=True,
            )

    session_id = "sess_legacy_failed_policy_recovery"
    store = FailingOrdinaryToolResultCloseStore()
    tool = LegacyCommandDenialTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_legacy_denial",
                    name="exec_command",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("continued from legacy denial"),
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
                messages=[Message.text("user", "run the command")],
            ),
        )
    )

    assert initial_events[-1].type == EventType.SESSION_FAILED
    legacy_failed = [event for event in initial_events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(legacy_failed) == 1
    assert legacy_failed[0].payload["result"]["structured"]["error"] == "command_denied"
    assert tool.calls == 1

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == 1
    assert all(
        event.type not in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED}
        for event in resumed_events
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    recovered_result = next(message for message in transcript if message.role == "tool").content[0]
    assert recovered_result.tool_call_id == "call_legacy_denial"
    assert recovered_result.structured["error"] == "command_denied"


def test_recorded_tool_outcomes_treats_terminal_without_result_as_ambiguous():
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery

    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call_1",
                name="side_effect",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=_tool_round_identity(),
    )

    outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=[
            Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="sess_1",
                payload={
                    "model_step_id": pending_round.model_step_id,
                    "model_attempt_id": pending_round.model_attempt_id,
                    "tool_round_id": pending_round.tool_round_id,
                    "tool_call_id": "call_1",
                },
            )
        ],
        pending_round=pending_round,
    )

    assert outcomes == {}
    assert started_ids == {"call_1"}


def test_cayu_app_resume_gates_pending_round_without_policy_outcome():
    store = FailingAfterPendingToolRoundCheckpointStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_not_started",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_not_started"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    assert not any(event.type == EventType.TOOL_CALL_STARTED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == []

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_not_started",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == []
    approval_events = [
        event
        for event in resume_events
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        and event.payload.get("recovered") is True
    ]
    assert len(approval_events) == 1
    durable_approval = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_tool_round_recover_not_started",
            approval_events,
        )
    )[0]
    approval = PendingToolApprovalEventView.from_event(durable_approval)
    assert approval.tool_round_id == checkpoint["pending_tool_round"]["tool_round_id"]
    assert approval.tool_call_id == "call_1"
    assert approval.tool_calls[0].policy_evidence == "ambiguous"
    assert approval.tool_calls[0].policy_decision is None
    assert approval.metadata == {
        "recovered": True,
        "policy_evaluation": "ambiguous",
    }
    assert resume_events[-1].type == EventType.SESSION_INTERRUPTED
    recovered_checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_not_started"))
    assert recovered_checkpoint is not None
    assert recovered_checkpoint["pending_tool_approval"]["approval_id"] == (approval.approval_id)
    assert recovered_checkpoint["pending_tool_round"]["policy_state"] == "planned"
    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_not_started"))
    assert [message.role for message in transcript] == ["user"]
    assert len(provider.requests) == 1


def test_cayu_app_recovers_selected_external_result_after_event_append_failure():
    store = FailingTerminalToolEventStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_unknown",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_unknown"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    assert any(event.type == EventType.TOOL_CALL_STARTED for event in initial_events)
    assert not any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == [{}]

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_unknown",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == [{}]
    # The tool returned a valid result and the effect owner selected it before
    # the separate event append failed. Recovery must retain that positive proof.
    assert len(provider.requests) == 2
    assert resume_events[-1].type is EventType.SESSION_COMPLETED
    persisted = asyncio.run(store.load_events("sess_tool_round_recover_unknown"))
    completed = [event for event in persisted if event.type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 1
    assert completed[0].payload["tool_call_id"] == "call_1"
    assert completed[0].payload["result"]["content"] == "recorded"
    assert not any(event.type is EventType.TOOL_CALL_FAILED for event in persisted)
    retained = asyncio.run(store.load_checkpoint("sess_tool_round_recover_unknown"))
    assert retained is not None
    assert "pending_tool_round" not in retained
    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_unknown"))
    tool_messages = [message for message in transcript if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content[0].content == "recorded"


def test_cayu_app_recovers_pending_tool_round_without_reusing_old_tool_call_id():
    store = FailingSecondTerminalToolEventStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"round": "old"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"round": "current"},
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_reused_id",
                messages=[Message.text("user", "use the tool twice")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_reused_id"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    private_initial_events = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_tool_round_recover_reused_id",
            initial_events,
        )
    )
    assert [
        event.payload["tool_round_id"]
        for event in private_initial_events
        if event.type == EventType.TOOL_CALL_STARTED
    ] == [
        private_initial_events[3].payload["tool_round_id"],
        checkpoint["pending_tool_round"]["tool_round_id"],
    ]
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == [{"round": "old"}, {"round": "current"}]

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_reused_id",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == [{"round": "old"}, {"round": "current"}]
    recovered_terminals = [
        event for event in resume_events if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert len(recovered_terminals) == 1
    durable_terminal = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_tool_round_recover_reused_id",
            recovered_terminals,
        )
    )[0]
    assert (
        durable_terminal.payload["tool_round_id"]
        == checkpoint["pending_tool_round"]["tool_round_id"]
    )
    assert durable_terminal.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_tool_round_recover_reused_id",
        tool_round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
    )
    assert durable_terminal.payload["result"]["content"] == "recorded"
    assert durable_terminal.payload["result"]["is_error"] is False
    assert resume_events[-1].type == EventType.SESSION_COMPLETED

    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_reused_id"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert transcript[2].content[0].content == "recorded"
    assert transcript[4].content[0].content == "recorded"


def test_recovery_plan_does_not_offer_unvalidated_external_effect_disposition() -> None:
    session_id = "sess_tool_effect_plan_no_unvalidated_disposition"
    app, store, tool, _checkpoint = _crashed_tool_round_app(session_id)

    async def scenario() -> None:
        before = await store.load_checkpoint(session_id)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=(session_id,)))
        )
        assert len(plan.items) == 1
        assert RecoveryPlanAction.TOOL_MARK_COMPLETED not in plan.items[0].allowed_actions
        assert RecoveryPlanAction.TOOL_MARK_FAILED not in plan.items[0].allowed_actions
        assert await store.load_checkpoint(session_id) == before
        assert tool.calls == [{}]
        assert RecoveryPlanAction.AUTOMATIC_REPAIR in plan.items[0].allowed_actions
        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(plan=plan, execution_id="repair-selected-effect")
        )
        assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
        events = await store.load_events(session_id)
        assert sum(event.type is EventType.TOOL_CALL_COMPLETED for event in events) == 1
        assert tool.calls == [{}]

    asyncio.run(scenario())


def test_recovery_plan_keeps_ambiguous_external_effect_for_receipt_reconciliation() -> None:
    class UnknownEffectTool(SideEffectTool):
        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            await super().run(ctx, args)
            raise ConnectionError("effect acknowledgement lost")

    async def scenario() -> None:
        tool = UnknownEffectTool()
        app = CayuApp(enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
        session_id = "sess_plan_ambiguous_external_effect"
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run the effect")],
            ),
        )
        before = await app.session_store.load_checkpoint(session_id)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=(session_id,)))
        )
        assert plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
        assert RecoveryBlockerCode.TOOL_EFFECT_OUTCOME_UNKNOWN in {
            blocker.code for blocker in plan.items[0].blockers
        }
        assert await app.session_store.load_checkpoint(session_id) == before
        assert tool.calls == [{}]

    asyncio.run(scenario())


@pytest.mark.parametrize("effect", ["none", "idempotent"])
def test_recovery_plan_preserves_manual_disposition_for_replay_safe_tools(effect) -> None:
    from cayu.tools.base import ToolEffect

    # Calls are instrumentation. The pure variant only computes a result; the
    # mutating variant's fake downstream collapses writes by the runtime key.
    downstream: dict[str, str] = {}

    class ReplaySafeTool(SideEffectTool):
        spec = ToolSpec(
            name="side_effect",
            description="Compute or idempotently store a fixed result.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect(effect),
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.calls.append(args)
            self.contexts.append(ctx)
            if effect == "idempotent":
                assert ctx.idempotency_key is not None
                result = downstream.setdefault(ctx.idempotency_key, "recorded")
            else:
                result = "recorded"
            return ToolResult(content=result)

    session_id = f"sess_tool_round_recovery_plan_{effect}"
    app, _store, tool, _checkpoint = _crashed_tool_round_app(session_id, tool=ReplaySafeTool())

    async def scenario() -> None:
        plan = await app.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=(session_id,)))
        )
        item = plan.items[0]

        assert len(item.pending_actions) == 1
        assert RecoveryBlockerCode.TOOL_EFFECT_OUTCOME_UNKNOWN in {
            blocker.code for blocker in item.blockers
        }
        assert RecoveryPlanAction.AUTOMATIC_REPAIR not in item.allowed_actions
        assert RecoveryPlanAction.TOOL_MARK_COMPLETED in item.allowed_actions

        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(
                plan=plan,
                execution_id="tool-effect-disposition",
                decisions=(
                    RecoveryDecision(
                        item_id=item.item_id,
                        action=RecoveryPlanAction.TOOL_MARK_COMPLETED,
                        message="operator supplied the recovered result",
                    ),
                ),
            )
        )

        assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
        assert receipt.items[0].final_session_status is SessionStatus.COMPLETED
        assert tool.calls == [{}]
        assert len(downstream) == (1 if effect == "idempotent" else 0)

    asyncio.run(scenario())


def test_tool_round_recovery_rejects_profile_drift_before_provider_preflight() -> None:
    session_id = "sess_tool_round_profile_before_provider_preflight"
    _, store, _, checkpoint = _crashed_tool_round_app(session_id)

    class ChangedSideEffectTool(SideEffectTool):
        spec = ToolSpec(
            name="side_effect",
            description="Changed tool declaration.",
            input_schema={"type": "object", "properties": {}},
        )

    provider = RecordingNativeStructuredOutputProvider([])
    replacement_app = CayuApp(session_store=store, enable_logging=False)
    replacement_app.register_provider(provider, default=True)
    replacement_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ChangedSideEffectTool()],
    )

    with pytest.raises(ExecutionProfileMismatchError):
        asyncio.run(
            collect_tool_round_recovery_events(
                replacement_app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified externally",
                    structured_output=StructuredOutputSpec(
                        name="recovered_output",
                        strategy="native",
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    ),
                ),
            )
        )

    assert provider.preflight_schemas == []


def test_tool_round_recovery_rejects_changed_finalization_before_provider_dispatch() -> None:
    session_id = "sess_tool_round_finalization_before_provider_dispatch"
    providers: list[FakeProvider] = []
    app, store, _tool, checkpoint = _crashed_tool_round_app(
        session_id,
        provider_capture=providers,
    )
    before = asyncio.run(store.load(session_id))
    assert before is not None
    assert len(providers) == 1
    initial_request_count = len(providers[0].requests)

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified externally",
                    max_steps=17,
                ),
            )
        )

    assert caught.value.changed_component_classes == (ExecutionProfileComponentClass.FINALIZATION,)
    assert len(providers[0].requests) == initial_request_count
    after = asyncio.run(store.load(session_id))
    assert after is not None
    assert after.status is before.status
    assert after.run_epoch == before.run_epoch


def test_tool_round_recovery_restores_original_run_limit_accounting() -> None:
    session_id = "sess_tool_round_recovery_restores_run_limits"
    providers: list[FakeProvider] = []
    app, _store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        provider_capture=providers,
        limits=RunLimits(max_tool_calls=1, scope="run"),
        tool=PureRecordingTool(),
        continuation_events=[
            ModelStreamEvent.tool_call(id="call_2", name="side_effect", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
    )

    events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="verified externally",
            ),
        )
    )

    limit_event = next(event for event in events if event.type is EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["limit"] == "tool_calls"
    assert limit_event.payload["actual"] == 2
    assert tool.calls == [{}]
    assert len(providers) == 1
    assert len(providers[0].requests) == 2


def test_command_policy_block_is_replayed_without_runner_execution_after_round_close_crash():
    session_id = "sess_command_policy_block_recovery"
    store = FailingOrdinaryToolResultCloseStore()

    class NeverCalledRunner(Runner):
        def __init__(self) -> None:
            self.calls = 0

        async def exec(
            self,
            command: ExecCommand,
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_s: int | None = None,
            stdin: str | None = None,
            output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
        ) -> ExecResult:
            del command, cwd, env, timeout_s, stdin, output_limit_bytes
            self.calls += 1
            return ExecResult(stdout="unexpected")

    class DenyCommandPolicy(CommandPolicy):
        async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
            del ctx, request
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY,
                reason="Command is not approved.",
            )

    runner = NeverCalledRunner()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_denied_command",
                    name="exec_command",
                    arguments={"argv": ["git", "push"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered without execution"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="command-recovery"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=DenyCommandPolicy())],
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "push the branch")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert runner.calls == 0
    stored_before_resume = asyncio.run(store.load_events(session_id))
    blocked_before_resume = [
        event for event in stored_before_resume if event.type == EventType.TOOL_CALL_BLOCKED
    ]
    assert len(blocked_before_resume) == 1
    assert blocked_before_resume[0].payload["denied_by"] == "command_policy"
    assert blocked_before_resume[0].payload["metadata"] == {}
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue safely")],
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert runner.calls == 0
    assert all(event.type != EventType.TOOL_CALL_BLOCKED for event in resumed_events)
    stored_after_resume = asyncio.run(store.load_events(session_id))
    assert (
        len([event for event in stored_after_resume if event.type == EventType.TOOL_CALL_BLOCKED])
        == 1
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    denied_result = next(message for message in transcript if message.role == "tool").content[0]
    assert denied_result.tool_call_id == "call_denied_command"
    assert denied_result.is_error is True
    assert denied_result.structured["error"] == "command_denied"


def test_approved_tool_policy_then_command_policy_denial_emits_one_command_block():
    session_id = "sess_approved_then_command_denied"

    class NeverCalledRunner(Runner):
        def __init__(self) -> None:
            self.calls = 0

        async def exec(
            self,
            command: ExecCommand,
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_s: int | None = None,
            stdin: str | None = None,
            output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
        ) -> ExecResult:
            del command, cwd, env, timeout_s, stdin, output_limit_bytes
            self.calls += 1
            return ExecResult(stdout="unexpected")

    class RequireToolApproval(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Approve access to the command tool.",
            )

    class DenyCommandPolicy(CommandPolicy):
        async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
            del ctx, request
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY,
                reason="Remote mutation is forbidden.",
            )

    runner = NeverCalledRunner()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_approved_command",
                    name="exec_command",
                    arguments={"argv": ["git", "push"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("The command remained blocked."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="approval-command"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=DenyCommandPolicy())],
        tool_policy=RequireToolApproval(),
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "push")],
            ),
        )
    )
    approval = next(
        event.payload["approval"]
        for event in initial_events
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )

    resolution_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval["approval_id"],
                tool_round_id=approval["tool_round_id"],
                tool_call_id=approval["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert runner.calls == 0
    assert (
        len([event for event in resolution_events if event.type == EventType.TOOL_CALL_APPROVED])
        == 1
    )
    blocked = [event for event in resolution_events if event.type == EventType.TOOL_CALL_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["denied_by"] == "command_policy"
    assert blocked[0].payload["decision"] == "deny"
    assert blocked[0].payload["metadata"] == {}
    private_initial = asyncio.run(
        _private_events_for_public_events(app.session_store, session_id, initial_events)
    )
    private_resolution = asyncio.run(
        _private_events_for_public_events(app.session_store, session_id, resolution_events)
    )
    private_approval_event = next(
        event for event in private_initial if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    private_blocked = next(
        event for event in private_resolution if event.type == EventType.TOOL_CALL_BLOCKED
    )
    assert private_blocked.payload["approval_id"] == private_approval_event.payload["approval_id"]
    assert all(event.type != EventType.TOOL_CALL_FAILED for event in resolution_events)
    assert all(event.type != EventType.TOOL_CALL_APPROVAL_DENIED for event in resolution_events)
    assert resolution_events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.parametrize("dynamic_scope", [False, True], ids=["static", "dynamic"])
def test_cayu_app_recover_tool_round_completed_outcome_resumes_without_unknown(
    dynamic_scope: bool,
):
    session_id = f"sess_tool_round_manual_completed_{dynamic_scope}"
    recovery_message = (
        "manual-recovery-secret-canary" if dynamic_scope else "side effect verified externally"
    )
    environment = (
        Environment(
            EnvironmentSpec(name="dynamic"),
            vault=StaticVault({"api_key": recovery_message}),
        )
        if dynamic_scope
        else None
    )
    provider_capture: list[FakeProvider] = []
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        environment=environment,
        provider_capture=provider_capture,
        tool=PureRecordingTool(),
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message=recovery_message,
                structured={
                    "verified": True,
                    "provided": recovery_message,
                    "isolated_tool_failure_code": "application_control_secret",
                    "tool_execution_boundary": "posix_process",
                    "tool_timeout_strength": "hard_process_deadline",
                },
                reason=recovery_message,
                metadata={"provided": recovery_message},
                resolved_by=ResolutionActor(
                    subject="operator@example.com",
                    tenant="tenant-a",
                    source=ResolutionActorSource.REQUEST,
                    claims={"role": "incident-responder"},
                ),
            ),
        )
    )

    # The tool was executed exactly once (before the crash) — never re-run.
    assert tool.calls == [{}]
    recovered = next(
        event
        for event in recovery_events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    )
    durable_recovered = asyncio.run(
        _private_events_for_public_events(store, session_id, [recovered])
    )[0]
    assert durable_recovered.payload["tool_round_id"] == round_id
    assert durable_recovered.payload["tool_call_id"] == "call_1"
    # The manual terminal event carries the same idempotency key as the crashed
    # execution's started event, and the same tool_round_id the auto-close ledger
    # reads — the continuation reuses this exact event instead of synthesizing.
    assert durable_recovered.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id=session_id,
        tool_round_id=round_id,
        tool_call_id="call_1",
    )
    expected_content = (
        "Externally verified tool output is unavailable because the invocation "
        "secret scope could not be reconstructed."
        if dynamic_scope
        else "side effect verified externally"
    )
    assert durable_recovered.payload["result"]["content"] == expected_content
    assert durable_recovered.payload["result"]["is_error"] is False
    assert durable_recovered.payload["reason"] == (None if dynamic_scope else recovery_message)
    assert durable_recovered.payload["metadata"] == (
        {} if dynamic_scope else {"provided": recovery_message}
    )
    assert recovered.payload["resolved_by"] == {
        "subject": "operator@example.com",
        "tenant": "tenant-a",
        "source": "request",
    }
    resumed = next(event for event in recovery_events if event.type == EventType.SESSION_RESUMED)
    assert resumed.payload["resolved_by"] == {
        "subject": "operator@example.com",
        "tenant": "tenant-a",
        "source": "request",
    }
    assert "claims" not in recovered.payload["resolved_by"]
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    assert not any(
        "outcome_unknown" in str(event.payload.get("result", "")) for event in recovery_events
    )
    assert_only_model_step_publication_checkpoint(asyncio.run(store.load_checkpoint(session_id)))

    transcript = asyncio.run(store.load_transcript(session_id))
    recovered_result = transcript[2].content[0]
    assert recovered_result.tool_call_id == "call_1"
    assert recovered_result.is_error is False
    assert recovered_result.content == expected_content
    assert recovered_result.structured == (
        {
            "error": "invalid_tool_output",
            "manual_recovery": True,
            "outcome_unknown": False,
            "reason": "invocation_secret_scope_unavailable",
        }
        if dynamic_scope
        else {"verified": True, "provided": recovery_message}
    )
    assert "isolated_tool_failure_code" not in json.dumps(
        [message.model_dump(mode="json") for message in provider_capture[0].requests[-1].messages]
    )
    if dynamic_scope:
        assert recovery_message not in repr(recovery_events)
        assert recovery_message not in repr(asyncio.run(store.load_events(session_id)))
        assert recovery_message not in repr(transcript)
        assert recovery_message not in repr(provider_capture[0].requests[-1].messages)

    # The round is closed and the checkpoint cleared — a second recovery rejects.
    with pytest.raises(RuntimeError, match="no pending tool round"):
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


def test_manual_tool_round_recovery_rebases_stale_running_operation() -> None:
    session_id = "sess_tool_round_manual_stale_running_operation"
    operation_id = "stale-manual-recovery-operation"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]
    active_profile = execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None

    async def simulate_crashed_continuation() -> None:
        def claim_crashed_continuation(current_session, current_checkpoint):
            claimed = _checkpoint_with_session_run_operation(
                checkpoint=current_checkpoint,
                current_session=current_session,
                operation_id=operation_id,
            )
            return execution_profiles_module.checkpoint_with_active_invocation_execution_profile(
                claimed,
                session_id=current_session.id,
                interaction_id=active_profile.interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=active_profile.profile,
                expected=active_profile,
            )

        await rebind_test_invocation(
            store,
            session_id,
            target_status=SessionStatus.RUNNING,
            checkpoint_transform=claim_crashed_continuation,
        )
        await store.release_run_fence(session_id)

    asyncio.run(simulate_crashed_continuation())
    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect verified externally",
            ),
        )
    )

    session = asyncio.run(store.load(session_id))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    assert recovery_events[-1].payload["session_run_operation_id"] == PRIVATE_EVENT_AUTHORITY
    completed_records = asyncio.run(
        store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_COMPLETED,
            )
        )
    )
    assert completed_records[-1].event.payload["session_run_operation_id"] == operation_id
    assert_only_model_step_publication_checkpoint(asyncio.run(store.load_checkpoint(session_id)))
    assert tool.calls == [{}]


def test_manual_tool_round_recovery_repairs_missing_prior_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "sess_tool_round_manual_missing_prior_terminal"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    active_profile = execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None
    resolved_profile_fingerprints: list[str] = []
    original_profile_resolver = execution_profile_admission.resolve_execution_profile_identity

    def count_profile_resolution(*args, **kwargs):
        resolved = original_profile_resolver(*args, **kwargs)
        resolved_profile_fingerprints.append(resolved.fingerprint)
        return resolved

    monkeypatch.setattr(
        execution_profile_admission,
        "resolve_execution_profile_identity",
        count_profile_resolution,
    )

    async def simulate_lost_failure_event_commit() -> None:
        # The in-memory store has no event-deletion API because durable history is
        # immutable. Remove only the synthetic fixture's final event from every
        # internal index to model a process/database failure between the status
        # commit and terminal append while preserving the real tool-round receipt.
        async with store._lock:
            failed_record = next(
                record
                for record in store._session_event_records[session_id]
                if record.event.type == EventType.SESSION_FAILED
            )
            failed_event = failed_record.event
            event_type = str(failed_event.type)
            store._events[session_id] = [
                event for event in store._events[session_id] if event.id != failed_event.id
            ]
            store._event_records.remove(failed_record)
            store._session_event_records[session_id].remove(failed_record)
            store._event_records_by_id.pop((session_id, failed_event.id))
            store._type_event_records[event_type].remove(failed_record)
            if not store._type_event_records[event_type]:
                store._type_event_records.pop(event_type)
            store._event_ids[session_id].remove(failed_event.id)
            store._persisted_event_side_effect_deliveries.pop(
                (session_id, failed_event.id),
                None,
            )
            store._pending_action_latest_barrier_records.pop(session_id, None)

    asyncio.run(simulate_lost_failure_event_commit())
    assert (
        asyncio.run(
            store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_FAILED,
                )
            )
        )
        == []
    )

    recovered = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect verified externally",
            ),
        )
    )
    terminal_records = asyncio.run(
        store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_FAILED,
                    EventType.SESSION_COMPLETED,
                ),
            )
        )
    )

    assert resolved_profile_fingerprints
    assert set(resolved_profile_fingerprints) == {active_profile.profile.fingerprint}
    assert recovered[-1].type == EventType.SESSION_COMPLETED
    assert [record.event.type for record in terminal_records] == [
        EventType.SESSION_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    assert terminal_records[0].event.payload["terminal_evidence_repaired"] is True
    assert "session_run_operation_id" not in terminal_records[0].event.payload
    assert "session_run_operation_id" in terminal_records[1].event.payload
    assert_only_model_step_publication_checkpoint(asyncio.run(store.load_checkpoint(session_id)))
    assert tool.calls == [{}]


def test_tool_round_recovery_reconciles_ambiguous_append_acknowledgement() -> None:
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
                raise RuntimeError("manual recovery commit acknowledgement lost")

    session_id = "sess_tool_round_recovery_ambiguous_append"
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        store=AmbiguousRecoveryAppendStore(),
        tool=PureRecordingTool(),
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    recovery = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect verified externally",
            ),
        )
    )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    assert recovery[-1].type == EventType.SESSION_INTERRUPTED
    assert recovery[-1].payload["manual_recovery_persisted"] is True
    persisted = asyncio.run(store.load_events(session_id))
    recovered = [
        event
        for event in persisted
        if event.payload.get("manual_recovery") is True
        and event.payload.get("tool_call_id") == "call_1"
    ]
    assert len(recovered) == 1

    resumed = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )
    assert resumed[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{}]


def test_tool_round_recovery_interrupts_when_append_reconciliation_is_unavailable() -> None:
    class UnreconcilableRecoveryAppendStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_recovery_ack = False
            self.failed_reconciliation = False

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            manual_recovery = any(event.payload.get("manual_recovery") is True for event in events)
            await super().append_events(session_id, events)
            if manual_recovery and not self.failed_recovery_ack:
                self.failed_recovery_ack = True
                raise RuntimeError("manual recovery commit acknowledgement lost")

        async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
            if (
                self.failed_recovery_ack
                and not self.failed_reconciliation
                and query is not None
                and query.event_id is not None
            ):
                self.failed_reconciliation = True
                raise RuntimeError("manual recovery reconciliation unavailable")
            return await super().query_events(query)

    session_id = "sess_tool_round_recovery_unreconcilable_append"
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        store=UnreconcilableRecoveryAppendStore(),
        tool=PureRecordingTool(),
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    recovery = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect verified externally",
            ),
        )
    )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    assert recovery[-1].type == EventType.SESSION_INTERRUPTED
    assert recovery[-1].payload["manual_recovery_persistence_unknown"] is True
    assert recovery[-1].payload["persistence_reconciliation_error_type"] == "RuntimeError"
    assert "manual_recovery_persisted" not in recovery[-1].payload

    resumed = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )
    assert resumed[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{}]


def test_tool_round_recovery_restores_status_when_append_did_not_commit() -> None:
    class PreCommitRecoveryAppendFailureStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            if any(event.payload.get("manual_recovery") is True for event in events):
                raise RuntimeError("manual recovery append rejected before commit")
            await super().append_events(session_id, events)

    session_id = "sess_tool_round_recovery_precommit_append_failure"
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        store=PreCommitRecoveryAppendFailureStore(),
        tool=PureRecordingTool(),
    )

    with pytest.raises(RuntimeError, match="append rejected before commit"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect verified externally",
                ),
            )
        )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.FAILED
    persisted = asyncio.run(store.load_events(session_id))
    assert not any(event.payload.get("manual_recovery") is True for event in persisted)
    restored_checkpoint = asyncio.run(store.load_checkpoint(session_id))
    original_profile = (
        execution_profiles_module.active_invocation_execution_profile_from_checkpoint(checkpoint)
    )
    restored_profile = (
        execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
            restored_checkpoint
        )
    )
    assert original_profile is not None
    assert restored_profile is not None
    assert restored_profile.session_id == original_profile.session_id
    assert restored_profile.interaction_id == original_profile.interaction_id
    assert restored_profile.profile == original_profile.profile
    assert restored_profile.run_epoch > original_profile.run_epoch
    assert checkpoint_without_active_invocation_profile(
        restored_checkpoint
    ) == checkpoint_without_active_invocation_profile(checkpoint)
    assert tool.calls == [{}]


@pytest.mark.parametrize(
    "grouped_cancellation",
    [False, True],
    ids=["ordinary-error", "grouped-cancellation"],
)
def test_tool_round_recovery_post_persist_fanout_failure_stays_resumable(
    grouped_cancellation: bool,
) -> None:
    failure_kind = "grouped" if grouped_cancellation else "ordinary"
    session_id = f"sess_tool_round_post_persist_failure_{failure_kind}"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]
    original_fan_out = app._event_writer.fan_out_persisted
    failed = {"value": False}
    fan_out_failure: BaseException = (
        BaseExceptionGroup(
            "manual recovery fan-out cancelled and failed",
            [asyncio.CancelledError("fan-out cancelled"), RuntimeError("fan-out failed")],
        )
        if grouped_cancellation
        else RuntimeError("manual recovery fan-out unavailable")
    )

    async def fail_recovery_fan_out(events: list[Event]) -> list[Event]:
        if not failed["value"] and any(
            event.payload.get("manual_recovery") is True for event in events
        ):
            failed["value"] = True
            raise fan_out_failure
        return await original_fan_out(events)

    app._event_writer.fan_out_persisted = fail_recovery_fan_out
    recovery_request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=round_id,
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )
    recovery_events: list[Event] = []
    if grouped_cancellation:
        with pytest.raises(BaseExceptionGroup) as raised:
            asyncio.run(collect_tool_round_recovery_events(app, recovery_request))
        assert raised.value is fan_out_failure
    else:
        recovery_events = asyncio.run(
            collect_tool_round_recovery_events(
                app,
                recovery_request,
            )
        )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    persisted = asyncio.run(store.load_events(session_id))
    terminal = [event for event in persisted if event.type == EventType.SESSION_INTERRUPTED][-1]
    if grouped_cancellation:
        assert terminal.payload.get("abandoned") is not True
    else:
        records = asyncio.run(store.query_events(EventQuery(session_id=session_id, limit=100)))
        assert (
            _private_record_for_public_event(records, recovery_events[-1]).event.id == terminal.id
        )
        assert terminal.payload["manual_recovery_persisted"] is True
    recovered = [
        event
        for event in persisted
        if event.payload.get("manual_recovery") is True
        and event.payload.get("tool_call_id") == "call_1"
    ]
    assert len(recovered) == 1
    assert tool.calls == [{}]

    with pytest.raises(RuntimeError, match="already has a terminal event"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="do not duplicate",
                ),
            )
        )

    resumed = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )
    assert resumed[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{}]


def test_tool_round_recovery_post_persist_failure_preserves_operator_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OperatorInterruptStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.operator_interrupt_committed: asyncio.Event | None = None

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
            if (
                to_status == SessionStatus.INTERRUPTING
                and self.operator_interrupt_committed is not None
            ):
                self.operator_interrupt_committed.set()
            return session

    session_id = "sess_tool_round_post_persist_operator_interrupt"
    store = OperatorInterruptStore()
    app, _store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]
    operator_app = CayuApp(session_store=store, enable_logging=False)
    operator_app.register_provider(FakeProvider([]), default=True)
    operator_app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        original_fan_out = app._event_writer.fan_out_persisted
        original_resumable_interrupt = (
            app._recovery_coordinator._interrupt_for_resumable_manual_recovery
        )
        failed = False
        interrupt_task: asyncio.Task[list[Event]] | None = None
        store.operator_interrupt_committed = asyncio.Event()
        resumable_interrupt_started = asyncio.Event()
        release_resumable_interrupt = asyncio.Event()

        async def block_resumable_interrupt(*args, **kwargs):
            resumable_interrupt_started.set()
            await release_resumable_interrupt.wait()
            async for event in original_resumable_interrupt(*args, **kwargs):
                yield event

        async def fail_after_operator_interrupt(events: list[Event]) -> list[Event]:
            nonlocal failed, interrupt_task
            if not failed and any(event.payload.get("manual_recovery") is True for event in events):
                failed = True
                interrupt_task = asyncio.create_task(
                    collect_interrupt_events(
                        operator_app,
                        InterruptSessionRequest(
                            session_id=session_id,
                            reason="operator requested stop",
                            metadata={"ticket": "OPS-1"},
                        ),
                    )
                )
                await asyncio.wait_for(store.operator_interrupt_committed.wait(), timeout=5)
                raise RuntimeError("manual recovery fan-out unavailable")
            return await original_fan_out(events)

        app._event_writer.fan_out_persisted = fail_after_operator_interrupt
        monkeypatch.setattr(
            app._recovery_coordinator,
            "_interrupt_for_resumable_manual_recovery",
            block_resumable_interrupt,
        )
        monkeypatch.setattr(
            recovery_ownership_module,
            "_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS",
            0.01,
        )
        try:
            recovery_events = await asyncio.wait_for(
                collect_tool_round_recovery_events(
                    app,
                    ToolRoundRecoveryRequest(
                        session_id=session_id,
                        round_id=round_id,
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="side effect verified externally",
                    ),
                ),
                timeout=5,
            )
        finally:
            release_resumable_interrupt.set()
        assert resumable_interrupt_started.is_set()
        assert interrupt_task is not None
        interrupt_events = await asyncio.wait_for(interrupt_task, timeout=5)

        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert recovery_events[-1].id == interrupt_events[-1].id
        assert sum(event.id == interrupt_events[-1].id for event in recovery_events) == 1
        assert recovery_events[-1].payload["interruption_type"] == "operator_requested"
        assert recovery_events[-1].payload["reason"] == "operator requested stop"
        assert recovery_events[-1].payload["metadata"] == {"ticket": "OPS-1"}
        persisted = await store.load_events(session_id)
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in persisted) == 1
        assert await app.drain_background_interruptions(timeout_s=1) is True
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_session_interrupt" not in checkpoint
        assert "pending_interruption_cascade" not in checkpoint
        assert "pending_tool_round" in checkpoint
        assert tool.calls == [{}]

    asyncio.run(scenario())
