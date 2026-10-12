"""Approval results contracts."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.runtime_budget_support import fake_budget_limit
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _pending_tool_approval_from_public_event,
    _private_events_for_public_events,
    _private_record_for_public_event,
    assert_only_model_step_publication_checkpoint,
    collect_events,
    collect_tool_approval_events,
)
from tests.runtime.runtime_tool_support import (
    EchoTool,
    RequireApprovalPolicy,
    SideEffectApprovalPolicy,
    SideEffectTool,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.budgets.run_limits import RunLimits
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME, StructuredOutputSpec
from cayu.events import EventType
from cayu.messages import Message
from cayu.observability.hooks import RuntimeHook, ToolCallHookContext
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools.base import ToolResult
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult


class FailingApprovalCloseStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed_close_once = False

    async def publish_runtime_publication(self, session_id: str, **kwargs):
        request = kwargs["request"]
        if not self.failed_close_once and request.kind == "approval-close":
            self.failed_close_once = True
            raise RuntimeError("approval close unavailable")
        return await super().publish_runtime_publication(session_id, **kwargs)


class DenyEchoRequireSideEffectApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "echo":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="echo is blocked",
            )
        if request.tool_name == "side_effect":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="side effect needs approval",
            )
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def test_cayu_app_preserves_structured_output_across_tool_approval():
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
                ModelStreamEvent.tool_call(
                    id="call_final",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "approved"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
    structured_output = StructuredOutputSpec(
        name="approval_answer",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_structured_output",
                messages=[Message.text("user", "use the tool")],
                structured_output=structured_output,
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]
    approval_id = approval["approval_id"]
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_approval_structured_output"))

    assert checkpoint is not None
    assert checkpoint["pending_tool_approval"]["structured_output"]["name"] == ("approval_answer")

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_structured_output",
                approval_id=approval_id,
                tool_round_id=approval["tool_round_id"],
                tool_call_id=approval["tool_call_id"],
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
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[9].payload["output"] == {"answer": "approved"}
    assert provider.requests[1].options["structured_output"]["name"] == "approval_answer"


def test_cayu_app_rejects_conflicting_structured_output_on_tool_approval():
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
                ModelStreamEvent.text_delta('{"answer":"approved"}'),
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
                session_id="sess_tool_approval_structured_output_conflict",
                messages=[Message.text("user", "use the tool")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                ),
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]
    approval_id = approval["approval_id"]

    with pytest.raises(ValueError, match="does not match the pending run contract"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_tool_approval_structured_output_conflict",
                    approval_id=approval_id,
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    structured_output=StructuredOutputSpec(
                        json_schema={
                            "type": "object",
                            "properties": {"different": {"type": "string"}},
                            "required": ["different"],
                        },
                    ),
                ),
            )
        )

    session = asyncio.run(store.load("sess_tool_approval_structured_output_conflict"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert tool.calls == []


def test_cayu_app_rejects_budget_limit_drift_before_approval_side_effects():
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
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
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
                session_id="sess_tool_approval_cost_limit",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]
    approval_id = approval["approval_id"]

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_tool_approval_cost_limit",
                    approval_id=approval_id,
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    budget_limits=(fake_budget_limit("0.002"),),
                ),
            )
        )

    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
    )
    assert tool.calls == []
    assert len(provider.requests) == 1

    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_approval_cost_limit"))
    assert checkpoint is not None and "pending_tool_approval" in checkpoint
    transcript = asyncio.run(store.load_transcript("sess_tool_approval_cost_limit"))
    assert [message.role for message in transcript] == ["user"]
    session = asyncio.run(store.load("sess_tool_approval_cost_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_rejects_limit_drift_with_one_executable_pending_tool():
    store = InMemorySessionStore()
    side_effect = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "first"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"value": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("round handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool(), side_effect],
        tool_policy=DenyEchoRequireSideEffectApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_limit_executable_only",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_approval_limit_executable_only",
                    approval_id=approval["approval_id"],
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    limits=RunLimits(max_tool_calls=1),
                ),
            )
        )

    assert caught.value.changed_component_classes == (ExecutionProfileComponentClass.FINALIZATION,)
    assert side_effect.calls == []
    assert len(provider.requests) == 1


def test_cayu_app_resolves_approved_multi_tool_round_in_order():
    store = InMemorySessionStore()
    side_effect = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "first"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"value": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("round handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool(), side_effect],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_tool_approval",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    private_approval_event = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_multi_tool_approval",
            [interrupt_events[4]],
        )
    )[0]
    approval = interrupt_events[4].payload["approval"]
    private_approval = private_approval_event.payload["approval"]
    assert [call["tool_call_id"] for call in private_approval["tool_calls"]] == [
        "call_1",
        "call_2",
    ]
    assert side_effect.calls == []

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_multi_tool_approval",
                approval_id=approval["approval_id"],
                tool_round_id=approval["tool_round_id"],
                tool_call_id=approval["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    tool_lifecycle = {
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
    }
    assert [event.type for event in events if event.type not in tool_lifecycle] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert side_effect.calls == [{"value": "second"}]

    private_tool_events = asyncio.run(
        _private_events_for_public_events(
            store, "sess_multi_tool_approval", [e for e in events if e.type in tool_lifecycle]
        )
    )
    assert len(private_tool_events) == 6
    for call_id in ("call_1", "call_2"):
        assert [e.type for e in private_tool_events if e.payload["tool_call_id"] == call_id] == [
            EventType.TOOL_CALL_APPROVED,
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_COMPLETED,
        ]

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert [part.tool_call_id for part in tool_result_message.content] == [
        "call_1",
        "call_2",
    ]
    assert [part.content for part in tool_result_message.content] == [
        "first",
        "recorded",
    ]


def test_cayu_app_resolves_denied_tool_call_and_continues_session():
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
                ModelStreamEvent.text_delta("denial handled"),
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
                session_id="sess_tool_approval_deny",
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
                session_id="sess_tool_approval_deny",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    private_resolution_events = asyncio.run(
        _private_events_for_public_events(store, "sess_tool_approval_deny", events)
    )
    assert private_resolution_events[1].payload[
        "idempotency_key"
    ] == tool_execution.tool_idempotency_key(
        session_id="sess_tool_approval_deny",
        tool_round_id=private_resolution_events[1].payload["tool_round_id"],
        tool_call_id="call_1",
        approval_id=private_approval.approval_id,
    )
    assert tool.calls == []
    assert provider.requests[1].messages[-1].role == "tool"
    tool_result = provider.requests[1].messages[-1].content[0]
    assert tool_result.type == "tool_result"
    assert tool_result.content == "Tool call denied by approval: not safe"
    assert tool_result.structured["denied_by_approval"] is True
    assert tool_result.structured["skipped_due_to_approval_denial"] is False
    assert tool_result.structured["tool_call_id"] == "call_1"
    assert tool_result.structured["tool_name"] == "side_effect"
    assert tool_result.is_error is True


def test_cayu_app_after_tool_call_hook_observes_approval_denial_result():
    class ApprovalDenialHook(RuntimeHook):
        def __init__(self) -> None:
            self.tool_event_type: EventType | str | None = None
            self.result: ToolResult | None = None

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.tool_event_type = context.tool_event.type
            self.result = context.result

    store = InMemorySessionStore()
    hook = ApprovalDenialHook()
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
                ModelStreamEvent.text_delta("denial handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
        runtime_hooks=[hook],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_deny_hook",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_deny_hook",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert hook.tool_event_type == EventType.TOOL_CALL_APPROVAL_DENIED
    assert hook.result is not None
    assert hook.result.content == "Tool call denied by approval: not safe"
    assert hook.result.is_error is True
    # after_tool_call runs before the result event is persisted, so the denial event lands last.
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.TOOL_CALL_APPROVAL_DENIED,
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        }
    ] == [
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    ]


def test_cayu_app_denied_multi_tool_round_marks_skipped_calls_explicitly():
    store = InMemorySessionStore()
    side_effect = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "first"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"value": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("denied round handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool(), side_effect],
        tool_policy=SideEffectApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_tool_approval_deny",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_multi_tool_approval_deny",
                approval_id=approval["approval_id"],
                tool_round_id=approval["tool_round_id"],
                tool_call_id=approval["tool_call_id"],
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert side_effect.calls == []

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert [part.tool_call_id for part in tool_result_message.content] == [
        "call_1",
        "call_2",
    ]

    skipped_result = tool_result_message.content[0]
    assert skipped_result.content == (
        "Tool call skipped because approval was denied for the same tool round: not safe"
    )
    assert skipped_result.structured["denied_by_approval"] is False
    assert skipped_result.structured["skipped_due_to_approval_denial"] is True
    assert skipped_result.structured["denied_tool_call_id"] == "call_2"
    assert skipped_result.structured["denied_tool_name"] == "side_effect"

    denied_result = tool_result_message.content[1]
    assert denied_result.content == "Tool call denied by approval: not safe"
    assert denied_result.structured["denied_by_approval"] is True
    assert denied_result.structured["skipped_due_to_approval_denial"] is False
    assert denied_result.structured["tool_call_id"] == "call_2"
    assert denied_result.structured["tool_name"] == "side_effect"


def test_cayu_app_keeps_pending_approval_if_atomic_resolution_close_fails():
    store = FailingApprovalCloseStore()
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
                ModelStreamEvent.text_delta("resumed after retry"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
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
                session_id="sess_approval_atomic_close_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_atomic_close_failure",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[-1].payload["interruption_type"] == "tool_approval_required"
    assert events[-1].payload["approval"]["approval_id"] == f"{events[-1].id}:approval_id"
    assert events[-1].payload["error"] == "approval close unavailable"

    session = asyncio.run(store.load("sess_approval_atomic_close_failure"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    checkpoint = asyncio.run(store.load_checkpoint("sess_approval_atomic_close_failure"))
    assert checkpoint is not None
    private_approval = asyncio.run(_pending_tool_approval_from_public_event(store, approval_event))
    assert checkpoint["pending_tool_approval"]["approval_id"] == private_approval.approval_id

    transcript = asyncio.run(store.load_transcript("sess_approval_atomic_close_failure"))
    assert [message.role for message in transcript] == ["user"]

    retry_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_atomic_close_failure",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert retry_events[-1].type == EventType.SESSION_COMPLETED
    assert_only_model_step_publication_checkpoint(
        asyncio.run(store.load_checkpoint("sess_approval_atomic_close_failure"))
    )
    assert provider.requests[1].messages[-2].role == "assistant"
    assert provider.requests[1].messages[-1].role == "tool"


def test_cayu_app_retries_approval_close_without_rerunning_completed_tool():
    store = FailingApprovalCloseStore()
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
                ModelStreamEvent.text_delta("resumed after retry"),
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
                session_id="sess_approval_approved_close_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_approved_close_failure",
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
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"value": "secret"}]

    retry_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_approved_close_failure",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in retry_events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == [{"value": "secret"}]
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].content == "recorded"


def test_tool_approval_close_acknowledgement_loss_replays_exact_receipt():
    class LostCloseAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lost_close_ack = False

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            result = await super().publish_runtime_publication(session_id, **kwargs)
            if not self.lost_close_ack and kwargs["request"].kind == "approval-close":
                self.lost_close_ack = True
                raise OSError("approval close acknowledgement lost")
            return result

    async def run() -> None:
        session_id = "sess_approval_close_ack_loss"
        store = LostCloseAcknowledgementStore()
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
        interrupted = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event for event in interrupted if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = await _pending_tool_approval_from_public_event(store, approval_event)
        request = ToolApprovalRequest(
            session_id=session_id,
            approval_id=approval.approval_id,
            tool_round_id=approval.tool_round_id,
            tool_call_id=approval.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
        )
        resolved = await collect_tool_approval_events(app, request)
        assert resolved[-1].type is EventType.SESSION_COMPLETED
        assert store.lost_close_ack is True
        assert tool.calls == [{"value": "secret"}]
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_tool_approval" not in checkpoint
        assert "pending_tool_round" not in checkpoint
        receipt = await store.load_runtime_publication_receipt(
            session_id,
            f"approval-close:{approval.approval_id}",
        )
        assert receipt is not None
        assert receipt.kind == "approval-close"

        replayed = await collect_tool_approval_events(app, request)
        assert [event.type for event in replayed] == [EventType.SESSION_CHECKPOINTED]
        records = await store.query_events(EventQuery(session_id=session_id))
        assert (
            _private_record_for_public_event(records, replayed[0]).event.id
            == (receipt.appended_event_ids[0])
        )
        assert tool.calls == [{"value": "secret"}]

        with pytest.raises(RuntimeError, match="conflicting identity or decision"):
            await collect_tool_approval_events(
                app,
                request.model_copy(update={"decision": ToolApprovalDecision.DENY}),
            )

        close_events = [
            event
            for event in await store.load_events(session_id)
            if event.id in receipt.appended_event_ids
        ]
        assert len(close_events) == 1
        tool_messages = [
            message for message in await store.load_transcript(session_id) if message.role == "tool"
        ]
        assert len(tool_messages) == 1

    asyncio.run(run())


def test_cayu_app_approval_retry_rejects_budget_drift_after_recorded_tool_outcome():
    store = FailingApprovalCloseStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
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
                session_id="sess_approval_recorded_outcome_limit",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_recorded_outcome_limit",
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
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"value": "secret"}]

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_approval_recorded_outcome_limit",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    budget_limits=(fake_budget_limit("0.001"),),
                ),
            )
        )

    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
    )
    assert tool.calls == [{"value": "secret"}]
    assert len(provider.requests) == 1
    checkpoint = asyncio.run(store.load_checkpoint("sess_approval_recorded_outcome_limit"))
    assert checkpoint is not None
    assert "pending_tool_approval" in checkpoint
    assert "pending_tool_round" in checkpoint

    transcript = asyncio.run(store.load_transcript("sess_approval_recorded_outcome_limit"))
    assert [message.role for message in transcript] == ["user"]
    assert sum(event.type == EventType.TOOL_CALL_COMPLETED for event in events) == 1


def test_cayu_app_rejects_denial_retry_after_approved_tool_executed():
    store = FailingApprovalCloseStore()
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
                session_id="sess_approval_reject_conflicting_deny",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = interrupt_events[4]
    approval_id = approval_event.payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_reject_conflicting_deny",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert tool.calls == [{"value": "secret"}]

    retry_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_reject_conflicting_deny",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.DENY,
                reason="changed mind",
            ),
        )
    )

    assert [event.type for event in retry_events] == [EventType.SESSION_INTERRUPTED]
    assert "cannot be retried as denied" in retry_events[0].payload["error"]
    assert tool.calls == [{"value": "secret"}]
    assert (
        asyncio.run(store.load_checkpoint("sess_approval_reject_conflicting_deny"))[
            "pending_tool_approval"
        ]["approval_id"]
        == asyncio.run(_pending_tool_approval_from_public_event(store, approval_event)).approval_id
    )
