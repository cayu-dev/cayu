"""Tool policy contracts."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from tests.runtime.runtime_environment_support import MemoryWorkspace
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailingTerminalToolEventStore
from tests.runtime.runtime_session_support import (
    _expected_failure_evidence,
    _private_events_for_public_events,
    _taint_aware_policy,
    collect_events,
    collect_resume_events,
    collect_tool_round_recovery_events,
    tool_round_identity_payload,
)
from tests.runtime.runtime_tool_support import (
    EchoTool,
    SideEffectTool,
    _ProtectedEmailTool,
    _TaintSourceTool,
)

import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.proxies.base import CredentialProxy
from cayu.proxies.passthrough import PassthroughProxy
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime.execution_profiles import EXECUTION_PROFILE_FINGERPRINT_FIELD
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec, _bound_policy_denial_text
from cayu.tools.policy import (
    AllowlistRule,
    GuardedToolPolicy,
    ParameterConstrainedToolPolicy,
    RequiredAllowlistRule,
    StaticToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults.static import StaticVault


@pytest.mark.parametrize("effect", ["none", "external"])
def test_cayu_app_recover_tool_round_taints_follow_up_rounds(effect):
    from cayu.tools.base import ToolEffect

    class RecordingTaintSourceTool(_TaintSourceTool):
        # The NONE variant computes a fixed untrusted result for manual recovery;
        # the EXTERNAL variant retains the production-default receipt boundary.
        spec = _TaintSourceTool.spec.model_copy(update={"effect": ToolEffect(effect)})

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.calls += 1
            return await super().run(ctx, args)

    session_id = "sess_tool_round_manual_taint"
    store = FailingTerminalToolEventStore()
    source_tool = RecordingTaintSourceTool()
    email_tool = _ProtectedEmailTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_read_web", name="read_web", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "page summary"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[source_tool, email_tool],
        tool_policy=_taint_aware_policy(),
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "read the page, then email it")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint

    if effect == "none":
        recovery_events = asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_read_web",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="untrusted page content",
                ),
            )
        )
    else:
        # The EXTERNAL result was already selected before publication failed.
        # Repair that evidence instead of overwriting it with a manual outcome.
        recovery_events = asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(session_id=session_id, messages=[Message.text("user", "continue")]),
            )
        )

    # Either recovered read_web completion still taints the session: the
    # follow-up round's protected send_email is denied, never executed.
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    blocked = [event for event in recovery_events if event.type == EventType.TOOL_CALL_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["tool_call_id"] == f"{blocked[0].id}:tool_call_id"
    assert email_tool.calls == []
    assert source_tool.calls == 1
    durable_events = asyncio.run(store.load_events(session_id))
    assert sum(event.type == EventType.TOOL_CALL_COMPLETED for event in durable_events) == 1


def test_cayu_app_blocks_tool_call_before_execution_with_tool_policy():
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
                ModelStreamEvent.text_delta("blocked handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=StaticToolPolicy(deny=["side_effect"]),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blocked_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == []
    blocked_event = events[4]
    assert blocked_event.tool_name == "side_effect"
    private_events = asyncio.run(
        _private_events_for_public_events(store, "sess_blocked_tool", events)
    )
    private_blocked = private_events[4]
    assert private_blocked.payload["tool_round_id"] == private_events[3].payload["tool_round_id"]
    timing_keys = (
        "tool_effect_completed_at",
        "tool_terminal_staged_at",
        "tool_terminal_publication_started_at",
    )
    timings = {key: private_blocked.payload[key] for key in timing_keys}
    timestamps = [datetime.fromisoformat(value) for value in timings.values()]
    assert all(value.tzinfo is not None for value in timestamps)
    assert timestamps == sorted(timestamps)
    assert private_blocked.payload == {
        **timings,
        "tool_call_id": "call_1",
        "idempotency_key": private_blocked.payload["idempotency_key"],
        **tool_round_identity_payload(private_blocked),
        "tool_name": "side_effect",
        "denied_by": "tool_policy",
        "decision": "deny",
        "reason": "Tool denied by policy: side_effect",
        "metadata": {},
        "argument_presence": {"keys": [], "key_count": 1, "unlisted_key_count": 1},
        "arguments_state": "unavailable",
        "arguments_exact": False,
        EXECUTION_PROFILE_FINGERPRINT_FIELD: private_blocked.payload[
            EXECUTION_PROFILE_FINGERPRINT_FIELD
        ],
        "result": {
            "content": "Tool denied by policy: side_effect",
            "structured": {
                "decision": "deny",
                "reason": "Tool denied by policy: side_effect",
                "metadata": {},
            },
            "artifacts": [],
            "is_error": True,
        },
    }

    assert provider.requests[1].messages[-1].role == "tool"
    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.tool_call_id == "call_1"
    assert tool_result_part.tool_name == "side_effect"
    assert tool_result_part.content == "Tool denied by policy: side_effect"
    assert tool_result_part.is_error is True


def test_cayu_app_blocks_tool_call_before_execution_with_parameter_policy():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "external"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("blocked handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ParameterConstrainedToolPolicy(
            {"side_effect": [AllowlistRule("value", values=["internal"])]}
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parameter_blocked_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert tool.calls == []
    blocked_event = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked_event.payload["denied_by"] == "tool_policy"
    assert blocked_event.payload["decision"] == "deny"
    assert blocked_event.payload["tool_name"] == "side_effect"
    assert blocked_event.payload["reason"] == "Parameter 'value' value is not allowed."
    assert blocked_event.payload["metadata"] == {
        "policy": "parameter_constrained",
        "tool_name": "side_effect",
        "parameter": "value",
        "rule": "AllowlistRule",
        "rule_index": 0,
    }
    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.content == "Parameter 'value' value is not allowed."
    assert tool_result_part.structured["metadata"]["policy"] == "parameter_constrained"
    assert tool_result_part.is_error is True


def test_cayu_app_required_allowlist_blocks_omitted_argument_before_execution():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("blocked handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ParameterConstrainedToolPolicy(
            {"side_effect": [RequiredAllowlistRule("value", values=["internal"])]}
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_required_allowlist_blocked_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert tool.calls == []
    blocked_event = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked_event.payload["reason"] == ("Required allowlisted parameter 'value' is missing.")
    assert blocked_event.payload["metadata"] == {
        "policy": "parameter_constrained",
        "tool_name": "side_effect",
        "parameter": "value",
        "rule": "RequiredAllowlistRule",
        "rule_index": 0,
        "violation": "missing",
    }
    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.content == "Required allowlisted parameter 'value' is missing."
    assert tool_result_part.structured["metadata"] == blocked_event.payload["metadata"]
    assert tool_result_part.is_error is True


def test_cayu_app_required_allowlist_executes_allowed_argument():
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "internal"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("allowed handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=InMemorySessionStore())
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ParameterConstrainedToolPolicy(
            {"side_effect": [RequiredAllowlistRule("value", values=["internal"])]}
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_required_allowlist_allowed_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert tool.calls == [{"value": "internal"}]
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)
    assert all(event.type != EventType.TOOL_CALL_BLOCKED for event in events)


def test_cayu_app_required_allowlist_can_request_durable_approval():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1", name="side_effect", arguments={"value": "external"}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ParameterConstrainedToolPolicy(
            {"side_effect": [RequiredAllowlistRule("value", values=["internal"])]},
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_required_allowlist_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert tool.calls == []
    approval_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = approval_event.payload["approval"]
    assert approval["reason"] == "Required allowlisted parameter 'value' value is not allowed."
    assert approval["metadata"] == {
        "policy": "parameter_constrained",
        "tool_name": "side_effect",
        "parameter": "value",
        "rule": "RequiredAllowlistRule",
        "rule_index": 0,
        "violation": "disallowed_value",
    }
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert events[-1].payload["interruption_type"] == "tool_approval_required"
    checkpoint = asyncio.run(store.load_checkpoint("sess_required_allowlist_approval"))
    assert checkpoint is not None
    private_approval_event = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_required_allowlist_approval",
            [approval_event],
        )
    )[0]
    assert (
        checkpoint["pending_tool_approval"]["approval_id"]
        == private_approval_event.payload["approval_id"]
    )


def test_cayu_app_tool_policy_allowlist_blocks_unlisted_registered_tools():
    store = InMemorySessionStore()
    blocked_tool = SideEffectTool()
    allowed_tool = EchoTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "blocked"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[allowed_tool, blocked_tool],
        tool_policy=StaticToolPolicy(allow=["echo"]),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_allowlist_tool_policy",
                messages=[Message.text("user", "use the blocked tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_BLOCKED
    assert events[4].payload["reason"] == "Tool not allowed by policy: side_effect"
    assert blocked_tool.calls == []


def test_cayu_app_tool_policy_receives_copied_tool_call_arguments():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            request.arguments["nested"]["value"] = "mutated"
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"nested": {"value": "original"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_argument_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].session.id == "sess_policy_argument_copy"
    assert policy.requests[0].agent.name == "assistant"
    assert policy.requests[0].tool_name == "side_effect"
    assert policy.requests[0].tool_call_id == "call_1"
    assert policy.requests[0].environment_name is None
    assert policy.requests[0].workspace_id is None
    assert policy.requests[0].metadata == {}
    assert tool.calls == [{"nested": {"value": "original"}}]


def test_cayu_app_tool_policy_receives_workspace_identity():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
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
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=MemoryWorkspace("workspace_1"),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_workspace",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].environment_name == "local"
    assert policy.requests[0].workspace_id == "workspace_1"


def test_cayu_app_wires_environment_proxy_into_tool_context():
    class ProxyProbeTool(Tool):
        spec = ToolSpec(
            name="probe_proxy",
            description="Record proxy identity.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            super().__init__()
            self.proxies: list[object] = []

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.proxies.append(ctx.proxy)
            return ToolResult(content="ok")

    vault = StaticVault({"api_key": "sk-secret-123"})
    proxy = PassthroughProxy(vault)
    tool = ProxyProbeTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="probe_proxy",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=proxy,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_context_proxy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(tool.proxies) == 1
    assert isinstance(tool.proxies[0], CredentialProxy)


def test_cayu_app_tool_policy_receives_run_request_metadata_copy():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            request.metadata["tenant"]["id"] = "mutated"
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
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
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_run_metadata",
                messages=[Message.text("user", "use the tool")],
                metadata={"tenant": {"id": "tenant_1"}},
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].metadata == {"tenant": {"id": "mutated"}}
    private_tool_started = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_policy_run_metadata",
            [events[3]],
        )
    )[0]
    assert tool.contexts[0].metadata["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_policy_run_metadata",
        tool_round_id=private_tool_started.payload["tool_round_id"],
        tool_call_id="call_1",
    )
    assert tool.contexts[0].metadata == {
        "tenant": {"id": "tenant_1"},
        "tool_call_id": "call_1",
        "idempotency_key": tool.contexts[0].metadata["idempotency_key"],
        "tool_effect": "external",
        EXECUTION_PROFILE_FINGERPRINT_FIELD: private_tool_started.payload[
            EXECUTION_PROFILE_FINGERPRINT_FIELD
        ],
    }
    tool.contexts[0].metadata["tenant"]["id"] = "tool-mutated"
    session = asyncio.run(app.session_store.load("sess_policy_run_metadata"))
    assert session is not None
    assert sessions_module.session_user_metadata(session.metadata) == {"tenant": {"id": "tenant_1"}}


@pytest.mark.parametrize("guard_depth", [0, 1, 2])
def test_cayu_app_taint_policy_requires_approval_from_prior_durable_tool_result(guard_depth):
    store = InMemorySessionStore()
    echo_tool = EchoTool()
    side_effect_tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_read",
                    name="echo",
                    arguments={"text": "external email body"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("stored"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    policy = TaintAwareToolPolicy(
        taint_sources={"echo": ["external_email"]},
        protected_tools={"side_effect": ["external_email"]},
    )
    for _ in range(guard_depth):
        policy = GuardedToolPolicy(guards=(), then=policy)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[echo_tool, side_effect_tool],
        tool_policy=policy,
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_taint_prior",
                messages=[Message.text("user", "read external data")],
            ),
        )
    )
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_taint_prior",
                messages=[Message.text("user", "send outbound action")],
            ),
        )
    )

    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    approval_event = next(
        event for event in resume_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = approval_event.payload["approval"]
    assert approval["tool_name"] == "side_effect"
    assert approval["metadata"]["policy"] == "taint_aware"
    assert approval["metadata"]["matched_taint_labels"] == ["external_email"]
    assert side_effect_tool.calls == []


@pytest.mark.parametrize("guard_depth", [0, 1, 2])
def test_cayu_app_taint_policy_applies_within_same_tool_round(guard_depth):
    echo_tool = EchoTool()
    side_effect_tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_read",
                    name="echo",
                    arguments={"text": "external email body"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_send",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    policy = TaintAwareToolPolicy(
        taint_sources={"echo": ["external_email"]},
        protected_tools={"side_effect": ["external_email"]},
    )
    for _ in range(guard_depth):
        policy = GuardedToolPolicy(guards=(), then=policy)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[echo_tool, side_effect_tool],
        tool_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_taint_same_round",
                messages=[Message.text("user", "read then send")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    approval_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = approval_event.payload["approval"]
    assert approval["tool_name"] == "side_effect"
    assert approval["metadata"]["policy"] == "taint_aware"
    assert approval["metadata"]["matched_taint_labels"] == ["external_email"]
    assert side_effect_tool.calls == []


def test_cayu_app_tool_policy_receives_resume_request_metadata_copy():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            request.metadata["resume"]["id"] = "mutated"
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
    tool = SideEffectTool()
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("ready"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_resume_metadata",
                messages=[Message.text("user", "first")],
                metadata={"original": {"id": "run"}},
            ),
        )
    )
    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_policy_resume_metadata",
                messages=[Message.text("user", "use the tool")],
                metadata={"resume": {"id": "resume_1"}},
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].metadata == {"resume": {"id": "mutated"}}
    private_tool_started = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_policy_resume_metadata",
            [events[3]],
        )
    )[0]
    assert tool.contexts[0].metadata["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_policy_resume_metadata",
        tool_round_id=private_tool_started.payload["tool_round_id"],
        tool_call_id="call_1",
    )
    assert tool.contexts[0].metadata == {
        "resume": {"id": "resume_1"},
        "tool_call_id": "call_1",
        "idempotency_key": tool.contexts[0].metadata["idempotency_key"],
        "tool_effect": "external",
        EXECUTION_PROFILE_FINGERPRINT_FIELD: private_tool_started.payload[
            EXECUTION_PROFILE_FINGERPRINT_FIELD
        ],
    }
    tool.contexts[0].metadata["resume"]["id"] = "tool-mutated"
    session = asyncio.run(store.load("sess_policy_resume_metadata"))
    assert session is not None
    assert sessions_module.session_user_metadata(session.metadata) == {"original": {"id": "run"}}


def test_cayu_app_tool_policy_blocked_event_uses_default_reason_when_omitted():
    class ReasonlessDenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY)

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
                ModelStreamEvent.text_delta("handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=ReasonlessDenyPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_default_reason",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    blocked_event = events[4]
    assert blocked_event.type == EventType.TOOL_CALL_BLOCKED
    assert blocked_event.payload["reason"] == "Tool call denied by policy."
    assert blocked_event.payload["result"]["content"] == "Tool call denied by policy."


def test_cayu_app_fails_session_when_tool_policy_raises_before_execution():
    class FailingToolPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            raise RuntimeError("policy unavailable")

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
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=FailingToolPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_raises",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert tool.calls == []
    assert events[-1].payload == {
        "failure_evidence": _expected_failure_evidence("sess_policy_raises", "RuntimeError"),
        "error": "policy unavailable",
        "error_type": "RuntimeError",
    }


def test_cayu_app_fails_session_when_tool_policy_returns_invalid_result():
    class InvalidToolPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return {"decision": "deny"}  # type: ignore[return-value]

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
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=InvalidToolPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_invalid_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == []
    assert events[-1].payload == {
        "failure_evidence": _expected_failure_evidence("sess_policy_invalid_result", "TypeError"),
        "error": "Tool policies must return ToolPolicyResult instances. Received dict.",
        "error_type": "TypeError",
    }


# --------------------------------------------------------------------------- #
# Pending tool approvals persist the original run's config (limits, budgets,
# max_steps, retry policy) so resolving does not restart with fresh defaults.
# --------------------------------------------------------------------------- #
def test_policy_denial_checkpoint_reasons_are_bounded_without_changing_approval_prompts():
    from cayu.approvals.tools import PendingToolCallApproval
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime import _resume_ledger as resume_ledger
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.vaults.redaction import SecretRedactor

    secret = "BOUNDARY_SECRET_checkpoint_value"
    oversized = "a" * 4050 + secret
    approval_prompt = "approval prompt 🙂 " * 500
    redactor = SecretRedactor(secret)
    denied_call = runtime_records.ToolCallRequest(id="call_deny", name="echo", arguments={})
    approval_call = runtime_records.ToolCallRequest(
        id="call_approval", name="side_effect", arguments={}
    )
    outcomes = [
        runtime_records.ToolCallPolicyOutcome(
            call=denied_call,
            result=ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason=oversized),
            evidence=runtime_records.ToolPolicyEvidence.AUTHORITATIVE,
        ),
        runtime_records.ToolCallPolicyOutcome(
            call=approval_call,
            result=ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason=approval_prompt,
            ),
            evidence=runtime_records.ToolPolicyEvidence.AUTHORITATIVE,
        ),
    ]
    expected_denial = _bound_policy_denial_text(redactor.redact_text(oversized))

    pending_round = tool_round_recovery.pending_tool_call_records(
        tool_calls=[denied_call, approval_call],
        policy_outcomes=outcomes,
        redactor=redactor,
    )
    pending_approval = approval_support.pending_tool_call_approvals(
        tool_calls=[denied_call, approval_call],
        policy_outcomes=outcomes,
        redactor=redactor,
    )

    for records in (pending_round, pending_approval):
        assert records[0].reason == expected_denial
        assert "BOUNDARY" not in records[0].reason
        assert records[1].reason == approval_prompt

    # Legacy checkpoints remain loadable without migration; their denial is
    # bounded when it reaches the canonical durable/model-facing boundary.
    historical = PendingToolCallApproval(
        tool_call_id="call_historical",
        tool_name="echo",
        policy_decision=ToolPolicyDecision.DENY.value,
        reason=oversized,
    )
    restored = resume_ledger.policy_result_from_pending_tool_call(historical)
    assert restored is not None
    assert restored.reason == oversized
    assert tool_execution.policy_denial_reason(restored) == oversized
