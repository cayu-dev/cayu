"""Tool hooks contracts."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events
from tests.runtime.runtime_tool_hook_support import (
    _RecordingEchoTool,
    _RewriteTextHook,
    _run_reauth_case,
    _tool_call_provider,
)
from tests.runtime.runtime_tool_support import EchoTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message, ToolCallPart
from cayu.observability.hooks import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    RuntimeHook,
    ToolCallHookContext,
)
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.tools.base import (
    _POLICY_DENIAL_TEXT_MAX_BYTES,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    _bound_policy_denial_text,
)
from cayu.tools.policy import (
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    AllowAllToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.tools.subagents import SubagentSpec, SubagentTool


def test_before_tool_call_hook_modifies_arguments():
    class ArgModifierHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            modified = dict(context.arguments)
            modified["text"] = "modified"
            return BeforeToolCallDecision(action="proceed_modified", modified_arguments=modified)

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[ArgModifierHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_mod",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == [{"text": "modified"}]
    transcript = asyncio.run(store.load_transcript("sess_before_mod"))
    assert transcript[2].content[0].structured == {"echoed": "modified"}
    hook_events = [
        event
        for event in events
        if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED}
        and event.payload["phase"] == "before_tool_call"
    ]
    assert [event.type for event in hook_events] == [
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    assert hook_events[0].interaction_id is not None
    assert all(event.interaction_id == hook_events[0].interaction_id for event in hook_events)
    assert events[-1].type == EventType.SESSION_COMPLETED


class _DenyTextPolicy(ToolPolicy):
    """Deny when arguments['text'] == 'forbidden'; allow otherwise."""

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.arguments.get("text") == "forbidden":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="forbidden text",
                metadata={"gate": "effective_arguments"},
            )
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def test_before_tool_call_modified_args_are_reauthorized_and_denied():
    # Policy allows the model's original "original", but a hook rewrites to "forbidden";
    # re-authorization on the effective args must block the call (the tool never runs).
    tool, events, transcript = _run_reauth_case(
        "sess_reauth_deny", _DenyTextPolicy(), [_RewriteTextHook("forbidden")]
    )
    assert tool.calls == []
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["blocked_by"] == "tool_policy_reauthorization"
    assert blocked.payload["denied_by"] == "tool_policy"
    assert blocked.payload["decision"] == "deny"
    assert blocked.payload["metadata"] == {"gate": "effective_arguments"}
    assert "effective_arguments" not in blocked.payload
    assert transcript[2].content[0].is_error is True


def test_before_tool_call_modified_args_pass_reauthorization():
    # A hook rewrites to an allowed value; re-authorization permits it and the tool runs with it.
    tool, _events, transcript = _run_reauth_case(
        "sess_reauth_allow", _DenyTextPolicy(), [_RewriteTextHook("allowed")]
    )
    assert tool.calls == [{"text": "allowed"}]
    assert transcript[2].content[0].structured == {"echoed": "allowed"}


def test_reauthorized_effective_args_reach_after_hook_and_event():
    seen: dict[str, object] = {}

    class CaptureArgs(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["args"] = context.arguments

    tool, events, _transcript = _run_reauth_case(
        "sess_effective_args",
        AllowAllToolPolicy(),
        [_RewriteTextHook("modified")],
        extra_hooks=[CaptureArgs()],
    )
    assert tool.calls == [{"text": "modified"}]
    # after_tool_call sees the EFFECTIVE args, and the result event records them for audit.
    assert seen["args"] == {"text": "modified"}
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["effective_arguments"] == {"text": "modified"}


def test_before_tool_call_modified_args_requiring_approval_are_blocked():
    reason = "modified arguments require approval 🙂 " * 500

    class ApprovalForModifiedPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if request.arguments.get("text") == "needs_approval":
                return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason=reason)
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    # v1 fail-safe: a hook that rewrites args into a REQUIRE_APPROVAL state is blocked, not resumed.
    tool, events, _transcript = _run_reauth_case(
        "sess_reauth_approval",
        ApprovalForModifiedPolicy(),
        [_RewriteTextHook("needs_approval")],
    )
    assert tool.calls == []
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["blocked_by"] == "tool_policy_reauthorization"
    assert blocked.payload["denied_by"] == "tool_policy"
    assert blocked.payload["decision"] == "require_approval"
    assert blocked.payload["metadata"] == {}
    assert blocked.payload["reason"] == _bound_policy_denial_text(reason)
    assert len(blocked.payload["result"]["content"].encode("utf-8")) <= (
        _POLICY_DENIAL_TEXT_MAX_BYTES
    )
    assert "effective_arguments" not in blocked.payload


class _RecordMarkerPolicy(ToolPolicy):
    """Allow everything; record the reauthorization marker seen on each authorize() call."""

    def __init__(self) -> None:
        self.markers: list[object] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.markers.append(request.metadata.get(TOOL_POLICY_REAUTHORIZATION_METADATA_KEY))
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def test_reauthorization_of_modified_args_is_marked_for_stateful_policies():
    policy = _RecordMarkerPolicy()
    tool, _events, _transcript = _run_reauth_case(
        "sess_reauth_marker", policy, [_RewriteTextHook("modified")]
    )
    # Two authorizations for a hook-modified call: initial (no marker), then re-auth (marked True)
    # so a stateful policy can re-check the limit without double-counting.
    assert tool.calls == [{"text": "modified"}]
    assert policy.markers == [None, True]


def test_unmodified_call_is_authorized_once_without_reauth_marker():
    policy = _RecordMarkerPolicy()
    tool, _events, _transcript = _run_reauth_case("sess_no_reauth", policy, [])
    # No hook modification → a single authorization, no re-auth, no marker.
    assert tool.calls == [{"text": "original"}]
    assert policy.markers == [None]


def test_after_tool_call_hook_sees_redacted_arguments():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-arg-secret"
    seen: dict[str, object] = {}

    class ObservingHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["arguments"] = context.arguments

    class InjectAfterCheckpoint(RuntimeHook):
        async def before_tool_call(
            self,
            context: BeforeToolCallHookContext,
        ) -> BeforeToolCallDecision:
            arguments = dict(context.arguments)
            arguments["text"] = secret_value
            return BeforeToolCallDecision(
                action="proceed_modified",
                modified_arguments=arguments,
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo", name="echo", arguments={"text": "public"}
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
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[InjectAfterCheckpoint(), ObservingHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_arg_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # The after-hook observes redacted arguments, not just a redacted result.
    assert secret_value not in str(seen["arguments"])
    assert seen["arguments"] == {"text": REDACTED_SECRET}

    started = next(event for event in events if event.type == EventType.TOOL_CALL_STARTED)
    assert started.payload["arguments_state"] == "quarantined"
    assert "arguments" not in started.payload
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["arguments_state"] == "finalized"
    assert completed.payload["arguments"] == {"text": "public"}

    transcript = asyncio.run(store.load_transcript("sess_arg_redact"))
    assistant_tool_call = next(
        part for message in transcript for part in message.content if type(part) is ToolCallPart
    )
    assert assistant_tool_call.arguments == {"text": "public"}
    assert secret_value not in str(
        [message.model_dump(mode="json") for message in provider.requests[1].messages]
    )


def test_composed_after_hook_never_sees_prior_hooks_raw_secret_result():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-x"
    seen: dict[str, object] = {}

    class InjectSecretHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify", modified_result=ToolResult(content=f"leak {secret_value}")
            )

    class ObserveResultHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["content"] = context.result.content

    store = InMemorySessionStore()
    # App-scope hook injects the secret via modify; agent-scope hook observes the threaded result.
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
        runtime_hooks=[InjectSecretHook()],
    )
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[ObserveResultHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compose_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # The second after-hook sees the first hook's modification redacted, never the raw secret.
    assert secret_value not in str(seen["content"])
    assert REDACTED_SECRET in str(seen["content"])


class _ModifyTextBeforeHook(RuntimeHook):
    def __init__(self, value: str) -> None:
        self._value = value

    async def before_tool_call(self, context: BeforeToolCallHookContext) -> BeforeToolCallDecision:
        modified = dict(context.arguments)
        modified["text"] = self._value
        return BeforeToolCallDecision(action="proceed_modified", modified_arguments=modified)


class _CaptureAfterArgsHook(RuntimeHook):
    def __init__(self) -> None:
        self.arguments: object = None

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        self.arguments = context.arguments


def test_short_circuit_without_invocation_scope_omits_effective_arguments():
    class ShortCircuitHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(
                action="short_circuit", synthetic_result=ToolResult(content="cached")
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    capture = _CaptureAfterArgsHook()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[_ModifyTextBeforeHook("modified-before-short"), ShortCircuitHook(), capture],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_eff_sc",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # No invocation scope was established, so modified private arguments remain unavailable.
    assert tool.calls == []
    assert capture.arguments == {}
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["arguments_state"] == "unavailable"
    assert "effective_arguments" not in completed.payload


def test_block_without_invocation_scope_omits_effective_arguments():
    class BlockHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(action="block", block_reason="blocked")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    capture = _CaptureAfterArgsHook()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[_ModifyTextBeforeHook("modified-before-block"), BlockHook(), capture],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_eff_block",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    assert capture.arguments == {}
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["arguments_state"] == "unavailable"
    assert "effective_arguments" not in blocked.payload


def test_invalid_after_tool_call_decision_fails_even_when_observe_only():
    class BadReturnHook(RuntimeHook):
        async def after_tool_call(self, context):
            return "not a decision"  # invalid: not an AfterToolCallDecision or None

    class DenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="no")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=DenyPolicy(),
        runtime_hooks=[BadReturnHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_observe",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # Observe-only (denial) path still validates the decision → hook.failed, not hook.completed.
    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["phase"] == "after_tool_call"
    assert hook_failed.payload["error_type"] == "TypeError"
    transcript = asyncio.run(store.load_transcript("sess_bad_observe"))
    assert transcript[2].content[0].is_error is True


def test_before_tool_call_hook_short_circuits_without_running_tool():
    class ShortCircuitHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(
                action="short_circuit",
                synthetic_result=ToolResult(content="cached", structured={"cached": True}),
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[ShortCircuitHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_short",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["short_circuited_by"] == "before_tool_call_hook"
    transcript = asyncio.run(store.load_transcript("sess_before_short"))
    assert transcript[2].content[0].content == "cached"
    assert transcript[2].content[0].structured == {"cached": True}
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_before_tool_call_hook_blocks_the_call():
    class BlockHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(action="block", block_reason="not allowed")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[BlockHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_block",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["blocked_by"] == "before_tool_call_hook"
    assert blocked.payload["reason"] == "not allowed"
    transcript = asyncio.run(store.load_transcript("sess_before_block"))
    result_part = transcript[2].content[0]
    assert result_part.content == "not allowed"
    assert result_part.is_error is True
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_before_tool_call_hook_failure_proceeds_unmodified():
    class FailingBeforeHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            raise RuntimeError("before hook broke")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[FailingBeforeHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_fail",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["phase"] == "before_tool_call"
    assert hook_failed.payload["error"] == "before hook broke"
    assert tool.calls == [{"text": "original"}]
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_before_tool_call_invalid_decision_proceeds_unmodified():
    class InvalidDecisionHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            # Missing modified_arguments raises ValidationError inside the hook body.
            return BeforeToolCallDecision(action="proceed_modified")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[InvalidDecisionHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_invalid",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["phase"] == "before_tool_call"
    assert hook_failed.payload["error_type"] == "ValidationError"
    assert tool.calls == [{"text": "original"}]
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_after_tool_call_hook_modifies_result_before_transcript():
    class ResultModifierHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content="rewritten", structured={"modified": True}),
            )

    store = InMemorySessionStore()
    provider = _tool_call_provider()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[ResultModifierHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_after_mod",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    transcript = asyncio.run(store.load_transcript("sess_after_mod"))
    assert transcript[2].content[0].content == "rewritten"
    assert transcript[2].content[0].structured == {"modified": True}
    # The model's next turn sees the modified result.
    assert provider.requests[1].messages[-1].content[0].structured == {"modified": True}
    # The result event is persisted after the hook lifecycle events.
    stored_events = asyncio.run(store.load_events("sess_after_mod"))
    assert [
        event.type
        for event in stored_events
        if event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        }
    ] == [
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
    ]


def test_after_tool_call_hooks_compose_app_then_agent_scope():
    class AppendHook(RuntimeHook):
        def __init__(self, suffix: str) -> None:
            self._suffix = suffix

        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content=context.result.content + self._suffix),
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_echo", name="echo", arguments={"text": "seed"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, runtime_hooks=[AppendHook("+app")])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[AppendHook("+agent")],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_after_compose",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    transcript = asyncio.run(store.load_transcript("sess_after_compose"))
    assert transcript[2].content[0].content == "seed+app+agent"


def test_after_tool_call_modification_is_redacted():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-injected-by-hook"

    class SecretInjectingHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content=f"leaked {secret_value}"),
            )

    store = InMemorySessionStore()
    provider = _tool_call_provider()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[SecretInjectingHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_after_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # Redaction runs after the hook's modification, so the injected secret is still scrubbed.
    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert secret_value not in str(tool_event.payload["result"])
    assert REDACTED_SECRET in tool_event.payload["result"]["content"]
    transcript = asyncio.run(store.load_transcript("sess_after_redact"))
    assert transcript[2].content[0].content == f"leaked {REDACTED_SECRET}"


def test_tool_call_hooks_apply_to_subagent_tool_calls():
    observed: list[tuple[str, str]] = []

    class RecordingAppHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision | None:
            observed.append(("before", context.tool_name))
            return None

        async def after_tool_call(
            self, context: ToolCallHookContext
        ) -> AfterToolCallDecision | None:
            observed.append(("after", context.tool_name))
            return None

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_sub",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "do it"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_child_echo", name="echo", arguments={"text": "child"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("child done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False, runtime_hooks=[RecordingAppHook()])
    subagent_tool = SubagentTool(
        app,
        agents={"reviewer": SubagentSpec(agent_name="reviewer", description="Review.")},
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[subagent_tool])
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"), tools=[EchoTool()])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_sub_hooks",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    # App-scope hooks fire on the parent's subagent call AND the child's echo call (propagation).
    assert ("before", "subagent") in observed
    assert ("before", "echo") in observed
    assert ("after", "echo") in observed


def test_after_tool_call_hook_sees_redacted_result():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-leaked-to-hook"
    seen: dict[str, object] = {}

    class LeakyEchoTool(Tool):
        spec = ToolSpec(
            name="echo",
            description="Return a secret.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content=f"token {secret_value}", structured={"token": secret_value})

    class ObservingHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["content"] = context.result.content
            seen["structured"] = context.result.structured

    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LeakyEchoTool()],
        runtime_hooks=[ObservingHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_hook_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # The hook observes the already-redacted result, never the raw secret.
    assert secret_value not in str(seen["content"])
    assert REDACTED_SECRET in str(seen["content"])
    assert seen["structured"] == {"token": REDACTED_SECRET}


def test_after_tool_call_modify_ignored_on_policy_denied_result():
    class DenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="nope")

    class RewriteHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify", modified_result=ToolResult(content="APPROVED")
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=DenyPolicy(),
        runtime_hooks=[RewriteHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_deny_modify",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    # The after-hook still fires on the denial, but its modify is ignored (observe-only).
    assert any(event.type == EventType.HOOK_STARTED for event in events)
    assert any(event.type == EventType.TOOL_CALL_BLOCKED for event in events)
    transcript = asyncio.run(store.load_transcript("sess_deny_modify"))
    result_part = transcript[2].content[0]
    assert result_part.content != "APPROVED"
    assert result_part.is_error is True


def test_after_tool_call_modify_applies_on_short_circuit_result():
    class ShortCircuitHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(
                action="short_circuit", synthetic_result=ToolResult(content="cached")
            )

    class RewriteHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content=f"rewritten-{context.result.content}"),
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[ShortCircuitHook(), RewriteHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sc_modify",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    transcript = asyncio.run(store.load_transcript("sess_sc_modify"))
    assert transcript[2].content[0].content == "rewritten-cached"


def test_before_tool_call_decision_copies_synthetic_result_payload() -> None:
    structured = {"nested": {"status": "original"}}
    artifacts = [{"id": "artifact-1", "metadata": {"source": "original"}}]
    original = ToolResult(
        content="cached",
        structured=structured,
        artifacts=artifacts,
        is_error=True,
    )

    decision = BeforeToolCallDecision(action="short_circuit", synthetic_result=original)

    assert decision.synthetic_result is not original
    structured["nested"]["status"] = "mutated"
    artifacts[0]["metadata"]["source"] = "mutated"
    with pytest.raises(TypeError, match="Frozen JSON values cannot be mutated"):
        original.structured["nested"]["status"] = "mutated"  # type: ignore[index]
    assert decision.synthetic_result == ToolResult(
        content="cached",
        structured={"nested": {"status": "original"}},
        artifacts=[{"id": "artifact-1", "metadata": {"source": "original"}}],
        is_error=True,
    )


def test_after_tool_call_decision_copies_modified_result_payload() -> None:
    structured = {"nested": {"status": "original"}}
    artifacts = [{"id": "artifact-1", "metadata": {"source": "original"}}]
    original = ToolResult(
        content="rewritten",
        structured=structured,
        artifacts=artifacts,
        is_error=False,
    )

    decision = AfterToolCallDecision(action="modify", modified_result=original)

    assert decision.modified_result is not original
    structured["nested"]["status"] = "mutated"
    artifacts[0]["metadata"]["source"] = "mutated"
    with pytest.raises(TypeError, match="Frozen JSON values cannot be mutated"):
        original.artifacts[0]["metadata"]["source"] = "mutated"  # type: ignore[index]
    assert decision.modified_result == ToolResult(
        content="rewritten",
        structured={"nested": {"status": "original"}},
        artifacts=[{"id": "artifact-1", "metadata": {"source": "original"}}],
        is_error=False,
    )


def test_before_tool_call_decision_rejects_stray_block_reason():
    with pytest.raises(ValidationError, match="block_reason is only valid with block"):
        BeforeToolCallDecision(action="proceed", block_reason="nope")
