from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_tool_policies import _request
from tests.core.test_user_input import _ScriptedProvider

from cayu import (
    AgentSpec,
    AllowAllToolPolicy,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    DenyPatternRule,
    Message,
    ParameterConstrainedToolPolicy,
    RequiredAllowlistRule,
    RequiredFieldRule,
    RunRequest,
    ToolPolicyDecision,
    UserInputTool,
    check_manifest,
)
from cayu.events import EventType
from cayu.tools.gateway import tool_argument_validation_error, tool_input_schema_supported


@pytest.mark.parametrize("arguments", [{}, {"question": ""}, {"question": "  "}])
def test_validity_precedes_earlier_approval_rule(arguments):
    policy = ParameterConstrainedToolPolicy(
        {"ask_user": [DenyPatternRule("action", patterns=[".*"]), RequiredFieldRule("question")]},
        decision=ToolPolicyDecision.REQUIRE_APPROVAL,
    )
    result = asyncio.run(
        policy.authorize(
            _request(
                tool_name="ask_user",
                arguments={
                    "action": "send",
                    **arguments,
                },
            )
        )
    )
    assert result.decision == ToolPolicyDecision.DENY
    assert "question" in result.reason
    assert "Correct the call" in result.reason


@pytest.mark.parametrize("value", [None, [], {}, 42, "  ", "outside"])
def test_required_allowlist_distinguishes_authority_from_validity(value):
    policy = ParameterConstrainedToolPolicy(
        {"send_email": [RequiredAllowlistRule("to", values=["allowed"])]},
        decision=ToolPolicyDecision.REQUIRE_APPROVAL,
    )
    result = asyncio.run(policy.authorize(_request(arguments={"to": value})))
    assert result.decision == (
        ToolPolicyDecision.REQUIRE_APPROVAL if value == "outside" else ToolPolicyDecision.DENY
    )


@pytest.mark.parametrize(
    "arguments,field",
    [
        ({}, "question"),
        ({"question": 42}, "question"),
        ({"question": "   "}, "question"),
        ({"question": "private-value", "unexpected": "private-value"}, "unexpected"),
    ],
)
def test_schema_invalid_call_never_reaches_policy_or_approval(arguments, field):
    class Policy(AlwaysRequireApprovalToolPolicy):
        async def authorize(self, request):
            raise AssertionError("Malformed arguments reached application policy")

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(_ScriptedProvider([("invalid", "ask_user", arguments)]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
            tool_policy=Policy(),
        )
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="invalid",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert not {
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
            EventType.SESSION_AWAITING_USER_INPUT,
        } & {e.type for e in events}
        blocked = next(e for e in events if e.type == EventType.TOOL_CALL_BLOCKED)
        assert field in json.dumps(blocked.payload)
        diagnostic = blocked.payload["argument_presence"]
        assert diagnostic["keys"] == sorted(set(arguments) & {"question", "options"})
        assert diagnostic["key_count"] == len(arguments)
        assert "private-value" not in json.dumps(diagnostic)

    asyncio.run(scenario())


def test_schema_diagnostic_never_contains_argument_or_schema_values():
    error = tool_argument_validation_error(
        {"question": "private-argument"},
        {"properties": {"question": {"enum": ["private-schema"]}}},
    )
    assert "question" in error
    assert "private-" not in error


def test_schema_denial_never_names_model_authored_keys():
    from cayu import Tool, ToolSpec

    secret = "sk-live-SECRETVALUE"
    schema = {
        "type": "object",
        "properties": {
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "items": {"type": "array", "items": {"type": "string"}},
        },
    }
    arguments = {"headers": {f"Authorization: Bearer {secret}": 5}}
    error = tool_argument_validation_error(arguments, schema)
    assert error.startswith("Invalid arguments: headers.<key> fails type.")
    assert tool_argument_validation_error({"items": ["ok", 5]}, schema).startswith(
        "Invalid arguments: items.1 fails type."
    )

    class HeadersTool(Tool):
        spec = ToolSpec(name="send_headers", description="Send headers", input_schema=schema)

        @property
        def _publish_arguments(self) -> bool:
            return False

        async def run(self, ctx, arguments):
            raise AssertionError("Schema-invalid arguments reached the tool")

    async def scenario():
        app = CayuApp(enable_logging=False)
        provider = _ScriptedProvider([("invalid", "send_headers", arguments)])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[HeadersTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
            )
        ]
        blocked = next(e for e in events if e.type == EventType.TOOL_CALL_BLOCKED)
        assert "headers.<key>" in blocked.payload["reason"]
        assert secret not in json.dumps([event.model_dump(mode="json") for event in events])

    asyncio.run(scenario())


def test_checker_describes_validity_rule_under_approval():
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool()],
        tool_policy=ParameterConstrainedToolPolicy(
            {"ask_user": [RequiredFieldRule("context")]},
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
        ),
    )
    diagnostic = next(
        d
        for d in check_manifest(app.describe()).diagnostics
        if d.code == "TOOL_APPROVAL_VALIDITY_RULE"
    )
    assert "ask_user" in diagnostic.message
    assert "RequiredFieldRule(context)" in diagnostic.message
    assert diagnostic.documentation_anchor.endswith("#tool-approval-validity-rule")


def test_profile_changes_only_for_approval_configurations():
    deny = ParameterConstrainedToolPolicy({"ask_user": [RequiredFieldRule("question")]})
    approve = ParameterConstrainedToolPolicy(
        {"ask_user": [RequiredFieldRule("question")]},
        decision=ToolPolicyDecision.REQUIRE_APPROVAL,
    )
    assert "validity_denials_version" not in deny._execution_profile_material()
    assert approve._execution_profile_material()["validity_denials_version"] == 1


def test_hook_rewrite_cannot_dispatch_schema_invalid_arguments():
    from tests.core.test_runtime import _RewriteTextHook, _run_reauth_case

    tool, events, _transcript = _run_reauth_case(
        "invalid_hook_rewrite", AllowAllToolPolicy(), [_RewriteTextHook(42)]
    )
    assert tool.calls == []
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["blocked_by"] == "tool_policy_reauthorization"
    assert blocked.payload["decision"] == "deny"
    assert blocked.payload["metadata"]["reason"] == "invalid_arguments"
    assert "text" in blocked.payload["reason"]
    assert EventType.TOOL_CALL_APPROVAL_REQUESTED not in {event.type for event in events}


def test_direct_user_input_fallback_identifies_missing_question():
    result = asyncio.run(UserInputTool().run(None, {}))
    assert result.is_error
    assert "question" in result.content
    assert "approval first" not in result.content


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"n": {"type": "number", "minimum": 0, "exclusiveMinimum": True}}},
        {"properties": {"n": {"type": "array", "items": [{"type": "string"}]}}},
        {"$ref": "https://invalid.example/schema.json"},
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"n": {"type": "number"}},
        },
    ],
)
def test_unsupported_schema_warns_but_preserves_tool_owned_validation(schema):
    from cayu import Tool, ToolResult, ToolSpec

    class LegacyTool(Tool):
        spec = ToolSpec(name="legacy", description="Legacy schema", input_schema=schema)

        async def run(self, ctx, arguments):
            return ToolResult(content="legacy ran")

    async def scenario():
        app = CayuApp(enable_logging=False)
        provider = _ScriptedProvider([("legacy-call", "legacy", {"n": "5"})])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[LegacyTool()])
        assert "TOOL_INPUT_SCHEMA_RUNTIME_UNSUPPORTED" in {
            d.code for d in check_manifest(app.describe()).diagnostics
        }
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)

    asyncio.run(scenario())


def test_declared_draft_2020_12_schema_keeps_runtime_validation():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "properties": {"n": {"type": "number"}},
    }
    assert tool_input_schema_supported(schema)
    assert not tool_input_schema_supported(
        {**schema, "$schema": "http://json-schema.org/draft-07/schema#"}
    )
    assert "n fails type" in tool_argument_validation_error({"n": "5"}, schema)


def test_blocked_static_arguments_remain_in_model_history():
    async def scenario():
        app = CayuApp(enable_logging=False)
        provider = _ScriptedProvider([("invalid", "ask_user", {"question": 42})])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        blocked = next(e for e in events if e.type == EventType.TOOL_CALL_BLOCKED)
        assert blocked.payload["arguments_state"] == "finalized"
        assert blocked.payload["arguments"] == {"question": 42}
        call = next(
            part
            for message in provider.requests[-1].messages
            for part in message.content
            if part.type == "tool_call"
        )
        assert call.arguments == {"question": 42}

    asyncio.run(scenario())


def test_argument_presence_is_captured_before_policy_and_bounded():
    from cayu.runtime._runtime_records import ToolCallRequest

    arguments = {"k" * 1000 + str(index): "private-value" for index in range(100)}
    call = ToolCallRequest(id="call", name="tool", arguments=arguments)
    arguments.clear()
    assert call.argument_presence["argument_presence"]["key_count"] == 100
    keys = call.argument_presence["argument_presence"]["keys"]
    assert len(keys) == 64 and all(len(key) == 64 for key in keys)
    assert "private-value" not in json.dumps(call.argument_presence)


def test_required_field_subclass_does_not_inherit_validity_authority():
    class AuthorityRule(RequiredFieldRule):
        def check(self, arguments):
            return "Needs permission"

    policy = ParameterConstrainedToolPolicy(
        {"send_email": [AuthorityRule("to")]},
        decision=ToolPolicyDecision.REQUIRE_APPROVAL,
    )
    result = asyncio.run(policy.authorize(_request(arguments={"to": "outside"})))
    assert result.decision == ToolPolicyDecision.REQUIRE_APPROVAL


def test_checker_omits_redundant_ask_user_validity_note():
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool()],
        tool_policy=ParameterConstrainedToolPolicy(
            {"ask_user": [RequiredFieldRule("question")]},
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
        ),
    )
    assert "TOOL_APPROVAL_VALIDITY_RULE" not in {
        d.code for d in check_manifest(app.describe()).diagnostics
    }
