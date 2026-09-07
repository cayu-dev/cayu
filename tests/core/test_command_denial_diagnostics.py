from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_runtime import FakeProvider
from tests.core.test_structured_commands import _policy_request, _profile

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunCommandTool,
    RunRequest,
    StructuredCommandToolPolicy,
)
from cayu._command_diagnostics import COMMAND_DENIAL_HINTS, CommandDenialCode
from cayu.providers import ModelStreamEvent
from cayu.runtime._approval_support import public_policy_denial_result
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult
from cayu.storage import SQLiteSessionStore


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"selector": "SECRET-selector"}, "unknown_selector"),
        ({"selector": "focused-test", "args": ["tests/test_unit.py"] * 1000}, "argument_count"),
        (
            {"selector": "focused-test", "args": ["tests/test_unit.py"], "timeoutSeconds": 99999},
            "timeout_ceiling",
        ),
        ({"selector": "focused-test", "args": ["SECRET/test_unit.py"]}, "disallowed_path"),
        (
            {
                "selector": "focused-test",
                "args": ["tests/test_unit.py"],
                "workingDirectory": "SECRET",
            },
            "working_directory",
        ),
        ({"selector": "focused-test", "args": "SECRET"}, "argument_shape"),
        (
            {"selector": "focused-test", "args": ["tests/test_unit.py"], "outputMode": "SECRET"},
            "output_mode",
        ),
    ],
)
def test_command_denial_survives_tool_round_and_sqlite_reload(tmp_path, args, code):
    async def run():
        profile = _profile()
        policy = StructuredCommandToolPolicy(toolchain_profile=profile)
        decision = await policy.authorize(_policy_request(args))
        assert decision.decision == ToolPolicyDecision.DENY
        assert decision.command_denial_code == code
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="denied", name="run_command", arguments=args),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        path = tmp_path / "denial.sqlite"
        store = SQLiteSessionStore(path)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RunCommandTool(toolchain_profile=profile)],
            tool_policy=policy,
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="denial",
                    messages=[Message.text("user", "check")],
                )
            )
        ]
        denied = next(e for e in events if e.type == EventType.TOOL_CALL_BLOCKED)
        assert denied.payload["reason"] == COMMAND_DENIAL_HINTS[CommandDenialCode(code)]
        assert code in json.dumps(denied.payload)
        assert "SECRET" not in json.dumps(denied.payload)
        assert any(
            COMMAND_DENIAL_HINTS[CommandDenialCode(code)] in str(m)
            for m in provider.requests[-1].messages
        )
        await store.close()
        reloaded = SQLiteSessionStore(path)
        try:
            persisted = await reloaded.load_events("denial")
            blocked = next(e for e in persisted if e.type == EventType.TOOL_CALL_BLOCKED)
            assert blocked.payload["reason"] == denied.payload["reason"]
            assert code in json.dumps(blocked.payload)
            assert "SECRET" not in json.dumps(blocked.payload)
        finally:
            await reloaded.close()

    asyncio.run(run())


@pytest.mark.parametrize("scope", ["static", "dynamic", "unknown"])
def test_custom_policy_cannot_mark_arbitrary_text_safe(scope):
    result = ToolPolicyResult(
        decision=ToolPolicyDecision.DENY,
        reason="SECRET",
        metadata={"safe": True, "command_denial_code": "SECRET"},
    )
    assert (
        public_policy_denial_result(
            secret_resolution_scope=scope, policy_result=result, publish_arguments=False
        ).reason
        is None
    )
    result.command_denial_code = CommandDenialCode.UNKNOWN_SELECTOR
    public = public_policy_denial_result(
        secret_resolution_scope=scope, policy_result=result, publish_arguments=False
    )
    assert "SECRET" not in public.model_dump_json()
    assert public.reason == COMMAND_DENIAL_HINTS[CommandDenialCode.UNKNOWN_SELECTOR]
    with pytest.raises(ValueError):
        ToolPolicyResult.model_validate({"decision": "deny", "command_denial_code": "SECRET"})


def test_command_diagnostic_round_checkpoint_roundtrip():
    from cayu.runtime._resume_ledger import policy_result_from_pending_tool_call
    from cayu.runtime.approvals import PendingToolCallApproval, copy_pending_tool_call_approval

    for code in CommandDenialCode:
        checkpoint = PendingToolCallApproval(
            tool_call_id="call",
            tool_name="run_command",
            arguments={"selector": "SECRET"},
            policy_decision=ToolPolicyDecision.DENY,
            command_denial_code=code,
            reason="SECRET",
            metadata={"secret": "SECRET"},
        )
        restored = copy_pending_tool_call_approval(
            PendingToolCallApproval.model_validate_json(checkpoint.model_dump_json())
        )
        policy = policy_result_from_pending_tool_call(restored)
        assert policy is not None
        public = public_policy_denial_result(
            secret_resolution_scope="unknown", policy_result=policy, publish_arguments=False
        )
        assert public.command_denial_code == code
        assert public.reason == COMMAND_DENIAL_HINTS[code]
        assert "SECRET" not in public.model_dump_json()
