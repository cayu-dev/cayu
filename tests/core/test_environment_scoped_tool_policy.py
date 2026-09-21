from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_policies import _request

from cayu import (
    AgentSpec,
    AllowAllToolPolicy,
    BrowserSessionTool,
    CayuApp,
    Environment,
    EnvironmentScopedToolPolicy,
    EnvironmentSpec,
    EventType,
    ExecCommandTool,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    ReadFileTool,
    RunRequest,
    ScriptedModelProvider,
    StaticToolPolicy,
    ToolPolicyDecision,
    WriteFileTool,
    check_manifest,
)
from cayu.runtime._execution_profile_admission import _cayu_policy_material


@pytest.mark.parametrize("environment", [None, "local", "research", "other"])
@pytest.mark.parametrize("tool", ["read_file", "exec_command", "unlisted"])
def test_trusted_environment_scope_denies_missing_and_unmatched_context(environment, tool):
    policy = EnvironmentScopedToolPolicy(
        allow={"read_file": ["local", "research"], "exec_command": ["research"]}
    )
    request = _request(tool_name=tool, arguments={"environment_name": "research"})
    request.environment_name = environment
    result = asyncio.run(policy.authorize(request))
    allowed = (tool == "read_file" and environment in {"local", "research"}) or (
        tool == "exec_command" and environment == "research"
    )
    assert result.decision == (ToolPolicyDecision.ALLOW if allowed else ToolPolicyDecision.DENY)


def test_scope_configuration_is_detached_and_fail_closed():
    environments = ["research"]
    scopes = {"read_file": environments, "write_file": []}
    policy = EnvironmentScopedToolPolicy(allow=scopes)
    environments.append("local")
    scopes.clear()
    assert policy.allow["read_file"] == frozenset({"research"})
    with pytest.raises(TypeError):
        policy.allow["read_file"] = frozenset({"local"})
    request = _request(tool_name="write_file")
    request.environment_name = "research"
    assert asyncio.run(policy.authorize(request)).decision == ToolPolicyDecision.DENY
    for invalid in ({"read_file": "research"}, {"read_file": [" "]}, {" ": ["research"]}):
        with pytest.raises((TypeError, ValueError)):
            EnvironmentScopedToolPolicy(allow=invalid)


class _CustomScope(EnvironmentScopedToolPolicy):
    pass


@pytest.mark.parametrize("kind", ["scoped", "custom", "static", "all"])
def test_checker_recognizes_only_maintained_scope_for_real_external_tools(kind):
    tools = [ReadFileTool(), WriteFileTool(), ExecCommandTool(), BrowserSessionTool()]
    scopes = {tool.name: ["research"] for tool in tools}
    policy = {
        "scoped": EnvironmentScopedToolPolicy(allow=scopes),
        "custom": _CustomScope(allow=scopes),
        "static": StaticToolPolicy(allow=scopes),
        "all": AllowAllToolPolicy(),
    }[kind]
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=tools, tool_policy=policy)
    manifest = app.describe()
    diagnostics = [
        d for d in check_manifest(manifest).diagnostics if d.code.startswith("EXTERNAL_TOOL")
    ]
    if kind == "scoped":
        assert not diagnostics
        assert all(t.policy_environment_names == ("research",) for t in manifest.agents[0].tools)
    else:
        assert len(diagnostics) == 4
        assert {d.code for d in diagnostics} == {
            "EXTERNAL_TOOL_COVERAGE_UNKNOWN" if kind == "custom" else "EXTERNAL_TOOL_UNGUARDED"
        }
        assert all(t.policy_environment_names is None for t in manifest.agents[0].tools)


@pytest.mark.parametrize(
    "environment,path,allowed",
    [
        ("research", "result.txt", True),
        ("local", "result.txt", False),
        ("research", "../escape.txt", False),
    ],
)
def test_native_workspace_execution_is_unattended_and_remains_confined(
    tmp_path, environment, path, allowed
):
    root = tmp_path / "workspace"
    root.mkdir()
    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.tool_call(
                    id="write",
                    name="write_file",
                    arguments={
                        "path": path,
                        "content": "proof",
                        "mode": "create",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
            (
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name=environment), workspace=LocalWorkspace(root)), default=True
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[WriteFileTool()],
        tool_policy=EnvironmentScopedToolPolicy(allow={"write_file": ["research"]}),
    )

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "write")])
            )
        ]

    events = asyncio.run(run())
    assert (root / "result.txt").exists() is allowed
    assert not (tmp_path / "escape.txt").exists()
    assert any(e.type == EventType.SESSION_COMPLETED for e in events)
    assert not any("approval" in e.type for e in events)
    assert any(e.type == EventType.TOOL_CALL_BLOCKED for e in events) is (environment == "local")


def test_scope_configuration_participates_in_maintained_identity():
    first = _cayu_policy_material(
        EnvironmentScopedToolPolicy(allow={"read_file": ["research", "local"]})
    )
    reordered = _cayu_policy_material(
        EnvironmentScopedToolPolicy(allow={"read_file": ["local", "research"]})
    )
    changed = _cayu_policy_material(EnvironmentScopedToolPolicy(allow={"read_file": ["research"]}))
    assert first is not None
    assert first == reordered
    assert first != changed
    assert _cayu_policy_material(_CustomScope(allow={"read_file": ["research"]})) is None


@pytest.mark.parametrize("tool", [ExecCommandTool(), BrowserSessionTool()])
def test_scope_does_not_admit_incompatible_native_environment(tmp_path, tool):
    from cayu import ExecutionRequirements, LocalRunner

    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="research"), runner=LocalRunner(tmp_path)), default=True
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        execution_requirements=ExecutionRequirements.untrusted(),
        tools=[tool],
        tool_policy=EnvironmentScopedToolPolicy(allow={tool.name: ["research"]}),
    )

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "run")])
            )
        ]

    events = asyncio.run(run())
    failure = next(e for e in events if e.type == EventType.SESSION_FAILED)
    assert failure.payload["execution_admission"]["refusals"]
    assert not any(e.type == EventType.MODEL_STARTED for e in events)


def test_manifest_scope_redaction_and_default_denial():
    from cayu import AppManifest, SecretRedactor

    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor("private-scope"))
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[ReadFileTool(), WriteFileTool(), ExecCommandTool()],
        tool_policy=EnvironmentScopedToolPolicy(
            allow={"read_file": ["private-scope"], "write_file": []}
        ),
    )
    manifest = app.describe()
    encoded = manifest.model_dump_json()
    assert "private-scope" not in encoded
    restored = AppManifest.model_validate_json(encoded)
    tools = {tool.name: tool for tool in restored.agents[0].tools}
    assert tools["read_file"].policy_coverage == "conditional"
    for name in ("write_file", "exec_command"):
        assert tools[name].policy_coverage == "denied"
        assert tools[name].policy_environment_names == ()


@pytest.mark.parametrize("command_allowed", [True, False])
def test_scoped_native_command_preserves_command_policy(tmp_path, command_allowed):
    import sys

    from cayu import LocalRunner, ProcessCommandPolicy

    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.tool_call(
                        id="exec",
                        name="exec_command",
                        arguments={
                            "argv": [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; Path('proof.txt').write_text('ok')",
                            ],
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
                (
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="research"), runner=LocalRunner(tmp_path)), default=True
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[
            ExecCommandTool(
                policy=ProcessCommandPolicy(
                    allowed_executables=[sys.executable] if command_allowed else [],
                    allowed_cwds=[str(tmp_path)],
                )
            )
        ],
        tool_policy=EnvironmentScopedToolPolicy(allow={"exec_command": ["research"]}),
    )

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "run")])
            )
        ]

    events = asyncio.run(run())
    assert (tmp_path / "proof.txt").exists() is command_allowed, [
        (e.type, e.payload) for e in events if "tool.call" in e.type
    ]
    assert any(e.type == EventType.SESSION_COMPLETED for e in events)
    assert not any("approval" in e.type for e in events)
    if not command_allowed:
        blocked = next(e for e in events if e.type == EventType.TOOL_CALL_BLOCKED)
        assert blocked.payload["denied_by"] == "command_policy"
