"""Generated policy inspection and real queued enforcement, without Docker effects."""

import asyncio
import importlib

import pytest
from tests.core.test_structured_commands import _profile

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    AppManifest,
    CayuApp,
    Environment,
    EnvironmentSpec,
    GuardedToolPolicy,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    ParameterConstrainedToolPolicy,
    RequiredAllowlistRule,
    RunRequest,
    ScriptedModelProvider,
    SQLiteTaskStore,
    StaticToolPolicy,
    StructuredCommandToolPolicy,
    TaskCreate,
    TaskQuery,
    Tool,
    ToolEffect,
    ToolSpec,
    WriteFileTool,
    check_manifest,
    complete_managed_task,
    run_task_worker,
)
from cayu.cli import main
from cayu.cli.project import project_context


class _ExternalTool(Tool):
    async def run(self, ctx, args):
        raise AssertionError("Inspection must never execute tools")


@pytest.mark.parametrize(
    "kind",
    [
        "none",
        "allow",
        "deny",
        "approval",
        "rules",
        "custom",
        "subclass",
        "nested",
        "cycle",
        "nested-cycle",
        "guarded-cycle",
        "mixed-cycle",
    ],
)
def test_maintained_wrapper_preserves_exact_base_coverage(kind):
    class CustomPolicy(StaticToolPolicy):
        pass

    class CustomWrapper(StructuredCommandToolPolicy):
        pass

    bases = {
        "none": None,
        "allow": StaticToolPolicy(),
        "deny": StaticToolPolicy(deny=["write_file", "run_command"]),
        "approval": AlwaysRequireApprovalToolPolicy(),
        "rules": ParameterConstrainedToolPolicy(
            {
                name: [RequiredAllowlistRule("path", values=["allowed.py"])]
                for name in ("write_file", "run_command")
            }
        ),
        "custom": CustomPolicy(deny=["write_file", "run_command"]),
    }
    policy = (CustomWrapper if kind == "subclass" else StructuredCommandToolPolicy)(
        toolchain_profile=_profile(), base_policy=bases.get(kind)
    )
    if kind == "nested":
        policy = StructuredCommandToolPolicy(toolchain_profile=_profile(), base_policy=policy)
    if kind == "cycle":
        policy._base_policy = policy
    if kind == "nested-cycle":
        policy._base_policy = StructuredCommandToolPolicy(
            toolchain_profile=_profile(), base_policy=policy
        )
    if kind == "guarded-cycle":
        policy = GuardedToolPolicy(guards=[], then=policy)
        policy._then = policy
    if kind == "mixed-cycle":
        policy._base_policy = GuardedToolPolicy(guards=[], then=policy)
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="agent", model="fixture"),
        tool_policy=policy,
        tools=[
            _ExternalTool(
                ToolSpec(
                    name=name,
                    effect=ToolEffect.EXTERNAL,
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            )
            for name in ("write_file", "run_command")
        ],
    )
    manifest = app.describe()
    expected = {
        "none": ("allowed", "conditional"),
        "allow": ("allowed", "conditional"),
        "nested": ("allowed", "conditional"),
        "deny": ("denied", "denied"),
        "approval": ("approval_required", "approval_required"),
        "rules": ("conditional", "conditional"),
        "custom": ("unknown", "unknown"),
        "subclass": ("unknown", "unknown"),
        "cycle": ("unknown", "unknown"),
        "nested-cycle": ("unknown", "unknown"),
        "guarded-cycle": ("unknown", "unknown"),
        "mixed-cycle": ("unknown", "unknown"),
    }[kind]
    tools = {tool.name: tool for tool in manifest.agents[0].tools}
    assert (tools["write_file"].policy_coverage, tools["run_command"].policy_coverage) == expected
    assert tools["write_file"].parameter_policy_decision == ("deny" if kind == "rules" else None)
    assert AppManifest.model_validate_json(manifest.model_dump_json()) == manifest
    diagnostics = check_manifest(manifest).diagnostics
    assert sum(d.code == "EXTERNAL_TOOL_COVERAGE_UNKNOWN" for d in diagnostics) == (
        2
        if kind in {"custom", "subclass", "cycle", "nested-cycle", "guarded-cycle", "mixed-cycle"}
        else 0
    )
    assert sum(d.code == "EXTERNAL_TOOL_UNGUARDED" for d in diagnostics) == (
        1 if kind in {"none", "allow", "nested"} else 0
    )


@pytest.mark.parametrize("path", ["allowed.py", "forbidden.py", ".git/config"])
def test_generated_rules_survive_composition_and_queued_worker(tmp_path, path):
    assert (
        main(
            [
                "new",
                "policy-coder",
                "--preset",
                "coding",
                "--execution",
                "docker",
                "--coding-toolchain",
                "python",
                "--dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    with project_context(tmp_path / "policy-coder"):
        composition = importlib.import_module("operations.coding")
        base = composition._primary_tool_policy()
        # The shipped extension recipe preserves existing generated restrictions.
        rules = dict(base.rules)
        rules["write_file"] = (
            *rules["write_file"],
            RequiredAllowlistRule("path", values=["allowed.py", ".git/config"]),
        )
        policy = StructuredCommandToolPolicy(
            toolchain_profile=_profile(), base_policy=ParameterConstrainedToolPolicy(rules)
        )

    async def exercise():
        store = SQLiteTaskStore(tmp_path / "tasks.db")
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        app = CayuApp(task_store=store, enable_logging=False)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="write",
                        name="write_file",
                        arguments={"path": path, "content": "verified", "mode": "create"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), workspace=LocalWorkspace(workspace_root)),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="agent", model="fixture"), tools=[WriteFileTool()], tool_policy=policy
        )
        assert not [
            d
            for d in check_manifest(app.describe()).diagnostics
            if d.code.startswith("EXTERNAL_TOOL_")
        ]
        await app.create_task(TaskCreate(task_id="queued", type="repair"))
        events = []

        async def handle(app, task, worker):
            events.extend(
                [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="agent",
                            session_id="session",
                            messages=[Message.text("user", "write the admitted file")],
                        )
                    )
                ]
            )
            await complete_managed_task(store, task, worker, {"session_id": "session"})

        try:
            async with asyncio.timeout(20):
                assert (
                    await run_task_worker(
                        app,
                        store,
                        handle,
                        worker_id="worker",
                        query=TaskQuery(type="repair"),
                        max_tasks=1,
                        reclaim=False,
                        recover_interrupted_handoffs=False,
                    )
                    == 1
                )
            assert events[-1].type.value == "session.completed"
            assert (workspace_root / "allowed.py").exists() is (path == "allowed.py")
            assert not (workspace_root / "forbidden.py").exists()
            assert not (workspace_root / ".git").exists()
            assert any(e.type.value == "tool.call.blocked" for e in events) is (
                path != "allowed.py"
            )
        finally:
            await store.close()
        reopened = SQLiteTaskStore(tmp_path / "tasks.db")
        try:
            task = await reopened.load_task("queued")
            assert task.status.value == "completed" and task.worker_id is None
            assert task.result == {"session_id": "session"}
        finally:
            await reopened.close()

    asyncio.run(exercise())
