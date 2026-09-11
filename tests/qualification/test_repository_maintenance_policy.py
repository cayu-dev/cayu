"""Fixed mutation scope through generated policy and real Runtime dispatch."""

import asyncio
import importlib

import pytest

from cayu import (
    AgentSpec,
    ApplyPatchTool,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventQuery,
    EventType,
    InMemorySessionStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    WriteFileTool,
)
from cayu.cli.project import project_context
from tests.qualification.repository_maintenance_application import maintenance_project_files
from tests.qualification.repository_maintenance_policy import (
    MaintenancePatchScopeRule,
    MaintenanceWriteBoundRule,
)


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "../range_ops.py",
        "./range_ops.py",
        "/workspace/range_ops.py",
        "tests/../range_ops.py",
        "range_ops.py ",
        True,
        None,
    ],
)
def test_patch_scope_rejects_noncanonical_or_unowned_paths(path):
    rule = MaintenancePatchScopeRule()
    assert rule.check({"operations": [{"type": "update", "path": path}]}) is not None
    for field in ("from_path", "to_path"):
        operation = {
            "type": "move",
            "from_path": "range_ops.py",
            "to_path": "tests/test_range_ops.py",
        }
        operation[field] = path
        assert rule.check({"operations": [operation]}) is not None


@pytest.mark.parametrize(
    "operations",
    [
        None,
        [],
        True,
        [None],
        [{"type": "future", "path": "range_ops.py"}],
        [{"type": "update"}],
        [{"type": "update", "path": "range_ops.py", "to_path": "tests/test_range_ops.py"}],
    ],
)
def test_patch_scope_rejects_missing_or_ambiguous_authority(operations):
    assert MaintenancePatchScopeRule().check({"operations": operations}) is not None


def test_patch_scope_checks_every_operation_before_admission():
    valid = {"type": "update", "path": "range_ops.py"}
    invalid = {"type": "update", "path": "pyproject.toml"}
    rule = MaintenancePatchScopeRule()
    assert rule.check({"operations": [valid]}) is None
    assert (
        rule.check({"operations": [valid, valid]}) is None
    )  # Tool owns duplicate-path validation.
    assert (
        rule.check(
            {
                "operations": [
                    {
                        "type": "move",
                        "from_path": "range_ops.py",
                        "to_path": "tests/test_range_ops.py",
                    }
                ]
            }
        )
        is None
    )
    assert rule.check({"operations": [valid, valid, valid]}) is not None
    assert rule.check({"operations": [valid, invalid]}) is not None
    assert rule.check({"operations": [invalid, valid]}) is not None


@pytest.mark.parametrize("maximum", [None, True, False, 0, -1, 16385, "16384", 16384.0])
def test_write_bound_requires_exact_positive_integer(maximum):
    assert MaintenanceWriteBoundRule().check({"max_bytes": maximum}) is not None


@pytest.mark.parametrize("maximum", [1, 16384])
def test_write_bound_accepts_declared_boundaries(maximum):
    assert MaintenanceWriteBoundRule().check({"max_bytes": maximum}) is None


@pytest.fixture(scope="module")
def generated_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("maintenance-policy-consumer")
    for relative, content in maintenance_project_files().items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


@pytest.mark.parametrize(
    "case", ["allowed_write", "allowed_patch", "outside_write", "unbounded_write", "partial_patch"]
)
def test_generated_policy_fences_real_runtime_mutations(generated_project, tmp_path, case):
    workspace = LocalWorkspace(tmp_path, workspace_id="policy-source")
    arguments: dict[str, object]
    if case in {"allowed_patch", "partial_patch"}:
        tool_name = "apply_patch"
        arguments = {
            "operations": [
                {"type": "create", "path": "range_ops.py", "content": "allowed = True\n"},
                {"type": "create", "path": "pyproject.toml", "content": "forbidden = true\n"},
            ]
        }
        if case == "allowed_patch":
            arguments = {
                "operations": [
                    {"type": "create", "path": "range_ops.py", "content": "allowed = True\n"}
                ]
            }
    else:
        tool_name = "write_file"
        arguments = {
            "path": "range_ops.py" if case != "outside_write" else "pyproject.toml",
            "mode": "create",
            "content": "allowed = True\n",
        }
        if case != "unbounded_write":
            arguments["max_bytes"] = 16384
    with project_context(generated_project):
        operations = importlib.import_module("operations.coding")
        policy = operations._primary_tool_policy()
        assert (
            policy.execution_profile_identity.name == "repository-maintenance.primary-tool-policy"
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="mutation", name=tool_name, arguments=arguments
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), workspace=workspace), default=True
        )
        app.register_agent(
            AgentSpec(name="worker", model="scripted-model"),
            tools=[WriteFileTool(), ApplyPatchTool(max_operations=2, max_file_bytes=16384)],
            tool_policy=policy,
        )

        async def scenario():
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="worker",
                        session_id="scope-test",
                        messages=[Message.text("user", "perform the fixture operation")],
                    )
                )
            ]
            blocked = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
            if case in {"allowed_write", "allowed_patch"}:
                assert not blocked
                assert (tmp_path / "range_ops.py").read_text() == "allowed = True\n"
                completed = [
                    event for event in events if event.type is EventType.TOOL_CALL_COMPLETED
                ]
                assert completed
            else:
                assert len(blocked) == 1
                assert blocked[0].payload["denied_by"] == "tool_policy"
                assert not (tmp_path / "range_ops.py").exists()
            assert not (tmp_path / "pyproject.toml").exists()
            assert any(event.type is EventType.SESSION_COMPLETED for event in events)
            stored_denials = await store.query_events(
                EventQuery(
                    session_id="scope-test",
                    event_type=EventType.TOOL_CALL_BLOCKED,
                )
            )
            assert len(stored_denials) == len(blocked)
            if blocked:
                assert stored_denials[0].event.payload["reason"] == blocked[0].payload["reason"]

        asyncio.run(scenario())
    # This proves policy delivery/dispatch fencing, not Docker execution.
