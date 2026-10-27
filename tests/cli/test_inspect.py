from __future__ import annotations

import json
import sys
from pathlib import Path

from cayu.cli import main


def test_inspect_json_discovers_nested_project_and_builds_factory_once(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = tmp_path / "project"
    nested = project / "agents" / "reviewer"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "inspect_project:build_app"\n',
        encoding="utf-8",
    )
    (project / "inspect_project.py").write_text(
        """from pathlib import Path

from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider


class ReviewTool(Tool):
    spec = ToolSpec(
        name="review",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "additionalProperties": False},
    )

    async def run(self, ctx, args):
        return ToolResult(content="reviewed")

build_count = 0


def build_app():
    global build_count
    build_count += 1
    Path("build-count.txt").write_text(str(build_count), encoding="utf-8")
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(AgentSpec(name="reviewer", model="test-model"), tools=[ReviewTool()])
    app.register_environment(Environment(EnvironmentSpec(name="optional")), default=False)
    return app
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(nested)
    sys.modules.pop("inspect_project", None)

    assert main(["inspect"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "17"
    assert output["agents"][0]["name"] == "reviewer"
    assert output["agents"][0]["resolved_provider"] == "scripted"
    assert output["defaults"]["environment"] is None
    assert output["environments"][0]["name"] == "optional"
    assert output["environments"][0]["is_default"] is False
    assert output["environments"][0]["workspace_branch_capabilities"] == {
        "detail_code": "workspace_branching_unsupported",
        "isolation": False,
        "lifecycle_inspection": "unsupported",
        "net_changes": False,
        "publication": "unsupported",
        "recovery": "unsupported",
        "retention": "unsupported",
    }
    assert output["environments"][0]["workspace_branch_lifecycle"] == {
        "attached_count": 0,
        "statuses": [],
        "truncated": False,
    }
    assert output["agents"][0]["registration_provenance"]["location"] == "inspect_project.py"
    assert (
        output["agents"][0]["tools"][0]["implementation_provenance"]["location"]
        == "inspect_project.py"
    )
    assert (project / "build-count.txt").read_text(encoding="utf-8") == "1"
    assert "inspect_project" not in sys.modules

    destination = project / "manifest.json"
    assert main(["inspect", "--output", str(destination)]) == 0
    assert json.loads(destination.read_text(encoding="utf-8"))["agents"][0]["name"] == ("reviewer")


def test_inspect_subject_filter_and_missing_subject_use_structured_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "filter_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "filter_project.py").write_text(
        """from cayu import AgentSpec, CayuApp
from cayu.evals import ScriptedModelProvider


def build_app():
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(AgentSpec(name="reviewer", model="test-model"))
    app.register_agent(AgentSpec(name="writer", model="test-model"))
    return app
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("filter_project", None)

    assert main(["inspect", "--agent", "reviewer", "--json"]) == 0
    filtered = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in filtered["agents"]] == ["reviewer"]
    assert filtered["providers"] == []

    assert main(["inspect", "--environment", "missing", "--json"]) == 1
    error = json.loads(capsys.readouterr().out)
    assert error["schema_version"] == "17"
    assert error["error"] == {
        "code": "SUBJECT_NOT_FOUND",
        "message": "Environment not found: missing.",
        "path": "environments.missing",
    }


def test_inspect_factory_failure_uses_current_manifest_schema_version(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "failed_inspect_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "failed_inspect_project.py").write_text(
        'def build_app():\n    raise RuntimeError("boot exploded")\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("failed_inspect_project", None)

    assert main(["inspect", "--json"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "schema_version": "17",
        "error": {
            "code": "PROJECT_BOOT_FAILED",
            "message": "Application factory failed (RuntimeError): boot exploded",
        },
    }


def test_inspect_lists_fenced_tasks_and_how_to_settle_them(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import asyncio
    from datetime import UTC, datetime, timedelta

    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.queries import TaskQuery

    async def fence() -> None:
        clock = {"value": datetime.now(UTC) - timedelta(hours=1)}
        store = SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=lambda: clock["value"])
        await store.create_task(TaskCreate(task_id="scan-1", type="scan"))
        await store.create_task(TaskCreate(task_id="scan-2", type="scan"))
        claimed = await store.claim_task("worker-old", TaskQuery(type="scan"), lease_seconds=30)
        assert claimed is not None and claimed.lease_expires_at is not None
        await store.mark_claimed_task_execution_started(
            claimed.id, "worker-old", claimed.lease_expires_at
        )
        clock["value"] = claimed.lease_expires_at + timedelta(seconds=1)
        await store.reclaim_expired(query=TaskQuery(type="scan"))
        await store.close()

    asyncio.run(fence())
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "fenced_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "fenced_project.py").write_text(
        """from cayu import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore


def build_app():
    return CayuApp(task_store=SQLiteTaskStore("tasks.db"), enable_logging=False)
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("fenced_project", None)

    assert main(["inspect", "--fenced-tasks", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["complete"] is True
    [task] = output["fenced_tasks"]
    assert task["task_id"] == "scan-1"
    assert task["worker_id"] == "worker-old"
    assert task["status"] == "claimed"
    assert task["error_code"] == "task_worker_lease_expired_after_dispatch"
    assert "reconcile_abandoned_task_cancellation" in task["settle"]

    sys.modules.pop("fenced_project", None)
    assert main(["inspect", "--fenced-tasks", "--table"]) == 0
    human = capsys.readouterr().out
    assert "1 fenced task(s)" in human
    assert "- scan-1 (scan): worker worker-old" in human
    assert "[task_worker_lease_expired_after_dispatch]" in human
