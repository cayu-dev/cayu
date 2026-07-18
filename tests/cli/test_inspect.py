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

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ScriptedModelProvider,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


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

    assert main(["inspect", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "3"
    assert output["agents"][0]["name"] == "reviewer"
    assert output["agents"][0]["resolved_provider"] == "scripted"
    assert output["defaults"]["environment"] is None
    assert output["environments"][0]["name"] == "optional"
    assert output["environments"][0]["is_default"] is False
    assert output["agents"][0]["registration_provenance"]["location"] == "inspect_project.py"
    assert (
        output["agents"][0]["tools"][0]["implementation_provenance"]["location"]
        == "inspect_project.py"
    )
    assert (project / "build-count.txt").read_text(encoding="utf-8") == "1"
    assert "inspect_project" not in sys.modules


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
        """from cayu import AgentSpec, CayuApp, ScriptedModelProvider


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
    assert error["schema_version"] == "3"
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
        "schema_version": "3",
        "error": {
            "code": "PROJECT_BOOT_FAILED",
            "message": "Application factory failed (RuntimeError): boot exploded",
        },
    }
