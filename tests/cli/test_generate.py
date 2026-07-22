from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.generate import GeneratorApplyError, apply_slice_plan, plan_slice, plan_tool
from cayu.runtime import APP_MANIFEST_SCHEMA_VERSION


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _stable_files(root: Path) -> dict[str, bytes]:
    """Snapshot project files whose bytes are not mutable runtime state."""

    cache_parts = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    return {
        relative: content
        for relative, content in _files(root).items()
        if Path(relative).parts[0] != "data" and cache_parts.isdisjoint(Path(relative).parts)
    }


def test_generate_slice_effect_help_routes_to_canonical_guide(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "slice", "--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "{none,idempotent,external}" in output
    assert "cayu guide tool-effects" in output


def test_generate_tool_attaches_first_tracer_bullet_to_scaffold_starter(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "invoice-analyst", "--dir", str(tmp_path), "--provider", "openai"]) == 0
    capsys.readouterr()
    project = tmp_path / "invoice-analyst"
    monkeypatch.chdir(project)
    command = [
        "generate",
        "tool",
        "calculate_total",
        "--agent",
        "invoice-analyst",
        "--effect",
        "none",
    ]

    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    after_apply = _files(project)
    assert "agents/invoice-analyst.py" not in after_apply
    assert after_apply["tools/__init__.py"] == b""
    assert "ToolSpec(" in after_apply["tools/calculate_total.py"].decode()
    assert '"required": ["input"]' in after_apply["tools/calculate_total.py"].decode()
    assert "starter_tools.append(CalculateTotalTool())" in after_apply["app.py"].decode()
    assert (
        "_WORKFLOW_TOOL_NAMES.append(CALCULATE_TOTAL_TOOL_NAME)"
        in after_apply["agents/agent.py"].decode()
    )

    for module_name in ("app", "agents.agent", "agents", "tools.calculate_total", "tools"):
        sys.modules.pop(module_name, None)
    assert main(["inspect", "--json"]) == 0
    agent = json.loads(capsys.readouterr().out)["agents"][0]
    assert agent["name"] == "invoice-analyst"
    assert agent["workflow_tool_names"] == ["calculate_total"]
    assert agent["authoring_state"] == "unfinished_generated_tracer_bullet"
    assert agent["tools"][0]["input_schema"]["required"] == ["input"]
    assert main(["check", "--json"]) == 0
    assert [item["code"] for item in json.loads(capsys.readouterr().out)["diagnostics"]] == [
        "AGENT_GENERATED_TRACER_BULLET_UNFINISHED"
    ]

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_calculate_total.py", "-q"],
        cwd=project,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    eval_output = project / "calculate-total-eval.json"
    assert (
        main(
            [
                "eval",
                "run",
                "evals.calculate_total:build_eval",
                "--output",
                str(eval_output),
            ]
        )
        == 0
    )
    assert json.loads(eval_output.read_text(encoding="utf-8"))["status"] == "passed"

    before_repeat = _stable_files(project)
    assert main([*command, "--json"]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["status"] == "already_present"
    assert repeated["edits"] == []
    assert _stable_files(project) == before_repeat

    agent_path = project / "agents" / "agent.py"
    completed_source = agent_path.read_text(encoding="utf-8").replace(
        '_AUTHORING_STATE = "unfinished_generated_tracer_bullet"',
        "_AUTHORING_STATE = None",
    )
    agent_path.write_text(completed_source, encoding="utf-8")
    for module_name in ("app", "agents.agent", "agents", "tools.calculate_total", "tools"):
        sys.modules.pop(module_name, None)
    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


def test_generate_tool_requires_intact_starter_markers_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8").replace(
            "    # <cayu:generated-starter-tools>\n    # </cayu:generated-starter-tools>\n",
            "",
        ),
        encoding="utf-8",
    )
    before = _files(project)
    monkeypatch.chdir(project)

    plan = plan_tool(tool_name="lookup", agent_name="project", effect="none")

    assert plan.status == "manual_action_required"
    assert "intact machine-owned starter markers" in plan.conflicts[0]["reason"]
    assert (
        main(
            [
                "generate",
                "tool",
                "lookup",
                "--agent",
                "project",
                "--effect",
                "none",
            ]
        )
        == 1
    )
    assert "manual_action_required" in capsys.readouterr().out
    assert _files(project) == before


def test_generate_slice_dry_run_is_deterministic_and_write_free(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    before = _files(project)
    monkeypatch.chdir(project)

    command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "none",
        "--dry-run",
        "--json",
    ]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert first["schema_version"] == APP_MANIFEST_SCHEMA_VERSION
    assert first["status"] == "ready"
    assert first["authoring_state"] == "unfinished_generated_tracer_bullet"
    assert [item["path"] for item in first["preconditions"]] == ["agents/agent.py"]
    assert [edit["path"] for edit in first["edits"]] == [
        "agents/analyst.py",
        "app.py",
        "evals/analyst.py",
        "tests/test_analyst.py",
        "tools/__init__.py",
        "tools/analyze_document.py",
    ]
    assert first["verification_commands"] == [
        "uv run cayu inspect --json",
        "uv run cayu check --json",
        "uv run pytest tests/test_analyst.py",
        "uv run cayu eval run evals.analyst:build_eval",
    ]
    assert _files(project) == before


def test_generate_slice_json_selects_format_while_dry_run_controls_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert (project / "agents" / "analyst.py").is_file()
    assert (project / "tools" / "analyze_document.py").is_file()


def test_generate_slice_applies_once_and_passes_public_verification(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path), "--provider", "anthropic"]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "external",
    ]

    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    after_apply = _files(project)
    app_source = after_apply["app.py"].decode()
    assert "AlwaysRequireApprovalToolPolicy" in app_source
    assert (
        "from tools.analyze_document import "
        "AnalyzeDocumentTool, ANALYZE_DOCUMENT_TOOL_NAME" in app_source
    )
    assert "tools=[ANALYZE_DOCUMENT_TOOL_NAME]" in app_source
    assert 'tools=["analyze_document"]' not in app_source
    assert after_apply["tools/__init__.py"] == b""
    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--no-cache", "."],
        cwd=project,
        text=True,
        capture_output=True,
        check=False,
    )
    assert formatted.returncode == 0, formatted.stdout + formatted.stderr

    assert main([*command, "--json"]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["status"] == "already_present"
    assert repeated["edits"] == []
    assert _files(project) == after_apply

    for module_name in (
        "app",
        "agents.analyst",
        "agents",
        "tools.analyze_document",
        "tools",
    ):
        sys.modules.pop(module_name, None)
    assert main(["inspect", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert [agent["name"] for agent in manifest["agents"]] == ["analyst", "project"]
    analyst = next(agent for agent in manifest["agents"] if agent["name"] == "analyst")
    assert analyst["workflow_tool_names"] == ["analyze_document"]
    assert analyst["authoring_state"] == "unfinished_generated_tracer_bullet"
    agent_source = after_apply["agents/analyst.py"].decode()
    tool_source = after_apply["tools/analyze_document.py"].decode()
    eval_source = after_apply["evals/analyst.py"].decode()
    assert "from tools.analyze_document import ANALYZE_DOCUMENT_TOOL_NAME" in agent_source
    assert "workflow_tool_names=(ANALYZE_DOCUMENT_TOOL_NAME,)" in agent_source
    assert 'ANALYZE_DOCUMENT_TOOL_NAME = "analyze_document"' in tool_source
    assert "name=ANALYZE_DOCUMENT_TOOL_NAME" in tool_source
    assert "name=ANALYZE_DOCUMENT_TOOL_NAME" in eval_source
    assert main(["check", "--json"]) == 0
    diagnostics = json.loads(capsys.readouterr().out)["diagnostics"]
    assert [item["code"] for item in diagnostics] == ["AGENT_GENERATED_TRACER_BULLET_UNFINISHED"]
    assert diagnostics[0]["path"] == "agents.analyst.authoring_state"
    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["diagnostics"] == diagnostics
    eval_output = project / "analyst-eval.json"
    assert (
        main(
            [
                "eval",
                "run",
                "evals.analyst:build_eval",
                "--output",
                str(eval_output),
            ]
        )
        == 0
    )
    assert json.loads(eval_output.read_text(encoding="utf-8"))["status"] == "passed"

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_analyst.py", "-q"],
        cwd=project,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_generate_slice_rejects_an_existing_logical_agent_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "reviewer", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "reviewer"
    before = _files(project)
    monkeypatch.chdir(project)

    plan = plan_slice(name="reviewer", tool_name="assess_submission", effect="none")

    assert plan.status == "conflict"
    assert plan.conflicts == (
        {
            "path": "app.py",
            "operation": "update_region",
            "reason": (
                "agent name 'reviewer' is already registered by agents.agent.AGENT; "
                "choose a different slice name or extend the existing agent explicitly"
            ),
        },
    )
    assert (
        main(
            [
                "generate",
                "slice",
                "reviewer",
                "--tool",
                "assess_submission",
                "--effect",
                "none",
            ]
        )
        == 1
    )
    assert "agent name 'reviewer' is already registered" in capsys.readouterr().out
    assert _files(project) == before
    assert main(["inspect", "--json"]) == 0
    assert [agent["name"] for agent in json.loads(capsys.readouterr().out)["agents"]] == [
        "reviewer"
    ]


def test_generate_slice_detects_keyword_registration_and_constant_agent_name(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "reviewer", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "reviewer"
    agent_path = project / "agents" / "agent.py"
    agent_source = agent_path.read_text(encoding="utf-8")
    agent_source = agent_source.replace(
        'AGENT = AgentSpec(\n    name="reviewer",',
        'AGENT_NAME = "reviewer"\n\n\nAGENT = AgentSpec(\n    name=AGENT_NAME,',
    )
    agent_path.write_text(agent_source, encoding="utf-8")
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8").replace(
            "        AGENT,\n",
            "        spec=AGENT,\n",
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    plan = plan_slice(name="reviewer", tool_name="assess_submission", effect="none")

    assert plan.status == "conflict"
    assert "agent name 'reviewer' is already registered" in plan.conflicts[0]["reason"]


def test_generate_slice_fails_closed_for_a_reassigned_agent_symbol(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "reviewer", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "reviewer"
    agent_path = project / "agents" / "agent.py"
    agent_path.write_text(
        agent_path.read_text(encoding="utf-8").replace(
            "AGENT = AgentSpec(\n",
            ('AGENT = AgentSpec(name="other", model="gpt-5.6-luna")\n\n\nAGENT = AgentSpec(\n'),
        ),
        encoding="utf-8",
    )
    before = _files(project)
    monkeypatch.chdir(project)

    plan = plan_slice(name="reviewer", tool_name="assess_submission", effect="none")

    assert plan.status == "manual_action_required"
    assert "cannot determine the registered agent name" in plan.conflicts[0]["reason"]
    assert _files(project) == before


def test_generate_slice_does_not_exempt_a_customized_same_symbol_registration(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "none",
    ]
    assert main(command) == 0
    capsys.readouterr()
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8").replace(
            (
                "app.register_agent(\n"
                "        _agent_for_provider_override(ANALYST_AGENT, provider),\n"
                "        tools=[AnalyzeDocumentTool()],\n"
                "    )"
            ),
            (
                "app.register_agent(\n"
                "        _agent_for_provider_override(ANALYST_AGENT, provider),\n"
                "    )"
            ),
        ),
        encoding="utf-8",
    )
    before = _files(project)

    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")

    assert plan.status == "conflict"
    assert "agent name 'analyst' is already registered" in plan.conflicts[0]["reason"]
    assert main(command) == 1
    assert "extend the existing agent explicitly" in capsys.readouterr().out
    assert _files(project) == before


def test_generate_slice_detects_a_bound_registration_alias_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "reviewer", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "reviewer"
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8").replace(
            "    app.register_agent(\n",
            "    register_agent = app.register_agent\n    register_agent(\n",
            1,
        ),
        encoding="utf-8",
    )
    assert "    register_agent(\n" in app_path.read_text(encoding="utf-8")
    before = _files(project)
    monkeypatch.chdir(project)

    plan = plan_slice(name="reviewer", tool_name="assess_submission", effect="none")

    assert plan.status == "conflict"
    assert "agent name 'reviewer' is already registered" in plan.conflicts[0]["reason"]
    assert _files(project) == before


def test_generate_slice_fails_closed_for_a_function_local_agent_shadow(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "reviewer", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "reviewer"
    agent_path = project / "agents" / "agent.py"
    agent_path.write_text(
        agent_path.read_text(encoding="utf-8").replace(
            'name="reviewer"',
            'name="other"',
        ),
        encoding="utf-8",
    )
    app_path = project / "app.py"
    app_source = app_path.read_text(encoding="utf-8")
    app_source = app_source.replace(
        "from cayu import (\n",
        "from cayu import (\n    AgentSpec,\n",
    )
    app_path.write_text(
        app_source.replace(
            "    app.register_agent(\n",
            (
                '    AGENT = AgentSpec(name="reviewer", model="gpt-5.6-luna")\n'
                "    app.register_agent(\n"
            ),
            1,
        ),
        encoding="utf-8",
    )
    before = _files(project)
    monkeypatch.chdir(project)

    plan = plan_slice(name="reviewer", tool_name="assess_submission", effect="none")

    assert plan.status == "manual_action_required"
    assert "cannot determine the registered agent name" in plan.conflicts[0]["reason"]
    assert _files(project) == before


def test_generate_slice_fails_closed_for_a_pattern_bound_agent_shadow(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "reviewer", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "reviewer"
    agent_path = project / "agents" / "agent.py"
    agent_path.write_text(
        agent_path.read_text(encoding="utf-8").replace(
            'name="reviewer"',
            'name="other"',
        ),
        encoding="utf-8",
    )
    app_path = project / "app.py"
    app_source = app_path.read_text(encoding="utf-8").replace(
        "    CayuApp,\n",
        "    AgentSpec,\n    CayuApp,\n",
    )
    app_path.write_text(
        app_source.replace(
            "    app.register_agent(\n",
            (
                '    match AgentSpec(name="reviewer", model="gpt-5.6-luna"):\n'
                "        case AGENT:\n"
                "            app.register_agent(\n"
            ),
            1,
        ),
        encoding="utf-8",
    )
    before = _files(project)
    monkeypatch.chdir(project)

    plan = plan_slice(name="reviewer", tool_name="assess_submission", effect="none")

    assert plan.status == "manual_action_required"
    assert "cannot determine the registered agent name" in plan.conflicts[0]["reason"]
    assert (
        main(
            [
                "generate",
                "slice",
                "reviewer",
                "--tool",
                "assess_submission",
                "--effect",
                "none",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["status"] == "manual_action_required"
    assert _files(project) == before
    assert main(["inspect", "--json"]) == 0
    assert [agent["name"] for agent in json.loads(capsys.readouterr().out)["agents"]] == [
        "reviewer"
    ]


def test_generate_slice_preserves_formatted_multiline_registrations(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    analyst_command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "external",
    ]
    assert main(analyst_command) == 0
    capsys.readouterr()
    app_path = project / "app.py"
    assert (
        "    app.register_agent(\n"
        "        _agent_for_provider_override(ANALYST_AGENT, provider),"
        in app_path.read_text(encoding="utf-8")
    )
    formatted = app_path.read_bytes()

    repeated = plan_slice(name="analyst", tool_name="analyze_document", effect="external")

    assert repeated.status == "already_present"
    assert repeated.edits == ()
    assert app_path.read_bytes() == formatted
    assert (
        main(
            [
                "generate",
                "slice",
                "writer",
                "--tool",
                "write_report",
                "--effect",
                "none",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["inspect", "--json"]) == 0
    assert [agent["name"] for agent in json.loads(capsys.readouterr().out)["agents"]] == [
        "analyst",
        "project",
        "writer",
    ]
    linted = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", "."],
        cwd=project,
        text=True,
        capture_output=True,
        check=False,
    )
    assert linted.returncode == 0, linted.stdout + linted.stderr


def test_generate_slice_fails_closed_for_a_dynamic_registered_agent_name(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    agent_path = project / "agents" / "agent.py"
    agent_source = agent_path.read_text(encoding="utf-8")
    agent_source = agent_source.replace(
        'AGENT = AgentSpec(\n    name="project",',
        (
            'def agent_name() -> str:\n    return "analyst"\n\n\n'
            "AGENT = AgentSpec(\n    name=agent_name(),"
        ),
    )
    agent_path.write_text(agent_source, encoding="utf-8")
    before = _files(project)
    monkeypatch.chdir(project)

    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")

    assert plan.status == "manual_action_required"
    assert "cannot determine the registered agent name" in plan.conflicts[0]["reason"]
    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
            ]
        )
        == 1
    )
    assert "without executing project code" in capsys.readouterr().out
    assert _files(project) == before


def test_generated_tool_package_wins_over_an_unrelated_installed_tools_package(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
            ]
        )
        == 0
    )
    capsys.readouterr()

    unrelated = tmp_path / "unrelated"
    unrelated_tools = unrelated / "tools"
    unrelated_tools.mkdir(parents=True)
    (unrelated_tools / "__init__.py").write_text("SOURCE = 'unrelated'\n", encoding="utf-8")
    framework_source = Path(__file__).parents[2] / "src"
    completed = subprocess.run(
        [sys.executable, "-m", "cayu", "inspect", "--json"],
        cwd=project,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(framework_source), str(unrelated))),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert [agent["name"] for agent in json.loads(completed.stdout)["agents"]] == [
        "analyst",
        "project",
    ]


def test_generate_slice_completion_is_explicit_and_preserves_customized_files(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "none",
    ]

    assert main(command) == 0
    capsys.readouterr()
    agent_path = project / "agents" / "analyst.py"
    tool_path = project / "tools" / "analyze_document.py"
    test_path = project / "tests" / "test_analyst.py"
    eval_path = project / "evals" / "analyst.py"

    agent_source = agent_path.read_text(encoding="utf-8").replace(
        "Use {ANALYZE_DOCUMENT_TOOL_NAME} when it directly answers the user's request.",
        "Use {ANALYZE_DOCUMENT_TOOL_NAME} to assess the submitted proposal.",
    )
    agent_path.write_text(agent_source, encoding="utf-8")
    for module_name in ("app", "agents.analyst", "agents", "tools.analyze_document", "tools"):
        sys.modules.pop(module_name, None)

    # Domain-looking source remains unfinished while the explicit marker is present.
    assert main(["check", "--json"]) == 0
    marked = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in marked["diagnostics"]] == [
        "AGENT_GENERATED_TRACER_BULLET_UNFINISHED"
    ]

    agent_source = agent_path.read_text(encoding="utf-8")
    agent_source = agent_source.replace("AgentAuthoringState, ", "")
    agent_source = agent_source.replace(
        "    authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET,\n",
        "",
    )
    agent_source += "\n# Completed domain text may still mention sample, echo, or tracer bullet.\n"
    agent_path.write_text(agent_source, encoding="utf-8")

    tool_source = tool_path.read_text(encoding="utf-8")
    tool_source = tool_source.replace(
        "Process one explicit input for the analyst agent.",
        "Assess one proposal document for the analyst agent.",
    )
    tool_source = tool_source.replace('"input"', '"document"')
    tool_source = tool_source.replace("args['input']", "args['document']")
    tool_path.write_text(tool_source, encoding="utf-8")

    for path in (test_path, eval_path):
        source = path.read_text(encoding="utf-8")
        source = source.replace('"input": "sample"', '"document": "proposal"')
        source = source.replace("Process sample", "Review proposal")
        source = source.replace("analyst completed sample.", "analyst completed review.")
        source = source.replace('FinalOutputContains("sample")', 'FinalOutputContains("review")')
        path.write_text(source, encoding="utf-8")

    for module_name in ("app", "agents.analyst", "agents", "tools.analyze_document", "tools"):
        sys.modules.pop(module_name, None)

    assert main(["inspect", "--agent", "analyst", "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["agents"][0]["authoring_state"] is None
    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_analyst.py", "-q"],
        cwd=project,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    eval_output = project / "analyst-completed-eval.json"
    assert (
        main(
            [
                "eval",
                "run",
                "evals.analyst:build_eval",
                "--output",
                str(eval_output),
            ]
        )
        == 0
    )
    assert json.loads(eval_output.read_text(encoding="utf-8"))["status"] == "passed"

    completed_files = _stable_files(project)
    assert main([*command, "--json"]) == 1
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["status"] == "conflict"
    assert repeated["authoring_state"] == "unfinished_generated_tracer_bullet"
    assert _stable_files(project) == completed_files


def test_generate_slice_missing_registration_seam_requires_manual_action_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8").replace(
            "    # <cayu:generated-registrations>\n    # </cayu:generated-registrations>\n",
            "",
        ),
        encoding="utf-8",
    )
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 1
    )
    plan = json.loads(capsys.readouterr().out)

    assert plan["status"] == "manual_action_required"
    assert plan["conflicts"][0]["path"] == "app.py"
    assert _files(project) == before


def test_generate_slice_conflict_preserves_user_files_and_registration(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    user_file = project / "agents" / "analyst.py"
    user_file.write_text("# user-owned\n", encoding="utf-8")
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 1
    )
    plan = json.loads(capsys.readouterr().out)

    assert plan["status"] == "conflict"
    assert plan["conflicts"] == [
        {
            "path": "agents/analyst.py",
            "operation": "create",
            "reason": "path exists with user-authored or different content",
        }
    ]
    assert _files(project) == before


def test_generate_slice_rejects_python_keywords_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "class",
                "--tool",
                "await",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)

    assert result["error"]["code"] == "GENERATOR_PLAN_FAILED"
    assert "Python keyword" in result["error"]["message"]
    assert _files(project) == before


def test_generate_slice_rejects_symlinked_generated_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    agents = project / "agents"
    agents.rename(project / "real-agents")
    agents.symlink_to(outside, target_is_directory=True)
    before = _files(project)
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "generate",
                "slice",
                "analyst",
                "--tool",
                "analyze_document",
                "--effect",
                "none",
                "--json",
            ]
        )
        == 1
    )
    plan = json.loads(capsys.readouterr().out)

    assert plan["status"] == "conflict"
    assert plan["conflicts"][0] == {
        "path": "agents/analyst.py",
        "operation": "create",
        "reason": "generated path contains a symbolic link: agents",
    }
    assert not (outside / "analyst.py").exists()
    assert _files(project) == before


def test_apply_slice_plan_rejects_stale_preimages_without_writes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    app_path = project / "app.py"
    app_path.write_text(app_path.read_text(encoding="utf-8") + "\n# concurrent edit\n")
    before = _files(project)

    with pytest.raises(GeneratorApplyError, match="changed after the plan was created"):
        apply_slice_plan(plan)

    assert _files(project) == before


def test_apply_slice_plan_rejects_a_stale_registered_agent_source(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    agent_path = project / "agents" / "agent.py"
    agent_path.write_text(
        agent_path.read_text(encoding="utf-8").replace(
            'name="project"',
            'name="analyst"',
        ),
        encoding="utf-8",
    )
    before = _files(project)

    with pytest.raises(GeneratorApplyError, match="agents/agent.py changed"):
        apply_slice_plan(plan)

    assert _files(project) == before


def test_apply_slice_plan_rejects_a_removed_tools_package_initializer(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    tools = project / "tools"
    tools.mkdir()
    init_path = tools / "__init__.py"
    init_path.write_text("# user package\n", encoding="utf-8")
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    init_path.unlink()
    before = _files(project)

    with pytest.raises(GeneratorApplyError, match="tools/__init__.py changed"):
        apply_slice_plan(plan)

    assert _files(project) == before


def test_apply_slice_plan_rejects_a_stale_preserved_slice_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    command = [
        "generate",
        "slice",
        "analyst",
        "--tool",
        "analyze_document",
        "--effect",
        "none",
    ]
    assert main(command) == 0
    capsys.readouterr()
    (project / "evals" / "analyst.py").unlink()
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    assert plan.status == "ready"
    tool_path = project / "tools" / "analyze_document.py"
    tool_path.write_text(
        tool_path.read_text(encoding="utf-8") + "\n# concurrent edit\n",
        encoding="utf-8",
    )
    before = _files(project)

    with pytest.raises(GeneratorApplyError, match="tools/analyze_document.py changed"):
        apply_slice_plan(plan)

    assert _files(project) == before


def test_apply_slice_plan_rejects_a_symlink_introduced_after_planning(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    agents = project / "agents"
    agents.rename(project / "real-agents")
    agents.symlink_to(outside, target_is_directory=True)
    before = _files(project)

    with pytest.raises(GeneratorApplyError, match="symbolic link: agents"):
        apply_slice_plan(plan)

    assert not (outside / "analyst.py").exists()
    assert _files(project) == before


def test_apply_slice_plan_rolls_back_a_mid_commit_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    monkeypatch.chdir(project)
    plan = plan_slice(name="analyst", tool_name="analyze_document", effect="none")
    before = _files(project)
    real_replace = Path.replace
    commits = 0

    def fail_second_commit(source: Path, target: Path) -> Path:
        nonlocal commits
        if ".cayu-generate-" in source.as_posix() and ".cayu-generate-" not in target.as_posix():
            commits += 1
            if commits == 2:
                raise OSError("simulated commit failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_commit)

    with pytest.raises(GeneratorApplyError, match="simulated commit failure"):
        apply_slice_plan(plan)

    assert _files(project) == before
    assert not list(project.glob(".cayu-generate-*"))
