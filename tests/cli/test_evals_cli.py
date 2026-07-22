from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EvalCase,
    EvalPlan,
    EvalStatus,
    EvalSuite,
    FinalOutputContains,
    Message,
    RunRequest,
    load_eval_run,
)
from cayu.cli import main
from cayu.cli.evals import add_eval_parser
from cayu.providers import ModelProvider, ModelStreamEvent


def _captured_eval_error(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1"
    assert payload["error"]["code"] == "EVAL_COMMAND_FAILED"
    return payload["error"]["message"]


def _write_cayu_project_config(
    root: Path,
    *,
    factory_target: str | None = "app:build_app",
    eval_declaration: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = ["[tool.cayu]"]
    if factory_target is not None:
        lines.append(f'factory = "{factory_target}"')
    if eval_declaration is not None:
        lines.append(eval_declaration)
    path = root / "pyproject.toml"
    path.write_text("\n".join((*lines, "")), encoding="utf-8")
    return path


class _SlowProvider(ModelProvider):
    name = "slow"

    async def stream(self, request):
        await asyncio.sleep(0.2)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def build_slow_eval_plan() -> EvalPlan:
    app = CayuApp(enable_logging=False)
    app.register_provider(_SlowProvider(), default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    suite = EvalSuite(
        id="slow-suite",
        cases=[
            EvalCase(
                id="slow-case",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                    max_steps=1,
                ),
                assertions=[FinalOutputContains("done")],
            )
        ],
    )
    return EvalPlan(app=app, suite=suite)


async def build_async_eval_plan() -> EvalPlan:
    await asyncio.sleep(0)
    return build_slow_eval_plan()


def test_eval_run_parses_optional_case_timeout_as_float() -> None:
    parser = argparse.ArgumentParser(prog="cayu")
    subparsers = parser.add_subparsers(dest="command")
    add_eval_parser(subparsers)

    configured = parser.parse_args(
        [
            "eval",
            "run",
            "example:build",
            "--case-timeout-seconds",
            "0.05",
            "--json",
        ]
    )
    timeout_omitted = parser.parse_args(["eval", "run", "example:build"])
    target_omitted = parser.parse_args(["eval", "run"])

    assert configured.case_timeout_seconds == 0.05
    assert configured.output_format == "json"
    assert timeout_omitted.case_timeout_seconds is None
    assert target_omitted.target is None


def test_eval_help_describes_configured_and_explicit_run_targets(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["eval", "--help"])

    assert raised.value.code == 0
    assert "Run a configured or explicit eval plan." in capsys.readouterr().out


def test_eval_run_discovers_async_default_target_from_nested_project_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    nested = project / "agents" / "reviewer"
    nested.mkdir(parents=True)
    _write_cayu_project_config(
        project,
        eval_declaration=f'eval_target = "{__name__}:build_async_eval_plan"',
    )
    monkeypatch.chdir(nested)

    assert main(["eval", "run", "--output", "eval-run.json"]) == 0

    report_path = project / "eval-run.json"
    assert report_path.is_file()
    assert load_eval_run(report_path).suite_id == "slow-suite"
    assert not (nested / "eval-run.json").exists()
    assert Path.cwd() == nested


def test_eval_run_explicit_target_overrides_configured_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_cayu_project_config(
        tmp_path,
        eval_declaration='eval_target = "missing_default_eval:build"',
    )
    output = tmp_path / "explicit-eval.json"
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "eval",
                "run",
                f"{__name__}:build_async_eval_plan",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert load_eval_run(output).suite_id == "slow-suite"


def test_eval_report_and_compare_do_not_require_project_discovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_path = tmp_path / "eval-run.json"
    report_path = tmp_path / "eval-report.json"
    comparison_path = tmp_path / "comparison.json"
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "eval",
                "run",
                f"{__name__}:build_async_eval_plan",
                "--output",
                str(run_path),
            ]
        )
        == 0
    )
    (tmp_path / "pyproject.toml").write_text("[tool.cayu\n", encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "report",
                str(run_path),
                "--format",
                "json",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "eval",
                "compare",
                str(run_path),
                str(run_path),
                "--output",
                str(comparison_path),
            ]
        )
        == 0
    )
    assert report_path.is_file()
    assert comparison_path.is_file()


def test_eval_run_does_not_climb_past_nearest_project_missing_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    parent = tmp_path / "parent"
    project = parent / "project"
    nested = project / "nested"
    nested.mkdir(parents=True)
    parent_marker = parent / "parent-eval-loaded.txt"
    _write_cayu_project_config(
        parent,
        factory_target="parent_app:build_app",
        eval_declaration='eval_target = "parent_eval:build"',
    )
    (parent / "parent_eval.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(parent_marker)!r}).write_text('loaded', encoding='utf-8')\n",
        encoding="utf-8",
    )
    project_pyproject = _write_cayu_project_config(
        project,
        factory_target="project_app:build_app",
    )
    monkeypatch.chdir(nested)

    assert main(["eval", "run"]) == 1

    error = _captured_eval_error(capsys)
    assert str(project_pyproject) in error
    assert "[tool.cayu].eval_target is not configured" in error
    assert not parent_marker.exists()


def test_eval_run_does_not_climb_past_eval_target_only_project(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    parent = tmp_path / "parent"
    project = parent / "project"
    nested = project / "nested"
    nested.mkdir(parents=True)
    parent_marker = parent / "parent-eval-loaded.txt"
    _write_cayu_project_config(
        parent,
        factory_target="parent_app:build_app",
        eval_declaration='eval_target = "parent_eval:build"',
    )
    (parent / "parent_eval.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(parent_marker)!r}).write_text('loaded', encoding='utf-8')\n",
        encoding="utf-8",
    )
    project_pyproject = _write_cayu_project_config(
        project,
        factory_target=None,
        eval_declaration='eval_target = "project_eval:build"',
    )
    monkeypatch.chdir(nested)

    assert main(["eval", "run"]) == 1

    error = _captured_eval_error(capsys)
    assert str(project_pyproject) in error
    assert "[tool.cayu].factory is not configured" in error
    assert 'factory = "module:build_app"' in error
    assert not parent_marker.exists()


def test_eval_run_reports_non_utf8_pyproject_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_bytes(b"[tool.cayu]\nfactory = \xff\n")
    monkeypatch.chdir(tmp_path)

    assert main(["eval", "run"]) == 1

    error = _captured_eval_error(capsys)
    assert f"Could not read {pyproject}" in error
    assert "utf-8" in error


def test_eval_run_reports_missing_or_malformed_configured_target(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    monkeypatch.chdir(tmp_path)

    assert main(["eval", "run"]) == 1
    missing_project_error = _captured_eval_error(capsys)
    assert "No Cayu project found" in missing_project_error
    assert '[tool.cayu] factory = "module:build_app"' in missing_project_error
    assert 'eval_target = "module:build_eval"' in missing_project_error
    assert "cayu eval run module:build_eval" in missing_project_error

    for declaration in ('eval_target = ""', "eval_target = 42"):
        _write_cayu_project_config(
            tmp_path,
            eval_declaration=declaration,
        )

        assert main(["eval", "run"]) == 1
        error = _captured_eval_error(capsys)
        assert str(pyproject) in error
        assert "[tool.cayu].eval_target must be a non-empty string" in error


def test_eval_run_timeout_returns_nonzero_and_saves_actionable_error(tmp_path: Path) -> None:
    output = tmp_path / "eval-run.json"

    exit_code = main(
        [
            "eval",
            "run",
            f"{__name__}:build_slow_eval_plan",
            "--case-timeout-seconds",
            "0.01",
            "--output",
            str(output),
        ]
    )

    report = load_eval_run(output)
    assert exit_code == 1
    assert report.status == EvalStatus.ERROR
    assert report.cases[0].status == EvalStatus.ERROR
    assert report.cases[0].error == "Eval case timed out after 0.01 seconds."


def test_eval_run_imports_target_from_cwd_without_leaking_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module_name = "repo_local_eval_target"
    first_project = tmp_path / "first"
    second_project = tmp_path / "second"
    first_project.mkdir()
    second_project.mkdir()
    (first_project / f"{module_name}.py").write_text(
        """from pathlib import Path

calls = 0


def build():
    global calls
    calls += 1
    Path("build-count.txt").write_text(str(calls), encoding="utf-8")
    return object()
""",
        encoding="utf-8",
    )
    project_paths = {str(first_project), str(second_project)}
    original_path = [entry for entry in sys.path if entry != "" and entry not in project_paths]
    monkeypatch.setattr(sys, "path", list(original_path))
    monkeypatch.chdir(first_project)
    sys.modules.pop(module_name, None)

    assert main(["eval", "run", f"{module_name}:build"]) == 1
    assert "Eval target must return EvalPlan" in _captured_eval_error(capsys)
    assert (first_project / "build-count.txt").read_text(encoding="utf-8") == "1"
    assert module_name not in sys.modules
    assert sys.path == original_path

    sys.modules.pop(module_name, None)
    monkeypatch.chdir(second_project)
    assert main(["eval", "run", f"{module_name}:build"]) == 1
    error = _captured_eval_error(capsys)
    assert "Command-line eval target could not be loaded" in error
    assert f"No module named '{module_name}'" in error
    assert sys.path == original_path


def test_eval_run_prioritizes_preexisting_cwd_path_and_restores_order(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module_name = "repo_local_eval_with_preexisting_path"
    installed = tmp_path / "installed"
    project = tmp_path / "project"
    installed.mkdir()
    project.mkdir()
    (installed / f"{module_name}.py").write_text(
        'source = "installed"\n\ndef build():\n    return object()\n',
        encoding="utf-8",
    )
    (project / f"{module_name}.py").write_text(
        """import json
import sys
from pathlib import Path

source = "local"
import_path = list(sys.path)
Path("import-state.json").write_text(
    json.dumps({"source": source, "import_path": import_path}),
    encoding="utf-8",
)


def build():
    return object()
""",
        encoding="utf-8",
    )
    cwd = str(project)
    excluded = {"", cwd, str(installed)}
    existing_path = [
        str(installed),
        cwd,
        *(entry for entry in sys.path if entry not in excluded),
    ]
    monkeypatch.setattr(sys, "path", list(existing_path))
    monkeypatch.chdir(project)
    sys.modules.pop(module_name, None)

    assert main(["eval", "run", f"{module_name}:build"]) == 1
    assert "Eval target must return EvalPlan" in _captured_eval_error(capsys)
    loaded = json.loads((project / "import-state.json").read_text(encoding="utf-8"))
    assert loaded["source"] == "local"
    assert loaded["import_path"][0] == cwd
    assert loaded["import_path"].count(cwd) == 1
    assert sys.path == existing_path
    assert module_name not in sys.modules


def test_eval_run_reports_clear_target_resolution_errors(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module_name = "repo_local_eval_with_missing_attribute"
    (tmp_path / f"{module_name}.py").write_text("present = object()\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop(module_name, None)

    assert main(["eval", "run", "not-a-target"]) == 1
    syntax_error = _captured_eval_error(capsys)
    assert "Command-line eval target must use module:attribute syntax" in syntax_error

    assert main(["eval", "run", "missing_repo_local_eval:build"]) == 1
    missing_error = _captured_eval_error(capsys)
    assert "Command-line eval target could not be loaded" in missing_error
    assert "No module named 'missing_repo_local_eval'" in missing_error

    assert main(["eval", "run", f"{module_name}:missing"]) == 1
    attribute_error = _captured_eval_error(capsys)
    assert "Command-line eval target could not be loaded" in attribute_error
    assert "has no attribute 'missing'" in attribute_error


def test_eval_run_configured_target_errors_identify_pyproject_source(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pyproject = _write_cayu_project_config(
        tmp_path,
        eval_declaration='eval_target = "not-a-target"',
    )
    monkeypatch.chdir(tmp_path)

    assert main(["eval", "run"]) == 1

    syntax_error = _captured_eval_error(capsys)
    assert f"Configured eval target from {pyproject}" in syntax_error
    assert "must use module:attribute syntax" in syntax_error

    _write_cayu_project_config(
        tmp_path,
        eval_declaration='eval_target = "missing_configured_eval:build"',
    )
    assert main(["eval", "run"]) == 1

    error = _captured_eval_error(capsys)
    assert f"Configured eval target from {pyproject}" in error
    assert "No module named 'missing_configured_eval'" in error

    (tmp_path / "invalid_configured_eval.py").write_text(
        "def build():\n    return object()\n",
        encoding="utf-8",
    )
    _write_cayu_project_config(
        tmp_path,
        eval_declaration='eval_target = "invalid_configured_eval:build"',
    )
    sys.modules.pop("invalid_configured_eval", None)

    assert main(["eval", "run"]) == 1

    invalid_error = _captured_eval_error(capsys)
    assert f"Configured eval target from {pyproject}" in invalid_error
    assert "returned an invalid eval plan" in invalid_error
    assert "Eval target must return EvalPlan" in invalid_error
