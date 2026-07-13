from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

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


def test_eval_run_parses_optional_case_timeout_as_float() -> None:
    parser = argparse.ArgumentParser(prog="cayu")
    subparsers = parser.add_subparsers(dest="command")
    add_eval_parser(subparsers)

    configured = parser.parse_args(
        ["eval", "run", "example:build", "--case-timeout-seconds", "0.05"]
    )
    omitted = parser.parse_args(["eval", "run", "example:build"])

    assert configured.case_timeout_seconds == 0.05
    assert omitted.case_timeout_seconds is None


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
        """calls = 0


def build():
    global calls
    calls += 1
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
    assert "Eval target must return EvalPlan" in capsys.readouterr().err
    assert vars(sys.modules[module_name])["calls"] == 1
    assert sys.path == original_path

    sys.modules.pop(module_name, None)
    monkeypatch.chdir(second_project)
    assert main(["eval", "run", f"{module_name}:build"]) == 1
    assert f"No module named '{module_name}'" in capsys.readouterr().err
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
        """import sys

source = "local"
import_path = list(sys.path)


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
    assert "Eval target must return EvalPlan" in capsys.readouterr().err
    loaded = vars(sys.modules[module_name])
    assert loaded["source"] == "local"
    assert loaded["import_path"][0] == cwd
    assert loaded["import_path"].count(cwd) == 1
    assert sys.path == existing_path
    sys.modules.pop(module_name, None)


def test_eval_run_reports_clear_target_resolution_errors(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module_name = "repo_local_eval_with_missing_attribute"
    (tmp_path / f"{module_name}.py").write_text("present = object()\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop(module_name, None)

    assert main(["eval", "run", "missing_repo_local_eval:build"]) == 1
    assert "No module named 'missing_repo_local_eval'" in capsys.readouterr().err

    assert main(["eval", "run", f"{module_name}:missing"]) == 1
    assert "has no attribute 'missing'" in capsys.readouterr().err
