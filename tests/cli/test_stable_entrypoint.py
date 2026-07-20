from __future__ import annotations

import importlib
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from cayu import ModelStreamEvent, ScriptedModelProvider, run_project_entrypoint
from cayu.cli import main as cayu_main
from cayu.cli.project import project_context


def _completed_provider(text: str) -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta(text),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )


def _scaffold(tmp_path: Path) -> Path:
    assert cayu_main(["new", "project", "--dir", str(tmp_path)]) == 0
    return tmp_path / "project"


def _run_generated_subprocess(project: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("CAYU_OPENAI_SUBSCRIPTION", None)
    return subprocess.run(
        [sys.executable, "run.py", *argv],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _generate_analyst(project: Path, monkeypatch) -> None:
    monkeypatch.chdir(project)
    assert (
        cayu_main(
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


@contextmanager
def _factory(project: Path, provider: ScriptedModelProvider):
    with project_context(project):
        app_module = importlib.import_module("app")
        yield lambda: app_module.build_app(provider=provider)


@contextmanager
def _generated_command(project: Path, provider: ScriptedModelProvider):
    with project_context(project):
        app_module = importlib.import_module("app")
        command_module = cast("Any", importlib.import_module("run"))
        command_module.build_app = lambda **_kwargs: app_module.build_app(provider=provider)
        yield command_module


def test_generated_command_runs_the_only_registered_agent(tmp_path: Path, capsys) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    provider = _completed_provider("Review result.")

    with _generated_command(project, provider) as command:
        result = command.main(["--message", "Review this change."])

    assert result == 0
    captured = capsys.readouterr()
    assert captured.out == "Review result.\n"
    assert captured.err == ""
    assert provider.requests[0].messages[-1].content[0].text == "Review this change."


def test_generated_command_auto_selects_a_renamed_starter(tmp_path: Path, capsys) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    agent_path = project / "agents" / "agent.py"
    agent_path.write_text(
        agent_path.read_text(encoding="utf-8").replace(
            'name="project"',
            'name="reviewer"',
            1,
        ),
        encoding="utf-8",
    )
    provider = _completed_provider("Renamed result.")

    with _generated_command(project, provider) as command:
        result = command.main(["--message", "Review this change."])

    assert result == 0
    assert capsys.readouterr().out == "Renamed result.\n"
    assert len(provider.requests) == 1


def test_generated_command_explains_how_to_configure_a_live_provider(
    tmp_path: Path,
    capsys,
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    completed = _run_generated_subprocess(
        project,
        "--message",
        "Review this change.",
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        "setup error: no live OpenAI provider is configured; set OPENAI_API_KEY, or run "
        "`cayu auth openai login` and set CAYU_OPENAI_SUBSCRIPTION=1.\n"
    )


def test_generated_command_rejects_steps_above_the_runtime_limit(
    tmp_path: Path,
    capsys,
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()

    completed = _run_generated_subprocess(
        project,
        "--message",
        "Review this change.",
        "--max-steps",
        "257",
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "setup error: --max-steps must be at most 256\n"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            ["--agent", "missing", "--message", "Choose."],
            "unknown agent 'missing'; available agents: analyst, project",
        ),
        (
            ["--message", "Choose."],
            "multiple agents are registered; pass --agent NAME (available: analyst, project)",
        ),
    ],
)
def test_generated_command_validates_agent_selection_before_provider_setup(
    tmp_path: Path,
    monkeypatch,
    capsys,
    arguments: list[str],
    expected: str,
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    _generate_analyst(project, monkeypatch)
    capsys.readouterr()

    completed = _run_generated_subprocess(project, *arguments)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == f"setup error: {expected}\n"


def test_generated_command_selects_a_generated_agent(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    _generate_analyst(project, monkeypatch)
    capsys.readouterr()
    provider = _completed_provider("Analyst result.")

    with _generated_command(project, provider) as command:
        result = command.main(
            ["--agent", "analyst", "--message", "Analyze this."],
        )

    assert result == 0
    assert capsys.readouterr().out == "Analyst result.\n"
    assert len(provider.requests) == 1


def test_generated_provider_validation_follows_model_pattern_routing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    routed_provider = _completed_provider("Routed result.")

    with project_context(project):
        app_module = importlib.import_module("app")
        command_module = cast("Any", importlib.import_module("run"))
        app = app_module.build_app()
        app.register_provider(routed_provider, model_patterns=("gpt-5.6-*",))
        command_module.build_app = lambda: app
        result = command_module.main(["--message", "Route this."])

    assert result == 0
    assert capsys.readouterr().out == "Routed result.\n"
    assert len(routed_provider.requests) == 1


def test_entrypoint_lists_agents_before_calling_provider(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    _generate_analyst(project, monkeypatch)
    capsys.readouterr()
    provider = _completed_provider("must not run")

    with _factory(project, provider) as factory:
        assert run_project_entrypoint(factory, ["--message", "Choose."]) == 2
        ambiguous = capsys.readouterr().err
        assert "pass --agent NAME" in ambiguous
        assert "analyst, project" in ambiguous

        assert (
            run_project_entrypoint(
                factory,
                ["--agent", "missing", "--message", "Choose."],
            )
            == 2
        )
        unknown = capsys.readouterr().err
        assert "unknown agent 'missing'" in unknown
        assert "analyst, project" in unknown

    assert provider.requests == []


def test_entrypoint_reports_run_and_project_configuration_failures(
    tmp_path: Path,
    capsys,
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    provider = ScriptedModelProvider([])

    with _factory(project, provider) as factory:
        assert run_project_entrypoint(factory, ["--message", "Run."]) == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    assert failed.err.startswith("run failed:")
    assert "session " in failed.err

    def broken_factory():
        raise RuntimeError("provider is not configured")

    assert run_project_entrypoint(broken_factory, ["--message", "Run."]) == 2
    assert capsys.readouterr().err == "setup error: provider is not configured\n"


def test_entrypoint_rejects_blank_messages_and_invalid_step_limits(capsys) -> None:
    def must_not_build():
        raise AssertionError("factory must not run")

    assert run_project_entrypoint(must_not_build, ["--message", "   "]) == 2
    assert "--message must not be blank" in capsys.readouterr().err

    assert (
        run_project_entrypoint(
            must_not_build,
            ["--message", "Run.", "--max-steps", "0"],
        )
        == 2
    )
    assert "--max-steps must be at least 1" in capsys.readouterr().err

    assert (
        run_project_entrypoint(
            must_not_build,
            ["--message", "Run.", "--max-steps", "257"],
        )
        == 2
    )
    assert "--max-steps must be at most 256" in capsys.readouterr().err
