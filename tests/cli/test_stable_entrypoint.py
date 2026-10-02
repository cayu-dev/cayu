from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from tests.core.test_openai_subscription_provider import StaticSubscriptionAuth

import cayu.entrypoint as entrypoint_module
from cayu import (
    AgentSpec,
    CayuApp,
    ModelStreamEvent,
    ScriptedModelProvider,
    run_project_entrypoint,
)
from cayu.cli import _build_parser
from cayu.cli import main as cayu_main
from cayu.cli.project import project_context
from cayu.configuration import MAX_STEPS
from cayu.providers import HttpxOpenAITransport, OpenAISubscriptionProvider
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


def test_every_cli_command_help_has_a_purpose_and_next_step() -> None:
    pending = [_build_parser()]
    descriptions: dict[str, str | None] = {}
    while pending:
        parser = pending.pop()
        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for _name, child in action.choices.items():
                descriptions[child.prog] = child.description
                pending.append(child)

    assert descriptions
    assert not {command: value for command, value in descriptions.items() if not value}


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


@pytest.mark.parametrize("max_steps", [64, 257, 10000])
def test_generated_command_runs_the_only_registered_agent(
    tmp_path: Path, capsys, max_steps
) -> None:
    project = _scaffold(tmp_path)
    capsys.readouterr()
    provider = _completed_provider("Review result.")

    with _generated_command(project, provider) as command:
        result = command.main(["--message", "Review this change.", "--max-steps", str(max_steps)])

    assert result == 0
    captured = capsys.readouterr()
    assert captured.out == "Review result.\n"
    assert captured.err == ""
    assert provider.requests[0].messages[-1].content[0].text == "Review this change."


def test_entrypoint_leaves_the_app_owned_step_default_unset() -> None:
    args = entrypoint_module._parser().parse_args(["--message", "Review this change."])

    assert args.max_steps is None


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
        "setup error: no provider is selected; set CAYU_PROVIDER to openai, anthropic, "
        "openrouter, cayu-gateway, or openai-subscription (credentials do not select a provider)\n"
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
        str(MAX_STEPS + 1),
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == f"setup error: --max-steps must be at most {MAX_STEPS}\n"


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


def test_generated_provider_validation_does_not_infer_intent_from_model_patterns(
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

    assert result == 2
    assert "no provider is selected" in capsys.readouterr().err
    assert routed_provider.requests == []


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
            ["--message", "Run.", "--max-steps", str(MAX_STEPS + 1)],
        )
        == 2
    )
    assert f"--max-steps must be at most {MAX_STEPS}" in capsys.readouterr().err


@pytest.mark.parametrize("show", [False, True])
def test_entrypoint_prints_provider_errors_only_when_asked(capsys, show) -> None:
    detail = "The 'gpt-5.4' model is not supported when using Codex with a ChatGPT account."
    transport = HttpxOpenAITransport()
    transport._client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"detail": detail}))
    )

    def factory() -> CayuApp:
        app = CayuApp()
        app.register_provider(
            OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))
        return app

    argv = ["--message", "Run.", *(["--show-provider-errors"] if show else [])]
    assert run_project_entrypoint(factory, argv) == 1
    err = capsys.readouterr().err

    assert err.startswith("run failed:")
    if show:
        assert f"provider error: HTTP 400; {detail}" in err
        assert "--show-provider-errors" not in err
    else:
        # The run failure already carries the backend's reason (#1974).
        assert detail in err
        assert "Rerun with --show-provider-errors" in err


@pytest.mark.parametrize("prefix_length", [0, 4070])
def test_entrypoint_redacts_workload_secrets_before_bounding_provider_errors(
    capsys, prefix_length
) -> None:
    secret = "synthetic-workload-secret-1970-boundary"
    detail = "x" * prefix_length + secret + " invalid input"
    transport = HttpxOpenAITransport()
    transport._client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"detail": detail}))
    )

    def factory() -> CayuApp:
        app = CayuApp(secret_redactor=SecretRedactor([secret]))
        app.register_provider(
            OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))
        return app

    assert run_project_entrypoint(factory, ["--message", "Run.", "--show-provider-errors"]) == 1
    err = capsys.readouterr().err
    assert "provider error: HTTP 400;" in err
    if prefix_length:
        assert "...[truncated]" in err
    else:
        assert REDACTED_SECRET in err
    assert secret[:12] not in err
    assert secret not in err


def test_entrypoint_closes_the_app_after_the_run(capsys) -> None:
    closed: list[str] = []

    class ClosedMarker:
        async def close(self) -> None:
            closed.append("closed")

    apps: list[CayuApp] = []

    def factory() -> CayuApp:
        app = CayuApp(enable_logging=False, owned_resources=(ClosedMarker(),))
        app.register_provider(_completed_provider("Answered."), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        apps.append(app)
        return app

    assert run_project_entrypoint(factory, ["--message", "Answer."]) == 0
    assert capsys.readouterr().out.strip() == "Answered."
    assert apps[0].lifecycle_state == "closed"
    assert closed == ["closed"]


def test_entrypoint_shutdown_owns_the_model_policy_stop(capsys) -> None:
    release = threading.Event()
    observed: list[str] = []
    apps: list[CayuApp] = []

    def factory() -> CayuApp:
        app = CayuApp(enable_logging=False)
        app.register_provider(_completed_provider("Answered."), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def blocked_stop_model_policy() -> None:
            # Unblocked only once shutdown is seen running, so the stop happens
            # inside aclose() rather than before it.
            await asyncio.to_thread(release.wait, 10)

        app.stop_model_policy = blocked_stop_model_policy  # ty: ignore[invalid-assignment]
        apps.append(app)
        return app

    def watch() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if apps and apps[0].lifecycle_state != "open":
                observed.append(apps[0].lifecycle_state)
                break
            time.sleep(0.01)
        release.set()

    watcher = threading.Thread(target=watch)
    watcher.start()
    assert run_project_entrypoint(factory, ["--message", "Answer."]) == 0
    watcher.join()
    assert observed == ["closing"]
    assert apps[0].lifecycle_state == "closed"
    assert capsys.readouterr().out.strip() == "Answered."
