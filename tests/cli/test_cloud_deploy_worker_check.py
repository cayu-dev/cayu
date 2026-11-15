"""`cayu cloud deploy` refuses, before uploading anything, a `[worker]` command that
`cayu check --deploy` reports as an error."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_deploy_check import (
    DEPLOY_CHECK_ENVIRONMENT_HINT,
    DEPLOY_CHECK_WORKER_ENVIRONMENT_HINT,
    deploy_check_hint,
    deploy_check_refusal,
    run_cloud_deploy_check,
)
from cayu.cli._worker_process_check import worker_command_findings

_MODULE = "cloud_deploy_worker_check_app"

_RECONCILING_WORKER = """\
from cayu import AgentSpec, CayuApp
from cayu.evals import ScriptedModelProvider
from cayu.tasks import run_task_worker


def build_app():
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="agent", model="scripted-model"))
    return app


async def run_shepherd(app, stop):
    options = {"reconcile_abandoned_after_s": 120, "restart_abandoned": True, "stop": stop}
    return await run_task_worker(app, app.task_store, None, **options)
"""

# Each case `cayu check --deploy` reports as an error, as (command, pyproject addition,
# manifest [env], code).
_ERRORS = [
    ("cayu worker shepherd", "", {}, "WORKER_SINGLE_PROCESS_REQUIRED"),
    ("cayu worker shepherd --processes 3", "", {}, "WORKER_SINGLE_PROCESS_REQUIRED"),
    ("cayu worker shepherd", "", {"CAYU_WORKER_PROCESSES": "4"}, "WORKER_SINGLE_PROCESS_REQUIRED"),
    (
        "cayu worker shepherd",
        '\n[tool.cayu.worker_processes]\nshepherd = "auto"\n',
        {},
        "WORKER_SINGLE_PROCESS_REQUIRED",
    ),
    (
        "cayu worker shepherd",
        "\n[tool.cayu.worker_processes]\nshepherd = 2\n",
        {},
        "WORKER_SINGLE_PROCESS_REQUIRED",
    ),
    ("cayu worker missing", "", {}, "WORKER_COMMAND_INVALID"),
    ("cayu worker shepherd --unknown-option", "", {}, "WORKER_COMMAND_INVALID"),
    (
        "cayu worker shepherd --processes 2 --reconcile-abandoned-after-seconds 60",
        "",
        {},
        "WORKER_COMMAND_INVALID",
    ),
    (
        "cayu worker shepherd",
        "\n[tool.cayu.worker_processes]\nshepherd = 0\n",
        {},
        "WORKER_COMMAND_INVALID",
    ),
    ("cayu worker shepherd --processes 1 --restart-abandoned", "", {}, "WORKER_COMMAND_INVALID"),
    (
        "cayu worker shepherd --processes 1 --max-abandoned-restarts 3",
        "",
        {},
        "WORKER_COMMAND_INVALID",
    ),
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # Only the manifest's [env] counts, never this shell's.
    monkeypatch.setenv("CAYU_WORKER_PROCESSES", "1")
    monkeypatch.setenv("CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S", "60")
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    sys.modules.pop(_MODULE, None)
    yield
    sys.modules.pop(_MODULE, None)


def _agent(
    root: Path, worker_command: str, *, extra: str = "", environment: dict[str, str] | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "worker-check"\nversion = "0.1.0"\ndependencies = ["cayu"]\n\n'
        f'[tool.cayu]\nfactory = "{_MODULE}:build_app"\n\n'
        f'[tool.cayu.workers]\nshepherd = "{_MODULE}:run_shepherd"\n{extra}',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / f"{_MODULE}.py").write_text(_RECONCILING_WORKER, encoding="utf-8")
    lines = [
        "schema_version = 2",
        'application = "worker-check"',
        'name = "Worker Check"',
        'version = "0.1.0"',
        f"entrypoint = {json.dumps(worker_command)}",
        'capabilities = ["model.generate"]',
        "cpu_millis = 1000",
        "memory_mb = 2048",
        "timeout_seconds = 900",
        'environment = "python"',
        'compatibility = "cayu>=0.1,<1"',
        'policy_version = "cayu-egress-v1"',
        "",
        "[worker]",
        f"command = {json.dumps(worker_command)}",
        "",
    ]
    if environment:
        lines += ["[env]", *(f"{key} = {json.dumps(value)}" for key, value in environment.items())]
    (root / "cayu-cloud.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class _RecordingClient:
    """Stands in for Cayu Cloud and records every request and upload it is asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("request", args))
        raise AssertionError("the deploy check must refuse before contacting Cayu Cloud")

    def upload_bytes(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("upload_bytes", args))
        raise AssertionError("the deploy check must refuse before uploading")


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    client = _RecordingClient()
    deployed: list[dict[str, Any]] = []
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *args, **kwargs: client)

    def deploy(arguments: object, **kwargs: Any) -> dict[str, Any]:
        deployed.append(kwargs)
        return {"operation": "deploy", "result": {}}

    monkeypatch.setattr(cloud_cli, "_deploy", deploy)
    return {"client": client, "deployed": deployed}


def _cloud(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, Any]:
    code = cloud_cli.run_cloud_cli(arguments)
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("command,extra,environment,code", _ERRORS)
def test_deploy_refuses_before_uploading(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cloud: dict[str, Any],
    command: str,
    extra: str,
    environment: dict[str, str],
    code: str,
) -> None:
    project = _agent(tmp_path / "agent", command, extra=extra, environment=environment)
    (expected,) = worker_command_findings(project.resolve(), command, environment)

    exit_code, payload = _cloud(["deploy", str(project)], capsys)

    assert exit_code == 2, payload
    assert cloud["client"].calls == []
    assert cloud["deployed"] == []
    error = payload["error"]
    assert error["category"] == "deploy_check_failed"
    assert "`cayu worker` process could not start" in error["message"]
    assert "Nothing was uploaded" in error["message"]
    assert "--skip-deploy-check" in error["message"]
    (blocking,) = error["deploy_check"]["blocking"]
    # The same finding `cayu check --deploy` reports: code, message and hint.
    assert blocking["code"] == code == expected.code
    assert blocking["message"] == expected.message
    assert blocking["hint"] == expected.hint
    assert blocking["path"] == "cayu-cloud.toml:[worker].command"
    assert blocking["documentation_anchor"] == expected.documentation_anchor


@pytest.mark.parametrize("command,extra,environment,code", _ERRORS)
def test_init_check_reports_the_same_refusal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cloud: dict[str, Any],
    command: str,
    extra: str,
    environment: dict[str, str],
    code: str,
) -> None:
    project = _agent(tmp_path / "agent", command, extra=extra, environment=environment)

    exit_code, payload = _cloud(["init", str(project), "--check"], capsys)

    assert exit_code == 2, payload
    assert payload["error"]["category"] == "deploy_check_failed"
    assert code in payload["error"]["message"]


@pytest.mark.parametrize(
    "command,extra,environment",
    [
        ("cayu worker shepherd --processes 1", "", {}),
        ("cayu worker shepherd", "\n[tool.cayu.worker_processes]\nshepherd = 1\n", {}),
        ("cayu worker shepherd", "", {"CAYU_WORKER_PROCESSES": "1"}),
    ],
)
def test_deploy_proceeds_for_a_worker_that_starts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cloud: dict[str, Any],
    command: str,
    extra: str,
    environment: dict[str, str],
) -> None:
    project = _agent(tmp_path / "agent", command, extra=extra, environment=environment)

    exit_code, payload = _cloud(["deploy", str(project)], capsys)

    assert exit_code == 0, payload
    (deployed,) = cloud["deployed"]
    assert deployed["deploy_check"] == {
        "command": "cayu check --deploy --fail-on warning --json",
        "status": "passed",
    }


def test_skip_deploy_check_uploads_anyway(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cloud: dict[str, Any]
) -> None:
    project = _agent(tmp_path / "agent", "cayu worker shepherd")

    exit_code, payload = _cloud(["deploy", str(project), "--skip-deploy-check"], capsys)

    assert exit_code == 0, payload
    (deployed,) = cloud["deployed"]
    assert deployed["deploy_check"] == {"status": "skipped"}


def test_a_worker_from_a_parent_project_is_not_checked(tmp_path: Path) -> None:
    # Only the uploaded directory reaches Cloud, so a parent's workers say nothing about it.
    parent = _agent(tmp_path / "parent", "cayu worker shepherd")
    child = parent / "agent"
    child.mkdir()

    check = run_cloud_deploy_check(
        child, serves_web=False, worker_command="cayu worker shepherd", worker_environment={}
    )

    assert check.status == "not_applicable"


def test_other_worker_commands_are_not_checked(tmp_path: Path) -> None:
    project = _agent(tmp_path / "agent", "python -m shepherd")

    check = run_cloud_deploy_check(
        project, serves_web=False, worker_command="python -m shepherd", worker_environment={}
    )

    assert check.status == "not_applicable"


def test_web_and_worker_findings_are_both_reported(tmp_path: Path) -> None:
    project = _agent(tmp_path / "agent", "cayu worker shepherd")

    check = run_cloud_deploy_check(
        project,
        serves_web=True,
        web_command="cayu serve --host 0.0.0.0",
        worker_command="cayu worker shepherd",
        worker_environment={},
    )

    assert check.status == "failed"
    assert "SERVE_SERVER_EXTRA_MISSING" in check.codes
    assert check.codes[-1] == "WORKER_SINGLE_PROCESS_REQUIRED"
    refusal = deploy_check_refusal(check)
    assert "`cayu serve` web process could not start" in refusal
    assert "`cayu worker` process could not start" in refusal
    assert "refuses the public service" not in refusal


def test_the_refusal_hint_says_where_the_worker_check_looks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cloud: dict[str, Any]
) -> None:
    project = _agent(tmp_path / "agent", "cayu worker shepherd")

    exit_code, payload = _cloud(["deploy", str(project)], capsys)

    assert exit_code == 2
    # Exporting the variable in this shell changes nothing for the worker check.
    assert payload["error"]["hint"] == DEPLOY_CHECK_WORKER_ENVIRONMENT_HINT


def test_web_and_worker_refusals_give_both_hints(tmp_path: Path) -> None:
    project = _agent(tmp_path / "agent", "cayu worker shepherd")
    check = run_cloud_deploy_check(
        project,
        serves_web=True,
        web_command="cayu serve --host 0.0.0.0",
        worker_command="cayu worker shepherd",
        worker_environment={},
    )

    assert deploy_check_hint(check) == (
        f"{DEPLOY_CHECK_ENVIRONMENT_HINT} {DEPLOY_CHECK_WORKER_ENVIRONMENT_HINT}"
    )
