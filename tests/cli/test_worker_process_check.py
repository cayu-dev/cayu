"""`cayu check --deploy` refuses a `[worker]` command that would start more processes than
its worker accepts, as `cayu worker` would at startup."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli._worker_process_check import worker_command_findings
from cayu.runtime._process_workers import single_process_worker_requirement

_MODULE = "worker_process_check_app"

# As cayu-shepherd's worker passed it: through an options dictionary.
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


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Only the worker's own environment counts, never this shell's.
    monkeypatch.setenv("CAYU_WORKER_PROCESSES", "1")
    monkeypatch.setenv("CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S", "60")
    sys.modules.pop(_MODULE, None)
    yield
    sys.modules.pop(_MODULE, None)


def _project(root: Path, *, source: str = _RECONCILING_WORKER, extra: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "worker-check"\nversion = "0.1.0"\ndependencies = ["cayu"]\n\n'
        f'[tool.cayu]\nfactory = "{_MODULE}:build_app"\n\n'
        f'[tool.cayu.workers]\nshepherd = "{_MODULE}:run_shepherd"\n{extra}',
        encoding="utf-8",
    )
    (root / f"{_MODULE}.py").write_text(source, encoding="utf-8")
    return root


def _codes(root: Path, command: str, environment: dict[str, str] | None = None) -> list[str]:
    return [item.code for item in worker_command_findings(root, command, environment or {})]


def test_the_default_of_one_process_per_cpu_fails_a_single_process_worker(
    tmp_path: Path,
) -> None:
    (finding,) = worker_command_findings(_project(tmp_path), "cayu worker shepherd", {})
    assert finding.code == "WORKER_SINGLE_PROCESS_REQUIRED"
    assert finding.severity.value == "error"
    assert finding.path == "cayu-cloud.toml:[worker].command"
    assert "deploy" in finding.tags
    assert f"in {_MODULE}.py" in finding.message
    assert "one process per CPU" in finding.message
    # The rule cayu worker enforces at startup, with both fixes.
    assert single_process_worker_requirement("reconcile_abandoned_after_s") in finding.message
    assert finding.hint is not None
    assert "cayu worker shepherd --processes 1" in finding.hint
    assert "shepherd = 1 in [tool.cayu.worker_processes]" in finding.hint
    assert finding.documentation_anchor == "cayu guide diagnostics#worker-single-process-required"


@pytest.mark.parametrize(
    "command,extra,environment",
    [
        ("cayu worker shepherd --processes 1", "", {}),
        ("cayu worker shepherd", "\n[tool.cayu.worker_processes]\nshepherd = 1\n", {}),
        ("cayu worker shepherd", "", {"CAYU_WORKER_PROCESSES": "1"}),
        ("cayu worker shepherd --reconcile-abandoned-after-seconds 60", "", {}),
        ("cayu worker shepherd", "", {"CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S": "60"}),
        ("cayu worker shepherd --until-idle", "", {}),
    ],
)
def test_a_command_that_starts_one_process_passes(
    tmp_path: Path, command: str, extra: str, environment: dict[str, str]
) -> None:
    assert _codes(_project(tmp_path, extra=extra), command, environment) == []


@pytest.mark.parametrize(
    "command,extra,environment,started,fix",
    [
        (
            "cayu worker shepherd --processes 3",
            "",
            {},
            "3 processes (from --processes)",
            "Set --processes 1 in the command (`cayu worker shepherd --processes 1`).",
        ),
        (
            "cayu worker shepherd",
            "",
            {"CAYU_WORKER_PROCESSES": "4"},
            "4 processes (from CAYU_WORKER_PROCESSES)",
            "or set CAYU_WORKER_PROCESSES to 1.",
        ),
        (
            "cayu worker shepherd",
            '\n[tool.cayu.worker_processes]\nshepherd = "auto"\n',
            {},
            "one process per CPU",
            "or set shepherd = 1 in [tool.cayu.worker_processes] in pyproject.toml.",
        ),
        (
            "cayu worker shepherd",
            "\n[tool.cayu.worker_processes]\nshepherd = 2\n",
            {},
            "2 processes (from [tool.cayu.worker_processes])",
            "or set shepherd = 1 in [tool.cayu.worker_processes] in pyproject.toml.",
        ),
    ],
)
def test_the_fix_named_is_the_one_that_takes_effect(
    tmp_path: Path,
    command: str,
    extra: str,
    environment: dict[str, str],
    started: str,
    fix: str,
) -> None:
    (finding,) = worker_command_findings(_project(tmp_path, extra=extra), command, environment)
    assert finding.code == "WORKER_SINGLE_PROCESS_REQUIRED"
    assert started in finding.message
    assert finding.hint is not None and finding.hint.endswith(fix)


@pytest.mark.parametrize(
    "source,expected",
    [
        # A keyword passed by a helper in the same module, which the entrypoint calls.
        (
            "from cayu.tasks import run_task_worker\n"
            "def _serve(app, stop):\n"
            "    return run_task_worker(app, app.task_store, None, stop=stop,\n"
            "                           reconcile_abandoned_after_s=30)\n"
            "async def run_shepherd(app, stop):\n"
            "    return await _serve(app, stop)\n",
            ["WORKER_SINGLE_PROCESS_REQUIRED"],
        ),
        # A module constant, and arithmetic on one.
        (
            "from cayu.tasks import run_task_worker\n"
            "_AFTER_MINUTES = 2\n"
            "_AFTER_SECONDS: float = _AFTER_MINUTES * 60\n"
            "async def run_shepherd(app, stop):\n"
            "    options = {'reconcile_abandoned_after_s': _AFTER_SECONDS, 'stop': stop}\n"
            "    return await run_task_worker(app, app.task_store, None, **options)\n",
            ["WORKER_SINGLE_PROCESS_REQUIRED"],
        ),
        (
            "from cayu.tasks import run_task_worker\n"
            "async def run_shepherd(app, stop):\n"
            "    return await run_task_worker(app, app.task_store, None, stop=stop,\n"
            "                                 reconcile_abandoned_after_s=None)\n",
            [],
        ),
        # A value only known at run time may be None, which turns recovery off.
        (
            "from cayu.tasks import run_task_worker\n"
            "def _serve(app, stop, after=None):\n"
            "    return run_task_worker(app, app.task_store, None, stop=stop,\n"
            "                           reconcile_abandoned_after_s=after)\n"
            "async def run_shepherd(app, stop):\n"
            "    return await _serve(app, stop)\n",
            [],
        ),
        (
            "from cayu.tasks import run_task_worker\n"
            "_AFTER = None\n"
            "async def run_shepherd(app, stop):\n"
            "    options = {'reconcile_abandoned_after_s': _AFTER, 'stop': stop}\n"
            "    return await run_task_worker(app, app.task_store, None, **options)\n",
            [],
        ),
        (
            "from cayu.tasks import run_task_worker\n"
            "async def run_shepherd(app, stop):\n"
            "    return await run_task_worker(app, app.task_store, None, stop=stop)\n",
            [],
        ),
        # Another worker in the same module reconciles; this one doesn't.
        (
            "from cayu.tasks import run_task_worker\n"
            "async def run_recovery(app, stop):\n"
            "    return await run_task_worker(app, app.task_store, None, stop=stop,\n"
            "                                 reconcile_abandoned_after_s=30)\n"
            "async def run_shepherd(app, stop):\n"
            "    return await run_task_worker(app, app.task_store, None, stop=stop)\n",
            [],
        ),
    ],
)
def test_only_what_the_entrypoint_runs_counts(
    tmp_path: Path, source: str, expected: list[str]
) -> None:
    assert _codes(_project(tmp_path, source=source), "cayu worker shepherd") == expected


@pytest.mark.parametrize(
    "command,extra",
    [
        ("cayu worker missing", ""),
        ("cayu worker shepherd --unknown-option", ""),
        ("cayu worker shepherd --processes 2 --reconcile-abandoned-after-seconds 60", ""),
        ("cayu worker shepherd", "\n[tool.cayu.worker_processes]\nshepherd = 0\n"),
        ("cayu worker shepherd --processes 1 --restart-abandoned", ""),
        ("cayu worker shepherd --processes 1 --max-abandoned-restarts 3", ""),
    ],
)
def test_a_command_cayu_worker_refuses_is_reported(
    tmp_path: Path, command: str, extra: str
) -> None:
    (finding,) = worker_command_findings(_project(tmp_path, extra=extra), command, {})
    assert finding.code == "WORKER_COMMAND_INVALID"
    assert finding.message.startswith(f"`{command}` would refuse to start: ")


def test_other_commands_are_not_checked(tmp_path: Path) -> None:
    project = _project(tmp_path)
    assert _codes(project, "python -m shepherd run") == []
    assert _codes(project, "cayu serve --host 0.0.0.0") == []


def _write_manifest(project: Path, worker_command: str) -> None:
    (project / "cayu-cloud.toml").write_text(
        "\n".join(
            (
                "schema_version = 2",
                'application = "worker-check"',
                'name = "Worker Check"',
                'version = "0.1.0"',
                'entrypoint = "cayu worker shepherd"',
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
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("pinned", [False, True])
def test_check_deploy_fails_on_the_cloud_worker_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pinned: bool,
) -> None:
    project = _project(tmp_path / "agent")
    _write_manifest(project, "cayu worker shepherd" + (" --processes 1" if pinned else ""))
    monkeypatch.chdir(project)

    code = main(["check", "--deploy", "--fail-on", "warning", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert "error" not in report
    codes = [item["code"] for item in report["diagnostics"]]
    if pinned:
        assert "WORKER_SINGLE_PROCESS_REQUIRED" not in codes
        return
    assert code == 1
    assert "WORKER_SINGLE_PROCESS_REQUIRED" in codes
    # A check outside --deploy doesn't read the Cloud manifest.
    main(["check", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert all(item["code"] != "WORKER_SINGLE_PROCESS_REQUIRED" for item in report["diagnostics"])
