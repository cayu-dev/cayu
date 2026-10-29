from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from cayu.cli import main


@contextmanager
def _running_worker(
    *,
    cwd: Path,
    name: str,
    shutdown_grace_seconds: str,
) -> Iterator[subprocess.Popen[str]]:
    environment = os.environ.copy()
    source_root = Path(__file__).parents[2] / "src"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if existing_pythonpath is None
        else f"{source_root}{os.pathsep}{existing_pythonpath}"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cayu",
            "worker",
            name,
            "--shutdown-grace-seconds",
            shutdown_grace_seconds,
        ],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def _wait_for_worker_start(process: subprocess.Popen[str], marker: Path) -> None:
    deadline = time.monotonic() + 20
    while not marker.is_file() and time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.02)
    assert marker.is_file()


def test_worker_closes_its_app_after_the_target_finishes(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "closing_worker:build_app"

[tool.cayu.workers]
once = "closing_worker:run_once"
""",
        encoding="utf-8",
    )
    (project / "closing_worker.py").write_text(
        """from pathlib import Path

from cayu import CayuApp


class ClosedMarker:
    async def close(self):
        Path("closed.txt").write_text("closed\\n", encoding="utf-8")


def build_app():
    return CayuApp(enable_logging=False, owned_resources=(ClosedMarker(),))


async def run_once(app, stop):
    assert app.lifecycle_state == "open"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    sys.modules.pop("closing_worker", None)

    assert main(["worker", "once"]) == 0

    assert (project / "closed.txt").read_text(encoding="utf-8") == "closed\n"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0"])
def test_worker_requires_a_finite_positive_shutdown_grace(value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["worker", "configured", "--shutdown-grace-seconds", value])

    assert excinfo.value.code == 2


def test_worker_discovers_named_target_and_builds_app_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    nested = project / "operations" / "workers"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "worker_project:build_app"

[tool.cayu.workers]
once = "worker_project:run_once"
""",
        encoding="utf-8",
    )
    (project / "worker_project.py").write_text(
        """from pathlib import Path

from cayu import CayuApp


def build_app():
    with Path("factory-log.txt").open("a", encoding="utf-8") as log:
        log.write(f"{Path.cwd()}\\n")
    return CayuApp(enable_logging=False)


async def run_once(app, stop):
    assert isinstance(app, CayuApp)
    assert not stop.is_set()
    Path("worker-log.txt").write_text("ran\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(nested)
    sys.modules.pop("worker_project", None)

    assert main(["worker", "once"]) == 0

    assert (project / "factory-log.txt").read_text(encoding="utf-8") == f"{project}\n"
    assert (project / "worker-log.txt").read_text(encoding="utf-8") == "ran\n"
    assert "worker_project" not in sys.modules


def test_worker_rejects_targets_outside_the_exact_async_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "invalid_worker:build_app"

[tool.cayu.workers]
invalid = "invalid_worker:extra_argument"
""",
        encoding="utf-8",
    )
    (tmp_path / "invalid_worker.py").write_text(
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def extra_argument(app, stop, optional=None):
    return None
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("invalid_worker", None)

    assert main(["worker", "invalid"]) == 1
    assert "must accept exactly (app, stop)" in capsys.readouterr().err


def test_worker_rejects_sync_target_before_app_construction_or_invocation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "sync_worker:build_app"

[tool.cayu.workers]
sync = "sync_worker:run"
""",
        encoding="utf-8",
    )
    (tmp_path / "sync_worker.py").write_text(
        """from pathlib import Path


def build_app():
    Path("factory-ran.txt").write_text("ran\\n", encoding="utf-8")


def run(app, stop):
    Path("worker-ran.txt").write_text("ran\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("sync_worker", None)

    assert main(["worker", "sync"]) == 1
    assert "must be declared with async def" in capsys.readouterr().err
    assert not (tmp_path / "factory-ran.txt").exists()
    assert not (tmp_path / "worker-ran.txt").exists()


def test_worker_reports_project_system_exit_as_startup_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "worker_app:build_app"

[tool.cayu.workers]
exiting = "exiting_worker:run"
""",
        encoding="utf-8",
    )
    (tmp_path / "worker_app.py").write_text(
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
""",
        encoding="utf-8",
    )
    (tmp_path / "exiting_worker.py").write_text(
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("worker_app", None)
    sys.modules.pop("exiting_worker", None)

    assert main(["worker", "exiting"]) == 1
    assert "Worker project startup raised SystemExit with status 9" in capsys.readouterr().err


def test_worker_unknown_name_fails_before_building_the_app(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "unknown_worker:build_app"

[tool.cayu.workers]
known = "unknown_worker:known"
""",
        encoding="utf-8",
    )
    (tmp_path / "unknown_worker.py").write_text(
        """def build_app():
    raise AssertionError("unknown names must fail before app construction")


async def known(app, stop):
    return None
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("unknown_worker", None)

    assert main(["worker", "missing"]) == 1
    error = capsys.readouterr().err
    assert "Unknown worker 'missing'; available workers: known" in error
    assert "unknown names must fail" not in error
    assert "unknown_worker" not in sys.modules


def test_worker_targets_use_real_deterministic_worker_seams(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "real_workers:build_app"

[tool.cayu.workers]
dispatch = "real_workers:dispatch"
fresh = "real_workers:fresh"
""",
        encoding="utf-8",
    )
    (tmp_path / "real_workers.py").write_text(
        """from pathlib import Path

from cayu import CayuApp
from cayu.tasks import InMemoryTaskStore, TaskStoreDispatcher, run_task_worker


def build_app():
    tasks = InMemoryTaskStore()
    with Path("factory-log.txt").open("a", encoding="utf-8") as log:
        log.write("built\\n")
    return CayuApp(
        task_store=tasks,
        dispatcher=TaskStoreDispatcher(tasks),
        enable_logging=False,
    )


async def dispatch(app, stop):
    stop.set()
    await app.dispatcher.run_worker(
        app,
        worker_id="dispatch-test-worker",
        stop=stop,
    )
    Path("dispatch-ran.txt").write_text("ran\\n", encoding="utf-8")


async def fresh(app, stop):
    async def handle(_app, _task, _worker_id):
        raise AssertionError("a stopped empty worker must not invoke its handler")

    stop.set()
    handled = await run_task_worker(
        app,
        app.task_store,
        handle,
        worker_id="fresh-test-worker",
        stop=stop,
    )
    assert handled == 0
    Path("fresh-ran.txt").write_text("ran\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("real_workers", None)

    assert main(["worker", "dispatch"]) == 0
    assert main(["worker", "fresh"]) == 0

    assert (tmp_path / "factory-log.txt").read_text(encoding="utf-8") == "built\nbuilt\n"
    assert (tmp_path / "dispatch-ran.txt").read_text(encoding="utf-8") == "ran\n"
    assert (tmp_path / "fresh-ran.txt").read_text(encoding="utf-8") == "ran\n"


def test_worker_surfaces_missing_task_store_from_configured_entrypoint(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "missing_store_worker:build_app"

[tool.cayu.workers]
fresh = "missing_store_worker:run_fresh"
""",
        encoding="utf-8",
    )
    (tmp_path / "missing_store_worker.py").write_text(
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run_fresh(app, stop):
    if app.task_store is None:
        raise RuntimeError("fresh worker requires app.task_store")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("missing_store_worker", None)

    assert main(["worker", "fresh"]) == 1
    assert "fresh worker requires app.task_store" in capsys.readouterr().err


def test_worker_target_may_request_stop_before_normal_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "self_stopping_worker:build_app"

[tool.cayu.workers]
self-stopping = "self_stopping_worker:run"
""",
        encoding="utf-8",
    )
    (tmp_path / "self_stopping_worker.py").write_text(
        """import asyncio
from pathlib import Path

from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run(app, stop):
    stop.set()
    await asyncio.sleep(0.01)
    Path("worker-ran.txt").write_text("ran\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("self_stopping_worker", None)

    assert main(["worker", "self-stopping"]) == 0
    assert (tmp_path / "worker-ran.txt").read_text(encoding="utf-8") == "ran\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handler contract")
def test_worker_restores_existing_process_signal_handlers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "handler_worker:build_app"

[tool.cayu.workers]
once = "handler_worker:run_once"
""",
        encoding="utf-8",
    )
    (tmp_path / "handler_worker.py").write_text(
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run_once(app, stop):
    return None
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("handler_worker", None)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigint(_signum, _frame) -> None:
        return None

    def handle_sigterm(_signum, _frame) -> None:
        return None

    try:
        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigterm)

        assert main(["worker", "once"]) == 0
        assert signal.getsignal(signal.SIGINT) is handle_sigint
        assert signal.getsignal(signal.SIGTERM) is handle_sigterm
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal exit contract")
@pytest.mark.parametrize(
    ("signum", "expected_exit"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_worker_signal_requests_cooperative_bounded_shutdown(
    tmp_path: Path,
    signum: signal.Signals,
    expected_exit: int,
) -> None:
    project = tmp_path / "project"
    nested = project / "operations"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "signal_worker:build_app"

[tool.cayu.workers]
wait = "signal_worker:wait_for_stop"
""",
        encoding="utf-8",
    )
    (project / "signal_worker.py").write_text(
        """from pathlib import Path

from cayu import CayuApp


class ClosedMarker:
    async def close(self):
        Path("closed.txt").write_text("closed\\n", encoding="utf-8")


def build_app():
    return CayuApp(enable_logging=False, owned_resources=(ClosedMarker(),))


async def wait_for_stop(app, stop):
    Path("started.txt").write_text("started\\n", encoding="utf-8")
    await stop.wait()
    Path("stopped.txt").write_text("stopped\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    with _running_worker(
        cwd=nested,
        name="wait",
        shutdown_grace_seconds="2",
    ) as process:
        _wait_for_worker_start(process, project / "started.txt")
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=5)

        assert process.returncode == expected_exit, stdout + stderr
        assert (project / "stopped.txt").read_text(encoding="utf-8") == "stopped\n"
        # The app was closed with the rest of the grace after the worker stopped.
        assert (project / "closed.txt").read_text(encoding="utf-8") == "closed\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal exit contract")
def test_worker_shutdown_timeout_is_bounded_and_actionable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "stuck_worker:build_app"

[tool.cayu.workers]
stuck = "stuck_worker:ignore_stop"
""",
        encoding="utf-8",
    )
    (project / "stuck_worker.py").write_text(
        """import asyncio
from pathlib import Path

from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def ignore_stop(app, stop):
    Path("started.txt").write_text("started\\n", encoding="utf-8")
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        Path("cancelled.txt").write_text("cancelled\\n", encoding="utf-8")
        await asyncio.Event().wait()
""",
        encoding="utf-8",
    )
    with _running_worker(
        cwd=project,
        name="stuck",
        shutdown_grace_seconds="0.1",
    ) as process:
        _wait_for_worker_start(process, project / "started.txt")
        process.send_signal(signal.SIGTERM)
        shutdown_started = time.monotonic()
        stdout, stderr = process.communicate(timeout=5)
        shutdown_elapsed = time.monotonic() - shutdown_started

        assert process.returncode == 124, stdout + stderr
        assert "did not stop within 0.1 seconds after SIGTERM" in stderr
        assert shutdown_elapsed < 1


def test_worker_passes_the_single_worker_declaration_to_task_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cayu.tasks.abandoned import RECONCILE_ABANDONED_AFTER_ENV

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "declared_worker:build_app"

[tool.cayu.workers]
once = "declared_worker:run_once"
""",
        encoding="utf-8",
    )
    (project / "declared_worker.py").write_text(
        f"""import os
from pathlib import Path

from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run_once(app, stop):
    Path("declared.txt").write_text(os.environ["{RECONCILE_ABANDONED_AFTER_ENV}"], encoding="utf-8")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    # main() writes the declaration into os.environ for the entrypoint. Setting
    # the variable through monkeypatch first records its original state, so
    # teardown removes what main() wrote. Deleting it at the end instead would
    # record "120.0" and restore it, leaking into later tests in this process.
    monkeypatch.setenv(RECONCILE_ABANDONED_AFTER_ENV, "")
    monkeypatch.delenv(RECONCILE_ABANDONED_AFTER_ENV)
    sys.modules.pop("declared_worker", None)

    assert main(["worker", "once", "--reconcile-abandoned-after-seconds", "120"]) == 0
    assert float((project / "declared.txt").read_text(encoding="utf-8")) == 120.0


def test_worker_passes_the_restart_choice_to_task_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cayu.tasks.abandoned import (
        MAX_ABANDONED_RESTARTS_ENV,
        RECONCILE_ABANDONED_AFTER_ENV,
        RESTART_ABANDONED_ENV,
    )

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "restart_worker:build_app"

[tool.cayu.workers]
once = "restart_worker:run_once"
""",
        encoding="utf-8",
    )
    (project / "restart_worker.py").write_text(
        f"""import os
from pathlib import Path

from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run_once(app, stop):
    Path("restart.txt").write_text(
        os.environ["{RESTART_ABANDONED_ENV}"] + " " + os.environ["{MAX_ABANDONED_RESTARTS_ENV}"],
        encoding="utf-8",
    )
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    # Record each variable's absence so teardown removes what main() writes.
    for name in (RECONCILE_ABANDONED_AFTER_ENV, RESTART_ABANDONED_ENV, MAX_ABANDONED_RESTARTS_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    sys.modules.pop("restart_worker", None)

    assert (
        main(
            [
                "worker",
                "once",
                "--reconcile-abandoned-after-seconds",
                "120",
                "--restart-abandoned",
                "--max-abandoned-restarts",
                "3",
            ]
        )
        == 0
    )
    assert (project / "restart.txt").read_text(encoding="utf-8") == "1 3"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--restart-abandoned"], "--restart-abandoned needs --reconcile-abandoned-after-seconds"),
        (
            ["--reconcile-abandoned-after-seconds", "60", "--max-abandoned-restarts", "1"],
            "--max-abandoned-restarts needs --restart-abandoned",
        ),
    ],
)
def test_worker_refuses_restart_options_without_what_they_need(
    tmp_path: Path,
    monkeypatch,
    capsys,
    arguments: list[str],
    message: str,
) -> None:
    from cayu.tasks.abandoned import RECONCILE_ABANDONED_AFTER_ENV

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "app:build_app"

[tool.cayu.workers]
once = "app:run_once"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    monkeypatch.delenv(RECONCILE_ABANDONED_AFTER_ENV, raising=False)

    assert main(["worker", "once", *arguments]) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("value", ["-1", "101", "two"])
def test_worker_requires_a_restart_count_in_range(value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["worker", "configured", "--max-abandoned-restarts", value])

    assert excinfo.value.code == 2


def test_worker_refuses_a_single_worker_declaration_with_several_processes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "app:build_app"

[tool.cayu.workers]
once = "app:run_once"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert (
        main(
            [
                "worker",
                "once",
                "--processes",
                "2",
                "--reconcile-abandoned-after-seconds",
                "60",
            ]
        )
        == 1
    )
    assert "can't be combined with --processes" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "-1", "soon"])
def test_worker_requires_a_finite_non_negative_declaration(value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["worker", "configured", "--reconcile-abandoned-after-seconds", value])

    assert excinfo.value.code == 2


def test_worker_refuses_an_environment_declaration_with_several_processes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from cayu.tasks.abandoned import RECONCILE_ABANDONED_AFTER_ENV

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "app:build_app"

[tool.cayu.workers]
once = "app:run_once"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    monkeypatch.setenv(RECONCILE_ABANDONED_AFTER_ENV, "0")

    assert main(["worker", "once", "--processes", "2"]) == 1
    assert "can't be combined with --processes" in capsys.readouterr().err


_UNTIL_IDLE_PROJECT = """from datetime import UTC, datetime
from pathlib import Path

from cayu import CayuApp
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskQuery,
    TaskSchedulePolicy,
    complete_managed_task,
    run_task_worker_until_idle,
    worker_until_idle_requested,
)

LATER = datetime(2099, 1, 1, tzinfo=UTC)


class CloseLogger:
    async def close(self):
        print("app closed", flush=True)


def build_app():
    return CayuApp(
        task_store=InMemoryTaskStore(),
        enable_logging=False,
        owned_resources=(CloseLogger(),),
    )


async def handle(app, task, worker_id):
    with Path("handled.txt").open("a", encoding="utf-8") as log:
        log.write(f"{task.id}\\n")
    await complete_managed_task(app.task_store, task, worker_id, {"ok": True})


async def run_jobs(app, stop):
    for task_id in ("due-1", "due-2"):
        await app.create_task(TaskCreate(task_id=task_id, type="job"))
    await app.create_task(
        TaskCreate(
            task_id="later",
            type="job",
            available_at=LATER,
            schedule_policy=TaskSchedulePolicy(),
        )
    )
    if worker_until_idle_requested():
        return await run_task_worker_until_idle(
            app,
            app.task_store,
            handle,
            worker_id="jobs",
            query=TaskQuery(type="job"),
            poll_interval_s=60,
            stop=stop,
        )
    Path("ran-without-flag.txt").write_text("ran\\n", encoding="utf-8")
"""


def _until_idle_project(tmp_path: Path, monkeypatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "until_idle_jobs:build_app"

[tool.cayu.workers]
jobs = "until_idle_jobs:run_jobs"
""",
        encoding="utf-8",
    )
    (project / "until_idle_jobs.py").write_text(_UNTIL_IDLE_PROJECT, encoding="utf-8")
    monkeypatch.chdir(project)
    sys.modules.pop("until_idle_jobs", None)
    return project


def test_worker_until_idle_runs_due_tasks_and_prints_the_next_wake(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = _until_idle_project(tmp_path, monkeypatch)

    assert main(["worker", "jobs", "--until-idle"]) == 0

    output = capsys.readouterr().out.splitlines()
    # The result is the last line, printed once the app has closed.
    assert output[0] == "app closed"
    assert [json.loads(line) for line in output[1:]] == [
        {
            "handled": 2,
            "idle": True,
            "next_available_at": "2099-01-01T00:00:00Z",
            "next_expiry_at": None,
            "maintenance_required": False,
            "continuations_pending": False,
        }
    ]
    assert (project / "handled.txt").read_text(encoding="utf-8").split() == ["due-1", "due-2"]


def test_worker_without_until_idle_does_not_request_it(tmp_path: Path, monkeypatch, capsys) -> None:
    project = _until_idle_project(tmp_path, monkeypatch)

    assert main(["worker", "jobs"]) == 0

    assert capsys.readouterr().out == "app closed\n"
    assert (project / "ran-without-flag.txt").is_file()


def test_worker_until_idle_requires_the_entrypoint_to_return_its_result(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "ignoring_worker:build_app"

[tool.cayu.workers]
once = "ignoring_worker:run_once"
""",
        encoding="utf-8",
    )
    (tmp_path / "ignoring_worker.py").write_text(
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run_once(app, stop):
    return None
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("ignoring_worker", None)

    assert main(["worker", "once", "--until-idle"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "doesn't support --until-idle" in captured.err


def test_worker_refuses_until_idle_with_several_processes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from cayu.tasks.abandoned import RECONCILE_ABANDONED_AFTER_ENV

    _until_idle_project(tmp_path, monkeypatch)
    monkeypatch.delenv(RECONCILE_ABANDONED_AFTER_ENV, raising=False)

    assert main(["worker", "jobs", "--until-idle", "--processes", "2"]) == 1
    assert "--until-idle can't be combined with --processes" in capsys.readouterr().err
