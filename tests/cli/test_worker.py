from __future__ import annotations

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
    deadline = time.monotonic() + 5
    while not marker.is_file() and time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.02)
    assert marker.is_file()


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

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    TaskStoreDispatcher,
    run_task_worker,
)


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


def build_app():
    return CayuApp(enable_logging=False)


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
