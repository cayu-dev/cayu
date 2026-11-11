"""``cayu worker`` ends the runs a lost worker left holding sessions ``running``.

A Cayu Cloud Agent's web process sleeps when nobody uses it, while its worker keeps
running scans. A deploy that replaces the worker mid-scan leaves the sessions it ran
``running``; the web process's scheduled reconciliation doesn't run while it sleeps,
so the worker reconciles them too, in its first process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from tests.runtime.test_orphaned_session_reconciliation import _orphan

from cayu import EventType, SessionStatus
from cayu.cli import main
from cayu.storage import SQLiteSessionStore

_PROJECT = """import asyncio
import os
from pathlib import Path

from tests.runtime.test_orphaned_session_reconciliation import _application

from cayu import SessionStatus
from cayu.storage import SQLiteSessionStore


def build_app():
    return _application(SQLiteSessionStore(os.environ["CAYU_ORPHAN_TEST_STORE"]))


async def run_once(app, stop):
    # The scan this worker runs goes on for a while; the old worker's session is
    # reconciled meanwhile, or not at all.
    for _attempt in range(int(os.environ["CAYU_ORPHAN_TEST_POLLS"])):
        if (await app.session_store.load("scan")).status is SessionStatus.FAILED:
            break
        await asyncio.sleep(0.05)
    Path("done.txt").write_text("done", encoding="utf-8")
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "orphan_worker:build_app"

[tool.cayu.workers]
once = "orphan_worker:run_once"
""",
        encoding="utf-8",
    )
    (root / "orphan_worker.py").write_text(_PROJECT, encoding="utf-8")
    store = tmp_path / "sessions.sqlite"
    _orphan(store, "scan", crash="setup")
    monkeypatch.chdir(root)
    monkeypatch.setenv("CAYU_ORPHAN_TEST_STORE", str(store))
    monkeypatch.setenv("CAYU_ORPHAN_TEST_POLLS", "200")
    monkeypatch.delenv("CAYU_WORKER_INDEX", raising=False)
    sys.modules.pop("orphan_worker", None)
    return root


def _status(project: Path) -> tuple[SessionStatus, list[EventType]]:
    import asyncio

    async def read() -> tuple[SessionStatus, list[EventType]]:
        store = SQLiteSessionStore(project.parent / "sessions.sqlite")
        try:
            session = await store.load("scan")
            return session.status, [event.type for event in await store.load_events("scan")]
        finally:
            await store.close()

    return asyncio.run(read())


def test_a_worker_reconciles_what_a_lost_worker_left_running(project: Path) -> None:
    assert (
        main(
            [
                "worker",
                "once",
                "--processes",
                "1",
                "--orphaned-session-inactive-after-seconds",
                "0",
            ]
        )
        == 0
    )

    status, events = _status(project)
    assert status is SessionStatus.FAILED
    assert EventType.SESSION_FAILED in events
    assert (project / "done.txt").is_file()


def test_a_worker_reconciles_a_fresh_orphan_by_default(project: Path) -> None:
    # The unavailable environment and lost owner qualify without the default hour.
    assert main(["worker", "once", "--processes", "1"]) == 0

    assert _status(project)[0] is SessionStatus.FAILED


@pytest.mark.parametrize(
    ("arguments", "environment"),
    [
        (["--orphaned-session-reconcile-interval-seconds", "off"], {}),
        # Only a worker's first process reconciles.
        ([], {"CAYU_WORKER_INDEX": "1"}),
    ],
)
def test_a_worker_leaves_reconciliation_off_or_to_its_first_process(
    project: Path, monkeypatch, arguments: list[str], environment: dict[str, str]
) -> None:
    monkeypatch.setenv("CAYU_ORPHAN_TEST_POLLS", "10")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert (
        main(
            [
                "worker",
                "once",
                "--processes",
                "1",
                "--orphaned-session-inactive-after-seconds",
                "0",
                *arguments,
            ]
        )
        == 0
    )

    assert _status(project)[0] is SessionStatus.RUNNING


def test_a_worker_run_until_idle_finishes_its_reconciliation_pass(project: Path) -> None:
    # The entrypoint returns at once; the worker still finishes the pass it started.
    (project / "orphan_worker.py").write_text(
        _PROJECT
        + """

async def run_idle(app, stop):
    from cayu.tasks.until_idle import TaskWorkerIdleResult

    return TaskWorkerIdleResult(handled=0, idle=True)
""",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "orphan_worker:build_app"

[tool.cayu.workers]
idle = "orphan_worker:run_idle"
""",
        encoding="utf-8",
    )

    assert (
        main(["worker", "idle", "--until-idle", "--orphaned-session-inactive-after-seconds", "0"])
        == 0
    )

    status, events = _status(project)
    assert status is SessionStatus.FAILED
    assert EventType.SESSION_FAILED in events


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--orphaned-session-reconcile-interval-seconds", "0"),
        ("--orphaned-session-reconcile-interval-seconds", "nan"),
        ("--orphaned-session-inactive-after-seconds", "-1"),
        ("--orphaned-session-inactive-after-seconds", "1.5"),
    ],
)
def test_a_worker_refuses_bad_reconciliation_timing(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["worker", "configured", flag, value])

    assert excinfo.value.code == 2


def _idle(app, stop):
    from cayu.tasks.until_idle import TaskWorkerIdleResult

    return TaskWorkerIdleResult(handled=0, idle=True)


def test_a_signal_during_the_idle_runs_reconciliation_stops_the_worker(monkeypatch, capsys) -> None:
    import asyncio
    import os
    import signal

    import cayu.cli.worker as worker_cli
    from cayu import CayuApp

    cancelled = []

    async def endless(app, stop, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def entrypoint(app, stop):
        # SIGTERM arrives while the run waits for its reconciliation pass.
        asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
        return _idle(app, stop)

    monkeypatch.setattr(worker_cli, "run_orphaned_session_reconciliation", endless)

    outcome = asyncio.run(
        worker_cli._invoke_worker(
            entrypoint,
            CayuApp(enable_logging=False),
            name="idle",
            shutdown_grace_seconds=5,
            until_idle=True,
            orphaned_session_reconcile_interval_seconds=600,
        )
    )

    assert outcome == 128 + signal.SIGTERM
    assert cancelled == [True]
    # A run a signal stopped reports no idle result.
    assert capsys.readouterr().out == ""


def test_a_reconciliation_that_outlasts_the_shutdown_grace_ends_the_process(
    monkeypatch,
) -> None:
    import asyncio

    import cayu.cli.worker as worker_cli
    from cayu import CayuApp

    class _Exited(Exception):
        pass

    def exit_after_timeout(message: str):
        raise _Exited(message)

    async def stubborn(app, stop, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Cleanup that outlasts the grace; the loop's own shutdown ends it.
            await asyncio.sleep(30)

    async def entrypoint(app, stop):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(worker_cli, "run_orphaned_session_reconciliation", stubborn)
    monkeypatch.setattr(worker_cli, "_exit_after_shutdown_timeout", exit_after_timeout)

    with pytest.raises(_Exited, match="did not stop reconciling orphaned sessions within 0.2"):
        asyncio.run(
            worker_cli._invoke_worker(
                entrypoint,
                CayuApp(enable_logging=False),
                name="stubborn",
                shutdown_grace_seconds=0.2,
                orphaned_session_reconcile_interval_seconds=600,
            )
        )
