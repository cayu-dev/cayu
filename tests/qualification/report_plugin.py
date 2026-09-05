"""Only allowlisted scalar evidence crosses the qualification report boundary."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

import pytest

from tests.qualification.postgres_cleanup import DATABASE_ENV
from tests.qualification.process_cleanup import (
    drain_process_groups,
    process_group_exists,
    register_process_group,
)

COUNTERS = frozenset(
    {
        "sessions_completed",
        "tasks_completed",
        "tool_calls",
        "event_count_min",
        "event_count_max",
        "active_tools",
        "active_claims",
        "active_fences",
        "retained_cleanups",
        "control_latency_ms",
        "empty_polls",
        "unfinished_tasks",
        "unfinished_sessions",
        "event_sequence_min",
        "event_sequence_max",
    }
)
_records = {}
_children = []
_original_popen = subprocess.Popen


class _TrackedProcess(_original_popen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qualification_own_group = (
            kwargs.get("start_new_session", False) or kwargs.get("process_group") == 0
        )
        if self._qualification_own_group:
            try:
                register_process_group(self.pid)
            except BaseException:
                with suppress(ProcessLookupError):
                    os.killpg(self.pid, signal.SIGKILL)
                self.wait(timeout=5)
                drain_process_groups({self.pid})
                raise
        _children.append(self)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def pytest_sessionstart(session):
    import cayu
    from cayu.build_provenance import current_runtime_build_provenance

    package = Path(cayu.__file__).resolve()
    expected = Path(os.environ["CAYU_QUALIFICATION_PACKAGE"]).resolve()
    if package != expected:
        pytest.exit("qualification imported a different Runtime build", returncode=4)
    provenance = current_runtime_build_provenance()
    _records["build"] = {
        "availability": provenance.availability.value,
        "origin": provenance.origin.value,
        "fingerprint": provenance.fingerprint,
    }
    _records["cases"] = {}
    _children.clear()
    subprocess.Popen = _TrackedProcess
    signal.signal(signal.SIGTERM, _interrupt)
    if os.environ.get("CAYU_QUALIFICATION_POSTGRES") == "1":
        from psycopg.conninfo import make_conninfo

        database = os.environ.get(DATABASE_ENV)
        if database is None or not database.startswith("cayu_qualification_"):
            pytest.exit("PostgreSQL qualification requires runner-owned provisioning", returncode=4)
        os.environ["CAYU_TEST_POSTGRES_DSN"] = make_conninfo(
            os.environ["CAYU_TEST_POSTGRES_DSN"],
            dbname=database,
        )


def pytest_runtest_logreport(report):
    identity = hashlib.sha256(report.nodeid.encode()).hexdigest()
    case = _records["cases"].setdefault(identity, {"phases": {}, "counters": {}})
    case["phases"][report.when] = "failed" if getattr(report, "wasxfail", None) else report.outcome
    for key, value in report.user_properties:
        if key in COUNTERS and type(value) is int and 0 <= value <= 1_000_000_000:
            case["counters"][key] = value


def pytest_sessionfinish(session, exitstatus):
    subprocess.Popen = _original_popen
    # poll() reaps leaders, but an owned group can outlive its leader.
    retained = [
        process
        for process in _children
        if process.poll() is None
        or (process._qualification_own_group and process_group_exists(process.pid))
    ]
    groups = {process.pid for process in _children if process._qualification_own_group}
    for process in retained:
        with suppress(ProcessLookupError):
            if process._qualification_own_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
    groups_remaining = drain_process_groups(groups)
    processes_remaining = sum(process.poll() is None for process in _children)
    _records["resources"] = {
        "subprocesses_started": len(_children),
        "subprocesses_retained": len(retained),
        "subprocess_groups_remaining": groups_remaining,
        "subprocesses_remaining": processes_remaining,
    }
    if retained or groups_remaining or processes_remaining:
        session.exitstatus = 1
    _records["exit_code"] = int(session.exitstatus)
    Path(os.environ["CAYU_QUALIFICATION_RESULT"]).write_text(json.dumps(_records))
