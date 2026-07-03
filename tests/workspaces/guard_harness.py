from __future__ import annotations

import subprocess
import sys
from typing import Any

from cayu.runners import ExecCommand, ExecResult
from cayu.workspaces._guest_guard import GUEST_PYTHON


def run_guard_locally(command: ExecCommand, stdin: str | None) -> ExecResult:
    """Execute a guest-guard ExecCommand on the local host for testing.

    The guard program is exactly what a runner would ship into the guest; the
    tests run it against a tmp_path-rooted workspace with the local Python.
    """

    argv = list(command.argv or [])
    assert argv and argv[0] == GUEST_PYTHON
    argv[0] = sys.executable
    completed = subprocess.run(
        argv,
        input=stdin or "",
        capture_output=True,
        text=True,
        timeout=30,
    )
    return ExecResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
    )


def make_local_guard_exec() -> Any:
    """Return an async fake ``runner.exec`` that runs the guard locally."""

    async def fake_exec(
        command: ExecCommand,
        *,
        stdin: str | None = None,
        **kwargs: Any,
    ) -> ExecResult:
        fake_exec.calls.append(command)  # type: ignore[attr-defined]
        return run_guard_locally(command, stdin)

    fake_exec.calls = []  # type: ignore[attr-defined]
    return fake_exec
