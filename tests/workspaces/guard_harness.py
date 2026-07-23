from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from cayu.runners import ExecCommand, ExecResult
from cayu.workspaces._guest_guard import GUEST_PYTHON

DIRECTORY_OPEN_BARRIER_MARKER = "        # CAYU_TEST_BARRIER_AFTER_DIRECTORY_OPEN"
WRITE_TARGET_PREOPEN_BARRIER_MARKER = "    # CAYU_TEST_BARRIER_BEFORE_WRITE_TARGET_OPEN"
WRITE_TARGET_OPEN_BARRIER_MARKER = "        # CAYU_TEST_BARRIER_AFTER_WRITE_TARGET_OPEN"
WRITE_TRUNCATE_MARKER = "        # CAYU_TEST_AFTER_WRITE_TRUNCATE"
BOUNDED_TAR_READ_MARKER = "# CAYU_TEST_BOUNDED_TAR_MEMBER_READS"
DIRECTORY_IDENTITY_ALIAS_MARKER = "    # CAYU_TEST_DIRECTORY_IDENTITY_ALIAS"


def instrument_directory_open_barrier(
    program: str,
    *,
    ready: Path,
    release: Path,
    component: str = "pivot",
) -> str:
    """Pause the real guest program after it pins one named directory descriptor."""

    assert program.count(DIRECTORY_OPEN_BARRIER_MARKER) == 1
    injected = f"""        if name == {component!r}:
            import time
            with open({str(ready)!r}, "w", encoding="utf-8") as barrier_file:
                barrier_file.write("ready")
            deadline = time.monotonic() + 10
            while not os.path.exists({str(release)!r}):
                if time.monotonic() >= deadline:
                    raise TimeoutError("test barrier timed out")
                time.sleep(0.001)"""
    return program.replace(DIRECTORY_OPEN_BARRIER_MARKER, injected)


def instrument_write_target_open_barrier(
    program: str,
    *,
    ready: Path,
    release: Path,
    target: str = "target.txt",
) -> str:
    """Pause after the final write target has been safely opened."""

    assert program.count(WRITE_TARGET_OPEN_BARRIER_MARKER) == 1
    injected = f"""        if name == {target!r}:
            import time
            with open({str(ready)!r}, "w", encoding="utf-8") as barrier_file:
                barrier_file.write("ready")
            deadline = time.monotonic() + 10
            while not os.path.exists({str(release)!r}):
                if time.monotonic() >= deadline:
                    raise TimeoutError("test barrier timed out")
                time.sleep(0.001)"""
    return program.replace(WRITE_TARGET_OPEN_BARRIER_MARKER, injected)


def instrument_write_target_preopen_barrier(
    program: str,
    *,
    ready: Path,
    release: Path,
    target: str = "target.txt",
) -> str:
    """Pause immediately before opening the final write target."""

    assert program.count(WRITE_TARGET_PREOPEN_BARRIER_MARKER) == 1
    injected = f"""    if name == {target!r}:
        import time
        with open({str(ready)!r}, "w", encoding="utf-8") as barrier_file:
            barrier_file.write("ready")
        deadline = time.monotonic() + 10
        while not os.path.exists({str(release)!r}):
            if time.monotonic() >= deadline:
                raise TimeoutError("test barrier timed out")
            time.sleep(0.001)"""
    return program.replace(WRITE_TARGET_PREOPEN_BARRIER_MARKER, injected)


def instrument_write_truncate_barrier(
    program: str,
    *,
    ready: Path,
    release: Path,
    target: str = "target.txt",
) -> str:
    """Pause after truncating one guarded write target."""

    assert program.count(WRITE_TRUNCATE_MARKER) == 1
    injected = f"""        if name == {target!r}:
            import time
            with open({str(ready)!r}, "w", encoding="utf-8") as barrier_file:
                barrier_file.write("ready")
            deadline = time.monotonic() + 10
            while not os.path.exists({str(release)!r}):
                if time.monotonic() >= deadline:
                    raise TimeoutError("test barrier timed out")
                time.sleep(0.001)"""
    return program.replace(WRITE_TRUNCATE_MARKER, injected)


def require_bounded_tar_member_reads(program: str) -> str:
    """Reject any guest-side tar member read that requests the complete payload."""

    assert program.count(BOUNDED_TAR_READ_MARKER) == 1
    injected = """_original_tar_member_read = tarfile.ExFileObject.read


def _bounded_tar_member_read(self, size=-1):
    if size is None or size < 0:
        raise AssertionError("tar member reads must be bounded")
    return _original_tar_member_read(self, size)


tarfile.ExFileObject.read = _bounded_tar_member_read"""
    return program.replace(BOUNDED_TAR_READ_MARKER, injected)


def instrument_directory_identity_alias(
    program: str,
    *,
    paths: tuple[str, ...],
) -> str:
    """Make distinct test paths report one directory identity, like bind aliases."""

    assert program.count(DIRECTORY_IDENTITY_ALIAS_MARKER) == 1
    injected = f"""    if prefix in {paths!r}:
        identity = (\"cayu-test-alias\", \"shared\")"""
    return program.replace(DIRECTORY_IDENTITY_ALIAS_MARKER, injected)


def run_guard_locally(
    command: ExecCommand,
    stdin: str | None,
    *,
    umask: int = -1,
) -> ExecResult:
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
        umask=umask,
    )
    return ExecResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
    )


def make_local_guard_exec(*, umask: int = -1) -> Any:
    """Return an async fake ``runner.exec`` that runs the guard locally."""

    async def fake_exec(
        command: ExecCommand,
        *,
        stdin: str | None = None,
        **kwargs: Any,
    ) -> ExecResult:
        fake_exec.calls.append(command)  # type: ignore[attr-defined]
        return run_guard_locally(command, stdin, umask=umask)

    fake_exec.calls = []  # type: ignore[attr-defined]
    return fake_exec
