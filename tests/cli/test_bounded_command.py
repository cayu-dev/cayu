"""Tests for CLI-owned bounded dependency commands."""

from __future__ import annotations

import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import IO

import pytest

from cayu.cli import _bounded_command
from cayu.cli._bounded_command import (
    BoundedCommandOutputOverflowError,
    BoundedCommandReadError,
    BoundedCommandStartError,
    BoundedCommandTimeoutError,
    run_bounded_command,
)


def test_bounded_command_retains_only_the_configured_output_prefix(tmp_path: Path) -> None:
    result = run_bounded_command(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 1048576)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_s=5,
        output_limit_bytes=128,
    )

    assert result.returncode == 0
    assert result.output == b"x" * 128
    assert result.output_truncated


def test_bounded_command_rejects_output_overflow_before_process_exit(
    tmp_path: Path,
) -> None:
    with pytest.raises(BoundedCommandOutputOverflowError):
        run_bounded_command(
            [
                sys.executable,
                "-c",
                "import sys, time; sys.stdout.buffer.write(b'x' * 1048576); "
                "sys.stdout.flush(); time.sleep(30)",
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_s=5,
            output_limit_bytes=128,
            reject_output_overflow=True,
        )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process-group ownership")
def test_bounded_command_timeout_stops_inherited_output_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-finished"
    descendant = (
        "import pathlib, time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped', encoding='utf-8')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}], "
        "stdout=sys.stdout, stderr=sys.stderr); time.sleep(30)"
    )

    with pytest.raises(BoundedCommandTimeoutError):
        run_bounded_command(
            [sys.executable, "-c", parent],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_s=0.1,
            output_limit_bytes=128,
        )
    time.sleep(0.6)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process-group ownership")
def test_bounded_command_reaps_descendants_after_direct_process_exits(tmp_path: Path) -> None:
    marker = tmp_path / "late-descendant-finished"
    descendant = (
        "import pathlib, time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped', encoding='utf-8')"
    )
    parent = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}], "
        "stdout=sys.stdout, stderr=sys.stderr)"
    )

    result = run_bounded_command(
        [sys.executable, "-c", parent],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_s=5,
        output_limit_bytes=128,
    )
    assert result.returncode == 0
    time.sleep(0.6)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX detached process")
def test_bounded_command_detached_pipe_holder_cannot_defeat_timeout(
    tmp_path: Path,
) -> None:
    child_pid_file = tmp_path / "detached-child-pid"
    descendant = "import time; time.sleep(30)"
    parent = (
        "import pathlib, subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}], "
        "stdout=sys.stdout, stderr=sys.stderr, start_new_session=True); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), "
        "encoding='utf-8')"
    )
    started = time.monotonic()
    try:
        with pytest.raises(BoundedCommandTimeoutError):
            run_bounded_command(
                [sys.executable, "-c", parent],
                cwd=tmp_path,
                env=os.environ.copy(),
                timeout_s=5,
                output_limit_bytes=128,
            )
    finally:
        if child_pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(child_pid_file.read_text(encoding="utf-8")), signal.SIGKILL)

    assert time.monotonic() - started < 3
    assert not any(
        thread.name == "cayu-bounded-command-output" and thread.is_alive()
        for thread in _bounded_command.threading.enumerate()
    )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process-group ownership")
def test_bounded_command_supervisory_exit_still_reaps_owned_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "supervisory-exit-process-finished"
    command = (
        "import pathlib, time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped', encoding='utf-8')"
    )
    real_monotonic = _bounded_command.time.monotonic
    calls = 0

    def interrupt_wait() -> float:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise SystemExit(17)
        return real_monotonic()

    monkeypatch.setattr(_bounded_command.time, "monotonic", interrupt_wait)
    with pytest.raises(SystemExit) as raised:
        run_bounded_command(
            [sys.executable, "-c", command],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_s=5,
            output_limit_bytes=128,
        )
    assert raised.value.code == 17
    time.sleep(0.6)
    assert not marker.exists()


def test_bounded_command_reader_start_failure_reaps_started_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "reader-start-failure-process-finished"
    command = (
        "import pathlib, time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped', encoding='utf-8')"
    )

    def fail_reader_start(_thread: object) -> None:
        raise RuntimeError("thread capacity exhausted")

    monkeypatch.setattr(_bounded_command.threading.Thread, "start", fail_reader_start)
    with pytest.raises(BoundedCommandStartError):
        run_bounded_command(
            [sys.executable, "-c", command],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_s=5,
            output_limit_bytes=128,
        )
    time.sleep(0.6)
    assert not marker.exists()


def test_bounded_command_read_failure_is_not_accepted_as_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "reader-failure-process-finished"
    command = (
        "import pathlib, time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped', encoding='utf-8')"
    )
    real_popen = _bounded_command.subprocess.Popen

    class FailingReader:
        def __init__(self, stream: IO[bytes]) -> None:
            self._stream = stream

        def __enter__(self) -> FailingReader:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def read(self, _size: int) -> bytes:
            raise OSError("private reader diagnostic")

        def fileno(self) -> int:
            return self._stream.fileno()

        def close(self) -> None:
            self._stream.close()

    def popen_with_failing_reader(*args: object, **kwargs: object):
        process = real_popen(*args, **kwargs)  # ty: ignore[no-matching-overload]
        assert process.stdout is not None
        process.stdout = FailingReader(process.stdout)  # ty: ignore[invalid-assignment]
        return process

    monkeypatch.setattr(_bounded_command.subprocess, "Popen", popen_with_failing_reader)
    with pytest.raises(BoundedCommandReadError) as raised:
        run_bounded_command(
            [sys.executable, "-c", command],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_s=5,
            output_limit_bytes=128,
        )
    assert "private reader diagnostic" not in str(raised.value)
    time.sleep(0.6)
    assert not marker.exists()
