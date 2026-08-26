"""Synchronous bounded subprocess ownership for CLI construction probes."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


class BoundedCommandStartError(RuntimeError):
    """The owned command could not be started."""


class BoundedCommandTimeoutError(RuntimeError):
    """The owned command did not settle before its deadline."""


class BoundedCommandOutputOverflowError(RuntimeError):
    """The owned command exceeded an authoritative output bound."""


class BoundedCommandReadError(RuntimeError):
    """The owned command output could not be read authoritatively."""


@dataclass(frozen=True, slots=True)
class BoundedCommandResult:
    """Content-bounded result from one settled command."""

    returncode: int
    output: bytes
    output_truncated: bool


def run_bounded_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    timeout_s: float,
    output_limit_bytes: int,
    capture_output: bool = True,
    reject_output_overflow: bool = False,
) -> BoundedCommandResult:
    """Run one command with bounded capture and platform-owned cleanup.

    POSIX commands run in a new session so the complete process group can be
    stopped and reaped. Windows uses a new process group and bounded best-effort
    cleanup of that group and the direct child; callers must not interpret this
    helper as a general Windows process-tree sandbox.
    """

    captured = bytearray()
    overflow = threading.Event()
    read_failed = threading.Event()
    stop_reader = threading.Event()
    closing_output = threading.Event()
    reader: threading.Thread | None = None
    timed_out = False
    rejected_overflow = False
    foreground_finished = False
    capture_unsettled = False
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if capture_output else subprocess.DEVNULL,
            start_new_session=os.name == "posix",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
            ),
            bufsize=0,
            text=False,
        )
    except OSError:
        raise BoundedCommandStartError from None

    try:
        if capture_output:
            if process.stdout is None:  # pragma: no cover - defensive subprocess invariant
                raise BoundedCommandStartError
            if os.name == "posix":
                try:
                    os.set_blocking(process.stdout.fileno(), False)
                except OSError:
                    raise BoundedCommandReadError from None

            def drain_output() -> None:
                assert process.stdout is not None
                try:
                    with process.stdout:
                        while not stop_reader.is_set():
                            chunk = process.stdout.read(16 * 1024)
                            if chunk is None:
                                stop_reader.wait(0.01)
                                continue
                            if not chunk:
                                return
                            remaining = output_limit_bytes + 1 - len(captured)
                            if remaining > 0:
                                captured.extend(chunk[:remaining])
                            if len(captured) > output_limit_bytes:
                                overflow.set()
                except (OSError, ValueError):
                    if not closing_output.is_set():
                        read_failed.set()

            try:
                reader = threading.Thread(
                    target=drain_output,
                    name="cayu-bounded-command-output",
                    daemon=True,
                )
                reader.start()
            except RuntimeError:
                raise BoundedCommandStartError from None

        deadline = time.monotonic() + timeout_s
        while process.poll() is None:
            if read_failed.is_set():
                break
            if reject_output_overflow and overflow.is_set():
                rejected_overflow = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=min(remaining, 0.05))
            except subprocess.TimeoutExpired:
                continue
        foreground_finished = True
    finally:
        # Cleanup begins immediately after Popen succeeds, so setup failures and
        # supervisory exits cannot leave the direct child unowned. On POSIX the
        # new session also makes descendant cleanup authoritative.
        _terminate_owned_process_group(process)
        if reader is not None and reader.ident is not None:
            if not foreground_finished or timed_out or rejected_overflow or read_failed.is_set():
                stop_reader.set()
            reader.join(timeout=1.0)
            if reader.is_alive():
                capture_unsettled = True
                stop_reader.set()
                reader.join(timeout=1.0)
            if reader.is_alive():
                # Popen uses an unbuffered stream, so closing it cannot wait on a
                # BufferedReader lock held by the collector.
                assert process.stdout is not None
                closing_output.set()
                with suppress(OSError):
                    process.stdout.close()
                reader.join(timeout=1.0)

    if reject_output_overflow and overflow.is_set():
        rejected_overflow = True

    if read_failed.is_set():
        raise BoundedCommandReadError
    if timed_out:
        raise BoundedCommandTimeoutError
    if rejected_overflow:
        raise BoundedCommandOutputOverflowError
    if capture_unsettled or (reader is not None and reader.is_alive()):
        raise BoundedCommandTimeoutError
    if process.returncode is None:  # pragma: no cover - termination helper settles it
        raise BoundedCommandTimeoutError
    return BoundedCommandResult(
        returncode=process.returncode,
        output=bytes(captured[:output_limit_bytes]),
        output_truncated=overflow.is_set(),
    )


def _terminate_owned_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop the owned POSIX group or perform bounded platform cleanup."""

    if os.name == "posix":
        with suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
    elif os.name == "nt":
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is not None:
            with suppress(OSError, ValueError):
                os.kill(process.pid, ctrl_break)
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
    elif process.poll() is None:  # pragma: no cover - supported platforms use branches above
        with suppress(OSError):
            process.kill()
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1.0)
    if process.poll() is None:
        with suppress(OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1.0)
