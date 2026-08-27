"""Linux subreaper owner for one process-isolated tool worker tree."""

from __future__ import annotations

import argparse
import ctypes
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import suppress
from math import isfinite
from pathlib import Path
from types import FrameType
from typing import Final

_PR_SET_CHILD_SUBREAPER: Final = 36
_POLL_SECONDS: Final = 0.01
_EXIT_SOFTWARE: Final = 70
_PROBE_CHILD_SUBREAPER_ARGUMENT: Final = "--probe-child-subreaper"
_SETTLEMENT_ACK_COMPLETED: Final = b"\x01\x00"
_SETTLEMENT_ACK_SUPERVISOR_FAILED: Final = b"\x01\x01"
_WORKER_ADMISSION: Final = b"\x01"

_shutdown_requested = False


def _request_shutdown(_signum: int, _frame: FrameType | None) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def _enable_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")


def _direct_children() -> tuple[int, ...]:
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    contents = children_path.read_text(encoding="ascii").strip()
    if not contents:
        return ()
    return tuple(int(value) for value in contents.split())


class _ChildReapObservationError(OSError):
    """Carry constant-size worker evidence across a retryable waitpid failure."""

    def __init__(self, worker_status: int | None) -> None:
        self.worker_status = worker_status
        super().__init__("Child-process reaping observation failed.")


def _reap_exited(
    *,
    worker_pid: int | None,
    worker_status: int | None,
) -> int | None:
    """Reap every exited child while retaining only the worker leader status."""

    while True:
        try:
            child_pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return worker_status
        except OSError as error:
            raise _ChildReapObservationError(worker_status) from error
        if child_pid == 0:
            return worker_status
        if worker_status is None and worker_pid is not None and child_pid == worker_pid:
            worker_status = status


def _signal_pid(child_pid: int, selected_signal: signal.Signals) -> None:
    with suppress(ProcessLookupError):
        os.kill(child_pid, selected_signal)


def _signal_group(process_group_id: int, selected_signal: signal.Signals) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process_group_id, selected_signal)


def _settle_owned_children(
    *,
    worker_process_group_id: int | None,
    worker_status: int | None,
    term_grace_seconds: float,
    kill_grace_seconds: float,
) -> int | None:
    phase_deadline = time.monotonic() + term_grace_seconds
    selected_signal = signal.SIGTERM
    while True:
        try:
            worker_status = _reap_exited(
                worker_pid=worker_process_group_id,
                worker_status=worker_status,
            )
            children = _direct_children()
        except _ChildReapObservationError as error:
            worker_status = error.worker_status
            # Losing observation cannot be converted into cleanup proof. Keep
            # this owner alive with any leader status already reaped so the
            # next observation cannot mistake an unreapable PID for live work.
            time.sleep(_POLL_SECONDS)
            continue
        except (OSError, ValueError):
            # Losing observation cannot be converted into cleanup proof. Keep
            # this owner alive so the parent remains fenced and can retry.
            time.sleep(_POLL_SECONDS)
            continue
        if not children:
            return worker_status
        try:
            # Once waitpid() has reaped the worker leader, Linux may reuse its
            # numeric PID as an unrelated process-group ID.  Adopted survivors
            # are direct children of this subreaper and are signalled below, so
            # retaining group authority after that point is both unnecessary
            # and unsafe.
            if worker_process_group_id is not None and worker_status is None:
                _signal_group(worker_process_group_id, selected_signal)
            for child_pid in children:
                _signal_pid(child_pid, selected_signal)
        except OSError:
            time.sleep(_POLL_SECONDS)
            continue
        now = time.monotonic()
        if selected_signal is signal.SIGTERM and now >= phase_deadline:
            selected_signal = signal.SIGKILL
            phase_deadline = now + kill_grace_seconds
        # The configured KILL grace bounds the public owner's first proof
        # attempt.  If an uninterruptible child remains, this supervisor must
        # retain ownership instead of exiting and orphaning it.
        time.sleep(_POLL_SECONDS if now < phase_deadline else min(0.05, _POLL_SECONDS * 5))


def _propagate_worker_status(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        selected_signal = os.WTERMSIG(status)
        if selected_signal not in {signal.SIGKILL, signal.SIGSTOP}:
            signal.signal(selected_signal, signal.SIG_DFL)
        os.kill(os.getpid(), selected_signal)
    return _EXIT_SOFTWARE


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--result-fd", type=int, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--settlement-fd", type=int, required=True)
    parser.add_argument("--worker-module", required=True)
    parser.add_argument("--term-grace-seconds", type=_non_negative_float, required=True)
    parser.add_argument("--kill-grace-seconds", type=_positive_float, required=True)
    return parser.parse_args(argv)


def _settle_before_worker_dispatch(
    arguments: argparse.Namespace,
    *,
    return_code: int,
) -> int:
    """Prove the subreaper has no children before publishing zero-dispatch cleanup."""

    with suppress(OSError):
        os.close(arguments.result_fd)
    with suppress(OSError):
        os.close(arguments.control_fd)
    acknowledgement_written = False
    try:
        _settle_owned_children(
            worker_process_group_id=None,
            worker_status=None,
            term_grace_seconds=arguments.term_grace_seconds,
            kill_grace_seconds=arguments.kill_grace_seconds,
        )
        acknowledgement = (
            _SETTLEMENT_ACK_COMPLETED if return_code == 0 else _SETTLEMENT_ACK_SUPERVISOR_FAILED
        )
        try:
            acknowledgement_written = os.write(
                arguments.settlement_fd,
                acknowledgement,
            ) == len(acknowledgement)
        except OSError:
            acknowledgement_written = False
    finally:
        with suppress(OSError):
            os.close(arguments.settlement_fd)
    return return_code if acknowledgement_written else _EXIT_SOFTWARE


def _await_worker_admission(control_poller: select.poll, control_fd: int) -> bool:
    """Wait for exact parent authority before allowing worker creation."""

    while not _shutdown_requested:
        try:
            events = control_poller.poll(max(1, int(_POLL_SECONDS * 1000)))
        except InterruptedError:
            continue
        if _shutdown_requested:
            return False
        if not events:
            continue
        try:
            admission = os.read(control_fd, len(_WORKER_ADMISSION) + 1)
        except InterruptedError:
            continue
        except OSError:
            return False
        return admission == _WORKER_ADMISSION and not _shutdown_requested
    return False


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "linux":
        return _EXIT_SOFTWARE
    arguments_list = sys.argv[1:] if argv is None else argv
    if arguments_list == [_PROBE_CHILD_SUBREAPER_ARGUMENT]:
        try:
            _enable_child_subreaper()
            _direct_children()
        except (AttributeError, OSError, TypeError, ValueError):
            return _EXIT_SOFTWARE
        return 0
    arguments = _parse_args(arguments_list)
    if (
        arguments.result_fd < 0
        or arguments.control_fd < 0
        or arguments.settlement_fd < 0
        or arguments.result_fd == arguments.control_fd
        or arguments.result_fd == arguments.settlement_fd
        or arguments.control_fd == arguments.settlement_fd
        or not arguments.worker_module
    ):
        return _EXIT_SOFTWARE
    try:
        _enable_child_subreaper()
        os.fstat(arguments.control_fd)
        os.fstat(arguments.settlement_fd)
    except (AttributeError, OSError, TypeError, ValueError):
        return _EXIT_SOFTWARE
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    try:
        control_poller = select.poll()
        control_poller.register(
            arguments.control_fd,
            select.POLLIN | select.POLLHUP | select.POLLERR | select.POLLNVAL,
        )
        worker_admitted = _await_worker_admission(
            control_poller,
            arguments.control_fd,
        )
    except Exception:
        return _settle_before_worker_dispatch(arguments, return_code=_EXIT_SOFTWARE)
    if not worker_admitted:
        # Process creation is not worker admission. The parent grants that
        # authority only after asyncio has returned the exact supervisor handle
        # and installed its wait/settlement owner. EOF, malformed input, or a
        # signal before then must win without creating a worker.
        return _settle_before_worker_dispatch(arguments, return_code=0)

    try:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-m",
                arguments.worker_module,
                "--result-fd",
                str(arguments.result_fd),
            ],
            close_fds=True,
            pass_fds=(arguments.result_fd,),
            process_group=0,
        )
    except Exception:
        # Popen may fail before returning a worker (for example under process
        # exhaustion).  The subreaper must still prove that it owns no adopted
        # children before acknowledging settlement; merely assuming that the
        # failed constructor created nothing would weaken the process-tree
        # contract.
        return _settle_before_worker_dispatch(arguments, return_code=_EXIT_SOFTWARE)
    except BaseException:
        # Process-control signals are not evidence that worker creation failed
        # before dispatch.  Preserve the fail-closed behavior and publish no
        # settlement acknowledgement.
        with suppress(OSError):
            os.close(arguments.result_fd)
        with suppress(OSError):
            os.close(arguments.control_fd)
        with suppress(OSError):
            os.close(arguments.settlement_fd)
        return _EXIT_SOFTWARE
    worker_status: int | None = None
    post_spawn_failed = False
    try:
        os.close(arguments.result_fd)
        while not _shutdown_requested and worker_status is None:
            try:
                if control_poller.poll(0):
                    _request_shutdown(0, None)
                    break
            except InterruptedError:
                time.sleep(_POLL_SECONDS)
                continue
            try:
                worker_status = _reap_exited(
                    worker_pid=worker.pid,
                    worker_status=worker_status,
                )
            except _ChildReapObservationError as error:
                worker_status = error.worker_status
                # Preserve a leader status reaped before the transient failure;
                # waitpid cannot return the same child to a later retry.
                time.sleep(_POLL_SECONDS)
                continue
            except OSError:
                # Do not abandon a live worker merely because observation is
                # temporarily unavailable.
                time.sleep(_POLL_SECONDS)
                continue
            if worker_status is None:
                time.sleep(_POLL_SECONDS)
    except Exception:
        post_spawn_failed = True
    finally:
        with suppress(OSError):
            os.close(arguments.result_fd)
        with suppress(OSError):
            os.close(arguments.control_fd)
        worker_status = _settle_owned_children(
            worker_process_group_id=worker.pid,
            worker_status=worker_status,
            term_grace_seconds=arguments.term_grace_seconds,
            kill_grace_seconds=arguments.kill_grace_seconds,
        )
        if worker_status is None:
            post_spawn_failed = True
        acknowledgement = (
            _SETTLEMENT_ACK_SUPERVISOR_FAILED if post_spawn_failed else _SETTLEMENT_ACK_COMPLETED
        )
        try:
            if os.write(arguments.settlement_fd, acknowledgement) != len(acknowledgement):
                post_spawn_failed = True
        except OSError:
            post_spawn_failed = True
        finally:
            with suppress(OSError):
                os.close(arguments.settlement_fd)
    if post_spawn_failed:
        return _EXIT_SOFTWARE
    if _shutdown_requested:
        return 0
    return _EXIT_SOFTWARE if worker_status is None else _propagate_worker_status(worker_status)


if __name__ == "__main__":  # pragma: no branch - module entry point
    raise SystemExit(main())
