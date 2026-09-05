"""Supervise fresh, same-host Cayu processes; durable work ownership stays in stores.

This is a POSIX process lifecycle primitive, not a task queue or a sandbox.
Children are never restarted automatically after possibly dispatching effects.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import threading
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


@dataclass(frozen=True)
class ProcessWorkerCommand:
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    log_path: Path | None = None


@dataclass(frozen=True)
class ProcessWorkerOutcome:
    exit_code: int
    child_exit_codes: tuple[int, ...]
    pids: tuple[int, ...]
    forced_shutdown: bool


def positive_process_count(value: str) -> int:
    import argparse

    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("process count must be an integer.") from None
    if not 1 <= count <= 256:
        raise argparse.ArgumentTypeError("process count must be between 1 and 256.")
    return count


def process_worker_environment(index: int, count: int) -> dict[str, str]:
    environment = os.environ.copy()
    # Spawn fresh interpreters, including when the caller came from a source tree.
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(map(str, sys.path)))
    environment["CAYU_WORKER_INDEX"] = str(index)
    environment["CAYU_WORKER_COUNT"] = str(count)
    environment["CAYU_SUPERVISOR_PID"] = str(os.getpid())
    return environment


async def watch_supervisor(request_stop: Callable[[int], None]) -> None:
    """A surviving child cooperatively stops when its original parent disappears."""
    parent = os.environ.get("CAYU_SUPERVISOR_PID")
    if parent is None:
        return
    expected = int(parent)
    while os.getppid() == expected:
        await asyncio.sleep(0.1)
    request_stop(signal.SIGTERM)


@contextlib.contextmanager
def supervisor_watchdog(shutdown_grace_seconds: float):
    """Cover parent loss during synchronous project import and factory startup too."""
    parent = os.environ.get("CAYU_SUPERVISOR_PID")
    if parent is None:
        yield
        return
    expected = int(parent)
    finished = threading.Event()

    def watch() -> None:
        while not finished.wait(0.1):
            if os.getppid() != expected:
                os.kill(os.getpid(), signal.SIGTERM)
                if not finished.wait(shutdown_grace_seconds):
                    os.kill(os.getpid(), signal.SIGKILL)
                return

    thread = threading.Thread(target=watch, name="cayu-supervisor-watch", daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join(timeout=1)


async def supervise_process_workers(
    commands: Sequence[ProcessWorkerCommand],
    *,
    shutdown_grace_seconds: float,
    ready: Callable[[tuple[asyncio.subprocess.Process, ...]], Awaitable[None]] | None = None,
) -> ProcessWorkerOutcome:
    """Run all children, stop siblings on failure, and reap them before returning.

    Call from the CLI's fresh event loop on the main thread. Signals are restored
    when this invocation ends. A ``ready`` callback may implement a pre-dispatch
    admission barrier; it must not dispatch domain work itself.
    """
    if os.name != "posix":
        raise ValueError("Multi-process Cayu execution currently requires POSIX.")
    if not commands or len(commands) > 256:
        raise ValueError("A process group requires between 1 and 256 workers.")
    if not isfinite(shutdown_grace_seconds) or shutdown_grace_seconds <= 0:
        raise ValueError("Process shutdown grace must be finite and positive.")
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    received_signal: int | None = None

    def request_stop(signum: int) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum
        stop.set()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    installed = []
    processes: list[asyncio.subprocess.Process] = []
    logs = []
    waits: list[asyncio.Task[int]] = []
    stop_wait = None
    admission = None
    forced = False
    drain_requested = asyncio.Event()
    group_drains: list[asyncio.Task[None]] = []

    def send(process: asyncio.subprocess.Process, signum: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signum)

    async def drain_group(process: asyncio.subprocess.Process, wait: asyncio.Task[int]) -> None:
        nonlocal forced
        requested = asyncio.create_task(drain_requested.wait())
        try:
            # Settle a group as soon as its leader exits, even while other
            # workers remain active. Do not retain old group IDs until run end.
            await asyncio.wait((wait, requested), return_when=asyncio.FIRST_COMPLETED)
        finally:
            requested.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await requested
        # A reaped leader does not imply that its process group is empty.
        send(process, signal.SIGTERM)
        deadline = loop.time() + shutdown_grace_seconds
        while True:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                forced = True
                send(process, signal.SIGKILL)
                return
            await asyncio.sleep(min(0.02, remaining))

    async def drain() -> None:
        drain_requested.set()
        # Each group has one cleanup task, shared by normal return and finally.
        # Once it settles, never signal its possibly reused numeric ID again.
        await asyncio.shield(asyncio.gather(*group_drains))
        await asyncio.shield(asyncio.gather(*waits))

    try:
        for sig in previous:
            loop.add_signal_handler(sig, request_stop, sig)
            installed.append(sig)
        for command in commands:
            if stop.is_set():
                break
            log = None
            if command.log_path is not None:
                log = command.log_path.open("xb")
                logs.append(log)
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *command.argv,
                    cwd=command.cwd,
                    env=command.environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT if log is not None else None,
                    start_new_session=True,
                )
            )
            cancelled = False
            while not spawn.done():
                try:
                    await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    cancelled = True
            process = spawn.result()
            processes.append(process)
            waits.append(asyncio.create_task(process.wait()))
            group_drains.append(asyncio.create_task(drain_group(process, waits[-1])))
            if cancelled:
                raise asyncio.CancelledError
        stop_wait = asyncio.create_task(stop.wait())
        if ready is not None and not stop.is_set():
            admission = asyncio.ensure_future(ready(tuple(processes)))
            done, _ = await asyncio.wait(
                [admission, stop_wait, *waits], return_when=asyncio.FIRST_COMPLETED
            )
            if stop_wait not in done:
                if admission not in done:
                    raise RuntimeError("A process worker exited before execution admission.")
                await admission
        pending = set(waits)
        failed = None
        while pending and not stop.is_set():
            done, _ = await asyncio.wait([*pending, stop_wait], return_when=asyncio.FIRST_COMPLETED)
            for task in done & pending:
                pending.remove(task)
                code = task.result()
                if code != 0 and failed is None:
                    failed = code if code > 0 else 128 - code
            if failed is not None:
                break
        await drain()
        forced = forced or any(task.result() == 124 for task in waits)
        exit_code = (
            124 if forced else 128 + received_signal if received_signal is not None else failed or 0
        )
        return ProcessWorkerOutcome(
            exit_code=exit_code,
            child_exit_codes=tuple(task.result() for task in waits),
            pids=tuple(process.pid for process in processes),
            forced_shutdown=forced,
        )
    finally:
        for task in (admission, stop_wait):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        # Also settle children after startup/admission failure or caller cancellation.
        cleanup = asyncio.create_task(drain())
        interrupted = False
        try:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    interrupted = True
            cleanup.result()
        finally:
            for log in logs:
                log.close()
            for sig in installed:
                loop.remove_signal_handler(sig)
                signal.signal(sig, previous[sig])
        if interrupted:
            raise asyncio.CancelledError
