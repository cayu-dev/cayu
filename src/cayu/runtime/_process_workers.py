"""Supervise fresh, same-host Cayu processes; durable work ownership stays in stores.

This is a POSIX process lifecycle primitive, not a task queue or a sandbox.
Children are never restarted automatically after possibly dispatching effects.
``cayu worker --processes`` and ``cayu eval run --processes`` use it, and an
application that runs its own task worker can use it to start that worker in
several processes; see docs/project-workers.md.
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
    """One child process: its argv, working directory and complete environment.

    With ``log_path``, the child's stdout and stderr go to that new file;
    otherwise they are inherited.
    """

    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    log_path: Path | None = None


@dataclass(frozen=True)
class ProcessWorkerOutcome:
    """How a supervised group ended once every child was reaped.

    ``exit_code`` is ``0`` when every child exited ``0``, the first failing
    child's code otherwise, ``128 + signal`` after SIGINT or SIGTERM, and
    ``124`` when a child had to be killed after the shutdown grace.
    """

    exit_code: int
    child_exit_codes: tuple[int, ...]
    pids: tuple[int, ...]
    forced_shutdown: bool


_MAX_PROCESS_COUNT = 256


def positive_process_count(value: str) -> int:
    import argparse

    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("process count must be an integer.") from None
    if not 1 <= count <= _MAX_PROCESS_COUNT:
        raise argparse.ArgumentTypeError("process count must be between 1 and 256.")
    return count


# Where Linux says which cgroups this process is in and where they are mounted.
_PROC_SELF = Path("/proc/self")


def available_cpu_count() -> int:
    """How many CPUs this process may use: at least 1.

    The CPUs it may be scheduled on (``os.process_cpu_count()`` where Python
    has it, else the scheduler affinity or ``os.cpu_count()``), lowered to a
    container's CPU limit (a cgroup CPU quota, rounded down), as on Cayu Cloud
    where it is the process's ``cpu_millis``.
    """

    count = _schedulable_cpu_count()
    limit = _cgroup_cpu_limit()
    if limit is not None:
        count = min(count, limit)
    return max(1, count)


def _schedulable_cpu_count() -> int:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return count or 1


def _cgroup_cpu_limit() -> int | None:
    """The whole CPUs this process's cgroups allow, or ``None`` without a quota.

    Reads the CPU quota of the process's own cgroup and of each ancestor up to
    its mount, cgroup v2 or v1, and keeps the tightest.
    """

    try:
        memberships = (_PROC_SELF / "cgroup").read_text(encoding="utf-8").splitlines()
        mounts = (_PROC_SELF / "mountinfo").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    limits = [
        limit
        for directory, mount_point, version in _cpu_cgroup_directories(memberships, mounts)
        for level in _cgroup_levels(directory, mount_point)
        if (limit := _cgroup_quota_cpus(level, version)) is not None
    ]
    return min(limits, default=None)


def _cpu_cgroup_directories(
    memberships: list[str], mounts: list[str]
) -> list[tuple[Path, Path, str]]:
    """Each (cgroup directory, its mount point, "v2" or "v1") that limits CPU."""

    # mountinfo: ID PARENT MAJOR:MINOR ROOT MOUNT_POINT OPTIONS [OPTIONAL...] - TYPE SOURCE SUPER
    mounted: list[tuple[str, str, Path, set[str]]] = []
    for line in mounts:
        before, separator, after = line.partition(" - ")
        fields, tail = before.split(), after.split()
        if not separator or len(fields) < 5 or not tail:
            continue
        options = set(tail[2].split(",")) if len(tail) > 2 else set()
        mounted.append(
            (
                tail[0],
                _unescape_mount_path(fields[3]),
                Path(_unescape_mount_path(fields[4])),
                options,
            )
        )
    found: list[tuple[Path, Path, str]] = []
    for line in memberships:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, cgroup_path = parts
        if controllers == "":
            version, wanted = "v2", ("cgroup2", None)
        elif "cpu" in controllers.split(","):
            version, wanted = "v1", ("cgroup", "cpu")
        else:
            continue
        for fstype, root, mount_point, options in mounted:
            if fstype != wanted[0] or (wanted[1] is not None and wanted[1] not in options):
                continue
            found.append((_cgroup_directory(cgroup_path, root, mount_point), mount_point, version))
    return found


def _unescape_mount_path(value: str) -> str:
    # mountinfo writes space, tab, newline and backslash as octal escapes.
    for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, plain)
    return value


def _cgroup_directory(cgroup_path: str, root: str, mount_point: Path) -> Path:
    """Where ``cgroup_path`` is under a mount of the hierarchy's ``root``."""

    if root == "/":
        relative = cgroup_path.lstrip("/")
    elif cgroup_path == root or cgroup_path.startswith(root.rstrip("/") + "/"):
        relative = cgroup_path[len(root) :].lstrip("/")
    else:
        # Outside what this mount shows, as across cgroup namespaces.
        return mount_point
    directory = mount_point / relative
    return (
        directory
        if directory.is_relative_to(mount_point) and ".." not in Path(relative).parts
        else mount_point
    )


def _cgroup_levels(directory: Path, mount_point: Path) -> list[Path]:
    """``directory`` and each of its ancestors up to ``mount_point``."""

    levels = [directory]
    while levels[-1] != mount_point and levels[-1].is_relative_to(mount_point):
        levels.append(levels[-1].parent)
    return levels


def _cgroup_quota_cpus(directory: Path, version: str) -> int | None:
    try:
        if version == "v2":
            fields = (directory / "cpu.max").read_text(encoding="ascii").split()
            if not fields or fields[0] == "max":
                return None
            quota, period = fields[0], fields[1] if len(fields) > 1 else "100000"
        else:
            quota = (directory / "cpu.cfs_quota_us").read_text(encoding="ascii").strip()
            period = (directory / "cpu.cfs_period_us").read_text(encoding="ascii").strip()
        quota_us, period_us = int(quota), int(period)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if quota_us <= 0 or period_us <= 0:
        # cgroup v1 states "no limit" as -1.
        return None
    return max(1, quota_us // period_us)


# How many worker processes one supervisor runs, in each of them.
WORKER_COUNT_ENVIRONMENT = "CAYU_WORKER_COUNT"


def process_worker_count() -> int:
    """How many processes this worker's supervisor runs: 1 without a supervisor."""

    value = os.environ.get(WORKER_COUNT_ENVIRONMENT, "").strip()
    if not value.isdecimal() or int(value) < 1:
        return 1
    return int(value)


def require_single_process_worker(operation: str) -> None:
    """Refuse sole-worker operations in an ambiguous or multi-process group.

    Without supervisor metadata, the caller owns the standalone declaration.
    Unlike the reporting helper above, malformed metadata is never proof of a
    single process. A child's local ``--processes 1`` does not erase its peers.
    """

    count = os.environ.get(WORKER_COUNT_ENVIRONMENT)
    index = os.environ.get("CAYU_WORKER_INDEX")
    if count is None and index is None and "CAYU_SUPERVISOR_PID" not in os.environ:
        return
    if count is not None and count.strip() == "1" and (index is None or index == "0"):
        return
    raise ValueError(
        f"{single_process_worker_requirement(operation)} "
        "Supervisor metadata must declare CAYU_WORKER_COUNT=1 and, when present, "
        "CAYU_WORKER_INDEX=0; do not override metadata in a multi-process worker."
    )


def single_process_worker_requirement(operation: str) -> str:
    """What :func:`require_single_process_worker` says ``operation`` needs, and the fixes.

    ``cayu check --deploy`` reports the same rule before a worker starts.
    """

    return (
        f"{operation} requires a single worker process. Run cayu worker with --processes 1 "
        "or set this worker to 1 in [tool.cayu.worker_processes]."
    )


def process_worker_environment(index: int, count: int) -> dict[str, str]:
    """This process's environment for child ``index`` of ``count``.

    Adds ``CAYU_WORKER_INDEX``, ``CAYU_WORKER_COUNT`` and
    ``CAYU_SUPERVISOR_PID``, which :func:`watch_supervisor` and
    :func:`supervisor_watchdog` use to stop the child when this process goes
    away, and passes ``sys.path`` on so the child imports the same code.
    """

    environment = os.environ.copy()
    # Spawn fresh interpreters, including when the caller came from a source tree.
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(map(str, sys.path)))
    environment["CAYU_WORKER_INDEX"] = str(index)
    environment[WORKER_COUNT_ENVIRONMENT] = str(count)
    environment["CAYU_SUPERVISOR_PID"] = str(os.getpid())
    return environment


async def watch_supervisor(request_stop: Callable[[int], None]) -> None:
    """In a child, call ``request_stop(SIGTERM)`` once its supervisor is gone.

    Run it as a task beside the child's work. It returns at once in a process
    that wasn't started with :func:`process_worker_environment`.
    """
    parent = os.environ.get("CAYU_SUPERVISOR_PID")
    if parent is None:
        return
    expected = int(parent)
    while os.getppid() == expected:
        await asyncio.sleep(0.1)
    request_stop(signal.SIGTERM)


@contextlib.contextmanager
def supervisor_watchdog(shutdown_grace_seconds: float):
    """In a child, send itself SIGTERM, then SIGKILL after the grace, if the supervisor goes.

    Unlike :func:`watch_supervisor` it works without a running event loop, so
    it also covers slow synchronous startup such as imports and app
    construction.
    """
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
    stop: asyncio.Event | None = None,
    handle_signals: bool = True,
) -> ProcessWorkerOutcome:
    """Run all children, stop siblings on failure, and reap them before returning.

    Each child starts in its own process group. When one exits non-zero, when
    ``stop`` is set, or on SIGINT or SIGTERM, every group gets SIGTERM and has
    ``shutdown_grace_seconds`` before SIGKILL. Children are never restarted.
    Cancelling the call stops and reaps the children too.

    With ``handle_signals`` (the default) the call installs SIGINT and SIGTERM
    handlers on the running loop, so it must run on the main thread, and
    restores the previous handlers when it ends. An application that handles
    signals itself, such as a web server, passes ``handle_signals=False`` and
    sets ``stop`` instead. A ``ready`` callback may implement a pre-dispatch
    admission barrier; it must not dispatch domain work itself.
    """
    if os.name != "posix":
        raise ValueError("Multi-process Cayu execution currently requires POSIX.")
    if not commands or len(commands) > 256:
        raise ValueError("A process group requires between 1 and 256 workers.")
    if not isfinite(shutdown_grace_seconds) or shutdown_grace_seconds <= 0:
        raise ValueError("Process shutdown grace must be finite and positive.")
    if stop is not None and not isinstance(stop, asyncio.Event):
        raise TypeError("stop must be an asyncio.Event.")
    if type(handle_signals) is not bool:
        raise TypeError("handle_signals must be a bool.")
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    received_signal: int | None = None

    def request_stop(signum: int) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum
        stop_requested.set()

    previous = (
        {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        if handle_signals
        else {}
    )
    installed = []
    processes: list[asyncio.subprocess.Process] = []
    logs = []
    waits: list[asyncio.Task[int]] = []
    stop_wait = None
    caller_stop_wait = None
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
        if stop is not None:
            # The caller's event stops the group the way a signal does.
            caller_stop_wait = asyncio.create_task(stop.wait())
            caller_stop_wait.add_done_callback(
                lambda task: None if task.cancelled() else stop_requested.set()
            )
            if stop.is_set():
                stop_requested.set()
        for command in commands:
            if stop_requested.is_set():
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
        stop_wait = asyncio.create_task(stop_requested.wait())
        if ready is not None and not stop_requested.is_set():
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
        while pending and not stop_requested.is_set():
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
        for task in (admission, stop_wait, caller_stop_wait):
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
