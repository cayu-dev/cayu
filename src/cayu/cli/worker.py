"""Run one named project worker with a process-scoped Cayu application."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import os
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from cayu.applications import CayuApp
from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli._worker_drain import DrainRequest, drain_notice, watch_cloud_drain
from cayu.cli.project import (
    _discover_configured_project,
    build_project_app,
    close_project_app,
    project_context,
)
from cayu.cli.project_control_plane import project_eval_worker
from cayu.runtime._process_workers import (
    ProcessWorkerCommand,
    positive_process_count,
    process_worker_environment,
    supervise_process_workers,
    supervisor_watchdog,
    watch_supervisor,
)
from cayu.tasks.abandoned import (
    MAX_ABANDONED_RESTARTS_ENV,
    RECONCILE_ABANDONED_AFTER_ENV,
    RESTART_ABANDONED_ENV,
)
from cayu.tasks.records import TASK_MAX_RESTARTS_LIMIT
from cayu.tasks.until_idle import TaskWorkerIdleResult, _worker_until_idle_requested

if TYPE_CHECKING:
    from cayu.cli._worker_evals import WorkerEvals
    from cayu.project_control_plane import ProjectControlPlaneContext
    from cayu.server.evals_worker import EvalRunCoordinator

# A worker that used its whole grace still lets the app seal and report.
_MIN_APP_SHUTDOWN_SECONDS = 0.1
# How long a worker that runs until idle lets its eval runs go on before it serves
# the tasks they may have created.
_EVAL_TASK_SERVICE_SECONDS = 5.0


class WorkerError(ValueError):
    """A configured worker cannot be started or stopped safely."""


@dataclass(frozen=True)
class _WorkerProject:
    root: Path
    factory_target: str
    worker_target: str


def add_worker_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "worker",
        help="Run a configured Cayu worker entrypoint.",
        description=(
            "Build the configured Cayu application once and run one named async "
            "worker until it completes or receives a cooperative stop request."
        ),
    )
    parser.add_argument("name", help="Worker name from [tool.cayu.workers].")
    parser.add_argument(
        "--processes",
        type=positive_process_count,
        default=1,
        help="Independent worker processes; each constructs its own app (default: 1).",
    )
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=_positive_seconds,
        default=30.0,
        help="Maximum cooperative shutdown time after SIGINT or SIGTERM (default: 30).",
    )
    parser.add_argument(
        "--reconcile-abandoned-after-seconds",
        type=_non_negative_seconds,
        default=None,
        metavar="SECONDS",
        help=(
            "Declare this the only worker for its tasks: run_task_worker settles a task "
            "fenced after another worker's lease expired at least SECONDS ago as cancelled. "
            "Off by default; can't be combined with --processes."
        ),
    )
    parser.add_argument(
        "--restart-abandoned",
        action="store_true",
        help=(
            "With --reconcile-abandoned-after-seconds, requeue a fenced task created with "
            "restartable=True instead of cancelling it. Other tasks are still cancelled."
        ),
    )
    parser.add_argument(
        "--max-abandoned-restarts",
        type=_restart_count,
        default=None,
        metavar="N",
        help=(
            "With --restart-abandoned, cancel a task once it was restarted N times after "
            "worker loss, unless the task sets its own max_restarts (default: 2)."
        ),
    )
    parser.add_argument(
        "--until-idle",
        action="store_true",
        help=(
            "Run the tasks that are due, then exit and print one JSON line with the "
            "number handled and when work is next due. The entrypoint must return "
            "run_task_worker_until_idle's result when worker_until_idle_requested() "
            "is true. Can't be combined with --processes."
        ),
    )


def run_worker(args: argparse.Namespace) -> int:
    with supervisor_watchdog(args.shutdown_grace_seconds):
        return _run_worker(args)


def _run_worker(args: argparse.Namespace) -> int:
    try:
        project = _resolve_worker_project(args.name)
        processes = getattr(args, "processes", 1)
        reconcile_after = getattr(args, "reconcile_abandoned_after_seconds", None)
        if processes > 1 and (
            reconcile_after is not None or os.environ.get(RECONCILE_ABANDONED_AFTER_ENV, "").strip()
        ):
            raise WorkerError(
                f"--reconcile-abandoned-after-seconds and {RECONCILE_ABANDONED_AFTER_ENV} "
                "declare a single worker and can't be combined with --processes."
            )
        until_idle = getattr(args, "until_idle", False)
        if processes > 1 and until_idle:
            raise WorkerError("--until-idle can't be combined with --processes.")
        restart_abandoned = getattr(args, "restart_abandoned", False)
        max_restarts = getattr(args, "max_abandoned_restarts", None)
        if restart_abandoned and (
            reconcile_after is None
            and not os.environ.get(RECONCILE_ABANDONED_AFTER_ENV, "").strip()
        ):
            raise WorkerError(
                "--restart-abandoned needs --reconcile-abandoned-after-seconds: only a "
                "declared single worker settles fenced tasks itself."
            )
        if max_restarts is not None and not restart_abandoned:
            raise WorkerError("--max-abandoned-restarts needs --restart-abandoned.")
        if reconcile_after is not None:
            # run_task_worker reads it when the worker doesn't pass the option itself.
            os.environ[RECONCILE_ABANDONED_AFTER_ENV] = repr(reconcile_after)
        if restart_abandoned:
            os.environ[RESTART_ABANDONED_ENV] = "1"
        if max_restarts is not None:
            os.environ[MAX_ABANDONED_RESTARTS_ENV] = str(max_restarts)
        if processes > 1:
            outcome = asyncio.run(
                supervise_process_workers(
                    [
                        ProcessWorkerCommand(
                            argv=(
                                sys.executable,
                                "-m",
                                "cayu",
                                "worker",
                                args.name,
                                "--shutdown-grace-seconds",
                                str(args.shutdown_grace_seconds),
                            ),
                            cwd=project.root,
                            environment=process_worker_environment(index, processes),
                        )
                        for index in range(processes)
                    ],
                    shutdown_grace_seconds=args.shutdown_grace_seconds + 1,
                )
            )
            return outcome.exit_code
        with project_context(project.root):
            try:
                target = _load_worker_target(project.worker_target, name=args.name)
                app = build_project_app(project.factory_target, command="Worker")
                evals = _worker_evals(project.root, args.name, app)
            except SystemExit as exc:
                status = exc.code if type(exc.code) is int else 1
                raise WorkerError(
                    f"Worker project startup raised SystemExit with status {status}."
                ) from exc
            try:
                return asyncio.run(
                    _invoke_worker(
                        target,
                        app,
                        name=args.name,
                        shutdown_grace_seconds=args.shutdown_grace_seconds,
                        until_idle=until_idle,
                        evals=None if evals is None else evals.coordinator,
                        eval_apps=() if evals is None else evals.apps,
                        eval_context=None if evals is None else evals.context,
                    )
                )
            except SystemExit as exc:
                status = exc.code if type(exc.code) is int else 1
                raise WorkerError(
                    f"Worker {args.name!r} entrypoint raised SystemExit with status {status}."
                ) from exc
            finally:
                if evals is not None:
                    evals.close()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _worker_evals(root: Path, name: str, app: CayuApp) -> WorkerEvals | None:
    if project_eval_worker(root) != name:
        return None
    try:
        from cayu.cli._worker_evals import worker_evals
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] not in {"fastapi", "starlette", "sse_starlette"}:
            raise
        raise WorkerError(
            f"[tool.cayu.evals].worker runs eval runs on worker {name!r}, which needs "
            'cayu\'s server extra. Install it with: pip install "cayu[server]"'
        ) from exc
    return worker_evals(root, name, app)


def _resolve_worker_project(name: str) -> _WorkerProject:
    discovered = _discover_configured_project(
        command="cayu worker",
        configuration_example=(
            '[tool.cayu] factory = "module:build_app" and '
            '[tool.cayu.workers] name = "module:run_worker"'
        ),
        explicit_target_example=None,
        discovery_keys=("workers", "factory"),
    )
    workers_value = discovered.config.get("workers")
    if not isinstance(workers_value, dict):
        raise WorkerError(f"{discovered.pyproject}: [tool.cayu].workers must be a table.")
    workers = cast("dict[str, Any]", workers_value)
    target = workers.get(name)
    if target is None:
        available = ", ".join(sorted(workers)) or "none"
        raise WorkerError(f"Unknown worker {name!r}; available workers: {available}.")
    if not isinstance(target, str) or not target.strip():
        raise WorkerError(
            f"{discovered.pyproject}: [tool.cayu.workers].{name} must be a non-empty string."
        )
    return _WorkerProject(
        root=discovered.root,
        factory_target=discovered.factory_target,
        worker_target=target.strip(),
    )


def _load_worker_target(target: str, *, name: str) -> Any:
    try:
        worker = load_target(
            target,
            label=f"Worker {name!r} target",
            normalize_errors=True,
        )
    except TargetResolutionError as exc:
        raise WorkerError(str(exc)) from exc
    if not callable(worker):
        raise WorkerError(f"Worker {name!r} target must be callable.")
    if not inspect.iscoroutinefunction(worker):
        raise WorkerError(
            f"Worker {name!r} target must be declared with async def and accept "
            "exactly (app, stop)."
        )
    try:
        signature = inspect.signature(worker)
    except (TypeError, ValueError) as exc:
        raise WorkerError(f"Worker {name!r} target must accept exactly (app, stop).") from exc
    parameters = tuple(signature.parameters.values())
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if (
        len(parameters) != 2
        or any(parameter.kind not in positional_kinds for parameter in parameters)
        or any(parameter.default is not inspect.Parameter.empty for parameter in parameters)
    ):
        raise WorkerError(f"Worker {name!r} target must accept exactly (app, stop).")
    return worker


async def _invoke_worker(
    target: Any,
    app: CayuApp,
    *,
    name: str,
    shutdown_grace_seconds: float,
    until_idle: bool = False,
    evals: EvalRunCoordinator | None = None,
    eval_apps: tuple[CayuApp, ...] = (),
    eval_context: ProjectControlPlaneContext | None = None,
) -> int:
    """Run ``target`` until it returns or a signal stops it.

    ``evals`` runs the project's eval runs beside it: until no run is left when
    the worker runs until idle, and otherwise until it stops. A drained worker
    finishes the eval run it holds before it exits; a signal releases it.
    ``eval_apps``, the applications the eval targets run on, close with ``app``, and
    ``eval_context`` closes its eval store in this loop, which its connections belong to.
    """

    stop = asyncio.Event()
    drain_requested = False
    signal_received = asyncio.Event()
    loop = asyncio.get_running_loop()
    received_signal: int | None = None
    stop_requested_at: float | None = None
    loop_handlers: list[signal.Signals] = []
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_stop(signum: int) -> None:
        nonlocal received_signal, stop_requested_at
        if received_signal is None:
            received_signal = signum
            stop_requested_at = loop.time()
        stop.set()
        signal_received.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop, signum)
        except (NotImplementedError, RuntimeError):
            signal.signal(
                signum,
                lambda received, _frame: loop.call_soon_threadsafe(request_stop, received),
            )
        else:
            loop_handlers.append(signum)

    def request_drain(request: DrainRequest) -> None:
        # Cayu Cloud kept this worker running past its replacement: stop claiming, as a
        # SIGTERM would, but without the shutdown grace, so held tasks finish in their time.
        nonlocal drain_requested
        if not stop.is_set():
            print(drain_notice(request), file=sys.stderr, flush=True)
        drain_requested = True
        stop.set()
        if evals is not None:
            # The replacement takes new eval runs from now on, whatever the entrypoint holds.
            evals.request_drain()

    worker_task: asyncio.Future[Any] | None = None
    idle_result: TaskWorkerIdleResult | None = None
    signal_wait: asyncio.Task[bool] | None = None
    supervisor_watch = asyncio.create_task(watch_supervisor(request_stop))
    drain_watch = asyncio.create_task(watch_cloud_drain(request_drain))
    try:
        # A no-op outside Cayu Cloud; close_project_app stops it.
        await app.start_managed_settings(role="worker")

        def start_entrypoint() -> asyncio.Future[Any]:
            # The entrypoint's task copies this context when it is created.
            with _worker_until_idle_requested() if until_idle else contextlib.nullcontext():
                result = target(app, stop)
                if not inspect.isawaitable(result):
                    raise WorkerError(f"Worker {name!r} target must return an awaitable.")
                return asyncio.ensure_future(result)

        if evals is not None and until_idle:
            worker_task = asyncio.ensure_future(
                _until_idle_with_evals(start_entrypoint, evals, name=name)
            )
        else:
            if evals is not None:
                evals.start()
            worker_task = start_entrypoint()
        if isinstance(worker_task, asyncio.Task):
            worker_task.set_name(f"cayu-worker-{name}")
        signal_wait = asyncio.create_task(
            signal_received.wait(),
            name=f"cayu-worker-{name}-signal",
        )
        done, _pending = await asyncio.wait(
            {worker_task, signal_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if worker_task in done and received_signal is None:
            outcome = await worker_task
            if until_idle:
                idle_result = _idle_result(outcome, name)
            if evals is None or until_idle:
                return 0
            # A drained worker finishes the eval run it holds; otherwise it releases it.
            finished = asyncio.ensure_future(evals.drain() if drain_requested else evals.stop())
            done, _pending = await asyncio.wait(
                {finished, signal_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if finished in done:
                await finished
                return 0
            # A signal came first: the eval runs are stopped below, within the grace.
            finished.cancel()
        if received_signal is None:
            raise AssertionError("The signal event was set without a process signal.")
        stopping = [asyncio.shield(worker_task)]
        if evals is not None:
            stopping.append(asyncio.ensure_future(evals.stop()))
        try:
            await asyncio.wait_for(
                asyncio.gather(*stopping, return_exceptions=True),
                timeout=shutdown_grace_seconds,
            )
        except TimeoutError:
            worker_task.cancel()
            _exit_after_shutdown_timeout(
                f"Worker {name!r} did not stop within "
                f"{shutdown_grace_seconds:g} seconds after "
                f"{signal.Signals(received_signal).name}."
            )
        return 128 + received_signal
    finally:
        supervisor_watch.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await supervisor_watch
        drain_watch.cancel()
        # Asking Cloud only ever informs the worker; a failure of it never fails the worker.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await drain_watch
        if signal_wait is not None and not signal_wait.done():
            signal_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await signal_wait
        if evals is not None:
            # Already stopped on every path but a failed worker target, which
            # releases the eval run it holds the same way.
            with contextlib.suppress(Exception):
                await evals.stop()
        for signum in loop_handlers:
            loop.remove_signal_handler(signum)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        # After a stop signal the apps get what is left of the same grace, so
        # the whole shutdown stays within the supervisor's bound.
        for closing in (*eval_apps, app):
            await close_project_app(
                closing,
                timeout_s=(
                    shutdown_grace_seconds
                    if stop_requested_at is None
                    else max(
                        stop_requested_at + shutdown_grace_seconds - loop.time(),
                        _MIN_APP_SHUTDOWN_SECONDS,
                    )
                ),
            )
        if eval_context is not None:
            await eval_context.close()
        # Last, so a host can read the final stdout line.
        if idle_result is not None:
            print(idle_result.model_dump_json(), flush=True)


def _idle_result(outcome: object, name: str) -> TaskWorkerIdleResult:
    if not isinstance(outcome, TaskWorkerIdleResult):
        raise WorkerError(
            f"Worker {name!r} doesn't support --until-idle: when "
            "worker_until_idle_requested() is true its entrypoint must return "
            "the result of run_task_worker_until_idle."
        )
    return outcome


async def _until_idle_with_evals(
    start_entrypoint: Callable[[], asyncio.Future[Any]],
    evals: EvalRunCoordinator,
    *,
    name: str,
) -> TaskWorkerIdleResult:
    """Run the entrypoint and the eval runs until neither has work left.

    An eval run's workflow may create tasks for this worker, so the entrypoint runs
    again while eval runs go on and once more after them. Eval runs admitted meanwhile,
    as by a task that pass handled, get another pass: the worker is idle once a round
    in which neither took any work. ``handled`` counts both.
    """

    outcome = _idle_result(await start_entrypoint(), name)
    handled = outcome.handled
    while True:
        running = asyncio.ensure_future(evals.run_until_idle())
        tasks_handled = 0
        try:
            while not running.done():
                await asyncio.wait({running}, timeout=_EVAL_TASK_SERVICE_SECONDS)
                outcome = _idle_result(await start_entrypoint(), name)
                tasks_handled += outcome.handled
            took, evals_idle = running.result()
        finally:
            if not running.done():
                running.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await running
        handled += took + tasks_handled
        if not evals_idle or not outcome.idle or (took == 0 and tasks_handled == 0):
            return outcome.model_copy(
                update={"handled": handled, "idle": outcome.idle and evals_idle}
            )


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("shutdown grace must be a number.") from None
    if not isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("shutdown grace must be finite and positive.")
    return seconds


def _non_negative_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number of seconds.") from None
    if not isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number of seconds.")
    return seconds


def _restart_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number.") from None
    if not 0 <= count <= TASK_MAX_RESTARTS_LIMIT:
        raise argparse.ArgumentTypeError(f"must be from 0 to {TASK_MAX_RESTARTS_LIMIT}.")
    return count


def _exit_after_shutdown_timeout(message: str) -> NoReturn:
    """Exit without waiting for a cancellation-resistant task during loop cleanup."""
    print(f"error: {message}", file=sys.stderr, flush=True)
    os._exit(124)
