"""Run one named project worker with a process-scoped Cayu application."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import os
import signal
import sys
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn, cast

from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli.project import (
    _discover_configured_project,
    build_project_app,
    project_context,
)
from cayu.runtime.app import CayuApp


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
        "--shutdown-grace-seconds",
        type=_positive_seconds,
        default=30.0,
        help="Maximum cooperative shutdown time after SIGINT or SIGTERM (default: 30).",
    )


def run_worker(args: argparse.Namespace) -> int:
    try:
        project = _resolve_worker_project(args.name)
        with project_context(project.root):
            try:
                target = _load_worker_target(project.worker_target, name=args.name)
                app = build_project_app(project.factory_target, command="Worker")
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
                    )
                )
            except SystemExit as exc:
                status = exc.code if type(exc.code) is int else 1
                raise WorkerError(
                    f"Worker {args.name!r} entrypoint raised SystemExit with status {status}."
                ) from exc
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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
) -> int:
    stop = asyncio.Event()
    signal_received = asyncio.Event()
    loop = asyncio.get_running_loop()
    received_signal: int | None = None
    loop_handlers: list[signal.Signals] = []
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_stop(signum: int) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum
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

    worker_task: asyncio.Future[Any] | None = None
    signal_wait: asyncio.Task[bool] | None = None
    try:
        result = target(app, stop)
        if not inspect.isawaitable(result):
            raise WorkerError(f"Worker {name!r} target must return an awaitable.")
        worker_task = asyncio.ensure_future(result)
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
            await worker_task
            return 0
        if received_signal is None:
            raise AssertionError("The signal event was set without a process signal.")
        try:
            await asyncio.wait_for(
                asyncio.shield(worker_task),
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
        if signal_wait is not None and not signal_wait.done():
            signal_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await signal_wait
        for signum in loop_handlers:
            loop.remove_signal_handler(signum)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("shutdown grace must be a number.") from None
    if not isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("shutdown grace must be finite and positive.")
    return seconds


def _exit_after_shutdown_timeout(message: str) -> NoReturn:
    """Exit without waiting for a cancellation-resistant task during loop cleanup."""
    print(f"error: {message}", file=sys.stderr, flush=True)
    os._exit(124)
