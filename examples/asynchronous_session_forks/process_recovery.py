"""Prove independent fork recovery across durable OS-process boundaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cayu import (
    AgentSpec,
    CayuApp,
    DispatchRequest,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    Message,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    SessionStatus,
    Task,
    TaskQuery,
    TaskStatus,
    TaskStoreDispatcher,
    ToolCapabilityCeiling,
)
from cayu.core import TextPart
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import ModelCompletionStageResult, RuntimePublicationRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore

_MODULE = "examples.asynchronous_session_forks.process_recovery"
_SOURCE_SESSION_ID = "process-fork-source"
_CHILD_SESSION_ID = "process-fork-child"
_DISPATCH_ID = "process-fork-dispatch"
_BOUNDARIES = ("claim", "provider_completion", "terminal_publication")
_CRASH_EXIT_CODES = {
    "claim": 71,
    "provider_completion": 72,
    "terminal_publication": 73,
}


@dataclass(frozen=True, slots=True)
class FreshProcessForkRecoveryTrace:
    """Bounded result from three isolated SQLite recovery scenarios."""

    boundaries: tuple[str, ...]
    queue_task_ids: dict[str, str]
    session_statuses: dict[str, str]
    task_statuses: dict[str, str]
    provider_calls: dict[str, dict[str, int]]


def _request_text(request: ModelRequest) -> str:
    return "\n".join(
        part.text
        for message in request.messages
        for part in message.content
        if isinstance(part, TextPart)
    )


def _append_fsynced_line(path: Path, line: str) -> None:
    payload = memoryview(f"{line}\n".encode())
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        while payload:
            written = os.write(descriptor, payload)
            if written <= 0:  # pragma: no cover - defensive OS contract guard
                raise OSError("Provider call evidence write made no progress.")
            payload = payload[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _DurableTraceProvider(ModelProvider):
    """Deterministic provider whose call evidence survives process death."""

    name = "process-fork-provider"

    def __init__(self, call_log: Path) -> None:
        self._call_log = call_log

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="examples:asynchronous-session-forks:process-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        operation = "child" if "process:child" in _request_text(request) else "source"
        _append_fsynced_line(self._call_log, operation)
        yield ModelStreamEvent.text_delta(f"{operation}:complete")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ExitAfterClaimTaskStore(SQLiteTaskStore):
    """Terminate only after SQLite has committed a task lease."""

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        task = await super().claim_task(
            worker_id,
            query,
            lease_seconds=lease_seconds,
        )
        if task is not None:
            os._exit(_CRASH_EXIT_CODES["claim"])
        return None


class _ExitAfterProviderCompletionSessionStore(SQLiteSessionStore):
    """Terminate only after SQLite has committed provider completion."""

    invocation_lifecycle_command_version = 1

    async def complete_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
    ) -> ModelCompletionStageResult:
        result = await super().complete_model_completion_stage(
            session_id,
            stage_id=stage_id,
            publication=publication,
        )
        if session_id == _CHILD_SESSION_ID:
            os._exit(_CRASH_EXIT_CODES["provider_completion"])
        return result


class _ExitAfterTerminalPublicationSessionStore(SQLiteSessionStore):
    """Terminate only after SQLite has committed the child terminal event."""

    invocation_lifecycle_command_version = 1

    async def append_event(self, session_id: str, event: Event) -> None:
        await super().append_event(session_id, event)
        if session_id == _CHILD_SESSION_ID and event.type is EventType.SESSION_COMPLETED:
            os._exit(_CRASH_EXIT_CODES["terminal_publication"])


def _initial_invocation() -> ResumeRequest:
    return ResumeRequest(
        session_id=_CHILD_SESSION_ID,
        messages=[Message.text("user", "process:child")],
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
        metadata={"application_worker": "fresh-process-child"},
        max_steps=1,
        retry_policy=RetryPolicy(
            max_attempts=1,
            max_unknown_attempts=1,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
        ),
    )


def _dispatch_request(invocation: ResumeRequest) -> DispatchRequest:
    return DispatchRequest(
        session_id=invocation.session_id,
        dispatch_id=_DISPATCH_ID,
        messages=invocation.messages,
        tool_capability_ceiling=invocation.tool_capability_ceiling,
        metadata=invocation.metadata,
        max_steps=invocation.max_steps,
        retry_policy=invocation.retry_policy,
    )


def _build_app(
    root: Path,
    *,
    session_store_type: type[SQLiteSessionStore] = SQLiteSessionStore,
    task_store_type: type[SQLiteTaskStore] = SQLiteTaskStore,
    task_clock: Callable[[], datetime] | None = None,
) -> tuple[CayuApp, TaskStoreDispatcher, SQLiteSessionStore, SQLiteTaskStore]:
    sessions = session_store_type(root / "sessions.sqlite")
    tasks = task_store_type(root / "tasks.sqlite", clock=task_clock)
    dispatcher = TaskStoreDispatcher(
        tasks,
        lease_seconds=1,
        recover_stalled_sessions_after_seconds=0,
    )
    app = CayuApp(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    app.register_provider(_DurableTraceProvider(root / "provider-calls.log"), default=True)
    app.register_agent(AgentSpec(name="worker", model="deterministic-model"))
    return app, dispatcher, sessions, tasks


async def _collect(stream) -> list[Any]:
    return [item async for item in stream]


async def _close_stores(sessions: SQLiteSessionStore, tasks: SQLiteTaskStore) -> None:
    await sessions.close()
    await tasks.close()


async def _setup(root: Path) -> None:
    app, _, sessions, tasks = _build_app(root)
    try:
        await _collect(
            app.run(
                RunRequest(
                    agent_name="worker",
                    session_id=_SOURCE_SESSION_ID,
                    causal_budget_id="process-fork-budget",
                    messages=[Message.text("user", "process:source")],
                )
            )
        )
        source = await app.snapshot_fork_source(_SOURCE_SESSION_ID)
        invocation = _initial_invocation()
        await _collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id=_SOURCE_SESSION_ID,
                    session_id=_CHILD_SESSION_ID,
                    expected_source=source,
                    initial_invocation=invocation,
                    initial_dispatch_id=_DISPATCH_ID,
                    metadata={"application_worker": "fresh-process-child"},
                )
            )
        )
        handle = await app.dispatch(_dispatch_request(invocation))
        (root / "state.json").write_text(
            json.dumps({"queue_task_id": handle.metadata["queue_task_id"]}),
            encoding="utf-8",
        )
    finally:
        await _close_stores(sessions, tasks)


async def _crash(root: Path, boundary: str) -> None:
    session_store_type: type[SQLiteSessionStore] = SQLiteSessionStore
    task_store_type: type[SQLiteTaskStore] = SQLiteTaskStore
    if boundary == "claim":
        task_store_type = _ExitAfterClaimTaskStore
    elif boundary == "provider_completion":
        session_store_type = _ExitAfterProviderCompletionSessionStore
    elif boundary == "terminal_publication":
        session_store_type = _ExitAfterTerminalPublicationSessionStore
    else:  # pragma: no cover - argparse and the parent validate this
        raise ValueError(f"Unsupported boundary: {boundary}")
    app, dispatcher, sessions, tasks = _build_app(
        root,
        session_store_type=session_store_type,
        task_store_type=task_store_type,
    )
    try:
        await dispatcher.process_next(app, worker_id=f"crashing-{boundary}-worker")
    finally:
        await _close_stores(sessions, tasks)
    raise AssertionError(f"The {boundary} process did not terminate at its commit boundary.")


async def _recover(root: Path, boundary: str) -> dict[str, str]:
    recovery_clock = datetime.now(UTC) + timedelta(days=1)
    app, dispatcher, sessions, tasks = _build_app(
        root,
        task_clock=lambda: recovery_clock,
    )
    try:
        source = await app.snapshot_fork_source(_SOURCE_SESSION_ID)
        invocation = _initial_invocation()
        await _collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id=_SOURCE_SESSION_ID,
                    session_id=_CHILD_SESSION_ID,
                    expected_source=source,
                    initial_invocation=invocation,
                    initial_dispatch_id=_DISPATCH_ID,
                    metadata={"application_worker": "fresh-process-child"},
                )
            )
        )
        replayed = await app.dispatch(_dispatch_request(invocation))
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if replayed.metadata["queue_task_id"] != state["queue_task_id"]:
            raise AssertionError("Fresh producer reconstruction changed queue identity.")

        reclaimed = await tasks.reclaim_expired(query=TaskQuery())
        if [task.id for task in reclaimed] != [state["queue_task_id"]]:
            raise AssertionError("Fresh worker did not reclaim the expired durable lease.")

        for ordinal in range(5):
            await dispatcher.process_next(
                app,
                worker_id=f"recovery-{boundary}-{ordinal}",
            )
            task = await tasks.load_task(state["queue_task_id"])
            if task is not None and task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                break
        task = await tasks.load_task(state["queue_task_id"])
        session = await sessions.load(_CHILD_SESSION_ID)
        if task is None or task.status is not TaskStatus.COMPLETED:
            raise AssertionError(f"Recovered task is not complete: {task!r}")
        expected_session_status = (
            SessionStatus.INTERRUPTED
            if boundary == "provider_completion"
            else SessionStatus.COMPLETED
        )
        if session is None or session.status is not expected_session_status:
            raise AssertionError(
                "Recovered child has the wrong terminal status: "
                f"expected {expected_session_status.value}, got {session!r}"
            )
        return {
            "queue_task_id": task.id,
            "session_status": session.status.value,
            "task_status": task.status.value,
        }
    finally:
        await _close_stores(sessions, tasks)


def _run_phase(*, phase: str, root: Path, boundary: str) -> None:
    if phase == "setup":
        asyncio.run(_setup(root))
    elif phase == "crash":
        asyncio.run(_crash(root, boundary))
    elif phase == "recover":
        print(json.dumps(asyncio.run(_recover(root, boundary)), sort_keys=True))
    else:  # pragma: no cover - argparse validates this
        raise ValueError(f"Unsupported phase: {phase}")


def _phase_command(*, phase: str, root: Path, boundary: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        _MODULE,
        "--phase",
        phase,
        "--root",
        str(root),
        "--boundary",
        boundary,
    ]


def run_fresh_process_fork_recovery_trace() -> FreshProcessForkRecoveryTrace:
    """Run isolated admission, crash, lease-expiry, and recovery processes."""

    queue_task_ids: dict[str, str] = {}
    session_statuses: dict[str, str] = {}
    task_statuses: dict[str, str] = {}
    provider_calls: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix="cayu-process-forks-") as temporary_root:
        root = Path(temporary_root)
        for boundary in _BOUNDARIES:
            scenario_root = root / boundary
            scenario_root.mkdir(mode=0o700)
            setup = subprocess.run(
                _phase_command(phase="setup", root=scenario_root, boundary=boundary),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if setup.returncode != 0:
                raise RuntimeError(f"{boundary} setup failed: {setup.stderr.strip()}")
            crashed = subprocess.run(
                _phase_command(phase="crash", root=scenario_root, boundary=boundary),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if crashed.returncode != _CRASH_EXIT_CODES[boundary]:
                raise RuntimeError(
                    f"{boundary} exited {crashed.returncode}, expected "
                    f"{_CRASH_EXIT_CODES[boundary]}: {crashed.stderr.strip()}"
                )
            recovered = subprocess.run(
                _phase_command(phase="recover", root=scenario_root, boundary=boundary),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if recovered.returncode != 0:
                raise RuntimeError(f"{boundary} recovery failed: {recovered.stderr.strip()}")
            evidence = json.loads(recovered.stdout)
            calls = Counter(
                (scenario_root / "provider-calls.log").read_text(encoding="utf-8").splitlines()
            )
            if calls != Counter({"source": 1, "child": 1}):
                raise AssertionError(f"{boundary} duplicated provider work: {calls!r}")
            queue_task_ids[boundary] = evidence["queue_task_id"]
            session_statuses[boundary] = evidence["session_status"]
            task_statuses[boundary] = evidence["task_status"]
            provider_calls[boundary] = dict(sorted(calls.items()))

    return FreshProcessForkRecoveryTrace(
        boundaries=_BOUNDARIES,
        queue_task_ids=queue_task_ids,
        session_statuses=session_statuses,
        task_statuses=task_statuses,
        provider_calls=provider_calls,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("setup", "crash", "recover"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--boundary", choices=_BOUNDARIES)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    internal_arguments = (arguments.phase, arguments.root, arguments.boundary)
    if all(value is None for value in internal_arguments):
        print(json.dumps(asdict(run_fresh_process_fork_recovery_trace()), indent=2, sort_keys=True))
        return
    if any(value is None for value in internal_arguments):
        raise SystemExit("--phase, --root, and --boundary must be supplied together")
    arguments.root.mkdir(parents=True, exist_ok=True)
    _run_phase(
        phase=arguments.phase,
        root=arguments.root,
        boundary=arguments.boundary,
    )


if __name__ == "__main__":
    main()
