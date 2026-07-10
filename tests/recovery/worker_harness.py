from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EventSink,
    IncompleteSessionRecoveryRequest,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    Session,
    SessionQuery,
    Task,
    TaskCreate,
    TaskQuery,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRequest,
    ToolRoundRecoveryRequest,
)
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tools import SubagentExecutionMode, SubagentSpec, SubagentTool

_WORKER_TIMEOUT_S = 20.0
_POLL_INTERVAL_S = 0.02
_PROVIDER_NAME = "recovery-harness"
_AGENT_NAME = "recovery-agent"


class RecoveryScenario(StrEnum):
    ORDINARY_TOOL = "ordinary_tool"
    APPROVAL = "approval"
    BACKGROUND_SUBAGENT = "background_subagent"
    TASK_CLAIM = "task_claim"


class RecoveryAction(StrEnum):
    START = "start"
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    APPROVE = "approve"
    DENY = "deny"
    RECOVER = "recover"
    START_UNATTACHED = "start_unattached"
    START_ATTACHED = "start_attached"
    RECOVER_UNATTACHED = "recover_unattached"
    RECOVER_ATTACHED = "recover_attached"


_SCENARIO_ACTIONS = {
    RecoveryScenario.ORDINARY_TOOL: frozenset(
        {RecoveryAction.START, RecoveryAction.AUTOMATIC, RecoveryAction.MANUAL}
    ),
    RecoveryScenario.APPROVAL: frozenset(
        {RecoveryAction.START, RecoveryAction.APPROVE, RecoveryAction.DENY}
    ),
    RecoveryScenario.BACKGROUND_SUBAGENT: frozenset({RecoveryAction.START, RecoveryAction.RECOVER}),
    RecoveryScenario.TASK_CLAIM: frozenset(
        {
            RecoveryAction.START_UNATTACHED,
            RecoveryAction.START_ATTACHED,
            RecoveryAction.RECOVER_UNATTACHED,
            RecoveryAction.RECOVER_ATTACHED,
        }
    ),
}


@dataclass(frozen=True)
class BackendConfig:
    kind: str
    session_path: str | None = None
    task_path: str | None = None
    dsn: str | None = None

    @classmethod
    def sqlite(cls, root: Path) -> BackendConfig:
        return cls(
            kind="sqlite",
            session_path=str(root / "sessions.sqlite"),
            task_path=str(root / "tasks.sqlite"),
        )

    @classmethod
    def postgres(cls, dsn: str) -> BackendConfig:
        return cls(kind="postgres", dsn=dsn)

    def as_json(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "session_path": self.session_path,
            "task_path": self.task_path,
            "dsn": self.dsn,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> BackendConfig:
        return cls(
            kind=value["kind"],
            session_path=value.get("session_path"),
            task_path=value.get("task_path"),
            dsn=value.get("dsn"),
        )


@dataclass(frozen=True)
class SessionState:
    session: Session | None
    checkpoint: dict[str, Any]
    transcript: list[Message]
    events: list[Event]


class WorkerHandle:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        phase_path: Path,
        result_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        stdout_stream,
        stderr_stream,
        durable_diagnostics: Callable[[], str],
    ) -> None:
        self.process = process
        self.phase_path = phase_path
        self.result_path = result_path
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self._stdout_stream = stdout_stream
        self._stderr_stream = stderr_stream
        self._durable_diagnostics = durable_diagnostics

    def wait_for_phase(self, expected: str) -> dict[str, Any]:
        deadline = time.monotonic() + _WORKER_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.phase_path.exists():
                phase = json.loads(self.phase_path.read_text(encoding="utf-8"))
                if phase.get("phase") == expected:
                    return phase
            returncode = self.process.poll()
            if returncode is not None:
                self._close_streams()
                raise AssertionError(
                    f"worker exited with status {returncode} before phase {expected!r}\n"
                    f"{self._diagnostics()}"
                )
            time.sleep(_POLL_INTERVAL_S)
        raise AssertionError(f"worker did not reach phase {expected!r}\n{self._diagnostics()}")

    def sigkill(self) -> None:
        if self.process.poll() is not None:
            self._close_streams()
            raise AssertionError(f"worker exited before SIGKILL\n{self._diagnostics()}")
        os.killpg(self.process.pid, signal.SIGKILL)
        returncode = self.process.wait(timeout=_WORKER_TIMEOUT_S)
        self._close_streams()
        if returncode != -signal.SIGKILL:
            raise AssertionError(
                f"worker exited with {returncode}, expected {-signal.SIGKILL}\n"
                f"{self._diagnostics()}"
            )

    def wait_success(self) -> dict[str, Any]:
        try:
            returncode = self.process.wait(timeout=_WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.terminate_if_running()
            raise AssertionError(f"worker timed out\n{self._diagnostics()}") from None
        finally:
            self._close_streams()
        if returncode != 0:
            raise AssertionError(f"worker exited with status {returncode}\n{self._diagnostics()}")
        if not self.result_path.exists():
            raise AssertionError(f"worker wrote no result\n{self._diagnostics()}")
        return json.loads(self.result_path.read_text(encoding="utf-8"))

    def terminate_if_running(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=_WORKER_TIMEOUT_S)
        self._close_streams()

    def _close_streams(self) -> None:
        if not self._stdout_stream.closed:
            self._stdout_stream.close()
        if not self._stderr_stream.closed:
            self._stderr_stream.close()

    def _diagnostics(self) -> str:
        phase = self.phase_path.read_text(encoding="utf-8") if self.phase_path.exists() else "none"
        stdout = self.stdout_path.read_text(encoding="utf-8") if self.stdout_path.exists() else ""
        stderr = self.stderr_path.read_text(encoding="utf-8") if self.stderr_path.exists() else ""
        return (
            f"phase: {phase}\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
            f"durable state:\n{self._durable_diagnostics()}"
        )


class RecoveryHarness:
    def __init__(self, root: Path, backend: BackendConfig) -> None:
        self.root = root
        self.backend = backend
        self.marker_path = root / "side-effects.jsonl"
        self._workers: list[WorkerHandle] = []
        self._session_ids: set[str] = set()
        self._task_ids: set[str] = set()

    def __enter__(self) -> RecoveryHarness:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for worker in self._workers:
            worker.terminate_if_running()
        self._cleanup_artifacts()

    def launch(
        self,
        *,
        scenario: str,
        action: str,
        session_id: str,
        **values: Any,
    ) -> WorkerHandle:
        scenario_name = RecoveryScenario(scenario)
        action_name = RecoveryAction(action)
        if action_name not in _SCENARIO_ACTIONS[scenario_name]:
            raise ValueError(f"Action {action!r} is not valid for scenario {scenario!r}")
        worker_id = uuid4().hex
        phase_path = self.root / f"phase-{worker_id}.json"
        result_path = self.root / f"result-{worker_id}.json"
        stdout_path = self.root / f"stdout-{worker_id}.log"
        stderr_path = self.root / f"stderr-{worker_id}.log"
        config_path = self.root / f"config-{worker_id}.json"
        config = {
            "scenario": scenario_name,
            "action": action_name,
            "session_id": session_id,
            "backend": self.backend.as_json(),
            "phase_path": str(phase_path),
            "result_path": str(result_path),
            "marker_path": str(self.marker_path),
            **values,
        }
        self._session_ids.add(session_id)
        child_session_id = values.get("child_session_id")
        if isinstance(child_session_id, str):
            self._session_ids.add(child_session_id)
        task_id = values.get("task_id")
        if isinstance(task_id, str):
            self._task_ids.add(task_id)
        _write_json_atomic(config_path, config)

        stdout_stream = stdout_path.open("wb")
        stderr_stream = stderr_path.open("wb")
        env = os.environ.copy()
        source_root = Path(__file__).resolve().parents[2] / "src"
        inherited_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(source_root)
            if not inherited_pythonpath
            else os.pathsep.join((str(source_root), inherited_pythonpath))
        )
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path)],
            env=env,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
        handle = WorkerHandle(
            process,
            phase_path=phase_path,
            result_path=result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_stream=stdout_stream,
            stderr_stream=stderr_stream,
            durable_diagnostics=lambda: self._durable_diagnostics(config),
        )
        self._workers.append(handle)
        return handle

    async def load_session_state(self, session_id: str) -> SessionState:
        store = _session_store(self.backend)
        try:
            session = await store.load(session_id)
            if session is None:
                return SessionState(
                    session=None,
                    checkpoint={},
                    transcript=[],
                    events=[],
                )
            return SessionState(
                session=session,
                checkpoint=await store.load_checkpoint(session_id) or {},
                transcript=await store.load_transcript(session_id),
                events=await store.load_events(session_id),
            )
        finally:
            await store.close()

    async def list_child_sessions(self, parent_session_id: str) -> list[Session]:
        store = _session_store(self.backend)
        try:
            result = await store.list_sessions(
                SessionQuery(parent_session_id=parent_session_id, limit=100)
            )
            return result.sessions
        finally:
            await store.close()

    async def list_causal_sessions(self, causal_budget_id: str) -> list[Session]:
        store = _session_store(self.backend)
        try:
            result = await store.list_sessions(
                SessionQuery(causal_budget_id=causal_budget_id, limit=100)
            )
            return result.sessions
        finally:
            await store.close()

    async def load_task(self, task_id: str) -> Task | None:
        store = _task_store(self.backend)
        try:
            return await store.load_task(task_id)
        finally:
            await store.close()

    def read_marker(self) -> list[dict[str, Any]]:
        if not self.marker_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.marker_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def _cleanup_artifacts(self) -> None:
        paths = {self.marker_path}
        for pattern in (
            "config-*.json",
            "phase-*.json",
            "result-*.json",
            "stdout-*.log",
            "stderr-*.log",
        ):
            paths.update(self.root.glob(pattern))
        if self.backend.kind == "sqlite":
            for raw_path in (self.backend.session_path, self.backend.task_path):
                if raw_path is None:
                    continue
                path = Path(raw_path)
                paths.update(
                    {
                        path,
                        Path(f"{path}-journal"),
                        Path(f"{path}-shm"),
                        Path(f"{path}-wal"),
                    }
                )
        for path in paths:
            path.unlink(missing_ok=True)
        if self.backend.kind == "postgres":
            asyncio.run(self._cleanup_postgres_rows())

    def _durable_diagnostics(self, config: dict[str, Any]) -> str:
        async def snapshot() -> dict[str, Any]:
            session_id = config["session_id"]
            state = await self.load_session_state(session_id)
            children = await self.list_child_sessions(session_id)
            task = None
            if isinstance(config.get("task_id"), str):
                task = await self.load_task(config["task_id"])
            return {
                "session": (
                    state.session.model_dump(mode="json") if state.session is not None else None
                ),
                "checkpoint": state.checkpoint,
                "transcript": [message.model_dump(mode="json") for message in state.transcript],
                "events": [event.model_dump(mode="json") for event in state.events],
                "task": task.model_dump(mode="json") if task is not None else None,
                "children": [child.model_dump(mode="json") for child in children],
            }

        try:
            return json.dumps(asyncio.run(snapshot()), indent=2, sort_keys=True)
        except Exception as exc:
            return f"unavailable: {type(exc).__name__}: {exc}"

    async def _cleanup_postgres_rows(self) -> None:
        if self.backend.dsn is None:
            return
        from psycopg import AsyncConnection

        session_ids = sorted(self._session_ids)
        task_ids = sorted(self._task_ids)
        async with await AsyncConnection.connect(self.backend.dsn) as connection:
            async with connection.cursor() as cursor:
                if task_ids or session_ids:
                    await cursor.execute(
                        """
                        DELETE FROM cayu_tasks
                        WHERE id = ANY(%s) OR session_id = ANY(%s)
                        """,
                        (task_ids, session_ids),
                    )
                if session_ids:
                    await cursor.execute(
                        "DELETE FROM cayu_sessions WHERE parent_session_id = ANY(%s)",
                        (session_ids,),
                    )
                    await cursor.execute(
                        "DELETE FROM cayu_sessions WHERE id = ANY(%s)",
                        (session_ids,),
                    )
            await connection.commit()


class _RecoveryProvider(ModelProvider):
    name = _PROVIDER_NAME

    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if self.mode in {"ordinary_start", "approval_start", "task_start"}:
            yield ModelStreamEvent.tool_call(
                id="call_side_effect",
                name="side_effect",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if self.mode == "subagent_start":
            yield ModelStreamEvent.tool_call(
                id="call_background_subagent",
                name="subagent",
                arguments={
                    "agent": "reviewer",
                    "task": "Perform the background review.",
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("Recovery completed.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _MarkerTool(Tool):
    def __init__(
        self,
        marker_path: Path,
        phase_path: Path | None,
        *,
        phase_name: str = "tool_side_effect_recorded",
    ) -> None:
        super().__init__(
            ToolSpec(
                name="side_effect",
                description="Record one externally visible side effect.",
                input_schema={"type": "object", "additionalProperties": False},
                parallel_safe=False,
                effect=ToolEffect.EXTERNAL,
            )
        )
        self.marker_path = marker_path
        self.phase_path = phase_path
        self.phase_name = phase_name

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.idempotency_key is None:
            raise AssertionError("ordinary tool execution did not receive an idempotency key")
        tool_call_id = ctx.metadata.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            raise AssertionError("ordinary tool execution did not receive a tool_call_id")
        record = {
            "idempotency_key": ctx.idempotency_key,
            "session_id": ctx.session_id,
            "tool_call_id": tool_call_id,
        }
        _append_json_line(self.marker_path, record)
        if self.phase_path is not None:
            _write_json_atomic(
                self.phase_path,
                {
                    "phase": self.phase_name,
                    **record,
                },
            )
            await asyncio.Event().wait()
        return ToolResult(content="side effect recorded", structured=record)


class _KillpointSink(EventSink):
    def __init__(
        self,
        phase_path: Path,
        killpoint: str,
        *,
        parent_session_id: str | None = None,
    ) -> None:
        self.phase_path = phase_path
        self.killpoint = killpoint
        self.parent_session_id = parent_session_id

    async def emit(self, event: Event) -> None:
        if (
            self.killpoint == "approval_requested"
            and event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        ):
            approval = event.payload["approval"]
            _write_json_atomic(
                self.phase_path,
                {
                    "phase": "approval_requested_persisted",
                    "approval_id": approval["approval_id"],
                    "event_id": event.id,
                },
            )
            await asyncio.Event().wait()
        if (
            self.killpoint == "subagent_child_started"
            and event.type == EventType.SESSION_STARTED
            and self.parent_session_id is not None
            and event.session_id.startswith(f"{self.parent_session_id}_subagent_")
        ):
            _write_json_atomic(
                self.phase_path,
                {
                    "phase": "subagent_child_started",
                    "child_session_id": event.session_id,
                    "event_id": event.id,
                },
            )
            await asyncio.Event().wait()


def _session_store(backend: BackendConfig):
    if backend.kind == "sqlite":
        if backend.session_path is None:
            raise ValueError("SQLite backend requires session_path")
        return SQLiteSessionStore(backend.session_path)
    if backend.kind == "postgres":
        if backend.dsn is None:
            raise ValueError("Postgres backend requires dsn")
        from cayu.storage import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        return PostgresSessionStore(backend.dsn, schema_mode=SchemaMode.CREATE)
    raise ValueError(f"Unknown backend: {backend.kind}")


def _task_store(backend: BackendConfig):
    if backend.kind == "sqlite":
        if backend.task_path is None:
            raise ValueError("SQLite backend requires task_path")
        return SQLiteTaskStore(backend.task_path)
    if backend.kind == "postgres":
        if backend.dsn is None:
            raise ValueError("Postgres backend requires dsn")
        from cayu.storage import PostgresTaskStore
        from cayu.storage.migrations import SchemaMode

        return PostgresTaskStore(backend.dsn, schema_mode=SchemaMode.CREATE)
    raise ValueError(f"Unknown backend: {backend.kind}")


def _ordinary_app(config: dict[str, Any], *, mode: str) -> CayuApp:
    backend = BackendConfig.from_json(config["backend"])
    store = _session_store(backend)
    phase_path = Path(config["phase_path"]) if mode == "ordinary_start" else None
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_RecoveryProvider(mode), default=True)
    app.register_agent(
        AgentSpec(name=_AGENT_NAME, model="recovery-model"),
        tools=[_MarkerTool(Path(config["marker_path"]), phase_path)],
    )
    return app


def _approval_app(config: dict[str, Any], *, initial: bool) -> CayuApp:
    backend = BackendConfig.from_json(config["backend"])
    store = _session_store(backend)
    sinks = [_KillpointSink(Path(config["phase_path"]), "approval_requested")] if initial else []
    app = CayuApp(session_store=store, event_sinks=sinks, enable_logging=False)
    app.register_provider(
        _RecoveryProvider("approval_start" if initial else "recovery"),
        default=True,
    )
    app.register_agent(
        AgentSpec(name=_AGENT_NAME, model="recovery-model"),
        tools=[_MarkerTool(Path(config["marker_path"]), None)],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools={"side_effect"}),
    )
    return app


def _subagent_app(config: dict[str, Any], *, initial: bool) -> CayuApp:
    backend = BackendConfig.from_json(config["backend"])
    store = _session_store(backend)
    sinks = (
        [
            _KillpointSink(
                Path(config["phase_path"]),
                "subagent_child_started",
                parent_session_id=config["session_id"],
            )
        ]
        if initial
        else []
    )
    app = CayuApp(session_store=store, event_sinks=sinks, enable_logging=False)
    app.register_provider(
        _RecoveryProvider("subagent_start" if initial else "recovery"),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="parent-agent", model="recovery-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="reviewer-agent",
                        mode=SubagentExecutionMode.BACKGROUND,
                    )
                },
            )
        ],
    )
    app.register_agent(AgentSpec(name="reviewer-agent", model="recovery-model"))
    return app


def _task_app(config: dict[str, Any], *, initial: bool, session_store, task_store) -> CayuApp:
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(
        _RecoveryProvider("task_start" if initial else "recovery"),
        default=True,
    )
    app.register_agent(
        AgentSpec(name=_AGENT_NAME, model="recovery-model"),
        tools=[
            _MarkerTool(
                Path(config["marker_path"]),
                Path(config["phase_path"]) if initial else None,
                phase_name="attached_tool_side_effect_recorded",
            )
        ],
    )
    return app


async def _run_ordinary_tool(config: dict[str, Any]) -> dict[str, Any]:
    action = config["action"]
    session_id = config["session_id"]
    app = _ordinary_app(
        config,
        mode="ordinary_start" if action == "start" else "recovery",
    )
    try:
        if action == "start":
            async for _ in app.run(
                RunRequest(
                    agent_name=_AGENT_NAME,
                    session_id=session_id,
                    messages=[Message.text("user", "Perform the side effect.")],
                )
            ):
                pass
            raise AssertionError("ordinary tool start unexpectedly returned")

        if action == "automatic":
            recovery = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            async for _ in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "Continue after recovery.")],
                )
            ):
                pass
            return {"actions": [str(item) for item in recovery.actions]}

        if action == "manual":
            checkpoint = await app.session_store.load_checkpoint(session_id) or {}
            pending = checkpoint["pending_tool_round"]
            async for _ in app.recover_tool_round(
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=pending["round_id"],
                    tool_call_id="call_side_effect",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="External marker verified the side effect.",
                    structured={"marker_verified": True},
                    resolved_by=ResolutionActor(
                        subject="recovery-operator",
                        source=ResolutionActorSource.REQUEST,
                    ),
                )
            ):
                pass
            return {"manual_recovery": True}
        raise ValueError(f"Unsupported ordinary tool action: {action}")
    finally:
        await app.session_store.close()


async def _run_approval(config: dict[str, Any]) -> dict[str, Any]:
    action = config["action"]
    session_id = config["session_id"]
    app = _approval_app(config, initial=action == "start")
    try:
        if action == "start":
            async for _ in app.run(
                RunRequest(
                    agent_name=_AGENT_NAME,
                    session_id=session_id,
                    messages=[Message.text("user", "Request the gated side effect.")],
                )
            ):
                pass
            raise AssertionError("approval start unexpectedly returned")

        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        checkpoint = await app.session_store.load_checkpoint(session_id) or {}
        approval_id = checkpoint["pending_tool_approval"]["approval_id"]
        request = ToolApprovalRequest(
            session_id=session_id,
            approval_id=approval_id,
            decision=(
                ToolApprovalDecision.APPROVE if action == "approve" else ToolApprovalDecision.DENY
            ),
            reason=f"SIGKILL recovery test {action}",
            resolved_by=ResolutionActor(
                subject="approval-operator",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        async for _ in app.resolve_tool_approval(request):
            pass

        retry_rejected = False
        try:
            async for _ in app.resolve_tool_approval(request):
                pass
        except (RuntimeError, ValueError):
            retry_rejected = True
        return {
            "approval_id": approval_id,
            "recovery_actions": [str(item) for item in recovery.actions],
            "retry_rejected": retry_rejected,
        }
    finally:
        await app.session_store.close()


async def _run_background_subagent(config: dict[str, Any]) -> dict[str, Any]:
    action = config["action"]
    parent_session_id = config["session_id"]
    app = _subagent_app(config, initial=action == "start")
    try:
        if action == "start":
            async for _ in app.run(
                RunRequest(
                    agent_name="parent-agent",
                    session_id=parent_session_id,
                    causal_budget_id="sigkill-subagent-causal",
                    messages=[Message.text("user", "Start a background review.")],
                )
            ):
                pass
            raise AssertionError("background subagent start unexpectedly returned")

        parent_recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=parent_session_id)
        )
        child_session_id = config["child_session_id"]
        child_recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=child_session_id)
        )
        async for _ in app.resume(
            ResumeRequest(
                session_id=parent_session_id,
                messages=[Message.text("user", "Continue after child recovery.")],
            )
        ):
            pass
        return {
            "parent_actions": [str(item) for item in parent_recovery.actions],
            "child_actions": [str(item) for item in child_recovery.actions],
        }
    finally:
        await app.session_store.close()


async def _wait_for_task_lease_expiry(task_store, task_id: str) -> None:
    deadline = time.monotonic() + _WORKER_TIMEOUT_S
    while time.monotonic() < deadline:
        task = await task_store.load_task(task_id)
        if task is None:
            raise AssertionError(f"task disappeared while waiting for lease expiry: {task_id}")
        if task.lease_expires_at is not None and task.lease_expires_at <= datetime.now(UTC):
            return
        await asyncio.sleep(_POLL_INTERVAL_S)
    raise AssertionError(f"task lease did not expire: {task_id}")


async def _run_task_claim(config: dict[str, Any]) -> dict[str, Any]:
    action = config["action"]
    backend = BackendConfig.from_json(config["backend"])
    task_store = _task_store(backend)
    session_store = None
    task_id = config["task_id"]
    task_type = config["task_type"]
    session_id = config["session_id"]
    try:
        if action in {"start_unattached", "start_attached"}:
            await task_store.create_task(TaskCreate(task_id=task_id, type=task_type))
            claimed = await task_store.claim_task(
                "worker-a",
                TaskQuery(type=task_type),
                lease_seconds=1,
            )
            if claimed is None or claimed.id != task_id:
                raise AssertionError("worker-a did not claim the expected task")

        if action == "start_unattached":
            _write_json_atomic(
                Path(config["phase_path"]),
                {
                    "phase": "unattached_task_claimed",
                    "task_id": task_id,
                    "worker_id": "worker-a",
                },
            )
            await asyncio.Event().wait()
            raise AssertionError("unattached task worker unexpectedly resumed")

        if action == "start_attached":
            session_store = _session_store(backend)
            app = _task_app(
                config,
                initial=True,
                session_store=session_store,
                task_store=task_store,
            )
            async for _ in app.run(
                RunRequest(
                    agent_name=_AGENT_NAME,
                    session_id=session_id,
                    task_id=task_id,
                    task_worker_id="worker-a",
                    messages=[Message.text("user", "Execute the attached task.")],
                )
            ):
                pass
            raise AssertionError("attached task start unexpectedly returned")

        await _wait_for_task_lease_expiry(task_store, task_id)
        reclaimed = await task_store.reclaim_expired(
            query=TaskQuery(type=task_type),
            max_reclaims=10,
        )
        worker_b_claim = await task_store.claim_task(
            "worker-b",
            TaskQuery(type=task_type),
            lease_seconds=30,
        )
        session_store = _session_store(backend)
        app = _task_app(
            config,
            initial=False,
            session_store=session_store,
            task_store=task_store,
        )

        if action == "recover_unattached":
            if worker_b_claim is None or worker_b_claim.id != task_id:
                raise AssertionError("worker-b did not claim the reclaimed task")
            async for _ in app.run(
                RunRequest(
                    agent_name=_AGENT_NAME,
                    session_id=session_id,
                    task_id=task_id,
                    task_worker_id="worker-b",
                    messages=[Message.text("user", "Complete the reclaimed task.")],
                )
            ):
                pass
        elif action == "recover_attached":
            if worker_b_claim is not None:
                raise AssertionError("an attached task was incorrectly returned to the free queue")
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            async for _ in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "Continue the attached task.")],
                )
            ):
                pass
        else:
            raise ValueError(f"Unsupported task claim action: {action}")

        return {
            "reclaimed_task_ids": [task.id for task in reclaimed],
            "worker_b_claimed": worker_b_claim.id if worker_b_claim is not None else None,
        }
    finally:
        if session_store is not None:
            await session_store.close()
        await task_store.close()


async def _run_worker(config: dict[str, Any]) -> dict[str, Any]:
    scenario = RecoveryScenario(config["scenario"])
    action = RecoveryAction(config["action"])
    if action not in _SCENARIO_ACTIONS[scenario]:
        raise ValueError(f"Action {action!r} is not valid for scenario {scenario!r}")
    handlers = {
        RecoveryScenario.ORDINARY_TOOL: _run_ordinary_tool,
        RecoveryScenario.APPROVAL: _run_approval,
        RecoveryScenario.BACKGROUND_SUBAGENT: _run_background_subagent,
        RecoveryScenario.TASK_CLAIM: _run_task_claim,
    }
    return await handlers[scenario](config)


def _append_json_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = asyncio.run(_run_worker(config))
    _write_json_atomic(Path(config["result_path"]), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
