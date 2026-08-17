"""Tests for the generic ``run_task_worker`` durable-worker helper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core.task_invocation_fixtures import (
    stored_session_invocation,
    task_backed_session_invocation,
)
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EventType,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    PendingActionQuery,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskHandlerOutcome,
    TaskInvocationSnapshot,
    TaskQuery,
    TaskStatus,
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    run_task_worker,
)
from cayu.runtime import SessionStatus
from cayu.runtime.sessions import (
    SessionIdentity,
    run_request_with_task_invocation,
)
from cayu.runtime.tasks import TaskExecutionSource, task_create_with_runtime_invocation
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def _build(
    tmp_path: Path,
    *,
    secret_redactor: SecretRedactor | None = None,
) -> tuple[CayuApp, SQLiteTaskStore]:
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    app = CayuApp(task_store=store, secret_redactor=secret_redactor)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    return app, store


def test_run_task_worker_rejects_nan_poll_interval(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        return None

    async def scenario() -> None:
        with pytest.raises(ValueError, match="poll_interval_s"):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker-a",
                poll_interval_s=float("nan"),
                max_tasks=0,
            )

    asyncio.run(scenario())


async def _run_handler(app: CayuApp, task: Task, worker_id: str) -> None:
    async for _event in app.run(
        RunRequest(
            agent_name="worker-agent",
            session_id=f"sess-{task.id}",
            task_id=task.id,
            task_worker_id=worker_id,
            messages=[Message.text("user", "go")],
        )
    ):
        pass


class _PublishChangeTool(Tool):
    spec = ToolSpec(
        name="publish_change",
        description="Publish one reviewed change.",
        input_schema={
            "type": "object",
            "properties": {"change": {"type": "string"}},
            "required": ["change"],
        },
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"Published {args['change']}",
            structured={"session_id": ctx.session_id},
        )


def _register_approval_agent(app: CayuApp) -> None:
    app.register_agent(
        AgentSpec(name="worker-agent", model="scripted-model"),
        tools=[_PublishChangeTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )


def test_run_task_worker_claims_runs_and_completes_a_task(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> tuple[int, Task | None]:
        created = await store.create_task(
            TaskCreate(type="job", assigned_agent_name="worker-agent")
        )
        handled = await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return handled, await store.load_task(created.id)

    handled, task = asyncio.run(scenario())
    assert handled == 1
    assert task is not None
    assert task.status == "completed"


def test_run_task_worker_reconciles_failure_terminalization_acknowledgement_loss() -> None:
    class CommitThenRaiseTaskStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

    store = CommitThenRaiseTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def scenario() -> tuple[int, Task | None]:
        await store.create_task(TaskCreate(task_id="worker-failure", type="job"))

        async def fail_handler(_app: CayuApp, _task: Task, _worker_id: str):
            raise RuntimeError("handler failed")

        handled = await run_task_worker(
            app,
            store,
            fail_handler,
            worker_id="worker-a",
            max_tasks=1,
        )
        return handled, await store.load_task("worker-failure")

    handled, task = asyncio.run(scenario())

    assert handled == 1
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert store.terminalize_calls == 1


def test_run_task_worker_continues_after_handler_terminalizes_then_raises() -> None:
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def scenario() -> tuple[int, Task | None, Task | None]:
        await store.create_task(TaskCreate(task_id="worker-first", type="job"))
        await store.create_task(TaskCreate(task_id="worker-second", type="job"))

        async def terminalize_then_raise(_app: CayuApp, task: Task, worker_id: str):
            await store.complete_task(task.id, {"winner": "handler"}, worker_id=worker_id)
            if task.id == "worker-first":
                raise RuntimeError("handler raised after terminalizing")

        handled = await run_task_worker(
            app,
            store,
            terminalize_then_raise,
            worker_id="worker-a",
            max_tasks=2,
        )
        return (
            handled,
            await store.load_task("worker-first"),
            await store.load_task("worker-second"),
        )

    handled, first, second = asyncio.run(scenario())
    assert handled == 2
    assert first is not None
    assert first.status is TaskStatus.COMPLETED
    assert first.result == {"winner": "handler"}
    assert second is not None
    assert second.status is TaskStatus.COMPLETED
    assert second.result == {"winner": "handler"}


def test_run_task_worker_keeps_same_key_changed_intent_conflict_explicit() -> None:
    class ChangedIntentStore(InMemoryTaskStore):
        async def terminalize_task(self, request: TaskTerminalizationRequest):
            await super().terminalize_task(
                TaskTerminalizationRequest(
                    task_id=request.task_id,
                    worker_id=request.worker_id,
                    kind=TaskTerminalKind.COMPLETED,
                    result={"winner": "other-intent"},
                    idempotency_key=request.idempotency_key,
                )
            )
            return await super().terminalize_task(request)

    store = ChangedIntentStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def scenario() -> Task | None:
        await store.create_task(TaskCreate(task_id="worker-conflict", type="job"))

        async def fail_handler(_app: CayuApp, _task: Task, _worker_id: str):
            raise RuntimeError("handler failed")

        with pytest.raises(TaskTerminalizationConflict):
            await run_task_worker(
                app,
                store,
                fail_handler,
                worker_id="worker-a",
                max_tasks=1,
            )
        return await store.load_task("worker-conflict")

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.result == {"winner": "other-intent"}


def test_run_task_worker_returns_immediately_when_stopped(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> int:
        stop = asyncio.Event()
        stop.set()
        return await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            reclaim=False,
            stop=stop,
        )

    assert asyncio.run(scenario()) == 0


def test_run_task_worker_rejects_negative_max_tasks(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> None:
        await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            max_tasks=-1,
        )

    with pytest.raises(ValueError, match="max_tasks must be non-negative"):
        asyncio.run(scenario())


def test_run_task_worker_fails_task_when_handler_leaves_it_active(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def no_terminal_state(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        return None

    async def scenario() -> Task | None:
        created = await store.create_task(
            TaskCreate(type="job", assigned_agent_name="worker-agent")
        )
        handled = await run_task_worker(
            app,
            store,
            no_terminal_state,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        assert handled == 1
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Task handler returned without completing or failing the task.",
    }


@pytest.mark.parametrize(
    ("rejected_text", "error_code"),
    [
        ("handler failure\u0000with invalid text", "nul_character"),
        ("handler failure\ud800with invalid text", "unicode_surrogate"),
    ],
)
def test_run_task_worker_terminalizes_nonportable_handler_failures(
    tmp_path: Path,
    rejected_text: str,
    error_code: str,
) -> None:
    app, store = _build(tmp_path)

    async def fail_handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise RuntimeError(rejected_text)

    async def scenario():
        created = await store.create_task(TaskCreate(type="job"))
        handled = await run_task_worker(
            app,
            store,
            fail_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        task = await store.load_task(created.id)
        reclaimed = await store.reclaim_expired(query=TaskQuery(type="job"))
        second_claim = await store.claim_task(
            "w2",
            TaskQuery(type="job"),
            lease_seconds=300,
        )
        return handled, task, reclaimed, second_claim

    handled, task, reclaimed, second_claim = asyncio.run(scenario())

    assert handled == 1
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Task handler failed with a non-portable diagnostic.",
        "durable_value_error_code": error_code,
        "durable_value_error_path": "$",
    }
    assert reclaimed == []
    assert second_claim is None


def test_run_task_worker_redacts_handler_failure_before_durable_write(tmp_path: Path) -> None:
    secret = "task-worker-diagnostic-secret-canary"
    app, store = _build(tmp_path, secret_redactor=SecretRedactor(secret))

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise RuntimeError(f"upstream rejected credential {secret}")

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            fail,
            worker_id="worker-1",
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())

    assert task is not None
    assert task.status == "failed"
    assert task.error is not None
    assert secret not in str(task.error)
    assert REDACTED_SECRET in task.error["message"]


def test_run_task_worker_redacts_secret_bearing_exception_type_before_durable_write(
    tmp_path: Path,
) -> None:
    secret = "TaskWorkerSecretTypeCanary"
    app, store = _build(tmp_path, secret_redactor=SecretRedactor(secret))
    secret_error_type = type(f"Failure{secret}", (Exception,), {})

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise secret_error_type

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            fail,
            worker_id="worker-1",
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())

    assert task is not None
    assert task.error == {
        "error": f"Failure{REDACTED_SECRET}",
        "message": f"Failure{REDACTED_SECRET}: task handler failed",
    }


def test_run_task_worker_continues_after_handler_error_with_broken_stringification(
    tmp_path: Path,
) -> None:
    app, store = _build(tmp_path)

    class BrokenStringError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("stringification failed")

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise BrokenStringError

    async def scenario() -> tuple[int, list[Task | None]]:
        created = [
            await store.create_task(TaskCreate(type="job")),
            await store.create_task(TaskCreate(type="job")),
        ]
        handled = await run_task_worker(
            app,
            store,
            fail,
            worker_id="worker-1",
            query=TaskQuery(type="job"),
            max_tasks=2,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return handled, [await store.load_task(task.id) for task in created]

    handled, tasks = asyncio.run(scenario())

    assert handled == 2
    assert all(task is not None and task.status == "failed" for task in tasks)
    assert [task.error for task in tasks if task is not None] == [
        {
            "error": "BrokenStringError",
            "message": "BrokenStringError: task handler failed",
        },
        {
            "error": "BrokenStringError",
            "message": "BrokenStringError: task handler failed",
        },
    ]


def test_task_worker_cancellation_during_failure_write_does_not_retain_raw_error() -> None:
    secret = "task-worker-cancelled-publication-secret-canary"

    class BlockingFailureStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.failure_started = asyncio.Event()

        async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
            self.failure_started.set()
            await asyncio.Event().wait()
            return await super().terminalize_task(request)

    store = BlockingFailureStore()
    app = CayuApp(
        task_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise RuntimeError(f"upstream rejected credential {secret}")

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        await store.create_task(TaskCreate(type="job"))
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                store,
                fail,
                worker_id="worker-1",
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await store.failure_started.wait()
        worker_task.cancel("stop worker")
        cancelling = worker_task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await worker_task
        return exc_info.value, cancelling, worker_task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("stop worker",)
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


def test_run_task_worker_rejects_secret_bearing_worker_authority(tmp_path: Path) -> None:
    secret = "task-worker-authority-secret-canary"
    app, store = _build(tmp_path, secret_redactor=SecretRedactor(secret))

    with pytest.raises(ValueError, match="durable task authority"):
        asyncio.run(
            run_task_worker(
                app,
                store,
                _run_handler,
                worker_id=f"worker-{secret}",
                max_tasks=0,
            )
        )


def test_run_task_worker_hands_interrupted_session_to_reconstructed_control_plane(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "sessions.sqlite"
    task_path = tmp_path / "tasks.sqlite"
    first_app = CayuApp(
        session_store=SQLiteSessionStore(session_path),
        task_store=SQLiteTaskStore(task_path),
        enable_logging=False,
    )
    first_app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-change",
                        name="publish_change",
                        arguments={"change": "reviewed-release"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    _register_approval_agent(first_app)
    assert first_app.task_store is not None

    async def await_approval(
        app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in app.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-handoff",
                task_id=task.id,
                task_worker_id=worker_id,
                messages=[Message.text("user", "Publish the reviewed change.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, SessionStatus, Task | None, SessionStatus]:
        await first_app.create_task(
            TaskCreate(
                task_id="task-handoff",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        handled = await run_task_worker(
            first_app,
            first_app.task_store,
            await_approval,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        assert handled == 1

        handed_off_task = await first_app.task_store.load_task("task-handoff")
        handed_off_session = await first_app.session_store.load("session-handoff")
        pending = await first_app.session_store.query_pending_actions(
            PendingActionQuery(session_id="session-handoff")
        )
        assert handed_off_task is not None
        assert handed_off_task.status == "running"
        assert handed_off_task.session_id == "session-handoff"
        assert handed_off_task.worker_id is None
        assert handed_off_task.lease_expires_at is None
        assert handed_off_session is not None
        assert handed_off_session.status == SessionStatus.INTERRUPTED
        assert len(pending.actions) == 1
        assert pending.actions[0].approval_id is not None

        reconstructed = CayuApp(
            session_store=SQLiteSessionStore(session_path),
            task_store=SQLiteTaskStore(task_path),
            enable_logging=False,
        )
        reconstructed.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("The durable work item is complete."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        _register_approval_agent(reconstructed)
        approval_id = pending.actions[0].approval_id
        assert approval_id is not None
        approval_round_id = pending.actions[0].round_id
        approval_tool_call_id = pending.actions[0].tool_call_id
        assert approval_round_id is not None
        assert approval_tool_call_id is not None
        async for _event in reconstructed.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="session-handoff",
                approval_id=approval_id,
                tool_round_id=approval_round_id,
                tool_call_id=approval_tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

        assert reconstructed.task_store is not None
        completed_task = await reconstructed.task_store.load_task("task-handoff")
        completed_session = await reconstructed.session_store.load("session-handoff")
        assert completed_session is not None
        return (
            handed_off_task,
            handed_off_session.status,
            completed_task,
            completed_session.status,
        )

    handed_off_task, handed_off_status, completed_task, completed_status = asyncio.run(scenario())
    assert handed_off_task is not None
    assert handed_off_status == SessionStatus.INTERRUPTED
    assert completed_task is not None
    assert completed_task.status == "completed"
    assert completed_status == SessionStatus.COMPLETED


@pytest.mark.parametrize(
    "session_status",
    [
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
    ],
)
def test_run_task_worker_fails_handoff_when_session_is_not_interrupted(
    tmp_path: Path,
    session_status: SessionStatus,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def invalid_handoff(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-invalid-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-invalid-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await task_store.create_task(TaskCreate(type="job"))
        await session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="session-invalid-handoff",
                    task_id=created.id,
                    messages=[Message.text("user", "original")],
                ),
                TaskInvocationSnapshot(
                    id=created.id,
                    session_id=created.session_id,
                    invocation=created.invocation,
                ),
            ),
            identity=SessionIdentity(
                provider_name="scripted",
                model="scripted-model",
            ),
        )
        if session_status is not SessionStatus.PENDING:
            await session_store.update_status("session-invalid-handoff", session_status)
        await run_task_worker(
            app,
            task_store,
            invalid_handoff,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await task_store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": (
            "Task handler requested an interrupted-session handoff while session "
            f"session-invalid-handoff was {session_status}."
        ),
    }


def test_run_task_worker_fails_handoff_for_missing_attached_session(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def missing_session_handoff(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await store.attach_task(
            task.id,
            session_id="session-missing",
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                "session-missing",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            missing_session_handoff,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Attached session not found: session-missing.",
    }


def test_run_task_worker_preserves_terminal_task_before_handoff_cleanup(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def terminal_race(
        _app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> TaskHandlerOutcome:
        await store.complete_task(task.id, {"winner": "terminal-state"})
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            terminal_race,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "completed"
    assert task.result == {"winner": "terminal-state"}


def test_resume_completes_the_running_task_already_attached_to_the_session(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario():
        task = await task_store.create_task(TaskCreate(task_id="task-resume", type="job"))
        await session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="session-resume",
                    task_id=task.id,
                    messages=[Message.text("user", "original")],
                ),
                TaskInvocationSnapshot(
                    id=task.id,
                    session_id=task.session_id,
                    invocation=task.invocation,
                ),
            ),
            identity=profiled_session_identity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await session_store.update_status("session-resume", SessionStatus.INTERRUPTED)
        await task_store.start_task(
            task.id,
            session_id="session-resume",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-resume",
            ),
        )

        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="session-resume",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        return await task_store.load_task(task.id), events

    task, events = asyncio.run(scenario())

    assert task is not None
    assert task.status == "completed"
    assert task.session_id == "session-resume"
    assert [event.type for event in events][-3:] == [
        EventType.TASK_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]


def test_resume_rejects_multiple_running_tasks_attached_to_the_same_session(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    provider = ScriptedModelProvider([])
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario() -> None:
        await session_store.create(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-ambiguous-tasks",
                messages=[Message.text("user", "original")],
            ),
            identity=SessionIdentity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await session_store.update_status(
            "session-ambiguous-tasks",
            SessionStatus.INTERRUPTED,
        )
        for task_id in ("task-ambiguous-a", "task-ambiguous-b"):
            session_invocation = await stored_session_invocation(
                session_store,
                "session-ambiguous-tasks",
            )
            await task_store.create_task(
                task_create_with_runtime_invocation(
                    TaskCreate(
                        task_id=task_id,
                        type="job",
                        session_id="session-ambiguous-tasks",
                    ),
                    source=TaskExecutionSource.SDK_TASK,
                    session_invocation=session_invocation,
                )
            )
            await task_store.start_task(
                task_id,
                session_id="session-ambiguous-tasks",
                session_invocation=session_invocation,
            )

        async for _ in app.resume(
            ResumeRequest(
                session_id="session-ambiguous-tasks",
                messages=[Message.text("user", "continue")],
            )
        ):
            pass

    with pytest.raises(
        RuntimeError,
        match="Session has multiple running tasks attached: session-ambiguous-tasks",
    ):
        asyncio.run(scenario())

    session = asyncio.run(session_store.load("session-ambiguous-tasks"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert provider.requests == []
