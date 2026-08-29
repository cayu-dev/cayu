"""Tests for the generic ``run_task_worker`` durable-worker helper."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core.task_invocation_fixtures import (
    stored_session_invocation,
    task_backed_session_invocation,
)
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.runtime.task_worker as task_worker_module
from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    Event,
    EventQuery,
    EventType,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
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
    TaskInterruptedHandoffConflict,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
    TaskInvocationSnapshot,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCapabilityCeiling,
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


async def _seed_interrupted_worker_handoff(
    app: CayuApp,
    task_store: TaskStore,
    *,
    task_id: str,
    session_id: str,
) -> None:
    created = await task_store.create_task(TaskCreate(task_id=task_id, type="job"))
    await app.session_store.create(
        run_request_with_task_invocation(
            RunRequest(
                agent_name="worker-agent",
                session_id=session_id,
                task_id=task_id,
                messages=[Message.text("user", "pause")],
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
    await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)


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


def test_run_task_worker_rejects_incomplete_interrupted_handoff_capability() -> None:
    class IncompleteHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True
        load_interrupted_task_handoff_receipt = TaskStore.load_interrupted_task_handoff_receipt

    store = IncompleteHandoffStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise AssertionError("Capability validation must precede task handling.")

    async def scenario() -> None:
        with pytest.raises(NotImplementedError, match="complete idempotent|capability requires"):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker-incomplete-handoff",
                poll_interval_s=0.001,
                max_tasks=1,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_method", ["reclaim_expired", "claim_task"])
def test_ordinary_task_worker_preserves_store_failure_identity_and_traceback(
    failure_method: str,
) -> None:
    class OrdinaryFailureStore:
        supports_verified_work_contracts = False
        hold_claimed_work_contract_task = TaskStore.hold_claimed_work_contract_task

        def __init__(self) -> None:
            self.failure = KeyError(f"ordinary {failure_method} failure")

        async def reclaim_expired(self, *, query=None):
            del query
            if failure_method == "reclaim_expired":
                raise self.failure
            return []

        async def claim_task(self, worker_id, query, *, lease_seconds):
            del worker_id, query, lease_seconds
            if failure_method == "claim_task":
                raise self.failure
            return None

    async def scenario() -> tuple[BaseException, OrdinaryFailureStore]:
        store = OrdinaryFailureStore()
        app = CayuApp(enable_logging=False)

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            raise AssertionError("An ordinary store failure must precede task handling.")

        with pytest.raises(KeyError) as raised:
            await run_task_worker(
                app,
                store,  # type: ignore[arg-type]
                handler,
                worker_id="ordinary-worker-failure",
                poll_interval_s=0.001,
                max_tasks=1,
            )
        return raised.value, store

    failure, store = asyncio.run(scenario())

    assert failure is store.failure
    traceback_names: list[str] = []
    traceback = failure.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert failure_method in traceback_names


@pytest.mark.anyio
async def test_one_second_task_lease_heartbeats_after_one_third(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elapsed = 0.0
    stop = asyncio.Event()
    observed_heartbeats: list[tuple[str, str, int, float]] = []

    class ExpiringLeaseStore:
        async def heartbeat(
            self,
            task_id: str,
            worker_id: str,
            *,
            extend_seconds: int,
        ) -> None:
            assert elapsed < 1.0
            observed_heartbeats.append((task_id, worker_id, extend_seconds, elapsed))
            stop.set()

    async def advance_clock(seconds: float, wait_stop: asyncio.Event) -> bool:
        nonlocal elapsed
        elapsed += seconds
        return wait_stop.is_set()

    monkeypatch.setattr(task_worker_module, "_wait_or_stop", advance_clock)

    await task_worker_module._heartbeat_until(
        ExpiringLeaseStore(),  # type: ignore[arg-type]
        "task-1",
        "worker-a",
        1,
        stop,
    )

    assert observed_heartbeats == [("task-1", "worker-a", 1, pytest.approx(1 / 3))]


def test_handler_may_finish_cleanup_after_terminalizing_its_task() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="terminal-cleanup", type="job"))
        cleanup_finished = asyncio.Event()

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            await store.complete_task(task.id, {"ok": True}, worker_id=worker_id)
            await asyncio.sleep(0.5)
            cleanup_finished.set()

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker-a",
                lease_seconds=1,
                poll_interval_s=0.01,
                max_tasks=1,
            )
            == 1
        )
        assert cleanup_finished.is_set()
        terminal = await store.load_task("terminal-cleanup")
        assert terminal is not None
        assert terminal.status is TaskStatus.COMPLETED

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
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:task-worker:publish-change-tool",
            behavior_version="1",
            implementation_version="1",
        ),
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


def test_running_ordinary_task_cancellation_is_worker_terminalized(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> None:
        started = asyncio.Event()
        stopped = asyncio.Event()
        created = await store.create_task(TaskCreate(task_id="cancel-live", type="job"))

        async def blocking_handler(
            _app: CayuApp,
            _task: Task,
            _worker_id: str,
        ) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                blocking_handler,
                worker_id="worker-cancel-live",
                query=TaskQuery(type="job"),
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        requested = await store.cancel_task(
            created.id,
            {"reason": "operator cancelled live builder"},
        )
        assert requested.status is TaskStatus.CLAIMED
        assert requested.worker_id == "worker-cancel-live"
        assert requested.lease_expires_at is not None
        assert requested.status_reason == "cancellation_requested"
        assert requested.status_payload is not None

        assert await asyncio.wait_for(worker, timeout=3) == 1
        assert stopped.is_set()

        terminal = await store.load_task(created.id)
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert terminal.worker_id is None
        assert terminal.lease_expires_at is None
        assert terminal.error == {"reason": "operator cancelled live builder"}
        key = requested.status_payload["terminalization_idempotency_key"]
        receipt = await store.load_task_terminalization_receipt(created.id, key)
        assert receipt is not None
        assert receipt.kind is TaskTerminalKind.CANCELLED
        assert receipt.task == terminal

    asyncio.run(scenario())


def test_run_task_worker_reconciles_failure_terminalization_acknowledgement_loss() -> None:
    class CommitThenRaiseTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

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
        verified_work_mutations_are_cancellation_quiescent = True

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
        verified_work_mutations_are_cancellation_quiescent = True

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

    async def scenario() -> tuple[
        Task | None,
        SessionStatus,
        Task | None,
        SessionStatus,
        list[tuple[str, str]],
    ]:
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
        handoff_events = await first_app.session_store.query_events(
            EventQuery(
                session_id="session-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )

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
            [
                (
                    str(record.event.payload["task_id"]),
                    str(record.event.payload["handoff_status"]),
                )
                for record in handoff_events
            ],
        )

    (
        handed_off_task,
        handed_off_status,
        completed_task,
        completed_status,
        handoff_evidence,
    ) = asyncio.run(scenario())
    assert handed_off_task is not None
    assert handed_off_status == SessionStatus.INTERRUPTED
    assert completed_task is not None
    assert completed_task.status == "completed"
    assert completed_status == SessionStatus.COMPLETED
    assert handoff_evidence == [
        ("task-handoff", "released"),
    ]


def test_terminal_peer_winner_during_handoff_does_not_stop_worker() -> None:
    class TerminalPeerStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            if request.task_id == "task-handoff-peer-winner":
                await self.terminalize_task(
                    TaskTerminalizationRequest(
                        task_id=request.task_id,
                        worker_id=request.worker_id,
                        kind=TaskTerminalKind.COMPLETED,
                        result={"winner": "peer"},
                        idempotency_key="terminal-peer-winner",
                    )
                )
            return await super().release_interrupted_task_worker(request)

    task_store = TerminalPeerStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome | None:
        if task.id == "task-handoff-peer-winner":
            await task_store.attach_task(
                task.id,
                session_id="session-handoff-peer-winner",
                session_invocation=await stored_session_invocation(
                    app.session_store,
                    "session-handoff-peer-winner",
                ),
                worker_id=worker_id,
            )
            return TaskHandlerOutcome.SESSION_INTERRUPTED
        await task_store.complete_task(
            task.id,
            {"handled": True},
            worker_id=worker_id,
        )
        return None

    async def scenario() -> tuple[int, Task | None, Task | None]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-handoff-peer-winner",
            session_id="session-handoff-peer-winner",
        )
        await task_store.create_task(TaskCreate(task_id="task-after-peer-winner", type="job"))
        handled = await run_task_worker(
            app,
            task_store,
            handler,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=2,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return (
            handled,
            await task_store.load_task("task-handoff-peer-winner"),
            await task_store.load_task("task-after-peer-winner"),
        )

    handled, peer_winner, following = asyncio.run(scenario())
    assert handled == 2
    assert peer_winner is not None
    assert peer_winner.status is TaskStatus.COMPLETED
    assert peer_winner.result == {"winner": "peer"}
    assert following is not None
    assert following.status is TaskStatus.COMPLETED
    assert following.result == {"handled": True}


@pytest.mark.parametrize(
    ("failure_point", "expected_calls", "expected_statuses"),
    [
        ("before_commit", 2, ["pending", "recovered"]),
        ("after_commit", 1, ["pending", "recovered"]),
    ],
)
def test_interrupted_handoff_retries_or_reads_back_without_failing_task(
    tmp_path: Path,
    failure_point: str,
    expected_calls: int,
    expected_statuses: list[str],
) -> None:
    class FaultingHandoffStore(SQLiteTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.release_calls = 0

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_calls += 1
            if failure_point == "before_commit" and self.release_calls == 1:
                raise RuntimeError("transient handoff write failure")
            receipt = await super().release_interrupted_task_worker(request)
            if failure_point == "after_commit" and self.release_calls == 1:
                raise RuntimeError("handoff acknowledgement lost")
            return receipt

    session_store = SQLiteSessionStore(tmp_path / f"sessions-{failure_point}.sqlite")
    task_store = FaultingHandoffStore(tmp_path / f"tasks-{failure_point}.sqlite")
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-faulted-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-faulted-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, list[str]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-faulted-handoff",
            session_id="session-faulted-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        task = await task_store.load_task("task-faulted-handoff")
        events = await session_store.query_events(
            EventQuery(
                session_id="session-faulted-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return task, [str(record.event.payload["handoff_status"]) for record in events]

    task, statuses = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-faulted-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error is None
    assert task_store.release_calls == expected_calls
    assert statuses == expected_statuses


@pytest.mark.parametrize(
    ("failure_point", "expected_statuses"),
    [
        ("before_commit", []),
        ("after_commit", ["released"]),
    ],
)
def test_interrupted_handoff_event_failure_never_owns_task_release(
    failure_point: str,
    expected_statuses: list[str],
) -> None:
    class FaultingEventStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type != EventType.TASK_INTERRUPTED_HANDOFF:
                await super().append_event(session_id, event)
                return
            if failure_point == "before_commit":
                raise RuntimeError("handoff event unavailable before commit")
            await super().append_event(session_id, event)
            raise RuntimeError("handoff event acknowledgement lost")

    session_store = FaultingEventStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-event-failure-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-event-failure-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, list[str]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-event-failure-handoff",
            session_id="session-event-failure-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        events = await session_store.query_events(
            EventQuery(
                session_id="session-event-failure-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return (
            await task_store.load_task("task-event-failure-handoff"),
            [str(record.event.payload["handoff_status"]) for record in events],
        )

    task, statuses = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-event-failure-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error is None
    assert statuses == expected_statuses


def test_interrupted_handoff_releases_before_slow_event_publication() -> None:
    class SlowEventStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if (
                event.type is EventType.TASK_INTERRUPTED_HANDOFF
                and event.payload.get("handoff_status") == "released"
            ):
                released = await task_store.load_task("task-slow-event-handoff")
                assert released is not None
                assert released.worker_id is None
                assert released.lease_expires_at is None
                await asyncio.sleep(1.1)
            await super().append_event(session_id, event)

    session_store = SlowEventStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-slow-event-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-slow-event-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-slow-event-handoff",
            session_id="session-slow-event-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        return await task_store.load_task("task-slow-event-handoff")

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id is None
    assert task.lease_expires_at is None


def test_interrupted_handoff_does_not_attest_secret_bearing_task_id() -> None:
    secret = "handoff-task-id-secret"
    task_id = f"task-{secret}-value"
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-secret-task-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-secret-task-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, list[object]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id=task_id,
            session_id="session-secret-task-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        events = await app.session_store.query_events(
            EventQuery(
                session_id="session-secret-task-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return await task_store.load_task(task_id), events

    task, events = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert events == []


def test_interrupted_handoff_rejects_malformed_store_receipt_without_retry() -> None:
    class MalformedReceiptStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_calls = 0

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            self.release_calls += 1
            return object()  # type: ignore[return-value]

    task_store = MalformedReceiptStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-malformed-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-malformed-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-malformed-handoff",
            session_id="session-malformed-handoff",
        )
        with pytest.raises(TaskInterruptedHandoffConflict, match="malformed receipt"):
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        return await task_store.load_task("task-malformed-handoff")

    task = asyncio.run(scenario())
    assert task_store.release_calls == 1
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id == "worker-a"
    assert task.error is None


def test_interrupted_handoff_exhaustion_recovers_and_resumes_original_task(
    tmp_path: Path,
) -> None:
    class UnavailableHandoffStore(SQLiteTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            raise RuntimeError("handoff store unavailable")

    session_path = tmp_path / "sessions-recovery.sqlite"
    task_path = tmp_path / "tasks-recovery.sqlite"
    session_store = SQLiteSessionStore(session_path)
    failing_store = UnavailableHandoffStore(task_path)
    first_app = CayuApp(
        session_store=session_store,
        task_store=failing_store,
        enable_logging=False,
    )
    first_app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-recovered-change",
                        name="publish_change",
                        arguments={"change": "recovered-release"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    _register_approval_agent(first_app)

    async def handoff_handler(
        app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in app.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-recovery-handoff",
                task_id=task.id,
                task_worker_id=worker_id,
                messages=[Message.text("user", "Publish after durable recovery.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def first_process() -> tuple[Task, str, str, str]:
        await first_app.create_task(
            TaskCreate(
                task_id="task-recovery-handoff",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        with pytest.raises(RuntimeError, match="handoff store unavailable"):
            await run_task_worker(
                first_app,
                failing_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        task = await failing_store.load_task("task-recovery-handoff")
        assert task is not None
        pending = await session_store.query_pending_actions(
            PendingActionQuery(session_id="session-recovery-handoff")
        )
        assert len(pending.actions) == 1
        approval = pending.actions[0]
        assert approval.approval_id is not None
        assert approval.round_id is not None
        assert approval.tool_call_id is not None
        await failing_store.close()
        return (
            task,
            approval.approval_id,
            approval.round_id,
            approval.tool_call_id,
        )

    stranded, approval_id, approval_round_id, approval_tool_call_id = asyncio.run(first_process())
    assert stranded.status is TaskStatus.RUNNING
    assert stranded.session_id == "session-recovery-handoff"
    assert stranded.worker_id == "worker-a"
    assert stranded.lease_expires_at is not None
    assert stranded.error is None

    async def second_process() -> tuple[
        Task | None,
        SessionStatus,
        list[str],
    ]:
        remaining = max(
            (stranded.lease_expires_at - datetime.now(UTC)).total_seconds(),
            0,
        )
        await asyncio.sleep(remaining + 0.05)
        recovered_store = SQLiteTaskStore(task_path)
        recovered_sessions = SQLiteSessionStore(session_path)
        recovered_app = CayuApp(
            session_store=recovered_sessions,
            task_store=recovered_store,
            enable_logging=False,
        )
        recovered_app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("The recovered task is complete."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        _register_approval_agent(recovered_app)
        stop = asyncio.Event()

        async def stop_after_recovery() -> None:
            await asyncio.sleep(0.05)
            stop.set()

        async def unexpected_handler(
            _app: CayuApp,
            _task: Task,
            _worker_id: str,
        ) -> None:
            raise AssertionError("Recovered attached work must not re-enter the fresh queue.")

        await asyncio.gather(
            run_task_worker(
                recovered_app,
                recovered_store,
                unexpected_handler,
                worker_id="worker-b",
                query=TaskQuery(type="job"),
                poll_interval_s=0.01,
                reclaim=False,
                stop=stop,
                max_tasks=1,
            ),
            stop_after_recovery(),
        )
        recovered = await recovered_store.load_task("task-recovery-handoff")
        assert recovered is not None
        assert recovered.status is TaskStatus.RUNNING
        assert recovered.session_id == "session-recovery-handoff"
        assert recovered.worker_id is None
        assert recovered.lease_expires_at is None

        async for _event in recovered_app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="session-recovery-handoff",
                approval_id=approval_id,
                tool_round_id=approval_round_id,
                tool_call_id=approval_tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

        task = await recovered_store.load_task("task-recovery-handoff")
        session = await recovered_sessions.load("session-recovery-handoff")
        assert session is not None
        events = await recovered_sessions.query_events(
            EventQuery(
                session_id="session-recovery-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        await recovered_store.close()
        return (
            task,
            session.status,
            [str(record.event.payload["handoff_status"]) for record in events],
        )

    completed, session_status, statuses = asyncio.run(second_process())
    assert completed is not None
    assert completed.status is TaskStatus.COMPLETED
    assert completed.session_id == "session-recovery-handoff"
    assert completed.worker_id is None
    assert completed.lease_expires_at is None
    assert session_status is SessionStatus.COMPLETED
    assert statuses == [
        "pending",
        "retrying",
        "retrying",
        "recovery_required",
        "recovered",
    ]


def test_interrupted_handoff_recovery_pages_past_ineligible_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_worker_module, "_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE", 2)
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def attach_expiring_task(
        *,
        task_id: str,
        session_id: str,
        interrupted: bool,
    ) -> None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id=task_id,
            session_id=session_id,
        )
        if not interrupted:
            await app.session_store.update_status(session_id, SessionStatus.RUNNING)
        claimed = await task_store.claim_task("expired-worker", lease_seconds=1)
        assert claimed is not None
        assert claimed.id == task_id
        await task_store.attach_task(
            task_id,
            session_id=session_id,
            session_invocation=await stored_session_invocation(
                app.session_store,
                session_id,
            ),
            worker_id="expired-worker",
        )

    async def handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> None:
        await task_store.complete_task(task.id, worker_id=worker_id, result={"ok": True})

    async def scenario() -> tuple[Task | None, list[Task | None]]:
        await attach_expiring_task(
            task_id="stale-ineligible-a",
            session_id="session-ineligible-a",
            interrupted=False,
        )
        await attach_expiring_task(
            task_id="stale-ineligible-b",
            session_id="session-ineligible-b",
            interrupted=False,
        )
        await attach_expiring_task(
            task_id="stale-interrupted",
            session_id="session-interrupted",
            interrupted=True,
        )
        await task_store.create_task(TaskCreate(task_id="trigger-a", type="trigger"))
        await asyncio.sleep(1.05)

        handled = await run_task_worker(
            app,
            task_store,
            handler,
            worker_id="recovery-worker",
            query=TaskQuery(type="trigger"),
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        assert handled == 1
        return (
            await task_store.load_task("stale-interrupted"),
            [
                await task_store.load_task("stale-ineligible-a"),
                await task_store.load_task("stale-ineligible-b"),
            ],
        )

    recovered, ineligible = asyncio.run(scenario())
    assert recovered is not None
    assert recovered.worker_id is None
    assert recovered.lease_expires_at is None
    assert all(task is not None and task.worker_id == "expired-worker" for task in ineligible)


def test_interrupted_handoff_recovery_skips_new_cancellation_request() -> None:
    class CancellationWinningStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def recover_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            await self.cancel_task(request.task_id, {"code": "operator_cancelled"})
            return await super().recover_interrupted_task_worker(request)

    task_store = CancellationWinningStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def scenario() -> tuple[Task | None, Task | None]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancellation-winner",
            session_id="session-cancellation-winner",
        )
        claimed = await task_store.claim_task("expired-worker", lease_seconds=1)
        assert claimed is not None
        await task_store.attach_task(
            claimed.id,
            session_id="session-cancellation-winner",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancellation-winner",
            ),
            worker_id="expired-worker",
        )
        await task_store.create_task(TaskCreate(task_id="fresh-after-cancel", type="fresh"))
        await asyncio.sleep(1.05)

        async def handler(
            _app: CayuApp,
            task: Task,
            worker_id: str,
        ) -> None:
            await task_store.complete_task(task.id, {"ok": True}, worker_id=worker_id)

        assert (
            await run_task_worker(
                app,
                task_store,
                handler,
                worker_id="recovery-worker",
                query=TaskQuery(type="fresh"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        return (
            await task_store.load_task("task-cancellation-winner"),
            await task_store.load_task("fresh-after-cancel"),
        )

    cancelled, fresh = asyncio.run(scenario())
    assert cancelled is not None
    assert cancelled.status is TaskStatus.RUNNING
    assert cancelled.status_reason == "cancellation_requested"
    assert fresh is not None
    assert fresh.status is TaskStatus.COMPLETED


def test_operator_cancellation_winning_live_handoff_is_terminalized() -> None:
    class CancellationWinningStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.cancellation_key: str | None = None

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            cancellation = await self.cancel_task(
                request.task_id,
                {"code": "operator_cancelled_during_handoff"},
            )
            assert cancellation.status_payload is not None
            key = cancellation.status_payload["terminalization_idempotency_key"]
            assert isinstance(key, str)
            self.cancellation_key = key
            return await super().release_interrupted_task_worker(request)

    task_store = CancellationWinningStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-live-cancellation-winner",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-live-cancellation-winner",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[int, Task | None, TaskTerminalizationReceipt | None]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-live-cancellation-winner",
            session_id="session-live-cancellation-winner",
        )
        handled = await run_task_worker(
            app,
            task_store,
            handoff_handler,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        terminal = await task_store.load_task("task-live-cancellation-winner")
        assert terminal is not None
        assert terminal.status_payload is None
        assert task_store.cancellation_key is not None
        return (
            handled,
            terminal,
            await task_store.load_task_terminalization_receipt(
                terminal.id,
                task_store.cancellation_key,
            ),
        )

    handled, task, receipt = asyncio.run(scenario())
    assert handled == 1
    assert task is not None
    assert task.status is TaskStatus.CANCELLED
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error == {"code": "operator_cancelled_during_handoff"}
    assert receipt is not None
    assert receipt.kind is TaskTerminalKind.CANCELLED
    assert receipt.task == task


def test_interrupted_handoff_recovery_retains_bounded_cursor_between_idle_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_worker_module, "_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE", 2)

    class CountingRecoveryStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.recovery_list_calls = 0

        async def list_expired_interrupted_task_handoff_candidates(
            self,
            *,
            after: tuple[datetime, str] | None = None,
            limit: int = 100,
        ) -> list[Task]:
            self.recovery_list_calls += 1
            return await super().list_expired_interrupted_task_handoff_candidates(
                after=after,
                limit=limit,
            )

    task_store = CountingRecoveryStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def attach_ineligible(task_id: str) -> None:
        session_id = f"session-{task_id}"
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id=task_id,
            session_id=session_id,
        )
        await app.session_store.update_status(session_id, SessionStatus.RUNNING)
        claimed = await task_store.claim_task("expired-worker", lease_seconds=1)
        assert claimed is not None
        await task_store.attach_task(
            task_id,
            session_id=session_id,
            session_invocation=await stored_session_invocation(
                app.session_store,
                session_id,
            ),
            worker_id="expired-worker",
        )

    async def scenario() -> None:
        await attach_ineligible("stale-bounded-a")
        await attach_ineligible("stale-bounded-b")
        await asyncio.sleep(1.05)
        stop = asyncio.Event()

        async def stop_worker() -> None:
            await asyncio.sleep(0.1)
            stop.set()

        async def unexpected_handler(
            _app: CayuApp,
            _task: Task,
            _worker_id: str,
        ) -> None:
            raise AssertionError("No fresh task should be claimed.")

        await asyncio.gather(
            run_task_worker(
                app,
                task_store,
                unexpected_handler,
                worker_id="recovery-worker",
                query=TaskQuery(type="fresh"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
                stop=stop,
            ),
            stop_worker(),
        )

    asyncio.run(scenario())
    assert task_store.recovery_list_calls == 2


def test_interrupted_handoff_cancellation_waits_for_dispatched_release() -> None:
    class BlockingHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_started.set()
            await self.allow_release.wait()
            return await super().release_interrupted_task_worker(request)

    task_store = BlockingHandoffStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-cancelled-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancelled-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, int, bool, tuple[object, ...], list[str]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancelled-handoff",
            session_id="session-cancelled-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await task_store.release_started.wait()
        worker_task.cancel("stop interrupted handoff")
        await asyncio.sleep(0)
        assert not worker_task.done()
        task_store.allow_release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        task = await task_store.load_task("task-cancelled-handoff")
        events = await app.session_store.query_events(
            EventQuery(
                session_id="session-cancelled-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return (
            task,
            worker_task.cancelling(),
            worker_task.cancelled(),
            raised.value.args,
            [str(record.event.payload["handoff_status"]) for record in events],
        )

    task, cancelling, cancelled, cancellation_args, statuses = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-cancelled-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error is None
    assert cancelling == 1
    assert cancelled is True
    assert cancellation_args == ("stop interrupted handoff",)
    assert statuses == []


def test_interrupted_handoff_pending_cancellation_owns_dispatched_release() -> None:
    class BlockingHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_started.set()
            await self.allow_release.wait()
            return await super().release_interrupted_task_worker(request)

    task_store = BlockingHandoffStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def scenario() -> tuple[Task | None, int, bool, tuple[object, ...]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-pending-cancelled-handoff",
            session_id="session-pending-cancelled-handoff",
        )
        claimed = await task_store.claim_task("worker-a", TaskQuery(type="job"))
        assert claimed is not None
        attached = await task_store.attach_task(
            claimed.id,
            session_id="session-pending-cancelled-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-pending-cancelled-handoff",
            ),
            worker_id="worker-a",
        )
        request = task_worker_module.interrupted_task_handoff_request(
            attached,
            session_run_epoch=1,
        )

        async def settle_with_pending_cancellation() -> TaskInterruptedHandoffReceipt:
            owner = asyncio.current_task()
            assert owner is not None
            owner.cancel("cancel before interrupted handoff settlement")
            return await task_worker_module._settle_interrupted_task_handoff_once(
                app,
                task_store,
                request,
                recover_expired=False,
            )

        settlement = asyncio.create_task(settle_with_pending_cancellation())
        await task_store.release_started.wait()
        await asyncio.sleep(0)
        assert not settlement.done()
        task_store.allow_release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await settlement
        return (
            await task_store.load_task(attached.id),
            settlement.cancelling(),
            settlement.cancelled(),
            raised.value.args,
        )

    task, cancelling, cancelled, cancellation_args = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-pending-cancelled-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert cancelling == 1
    assert cancelled is True
    assert cancellation_args == ("cancel before interrupted handoff settlement",)


def test_interrupted_handoff_cancellation_sanitizes_concurrent_store_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "handoff-store-secret-canary"

    class BlockingFailureStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_failure = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            self.release_started.set()
            await self.allow_failure.wait()
            raise RuntimeError(f"store cleanup exposed {secret}")

    task_store = BlockingFailureStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-cancelled-failing-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancelled-failing-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[asyncio.CancelledError, Task | None, int, bool]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancelled-failing-handoff",
            session_id="session-cancelled-failing-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await task_store.release_started.wait()
        worker_task.cancel("cancel handoff with failing cleanup")
        await asyncio.sleep(0)
        assert not worker_task.done()
        task_store.allow_failure.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        return (
            raised.value,
            await task_store.load_task("task-cancelled-failing-handoff"),
            worker_task.cancelling(),
            worker_task.cancelled(),
        )

    with caplog.at_level("DEBUG"):
        cancellation, task, cancelling, cancelled = asyncio.run(scenario())
    rendered: list[str] = []
    pending: list[BaseException] = [cancellation]
    seen: set[int] = set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        rendered.extend((str(failure), repr(failure)))
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)

    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id == "worker-a"
    assert task.error is None
    assert cancelling == 1
    assert cancelled is True
    assert REDACTED_SECRET in " ".join(rendered)
    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            *rendered,
            captured.out,
            captured.err,
            *[record.getMessage() for record in caplog.records],
            *[str(warning.message) for warning in recwarn],
        ]
    )
    assert secret not in diagnostics


def test_interrupted_handoff_event_cancellation_drops_store_failure_context(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "handoff-event-context-secret-canary"

    class FailingHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            raise RuntimeError(f"store handoff exposed {secret}")

    class BlockingEventStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.publication_started = asyncio.Event()

        async def append_event(self, session_id: str, event: Event) -> None:
            if (
                event.type is EventType.TASK_INTERRUPTED_HANDOFF
                and event.payload.get("handoff_status") == "pending"
            ):
                self.publication_started.set()
                await asyncio.Event().wait()
            await super().append_event(session_id, event)

    task_store = FailingHandoffStore()
    session_store = BlockingEventStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-event-cancelled-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-event-cancelled-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-event-cancelled-handoff",
            session_id="session-event-cancelled-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await session_store.publication_started.wait()
        worker_task.cancel("cancel pending handoff event")
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        return raised.value, worker_task.cancelling(), worker_task.cancelled()

    with caplog.at_level("DEBUG"):
        cancellation, cancelling, cancelled = asyncio.run(scenario())
    rendered: list[str] = []
    pending: list[BaseException] = [cancellation]
    seen: set[int] = set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        rendered.extend((str(failure), repr(failure)))
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("cancel pending handoff event",)
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            *rendered,
            captured.out,
            captured.err,
            *[record.getMessage() for record in caplog.records],
            *[str(warning.message) for warning in recwarn],
        ]
    )
    assert secret not in diagnostics


def test_interrupted_handoff_cancellation_validates_commit_receipt_before_redelivery(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "handoff-malformed-receipt-secret-canary"

    class SecretBearingValue:
        def __eq__(self, _other: object) -> bool:
            raise RuntimeError(secret)

        def __repr__(self) -> str:
            return secret

    class BlockingMalformedReceiptStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_started.set()
            await self.allow_release.wait()
            receipt = await super().release_interrupted_task_worker(request)
            object.__setattr__(receipt, "request", SecretBearingValue())
            return receipt

    task_store = BlockingMalformedReceiptStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-cancelled-malformed-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancelled-malformed-handoff",
            ),
            worker_id=worker_id,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[asyncio.CancelledError, Task | None, int, bool]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancelled-malformed-handoff",
            session_id="session-cancelled-malformed-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await task_store.release_started.wait()
        worker_task.cancel("cancel handoff with malformed receipt")
        await asyncio.sleep(0)
        assert not worker_task.done()
        task_store.allow_release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        return (
            raised.value,
            await task_store.load_task("task-cancelled-malformed-handoff"),
            worker_task.cancelling(),
            worker_task.cancelled(),
        )

    with caplog.at_level("DEBUG"):
        cancellation, task, cancelling, cancelled = asyncio.run(scenario())
    rendered: list[str] = []
    pending: list[BaseException] = [cancellation]
    seen: set[int] = set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        rendered.extend((str(failure), repr(failure)))
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)

    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert cancelling == 1
    assert cancelled is True
    assert "malformed receipt evidence" in " ".join(rendered)
    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            *rendered,
            captured.out,
            captured.err,
            *[record.getMessage() for record in caplog.records],
            *[str(warning.message) for warning in recwarn],
        ]
    )
    assert secret not in diagnostics


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
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
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
