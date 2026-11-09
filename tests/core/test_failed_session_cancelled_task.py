"""A session failure settles a task whose own worker already requested cancellation.

A worker cancelled while its handler runs durably requests the task's
cancellation and waits for the handler. When that handler's session then fails,
the request wins task terminalization: the task is cancelled through it, and the
session still ends FAILED with its own failure evidence, live or after recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import BeforeStopContext, BeforeStopDecision, LoopPolicy
from cayu.runtime._continuation_task_failure import runtime_task_failure_terminalization_request
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._session_finalization import SessionFinalization
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import InMemorySessionStore
from cayu.storage.tasks_sqlite import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus
from cayu.tasks import worker as worker_module
from cayu.tasks._cancellation import _task_cancellation_terminalization_request
from cayu.tasks._terminalization import (
    _terminalize_claimed_task,
    _terminalize_claimed_task_or_detect_peer_winner,
    _terminalize_claimed_task_or_requested_cancellation,
)
from cayu.tasks.cancellation import _task_cancellation_requested
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskClaimLost
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)
from cayu.tasks.worker import run_task_worker

SESSION_ID = "failed-session-cancelled-task"
WORKER_ID = "worker-a"
TASK_TYPE = "answer"


class _Clock:
    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now


class _FailingStopPolicy(LoopPolicy):
    """Waits in ``before_stop`` until released, then fails the run."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:failed-session-cancelled-task",
            behavior_version="1",
            implementation_version="1",
        )

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        self.entered.set()
        await self.release.wait()
        raise RuntimeError("stop policy failed")


def _app(store: InMemorySessionStore, tasks: TaskStore) -> CayuApp:
    app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted"))
    return app


async def _fail_after_worker_cancellation(app: CayuApp, tasks: TaskStore, task_id: str) -> None:
    """Cancel the worker mid-run, then fail the run its handler keeps running."""

    policy = _FailingStopPolicy()

    async def handler(app: CayuApp, task, worker_id: str) -> None:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=SESSION_ID,
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "answer")],
                loop_policies=(policy,),
            )
        ):
            pass

    worker = asyncio.create_task(
        run_task_worker(
            app,
            tasks,
            handler,
            worker_id=WORKER_ID,
            query=TaskQuery(type=TASK_TYPE),
            max_tasks=1,
            poll_interval_s=0.005,
            reclaim=False,
            recover_interrupted_handoffs=False,
        )
    )
    await asyncio.wait_for(policy.entered.wait(), timeout=10)
    worker.cancel()
    for _ in range(1000):
        task = await tasks.load_task(task_id)
        if task is not None and _task_cancellation_requested(task):
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError("the worker did not request the task's cancellation")
    policy.release.set()
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(worker, timeout=10)


async def _assert_failed_session_with_cancelled_task(
    store: InMemorySessionStore, tasks: TaskStore, task_id: str
) -> None:
    session = await store.load(SESSION_ID)
    assert session is not None and session.status is SessionStatus.FAILED
    task = await tasks.load_task(task_id)
    assert task is not None and task.status is TaskStatus.CANCELLED
    events = await store.load_events(SESSION_ID)
    types = [event.type for event in events]
    assert EventType.SESSION_FAILED in types
    assert EventType.TASK_CANCELLED in types
    assert EventType.TASK_FAILED not in types
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error"] == "stop policy failed"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_session_failure_cancels_a_task_through_its_workers_request(
    tmp_path: Path, backend: str
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = InMemorySessionStore(ownership_clock=clock)
        tasks = (
            InMemoryTaskStore(clock=clock, ownership_clock=clock)
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite", clock=clock, ownership_clock=clock)
        )
        task = await tasks.create_task(TaskCreate(type=TASK_TYPE))
        app = _app(store, tasks)
        try:
            await _fail_after_worker_cancellation(app, tasks, task.id)
            await _assert_failed_session_with_cancelled_task(store, tasks, task.id)
        finally:
            await app.aclose()
            if backend == "sqlite":
                await tasks.close()

    asyncio.run(scenario())


def test_recovery_finishes_a_failed_session_whose_task_its_request_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = InMemorySessionStore(ownership_clock=clock)
        tasks = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        task = await tasks.create_task(TaskCreate(type=TASK_TYPE))
        original_fail_task = SessionFinalization.fail_task

        async def cancel_then_lose_acknowledgement(self, **kwargs):
            await original_fail_task(self, **kwargs)
            raise ConnectionError("task store acknowledgement lost")

        # The task is cancelled through its request, but the failing run never
        # learns it, so its session stays unsettled for recovery.
        monkeypatch.setattr(SessionFinalization, "fail_task", cancel_then_lose_acknowledgement)
        app = _app(store, tasks)
        try:
            await _fail_after_worker_cancellation(app, tasks, task.id)
        finally:
            await app.aclose()
        monkeypatch.setattr(SessionFinalization, "fail_task", original_fail_task)
        session = await store.load(SESSION_ID)
        assert session is not None and session.status is SessionStatus.RUNNING
        assert (await tasks.load_task(task.id)).status is TaskStatus.CANCELLED

        clock.now += timedelta(days=1)
        recovery = _app(store, tasks)
        try:
            await recovery.recover_abandoned_sessions(inactive_for_seconds=0)
            await _assert_failed_session_with_cancelled_task(store, tasks, task.id)
        finally:
            await recovery.aclose()

    asyncio.run(scenario())


def test_a_repeated_failure_returns_the_task_its_request_cancelled() -> None:
    async def scenario() -> None:
        tasks = InMemoryTaskStore()
        await tasks.create_task(TaskCreate(type=TASK_TYPE))
        task = await tasks.claim_task(WORKER_ID, TaskQuery(type=TASK_TYPE))
        assert task is not None
        task = await tasks.request_claimed_task_cancellation(
            task.id, WORKER_ID, task.lease_expires_at
        )

        def failure(**changes) -> TaskTerminalizationRequest:
            return runtime_task_failure_terminalization_request(
                task_id=task.id,
                task_worker_id=WORKER_ID,
                task_lease_expires_at=task.lease_expires_at,
                task_handoff_id=None,
                session_id=SESSION_ID,
                error={"error": "run failed"},
            ).model_copy(update=changes)

        # Another lease than the failing request's is no claim of this request.
        with pytest.raises(TaskClaimLost):
            await _terminalize_claimed_task_or_requested_cancellation(
                tasks, failure(lease_expires_at=task.lease_expires_at + timedelta(seconds=1))
            )
        first = await _terminalize_claimed_task_or_requested_cancellation(tasks, failure())
        again = await _terminalize_claimed_task_or_requested_cancellation(tasks, failure())
        assert first.status is TaskStatus.CANCELLED
        assert again == first
        with pytest.raises(TaskTerminalizationConflict):
            await _terminalize_claimed_task_or_requested_cancellation(
                tasks,
                failure(worker_id="worker-b"),
            )

    asyncio.run(scenario())


def test_recovery_leaves_a_task_to_its_request_and_never_reports_it_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = InMemorySessionStore(ownership_clock=clock)
        tasks = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        task = await tasks.create_task(TaskCreate(type=TASK_TYPE))

        async def lose_task_failure(self, **kwargs):
            raise ConnectionError("task store unavailable")

        async def lose_worker_cancellation(*args, **kwargs):
            raise ConnectionError("worker lost")

        # Neither the run nor its worker settles the task, so it stays running
        # with its worker's cancellation request behind a committed failure.
        monkeypatch.setattr(SessionFinalization, "fail_task", lose_task_failure)
        monkeypatch.setattr(
            worker_module,
            "_terminalize_claimed_task_or_detect_peer_winner",
            lose_worker_cancellation,
        )
        app = _app(store, tasks)
        try:
            await _fail_after_worker_cancellation(app, tasks, task.id)
        finally:
            await app.aclose()
        monkeypatch.undo()
        running = await tasks.load_task(task.id)
        assert running.status is TaskStatus.RUNNING and _task_cancellation_requested(running)

        # A recovery pass is cut off right after it publishes a task event, if any.
        original_fan_out = RuntimeEventWriter.fan_out_persisted
        task_events = {EventType.TASK_FAILED, EventType.TASK_CANCELLED}
        cut_off = False

        async def cut_off_after_task_event(self, events, *args, **kwargs):
            nonlocal cut_off
            if not cut_off and any(event.type in task_events for event in events):
                cut_off = True
                raise ConnectionError("delivery interrupted")
            return await original_fan_out(self, events, *args, **kwargs)

        monkeypatch.setattr(RuntimeEventWriter, "fan_out_persisted", cut_off_after_task_event)
        clock.now += timedelta(days=1)
        recovery = _app(store, tasks)
        try:
            await recovery.recover_abandoned_sessions(inactive_for_seconds=0)
            # Its request then settles the task within its live lease, and recovery
            # runs again.
            clock.now -= timedelta(days=1)
            running = await tasks.load_task(task.id)
            cancellation = _task_cancellation_terminalization_request(
                running, worker_id=WORKER_ID, claim_generation=running.claim_generation
            )
            await _terminalize_claimed_task(tasks, cancellation)
            clock.now += timedelta(days=1)
            await recovery.recover_abandoned_sessions(inactive_for_seconds=0)
            session = await store.load(SESSION_ID)
            assert session is not None and session.status is SessionStatus.FAILED
            assert (await tasks.load_task(task.id)).status is TaskStatus.CANCELLED
            events = await store.load_events(SESSION_ID)
            assert EventType.SESSION_FAILED in [event.type for event in events]
            assert not [event for event in events if event.type in task_events]
        finally:
            await recovery.aclose()

    asyncio.run(scenario())


def test_a_worker_failure_settles_its_cancellation_after_its_lease_was_renewed() -> None:
    async def scenario() -> None:
        tasks = InMemoryTaskStore()
        await tasks.create_task(TaskCreate(type=TASK_TYPE))
        claimed = await tasks.claim_task(WORKER_ID, TaskQuery(type=TASK_TYPE))
        assert claimed is not None
        generation = claimed.claim_generation
        await tasks.mark_claimed_task_execution_started(
            claimed.id, WORKER_ID, claim_generation=generation
        )
        await tasks.request_claimed_task_cancellation(
            claimed.id, WORKER_ID, claim_generation=generation
        )
        # The claim's heartbeat renews its lease after the failing request was built.
        renewed = await tasks.heartbeat(
            claimed.id, WORKER_ID, claim_generation=generation, extend_seconds=600
        )
        assert renewed.lease_expires_at != claimed.lease_expires_at
        failure = TaskTerminalizationRequest(
            task_id=claimed.id,
            worker_id=WORKER_ID,
            lease_expires_at=claimed.lease_expires_at,
            claim_generation=generation,
            kind=TaskTerminalKind.FAILED,
            error={"error": "handler failed"},
            idempotency_key="worker-failure:renewed-lease",
        )
        assert await _terminalize_claimed_task_or_detect_peer_winner(tasks, failure) is True
        assert (await tasks.load_task(claimed.id)).status is TaskStatus.CANCELLED

    asyncio.run(scenario())
