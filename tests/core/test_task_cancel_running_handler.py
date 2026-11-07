"""Cancelling a task whose handler is running reaches the handler."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.outcomes import run_to_completion
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCancellationSignal,
    TaskCreate,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryPolicy,
    TaskStatus,
    current_task_cancellation,
    run_task_worker,
)
from cayu.tasks._execution_settlement import TaskExecutionSettlement
from cayu.tasks.worker import complete_managed_task

GRACE_S = 0.2
# Old releases never end these handlers; stop them after this long so a
# regression fails instead of hanging.
GIVE_UP_S = 8.0
OPERATOR_ERROR = {"code": "task_cancelled", "message": "operator asked to stop"}


class StalledProvider(ScriptedModelProvider):
    async def stream(self, request):
        await asyncio.Event().wait()
        yield ModelStreamEvent.completed()


def _app(store) -> CayuApp:
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(StalledProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted"))
    return app


async def _wait_for(predicate, give_up: asyncio.Event):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + GIVE_UP_S
    while loop.time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(0.02)
    give_up.set()
    return False


def _session_running(app, session_id):
    async def check():
        session = await app.session_store.load(session_id)
        return session is not None and session.status is SessionStatus.RUNNING

    return check


def _worker(app, store, handler, **options):
    handler_tasks = []

    async def recorded(app, task, worker_id):
        handler_tasks.append(asyncio.current_task())
        return await handler(app, task, worker_id)

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            recorded,
            worker_id="w",
            max_tasks=1,
            lease_seconds=3,
            poll_interval_s=0.02,
            **options,
        )
    )
    return worker, handler_tasks


async def _finish(worker, handler_tasks):
    done, _ = await asyncio.wait({worker}, timeout=GIVE_UP_S)
    if worker in done:
        return worker.result()
    # A release that never delivers the request leaves the handler running.
    for handler_task in handler_tasks:
        handler_task.cancel()
    await asyncio.wait({worker}, timeout=GIVE_UP_S)
    pytest.fail("The worker did not stop its handler after the task was cancelled.")


@pytest.mark.parametrize("execution_timeout_s", [None, 600.0])
def test_cancel_interrupts_the_handler_and_its_running_session(execution_timeout_s):
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        give_up = asyncio.Event()
        seen = {}

        async def handler(app, task, worker_id):
            # The reported shape: sessions awaited directly beside other work.
            try:
                await asyncio.gather(
                    run_to_completion(
                        app,
                        RunRequest(
                            agent_name="worker",
                            session_id="in-flight",
                            messages=[Message.text("user", "go")],
                        ),
                    ),
                    asyncio.wait_for(give_up.wait(), 60),
                )
            except asyncio.CancelledError:
                seen["cancelled"] = asyncio.get_running_loop().time()
                raise

        worker = _worker(
            app,
            store,
            handler,
            execution_timeout_s=execution_timeout_s,
            execution_grace_s=GRACE_S,
        )
        assert await _wait_for(_session_running(app, "in-flight"), give_up)
        cancelled_at = asyncio.get_running_loop().time()
        requested = await store.cancel_task(task.id, OPERATOR_ERROR)
        assert requested.status is TaskStatus.CLAIMED
        assert requested.status_reason == "cancellation_requested"

        assert await _finish(*worker) == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.CANCELLED
        assert final.error == OPERATOR_ERROR
        # Delivered within a heartbeat (lease_seconds / 3) plus the grace period.
        assert seen["cancelled"] - cancelled_at < 1.0 + GRACE_S + 1.0
        session = await app.session_store.load("in-flight")
        assert session.status is SessionStatus.INTERRUPTED
        events = await app.session_store.load_events("in-flight")
        assert events[-1].type.value == "session.interrupted"

    asyncio.run(scenario())


def test_handler_sees_the_request_and_new_runs_are_refused_in_the_grace_period():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        seen = {}
        entered = asyncio.Event()

        async def handler(app, task, worker_id):
            signal = current_task_cancellation()
            seen["before"] = signal.requested
            entered.set()
            await asyncio.wait_for(signal.wait(), GIVE_UP_S)
            seen["after"] = signal.requested
            seen["retry"] = await run_to_completion(
                app, RunRequest(agent_name="worker", messages=[Message.text("user", "again")])
            )
            # Settling it itself does not override the operator's request.
            await complete_managed_task(app.task_store, task, worker_id, {"delivered": True})

        worker = _worker(app, store, handler, execution_grace_s=2.0)
        await asyncio.wait_for(entered.wait(), GIVE_UP_S)
        await store.cancel_task(task.id, OPERATOR_ERROR)
        assert await _finish(*worker) == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.CANCELLED
        assert final.error == OPERATOR_ERROR
        assert seen["before"] is False and seen["after"] is True
        assert seen["retry"].status is SessionStatus.FAILED
        assert "TaskCancellationRequested" in seen["retry"].error

    asyncio.run(scenario())


def test_interrupt_on_cancel_false_only_signals_the_handler():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        entered = asyncio.Event()
        seen = {"cancelled": False}

        async def handler(app, task, worker_id):
            entered.set()
            try:
                await asyncio.wait_for(current_task_cancellation().wait(), GIVE_UP_S)
                # Well past the grace period: the worker leaves the handler alone.
                await asyncio.sleep(GRACE_S + 0.5)
            except asyncio.CancelledError:
                seen["cancelled"] = True
                raise

        worker = _worker(app, store, handler, execution_grace_s=GRACE_S, interrupt_on_cancel=False)
        await asyncio.wait_for(entered.wait(), GIVE_UP_S)
        await store.cancel_task(task.id, OPERATOR_ERROR)
        assert await _finish(*worker) == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.CANCELLED
        assert final.error == OPERATOR_ERROR
        assert seen["cancelled"] is False

    asyncio.run(scenario())


def test_cancel_interrupts_a_retry_policy_attempt():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(
            TaskCreate(
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=3, initial_backoff_seconds=60.0),
            )
        )
        give_up = asyncio.Event()

        async def handler(app, task, worker_id):
            await run_to_completion(
                app,
                RunRequest(
                    agent_name="worker",
                    session_id="attempt",
                    messages=[Message.text("user", "go")],
                ),
            )
            await asyncio.wait_for(give_up.wait(), 60)

        worker = _worker(app, store, handler, execution_grace_s=GRACE_S)
        assert await _wait_for(_session_running(app, "attempt"), give_up)
        await store.cancel_task(task.id, OPERATOR_ERROR)
        assert await _finish(*worker) == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.CANCELLED
        assert final.retry_series.disposition.value == "cancelled"
        session = await app.session_store.load("attempt")
        assert session.status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_a_handler_its_time_limit_cancelled_is_not_cancelled_again():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        seen = {"cancellations": 0, "cleaned": False}

        async def handler(app, task, worker_id):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                seen["cancellations"] += 1
                # Cleanup that outlasts the time-limit fence and the next heartbeat.
                try:
                    await asyncio.sleep(3.5)
                except asyncio.CancelledError:
                    seen["cancellations"] += 1
                    raise
                seen["cleaned"] = True
                raise

        worker = _worker(app, store, handler, execution_timeout_s=0.5, execution_grace_s=0.0)
        assert await _finish(*worker) == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.CANCELLED
        assert final.error["code"] == "task_handler_timeout"
        assert seen == {"cancellations": 1, "cleaned": True}

    asyncio.run(scenario())


def test_a_returned_retry_report_survives_a_slow_acknowledgement(monkeypatch):
    original_attempt = TaskExecutionSettlement.attempt

    async def slow_attempt(self):
        await asyncio.sleep(0.6)
        return await original_attempt(self)

    monkeypatch.setattr(TaskExecutionSettlement, "attempt", slow_attempt)

    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(
            TaskCreate(
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=3, initial_backoff_seconds=60.0),
            )
        )
        entered = asyncio.Event()

        async def handler(app, task, worker_id):
            entered.set()
            await asyncio.wait_for(current_task_cancellation().wait(), GIVE_UP_S)
            return TaskRetryAttemptReport(
                idempotency_key="stopped-on-request",
                disposition=TaskRetryAttemptDisposition.CANCELLED,
                error={"code": "stopped_on_request"},
                token_count=1234,
                estimated_cost=Decimal("0.25"),
            )

        worker = _worker(app, store, handler, execution_grace_s=GRACE_S)
        await asyncio.wait_for(entered.wait(), GIVE_UP_S)
        await store.cancel_task(task.id, OPERATOR_ERROR)
        assert await _finish(*worker) == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.CANCELLED
        assert final.retry_series.disposition.value == "cancelled"
        # The report returned before the grace passed keeps its usage.
        assert final.retry_series.cumulative_tokens == 1234
        assert final.retry_series.cumulative_estimated_cost == Decimal("0.25")

    asyncio.run(scenario())


def test_a_hand_written_worker_can_bind_its_own_signal():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        signal = TaskCancellationSignal()
        assert current_task_cancellation() is None
        with signal.bind():
            assert current_task_cancellation() is signal
            assert signal.requested is False
            signal.request()
            signal.request()  # Idempotent.
            assert signal.requested is True
            await asyncio.wait_for(signal.wait(), 1)
            outcome = await run_to_completion(
                app, RunRequest(agent_name="worker", messages=[Message.text("user", "go")])
            )
            assert outcome.status is SessionStatus.FAILED
            assert "TaskCancellationRequested" in outcome.error
        assert current_task_cancellation() is None

    asyncio.run(scenario())
