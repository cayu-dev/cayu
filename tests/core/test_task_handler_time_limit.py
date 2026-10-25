"""run_task_worker(execution_timeout_s=...) bounds the whole handler call."""

from __future__ import annotations

import asyncio

import pytest

from cayu import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.deadlines import (
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    HandlerTimeLimit,
    admit_handler_run,
    current_handler_time_limit,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.outcomes import run_to_completion
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryPolicy,
    TaskStatus,
    run_task_worker,
)
from cayu.tasks.worker import complete_managed_task

# The grace period is the end of the limit: new runs are refused at 0.3 s.
TIMEOUT_S = 0.5
GRACE_S = 0.2
# Old releases never end these handlers; the tests stop them after this long
# so a regression fails instead of hanging.
GIVE_UP_S = 8.0


class StalledProvider(ScriptedModelProvider):
    async def stream(self, request):
        await asyncio.Event().wait()
        yield ModelStreamEvent.completed()


class StalledTool(Tool):
    spec = ToolSpec(name="wait", effect=ToolEffect.NONE)

    async def run(self, ctx, arguments):
        await asyncio.Event().wait()
        return ToolResult(content="unreachable")


def _app(store, provider=None) -> CayuApp:
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(provider or StalledProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted"), tools=[StalledTool()])
    return app


async def _wait_for_status(store, task_id, statuses, give_up):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + GIVE_UP_S
    while loop.time() < deadline:
        task = await store.load_task(task_id)
        if task.status in statuses:
            return task
        await asyncio.sleep(0.05)
    give_up.set()
    return await store.load_task(task_id)


def _retry_forever(give_up, *, explicit_deadline):
    """The handler authors write: unattached runs, retried until one works."""

    sessions = []

    async def handler(app, task, worker_id):
        attempt = 0
        while not give_up.is_set():
            attempt += 1
            extra = (
                {"execution_deadline": ExecutionDeadline.after(0.15)} if explicit_deadline else {}
            )
            try:
                outcome = await run_to_completion(
                    app,
                    RunRequest(
                        agent_name="worker",
                        session_id=f"{task.id}-a{attempt}",
                        messages=[Message.text("user", "go")],
                        **extra,
                    ),
                )
                sessions.append(outcome.session_id)
                if outcome.ok:
                    await complete_managed_task(app.task_store, task, worker_id, {})
                    return None
            except Exception:
                pass
            await asyncio.sleep(0.02)
        return None

    return handler, sessions


@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize("explicit_deadline", [False, True])
def test_retry_forever_handler_fails_its_task_at_the_limit(
    sqlite, explicit_deadline, sqlite_resources
):
    async def scenario():
        async with sqlite_resources as resources:
            store = (
                resources.own(SQLiteTaskStore(resources.path())) if sqlite else InMemoryTaskStore()
            )
            # Warm lazy provider/session setup before measuring the short handler budget.
            warm = _app(store, ScriptedModelProvider([ModelStreamEvent.completed()]))
            await run_to_completion(
                warm, RunRequest(agent_name="worker", messages=[Message.text("user", "warm")])
            )
            await warm.aclose()
            app = _app(store)
            task = await store.create_task(TaskCreate(type="job", restartable=True))
            give_up = asyncio.Event()
            retry, _sessions = _retry_forever(give_up, explicit_deadline=explicit_deadline)
            entered = asyncio.Event()

            async def handler(app, task, worker_id):
                entered.set()
                return await retry(app, task, worker_id)

            started = asyncio.get_running_loop().time()
            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="w",
                    max_tasks=1,
                    poll_interval_s=0.02,
                    execution_timeout_s=TIMEOUT_S,
                    execution_grace_s=GRACE_S,
                )
            )
            final = await _wait_for_status(store, task.id, {TaskStatus.FAILED}, give_up)
            elapsed = asyncio.get_running_loop().time() - started
            assert await asyncio.wait_for(worker, 5) == 1

            assert final.status is TaskStatus.FAILED, final
            assert elapsed < TIMEOUT_S + GRACE_S + 2
            assert final.worker_id is None and final.lease_expires_at is None
            # A restartable task is not requeued by its time limit.
            assert final.restart_count == 0
            error = final.error
            assert error["error"] == "ExecutionDeadlineExceeded"
            assert error["scope"] == "task_handler"
            assert error["timeout_seconds"] == TIMEOUT_S
            assert error["grace_seconds"] == GRACE_S
            assert error["handler_cancelled"] is True
            assert error["execution_deadline"]["scope"] == "task_handler"
            assert entered.is_set()
            # A short per-run deadline may expire during admission on a busy
            # host. The handler must still fail even if no session was admitted.
            assert len(error["session_ids"]) == min(error["sessions_started"], 20)
            assert all(identity.startswith(f"{task.id}-a") for identity in error["session_ids"])
            # No session the call started is left running. One started just
            # before the limit can fail at setup when its deadline runs out.
            for session_id in error["session_ids"]:
                session = await app.session_store.load(session_id)
                assert session.status in {SessionStatus.INTERRUPTED, SessionStatus.FAILED}

    asyncio.run(scenario())


@pytest.mark.parametrize("hang", ["model", "tool"])
def test_run_in_flight_at_the_hard_limit_ends_interrupted(hang):
    async def scenario():
        store = InMemoryTaskStore()
        provider = (
            StalledProvider([])
            if hang == "model"
            else ScriptedModelProvider(
                [
                    ModelStreamEvent.tool_call(id="wait", name="wait", arguments={}),
                    ModelStreamEvent.completed(),
                ]
            )
        )
        app = _app(store, provider)
        task = await store.create_task(TaskCreate(type="job"))

        async def handler(app, task, worker_id):
            # Its own deadline outlives the handler's limit and grace period.
            await run_to_completion(
                app,
                RunRequest(
                    agent_name="worker",
                    session_id="in-flight",
                    messages=[Message.text("user", "go")],
                    execution_deadline=ExecutionDeadline.after(60),
                ),
            )

        handled = await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                poll_interval_s=0.02,
                execution_timeout_s=TIMEOUT_S,
                execution_grace_s=GRACE_S,
            ),
            GIVE_UP_S,
        )
        assert handled == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.FAILED
        assert final.error["handler_cancelled"] is True
        assert final.error["session_ids"] == ["in-flight"]
        session = await app.session_store.load("in-flight")
        assert session.status is SessionStatus.INTERRUPTED
        events = await app.session_store.load_events("in-flight")
        assert events[-1].type.value == "session.interrupted"

    asyncio.run(scenario())


def test_grace_period_lets_the_handler_deliver_and_refuses_new_runs():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        seen = {}

        async def handler(app, task, worker_id):
            limit = current_handler_time_limit()
            seen["before"] = limit.expired
            await asyncio.sleep(TIMEOUT_S - GRACE_S + 0.05)  # The agent's work ran long.
            seen["after"] = limit.expired
            outcome = await run_to_completion(
                app, RunRequest(agent_name="worker", messages=[Message.text("user", "again")])
            )
            seen["retry"] = outcome
            await asyncio.sleep(0.05)  # Delivering the result.
            await complete_managed_task(app.task_store, task, worker_id, {"delivered": True})

        handled = await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                poll_interval_s=0.02,
                execution_timeout_s=TIMEOUT_S - GRACE_S + 1.0,
                execution_grace_s=1.0,
            ),
            GIVE_UP_S,
        )
        assert handled == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.COMPLETED
        assert final.result == {"delivered": True}
        assert seen["before"] is False and seen["after"] is True
        assert seen["retry"].status is SessionStatus.FAILED
        assert "ExecutionDeadlineExceeded" in seen["retry"].error
        assert current_handler_time_limit() is None

    asyncio.run(scenario())


def test_handler_that_returns_unsettled_after_the_limit_fails_with_the_timeout():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))

        async def handler(app, task, worker_id):
            await asyncio.sleep(TIMEOUT_S - GRACE_S + 0.05)
            raise RuntimeError("delivery failed")

        await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                poll_interval_s=0.02,
                execution_timeout_s=TIMEOUT_S - GRACE_S + 1.0,
                execution_grace_s=1.0,
            ),
            GIVE_UP_S,
        )
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.FAILED
        assert final.error["error"] == "ExecutionDeadlineExceeded"
        assert final.error["handler_cancelled"] is False
        assert final.error["handler_failure"]["error"] == "RuntimeError"
        assert final.error["session_ids"] == []

    asyncio.run(scenario())


def test_operator_cancellation_wins_over_the_time_limit():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        give_up = asyncio.Event()
        handler, _sessions = _retry_forever(give_up, explicit_deadline=True)
        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                poll_interval_s=0.02,
                execution_timeout_s=TIMEOUT_S,
                execution_grace_s=GRACE_S,
            )
        )
        while (await store.load_task(task.id)).status is TaskStatus.PENDING:
            await asyncio.sleep(0.02)
        await store.cancel_task(task.id, {"code": "operator"})
        final = await _wait_for_status(store, task.id, {TaskStatus.CANCELLED}, give_up)
        await asyncio.wait_for(worker, 5)
        assert final.status is TaskStatus.CANCELLED
        assert final.error == {"code": "operator"}

    asyncio.run(scenario())


def test_cancelling_the_worker_drains_within_the_time_limit():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        give_up = asyncio.Event()
        handler, _sessions = _retry_forever(give_up, explicit_deadline=True)
        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                poll_interval_s=0.02,
                execution_timeout_s=TIMEOUT_S,
                execution_grace_s=GRACE_S,
            )
        )
        while (await store.load_task(task.id)).status is TaskStatus.PENDING:
            await asyncio.sleep(0.02)
        worker.cancel()
        # The drain waits for the handler; the time limit still ends it.
        final = await _wait_for_status(store, task.id, {TaskStatus.CANCELLED}, give_up)
        done, _ = await asyncio.wait({worker}, timeout=5)
        assert done == {worker}
        assert final.status is TaskStatus.CANCELLED
        assert final.error == {"code": "task_worker_owner_cancelled"}

    asyncio.run(scenario())


def test_handler_that_ignores_cancellation_keeps_its_lease_under_a_fence():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        release = asyncio.Event()
        ignored = []

        async def handler(app, task, worker_id):
            while not release.is_set():
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    ignored.append(True)

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                lease_seconds=3,
                poll_interval_s=0.02,
                execution_timeout_s=TIMEOUT_S,
                execution_grace_s=GRACE_S,
            )
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + GIVE_UP_S
        while loop.time() < deadline:
            current = await store.load_task(task.id)
            if current.status_reason == "cancellation_requested":
                break
            await asyncio.sleep(0.05)
        try:
            assert ignored
            assert current.status is TaskStatus.CLAIMED
            assert current.worker_id == "w"
            assert current.status_payload["error"]["code"] == "task_handler_timeout"
            await asyncio.sleep(3.2)  # Longer than one lease: still renewed.
            held = await store.load_task(task.id)
            assert held.status is TaskStatus.CLAIMED and held.worker_id == "w"
        finally:
            release.set()
        assert await asyncio.wait_for(worker, 5) == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.CANCELLED
        assert final.error["code"] == "task_handler_timeout"

    asyncio.run(scenario())


def test_retry_policy_task_records_a_retryable_attempt():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(
            TaskCreate(
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2, initial_backoff_seconds=60.0),
            )
        )
        give_up = asyncio.Event()
        handler, _sessions = _retry_forever(give_up, explicit_deadline=False)
        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                poll_interval_s=0.02,
                execution_timeout_s=TIMEOUT_S,
                execution_grace_s=GRACE_S,
            )
        )
        final = await _wait_for_status(store, task.id, {TaskStatus.FAILED}, give_up)
        await asyncio.wait_for(worker, 5)
        assert final.status is TaskStatus.FAILED
        assert final.error["error"] == "ExecutionDeadlineExceeded"
        assert final.retry_series.disposition.value == "retry_scheduled"
        successor = await store.load_task(final.retry_series.successor_task_id)
        assert successor.status is TaskStatus.PENDING

    asyncio.run(scenario())


def test_handler_time_limit_works_without_the_worker():
    async def scenario():
        with pytest.raises(ExecutionDeadlineExceeded) as raised:
            async with HandlerTimeLimit(0.1, grace_s=0.05) as limit:
                await asyncio.Event().wait()
        assert limit.expired and limit.cancelled
        assert raised.value.diagnostics["scope"] == "task_handler"
        assert raised.value.diagnostics["grace_seconds"] == 0.05

        async with HandlerTimeLimit(5, grace_s=1) as quick:
            assert current_handler_time_limit() is quick
            await asyncio.sleep(0)
        assert not quick.expired and not quick.cancelled
        assert current_handler_time_limit() is None

        # A TimeoutError raised by the block itself is not the limit's.
        with pytest.raises(TimeoutError) as own:
            async with HandlerTimeLimit(5, grace_s=1):
                raise TimeoutError("provider")
        assert type(own.value) is TimeoutError

        # Cancellation from outside passes through unchanged.
        async def body():
            async with HandlerTimeLimit(5, grace_s=1):
                await asyncio.Event().wait()

        outer = asyncio.create_task(body())
        await asyncio.sleep(0.01)
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer

    asyncio.run(scenario())


def test_grace_period_is_the_end_of_the_limit_not_extra_time():
    """A 600 s limit with 30 s grace refuses new runs at 570 s and cancels at 600 s."""

    async def scenario():
        loop = asyncio.get_running_loop()
        # The 600/30 example from the guide, scaled down a thousand times.
        async with HandlerTimeLimit(0.6, grace_s=0.03) as limit:
            started = loop.time()
            assert limit.soft_timeout_s == pytest.approx(0.57)
            # The hard limit is fixed at entry: the timeout itself, not 0.63.
            assert limit._timer.when() - started == pytest.approx(0.6, abs=0.005)
            assert limit.deadline.remaining_seconds() == pytest.approx(0.57, abs=0.02)
            assert limit.inspection()["runs_refused_after_seconds"] == pytest.approx(0.57)
            assert admit_handler_run() is limit
        assert not limit.expired

        with pytest.raises(ExecutionDeadlineExceeded):
            async with HandlerTimeLimit(0.6, grace_s=0.03) as limit:
                started = loop.time()
                while not limit.expired:
                    await asyncio.sleep(0.005)
                soft_at = loop.time() - started
                with pytest.raises(ExecutionDeadlineExceeded):
                    admit_handler_run()
                try:
                    await asyncio.Event().wait()
                finally:
                    hard_at = loop.time() - started
        assert limit.cancelled
        assert 0.56 <= soft_at < 0.6
        assert 0.6 <= hard_at < 0.63

    asyncio.run(scenario())


def test_worker_handler_ends_at_execution_timeout_with_grace_inside():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        marks = {}

        async def handler(app, task, worker_id):
            loop = asyncio.get_running_loop()
            started = loop.time()
            limit = current_handler_time_limit()
            try:
                while not limit.expired:
                    await asyncio.sleep(0.005)
                marks["soft"] = loop.time() - started
                await asyncio.Event().wait()
            finally:
                marks["hard"] = loop.time() - started

        await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                poll_interval_s=0.02,
                execution_timeout_s=0.6,
                execution_grace_s=0.03,
            ),
            GIVE_UP_S,
        )
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.FAILED
        assert final.error["timeout_seconds"] == 0.6
        assert final.error["grace_seconds"] == 0.03
        assert final.error["handler_cancelled"] is True
        assert 0.55 <= marks["soft"] < 0.6
        assert 0.59 <= marks["hard"] < 0.63

    asyncio.run(scenario())


def test_default_run_deadline_comes_from_the_entered_limit():
    """A run ended by its default deadline always finds the handler limit expired."""

    async def scenario():
        async with HandlerTimeLimit(5, grace_s=1) as limit:
            run = limit.run_deadline()
            assert run.expires_at == limit.deadline.expires_at
            assert run._monotonic_expiry == limit.deadline._monotonic_expiry
            assert run.scope == "execution" and run.source == limit.source

        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))
        seen = {}

        async def handler(app, task, worker_id):
            # No execution_deadline of its own: the worker's default ends it.
            outcome = await run_to_completion(
                app, RunRequest(agent_name="worker", messages=[Message.text("user", "go")])
            )
            seen["status"] = outcome.status
            seen["expired"] = current_handler_time_limit().expired
            return None  # Returns straight away, without settling the task.

        await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                poll_interval_s=0.02,
                execution_timeout_s=1.0,
                execution_grace_s=0.5,
            ),
            GIVE_UP_S,
        )
        # On a loaded host the deadline can also end the run during setup.
        assert seen["status"] in {SessionStatus.INTERRUPTED, SessionStatus.FAILED}
        assert seen["expired"] is True
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.FAILED
        assert final.error["error"] == "ExecutionDeadlineExceeded"
        assert final.error["handler_cancelled"] is False

    asyncio.run(scenario())


def test_default_grace_fits_inside_short_limits_and_explicit_grace_must():
    assert HandlerTimeLimit(600).grace_s == 30
    assert HandlerTimeLimit(20).grace_s == 10  # Half the limit when 30 s would not fit.
    assert HandlerTimeLimit(20).soft_timeout_s == 10
    assert HandlerTimeLimit(1, grace_s=0).soft_timeout_s == 1
    assert HandlerTimeLimit(0).grace_s == 0
    for timeout_s, grace_s in ((30, 30), (10, 30), (0, 1)):
        with pytest.raises(ValueError, match="smaller than timeout_s"):
            HandlerTimeLimit(timeout_s, grace_s=grace_s)
    store = InMemoryTaskStore()

    async def handler(app, task, worker_id):
        return None

    with pytest.raises(ValueError, match="smaller than timeout_s"):
        asyncio.run(
            run_task_worker(
                CayuApp(task_store=store, enable_logging=False),
                store,
                handler,
                worker_id="w",
                execution_timeout_s=10,
                execution_grace_s=30,
            )
        )


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True])
def test_time_limit_settings_are_validated(bad):
    with pytest.raises(ValueError):
        HandlerTimeLimit(bad)
    with pytest.raises(ValueError):
        HandlerTimeLimit(1, grace_s=bad)
    store = InMemoryTaskStore()

    async def handler(app, task, worker_id):
        return None

    with pytest.raises(ValueError):
        asyncio.run(
            run_task_worker(
                CayuApp(task_store=store, enable_logging=False),
                store,
                handler,
                worker_id="w",
                execution_timeout_s=1,
                execution_grace_s=bad,
            )
        )


@pytest.mark.parametrize("external", [False, True])
def test_grouped_cancellation_retains_timeout_ownership(external):
    async def scenario():
        limit = HandlerTimeLimit(0.01, grace_s=0)

        async def body():
            async with limit:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    if external:
                        asyncio.current_task().cancel()
                    raise BaseExceptionGroup(
                        "cleanup", [cancellation, RuntimeError("cleanup")]
                    ) from None

        task = asyncio.create_task(body())
        expected = BaseExceptionGroup if external else ExecutionDeadlineExceeded
        with pytest.raises(expected) as caught:
            await task
        if not external:
            assert isinstance(caught.value.__cause__, BaseExceptionGroup)

    asyncio.run(scenario())


def test_grouped_handler_timeout_settles_task():
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(TaskCreate(type="job"))

        async def handler(*args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "cleanup", [cancellation, RuntimeError("cleanup")]
                ) from None

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                execution_timeout_s=0.05,
                execution_grace_s=0,
            )
            == 1
        )
        settled = await store.load_task(task.id)
        assert settled.status is TaskStatus.FAILED
        assert settled.error["error"] == "ExecutionDeadlineExceeded"
        assert settled.worker_id is None

    asyncio.run(scenario())


def test_handler_limit_records_only_admitted_sessions():
    async def scenario():
        app = _app(InMemoryTaskStore())
        async with HandlerTimeLimit(5) as limit:
            with pytest.raises(KeyError, match="Agent not registered"):
                async for _ in app.run(
                    RunRequest(agent_name="unknown", session_id="absent", messages=[])
                ):
                    pass
        assert limit.session_ids == ()
        assert limit.sessions_started == 0

    asyncio.run(scenario())


def test_settlement_latency_does_not_change_handler_timeout_disposition(monkeypatch):
    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(
            TaskCreate(type="job", retry_policy=TaskRetryPolicy(max_attempts=2))
        )
        from cayu.tasks._execution_settlement import TaskExecutionSettlement

        settle = TaskExecutionSettlement.attempt

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.15)
            return await settle(*args, **kwargs)

        monkeypatch.setattr(TaskExecutionSettlement, "attempt", slow)

        async def handler(*args):
            return None

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                execution_timeout_s=0.05,
                execution_grace_s=0.01,
            )
            == 1
        )
        settled = await store.load_task(task.id)
        assert settled.status is TaskStatus.FAILED
        assert settled.error["error"] == "RuntimeError"
        assert settled.retry_series.disposition.value == "non_retryable_failure"

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["error", "timeout", "expired_return"])
def test_continuation_failure_settles_with_owned_handoff(outcome):
    from tests.core.test_task_worker import _seed_receipt_backed_continuation

    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        await _seed_receipt_backed_continuation(
            app, store, task_id="continued", session_id="paused"
        )

        async def handler(*args):
            if outcome == "error":
                raise RuntimeError("handler failed")
            if outcome == "timeout":
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(0.1)

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                recovered_interrupted_task_handler=handler,
                recover_interrupted_handoffs=False,
                reclaim=False,
                execution_timeout_s=0.15,
                execution_grace_s=0.1,
            )
            == 1
        )
        settled = await store.load_task("continued")
        assert settled.status is TaskStatus.FAILED
        assert settled.worker_id is None
        assert settled.error["error"] == (
            "RuntimeError" if outcome == "error" else "ExecutionDeadlineExceeded"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_grouped_timeout_preserves_process_control(signal):
    async def scenario():
        with pytest.raises(BaseExceptionGroup) as raised:
            async with HandlerTimeLimit(0.01, grace_s=0):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    raise BaseExceptionGroup("cleanup", [cancellation, signal()]) from None
        assert isinstance(raised.value.exceptions[1], signal)

    asyncio.run(scenario())


def test_stopped_handler_timeout_is_retryable_during_slow_settlement(monkeypatch):
    from cayu.tasks._execution_settlement import TaskExecutionSettlement

    async def scenario():
        store = InMemoryTaskStore()
        app = _app(store)
        task = await store.create_task(
            TaskCreate(type="job", retry_policy=TaskRetryPolicy(max_attempts=2))
        )
        settle = TaskExecutionSettlement.attempt
        stopped = asyncio.Event()

        async def slow(*args, **kwargs):
            assert stopped.is_set()
            # Longer than the watcher's allowance after the hard limit.
            await asyncio.sleep(1.4)
            return await settle(*args, **kwargs)

        monkeypatch.setattr(TaskExecutionSettlement, "attempt", slow)

        async def handler(*args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopped.set()
                return None

        await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="w",
                max_tasks=1,
                execution_timeout_s=0.05,
                execution_grace_s=0.01,
            ),
            GIVE_UP_S,
        )
        settled = await store.load_task(task.id)
        assert settled.status is TaskStatus.FAILED
        assert settled.error["error"] == "ExecutionDeadlineExceeded"
        assert settled.retry_series.disposition.value == "retry_scheduled"
        successor = await store.load_task(settled.retry_series.successor_task_id)
        assert successor.status is TaskStatus.PENDING

    asyncio.run(scenario())


def test_nested_handler_limits_preserve_outer_admission_and_session_accounting():
    async def scenario():
        app = _app(
            InMemoryTaskStore(),
            ScriptedModelProvider([[ModelStreamEvent.completed()]]),
        )
        async with HandlerTimeLimit(60) as outer, HandlerTimeLimit(60) as inner:
            result = await run_to_completion(
                app,
                RunRequest(
                    agent_name="worker", session_id="nested", messages=[Message.text("user", "go")]
                ),
            )
            assert result.ok
            assert inner.session_ids == outer.session_ids == ("nested",)
            assert inner.sessions_started == outer.sessions_started == 1
            outer._deadline = ExecutionDeadline.after(0, scope="task_handler")
            with pytest.raises(ExecutionDeadlineExceeded):
                admit_handler_run()
            refused = await run_to_completion(
                app,
                RunRequest(
                    agent_name="worker",
                    session_id="refused",
                    messages=[Message.text("user", "go")],
                    execution_deadline=ExecutionDeadline.after(60),
                ),
            )
            assert not refused.ok
            assert await app.session_store.load("refused") is None
            assert inner.sessions_started == outer.sessions_started == 1

    asyncio.run(scenario())
