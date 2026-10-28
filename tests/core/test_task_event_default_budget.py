"""The default event budget caps attempts, not time, and short budgets are visible."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from cayu import CayuApp
from cayu.storage import SQLiteTaskStore
from cayu.tasks import (
    DEFAULT_TASK_RETRY_BUDGET,
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryBudget,
    TaskStatus,
    run_task_worker,
)
from cayu.tasks.retry_budgets import (
    TaskRetryBudgetExhausted,
    TaskRetryKeyStatus,
    reconcile_task_event_deadlines,
)
from cayu.tasks.worker import complete_managed_task

_WORKER_LOGGER = "cayu.tasks.worker"


def _store(backend, tmp_path, *, clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    if backend == "memory":
        return InMemoryTaskStore(**kwargs)
    return SQLiteTaskStore(tmp_path / "events.sqlite", **kwargs)


def _budget_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _WORKER_LOGGER and "event budget of" in record.getMessage()
    ]


async def _complete(app, task, worker_id):
    await complete_managed_task(app.task_store, task, worker_id, {"ok": True})


def test_default_budget_caps_attempts_without_an_elapsed_limit():
    assert TaskRetryBudget(max_attempts=3, max_elapsed_seconds=None) == DEFAULT_TASK_RETRY_BUDGET
    # Declaring TaskRetryBudget() still means what it says.
    assert TaskRetryBudget() == TaskRetryBudget(max_attempts=3, max_elapsed_seconds=1800)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_undeclared_event_keeps_running_past_thirty_minutes(backend, tmp_path):
    now = [datetime(2026, 10, 28, tzinfo=UTC)]

    async def scenario():
        store = _store(backend, tmp_path, clock=lambda: now[0])
        task = await store.create_task(TaskCreate(type="scan"))
        state = await store.load_task_retry_key(task.invocation.event_retry_key)
        assert state.budget == DEFAULT_TASK_RETRY_BUDGET
        assert state.elapsed_deadline is None
        now[0] += timedelta(hours=6)
        await reconcile_task_event_deadlines(store)
        assert (await store.load_task(task.id)).status is TaskStatus.PENDING
        assert await store.task_event_remaining_seconds(task.invocation.event_retry_key) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_default_attempt_cap_still_stops_a_resubmission_loop(backend, tmp_path):
    async def scenario():
        store = _store(backend, tmp_path)
        for _ in range(3):
            task = await store.create_task(TaskCreate(type="scan", event_id="evt-1"))
            await store.fail_task(task.id, {"code": "failed"})
        with pytest.raises(
            TaskRetryBudgetExhausted, match="default a task gets when it declares no retry_budget"
        ) as refused:
            await store.create_task(TaskCreate(type="scan", event_id="evt-1"))
        assert refused.value.state.status is TaskRetryKeyStatus.GIVEN_UP
        assert refused.value.state.given_up_reason.value == "attempts_exhausted"

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_declared_elapsed_budget_still_cancels_and_says_which_budget(backend, tmp_path):
    now = [datetime(2026, 10, 28, tzinfo=UTC)]

    async def scenario():
        store = _store(backend, tmp_path, clock=lambda: now[0])
        declared = await store.create_task(
            TaskCreate(
                type="scan",
                event_id="declared",
                retry_budget=TaskRetryBudget(max_attempts=None, max_elapsed_seconds=60),
            )
        )
        # TaskRetryBudget() is the budget 0.16.0 gave undeclared tasks.
        legacy = await store.create_task(
            TaskCreate(type="scan", event_id="legacy", retry_budget=TaskRetryBudget())
        )
        now[0] += timedelta(hours=1)
        await reconcile_task_event_deadlines(store)

        declared = await store.load_task(declared.id)
        assert declared.status is TaskStatus.CANCELLED
        assert declared.error["code"] == "event_elapsed_exhausted"
        assert declared.error["event_max_elapsed_seconds"] == 60
        assert "60 seconds" in declared.error["message"]
        assert (
            "retry_budget=TaskRetryBudget(max_elapsed_seconds=...)" in (declared.error["message"])
        )
        assert "0.16.0" not in declared.error["message"]

        legacy = await store.load_task(legacy.id)
        assert legacy.status is TaskStatus.CANCELLED
        assert "1,800 seconds" in legacy.error["message"]
        assert "Cayu 0.16.0 gave tasks created without retry_budget" in legacy.error["message"]

    asyncio.run(scenario())


def test_worker_warns_once_when_a_declared_budget_is_shorter_than_its_limit(caplog):
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        short = TaskRetryBudget(max_attempts=None, max_elapsed_seconds=600)
        try:
            for index in range(2):
                await app.create_task(
                    TaskCreate(type="scan", event_id=f"scan-{index}", retry_budget=short)
                )
            await app.create_task(TaskCreate(type="report"))
            with caplog.at_level(logging.WARNING, logger=_WORKER_LOGGER):
                handled = await run_task_worker(
                    app,
                    store,
                    _complete,
                    worker_id="scanner",
                    max_tasks=3,
                    execution_timeout_s=7200,
                    reclaim=False,
                )
            assert handled == 3
            # The declared budget still applied; the tasks finished within it.
            tasks = await store.list_tasks()
            assert {task.status for task in tasks} == {TaskStatus.COMPLETED}
        finally:
            await app.aclose()

    asyncio.run(scenario())
    warnings = _budget_warnings(caplog)
    assert len(warnings) == 1
    assert "600 seconds" in warnings[0]
    assert "execution_timeout_s=7200" in warnings[0]
    assert "retry_budget=TaskRetryBudget(max_elapsed_seconds=...)" in warnings[0]
    assert "'scan'" in warnings[0] or "task_type=scan" in warnings[0]


@pytest.mark.parametrize(
    ("budget", "execution_timeout_s"),
    [
        (None, 7200),  # The default has no elapsed limit.
        (None, None),
        (TaskRetryBudget(max_elapsed_seconds=7200), 7200),
        # Without its own limit, the worker is bounded by the declared budget.
        (TaskRetryBudget(max_elapsed_seconds=600), None),
    ],
)
def test_worker_stays_quiet_when_the_budget_does_not_cut_its_limit_short(
    caplog, budget, execution_timeout_s
):
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            await app.create_task(TaskCreate(type="scan", event_id="one", retry_budget=budget))
            with caplog.at_level(logging.WARNING, logger=_WORKER_LOGGER):
                assert (
                    await run_task_worker(
                        app,
                        store,
                        _complete,
                        worker_id="scanner",
                        max_tasks=1,
                        execution_timeout_s=execution_timeout_s,
                        reclaim=False,
                    )
                    == 1
                )
        finally:
            await app.aclose()

    asyncio.run(scenario())
    assert _budget_warnings(caplog) == []


def test_worker_without_a_limit_warns_about_an_event_queued_under_0_16_0(caplog):
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            # The value an undeclared 0.16.0 event was admitted with.
            await app.create_task(
                TaskCreate(type="scan", event_id="old", retry_budget=TaskRetryBudget())
            )
            with caplog.at_level(logging.WARNING, logger=_WORKER_LOGGER):
                await run_task_worker(
                    app, store, _complete, worker_id="scanner", max_tasks=1, reclaim=False
                )
        finally:
            await app.aclose()

    asyncio.run(scenario())
    (warning,) = _budget_warnings(caplog)
    assert "sets no execution_timeout_s" in warning
    assert "Cayu 0.16.0" in warning


def test_handler_cut_off_by_an_elapsed_budget_fails_with_the_reason(caplog):
    offset = [timedelta()]

    async def scenario():
        store = InMemoryTaskStore(clock=lambda: datetime.now(UTC) + offset[0])
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            task = await app.create_task(
                TaskCreate(type="scan", event_id="old", retry_budget=TaskRetryBudget())
            )
            # The task waited in the queue for almost all of its 30 minutes.
            offset[0] = timedelta(seconds=1799.5)

            async def hang(app, task, worker_id):
                await asyncio.Event().wait()

            with caplog.at_level(logging.WARNING, logger=_WORKER_LOGGER):
                await asyncio.wait_for(
                    run_task_worker(
                        app,
                        store,
                        hang,
                        worker_id="scanner",
                        max_tasks=1,
                        execution_timeout_s=7200,
                        reclaim=False,
                    ),
                    10,
                )
            final = await store.load_task(task.id)
            assert final.status is TaskStatus.FAILED
            assert final.error["code"] == "event_elapsed_exhausted"
            assert final.error["event_max_elapsed_seconds"] == 1800
            assert "1,800 seconds" in final.error["message"]
            assert "Cayu 0.16.0" in final.error["message"]
            assert "retry_budget" in final.error["message"]
        finally:
            await app.aclose()

    asyncio.run(scenario())
    failures = [
        record.getMessage()
        for record in caplog.records
        if "exceeded its time limit" in record.getMessage()
    ]
    assert failures and "source=task_event" in failures[0]
    assert "Cayu 0.16.0" in failures[0]
