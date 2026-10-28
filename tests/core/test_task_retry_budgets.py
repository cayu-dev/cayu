"""A retry key caps how many tasks an application may create for one event."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.cli import main
from cayu.storage import SQLiteTaskStore
from cayu.tasks import (
    DEFAULT_TASK_RETRY_BUDGET,
    InMemoryTaskStore,
    TaskCreate,
    TaskQuery,
    TaskRetryBudget,
    TaskRetryBudgetExhausted,
    TaskRetryKeyStatus,
    TaskStatus,
    run_task_worker,
)
from cayu.tasks.graphs import TaskGraphNode
from cayu.tasks.records import TaskRetrySeriesDisposition


def _store(backend, tmp_path, request):
    if backend == "memory":
        return InMemoryTaskStore()
    if backend == "sqlite":
        return SQLiteTaskStore(tmp_path / "retry-keys.sqlite")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    return PostgresTaskStore(request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE)


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def _attempt(event_id: str, attempt: int, **fields) -> TaskCreate:
    """The shape the evidence apps' sweepers use: a new task ID per attempt."""
    return TaskCreate(
        task_id=f"evt-{event_id}-{attempt}",
        type="support.webhook_event",
        input={"event_id": event_id, "attempt": attempt},
        restartable=fields.pop("restartable", True),
        retry_key=f"event:{event_id}",
        **fields,
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_default_budget_refuses_the_fourth_task_for_a_key_and_stays_given_up(
    backend, tmp_path, request
):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            for attempt in (1, 2, 3):
                created = await app.create_task(_attempt("e1", attempt))
                assert created.id == f"evt-e1-{attempt}"
            # A duplicate task ID is still reported as a duplicate and costs nothing.
            with pytest.raises(ValueError, match="already exists"):
                await app.create_task(_attempt("e1", 3))
            state = await store.load_task_retry_key("event:e1")
            assert state.status is TaskRetryKeyStatus.ACTIVE
            assert state.attempts == 3 and state.attempts_remaining == 0
            assert state.latest_task_id == "evt-e1-3"
            assert state.budget == DEFAULT_TASK_RETRY_BUDGET

            with pytest.raises(TaskRetryBudgetExhausted) as refused:
                await app.create_task(_attempt("e1", 4))
            assert refused.value.state.status is TaskRetryKeyStatus.ACTIVE
            for attempt in (1, 2, 3):
                await store.fail_task(f"evt-e1-{attempt}", {"code": "failed"})
            given_up = await store.load_task_retry_key("event:e1")
            assert given_up.status is TaskRetryKeyStatus.GIVEN_UP
            assert given_up.given_up_reason is TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED
            assert given_up.refused_task_id is None
            assert given_up.attempts == 3
            assert await store.load_task("evt-e1-4") is None
            # Durable and terminal: a later, larger budget does not reopen the key.
            assert await store.load_task_retry_key("event:e1") == given_up
            with pytest.raises(TaskRetryBudgetExhausted):
                await store.create_task(
                    _attempt("e1", 5, retry_budget=TaskRetryBudget(max_attempts=50))
                )
            assert await store.load_task("evt-e1-5") is None
            assert (await store.load_task_retry_key("event:e1")).refused_task_id is None

            # Separate keys and independent unkeyed jobs have separate event budgets.
            await app.create_task(_attempt("e2", 1))
            await app.create_task(TaskCreate(task_id="plain", type="support.webhook_event"))
            assert (await store.load_task_retry_key("event:e2")).attempts == 1
            assert await store.load_task_retry_key("event:never") is None
            ids = {task.id for task in await store.list_tasks(TaskQuery(limit=100))}
            assert ids == {"evt-e1-1", "evt-e1-2", "evt-e1-3", "evt-e2-1", "plain"}
        finally:
            await app.aclose()
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_authors_can_raise_or_disable_the_budget(backend, tmp_path, request):
    async def scenario():
        store = _store(backend, tmp_path, request)
        raised = TaskRetryBudget(max_attempts=5, max_elapsed_seconds=4 * 3600)
        for attempt in range(1, 6):
            await store.create_task(_attempt("raised", attempt, retry_budget=raised))
        with pytest.raises(TaskRetryBudgetExhausted):
            await store.create_task(_attempt("raised", 6, retry_budget=raised))

        unlimited = TaskRetryBudget.unlimited()
        for attempt in range(1, 12):
            await store.create_task(_attempt("free", attempt, retry_budget=unlimited))
        state = await store.load_task_retry_key("event:free")
        assert state.status is TaskRetryKeyStatus.ACTIVE
        assert state.attempts == 11
        assert state.attempts_remaining is None and state.elapsed_deadline is None

        # The first admission fixes policy; later submissions cannot reset its budget.
        for attempt in (1, 2, 3):
            await store.create_task(_attempt("later", attempt))
        with pytest.raises(ValueError, match="fixed by its first admission"):
            await store.create_task(_attempt("later", 4, retry_budget=raised))
        assert (await store.load_task_retry_key("event:later")).attempts == 3
        await _close(store)

    asyncio.run(scenario())


def test_elapsed_budget_is_measured_from_the_first_task_with_store_time():
    now = [datetime(2026, 10, 24, 12, tzinfo=UTC)]
    store = InMemoryTaskStore(clock=lambda: now[0])
    budget = TaskRetryBudget(max_attempts=None, max_elapsed_seconds=1800)

    async def scenario():
        await store.create_task(_attempt("slow", 1, retry_budget=budget))
        now[0] += timedelta(minutes=29)
        await store.create_task(_attempt("slow", 2, retry_budget=budget))
        now[0] += timedelta(minutes=1)
        with pytest.raises(TaskRetryBudgetExhausted) as refused:
            await store.create_task(_attempt("slow", 3, retry_budget=budget))
        state = refused.value.state
        assert state.given_up_reason is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
        assert state.elapsed_deadline == datetime(2026, 10, 24, 12, 30, tzinfo=UTC)
        assert state.given_up_at == now[0]

    asyncio.run(scenario())


def test_retry_key_shape_is_validated():
    assert TaskCreate(type="job", retry_budget=TaskRetryBudget()).retry_key is None
    with pytest.raises(ValueError, match="unattached queue work"):
        TaskCreate(type="job", session_id="s1", retry_budget=TaskRetryBudget())
    with pytest.raises(ValueError, match="unattached queue work"):
        TaskCreate(type="job", session_id="s1", retry_key="k")
    with pytest.raises(ValueError, match="at most"):
        TaskCreate(type="job", retry_key="k" * 257)
    with pytest.raises(ValueError):
        TaskRetryBudget(max_attempts=0)
    with pytest.raises(ValueError, match="graphs or groups"):
        TaskGraphNode(task=TaskCreate(task_id="n1", type="job", retry_key="k"))
    # Unkeyed requests serialize exactly as before, so retained creation digests hold.
    assert "retry_key" not in TaskCreate(type="job").model_dump(mode="json")


def test_a_custom_store_without_retry_keys_refuses_a_keyed_task():
    class LegacyStore(InMemoryTaskStore):
        supports_task_retry_keys = False

    async def scenario():
        app = CayuApp(task_store=LegacyStore(), enable_logging=False)
        with pytest.raises(NotImplementedError, match="retry keys"):
            await app.create_task(_attempt("x", 1))
        await app.aclose()

    asyncio.run(scenario())


def test_budget_stops_a_sweeper_that_resubmits_a_hung_event():
    """Each task hits the handler time limit; the sweeper's fourth task is refused."""

    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    handled: list[str] = []

    async def hung_handler(app, task, worker_id):
        handled.append(task.id)
        await asyncio.Event().wait()

    async def scenario():
        stop = asyncio.Event()
        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                hung_handler,
                worker_id="w",
                poll_interval_s=0.02,
                stop=stop,
                execution_timeout_s=0.2,
                execution_grace_s=0.05,
            )
        )
        attempt = 1
        await app.create_task(_attempt("hang", attempt))
        given_up = None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10
        while given_up is None and loop.time() < deadline:
            # The sweeper authors write: a terminal task means "try the event again".
            current = await store.load_task(f"evt-hang-{attempt}")
            if current.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
                try:
                    attempt += 1
                    await app.create_task(_attempt("hang", attempt))
                except TaskRetryBudgetExhausted as exhausted:
                    given_up = exhausted.state
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(worker, 5)
        assert given_up is not None
        assert given_up.status is TaskRetryKeyStatus.GIVEN_UP
        assert handled == ["evt-hang-1", "evt-hang-2", "evt-hang-3"]
        for task_id in handled:
            task = await store.load_task(task_id)
            assert task.status is TaskStatus.FAILED
            assert task.error["error"] == "ExecutionDeadlineExceeded"
        await app.aclose()

    asyncio.run(scenario())


def test_cayu_inspect_shows_a_given_up_retry_key(tmp_path: Path, monkeypatch, capsys):
    database = tmp_path / "tasks.sqlite"

    async def give_up():
        store = SQLiteTaskStore(database)
        for attempt in (1, 2, 3):
            task = await store.create_task(_attempt("seen", attempt))
            await store.fail_task(task.id, {"code": "failed"})
        await store.close()

    asyncio.run(give_up())
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "retry_key_project:build_app"\n', encoding="utf-8"
    )
    (tmp_path / "retry_key_project.py").write_text(
        f"""from cayu import CayuApp
from cayu.storage import SQLiteTaskStore


def build_app():
    return CayuApp(task_store=SQLiteTaskStore({str(database)!r}), enable_logging=False)
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("retry_key_project", None)

    assert main(["inspect", "--retry-key", "event:seen", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)["retry_key"]
    assert shown["status"] == "given_up"
    assert shown["given_up_reason"] == "attempts_exhausted"
    assert shown["attempts"] == 3 and shown["attempts_remaining"] == 0
    assert shown["latest_task_id"] == "evt-seen-3"
    assert shown["refused_task_id"] is None

    assert main(["inspect", "--retry-key", "event:seen", "--table"]) == 0
    text = capsys.readouterr().out
    assert "Retry key event:seen: given_up" in text
    assert "Gave up:" in text and "attempts_exhausted" in text

    assert main(["inspect", "--retry-key", "event:other", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "SUBJECT_NOT_FOUND"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_scheduled_keyed_task_past_the_budget_keeps_the_exception_and_state(
    backend, tmp_path, request
):
    """A sweeper that schedules each retry with a backoff still sees the refusal."""

    from cayu.tasks import TaskSchedulePolicy

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = CayuApp(task_store=store, enable_logging=False)
        budget = TaskRetryBudget(max_attempts=2)
        try:
            for attempt in (1, 2, 3):
                scheduled = _attempt(
                    "later",
                    attempt,
                    retry_budget=budget,
                    available_at=datetime.now(UTC) + timedelta(seconds=attempt),
                    schedule_policy=TaskSchedulePolicy(),
                    restartable=False,
                )
                if attempt < 3:
                    await app.create_task(scheduled)
                    continue
                with pytest.raises(TaskRetryBudgetExhausted) as refused:
                    await app.create_task(scheduled)
            state = refused.value.state
            assert state.status is TaskRetryKeyStatus.ACTIVE
            assert state.given_up_reason is None
            assert state.refused_task_id is None
            assert await store.load_task_retry_key("event:later") == state
        finally:
            await app.aclose()
            await _close(store)

    asyncio.run(scenario())


def test_postgres_elapsed_budget_uses_store_time_after_the_key_lock(request):
    """A creator that waits for the key lock is judged at the time it gets it."""

    import psycopg

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")
    now = [datetime(2026, 10, 24, 12, tzinfo=UTC)]
    budget = TaskRetryBudget(max_attempts=None, max_elapsed_seconds=60)

    async def scenario():
        store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=lambda: now[0])
        try:
            await store.create_task(_attempt("locked", 1, retry_budget=budget))
            now[0] += timedelta(seconds=59)
            async with await psycopg.AsyncConnection.connect(dsn) as holder:
                await holder.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 449))",
                    ("cayu:verified-work:retry-key:event:locked",),
                )
                creator = asyncio.create_task(
                    store.create_task(_attempt("locked", 2, retry_budget=budget))
                )
                async with await psycopg.AsyncConnection.connect(dsn) as observer:
                    for _ in range(500):
                        cursor = await observer.execute(
                            "SELECT count(*) FROM pg_locks "
                            "WHERE locktype = 'advisory' AND NOT granted"
                        )
                        if (await cursor.fetchone())[0]:
                            break
                        await asyncio.sleep(0.01)
                    else:
                        raise AssertionError("The creator never waited for the key lock.")
                now[0] += timedelta(seconds=3)  # The wait outlasts the budget.
                await holder.rollback()
            with pytest.raises(TaskRetryBudgetExhausted) as refused:
                await creator
            state = refused.value.state
            assert state.given_up_reason is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
            assert state.given_up_at == now[0]
            assert await store.load_task("evt-locked-2") is None
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("secret", ['db"password', "plainpassword"])
def test_boundary_withholds_a_refusal_whose_state_holds_a_secret(secret):
    from cayu.runtime._task_store_operation_boundary import _detached_task_store_failure
    from cayu.tasks.retry_budgets import TaskRetryKeyState
    from cayu.vaults.redaction import SecretRedactor

    at = datetime(2026, 10, 24, 12, tzinfo=UTC)
    state = TaskRetryKeyState(
        retry_key="event:secret",
        status=TaskRetryKeyStatus.GIVEN_UP,
        attempts=1,
        budget=TaskRetryBudget(max_attempts=1),
        first_attempt_at=at,
        latest_attempt_at=at,
        latest_task_id=f"evt-{secret}",
        given_up_at=at,
        given_up_reason=TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED,
        refused_task_id="evt-2",
    )
    error = TaskRetryBudgetExhausted(state)
    kept = _detached_task_store_failure(
        error, operation_name="Task creation", redactor=SecretRedactor()
    )
    assert type(kept) is TaskRetryBudgetExhausted and kept.state == state
    withheld = _detached_task_store_failure(
        error, operation_name="Task creation", redactor=SecretRedactor([secret])
    )
    assert type(withheld) is not TaskRetryBudgetExhausted
    assert secret not in str(withheld)
