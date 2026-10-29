"""Saved task step results: one contract suite for memory, SQLite and PostgreSQL."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from uuid import uuid4

import pytest

from cayu.storage import SQLiteTaskStepStore, SQLiteTaskStore
from cayu.storage import migrations as schema
from cayu.storage._task_step_schema import TASK_STEPS_REVISION
from cayu.tasks import (
    TASK_STEP_RESULT_MAX_BYTES,
    InMemoryTaskStepStore,
    TaskCreate,
    TaskStepRecord,
    TaskSteps,
    TaskStepStore,
    complete_managed_task,
    requeue_restartable_task,
    task_steps,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def open_store(request, tmp_path):
    """Return a factory; each call opens a new store instance on one database."""

    backend = request.param
    address = (
        request.getfixturevalue("postgres_dsn")
        if backend == "postgres"
        else str(tmp_path / "steps.sqlite")
    )
    memory = InMemoryTaskStepStore()
    opened: list[TaskStepStore] = []

    def factory() -> TaskStepStore:
        if backend == "memory":
            return memory
        if backend == "sqlite":
            store: TaskStepStore = SQLiteTaskStepStore(address)
        else:
            from cayu.storage import PostgresTaskStepStore

            store = PostgresTaskStepStore(address, schema_mode=schema.SchemaMode.CREATE)
        opened.append(store)
        return store

    yield factory
    for store in opened:
        await store.close()


async def test_save_load_and_first_save_wins(open_store) -> None:
    store = open_store()
    assert await store.load_step("task-1", "checkout") is None

    saved = await store.save_step("task-1", "checkout", {"sha": "abc", "files": [1, 2]})
    assert isinstance(saved, TaskStepRecord)
    assert (saved.task_id, saved.step, saved.result) == (
        "task-1",
        "checkout",
        {"sha": "abc", "files": [1, 2]},
    )
    assert isinstance(saved.saved_at, datetime) and saved.saved_at.tzinfo is not None

    again = await store.save_step("task-1", "checkout", {"sha": "different"})
    assert again == saved

    reopened = open_store()
    assert await reopened.load_step("task-1", "checkout") == saved
    assert await reopened.load_step("task-2", "checkout") is None


@pytest.mark.parametrize("result", [None, 0, "text", [1, "two"], {"nested": {"ok": True}}])
async def test_any_json_value_round_trips(open_store, result) -> None:
    store = open_store()
    task_id = f"task-json-{uuid4().hex}"  # PostgreSQL keeps one database per module
    assert (await store.save_step(task_id, "step", result)).result == result
    loaded = await store.load_step(task_id, "step")
    assert loaded is not None and loaded.result == result


async def test_list_orders_by_step_pages_and_isolates_tasks(open_store) -> None:
    store = open_store()
    for step in ("verify:F2", "propose:F1", "verify:F1"):
        await store.save_step("task-a", step, step)
    await store.save_step("task-b", "other", 1)

    assert [r.step for r in await store.list_steps("task-a")] == [
        "propose:F1",
        "verify:F1",
        "verify:F2",
    ]
    assert [r.step for r in await store.list_steps("task-a", limit=2)] == [
        "propose:F1",
        "verify:F1",
    ]
    assert [r.step for r in await store.list_steps("task-a", after="verify:F1")] == ["verify:F2"]
    assert await store.list_steps("missing") == []
    with pytest.raises(ValueError, match="limit"):
        await store.list_steps("task-a", limit=0)


async def test_delete_one_step_or_every_step_of_a_task(open_store) -> None:
    store = open_store()
    for step in ("a", "b", "c"):
        await store.save_step("task-d", step, step)
    await store.save_step("task-e", "a", "kept")

    assert await store.delete_steps("task-d", step="b") == 1
    assert await store.delete_steps("task-d", step="b") == 0
    assert [r.step for r in await store.list_steps("task-d")] == ["a", "c"]
    assert await store.delete_steps("task-d") == 2
    assert await store.list_steps("task-d") == []
    kept = await store.load_step("task-e", "a")
    assert kept is not None and kept.result == "kept"
    # A deleted step can be saved again with a new result.
    assert (await store.save_step("task-d", "a", "again")).result == "again"


async def test_results_are_bounded_and_must_be_json(open_store) -> None:
    store = open_store()
    with pytest.raises(ValueError):
        await store.save_step("task-big", "step", "x" * TASK_STEP_RESULT_MAX_BYTES)
    assert await store.load_step("task-big", "step") is None
    fits = "x" * (TASK_STEP_RESULT_MAX_BYTES - 2)  # two bytes for the JSON quotes
    assert (await store.save_step("task-big", "step", fits)).result == fits

    for value in (float("nan"), {1, 2}, ("tu", "ple"), object(), {1: "non-string key"}):
        with pytest.raises((TypeError, ValueError)):
            await store.save_step("task-bad", "step", value)
    assert await store.list_steps("task-bad") == []


@pytest.mark.parametrize(
    ("task_id", "step"),
    [
        ("", "step"),
        ("task", " "),
        ("task", " padded"),
        ("task", "x" * 513),
        (1, "step"),
        ("task", None),
    ],
)
async def test_keys_are_validated(open_store, task_id, step) -> None:
    store = open_store()
    with pytest.raises((TypeError, ValueError)):
        await store.save_step(task_id, step, 1)
    with pytest.raises((TypeError, ValueError)):
        await store.load_step(task_id, step)


async def test_run_calls_each_step_once_across_reruns(open_store) -> None:
    calls: list[str] = []

    async def checkout() -> dict:
        calls.append("checkout")
        return {"sha": "abc"}

    def propose() -> list:
        calls.append("propose")
        return ["F1", 3]

    async def handler(store: TaskStepStore) -> list:
        steps = TaskSteps(store, "task-run")
        return [await steps.run("checkout", checkout), await steps.run("propose:F1", propose)]

    first = await handler(open_store())
    second = await handler(open_store())

    assert first == second == [{"sha": "abc"}, ["F1", 3]]
    assert calls == ["checkout", "propose"]


async def test_failed_step_saves_nothing_and_forget_reruns(open_store) -> None:
    steps = task_steps(open_store(), "task-fail")
    attempts = 0

    async def flaky() -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        return attempts

    with pytest.raises(RuntimeError, match="boom"):
        await steps.run("flaky", flaky)
    assert await steps.load("flaky") is None
    assert await steps.run("flaky", flaky) == 2
    assert await steps.run("flaky", flaky) == 2

    assert await steps.forget("flaky") is True
    assert await steps.forget("flaky") is False
    await steps.save("keep", 1)
    with pytest.raises(TypeError):
        await steps.forget(None)  # type: ignore[arg-type]
    assert [r.step for r in await steps.saved()] == ["keep"]
    assert await steps.forget("keep") is True
    assert await steps.run("flaky", flaky) == 3

    await steps.save("other", {"ok": True})
    assert [r.step for r in await steps.saved()] == ["flaky", "other"]
    assert await steps.clear() == 2
    assert await steps.saved() == []


async def test_task_steps_accepts_a_task_or_its_id(tmp_path) -> None:
    store = InMemoryTaskStepStore()
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    try:
        task = await task_store.create_task(TaskCreate(task_id="task-object", type="job"))
    finally:
        await task_store.close()
    assert task_steps(store, task).task_id == "task-object"
    assert task_steps(store, "task-id").task_id == "task-id"
    with pytest.raises(TypeError):
        task_steps(store, 42)
    with pytest.raises(TypeError):
        TaskSteps(object(), "task")  # type: ignore[arg-type]


async def test_restarted_task_skips_steps_saved_before_the_restart(tmp_path) -> None:
    """Steps live beside the task in one SQLite file and survive a requeue."""

    path = tmp_path / "app.sqlite"
    task_store = SQLiteTaskStore(path)
    step_store = SQLiteTaskStepStore(path)
    calls: list[str] = []

    async def step(name: str) -> str:
        calls.append(name)
        return f"{name}-done"

    try:
        created = await task_store.create_task(TaskCreate(type="pipeline", restartable=True))
        first = await task_store.claim_task("worker-1")
        assert first is not None and first.id == created.id
        steps = task_steps(step_store, first)
        await steps.run("rank", lambda: step("rank"))
        # The first worker stops here and its task is requeued.
        requeued = await requeue_restartable_task(task_store, first, "worker-1")
        assert requeued.restart_count == 1

        second = await task_store.claim_task("worker-2")
        assert second is not None and second.id == created.id
        steps = task_steps(step_store, second)
        results = [await steps.run(name, lambda name=name: step(name)) for name in ("rank", "hunt")]
        await complete_managed_task(task_store, second, "worker-2", {"steps": results})
    finally:
        await step_store.close()
        await task_store.close()

    assert results == ["rank-done", "hunt-done"]
    assert calls == ["rank", "hunt"]


def test_step_revision_is_additive_and_only_step_stores_require_it(tmp_path) -> None:
    revision = schema.revision(TASK_STEPS_REVISION)
    assert revision.kind is schema.RevisionKind.ADDITIVE
    assert revision.compatible_from == schema.revision(TASK_STEPS_REVISION - 1).compatible_from

    # Task stores never use the step table: one validates a current database without it.
    current = tmp_path / "current-without-steps.sqlite"
    asyncio.run(SQLiteTaskStore(current).close())
    with sqlite3.connect(current) as connection:
        connection.execute("DROP TABLE cayu_task_steps")
    asyncio.run(SQLiteTaskStore(current, schema_mode=schema.SchemaMode.VALIDATE).close())

    path = tmp_path / "revision-before-steps.sqlite"
    asyncio.run(SQLiteTaskStore(path).close())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE cayu_task_steps")
        connection.execute(
            "DELETE FROM cayu_schema_migrations WHERE revision >= ?", (TASK_STEPS_REVISION,)
        )
        connection.execute(f"PRAGMA user_version = {TASK_STEPS_REVISION - 1}")

    # A step store asks for the step revision itself. A task store's floor comes
    # only from a later breaking revision, never from the step table.
    with pytest.raises(schema.SchemaTooOld, match=f"requires >= {TASK_STEPS_REVISION}\\."):
        SQLiteTaskStepStore(path)
    task_floor = next(
        candidate.revision
        for candidate in reversed(schema.REVISIONS)
        if candidate.kind is schema.RevisionKind.BREAKING
    )
    assert task_floor > TASK_STEPS_REVISION
    with pytest.raises(schema.SchemaTooOld, match=f"requires >= {task_floor}\\."):
        asyncio.run(SQLiteTaskStore(path, schema_mode=schema.SchemaMode.VALIDATE).close())

    async def migrate_and_use() -> object:
        await SQLiteTaskStore(path, schema_mode=schema.SchemaMode.MIGRATE).close()
        store = SQLiteTaskStepStore(path, schema_mode=schema.SchemaMode.VALIDATE)
        try:
            await store.save_step("task", "step", {"ok": True})
            return (await store.load_step("task", "step")).result
        finally:
            await store.close()

    assert asyncio.run(migrate_and_use()) == {"ok": True}
