"""Task progress counters: one contract suite for memory, SQLite and PostgreSQL."""

from __future__ import annotations

import asyncio
import sqlite3
from uuid import uuid4

import pytest

from cayu.storage import SQLiteTaskStore
from cayu.storage import migrations as schema
from cayu.storage._task_progress_schema import TASK_PROGRESS_REVISION
from cayu.storage.task_progress_sqlite import SQLiteTaskProgressStore
from cayu.tasks.progress import (
    TASK_PROGRESS_MAX_COUNTS,
    InMemoryTaskProgressStore,
    TaskProgressStore,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def open_store(request, tmp_path):
    """Return a factory; each call opens a new store instance on one database."""

    backend = request.param
    address = (
        request.getfixturevalue("postgres_dsn")
        if backend == "postgres"
        else str(tmp_path / "progress.sqlite")
    )
    memory = InMemoryTaskProgressStore()
    opened: list[TaskProgressStore] = []

    def factory() -> TaskProgressStore:
        if backend == "memory":
            return memory
        if backend == "sqlite":
            store: TaskProgressStore = SQLiteTaskProgressStore(address)
        else:
            from cayu.storage.task_progress_postgres import PostgresTaskProgressStore

            store = PostgresTaskProgressStore(address, schema_mode=schema.SchemaMode.CREATE)
        opened.append(store)
        return store

    yield factory
    for store in opened:
        await store.close()


async def test_progress_is_set_added_read_and_deleted(open_store) -> None:
    store = open_store()
    task_id = f"review-{uuid4().hex}"
    assert await store.load_progress(task_id) is None

    started = await store.set_progress(task_id, total=40)
    assert (started.done, started.total, started.counts) == (0, 40, {})
    await store.add_progress(task_id, done=1, counts={"approved": 1})
    await store.add_progress(task_id, done=2, counts={"approved": 1, "failed": 1})
    # Another instance on the same database reads what this one wrote.
    progress = await open_store().load_progress(task_id)
    assert progress is not None
    assert (progress.done, progress.total, progress.counts) == (3, 40, {"approved": 2, "failed": 1})
    assert progress.updated_at >= started.updated_at

    replaced = await store.set_progress(task_id, done=5, counts={"approved": 5})
    assert (replaced.done, replaced.total, replaced.counts) == (5, 40, {"approved": 5})
    assert await store.delete_progress(task_id) is True
    assert await store.delete_progress(task_id) is False
    assert await store.load_progress(task_id) is None


async def test_concurrent_additions_are_all_counted(open_store) -> None:
    task_id = f"review-{uuid4().hex}"
    stores = [open_store() for _ in range(4)]
    await asyncio.gather(
        *(
            store.add_progress(task_id, done=1, counts={"approved": 1})
            for store in stores
            for _ in range(5)
        )
    )
    progress = await stores[0].load_progress(task_id)
    assert progress is not None and progress.done == 20 and progress.counts == {"approved": 20}


async def test_bad_values_are_refused_and_nothing_is_written(open_store) -> None:
    store = open_store()
    task_id = f"review-{uuid4().hex}"
    for call in (
        lambda: store.set_progress(task_id, done=-1),
        lambda: store.set_progress(task_id, total=True),
        lambda: store.set_progress(task_id, counts={"": 1}),
        lambda: store.add_progress(task_id, done=-1),
        lambda: store.set_progress(
            task_id, counts={f"o{index}": 1 for index in range(TASK_PROGRESS_MAX_COUNTS + 1)}
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            await call()
    assert await store.load_progress(task_id) is None


def test_the_progress_revision_is_additive_and_task_stores_do_not_need_it(tmp_path) -> None:
    revision = schema.revision(TASK_PROGRESS_REVISION)
    assert revision.kind is schema.RevisionKind.ADDITIVE
    current = tmp_path / "current-without-progress.sqlite"
    asyncio.run(SQLiteTaskStore(current).close())
    with sqlite3.connect(current) as connection:
        connection.execute("DROP TABLE cayu_task_progress")
    asyncio.run(SQLiteTaskStore(current, schema_mode=schema.SchemaMode.VALIDATE).close())
