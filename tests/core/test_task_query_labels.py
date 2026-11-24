"""``TaskQuery.labels``: list, latest and index reads by the labels a task was created with."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.index import TaskIndexQuery
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery, copy_task_query
from cayu.tasks.store import TaskStore


async def _seed(store: TaskStore) -> None:
    for task_id, labels in (
        ("a-1", {"person": "ana", "job": "close"}),
        ("a-2", {"person": "ana"}),
        ("b-1", {"person": "ben", "job": "close"}),
        ("none", {}),
    ):
        await store.create_task(TaskCreate(task_id=task_id, type="scan", labels=labels))


async def _check(store: TaskStore) -> None:
    await _seed(store)

    async def listed(**labels: str) -> list[str]:
        return sorted(task.id for task in await store.list_tasks(TaskQuery(labels=labels)))

    assert await listed(person="ana") == ["a-1", "a-2"]
    assert await listed(person="ana", job="close") == ["a-1"]
    assert await listed(job="close") == ["a-1", "b-1"]
    assert await listed(person="nobody") == []
    assert await listed() == ["a-1", "a-2", "b-1", "none"]

    index = await store.query_task_index(TaskIndexQuery(filters=TaskQuery(labels={"job": "close"})))
    assert sorted(task.id for task in index.items) == ["a-1", "b-1"]
    assert index.total_count == 2
    assert index.counts_by_status.pending == 2

    # A predicate read visits the index with the store's filter clauses.
    visited = await store.query_task_index(
        TaskIndexQuery(filters=TaskQuery(labels={"person": "ana"})), matches=lambda task: True
    )
    assert sorted(task.id for task in visited.items) == ["a-1", "a-2"]

    with pytest.raises(ValueError, match="labels"):
        await store.claim_task("worker", TaskQuery(labels={"person": "ana"}), lease_seconds=30)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_tasks_are_listed_by_their_labels(backend: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        store: TaskStore = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "t.db")
        )
        await _check(store)

    asyncio.run(scenario())


def test_postgres_lists_tasks_by_their_labels(postgres_dsn: str) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    async def scenario() -> None:
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store._ensure_ready()
            async with store._connection() as conn, conn.cursor() as cur:
                await cur.execute("TRUNCATE cayu_tasks CASCADE")
                await conn.commit()
            await _check(store)
        finally:
            await store.close()

    asyncio.run(scenario())


def test_label_filters_are_validated_and_survive_copies() -> None:
    query = TaskQuery(labels={"person": "ana"})
    assert copy_task_query(query).labels == {"person": "ana"}
    assert "labels" not in TaskQuery().model_dump()
    with pytest.raises(ValidationError):
        TaskQuery(labels={"person": 1})  # type: ignore[dict-item]
