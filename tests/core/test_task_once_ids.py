"""create_task_once with task IDs the application chooses."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, create_task_once
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus
from cayu.tasks.store import TaskStore


def _store(backend: str, tmp_path: Path) -> TaskStore:
    if backend == "memory":
        return InMemoryTaskStore()
    return SQLiteTaskStore(tmp_path / "tasks.db")


def _ids(key: str, attempt: int) -> str:
    return f"{key}:{attempt}"


async def _fail_next(store: TaskStore, task_type: str) -> None:
    claimed = await store.claim_task("worker", TaskQuery(type=task_type))
    assert claimed is not None
    await store.fail_task(
        claimed.id,
        {"message": "failed"},
        worker_id="worker",
        claim_generation=claimed.claim_generation,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_chosen_ids_keep_one_task_per_key_and_attempt(backend: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _store(backend, tmp_path)
        request = TaskCreate(type="reset", retry_key="reset-owner-1")
        racing = await asyncio.gather(
            *(create_task_once(store, request, attempts=2, task_id_for=_ids) for _ in range(3))
        )
        assert sum(item.created for item in racing) == 1
        assert {item.task.id for item in racing if item.task is not None} == {"reset-owner-1:1"}

        await _fail_next(store, "reset")
        again = await asyncio.gather(
            *(create_task_once(store, request, task_id_for=_ids) for _ in range(3))
        )
        assert sum(item.created for item in again) == 1
        assert {item.task.id for item in again if item.task is not None} == {"reset-owner-1:2"}

        # The budget of two is used: the failed second attempt is returned.
        await _fail_next(store, "reset")
        last = await create_task_once(store, request, task_id_for=_ids)
        assert not last.created and last.task is not None
        assert (last.task.id, last.task.status) == ("reset-owner-1:2", TaskStatus.FAILED)
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_chosen_ids_are_checked() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="taken:1", type="other"))
        with pytest.raises(ValueError):
            await create_task_once(
                store, TaskCreate(type="reset", retry_key="taken"), task_id_for=_ids
            )

        with pytest.raises(TypeError, match="as a string"):
            await create_task_once(
                store,
                TaskCreate(type="reset", retry_key="k-type"),
                task_id_for=lambda key, attempt: attempt,  # type: ignore[arg-type,return-value]
            )

        same = TaskCreate(type="reset", retry_key="k-same")
        await create_task_once(store, same, attempts=2, task_id_for=lambda key, attempt: key)
        await _fail_next(store, "reset")
        with pytest.raises(ValueError, match="different ID for each attempt"):
            await create_task_once(store, same, task_id_for=lambda key, attempt: key)

        with pytest.raises(ValueError, match="leave task_id unset"):
            await create_task_once(
                store, TaskCreate(task_id="mine", type="reset", retry_key="k"), task_id_for=_ids
            )

    asyncio.run(scenario())
