"""Single-worker recovery must not run in a supervised multi-process worker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import complete_managed_task, run_task_worker, run_task_worker_until_idle
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import Task, TaskStatus


@pytest.fixture(autouse=True)
def clean_worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CAYU_WORKER_COUNT",
        "CAYU_WORKER_INDEX",
        "CAYU_SUPERVISOR_PID",
        "CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S",
        "CAYU_WORKER_RESTART_ABANDONED",
        "CAYU_WORKER_MAX_ABANDONED_RESTARTS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("until_idle", [False, True])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("declaration", ["code", "environment"])
def test_multiple_processes_refused_before_task_store_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    until_idle: bool,
    restart: bool,
    declaration: str,
) -> None:
    monkeypatch.setenv("CAYU_WORKER_COUNT", "2")
    monkeypatch.setenv("CAYU_WORKER_INDEX", "0")
    if declaration == "environment":
        monkeypatch.setenv("CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S", "0")

    async def scenario() -> None:
        store = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        task = await store.create_task(TaskCreate(type="job", restartable=True))
        before = await store.load_task(task.id)
        app = CayuApp(task_store=store, enable_logging=False)
        accesses: list[str] = []

        # Observe actual store access, not just handler dispatch. Reconciliation
        # and recovery must not have touched durable state before admission fails.
        for name in ("claim_task", "reclaim_expired", "list_tasks"):
            method = getattr(type(store), name)

            async def touched(
                *args: Any, _name: str = name, _method: Any = method, **kwargs: Any
            ) -> Any:
                accesses.append(_name)
                return await _method(*args, **kwargs)

            monkeypatch.setattr(type(store), name, touched)

        async def handle(app: CayuApp, task: Task, worker_id: str) -> None:
            await complete_managed_task(store, task, worker_id, {"ok": True})

        run = run_task_worker_until_idle if until_idle else run_task_worker
        with pytest.raises(ValueError, match="single worker process.*--processes 1"):
            await run(
                app,
                store,
                handle,
                worker_id="worker",
                max_tasks=1,
                reconcile_abandoned_after_s=0 if declaration == "code" else None,
                restart_abandoned=restart,
            )
        assert accesses == []
        assert await store.load_task(task.id) == before
        if backend == "sqlite":
            assert await SQLiteTaskStore(tmp_path / "tasks.db").load_task(task.id) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("count", ["", "0", "-1", "true", "False", "1.0", "many", "257"])
def test_invalid_process_count_cannot_authorize_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    count: str,
) -> None:
    monkeypatch.setenv("CAYU_WORKER_COUNT", count)

    async def scenario() -> None:
        async def handle(app: CayuApp, task: Task, worker_id: str) -> None:
            raise AssertionError("handler must not run")

        with pytest.raises(ValueError, match="CAYU_WORKER_COUNT"):
            await run_task_worker(
                CayuApp(enable_logging=False),
                InMemoryTaskStore(),
                handle,
                worker_id="worker",
                max_tasks=0,
                reconcile_abandoned_after_s=0,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "metadata",
    [
        {"CAYU_WORKER_INDEX": "0"},
        {"CAYU_SUPERVISOR_PID": "123"},
        {"CAYU_WORKER_COUNT": "1", "CAYU_WORKER_INDEX": "1"},
        {"CAYU_WORKER_COUNT": "1", "CAYU_WORKER_INDEX": "true"},
    ],
)
def test_incomplete_or_conflicting_supervisor_metadata_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, str],
) -> None:
    for name, value in metadata.items():
        monkeypatch.setenv(name, value)

    async def scenario() -> None:
        async def handle(app: CayuApp, task: Task, worker_id: str) -> None:
            raise AssertionError("handler must not run")

        with pytest.raises(ValueError, match="single worker process"):
            await run_task_worker_until_idle(
                CayuApp(enable_logging=False),
                InMemoryTaskStore(),
                handle,
                worker_id="worker",
                reconcile_abandoned_after_s=0,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("count,reconcile", [(None, 0), ("1", 0), ("2", None)])
@pytest.mark.parametrize("until_idle", [False, True])
def test_standalone_single_process_and_ordinary_multi_process_workers_still_handle_tasks(
    monkeypatch: pytest.MonkeyPatch,
    count: str | None,
    reconcile: float | None,
    until_idle: bool,
) -> None:
    if count is not None:
        monkeypatch.setenv("CAYU_WORKER_COUNT", count)
        monkeypatch.setenv("CAYU_WORKER_INDEX", "0")

    async def scenario() -> None:
        store = InMemoryTaskStore()
        task = await store.create_task(TaskCreate(type="job"))

        async def handle(app: CayuApp, task: Task, worker_id: str) -> None:
            await complete_managed_task(store, task, worker_id, {"ok": True})

        run = run_task_worker_until_idle if until_idle else run_task_worker
        await run(
            CayuApp(task_store=store, enable_logging=False),
            store,
            handle,
            worker_id="worker",
            max_tasks=1,
            reconcile_abandoned_after_s=reconcile,
        )
        completed = await store.load_task(task.id)
        assert completed is not None and completed.status is TaskStatus.COMPLETED

    asyncio.run(scenario())
