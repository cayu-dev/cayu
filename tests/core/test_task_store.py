from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from cayu import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskStore,
)

StoreFactory = Callable[[object], TaskStore]


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_create_load_and_copy_boundary(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        request_input = {"invoice_id": "inv_123", "lines": [{"amount": 25}]}
        task = await store.create_task(
            TaskCreate(
                task_id="task_invoice",
                type="process_invoice",
                title="Process invoice",
                description="Extract and post invoice fields.",
                session_id="sess_invoice",
                assigned_agent_name="invoice_agent",
                input=request_input,
                metadata={"source": "webhook"},
            )
        )
        request_input["lines"][0]["amount"] = 999

        loaded = await store.load_task("task_invoice")
        assert loaded is not None
        assert task.status == TaskStatus.PENDING
        assert loaded.input == {"invoice_id": "inv_123", "lines": [{"amount": 25}]}
        assert loaded.metadata == {"source": "webhook"}

        loaded.input["invoice_id"] = "mutated"
        loaded_again = await store.load_task("task_invoice")
        assert loaded_again is not None
        assert loaded_again.input["invoice_id"] == "inv_123"
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_lifecycle_and_terminal_guards(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_lifecycle", type="analyze_repository"))

        running = await store.start_task("task_lifecycle", session_id="sess_analysis")
        assert running.status == TaskStatus.RUNNING
        assert running.session_id == "sess_analysis"
        assert running.started_at is not None
        assert running.completed_at is None

        completed = await store.complete_task(
            "task_lifecycle",
            {"summary": "done"},
        )
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == {"summary": "done"}
        assert completed.error is None
        assert completed.completed_at is not None

        with pytest.raises(ValueError, match="already terminal"):
            await store.fail_task("task_lifecycle", {"message": "too late"})

        with pytest.raises(KeyError, match="Task not found"):
            await store.start_task("missing_task")

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_list_tasks_with_filters_and_pagination(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(
            TaskCreate(
                task_id="task_1",
                type="process_invoice",
                session_id="sess_1",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_2",
                type="process_invoice",
                session_id="sess_2",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_3",
                type="review_report",
                parent_task_id="task_2",
                assigned_agent_name="reviewer",
            )
        )
        await store.start_task("task_1")
        await store.complete_task("task_2", {"posted": True})

        invoice_tasks = await store.list_tasks(
            TaskQuery(type="process_invoice", order_by=TaskOrder.CREATED_AT_ASC)
        )
        invoice_agent_tasks = await store.list_tasks(
            TaskQuery(
                assigned_agent_name="invoice_agent",
                order_by=TaskOrder.CREATED_AT_ASC,
            )
        )
        completed_tasks = await store.list_tasks(TaskQuery(status=TaskStatus.COMPLETED))
        child_tasks = await store.list_tasks(TaskQuery(parent_task_id="task_2"))
        paged_tasks = await store.list_tasks(
            TaskQuery(limit=1, offset=1, order_by=TaskOrder.CREATED_AT_ASC)
        )

        assert [task.id for task in invoice_tasks] == ["task_1", "task_2"]
        assert [task.id for task in invoice_agent_tasks] == ["task_1", "task_2"]
        assert [task.id for task in completed_tasks] == ["task_2"]
        assert [task.id for task in child_tasks] == ["task_3"]
        assert [task.id for task in paged_tasks] == ["task_2"]
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_duplicate_tasks_and_invalid_payloads(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="JSON-compatible"):
            await store.complete_task("task_duplicate", {"bad": object()})

        with pytest.raises(ValueError, match="JSON object"):
            await store.fail_task("task_duplicate", ["not", "an", "object"])  # type: ignore[arg-type]

        await _close_store(store)

    asyncio.run(run_store_operations())


def test_sqlite_task_store_persists_tasks_across_reopen(tmp_path):
    db_path = tmp_path / "tasks.sqlite"
    store = SQLiteTaskStore(db_path)

    async def create_task() -> None:
        await store.create_task(
            TaskCreate(
                task_id="task_persisted",
                type="process_invoice",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.start_task("task_persisted", session_id="sess_persisted")
        await store.fail_task("task_persisted", {"message": "external API failed"})
        await store.close()

    asyncio.run(create_task())

    reopened = SQLiteTaskStore(db_path)

    async def assert_persisted_task() -> None:
        task = await reopened.load_task("task_persisted")
        assert task is not None
        assert task.type == "process_invoice"
        assert task.status == TaskStatus.FAILED
        assert task.session_id == "sess_persisted"
        assert task.error == {"message": "external API failed"}
        assert task.started_at is not None
        assert task.completed_at is not None
        await reopened.close()

    asyncio.run(assert_persisted_task())


def test_sqlite_task_store_rejects_stale_cross_connection_transitions(tmp_path):
    db_path = tmp_path / "tasks.sqlite"
    first = SQLiteTaskStore(db_path)
    second = SQLiteTaskStore(db_path)

    async def run_store_operations() -> None:
        await first.create_task(TaskCreate(task_id="task_claim", type="demo"))

        await first.start_task("task_claim", session_id="session_one")
        with pytest.raises(ValueError, match="cannot transition to running"):
            await second.start_task("task_claim", session_id="session_two")

        completed = await second.complete_task("task_claim", {"ok": True})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.session_id == "session_one"

        with pytest.raises(ValueError, match="already terminal"):
            await first.fail_task("task_claim", {"message": "too late"})

        await first.close()
        await second.close()

    asyncio.run(run_store_operations())


def _make_store(store_factory: StoreFactory, tmp_path) -> TaskStore:
    if store_factory is SQLiteTaskStore:
        return SQLiteTaskStore(tmp_path / "tasks.sqlite")
    return store_factory()


async def _close_store(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()
