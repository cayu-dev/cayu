"""Postgres TaskStore parity tests.

Mirror the conformance assertions in ``test_task_store.py`` against a real
Dockerized Postgres. They skip automatically when Docker is unavailable.
"""

from __future__ import annotations

import asyncio

import pytest

from cayu import TaskCreate, TaskOrder, TaskQuery, TaskStatus

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
    "cayu_events",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


async def _truncate(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_store(dsn: str):
    from cayu import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    # Tests own a throwaway database and (re)create the schema each run.
    return PostgresTaskStore(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)


def _run(dsn: str, coro_factory) -> object:
    async def runner():
        await _truncate(dsn)
        store = _new_store(dsn)
        try:
            return await coro_factory(store)
        finally:
            await store.close()

    return asyncio.run(runner())


def test_postgres_task_store_create_load_and_copy_boundary(postgres_dsn):
    async def ops(store):
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

    _run(postgres_dsn, ops)


def test_postgres_task_store_lifecycle_and_terminal_guards(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_lifecycle", type="analyze_repository"))

        running = await store.start_task("task_lifecycle", session_id="sess_analysis")
        assert running.status == TaskStatus.RUNNING
        assert running.session_id == "sess_analysis"
        assert running.started_at is not None
        assert running.completed_at is None

        completed = await store.complete_task("task_lifecycle", {"summary": "done"})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == {"summary": "done"}
        assert completed.error is None
        assert completed.completed_at is not None

        with pytest.raises(ValueError, match="already terminal"):
            await store.fail_task("task_lifecycle", {"message": "too late"})

        with pytest.raises(KeyError, match="Task not found"):
            await store.start_task("missing_task")

    _run(postgres_dsn, ops)


def test_postgres_task_store_list_tasks_with_filters_and_pagination(postgres_dsn):
    async def ops(store):
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
            TaskQuery(assigned_agent_name="invoice_agent", order_by=TaskOrder.CREATED_AT_ASC)
        )
        completed_tasks = await store.list_tasks(TaskQuery(status=TaskStatus.COMPLETED))
        child_tasks = await store.list_tasks(TaskQuery(parent_task_id="task_2"))
        paged_tasks = await store.list_tasks(
            TaskQuery(limit=1, offset=1, order_by=TaskOrder.CREATED_AT_ASC)
        )

        assert [t.id for t in invoice_tasks] == ["task_1", "task_2"]
        assert [t.id for t in invoice_agent_tasks] == ["task_1", "task_2"]
        assert [t.id for t in completed_tasks] == ["task_2"]
        assert [t.id for t in child_tasks] == ["task_3"]
        assert [t.id for t in paged_tasks] == ["task_2"]

    _run(postgres_dsn, ops)


def test_postgres_task_store_reject_duplicate_tasks_and_invalid_payloads(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="JSON-compatible"):
            await store.complete_task("task_duplicate", {"bad": object()})

        with pytest.raises(ValueError, match="JSON object"):
            await store.fail_task("task_duplicate", ["not", "an", "object"])  # type: ignore[arg-type]

    _run(postgres_dsn, ops)


def test_postgres_task_store_cancel_and_persistence(postgres_dsn):
    async def ops(store):
        await store.create_task(
            TaskCreate(
                task_id="task_cancel",
                type="process_invoice",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.start_task("task_cancel", session_id="sess_cancel")
        cancelled = await store.cancel_task("task_cancel", {"reason": "operator stop"})
        assert cancelled.status == TaskStatus.CANCELLED
        assert cancelled.error == {"reason": "operator stop"}
        assert cancelled.started_at is not None
        assert cancelled.completed_at is not None

        # Reload from a fresh store/pool to confirm durability.
        reopened = _new_store(postgres_dsn)
        try:
            loaded = await reopened.load_task("task_cancel")
            assert loaded is not None
            assert loaded.status == TaskStatus.CANCELLED
            assert loaded.session_id == "sess_cancel"
            assert loaded.error == {"reason": "operator stop"}
        finally:
            await reopened.close()

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_stale_cross_pool_transitions(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_claim", type="demo"))

        second = _new_store(postgres_dsn)
        try:
            await store.start_task("task_claim", session_id="session_one")
            with pytest.raises(ValueError, match="cannot transition to running"):
                await second.start_task("task_claim", session_id="session_two")

            completed = await second.complete_task("task_claim", {"ok": True})
            assert completed.status == TaskStatus.COMPLETED
            assert completed.session_id == "session_one"

            with pytest.raises(ValueError, match="already terminal"):
                await store.fail_task("task_claim", {"message": "too late"})
        finally:
            await second.close()

    _run(postgres_dsn, ops)
