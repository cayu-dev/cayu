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
    "cayu_event_watcher_state",
    "cayu_events",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_session_operations",
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


def test_postgres_task_store_creates_running_task_atomically(postgres_dsn):
    async def ops(store):
        running = await store.create_running_task(
            TaskCreate(
                task_id="task_atomic_run",
                type="run",
                session_id="sess_atomic_run",
                input={"prompt": "hello"},
            )
        )

        assert running.status is TaskStatus.RUNNING
        assert running.session_id == "sess_atomic_run"
        assert running.started_at is not None
        assert running.completed_at is None
        assert await store.claim_task("worker_a") is None

        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_running_task(
                TaskCreate(
                    task_id="task_atomic_run",
                    type="duplicate",
                    session_id="sess_other",
                )
            )
        with pytest.raises(ValueError, match="session_id is required"):
            await store.create_running_task(TaskCreate(task_id="task_missing_session", type="run"))
        assert await store.load_task("task_missing_session") is None

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


def test_postgres_task_store_hold_resume_and_attention_states(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_blocked", type="review"))
        await store.create_task(TaskCreate(task_id="task_attention", type="review"))
        await store.create_task(TaskCreate(task_id="task_pause_claim", type="review"))

        blocked = await store.block_task(
            "task_blocked",
            reason="Waiting on vendor API",
            payload={"dependency": "vendor_api"},
        )
        assert blocked.status == TaskStatus.BLOCKED
        assert blocked.status_reason == "Waiting on vendor API"
        assert blocked.status_payload == {"dependency": "vendor_api"}

        attention = await store.mark_task_needs_attention(
            "task_attention",
            reason="Operator approval required",
            payload={"field": "amount"},
        )
        assert attention.status == TaskStatus.NEEDS_ATTENTION

        claimed = await store.claim_task(
            "worker_a",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
        )
        assert claimed is not None
        assert claimed.id == "task_pause_claim"

        paused = await store.pause_task("task_pause_claim", reason="Worker shutting down")
        assert paused.status == TaskStatus.PAUSED
        assert paused.worker_id is None
        assert paused.lease_expires_at is None

        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None

        resumed = await store.resume_task("task_blocked")
        assert resumed.status == TaskStatus.PENDING
        assert resumed.status_reason is None
        assert resumed.status_payload is None

        claimed_after_resume = await store.claim_task("worker_c", TaskQuery(type="review"))
        assert claimed_after_resume is not None
        assert claimed_after_resume.id == "task_blocked"

        with pytest.raises(ValueError, match="not paused, blocked, or waiting"):
            await store.resume_task("task_blocked")

        escalated = await store.block_task(
            "task_attention",
            reason="Waiting on supervisor decision",
        )
        assert escalated.status == TaskStatus.BLOCKED
        assert escalated.status_reason == "Waiting on supervisor decision"
        assert escalated.status_payload is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_does_not_hold_attached_running_tasks(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_hold", type="review"))
        await store.start_task("task_attached_hold", session_id="sess_attached_hold")

        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.pause_task("task_attached_hold", reason="not allowed")
        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.block_task("task_attached_hold", reason="not allowed")
        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.mark_task_needs_attention("task_attached_hold", reason="not allowed")

        loaded = await store.load_task("task_attached_hold")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.status_reason is None
        assert loaded.status_payload is None

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
        search_tasks = await store.list_tasks(
            TaskQuery(q="invoice", order_by=TaskOrder.CREATED_AT_ASC)
        )
        search_parent_tasks = await store.list_tasks(
            TaskQuery(q="TASK_2", order_by=TaskOrder.CREATED_AT_ASC)
        )
        paged_tasks = await store.list_tasks(
            TaskQuery(limit=1, offset=1, order_by=TaskOrder.CREATED_AT_ASC)
        )

        assert [t.id for t in invoice_tasks] == ["task_1", "task_2"]
        assert [t.id for t in invoice_agent_tasks] == ["task_1", "task_2"]
        assert [t.id for t in completed_tasks] == ["task_2"]
        assert [t.id for t in child_tasks] == ["task_3"]
        assert [t.id for t in search_tasks] == ["task_1", "task_2"]
        assert [t.id for t in search_parent_tasks] == ["task_2", "task_3"]
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


def test_postgres_task_store_claim_heartbeat_and_release_task(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_a", type="review"))
        await store.create_task(TaskCreate(task_id="task_b", type="review"))
        await store.create_task(
            TaskCreate(task_id="task_session_linked", type="review", session_id="sess_linked")
        )

        first = await store.claim_task(
            "worker_a",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=300,
        )
        assert first is not None
        assert first.id == "task_a"
        assert first.status == TaskStatus.CLAIMED
        assert first.worker_id == "worker_a"
        assert first.lease_expires_at is not None
        assert first.started_at is None

        second = await store.claim_task(
            "worker_b",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=300,
        )
        assert second is not None
        assert second.id == "task_b"
        assert second.worker_id == "worker_b"

        assert await store.claim_task("worker_c", TaskQuery(type="review")) is None
        linked = await store.load_task("task_session_linked")
        assert linked is not None
        assert linked.status == TaskStatus.PENDING
        assert linked.worker_id is None

        heartbeat = await store.heartbeat("task_a", "worker_a", extend_seconds=600)
        assert heartbeat.lease_expires_at is not None
        assert heartbeat.lease_expires_at > first.lease_expires_at

        with pytest.raises(ValueError, match="does not own"):
            await store.heartbeat("task_a", "worker_b")

        released = await store.release_task("task_a", "worker_a")
        assert released.status == TaskStatus.PENDING
        assert released.worker_id is None
        assert released.lease_expires_at is None

        reclaimed = await store.claim_task("worker_c", TaskQuery(type="review"))
        assert reclaimed is not None
        assert reclaimed.id == "task_a"
        assert reclaimed.worker_id == "worker_c"

        completed = await store.complete_task("task_a", {"ok": True})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.worker_id is None
        assert completed.lease_expires_at is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_attach_task_starts_claimed_task(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_claimed", type="review"))

        with pytest.raises(ValueError, match="not claimed by worker worker_a"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_unclaimed",
                worker_id="worker_a",
            )
        unclaimed = await store.load_task("task_claimed")
        assert unclaimed is not None
        assert unclaimed.status == TaskStatus.PENDING
        assert unclaimed.session_id is None

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        assert claimed.status == TaskStatus.CLAIMED
        assert claimed.worker_id == "worker_a"
        assert claimed.session_id is None
        assert claimed.lease_expires_at is not None

        with pytest.raises(ValueError, match="session_id"):
            await store.attach_task("task_claimed", session_id="", worker_id="worker_a")
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_claimed", session_id="sess_wrong")
        with pytest.raises(ValueError, match="does not own"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_wrong",
                worker_id="worker_b",
            )

        started = await store.attach_task(
            "task_claimed",
            session_id="sess_claimed",
            worker_id="worker_a",
        )
        assert started.status == TaskStatus.RUNNING
        assert started.session_id == "sess_claimed"
        assert started.worker_id == "worker_a"
        assert started.lease_expires_at == claimed.lease_expires_at

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_expired_claim_handoff(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_expired_handoff", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None

        await asyncio.sleep(1.05)
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_expired_handoff", session_id="sess_expired")
        with pytest.raises(ValueError, match="lease for worker worker_a has expired"):
            await store.heartbeat("task_expired_handoff", "worker_a")

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_release_after_session_attachment(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_release", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        await store.attach_task(
            "task_attached_release",
            session_id="sess_attached",
            worker_id="worker_a",
        )

        with pytest.raises(ValueError, match="already attached to session sess_attached"):
            await store.release_task("task_attached_release", "worker_a")

        loaded = await store.load_task("task_attached_release")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.session_id == "sess_attached"
        assert loaded.worker_id == "worker_a"

    _run(postgres_dsn, ops)


def test_postgres_task_store_does_not_reclaim_attached_expired_leases(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_expired", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None
        await store.attach_task(
            "task_attached_expired",
            session_id="sess_attached_expired",
            worker_id="worker_a",
        )

        await asyncio.sleep(1.05)
        reclaimed = await store.reclaim_expired(query=TaskQuery(type="review"))
        assert reclaimed == []

        loaded = await store.load_task("task_attached_expired")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.session_id == "sess_attached_expired"
        assert loaded.worker_id == "worker_a"

    _run(postgres_dsn, ops)


def test_postgres_task_store_reclaim_expired_leases(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_expired", type="demo"))
        await store.create_task(TaskCreate(task_id="task_waiting", type="demo"))
        await store.claim_task(
            "worker_a",
            TaskQuery(type="demo", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=1,
        )

        await asyncio.sleep(1.05)
        reclaimed = await store.reclaim_expired(
            query=TaskQuery(type="demo"),
            max_reclaims=1,
        )
        assert [task.id for task in reclaimed] == ["task_expired"]
        assert reclaimed[0].status == TaskStatus.PENDING
        assert reclaimed[0].worker_id is None
        assert reclaimed[0].lease_expires_at is None

        loaded = await store.load_task("task_expired")
        assert loaded is not None
        assert loaded.status == TaskStatus.PENDING

        assert await store.reclaim_expired(query=TaskQuery(status=TaskStatus.PENDING)) == []

    _run(postgres_dsn, ops)


def test_postgres_task_store_validate_worker_lease_inputs(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_validate_worker", type="demo"))

        with pytest.raises(ValueError, match="lease_seconds must be >= 1"):
            await store.claim_task("worker_a", lease_seconds=0)
        with pytest.raises(TypeError, match="lease_seconds must be an integer"):
            await store.claim_task("worker_a", lease_seconds=True)  # type: ignore[arg-type]

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None

        with pytest.raises(ValueError, match="extend_seconds must be >= 1"):
            await store.heartbeat("task_validate_worker", "worker_a", extend_seconds=0)
        with pytest.raises(ValueError, match="max_reclaims must be >= 1"):
            await store.reclaim_expired(max_reclaims=0)
        with pytest.raises(ValueError, match="do not support session_id"):
            await store.claim_task("worker_b", TaskQuery(session_id="sess_1"))
        with pytest.raises(ValueError, match="do not support session_id"):
            await store.reclaim_expired(query=TaskQuery(session_id="sess_1"))
        with pytest.raises(ValueError, match="do not support limit"):
            await store.claim_task("worker_b", TaskQuery(limit=2))
        with pytest.raises(ValueError, match="do not support offset"):
            await store.reclaim_expired(query=TaskQuery(offset=1))

    _run(postgres_dsn, ops)


def test_postgres_task_store_concurrent_claims_do_not_duplicate_tasks(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_a", type="review"))
        await store.create_task(TaskCreate(task_id="task_b", type="review"))

        second = _new_store(postgres_dsn)
        try:
            claimed = await asyncio.gather(
                store.claim_task(
                    "worker_a",
                    TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
                ),
                second.claim_task(
                    "worker_b",
                    TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
                ),
            )
            claimed_ids = sorted(task.id for task in claimed if task is not None)
            worker_ids = sorted(task.worker_id for task in claimed if task is not None)

            assert claimed_ids == ["task_a", "task_b"]
            assert worker_ids == ["worker_a", "worker_b"]

            loaded_a = await store.load_task("task_a")
            loaded_b = await second.load_task("task_b")
            assert loaded_a is not None
            assert loaded_b is not None
            assert {loaded_a.worker_id, loaded_b.worker_id} == {"worker_a", "worker_b"}
            assert loaded_a.id != loaded_b.id
        finally:
            await second.close()

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
