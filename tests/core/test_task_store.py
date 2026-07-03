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
def test_task_stores_hold_resume_and_attention_states(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
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
        assert blocked.status_payload is not None
        blocked.status_payload["dependency"] = "mutated"

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

        paused = await store.pause_task(
            "task_pause_claim",
            reason="Worker shutting down",
        )
        assert paused.status == TaskStatus.PAUSED
        assert paused.worker_id is None
        assert paused.lease_expires_at is None

        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None

        resumed = await store.resume_task("task_blocked")
        assert resumed.status == TaskStatus.PENDING
        assert resumed.status_reason is None
        assert resumed.status_payload is None

        reloaded = await store.load_task("task_blocked")
        assert reloaded is not None
        assert reloaded.status == TaskStatus.PENDING
        assert reloaded.status_payload is None

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

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_do_not_hold_attached_running_tasks(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
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


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_claim_heartbeat_and_release_task(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
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
        assert first.status == TaskStatus.RUNNING
        assert first.worker_id == "worker_a"
        assert first.lease_expires_at is not None
        assert first.started_at is not None

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
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_start_task_attaches_claimed_task(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_claimed", type="review"))

        with pytest.raises(ValueError, match="not claimed by worker worker_a"):
            await store.start_task(
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
        assert claimed.status == TaskStatus.RUNNING
        assert claimed.worker_id == "worker_a"
        assert claimed.session_id is None
        assert claimed.lease_expires_at is not None

        with pytest.raises(ValueError, match="worker handoff requires session_id"):
            await store.start_task("task_claimed", worker_id="worker_a")
        with pytest.raises(ValueError, match="does not own"):
            await store.start_task("task_claimed", session_id="sess_wrong")
        with pytest.raises(ValueError, match="does not own"):
            await store.start_task(
                "task_claimed",
                session_id="sess_wrong",
                worker_id="worker_b",
            )

        started = await store.start_task(
            "task_claimed",
            session_id="sess_claimed",
            worker_id="worker_a",
        )
        assert started.status == TaskStatus.RUNNING
        assert started.session_id == "sess_claimed"
        assert started.worker_id == "worker_a"
        assert started.lease_expires_at == claimed.lease_expires_at

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_expired_claim_handoff(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_expired_handoff", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None

        await asyncio.sleep(1.05)
        with pytest.raises(ValueError, match="cannot transition to running from running"):
            await store.start_task("task_expired_handoff", session_id="sess_expired")
        with pytest.raises(ValueError, match="lease for worker worker_a has expired"):
            await store.heartbeat("task_expired_handoff", "worker_a")

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_release_after_session_attachment(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_attached_release", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        await store.start_task(
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

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_do_not_reclaim_attached_expired_leases(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_attached_expired", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None
        await store.start_task(
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

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reclaim_expired_leases(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
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
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_validate_worker_lease_inputs(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
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

        await _close_store(store)

    asyncio.run(run_store_operations())


def test_sqlite_task_store_concurrent_claims_do_not_duplicate_tasks(tmp_path):
    db_path = tmp_path / "tasks.sqlite"
    first = SQLiteTaskStore(db_path)
    second = SQLiteTaskStore(db_path)

    async def run_store_operations() -> None:
        await first.create_task(TaskCreate(task_id="task_a", type="review"))
        await first.create_task(TaskCreate(task_id="task_b", type="review"))

        claimed = await asyncio.gather(
            first.claim_task(
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

        loaded_a = await first.load_task("task_a")
        loaded_b = await second.load_task("task_b")
        assert loaded_a is not None
        assert loaded_b is not None
        assert {loaded_a.worker_id, loaded_b.worker_id} == {"worker_a", "worker_b"}
        assert loaded_a.id != loaded_b.id

        await first.close()
        await second.close()

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


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_claim_is_fifo_regardless_of_display_order(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_old", type="review"))
        await store.create_task(TaskCreate(task_id="task_new", type="review"))

        # Even when the query asks for a descending display order, claiming stays
        # FIFO and dispatches the oldest pending task first.
        first = await store.claim_task(
            "worker_a",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_DESC),
        )
        assert first is not None
        assert first.id == "task_old"

        second = await store.claim_task(
            "worker_b",
            TaskQuery(type="review", order_by=TaskOrder.UPDATED_AT_DESC),
        )
        assert second is not None
        assert second.id == "task_new"
        await _close_store(store)

    asyncio.run(run_store_operations())


def _make_store(store_factory: StoreFactory, tmp_path) -> TaskStore:
    if store_factory is SQLiteTaskStore:
        return SQLiteTaskStore(tmp_path / "tasks.sqlite")
    return store_factory()


async def _close_store(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()
