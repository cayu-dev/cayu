from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)
from tests.core.task_store_conformance import (
    assert_exact_claimed_task_cancellation_conformance,
    assert_interrupted_continuation_scan_bound_conformance,
    assert_task_claim_lost_conformance,
    assert_task_session_invocation_binding_conformance,
    assert_worker_terminalization_generation_conformance,
)
from tests.core.task_terminalization_conformance import (
    assert_attached_task_recovery_terminalization_conformance,
    assert_live_ordinary_cancellation_conformance,
    assert_owner_lost_ordinary_cancellation_reconciliation_conformance,
    assert_recovered_continuation_terminalization_conformance,
    assert_task_terminalization_acknowledgement_conformance,
    ordinary_cancellation_reconciliation_request,
)

from cayu import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskInterruptedHandoffConflict,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    interrupted_task_handoff_request,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    extract_durable_value_error,
)
from cayu.runtime.tasks import (
    _legacy_task_terminalization_request_sha256,
    _require_interrupted_task_handoff_authority,
    _task_terminalization_request_matches_sha256,
    prepare_interrupted_task_handoff,
    prepare_task_terminalization,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations

StoreFactory = Callable[[object], TaskStore]


async def _exact_task_lease(store: TaskStore, task_id: str) -> datetime:
    task = await store.load_task(task_id)
    assert task is not None
    assert task.lease_expires_at is not None
    return task.lease_expires_at


def _interrupted_handoff_request(
    task: Task,
    *,
    handoff_id: str,
    session_run_epoch: int = 1,
    worker_id: str | None = None,
) -> TaskInterruptedHandoffRequest:
    assert task.worker_id is not None
    assert task.lease_expires_at is not None
    assert task.session_id is not None
    assert task.session_instance_id is not None
    return TaskInterruptedHandoffRequest(
        task_id=task.id,
        worker_id=task.worker_id if worker_id is None else worker_id,
        lease_expires_at=task.lease_expires_at,
        session_id=task.session_id,
        session_instance_id=task.session_instance_id,
        session_run_epoch=session_run_epoch,
        handoff_id=handoff_id,
    )


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_bound_continuation_scans_before_applying_query_filters(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_interrupted_continuation_scan_bound_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_claim_exact_interrupted_continuation(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            for task_id in ("exact-continuation-a", "exact-continuation-b"):
                await store.create_task(TaskCreate(task_id=task_id, type="exact-continuation"))
                claimed = await store.claim_task(
                    f"prior-{task_id}",
                    TaskQuery(type="exact-continuation"),
                )
                assert claimed is not None and claimed.id == task_id
                attached = await store.attach_task(
                    task_id,
                    session_id=f"session-{task_id}",
                    session_invocation=await task_backed_session_invocation(
                        store,
                        task_id,
                        f"session-{task_id}",
                    ),
                    worker_id=claimed.worker_id,
                    lease_expires_at=claimed.lease_expires_at,
                )
                await store.release_interrupted_task_worker(
                    interrupted_task_handoff_request(attached, session_run_epoch=1)
                )

            handoff_id = str(uuid4())
            page = await store.claim_interrupted_task_continuation(
                "exact-continuation-worker",
                handoff_id=handoff_id,
                task_id="exact-continuation-b",
                scan_limit=1,
            )

            assert page.task is not None
            assert page.task.id == "exact-continuation-b"
            untouched = await store.load_task("exact-continuation-a")
            assert untouched is not None and untouched.worker_id is None
            replay = await store.claim_interrupted_task_continuation(
                "exact-continuation-worker",
                handoff_id=handoff_id,
                task_id="exact-continuation-b",
                scan_limit=1,
            )
            assert replay.replayed is True
            assert replay.task == page.task
            with pytest.raises(TaskClaimLost):
                await store.claim_interrupted_task_continuation(
                    "exact-continuation-worker",
                    handoff_id=handoff_id,
                    task_id="exact-continuation-a",
                    scan_limit=1,
                )
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


def test_sqlite_consumed_continuation_generation_survives_restart(tmp_path) -> None:
    db_path = tmp_path / "tasks.sqlite"

    async def run_store_operations() -> None:
        store = SQLiteTaskStore(db_path)
        handoff_id = str(uuid4())
        try:
            await store.create_task(TaskCreate(task_id="durable-generation", type="review"))
            claimed = await store.claim_task("prior-worker", TaskQuery(type="review"))
            assert claimed is not None
            attached = await store.attach_task(
                claimed.id,
                session_id="durable-generation-session",
                session_invocation=await task_backed_session_invocation(
                    store,
                    claimed.id,
                    "durable-generation-session",
                ),
                worker_id="prior-worker",
                lease_expires_at=claimed.lease_expires_at,
            )
            await store.release_interrupted_task_worker(
                interrupted_task_handoff_request(attached, session_run_epoch=1)
            )
            continuation = await store.claim_interrupted_task_continuation(
                "continuation-worker",
                TaskQuery(type="review"),
                handoff_id=handoff_id,
            )
            assert continuation.task is not None
            await store.release_interrupted_task_worker(
                interrupted_task_handoff_request(
                    continuation.task,
                    session_run_epoch=2,
                )
            )
        finally:
            await store.close()

        reopened = SQLiteTaskStore(db_path)
        try:
            with pytest.raises(TaskClaimLost):
                await reopened.claim_interrupted_task_continuation(
                    "continuation-worker",
                    TaskQuery(type="review"),
                    handoff_id=handoff_id,
                )
            replacement = await reopened.claim_interrupted_task_continuation(
                "continuation-worker",
                TaskQuery(type="review"),
                handoff_id=str(uuid4()),
            )
            assert replacement.task is not None
        finally:
            await reopened.close()

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_task_claim_lost_conformance(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_task_claim_lost_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_worker_terminalization_generation_conformance(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_worker_terminalization_generation_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_exact_claimed_cancellation_conformance(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_exact_claimed_task_cancellation_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_competing_terminalization_after_winner(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_conflict", type="review"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None
        winner = await store.terminalize_task(
            TaskTerminalizationRequest(
                task_id="task_conflict",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "winner"},
                idempotency_key="winner-key",
            )
        )

        with pytest.raises(TaskTerminalizationConflict):
            await store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id="task_conflict",
                    worker_id="worker_a",
                    lease_expires_at=claimed.lease_expires_at,
                    kind=TaskTerminalKind.FAILED,
                    error={"message": "loser"},
                    idempotency_key="loser-key",
                )
            )

        loaded = await store.load_task("task_conflict")
        assert loaded == winner
        assert loaded is not None
        assert loaded.status is TaskStatus.COMPLETED
        assert loaded.result == {"summary": "winner"}
        assert await store.load_task_terminalization_receipt("task_conflict", "loser-key") is None
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_wrong_worker_before_terminalization(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_wrong_worker", type="review"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None
        with pytest.raises(TaskClaimLost):
            await store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id="task_wrong_worker",
                    worker_id="worker_b",
                    lease_expires_at=claimed.lease_expires_at,
                    kind=TaskTerminalKind.COMPLETED,
                    result={"summary": "unauthorized"},
                    idempotency_key="wrong-worker",
                )
            )
        task = await store.load_task("task_wrong_worker")
        assert task is not None
        assert task.status is TaskStatus.CLAIMED
        assert (
            await store.load_task_terminalization_receipt("task_wrong_worker", "wrong-worker")
            is None
        )
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_store_retry_conformance_for_acknowledgement_failures(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await assert_task_terminalization_acknowledgement_conformance(store)
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_live_ordinary_cancellation_conformance(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_live_ordinary_cancellation_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_recovered_continuations_terminalize_all_ordinary_kinds(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_recovered_continuation_terminalization_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_attached_task_recovery_terminalization_conformance(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_attached_task_recovery_terminalization_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_owner_lost_ordinary_cancellation_reconciliation_conformance(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_owner_lost_ordinary_cancellation_reconciliation_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


def test_sqlite_ordinary_cancellation_reconciliation_replays_after_restart(tmp_path):
    async def run_store_operations() -> None:
        path = tmp_path / "ordinary-reconciliation-restart.sqlite"
        store = SQLiteTaskStore(path)
        await store.create_task(TaskCreate(task_id="ordinary_restart", type="review"))
        claimed = await store.claim_task("ordinary-prior-worker", lease_seconds=60)
        assert claimed is not None
        attached = await store.attach_task(
            claimed.id,
            session_id="ordinary-restart-session",
            session_invocation=await task_backed_session_invocation(
                store,
                claimed.id,
                "ordinary-restart-session",
            ),
            worker_id="ordinary-prior-worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        await store.release_interrupted_task_worker(
            interrupted_task_handoff_request(attached, session_run_epoch=1)
        )
        recovery_owner = (
            await store.claim_interrupted_task_continuation(
                "ordinary-restart-worker",
                handoff_id=str(uuid4()),
                lease_seconds=1,
            )
        ).task
        assert recovery_owner is not None
        assert recovery_owner.interrupted_handoff_id is not None
        requested = await store.cancel_task(recovery_owner.id, {"code": "operator"})
        request = ordinary_cancellation_reconciliation_request(requested)
        await asyncio.sleep(1.05)
        result = await store.reconcile_task_cancellation(request)
        await store.close()

        reopened = SQLiteTaskStore(path)
        try:
            assert await reopened.reconcile_task_cancellation(request) == result
            assert result.task.interrupted_handoff_id is None
            assert (
                await reopened.load_task_terminalization_receipt(
                    request.task_id,
                    request.cancellation_idempotency_key,
                )
                == result.terminalization_receipt
            )
        finally:
            await reopened.close()

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_unreconcilable_live_cancellation_identity(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await store.create_task(TaskCreate(task_id="ordinary_bounded", type="review"))
            oversized_worker = "w" * 1025
            claimed = await store.claim_task(oversized_worker)
            assert claimed is not None
            with pytest.raises(ValueError, match="worker_id must be at most"):
                await store.cancel_task(claimed.id, {"code": "operator"})
            unchanged = await store.load_task(claimed.id)
            assert unchanged == claimed
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_replay_exact_terminalization_after_lease_clearance(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_terminal", type="review"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None

        first_request = TaskTerminalizationRequest(
            task_id="task_terminal",
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done", "metrics": {"changed": 2, "checked": 4}},
            idempotency_key="terminal-attempt-1",
        )
        first = await store.terminalize_task(first_request)
        replayed = await store.terminalize_task(
            TaskTerminalizationRequest(
                task_id="task_terminal",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"metrics": {"checked": 4, "changed": 2}, "summary": "done"},
                idempotency_key="terminal-attempt-1",
            )
        )

        assert first == replayed
        assert replayed.status is TaskStatus.COMPLETED
        assert replayed.result == {
            "summary": "done",
            "metrics": {"changed": 2, "checked": 4},
        }
        assert replayed.worker_id is None
        assert replayed.lease_expires_at is None
        receipt = await store.load_task_terminalization_receipt(
            "task_terminal", "terminal-attempt-1"
        )
        assert receipt is not None
        assert receipt.request_sha256 == prepare_task_terminalization(first_request)[1]

        assert first.result is not None
        first.result["summary"] = "mutated"
        replayed_again = await store.terminalize_task(
            TaskTerminalizationRequest(
                task_id="task_terminal",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done", "metrics": {"changed": 2, "checked": 4}},
                idempotency_key="terminal-attempt-1",
            )
        )
        assert replayed_again.result == {
            "summary": "done",
            "metrics": {"changed": 2, "checked": 4},
        }
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_task_terminalization_digest_accepts_pre_lease_receipt() -> None:
    request = TaskTerminalizationRequest(
        task_id="task_terminal",
        worker_id="worker_a",
        lease_expires_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
        kind=TaskTerminalKind.COMPLETED,
        result={"summary": "done", "metrics": {"changed": 2, "checked": 4}},
        idempotency_key="terminal-attempt-1",
    )
    request, request_sha256 = prepare_task_terminalization(request)
    legacy_sha256 = _legacy_task_terminalization_request_sha256(request)

    assert request_sha256 != legacy_sha256
    assert legacy_sha256 == "f44314f4f13d93a708c544e83a90ecb2e2dea4d6dd7f4ceb0512b2f895d364a8"
    assert _task_terminalization_request_matches_sha256(
        request,
        request_sha256=request_sha256,
        candidate_sha256=legacy_sha256,
    )


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_expose_detached_terminalization_receipt(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        assert await store.load_task_terminalization_receipt("task_receipt", "failure-1") is None
        await store.create_task(TaskCreate(task_id="task_receipt", type="review"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None
        terminal_task = await store.terminalize_task(
            TaskTerminalizationRequest(
                task_id="task_receipt",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error={"message": "provider unavailable"},
                idempotency_key="failure-1",
            )
        )

        receipt = await store.load_task_terminalization_receipt("task_receipt", "failure-1")
        assert type(receipt) is TaskTerminalizationReceipt
        assert receipt.task_id == "task_receipt"
        assert receipt.idempotency_key == "failure-1"
        assert receipt.worker_id == "worker_a"
        assert receipt.kind is TaskTerminalKind.FAILED
        assert len(receipt.request_sha256) == 64
        assert receipt.task == terminal_task

        assert receipt.task.error is not None
        receipt.task.error["message"] = "mutated"
        loaded_again = await store.load_task_terminalization_receipt("task_receipt", "failure-1")
        assert loaded_again is not None
        assert loaded_again.task.error == {"message": "provider unavailable"}
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_bind_session_identity_to_invocation(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        try:
            await assert_task_session_invocation_binding_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run_store_operations())


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
def test_task_stores_create_running_task_atomically(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        running = await store.create_running_task(
            TaskCreate(
                task_id="task_atomic_run",
                type="run",
                session_id="sess_atomic_run",
                input={"prompt": "hello"},
            ),
            session_invocation=unattributed_session_invocation_binding("sess_atomic_run"),
        )

        assert running.status is TaskStatus.RUNNING
        assert running.session_id == "sess_atomic_run"
        assert running.started_at is not None
        assert running.completed_at is None
        assert await store.claim_task("worker_a") is None

        loaded = await store.load_task("task_atomic_run")
        assert loaded == running
        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_running_task(
                TaskCreate(
                    task_id="task_atomic_run",
                    type="duplicate",
                    session_id="sess_other",
                ),
                session_invocation=unattributed_session_invocation_binding("sess_other"),
            )
        with pytest.raises(ValueError, match="session_id is required"):
            await store.create_running_task(
                TaskCreate(task_id="task_missing_session", type="run"),
                session_invocation=unattributed_session_invocation_binding("sess_missing"),
            )
        assert await store.load_task("task_missing_session") is None
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_lifecycle_and_terminal_guards(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_lifecycle", type="analyze_repository"))

        running = await store.start_task(
            "task_lifecycle",
            session_id="sess_analysis",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_lifecycle",
                "sess_analysis",
            ),
        )
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
        await store.start_task(
            "task_attached_hold",
            session_id="sess_attached_hold",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_attached_hold",
                "sess_attached_hold",
            ),
        )

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
        await store.start_task(
            "task_1",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_1",
                "sess_1",
            ),
        )
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
def test_task_stores_revalidate_portable_values_before_atomic_mutation(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        poisoned_create = TaskCreate(
            task_id="task_poisoned_create",
            type="demo",
            input={"safe": True},
        )
        poisoned_create.input["bad"] = float("nan")
        with pytest.raises((DurableValueError, ValidationError)) as invalid_create:
            await store.create_task(poisoned_create)
        create_error = extract_durable_value_error(invalid_create.value)
        assert create_error is not None
        assert create_error.code == "non_finite_number"
        assert await store.load_task("task_poisoned_create") is None

        request = TaskCreate(
            task_id="task_portable_numbers",
            type="demo",
            input={"numbers": {"safe": True}},
            metadata={"numbers": {"safe": True}},
        )
        numbers = {
            "ordinary": 1.0,
            "negative_zero": -0.0,
            "large": 1e18,
            "fractional": 1e-7,
        }
        request.input["numbers"] = dict(numbers)
        request.metadata["numbers"] = dict(numbers)
        await store.create_task(request)

        with pytest.raises(DurableValueError) as invalid_result:
            await store.complete_task(
                "task_portable_numbers",
                {"bad": MAX_DURABLE_JSON_INTEGER + 1},
            )
        assert invalid_result.value.code == "integer_out_of_range"
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        with pytest.raises(DurableValueError) as invalid_reason:
            await store.pause_task("task_portable_numbers", reason="poisoned\x00reason")
        assert invalid_reason.value.code == "nul_character"
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        for invalid_text, code in (
            ("workload-secret-value\x00", "nul_character"),
            ("workload-secret-value\ud800", "unicode_surrogate"),
        ):
            with pytest.raises(DurableValueError) as invalid_session_id:
                await store.start_task(
                    "task_portable_numbers",
                    session_id=invalid_text,
                )
            assert invalid_session_id.value.code == code
            assert "workload-secret-value" not in str(invalid_session_id.value)

            forged_query = TaskQuery(q="safe")
            forged_query.q = invalid_text
            with pytest.raises(ValidationError) as invalid_query:
                await store.list_tasks(forged_query)
            query_error = extract_durable_value_error(invalid_query.value)
            assert query_error is not None
            assert query_error.code == code
            assert "workload-secret-value" not in str(invalid_query.value)

            pending = await store.load_task("task_portable_numbers")
            assert pending is not None
            assert pending.status is TaskStatus.PENDING
            assert pending.session_id is None

        forged_query = TaskQuery()
        forged_query.offset = MAX_DURABLE_JSON_INTEGER + 1
        with pytest.raises(ValidationError):
            await store.list_tasks(forged_query)
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        completed = await store.complete_task(
            "task_portable_numbers",
            {"numbers": numbers},
        )
        assert completed.status is TaskStatus.COMPLETED

        loaded = await store.load_task("task_portable_numbers")
        assert loaded is not None
        assert loaded.result is not None
        for value in (
            loaded.input["numbers"],
            loaded.metadata["numbers"],
            loaded.result["numbers"],
        ):
            assert value == {
                "ordinary": 1,
                "negative_zero": 0,
                "large": 1_000_000_000_000_000_000,
                "fractional": 1e-7,
            }
            assert type(value["ordinary"]) is int
            assert type(value["negative_zero"]) is int
            assert type(value["large"]) is int
            assert type(value["fractional"]) is float

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

        heartbeat = await store.heartbeat(
            "task_a",
            "worker_a",
            lease_expires_at=first.lease_expires_at,
            extend_seconds=600,
        )
        assert heartbeat.lease_expires_at is not None
        assert heartbeat.lease_expires_at > first.lease_expires_at

        with pytest.raises(ValueError, match="does not own"):
            await store.heartbeat(
                "task_a",
                "worker_b",
                lease_expires_at=heartbeat.lease_expires_at,
            )

        released = await store.release_task(
            "task_a",
            "worker_a",
            lease_expires_at=heartbeat.lease_expires_at,
        )
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
def test_task_stores_attach_task_starts_claimed_task(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_claimed", type="review"))

        with pytest.raises(ValueError, match="not claimed by worker worker_a"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_unclaimed",
                session_invocation=await task_backed_session_invocation(
                    store,
                    "task_claimed",
                    "sess_unclaimed",
                ),
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
            await store.attach_task(
                "task_claimed",
                session_id="",
                session_invocation=unattributed_session_invocation_binding("unused_session"),
                worker_id="worker_a",
            )
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_claimed", session_id="sess_wrong")
        with pytest.raises(ValueError, match="does not own"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_wrong",
                session_invocation=await task_backed_session_invocation(
                    store,
                    "task_claimed",
                    "sess_wrong",
                ),
                worker_id="worker_b",
            )

        started = await store.attach_task(
            "task_claimed",
            session_id="sess_claimed",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_claimed",
                "sess_claimed",
            ),
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
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
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_expired_handoff", session_id="sess_expired")
        with pytest.raises(TaskClaimLost, match="lease for worker worker_a has expired"):
            await store.heartbeat(
                "task_expired_handoff",
                "worker_a",
                lease_expires_at=claimed.lease_expires_at,
            )

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
        await store.attach_task(
            "task_attached_release",
            session_id="sess_attached",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_attached_release",
                "sess_attached",
            ),
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
        )

        with pytest.raises(ValueError, match="already attached to session sess_attached"):
            await store.release_task(
                "task_attached_release",
                "worker_a",
                lease_expires_at=claimed.lease_expires_at,
            )

        loaded = await store.load_task("task_attached_release")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.session_id == "sess_attached"
        assert loaded.worker_id == "worker_a"

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_release_attached_task_worker_without_requeueing(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="parent_review", type="review"))
        await store.complete_task("parent_review", {"status": "delegated"})
        await store.create_task(
            TaskCreate(
                task_id="task_attached_handoff",
                type="review",
                parent_task_id="parent_review",
                assigned_agent_name="reviewer",
                input={"pull_request": 382},
                metadata={"tenant": "acme"},
            )
        )
        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        attached = await store.attach_task(
            "task_attached_handoff",
            session_id="sess_attached_handoff",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_attached_handoff",
                "sess_attached_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
        )

        released = await store.release_attached_task_worker(
            "task_attached_handoff",
            "worker_a",
            lease_expires_at=attached.lease_expires_at,
        )

        assert released.status == TaskStatus.RUNNING
        assert released.session_id == "sess_attached_handoff"
        assert released.worker_id is None
        assert released.lease_expires_at is None
        assert released.parent_task_id == "parent_review"
        assert released.assigned_agent_name == "reviewer"
        assert released.input == {"pull_request": 382}
        assert released.metadata == {"tenant": "acme"}
        assert released.created_at == attached.created_at
        assert released.started_at == attached.started_at
        assert released.completed_at is None
        assert released.updated_at >= attached.updated_at

        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None
        assert await store.reclaim_expired(query=TaskQuery(type="review")) == []

        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_interrupted_task_handoff_is_exact_and_replay_safe(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_exact_handoff", type="review"))
        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        attached = await store.attach_task(
            "task_exact_handoff",
            session_id="session_exact_handoff",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_exact_handoff",
                "session_exact_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-exact",
        )

        receipt = await store.release_interrupted_task_worker(request)
        replayed = await store.release_interrupted_task_worker(request)

        assert type(receipt) is TaskInterruptedHandoffReceipt
        assert replayed == receipt
        assert receipt.request == request
        assert receipt.task.status is TaskStatus.RUNNING
        assert receipt.task.session_id == "session_exact_handoff"
        assert receipt.task.session_instance_id == attached.session_instance_id
        assert receipt.task.worker_id is None
        assert receipt.task.lease_expires_at is None
        assert (
            await store.load_interrupted_task_handoff_receipt(
                request.task_id,
                request.handoff_id,
            )
            == receipt
        )
        changed_fields = (
            {"worker_id": "worker_b"},
            {"lease_expires_at": request.lease_expires_at + timedelta(seconds=1)},
            {"session_id": "session_other_handoff"},
            {"session_instance_id": "11111111-1111-4111-8111-111111111111"},
            {"session_run_epoch": 2},
            {"handoff_id": "handoff-other"},
        )
        for changed in changed_fields:
            with pytest.raises(TaskInterruptedHandoffConflict):
                await store.release_interrupted_task_worker(request.model_copy(update=changed))
        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None
        assert await store.reclaim_expired(query=TaskQuery(type="review")) == []
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_base_direct_resume_keeps_legacy_custom_stores_compatible_and_recovery_safe() -> None:
    class LegacyCustomTaskStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = False
        load_direct_attached_task_resume = TaskStore.load_direct_attached_task_resume

    async def scenario() -> None:
        store = LegacyCustomTaskStore()
        await store.create_task(TaskCreate(task_id="legacy-direct", type="review"))
        claimed = await store.claim_task("legacy-worker")
        assert claimed is not None
        attached = await store.attach_task(
            "legacy-direct",
            session_id="legacy-session",
            session_invocation=await task_backed_session_invocation(
                store,
                "legacy-direct",
                "legacy-session",
            ),
            worker_id="legacy-worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        released = await store.release_attached_task_worker(
            attached.id,
            "legacy-worker",
            lease_expires_at=attached.lease_expires_at,
        )
        assert released.session_instance_id is not None
        loaded = await store.load_direct_attached_task_resume(
            released.id,
            session_id="legacy-session",
            session_instance_id=released.session_instance_id,
        )
        assert loaded == released

        recovery_store = InMemoryTaskStore()
        with pytest.raises(NotImplementedError, match="must implement an atomic"):
            await TaskStore.load_direct_attached_task_resume(
                recovery_store,
                released.id,
                session_id="legacy-session",
                session_instance_id=released.session_instance_id,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_point", ["before_commit", "after_commit"])
def test_in_memory_interrupted_task_handoff_faults_reconcile_exactly(
    failure_point: str,
) -> None:
    class FaultingStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.calls += 1
            if failure_point == "before_commit" and self.calls == 1:
                raise RuntimeError("fail before interrupted handoff commit")
            receipt = await super().release_interrupted_task_worker(request)
            if failure_point == "after_commit" and self.calls == 1:
                raise RuntimeError("fail after interrupted handoff commit")
            return receipt

    store = FaultingStore()

    async def scenario() -> None:
        await store.create_task(TaskCreate(task_id="task_memory_fault_handoff", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_memory_fault_handoff",
            session_id="session_memory_fault_handoff",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_memory_fault_handoff",
                "session_memory_fault_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(
                store,
                "task_memory_fault_handoff",
            ),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id=f"handoff-{failure_point}",
        )
        with pytest.raises(RuntimeError, match=f"{failure_point.split('_')[0]}.*commit"):
            await store.release_interrupted_task_worker(request)

        receipt = await store.load_interrupted_task_handoff_receipt(
            request.task_id,
            request.handoff_id,
        )
        if failure_point == "before_commit":
            assert receipt is None
            unchanged = await store.load_task(request.task_id)
            assert unchanged is not None
            assert unchanged.worker_id == request.worker_id
            assert unchanged.lease_expires_at == request.lease_expires_at
        else:
            assert receipt is not None
            assert receipt.task.worker_id is None

        replayed = await store.release_interrupted_task_worker(request)
        assert replayed == await store.load_interrupted_task_handoff_receipt(
            request.task_id,
            request.handoff_id,
        )
        assert replayed.task.worker_id is None
        assert replayed.task.session_id == request.session_id

    asyncio.run(scenario())


def test_interrupted_task_handoff_revalidates_mutated_request_before_serialization(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mutated-handoff-request-secret"

    class SecretBearingValue:
        def __repr__(self) -> str:
            return secret

        def __str__(self) -> str:
            return secret

    store = InMemoryTaskStore()

    async def scenario() -> BaseException:
        await store.create_task(TaskCreate(task_id="task_mutated_handoff", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_mutated_handoff",
            session_id="session_mutated_handoff",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_mutated_handoff",
                "session_mutated_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, "task_mutated_handoff"),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-mutated-request",
        )
        object.__setattr__(request, "task_id", SecretBearingValue())
        with pytest.raises(ValidationError) as raised:
            await store.release_interrupted_task_worker(request)
        return raised.value

    failure = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic = "\n".join(
        [
            str(failure),
            repr(failure),
            captured.out,
            captured.err,
            "\n".join(record.getMessage() for record in caplog.records),
            "\n".join(str(record.message) for record in recwarn),
        ]
    )
    assert secret not in diagnostic


def test_interrupted_task_handoff_receipt_copies_mutated_task_without_serializer_warning(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mutated-handoff-receipt-task-secret"

    class SecretBearingValue:
        def __repr__(self) -> str:
            return secret

        def __str__(self) -> str:
            return secret

    store = InMemoryTaskStore()

    async def scenario() -> BaseException:
        await store.create_task(TaskCreate(task_id="task_mutated_receipt", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_mutated_receipt",
            session_id="session_mutated_receipt",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_mutated_receipt",
                "session_mutated_receipt",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, "task_mutated_receipt"),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-mutated-receipt-task",
        )
        object.__setattr__(attached, "status", SecretBearingValue())
        with pytest.raises(ValidationError) as raised:
            TaskInterruptedHandoffReceipt(
                request=request,
                request_sha256="0" * 64,
                task=attached,
                committed_at=attached.updated_at,
            )
        return raised.value

    failure = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic = "\n".join(
        [
            str(failure),
            repr(failure),
            captured.out,
            captured.err,
            "\n".join(record.getMessage() for record in caplog.records),
            "\n".join(str(record.message) for record in recwarn),
        ]
    )
    assert secret not in diagnostic


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_interrupted_task_handoff_rejects_wrong_owner_and_recovers_expired_owner(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_expired_handoff", type="review"))
        await store.claim_task("worker_a", lease_seconds=1)
        attached = await store.attach_task(
            "task_expired_handoff",
            session_id="session_expired_handoff",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_expired_handoff",
                "session_expired_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, "task_expired_handoff"),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-expired",
        )
        with pytest.raises(TaskInterruptedHandoffConflict):
            await store.release_interrupted_task_worker(
                _interrupted_handoff_request(
                    attached,
                    handoff_id="handoff-wrong-worker",
                    worker_id="worker_b",
                )
            )

        await asyncio.sleep(1.05)
        with pytest.raises(TaskInterruptedHandoffConflict, match="live worker lease"):
            await store.release_interrupted_task_worker(request)
        candidates = await store.list_expired_interrupted_task_handoff_candidates()
        assert [candidate.id for candidate in candidates] == ["task_expired_handoff"]

        receipt = await store.recover_interrupted_task_worker(request)
        assert receipt.task.status is TaskStatus.RUNNING
        assert receipt.task.session_id == "session_expired_handoff"
        assert receipt.task.worker_id is None
        assert receipt.task.lease_expires_at is None
        assert await store.list_expired_interrupted_task_handoff_candidates() == []
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_interrupted_task_handoff_rejects_retry_cancellation_marker() -> None:
    store = InMemoryTaskStore()

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_retry_cancel_marker", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_retry_cancel_marker",
            session_id="session_retry_cancel_marker",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_retry_cancel_marker",
                "session_retry_cancel_marker",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(
                store,
                "task_retry_cancel_marker",
            ),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-retry-cancel-marker",
        )

        # Retry-series attempts cannot attach to sessions through supported public
        # APIs. Still fail closed if corrupted/custom-store evidence presents that
        # cancellation marker at the shared built-in-store authority boundary.
        malformed = attached.model_copy(update={"status_reason": "retry_cancellation_requested"})
        with pytest.raises(TaskInterruptedHandoffConflict, match="cancellation is still draining"):
            _require_interrupted_task_handoff_authority(
                malformed,
                request,
                now=attached.updated_at,
                recover_expired=False,
            )

        unchanged = await store.load_task(attached.id)
        assert unchanged == attached
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_interrupted_task_handoff_candidate_pages_have_stable_store_order(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        with pytest.raises(ValueError, match="limit must be <= 100"):
            await store.list_expired_interrupted_task_handoff_candidates(limit=101)
        with pytest.raises(ValueError, match="scan_limit must be <= 100"):
            await store.claim_interrupted_task_continuation(
                "continuation-worker",
                handoff_id=str(uuid4()),
                scan_limit=101,
            )
        for task_id in ("task_handoff_page_a", "task_handoff_page_b"):
            await store.create_task(TaskCreate(task_id=task_id, type="review"))
            claimed = await store.claim_task("worker_a", lease_seconds=1)
            assert claimed is not None
            assert claimed.id == task_id
            await store.attach_task(
                task_id,
                session_id=f"session_{task_id}",
                session_invocation=await task_backed_session_invocation(
                    store,
                    task_id,
                    f"session_{task_id}",
                ),
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
            )
        await asyncio.sleep(1.05)

        first_page = await store.list_expired_interrupted_task_handoff_candidates(limit=1)
        assert len(first_page) == 1
        first = first_page[0]
        assert first.lease_expires_at is not None
        second_page = await store.list_expired_interrupted_task_handoff_candidates(
            after=(first.lease_expires_at, first.id),
            limit=1,
        )
        assert [first.id, second_page[0].id] == [
            "task_handoff_page_a",
            "task_handoff_page_b",
        ]
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_interrupted_task_handoff_converges_with_concurrent_terminalization(
    store_factory: StoreFactory,
    tmp_path,
) -> None:
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_handoff_terminal_race", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_handoff_terminal_race",
            session_id="session_handoff_terminal_race",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_handoff_terminal_race",
                "session_handoff_terminal_race",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(
                store,
                "task_handoff_terminal_race",
            ),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-terminal-race",
        )
        terminal_request = TaskTerminalizationRequest(
            task_id=attached.id,
            worker_id="worker_a",
            lease_expires_at=attached.lease_expires_at,
            kind=TaskTerminalKind.COMPLETED,
            result={"winner": "terminal"},
            idempotency_key="terminal-handoff-race",
        )
        outcomes = await asyncio.gather(
            store.release_interrupted_task_worker(request),
            store.terminalize_task(terminal_request),
            return_exceptions=True,
        )
        assert (
            sum(type(outcome) is TaskInterruptedHandoffReceipt for outcome in outcomes)
            + sum(type(outcome) is Task for outcome in outcomes)
            == 1
        )
        assert sum(isinstance(outcome, Exception) for outcome in outcomes) == 1

        durable = await store.load_task(attached.id)
        handoff_receipt = await store.load_interrupted_task_handoff_receipt(
            request.task_id,
            request.handoff_id,
        )
        assert durable is not None
        if handoff_receipt is None:
            assert durable.status is TaskStatus.COMPLETED
            assert durable.result == {"winner": "terminal"}
        else:
            assert durable == handoff_receipt.task
            assert durable.status is TaskStatus.RUNNING
            assert durable.session_id == request.session_id
            assert durable.worker_id is None
            assert durable.lease_expires_at is None
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_sqlite_interrupted_task_handoff_receipt_survives_restart(tmp_path) -> None:
    db_path = tmp_path / "interrupted-handoff.sqlite"
    first = SQLiteTaskStore(db_path)

    async def first_process() -> tuple[
        TaskInterruptedHandoffRequest, TaskInterruptedHandoffReceipt
    ]:
        await first.create_task(TaskCreate(task_id="task_restart_handoff", type="review"))
        await first.claim_task("worker_a", lease_seconds=300)
        attached = await first.attach_task(
            "task_restart_handoff",
            session_id="session_restart_handoff",
            session_invocation=await task_backed_session_invocation(
                first,
                "task_restart_handoff",
                "session_restart_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(first, "task_restart_handoff"),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-restart",
        )
        receipt = await first.release_interrupted_task_worker(request)
        await first.close()
        return request, receipt

    request, receipt = asyncio.run(first_process())
    reopened = SQLiteTaskStore(db_path)

    async def second_process() -> None:
        assert await reopened.release_interrupted_task_worker(request) == receipt
        assert (
            await reopened.load_interrupted_task_handoff_receipt(
                request.task_id,
                request.handoff_id,
            )
            == receipt
        )
        await reopened.close()

    asyncio.run(second_process())


def test_sqlite_interrupted_task_handoff_rejects_rewritten_receipt_authority(
    tmp_path,
) -> None:
    db_path = tmp_path / "interrupted-handoff-corrupt.sqlite"
    store = SQLiteTaskStore(db_path)

    async def publish() -> TaskInterruptedHandoffRequest:
        await store.create_task(TaskCreate(task_id="task_corrupt_handoff", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_corrupt_handoff",
            session_id="session_corrupt_handoff",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_corrupt_handoff",
                "session_corrupt_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, "task_corrupt_handoff"),
        )
        request = _interrupted_handoff_request(
            attached,
            handoff_id="handoff-corrupt",
        )
        await store.release_interrupted_task_worker(request)
        await store.close()
        return request

    request = asyncio.run(publish())
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE cayu_task_interrupted_handoff_receipts "
        "SET request_json = json_set(request_json, '$.session_run_epoch', 2) "
        "WHERE task_id = ? AND handoff_id = ?",
        (request.task_id, request.handoff_id),
    )
    connection.commit()
    connection.close()
    reopened = SQLiteTaskStore(db_path)

    async def reject() -> None:
        with pytest.raises(TaskInterruptedHandoffConflict, match="malformed"):
            await reopened.load_interrupted_task_handoff_receipt(
                request.task_id,
                request.handoff_id,
            )
        with pytest.raises(TaskInterruptedHandoffConflict, match="malformed"):
            await reopened.release_interrupted_task_worker(request)
        await reopened.close()

    asyncio.run(reject())


def test_sqlite_interrupted_task_handoff_rejects_receipt_from_another_storage_key(
    tmp_path,
) -> None:
    db_path = tmp_path / "interrupted-handoff-key-conflict.sqlite"
    store = SQLiteTaskStore(db_path)

    async def publish(
        task_id: str,
        session_id: str,
        handoff_id: str,
    ) -> TaskInterruptedHandoffRequest:
        await store.create_task(TaskCreate(task_id=task_id, type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            task_id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task_id,
                session_id,
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, task_id),
        )
        request = _interrupted_handoff_request(attached, handoff_id=handoff_id)
        await store.release_interrupted_task_worker(request)
        return request

    async def publish_both() -> tuple[
        TaskInterruptedHandoffRequest,
        TaskInterruptedHandoffRequest,
    ]:
        first = await publish("task_key_first", "session_key_first", "handoff-key-first")
        second = await publish("task_key_second", "session_key_second", "handoff-key-second")
        await store.close()
        return first, second

    first, second = asyncio.run(publish_both())
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE cayu_task_interrupted_handoff_receipts AS target "
        "SET request_sha256 = source.request_sha256, "
        "request_json = source.request_json, task_json = source.task_json, "
        "committed_at = source.committed_at "
        "FROM cayu_task_interrupted_handoff_receipts AS source "
        "WHERE target.task_id = ? AND target.handoff_id = ? "
        "AND source.task_id = ? AND source.handoff_id = ?",
        (
            first.task_id,
            first.handoff_id,
            second.task_id,
            second.handoff_id,
        ),
    )
    connection.commit()
    connection.close()
    reopened = SQLiteTaskStore(db_path)

    async def reject() -> None:
        with pytest.raises(TaskInterruptedHandoffConflict, match="malformed"):
            await reopened.load_interrupted_task_handoff_receipt(
                first.task_id,
                first.handoff_id,
            )
        with pytest.raises(TaskInterruptedHandoffConflict, match="malformed"):
            await reopened.release_interrupted_task_worker(first)
        await reopened.close()

    asyncio.run(reject())


def test_sqlite_interrupted_task_handoff_survives_real_process_loss(tmp_path) -> None:
    db_path = tmp_path / "interrupted-handoff-process-loss.sqlite"
    store = SQLiteTaskStore(db_path)

    async def prepare() -> TaskInterruptedHandoffRequest:
        await store.create_task(TaskCreate(task_id="task_process_handoff", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_process_handoff",
            session_id="session_process_handoff",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_process_handoff",
                "session_process_handoff",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, "task_process_handoff"),
        )
        await store.close()
        return _interrupted_handoff_request(
            attached,
            handoff_id="handoff-process-loss",
        )

    request = asyncio.run(prepare())
    repository_root = Path(__file__).parents[2]
    python_path = str(repository_root / "src")
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path = os.pathsep.join((python_path, inherited_python_path))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.core._interrupted_handoff_process_worker",
            "sqlite",
            str(db_path),
            request.model_dump_json(),
        ],
        check=False,
        capture_output=True,
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": python_path},
        text=True,
        timeout=30,
    )
    assert completed.returncode == 23, completed.stderr

    reopened = SQLiteTaskStore(db_path)

    async def reconcile() -> None:
        receipt = await reopened.load_interrupted_task_handoff_receipt(
            request.task_id,
            request.handoff_id,
        )
        assert receipt is not None
        assert await reopened.release_interrupted_task_worker(request) == receipt
        task = await reopened.load_task(request.task_id)
        assert task == receipt.task
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert task.session_id == request.session_id
        assert task.worker_id is None
        assert task.lease_expires_at is None
        await reopened.close()

    asyncio.run(reconcile())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_invalid_attached_worker_release(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        with pytest.raises(KeyError, match="Task not found"):
            await store.release_attached_task_worker(
                "missing",
                "worker_a",
                lease_expires_at=datetime.now(UTC),
            )

        await store.create_task(TaskCreate(task_id="task_unattached", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        unattached_before = await store.load_task("task_unattached")
        assert unattached_before is not None
        with pytest.raises(ValueError, match="not running"):
            await store.release_attached_task_worker(
                "task_unattached",
                "worker_a",
                lease_expires_at=unattached_before.lease_expires_at,
            )

        await store.create_task(TaskCreate(task_id="task_wrong_worker", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        await store.attach_task(
            "task_wrong_worker",
            session_id="sess_wrong_worker",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_wrong_worker",
                "sess_wrong_worker",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, "task_wrong_worker"),
        )
        wrong_worker_before = await store.load_task("task_wrong_worker")
        assert wrong_worker_before is not None
        with pytest.raises(ValueError, match="does not own"):
            await store.release_attached_task_worker(
                "task_wrong_worker",
                "worker_b",
                lease_expires_at=wrong_worker_before.lease_expires_at,
            )

        await store.create_task(TaskCreate(task_id="task_expired_worker", type="review"))
        await store.claim_task("worker_a", lease_seconds=1)
        await store.attach_task(
            "task_expired_worker",
            session_id="sess_expired_worker",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_expired_worker",
                "sess_expired_worker",
            ),
            worker_id="worker_a",
            lease_expires_at=await _exact_task_lease(store, "task_expired_worker"),
        )
        await asyncio.sleep(1.05)
        expired_before = await store.load_task("task_expired_worker")
        assert expired_before is not None
        with pytest.raises(TaskClaimLost, match="lease for worker worker_a has expired"):
            await store.release_attached_task_worker(
                "task_expired_worker",
                "worker_a",
                lease_expires_at=expired_before.lease_expires_at,
            )

        await store.create_task(TaskCreate(task_id="task_terminal", type="review"))
        terminal_before = await store.complete_task(
            "task_terminal",
            {"winner": "terminal-state"},
        )
        with pytest.raises(ValueError, match="running"):
            await store.release_attached_task_worker(
                "task_terminal",
                "worker_a",
                lease_expires_at=datetime.now(UTC),
            )

        unattached_after = await store.load_task("task_unattached")
        wrong_worker_after = await store.load_task("task_wrong_worker")
        expired_after = await store.load_task("task_expired_worker")
        terminal_after = await store.load_task("task_terminal")
        assert unattached_after == unattached_before
        assert wrong_worker_after == wrong_worker_before
        assert expired_after == expired_before
        assert terminal_after is not None
        assert terminal_after == terminal_before

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
        await store.attach_task(
            "task_attached_expired",
            session_id="sess_attached_expired",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_attached_expired",
                "sess_attached_expired",
            ),
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
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
            await store.heartbeat(
                "task_validate_worker",
                "worker_a",
                lease_expires_at=claimed.lease_expires_at,
                extend_seconds=0,
            )
        with pytest.raises(ValueError, match="max_reclaims must be >= 1"):
            await store.reclaim_expired(max_reclaims=0)
        with pytest.raises(ValueError, match="do not support q"):
            await store.claim_task("worker_b", TaskQuery(q="invoice"))
        with pytest.raises(ValueError, match="do not support q"):
            await store.reclaim_expired(query=TaskQuery(q="invoice"))
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


def test_sqlite_claimed_cancellation_rechecks_lease_after_writer_wait(tmp_path):
    db_path = tmp_path / "tasks.sqlite"
    clock_lock = threading.Lock()
    clock_value = [datetime(2026, 1, 1, tzinfo=UTC)]

    def ownership_clock() -> datetime:
        with clock_lock:
            return clock_value[0]

    owner = SQLiteTaskStore(db_path, ownership_clock=ownership_clock)
    contender = SQLiteTaskStore(db_path, ownership_clock=ownership_clock)

    async def prepare() -> Task:
        await owner.create_task(TaskCreate(task_id="sqlite-cancellation-race", type="review"))
        claimed = await owner.claim_task("worker-a", lease_seconds=1)
        assert claimed is not None
        assert claimed.lease_expires_at is not None
        return claimed

    claimed = asyncio.run(prepare())
    writer = sqlite3.connect(db_path, check_same_thread=False)
    writer.execute("PRAGMA busy_timeout = 5000")
    writer.execute("BEGIN IMMEDIATE")
    transaction_started = threading.Event()
    contender._connection.set_trace_callback(
        lambda statement: (
            transaction_started.set() if statement.strip().upper() == "BEGIN IMMEDIATE" else None
        )
    )
    failures: list[BaseException] = []

    def request_cancellation() -> None:
        try:
            asyncio.run(
                contender.request_claimed_task_cancellation(
                    claimed.id,
                    "worker-a",
                    claimed.lease_expires_at,
                    {"reason": "lease-expired-while-waiting"},
                )
            )
        except BaseException as exc:
            failures.append(exc)

    requester = threading.Thread(target=request_cancellation)
    requester.start()
    try:
        assert transaction_started.wait(timeout=2)
        with clock_lock:
            clock_value[0] += timedelta(seconds=2)
        writer.commit()
        requester.join(timeout=5)
        assert not requester.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], TaskClaimLost)

        current = asyncio.run(owner.load_task(claimed.id))
        assert current is not None
        assert current.status_reason is None
        assert current.worker_id == "worker-a"
        assert current.lease_expires_at == claimed.lease_expires_at
    finally:
        if writer.in_transaction:
            writer.rollback()
        writer.close()
        if requester.is_alive():
            requester.join(timeout=5)
        asyncio.run(owner.close())
        asyncio.run(contender.close())


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
        await store.start_task(
            "task_persisted",
            session_id="sess_persisted",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_persisted",
                "sess_persisted",
            ),
        )
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

        await first.start_task(
            "task_claim",
            session_id="session_one",
            session_invocation=await task_backed_session_invocation(
                first,
                "task_claim",
                "session_one",
            ),
        )
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


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_search_tasks(store_factory: StoreFactory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(
            TaskCreate(
                task_id="task_billing_export",
                type="sync",
                title="Wait for billing export",
                assigned_agent_name="billing-agent",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_invoice_review",
                type="review",
                title="Review invoice",
                assigned_agent_name="invoice-agent",
            )
        )
        await store.block_task("task_billing_export", reason="Waiting on upstream export")

        by_title = await store.list_tasks(TaskQuery(q="billing", order_by=TaskOrder.CREATED_AT_ASC))
        assert [task.id for task in by_title] == ["task_billing_export"]

        by_reason = await store.list_tasks(
            TaskQuery(q="UPSTREAM", order_by=TaskOrder.CREATED_AT_ASC)
        )
        assert [task.id for task in by_reason] == ["task_billing_export"]

        by_agent = await store.list_tasks(TaskQuery(q="invoice-agent"))
        assert [task.id for task in by_agent] == ["task_invoice_review"]
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_reject_same_key_with_changed_logical_intent(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_changed", type="review"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None
        winner = TaskTerminalizationRequest(
            task_id="task_changed",
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "winner"},
            idempotency_key="shared-key",
        )
        terminal = await store.terminalize_task(winner)

        conflicting_requests = (
            winner.model_copy(update={"worker_id": "worker_b"}),
            TaskTerminalizationRequest(
                task_id="task_changed",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error={"message": "failed"},
                idempotency_key="shared-key",
            ),
            winner.model_copy(update={"result": {"summary": "changed"}}),
        )
        for conflicting in conflicting_requests:
            with pytest.raises(TaskTerminalizationConflict):
                await store.terminalize_task(conflicting)

        assert await store.load_task("task_changed") == terminal
        receipt = await store.load_task_terminalization_receipt("task_changed", "shared-key")
        assert receipt is not None
        assert receipt.task == terminal
        await _close_store(store)

    asyncio.run(run_store_operations())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_stores_concurrent_retries_converge_and_conflicts_choose_one_winner(
    store_factory: StoreFactory,
    tmp_path,
):
    store = _make_store(store_factory, tmp_path)

    async def run_store_operations() -> None:
        await store.create_task(TaskCreate(task_id="task_exact_race", type="review"))
        exact_claim = await store.claim_task("worker_a")
        assert exact_claim is not None
        exact = TaskTerminalizationRequest(
            task_id="task_exact_race",
            worker_id="worker_a",
            lease_expires_at=exact_claim.lease_expires_at,
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="race-key",
        )
        exact_results = await asyncio.gather(*(store.terminalize_task(exact) for _ in range(8)))
        assert all(result == exact_results[0] for result in exact_results)

        await store.create_task(TaskCreate(task_id="task_conflict_race", type="review"))
        conflict_claim = await store.claim_task("worker_b")
        assert conflict_claim is not None
        requests = (
            TaskTerminalizationRequest(
                task_id="task_conflict_race",
                worker_id="worker_b",
                lease_expires_at=conflict_claim.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"winner": "completed"},
                idempotency_key="conflict-key",
            ),
            TaskTerminalizationRequest(
                task_id="task_conflict_race",
                worker_id="worker_b",
                lease_expires_at=conflict_claim.lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error={"winner": "failed"},
                idempotency_key="conflict-key",
            ),
        )

        async def apply(request: TaskTerminalizationRequest):
            try:
                return await store.terminalize_task(request)
            except TaskTerminalizationConflict as exc:
                return exc

        outcomes = await asyncio.gather(*(apply(request) for request in requests))
        winners = [outcome for outcome in outcomes if type(outcome) is Task]
        conflicts = [
            outcome for outcome in outcomes if isinstance(outcome, TaskTerminalizationConflict)
        ]
        assert len(winners) == 1
        assert len(conflicts) == 1
        assert await store.load_task("task_conflict_race") == winners[0]
        await _close_store(store)

    asyncio.run(run_store_operations())


def test_sqlite_task_terminalization_receipt_survives_restart(tmp_path) -> None:
    db_path = tmp_path / "tasks.sqlite"
    store = SQLiteTaskStore(db_path)

    async def first_process() -> tuple[Task, TaskTerminalizationRequest]:
        await store.create_task(TaskCreate(task_id="task_restart", type="review"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None
        request = TaskTerminalizationRequest(
            task_id="task_restart",
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="restart-key",
        )
        terminal = await store.terminalize_task(request)
        await store.close()
        return terminal, request

    terminal, request = asyncio.run(first_process())
    reopened = SQLiteTaskStore(db_path)

    async def second_process() -> None:
        assert await reopened.terminalize_task(request) == terminal
        receipt = await reopened.load_task_terminalization_receipt("task_restart", "restart-key")
        assert receipt is not None
        assert receipt.task == terminal
        await reopened.close()

    asyncio.run(second_process())


def test_sqlite_task_terminalization_converges_across_connections(tmp_path) -> None:
    db_path = tmp_path / "tasks.sqlite"
    first_store = SQLiteTaskStore(db_path)
    second_store = SQLiteTaskStore(db_path)

    async def run() -> None:
        try:
            await first_store.create_task(TaskCreate(task_id="task_connection_race", type="review"))
            claim = await first_store.claim_task("worker_a")
            assert claim is not None
            request = TaskTerminalizationRequest(
                task_id="task_connection_race",
                worker_id="worker_a",
                lease_expires_at=claim.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done"},
                idempotency_key="connection-race",
            )
            first, replayed = await asyncio.gather(
                first_store.terminalize_task(request),
                second_store.terminalize_task(request),
            )
            assert first == replayed

            await first_store.create_task(
                TaskCreate(task_id="task_connection_conflict", type="review")
            )
            conflict_claim = await first_store.claim_task("worker_b")
            assert conflict_claim is not None
            completed = TaskTerminalizationRequest(
                task_id="task_connection_conflict",
                worker_id="worker_b",
                lease_expires_at=conflict_claim.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"winner": "completed"},
                idempotency_key="connection-conflict",
            )
            failed = TaskTerminalizationRequest(
                task_id="task_connection_conflict",
                worker_id="worker_b",
                lease_expires_at=conflict_claim.lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error={"winner": "failed"},
                idempotency_key="connection-conflict",
            )

            async def apply(store: TaskStore, request: TaskTerminalizationRequest):
                try:
                    return await store.terminalize_task(request)
                except TaskTerminalizationConflict as exc:
                    return exc

            outcomes = await asyncio.gather(
                apply(first_store, completed),
                apply(second_store, failed),
            )
            assert sum(type(outcome) is Task for outcome in outcomes) == 1
            assert (
                sum(isinstance(outcome, TaskTerminalizationConflict) for outcome in outcomes) == 1
            )
        finally:
            await first_store.close()
            await second_store.close()

    asyncio.run(run())


def test_sqlite_revision_thirty_nine_rejects_populated_store_before_receipt_migration(
    tmp_path,
) -> None:
    db_path = tmp_path / "tasks.sqlite"
    current = SQLiteTaskStore(db_path)

    async def create_existing_terminal() -> None:
        await current.create_task(TaskCreate(task_id="existing_terminal", type="review"))
        await current.complete_task("existing_terminal", {"summary": "existing"})
        await current.close()

    asyncio.run(create_existing_terminal())
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_task_terminalization_receipts")
        connection.execute("ALTER TABLE cayu_tasks DROP COLUMN invocation_json")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 38")
        connection.execute("PRAGMA user_version = 37")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="revision 39 requires invocation provenance"):
        SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        revision = connection.execute("SELECT MAX(revision) FROM cayu_schema_migrations").fetchone()
        receipt_table = connection.execute(
            "SELECT type FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cayu_task_terminalization_receipts'"
        ).fetchone()
        task = connection.execute(
            "SELECT status, result_json FROM cayu_tasks WHERE id = 'existing_terminal'"
        ).fetchone()
    finally:
        connection.close()
    assert revision == (37,)
    assert receipt_table is None
    assert task == (str(TaskStatus.COMPLETED), '{"summary":"existing"}')


def test_sqlite_task_store_rejects_missing_terminalization_receipt_table(tmp_path) -> None:
    db_path = tmp_path / "tasks.sqlite"
    store = SQLiteTaskStore(db_path)
    asyncio.run(store.close())
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_task_terminalization_receipts")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="terminalization receipt"):
        SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_task_store_validate_rejects_pre_handoff_generation_schema(tmp_path) -> None:
    db_path = tmp_path / "tasks.sqlite"
    store = SQLiteTaskStore(db_path)
    asyncio.run(store.close())
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_tasks_interrupted_handoff_continuation")
        connection.execute("DROP INDEX idx_cayu_tasks_interrupted_handoff_generation")
        connection.execute("DROP TABLE cayu_task_interrupted_continuation_claims")
        connection.execute("ALTER TABLE cayu_tasks DROP COLUMN interrupted_handoff_id")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 76")
        connection.execute("PRAGMA user_version = 75")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(schema_migrations.SchemaTooOld, match="requires >= 76"):
        SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_task_store_rejects_missing_continuation_claim_registry(tmp_path) -> None:
    db_path = tmp_path / "tasks.sqlite"
    store = SQLiteTaskStore(db_path)
    asyncio.run(store.close())
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_task_interrupted_continuation_claims")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="task handoff generation"):
        SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_revision_seventy_six_backfills_live_handoff_authority(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "tasks-handoff-migration.sqlite"
    monkeypatch.setattr(sqlite_support, "_INTERRUPTED_HANDOFF_MIGRATION_BATCH_SIZE", 1)

    async def seed() -> tuple[TaskInterruptedHandoffRequest, ...]:
        store = SQLiteTaskStore(db_path)
        try:

            async def release(
                task_id: str,
                task_type: str,
                session_id: str,
            ) -> TaskInterruptedHandoffRequest:
                await store.create_task(TaskCreate(task_id=task_id, type=task_type))
                worker_id = f"prior-{task_id}"
                claimed = await store.claim_task(worker_id, TaskQuery(type=task_type))
                assert claimed is not None
                attached = await store.attach_task(
                    task_id,
                    session_id=session_id,
                    session_invocation=await task_backed_session_invocation(
                        store,
                        task_id,
                        session_id,
                    ),
                    worker_id=worker_id,
                    lease_expires_at=claimed.lease_expires_at,
                )
                request = _interrupted_handoff_request(
                    attached,
                    handoff_id=f"{task_id}-generation-1",
                )
                await store.release_interrupted_task_worker(request)
                return request

            live = await release("migrated-live", "migration-live", "migrated-live-session")
            historical = await release(
                "migrated-history",
                "migration-history",
                "migrated-history-session",
            )
            history_owner = (
                await store.claim_interrupted_task_continuation(
                    "history-owner",
                    TaskQuery(type="migration-history"),
                    handoff_id=str(uuid4()),
                )
            ).task
            assert history_owner is not None
            current = interrupted_task_handoff_request(history_owner, session_run_epoch=2)
            await store.release_interrupted_task_worker(current)

            terminal = await release(
                "migrated-terminal",
                "migration-terminal",
                "migrated-terminal-session",
            )
            terminal_owner = (
                await store.claim_interrupted_task_continuation(
                    "terminal-owner",
                    TaskQuery(type="migration-terminal"),
                    handoff_id=str(uuid4()),
                )
            ).task
            assert terminal_owner is not None
            await store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id=terminal_owner.id,
                    worker_id="terminal-owner",
                    lease_expires_at=terminal_owner.lease_expires_at,
                    handoff_id=terminal_owner.interrupted_handoff_id,
                    kind=TaskTerminalKind.COMPLETED,
                    result={"outcome": "done"},
                    idempotency_key="migration-terminal-complete",
                )
            )
            return live, historical, current, terminal
        finally:
            await store.close()

    live, historical, current, terminal = asyncio.run(seed())
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE cayu_task_interrupted_handoff_receipts "
            "SET task_json = json_remove(task_json, '$.interrupted_handoff_id')"
        )
        connection.execute("DROP INDEX idx_cayu_tasks_interrupted_handoff_continuation")
        connection.execute("DROP INDEX idx_cayu_tasks_interrupted_handoff_generation")
        connection.execute("DROP TABLE cayu_task_interrupted_continuation_claims")
        connection.execute("ALTER TABLE cayu_tasks DROP COLUMN interrupted_handoff_id")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 76")
        connection.execute("PRAGMA user_version = 75")
        connection.commit()
    finally:
        connection.close()

    migrated = SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    async def verify() -> None:
        try:
            task = await migrated.load_task(live.task_id)
            assert task is not None
            assert task.interrupted_handoff_id == live.handoff_id
            for request in (live, historical, current, terminal):
                receipt = await migrated.load_interrupted_task_handoff_receipt(
                    request.task_id,
                    request.handoff_id,
                )
                assert receipt is not None
                assert receipt.task.interrupted_handoff_id == request.handoff_id
            historical_task = await migrated.load_task(historical.task_id)
            assert historical_task is not None
            assert historical_task.interrupted_handoff_id == current.handoff_id
            terminal_task = await migrated.load_task(terminal.task_id)
            assert terminal_task is not None
            assert terminal_task.status is TaskStatus.COMPLETED
            assert terminal_task.interrupted_handoff_id is None
            owner = (
                await migrated.claim_interrupted_task_continuation(
                    "migration-owner",
                    TaskQuery(type="migration-live"),
                    handoff_id=str(uuid4()),
                )
            ).task
            assert owner is not None
            assert owner.worker_id == "migration-owner"
            assert owner.interrupted_handoff_id not in {None, live.handoff_id}
        finally:
            await migrated.close()

    asyncio.run(verify())


def test_sqlite_revision_seventy_six_rejects_ambiguous_handoff_authority(tmp_path) -> None:
    db_path = tmp_path / "tasks-ambiguous-handoff-migration.sqlite"

    async def seed() -> TaskInterruptedHandoffRequest:
        store = SQLiteTaskStore(db_path)
        try:
            await store.create_task(TaskCreate(task_id="ambiguous-handoff", type="review"))
            claimed = await store.claim_task("prior-worker")
            assert claimed is not None
            attached = await store.attach_task(
                "ambiguous-handoff",
                session_id="ambiguous-session",
                session_invocation=await task_backed_session_invocation(
                    store,
                    "ambiguous-handoff",
                    "ambiguous-session",
                ),
                worker_id="prior-worker",
                lease_expires_at=claimed.lease_expires_at,
            )
            request = _interrupted_handoff_request(
                attached,
                handoff_id="ambiguous-generation-a",
            )
            await store.release_interrupted_task_worker(request)
            return request
        finally:
            await store.close()

    first = asyncio.run(seed())
    second, second_sha256 = prepare_interrupted_task_handoff(
        first.model_copy(update={"handoff_id": "ambiguous-generation-b"})
    )
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE cayu_task_interrupted_handoff_receipts "
            "SET task_json = json_remove(task_json, '$.interrupted_handoff_id')"
        )
        connection.execute(
            """
            INSERT INTO cayu_task_interrupted_handoff_receipts (
                task_id, handoff_id, request_sha256, request_json, task_json, committed_at
            )
            SELECT task_id, ?, ?, ?, task_json, committed_at
            FROM cayu_task_interrupted_handoff_receipts
            WHERE task_id = ? AND handoff_id = ?
            """,
            (
                second.handoff_id,
                second_sha256,
                json.dumps(second.model_dump(mode="json")),
                first.task_id,
                first.handoff_id,
            ),
        )
        connection.execute("DROP INDEX idx_cayu_tasks_interrupted_handoff_continuation")
        connection.execute("DROP INDEX idx_cayu_tasks_interrupted_handoff_generation")
        connection.execute("DROP TABLE cayu_task_interrupted_continuation_claims")
        connection.execute("ALTER TABLE cayu_tasks DROP COLUMN interrupted_handoff_id")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 76")
        connection.execute("PRAGMA user_version = 75")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="cannot determine one interrupted-task handoff"):
        SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)


def test_sqlite_revision_thirty_eight_rejects_conflicting_table_before_recording(
    tmp_path,
) -> None:
    db_path = tmp_path / "tasks.sqlite"
    store = SQLiteTaskStore(db_path)
    asyncio.run(store.close())
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_task_terminalization_receipts")
        connection.execute("ALTER TABLE cayu_tasks DROP COLUMN invocation_json")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 38")
        connection.execute("PRAGMA user_version = 37")
        connection.execute(
            "CREATE TABLE cayu_task_terminalization_receipts (task_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="terminalization receipt table"):
        SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        revision = connection.execute("SELECT MAX(revision) FROM cayu_schema_migrations").fetchone()
    finally:
        connection.close()
    assert revision == (37,)


def _make_store(store_factory: StoreFactory, tmp_path) -> TaskStore:
    if store_factory is SQLiteTaskStore:
        return SQLiteTaskStore(tmp_path / "tasks.sqlite")
    return store_factory()


async def _close_store(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()
