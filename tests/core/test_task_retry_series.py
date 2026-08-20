from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    SecretRedactor,
    SQLiteTaskStore,
    TaskClaimLost,
    TaskCreate,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryEventType,
    TaskRetryPolicy,
    TaskRetrySeriesDisposition,
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    TaskStatus,
    TaskTerminalizationConflict,
    run_task_worker,
    settle_task_retry_attempt_with_retry,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.storage import migrations as schema_migrations


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _store_for_kind(store_kind: str, path: Path, clock: _MutableClock):
    return (
        InMemoryTaskStore(clock=clock)
        if store_kind == "memory"
        else SQLiteTaskStore(path, clock=clock)
    )


def _retry_causal_budget_id(task) -> str:
    assert task.retry_series is not None
    return task.retry_series.causal_budget_id


@pytest.mark.parametrize(
    "policy",
    [
        {"max_attempts": 0},
        {"max_attempts": 2, "max_elapsed_seconds": float("nan")},
        {"max_attempts": 2, "max_total_tokens": 0},
        {"max_attempts": 2, "max_estimated_cost": Decimal(0)},
        {"max_attempts": 2, "max_estimated_cost": Decimal("Infinity")},
        {"max_attempts": 2, "max_estimated_cost": Decimal("1e65")},
        {"max_attempts": 2, "max_estimated_cost": Decimal("1e-65")},
        {"max_attempts": 2, "cost_currency": "   "},
        {"max_attempts": 2, "cost_currency": "X" * 17},
    ],
)
def test_task_retry_policy_rejects_unbounded_or_unusable_authority(policy) -> None:
    with pytest.raises(ValueError):
        TaskRetryPolicy.model_validate(policy)


def test_task_retry_policy_round_trips_as_normalized_public_json() -> None:
    policy = TaskRetryPolicy(
        max_attempts=4,
        max_elapsed_seconds=30,
        max_total_tokens=500,
        max_estimated_cost=Decimal("2.50"),
        cost_currency="usd",
    )

    assert TaskRetryPolicy.model_validate_json(policy.model_dump_json()) == policy
    assert policy.cost_currency == "USD"


def test_task_retry_migration_documentation_matches_schema_revision() -> None:
    contract = Path("docs/runtime-contracts.md").read_text(encoding="utf-8")

    assert schema_migrations.LATEST_REVISION == 45
    assert "Breaking schema revision\n45 adds the retry-series" in contract
    assert "Pre-45 task workers" in contract


def test_cayu_app_rejects_retry_policy_before_unsupported_store_create() -> None:
    class UnsupportedRetryStore(InMemoryTaskStore):
        supports_task_retry_series = False

        def __init__(self) -> None:
            super().__init__()
            self.create_calls = 0

        async def create_task(self, request):
            self.create_calls += 1
            return await super().create_task(request)

    async def run() -> None:
        store = UnsupportedRetryStore()
        app = CayuApp(task_store=store, enable_logging=False)
        with pytest.raises(NotImplementedError, match="does not support task retry series"):
            await app.create_task(
                TaskCreate(
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
        assert store.create_calls == 0

    asyncio.run(run())


@pytest.mark.parametrize("secret", ["SECRETCURRENCY", "SecretCurrency", "secretcurrency"])
def test_cayu_app_rejects_secret_bearing_retry_currency_before_store_create(
    secret: str,
) -> None:
    class RecordingStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.create_calls = 0

        async def create_task(self, request):
            self.create_calls += 1
            return await super().create_task(request)

    async def run() -> None:
        store = RecordingStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        with pytest.raises(ValueError, match="cost currency contains a workload secret"):
            await app.create_task(
                TaskCreate(
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        cost_currency=secret,
                    ),
                )
            )
        assert store.create_calls == 0

    asyncio.run(run())


def test_task_retry_maximum_attempt_reports_remain_durably_representable() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        maximum_report = MAX_DURABLE_JSON_INTEGER // 100
        await store.create_task(
            TaskCreate(
                task_id="maximum-cumulative-token-accounting",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=100,
                    initial_backoff_seconds=0,
                ),
            )
        )

        receipt = None
        for attempt in range(1, 101):
            claimed = await store.claim_task("worker")
            assert claimed is not None
            succeeded = attempt == 100
            receipt = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="worker",
                    idempotency_key=f"maximum-token-report-{attempt}",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=(
                        TaskRetryAttemptDisposition.SUCCEEDED
                        if succeeded
                        else TaskRetryAttemptDisposition.RETRYABLE_FAILURE
                    ),
                    result={"ok": True} if succeeded else None,
                    error=None if succeeded else {"code": "temporary"},
                    token_count=maximum_report,
                )
            )

        assert receipt is not None
        assert receipt.task.retry_series is not None
        assert receipt.task.retry_series.cumulative_tokens == maximum_report * 100
        assert receipt.task.retry_series.disposition is TaskRetrySeriesDisposition.SUCCEEDED

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_series_expires_queued_attempt_before_late_claim(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started_at = datetime(2026, 8, 19, 15, 30, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = _store_for_kind(store_kind, tmp_path / "elapsed-claim.sqlite", clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="task-elapsed-claim",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=3,
                        max_elapsed_seconds=10,
                        initial_backoff_seconds=2,
                    ),
                )
            )
            claimed = await store.claim_task("first-worker")
            assert claimed is not None
            first = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id="task-elapsed-claim",
                    worker_id="first-worker",
                    idempotency_key="first-elapsed-report",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                )
            )
            assert first.successor is not None
            successor_id = first.successor.id

            clock.value = started_at + timedelta(seconds=11)
            assert await store.claim_task("late-worker") is None
            expired = await store.load_task(successor_id)
            assert expired is not None
            assert expired.status is TaskStatus.FAILED
            assert expired.retry_series is not None
            assert expired.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
            assert expired.error == {"code": "elapsed_exhausted"}
            assert expired.status_payload is not None
            receipt_key = expired.status_payload["settlement_idempotency_key"]
            assert isinstance(receipt_key, str)
            receipt = await store.load_task_retry_settlement(successor_id, receipt_key)
            assert receipt is not None
            assert receipt.task == expired
            assert receipt.successor is None
            assert [event.type for event in receipt.events] == [
                TaskRetryEventType.ATTEMPT_SETTLED,
                TaskRetryEventType.SERIES_TERMINAL,
            ]
        finally:
            await _close(store)

    asyncio.run(run())


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_series_succeeds_after_one_delayed_retry(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 12, tzinfo=UTC))
        store = _store_for_kind(
            store_kind,
            tmp_path / "retry-series.sqlite",
            clock,
        )
        policy = TaskRetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0,
            max_total_tokens=20,
            max_estimated_cost=Decimal("0.50"),
        )
        first = await store.create_task(
            TaskCreate(task_id="series-first", type="job", retry_policy=policy)
        )
        claimed = await store.claim_task("worker-a")
        assert claimed is not None
        assert claimed.id == first.id

        retry = await store.settle_task_retry_attempt(
            TaskRetrySettlementRequest(
                task_id=claimed.id,
                worker_id="worker-a",
                idempotency_key="first-failure",
                causal_budget_id=_retry_causal_budget_id(claimed),
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
                token_count=7,
                estimated_cost=Decimal("0.10"),
            )
        )

        assert retry.task.status is TaskStatus.FAILED
        assert retry.successor is not None
        assert retry.successor.status is TaskStatus.PENDING
        assert retry.successor.retry_series is not None
        assert retry.successor.retry_series.series_id == first.retry_series.series_id
        assert (
            retry.successor.retry_series.causal_budget_id
            == first.retry_series.causal_budget_id
            == first.retry_series.series_id
        )
        assert retry.successor.retry_series.attempt == 2
        assert retry.successor.retry_series.cumulative_tokens == 7
        assert retry.successor.retry_series.cumulative_estimated_cost == Decimal("0.10")
        assert retry.task.retry_series is not None
        assert retry.task.retry_series.attempts_remaining == 2
        assert retry.task.retry_series.tokens_remaining == 13
        assert retry.task.retry_series.estimated_cost_remaining == Decimal("0.40")
        assert [event.type for event in retry.events] == [
            TaskRetryEventType.ATTEMPT_SETTLED,
            TaskRetryEventType.RETRY_SCHEDULED,
        ]
        assert all("temporary" not in event.model_dump_json() for event in retry.events)

        second = await store.claim_task("worker-b")
        assert second is not None
        assert second.id == retry.successor.id
        completed = await store.settle_task_retry_attempt(
            TaskRetrySettlementRequest(
                task_id=second.id,
                worker_id="worker-b",
                idempotency_key="second-success",
                causal_budget_id=_retry_causal_budget_id(second),
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
                token_count=5,
                estimated_cost=Decimal("0.20"),
            )
        )

        assert completed.successor is None
        assert completed.task.status is TaskStatus.COMPLETED
        assert completed.task.retry_series is not None
        assert completed.task.retry_series.disposition is TaskRetrySeriesDisposition.SUCCEEDED
        assert completed.task.retry_series.cumulative_tokens == 12
        assert completed.task.retry_series.cumulative_estimated_cost == Decimal("0.30")
        await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("retry_series", "causal_budget_id"), "changed-budget"),
        (("retry_series", "policy", "max_attempts"), 4),
        (("retry_series", "started_at"), datetime(2026, 8, 20, tzinfo=UTC)),
        (("retry_series", "cumulative_tokens"), 0),
        (("retry_series", "cumulative_estimated_cost"), Decimal("0.20")),
        (("retry_series", "tokens_remaining"), 20),
        (("retry_series", "estimated_cost_remaining"), Decimal("0.50")),
        (("input",), {"changed": True}),
        (("metadata",), {"changed": True}),
        (("invocation", "origin", "subject"), "changed-subject"),
        (("title",), "changed title"),
        (("available_at",), datetime(2026, 8, 20, tzinfo=UTC)),
        (("created_at",), datetime(2026, 8, 20, tzinfo=UTC)),
    ],
)
def test_task_retry_receipt_rejects_changed_successor_authority(
    path: tuple[str, ...],
    replacement,
) -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(
                task_id="authority-attempt-1",
                type="job",
                title="stable title",
                input={"stable": True},
                metadata={"stable": True},
                retry_policy=TaskRetryPolicy(
                    max_attempts=3,
                    max_total_tokens=20,
                    max_estimated_cost=Decimal("0.50"),
                    initial_backoff_seconds=0,
                ),
            )
        )
        claimed = await store.claim_task("worker")
        assert claimed is not None
        receipt = await store.settle_task_retry_attempt(
            TaskRetrySettlementRequest(
                task_id=claimed.id,
                worker_id="worker",
                idempotency_key="authority-report",
                causal_budget_id=_retry_causal_budget_id(claimed),
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
                token_count=3,
                estimated_cost=Decimal("0.10"),
            )
        )
        payload = receipt.model_dump(mode="python", warnings=False)
        successor = payload["successor"]
        assert isinstance(successor, dict)
        target = successor
        for component in path[:-1]:
            child = target[component]
            assert isinstance(child, dict)
            target = child
        target[path[-1]] = replacement

        with pytest.raises(ValueError, match="retry|successor|authority|cumulative"):
            TaskRetrySettlementResult.model_validate(payload)

    asyncio.run(run())


def test_task_retry_settlement_boundary_revalidates_mutated_custom_store_receipt() -> None:
    class MutatingReceiptStore(InMemoryTaskStore):
        async def settle_task_retry_attempt(self, request):
            receipt = await super().settle_task_retry_attempt(request)
            assert receipt.successor is not None
            object.__setattr__(receipt.successor, "input", {"changed": True})
            return receipt

    async def run() -> None:
        store = MutatingReceiptStore()
        await store.create_task(
            TaskCreate(
                task_id="mutated-receipt-attempt",
                type="job",
                input={"stable": True},
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    initial_backoff_seconds=0,
                ),
            )
        )
        claimed = await store.claim_task("worker")
        assert claimed is not None

        with pytest.raises(ValueError, match="retry-series authority"):
            await settle_task_retry_attempt_with_retry(
                store,
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="worker",
                    idempotency_key="mutated-receipt-report",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                ),
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("idempotency_key", "wrong-settlement-operation"),
        ("request_sha256", "0" * 64),
    ],
)
def test_task_retry_settlement_rejects_immediate_receipt_for_another_operation(
    field_name: str,
    wrong_value: str,
) -> None:
    class WrongOperationReceiptStore(InMemoryTaskStore):
        async def settle_task_retry_attempt(self, request):
            receipt = await super().settle_task_retry_attempt(request)
            return receipt.model_copy(update={field_name: wrong_value})

    async def run() -> None:
        store = WrongOperationReceiptStore()
        await store.create_task(
            TaskCreate(
                task_id=f"wrong-receipt-{field_name}",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=1),
            )
        )
        claimed = await store.claim_task("worker")
        assert claimed is not None

        with pytest.raises(TaskTerminalizationConflict, match="requested operation"):
            await settle_task_retry_attempt_with_retry(
                store,
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="worker",
                    idempotency_key="expected-settlement-operation",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"ok": True},
                ),
            )

    asyncio.run(run())


def test_task_retry_settlement_rejects_immediate_receipt_for_another_task() -> None:
    class WrongTaskReceiptStore(InMemoryTaskStore):
        wrong_receipt: TaskRetrySettlementResult | None = None

        async def settle_task_retry_attempt(self, request):
            assert self.wrong_receipt is not None
            return self.wrong_receipt

    async def run() -> None:
        store = WrongTaskReceiptStore()
        await store.create_task(
            TaskCreate(
                task_id="wrong-receipt-source",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=1),
            )
        )
        source = await store.claim_task("source-worker")
        assert source is not None
        store.wrong_receipt = await InMemoryTaskStore.settle_task_retry_attempt(
            store,
            TaskRetrySettlementRequest(
                task_id=source.id,
                worker_id="source-worker",
                idempotency_key="source-settlement",
                causal_budget_id=_retry_causal_budget_id(source),
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"source": True},
            ),
        )

        await store.create_task(
            TaskCreate(
                task_id="wrong-receipt-target",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=1),
            )
        )
        target = await store.claim_task("target-worker")
        assert target is not None

        with pytest.raises(TaskTerminalizationConflict, match="requested operation"):
            await settle_task_retry_attempt_with_retry(
                store,
                TaskRetrySettlementRequest(
                    task_id=target.id,
                    worker_id="target-worker",
                    idempotency_key="target-settlement",
                    causal_budget_id=_retry_causal_budget_id(target),
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"target": True},
                ),
            )
        unchanged = await store.load_task(target.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.CLAIMED
        assert unchanged.worker_id == "target-worker"

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("tokens", "cost", "expected"),
    [
        (6, Decimal("0.10"), TaskRetrySeriesDisposition.TOKENS_EXHAUSTED),
        (1, Decimal("0.26"), TaskRetrySeriesDisposition.COST_EXHAUSTED),
    ],
)
def test_task_retry_settlement_terminalizes_overspend_without_replay(
    store_kind: str,
    tokens: int,
    cost: Decimal,
    expected: TaskRetrySeriesDisposition,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 12, 30, tzinfo=UTC))
        store = _store_for_kind(store_kind, tmp_path / f"overspend-{expected}.sqlite", clock)
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id=f"overspend-{expected}",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_total_tokens=5,
                        max_estimated_cost=Decimal("0.25"),
                        initial_backoff_seconds=0,
                    ),
                )
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            receipt = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=created.id,
                    worker_id="worker",
                    idempotency_key=f"overspend-{expected}",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                    token_count=tokens,
                    estimated_cost=cost,
                )
            )

            assert receipt.successor is None
            assert receipt.task.status is TaskStatus.FAILED
            assert receipt.task.result is None
            assert receipt.task.error == {"code": expected.value}
            assert receipt.task.retry_series is not None
            assert receipt.task.retry_series.disposition is expected
            assert receipt.task.retry_series.cumulative_tokens == tokens
            assert receipt.task.retry_series.cumulative_estimated_cost == cost
            assert receipt.task.retry_series.tokens_remaining == max(0, 5 - tokens)
            assert receipt.task.retry_series.estimated_cost_remaining == max(
                Decimal(0),
                Decimal("0.25") - cost,
            )
            assert (
                await store.load_task_retry_settlement(
                    created.id,
                    f"overspend-{expected}",
                )
                == receipt
            )
            assert len(await store.list_tasks()) == 1
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_late_active_attempt_cannot_succeed(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started_at = datetime(2026, 8, 19, 12, 40, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = _store_for_kind(store_kind, tmp_path / "late-active.sqlite", clock)
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id="late-active-task",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_elapsed_seconds=1,
                    ),
                )
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            clock.value = started_at + timedelta(seconds=2)
            receipt = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=created.id,
                    worker_id="worker",
                    idempotency_key="late-success",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"too_late": True},
                )
            )

            assert receipt.successor is None
            assert receipt.task.status is TaskStatus.FAILED
            assert receipt.task.result is None
            assert receipt.task.error == {"code": "elapsed_exhausted"}
            assert receipt.task.retry_series is not None
            assert (
                receipt.task.retry_series.disposition
                is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
            )
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_settlement_authenticates_causal_budget_identity(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 12, 45, tzinfo=UTC))
        store = _store_for_kind(store_kind, tmp_path / "causal-budget.sqlite", clock)
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id="causal-budget-task",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            assert created.retry_series is not None
            assert created.retry_series.causal_budget_id == created.retry_series.series_id
            with pytest.raises(TaskTerminalizationConflict, match="causal budget"):
                await store.settle_task_retry_attempt(
                    TaskRetrySettlementRequest(
                        task_id=created.id,
                        worker_id="worker",
                        idempotency_key="wrong-causal-budget",
                        causal_budget_id="another-budget",
                        disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                        error={"code": "temporary"},
                    )
                )
            assert await store.load_task(created.id) == claimed
            assert (
                await store.load_task_retry_settlement(
                    created.id,
                    "wrong-causal-budget",
                )
                is None
            )
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_direct_cancellation_writes_terminal_receipt(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 12, 50, tzinfo=UTC))
        store = _store_for_kind(store_kind, tmp_path / "cancel-receipt.sqlite", clock)
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id="cancel-receipt-task",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            cancelled = await store.cancel_task(created.id, {"code": "operator"})
            assert cancelled.status is TaskStatus.CANCELLED
            assert cancelled.status_payload is not None
            receipt_key = cancelled.status_payload["settlement_idempotency_key"]
            assert isinstance(receipt_key, str)
            receipt = await store.load_task_retry_settlement(created.id, receipt_key)
            assert receipt is not None
            assert receipt.task == cancelled
            assert receipt.successor is None
            assert [event.type for event in receipt.events] == [
                TaskRetryEventType.ATTEMPT_SETTLED,
                TaskRetryEventType.SERIES_TERMINAL,
            ]
            assert all("operator" not in event.model_dump_json() for event in receipt.events)
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_active_cancellation_retains_claim_until_handler_quiesces(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "active-cancellation.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="active-cancellation-drain",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_total_tokens=5,
                    max_estimated_cost=Decimal("1.00"),
                ),
            )
        )
        started = asyncio.Event()
        cancellation_delivered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancellation_delivered.set()
                await release.wait()
                return TaskRetryAttemptReport(
                    idempotency_key="cancelled-handler-accounting",
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "handler_quiesced"},
                    token_count=7,
                    estimated_cost=Decimal("1.50"),
                )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="cancellation-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            requested = await store.cancel_task(
                "active-cancellation-drain",
                {"code": "operator"},
            )
            assert requested.status is TaskStatus.CLAIMED
            assert requested.worker_id == "cancellation-owner"
            assert requested.status_reason == "retry_cancellation_requested"

            await asyncio.wait_for(cancellation_delivered.wait(), timeout=3)
            await asyncio.sleep(1.1)
            draining = await store.load_task("active-cancellation-drain")
            assert draining is not None
            assert draining.status is TaskStatus.CLAIMED
            assert draining.worker_id == "cancellation-owner"
            assert draining.lease_expires_at is not None
            assert await store.claim_task("competing-worker") is None

            release.set()
            assert await asyncio.wait_for(worker, timeout=3) == 1
            terminal = await store.load_task("active-cancellation-drain")
            assert terminal is not None
            assert terminal.status is TaskStatus.CANCELLED
            assert terminal.worker_id is None
            assert terminal.error == {"code": "operator"}
            assert terminal.retry_series is not None
            assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
            assert terminal.retry_series.cumulative_tokens == 7
            assert terminal.retry_series.cumulative_estimated_cost == Decimal("1.50")
            assert terminal.retry_series.tokens_remaining == 0
            assert terminal.retry_series.estimated_cost_remaining == Decimal(0)
            assert terminal.status_payload is not None
            receipt_key = terminal.status_payload["settlement_idempotency_key"]
            assert isinstance(receipt_key, str)
            assert await store.load_task_retry_settlement(terminal.id, receipt_key) is not None
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_deadline_drain_honors_late_operator_cancellation(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "deadline-cancellation-race.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="deadline-cancellation-race",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2, max_elapsed_seconds=0.1),
            )
        )
        cancellation_delivered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancellation_delivered.set()
                await release.wait()
                return TaskRetryAttemptReport(
                    idempotency_key="deadline-cancellation-accounting",
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "handler_quiesced"},
                    token_count=9,
                    estimated_cost=Decimal("2.25"),
                )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="deadline-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(cancellation_delivered.wait(), timeout=3)
            requested = await store.cancel_task(
                "deadline-cancellation-race",
                {"code": "operator"},
            )
            assert requested.status_reason == "retry_cancellation_requested"
            release.set()

            assert await asyncio.wait_for(worker, timeout=3) == 1
            terminal = await store.load_task("deadline-cancellation-race")
            assert terminal is not None
            assert terminal.status is TaskStatus.CANCELLED
            assert terminal.error == {"code": "operator"}
            assert terminal.retry_series is not None
            assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
            assert terminal.retry_series.cumulative_tokens == 9
            assert terminal.retry_series.cumulative_estimated_cost == Decimal("2.25")
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_worker_shutdown_releases_quiescent_unreported_attempt(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "shutdown-release.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="shutdown-release",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        started = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            started.set()
            await asyncio.sleep(30)

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="shutdown-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            worker.cancel("shutdown worker")
            with pytest.raises(asyncio.CancelledError, match="shutdown worker"):
                await asyncio.wait_for(worker, timeout=3)
            assert worker.cancelled()

            released = await store.load_task("shutdown-release")
            assert released is not None
            assert released.status is TaskStatus.PENDING
            assert released.worker_id is None
            assert released.lease_expires_at is None
            assert await store.claim_task("replacement-owner") is not None
        finally:
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_worker_shutdown_settles_quiescent_handler_report(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "shutdown-report.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="shutdown-report",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        started = asyncio.Event()
        cancellation_delivered = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancellation_delivered.set()
                return TaskRetryAttemptReport(
                    idempotency_key="shutdown-report-settlement",
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "worker_shutdown"},
                    token_count=5,
                    estimated_cost=Decimal("1.25"),
                    retry_after_seconds=0,
                )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="shutdown-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            worker.cancel("shutdown worker")
            await asyncio.wait_for(cancellation_delivered.wait(), timeout=3)

            with pytest.raises(asyncio.CancelledError, match="shutdown worker"):
                await asyncio.wait_for(worker, timeout=3)
            assert worker.cancelled()
            settled = await store.load_task("shutdown-report")
            assert settled is not None
            assert settled.status is TaskStatus.FAILED
            assert settled.retry_series is not None
            assert settled.retry_series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
            assert settled.retry_series.cumulative_tokens == 5
            assert settled.retry_series.cumulative_estimated_cost == Decimal("1.25")
            assert settled.retry_series.successor_task_id is not None
            successor = await store.load_task(settled.retry_series.successor_task_id)
            assert successor is not None
            assert successor.status is TaskStatus.PENDING
        finally:
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_worker_shutdown_reconciles_cancellation_requested_while_draining(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "shutdown-late-cancellation.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="shutdown-late-cancellation",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        started = asyncio.Event()
        cancellation_delivered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancellation_delivered.set()
                await release.wait()
                return TaskRetryAttemptReport(
                    idempotency_key="shutdown-late-accounting",
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "handler_quiesced"},
                    token_count=11,
                    estimated_cost=Decimal("2.75"),
                )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="shutdown-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            worker.cancel("shutdown worker")
            await asyncio.wait_for(cancellation_delivered.wait(), timeout=3)
            requested = await store.cancel_task(
                "shutdown-late-cancellation",
                {"code": "operator"},
            )
            assert requested.status_reason == "retry_cancellation_requested"
            with pytest.raises(TaskTerminalizationConflict, match="still draining"):
                await store.release_task(requested.id, "shutdown-owner")
            release.set()

            with pytest.raises(asyncio.CancelledError, match="shutdown worker"):
                await asyncio.wait_for(worker, timeout=3)
            assert worker.cancelled()
            terminal = await store.load_task("shutdown-late-cancellation")
            assert terminal is not None
            assert terminal.status is TaskStatus.CANCELLED
            assert terminal.error == {"code": "operator"}
            assert terminal.retry_series is not None
            assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
            assert terminal.retry_series.cumulative_tokens == 11
            assert terminal.retry_series.cumulative_estimated_cost == Decimal("2.75")
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_owner_shutdown_settles_existing_cancellation_request(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "shutdown-cancellation.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="shutdown-cancellation",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        started = asyncio.Event()
        cancellation_delivered = asyncio.Event()
        release = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancellation_delivered.set()
                await release.wait()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="cancellation-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            requested = await store.cancel_task(
                "shutdown-cancellation",
                {"code": "operator"},
            )
            assert requested.status_reason == "retry_cancellation_requested"

            worker.cancel("shutdown worker")
            await asyncio.wait_for(cancellation_delivered.wait(), timeout=3)
            await asyncio.sleep(1.1)

            draining = await store.load_task("shutdown-cancellation")
            assert draining is not None
            assert draining.status is TaskStatus.CLAIMED
            assert draining.worker_id == "cancellation-owner"
            assert draining.lease_expires_at is not None
            assert await store.claim_task("competing-worker") is None

            release.set()
            with pytest.raises(asyncio.CancelledError, match="shutdown worker"):
                await asyncio.wait_for(worker, timeout=3)
            assert worker.cancelled()

            terminal = await store.load_task("shutdown-cancellation")
            assert terminal is not None
            assert terminal.status is TaskStatus.CANCELLED
            assert terminal.worker_id is None
            assert terminal.error == {"code": "operator"}
            assert terminal.retry_series is not None
            assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_durable_cancellation_wins_over_later_deadline(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started_at = datetime(2026, 8, 20, 8, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = _store_for_kind(
            store_kind,
            tmp_path / f"cancellation-before-deadline-{store_kind}.sqlite",
            clock,
        )
        try:
            await store.create_task(
                TaskCreate(
                    task_id="cancellation-before-deadline",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_elapsed_seconds=10,
                    ),
                )
            )
            claimed = await store.claim_task("cancellation-owner")
            assert claimed is not None
            requested = await store.cancel_task(
                claimed.id,
                {"code": "operator"},
            )
            assert requested.status_reason == "retry_cancellation_requested"
            assert requested.status_payload is not None
            settlement_key = requested.status_payload["settlement_idempotency_key"]
            assert isinstance(settlement_key, str)

            clock.value = started_at + timedelta(seconds=11)
            assert await store.enforce_task_retry_deadline(claimed.id, "cancellation-owner") is None
            receipt = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="cancellation-owner",
                    idempotency_key=settlement_key,
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "operator"},
                )
            )

            assert receipt.successor is None
            assert receipt.task.status is TaskStatus.CANCELLED
            assert receipt.task.error == {"code": "operator"}
            assert receipt.task.retry_series is not None
            assert receipt.task.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "action_name",
    ["pause_task", "block_task", "mark_task_needs_attention"],
)
def test_task_retry_cancellation_drain_rejects_held_state_transition(
    store_kind: str,
    action_name: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / f"cancellation-hold-{action_name}.sqlite")
        )
        try:
            await store.create_task(
                TaskCreate(
                    task_id="cancellation-hold",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            claimed = await store.claim_task("cancellation-owner")
            assert claimed is not None
            requested = await store.cancel_task(
                claimed.id,
                {"code": "operator"},
            )

            action = getattr(store, action_name)
            with pytest.raises(TaskTerminalizationConflict, match="still draining"):
                await action(claimed.id)

            preserved = await store.load_task(claimed.id)
            assert preserved == requested
            assert preserved is not None
            assert preserved.worker_id == "cancellation-owner"
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("policy", "tokens", "cost", "expected"),
    [
        (
            TaskRetryPolicy(max_attempts=1, initial_backoff_seconds=0),
            0,
            Decimal(0),
            TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED,
        ),
        (
            TaskRetryPolicy(
                max_attempts=2,
                max_elapsed_seconds=5,
                initial_backoff_seconds=5,
            ),
            0,
            Decimal(0),
            TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED,
        ),
        (
            TaskRetryPolicy(
                max_attempts=2,
                max_total_tokens=5,
                initial_backoff_seconds=0,
            ),
            5,
            Decimal(0),
            TaskRetrySeriesDisposition.TOKENS_EXHAUSTED,
        ),
        (
            TaskRetryPolicy(
                max_attempts=2,
                max_estimated_cost=Decimal("0.25"),
                initial_backoff_seconds=0,
            ),
            0,
            Decimal("0.25"),
            TaskRetrySeriesDisposition.COST_EXHAUSTED,
        ),
    ],
)
def test_task_retry_series_cumulative_ceiling_prevents_successor(
    store_kind: str,
    policy: TaskRetryPolicy,
    tokens: int,
    cost: Decimal,
    expected: TaskRetrySeriesDisposition,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 13, tzinfo=UTC))
        store = _store_for_kind(store_kind, tmp_path / f"{expected}.sqlite", clock)
        try:
            await store.create_task(
                TaskCreate(task_id=f"task-{expected}", type="job", retry_policy=policy)
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            receipt = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="worker",
                    idempotency_key=f"settle-{expected}",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "retryable"},
                    token_count=tokens,
                    estimated_cost=cost,
                )
            )
            assert receipt.successor is None
            assert receipt.task.retry_series is not None
            assert receipt.task.retry_series.disposition is expected
            assert receipt.task.retry_series.next_eligible_at is None
            assert receipt.task.status_payload is not None
            assert receipt.task.status_payload["next_eligible_at"] is None
            assert receipt.events[-1].type is TaskRetryEventType.SERIES_TERMINAL
            assert await store.claim_task("other-worker") is None
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_settlement_replays_exact_intent_and_rejects_changed_intent(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 14, tzinfo=UTC))
        store = _store_for_kind(store_kind, tmp_path / "replay.sqlite", clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="task-replay",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        initial_backoff_seconds=0,
                    ),
                )
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            request = TaskRetrySettlementRequest(
                task_id="task-replay",
                worker_id="worker",
                idempotency_key="retry-report",
                causal_budget_id=_retry_causal_budget_id(claimed),
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
            )
            first = await store.settle_task_retry_attempt(request)
            replay = await store.settle_task_retry_attempt(request)
            assert replay == first
            assert await store.load_task_retry_settlement("task-replay", "retry-report") == first
            with pytest.raises(TaskTerminalizationConflict):
                await store.settle_task_retry_attempt(
                    request.model_copy(update={"error": {"code": "changed"}})
                )
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_task_retry_settlement_converges_across_connections(tmp_path: Path) -> None:
    async def run() -> None:
        path = tmp_path / "race.sqlite"
        clock = _MutableClock(datetime(2026, 8, 19, 15, tzinfo=UTC))
        first = SQLiteTaskStore(path, clock=clock)
        second = SQLiteTaskStore(path, clock=clock)
        try:
            await first.create_task(
                TaskCreate(
                    task_id="task-race",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        initial_backoff_seconds=0,
                    ),
                )
            )
            claimed = await first.claim_task("worker")
            assert claimed is not None
            request = TaskRetrySettlementRequest(
                task_id="task-race",
                worker_id="worker",
                idempotency_key="same-report",
                causal_budget_id=_retry_causal_budget_id(claimed),
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
            )
            settled = await asyncio.gather(
                first.settle_task_retry_attempt(request),
                second.settle_task_retry_attempt(request),
            )
            assert settled[0] == settled[1]
            tasks = await first.list_tasks()
            assert len(tasks) == 2
            assert len({task.id for task in tasks}) == 2
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_sqlite_task_retry_backoff_survives_restart_and_cancellation(tmp_path: Path) -> None:
    async def run() -> None:
        path = tmp_path / "restart.sqlite"
        started_at = datetime(2026, 8, 19, 16, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = SQLiteTaskStore(path, clock=clock)
        await store.create_task(
            TaskCreate(
                task_id="task-restart",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    initial_backoff_seconds=30,
                ),
            )
        )
        claimed = await store.claim_task("worker")
        assert claimed is not None
        receipt = await store.settle_task_retry_attempt(
            TaskRetrySettlementRequest(
                task_id="task-restart",
                worker_id="worker",
                idempotency_key="restart-report",
                causal_budget_id=_retry_causal_budget_id(claimed),
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
            )
        )
        assert receipt.successor is not None
        successor_id = receipt.successor.id
        await store.close()

        clock.value = started_at + timedelta(seconds=29)
        reopened = SQLiteTaskStore(path, clock=clock)
        try:
            assert await reopened.claim_task("early-worker") is None
            cancelled = await reopened.cancel_task(
                successor_id,
                {"code": "operator_cancelled"},
            )
            assert cancelled.retry_series is not None
            assert cancelled.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
            clock.value = started_at + timedelta(seconds=31)
            assert await reopened.claim_task("late-worker") is None
        finally:
            await reopened.close()

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_series_lease_loss_and_legacy_terminalization_fail_closed(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started_at = datetime(2026, 8, 19, 17, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = _store_for_kind(store_kind, tmp_path / "lease.sqlite", clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="task-lease",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            claimed = await store.claim_task("worker", lease_seconds=1)
            assert claimed is not None
            with pytest.raises(ValueError, match="settle_task_retry_attempt"):
                await store.complete_task("task-lease", {"wrong": "boundary"})
            await asyncio.sleep(1.05)
            with pytest.raises(TaskClaimLost):
                await store.settle_task_retry_attempt(
                    TaskRetrySettlementRequest(
                        task_id="task-lease",
                        worker_id="worker",
                        idempotency_key="expired-report",
                        causal_budget_id=_retry_causal_budget_id(claimed),
                        disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                        error={"code": "temporary"},
                    )
                )
            loaded = await store.load_task("task-lease")
            assert loaded is not None
            assert loaded.status is TaskStatus.CLAIMED
            assert await store.load_task_retry_settlement("task-lease", "expired-report") is None
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_series_cannot_enter_generic_session_lifecycle(
    store_kind: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unattached queue work"):
        TaskCreate(
            task_id="invalid-attached-retry",
            type="job",
            session_id="session",
            retry_policy=TaskRetryPolicy(max_attempts=2),
        )

    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 17, 30, tzinfo=UTC))
        store = _store_for_kind(store_kind, tmp_path / "attachment.sqlite", clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry-queue-task",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            with pytest.raises(ValueError, match="cannot attach to sessions"):
                await store.start_task("retry-queue-task")
            loaded = await store.load_task("retry-queue-task")
            assert loaded is not None
            assert loaded.status is TaskStatus.PENDING
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_worker_uses_application_retry_dispositions_across_fresh_workers(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        clock = _MutableClock(datetime(2026, 8, 19, 18, tzinfo=UTC))
        store = _store_for_kind(store_kind, tmp_path / "worker.sqlite", clock)
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="task-worker-series",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        initial_backoff_seconds=0,
                        max_total_tokens=10,
                    ),
                )
            )

            async def retry_handler(_app, task, _worker_id):
                assert task.retry_series is not None
                assert task.retry_series.attempt == 1
                return TaskRetryAttemptReport(
                    idempotency_key="worker-attempt-1",
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "application_classified"},
                    token_count=3,
                )

            assert (
                await run_task_worker(
                    app,
                    store,
                    retry_handler,
                    worker_id="fresh-worker-a",
                    max_tasks=1,
                    poll_interval_s=0.01,
                )
                == 1
            )

            async def success_handler(_app, task, _worker_id):
                assert task.retry_series is not None
                assert task.retry_series.attempt == 2
                assert task.retry_series.cumulative_tokens == 3
                return TaskRetryAttemptReport(
                    idempotency_key="worker-attempt-2",
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"ok": True},
                    token_count=2,
                )

            assert (
                await run_task_worker(
                    app,
                    store,
                    success_handler,
                    worker_id="fresh-worker-b",
                    max_tasks=1,
                    poll_interval_s=0.01,
                )
                == 1
            )
            tasks = await store.list_tasks()
            assert [task.retry_series.attempt for task in tasks if task.retry_series] == [2, 1]
            terminal = next(task for task in tasks if task.status is TaskStatus.COMPLETED)
            assert terminal.retry_series is not None
            assert terminal.retry_series.cumulative_tokens == 5
            assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.SUCCEEDED
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_retry_deadline_probe_retains_live_claim(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started_at = datetime(2026, 8, 19, 18, 30, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = _store_for_kind(store_kind, tmp_path / "deadline-probe.sqlite", clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="deadline-probe",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_elapsed_seconds=10,
                    ),
                )
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            assert not await store.task_retry_deadline_elapsed(claimed.id, "worker")

            clock.value += timedelta(seconds=11)
            assert await store.task_retry_deadline_elapsed(claimed.id, "worker")
            still_owned = await store.load_task(claimed.id)
            assert still_owned is not None
            assert still_owned.status is TaskStatus.CLAIMED
            assert still_owned.worker_id == "worker"
            assert await store.claim_task("competitor") is None
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_task_worker_cancels_active_handler_at_series_deadline(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = (
            InMemoryTaskStore()
            if store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "worker-deadline.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id="task-worker-elapsed",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_elapsed_seconds=0.2,
                        initial_backoff_seconds=0,
                    ),
                )
            )
            cancelled = asyncio.Event()

            async def handler(_app, _task, _worker_id):
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            assert (
                await run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="deadline-worker",
                    lease_seconds=1,
                    max_tasks=1,
                    poll_interval_s=0.01,
                )
                == 1
            )
            assert cancelled.is_set()
            terminal = await store.load_task(created.id)
            assert terminal is not None
            assert terminal.status is TaskStatus.FAILED
            assert terminal.retry_series is not None
            assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
            assert terminal.status_payload is not None
            receipt_key = terminal.status_payload["settlement_idempotency_key"]
            assert isinstance(receipt_key, str)
            assert await store.load_task_retry_settlement(created.id, receipt_key) is not None
        finally:
            await _close(store)

    asyncio.run(run())


def test_task_worker_never_uses_its_wall_clock_as_retry_deadline_authority() -> None:
    async def run() -> None:
        store_clock = _MutableClock(datetime(2020, 1, 1, tzinfo=UTC))
        store = InMemoryTaskStore(clock=store_clock)
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="store-authoritative-deadline",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_elapsed_seconds=3600,
                ),
            )
        )
        called = False

        async def handler(_app, task, _worker_id):
            nonlocal called
            called = True
            assert task.retry_series is not None
            assert task.retry_series.elapsed_deadline == store_clock.value + timedelta(hours=1)
            return TaskRetryAttemptReport(
                idempotency_key="store-authoritative-success",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="skewed-worker",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
            == 1
        )
        assert called
        terminal = await store.load_task("store-authoritative-deadline")
        assert terminal is not None
        assert terminal.status is TaskStatus.COMPLETED

    asyncio.run(run())


def test_task_worker_bounds_noncooperative_deadline_cancellation() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="noncooperative-deadline-handler",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_elapsed_seconds=0.1,
                    max_total_tokens=5,
                    max_estimated_cost=Decimal("1.00"),
                ),
            )
        )
        release = asyncio.Event()
        cancellation_delivered = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancellation_delivered.set()
                await release.wait()
                return TaskRetryAttemptReport(
                    idempotency_key="deadline-handler-accounting",
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "handler_quiesced"},
                    token_count=13,
                    estimated_cost=Decimal("3.25"),
                )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="noncooperative-worker",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(cancellation_delivered.wait(), timeout=3)
            await asyncio.sleep(1.1)
            assert not worker.done()
            draining = await store.load_task("noncooperative-deadline-handler")
            assert draining is not None
            assert draining.status is TaskStatus.CLAIMED
            assert draining.worker_id == "noncooperative-worker"
            assert draining.lease_expires_at is not None
            assert await store.claim_task("competing-worker") is None

            release.set()
            assert await asyncio.wait_for(worker, timeout=3) == 1
            terminal = await store.load_task("noncooperative-deadline-handler")
            assert terminal is not None
            assert terminal.status is TaskStatus.FAILED
            assert terminal.retry_series is not None
            assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
            assert terminal.retry_series.cumulative_tokens == 13
            assert terminal.retry_series.cumulative_estimated_cost == Decimal("3.25")
            assert terminal.retry_series.tokens_remaining == 0
            assert terminal.retry_series.estimated_cost_remaining == Decimal(0)
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker

    asyncio.run(run())


def test_task_worker_cancellation_during_deadline_drain_preserves_claim_until_quiescent() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="cancelled-deadline-drain",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_elapsed_seconds=0.1,
                ),
            )
        )
        release = asyncio.Event()
        cancellation_delivered = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancellation_delivered.set()
                await release.wait()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="deadline-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        await asyncio.wait_for(cancellation_delivered.wait(), timeout=3)
        worker.cancel()
        await asyncio.sleep(1.1)

        assert worker.cancelling() == 1
        draining = await store.load_task("cancelled-deadline-drain")
        assert draining is not None
        assert draining.status is TaskStatus.CLAIMED
        assert draining.worker_id == "deadline-owner"
        assert await store.claim_task("competing-worker") is None

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, timeout=3)
        assert worker.cancelled()
        terminal = await store.load_task("cancelled-deadline-drain")
        assert terminal is not None
        assert terminal.status is TaskStatus.FAILED
        assert terminal.retry_series is not None
        assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED

    asyncio.run(run())


def test_task_worker_defers_cancellation_during_deadline_terminalization() -> None:
    class BlockingDeadlineStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalization_started = asyncio.Event()
            self.allow_terminalization = asyncio.Event()

        async def enforce_task_retry_deadline(self, task_id, worker_id, **kwargs):
            self.terminalization_started.set()
            await self.allow_terminalization.wait()
            return await super().enforce_task_retry_deadline(task_id, worker_id, **kwargs)

    async def run() -> None:
        store = BlockingDeadlineStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="cancelled-deadline-terminalization",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_elapsed_seconds=0.1,
                ),
            )
        )

        async def handler(_app, _task, _worker_id):
            await asyncio.sleep(30)

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="deadline-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        await asyncio.wait_for(store.terminalization_started.wait(), timeout=3)
        worker.cancel()
        await asyncio.sleep(0)

        assert worker.cancelling() == 1
        assert not worker.done()
        claimed = await store.load_task("cancelled-deadline-terminalization")
        assert claimed is not None
        assert claimed.status is TaskStatus.CLAIMED
        assert claimed.worker_id == "deadline-owner"

        store.allow_terminalization.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, timeout=3)
        assert worker.cancelled()
        terminal = await store.load_task("cancelled-deadline-terminalization")
        assert terminal is not None
        assert terminal.status is TaskStatus.FAILED
        assert terminal.retry_series is not None
        assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED

    asyncio.run(run())


def test_task_worker_reconciles_lost_deadline_enforcement_acknowledgement() -> None:
    class _AckLostDeadlineStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.lost = False

        async def enforce_task_retry_deadline(self, task_id, worker_id, **kwargs):
            receipt = await super().enforce_task_retry_deadline(task_id, worker_id, **kwargs)
            if receipt is not None and not self.lost:
                self.lost = True
                raise ConnectionError("deadline commit acknowledgement lost")
            return receipt

    async def run() -> None:
        store = _AckLostDeadlineStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="deadline-ack-lost",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_elapsed_seconds=0.1,
                ),
            )
        )
        cancelled = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                return TaskRetryAttemptReport(
                    idempotency_key="deadline-ack-lost-accounting",
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "handler_quiesced"},
                    token_count=19,
                    estimated_cost=Decimal("4.75"),
                )

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="deadline-ack-worker",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
            == 1
        )
        assert store.lost
        assert cancelled.is_set()
        terminal = await store.load_task("deadline-ack-lost")
        assert terminal is not None
        assert terminal.status is TaskStatus.FAILED
        assert terminal.retry_series is not None
        assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
        assert terminal.retry_series.cumulative_tokens == 19
        assert terminal.retry_series.cumulative_estimated_cost == Decimal("4.75")

    asyncio.run(run())


def test_task_worker_retries_indeterminate_deadline_enforcement() -> None:
    class _TransientDeadlineStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.enforcement_calls = 0

        async def enforce_task_retry_deadline(self, task_id, worker_id, **kwargs):
            self.enforcement_calls += 1
            if self.enforcement_calls == 1:
                raise ConnectionError("deadline check unavailable")
            return await super().enforce_task_retry_deadline(task_id, worker_id, **kwargs)

    async def run() -> None:
        store = _TransientDeadlineStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="deadline-transient-failure",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_elapsed_seconds=0.1,
                ),
            )
        )
        cancelled = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="deadline-transient-worker",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
            == 1
        )
        assert store.enforcement_calls == 2
        assert cancelled.is_set()
        terminal = await store.load_task("deadline-transient-failure")
        assert terminal is not None
        assert terminal.status is TaskStatus.FAILED
        assert terminal.retry_series is not None
        assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED

    asyncio.run(run())


def test_task_worker_reconciles_lost_retry_settlement_acknowledgement() -> None:
    class _AckLostStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.lost = False

        async def settle_task_retry_attempt(self, request):
            receipt = await super().settle_task_retry_attempt(request)
            if not self.lost:
                self.lost = True
                raise ConnectionError("commit acknowledgement lost")
            return receipt

    async def run() -> None:
        store = _AckLostStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="task-ack-lost",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=1),
            )
        )

        async def handler(_app, _task, _worker_id):
            return TaskRetryAttemptReport(
                idempotency_key="ack-lost-report",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker",
                max_tasks=1,
                poll_interval_s=0.01,
            )
            == 1
        )
        task = await store.load_task("task-ack-lost")
        assert task is not None
        assert task.status is TaskStatus.COMPLETED
        assert store.lost is True

    asyncio.run(run())


def test_task_retry_worker_retains_lease_and_settles_before_redelivering_cancellation() -> None:
    class BlockingSettlementStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.settlement_started = asyncio.Event()
            self.allow_settlement = asyncio.Event()

        async def settle_task_retry_attempt(self, request):
            self.settlement_started.set()
            await self.allow_settlement.wait()
            return await super().settle_task_retry_attempt(request)

    async def run() -> None:
        store = BlockingSettlementStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="cancelled-during-attempt-settlement",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=1),
            )
        )
        effects: list[str] = []

        async def handler(_app, _task, _worker_id):
            effects.append("completed")
            return TaskRetryAttemptReport(
                idempotency_key="cancelled-settlement-report",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="settlement-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(store.settlement_started.wait(), timeout=3)
            worker.cancel("shutdown during attempt settlement")
            await asyncio.sleep(1.1)

            assert worker.cancelling() == 1
            assert not worker.done()
            assert await store.reclaim_expired(query=None) == []
            assert await store.claim_task("competing-worker") is None

            store.allow_settlement.set()
            with pytest.raises(asyncio.CancelledError, match="shutdown during attempt settlement"):
                await asyncio.wait_for(worker, timeout=3)
            assert worker.cancelled()
            settled = await store.load_task("cancelled-during-attempt-settlement")
            assert settled is not None
            assert settled.status is TaskStatus.COMPLETED
            assert settled.worker_id is None
            assert settled.retry_series is not None
            assert settled.retry_series.disposition is TaskRetrySeriesDisposition.SUCCEEDED
            assert effects == ["completed"]
            assert await store.claim_task("replacement-worker") is None
        finally:
            store.allow_settlement.set()
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker

    asyncio.run(run())


def test_task_retry_worker_classifies_child_cancellation_as_handler_failure() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="child-cancelled-handler",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        effects: list[str] = []

        async def handler(_app, _task, _worker_id):
            effects.append("dispatched")
            raise asyncio.CancelledError("provider child cancelled")

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="child-cancellation-worker",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
            == 1
        )
        owner = asyncio.current_task()
        assert owner is not None
        assert owner.cancelling() == 0
        terminal = await store.load_task("child-cancelled-handler")
        assert terminal is not None
        assert terminal.status is TaskStatus.FAILED
        assert terminal.retry_series is not None
        assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.NON_RETRYABLE_FAILURE
        assert terminal.error == {
            "error": "RuntimeError",
            "message": (
                "Task retry handler was unexpectedly cancelled without cancellation "
                "of its owning worker."
            ),
        }
        assert terminal.retry_series.successor_task_id is None
        assert effects == ["dispatched"]
        assert await store.claim_task("replacement-worker") is None

    asyncio.run(run())


def test_task_retry_worker_does_not_classify_heartbeat_failure_as_handler_failure() -> None:
    class FailingHeartbeatStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_failed = asyncio.Event()

        async def heartbeat(self, task_id, worker_id, *, extend_seconds=300):
            self.heartbeat_failed.set()
            raise ConnectionError("transient task heartbeat failure")

    async def run() -> None:
        store = FailingHeartbeatStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="retry-heartbeat-failure",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        handler_cancelled = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise

        with pytest.raises(ConnectionError, match="transient task heartbeat failure"):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="heartbeat-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )

        assert store.heartbeat_failed.is_set()
        assert handler_cancelled.is_set()
        retained = await store.load_task("retry-heartbeat-failure")
        assert retained is not None
        assert retained.status is TaskStatus.CLAIMED
        assert retained.worker_id == "heartbeat-owner"
        assert retained.error is None
        assert retained.retry_series is not None
        assert retained.retry_series.disposition is TaskRetrySeriesDisposition.ACTIVE
        assert retained.retry_series.successor_task_id is None

    asyncio.run(run())


def test_task_retry_worker_preserves_cancellation_when_heartbeat_fails_during_settlement() -> None:
    class FailingSettlementHeartbeatStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.settlement_started = asyncio.Event()
            self.heartbeat_failed = asyncio.Event()
            self.allow_settlement = asyncio.Event()

        async def heartbeat(self, task_id, worker_id, *, extend_seconds=300):
            if self.settlement_started.is_set():
                self.heartbeat_failed.set()
                raise ConnectionError("settlement heartbeat failure")
            return await super().heartbeat(
                task_id,
                worker_id,
                extend_seconds=extend_seconds,
            )

        async def settle_task_retry_attempt(self, request):
            self.settlement_started.set()
            await self.allow_settlement.wait()
            return await super().settle_task_retry_attempt(request)

    async def run() -> None:
        store = FailingSettlementHeartbeatStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="cancelled-with-settlement-heartbeat-failure",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=1),
            )
        )

        async def handler(_app, _task, _worker_id):
            return TaskRetryAttemptReport(
                idempotency_key="settlement-heartbeat-report",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="settlement-heartbeat-owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(store.heartbeat_failed.wait(), timeout=3)
            worker.cancel("shutdown after settlement heartbeat failure")
            store.allow_settlement.set()
            with pytest.raises(
                asyncio.CancelledError,
                match="shutdown after settlement heartbeat failure",
            ) as exc_info:
                await asyncio.wait_for(worker, timeout=3)

            assert worker.cancelled()
            assert isinstance(exc_info.value.__cause__, ConnectionError)
            assert str(exc_info.value.__cause__) == "settlement heartbeat failure"
            settled = await store.load_task("cancelled-with-settlement-heartbeat-failure")
            assert settled is not None
            assert settled.status is TaskStatus.COMPLETED
            assert settled.worker_id is None
            assert settled.retry_series is not None
            assert settled.retry_series.disposition is TaskRetrySeriesDisposition.SUCCEEDED
        finally:
            store.allow_settlement.set()
            if not worker.done():
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker

    asyncio.run(run())


def test_task_retry_worker_classifies_grouped_child_cancellation_as_handler_failure() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="grouped-child-cancelled-handler",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        effects: list[str] = []

        async def handler(_app, _task, _worker_id):
            effects.append("dispatched")
            raise BaseExceptionGroup(
                "provider child cancellation",
                [asyncio.CancelledError("provider child cancelled")],
            )

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="grouped-child-cancellation-worker",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
            == 1
        )
        owner = asyncio.current_task()
        assert owner is not None
        assert owner.cancelling() == 0
        terminal = await store.load_task("grouped-child-cancelled-handler")
        assert terminal is not None
        assert terminal.status is TaskStatus.FAILED
        assert terminal.retry_series is not None
        assert terminal.retry_series.disposition is TaskRetrySeriesDisposition.NON_RETRYABLE_FAILURE
        assert terminal.error == {
            "error": "RuntimeError",
            "message": (
                "Task retry handler was unexpectedly cancelled without cancellation "
                "of its owning worker."
            ),
        }
        assert terminal.retry_series.successor_task_id is None
        assert effects == ["dispatched"]
        assert await store.claim_task("replacement-worker") is None

    asyncio.run(run())


def test_task_retry_worker_preserves_owner_cancellation_when_handler_cleanup_groups_it() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="grouped-owner-cancelled-handler",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        handler_started = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            handler_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "handler cancellation cleanup",
                    [cancellation, RuntimeError("handler cleanup failed")],
                ) from None

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="grouped-owner-cancellation-worker",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        await asyncio.wait_for(handler_started.wait(), timeout=3)
        worker.cancel("shutdown grouped handler")
        with pytest.raises(asyncio.CancelledError, match="shutdown grouped handler") as exc_info:
            await asyncio.wait_for(worker, timeout=3)

        assert worker.cancelled()
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "handler cleanup failed"
        released = await store.load_task("grouped-owner-cancelled-handler")
        assert released is not None
        assert released.status is TaskStatus.PENDING
        assert released.worker_id is None
        assert released.lease_expires_at is None
        assert released.retry_series is not None
        assert released.retry_series.disposition is TaskRetrySeriesDisposition.ACTIVE

    asyncio.run(run())


def test_task_worker_rejects_retry_report_for_legacy_task_without_stopping() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="legacy-task", type="legacy"))
        app = CayuApp(task_store=store, enable_logging=False)

        async def handler(_app, _task, _worker_id):
            return TaskRetryAttemptReport(
                idempotency_key="legacy-report",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="legacy-worker",
                max_tasks=1,
                poll_interval_s=0.01,
            )
            == 1
        )
        task = await store.load_task("legacy-task")
        assert task is not None
        assert task.status is TaskStatus.FAILED
        assert task.error == {
            "error": "TypeError",
            "message": "TaskRetryAttemptReport requires a task retry policy.",
        }

    asyncio.run(run())


def test_task_worker_rejects_corrupted_retry_authority_before_handler() -> None:
    class CorruptingStore(InMemoryTaskStore):
        async def claim_task(self, worker_id, query=None, *, lease_seconds=300):
            claimed = await super().claim_task(
                worker_id,
                query,
                lease_seconds=lease_seconds,
            )
            if claimed is None:
                return None
            return claimed.model_copy(update={"input": {"changed": True}}, deep=True)

    async def run() -> None:
        store = CorruptingStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="corrupted-authority",
                type="job",
                input={"stable": True},
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        handler_calls = 0

        async def handler(_app, _task, _worker_id):
            nonlocal handler_calls
            handler_calls += 1
            return None

        with pytest.raises(ValueError, match="retry-series authority"):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker",
                max_tasks=1,
                poll_interval_s=0.01,
            )
        assert handler_calls == 0

    asyncio.run(run())


def test_task_retry_worker_example_continues_across_fresh_processes() -> None:
    repository = Path(__file__).parents[2]
    completed = subprocess.run(
        [sys.executable, str(repository / "examples" / "task_retry_worker.py")],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        timeout=30,
    )
    assert "attempts 2 tokens 7 disposition succeeded" in completed.stdout


def test_sqlite_revision_45_migrates_existing_non_retry_tasks(tmp_path: Path) -> None:
    path = tmp_path / "migrate.sqlite"

    async def seed() -> None:
        store = SQLiteTaskStore(path)
        await store.create_task(TaskCreate(task_id="existing-task", type="job"))
        await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TABLE cayu_task_retry_settlements")
        connection.execute("ALTER TABLE cayu_tasks DROP COLUMN retry_series_json")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 45")
        connection.execute("PRAGMA user_version = 44")
        connection.commit()
    finally:
        connection.close()

    async def migrate() -> None:
        store = SQLiteTaskStore(path, schema_mode=schema_migrations.SchemaMode.MIGRATE)
        try:
            existing = await store.load_task("existing-task")
            assert existing is not None
            assert existing.retry_series is None
            created = await store.create_task(
                TaskCreate(
                    task_id="retry-task",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            assert created.retry_series is not None
        finally:
            await store.close()

    asyncio.run(migrate())


def test_sqlite_revision_45_validation_rejects_missing_retry_receipts(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite"

    async def create() -> None:
        store = SQLiteTaskStore(path)
        await store.close()

    asyncio.run(create())
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TABLE cayu_task_retry_settlements")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="retry-series schema"):
        SQLiteTaskStore(path, schema_mode=schema_migrations.SchemaMode.VALIDATE)
