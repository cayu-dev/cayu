"""Scrubbing stored task failure text, on every task store."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from tests.core.test_sleep_until import _own_postgres_database

from cayu.storage import PostgresTaskStore, SQLiteTaskStore
from cayu.storage.migrations import SchemaMode
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)
from cayu.tasks._failure_scrub import (
    SCRUB_RECORD_KEY,
    TaskFailureScrub,
    failure_text_redaction,
    scrub_task_failure,
)
from cayu.tasks.records import Task

# Made up at runtime so no credential-shaped literal sits in the source.
_TOKEN = "gh" + "s_" + "Q7" * 18
_HEADER = "Authorization: Basic " + "eC1hY2Nlc3M" + "tdG9rZW46" + "Zm9v" * 6
_MESSAGE = f"git -c http.extraheader='{_HEADER}' fetch timed out using {_TOKEN}"
_AT = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def _scrub(*, dry_run: bool = False) -> TaskFailureScrub:
    return TaskFailureScrub(redact=failure_text_redaction(), scrubbed_at=_AT, dry_run=dry_run)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryTaskStore()
    if request.param == "sqlite":
        return SQLiteTaskStore(tmp_path / "tasks.db")
    return PostgresTaskStore(_own_postgres_database(request), schema_mode=SchemaMode.CREATE)


async def _failed(store, task_id: str, error: dict) -> tuple[Task, TaskTerminalizationRequest]:
    await store.create_task(TaskCreate(task_id=task_id, type="scout_run"))
    claimed = await store.claim_task("worker_a")
    assert claimed is not None and claimed.id == task_id
    request = TaskTerminalizationRequest(
        task_id=task_id,
        worker_id="worker_a",
        lease_expires_at=claimed.lease_expires_at,
        kind=TaskTerminalKind.FAILED,
        error=error,
        idempotency_key=f"terminal-{task_id}",
    )
    return await store.terminalize_task(request), request


def _text(value) -> str:
    return json.dumps(value.model_dump(mode="json") if isinstance(value, Task) else value)


def test_scrub_masks_shapes_records_counts_and_is_deterministic():
    task = Task(
        id="t1",
        type="scout_run",
        error={"code": "TimeoutExpired", "message": _MESSAGE, "details": [_TOKEN, 3]},
        invocation=_invocation(),
    )
    scrubbed, fields, masked = scrub_task_failure(task, _scrub())
    assert fields == ("error",)
    assert masked == 2
    assert _TOKEN not in _text(scrubbed) and "Zm9v" not in _text(scrubbed)
    assert scrubbed.error["code"] == "TimeoutExpired"
    assert scrubbed.error[SCRUB_RECORD_KEY] == {
        "scrubs": [
            {
                "scrubbed_at": _AT.isoformat(),
                "redaction": "secret_redactor+credential_shapes",
                "masked_values": 2,
            }
        ]
    }
    # The same snapshot scrubbed again elsewhere (a receipt copy) is equal.
    assert scrub_task_failure(task, _scrub())[0] == scrubbed
    # A second pass finds nothing more to mask and leaves the task as it is.
    assert scrub_task_failure(scrubbed, _scrub()) == (scrubbed, (), 0)


def test_scrubbing_again_keeps_the_words_after_a_masked_header():
    redact = failure_text_redaction()
    text = "request Authorization: Basic abc failed with status 401"
    once = redact(text)
    assert once == "request Authorization: [REDACTED_SECRET] failed with status 401"
    assert redact(redact(redact(once))) == once
    # A registered secret masked first leaves its marker; the scheme before it goes too.
    masked = redact("Authorization: Basic [REDACTED_SECRET] failed")
    assert masked == "Authorization: [REDACTED_SECRET] failed"
    assert redact(masked) == masked
    # A registered prefix of a longer credential: the rest of the word goes with it.
    secrets = failure_text_redaction(lambda text: text.replace("known-prefix", "[REDACTED_SECRET]"))
    rotated = secrets("request Authorization: Bearer known-prefix-rotated-secret failed")
    assert rotated == "request Authorization: [REDACTED_SECRET] failed"
    assert secrets(rotated) == rotated
    quoted = redact('{"Authorization": "[REDACTED_SECRET]", "status": 401}')
    assert quoted == '{"Authorization": "[REDACTED_SECRET]", "status": 401}'


def test_reconciliation_rereads_a_receipt_a_scrub_rewrote_between_its_reads():
    from decimal import Decimal

    from cayu.tasks import terminalize_task_with_retry
    from cayu.tasks._terminalization import settle_task_retry_attempt_with_retry
    from cayu.tasks.records import TaskRetryPolicy
    from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetrySettlementRequest

    class ScrubbedBetweenReads(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def terminalize_task(self, request):
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

        async def settle_task_retry_attempt(self, request):
            await super().settle_task_retry_attempt(request)
            raise ConnectionError("acknowledgement lost")

        async def load_task_terminalization_receipt(self, task_id, key):
            receipt = await super().load_task_terminalization_receipt(task_id, key)
            await self.scrub_task_failure_text(task_id, _scrub())
            return receipt

        async def load_task_retry_settlement(self, task_id, key):
            settled = await super().load_task_retry_settlement(task_id, key)
            await self.scrub_task_failure_text(task_id, _scrub())
            return settled

    async def run() -> None:
        store = ScrubbedBetweenReads()
        await store.create_task(TaskCreate(task_id="plain", type="scout_run"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None
        outcome = await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="plain",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error={"message": _MESSAGE},
                idempotency_key="terminal-plain",
            ),
        )
        assert outcome.receipt_reconciled
        assert outcome.task == await store.load_task("plain")
        assert _TOKEN not in _text(outcome.task)

        policy = TaskRetryPolicy(max_attempts=2, initial_backoff_seconds=0, max_total_tokens=20)
        await store.create_task(TaskCreate(task_id="series", type="scout_run", retry_policy=policy))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None and claimed.retry_series is not None
        settled = await settle_task_retry_attempt_with_retry(
            store,
            TaskRetrySettlementRequest(
                task_id="series",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                idempotency_key="attempt-1",
                causal_budget_id=claimed.retry_series.causal_budget_id,
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"message": _MESSAGE},
                token_count=1,
                estimated_cost=Decimal("0"),
            ),
        )
        assert settled.task == await store.load_task("series")
        assert _TOKEN not in _text(settled.task)

    asyncio.run(run())


def test_scrub_covers_a_pending_cancellation_error_and_registered_secrets():
    task = Task(
        id="t2",
        type="scout_run",
        status_payload={"error": {"message": f"cancel: registered-value {_TOKEN}"}, "n": 1},
        invocation=_invocation(),
    )
    redact = failure_text_redaction(lambda text: text.replace("registered-value", "[R]"))
    scrubbed, fields, masked = scrub_task_failure(
        task, TaskFailureScrub(redact=redact, scrubbed_at=_AT)
    )
    assert fields == ("status_payload.error",)
    assert masked == 1
    assert scrubbed.status_payload["n"] == 1
    assert scrubbed.status_payload["error"]["message"] == "cancel: [R] [REDACTED_SECRET]"


def test_store_scrubs_row_and_receipt_together(store):
    async def run() -> None:
        failed, request = await _failed(
            store, "task-a", {"code": "TimeoutExpired", "message": _MESSAGE}
        )
        assert _TOKEN in _text(failed)

        preview = await store.scrub_task_failure_text("task-a", _scrub(dry_run=True))
        assert preview.fields == ("error",) and preview.receipts == 1
        assert preview.masked_values == 1 and not preview.applied
        assert _TOKEN in _text(await store.load_task("task-a"))

        result = await store.scrub_task_failure_text("task-a", _scrub())
        assert result.applied and result.fields == ("error",) and result.receipts == 1
        current = await store.load_task("task-a")
        assert _TOKEN not in _text(current) and "Zm9v" not in _text(current)
        assert current.error[SCRUB_RECORD_KEY]["scrubs"][0]["masked_values"] == 1
        assert current.updated_at == failed.updated_at
        receipt = await store.load_task_terminalization_receipt("task-a", "terminal-task-a")
        assert receipt is not None and receipt.task == current
        # The worker's retried terminalization still replays against the scrubbed row.
        assert await store.terminalize_task(request) == current

        again = await store.scrub_task_failure_text("task-a", _scrub())
        assert (again.fields, again.receipts, again.applied) == ((), 0, False)

    asyncio.run(run())


def test_store_scrubs_a_retry_settlement_receipt_and_its_replay_still_matches(store):
    from decimal import Decimal

    from cayu.tasks.records import TaskRetryPolicy
    from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetrySettlementRequest

    async def run() -> None:
        policy = TaskRetryPolicy(max_attempts=2, initial_backoff_seconds=0, max_total_tokens=20)
        await store.create_task(
            TaskCreate(task_id="series-1", type="scout_run", retry_policy=policy)
        )
        claimed = await store.claim_task("worker_a")
        assert claimed is not None and claimed.retry_series is not None
        request = TaskRetrySettlementRequest(
            task_id=claimed.id,
            worker_id="worker_a",
            lease_expires_at=claimed.lease_expires_at,
            idempotency_key="attempt-1",
            causal_budget_id=claimed.retry_series.causal_budget_id,
            disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
            error={"code": "TimeoutExpired", "message": _MESSAGE},
            token_count=1,
            estimated_cost=Decimal("0"),
        )
        settled = await store.settle_task_retry_attempt(request)
        assert _TOKEN in _text(settled.task)

        result = await store.scrub_task_failure_text("series-1", _scrub())
        assert result.applied and result.fields == ("error",) and result.receipts == 1
        current = await store.load_task("series-1")
        assert _TOKEN not in _text(current)
        replayed = await store.settle_task_retry_attempt(request)
        assert replayed.task == current
        assert replayed.successor == settled.successor

    asyncio.run(run())


def test_store_skips_a_leased_task_and_refuses_an_unknown_one(store):
    async def run() -> None:
        await store.create_task(TaskCreate(task_id="task-held", type="scout_run"))
        assert await store.claim_task("worker_a") is not None
        result = await store.scrub_task_failure_text("task-held", _scrub())
        assert result.skipped_reason == "leased" and not result.applied
        with pytest.raises(KeyError):
            await store.scrub_task_failure_text("task-missing", _scrub())

    asyncio.run(run())


def test_store_lists_tasks_with_failure_text_in_pages(store):
    async def run() -> None:
        for name in ("task-c", "task-a", "task-b"):
            await _failed(store, name, {"message": "plain"})
        await store.create_task(TaskCreate(task_id="task-ok", type="scout_run"))
        first = await store.list_task_ids_with_failure_text(limit=2)
        assert first == ["task-a", "task-b"]
        assert await store.list_task_ids_with_failure_text(after_task_id="task-b") == ["task-c"]
        with pytest.raises(ValueError):
            await store.list_task_ids_with_failure_text(limit=101)

    asyncio.run(run())


def _invocation():
    from tests.core.task_invocation_fixtures import unattributed_task_invocation

    return unattributed_task_invocation()


def test_errors_other_records_bind_exactly_are_left_alone():
    from cayu.tasks._failure_scrub import scrub_skip_reason

    session_failure = Task(
        id="t3",
        type="scout_run",
        error={"message": _MESSAGE, "runtime_task_failure": {"schema": "made-up"}},
        invocation=_invocation(),
    )
    assert scrub_skip_reason(session_failure) == "session_failure"
    reconciled = Task(
        id="t4",
        type="scout_run",
        error={"message": _MESSAGE},
        status_payload={"cancellation_reconciliation": {"made_up": True}},
        invocation=_invocation(),
    )
    assert scrub_skip_reason(reconciled) == "cancellation_reconciliation"
    plain = Task(id="t5", type="scout_run", error={"message": _MESSAGE}, invocation=_invocation())
    assert scrub_skip_reason(plain) is None


def test_store_leaves_a_session_failure_error_unchanged(store):
    async def run() -> None:
        error = {"message": _MESSAGE, "runtime_task_failure": {"schema": "made-up"}}
        failed, _ = await _failed(store, "task-s", error)
        result = await store.scrub_task_failure_text("task-s", _scrub())
        assert result.skipped_reason == "session_failure"
        assert (result.fields, result.receipts, result.applied) == ((), 0, False)
        assert await store.load_task("task-s") == failed

    asyncio.run(run())


def test_reconciliation_retries_when_the_reread_after_a_scrub_fails():
    from cayu.tasks import terminalize_task_with_retry
    from cayu.tasks.terminalization import TaskTerminalizationRetryPolicy

    class RereadFailsOnce(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        loads = 0

        async def terminalize_task(self, request):
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

        async def load_task_terminalization_receipt(self, task_id, key):
            self.loads += 1
            if self.loads == 2:
                raise ConnectionError("reread lost")
            receipt = await super().load_task_terminalization_receipt(task_id, key)
            if self.loads == 1:
                await self.scrub_task_failure_text(task_id, _scrub())
            return receipt

    async def run() -> None:
        store = RereadFailsOnce()
        await store.create_task(TaskCreate(task_id="plain", type="scout_run"))
        claimed = await store.claim_task("worker_a")
        assert claimed is not None
        outcome = await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="plain",
                worker_id="worker_a",
                lease_expires_at=claimed.lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error={"message": _MESSAGE},
                idempotency_key="terminal-plain",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=3, initial_backoff_seconds=0.0, max_backoff_seconds=0.0
            ),
        )
        assert outcome.receipt_reconciled and outcome.attempt_count >= 2
        assert outcome.task == await store.load_task("plain")

    asyncio.run(run())
