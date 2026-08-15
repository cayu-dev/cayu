from __future__ import annotations

import asyncio
import errno
import sqlite3
from datetime import UTC, datetime

import pytest
from psycopg.errors import DiskFull
from pydantic import ValidationError

from cayu import (
    InMemoryTaskStore,
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalizationRetryPolicy,
    TaskTerminalizationUncertain,
    TaskTerminalKind,
    terminalize_task_with_retry,
)


def _request(task_id: str = "task_retry_classification") -> TaskTerminalizationRequest:
    return TaskTerminalizationRequest(
        task_id=task_id,
        worker_id="worker_a",
        kind=TaskTerminalKind.COMPLETED,
        result={"summary": "done"},
        idempotency_key="terminal-classification",
    )


def _sqlite_full_error() -> sqlite3.OperationalError:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA page_size=512")
        connection.execute("PRAGMA max_page_count=2")
        connection.execute("CREATE TABLE payloads(value BLOB)")
        with pytest.raises(sqlite3.OperationalError) as captured:
            connection.execute("INSERT INTO payloads VALUES (zeroblob(100000))")
        return captured.value
    finally:
        connection.close()


def test_terminalization_request_is_detached_and_bounded() -> None:
    result = {"summary": {"changed": 2}}
    request = TaskTerminalizationRequest(
        task_id="task_detached",
        worker_id="worker_a",
        kind=TaskTerminalKind.COMPLETED,
        result=result,
        idempotency_key="é" * 128,
    )
    result["summary"]["changed"] = 999
    assert request.result == {"summary": {"changed": 2}}

    with pytest.raises(ValidationError, match="at most 256 UTF-8 bytes"):
        TaskTerminalizationRequest(
            task_id="task_too_long",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={},
            idempotency_key="é" * 129,
        )
    with pytest.raises(ValidationError, match="requires result and forbids error"):
        TaskTerminalizationRequest(
            task_id="task_invalid_payload",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            error={"message": "wrong field"},
            idempotency_key="invalid-payload",
        )


def test_terminalization_retry_policy_rejects_coerced_or_unbounded_values() -> None:
    with pytest.raises(ValidationError):
        TaskTerminalizationRetryPolicy(max_attempts="3")
    with pytest.raises(ValidationError):
        TaskTerminalizationRetryPolicy(initial_backoff_seconds="0.1")
    with pytest.raises(ValidationError):
        TaskTerminalizationRetryPolicy(max_attempts=11)
    with pytest.raises(ValidationError):
        TaskTerminalizationRetryPolicy(attempt_timeout_seconds=0)


def test_terminalization_helper_requires_store_capability() -> None:
    class LegacyStore(InMemoryTaskStore):
        supports_idempotent_terminalization = False

    async def run() -> None:
        with pytest.raises(ValueError, match="must support idempotent"):
            await terminalize_task_with_retry(LegacyStore(), _request())

    asyncio.run(run())


def test_terminalization_helper_reconstructs_commit_before_error() -> None:
    class CommitThenRaiseStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

    async def run() -> None:
        store = CommitThenRaiseStore()
        await store.create_task(TaskCreate(task_id="task_commit", type="review"))
        assert await store.claim_task("worker_a") is not None
        result = await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="task_commit",
                worker_id="worker_a",
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done"},
                idempotency_key="terminal-1",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            ),
        )

        assert result.task.result == {"summary": "done"}
        assert result.attempt_count == 1
        assert result.receipt_reconciled is True
        assert result.elapsed_seconds >= 0
        assert result.applied_backoff_seconds == 0
        assert store.terminalize_calls == 1

    asyncio.run(run())


def test_terminalization_helper_rejects_receipt_current_task_inconsistency() -> None:
    class InconsistentReceiptStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.reported_current_task: Task | None = None

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.reported_current_task = await super().load_task(request.task_id)
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

        async def load_task(self, task_id: str):
            if self.reported_current_task is not None:
                return self.reported_current_task.model_copy(deep=True)
            return await super().load_task(task_id)

    async def run() -> None:
        store = InconsistentReceiptStore()
        await store.create_task(TaskCreate(task_id="task_inconsistent", type="review"))
        assert await store.claim_task("worker_a") is not None

        with pytest.raises(TaskTerminalizationConflict, match="current terminal task"):
            await terminalize_task_with_retry(
                store,
                TaskTerminalizationRequest(
                    task_id="task_inconsistent",
                    worker_id="worker_a",
                    kind=TaskTerminalKind.COMPLETED,
                    result={"summary": "done"},
                    idempotency_key="terminal-inconsistent",
                ),
                policy=TaskTerminalizationRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=0,
                    max_backoff_seconds=0,
                ),
            )

    asyncio.run(run())


def test_terminalization_helper_retries_exact_request_after_precommit_error() -> None:
    class FailBeforeCommitStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            if self.terminalize_calls == 1:
                raise ConnectionError("write acknowledgement unavailable")
            return await super().terminalize_task(request)

    async def run() -> None:
        store = FailBeforeCommitStore()
        await store.create_task(TaskCreate(task_id="task_retry", type="review"))
        assert await store.claim_task("worker_a") is not None
        result = await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="task_retry",
                worker_id="worker_a",
                kind=TaskTerminalKind.FAILED,
                error={"message": "provider unavailable"},
                idempotency_key="terminal-1",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.01,
                max_backoff_seconds=0.01,
            ),
        )

        assert result.task.error == {"message": "provider unavailable"}
        assert result.attempt_count == 2
        assert result.receipt_reconciled is False
        assert result.applied_backoff_seconds == pytest.approx(0.01)
        assert result.elapsed_seconds >= result.applied_backoff_seconds
        assert store.terminalize_calls == 2

    asyncio.run(run())


def test_terminalization_helper_detaches_request_for_every_retry() -> None:
    class MutatingFailureStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            if self.terminalize_calls == 1:
                assert request.result is not None
                request.result["summary"] = "mutated by store"
                raise ConnectionError("write unavailable")
            assert request.result == {"summary": "done"}
            return await super().terminalize_task(request)

    async def run() -> None:
        store = MutatingFailureStore()
        await store.create_task(TaskCreate(task_id="task_detached_retry", type="review"))
        assert await store.claim_task("worker_a") is not None
        outcome = await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="task_detached_retry",
                worker_id="worker_a",
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done"},
                idempotency_key="detached-retry",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            ),
        )
        assert outcome.task.result == {"summary": "done"}
        assert store.terminalize_calls == 2

    asyncio.run(run())


def test_terminalization_helper_exhaustion_is_bounded_and_content_free() -> None:
    class AlwaysUnavailableStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            raise ConnectionError("secret backend diagnostic")

    async def run() -> None:
        store = AlwaysUnavailableStore()
        await store.create_task(TaskCreate(task_id="task_uncertain", type="review"))
        assert await store.claim_task("worker_a") is not None
        request = TaskTerminalizationRequest(
            task_id="task_uncertain",
            worker_id="worker_a",
            kind=TaskTerminalKind.FAILED,
            error={"secret_payload": "must-not-leak"},
            idempotency_key="terminal-uncertain",
        )

        with pytest.raises(TaskTerminalizationUncertain) as captured:
            await terminalize_task_with_retry(
                store,
                request,
                policy=TaskTerminalizationRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=0.01,
                    max_backoff_seconds=0.01,
                ),
            )

        error = captured.value
        assert error.task_id == "task_uncertain"
        assert error.idempotency_key == "terminal-uncertain"
        assert error.attempt_count == 3
        assert error.error_category == "connection"
        assert error.applied_backoff_seconds == pytest.approx(0.02)
        assert error.elapsed_seconds >= error.applied_backoff_seconds
        assert store.terminalize_calls == 3
        assert "must-not-leak" not in str(error)
        assert "secret backend diagnostic" not in str(error)

    asyncio.run(run())


def test_terminalization_uncertain_bounds_each_evidence_field_in_utf8_bytes() -> None:
    error = TaskTerminalizationUncertain(
        task_id="é" * 300,
        idempotency_key="x" * 300,
        attempt_count=3,
        error_category="connection",
    )

    assert len(error.task_id.encode("utf-8")) <= 256
    assert len(error.idempotency_key.encode("utf-8")) <= 256
    assert error.task_id.endswith("]")
    assert error.idempotency_key.endswith("]")


def test_terminalization_helper_bounds_each_store_attempt() -> None:
    class HungStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            await asyncio.Event().wait()

    async def run() -> None:
        store = HungStore()
        with pytest.raises(TaskTerminalizationUncertain) as captured:
            await terminalize_task_with_retry(
                store,
                _request(),
                policy=TaskTerminalizationRetryPolicy(
                    max_attempts=2,
                    attempt_timeout_seconds=0.01,
                    initial_backoff_seconds=0,
                    max_backoff_seconds=0,
                ),
            )
        assert captured.value.attempt_count == 2
        assert captured.value.error_category == "timeout"
        assert store.terminalize_calls == 2

    asyncio.run(run())


def test_terminalization_helper_does_not_retry_caller_cancellation() -> None:
    class CancelledStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            raise asyncio.CancelledError

    async def run() -> None:
        store = CancelledStore()
        await store.create_task(TaskCreate(task_id="task_cancel", type="review"))
        assert await store.claim_task("worker_a") is not None
        with pytest.raises(asyncio.CancelledError):
            await terminalize_task_with_retry(
                store,
                TaskTerminalizationRequest(
                    task_id="task_cancel",
                    worker_id="worker_a",
                    kind=TaskTerminalKind.COMPLETED,
                    result={"summary": "done"},
                    idempotency_key="terminal-cancel",
                ),
                policy=TaskTerminalizationRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=0,
                    max_backoff_seconds=0,
                ),
            )
        assert store.terminalize_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "raised",
    [
        TaskClaimLost("claim lost"),
        TaskTerminalizationConflict("conflicting intent"),
        ValueError("invalid request"),
    ],
)
def test_terminalization_helper_does_not_retry_deterministic_failures(raised: Exception) -> None:
    class DeterministicFailureStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            raise raised

    async def run() -> None:
        store = DeterministicFailureStore()
        with pytest.raises(type(raised), match=str(raised)):
            await terminalize_task_with_retry(
                store,
                _request(),
                policy=TaskTerminalizationRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=0,
                    max_backoff_seconds=0,
                ),
            )
        assert store.terminalize_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "raised",
    [
        PermissionError(errno.EACCES, "permission denied"),
        FileNotFoundError(errno.ENOENT, "database path missing"),
        OSError(errno.ENOSPC, "disk full"),
        _sqlite_full_error(),
        sqlite3.OperationalError("no such table: task_terminalization_receipts"),
        DiskFull("disk full"),
    ],
)
def test_terminalization_helper_does_not_retry_deterministic_store_errors(
    raised: Exception,
) -> None:
    class DeterministicStoreFailureStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0
            self.receipt_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            raise raised

        async def load_task_terminalization_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ):
            self.receipt_calls += 1
            return await super().load_task_terminalization_receipt(task_id, idempotency_key)

    async def run() -> None:
        store = DeterministicStoreFailureStore()
        with pytest.raises(type(raised)):
            await terminalize_task_with_retry(
                store,
                _request(),
                policy=TaskTerminalizationRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=0,
                    max_backoff_seconds=0,
                ),
            )
        assert store.terminalize_calls == 1
        assert store.receipt_calls == 0

    asyncio.run(run())


def test_terminalization_helper_does_not_catch_base_exception() -> None:
    class WorkerAbort(BaseException):
        pass

    class AbortedStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            raise WorkerAbort

    async def run() -> None:
        store = AbortedStore()
        with pytest.raises(WorkerAbort):
            await terminalize_task_with_retry(store, _request())
        assert store.terminalize_calls == 1

    asyncio.run(run())


def test_terminalization_helper_exhausts_when_write_and_receipt_acknowledgements_fail() -> None:
    class ReceiptUnavailableStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0
            self.receipt_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            if self.terminalize_calls == 1:
                await super().terminalize_task(request)
            raise ConnectionError("write acknowledgement unavailable")

        async def load_task_terminalization_receipt(self, task_id: str, idempotency_key: str):
            self.receipt_calls += 1
            raise TimeoutError("receipt acknowledgement unavailable")

    async def run() -> None:
        store = ReceiptUnavailableStore()
        await store.create_task(TaskCreate(task_id="task_retry_classification", type="review"))
        assert await store.claim_task("worker_a") is not None
        with pytest.raises(TaskTerminalizationUncertain) as captured:
            await terminalize_task_with_retry(
                store,
                _request(),
                policy=TaskTerminalizationRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=0,
                    max_backoff_seconds=0,
                ),
            )
        assert captured.value.attempt_count == 3
        assert captured.value.error_category == "timeout"
        assert store.terminalize_calls == 3
        assert store.receipt_calls == 3

    asyncio.run(run())


def test_terminalization_boundaries_reject_hostile_subclasses() -> None:
    class HostileRequest(TaskTerminalizationRequest):
        pass

    class HostileTask(Task):
        pass

    class HostileReceipt(TaskTerminalizationReceipt):
        pass

    class HostileResultStore(InMemoryTaskStore):
        async def terminalize_task(self, request: TaskTerminalizationRequest):
            task = await super().terminalize_task(request)
            return HostileTask.model_validate(task.model_dump(mode="python"))

    class HostileReceiptStore(InMemoryTaskStore):
        async def terminalize_task(self, request: TaskTerminalizationRequest):
            raise ConnectionError("acknowledgement unavailable")

        async def load_task_terminalization_receipt(self, task_id: str, idempotency_key: str):
            return HostileReceipt(
                task_id=task_id,
                idempotency_key=idempotency_key,
                worker_id="worker_a",
                kind=TaskTerminalKind.COMPLETED,
                request_sha256="0" * 64,
                task=Task(
                    id=task_id,
                    type="review",
                    status="completed",
                    result={"summary": "done"},
                    completed_at=datetime.now(UTC),
                ),
                committed_at=datetime.now(UTC),
            )

    async def run() -> None:
        store = HostileResultStore()
        await store.create_task(TaskCreate(task_id="task_hostile", type="review"))
        assert await store.claim_task("worker_a") is not None
        request = TaskTerminalizationRequest(
            task_id="task_hostile",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="terminal-hostile",
        )
        hostile_request = HostileRequest.model_validate(request.model_dump(mode="python"))
        with pytest.raises(TypeError, match="TaskTerminalizationRequest"):
            await terminalize_task_with_retry(store, hostile_request)
        with pytest.raises(TypeError, match="Task instance"):
            await terminalize_task_with_retry(store, request)
        with pytest.raises(TypeError, match="TaskTerminalizationReceipt"):
            await terminalize_task_with_retry(HostileReceiptStore(), _request())

    asyncio.run(run())
