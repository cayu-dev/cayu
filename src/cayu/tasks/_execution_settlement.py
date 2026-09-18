"""Process-local ownership of exact, positively observed execution settlement."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from cayu._exception_groups import exception_cause, iter_exception_tree, set_exception_cause
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_result_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.tasks.base import Task, TaskStore, copy_task
from cayu.tasks.groups import TaskGroupConflict
from cayu.vaults.redaction import SecretRedactor

_OBSERVATION_TIMEOUT_SECONDS = 1.0


class _WithheldSettlementRedactor(SecretRedactor):
    """Dispatch runtimes do not expose their private secret registry."""

    def redact_text(self, value: str) -> str:
        return "Task group settlement diagnostic withheld."

    def redact_text_bounded(self, value: str, *, max_bytes: int) -> str:
        return self.redact_text(value)[:max_bytes]


class TaskExecutionSettlementPending(RuntimeError):
    """The caller owns ``settlement.retry()``; it never invokes user work.

    Normal task disposition has been attempted unless execution entry failed
    before dispatch. Retain this handle if acknowledgement storage remains
    unavailable. A timed-out write stays owned
    by this handle; retry joins it without redispatch or cancellation. There is
    no background retry loop or global registry. Process loss deliberately
    leaves an unacknowledged durable barrier fenced.
    """

    def __init__(self, settlement: TaskExecutionSettlement) -> None:
        super().__init__("Task group execution acknowledgement remains pending.")
        self.settlement = settlement


class TaskExecutionSettlement:
    """One immutable authority snapshot, one result, at most three calls per retry.

    Only the owner observing natural callback return or proven nondispatch may
    call ``observe``. A failed acknowledgement is not a callback failure. The
    original result remains available to disposition code and on the retry
    handle, including when storage committed but its acknowledgement was lost.
    """

    def __init__(self, store: TaskStore, redactor: SecretRedactor | None = None) -> None:
        self._store = store
        self._redactor = redactor if redactor is not None else _WithheldSettlementRedactor()
        self._authority: Task | None = None
        self.result: Any = None
        self._pending = False
        self._failure: BaseException | None = None
        self._lock = asyncio.Lock()
        self._operation: asyncio.Task[TaskStoreOperationOutcome[None]] | None = None
        self._entry_claim: Task | None = None
        self._entry_operation: asyncio.Task[TaskStoreOperationOutcome[Task]] | None = None
        self._return_verification: Callable[[], Awaitable[bool]] | None = None

    async def enter(self, claim: Task, nondispatch_result: Any = None) -> Task:
        """Own one start publication until its exact acknowledgement is known.

        Cancellation stops observation, not the store call. If entry fails, the
        caller must not dispatch; the existing retry handle joins that same call
        and reconciles proven nondispatch instead of issuing another start.
        """
        if not self._store.supports_task_group_quiescence:
            assert claim.worker_id is not None and claim.lease_expires_at is not None
            return await self._store.mark_claimed_task_execution_started(
                claim.id, claim.worker_id, claim.lease_expires_at
            )
        if self._entry_claim is not None or self._authority is not None:
            raise RuntimeError("Execution entry is already owned.")
        claim = copy_task(claim)
        self._entry_claim = claim
        self.result = nondispatch_result

        async def mark() -> Task:
            assert claim.worker_id is not None and claim.lease_expires_at is not None
            result = await self._store.mark_claimed_task_execution_started(
                claim.id, claim.worker_id, claim.lease_expires_at
            )
            return self._require_entry(result, readback=False)

        self._entry_operation = asyncio.create_task(
            capture_task_store_operation(
                mark,
                operation_name="Task group execution entry",
                redactor=self._redactor,
                mutation_store=self._store,
                mutation_method_name="mark_claimed_task_execution_started",
            ),
            name="cayu-group-execution-entry",
        )
        try:
            outcome = await asyncio.shield(self._entry_operation)
            if outcome.failure is not None:
                raise_task_store_operation_failure(outcome.failure)
            assert outcome.result is not None
        except BaseException:
            self._pending = True
            raise
        self._entry_operation = None
        self._entry_claim = None
        return outcome.result

    def _require_entry(self, value: object, *, readback: bool) -> Task:
        claim = self._entry_claim
        assert claim is not None

        def validate() -> Task:
            if type(value) is not Task:
                raise TaskGroupConflict("Execution entry lost its exact task claim.")
            recorded = copy_task(value)
            ignored: dict[str, Any] = {
                "started_at": claim.started_at,
                "updated_at": claim.updated_at,
            }
            if readback:
                # Election may request cancellation, but cannot replace the
                # exact claim or attach an invocation on this worker's behalf.
                ignored.update(
                    status_reason=claim.status_reason,
                    status_payload=claim.status_payload,
                    error=claim.error,
                )
            if (
                (readback and claim.started_at is not None)
                or (not readback and recorded.started_at is None)
                or recorded.model_copy(update=ignored) != claim
            ):
                raise TaskGroupConflict("Execution entry changed its exact task claim.")
            return recorded

        validated = capture_sensitive_result_validation(
            validate, operation_name="Task group execution entry authority", redactor=self._redactor
        )
        if validated.failure is not None:
            raise_task_store_operation_failure(validated.failure)
        assert validated.result is not None
        return validated.result

    async def _acknowledge(self) -> None:
        if self._return_verification is not None:
            verified = await self._return_verification()
            if verified is False:
                # The callback established no group obligation. Ordinary
                # invocation recovery remains responsible for its release.
                self._return_verification = None
                return
            if verified is not True:
                raise TaskGroupConflict("Exact execution release has not been established.")
            self._return_verification = None
        if self._entry_claim is not None:
            assert self._entry_operation is not None
            outcome = await asyncio.shield(self._entry_operation)
            if outcome.failure is None:
                recorded = self._require_entry(outcome.result, readback=False)
            else:
                recorded = self._require_entry(
                    await self._store.load_task(self._entry_claim.id), readback=True
                )
            # Keep the reconciled marker across acknowledgement loss. A later
            # retry need not derive authority from a terminal task's cleared lease.
            self._authority = recorded
            self._entry_claim = None
            self._entry_operation = None
        assert self._authority is not None
        if self._authority.started_at is not None:
            await self._store._settle_task_group_execution(copy_task(self._authority))

    def observe(self, authority: Task, result: Any = None) -> None:
        if self._authority is not None:
            raise RuntimeError("Execution settlement was already observed.")
        self._authority = copy_task(authority)
        self.result = result
        self._pending = self._store.supports_task_group_quiescence

    async def verify_returned_execution(
        self, authority: Task, verify: Callable[[], Awaitable[bool]], result: Any = None
    ) -> None:
        """Retain exact verification after owner return, before awaiting readback.

        Retry joins an in-flight read or repeats a failed read before attempting
        settlement. True proves non-admission or exact released ownership; False
        proves there is no group obligation requiring acknowledgement. Pending
        release, unknown or unavailable evidence must raise, never return False.
        """
        if not self._store.supports_task_group_quiescence:
            await verify()
            return
        self.observe(authority, result)
        self._return_verification = verify
        await self.attempt()

    async def attempt(self) -> None:
        """Bound observation, not the write; retain it across timeout/cancel.

        Concurrent retry callers never queue unboundedly on a slow observer.
        They receive the same pending owner and can join its call on retry.
        """
        if self._lock.locked():
            return
        async with self._lock:
            if not self._pending:
                return
            assert self._authority is not None or self._entry_claim is not None
            for _attempt in range(3):
                if self._operation is None:
                    self._operation = asyncio.create_task(
                        capture_task_store_operation(
                            self._acknowledge,
                            operation_name="Task group execution settlement",
                            redactor=self._redactor,
                            mutation_store=self._store,
                            mutation_method_name="_settle_task_group_execution",
                        ),
                        name="cayu-group-execution-settlement",
                    )
                # asyncio.wait never forwards caller cancellation or timeout to
                # its children. The handle, not an orphaned callback, owns the
                # still-running operation and its eventual detached result.
                completed, _ = await asyncio.wait(
                    (self._operation,), timeout=_OBSERVATION_TIMEOUT_SECONDS
                )
                if not completed:
                    return
                outcome = self._operation.result()
                self._operation = None
                self._failure = outcome.failure
                if outcome.failure is None:
                    self._pending = False
                    return
                if not isinstance(outcome.failure, Exception):
                    raise_task_store_operation_failure(outcome.failure)

    async def retry(self) -> Any:
        """Retry only exact acknowledgement, returning the retained outcome."""
        primary: BaseException | None = None
        try:
            await self.attempt()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self.finish(primary)
        return self.result

    def finish(self, primary: BaseException | None = None) -> None:
        """Transfer unresolved ownership without replacing a primary signal."""
        if not self._pending:
            return
        existing = _failure_chain(primary)
        if any(
            isinstance(error, TaskExecutionSettlementPending) and error.settlement is self
            for error in existing
        ):
            return
        pending = TaskExecutionSettlementPending(self)
        if self._failure is not None and not any(
            error is retained for error in _failure_chain(self._failure) for retained in existing
        ):
            set_exception_cause(pending, self._failure)
        if primary is None:
            raise pending from self._failure
        cause = exception_cause(primary)
        set_exception_cause(
            primary,
            pending
            if cause is None
            else BaseExceptionGroup("Execution and settlement failures", [cause, pending]),
        )


def _failure_chain(error: BaseException | None) -> list[BaseException]:
    """Inspect identity only, never infer current cancellation from history."""
    observed: dict[int, BaseException] = {}
    pending = [] if error is None else [error]
    while pending:
        for candidate in iter_exception_tree(pending.pop()):
            if id(candidate) in observed:
                continue
            observed[id(candidate)] = candidate
            cause = exception_cause(candidate)
            if cause is not None:
                pending.append(cause)
    return list(observed.values())


def has_task_execution_settlement_pending(error: BaseException) -> bool:
    """An outer loop must not discard a retry owner carried by another failure."""
    return any(isinstance(item, TaskExecutionSettlementPending) for item in _failure_chain(error))
