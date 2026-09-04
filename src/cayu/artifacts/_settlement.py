"""Owned task settlement for built-in artifact writes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextvars import copy_context
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from threading import RLock
from typing import Any, cast
from uuid import uuid4

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_suppresses_context,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.artifacts.base import ArtifactMetadata, ArtifactStoreUnavailableError
from cayu.artifacts.settlement import (
    ArtifactWriteSettlementEvidence,
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementObservation,
    ArtifactWriteSettlementPhase,
    ArtifactWriteSettlementStatus,
    _attach_artifact_write_settlement,
    _current_artifact_write_observer,
    _log_late_artifact_write_settlement,
    _store_identity_sha256,
)

_DEFAULT_SETTLEMENT_TIMEOUT_SECONDS = 5.0
_DEFAULT_MAX_RETAINED_WRITES = 64


@dataclass(frozen=True)
class _ArtifactWriteMutationOutcome:
    status: ArtifactWriteSettlementStatus
    phase: ArtifactWriteSettlementPhase
    artifact: ArtifactMetadata | None = None
    error: BaseException | None = None
    cancellation_error: BaseException | None = None
    child_cancellation_error: asyncio.CancelledError | None = None
    failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...] = ()
    backend_locator: str | None = None
    backend_version: str | None = None

    def __post_init__(self) -> None:
        if self.status is ArtifactWriteSettlementStatus.COMMITTED:
            if type(self.artifact) is not ArtifactMetadata:
                raise TypeError("Committed artifact writes require exact ArtifactMetadata.")
        elif self.artifact is not None:
            raise ValueError("Only committed artifact writes may carry ArtifactMetadata.")
        if self.status is not ArtifactWriteSettlementStatus.COMMITTED and self.error is None:
            raise ValueError("Uncommitted artifact writes require an authoritative error.")
        if self.error is not None and not isinstance(self.error, BaseException):
            raise TypeError("Artifact write outcome errors must be BaseException instances.")
        if self.cancellation_error is not None and not isinstance(
            self.cancellation_error, BaseException
        ):
            raise TypeError("Artifact write cancellation evidence must be BaseException.")
        if self.child_cancellation_error is not None and not isinstance(
            self.child_cancellation_error, asyncio.CancelledError
        ):
            raise TypeError("Artifact write child cancellation must be CancelledError.")
        if len(set(self.failure_codes)) != len(self.failure_codes):
            raise ValueError("Artifact write outcome failure codes must be unique.")


class _ArtifactWritePhaseReporter:
    def __init__(self, *, operation_id: str, observer: Any) -> None:
        self._operation_id = operation_id
        self._observer = observer
        self._phase = ArtifactWriteSettlementPhase.PRE_DISPATCH
        self._child_cancellation: asyncio.CancelledError | None = None
        self._lock = RLock()

    @property
    def phase(self) -> ArtifactWriteSettlementPhase:
        with self._lock:
            return self._phase

    def set(self, phase: ArtifactWriteSettlementPhase) -> None:
        if not isinstance(phase, ArtifactWriteSettlementPhase):
            raise TypeError("Artifact write phase must be ArtifactWriteSettlementPhase.")
        with self._lock:
            self._phase = phase
        if self._observer is not None:
            self._observer._set_phase(self._operation_id, phase)

    @property
    def child_cancellation(self) -> asyncio.CancelledError | None:
        with self._lock:
            return self._child_cancellation

    def record_child_cancellation(self, cancellation: asyncio.CancelledError) -> None:
        if not isinstance(cancellation, asyncio.CancelledError):
            raise TypeError("Child cancellation must be a CancelledError.")
        with self._lock:
            if self._child_cancellation is None:
                self._child_cancellation = cancellation


async def _await_owned_sync_call(
    reporter: _ArtifactWritePhaseReporter,
    callback: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Keep one dispatched synchronous call owned through task cancellation."""

    if not callable(callback):
        raise TypeError("Owned synchronous artifact callback must be callable.")
    loop = asyncio.get_running_loop()
    context = copy_context()
    future = loop.run_in_executor(
        None,
        partial(context.run, callback, *args, **kwargs),
    )
    while True:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError as cancellation:
            current_task = asyncio.current_task()
            if current_task is None or current_task.cancelling() == 0:
                # The synchronous callable itself raised CancelledError (or its
                # executor future was cancelled before dispatch). Let the
                # adapter classify that terminal outcome normally.
                raise
            reporter.record_child_cancellation(cancellation)
            consume_pending_task_cancellation(cancellation)


class _ArtifactWriteRegistry:
    """Retain a bounded set of dispatched artifact-write tasks."""

    def __init__(self, *, max_operations: int = _DEFAULT_MAX_RETAINED_WRITES) -> None:
        if type(max_operations) is not int:
            raise TypeError("max_operations must be an integer.")
        if max_operations <= 0:
            raise ValueError("max_operations must be greater than zero.")
        self._max_operations = max_operations
        self._reservations = 0
        self._tasks: set[asyncio.Task[object]] = set()
        self._lock = RLock()

    def reserve(self) -> bool:
        with self._lock:
            if self._reservations + len(self._tasks) >= self._max_operations:
                return False
            self._reservations += 1
            return True

    def release_reservation(self) -> None:
        with self._lock:
            if self._reservations <= 0:
                raise RuntimeError("Artifact write registry has no reservation to release.")
            self._reservations -= 1

    def track(self, task: asyncio.Task[object]) -> None:
        with self._lock:
            if self._reservations <= 0:
                raise RuntimeError("Artifact write registry has no reservation to track.")
            self._reservations -= 1
            self._tasks.add(task)

    def release(self, task: asyncio.Task[object]) -> None:
        with self._lock:
            self._tasks.discard(task)

    def __len__(self) -> int:
        with self._lock:
            return self._reservations + len(self._tasks)


def _committed_artifact_write(
    artifact: ArtifactMetadata,
    *,
    error: BaseException | None = None,
    failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...] = (),
) -> _ArtifactWriteMutationOutcome:
    return _ArtifactWriteMutationOutcome(
        status=ArtifactWriteSettlementStatus.COMMITTED,
        phase=ArtifactWriteSettlementPhase.SETTLED,
        artifact=artifact,
        error=error,
        failure_codes=failure_codes,
    )


def _absent_artifact_write(
    error: BaseException,
    *,
    phase: ArtifactWriteSettlementPhase,
    failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...],
    cancellation_error: BaseException | None = None,
) -> _ArtifactWriteMutationOutcome:
    return _ArtifactWriteMutationOutcome(
        status=ArtifactWriteSettlementStatus.ABSENT,
        phase=phase,
        error=error,
        cancellation_error=cancellation_error,
        failure_codes=failure_codes,
    )


def _unsettled_artifact_write(
    error: BaseException,
    *,
    phase: ArtifactWriteSettlementPhase,
    failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...],
    cancellation_error: BaseException | None = None,
    backend_locator: str | None = None,
    backend_version: str | None = None,
) -> _ArtifactWriteMutationOutcome:
    return _ArtifactWriteMutationOutcome(
        status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
        phase=phase,
        error=error,
        cancellation_error=cancellation_error,
        failure_codes=failure_codes,
        backend_locator=backend_locator,
        backend_version=backend_version,
    )


async def _settle_artifact_write(
    *,
    registry: _ArtifactWriteRegistry,
    store_id: str,
    artifact_id: str,
    operation: Callable[[_ArtifactWritePhaseReporter], Awaitable[_ArtifactWriteMutationOutcome]],
    operation_name: str = "Artifact write mutation",
    settlement_timeout_s: float = _DEFAULT_SETTLEMENT_TIMEOUT_SECONDS,
) -> ArtifactMetadata:
    """Run one complete artifact mutation under bounded retained ownership."""

    if type(settlement_timeout_s) is not float or settlement_timeout_s < 0:
        raise ValueError("settlement_timeout_s must be a non-negative float.")
    operation_id = f"artifact_write_{uuid4().hex}"
    store_identity_sha256 = _store_identity_sha256(store_id)
    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    observer = _current_artifact_write_observer()

    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as cancellation:
        evidence = _settlement_evidence(
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
            status=ArtifactWriteSettlementStatus.ABSENT,
            phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
            observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
            failure_codes=(),
        )
        _attach_artifact_write_settlement(cancellation, evidence)
        if observer is not None:
            observer._record_external(evidence)
        raise

    if not registry.reserve():
        error = ArtifactStoreUnavailableError("Artifact write settlement capacity is exhausted.")
        evidence = _settlement_evidence(
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
            status=ArtifactWriteSettlementStatus.ABSENT,
            phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
            observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
            failure_codes=(ArtifactWriteSettlementFailureCode.CAPACITY_EXHAUSTED,),
        )
        _attach_artifact_write_settlement(error, evidence)
        if observer is not None:
            observer._record_external(evidence)
        raise error

    observer_reserved = False
    if observer is not None:
        observer_reserved = observer._reserve(
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        if not observer_reserved:
            registry.release_reservation()
            error = ArtifactStoreUnavailableError(
                "Artifact write settlement observer capacity is exhausted."
            )
            evidence = _settlement_evidence(
                operation_id=operation_id,
                artifact_id=artifact_id,
                store_identity_sha256=store_identity_sha256,
                started_at=started_at,
                started_monotonic=started_monotonic,
                status=ArtifactWriteSettlementStatus.ABSENT,
                phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
                failure_codes=(ArtifactWriteSettlementFailureCode.CAPACITY_EXHAUSTED,),
            )
            _attach_artifact_write_settlement(error, evidence)
            observer._record_external(evidence)
            raise error

    reporter = _ArtifactWritePhaseReporter(operation_id=operation_id, observer=observer)
    try:
        task = asyncio.create_task(capture_awaitable_outcome(lambda: operation(reporter)))
    except BaseException:
        registry.release_reservation()
        if observer_reserved and observer is not None:
            observer._release(operation_id)
        raise
    tracked_task = cast("asyncio.Task[object]", task)
    registry.track(tracked_task)

    try:
        waited = await await_shielded_task_outcome(
            task,
            timeout_after_cancellation_s=settlement_timeout_s,
        )
    except BaseException as abandonment:
        evidence = _settlement_evidence(
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
            status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
            phase=reporter.phase,
            observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
            failure_codes=(ArtifactWriteSettlementFailureCode.WAITER_ABANDONED,),
        )
        _attach_artifact_write_settlement(abandonment, evidence)
        if observer is not None:
            observer._record_owned(evidence, release=False)
        _retain_late_artifact_write(
            task=task,
            registry=registry,
            observer=observer,
            reporter=reporter,
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
            operation_name=operation_name,
        )
        raise

    cancellation = waited.cancellation
    if waited.timed_out:
        evidence = _settlement_evidence(
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
            status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
            phase=reporter.phase,
            observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
            failure_codes=(ArtifactWriteSettlementFailureCode.SETTLEMENT_DEADLINE_EXPIRED,),
        )
        if cancellation is None:  # pragma: no cover - only post-cancellation timeout is configured
            cancellation = asyncio.CancelledError()
        _attach_artifact_write_settlement(cancellation, evidence)
        if observer is not None:
            observer._record_owned(evidence, release=False)
        _retain_late_artifact_write(
            task=task,
            registry=registry,
            observer=observer,
            reporter=reporter,
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
            operation_name=operation_name,
        )
        restore_task_cancellation_requests(
            waited.cancellation_requests_consumed,
            cancellation=cancellation,
        )
        raise cancellation

    captured = waited.result
    if captured is None:  # pragma: no cover - capture helper always returns an outcome
        decision = _unexpected_outcome(
            RuntimeError("Artifact write task returned no captured outcome."),
            phase=reporter.phase,
            operation_name=operation_name,
        )
    elif captured.error is not None:
        decision = _unexpected_outcome(
            captured.error,
            phase=reporter.phase,
            operation_name=operation_name,
        )
    elif type(captured.result) is not _ArtifactWriteMutationOutcome:
        decision = _unexpected_outcome(
            TypeError("Artifact write adapter returned an invalid settlement outcome."),
            phase=reporter.phase,
            operation_name=operation_name,
        )
    else:
        decision = captured.result

    decision = _normalize_child_cancellation(
        decision,
        operation_name=operation_name,
        observed_cancellation=reporter.child_cancellation,
    )

    registry.release(tracked_task)
    observation = (
        ArtifactWriteSettlementObservation.LATE
        if observer is not None and observer._boundary_recorded(operation_id)
        else ArtifactWriteSettlementObservation.CALLER_BOUNDARY
    )
    if (
        cancellation is not None
        or decision.status is not ArtifactWriteSettlementStatus.COMMITTED
        or decision.error is not None
        or observation is ArtifactWriteSettlementObservation.LATE
    ):
        evidence = _evidence_from_decision(
            decision,
            operation_id=operation_id,
            artifact_id=artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
            observation=observation,
        )
        if cancellation is not None:
            _attach_artifact_write_settlement(cancellation, evidence)
        elif decision.error is not None:
            _attach_artifact_write_settlement(decision.error, evidence)
        if observer is not None:
            observer._record_owned(evidence, release=True)
        if observation is ArtifactWriteSettlementObservation.LATE:
            _log_late_artifact_write_settlement(evidence)
    elif observer is not None:
        observer._release(operation_id)

    cancellation_cause = None
    cancellation_errors = [
        error
        for error in (
            decision.cancellation_error or decision.error,
            decision.child_cancellation_error,
        )
        if error is not None
    ]
    if cancellation is not None and cancellation_errors:
        cancellation_cause = _ordered_exception_evidence(
            cancellation,
            cancellation_errors,
            message="Artifact write also failed while caller cancellation was pending.",
        )
        cancellation.add_note("Artifact write also failed while caller cancellation was pending.")
    if cancellation is not None:
        restore_task_cancellation_requests(
            waited.cancellation_requests_consumed,
            cancellation=cancellation,
        )
        if cancellation_cause is not None:
            raise cancellation from cancellation_cause
        raise cancellation
    if decision.error is not None:
        if decision.child_cancellation_error is not None:
            child_cancellation_cause = _ordered_exception_evidence(
                decision.error,
                [decision.child_cancellation_error],
                message="Artifact write failed after its owned child was cancelled.",
            )
            if child_cancellation_cause is not None:
                raise decision.error from child_cancellation_cause
        raise decision.error
    return cast("ArtifactMetadata", decision.artifact)


def _unexpected_outcome(
    error: BaseException,
    *,
    phase: ArtifactWriteSettlementPhase,
    operation_name: str = "Artifact write mutation",
) -> _ArtifactWriteMutationOutcome:
    if issubclass(type(error), asyncio.CancelledError):
        error = unexpected_child_cancellation_error(
            cast("asyncio.CancelledError", error),
            operation=operation_name,
        )
        failure_codes = (ArtifactWriteSettlementFailureCode.CHILD_CANCELLED,)
    else:
        failure_codes = (ArtifactWriteSettlementFailureCode.MUTATION_FAILED,)
    return _unsettled_artifact_write(error, phase=phase, failure_codes=failure_codes)


def _normalize_child_cancellation(
    decision: _ArtifactWriteMutationOutcome,
    *,
    operation_name: str,
    observed_cancellation: asyncio.CancelledError | None = None,
) -> _ArtifactWriteMutationOutcome:
    decision_cancellation = (
        cast("asyncio.CancelledError", decision.error)
        if decision.error is not None and issubclass(type(decision.error), asyncio.CancelledError)
        else None
    )
    if decision_cancellation is None and observed_cancellation is None:
        return decision
    failure_codes = tuple(
        dict.fromkeys((*decision.failure_codes, ArtifactWriteSettlementFailureCode.CHILD_CANCELLED))
    )
    error = decision.error
    cancellation_error = decision.cancellation_error
    child_cancellation_error = decision.child_cancellation_error
    if decision_cancellation is not None:
        error = unexpected_child_cancellation_error(
            decision_cancellation,
            operation=operation_name,
        )
        cancellation_error = None
        child_cancellation_error = (
            observed_cancellation
            if observed_cancellation is not None
            and observed_cancellation is not decision_cancellation
            else None
        )
    elif error is None:
        if observed_cancellation is None:  # pragma: no cover - excluded above
            raise AssertionError("Child cancellation normalization lost its signal.")
        error = unexpected_child_cancellation_error(
            observed_cancellation,
            operation=operation_name,
        )
        cancellation_error = None
        child_cancellation_error = None
    elif observed_cancellation is not None:
        child_cancellation_error = observed_cancellation
    return _ArtifactWriteMutationOutcome(
        status=decision.status,
        phase=decision.phase,
        artifact=decision.artifact,
        error=error,
        cancellation_error=cancellation_error,
        child_cancellation_error=child_cancellation_error,
        failure_codes=failure_codes,
        backend_locator=decision.backend_locator,
        backend_version=decision.backend_version,
    )


def _visible_exception_evidence(error: BaseException) -> BaseException | None:
    cause = exception_cause(error)
    if cause is not None:
        return cause
    context = exception_context(error)
    if context is not None and not exception_suppresses_context(error):
        return context
    return None


def _ordered_exception_evidence(
    authoritative: BaseException,
    additions: list[BaseException],
    *,
    message: str,
) -> BaseException | None:
    evidence: list[BaseException] = []
    for candidate in (_visible_exception_evidence(authoritative), *additions):
        if candidate is None or candidate is authoritative:
            continue
        if any(existing is candidate for existing in evidence):
            continue
        evidence.append(candidate)
    if not evidence:
        return None
    if len(evidence) == 1:
        return evidence[0]
    return BaseExceptionGroup(message, evidence)


def _retain_late_artifact_write(
    *,
    task: Any,
    registry: _ArtifactWriteRegistry,
    observer: Any,
    reporter: _ArtifactWritePhaseReporter,
    operation_id: str,
    artifact_id: str,
    store_identity_sha256: str,
    started_at: datetime,
    started_monotonic: float,
    operation_name: str,
) -> None:
    def observe_late(completed: Any) -> None:
        try:
            captured = completed.result()
            if captured.error is not None:
                decision = _unexpected_outcome(
                    captured.error,
                    phase=reporter.phase,
                    operation_name=operation_name,
                )
            elif type(captured.result) is _ArtifactWriteMutationOutcome:
                decision = captured.result
            else:
                decision = _unexpected_outcome(
                    TypeError("Artifact write adapter returned an invalid settlement outcome."),
                    phase=reporter.phase,
                    operation_name=operation_name,
                )
            decision = _normalize_child_cancellation(
                decision,
                operation_name=operation_name,
                observed_cancellation=reporter.child_cancellation,
            )
            evidence = _evidence_from_decision(
                decision,
                operation_id=operation_id,
                artifact_id=artifact_id,
                store_identity_sha256=store_identity_sha256,
                started_at=started_at,
                started_monotonic=started_monotonic,
                observation=ArtifactWriteSettlementObservation.LATE,
            )
            if observer is not None:
                observer._record_owned(evidence, release=True)
            _log_late_artifact_write_settlement(evidence)
        except BaseException:
            if observer is not None:
                observer._release(operation_id)
            # Do not inspect the reporting exception: extension-controlled
            # values must not escape through this final ownership callback.
        finally:
            registry.release(cast("asyncio.Task[object]", completed))

    task.add_done_callback(observe_late)


def _evidence_from_decision(
    decision: _ArtifactWriteMutationOutcome,
    *,
    operation_id: str,
    artifact_id: str,
    store_identity_sha256: str,
    started_at: datetime,
    started_monotonic: float,
    observation: ArtifactWriteSettlementObservation,
) -> ArtifactWriteSettlementEvidence:
    return _settlement_evidence(
        operation_id=operation_id,
        artifact_id=artifact_id,
        store_identity_sha256=store_identity_sha256,
        started_at=started_at,
        started_monotonic=started_monotonic,
        status=decision.status,
        phase=decision.phase,
        observation=observation,
        failure_codes=decision.failure_codes,
        backend_locator=decision.backend_locator,
        backend_version=decision.backend_version,
    )


def _settlement_evidence(
    *,
    operation_id: str,
    artifact_id: str,
    store_identity_sha256: str,
    started_at: datetime,
    started_monotonic: float,
    status: ArtifactWriteSettlementStatus,
    phase: ArtifactWriteSettlementPhase,
    observation: ArtifactWriteSettlementObservation,
    failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...],
    backend_locator: str | None = None,
    backend_version: str | None = None,
) -> ArtifactWriteSettlementEvidence:
    observed_at = datetime.now(UTC)
    if observed_at < started_at:
        observed_at = started_at
    unique_codes = tuple(dict.fromkeys(failure_codes))
    return ArtifactWriteSettlementEvidence(
        operation_id=operation_id,
        artifact_id=artifact_id,
        store_identity_sha256=store_identity_sha256,
        status=status,
        phase=phase,
        observation=observation,
        started_at=started_at,
        observed_at=observed_at,
        elapsed_ms=min(
            int(max(time.monotonic() - started_monotonic, 0.0) * 1000),
            MAX_DURABLE_JSON_INTEGER,
        ),
        backend_locator=backend_locator,
        backend_version=backend_version,
        failure_codes=unique_codes,
    )
