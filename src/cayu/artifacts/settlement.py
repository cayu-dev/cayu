"""Bounded diagnostic evidence for interrupted artifact writes."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from enum import StrEnum
from itertools import islice
from threading import RLock
from types import TracebackType
from typing import Any, Literal, Self, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._exception_groups import exception_cause, exception_context, exception_group_children
from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_clean_nonblank, require_durable_text

_SETTLEMENTS_ATTRIBUTE = "_cayu_artifact_write_settlements"
_MAX_IDENTITY_UTF8_BYTES = 256
_MAX_FAILURE_CODES = 8
_MAX_ATTACHED_SETTLEMENTS = 16
_MAX_COLLECTED_SETTLEMENTS = 64
_MAX_EXCEPTION_GRAPH_NODES = 256
_STORE_IDENTITY_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_CURRENT_ARTIFACT_WRITE_OBSERVER: ContextVar[ArtifactWriteSettlementObserver | None]
_LOGGER = logging.getLogger("cayu.artifacts.settlement")


class ArtifactWriteSettlementStatus(StrEnum):
    """Truthful terminal classifications for an interrupted artifact write."""

    COMMITTED = "committed"
    ABSENT = "absent"
    RECONCILIATION_REQUIRED = "reconciliation_required"


class ArtifactWriteSettlementPhase(StrEnum):
    """Last bounded phase observed for an artifact write."""

    PRE_DISPATCH = "pre_dispatch"
    CONTENT = "content"
    COMMIT = "commit"
    CLEANUP = "cleanup"
    RECONCILIATION = "reconciliation"
    SETTLED = "settled"


class ArtifactWriteSettlementObservation(StrEnum):
    """Boundary at which settlement evidence was observed."""

    CALLER_BOUNDARY = "caller_boundary"
    LATE = "late"


class ArtifactWriteSettlementFailureCode(StrEnum):
    """Content-free failure classifications retained by settlement evidence."""

    CAPACITY_EXHAUSTED = "capacity_exhausted"
    MUTATION_FAILED = "mutation_failed"
    COMMIT_FAILED = "commit_failed"
    CLEANUP_FAILED = "cleanup_failed"
    RECONCILIATION_FAILED = "reconciliation_failed"
    CHILD_CANCELLED = "child_cancelled"
    SETTLEMENT_DEADLINE_EXPIRED = "settlement_deadline_expired"
    WAITER_ABANDONED = "waiter_abandoned"


class ArtifactWriteSettlementEvidence(BaseModel):
    """Bounded, non-authoritative evidence for one artifact-write invocation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[1] = 1
    operation_id: str
    artifact_id: str
    store_identity_sha256: str
    status: ArtifactWriteSettlementStatus
    phase: ArtifactWriteSettlementPhase
    observation: ArtifactWriteSettlementObservation
    started_at: datetime
    observed_at: datetime
    elapsed_ms: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    backend_locator: str | None = None
    backend_version: str | None = None
    failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...] = ()

    @field_validator("operation_id", "artifact_id", "backend_locator", "backend_version")
    @classmethod
    def validate_bounded_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_identity(value, info.field_name)

    @field_validator("store_identity_sha256")
    @classmethod
    def validate_store_identity_sha256(cls, value: str) -> str:
        if type(value) is not str or _STORE_IDENTITY_PATTERN.fullmatch(value) is None:
            raise ValueError("`store_identity_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("started_at", "observed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("failure_codes", mode="before")
    @classmethod
    def copy_failure_codes(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, str | bytes | Mapping) or not isinstance(value, Iterable):
            raise TypeError("failure_codes must be a sequence of settlement failure codes.")
        copied: list[object] = []
        for item in value:
            if len(copied) >= _MAX_FAILURE_CODES:
                raise ValueError(f"failure_codes may contain at most {_MAX_FAILURE_CODES} entries.")
            copied.append(item)
        return tuple(copied)

    @model_validator(mode="after")
    def validate_shape(self) -> ArtifactWriteSettlementEvidence:
        if self.observed_at < self.started_at:
            raise ValueError("observed_at cannot precede started_at.")
        if len(self.failure_codes) > _MAX_FAILURE_CODES:
            raise ValueError(f"failure_codes may contain at most {_MAX_FAILURE_CODES} entries.")
        if len(set(self.failure_codes)) != len(self.failure_codes):
            raise ValueError("failure_codes must not contain duplicates.")
        if (
            self.status is ArtifactWriteSettlementStatus.COMMITTED
            and self.phase is not ArtifactWriteSettlementPhase.SETTLED
        ):
            raise ValueError("Committed settlement evidence must be settled.")
        if (
            self.status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
            and self.phase is ArtifactWriteSettlementPhase.SETTLED
        ):
            raise ValueError("Reconciliation-required evidence cannot be settled.")
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return an independently validated immutable evidence record."""

        payload = _artifact_write_settlement_payload(self)
        if update is not None:
            if not isinstance(update, Mapping):
                raise TypeError("update must be a mapping or None.")
            keys = tuple(islice(iter(update), len(_SETTLEMENT_FIELD_NAMES) + 1))
            if len(keys) > len(_SETTLEMENT_FIELD_NAMES):
                raise ValueError("Artifact write settlement updates contain too many fields.")
            for key in keys:
                payload[key] = update[key]
        copied = type(self).model_validate(payload)
        raw_fields_set = object.__getattribute__(self, "__pydantic_fields_set__")
        if type(raw_fields_set) is not set:
            raise TypeError("Artifact write settlement fields-set state is invalid.")
        fields_set = {name for name in _SETTLEMENT_FIELD_NAMES if name in raw_fields_set}
        if update is not None:
            fields_set.update(keys)
        object.__setattr__(copied, "__pydantic_fields_set__", fields_set)
        return copied


_SETTLEMENT_FIELD_NAMES = tuple(ArtifactWriteSettlementEvidence.model_fields)


def _artifact_write_settlement_payload(
    value: ArtifactWriteSettlementEvidence,
) -> dict[str, object]:
    """Read only the fixed evidence schema before bounded field validation."""

    return {name: object.__getattribute__(value, name) for name in _SETTLEMENT_FIELD_NAMES}


def copy_artifact_write_settlement(
    value: ArtifactWriteSettlementEvidence,
) -> ArtifactWriteSettlementEvidence:
    """Defensively reconstruct one exact settlement evidence model."""

    if type(value) is not ArtifactWriteSettlementEvidence:
        raise TypeError("Artifact write settlement evidence has an invalid type.")
    payload = _artifact_write_settlement_payload(value)
    return ArtifactWriteSettlementEvidence.model_validate(payload)


def artifact_write_settlements(error: BaseException) -> tuple[ArtifactWriteSettlementEvidence, ...]:
    """Read bounded settlement evidence without invoking extension exception accessors."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be a BaseException.")
    records: list[ArtifactWriteSettlementEvidence] = []
    seen_records: set[tuple[str, ArtifactWriteSettlementObservation]] = set()
    pending = [error]
    seen_errors: set[int] = set()
    while pending:
        if len(seen_errors) >= _MAX_EXCEPTION_GRAPH_NODES:
            break
        candidate = pending.pop()
        if id(candidate) in seen_errors:
            continue
        seen_errors.add(id(candidate))
        for record in _exception_settlements(candidate):
            key = (record.operation_id, record.observation)
            if key not in seen_records:
                records.append(record)
                seen_records.add(key)
                if len(records) >= _MAX_COLLECTED_SETTLEMENTS:
                    return tuple(records)
        remaining = _MAX_EXCEPTION_GRAPH_NODES - len(seen_errors) - len(pending)
        if remaining <= 0:
            continue
        successors: list[BaseException] = []
        context = exception_context(candidate)
        if context is not None:
            successors.append(context)
        cause = exception_cause(candidate)
        if cause is not None and len(successors) < remaining:
            successors.append(cause)
        if issubclass(type(candidate), BaseExceptionGroup) and len(successors) < remaining:
            children = exception_group_children(
                cast("BaseExceptionGroup", candidate),
                maximum=remaining - len(successors),
            )
            if children is not None:
                successors.extend(children)
        pending.extend(reversed(successors))
    return tuple(records)


def record_artifact_write_settlement(
    evidence: ArtifactWriteSettlementEvidence,
    *,
    error: BaseException | None = None,
) -> None:
    """Publish one third-party-store settlement record to diagnostics."""

    if error is not None and not isinstance(error, BaseException):
        raise TypeError("error must be a BaseException or None.")
    copied = copy_artifact_write_settlement(evidence)
    observer = _CURRENT_ARTIFACT_WRITE_OBSERVER.get()
    if observer is not None:
        copied = observer._record_external(copied)
    if error is not None:
        _attach_artifact_write_settlement(error, copied)
    if copied.observation is ArtifactWriteSettlementObservation.LATE:
        _log_late_artifact_write_settlement(copied)


def artifact_store_identity_sha256(store_id: str) -> str:
    """Return the bounded diagnostic identity used by settlement evidence."""

    return _store_identity_sha256(store_id)


class _ObservedArtifactWrite:
    __slots__ = (
        "artifact_id",
        "boundary_recorded",
        "operation_id",
        "phase",
        "started_at",
        "started_monotonic",
        "store_identity_sha256",
    )

    def __init__(
        self,
        *,
        operation_id: str,
        artifact_id: str,
        store_identity_sha256: str,
        started_at: datetime,
        started_monotonic: float,
    ) -> None:
        self.operation_id = operation_id
        self.artifact_id = artifact_id
        self.store_identity_sha256 = store_identity_sha256
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.phase = ArtifactWriteSettlementPhase.PRE_DISPATCH
        self.boundary_recorded = False


class ArtifactWriteSettlementObserver:
    """Bounded non-recursive collector for interrupted artifact-write evidence."""

    def __init__(self, *, max_operations: int = 64) -> None:
        if type(max_operations) is not int:
            raise TypeError("max_operations must be an integer.")
        if max_operations <= 0:
            raise ValueError("max_operations must be greater than zero.")
        self._max_operations = max_operations
        self._active: dict[str, _ObservedArtifactWrite] = {}
        self._records: deque[ArtifactWriteSettlementEvidence] = deque(maxlen=max_operations * 2)
        self._record_keys: set[tuple[str, ArtifactWriteSettlementObservation]] = set()
        self._lock = RLock()
        self._token: Token[ArtifactWriteSettlementObserver | None] | None = None

    def __enter__(self) -> ArtifactWriteSettlementObserver:
        with self._lock:
            if self._token is not None:
                raise RuntimeError("Artifact write settlement observer is already installed.")
            self._token = _CURRENT_ARTIFACT_WRITE_OBSERVER.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with self._lock:
            token = self._token
            if token is None:
                raise RuntimeError("Artifact write settlement observer is not installed.")
            self._token = None
        _CURRENT_ARTIFACT_WRITE_OBSERVER.reset(token)

    def snapshot(self) -> tuple[ArtifactWriteSettlementEvidence, ...]:
        """Return defensive copies of every retained record."""

        with self._lock:
            return tuple(copy_artifact_write_settlement(record) for record in self._records)

    def drain(self) -> tuple[ArtifactWriteSettlementEvidence, ...]:
        """Return and clear retained records without abandoning active operations."""

        with self._lock:
            records = tuple(copy_artifact_write_settlement(record) for record in self._records)
            self._records.clear()
            self._record_keys.clear()
            return records

    def record_active_candidates(self) -> tuple[ArtifactWriteSettlementEvidence, ...]:
        """Record bounded candidates when this observer's caller stops waiting."""

        with self._lock:
            records: list[ArtifactWriteSettlementEvidence] = []
            for active in self._active.values():
                if active.boundary_recorded:
                    existing = self._find_record(
                        active.operation_id,
                        ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
                    )
                    if (
                        existing is not None
                        and existing.status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
                    ):
                        records.append(copy_artifact_write_settlement(existing))
                        continue
                evidence = _active_candidate(active)
                active.boundary_recorded = True
                self._append_record(evidence)
                records.append(copy_artifact_write_settlement(evidence))
            return tuple(records)

    def _reserve(
        self,
        *,
        operation_id: str,
        artifact_id: str,
        store_identity_sha256: str,
        started_at: datetime,
        started_monotonic: float,
    ) -> bool:
        with self._lock:
            if operation_id in self._active or len(self._active) >= self._max_operations:
                return False
            self._active[operation_id] = _ObservedArtifactWrite(
                operation_id=operation_id,
                artifact_id=artifact_id,
                store_identity_sha256=store_identity_sha256,
                started_at=started_at,
                started_monotonic=started_monotonic,
            )
            return True

    def _reserve_registration(
        self,
        *,
        operation_id: str,
        artifact_id: str,
        store_identity_sha256: str,
        started_at: datetime,
        started_monotonic: float,
    ) -> None:
        with self._lock:
            if operation_id in self._active or any(
                recorded_operation_id == operation_id
                for recorded_operation_id, _observation in self._record_keys
            ):
                raise ValueError(
                    "Artifact write settlement operation_id is already active or retained."
                )
            if len(self._active) >= self._max_operations:
                raise RuntimeError("Artifact write settlement observer capacity is exhausted.")
            self._active[operation_id] = _ObservedArtifactWrite(
                operation_id=operation_id,
                artifact_id=artifact_id,
                store_identity_sha256=store_identity_sha256,
                started_at=started_at,
                started_monotonic=started_monotonic,
            )

    def _set_phase(
        self,
        operation_id: str,
        phase: ArtifactWriteSettlementPhase,
    ) -> None:
        with self._lock:
            active = self._active.get(operation_id)
            if active is not None:
                active.phase = phase

    def _boundary_recorded(self, operation_id: str) -> bool:
        with self._lock:
            active = self._active.get(operation_id)
            return active is not None and active.boundary_recorded

    def _record_owned(
        self,
        evidence: ArtifactWriteSettlementEvidence,
        *,
        release: bool,
    ) -> None:
        copied = copy_artifact_write_settlement(evidence)
        with self._lock:
            active = self._active.get(copied.operation_id)
            if (
                active is not None
                and copied.observation is ArtifactWriteSettlementObservation.CALLER_BOUNDARY
            ):
                active.boundary_recorded = True
            self._append_record(copied)
            if release:
                self._active.pop(copied.operation_id, None)

    def _release(self, operation_id: str) -> None:
        with self._lock:
            self._active.pop(operation_id, None)

    def _record_external(
        self,
        evidence: ArtifactWriteSettlementEvidence,
    ) -> ArtifactWriteSettlementEvidence:
        copied = copy_artifact_write_settlement(evidence)
        with self._lock:
            if copied.operation_id in self._active:
                raise RuntimeError(
                    "Active registered artifact writes must be finalized through "
                    "registration.record()."
                )
            self._append_record(copied)
        return copy_artifact_write_settlement(copied)

    def _record_registration(
        self,
        evidence: ArtifactWriteSettlementEvidence,
        *,
        final: bool,
    ) -> ArtifactWriteSettlementEvidence:
        copied = copy_artifact_write_settlement(evidence)
        with self._lock:
            active = self._active.get(copied.operation_id)
            if active is None:
                raise RuntimeError("Artifact write settlement registration is not active.")
            if (
                copied.artifact_id != active.artifact_id
                or copied.store_identity_sha256 != active.store_identity_sha256
                or copied.started_at != active.started_at
            ):
                raise ValueError(
                    "Artifact write settlement evidence does not match its registration."
                )
            active.phase = copied.phase
            if active.boundary_recorded and not final:
                existing = self._find_record(
                    active.operation_id,
                    ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
                )
                if existing is None:
                    existing = _active_candidate(active)
                    self._append_record(existing)
                return copy_artifact_write_settlement(existing)
            if final and active.boundary_recorded:
                observed_at = datetime.now(UTC)
                if observed_at < active.started_at:
                    observed_at = active.started_at
                copied = copied.model_copy(
                    update={
                        "observation": ArtifactWriteSettlementObservation.LATE,
                        "observed_at": observed_at,
                        "elapsed_ms": min(
                            int(max(time.monotonic() - active.started_monotonic, 0.0) * 1000),
                            MAX_DURABLE_JSON_INTEGER,
                        ),
                    }
                )
            elif copied.observation is ArtifactWriteSettlementObservation.CALLER_BOUNDARY:
                active.boundary_recorded = True
            self._append_record(copied)
            if final:
                self._active.pop(copied.operation_id, None)
        return copy_artifact_write_settlement(copied)

    def _append_record(self, evidence: ArtifactWriteSettlementEvidence) -> None:
        key = (evidence.operation_id, evidence.observation)
        if key in self._record_keys:
            existing = self._find_record(evidence.operation_id, evidence.observation)
            if existing != evidence:
                raise ValueError(
                    "Artifact write settlement evidence conflicts with a retained operation."
                )
            return
        if len(self._records) == self._records.maxlen and self._records:
            removed = self._records[0]
            self._record_keys.discard((removed.operation_id, removed.observation))
        self._records.append(evidence)
        self._record_keys.add(key)

    def _find_record(
        self,
        operation_id: str,
        observation: ArtifactWriteSettlementObservation,
    ) -> ArtifactWriteSettlementEvidence | None:
        for record in reversed(self._records):
            if record.operation_id == operation_id and record.observation is observation:
                return record
        return None


_CURRENT_ARTIFACT_WRITE_OBSERVER = ContextVar(
    "cayu_artifact_write_settlement_observer",
    default=None,
)


def _current_artifact_write_observer() -> ArtifactWriteSettlementObserver | None:
    return _CURRENT_ARTIFACT_WRITE_OBSERVER.get()


class ArtifactWriteSettlementRegistration:
    """Operation-scoped observer registration for third-party artifact stores."""

    __slots__ = (
        "_boundary_recorded",
        "_closed",
        "_lock",
        "_observer",
        "_phase",
        "_started_monotonic",
        "artifact_id",
        "operation_id",
        "started_at",
        "store_identity_sha256",
    )

    def __init__(
        self,
        *,
        observer: ArtifactWriteSettlementObserver | None,
        operation_id: str,
        artifact_id: str,
        store_identity_sha256: str,
        started_at: datetime,
        started_monotonic: float,
    ) -> None:
        self._observer = observer
        self._boundary_recorded = False
        self._closed = False
        self._lock = RLock()
        self._phase = ArtifactWriteSettlementPhase.PRE_DISPATCH
        self._started_monotonic = started_monotonic
        self.operation_id = operation_id
        self.artifact_id = artifact_id
        self.store_identity_sha256 = store_identity_sha256
        self.started_at = started_at

    @property
    def phase(self) -> ArtifactWriteSettlementPhase:
        with self._lock:
            return self._phase

    def set_phase(self, phase: ArtifactWriteSettlementPhase) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Artifact write settlement registration is closed.")
            if not isinstance(phase, ArtifactWriteSettlementPhase):
                raise TypeError("Artifact write phase must be ArtifactWriteSettlementPhase.")
            self._phase = phase
            if self._observer is not None:
                self._observer._set_phase(self.operation_id, phase)

    def record(
        self,
        *,
        status: ArtifactWriteSettlementStatus,
        phase: ArtifactWriteSettlementPhase | None = None,
        error: BaseException | None = None,
        failure_codes: Iterable[ArtifactWriteSettlementFailureCode] = (),
        backend_locator: str | None = None,
        backend_version: str | None = None,
        final: bool = True,
    ) -> ArtifactWriteSettlementEvidence:
        """Record bounded caller-boundary or late evidence for this operation."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Artifact write settlement registration is closed.")
            if type(final) is not bool:
                raise TypeError("final must be a bool.")
            if error is not None and not isinstance(error, BaseException):
                raise TypeError("error must be a BaseException or None.")
            resolved_phase = self._phase if phase is None else phase
            if not isinstance(resolved_phase, ArtifactWriteSettlementPhase):
                raise TypeError("Artifact write phase must be ArtifactWriteSettlementPhase.")
            observed_at = datetime.now(UTC)
            if observed_at < self.started_at:
                observed_at = self.started_at
            evidence = ArtifactWriteSettlementEvidence.model_validate(
                {
                    "operation_id": self.operation_id,
                    "artifact_id": self.artifact_id,
                    "store_identity_sha256": self.store_identity_sha256,
                    "status": status,
                    "phase": resolved_phase,
                    "observation": ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
                    "started_at": self.started_at,
                    "observed_at": observed_at,
                    "elapsed_ms": min(
                        int(max(time.monotonic() - self._started_monotonic, 0.0) * 1000),
                        MAX_DURABLE_JSON_INTEGER,
                    ),
                    "backend_locator": backend_locator,
                    "backend_version": backend_version,
                    "failure_codes": failure_codes,
                }
            )
            if (
                not final
                and evidence.status is not ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
            ):
                raise ValueError("A non-final settlement record must require reconciliation.")
            self._phase = resolved_phase
            copied = evidence
            if self._observer is not None:
                copied = self._observer._record_registration(copied, final=final)
            elif final and self._boundary_recorded:
                copied = copied.model_copy(
                    update={"observation": ArtifactWriteSettlementObservation.LATE}
                )
            if not final:
                self._boundary_recorded = True
            if final:
                self._closed = True
            if error is not None:
                _attach_artifact_write_settlement(error, copied)
            if copied.observation is ArtifactWriteSettlementObservation.LATE:
                _log_late_artifact_write_settlement(copied)
            return copy_artifact_write_settlement(copied)

    def close(self) -> None:
        """Release observer ownership only after the operation is terminal."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._observer is not None:
                self._observer._release(self.operation_id)


def register_artifact_write_operation(
    *,
    artifact_id: str,
    store_id: str,
    operation_id: str | None = None,
) -> ArtifactWriteSettlementRegistration:
    """Register a third-party write before dispatch so runtime timeouts can identify it."""

    resolved_operation_id = _bounded_identity(
        f"artifact_write_{uuid4().hex}" if operation_id is None else operation_id,
        "operation_id",
    )
    resolved_artifact_id = _bounded_identity(artifact_id, "artifact_id")
    store_identity_sha256 = _store_identity_sha256(store_id)
    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    observer = _current_artifact_write_observer()
    if observer is not None:
        observer._reserve_registration(
            operation_id=resolved_operation_id,
            artifact_id=resolved_artifact_id,
            store_identity_sha256=store_identity_sha256,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
    return ArtifactWriteSettlementRegistration(
        observer=observer,
        operation_id=resolved_operation_id,
        artifact_id=resolved_artifact_id,
        store_identity_sha256=store_identity_sha256,
        started_at=started_at,
        started_monotonic=started_monotonic,
    )


def _store_identity_sha256(store_id: str) -> str:
    value = require_durable_text(store_id, "store_id")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_identity(value: str, field_name: str) -> str:
    value = require_clean_nonblank(require_durable_text(value, field_name), field_name)
    if len(value.encode("utf-8")) > _MAX_IDENTITY_UTF8_BYTES:
        raise ValueError(f"`{field_name}` must be at most {_MAX_IDENTITY_UTF8_BYTES} UTF-8 bytes.")
    return value


def _log_late_artifact_write_settlement(
    evidence: ArtifactWriteSettlementEvidence,
) -> None:
    try:
        _LOGGER.info(
            "artifact write settled after caller stopped waiting: "
            "operation_id=%s artifact_id=%s store_identity_sha256=%s "
            "phase=%s elapsed_ms=%s status=%s",
            evidence.operation_id,
            evidence.artifact_id,
            evidence.store_identity_sha256,
            evidence.phase.value,
            evidence.elapsed_ms,
            evidence.status.value,
        )
    except Exception:
        # Diagnostic handlers cannot own or replace artifact settlement.
        return


def _active_candidate(active: _ObservedArtifactWrite) -> ArtifactWriteSettlementEvidence:
    observed_at = datetime.now(UTC)
    if observed_at < active.started_at:
        observed_at = active.started_at
    return ArtifactWriteSettlementEvidence(
        operation_id=active.operation_id,
        artifact_id=active.artifact_id,
        store_identity_sha256=active.store_identity_sha256,
        status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
        phase=active.phase,
        observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
        started_at=active.started_at,
        observed_at=observed_at,
        elapsed_ms=min(
            int(max(time.monotonic() - active.started_monotonic, 0.0) * 1000),
            MAX_DURABLE_JSON_INTEGER,
        ),
        failure_codes=(ArtifactWriteSettlementFailureCode.SETTLEMENT_DEADLINE_EXPIRED,),
    )


def _attach_artifact_write_settlement(
    error: BaseException,
    evidence: ArtifactWriteSettlementEvidence,
) -> None:
    copied = copy_artifact_write_settlement(evidence)
    try:
        namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    except BaseException:
        return
    if type(namespace) is not dict:
        return
    existing = dict.get(namespace, _SETTLEMENTS_ATTRIBUTE)
    records = list(existing[-_MAX_ATTACHED_SETTLEMENTS:]) if type(existing) is tuple else []
    key = (copied.operation_id, copied.observation)
    for record in records:
        if (
            type(record) is ArtifactWriteSettlementEvidence
            and (record.operation_id, record.observation) == key
        ):
            return
    records = [*records[-(_MAX_ATTACHED_SETTLEMENTS - 1) :], copied]
    dict.__setitem__(namespace, _SETTLEMENTS_ATTRIBUTE, tuple(records))


def _exception_settlements(
    error: BaseException,
) -> tuple[ArtifactWriteSettlementEvidence, ...]:
    try:
        namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    except BaseException:
        return ()
    if type(namespace) is not dict:
        return ()
    raw = dict.get(namespace, _SETTLEMENTS_ATTRIBUTE)
    if type(raw) is not tuple:
        return ()
    copied: list[ArtifactWriteSettlementEvidence] = []
    for value in raw[-_MAX_ATTACHED_SETTLEMENTS:]:
        try:
            copied.append(copy_artifact_write_settlement(value))
        except (TypeError, ValueError):
            continue
    return tuple(copied)
