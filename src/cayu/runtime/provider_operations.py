from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.core.events import Event, EventType
from cayu.providers import (
    ModelStreamEvent,
    ModelStreamEventType,
    ProviderOperationRecoveryMetadata,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_model_stream_event,
)
from cayu.providers.operations import (
    PROVIDER_OPERATION_ID_MAX_CHARS,
    PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS,
)
from cayu.runtime.budgets import budget_settlement_event_id, budget_settlement_id
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    ModelCompletionStage,
    SessionOperationPublication,
    SessionStore,
)
from cayu.runtime.usage import is_conversational_model_completion_payload

_INSPECTION_ATTEMPT_EVENT_TYPES = (
    EventType.PROVIDER_OPERATION_STARTING,
    EventType.PROVIDER_OPERATION_STARTED,
    EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
    EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_ERROR,
    EventType.MODEL_ATTEMPT_DISCARDED,
)
_INSPECTION_EVENT_LIMIT = 8
_RECOVERY_EVIDENCE_LIMIT = 2
_MODEL_IDENTITY_PAYLOAD_FIELDS = (
    "step",
    "attempt",
    "max_attempts",
    "model_step_id",
    "model_attempt_id",
)
_RECOVERY_EVENT_TYPES = frozenset(
    {
        EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
        EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
        EventType.PROVIDER_OPERATION_RECONCILED,
    }
)
_RECOVERY_OUTPUT_EVENT_TYPES = (
    EventType.MODEL_TEXT_DELTA,
    EventType.MODEL_THINKING_DELTA,
    EventType.MODEL_ERROR,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_ATTEMPT_DISCARDED,
    EventType.PROVIDER_OPERATION_PROGRESS,
)
_PROVIDER_OPERATION_PROGRESS_EVENT_TYPES = (
    EventType.MODEL_TEXT_DELTA,
    EventType.MODEL_THINKING_DELTA,
    EventType.MODEL_ERROR,
    EventType.MODEL_COMPLETED,
    EventType.PROVIDER_OPERATION_PROGRESS,
)
_PROVIDER_OPERATION_PROGRESS_RECORD_TYPE = "cayu.provider-operation-progress"
_PROVIDER_OPERATION_PROGRESS_SCHEMA_VERSION = 1
_PROVIDER_OPERATION_PROGRESS_KEY_PREFIX = "cayu.provider-operation-progress:v1:"
_PROVIDER_OPERATION_PROGRESS_PAGE_SIZE = 1000


class ProviderOperationInspectionStatus(StrEnum):
    SYNCHRONOUS = "synchronous"
    PROVIDER_OPERATION_IN_PROGRESS = "provider_operation_in_progress"
    RECONNECT_SCHEDULED = "reconnect_scheduled"
    RECONNECT_IN_PROGRESS = "reconnect_in_progress"
    PROVIDER_OPERATION_RECONCILED = "provider_operation_reconciled"


class ProviderOperationCancellationStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    UNSUPPORTED = "unsupported"
    PENDING = "pending"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class ProviderOperationAccountingStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    RESERVED = "reserved"
    SETTLED = "settled"


class ProviderOperationInspection(BaseModel):
    """Bounded public view of the latest model attempt's dispatch mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ProviderOperationInspectionStatus
    provider: str | None = Field(default=None, max_length=256)
    operation_id: str | None = Field(
        default=None,
        max_length=PROVIDER_OPERATION_ID_MAX_CHARS,
    )
    stream_protocol: str | None = Field(
        default=None,
        max_length=PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS,
    )
    cancellation_status: ProviderOperationCancellationStatus = (
        ProviderOperationCancellationStatus.NOT_REQUESTED
    )
    accounting_status: ProviderOperationAccountingStatus = (
        ProviderOperationAccountingStatus.NOT_APPLICABLE
    )
    reservation_count: int = Field(default=0, ge=0, le=32)


class ProviderOperationEvidenceError(RuntimeError):
    """Durable provider-operation evidence is malformed or contradictory."""


class ProviderOperationProgressEnvelope(BaseModel):
    """Runtime-private normalized provider event and its exact operation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    state_version: Literal[1] = 1
    operation_id: str = Field(max_length=PROVIDER_OPERATION_ID_MAX_CHARS)
    stream_protocol: str = Field(max_length=PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS)
    stream_event: ModelStreamEvent

    @model_validator(mode="after")
    def require_recovery_metadata(self) -> ProviderOperationProgressEnvelope:
        if self.stream_event.recovery_metadata is None:
            raise ValueError("Provider-operation progress requires recovery metadata.")
        if self.stream_event.recovery_metadata.cursor is None:
            raise ValueError("Provider-operation progress requires a monotonic cursor.")
        return self

    @property
    def recovery_metadata(self) -> ProviderOperationRecoveryMetadata:
        metadata = self.stream_event.recovery_metadata
        if metadata is None:  # pragma: no cover - model validator owns this invariant
            raise AssertionError("Validated provider progress lost recovery metadata.")
        return metadata


class _ProviderOperationProgressRecord(BaseModel):
    """Bounded latest accepted provider event for one active completion stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.provider-operation-progress"] = (
        _PROVIDER_OPERATION_PROGRESS_RECORD_TYPE
    )
    schema_version: Literal[1] = _PROVIDER_OPERATION_PROGRESS_SCHEMA_VERSION
    session_id: str
    stage_id: str
    model_step_id: str
    model_attempt_id: str
    state_version: Literal[1] = 1
    operation_id: str = Field(max_length=PROVIDER_OPERATION_ID_MAX_CHARS)
    stream_protocol: str = Field(max_length=PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS)
    recovery_metadata: ProviderOperationRecoveryMetadata
    event_id: str
    event_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ProviderOperationProgressCommit:
    """One newly committed provider boundary or an exact replay of it."""

    state: ProviderOperationState
    event: Event
    replayed: bool


class _ProviderOperationProgressReplay(RuntimeError):
    """Internal transaction-abort signal for an exact already-durable event."""


@dataclass(frozen=True, slots=True)
class RecoverableProviderOperation:
    """Exact durable operation identity bound to one active completion stage."""

    interaction_id: str
    provider: str
    model: str
    model_attempt_identity: ModelAttemptIdentity
    state: ProviderOperationState
    status: ProviderOperationStatus
    step: int
    attempt: int
    max_attempts: int
    source_run_epoch: int
    accepted_stream_events: tuple[ModelStreamEvent, ...] = ()


class ProviderOperationRecoveryStatus(StrEnum):
    """Outcome of one fenced provider-operation retrieval attempt."""

    PENDING = "pending"
    RECONCILED = "reconciled"


@dataclass(frozen=True, slots=True)
class ProviderOperationRecoveryResult:
    """Events and state produced by one exact-operation recovery attempt."""

    status: ProviderOperationRecoveryStatus
    events: tuple[Event, ...]
    completion_event: Event | None = None


def _model_identity(event: Event, *, label: str) -> tuple[object, ...]:
    values = tuple(event.payload.get(field) for field in _MODEL_IDENTITY_PAYLOAD_FIELDS)
    step, attempt, max_attempts, model_step_id, model_attempt_id = values
    if (
        event.interaction_id is None
        or type(step) is not int
        or step < 1
        or type(attempt) is not int
        or attempt < 1
        or type(max_attempts) is not int
        or max_attempts < attempt
        or type(model_step_id) is not str
        or not model_step_id.strip()
        or type(model_attempt_id) is not str
        or not model_attempt_id.strip()
    ):
        raise ProviderOperationEvidenceError(f"{label} identity is malformed.")
    return (event.interaction_id, *values)


def _provider_scope(event: Event, *, label: str) -> tuple[str, str]:
    provider = event.payload.get("provider")
    model = event.payload.get("model")
    if (
        type(provider) is not str
        or not provider.strip()
        or type(model) is not str
        or not model.strip()
    ):
        raise ProviderOperationEvidenceError(f"{label} provider scope is malformed.")
    return provider, model


def _provider_operation_epoch(event: Event, *, label: str) -> int:
    source_run_epoch = event.payload.get("source_run_epoch")
    if type(source_run_epoch) is not int or source_run_epoch < 1:
        raise ProviderOperationEvidenceError(f"{label} run-epoch evidence is malformed.")
    return source_run_epoch


def _completion_scope(event: Event, *, label: str) -> tuple[str, str]:
    provider = event.payload.get("provider_name")
    model = event.payload.get("requested_model")
    if (
        type(provider) is not str
        or not provider.strip()
        or type(model) is not str
        or not model.strip()
    ):
        raise ProviderOperationEvidenceError(f"{label} provider scope is malformed.")
    return provider, model


def provider_operation_progress_storage_key(stage_id: str) -> str:
    """Return the private latest-progress key for one immutable model stage."""

    stage_id = require_durable_clean_nonblank(stage_id, "stage_id")
    return _PROVIDER_OPERATION_PROGRESS_KEY_PREFIX + sha256(stage_id.encode()).hexdigest()


def provider_operation_progress_event_id(stage_id: str, cursor: int) -> str:
    """Return a stable event id so inclusive provider replay has one identity."""

    stage_id = require_durable_clean_nonblank(stage_id, "stage_id")
    if type(cursor) is not int or not 0 <= cursor <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Provider-operation progress cursor is invalid.")
    material = canonical_durable_json_bytes(
        {"schema_version": 1, "stage_id": stage_id, "cursor": cursor},
        "provider_operation_progress_identity",
    )
    return f"provider-progress:v1:{sha256(material).hexdigest()}"


def provider_operation_progress_envelope(
    state: ProviderOperationState,
    stream_event: ModelStreamEvent,
) -> ProviderOperationProgressEnvelope:
    """Copy one normalized reconnectable event into its private durable envelope."""

    if type(state) is not ProviderOperationState:
        raise TypeError("Provider-operation progress requires ProviderOperationState.")
    return ProviderOperationProgressEnvelope(
        state_version=state.version,
        operation_id=state.operation_id,
        stream_protocol=state.stream_protocol,
        stream_event=copy_model_stream_event(stream_event),
    )


def provider_operation_progress_payload(
    state: ProviderOperationState,
    stream_event: ModelStreamEvent,
) -> dict[str, Any]:
    """Return the exact internal payload attached to the corresponding runtime event."""

    return provider_operation_progress_envelope(state, stream_event).model_dump(mode="json")


def _provider_operation_progress_digest(
    envelope: ProviderOperationProgressEnvelope,
) -> str:
    return sha256(
        canonical_durable_json_bytes(
            envelope.model_dump(mode="json"),
            "provider_operation_progress",
        )
    ).hexdigest()


def _provider_operation_progress_record(
    *,
    stage: ModelCompletionStage,
    model_attempt_identity: ModelAttemptIdentity,
    envelope: ProviderOperationProgressEnvelope,
    event: Event,
) -> _ProviderOperationProgressRecord:
    return _ProviderOperationProgressRecord(
        session_id=stage.session_id,
        stage_id=stage.stage_id,
        model_step_id=model_attempt_identity.model_step_id,
        model_attempt_id=model_attempt_identity.model_attempt_id,
        state_version=envelope.state_version,
        operation_id=envelope.operation_id,
        stream_protocol=envelope.stream_protocol,
        recovery_metadata=envelope.recovery_metadata,
        event_id=event.id,
        event_digest=_provider_operation_progress_digest(envelope),
    )


def _parse_progress_envelope(event: Event) -> ProviderOperationProgressEnvelope:
    try:
        return ProviderOperationProgressEnvelope.model_validate(
            event.payload.get("provider_operation_progress")
        )
    except (TypeError, ValueError):
        raise ProviderOperationEvidenceError(
            "Provider-operation progress evidence is malformed."
        ) from None


async def commit_provider_operation_progress(
    session_store: SessionStore,
    *,
    stage: ModelCompletionStage,
    model_attempt_identity: ModelAttemptIdentity,
    current_state: ProviderOperationState,
    stream_event: ModelStreamEvent,
    event: Event,
    expected_run_epoch: int,
) -> ProviderOperationProgressCommit:
    """Atomically accept one normalized event and advance its reconnect state.

    The latest state record and corresponding runtime event share the store's
    fenced session-operation transaction. Exact inclusive replay aborts that
    transaction before append and returns the already-durable event instead.
    """

    if type(stage) is not ModelCompletionStage:
        raise TypeError("stage must be a ModelCompletionStage.")
    if type(model_attempt_identity) is not ModelAttemptIdentity:
        raise TypeError("model_attempt_identity must be a ModelAttemptIdentity.")
    if type(current_state) is not ProviderOperationState:
        raise TypeError("current_state must be a ProviderOperationState.")
    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    if event.session_id != stage.session_id:
        raise ValueError("Provider-operation progress event belongs to another session.")
    envelope = provider_operation_progress_envelope(current_state, stream_event)
    expected_payload = envelope.model_dump(mode="json")
    if event.payload.get("provider_operation_progress") != expected_payload:
        raise ValueError("Provider-operation progress event lost its exact private envelope.")
    cursor = envelope.recovery_metadata.cursor
    if cursor is None:  # pragma: no cover - envelope validation owns this invariant
        raise AssertionError("Validated provider-operation cursor disappeared.")
    expected_event_id = provider_operation_progress_event_id(stage.stage_id, cursor)
    if event.id != expected_event_id:
        raise ValueError("Provider-operation progress event id is not cursor-stable.")

    storage_key = provider_operation_progress_storage_key(stage.stage_id)
    requested = _provider_operation_progress_record(
        stage=stage,
        model_attempt_identity=model_attempt_identity,
        envelope=envelope,
        event=event,
    )
    outcome: dict[str, _ProviderOperationProgressRecord | bool] = {}

    def transform(_session, checkpoint, current_record):
        if checkpoint is None:
            raise ProviderOperationEvidenceError(
                "Provider-operation progress requires a durable session checkpoint."
            )
        if current_record is None:
            current_cursor = current_state.recovery_metadata.cursor
            current_cursor = -1 if current_cursor is None else current_cursor
        else:
            try:
                current = _ProviderOperationProgressRecord.model_validate(current_record)
            except (TypeError, ValueError):
                raise ProviderOperationEvidenceError(
                    "Provider-operation latest-progress evidence is malformed."
                ) from None
            if (
                current.session_id != stage.session_id
                or current.stage_id != stage.stage_id
                or current.model_step_id != model_attempt_identity.model_step_id
                or current.model_attempt_id != model_attempt_identity.model_attempt_id
                or current.state_version != current_state.version
                or current.operation_id != current_state.operation_id
                or current.stream_protocol != current_state.stream_protocol
            ):
                raise ProviderOperationEvidenceError(
                    "Provider-operation latest progress belongs to another operation."
                )
            current_cursor = current.recovery_metadata.cursor
            if current_cursor is None:
                raise ProviderOperationEvidenceError(
                    "Provider-operation latest progress has no monotonic cursor."
                )
            if cursor <= current_cursor:
                outcome["replayed"] = True
                outcome["record"] = current
                raise _ProviderOperationProgressReplay
            if current_state.recovery_metadata != current.recovery_metadata:
                raise ProviderOperationEvidenceError(
                    "Provider-operation progress advanced from stale continuation state."
                )
        if cursor != current_cursor + 1:
            raise ProviderOperationEvidenceError("Provider-operation cursor advanced with a gap.")
        outcome["replayed"] = False
        outcome["record"] = requested
        return SessionOperationPublication(
            checkpoint=checkpoint,
            operation_records={storage_key: requested.model_dump(mode="json")},
        )

    try:
        await session_store.publish_session_operation_guarded(
            stage.session_id,
            idempotency_key=storage_key,
            operation_transform=transform,
            commit_guard=lambda: None,
            events=[event],
            expected_run_epoch=expected_run_epoch,
        )
    except _ProviderOperationProgressReplay:
        record = outcome.get("record")
        if type(record) is not _ProviderOperationProgressRecord:
            raise AssertionError("Provider-operation replay lost its durable record.") from None
        records = await session_store.query_events(
            EventQuery(session_id=stage.session_id, event_id=event.id, limit=2)
        )
        if len(records) != 1:
            raise ProviderOperationEvidenceError(
                "Provider-operation replay has no unique durable event."
            ) from None
        historical_event = records[0].event
        historical_digest = _provider_operation_progress_digest(
            _parse_progress_envelope(historical_event)
        )
        if historical_digest != requested.event_digest:
            raise ProviderOperationEvidenceError(
                "Provider-operation cursor regressed or was reused for different output."
            ) from None
        record_cursor = record.recovery_metadata.cursor
        if record_cursor == cursor and (
            record.event_id != event.id or record.event_digest != requested.event_digest
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation latest progress conflicts with its durable event."
            ) from None
        return ProviderOperationProgressCommit(
            state=ProviderOperationState(
                version=record.state_version,
                operation_id=record.operation_id,
                stream_protocol=record.stream_protocol,
                recovery_metadata=record.recovery_metadata,
            ),
            event=historical_event,
            replayed=True,
        )

    return ProviderOperationProgressCommit(
        state=ProviderOperationState(
            version=requested.state_version,
            operation_id=requested.operation_id,
            stream_protocol=requested.stream_protocol,
            recovery_metadata=requested.recovery_metadata,
        ),
        event=event,
        replayed=False,
    )


def _parse_operation_event(event: Event) -> RecoverableProviderOperation:
    try:
        _model_identity(event, label="Provider-operation recovery evidence")
        provider, model = _provider_scope(
            event,
            label="Provider-operation recovery evidence",
        )
        source_run_epoch = _provider_operation_epoch(
            event,
            label="Provider-operation recovery evidence",
        )
        identity = ModelAttemptIdentity.model_validate(
            {
                "model_step_id": event.payload.get("model_step_id"),
                "model_attempt_id": event.payload.get("model_attempt_id"),
            }
        )
        state = ProviderOperationState.model_validate(
            {
                "version": event.payload.get("state_version"),
                "operation_id": event.payload.get("operation_id"),
                "stream_protocol": event.payload.get("stream_protocol"),
                "recovery_metadata": event.payload.get("recovery_metadata", {}),
            }
        )
        status = ProviderOperationStatus(event.payload.get("status"))
        step = event.payload.get("step")
        attempt = event.payload.get("attempt")
        max_attempts = event.payload.get("max_attempts")
        start_id = event.payload.get("start_id")
        if (
            event.interaction_id is None
            or type(step) is not int
            or type(attempt) is not int
            or type(max_attempts) is not int
            or type(start_id) is not str
        ):
            raise ValueError
        start_id = require_durable_clean_nonblank(start_id, "start_id")
        if len(start_id) > 1024:
            raise ValueError
    except (ProviderOperationEvidenceError, TypeError, ValueError):
        raise ProviderOperationEvidenceError(
            "Provider-operation recovery evidence is malformed."
        ) from None
    return RecoverableProviderOperation(
        interaction_id=event.interaction_id,
        provider=provider,
        model=model,
        model_attempt_identity=identity,
        state=state,
        status=status,
        step=step,
        attempt=attempt,
        max_attempts=max_attempts,
        source_run_epoch=source_run_epoch,
    )


def _progress_event_matches_stream_type(
    event_type: EventType | str,
    stream_type: ModelStreamEventType,
) -> bool:
    return (
        (
            event_type == EventType.MODEL_TEXT_DELTA
            and stream_type is ModelStreamEventType.TEXT_DELTA
        )
        or (
            event_type == EventType.MODEL_THINKING_DELTA
            and stream_type is ModelStreamEventType.THINKING
        )
        or (event_type == EventType.MODEL_ERROR and stream_type is ModelStreamEventType.ERROR)
        or (
            event_type == EventType.MODEL_COMPLETED
            and stream_type is ModelStreamEventType.COMPLETED
        )
        or (
            event_type == EventType.PROVIDER_OPERATION_PROGRESS
            and stream_type in {ModelStreamEventType.TOOL_CALL, ModelStreamEventType.THINKING}
        )
    )


async def _load_accepted_provider_operation_progress(
    session_store: SessionStore,
    *,
    stage: ModelCompletionStage,
    operation: RecoverableProviderOperation,
    after_sequence: int,
) -> tuple[ProviderOperationState, tuple[ModelStreamEvent, ...]]:
    accepted: list[ModelStreamEvent] = []
    latest_sequence = after_sequence
    initial_cursor = operation.state.recovery_metadata.cursor
    expected_cursor: int = -1 if initial_cursor is None else initial_cursor
    expected_model_identity = (
        operation.interaction_id,
        operation.step,
        operation.attempt,
        operation.max_attempts,
        operation.model_attempt_identity.model_step_id,
        operation.model_attempt_identity.model_attempt_id,
    )
    while True:
        page = await session_store.query_events(
            EventQuery(
                session_id=stage.session_id,
                event_types=_PROVIDER_OPERATION_PROGRESS_EVENT_TYPES,
                after_sequence=latest_sequence,
                order_by=EventOrder.SEQUENCE_ASC,
                limit=_PROVIDER_OPERATION_PROGRESS_PAGE_SIZE,
            )
        )
        if not page:
            break
        for record in page:
            latest_sequence = record.sequence
            event = record.event
            if (
                event.payload.get("model_step_id") != operation.model_attempt_identity.model_step_id
                or event.payload.get("model_attempt_id")
                != operation.model_attempt_identity.model_attempt_id
            ):
                continue
            if (
                _model_identity(event, label="Provider-operation progress evidence")
                != expected_model_identity
            ):
                raise ProviderOperationEvidenceError(
                    "Provider-operation progress changed its owning interaction or attempt."
                )
            envelope = _parse_progress_envelope(event)
            if (
                envelope.state_version != operation.state.version
                or envelope.operation_id != operation.state.operation_id
                or envelope.stream_protocol != operation.state.stream_protocol
            ):
                raise ProviderOperationEvidenceError(
                    "Provider-operation progress changed operation identity."
                )
            if not _progress_event_matches_stream_type(event.type, envelope.stream_event.type):
                raise ProviderOperationEvidenceError(
                    "Provider-operation progress event type conflicts with normalized output."
                )
            cursor = envelope.recovery_metadata.cursor
            if cursor is None:  # pragma: no cover - envelope validation owns this invariant
                raise AssertionError("Validated provider progress lost its cursor.")
            if cursor != expected_cursor + 1:
                raise ProviderOperationEvidenceError(
                    "Provider-operation durable progress is not monotonic and contiguous."
                )
            if event.id != provider_operation_progress_event_id(stage.stage_id, cursor):
                raise ProviderOperationEvidenceError(
                    "Provider-operation progress event identity conflicts with its cursor."
                )
            expected_cursor = cursor
            accepted.append(copy_model_stream_event(envelope.stream_event))
        if len(page) < _PROVIDER_OPERATION_PROGRESS_PAGE_SIZE:
            break

    storage_key = provider_operation_progress_storage_key(stage.stage_id)
    raw_latest = await session_store.load_session_operation(stage.session_id, storage_key)
    if not accepted:
        if raw_latest is not None:
            raise ProviderOperationEvidenceError(
                "Provider-operation latest progress has no corresponding durable event."
            )
        return operation.state, ()
    try:
        latest = _ProviderOperationProgressRecord.model_validate(raw_latest)
    except (TypeError, ValueError):
        raise ProviderOperationEvidenceError(
            "Provider-operation latest-progress evidence is malformed."
        ) from None
    final_event = accepted[-1]
    final_metadata = final_event.recovery_metadata
    if final_metadata is None:  # pragma: no cover - envelope validation owns this invariant
        raise AssertionError("Accepted provider progress lost recovery metadata.")
    final_cursor = final_metadata.cursor
    if final_cursor is None:  # pragma: no cover - envelope validation owns this invariant
        raise AssertionError("Accepted provider progress lost its cursor.")
    final_envelope = provider_operation_progress_envelope(operation.state, final_event)
    if (
        latest.session_id != stage.session_id
        or latest.stage_id != stage.stage_id
        or latest.model_step_id != operation.model_attempt_identity.model_step_id
        or latest.model_attempt_id != operation.model_attempt_identity.model_attempt_id
        or latest.state_version != operation.state.version
        or latest.operation_id != operation.state.operation_id
        or latest.stream_protocol != operation.state.stream_protocol
        or latest.recovery_metadata != final_metadata
        or latest.event_id != provider_operation_progress_event_id(stage.stage_id, final_cursor)
        or latest.event_digest != _provider_operation_progress_digest(final_envelope)
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation latest progress conflicts with durable event history."
        )
    if final_event.type in {ModelStreamEventType.ERROR, ModelStreamEventType.COMPLETED}:
        raise ProviderOperationEvidenceError(
            "Provider-operation terminal output already crossed Cayu's durable boundary."
        )
    return (
        ProviderOperationState(
            version=operation.state.version,
            operation_id=operation.state.operation_id,
            stream_protocol=operation.state.stream_protocol,
            recovery_metadata=final_metadata,
        ),
        tuple(accepted),
    )


async def load_recoverable_provider_operation(
    session_store: SessionStore,
    stage: ModelCompletionStage,
) -> RecoverableProviderOperation | None:
    """Load one bounded operation identity that exactly matches an active stage."""

    records = await session_store.query_events(
        EventQuery(
            session_id=stage.session_id,
            event_type=EventType.PROVIDER_OPERATION_STARTED,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=_RECOVERY_EVIDENCE_LIMIT,
        )
    )
    if not records:
        return None
    latest = _parse_operation_event(records[0].event)
    raw_attempt_id = stage.intent.get("model_attempt_id")
    raw_provider = stage.intent.get("provider_name")
    raw_model = stage.intent.get("requested_model")
    if (
        latest.model_attempt_identity.model_step_id != stage.logical_step_id
        or latest.model_attempt_identity.model_attempt_id != raw_attempt_id
        or latest.provider != raw_provider
        or latest.model != raw_model
        or latest.source_run_epoch != stage.source_run_epoch
    ):
        return None
    started_records = await session_store.query_events(
        EventQuery(
            session_id=stage.session_id,
            event_type=EventType.MODEL_STARTED,
            before_sequence=records[0].sequence,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not started_records:
        raise ProviderOperationEvidenceError(
            "Provider-operation recovery evidence has no authoritative model-started owner."
        )
    started_event = started_records[0].event
    started_identity = _model_identity(
        started_event,
        label="Authoritative model-started evidence",
    )
    started_scope = _provider_scope(
        started_event,
        label="Authoritative model-started evidence",
    )
    if (
        started_identity[-2] != stage.logical_step_id
        or started_identity[-1] != raw_attempt_id
        or started_scope != (raw_provider, raw_model)
    ):
        return None
    operation_identity = _model_identity(
        records[0].event,
        label="Provider-operation recovery evidence",
    )
    if operation_identity != started_identity or (latest.provider, latest.model) != started_scope:
        raise ProviderOperationEvidenceError(
            "Provider-operation recovery evidence does not match its authoritative "
            "model-started owner."
        )
    if len(records) > 1:
        prior = _parse_operation_event(records[1].event)
        if prior.model_attempt_identity == latest.model_attempt_identity:
            raise ProviderOperationEvidenceError(
                "Active model attempt has more than one durable provider-operation identity."
            )
    state, accepted_stream_events = await _load_accepted_provider_operation_progress(
        session_store,
        stage=stage,
        operation=latest,
        after_sequence=records[0].sequence,
    )
    expected_output_identity = (
        latest.interaction_id,
        latest.step,
        latest.attempt,
        latest.max_attempts,
        latest.model_attempt_identity.model_step_id,
        latest.model_attempt_identity.model_attempt_id,
    )
    output_sequence = records[0].sequence
    while True:
        output_page = await session_store.query_events(
            EventQuery(
                session_id=stage.session_id,
                event_types=_RECOVERY_OUTPUT_EVENT_TYPES,
                after_sequence=output_sequence,
                order_by=EventOrder.SEQUENCE_ASC,
                limit=_PROVIDER_OPERATION_PROGRESS_PAGE_SIZE,
            )
        )
        if not output_page:
            break
        for output_record in output_page:
            output_sequence = output_record.sequence
            output_event = output_record.event
            output_identity = _model_identity(
                output_event,
                label="Provider-operation output evidence",
            )
            if output_identity != expected_output_identity:
                continue
            if (
                output_event.type == EventType.MODEL_ATTEMPT_DISCARDED
                or output_event.payload.get("provider_operation_progress") is None
            ):
                raise ProviderOperationEvidenceError(
                    "Legacy provider-operation recovery is unsafe after provider output crossed "
                    "Cayu's durable event boundary without reconnect metadata."
                )
        if len(output_page) < _PROVIDER_OPERATION_PROGRESS_PAGE_SIZE:
            break
    return RecoverableProviderOperation(
        interaction_id=latest.interaction_id,
        provider=latest.provider,
        model=latest.model,
        model_attempt_identity=latest.model_attempt_identity,
        state=state,
        status=latest.status,
        step=latest.step,
        attempt=latest.attempt,
        max_attempts=latest.max_attempts,
        source_run_epoch=latest.source_run_epoch,
        accepted_stream_events=accepted_stream_events,
    )


async def inspect_provider_operation(
    session_store: SessionStore,
    session_id: str,
) -> ProviderOperationInspection:
    """Inspect at most one latest model attempt without hydrating stream deltas."""

    started_records = await session_store.query_events(
        EventQuery(
            session_id=session_id,
            event_type=EventType.MODEL_STARTED,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not started_records:
        return ProviderOperationInspection(status=ProviderOperationInspectionStatus.SYNCHRONOUS)
    started_record = started_records[0]
    current_identity = _model_identity(
        started_record.event,
        label="Latest model-attempt evidence",
    )
    current_provider_scope = _provider_scope(
        started_record.event,
        label="Latest model-attempt evidence",
    )
    attempt_records = await session_store.query_events(
        EventQuery(
            session_id=session_id,
            event_types=_INSPECTION_ATTEMPT_EVENT_TYPES,
            after_sequence=started_record.sequence,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=_INSPECTION_EVENT_LIMIT,
        )
    )
    if len(attempt_records) == _INSPECTION_EVENT_LIMIT:
        raise ProviderOperationEvidenceError(
            "Latest model-attempt evidence exceeds the bounded inspection window."
        )
    recovery_records = await session_store.query_events(
        EventQuery(
            session_id=session_id,
            event_types=tuple(_RECOVERY_EVENT_TYPES),
            after_sequence=started_record.sequence,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    later_events = [
        record.event
        for record in sorted(
            [*attempt_records, *recovery_records],
            key=lambda record: record.sequence,
            reverse=True,
        )
    ]

    starting_events = [
        event for event in later_events if event.type == EventType.PROVIDER_OPERATION_STARTING
    ]
    operation_events = [
        event for event in later_events if event.type == EventType.PROVIDER_OPERATION_STARTED
    ]
    cancellation_events = [
        event
        for event in later_events
        if event.type
        in {
            EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
            EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
        }
    ]
    recovery_events = [event for event in later_events if event.type in _RECOVERY_EVENT_TYPES]
    owning_events = [
        event
        for event in later_events
        if event.type
        in {
            EventType.PROVIDER_OPERATION_STARTING,
            EventType.PROVIDER_OPERATION_STARTED,
            EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
            EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
            *_RECOVERY_EVENT_TYPES,
            EventType.MODEL_ERROR,
            EventType.MODEL_ATTEMPT_DISCARDED,
        }
        or (
            event.type == EventType.MODEL_COMPLETED
            and is_conversational_model_completion_payload(event.payload)
        )
    ]
    for event in owning_events:
        if _model_identity(event, label="Latest model-attempt event") != current_identity:
            raise ProviderOperationEvidenceError(
                "Latest model-attempt history contains mismatched identity evidence."
            )
        if (
            event.type == EventType.MODEL_COMPLETED
            and _completion_scope(event, label="Model completion evidence")
            != current_provider_scope
        ):
            raise ProviderOperationEvidenceError(
                "Model completion evidence is bound to a different provider or model."
            )
    provider_evidence = [
        *starting_events,
        *operation_events,
        *cancellation_events,
        *recovery_events,
    ]
    for evidence in provider_evidence:
        if _provider_scope(evidence, label="Provider-operation evidence") != current_provider_scope:
            raise ProviderOperationEvidenceError(
                "Provider-operation evidence is bound to a different provider or model."
            )
    epochs = [
        _provider_operation_epoch(event, label="Provider-operation evidence")
        for event in provider_evidence
    ]
    if epochs and any(epoch != epochs[0] for epoch in epochs[1:]):
        raise ProviderOperationEvidenceError(
            "Provider-operation evidence has contradictory run epochs."
        )
    terminal_seen = any(
        event.type == EventType.MODEL_COMPLETED
        and is_conversational_model_completion_payload(event.payload)
        and _model_identity(event, label="Model completion evidence") == current_identity
        for event in owning_events
    )
    if recovery_events and not operation_events:
        raise ProviderOperationEvidenceError(
            "Provider-operation recovery evidence has no durable operation identity."
        )
    if cancellation_events and not operation_events:
        raise ProviderOperationEvidenceError(
            "Provider-operation cancellation evidence has no durable operation identity."
        )
    if not operation_events and not starting_events:
        return ProviderOperationInspection(status=ProviderOperationInspectionStatus.SYNCHRONOUS)
    parsed_starting: list[tuple[str, str]] = []
    for starting_event in starting_events:
        try:
            raw_provider = starting_event.payload.get("provider")
            raw_start_id = starting_event.payload.get("start_id")
            if type(raw_provider) is not str or type(raw_start_id) is not str:
                raise ValueError
            provider = require_durable_clean_nonblank(
                raw_provider,
                "provider",
            )
            start_id = require_durable_clean_nonblank(
                raw_start_id,
                "start_id",
            )
            if len(provider) > 256 or len(start_id) > 1024:
                raise ValueError
        except (TypeError, ValueError):
            raise ProviderOperationEvidenceError(
                "Provider-operation starting evidence is malformed."
            ) from None
        parsed_starting.append((provider, start_id))
    if parsed_starting and any(
        candidate != parsed_starting[0] for candidate in parsed_starting[1:]
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation starting evidence is contradictory for the latest model attempt."
        )
    if not operation_events:
        provider, _start_id = parsed_starting[0]
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS,
            provider=provider,
        )
    parsed_evidence: list[tuple[ProviderOperationState, ProviderOperationStatus, str, str]] = []
    for operation_event in operation_events:
        try:
            state = ProviderOperationState.model_validate(
                {
                    "version": operation_event.payload.get("state_version"),
                    "operation_id": operation_event.payload.get("operation_id"),
                    "stream_protocol": operation_event.payload.get("stream_protocol"),
                    "recovery_metadata": operation_event.payload.get("recovery_metadata", {}),
                }
            )
            status = ProviderOperationStatus(operation_event.payload.get("status"))
            provider = operation_event.payload.get("provider")
            start_id = operation_event.payload.get("start_id")
            if type(provider) is not str or type(start_id) is not str:
                raise ValueError
            provider = require_durable_clean_nonblank(provider, "provider")
            start_id = require_durable_clean_nonblank(start_id, "start_id")
            if len(provider) > 256 or len(start_id) > 1024:
                raise ValueError
        except (TypeError, ValueError):
            raise ProviderOperationEvidenceError(
                "Provider-operation evidence is malformed."
            ) from None
        parsed_evidence.append((state, status, provider, start_id))

    state, status, provider, start_id = parsed_evidence[0]
    if any(candidate != (state, status, provider, start_id) for candidate in parsed_evidence[1:]):
        raise ProviderOperationEvidenceError(
            "Provider-operation evidence is contradictory for the latest model attempt."
        )
    if parsed_starting and parsed_starting[0] != (provider, start_id):
        raise ProviderOperationEvidenceError(
            "Provider-operation starting and started evidence is contradictory for the latest "
            "model attempt."
        )
    for recovery_event in recovery_events:
        if (
            recovery_event.payload.get("operation_id") != state.operation_id
            or recovery_event.payload.get("stream_protocol") != state.stream_protocol
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation recovery evidence is bound to a different operation."
            )
    cancellation_status = ProviderOperationCancellationStatus.NOT_REQUESTED
    if cancellation_events:
        latest_cancellation = cancellation_events[0]
        if (
            latest_cancellation.payload.get("operation_id") != state.operation_id
            or latest_cancellation.payload.get("stream_protocol") != state.stream_protocol
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation cancellation evidence is bound to a different operation."
            )
        try:
            cancellation_status = ProviderOperationCancellationStatus(
                latest_cancellation.payload.get("cancellation_status")
            )
        except ValueError:
            raise ProviderOperationEvidenceError(
                "Provider-operation cancellation evidence has an invalid status."
            ) from None
        if (
            latest_cancellation.type is EventType.PROVIDER_OPERATION_CANCEL_REQUESTED
            and cancellation_status is not ProviderOperationCancellationStatus.REQUESTED
        ) or (
            latest_cancellation.type is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED
            and cancellation_status
            in {
                ProviderOperationCancellationStatus.NOT_REQUESTED,
                ProviderOperationCancellationStatus.REQUESTED,
            }
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation cancellation status conflicts with its event type."
            )
    active_stage = await session_store.load_active_model_completion_stage(session_id)
    reservation_ids: tuple[str, ...] = ()
    if active_stage is not None:
        active_attempt_id = active_stage.stage.intent.get("model_attempt_id")
        if (
            active_stage.stage.logical_step_id == current_identity[-2]
            and active_attempt_id == current_identity[-1]
        ):
            reservation_ids = active_stage.stage.reservation_ids
    accounting_status = ProviderOperationAccountingStatus.NOT_APPLICABLE
    reservation_count = len(reservation_ids)
    if reservation_ids:
        accounting_status = ProviderOperationAccountingStatus.RESERVED
        settled_ids: set[str] = set()
        for reservation_id in reservation_ids:
            settlement_event_id = budget_settlement_event_id(budget_settlement_id(reservation_id))
            settlement_records = await session_store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=settlement_event_id,
                    after_sequence=started_record.sequence,
                    limit=1,
                )
            )
            if not settlement_records:
                continue
            settlement = settlement_records[0].event
            if (
                settlement.id != settlement_event_id
                or settlement.type is not EventType.BUDGET_RECONCILED
                or settlement.session_id != session_id
                or settlement.payload.get("reservation_id") != reservation_id
            ):
                raise ProviderOperationEvidenceError(
                    "Provider-operation accounting settlement evidence is contradictory."
                )
            settled_ids.add(reservation_id)
        if set(reservation_ids) <= settled_ids:
            accounting_status = ProviderOperationAccountingStatus.SETTLED
    elif terminal_seen:
        completed_event = next(
            event
            for event in owning_events
            if event.type is EventType.MODEL_COMPLETED
            and is_conversational_model_completion_payload(event.payload)
        )
        settlements = completed_event.payload.get("budget_settlements")
        if type(settlements) is list and settlements:
            reservation_count = len(settlements)
            if reservation_count > 32:
                raise ProviderOperationEvidenceError(
                    "Provider-operation accounting evidence exceeds its bounded reservation set."
                )
            accounting_status = ProviderOperationAccountingStatus.SETTLED
    inspection_fields = {
        "cancellation_status": cancellation_status,
        "accounting_status": accounting_status,
        "reservation_count": reservation_count,
    }
    latest_recovery_type = recovery_events[0].type if recovery_events else None
    if terminal_seen or latest_recovery_type == EventType.PROVIDER_OPERATION_RECONCILED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            **inspection_fields,
        )
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RECONNECT_STARTED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.RECONNECT_IN_PROGRESS,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            **inspection_fields,
        )
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.RECONNECT_SCHEDULED,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            **inspection_fields,
        )
    if status.terminal:
        return ProviderOperationInspection(status=ProviderOperationInspectionStatus.SYNCHRONOUS)
    return ProviderOperationInspection(
        status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS,
        provider=provider,
        operation_id=state.operation_id,
        stream_protocol=state.stream_protocol,
        **inspection_fields,
    )


__all__ = [
    "ProviderOperationAccountingStatus",
    "ProviderOperationCancellationStatus",
    "ProviderOperationEvidenceError",
    "ProviderOperationInspection",
    "ProviderOperationInspectionStatus",
    "ProviderOperationRecoveryResult",
    "ProviderOperationRecoveryStatus",
    "RecoverableProviderOperation",
    "inspect_provider_operation",
    "load_recoverable_provider_operation",
]
