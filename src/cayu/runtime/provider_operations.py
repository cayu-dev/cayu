from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import require_durable_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.providers import ProviderOperationState, ProviderOperationStatus
from cayu.providers.operations import (
    PROVIDER_OPERATION_ID_MAX_CHARS,
    PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS,
)
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    ModelCompletionStage,
    SessionStore,
)
from cayu.runtime.usage import is_conversational_model_completion_payload

_INSPECTION_ATTEMPT_EVENT_TYPES = (
    EventType.PROVIDER_OPERATION_STARTING,
    EventType.PROVIDER_OPERATION_STARTED,
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
)


class ProviderOperationInspectionStatus(StrEnum):
    SYNCHRONOUS = "synchronous"
    PROVIDER_OPERATION_IN_PROGRESS = "provider_operation_in_progress"
    RECONNECT_SCHEDULED = "reconnect_scheduled"
    RECONNECT_IN_PROGRESS = "reconnect_in_progress"
    PROVIDER_OPERATION_RECONCILED = "provider_operation_reconciled"


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


class ProviderOperationEvidenceError(RuntimeError):
    """Durable provider-operation evidence is malformed or contradictory."""


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
    if len(records) > 1:
        prior = _parse_operation_event(records[1].event)
        if prior.model_attempt_identity == latest.model_attempt_identity:
            raise ProviderOperationEvidenceError(
                "Active model attempt has more than one durable provider-operation identity."
            )
    for event_type in _RECOVERY_OUTPUT_EVENT_TYPES:
        accepted = await session_store.query_events(
            EventQuery(
                session_id=stage.session_id,
                event_type=event_type,
                after_sequence=records[0].sequence,
                order_by=EventOrder.SEQUENCE_ASC,
                limit=1,
            )
        )
        if accepted:
            raise ProviderOperationEvidenceError(
                "Provider-operation recovery is unsafe after provider output crossed "
                "Cayu's durable event boundary."
            )
    return latest


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
    recovery_events = [event for event in later_events if event.type in _RECOVERY_EVENT_TYPES]
    owning_events = [
        event
        for event in later_events
        if event.type
        in {
            EventType.PROVIDER_OPERATION_STARTING,
            EventType.PROVIDER_OPERATION_STARTED,
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
    provider_evidence = [*starting_events, *operation_events, *recovery_events]
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
    latest_recovery_type = recovery_events[0].type if recovery_events else None
    if terminal_seen or latest_recovery_type == EventType.PROVIDER_OPERATION_RECONCILED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
        )
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RECONNECT_STARTED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.RECONNECT_IN_PROGRESS,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
        )
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.RECONNECT_SCHEDULED,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
        )
    if status.terminal:
        return ProviderOperationInspection(status=ProviderOperationInspectionStatus.SYNCHRONOUS)
    return ProviderOperationInspection(
        status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS,
        provider=provider,
        operation_id=state.operation_id,
        stream_protocol=state.stream_protocol,
    )


__all__ = [
    "ProviderOperationEvidenceError",
    "ProviderOperationInspection",
    "ProviderOperationInspectionStatus",
    "ProviderOperationRecoveryResult",
    "ProviderOperationRecoveryStatus",
    "RecoverableProviderOperation",
    "inspect_provider_operation",
    "load_recoverable_provider_operation",
]
