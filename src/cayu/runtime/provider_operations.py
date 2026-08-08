from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import require_durable_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.providers import ProviderOperationState, ProviderOperationStatus
from cayu.providers.operations import (
    PROVIDER_OPERATION_ID_MAX_CHARS,
    PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS,
)
from cayu.runtime.sessions import EventOrder, EventQuery, SessionStore
from cayu.runtime.usage import is_conversational_model_completion_payload

_INSPECTION_EVENT_TYPES = (
    EventType.MODEL_STARTED,
    EventType.PROVIDER_OPERATION_STARTING,
    EventType.PROVIDER_OPERATION_STARTED,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_ERROR,
    EventType.MODEL_ATTEMPT_DISCARDED,
)
_INSPECTION_EVENT_LIMIT = 8
_MODEL_IDENTITY_PAYLOAD_FIELDS = (
    "step",
    "attempt",
    "max_attempts",
    "model_step_id",
    "model_attempt_id",
)


class ProviderOperationInspectionStatus(StrEnum):
    SYNCHRONOUS = "synchronous"
    PROVIDER_OPERATION_IN_PROGRESS = "provider_operation_in_progress"


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


async def inspect_provider_operation(
    session_store: SessionStore,
    session_id: str,
) -> ProviderOperationInspection:
    """Inspect at most one latest model attempt without hydrating stream deltas."""

    records = await session_store.query_events(
        EventQuery(
            session_id=session_id,
            event_types=_INSPECTION_EVENT_TYPES,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=_INSPECTION_EVENT_LIMIT,
        )
    )
    current_identity: tuple[object, ...] | None = None
    current_provider_scope: tuple[str, str] | None = None
    later_events: list[Event] = []
    for record in records:
        event = record.event
        if event.type == EventType.MODEL_STARTED:
            current_identity = _model_identity(event, label="Latest model-attempt evidence")
            current_provider_scope = _provider_scope(
                event,
                label="Latest model-attempt evidence",
            )
            break
        later_events.append(event)
    if current_identity is None:
        if len(records) == _INSPECTION_EVENT_LIMIT:
            raise ProviderOperationEvidenceError(
                "Latest model-attempt evidence exceeds the bounded inspection window."
            )
        return ProviderOperationInspection(status=ProviderOperationInspectionStatus.SYNCHRONOUS)
    if current_provider_scope is None:  # pragma: no cover - assigned with current_identity
        raise ProviderOperationEvidenceError("Latest model-attempt provider scope is unavailable.")

    starting_events = [
        event for event in later_events if event.type == EventType.PROVIDER_OPERATION_STARTING
    ]
    operation_events = [
        event for event in later_events if event.type == EventType.PROVIDER_OPERATION_STARTED
    ]
    owning_events = [
        event
        for event in later_events
        if event.type
        in {
            EventType.PROVIDER_OPERATION_STARTING,
            EventType.PROVIDER_OPERATION_STARTED,
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
    provider_evidence = [*starting_events, *operation_events]
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
    if terminal_seen or status.terminal:
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
    "inspect_provider_operation",
]
