from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import (
    Event,
    EventType,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.providers import (
    ModelStreamEvent,
    ModelStreamEventType,
    ProviderOperationRecoveryMetadata,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_model_stream_event,
)
from cayu.providers.operations import (
    PROVIDER_OPERATION_ID_MAX_CHARS,
    PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS,
)
from cayu.runtime.approvals import (
    ResolutionActor,
    copy_resolution_actor,
    resolution_actor_payload,
)
from cayu.runtime.budgets import budget_settlement_event_id, budget_settlement_id
from cayu.runtime.execution_profiles import (
    event_with_execution_profile_fingerprint_authority,
)
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.interactions import InteractionStatus, InteractionSummaryEvidence
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    ModelCompletionStage,
    ModelCompletionStageRelease,
    SessionOperationPublication,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
)
from cayu.runtime.usage import is_conversational_model_completion_payload
from cayu.vaults import SecretRedactor

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
        EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
        EventType.PROVIDER_OPERATION_RESOLVED,
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
_PROVIDER_OPERATION_RESOLUTION_RECORD_TYPE = "cayu.provider-operation-resolution"
_PROVIDER_OPERATION_RESOLUTION_KEY_PREFIX = "cayu.provider-operation-resolution:v1:"
_PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY = (
    "provider_operation_fallback_dispatch_ordinals"
)
_PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY = (
    "provider_operation_pending_resolution_disposition"
)
PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES = 16 * 1024


class ProviderOperationInspectionStatus(StrEnum):
    SYNCHRONOUS = "synchronous"
    PROVIDER_OPERATION_IN_PROGRESS = "provider_operation_in_progress"
    RECONNECT_SCHEDULED = "reconnect_scheduled"
    RECONNECT_IN_PROGRESS = "reconnect_in_progress"
    PROVIDER_OPERATION_RECONCILED = "provider_operation_reconciled"
    PROVIDER_OPERATION_UNAVAILABLE = "provider_operation_unavailable"
    AMBIGUOUS_SUBMISSION = "ambiguous_submission"
    FALLBACK_RETRY = "fallback_retry"


class ProviderOperationUnavailableReason(StrEnum):
    """Why exact continuation of one submitted model attempt is unavailable."""

    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    MALFORMED = "malformed"
    WRONG_PROVIDER = "wrong_provider"
    UNAVAILABLE = "unavailable"
    AMBIGUOUS_SUBMISSION = "ambiguous_submission"


class ProviderOperationResolutionAction(StrEnum):
    """Explicit dispositions for a provider operation Cayu cannot continue exactly."""

    FALLBACK_RETRY = "fallback_retry"
    FAIL = "fail"


def copy_provider_operation_resolution_metadata(value: object) -> dict[str, Any]:
    """Own and enforce the exact durable metadata ceiling for one resolution."""

    copied = copy_durable_json_object(value, "metadata")
    if (
        len(canonical_durable_json_bytes(copied, "metadata"))
        > PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES
    ):
        raise ValueError("Provider-operation resolution metadata is too large.")
    return copied


class ProviderOperationResolutionRequest(BaseModel):
    """One explicit, run-epoch-fenced disposition of unavailable provider work."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    task_worker_id: str | None = None
    task_handoff_id: str | None = None
    stage_id: str
    expected_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    action: ProviderOperationResolutionAction
    reason: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None

    @field_validator("session_id", "stage_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("task_worker_id", "task_handoff_id")
    @classmethod
    def validate_task_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "task continuation authority")

    @model_validator(mode="after")
    def validate_task_handoff_authority(self) -> ProviderOperationResolutionRequest:
        if self.task_handoff_id is not None and self.task_worker_id is None:
            raise ValueError("task_handoff_id requires task_worker_id.")
        return self

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, "reason")

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: object) -> dict[str, Any]:
        return copy_provider_operation_resolution_metadata(value)

    @field_validator("resolved_by")
    @classmethod
    def validate_resolved_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)


def copy_provider_operation_resolution_request(
    request: ProviderOperationResolutionRequest,
    *,
    session_id: str | None = None,
) -> ProviderOperationResolutionRequest:
    """Own one public resolution request without serializing caller-owned state."""

    if type(request) is not ProviderOperationResolutionRequest:
        raise TypeError("request must be a ProviderOperationResolutionRequest.")
    return ProviderOperationResolutionRequest(
        session_id=request.session_id if session_id is None else session_id,
        task_worker_id=request.task_worker_id,
        task_handoff_id=request.task_handoff_id,
        stage_id=request.stage_id,
        expected_run_epoch=request.expected_run_epoch,
        action=request.action,
        reason=request.reason,
        metadata=copy_provider_operation_resolution_metadata(request.metadata),
        resolved_by=copy_resolution_actor(request.resolved_by),
    )


def prepare_provider_operation_resolution_request(
    request: ProviderOperationResolutionRequest,
    *,
    redactor: SecretRedactor,
    session_id: str | None = None,
) -> ProviderOperationResolutionRequest:
    """Own and redact one resolution before hashing or durable publication."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    copied = copy_provider_operation_resolution_request(request, session_id=session_id)
    redactor.require_no_secret_keys(
        copied.metadata,
        field_name="ProviderOperationResolutionRequest.metadata",
        match_short_substrings=True,
    )
    metadata = redactor.redact_json_values(copied.metadata)
    if type(metadata) is not dict:
        raise AssertionError("Provider-operation resolution metadata redaction failed.")
    actor = copy_resolution_actor(copied.resolved_by)
    if actor is not None:
        redactor.require_no_secret_keys(
            actor.claims,
            field_name="ProviderOperationResolutionRequest.resolved_by.claims",
            match_short_substrings=True,
        )
        claims = redactor.redact_json_values(actor.claims)
        if type(claims) is not dict:
            raise AssertionError("Provider-operation actor-claim redaction failed.")
        actor = ResolutionActor(
            subject=redactor.redact_text(actor.subject),
            tenant=(None if actor.tenant is None else redactor.redact_text(actor.tenant)),
            source=actor.source,
            claims=claims,
        )
    return ProviderOperationResolutionRequest(
        session_id=copied.session_id,
        task_worker_id=copied.task_worker_id,
        stage_id=copied.stage_id,
        expected_run_epoch=copied.expected_run_epoch,
        action=copied.action,
        reason=(None if copied.reason is None else redactor.redact_text(copied.reason)),
        metadata=metadata,
        resolved_by=actor,
    )


class ProviderOperationResolutionRecord(BaseModel):
    """Immutable audit record for one provider-operation disposition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.provider-operation-resolution"] = (
        _PROVIDER_OPERATION_RESOLUTION_RECORD_TYPE
    )
    schema_version: Literal[1] = 1
    resolution_id: str
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_attempt_id: str
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    resolved_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    action: ProviderOperationResolutionAction
    recovery_reason: ProviderOperationUnavailableReason
    duplicate_request_risk: bool
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: dict[str, Any] | None = None
    resolved_at: datetime
    event_id: str
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_profile_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    record_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProviderOperationPendingDisposition(BaseModel):
    """Durable retry ownership for one accepted provider-operation resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.provider-operation-resolution-disposition"] = (
        "cayu.provider-operation-resolution-disposition"
    )
    schema_version: Literal[1] = 1
    session_id: str
    stage_id: str
    resolution_id: str
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: ProviderOperationResolutionAction
    resolved_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    logical_step_id: str
    source_step: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    source_dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    target_dispatch_ordinal: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    execution_profile_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    execution_claimed: bool = False
    execution_task_worker_id: str | None = None
    execution_task_handoff_id: str | None = None

    @field_validator("session_id", "stage_id", "resolution_id", "logical_step_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("execution_task_worker_id", "execution_task_handoff_id")
    @classmethod
    def validate_execution_task_authority(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_target_dispatch(self) -> ProviderOperationPendingDisposition:
        if self.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            if self.target_dispatch_ordinal != self.source_dispatch_ordinal + 1:
                raise ValueError(
                    "Fallback provider-operation disposition has an invalid target ordinal."
                )
        elif self.target_dispatch_ordinal is not None:
            raise ValueError("Fail provider-operation disposition cannot target a dispatch.")
        if not self.execution_claimed and (
            self.execution_task_worker_id is not None or self.execution_task_handoff_id is not None
        ):
            raise ValueError("Unclaimed provider-operation execution cannot name task authority.")
        if self.execution_task_handoff_id is not None and self.execution_task_worker_id is None:
            raise ValueError(
                "Provider-operation execution handoff authority requires a task worker."
            )
        return self


def checkpoint_with_provider_operation_disposition_execution_owner(
    checkpoint: dict[str, Any] | None,
    *,
    expected: ProviderOperationPendingDisposition,
    task_worker_id: str | None,
    task_handoff_id: str | None,
) -> dict[str, Any]:
    """Bind one pending disposition to its current execution owner by exact CAS."""

    if type(expected) is not ProviderOperationPendingDisposition:
        raise TypeError("expected must be a ProviderOperationPendingDisposition.")
    if task_worker_id is not None:
        task_worker_id = require_durable_clean_nonblank(task_worker_id, "task_worker_id")
    if task_handoff_id is not None:
        task_handoff_id = require_durable_clean_nonblank(
            task_handoff_id,
            "task_handoff_id",
        )
    if task_handoff_id is not None and task_worker_id is None:
        raise ValueError("Provider-operation execution handoff authority requires a task worker.")
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    current = pending_provider_operation_disposition_from_checkpoint(updated)
    if current is None or current != expected:
        raise ProviderOperationResolutionConflict(
            "Provider-operation execution ownership changed before it was claimed."
        )
    claimed = current.model_copy(
        update={
            "execution_claimed": True,
            "execution_task_worker_id": task_worker_id,
            "execution_task_handoff_id": task_handoff_id,
        }
    )
    updated[_PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY] = claimed.model_dump(
        mode="json"
    )
    return updated


class ProviderOperationResolutionResult(BaseModel):
    """A newly committed provider disposition or its exact replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: ProviderOperationResolutionRecord
    event: Event
    replayed: bool


class ProviderOperationResolutionConflict(RuntimeError):
    """A provider operation already has a different durable disposition."""


class _ProviderOperationResolutionReplay(RuntimeError):
    pass


def _provider_operation_stage_profile_fingerprint(
    stage: ModelCompletionStage,
) -> str | None:
    """Return profile authority carried by a stage, or ``None`` for legacy stages."""

    raw_recovery_context = stage.intent.get("recovery_context")
    if raw_recovery_context is None:
        return None
    if type(raw_recovery_context) is not dict:
        raise ProviderOperationEvidenceError(
            "Provider-operation stage has malformed recovery authority."
        )
    if "execution_profile_fingerprint" not in raw_recovery_context:
        return None
    profile_fingerprint = raw_recovery_context.get("execution_profile_fingerprint")
    if (
        type(profile_fingerprint) is not str
        or len(profile_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in profile_fingerprint)
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation stage has no execution-profile authority."
        )
    return profile_fingerprint


def _require_provider_operation_event_profile(
    event: Event,
    *,
    stage: ModelCompletionStage,
    label: str,
) -> None:
    """Require event attribution whenever the owning stage is profiled.

    Historical stages predate execution-profile attribution, so their events
    remain readable without the field. A profiled stage is positive evidence
    that every governed event belongs to the new protocol and must carry the
    exact same fingerprint.
    """

    expected = _provider_operation_stage_profile_fingerprint(stage)
    if expected is None and "execution_profile_fingerprint" not in event.payload:
        return
    if "execution_profile_fingerprint" not in event.payload:
        raise ProviderOperationEvidenceError(
            f"{label} has no execution-profile evidence for its profiled stage."
        )
    observed = event.payload.get("execution_profile_fingerprint")
    if (
        type(observed) is not str
        or len(observed) != 64
        or any(character not in "0123456789abcdef" for character in observed)
    ):
        raise ProviderOperationEvidenceError(f"{label} has malformed execution-profile evidence.")
    if expected is None or observed != expected:
        raise ProviderOperationEvidenceError(
            f"{label} conflicts with its active execution profile."
        )


def provider_operation_unavailable_reason(
    status: ProviderOperationStatus,
) -> ProviderOperationUnavailableReason | None:
    """Map a terminal provider status to its bounded unavailable reason."""

    if type(status) is not ProviderOperationStatus:
        raise TypeError("status must be a ProviderOperationStatus.")
    return {
        ProviderOperationStatus.FAILED: ProviderOperationUnavailableReason.FAILED,
        ProviderOperationStatus.EXPIRED: ProviderOperationUnavailableReason.EXPIRED,
        ProviderOperationStatus.CANCELLED: ProviderOperationUnavailableReason.CANCELLED,
        ProviderOperationStatus.UNAVAILABLE: ProviderOperationUnavailableReason.UNAVAILABLE,
        ProviderOperationStatus.COMPLETED: ProviderOperationUnavailableReason.MALFORMED,
    }.get(status)


def provider_operation_duplicate_request_risk(
    reason: ProviderOperationUnavailableReason,
) -> bool:
    """Return whether fallback may overlap provider work with an unknown outcome."""

    if type(reason) is not ProviderOperationUnavailableReason:
        raise TypeError("reason must be a ProviderOperationUnavailableReason.")
    return reason in {
        ProviderOperationUnavailableReason.MALFORMED,
        ProviderOperationUnavailableReason.WRONG_PROVIDER,
        ProviderOperationUnavailableReason.UNAVAILABLE,
        ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION,
    }


def _parse_provider_operation_resolution_event(
    event: Event,
) -> tuple[
    ProviderOperationResolutionAction,
    ProviderOperationUnavailableReason,
    str,
]:
    try:
        resolution_action = ProviderOperationResolutionAction(
            event.payload.get("resolution_action")
        )
        recovery_reason = ProviderOperationUnavailableReason(event.payload.get("recovery_reason"))
        raw_resolution_id = event.payload.get("resolution_id")
        if type(raw_resolution_id) is not str:
            raise ValueError
        resolution_id = require_durable_clean_nonblank(
            raw_resolution_id,
            "resolution_id",
        )
    except (TypeError, ValueError):
        raise ProviderOperationEvidenceError(
            "Provider-operation resolution evidence is malformed."
        ) from None
    return resolution_action, recovery_reason, resolution_id


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
    stage_id: str | None = Field(default=None, max_length=256)
    run_epoch: int | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    recovery_reason: ProviderOperationUnavailableReason | None = None
    duplicate_request_risk: bool = False
    allowed_resolutions: tuple[ProviderOperationResolutionAction, ...] = ()
    resolution_action: ProviderOperationResolutionAction | None = None
    resolution_id: str | None = Field(default=None, max_length=256)


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


@dataclass(frozen=True, slots=True)
class RecoverableProviderOperationStart:
    """Durable start-only evidence bound to one active model attempt."""

    interaction_id: str
    provider: str
    model: str
    model_attempt_identity: ModelAttemptIdentity
    start_id: str
    idempotency_support: ProviderOperationStartIdempotencySupport
    step: int
    attempt: int
    max_attempts: int
    source_run_epoch: int


class ProviderOperationRecoveryStatus(StrEnum):
    """Outcome of one fenced provider-operation retrieval attempt."""

    PENDING = "pending"
    RECONCILED = "reconciled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ProviderOperationRecoveryResult:
    """Events and state produced by one exact-operation recovery attempt."""

    status: ProviderOperationRecoveryStatus
    events: tuple[Event, ...]
    completion_event: Event | None = None
    unavailable_reason: ProviderOperationUnavailableReason | None = None


def _model_identity(event: Event, *, label: str) -> tuple[str, int, int, int, str, str]:
    step = event.payload.get("step")
    attempt = event.payload.get("attempt")
    max_attempts = event.payload.get("max_attempts")
    model_step_id = event.payload.get("model_step_id")
    model_attempt_id = event.payload.get("model_attempt_id")
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
    return (
        event.interaction_id,
        step,
        attempt,
        max_attempts,
        model_step_id,
        model_attempt_id,
    )


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
            _require_provider_operation_event_profile(
                event,
                stage=stage,
                label="Provider-operation progress evidence",
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
    if final_event.type is ModelStreamEventType.COMPLETED or (
        final_event.type is ModelStreamEventType.ERROR
        and (
            final_event.provider_operation_status is None
            or final_event.provider_operation_status.terminal
        )
    ):
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
    _require_provider_operation_event_profile(
        records[0].event,
        stage=stage,
        label="Provider-operation recovery evidence",
    )
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
    _require_provider_operation_event_profile(
        started_event,
        stage=stage,
        label="Authoritative model-started evidence",
    )
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
            _require_provider_operation_event_profile(
                output_event,
                stage=stage,
                label="Provider-operation output evidence",
            )
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


async def load_recoverable_provider_operation_start(
    session_store: SessionStore,
    stage: ModelCompletionStage,
) -> RecoverableProviderOperationStart | None:
    """Load one start-only attempt as exact-recovery or ambiguous evidence."""

    records = await session_store.query_events(
        EventQuery(
            session_id=stage.session_id,
            event_type=EventType.PROVIDER_OPERATION_STARTING,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=_RECOVERY_EVIDENCE_LIMIT,
        )
    )
    model_attempt_id = stage.intent.get("model_attempt_id")
    matching = [
        record.event
        for record in records
        if record.event.payload.get("model_step_id") == stage.logical_step_id
        and record.event.payload.get("model_attempt_id") == model_attempt_id
    ]
    if not matching:
        return None
    if len(matching) != 1:
        raise ProviderOperationEvidenceError(
            "Provider-operation start-only evidence is contradictory."
        )
    event = matching[0]
    _require_provider_operation_event_profile(
        event,
        stage=stage,
        label="Provider-operation starting evidence",
    )
    try:
        identity = _model_identity(event, label="Provider-operation starting evidence")
        provider, model = _provider_scope(
            event,
            label="Provider-operation starting evidence",
        )
        source_run_epoch = _provider_operation_epoch(
            event,
            label="Provider-operation starting evidence",
        )
        support = ProviderOperationStartIdempotencySupport(
            event.payload.get("start_idempotency_support")
        )
        raw_start_id = event.payload.get("start_id")
        if type(raw_start_id) is not str:
            raise ValueError
        start_id = require_durable_clean_nonblank(raw_start_id, "start_id")
    except (ProviderOperationEvidenceError, TypeError, ValueError):
        raise ProviderOperationEvidenceError(
            "Provider-operation starting evidence is malformed."
        ) from None
    replay = stage.intent.get("provider_operation_start")
    if type(replay) is not dict:
        raise ProviderOperationEvidenceError("Provider-operation start intent is malformed.")
    try:
        if (
            replay.get("schema_version") != 1
            or ProviderOperationStartIdempotencySupport(replay.get("idempotency_support"))
            is not support
            or replay.get("idempotency_key") != start_id
            or set(replay)
            != {
                "schema_version",
                "idempotency_support",
                "idempotency_key",
            }
        ):
            raise ValueError
        request_fingerprint = stage.intent.get("request_fingerprint")
        if (
            type(request_fingerprint) is not str
            or len(request_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in request_fingerprint)
        ):
            raise ValueError
    except (TypeError, ValueError):
        raise ProviderOperationEvidenceError(
            "Provider-operation start intent is malformed."
        ) from None
    interaction_id, step, attempt, max_attempts, model_step_id, parsed_attempt_id = identity
    if (
        interaction_id is None
        or model_step_id != stage.logical_step_id
        or parsed_attempt_id != model_attempt_id
        or source_run_epoch != stage.source_run_epoch
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation starting evidence conflicts with its active stage."
        )
    return RecoverableProviderOperationStart(
        interaction_id=interaction_id,
        provider=provider,
        model=model,
        model_attempt_identity=ModelAttemptIdentity(
            model_step_id=model_step_id,
            model_attempt_id=parsed_attempt_id,
        ),
        start_id=start_id,
        idempotency_support=support,
        step=step,
        attempt=attempt,
        max_attempts=max_attempts,
        source_run_epoch=source_run_epoch,
    )


def provider_operation_resolution_storage_key(stage_id: str) -> str:
    """Return the immutable disposition key for one model-completion stage."""

    stage_id = require_durable_clean_nonblank(stage_id, "stage_id")
    return _PROVIDER_OPERATION_RESOLUTION_KEY_PREFIX + sha256(stage_id.encode()).hexdigest()


def provider_operation_started_event_id(start_id: str) -> str:
    """Return one stable operation-identity publication event id per start key."""

    start_id = require_durable_clean_nonblank(start_id, "start_id")
    return f"provider-operation-started:v1:{sha256(start_id.encode()).hexdigest()}"


def provider_operation_resolution_outcome_event_id(
    resolution_id: str,
    outcome: Literal[
        "model_error",
        "task_failed",
        "interaction_failed",
        "session_failed",
    ],
) -> str:
    """Return one stable event identity for fail-resolution terminalization."""

    resolution_id = require_durable_clean_nonblank(resolution_id, "resolution_id")
    return f"{resolution_id}:{outcome}"


def validate_provider_operation_resolution_outcome_event(
    event: Event,
    *,
    resolution_event: Event,
    outcome: Literal["model_error", "interaction_failed", "session_failed"],
    expected_execution_profile_fingerprint: str | None = None,
) -> None:
    """Require exact durable evidence for one explicit-failure outcome."""

    if resolution_event.type != EventType.PROVIDER_OPERATION_RESOLVED:
        raise ProviderOperationEvidenceError(
            "Provider-operation failure outcome has no resolution authority."
        )
    resolution_id = resolution_event.payload.get("resolution_id")
    if type(resolution_id) is not str:
        raise ProviderOperationEvidenceError(
            "Provider-operation failure outcome has malformed resolution authority."
        )
    expected_type = {
        "model_error": EventType.MODEL_ERROR,
        "interaction_failed": EventType.INTERACTION_FAILED,
        "session_failed": EventType.SESSION_FAILED,
    }[outcome]
    expected_interaction_id = (
        None if outcome == "session_failed" else resolution_event.interaction_id
    )
    if (
        event.id != provider_operation_resolution_outcome_event_id(resolution_id, outcome)
        or event.type != expected_type
        or event.session_id != resolution_event.session_id
        or event.interaction_id != expected_interaction_id
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation failure outcome has contradictory event identity."
        )

    if outcome == "interaction_failed":
        try:
            evidence = InteractionSummaryEvidence.model_validate(event.payload)
        except (TypeError, ValueError):
            raise ProviderOperationEvidenceError(
                "Provider-operation interaction failure evidence is malformed."
            ) from None
        if evidence.status is not InteractionStatus.FAILED:
            raise ProviderOperationEvidenceError(
                "Provider-operation interaction failure has contradictory status."
            )
        return

    if expected_execution_profile_fingerprint is not None and (
        type(expected_execution_profile_fingerprint) is not str
        or len(expected_execution_profile_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_execution_profile_fingerprint
        )
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation failure outcome has malformed expected profile authority."
        )
    resolution_has_profile = "execution_profile_fingerprint" in resolution_event.payload
    outcome_has_profile = "execution_profile_fingerprint" in event.payload
    resolution_profile = resolution_event.payload.get("execution_profile_fingerprint")
    outcome_profile = event.payload.get("execution_profile_fingerprint")
    if resolution_has_profile:
        profiles_match = (
            outcome_has_profile
            and type(resolution_profile) is str
            and type(outcome_profile) is str
            and len(resolution_profile) == 64
            and all(character in "0123456789abcdef" for character in resolution_profile)
            and outcome_profile == resolution_profile
            and (
                expected_execution_profile_fingerprint is None
                or resolution_profile == expected_execution_profile_fingerprint
            )
        )
    elif outcome_has_profile:
        # Schema-v1 resolution evidence written before profile attribution can
        # be completed after an upgrade. In that mixed-version case, the exact
        # pending disposition/source stage supplies the independently frozen
        # profile authority for the newly written outcome.
        profiles_match = (
            expected_execution_profile_fingerprint is not None
            and type(outcome_profile) is str
            and outcome_profile == expected_execution_profile_fingerprint
        )
    else:
        # Historical resolution and outcome events both predate profile attribution.
        profiles_match = True
    if not profiles_match:
        raise ProviderOperationEvidenceError(
            "Provider-operation failure outcome conflicts with its execution profile."
        )

    for field in (
        "provider",
        "model",
        "step",
        "attempt",
        "max_attempts",
        "model_step_id",
        "model_attempt_id",
        "source_run_epoch",
        "run_epoch",
        "stage_id",
        "resolution_id",
        "resolution_action",
        "recovery_reason",
        "duplicate_request_risk",
        "reason",
        "metadata",
        "resolved_by",
    ):
        if (
            field not in event.payload
            or field not in resolution_event.payload
            or event.payload[field] != resolution_event.payload[field]
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation failure outcome conflicts with its resolution."
            )
    if outcome == "model_error" and (
        event.payload.get("error_type") != "provider_operation_unavailable"
        or event.payload.get("stage") != "provider_operation_recovery"
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation model failure has contradictory classification."
        )
    if outcome == "session_failed" and (
        event.payload.get("failure_type") != "provider_operation_unavailable"
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation session failure has contradictory classification."
        )


def provider_operation_resolution_request_digest(
    request: ProviderOperationResolutionRequest,
) -> str:
    """Bind the exact audit-bearing operator request, excluding authorization claims."""

    if type(request) is not ProviderOperationResolutionRequest:
        raise TypeError("request must be a ProviderOperationResolutionRequest.")
    material = request.model_dump(
        mode="json",
        exclude={"resolved_by", "task_worker_id"},
    )
    material["resolved_by"] = resolution_actor_payload(request.resolved_by)
    return sha256(
        canonical_durable_json_bytes(material, "provider_operation_resolution_request")
    ).hexdigest()


def fallback_dispatch_ordinal_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    logical_step_id: str,
) -> int:
    """Return the durable next ordinal authorized by prior explicit fallback decisions."""

    logical_step_id = require_durable_clean_nonblank(logical_step_id, "logical_step_id")
    if checkpoint is None:
        return 0
    raw = checkpoint.get(_PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY)
    if raw is None:
        return 0
    if type(raw) is not dict:
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback ordinal evidence is malformed."
        )
    value = raw.get(logical_step_id, 0)
    if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ProviderOperationEvidenceError(
            "Provider-operation fallback ordinal evidence is malformed."
        )
    return value


def _resolution_record_digest(payload: dict[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key != "record_digest"}
    return sha256(
        canonical_durable_json_bytes(material, "provider_operation_resolution_record")
    ).hexdigest()


def _parse_provider_operation_resolution_record(
    raw: object,
    *,
    session_id: str,
    stage_id: str,
) -> ProviderOperationResolutionRecord:
    try:
        record = ProviderOperationResolutionRecord.model_validate(raw)
    except (TypeError, ValueError):
        raise ProviderOperationResolutionConflict(
            "Provider-operation resolution evidence is malformed."
        ) from None
    if record.session_id != session_id or record.stage_id != stage_id:
        raise ProviderOperationResolutionConflict(
            "Provider-operation resolution belongs to another stage."
        )
    if record.execution_profile_fingerprint is None and (
        type(raw) is not dict or "execution_profile_fingerprint" in raw
    ):
        raise ProviderOperationResolutionConflict(
            "Provider-operation resolution profile evidence is malformed."
        )
    record_payload = record.model_dump(mode="json")
    expected_digest = _resolution_record_digest(record_payload)
    if record.record_digest != expected_digest and record.execution_profile_fingerprint is None:
        # Schema-v1 records written before execution-profile attribution did not
        # carry this optional authority field. Preserve their original digest
        # material while requiring all newly written records to bind it.
        record_payload.pop("execution_profile_fingerprint")
        expected_digest = _resolution_record_digest(record_payload)
    if record.record_digest != expected_digest:
        raise ProviderOperationResolutionConflict(
            "Provider-operation resolution digest is invalid."
        )
    return record


async def _load_provider_operation_resolution_event(
    session_store: SessionStore,
    record: ProviderOperationResolutionRecord,
) -> Event:
    records = await session_store.query_events(
        EventQuery(session_id=record.session_id, event_id=record.event_id, limit=2)
    )
    if len(records) != 1:
        raise ProviderOperationResolutionConflict(
            "Provider-operation resolution event is missing or duplicated."
        )
    event = records[0].event
    if (
        event.type != EventType.PROVIDER_OPERATION_RESOLVED
        or event.id != record.event_id
        or event.session_id != record.session_id
        or any(
            field not in event.payload
            for field in (
                "resolution_id",
                "stage_id",
                "model_step_id",
                "model_attempt_id",
                "source_run_epoch",
                "run_epoch",
                "resolution_action",
                "recovery_reason",
                "duplicate_request_risk",
                "reason",
                "metadata",
                "resolved_by",
            )
        )
        or (
            ("execution_profile_fingerprint" in event.payload)
            != (record.execution_profile_fingerprint is not None)
        )
        or event.payload.get("execution_profile_fingerprint")
        != record.execution_profile_fingerprint
        or event.payload.get("resolution_id") != record.resolution_id
        or event.payload.get("stage_id") != record.stage_id
        or event.payload.get("model_step_id") != record.logical_step_id
        or event.payload.get("model_attempt_id") != record.model_attempt_id
        or event.payload.get("source_run_epoch") != record.source_run_epoch
        or event.payload.get("run_epoch") != record.resolved_run_epoch
        or event.payload.get("resolution_action") != record.action.value
        or event.payload.get("recovery_reason") != record.recovery_reason.value
        or event.payload.get("duplicate_request_risk") != record.duplicate_request_risk
        or event.payload.get("reason") != record.reason
        or event.payload.get("metadata") != record.metadata
        or event.payload.get("resolved_by") != record.resolved_by
        or event.timestamp != record.resolved_at
    ):
        raise ProviderOperationResolutionConflict(
            "Provider-operation resolution event conflicts with its record."
        )
    return event


async def load_provider_operation_resolution(
    session_store: SessionStore,
    session_id: str,
    stage_id: str,
) -> ProviderOperationResolutionResult | None:
    """Load and verify one immutable provider-operation disposition."""

    storage_key = provider_operation_resolution_storage_key(stage_id)
    raw = await session_store.load_session_operation(session_id, storage_key)
    if raw is None:
        return None
    record = _parse_provider_operation_resolution_record(
        raw,
        session_id=session_id,
        stage_id=stage_id,
    )
    event = await _load_provider_operation_resolution_event(session_store, record)
    return ProviderOperationResolutionResult(record=record, event=event, replayed=True)


def pending_provider_operation_disposition_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> ProviderOperationPendingDisposition | None:
    """Reconstruct the exact accepted disposition still owned by recovery."""

    if checkpoint is None:
        return None
    raw = checkpoint.get(_PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY)
    if raw is None:
        return None
    try:
        return ProviderOperationPendingDisposition.model_validate(raw)
    except (TypeError, ValueError):
        raise ProviderOperationEvidenceError(
            "Pending provider-operation disposition evidence is malformed."
        ) from None


async def load_pending_provider_operation_disposition(
    session_store: SessionStore,
    session_id: str,
    *,
    checkpoint: dict[str, Any] | None = None,
) -> tuple[ProviderOperationPendingDisposition, ProviderOperationResolutionResult] | None:
    """Load and cross-check pending retry ownership with immutable resolution evidence."""

    session_id = require_durable_clean_nonblank(session_id, "session_id")
    if checkpoint is None:
        checkpoint = await session_store.load_checkpoint(session_id)
    pending = pending_provider_operation_disposition_from_checkpoint(checkpoint)
    if pending is None:
        return None
    if pending.session_id != session_id:
        raise ProviderOperationEvidenceError(
            "Pending provider-operation disposition belongs to another session."
        )
    resolution = await load_provider_operation_resolution(
        session_store,
        session_id,
        pending.stage_id,
    )
    if resolution is None:
        raise ProviderOperationEvidenceError(
            "Pending provider-operation disposition has no immutable resolution record."
        )
    record = resolution.record
    if (
        record.resolution_id != pending.resolution_id
        or record.request_digest != pending.request_digest
        or record.action is not pending.action
        or record.resolved_run_epoch != pending.resolved_run_epoch
        or record.logical_step_id != pending.logical_step_id
        or record.dispatch_ordinal != pending.source_dispatch_ordinal
        or (
            record.execution_profile_fingerprint is not None
            and record.execution_profile_fingerprint != pending.execution_profile_fingerprint
        )
        or resolution.event.payload.get("step") != pending.source_step
    ):
        raise ProviderOperationEvidenceError(
            "Pending provider-operation disposition conflicts with its resolution record."
        )
    stage = await session_store.load_model_completion_stage(session_id, pending.stage_id)
    if stage is None:
        raise ProviderOperationEvidenceError(
            "Pending provider-operation disposition has no source stage."
        )
    model_attempt_id = stage.intent.get("model_attempt_id")
    if (
        stage.session_id != record.session_id
        or stage.stage_id != record.stage_id
        or stage.logical_step_id != record.logical_step_id
        or stage.dispatch_ordinal != record.dispatch_ordinal
        or stage.preparation_digest != record.preparation_digest
        or model_attempt_id != record.model_attempt_id
        or stage.source_run_epoch != record.source_run_epoch
        or _provider_operation_stage_profile_fingerprint(stage)
        != pending.execution_profile_fingerprint
    ):
        raise ProviderOperationEvidenceError(
            "Pending provider-operation disposition conflicts with its source stage."
        )
    return pending, resolution


async def clear_pending_provider_operation_disposition(
    session_store: SessionStore,
    pending: ProviderOperationPendingDisposition,
) -> None:
    """Retire exact disposition ownership after its durable effect is reconstructable."""

    if type(pending) is not ProviderOperationPendingDisposition:
        raise TypeError("pending must be a ProviderOperationPendingDisposition.")

    def clear_marker(
        _session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        current = pending_provider_operation_disposition_from_checkpoint(checkpoint)
        if current is None:
            return checkpoint
        if current != pending:
            raise ProviderOperationResolutionConflict(
                "Provider-operation disposition ownership changed before completion."
            )
        updated = copy_durable_json_object(checkpoint, "checkpoint")
        updated.pop(_PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY)
        return updated

    await session_store.transform_checkpoint(pending.session_id, clear_marker)


async def resolve_provider_operation_stage(
    session_store: SessionStore,
    request: ProviderOperationResolutionRequest,
    *,
    redactor: SecretRedactor,
    before_resolution: Callable[[], Awaitable[None]] | None = None,
) -> ProviderOperationResolutionResult:
    """Atomically record one disposition and release only its exact active stage."""

    if type(request) is not ProviderOperationResolutionRequest:
        raise TypeError("request must be a ProviderOperationResolutionRequest.")
    request = prepare_provider_operation_resolution_request(request, redactor=redactor)
    request_digest = provider_operation_resolution_request_digest(request)
    existing = await load_provider_operation_resolution(
        session_store,
        request.session_id,
        request.stage_id,
    )
    if existing is not None:
        if existing.record.request_digest != request_digest:
            raise ProviderOperationResolutionConflict(
                "Provider operation was already resolved by a conflicting request."
            )
        active = await session_store.load_active_model_completion_stage(request.session_id)
        if active is not None and active.stage.stage_id == request.stage_id:
            raise ProviderOperationResolutionConflict(
                "Existing provider-operation resolution does not prove atomic source-stage release."
            )
        if before_resolution is not None:
            await before_resolution()
        return existing
    if not session_store._supports_atomic_model_completion_stage_release_protocol():
        raise ProviderOperationResolutionConflict(
            "The session store does not support atomic provider-operation stage release."
        )

    session = await session_store.load(request.session_id)
    if session is None:
        raise KeyError(f"Session not found: {request.session_id}")
    if session.run_epoch != request.expected_run_epoch:
        raise SessionRunFenced(
            "Provider-operation resolution run epoch is stale: expected "
            f"{request.expected_run_epoch}, current {session.run_epoch}."
        )
    if session.status is not SessionStatus.INTERRUPTED:
        raise SessionStatusConflict(
            "A new provider-operation resolution requires an interrupted session."
        )
    active = await session_store.load_active_model_completion_stage(request.session_id)
    if active is None or active.stage.stage_id != request.stage_id:
        raise ProviderOperationResolutionConflict(
            "The requested provider-operation stage is no longer active."
        )
    stage = active.stage
    if stage.state != "in_flight":
        raise ProviderOperationResolutionConflict(
            "A completed model-completion stage cannot be resolved as unavailable."
        )
    try:
        profile_fingerprint = _provider_operation_stage_profile_fingerprint(stage)
    except ProviderOperationEvidenceError as exc:
        raise ProviderOperationResolutionConflict(str(exc)) from None
    if profile_fingerprint is None:
        raise ProviderOperationResolutionConflict(
            "Legacy provider-operation stages cannot enter a profiled resolution."
        )
    inspection = await inspect_provider_operation(session_store, request.session_id)
    if (
        inspection.status
        not in {
            ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE,
            ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION,
        }
        or inspection.recovery_reason is None
        or request.action not in inspection.allowed_resolutions
    ):
        raise ProviderOperationResolutionConflict(
            "The provider operation does not currently require this resolution."
        )
    model_attempt_id = stage.intent.get("model_attempt_id")
    if type(model_attempt_id) is not str or not model_attempt_id.strip():
        raise ProviderOperationResolutionConflict(
            "The active model-completion stage has no model-attempt identity."
        )
    model_records = await session_store.query_events(
        EventQuery(
            session_id=request.session_id,
            event_type=EventType.MODEL_STARTED,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not model_records:
        raise ProviderOperationResolutionConflict(
            "The active model-completion stage has no model-started evidence."
        )
    model_event = model_records[0].event
    model_identity = _model_identity(model_event, label="Provider resolution model evidence")
    if model_identity[-2:] != (stage.logical_step_id, model_attempt_id):
        raise ProviderOperationResolutionConflict(
            "The latest model-attempt evidence belongs to another stage."
        )
    resolved_at = datetime.now(UTC)
    resolution_material = canonical_durable_json_bytes(
        {"schema_version": 1, "session_id": request.session_id, "stage_id": stage.stage_id},
        "provider_operation_resolution_identity",
    )
    resolution_hash = sha256(resolution_material).hexdigest()
    resolution_id = f"provider-resolution:v1:{resolution_hash}"
    event_id = f"provider-resolution-event:v1:{resolution_hash}"
    step, attempt, max_attempts = model_identity[1:4]
    event_payload: dict[str, Any] = {
        "provider": inspection.provider,
        "model": stage.intent.get("requested_model"),
        "step": step,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "model_step_id": stage.logical_step_id,
        "model_attempt_id": model_attempt_id,
        "source_run_epoch": stage.source_run_epoch,
        "run_epoch": request.expected_run_epoch,
        "stage_id": stage.stage_id,
        "resolution_id": resolution_id,
        "resolution_action": request.action.value,
        "recovery_reason": inspection.recovery_reason.value,
        "duplicate_request_risk": inspection.duplicate_request_risk,
        "reason": request.reason,
        "metadata": request.metadata,
        "resolved_by": resolution_actor_payload(request.resolved_by),
    }
    if inspection.operation_id is not None:
        event_payload["operation_id"] = inspection.operation_id
    if inspection.stream_protocol is not None:
        event_payload["stream_protocol"] = inspection.stream_protocol
    resolution_event = Event(
        id=event_id,
        type=EventType.PROVIDER_OPERATION_RESOLVED,
        session_id=request.session_id,
        interaction_id=model_event.interaction_id,
        agent_name=model_event.agent_name,
        environment_name=model_event.environment_name,
        timestamp=resolved_at,
        payload=event_payload,
    )
    resolution_event = event_with_execution_profile_fingerprint_authority(
        resolution_event,
        profile_fingerprint,
    )
    resolution_event = event_with_runtime_payload_authority(
        resolution_event,
        "model_attempt_id",
        "model_step_id",
        "resolution_id",
        "stage_id",
        *(field for field in ("operation_id", "stream_protocol") if field in event_payload),
    )
    resolution_event = event_with_runtime_generated_id(resolution_event)
    record_payload = {
        "record_type": _PROVIDER_OPERATION_RESOLUTION_RECORD_TYPE,
        "schema_version": 1,
        "resolution_id": resolution_id,
        "session_id": request.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "preparation_digest": stage.preparation_digest,
        "model_attempt_id": model_attempt_id,
        "source_run_epoch": stage.source_run_epoch,
        "resolved_run_epoch": request.expected_run_epoch,
        "action": request.action.value,
        "recovery_reason": inspection.recovery_reason.value,
        "duplicate_request_risk": inspection.duplicate_request_risk,
        "reason": request.reason,
        "metadata": request.metadata,
        "resolved_by": resolution_actor_payload(request.resolved_by),
        "resolved_at": resolved_at.isoformat().replace("+00:00", "Z"),
        "event_id": event_id,
        "request_digest": request_digest,
        "execution_profile_fingerprint": profile_fingerprint,
    }
    record_payload["record_digest"] = _resolution_record_digest(record_payload)
    record = ProviderOperationResolutionRecord.model_validate(record_payload)
    pending_disposition = ProviderOperationPendingDisposition(
        session_id=request.session_id,
        stage_id=stage.stage_id,
        resolution_id=resolution_id,
        request_digest=request_digest,
        action=request.action,
        resolved_run_epoch=request.expected_run_epoch,
        logical_step_id=stage.logical_step_id,
        source_step=step,
        source_dispatch_ordinal=stage.dispatch_ordinal,
        target_dispatch_ordinal=(
            stage.dispatch_ordinal + 1
            if request.action is ProviderOperationResolutionAction.FALLBACK_RETRY
            else None
        ),
        execution_profile_fingerprint=profile_fingerprint,
    )
    storage_key = provider_operation_resolution_storage_key(stage.stage_id)

    def transform(_session, checkpoint, current_record):
        if current_record is not None:
            current = _parse_provider_operation_resolution_record(
                current_record,
                session_id=request.session_id,
                stage_id=request.stage_id,
            )
            if current.request_digest != request_digest:
                raise ProviderOperationResolutionConflict(
                    "Provider operation was already resolved by a conflicting request."
                )
            raise _ProviderOperationResolutionReplay
        updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
        current_pending = pending_provider_operation_disposition_from_checkpoint(updated)
        if current_pending is not None and current_pending != pending_disposition:
            raise ProviderOperationResolutionConflict(
                "Another provider-operation disposition is still pending."
            )
        if request.action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            raw_ordinals = updated.get(_PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY, {})
            if type(raw_ordinals) is not dict:
                raise ProviderOperationEvidenceError(
                    "Provider-operation fallback ordinal evidence is malformed."
                )
            ordinals = copy_durable_json_object(raw_ordinals, "fallback_dispatch_ordinals")
            current_ordinal = ordinals.get(stage.logical_step_id, 0)
            if current_ordinal != stage.dispatch_ordinal:
                raise ProviderOperationResolutionConflict(
                    "Provider-operation fallback advanced from a stale dispatch ordinal."
                )
            ordinals[stage.logical_step_id] = stage.dispatch_ordinal + 1
            updated[_PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY] = ordinals
        updated[_PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY] = (
            pending_disposition.model_dump(mode="json")
        )
        return SessionOperationPublication(
            checkpoint=updated,
            operation_records={storage_key: record.model_dump(mode="json")},
            model_completion_stage_release=ModelCompletionStageRelease(
                stage_id=stage.stage_id,
                preparation_digest=stage.preparation_digest,
            ),
        )

    if before_resolution is not None:
        await before_resolution()
    try:
        await session_store.publish_session_operation(
            request.session_id,
            idempotency_key=storage_key,
            operation_transform=transform,
            events=[resolution_event],
            expected_statuses={SessionStatus.INTERRUPTED},
            expected_run_epoch=request.expected_run_epoch,
            expected_transcript_cursor=stage.source_transcript_cursor,
        )
    except _ProviderOperationResolutionReplay:
        replayed = await load_provider_operation_resolution(
            session_store,
            request.session_id,
            request.stage_id,
        )
        if replayed is None:
            raise ProviderOperationResolutionConflict(
                "Provider-operation resolution replay lost its durable record."
            ) from None
        return replayed
    return ProviderOperationResolutionResult(
        record=record,
        event=resolution_event,
        replayed=False,
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
    active_stage = await session_store.load_active_model_completion_stage(session_id)
    resolution_stage_id: str | None = None
    resolution_run_epoch: int | None = None
    if active_stage is not None:
        active_attempt_id = active_stage.stage.intent.get("model_attempt_id")
        if (
            active_stage.stage.logical_step_id == current_identity[-2]
            and active_attempt_id == current_identity[-1]
        ):
            session = await session_store.load(session_id)
            if session is None:
                raise ProviderOperationEvidenceError(
                    "Provider-operation active stage has no owning session."
                )
            resolution_stage_id = active_stage.stage.stage_id
            resolution_run_epoch = session.run_epoch
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
    if (
        recovery_events
        and not operation_events
        and recovery_events[0].type
        not in {
            EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
            EventType.PROVIDER_OPERATION_RESOLVED,
        }
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation recovery evidence has no durable operation identity."
        )
    if cancellation_events and not operation_events:
        raise ProviderOperationEvidenceError(
            "Provider-operation cancellation evidence has no durable operation identity."
        )
    if not operation_events and not starting_events:
        return ProviderOperationInspection(status=ProviderOperationInspectionStatus.SYNCHRONOUS)
    parsed_starting: list[tuple[str, str, ProviderOperationStartIdempotencySupport]] = []
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
            start_idempotency_support = ProviderOperationStartIdempotencySupport(
                starting_event.payload.get(
                    "start_idempotency_support",
                    ProviderOperationStartIdempotencySupport.UNSUPPORTED.value,
                )
            )
        except (TypeError, ValueError):
            raise ProviderOperationEvidenceError(
                "Provider-operation starting evidence is malformed."
            ) from None
        parsed_starting.append((provider, start_id, start_idempotency_support))
    if parsed_starting and any(
        candidate != parsed_starting[0] for candidate in parsed_starting[1:]
    ):
        raise ProviderOperationEvidenceError(
            "Provider-operation starting evidence is contradictory for the latest model attempt."
        )
    if not operation_events:
        provider, _start_id, _start_support = parsed_starting[0]
        if recovery_events and recovery_events[0].type == EventType.PROVIDER_OPERATION_RESOLVED:
            latest_resolution = recovery_events[0]
            resolution_action, recovery_reason, resolution_id = (
                _parse_provider_operation_resolution_event(latest_resolution)
            )
            return ProviderOperationInspection(
                status=(
                    ProviderOperationInspectionStatus.FALLBACK_RETRY
                    if resolution_action is ProviderOperationResolutionAction.FALLBACK_RETRY
                    else ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
                ),
                provider=provider,
                recovery_reason=recovery_reason,
                duplicate_request_risk=True,
                resolution_action=resolution_action,
                resolution_id=resolution_id,
            )
        recovery_reason = ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION
        status = ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
        if recovery_events:
            latest_recovery = recovery_events[0]
            if latest_recovery.type is not EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED:
                raise ProviderOperationEvidenceError(
                    "Start-only provider-operation recovery evidence is malformed."
                )
            try:
                recovery_reason = ProviderOperationUnavailableReason(
                    latest_recovery.payload.get("recovery_reason")
                )
            except ValueError:
                raise ProviderOperationEvidenceError(
                    "Start-only provider-operation recovery reason is malformed."
                ) from None
            status = (
                ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
                if recovery_reason is ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION
                else ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
            )
        return ProviderOperationInspection(
            status=status,
            provider=provider,
            recovery_reason=recovery_reason,
            duplicate_request_risk=True,
            allowed_resolutions=(
                ProviderOperationResolutionAction.FALLBACK_RETRY,
                ProviderOperationResolutionAction.FAIL,
            ),
            stage_id=resolution_stage_id,
            run_epoch=resolution_run_epoch,
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
    if parsed_starting and parsed_starting[0][:2] != (provider, start_id):
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
    latest_recovery_type = recovery_events[0].type if recovery_events else None
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RESOLVED:
        latest_resolution = recovery_events[0]
        resolution_action, recovery_reason, resolution_id = (
            _parse_provider_operation_resolution_event(latest_resolution)
        )
        return ProviderOperationInspection(
            status=(
                ProviderOperationInspectionStatus.FALLBACK_RETRY
                if resolution_action is ProviderOperationResolutionAction.FALLBACK_RETRY
                else ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
            ),
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            recovery_reason=recovery_reason,
            duplicate_request_risk=bool(latest_resolution.payload.get("duplicate_request_risk")),
            resolution_action=resolution_action,
            resolution_id=resolution_id,
            cancellation_status=cancellation_status,
            accounting_status=accounting_status,
            reservation_count=reservation_count,
        )
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED:
        try:
            recovery_reason = ProviderOperationUnavailableReason(
                recovery_events[0].payload.get("recovery_reason")
            )
        except ValueError:
            raise ProviderOperationEvidenceError(
                "Provider-operation recovery evidence has an invalid reason."
            ) from None
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            recovery_reason=recovery_reason,
            duplicate_request_risk=provider_operation_duplicate_request_risk(recovery_reason),
            allowed_resolutions=(
                ProviderOperationResolutionAction.FALLBACK_RETRY,
                ProviderOperationResolutionAction.FAIL,
            ),
            stage_id=resolution_stage_id,
            run_epoch=resolution_run_epoch,
            cancellation_status=cancellation_status,
            accounting_status=accounting_status,
            reservation_count=reservation_count,
        )
    if terminal_seen or latest_recovery_type == EventType.PROVIDER_OPERATION_RECONCILED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            cancellation_status=cancellation_status,
            accounting_status=accounting_status,
            reservation_count=reservation_count,
        )
    terminal_progress_statuses: set[ProviderOperationStatus] = set()
    for event in owning_events:
        if event.type != EventType.MODEL_ERROR:
            continue
        if event.payload.get("provider_operation_progress") is None:
            continue
        envelope = _parse_progress_envelope(event)
        if (
            envelope.state_version != state.version
            or envelope.operation_id != state.operation_id
            or envelope.stream_protocol != state.stream_protocol
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation terminal progress changed operation identity."
            )
        terminal_status = envelope.stream_event.provider_operation_status
        if terminal_status is not None and terminal_status.terminal:
            terminal_progress_statuses.add(terminal_status)
    if len(terminal_progress_statuses) > 1:
        raise ProviderOperationEvidenceError(
            "Provider-operation terminal progress contains contradictory statuses."
        )
    if terminal_progress_statuses:
        [terminal_progress_status] = terminal_progress_statuses
        recovery_reason = provider_operation_unavailable_reason(terminal_progress_status)
        if recovery_reason is None:
            raise ProviderOperationEvidenceError(
                "Provider-operation terminal progress has no recovery reason."
            )
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            recovery_reason=recovery_reason,
            duplicate_request_risk=provider_operation_duplicate_request_risk(recovery_reason),
            allowed_resolutions=(
                ProviderOperationResolutionAction.FALLBACK_RETRY,
                ProviderOperationResolutionAction.FAIL,
            ),
            stage_id=resolution_stage_id,
            run_epoch=resolution_run_epoch,
            cancellation_status=cancellation_status,
            accounting_status=accounting_status,
            reservation_count=reservation_count,
        )
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RECONNECT_STARTED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.RECONNECT_IN_PROGRESS,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            cancellation_status=cancellation_status,
            accounting_status=accounting_status,
            reservation_count=reservation_count,
        )
    if latest_recovery_type == EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED:
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.RECONNECT_SCHEDULED,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            cancellation_status=cancellation_status,
            accounting_status=accounting_status,
            reservation_count=reservation_count,
        )
    if status.terminal:
        recovery_reason = provider_operation_unavailable_reason(status)
        if recovery_reason is None:
            raise ProviderOperationEvidenceError(
                "Provider operation has terminal status without completion evidence."
            )
        return ProviderOperationInspection(
            status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE,
            provider=provider,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            recovery_reason=recovery_reason,
            duplicate_request_risk=provider_operation_duplicate_request_risk(recovery_reason),
            allowed_resolutions=(
                ProviderOperationResolutionAction.FALLBACK_RETRY,
                ProviderOperationResolutionAction.FAIL,
            ),
            stage_id=resolution_stage_id,
            run_epoch=resolution_run_epoch,
            cancellation_status=cancellation_status,
            accounting_status=accounting_status,
            reservation_count=reservation_count,
        )
    return ProviderOperationInspection(
        status=ProviderOperationInspectionStatus.PROVIDER_OPERATION_IN_PROGRESS,
        provider=provider,
        operation_id=state.operation_id,
        stream_protocol=state.stream_protocol,
        cancellation_status=cancellation_status,
        accounting_status=accounting_status,
        reservation_count=reservation_count,
    )


__all__ = [
    "PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES",
    "ProviderOperationAccountingStatus",
    "ProviderOperationCancellationStatus",
    "ProviderOperationEvidenceError",
    "ProviderOperationInspection",
    "ProviderOperationInspectionStatus",
    "ProviderOperationPendingDisposition",
    "ProviderOperationRecoveryResult",
    "ProviderOperationRecoveryStatus",
    "ProviderOperationResolutionAction",
    "ProviderOperationResolutionConflict",
    "ProviderOperationResolutionRecord",
    "ProviderOperationResolutionRequest",
    "ProviderOperationResolutionResult",
    "ProviderOperationUnavailableReason",
    "RecoverableProviderOperation",
    "RecoverableProviderOperationStart",
    "checkpoint_with_provider_operation_disposition_execution_owner",
    "clear_pending_provider_operation_disposition",
    "copy_provider_operation_resolution_metadata",
    "copy_provider_operation_resolution_request",
    "inspect_provider_operation",
    "load_pending_provider_operation_disposition",
    "load_provider_operation_resolution",
    "load_recoverable_provider_operation",
    "load_recoverable_provider_operation_start",
    "pending_provider_operation_disposition_from_checkpoint",
    "prepare_provider_operation_resolution_request",
    "provider_operation_duplicate_request_risk",
    "provider_operation_resolution_request_digest",
    "provider_operation_resolution_storage_key",
    "provider_operation_unavailable_reason",
    "resolve_provider_operation_stage",
    "validate_provider_operation_resolution_outcome_event",
]
