"""Durable model-completion stage records, outcomes and recovery authority."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, copy_event
from cayu.sessions import authority as session_authority_rules
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions._durable_operation_ownership import DurableOperationOwnership
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition,
    ModelCompletionStageSettlementRequest,
)
from cayu.sessions._operation_records import MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.records import Session, SessionStatus, copy_session


class SessionModelCompletionStageConflict(SessionRuntimePublicationConflict):
    """A logical model-completion stage conflicts with immutable durable evidence."""


class SessionModelCompletionDispatchAlreadyAuthorized(SessionModelCompletionStageConflict):
    """A provider dispatch won admission before terminal-decision election."""


class SessionModelCompletionStageIncomplete(RuntimeError):
    """A logical model step was prepared but has no durable terminal completion."""


class ModelCompletionStageSettlement(BaseModel):
    """Append-only proof that one exact prepared model effect is no longer active."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.model-completion-stage.settled"] = (
        "cayu.model-completion-stage.settled"
    )
    schema_version: Literal[1] = 1
    session_id: str
    stage_id: str
    logical_step_id: str
    model_attempt_id: str
    interaction_id: str
    request_fingerprint: str
    provider_effect_id: str
    execution_profile_fingerprint: str | None = None
    preparation_digest: str
    settlement_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    settled_reservation_ids: tuple[str, ...]
    disposition: ModelCompletionStageDisposition
    reason_code: str
    superseding_stage_id: str | None = None
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    settled_at: datetime
    record_digest: str

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")

    @field_validator("settled_at")
    @classmethod
    def normalize_settled_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("settled_at must include a timezone.")
        return value.astimezone(UTC)

    @field_validator("record_digest")
    @classmethod
    def validate_record_digest(cls, value: str) -> str:
        session_authority_rules._require_raw_sha256_digest(value)
        return value

    @model_validator(mode="after")
    def validate_request_shape(self) -> ModelCompletionStageSettlement:
        ModelCompletionStageSettlementRequest.model_validate(
            self.model_dump(
                mode="python",
                include=set(ModelCompletionStageSettlementRequest.model_fields),
            )
        )
        return self


class ModelCompletionManualRecoveryRequest(BaseModel):
    """Runtime-owned terminal disposition for one ambiguous model dispatch."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    stage_id: str = Field(max_length=256)
    expected_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    terminal_status: SessionStatus
    terminalization_only: StrictBool = False
    expected_session_instance_id: str | None = None
    inactive_for_seconds: StrictInt = Field(default=30, ge=0)

    @model_validator(mode="after")
    def validate_terminalization_identity(self) -> ModelCompletionManualRecoveryRequest:
        if self.terminalization_only and not self.expected_session_instance_id:
            raise ValueError(
                "Terminalization-only recovery requires the exact session incarnation."
            )
        return self

    @field_validator("session_id", "stage_id")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("terminal_status")
    @classmethod
    def validate_terminal_status(cls, value: SessionStatus) -> SessionStatus:
        if value not in {SessionStatus.FAILED, SessionStatus.INTERRUPTED}:
            raise ValueError("terminal_status must be failed or interrupted.")
        return value


def copy_model_completion_manual_recovery_request(
    request: ModelCompletionManualRecoveryRequest,
    *,
    session_id: str | None = None,
) -> ModelCompletionManualRecoveryRequest:
    """Detach one public manual model-recovery request."""

    if type(request) is not ModelCompletionManualRecoveryRequest:
        raise TypeError("request must be a ModelCompletionManualRecoveryRequest.")
    return ModelCompletionManualRecoveryRequest(
        session_id=request.session_id if session_id is None else session_id,
        stage_id=request.stage_id,
        expected_run_epoch=request.expected_run_epoch,
        terminal_status=request.terminal_status,
        terminalization_only=request.terminalization_only,
        expected_session_instance_id=request.expected_session_instance_id,
        inactive_for_seconds=request.inactive_for_seconds,
    )


class ModelCompletionManualRecoveryResult(BaseModel):
    """Verified accounting and stage settlement from manual model recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: Session
    settlement: ModelCompletionStageSettlement
    budget_events: tuple[Event, ...] = ()
    replayed: StrictBool = False

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("settlement")
    @classmethod
    def copy_settlement(
        cls,
        value: ModelCompletionStageSettlement,
    ) -> ModelCompletionStageSettlement:
        if type(value) is not ModelCompletionStageSettlement:
            raise TypeError("settlement must be a ModelCompletionStageSettlement.")
        return value.model_copy(deep=True)

    @field_validator("budget_events", mode="before")
    @classmethod
    def copy_budget_events(cls, value) -> tuple[Event, ...]:
        return tuple(copy_event(event) for event in value)


MODEL_COMPLETION_STAGE_RECORD_TYPE = "cayu.model-completion-stage"


MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE = "cayu.model-completion-stage.prepared"


MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE = "cayu.model-completion-stage.completed"


MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE = "cayu.model-completion-stage.active"


MODEL_COMPLETION_WINNER_RECORD_TYPE = "cayu.model-completion-stage.winner"


MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE = "cayu.model-completion-stage.abandoned"


MODEL_COMPLETION_STAGE_SETTLEMENT_RECORD_TYPE = "cayu.model-completion-stage.settled"


MODEL_COMPLETION_STAGE_DISPATCH_RECORD_TYPE = "cayu.model-completion-stage.dispatched"


MODEL_COMPLETION_STAGE_SCHEMA_VERSION = 1


MODEL_COMPLETION_STAGE_DISPATCH_OPERATION_KEY_PREFIX = (
    MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "dispatched:"
)


MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY = MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "active"


MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES = 2 * 1024 * 1024


MODEL_COMPLETION_PROVIDER_START_INTENT_MAX_BYTES = 16 * 1024


ModelCompletionPurpose = Literal["assistant-turn", "context-compaction", "auxiliary-inference"]


class ModelCompletionStageRequest(BaseModel):
    """Immutable identity and accounting link prepared before provider dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose = "assistant-turn"
    intent: dict[str, Any]
    reservation_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("stage_id", "logical_step_id")
    @classmethod
    def validate_stage_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "intent")
        recovery_context = copied.get("recovery_context")
        if recovery_context is not None:
            if type(recovery_context) is not dict:
                raise ValueError("intent.recovery_context must be an object.")
            if (
                len(
                    canonical_durable_json_bytes(
                        recovery_context,
                        "intent.recovery_context",
                    )
                )
                > MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES
            ):
                raise ValueError(
                    "intent.recovery_context exceeds the durable byte limit of "
                    f"{MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES}."
                )
        provider_start = copied.get("provider_operation_start")
        if provider_start is not None:
            if type(provider_start) is not dict:
                raise ValueError("intent.provider_operation_start must be an object.")
            if (
                len(
                    canonical_durable_json_bytes(
                        provider_start,
                        "intent.provider_operation_start",
                    )
                )
                > MODEL_COMPLETION_PROVIDER_START_INTENT_MAX_BYTES
            ):
                raise ValueError(
                    "intent.provider_operation_start exceeds the durable byte limit of "
                    f"{MODEL_COMPLETION_PROVIDER_START_INTENT_MAX_BYTES}."
                )
        return copied

    @field_validator("reservation_ids", mode="before")
    @classmethod
    def copy_reservation_ids(cls, value) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("reservation_ids must be a list or tuple of reservation ids.")
        copied = tuple(
            require_clean_nonblank(reservation_id, "reservation_ids item")
            for reservation_id in value
        )
        if len(copied) > 32:
            raise ValueError("reservation_ids cannot contain more than 32 ids.")
        if len(set(copied)) != len(copied):
            raise ValueError("reservation_ids must not contain duplicate ids.")
        return copied


class ModelCompletionStage(BaseModel):
    """Verified view assembled from append-only preparation and completion records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage"] = MODEL_COMPLETION_STAGE_RECORD_TYPE
    schema_version: Literal[1] = MODEL_COMPLETION_STAGE_SCHEMA_VERSION
    session_id: str
    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    state: Literal["in_flight", "completed"]
    intent: dict[str, Any]
    reservation_ids: tuple[str, ...]
    preparation_request_digest: str
    preparation_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    prepared_at: datetime
    publication: RuntimePublicationRequest | None = None
    publication_material_digest: str | None = None
    completion_digest: str | None = None
    completed_at: datetime | None = None

    @field_validator("session_id", "stage_id", "logical_step_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "intent")

    @field_validator("reservation_ids", mode="before")
    @classmethod
    def copy_reservation_ids(cls, value) -> tuple[str, ...]:
        return ModelCompletionStageRequest.copy_reservation_ids(value)

    @field_validator(
        "preparation_request_digest",
        "preparation_digest",
        "publication_material_digest",
        "completion_digest",
    )
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is None and info.field_name in {
            "publication_material_digest",
            "completion_digest",
        }:
            return None
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("prepared_at", "completed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must include a timezone.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_state(self) -> ModelCompletionStage:
        terminal_fields = (
            self.publication,
            self.publication_material_digest,
            self.completion_digest,
            self.completed_at,
        )
        if self.state == "in_flight" and any(value is not None for value in terminal_fields):
            raise ValueError("An in-flight model-completion stage cannot contain terminal data.")
        if self.state == "completed" and any(value is None for value in terminal_fields):
            raise ValueError("A completed model-completion stage requires all terminal data.")
        if self.completed_at is not None and self.completed_at < self.prepared_at:
            raise ValueError("completed_at cannot precede prepared_at.")
        return self


class ModelCompletionStageDispatch(BaseModel):
    """Append-only proof that one exact prepared request may reach its provider."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.model-completion-stage.dispatched"] = (
        MODEL_COMPLETION_STAGE_DISPATCH_RECORD_TYPE
    )
    schema_version: Literal[1] = MODEL_COMPLETION_STAGE_SCHEMA_VERSION
    session_id: str
    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    model_attempt_id: str = Field(max_length=256)
    interaction_id: str = Field(max_length=256)
    request_fingerprint: str
    provider_effect_id: str = Field(max_length=512)
    execution_profile_fingerprint: str | None = None
    preparation_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    dispatched_at: datetime
    record_digest: str

    @field_validator(
        "session_id",
        "stage_id",
        "logical_step_id",
        "model_attempt_id",
        "interaction_id",
        "provider_effect_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "request_fingerprint",
        "execution_profile_fingerprint",
        "preparation_digest",
        "record_digest",
    )
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is None:
            if info.field_name == "execution_profile_fingerprint":
                return None
            raise ValueError(f"{info.field_name} is required.")
        try:
            session_authority_rules._require_raw_sha256_digest(value)
        except ValueError as exc:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.") from exc
        return value

    @field_validator("dispatched_at")
    @classmethod
    def normalize_dispatched_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("dispatched_at must include a timezone.")
        return value.astimezone(UTC)


class ModelCompletionStageResult(BaseModel):
    """Result of preparing or terminally completing one immutable stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ModelCompletionStage
    replayed: StrictBool
    dispatch_authorized: StrictBool
    prepared_events: tuple[Event, ...] = Field(
        default_factory=tuple, exclude_if=lambda events: not events
    )

    @field_validator("prepared_events", mode="before")
    @classmethod
    def copy_prepared_events(cls, value):
        if type(value) not in (list, tuple) or len(value) > 1:
            raise ValueError("Stage preparation may publish at most one selection event.")
        return tuple(copy_event(event) for event in value)

    @model_validator(mode="after")
    def validate_dispatch_authorization(self) -> ModelCompletionStageResult:
        expected = self.stage.state == "in_flight" and not self.replayed
        if self.dispatch_authorized != expected:
            raise ValueError(
                "dispatch_authorized must be true exactly for a newly inserted "
                "in-flight preparation."
            )
        if self.prepared_events and not expected:
            raise ValueError("Only a newly prepared stage returns newly published events.")
        return self


class ActiveModelCompletionStage(BaseModel):
    """Verified snapshot of the model dispatch currently eligible to publish."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ModelCompletionStage
    activated_at: datetime
    marker_digest: str

    @field_validator("activated_at")
    @classmethod
    def normalize_activated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("activated_at must include a timezone.")
        return value.astimezone(UTC)

    @field_validator("marker_digest")
    @classmethod
    def validate_marker_digest(cls, value: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("marker_digest must be a lowercase SHA-256 digest.")
        return value


class ModelCompletionStageAbandonment(BaseModel):
    """Durable proof that an exact pre-dispatch preparation was abandoned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.abandoned"] = (
        MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE
    )
    schema_version: Literal[1] = MODEL_COMPLETION_STAGE_SCHEMA_VERSION
    session_id: str
    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_request_digest: str
    preparation_digest: str
    active_marker_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    abandoned_at: datetime
    abandonment_digest: str

    @field_validator("session_id", "stage_id", "logical_step_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "preparation_request_digest",
        "preparation_digest",
        "active_marker_digest",
        "abandonment_digest",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("abandoned_at")
    @classmethod
    def normalize_abandoned_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("abandoned_at must include a timezone.")
        return value.astimezone(UTC)


class ModelCompletionStageAbandonmentResult(BaseModel):
    """Result of a new abandonment or an exact durable replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    abandonment: ModelCompletionStageAbandonment
    replayed: StrictBool


class _ModelCompletionStagePreparationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.prepared"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    intent: dict[str, Any]
    reservation_ids: tuple[str, ...]
    request_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    prepared_at: datetime
    record_digest: str


class _ModelCompletionStageTerminalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.completed"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_digest: str
    publication: RuntimePublicationRequest
    publication_material_digest: str
    completed_at: datetime
    record_digest: str


class ModelCompletionStageRecoveryFence(BaseModel):
    """Expected live owner for inserting recovery-generated terminal material.

    This is an admission fence, not evidence that a provider completed. Exact
    terminal replay is checked before this transient lease is required again.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    expected_session_instance_id: str
    expected_run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    expected_active_profile: ActiveInvocationExecutionProfile
    recovery_claim_id: str
    preparation_digest: str
    recovery_plan_ownership: DurableOperationOwnership | None = None

    @field_validator("expected_session_instance_id", "recovery_claim_id", "preparation_digest")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        value = require_durable_clean_nonblank(value, info.field_name)
        if info.field_name == "preparation_digest" and (
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("preparation_digest must be a lowercase SHA-256 digest.")
        return value

    @field_validator("expected_active_profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> ActiveInvocationExecutionProfile:
        if type(value) is ActiveInvocationExecutionProfile:
            value = value.model_dump(mode="python")
        return ActiveInvocationExecutionProfile.model_validate(value)


class _ActiveModelCompletionStageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.active"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    activated_at: datetime
    record_digest: str


class _ModelCompletionStageWinnerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.winner"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_digest: str
    completion_digest: str
    publication_material_digest: str
    receipt_request_digest: str
    publication_digest: str
    published_at: datetime
    record_digest: str


class _ModelCompletionStageAbandonmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.abandoned"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_request_digest: str
    preparation_digest: str
    active_marker_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    abandoned_at: datetime
    record_digest: str
