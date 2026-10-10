"""Durable model-step publication identity and exact stage-settlement contracts."""

from __future__ import annotations

from enum import StrEnum
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
    copy_durable_json_object,
    copy_durable_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.sessions import authority as session_authority_rules

LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY = "last_model_step_publication"
MODEL_STEP_PUBLICATION_CHECKPOINT_RECORD_TYPE = "cayu.model-step-publication"
MODEL_STEP_PUBLICATION_CHECKPOINT_SCHEMA_VERSION = 2
_NON_TURN_CLASSIFICATIONS = frozenset({"failed", "filtered", "invalid", "length"})
_MESSAGELESS_CLASSIFICATIONS = _NON_TURN_CLASSIFICATIONS | {"continue"}
_CLASSIFICATIONS = frozenset(
    {
        "continue",
        "failed",
        "filtered",
        "final",
        "invalid",
        "length",
        "think_only",
    }
)


class ModelStepPublicationCheckpoint(BaseModel):
    """Bounded pointer from session state to an immutable publication receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-step-publication"] = (
        MODEL_STEP_PUBLICATION_CHECKPOINT_RECORD_TYPE
    )
    schema_version: Literal[2] = MODEL_STEP_PUBLICATION_CHECKPOINT_SCHEMA_VERSION
    logical_step_id: str = Field(max_length=256)
    stage_id: str = Field(max_length=256)
    source_transcript_cursor: StrictInt = Field(
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    transcript_end_cursor: StrictInt = Field(
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    completion_event_id: str
    classification: dict[str, Any]
    assistant_message_published: StrictBool
    assistant_message_deferred: StrictBool = False
    tool_round_id: str | None = None

    @field_validator(
        "logical_step_id",
        "stage_id",
        "completion_event_id",
        "tool_round_id",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("classification", mode="before")
    @classmethod
    def copy_classification(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "classification")
        classification_type = copied.get("type")
        if type(classification_type) is not str:
            raise ValueError("classification.type must be a string.")
        require_clean_nonblank(classification_type, "classification.type")
        if classification_type not in _CLASSIFICATIONS:
            raise ValueError("classification.type is not a supported model-step outcome.")
        return copied

    @model_validator(mode="after")
    def validate_transcript_boundary(self) -> ModelStepPublicationCheckpoint:
        expected_end = self.source_transcript_cursor + int(self.assistant_message_published)
        if self.transcript_end_cursor != expected_end:
            raise ValueError(
                "transcript_end_cursor must reflect the authoritative assistant message."
            )
        if (
            not self.assistant_message_published
            and not self.assistant_message_deferred
            and self.classification.get("type") not in _MESSAGELESS_CLASSIFICATIONS
        ):
            raise ValueError(
                "A model-step pointer without an assistant message requires a "
                "non-turn or continue classification."
            )
        if self.assistant_message_published and self.assistant_message_deferred:
            raise ValueError(
                "An assistant message cannot be published and deferred simultaneously."
            )
        if self.assistant_message_deferred and self.tool_round_id is None:
            raise ValueError("A deferred assistant message requires a pending tool round.")
        if self.tool_round_id is not None and not (
            self.assistant_message_published or self.assistant_message_deferred
        ):
            raise ValueError(
                "A model-step pointer cannot own a tool round without an assistant message."
            )
        return self


def model_step_publication_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> ModelStepPublicationCheckpoint | None:
    if checkpoint is None:
        return None
    raw_pointer = checkpoint.get(LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY)
    if raw_pointer is None:
        return None
    copied = copy_durable_json_value(
        raw_pointer,
        LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    )
    if type(copied) is not dict:
        raise ValueError("Last model-step publication checkpoint must be an object.")
    return ModelStepPublicationCheckpoint.model_validate(copied)


class ModelCompletionStageDisposition(StrEnum):
    """Durable non-success outcome of one exact model-completion dispatch."""

    FAILED_BEFORE_PROVIDER_EFFECT = "failed_before_provider_effect"
    PROVIDER_EFFECT_OUTCOME_UNKNOWN = "provider_effect_outcome_unknown"
    SUPERSEDED = "superseded"


class ModelCompletionStageSettlementRequest(BaseModel):
    """Exact authority required to terminally settle one active model stage."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    model_attempt_id: str = Field(max_length=256)
    interaction_id: str = Field(max_length=256)
    request_fingerprint: str
    provider_effect_id: str = Field(max_length=512)
    execution_profile_fingerprint: str | None = None
    preparation_digest: str
    settlement_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    settled_reservation_ids: tuple[str, ...]
    disposition: ModelCompletionStageDisposition
    reason_code: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9_.-]+$")
    superseding_stage_id: str | None = Field(default=None, max_length=256)

    @field_validator(
        "stage_id",
        "logical_step_id",
        "model_attempt_id",
        "interaction_id",
        "provider_effect_id",
        "superseding_stage_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "request_fingerprint",
        "execution_profile_fingerprint",
        "preparation_digest",
    )
    @classmethod
    def validate_digests(cls, value: str | None, info) -> str | None:
        if value is None:
            if info.field_name == "execution_profile_fingerprint":
                return None
            raise ValueError(f"{info.field_name} is required.")
        try:
            session_authority_rules._require_raw_sha256_digest(value)
        except ValueError as exc:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.") from exc
        return value

    @field_validator("settled_reservation_ids")
    @classmethod
    def validate_settled_reservation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if type(value) is not tuple:
            raise TypeError("settled_reservation_ids must be a tuple.")
        copied = tuple(
            require_durable_clean_nonblank(reservation_id, "settled_reservation_ids")
            for reservation_id in value
        )
        if len(set(copied)) != len(copied):
            raise ValueError("settled_reservation_ids must be distinct.")
        return copied

    @model_validator(mode="after")
    def validate_supersession(self) -> ModelCompletionStageSettlementRequest:
        if self.disposition is ModelCompletionStageDisposition.SUPERSEDED:
            if self.superseding_stage_id is None:
                raise ValueError("A superseded stage requires superseding_stage_id.")
            if self.superseding_stage_id == self.stage_id:
                raise ValueError("A model-completion stage cannot supersede itself.")
        elif self.superseding_stage_id is not None:
            raise ValueError("superseding_stage_id is valid only for a superseded stage.")
        return self
