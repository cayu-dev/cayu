"""Durable identity for the latest atomically published assistant model step."""

from __future__ import annotations

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
)

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
