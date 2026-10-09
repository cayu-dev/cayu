"""Deferred transcript input, durable payloads and initial-publication value rules."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.messages import Message, copy_message, detach_message


class DeferredInteractionInput(BaseModel):
    """Source messages durably admitted but not yet visible in the transcript."""

    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    source_messages: list[Message]
    initial_transcript_messages: list[Message] | None = None

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "interaction_id")

    @field_validator("source_messages")
    @classmethod
    def copy_source_messages(cls, value: list[Message]) -> list[Message]:
        return [copy_message(message) for message in value]

    @field_validator("initial_transcript_messages")
    @classmethod
    def copy_initial_transcript_messages(
        cls,
        value: list[Message] | None,
    ) -> list[Message] | None:
        if value is None:
            return None
        return [copy_message(message) for message in value]

    @model_validator(mode="after")
    def validate_initial_transcript_projection(self) -> DeferredInteractionInput:
        initial = self.initial_transcript_messages
        source = self.source_messages
        if initial is not None and (
            len(initial) < len(source) or (source and initial[-len(source) :] != source)
        ):
            raise ValueError(
                "Initial transcript projection must preserve the deferred source suffix."
            )
        return self


def deferred_interaction_input_storage_payload(
    value: DeferredInteractionInput,
) -> dict[str, Any]:
    if type(value) is not DeferredInteractionInput:
        raise TypeError("Deferred interaction storage requires DeferredInteractionInput.")
    stable = DeferredInteractionInput.model_validate(
        value.model_dump(mode="python", warnings=False)
    )
    payload = stable.model_dump(mode="json", warnings=False)
    payload.pop("interaction_id")
    return payload


def deferred_interaction_input_from_storage_payload(
    interaction_id: str,
    payload: object,
) -> DeferredInteractionInput:
    if type(payload) is not dict or set(payload) != {
        "source_messages",
        "initial_transcript_messages",
    }:
        raise ValueError("Deferred interaction input has invalid durable payload.")
    stable_payload = cast("dict[str, object]", payload)
    return DeferredInteractionInput.model_validate(
        {
            "interaction_id": interaction_id,
            "source_messages": stable_payload["source_messages"],
            "initial_transcript_messages": stable_payload["initial_transcript_messages"],
        }
    )


def require_deferred_initial_transcript_replacement(
    deferred: DeferredInteractionInput,
    *,
    expected_messages: list[Message],
    replacement_messages: list[Message],
) -> None:
    """Require source identity and any retained complete transcript authority."""

    if deferred.source_messages != expected_messages:
        raise RuntimeError("Deferred interaction input changed before finalization.")
    authenticated = deferred.initial_transcript_messages
    if authenticated is not None and authenticated != replacement_messages:
        raise RuntimeError(
            "Initial transcript replacement conflicts with its authenticated projection."
        )


def copy_transcript_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    return [detach_message(message) for message in messages]


def _initial_transcript_prefix_count(
    expected: list[Message],
    replacement: list[Message],
    *,
    runtime_suffix_count: int,
) -> int:
    """Validate the admitted source segment and return its bootstrap offset."""

    if type(runtime_suffix_count) is not int:
        raise TypeError("runtime_suffix_count must be an int.")
    if runtime_suffix_count < 0 or runtime_suffix_count > len(replacement):
        raise ValueError("runtime_suffix_count is outside the replacement transcript.")
    prefix_count = len(replacement) - len(expected) - runtime_suffix_count
    if prefix_count < 0 or replacement[prefix_count : prefix_count + len(expected)] != expected:
        raise RuntimeError(
            "Initial transcript must preserve the admitted source before its runtime suffix."
        )
    return prefix_count
