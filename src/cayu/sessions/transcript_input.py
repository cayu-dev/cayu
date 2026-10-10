"""Deferred transcript input, durable payloads and initial-publication value rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.messages import Message, MessageRole, PeerContentPart, copy_message, detach_message


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


SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY = "input_contract"


@dataclass(frozen=True, slots=True)
class SessionInputContractEvidence:
    """Runtime-owned facts needed to identify replayable fresh input."""

    message_start_index: int
    message_count: int
    redactions_applied: bool
    structured_output_requested: bool
    messages_sha256: str


def parse_session_input_contract_evidence(value: object) -> SessionInputContractEvidence:
    """Parse one canonical versioned fresh-input contract marker."""

    if type(value) is not str:
        raise ValueError("input_contract must be a canonical string.")
    parts = value.split(":")
    if len(parts) != 7 or parts[0] != "v1" or parts[5] != "sha256":
        raise ValueError("input_contract must use the supported v1 format.")
    raw_start_index, raw_count, redaction_mode, output_mode, _, messages_sha256 = parts[1:]
    max_integer = str(MAX_DURABLE_JSON_INTEGER)
    parsed_integers: list[int] = []
    for raw_value, field_name in (
        (raw_start_index, "message start index"),
        (raw_count, "message count"),
    ):
        if (
            not raw_value
            or not raw_value.isascii()
            or not raw_value.isdecimal()
            or (len(raw_value) > 1 and raw_value.startswith("0"))
        ):
            raise ValueError(f"input_contract {field_name} must be canonical.")
        if len(raw_value) > len(max_integer) or (
            len(raw_value) == len(max_integer) and raw_value > max_integer
        ):
            raise ValueError(f"input_contract {field_name} exceeds the durable integer limit.")
        parsed_integers.append(int(raw_value))
    start_index, count = parsed_integers
    if redaction_mode not in {"original", "redacted"}:
        raise ValueError("input_contract redaction mode is unsupported.")
    if output_mode not in {"text", "structured"}:
        raise ValueError("input_contract output mode is unsupported.")
    if len(messages_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in messages_sha256
    ):
        raise ValueError("input_contract message digest must be lowercase SHA-256.")
    return SessionInputContractEvidence(
        message_start_index=start_index,
        message_count=count,
        redactions_applied=redaction_mode == "redacted",
        structured_output_requested=output_mode == "structured",
        messages_sha256=messages_sha256,
    )


def session_input_messages_sha256(messages: Sequence[Message]) -> str:
    """Hash the exact canonical messages crossing the fresh-run input boundary."""

    if type(messages) not in {list, tuple}:
        raise TypeError("messages must be a list or tuple.")
    document: list[dict[str, Any]] = []
    for message in messages:
        if type(message) is not Message:
            raise TypeError("messages must contain exact Message instances.")
        document.append(message.model_dump(mode="json"))
    return sha256(canonical_durable_json_bytes(document, "messages")).hexdigest()


def system_prompt_messages_sha256(messages: Sequence[Message]) -> str:
    """Hash only the canonical system messages that define one prompt anatomy."""

    if type(messages) not in {list, tuple}:
        raise TypeError("messages must be a list or tuple.")
    system_messages: list[dict[str, Any]] = []
    for message in messages:
        if type(message) is not Message:
            raise TypeError("messages must contain exact Message instances.")
        if message.role == MessageRole.SYSTEM:
            system_messages.append(message.model_dump(mode="json"))
    return sha256(
        canonical_durable_json_bytes(system_messages, "system_prompt_messages")
    ).hexdigest()


def session_messages_input_contract_evidence(
    messages: Sequence[Message],
    *,
    message_start_index: int,
    redactions_applied: bool,
    structured_output_requested: bool,
) -> str:
    """Bind one runtime-owned input batch to its exact transcript position."""

    if type(messages) not in {list, tuple}:
        raise TypeError("messages must be a list or tuple.")
    if type(message_start_index) is not int:
        raise TypeError("message_start_index must be an integer.")
    if not 0 <= message_start_index <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("message_start_index exceeds the durable integer limit.")
    if len(messages) > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("messages exceeds the durable message-count limit.")
    if type(redactions_applied) is not bool:
        raise TypeError("redactions_applied must be a bool.")
    if type(structured_output_requested) is not bool:
        raise TypeError("structured_output_requested must be a bool.")
    redaction_mode = "redacted" if redactions_applied else "original"
    output_mode = "structured" if structured_output_requested else "text"
    messages_sha256 = session_input_messages_sha256(messages)
    return (
        f"v1:{message_start_index}:{len(messages)}:{redaction_mode}:"
        f"{output_mode}:sha256:{messages_sha256}"
    )


def _copy_caller_input_message(message: Message) -> Message:
    copied = detach_message(message)
    if any(isinstance(part, PeerContentPart) for part in copied.content):
        raise ValueError("Peer content requires authenticated peer delivery.")
    return copied
