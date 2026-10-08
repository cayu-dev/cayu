"""Bounded private tool-argument records and session-store access rules.

Runtime owns original-argument capture, redaction and model-only restoration.
The native publication transaction owns writes and exact replay settlement.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu.messages import Message, ToolCallPart

if TYPE_CHECKING:
    from cayu.sessions.records import Session

STORAGE_KEY = "cayu:private-argument-continuity"
MAX_ROUNDS = 16
MAX_ROUND_BYTES = 8192
MAX_CALL_BYTES = 4096
MAX_RECORD_BYTES = 16384
_reading: ContextVar[bool] = ContextVar("private_argument_continuity_read", default=False)


@contextmanager
def private_read_scope() -> Iterator[None]:
    token = _reading.set(True)
    try:
        yield
    finally:
        _reading.reset(token)


def require_private_key_access(key: str, *, read: bool) -> None:
    if key.startswith(STORAGE_KEY) and not (read and _reading.get()):
        raise ValueError("Private argument continuity is runtime-owned.")


class ArgumentContinuity(BaseModel):
    """Sealed original arguments retained with one tool-round publication."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    # A private nonce prevents public request digests becoming dictionary oracles.
    nonce: str = Field(min_length=32, max_length=32, repr=False, pattern=r"^[0-9a-f]{32}$")
    profile: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    scope: str = Field(default="0" * 64, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    arguments: dict[str, Any] = Field(repr=False)

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "private arguments")
        if not copied or len(copied) > 32 or any(type(v) is not dict for v in copied.values()):
            raise ValueError("Private argument batch is malformed.")
        if any(
            len(canonical_durable_json_bytes(v, "private arguments")) > MAX_CALL_BYTES
            for v in copied.values()
        ):
            raise ValueError("Private call arguments exceed their bound.")
        if len(canonical_durable_json_bytes(copied, "private arguments")) > MAX_ROUND_BYTES:
            raise ValueError("Private argument batch exceeds its bound.")
        return copied


def append_record(
    current: dict[str, Any] | None,
    *,
    continuity: ArgumentContinuity,
    session: Session,
    messages: Iterable[Message],
    request_digest: str,
) -> dict[str, Any]:
    """Build bounded private state before any native transaction mutation."""
    calls = [
        part
        for message in messages
        for part in message.content
        if isinstance(part, ToolCallPart) and part.tool_call_id in continuity.arguments
    ]
    if len(calls) != len(continuity.arguments) or any(part.arguments for part in calls):
        raise ValueError("Private arguments require exact omitted transcript calls.")
    if len({part.tool_call_id for part in calls}) != len(calls):
        raise ValueError("Private argument transcript calls must be distinct.")
    if any(not part.tool_round_id or not part.model_attempt_id for part in calls):
        raise ValueError("Private argument transcript calls require runtime identity.")
    if len({part.tool_round_id for part in calls}) != 1:
        raise ValueError("Private argument batch must belong to one published round.")
    if len(request_digest) != 64 or any(c not in "0123456789abcdef" for c in request_digest):
        raise ValueError("Private argument batch requires its publication digest.")
    records = [] if current is None else [record for record, _ in validate_records(current)]
    record = {
        "instance": session.instance_id,
        "session_id": session.id,
        "publication_id": f"tool-round:{calls[0].tool_round_id}",
        "request_digest": request_digest,
        "continuity": continuity.model_dump(mode="json"),
        "calls": [part.model_dump(mode="json") for part in calls],
    }
    record["digest"] = sha256(canonical_durable_json_bytes(record, "private record")).hexdigest()
    if len(canonical_durable_json_bytes(record, "private record")) > MAX_RECORD_BYTES:
        return {"records": records}
    return {"records": [*records[-(MAX_ROUNDS - 1) :], record]}


def validate_records(raw: dict[str, Any]) -> list[tuple[dict[str, Any], ArgumentContinuity]]:
    if type(raw) is not dict or set(raw) != {"records"}:
        raise ValueError("Private argument continuity state is malformed.")
    records = raw["records"]
    if type(records) is not list or len(records) > MAX_ROUNDS:
        raise ValueError("Private argument continuity state exceeds its bound.")
    validated = []
    for record in records:
        if type(record) is not dict or set(record) != {
            "instance",
            "session_id",
            "publication_id",
            "request_digest",
            "continuity",
            "calls",
            "digest",
        }:
            raise ValueError("Private argument continuity record is malformed.")
        if len(canonical_durable_json_bytes(record, "private record")) > MAX_RECORD_BYTES:
            raise ValueError("Private argument continuity record exceeds its bound.")
        material = {key: value for key, value in record.items() if key != "digest"}
        if (
            sha256(canonical_durable_json_bytes(material, "private record")).hexdigest()
            != record["digest"]
        ):
            raise ValueError("Private argument continuity record failed integrity validation.")
        value = ArgumentContinuity.model_validate(record["continuity"])
        if type(record["calls"]) is not list or len(record["calls"]) != len(value.arguments):
            raise ValueError("Private argument continuity record lost its call association.")
        parts = [ToolCallPart.model_validate(call) for call in record["calls"]]
        if {part.tool_call_id for part in parts} != set(value.arguments):
            raise ValueError("Private argument continuity record has conflicting call identities.")
        if any(record["publication_id"] != f"tool-round:{part.tool_round_id}" for part in parts):
            raise ValueError("Private argument continuity record lost publication linkage.")
        validated.append((record, value))
    return validated
