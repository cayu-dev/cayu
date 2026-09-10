"""Bounded, runtime-private continuity for omitted model-authored arguments.

This is a best-effort cache of original inputs, not knowledge read permission or
a second transcript. Public transcript material remains the authority for which
calls survive context selection. The native publication transaction owns writes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu.core.messages import Message, ProviderStatePart, ToolCallPart
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime._runtime_records import ToolCallRequest
    from cayu.runtime.sessions import Session, SessionStore
    from cayu.storage.memory import KnowledgeAccessScope

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


def scope_digest(scope: KnowledgeAccessScope | None) -> str:
    if scope is None:
        return "0" * 64
    from cayu.storage.memory import KnowledgeAccessScope

    if type(scope) is not KnowledgeAccessScope:
        raise TypeError("Argument continuity requires a native knowledge access scope.")
    return sha256(canonical_durable_json_bytes(scope.model_dump(mode="json"), "scope")).hexdigest()


def capture_arguments(
    tool_calls: Iterable[ToolCallRequest],
    *,
    names: frozenset[str],
    profile: str | None,
    redactor: SecretRedactor,
    scope: KnowledgeAccessScope | None = None,
) -> ArgumentContinuity | None:
    if not names or profile is None:
        return None
    retained: dict[str, Any] = {}
    for call in tool_calls:
        if call.name not in names:
            continue
        # Targeted gateways have a distinct outer model envelope and grant
        # authority. Never substitute their resolved inner execution arguments
        # for that model-authored envelope.
        if (
            call.model_tool_name is not None
            or call.targeted_tool_grant_id is not None
            or call.targeted_tool_invocation is not None
            or call.targeted_tool_rejection is not None
        ):
            continue
        arguments = redactor.redact_json(call.arguments)
        if len(canonical_durable_json_bytes(arguments, "private arguments")) > MAX_CALL_BYTES:
            continue
        candidate = {**retained, call.id: arguments}
        if (
            len(candidate) <= 32
            and len(canonical_durable_json_bytes(candidate, "private arguments")) <= MAX_ROUND_BYTES
        ):
            retained = candidate
    return (
        None
        if not retained
        else ArgumentContinuity(
            nonce=uuid4().hex, profile=profile, scope=scope_digest(scope), arguments=retained
        )
    )


def redact_continuity(
    value: ArgumentContinuity | None, redactor: SecretRedactor
) -> ArgumentContinuity | None:
    if value is None:
        return None
    # Redaction can expand strings; exceeding the cache bound withholds continuity
    # instead of changing the outcome of an already executed tool.
    arguments = {
        call_id: redactor.redact_json(arguments) for call_id, arguments in value.arguments.items()
    }
    arguments = {
        key: value
        for key, value in arguments.items()
        if len(canonical_durable_json_bytes(value, "private arguments")) <= MAX_CALL_BYTES
    }
    if not arguments:
        return None
    if len(canonical_durable_json_bytes(arguments, "private arguments")) > MAX_ROUND_BYTES:
        return None
    return ArgumentContinuity(
        nonce=value.nonce, profile=value.profile, scope=value.scope, arguments=arguments
    )


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


async def materialize(
    *,
    store: SessionStore,
    session: Session,
    profile: str | None,
    messages: list[Message],
    names: frozenset[str],
    redactor: SecretRedactor,
    scope: KnowledgeAccessScope | None = None,
) -> list[Message]:
    """Overlay a detached selected view; never supply private inputs to compaction."""
    if not names or profile is None:
        return messages
    selected = [
        part
        for message in messages
        if message.role == "assistant"
        for part in message.content
        if isinstance(part, ToolCallPart) and part.tool_name in names and not part.arguments
    ]
    if not selected:
        return messages
    if len({(part.tool_round_id, part.tool_call_id) for part in selected}) != len(selected):
        return messages
    with private_read_scope():
        raw = await store.load_session_operation(session.id, STORAGE_KEY)
    if raw is None:
        return messages
    by_identity = {}
    current_scope = scope_digest(scope)
    for record, continuity in validate_records(raw):
        if record.get("instance") != session.instance_id or record.get("session_id") != session.id:
            continue
        if continuity.profile != profile or continuity.scope != current_scope:
            continue
        for call in record["calls"]:
            identity = canonical_durable_json_bytes(call, "private call association")
            by_identity[identity] = continuity.arguments.get(call["tool_call_id"])
    updates = {}
    for part in selected:
        identity = canonical_durable_json_bytes(part.model_dump(mode="json"), "selected call")
        arguments = by_identity.get(identity)
        if arguments is not None:
            updates[id(part)] = part.model_copy(
                update={"arguments": redactor.redact_json(arguments)}, deep=True
            )
    if not updates:
        return messages
    # OpenAI replays provider-native function-call items in preference to their
    # neutral counterparts. Restore only the matching omitted envelope in this
    # same selected message; never strip opaque reasoning or mutate audit state.
    provider_updates: dict[int, ProviderStatePart] = {}
    for message in messages:
        restored = {
            (part.tool_call_id, part.tool_name): updates[id(part)]
            for part in message.content
            if isinstance(part, ToolCallPart) and id(part) in updates
        }
        if not restored:
            continue
        for part in message.content:
            if (
                isinstance(part, ProviderStatePart)
                and part.provider == "openai"
                and part.state.get("type") == "function_call"
                and part.state.get("arguments") == "{}"
            ):
                call_id, name = part.state.get("call_id"), part.state.get("name")
                if type(call_id) is not str or type(name) is not str:
                    continue
                call = restored.get((call_id, name))
                if call is not None:
                    provider_updates[id(part)] = part.model_copy(
                        update={
                            "state": {
                                **part.state,
                                "arguments": canonical_durable_json_bytes(
                                    call.arguments, "private provider arguments"
                                ).decode("utf-8"),
                            }
                        },
                        deep=True,
                    )
    replacements = {**updates, **provider_updates}
    return [
        message.model_copy(
            update={"content": tuple(replacements.get(id(part), part) for part in message.content)},
            deep=True,
        )
        if message.role == "assistant" and any(id(part) in updates for part in message.content)
        else message
        for message in messages
    ]
