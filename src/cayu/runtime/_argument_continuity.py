"""Bounded, runtime-private continuity for omitted model-authored arguments.

This is a best-effort cache of original inputs, not knowledge read permission or
a second transcript. Public transcript material remains the authority for which
calls survive context selection. The native publication transaction owns writes.
"""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cayu._validation import canonical_durable_json_bytes
from cayu.messages import Message, ProviderStatePart, ToolCallPart
from cayu.sessions._argument_continuity import (
    MAX_CALL_BYTES as MAX_CALL_BYTES,
)
from cayu.sessions._argument_continuity import (
    MAX_RECORD_BYTES as MAX_RECORD_BYTES,
)
from cayu.sessions._argument_continuity import (
    MAX_ROUND_BYTES as MAX_ROUND_BYTES,
)
from cayu.sessions._argument_continuity import (
    MAX_ROUNDS as MAX_ROUNDS,
)
from cayu.sessions._argument_continuity import (
    STORAGE_KEY as STORAGE_KEY,
)
from cayu.sessions._argument_continuity import (
    ArgumentContinuity as ArgumentContinuity,
)
from cayu.sessions._argument_continuity import (
    _reading as _reading,
)
from cayu.sessions._argument_continuity import (
    append_record as append_record,
)
from cayu.sessions._argument_continuity import (
    private_read_scope as private_read_scope,
)
from cayu.sessions._argument_continuity import (
    require_private_key_access as require_private_key_access,
)
from cayu.sessions._argument_continuity import (
    validate_records as validate_records,
)
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.knowledge.scopes import KnowledgeAccessScope
    from cayu.runtime._runtime_records import ToolCallRequest
    from cayu.sessions.base import SessionStore
    from cayu.sessions.records import Session


def scope_digest(scope: KnowledgeAccessScope | None) -> str:
    if scope is None:
        return "0" * 64
    from cayu.knowledge.scopes import KnowledgeAccessScope

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
            # Only this identity-bound private view regains available arguments;
            # the persisted audit projection remains unavailable.
            updates[id(part)] = part.model_copy(
                update={
                    "arguments": redactor.redact_json(arguments),
                    "arguments_state": "finalized",
                },
                deep=True,
            )
    if not updates:
        return messages
    # OpenAI replays provider-native function-call items in preference to their
    # neutral counterparts. Restore only the matching omitted envelope in this
    # same selected message; never strip opaque reasoning or mutate audit state.
    provider_updates: dict[int, ProviderStatePart] = {}
    for message in messages:
        restored = {
            (part.tool_call_id, part.tool_name): (part, updates[id(part)])
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
            ):
                call_id, name = part.state.get("call_id"), part.state.get("name")
                if type(call_id) is not str or type(name) is not str:
                    continue
                pair = restored.get((call_id, name))
                if pair is not None:
                    original, call = pair
                    expected = canonical_durable_json_bytes(
                        original.continuation_arguments(), "omitted provider arguments"
                    ).decode("utf-8")
                    if part.state.get("arguments") not in ("{}", expected):
                        continue
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
