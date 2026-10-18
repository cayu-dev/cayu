"""Keyed, non-authoritative correlation values for public tool, model and supervision reads.

Pure derivations from durable records and an alias codec. They sit below the
runtime and the fork-controller layer so both can publish the same values; the
runtime's process-local fallback key stays in ``cayu.runtime._tool_call_correlation``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from cayu import _event_schema
from cayu.events import Event, EventType
from cayu.tools.outcomes import _TOOL_CALL_TERMINAL_DISPOSITIONS

if TYPE_CHECKING:
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec

TOOL_CALL_CORRELATION_FIELD = "tool_call_correlation_id"
TOOL_ROUND_CORRELATION_FIELD = "tool_round_correlation_id"
MODEL_STEP_CORRELATION_FIELD = "model_step_correlation_id"
MODEL_ATTEMPT_CORRELATION_FIELD = "model_attempt_correlation_id"
AUXILIARY_OPERATION_CORRELATION_FIELD = "auxiliary_operation_correlation_id"
# Every payload field a correlated read may publish. Readers drop these names
# from every payload before adding derived values, so producer data never
# poses as correlation.
CORRELATION_FIELDS = (
    TOOL_CALL_CORRELATION_FIELD,
    TOOL_ROUND_CORRELATION_FIELD,
    MODEL_STEP_CORRELATION_FIELD,
    MODEL_ATTEMPT_CORRELATION_FIELD,
    AUXILIARY_OPERATION_CORRELATION_FIELD,
)
# The codec field name is the domain: values of one kind never equal values of
# another kind, and no resolver accepts any of these field names.
TOOL_CALL_CORRELATION_DOMAIN = "tool_call_correlation"
_TOOL_ROUND_DOMAIN = "tool_round_correlation"
_MODEL_STEP_DOMAIN = "model_step_correlation"
_MODEL_ATTEMPT_DOMAIN = "model_attempt_correlation"
_AUXILIARY_OPERATION_DOMAIN = "auxiliary_operation_correlation"
TOOL_CALL_CORRELATION_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_STARTED,
        *_TOOL_CALL_TERMINAL_DISPOSITIONS,
    }
)
_MODEL_CORRELATION_EVENT_TYPES = frozenset(
    {
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_ERROR,
    }
)
_AUXILIARY_CORRELATION_EVENT_TYPES = frozenset(
    {
        EventType.MODEL_AUXILIARY_ATTEMPT_STARTED,
        EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
    }
)


def _tool_call_value(
    codec: PublicAuthorityAliasCodec, *, session_id: str, round_id: str, call_id: str
) -> str:
    return codec.encode(
        json.dumps([round_id, call_id], ensure_ascii=False, separators=(",", ":")),
        field_name=TOOL_CALL_CORRELATION_DOMAIN,
        session_id=session_id,
    )


def tool_call_correlation_value(event: Event, codec: PublicAuthorityAliasCodec) -> str | None:
    """The session-scoped correlation of one tool call's lifecycle events, or ``None``."""

    if event.type not in TOOL_CALL_CORRELATION_EVENT_TYPES:
        return None
    call_id = _event_schema.private_event_linkage_value(event, field_name="tool_call_id")
    round_id = _event_schema.private_event_linkage_value(event, field_name="tool_round_id")
    if call_id is None or round_id is None:
        return None
    return _tool_call_value(codec, session_id=event.session_id, round_id=round_id, call_id=call_id)


def _schema_identity(event: Event, field_name: str) -> str | None:
    """Read one private identity the event's schema owns as authority, or nothing."""

    policy = _event_schema.event_payload_policy(event.type)
    if field_name in policy.aliased_authority_keys:
        return _event_schema.private_event_linkage_value(event, field_name=field_name)
    if field_name not in policy.authority_keys:
        return None
    value = event.payload.get(field_name)
    return value if type(value) is str and value.strip() else None


def _auxiliary_operation_id(event: Event) -> str | None:
    if event.type not in _AUXILIARY_CORRELATION_EVENT_TYPES:
        return None
    attribution = event.payload.get("auxiliary_inference")
    value = attribution.get("operation_id") if type(attribution) is dict else None
    return value if type(value) is str and value.strip() else None


def _encoded(
    codec: PublicAuthorityAliasCodec, value: str | None, *, domain: str, session_id: str
) -> str | None:
    return None if value is None else codec.encode(value, field_name=domain, session_id=session_id)


def event_correlation_ids(event: Event, codec: PublicAuthorityAliasCodec) -> dict[str, str]:
    """Derive keyed correlation values from one private durable event.

    Tool starts and terminals carry their call and round, and a model
    completion that proposed tools carries that round. Tool starts, model
    starts, completions and errors, and auxiliary attempts carry their model
    step and attempt; auxiliary attempts also carry their operation. Values are
    session-scoped HMACs under the store keyring, so they are stable across
    reads and restarts and say nothing about the private identity.
    """

    session_id = event.session_id
    ids: dict[str, str | None] = {}
    ids[TOOL_CALL_CORRELATION_FIELD] = tool_call_correlation_value(event, codec)
    if event.type in TOOL_CALL_CORRELATION_EVENT_TYPES | _MODEL_CORRELATION_EVENT_TYPES:
        ids[TOOL_ROUND_CORRELATION_FIELD] = _encoded(
            codec,
            _schema_identity(event, "tool_round_id"),
            domain=_TOOL_ROUND_DOMAIN,
            session_id=session_id,
        )
    if event.type in {
        EventType.TOOL_CALL_STARTED,
        *_MODEL_CORRELATION_EVENT_TYPES,
        *_AUXILIARY_CORRELATION_EVENT_TYPES,
    }:
        ids[MODEL_STEP_CORRELATION_FIELD] = _encoded(
            codec,
            _schema_identity(event, "model_step_id"),
            domain=_MODEL_STEP_DOMAIN,
            session_id=session_id,
        )
        ids[MODEL_ATTEMPT_CORRELATION_FIELD] = _encoded(
            codec,
            _schema_identity(event, "model_attempt_id"),
            domain=_MODEL_ATTEMPT_DOMAIN,
            session_id=session_id,
        )
    ids[AUXILIARY_OPERATION_CORRELATION_FIELD] = _encoded(
        codec,
        _auxiliary_operation_id(event),
        domain=_AUXILIARY_OPERATION_DOMAIN,
        session_id=session_id,
    )
    return {field: value for field, value in ids.items() if value is not None}


def supervision_correlation_ids(
    codec: PublicAuthorityAliasCodec,
    *,
    session_id: str,
    tool_round_id: str | None,
    tool_call_ids: Sequence[str],
    model_step_id: str | None,
    model_attempt_id: str | None,
) -> dict[str, str | list[str] | None]:
    """Derive the same keyed values for one supervision touch's private linkage."""

    return {
        "tool_call_correlation_ids": (
            []
            if tool_round_id is None
            else [
                _tool_call_value(
                    codec, session_id=session_id, round_id=tool_round_id, call_id=call_id
                )
                for call_id in tool_call_ids
            ]
        ),
        TOOL_ROUND_CORRELATION_FIELD: _encoded(
            codec, tool_round_id, domain=_TOOL_ROUND_DOMAIN, session_id=session_id
        ),
        MODEL_STEP_CORRELATION_FIELD: _encoded(
            codec, model_step_id, domain=_MODEL_STEP_DOMAIN, session_id=session_id
        ),
        MODEL_ATTEMPT_CORRELATION_FIELD: _encoded(
            codec, model_attempt_id, domain=_MODEL_ATTEMPT_DOMAIN, session_id=session_id
        ),
    }
