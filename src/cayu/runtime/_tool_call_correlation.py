"""Non-authoritative correlation for public tool lifecycle observations."""

from __future__ import annotations

import base64
import json
import secrets

from pydantic import SecretStr

from cayu import _event_schema
from cayu.events import Event, EventType
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.tools.outcomes import _TOOL_CALL_TERMINAL_DISPOSITIONS

TOOL_CALL_CORRELATION_FIELD = "tool_call_correlation_id"
TOOL_CALL_CORRELATION_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_STARTED,
        *_TOOL_CALL_TERMINAL_DISPOSITIONS,
    }
)
# Stores without a configured keyring can still correlate one consumed stream.
# This key never signs resolution authority and never enters a checkpoint.
_PROCESS_CODEC = PublicAuthorityAliasCodec(
    PublicAuthorityAliasKeyring(
        active_key_id="process",
        keys={
            "process": SecretStr(
                base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
            )
        },
    )
)


def public_tool_call_correlation(
    event: Event,
    codec: PublicAuthorityAliasCodec | None,
) -> str | None:
    """Group one call across record-scoped aliases without publishing authority.

    Use the existing store keyring, a distinct field namespace and exact session,
    round and call identities. This handle is never accepted by resolution APIs.
    A changed active alias key produces a new presentation handle. Stores with
    no configured keyring use a process-local key. The handle is suitable for
    one collected stream, not a durable application identifier.
    """
    if event.type not in TOOL_CALL_CORRELATION_EVENT_TYPES:
        return None
    codec = codec or _PROCESS_CODEC
    call_id = _event_schema.private_event_linkage_value(event, field_name="tool_call_id")
    round_id = _event_schema.private_event_linkage_value(event, field_name="tool_round_id")
    if call_id is None or round_id is None:
        return None
    return codec.encode(
        json.dumps([round_id, call_id], ensure_ascii=False, separators=(",", ":")),
        field_name="tool_call_correlation",
        session_id=event.session_id,
    )
