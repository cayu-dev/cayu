"""Non-authoritative correlation for public tool lifecycle observations."""

from __future__ import annotations

import base64
import secrets

from pydantic import SecretStr

from cayu.events import Event
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions._correlation import tool_call_correlation_value, tool_call_correlation_with

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
    return tool_call_correlation_value(event, codec or _PROCESS_CODEC)


def public_tool_call_handle(
    *,
    session_id: str,
    round_id: str,
    call_id: str,
    codec: PublicAuthorityAliasCodec | None,
) -> str:
    """The `tool_call_correlation_id` that call's public tool events carry."""

    return tool_call_correlation_with(
        codec or _PROCESS_CODEC, session_id=session_id, round_id=round_id, call_id=call_id
    )
