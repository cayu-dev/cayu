"""The runtime-owned record of the one call a proposal session may raise.

``tool_call_proposal_request`` binds this record to the ``RunRequest`` it
builds, and session creation stores it under a reserved metadata key. No HTTP
route, request metadata or metadata update can write that key, so a call that
matches it was proposed by application code, not written into a transcript by
whoever can start a run. Proposal-only tools are admitted only through it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any

from cayu._validation import canonical_durable_json_bytes

TOOL_CALL_PROPOSAL_METADATA_KEY = "cayu:tool_call_proposal"
_RECORD_VERSION = 1
_RUNTIME_TOOL_CALL_PROPOSAL_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _RuntimeToolCallProposal:
    """Process-local proof that ``tool_call_proposal_request`` wrote the record."""

    token: object
    record: Mapping[str, Any]


def proposal_arguments_sha256(arguments: Mapping[str, Any]) -> str:
    """Digest of a call's arguments in canonical durable JSON."""

    return sha256(canonical_durable_json_bytes(dict(arguments), "arguments")).hexdigest()


def tool_call_proposal_record(
    *, tool_name: str, tool_call_id: str, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "version": _RECORD_VERSION,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "arguments_sha256": proposal_arguments_sha256(arguments),
    }


def runtime_tool_call_proposal_holder(record: Mapping[str, Any]) -> _RuntimeToolCallProposal:
    return _RuntimeToolCallProposal(
        _RUNTIME_TOOL_CALL_PROPOSAL_TOKEN, MappingProxyType(dict(record))
    )


def is_runtime_tool_call_proposal_holder(value: object) -> bool:
    return held_tool_call_proposal(value) is not None


def held_tool_call_proposal(value: object) -> dict[str, Any] | None:
    """The record in a genuine runtime holder; None for any other value."""

    if (
        type(value) is not _RuntimeToolCallProposal
        or value.token is not _RUNTIME_TOOL_CALL_PROPOSAL_TOKEN
    ):
        return None
    return dict(value.record)


def session_tool_call_proposal(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """The record a session was created with, or None when it has no valid one."""

    record = metadata.get(TOOL_CALL_PROPOSAL_METADATA_KEY)
    if (
        type(record) is not dict
        or record.get("version") != _RECORD_VERSION
        or set(record) != {"version", "tool_name", "tool_call_id", "arguments_sha256"}
        or not all(
            type(record[key]) is str and record[key]
            for key in ("tool_name", "tool_call_id", "arguments_sha256")
        )
    ):
        return None
    return dict(record)


def call_matches_tool_call_proposal(
    record: Mapping[str, Any], *, tool_name: str, tool_call_id: str, arguments: Mapping[str, Any]
) -> bool:
    try:
        digest = proposal_arguments_sha256(arguments)
    except (TypeError, ValueError):
        return False
    return (
        record["tool_name"] == tool_name
        and record["tool_call_id"] == tool_call_id
        and record["arguments_sha256"] == digest
    )
