"""Rebuild events PostgreSQL stores wrote into ``cayu_events.event`` documents.

The document is the event's durable envelope, written by the store after its
ingestion boundary validated the event. ``event_from_document`` rebuilds it
through ``event_from_trusted_store_record``, which lists the checks that still
run; a document that is not a complete envelope object takes the validating
``Event`` constructor and raises what it raised before.
"""

from __future__ import annotations

from typing import Any

from cayu.events import Event, event_from_trusted_store_record, store_payload_json
from cayu.sessions.event_delivery import (
    persisted_event_authority_candidates,
    restore_persisted_event_authority,
)
from cayu.sessions.records import EventRecord, stored_event_record
from cayu.storage import _postgres_support as pg_support

_ENVELOPE_KEYS = frozenset(Event.model_fields)


def event_from_document(
    document: Any,
    *,
    restore_authority: bool = False,
    input_contract_runtime_owned: bool = False,
    file_attachment_attestations_runtime_owned: bool = False,
) -> Event:
    """Rebuild one stored event document without re-walking its payload.

    With ``restore_authority``, the store's private authority is restored as
    ``restore_persisted_event_authority`` restores it; without, the event has
    none, as ``Event(**document)`` had.
    """

    document = pg_support._json_obj(document)
    candidates = (
        persisted_event_authority_candidates(
            document.get("type") if type(document) is dict else None,
            input_contract_runtime_owned=input_contract_runtime_owned,
            file_attachment_attestations_runtime_owned=file_attachment_attestations_runtime_owned,
        )
        if restore_authority
        else ()
    )
    payload_json = _payload_text(document)
    if payload_json is None:
        event = Event(**document)
        if not restore_authority:
            return event
        return restore_persisted_event_authority(
            event,
            input_contract_runtime_owned=input_contract_runtime_owned,
            file_attachment_attestations_runtime_owned=file_attachment_attestations_runtime_owned,
        )
    return event_from_trusted_store_record(
        event_type=document["type"],
        session_id=document["session_id"],
        interaction_id=document["interaction_id"],
        event_id=document["id"],
        timestamp=document["timestamp"],
        agent_name=document["agent_name"],
        environment_name=document["environment_name"],
        workflow_name=document["workflow_name"],
        tool_name=document["tool_name"],
        payload_json=payload_json,
        runtime_payload_authority_candidates=candidates,
    )


def event_record_from_document(sequence: int, document: Any, **authority: bool) -> EventRecord:
    """``EventRecord`` for one stored event document, without a defensive copy."""

    return stored_event_record(sequence, event_from_document(document, **authority))


def _payload_text(document: Any) -> str | None:
    """The payload's stored JSON text, or None when the document needs validation."""

    if type(document) is not dict or document.keys() != _ENVELOPE_KEYS:
        return None
    payload = document["payload"]
    if type(payload) is not dict:
        return None
    try:
        return store_payload_json(payload)
    except (TypeError, ValueError, RecursionError):
        return None
