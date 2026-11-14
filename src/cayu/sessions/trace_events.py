"""Events as the session trace reads them: their columns and a few payload members.

The trace places each event under its turn, model step, tool round, tool call,
command or wait from identity members of its payload, and reads a span's status
from a few outcome members. :func:`reduce_trace_payload` keeps exactly those
members, so a reader folding :class:`TraceEvent` rows sees the same topology as
one folding whole events. Native stores keep the reduced payload in
``cayu_trace_events`` as each event is written (see
:mod:`cayu.storage._trace_event_schema`); :func:`trace_events_of` is the
reference that derives the same rows from whole events.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu._trace_payload import (
    TRACE_PAYLOAD_KEYS,
    TRACE_RESULT_STRUCTURED_EVENT_TYPES,
    TRACE_RESULT_STRUCTURED_KEYS,
    TRACE_WHOLE_MEMBERS,
    reduce_trace_payload,
)
from cayu.events import Event
from cayu.messages import ToolCallPart
from cayu.sessions.records import EventHead, EventRecord

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One stored event with its columns, its payload's size and its reduced payload.

    ``payload_bytes`` is the size of the payload as the store keeps it, which
    differs slightly between stores; it is for display, not comparison.
    ``event`` is an unvalidated ``Event`` carrying the reduced payload, for the
    trace's own topology rules only. Never publish it.
    """

    sequence: int
    event: Event
    payload_bytes: int


def stored_trace_event(
    head: EventHead,
    *,
    interaction_id: str | None,
    agent_name: str | None,
    environment_name: str | None,
    workflow_name: str | None,
    tool_name: str | None,
    payload_bytes: int,
    payload: Mapping[str, Any] | str,
) -> TraceEvent:
    """A trace event from a stored head, the event's other columns and its reduced payload."""

    reduced = json.loads(payload) if isinstance(payload, str) else dict(payload)
    return TraceEvent(
        sequence=head.sequence,
        payload_bytes=payload_bytes,
        # The stored event was validated when it was written; this copy carries
        # only what the trace reads, so it is built without validating again.
        event=Event.model_construct(
            id=head.event_id,
            type=head.type,
            session_id=head.session_id,
            interaction_id=interaction_id,
            timestamp=head.timestamp,
            agent_name=agent_name,
            environment_name=environment_name,
            workflow_name=workflow_name,
            tool_name=tool_name,
            payload=reduced,
        ),
    )


def _payload_bytes(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())


def trace_event_of(record: EventRecord) -> TraceEvent:
    """Derive one trace event from a whole stored event: the reference for native stores."""

    event = record.event
    return stored_trace_event(
        EventHead.of(record),
        interaction_id=event.interaction_id,
        agent_name=event.agent_name,
        environment_name=event.environment_name,
        workflow_name=event.workflow_name,
        tool_name=event.tool_name,
        payload_bytes=_payload_bytes(event.payload),
        payload=reduce_trace_payload(str(event.type), event.payload),
    )


def trace_events_of(records: Iterable[EventRecord]) -> list[TraceEvent]:
    return [trace_event_of(record) for record in records]


@dataclass(frozen=True, slots=True)
class TranscriptToolCall:
    """One tool call part of a transcript message, where it sits in the transcript."""

    message_index: int
    part_index: int
    interaction_id: str | None
    part: ToolCallPart


_TRANSCRIPT_PAGE = 1000


async def transcript_tool_calls_of(
    store: SessionStore, session_id: str, interaction_ids: Iterable[str]
) -> list[TranscriptToolCall]:
    """Every tool call part of these interactions' transcript messages, read whole.

    The reference for native stores, which read only the parts.
    """

    from cayu.sessions.transcript_queries import TranscriptQuery

    calls: list[TranscriptToolCall] = []
    for interaction_id in sorted(set(interaction_ids)):
        offset = 0
        while True:
            page = await store.query_transcript(
                TranscriptQuery(
                    session_id=session_id,
                    interaction_id=interaction_id,
                    offset=offset,
                    limit=_TRANSCRIPT_PAGE,
                )
            )
            for record in page.records:
                for position, part in enumerate(record.message.content):
                    if type(part) is ToolCallPart:
                        calls.append(
                            TranscriptToolCall(record.index, position, record.interaction_id, part)
                        )
            offset += _TRANSCRIPT_PAGE
            if offset >= page.total_records:
                break
    return calls


def stored_transcript_tool_call(
    message_index: int, part_index: int, interaction_id: str | None, part: Any
) -> TranscriptToolCall:
    document = json.loads(part) if isinstance(part, str | bytes) else part
    return TranscriptToolCall(
        message_index, part_index, interaction_id, ToolCallPart.model_validate(document)
    )


_CHECK_PAGE = 5000


@dataclass(frozen=True, slots=True)
class TraceEventCheck:
    """How a session's stored trace events compare with a recompute from its events."""

    session_id: str
    checked_events: int
    #: Sequences whose stored row differs from the recompute, or exists on one side only.
    mismatched_sequences: tuple[int, ...]


def _comparable(row: TraceEvent) -> tuple[Any, ...]:
    event = row.event
    return (
        event.id,
        str(event.type),
        event.session_id,
        event.interaction_id,
        event.timestamp,
        event.agent_name,
        event.environment_name,
        event.workflow_name,
        event.tool_name,
        event.payload,
    )


async def check_trace_events(store: SessionStore, session_id: str) -> TraceEventCheck:
    """Compare ``store``'s trace events for one session with the reference derivation.

    Reads the session's events twice, through the store's own
    ``query_trace_events`` and through the default that reduces whole events,
    and reports every sequence where they differ. Payload sizes are not
    compared: each store measures the payload as it keeps it.
    """

    from cayu.sessions.event_queries import EventQuery
    from cayu.sessions.store import SessionStore

    stored: dict[int, tuple[Any, ...]] = {}
    recomputed: dict[int, tuple[Any, ...]] = {}
    for target, read in (
        (stored, store.query_trace_events),
        (recomputed, lambda query: SessionStore.query_trace_events(store, query)),
    ):
        after = 0
        while True:
            page = await read(
                EventQuery(session_id=session_id, after_sequence=after, limit=_CHECK_PAGE)
            )
            for row in page:
                target[row.sequence] = _comparable(row)
            if len(page) < _CHECK_PAGE:
                break
            after = page[-1].sequence
    mismatched = sorted(
        sequence
        for sequence in stored.keys() | recomputed.keys()
        if stored.get(sequence) != recomputed.get(sequence)
    )
    return TraceEventCheck(session_id, len(recomputed), tuple(mismatched))


__all__ = [
    "TRACE_PAYLOAD_KEYS",
    "TRACE_RESULT_STRUCTURED_EVENT_TYPES",
    "TRACE_RESULT_STRUCTURED_KEYS",
    "TRACE_WHOLE_MEMBERS",
    "TraceEvent",
    "TraceEventCheck",
    "TranscriptToolCall",
    "check_trace_events",
    "reduce_trace_payload",
    "stored_trace_event",
    "stored_transcript_tool_call",
    "trace_event_of",
    "trace_events_of",
    "transcript_tool_calls_of",
]
