"""Session event pages: whole events, and heads without their payloads.

``GET /sessions/{session_id}/events`` and ``GET /sessions/{session_id}/event-heads``
take the same filters and cursors and return the same matches in the same order.
A head carries the event's sequence, public ID, type, time, names and payload
size, and is read without reading the payload, so a page of heads costs the
same however large its events are. Load one event's payload from ``/events``
with ``event_id``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import StringConstraints, ValidationError

from cayu.server.contracts import (
    SESSION_EVENTS_MAX_WAIT_SECONDS,
    ListSessionEventHeadsResponse,
    ListSessionEventsResponse,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import EventRecord, SessionStatus
from cayu.sessions.store import SessionStore
from cayu.sessions.trace_events import TraceEvent

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

EVENT_PAGE_LIMIT_MAX = 1000
REPLAY_ACTIVE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
}
# Replays check quickly after reconnect, then back off while a live session is
# quiet so idle streams do not continuously hammer the durable stores.
REPLAY_POLL_INTERVAL_MIN_S = 0.05
REPLAY_POLL_INTERVAL_MAX_S = 1.0


def next_replay_poll_interval(current: float, *, received_events: bool) -> float:
    if received_events:
        return REPLAY_POLL_INTERVAL_MIN_S
    return min(current * 2, REPLAY_POLL_INTERVAL_MAX_S)


def register_session_events(
    router: APIRouter,
    *,
    protected: list[Any],
    store: SessionStore,
    serialize_event: Callable[[EventRecord], dict[str, Any]],
    project_session: Callable[[str], str],
    project_head: Callable[[TraceEvent], dict[str, Any]],
    resolve_session: Callable[[str], Awaitable[str]],
    resolve_interaction: Callable[..., Awaitable[str]],
    resolve_event_record: Callable[[str, str], Awaitable[tuple[EventRecord | None, bool]]],
) -> None:
    def endpoint(*, heads: bool):
        async def list_session_events(
            session_id: NonBlankString,
            http_request: Request,
            event_id: str | None = Query(
                default=None,
                description="Return the event with this exact session-scoped event ID, if it exists.",
            ),
            event_type: str | None = None,
            interaction_id: str | None = Query(
                default=None,
                description="Return only events attributed to this interaction.",
            ),
            exclude_event_type: str | None = Query(
                default=None,
                description="Exclude one event type before applying pagination.",
            ),
            tool_name: str | None = None,
            agent_name: str | None = None,
            environment_name: str | None = None,
            workflow_name: str | None = None,
            after_sequence: int | None = Query(
                default=None,
                ge=0,
                description="Return only events with a greater durable sequence.",
            ),
            before_sequence: int | None = Query(
                default=None,
                ge=1,
                description="Return only events with a smaller durable sequence.",
            ),
            order_by: Annotated[
                EventOrder,
                Query(description="Return events in durable sequence order."),
            ] = EventOrder.SEQUENCE_ASC,
            limit: int = Query(default=100, ge=1, le=EVENT_PAGE_LIMIT_MAX),
            wait_seconds: float | None = Query(
                default=None,
                ge=0,
                le=SESSION_EVENTS_MAX_WAIT_SECONDS,
                description=(
                    "Long-poll fallback for clients that cannot hold the follow stream "
                    "open. Wait up to this many seconds for a matching event after "
                    "`after_sequence` while the session is pending, running, or "
                    "interrupting, then return the page (possibly empty)."
                ),
            ),
        ):
            if wait_seconds and (
                before_sequence is not None
                or event_id is not None
                or order_by != EventOrder.SEQUENCE_ASC
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "wait_seconds only applies to ascending tail reads without "
                        "before_sequence or event_id."
                    ),
                )
            session_id = await resolve_session(session_id)
            public_session_id = project_session(session_id)
            state = await store.load_state(session_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Session not found")

            if interaction_id is not None:
                interaction_id = await resolve_interaction(
                    session_id=session_id,
                    value=interaction_id,
                )

            private_event_id = event_id
            unresolved_public_event_id = False
            if event_id is not None:
                resolved_record, handled = await resolve_event_record(
                    session_id,
                    event_id,
                )
                if handled:
                    if resolved_record is None:
                        private_event_id = None
                        unresolved_public_event_id = True
                    else:
                        private_event_id = resolved_record.event.id

            has_event_filters = any(
                value is not None
                for value in (
                    event_id,
                    event_type,
                    interaction_id,
                    exclude_event_type,
                    tool_name,
                    agent_name,
                    environment_name,
                    workflow_name,
                )
            )
            try:
                query = EventQuery(
                    session_id=session_id,
                    event_id=private_event_id,
                    event_type=event_type,
                    interaction_id=interaction_id,
                    exclude_event_types=(exclude_event_type,)
                    if exclude_event_type is not None
                    else (),
                    tool_name=tool_name,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    workflow_name=workflow_name,
                    after_sequence=after_sequence,
                    before_sequence=before_sequence,
                    limit=limit + 1,
                    order_by=order_by,
                )
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=exc.errors(include_context=False, include_url=False),
                ) from exc

            if not wait_seconds:
                return await _read_session_events_page(
                    session_id=session_id,
                    public_session_id=public_session_id,
                    query=query,
                    limit=limit,
                    has_event_filters=has_event_filters,
                    unresolved_public_event_id=unresolved_public_event_id,
                    heads=heads,
                )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait_seconds
            poll_interval = REPLAY_POLL_INTERVAL_MIN_S
            while True:
                result = await _read_session_events_page(
                    session_id=session_id,
                    public_session_id=public_session_id,
                    query=query,
                    limit=limit,
                    has_event_filters=has_event_filters,
                    unresolved_public_event_id=unresolved_public_event_id,
                    heads=heads,
                )
                remaining = deadline - loop.time()
                if result["events"] or remaining <= 0:
                    return result
                current = await store.load_state(session_id)
                if current is None or current.status not in REPLAY_ACTIVE_SESSION_STATUSES:
                    return result
                if await http_request.is_disconnected():
                    return result
                # Same backoff as the follow stream: quick first checks, then at
                # most one store read per second while the session is quiet.
                await asyncio.sleep(min(poll_interval, remaining))
                poll_interval = next_replay_poll_interval(poll_interval, received_events=False)

        return list_session_events

    router.get(
        "/sessions/{session_id}/events",
        response_model=ListSessionEventsResponse,
        dependencies=protected,
    )(endpoint(heads=False))
    router.get(
        "/sessions/{session_id}/event-heads",
        response_model=ListSessionEventHeadsResponse,
        dependencies=protected,
        name="list_session_event_heads",
        description=(
            "Return event heads with the filters, cursors and order of "
            "`GET /sessions/{session_id}/events`: each event's sequence, ID, type, "
            "time, names and payload size, without its payload."
        ),
    )(endpoint(heads=True))

    async def _read_session_events_page(
        *,
        session_id: str,
        public_session_id: str,
        query: EventQuery,
        limit: int,
        has_event_filters: bool,
        unresolved_public_event_id: bool,
        heads: bool,
    ) -> dict[str, Any]:
        after_sequence = query.after_sequence
        before_sequence = query.before_sequence
        order_by = query.order_by
        raw_scan_sequence: int | None = None
        if before_sequence is None and has_event_filters:
            # Capture the raw high-water mark before the filtered read. Advancing
            # through a sequence observed after that read could skip a matching
            # event committed between the two queries.
            latest_records = await store.query_event_heads(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if latest_records:
                raw_scan_sequence = latest_records[0].sequence

        records: Sequence[EventRecord | TraceEvent]
        if heads:
            # Heads read each event's columns and payload size, never its payload.
            records = [] if unresolved_public_event_id else await store.query_trace_events(query)
        else:
            records = [] if unresolved_public_event_id else await store.query_events(query)
        page = records[:limit]
        has_more = len(records) > limit
        cursor = after_sequence if order_by == EventOrder.SEQUENCE_ASC else before_sequence
        next_sequence = page[-1].sequence if page else cursor

        scan_through_sequence: int | None = None
        if before_sequence is None:
            scan_candidates = [sequence for sequence in (after_sequence,) if sequence is not None]
            if order_by == EventOrder.SEQUENCE_ASC and has_more:
                # A matching event remains beyond this response, so the tail
                # cursor must stop at the last event actually returned.
                if page:
                    scan_candidates.append(page[-1].sequence)
            else:
                scan_candidates.extend(record.sequence for record in page)
                if raw_scan_sequence is not None:
                    scan_candidates.append(raw_scan_sequence)
            if scan_candidates:
                scan_through_sequence = max(scan_candidates)

        return {
            "session_id": public_session_id,
            "events": [
                project_head(record) if isinstance(record, TraceEvent) else serialize_event(record)
                for record in page
            ],
            "order_by": order_by,
            "next_sequence": next_sequence,
            "scan_through_sequence": scan_through_sequence,
            "has_more": has_more,
        }
