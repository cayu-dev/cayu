"""HTTP assembly for the private-identity session trace reader."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import StringConstraints

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.server.contracts import SessionTraceResponse, SessionTraceSpanEventsResponse
from cayu.sessions.trace import SessionTraceReader, TraceReadConflict, TraceSpanNotFound

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def register_session_trace(
    router: APIRouter,
    *,
    protected: list[Any],
    reader: SessionTraceReader,
    resolve_session: Callable[[str], Awaitable[str]],
) -> None:
    @router.get(
        "/sessions/{session_id}/trace",
        response_model=SessionTraceResponse,
        dependencies=protected,
    )
    async def get_session_trace(
        session_id: NonBlankString,
        session_instance_id: Annotated[str | None, Query(min_length=1)] = None,
        after_sequence: int = Query(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER),
        through_sequence: int | None = Query(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER),
        limit: int = Query(default=100, ge=1, le=1000),
        include_events: bool = Query(
            default=True,
            description=(
                "Include the page's events. With `false` the response carries the "
                "page's spans and their event IDs and `events` is empty; load a "
                "selected span's events from `GET /sessions/{session_id}/events`."
            ),
        ),
    ):
        session_id = await resolve_session(session_id)
        try:
            result = await reader.read(
                session_id,
                after_sequence=after_sequence,
                through_sequence=through_sequence,
                limit=limit,
                session_instance_id=session_instance_id,
                include_events=include_events,
            )
        except TraceReadConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        if result is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return result

    @router.get(
        "/sessions/{session_id}/trace/spans/{span_id}/events",
        response_model=SessionTraceSpanEventsResponse,
        dependencies=protected,
        description=(
            "Return one trace span's events, and with `include_children` its direct "
            "children's, whole and in sequence order, for a span detail view over a trace read with "
            "`include_events=false`. Spans are those of the whole retained trace "
            "through `through_sequence`."
        ),
    )
    async def get_session_trace_span_events(
        session_id: NonBlankString,
        span_id: NonBlankString,
        session_instance_id: Annotated[str | None, Query(min_length=1)] = None,
        through_sequence: int | None = Query(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER),
        include_children: bool = Query(
            default=False, description="Also return the span's direct children's events."
        ),
    ):
        session_id = await resolve_session(session_id)
        try:
            result = await reader.read_span_events(
                session_id,
                span_id,
                through_sequence=through_sequence,
                session_instance_id=session_instance_id,
                include_children=include_children,
            )
        except TraceReadConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except TraceSpanNotFound:
            raise HTTPException(status_code=404, detail="Trace span not found") from None
        if result is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return result
