"""HTTP assembly for the private-identity session trace reader."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import StringConstraints

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.server.contracts import SessionTraceResponse
from cayu.sessions.trace import SessionTraceReader, TraceReadConflict

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
    ):
        session_id = await resolve_session(session_id)
        try:
            result = await reader.read(
                session_id,
                after_sequence=after_sequence,
                through_sequence=through_sequence,
                limit=limit,
                session_instance_id=session_instance_id,
            )
        except TraceReadConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        if result is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return result
