"""HTTP assembly for one session's model-call usage and paired tool calls, and rollups.

`GET /sessions/{id}/model-calls` and `GET /sessions/{id}/tool-calls` serve the views behind
`cayu session usage` and `cayu session tools` (`cayu.sessions._call_inspection`). They pair
events by private identity on the server and publish only what the public event API does: a
tool call is named by its `tool_call_correlation_id`, and model step, attempt, round and
budget reservation ids are withheld.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.routing import APIRoute
from pydantic import StringConstraints

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.budgets.pricing import PriceBook
from cayu.runtime._tool_call_correlation import public_tool_call_handle
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.server.contracts import (
    SessionModelCallsRequest,
    SessionModelCallsResponse,
    SessionToolCallsResponse,
)
from cayu.sessions._call_inspection import (
    ArgumentDisclosure,
    InspectionTooLarge,
    InspectionWindow,
    inspect_model_calls,
    inspect_tool_calls,
)
from cayu.sessions._rollup import RollupPage, SessionRollupQuery, rollup_page
from cayu.sessions.store import SessionStore

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_PAGE_LIMIT_MAX = 1000
_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"description": "The session does not exist."},
    413: {"description": "The event window holds more than one view reads; narrow it."},
}


def register_session_call_routes(
    router: APIRouter,
    *,
    protected: list[Any],
    store: SessionStore,
    resolve_session: Callable[[str], Awaitable[str]],
    resolve_interaction: Callable[[str, str], Awaitable[str]],
    project_session: Callable[[str], str],
    redact: Callable[[Any], Any],
    codec: Callable[[], PublicAuthorityAliasCodec | None],
    pricing_route: type[APIRoute] = APIRoute,
) -> None:
    async def window(
        session_id: str,
        after_sequence: int | None,
        before_sequence: int | None,
        interaction_id: str | None,
    ) -> tuple[str, InspectionWindow]:
        private_id = await resolve_session(session_id)
        if await store.load_state(private_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if interaction_id is not None:
            interaction_id = await resolve_interaction(private_id, interaction_id)
        return private_id, InspectionWindow(
            after_sequence=after_sequence,
            before_sequence=before_sequence,
            interaction_id=interaction_id,
        )

    async def model_calls(
        session_id: str,
        response: Response,
        *,
        offset: int,
        limit: int,
        after_sequence: int | None,
        before_sequence: int | None,
        interaction_id: str | None,
        pricing: PriceBook | None,
    ) -> dict[str, Any]:
        private_id, selected = await window(
            session_id, after_sequence, before_sequence, interaction_id
        )
        try:
            result = await inspect_model_calls(
                store, private_id, selected, offset=offset, limit=limit, pricing=pricing
            )
        except InspectionTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None
        response.headers["Cache-Control"] = "private, no-store"
        return {
            "session_id": project_session(private_id),
            **result,
            "unmatched_ledger": [
                {**ledger, "reservation_id": None} for ledger in result["unmatched_ledger"]
            ],
        }

    @router.get(
        "/sessions/{session_id}/model-calls",
        response_model=SessionModelCallsResponse,
        responses=_RESPONSES,
        dependencies=protected,
        description=(
            "Per-model-call token usage, budget reservations no call claims, and aggregate "
            "counters for one session, as `cayu session usage` reports them. `offset` and "
            "`limit` page the calls and the reservations alike. POST a price book to price "
            "each call as well."
        ),
    )
    async def get_session_model_calls(
        session_id: NonBlankString,
        response: Response,
        offset: Annotated[int, Query(ge=0, le=MAX_DURABLE_JSON_INTEGER)] = 0,
        limit: Annotated[int, Query(ge=1, le=_PAGE_LIMIT_MAX)] = 100,
        after_sequence: Annotated[int | None, Query(ge=0)] = None,
        before_sequence: Annotated[int | None, Query(ge=1)] = None,
        interaction_id: Annotated[NonBlankString | None, Query()] = None,
    ):
        return await model_calls(
            session_id,
            response,
            offset=offset,
            limit=limit,
            after_sequence=after_sequence,
            before_sequence=before_sequence,
            interaction_id=interaction_id,
            pricing=None,
        )

    async def post_session_model_calls(
        session_id: NonBlankString,
        body: SessionModelCallsRequest,
        response: Response,
        offset: Annotated[int, Query(ge=0, le=MAX_DURABLE_JSON_INTEGER)] = 0,
        limit: Annotated[int, Query(ge=1, le=_PAGE_LIMIT_MAX)] = 100,
        after_sequence: Annotated[int | None, Query(ge=0)] = None,
        before_sequence: Annotated[int | None, Query(ge=1)] = None,
        interaction_id: Annotated[NonBlankString | None, Query()] = None,
    ):
        return await model_calls(
            session_id,
            response,
            offset=offset,
            limit=limit,
            after_sequence=after_sequence,
            before_sequence=before_sequence,
            interaction_id=interaction_id,
            pricing=body.pricing,
        )

    router.add_api_route(
        "/sessions/{session_id}/model-calls",
        post_session_model_calls,
        methods=["POST"],
        response_model=SessionModelCallsResponse,
        responses=_RESPONSES,
        dependencies=protected,
        route_class_override=pricing_route,
        description=(
            "The same view as `GET`, with every call priced on its own date from the "
            "body's price book. `aggregate.estimated_cost` sums priced calls per currency "
            "and `aggregate.unpriced_calls` counts calls the book has no price for."
        ),
    )

    @router.get(
        "/sessions/{session_id}/tool-calls",
        response_model=SessionToolCallsResponse,
        responses=_RESPONSES,
        dependencies=protected,
        description=(
            "One session's tool calls, each start paired with its outcome and approval, "
            "without result bodies, as `cayu session tools` reports them. `tool_call_id` "
            "is the call's `tool_call_correlation_id` from its tool events."
        ),
    )
    async def get_session_tool_calls(
        session_id: NonBlankString,
        response: Response,
        offset: Annotated[int, Query(ge=0, le=MAX_DURABLE_JSON_INTEGER)] = 0,
        limit: Annotated[int, Query(ge=1, le=_PAGE_LIMIT_MAX)] = 100,
        after_sequence: Annotated[int | None, Query(ge=0)] = None,
        before_sequence: Annotated[int | None, Query(ge=1)] = None,
        interaction_id: Annotated[NonBlankString | None, Query()] = None,
    ):
        private_id, selected = await window(
            session_id, after_sequence, before_sequence, interaction_id
        )
        try:
            result = await inspect_tool_calls(
                store,
                private_id,
                selected,
                offset=offset,
                limit=limit,
                disclosure=ArgumentDisclosure(quarantine_pre_execution=True, redact=redact),
            )
        except InspectionTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None
        alias_codec = codec()
        response.headers["Cache-Control"] = "private, no-store"
        return {
            "session_id": project_session(private_id),
            **result,
            "calls": [_public_tool_call(row, private_id, alias_codec) for row in result["calls"]],
        }


def _public_tool_call(
    row: dict[str, Any],
    session_id: str,
    codec: PublicAuthorityAliasCodec | None,
) -> dict[str, Any]:
    round_id, call_id = row.get("tool_round_id"), row.get("tool_call_id")
    handle = (
        public_tool_call_handle(
            session_id=session_id, round_id=round_id, call_id=call_id, codec=codec
        )
        if isinstance(round_id, str) and isinstance(call_id, str)
        else None
    )
    return {
        **row,
        "tool_call_id": handle,
        "model_step_id": None,
        "model_attempt_id": None,
        "tool_round_id": None,
    }


def register_session_rollup_route(
    router: APIRouter,
    *,
    protected: list[Any],
    store: SessionStore,
    redact: Callable[[Any], Any],
    serialize_cursor: Callable[[str | None], str | None],
    pricing_route: type[APIRoute] = APIRoute,
) -> None:
    async def post_session_rollup(query: SessionRollupQuery, response: Response):
        try:
            page = await rollup_page(store, query)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        response.headers["Cache-Control"] = "private, no-store"
        return RollupPage(
            groups=[
                group.model_copy(
                    update={
                        "key": redact(group.key),
                        "failure_kinds": [
                            kind.model_copy(update={"tool": redact(kind.tool)})
                            for kind in group.failure_kinds
                        ],
                    }
                )
                for group in page.groups
            ],
            sessions_read=page.sessions_read,
            next_cursor=serialize_cursor(page.next_cursor),
        )

    router.add_api_route(
        "/sessions/rollup",
        post_session_rollup,
        methods=["POST"],
        response_model=RollupPage,
        dependencies=protected,
        route_class_override=pricing_route,
        description=(
            "One page of `cayu session rollup`: usage, cost, time by phase and tool failures "
            "by error kind for up to `max_sessions` sessions created in the window, grouped "
            "by agent, UTC day, Release or `label:KEY`. Groups merge by adding; pass "
            "`next_cursor` back as `cursor` for the next page."
        ),
    )
