"""HTTP routes of a person's scoped inbox, mounted by ``create_resource_router(inbox=True)``.

The paths, bodies and responses are those of the Developer dashboard's ``/api``
inbox and human-review routes, under the resource router's prefix, so a client
changes only its base URL. Each route calls ``ScopedCayuAccess.inbox``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.approvals.review import HumanReviewLongFieldView, HumanReviewView
from cayu.resource_access import ScopedCayuAccess
from cayu.server._inbox_routes import ItemIdPath
from cayu.server.contracts import ApiErrorResponse
from cayu.server.inbox import (
    InboxConflictResponse,
    InboxCounts,
    InboxItemKind,
    InboxItemState,
    InboxPage,
    InboxSettleRequest,
    InboxSettleResponse,
)
from cayu.server.routes import HumanReviewLongFieldBody
from cayu.server.scoped_inbox import InboxRefused

_NO_STORE = {"Cache-Control": "private, no-store"}
_REVIEW_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}
_UNAVAILABLE = "Resource unavailable."


async def _answer(operation: Awaitable[Any]) -> Any:
    try:
        return await operation
    except ResourceAccessDenied:
        # A hidden, foreign or absent item gets the same answer.
        raise HTTPException(status_code=404, detail=_UNAVAILABLE, headers=_NO_STORE) from None
    except InboxRefused as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.detail, headers=_NO_STORE
        ) from None


def register_scoped_inbox_routes(
    router: APIRouter, *, access_dependency: Any, person_dependency: Any
) -> None:
    """Serve the authenticated person's inbox on ``router``.

    The dependencies are ``fastapi.Depends(...)`` markers resolving to the person's
    ``ScopedCayuAccess`` and to their ``(subject, tenant)``.
    """

    @router.get(
        "/inbox",
        response_model=InboxPage,
        responses={404: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    async def scoped_inbox(
        response: Response,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: Annotated[str | None, Query(min_length=1, max_length=8192)] = None,
        session_id: Annotated[str | None, Query(max_length=4096)] = None,
        agent_name: Annotated[str | None, Query(max_length=4096)] = None,
        kind: InboxItemKind | None = None,
        q: Annotated[str | None, Query(max_length=4096)] = None,
        handle: ScopedCayuAccess = access_dependency,
    ) -> InboxPage:
        response.headers.update(_NO_STORE)
        try:
            return await _answer(
                handle.inbox.list(
                    kind=kind,
                    session_id=session_id,
                    cursor=cursor,
                    limit=limit,
                    agent_name=agent_name,
                    q=q,
                )
            )
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid inbox cursor.") from None

    @router.get("/inbox/counts", response_model=InboxCounts)
    async def scoped_inbox_counts(
        response: Response,
        session_id: Annotated[str | None, Query(max_length=4096)] = None,
        agent_name: Annotated[str | None, Query(max_length=4096)] = None,
        q: Annotated[str | None, Query(max_length=4096)] = None,
        handle: ScopedCayuAccess = access_dependency,
    ) -> InboxCounts:
        response.headers.update(_NO_STORE)
        return await _answer(handle.inbox.counts(session_id=session_id, agent_name=agent_name, q=q))

    @router.get(
        "/inbox/{item_id}",
        response_model=InboxItemState,
        responses={404: {"model": ApiErrorResponse}},
    )
    async def scoped_inbox_item(
        item_id: ItemIdPath, response: Response, handle: ScopedCayuAccess = access_dependency
    ) -> InboxItemState:
        response.headers.update(_NO_STORE)
        return await _answer(handle.inbox.get(item_id))

    @router.post(
        "/inbox/{item_id}/settle",
        response_model=InboxSettleResponse,
        responses={
            404: {"model": ApiErrorResponse},
            409: {"model": InboxConflictResponse},
            422: {"model": ApiErrorResponse},
        },
    )
    async def settle_scoped_inbox_item(
        item_id: ItemIdPath,
        body: InboxSettleRequest,
        response: Response,
        who: tuple[str, str | None] = person_dependency,
        handle: ScopedCayuAccess = access_dependency,
    ) -> InboxSettleResponse:
        response.headers.update(_NO_STORE)
        return await _answer(handle.inbox.settle(item_id, body, tenant=who[1]))

    @router.get("/sessions/{session_id}/human-review", response_model=HumanReviewView)
    async def scoped_human_review(
        session_id: str,
        purpose: Annotated[str, Query(min_length=1, max_length=128)],
        who: tuple[str, str | None] = person_dependency,
        handle: ScopedCayuAccess = access_dependency,
    ):
        view = await _answer(handle.inbox.review(session_id, purpose, tenant=who[1]))
        return JSONResponse(view.model_dump(mode="json"), headers=_REVIEW_HEADERS)

    @router.post(
        "/sessions/{session_id}/human-review/long-field",
        response_model=HumanReviewLongFieldView,
    )
    async def scoped_human_review_long_field(
        session_id: str,
        body: HumanReviewLongFieldBody,
        who: tuple[str, str | None] = person_dependency,
        handle: ScopedCayuAccess = access_dependency,
    ):
        field = await _answer(
            handle.inbox.review_long_field(
                session_id, body.review_reference, body.key, tenant=who[1]
            )
        )
        return JSONResponse(field.model_dump(mode="json"), headers=_REVIEW_HEADERS)
