"""Read public session labels with the store and run identity that owns them."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Response

from cayu.cloud_reporting import _store_identity
from cayu.runtime.public_authority import PublicSessionAliasNotFound
from cayu.server.contracts import SessionDisplaysResponse
from cayu.sessions._run_identity import session_run_id
from cayu.sessions.records import Session

if TYPE_CHECKING:
    from cayu.applications import CayuApp


def register_session_displays(
    router: APIRouter,
    *,
    app: CayuApp,
    protected: list[Any],
    resolve: Callable[[str], Awaitable[str]],
    title: Callable[[Session], str | None],
) -> None:
    @router.get("/sessions/display", response_model=SessionDisplaysResponse, dependencies=protected)
    async def session_displays(response: Response, ids: Annotated[list[str], Query()]):
        response.headers["Cache-Control"] = "private, no-store"
        try:
            generation = await _store_identity(app)
        except NotImplementedError:
            raise HTTPException(501, "This store does not expose a stable identity.") from None
        rows = []
        for public_id in dict.fromkeys(ids):
            try:
                private_id = await resolve(public_id)
            except PublicSessionAliasNotFound:
                continue
            except ValueError as exc:
                raise HTTPException(409, detail=str(exc)) from exc
            session = await app.session_store.load(private_id)
            if session is not None:
                rows.append(
                    {
                        "session_id": app.project_session_id_for_exposure(session.id),
                        "run_id": session_run_id(session),
                        "title": title(session),
                    }
                )
        if await _store_identity(app) != generation:
            raise HTTPException(409, "The session store changed; read the labels again.")
        return {"generation": generation, "items": rows}
