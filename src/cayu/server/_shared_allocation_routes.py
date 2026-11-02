"""Routes that list and release the shared allocations workflow runs record.

``GET /api/shared-allocations`` and ``POST /api/shared-allocations/release``
wrap ``CayuApp.list_shared_allocations`` and
``CayuApp.release_shared_allocation``, for an operator cleaning up after a
workflow run that crashed and won't be resumed. Neither is in the
read-and-decide role list, so under ``ControlPlaneRoles`` both need full
access (developer or admin), like the other session mutations. `cayu
environment allocations list|release` calls them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import HTTPException, Query, Response

from cayu.environments.shared_allocations import SharedAllocation
from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    ApiSharedAllocationRelease,
    ReleaseSharedAllocationBody,
    SharedAllocationsResponse,
)

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp

_LOGGER = logging.getLogger(__name__)
_MAX_QUERY_CHARS = 4096


def register_shared_allocation_routes(
    router: APIRouter,
    *,
    mutation_router: APIRouter,
    cayu_app: CayuApp,
    protected: list[Any],
    optional_auth_context: Depends,
) -> None:
    """Add the listing to ``router`` and the release to the bounded ``mutation_router``."""

    # The dependency both guards the release and names the caller for its log line.
    auth_default = cast("AuthContext | None", optional_auth_context)

    @router.get(
        "/shared-allocations",
        response_model=SharedAllocationsResponse,
        dependencies=protected,
        summary="List the shared allocations workflow runs recorded",
    )
    async def list_shared_allocations(
        response: Response,
        workflow_run_id: Annotated[str | None, Query(max_length=_MAX_QUERY_CHARS)] = None,
        environment_name: Annotated[str | None, Query(max_length=_MAX_QUERY_CHARS)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
        cursor: Annotated[str | None, Query(max_length=_MAX_QUERY_CHARS)] = None,
        inactive_for_seconds: Annotated[int | None, Query(ge=0, le=2**53 - 1)] = None,
    ):
        """Read the records of up to ``limit`` workflow runs, newest first.

        Each allocation says whether its run looks like it is running, has
        stopped, or has ended. Reconnect metadata and binding state are never
        returned.
        """

        response.headers["Cache-Control"] = "private, no-store"
        try:
            page = await cayu_app.list_shared_allocations(
                workflow_run_id=workflow_run_id or None,
                environment_name=environment_name or None,
                limit=limit,
                cursor=cursor or None,
                inactive_for_seconds=inactive_for_seconds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "allocations": [_serialize(allocation) for allocation in page.allocations],
            "next_cursor": page.next_cursor,
        }

    @mutation_router.post(
        "/shared-allocations/release",
        response_model=ApiSharedAllocationRelease,
        summary="Copy back or discard a workflow run's shared allocation",
        responses={
            404: {"description": "No such allocation is recorded (or it was already released)."},
            409: {"description": "Refused: the run may still be running, or the mode can't apply."},
        },
    )
    async def release_shared_allocation(
        body: ReleaseSharedAllocationBody,
        response: Response,
        auth_context: AuthContext | None = auth_default,
    ):
        """Release one recorded allocation, fenced like recovery's reaping.

        Refused with 409 while this server process runs the workflow run, while
        the run looks like it is running elsewhere (unless ``force``), when the
        environment isn't registered here, and for a copy-back of an allocation
        that can't be reconnected or holds no work.
        """

        response.headers["Cache-Control"] = "private, no-store"
        try:
            release = await cayu_app.release_shared_allocation(
                body.allocation_id,
                mode=body.mode,
                force=body.force,
                inactive_for_seconds=body.inactive_for_seconds,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Shared allocation not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=cayu_app.redact_json(str(exc))) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=cayu_app.redact_json(str(exc))) from exc
        _LOGGER.warning(
            "Shared allocation %s released (%s) by %s.",
            body.allocation_id,
            body.mode,
            "an unauthenticated caller" if auth_context is None else auth_context.subject,
        )
        return {
            "allocation": _serialize(release.allocation),
            "mode": release.mode.value,
            "forced": release.forced,
            "message": release.message,
        }


def _serialize(allocation: SharedAllocation) -> dict[str, Any]:
    return asdict(allocation)
