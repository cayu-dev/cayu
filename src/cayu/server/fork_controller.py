"""HTTP entrance for enrolled fork controllers.

Mount this router in a host application that authenticates its callers. The
host's ``authenticate`` callback names the subject; request bodies never select
the controller or carry its authority.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import aclosing

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from cayu.applications import CayuApp
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.fork_controllers import ForkControllerAccess, ForkControllerDenied
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionRejected
from cayu.sessions.forks import ForkRequestConflict
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest

_DENIED = "Fork controller access denied."


async def _stream(events):
    # Admission failures surface before any successful HTTP headers are sent.
    # Details are fixed: an absent session and a foreign one look the same.
    try:
        first = await anext(events)
    except StopAsyncIteration:
        return StreamingResponse(iter(()), media_type="text/event-stream")
    except (ForkControllerDenied, KeyError):
        await events.aclose()
        raise HTTPException(status_code=403, detail=_DENIED) from None
    except ExecutionProfileAdoptionRejected:
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The fork's execution profile was not adopted."
        ) from None
    except ForkRequestConflict:
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The fork destination exists with a different request."
        ) from None
    except ValueError:
        await events.aclose()
        raise HTTPException(status_code=422, detail="Invalid fork-controller request.") from None

    async def body():
        async with aclosing(events):
            yield "data: " + first.model_dump_json() + "\n\n"
            async for event in events:
                yield "data: " + event.model_dump_json() + "\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


def _with_authenticated_actor(body, subject: str):
    """Adoption provenance comes from authentication, never from the request body."""

    intent = body.profile_adoption
    if intent is None:
        return body
    if intent.requested_by.subject != subject:
        # Never silently replace a claimed actor; the audit trail would disagree.
        raise HTTPException(status_code=403, detail=_DENIED)
    return body.model_copy(
        update={
            "profile_adoption": intent.model_copy(
                update={
                    "requested_by": ResolutionActor(
                        subject=subject, source=ResolutionActorSource.HTTP_AUTH
                    )
                }
            )
        }
    )


def create_fork_controller_router(
    app: CayuApp,
    *,
    authenticate: Callable[[Request], Awaitable[str]],
    prefix: str = "/fork-controller",
) -> APIRouter:
    """Mount source snapshots, ``POST /fork`` and ``POST /resume`` for enrolled controllers."""

    if not callable(authenticate):
        raise TypeError("A trusted authentication callback is required.")
    router = APIRouter(prefix=prefix)

    async def access(request: Request) -> ForkControllerAccess:
        subject = await authenticate(request)
        if type(subject) is not str or not subject.strip() or subject != subject.strip():
            raise HTTPException(status_code=401, detail="Authentication required.")
        try:
            return app.fork_controller_access(subject)
        except ForkControllerDenied:
            raise HTTPException(status_code=403, detail=_DENIED) from None

    access_dependency = Depends(access)

    @router.get("/sources/{session_id}/snapshot")
    async def snapshot(session_id: str, handle: ForkControllerAccess = access_dependency):
        try:
            return await handle.snapshot_source(session_id)
        except (ForkControllerDenied, KeyError):
            raise HTTPException(status_code=403, detail=_DENIED) from None

    @router.post("/fork")
    async def fork(body: ForkSessionRequest, handle: ForkControllerAccess = access_dependency):
        return await _stream(handle.fork(_with_authenticated_actor(body, handle.subject)))

    @router.post("/resume")
    async def resume(body: ResumeRequest, handle: ForkControllerAccess = access_dependency):
        return await _stream(handle.resume(_with_authenticated_actor(body, handle.subject)))

    return router


__all__ = ["create_fork_controller_router"]
