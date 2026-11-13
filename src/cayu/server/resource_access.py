"""Product HTTP entrance for application-authenticated resource handles.

The packaged control plane remains an operator interface. Mount this router in
an application that authenticates requests; never derive its subject from a
request body, scope document, tenant header, or Runtime provenance field.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import Base64Bytes, BaseModel, ConfigDict, Field

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.applications import CayuApp
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeListQuery, KnowledgeQuery
from cayu.resource_access import ScopedCayuAccess
from cayu.server._inbox_cursor import InboxCursorSeal, share_inbox_cursor_seal
from cayu.server._provider_setup_errors import provider_setup_http_error
from cayu.server.service import ProductPrincipal
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.queries import SessionQuery
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.sessions.supervision_events import (
    SupervisionEventCursorInvalid,
    SupervisionEventQuery,
    SupervisionEventResultTooLarge,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.queries import TaskQuery


class _Labels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: dict[str, str]


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metadata: dict


class _TaskCreation(_Labels):
    task: TaskCreate


class _ArtifactCreation(_Labels):
    filename: str
    content: Base64Bytes = Field(max_length=4_194_304)
    content_type: str | None = None


async def _checked(operation):
    try:
        return await operation
    except ResourceAccessDenied:
        # Known foreign IDs and absent IDs have the same public response.
        raise HTTPException(status_code=404, detail="Resource unavailable.") from None


async def _stream(events, revalidate, *, app: CayuApp | None = None, session_id: str | None = None):
    # Do not commit successful HTTP headers before the first admission check.
    try:
        first = await anext(events)
    except StopAsyncIteration:
        return StreamingResponse(iter(()), media_type="text/event-stream")
    except ResourceAccessDenied:
        await events.aclose()
        raise HTTPException(status_code=404, detail="Resource unavailable.") from None
    except Exception as exc:
        # A provider this server hasn't set up: 503 with what to set, or 409 naming
        # a provider it doesn't have.
        setup_error = (
            None
            if app is None
            else await provider_setup_http_error(app, exc, session_id=session_id)
        )
        if setup_error is None:
            raise
        await events.aclose()
        raise setup_error from exc

    async def body():
        async with aclosing(events):
            try:
                await revalidate(first)
                yield "data: " + first.model_dump_json() + "\n\n"
                async for event in events:
                    await revalidate(event)
                    yield "data: " + event.model_dump_json() + "\n\n"
            except ResourceAccessDenied:
                yield 'event: access_revoked\ndata: {"detail":"Resource unavailable."}\n\n'

    return StreamingResponse(body(), media_type="text/event-stream")


def create_resource_router(
    app: CayuApp,
    *,
    authenticate: Callable[[Request], Awaitable[str | ProductPrincipal]],
    principal_subject: Callable[[ProductPrincipal], str] | None = None,
    prefix: str = "/resources",
    artifact_stores: dict | None = None,
    knowledge_scope: KnowledgeAccessScope | None = None,
    inbox: bool = False,
    inbox_cursor_key: bytes | None = None,
) -> APIRouter:
    """Mount only scoped endpoints. ``authenticate`` is trusted host code.

    It returns the authenticated subject, or a ``ProductPrincipal``, so a product
    dependency such as ``ProxyIdentityAuth.product_dependency()`` can be passed as it
    is. The resource access policy resolves a subject string alone, so a principal
    needs ``principal_subject``, which says what subject it is: without it, two
    tenants' people with the same ``subject_id`` would get each other's resources.
    Pass ``lambda principal: principal.subject_id`` only when subject ids are unique
    across tenants; otherwise return one that names the tenant too, or refuse a
    tenant the app doesn't serve by raising ``HTTPException``. A principal without
    ``principal_subject`` raises ``TypeError``.

    ``inbox=True`` also serves the person's inbox (``ScopedCayuAccess.inbox``) on the
    paths and bodies of the Developer dashboard's ``/api``: ``GET /inbox``,
    ``GET /inbox/counts``, ``GET /inbox/{item_id}``, ``POST /inbox/{item_id}/settle``,
    ``GET /sessions/{id}/human-review`` and ``POST /sessions/{id}/human-review/long-field``.
    A principal's ``tenant_id`` is recorded as the decider's tenant. Its cursors are
    sealed with the deployment's ``inbox_cursor_key``: the one given here, or else the
    one ``create_server``, ``mount_cayu`` or ``create_router`` was given for ``app``.
    Without either, each process makes its own key, so a cursor works only on the
    replica that issued it.
    """
    if not callable(authenticate):
        raise TypeError("A trusted authentication callback is required.")
    if principal_subject is not None and not callable(principal_subject):
        raise TypeError("`principal_subject` must be callable.")
    if inbox_cursor_key is not None:
        share_inbox_cursor_seal(app, InboxCursorSeal(inbox_cursor_key))
    router = APIRouter(prefix=prefix)
    configured_artifacts = dict(artifact_stores or {})

    async def person(request: Request) -> tuple[str, str | None]:
        identity = await authenticate(request)
        tenant = None
        subject = identity
        if isinstance(identity, ProductPrincipal):
            if principal_subject is None:
                raise TypeError(
                    "create_resource_router's authenticate returned a ProductPrincipal; "
                    "pass principal_subject= to say which subject its tenant's person is."
                )
            subject = principal_subject(identity)
            tenant = identity.tenant_id
        if type(subject) is not str or not subject.strip() or subject != subject.strip():
            raise HTTPException(status_code=401, detail="Authentication required.")
        return subject, tenant

    person_dependency = Depends(person)

    async def access(who: tuple[str, str | None] = person_dependency) -> ScopedCayuAccess:
        return await _checked(app.access(who[0]))

    access_dependency = Depends(access)

    async def private_session_id(session_id: str) -> str:
        # Inbox items and events name a session by its public alias when its ID
        # carries a secret; the scoped access check runs on the private ID.
        try:
            return await app._resolve_public_session_id(session_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Resource unavailable.") from None

    @router.post("/sessions/query")
    async def sessions(query: SessionQuery, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.sessions.list_sessions(query))

    @router.get("/sessions/{session_id}")
    async def session(session_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.sessions.load(await private_session_id(session_id)))

    @router.put("/sessions/{session_id}/labels")
    async def labels(session_id: str, body: _Labels, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.sessions.update_labels(session_id, body.labels))

    @router.put("/sessions/{session_id}/metadata")
    async def metadata(
        session_id: str, body: _Metadata, handle: ScopedCayuAccess = access_dependency
    ):
        return await _checked(handle.sessions.update_metadata(session_id, body.metadata))

    @router.delete("/sessions/{session_id}", status_code=204)
    async def delete(session_id: str, handle: ScopedCayuAccess = access_dependency):
        await _checked(handle.sessions.delete_session(session_id))

    @router.get("/sessions/{session_id}/records/{kind}")
    async def records(
        session_id: str,
        kind: Literal["events", "transcript", "checkpoint", "access_audit"],
        offset: int = Query(default=0, ge=0, le=1_000_000),
        limit: int = Query(default=100, ge=1, le=1000),
        max_bytes: int = Query(default=1_048_576, ge=1, le=4_194_304),
        handle: ScopedCayuAccess = access_dependency,
    ):
        return asdict(
            await _checked(
                handle.sessions.read_records(
                    await private_session_id(session_id),
                    kind=kind,
                    offset=offset,
                    limit=limit,
                    max_bytes=max_bytes,
                )
            )
        )

    @router.post("/tasks/query")
    async def tasks(query: TaskQuery, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.list(query))

    @router.get("/tasks/{task_id}")
    async def task(task_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.load(task_id))

    @router.post("/tasks", status_code=201)
    async def create_task(body: _TaskCreation, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.create(body.task, labels=body.labels))

    def artifact_handle(handle, store_name):
        configured = configured_artifacts.get(store_name)
        if configured is None:
            raise HTTPException(status_code=404, detail="Resource unavailable.")
        store, environment_name = configured
        return handle.artifacts(store, environment_name=environment_name)

    @router.get("/artifacts/{store_name}")
    async def artifacts(
        store_name: str,
        limit: int = Query(default=100, ge=1, le=1000),
        handle: ScopedCayuAccess = access_dependency,
    ):
        return await _checked(artifact_handle(handle, store_name).list(limit=limit))

    @router.post("/artifacts/{store_name}", status_code=201)
    async def create_artifact(
        store_name: str, body: _ArtifactCreation, handle: ScopedCayuAccess = access_dependency
    ):
        return await _checked(
            artifact_handle(handle, store_name).put_bytes(
                body.content,
                filename=body.filename,
                content_type=body.content_type,
                labels=body.labels,
            )
        )

    @router.get("/artifacts/{store_name}/{artifact_id}")
    async def artifact(
        store_name: str,
        artifact_id: str,
        max_bytes: int = Query(default=1_048_576, ge=1, le=4_194_304),
        handle: ScopedCayuAccess = access_dependency,
    ):
        result = await _checked(
            artifact_handle(handle, store_name).read_bytes(artifact_id, max_bytes=max_bytes)
        )
        return Response(
            result.content,
            media_type="application/octet-stream",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @router.delete("/artifacts/{store_name}/{artifact_id}", status_code=204)
    async def delete_artifact(
        store_name: str, artifact_id: str, handle: ScopedCayuAccess = access_dependency
    ):
        await _checked(artifact_handle(handle, store_name).delete(artifact_id))

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.cancel(task_id))

    @router.post("/tasks/{task_id}/pause")
    async def pause_task(task_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.pause(task_id))

    @router.post("/tasks/{task_id}/resume")
    async def resume_task(task_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.resume(task_id))

    @router.post("/events/query")
    async def events(
        query: EventQuery,
        max_bytes: int = Query(default=1_048_576, ge=1, le=4_194_304),
        handle: ScopedCayuAccess = access_dependency,
    ):
        return await _checked(handle.sessions.events(query, max_bytes=max_bytes))

    @router.post("/usage/query")
    async def usage(
        query: EventQuery, by_session: bool = False, handle: ScopedCayuAccess = access_dependency
    ):
        return await _checked(handle.sessions.usage(query, by_session=by_session))

    @router.post("/supervision-events/query")
    async def supervision_events(
        query: SupervisionEventQuery, handle: ScopedCayuAccess = access_dependency
    ):
        try:
            return await _checked(handle.sessions.supervision_events(query))
        except SupervisionEventCursorInvalid:
            raise HTTPException(
                status_code=422, detail="Invalid supervision event cursor."
            ) from None
        except NotImplementedError:
            raise HTTPException(
                status_code=501,
                detail="The configured session store does not support supervision events.",
            ) from None
        except SupervisionEventResultTooLarge:
            raise HTTPException(
                status_code=413,
                detail="Supervision events exceed the byte limit; request a smaller page.",
            ) from None

    @router.get("/graphs/{graph_id}")
    async def graph(graph_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.graph(graph_id))

    @router.get("/groups/{group_id}")
    async def group(group_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.group(group_id))

    @router.post("/runs")
    async def run(body: RunRequest, handle: ScopedCayuAccess = access_dependency):
        return await _stream(handle.run(body), handle.revalidate_delivery, app=app, session_id=None)

    @router.post("/resume")
    async def resume(body: ResumeRequest, handle: ScopedCayuAccess = access_dependency):
        return await _stream(
            handle.resume(body), handle.revalidate_delivery, app=app, session_id=body.session_id
        )

    @router.post("/fork")
    async def fork(body: ForkSessionRequest, handle: ScopedCayuAccess = access_dependency):
        return await _stream(
            handle.fork(body),
            handle.revalidate_delivery,
            app=app,
            session_id=body.source_session_id,
        )

    if knowledge_scope is not None:

        @router.post("/knowledge/query")
        async def knowledge_list(
            query: KnowledgeListQuery, handle: ScopedCayuAccess = access_dependency
        ):
            return await _checked(
                handle.knowledge(access_scope=knowledge_scope).list_entries(query)
            )

        @router.post("/knowledge/search")
        async def knowledge_search(
            query: KnowledgeQuery, handle: ScopedCayuAccess = access_dependency
        ):
            return await _checked(handle.knowledge(access_scope=knowledge_scope).search(query))

        @router.get("/knowledge/{entry_id}")
        async def knowledge_entry(entry_id: str, handle: ScopedCayuAccess = access_dependency):
            entry = await _checked(
                handle.knowledge(access_scope=knowledge_scope).get_entry(entry_id)
            )
            if entry is None:
                raise HTTPException(status_code=404, detail="Resource unavailable.")
            return entry

    if inbox:
        from cayu.server._scoped_inbox_routes import register_scoped_inbox_routes

        register_scoped_inbox_routes(
            router, access_dependency=access_dependency, person_dependency=person_dependency
        )

    return router
