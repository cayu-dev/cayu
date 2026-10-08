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
from cayu.sessions.base import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.queries import SessionQuery
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


async def _stream(events, revalidate):
    # Do not commit successful HTTP headers before the first admission check.
    try:
        first = await anext(events)
    except StopAsyncIteration:
        return StreamingResponse(iter(()), media_type="text/event-stream")
    except ResourceAccessDenied:
        await events.aclose()
        raise HTTPException(status_code=404, detail="Resource unavailable.") from None

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
    authenticate: Callable[[Request], Awaitable[str]],
    prefix: str = "/resources",
    artifact_stores: dict | None = None,
    knowledge_scope: KnowledgeAccessScope | None = None,
) -> APIRouter:
    """Mount only scoped endpoints. ``authenticate`` is trusted host code."""
    if not callable(authenticate):
        raise TypeError("A trusted authentication callback is required.")
    router = APIRouter(prefix=prefix)
    configured_artifacts = dict(artifact_stores or {})

    async def access(request: Request) -> ScopedCayuAccess:
        subject = await authenticate(request)
        if type(subject) is not str or not subject.strip() or subject != subject.strip():
            raise HTTPException(status_code=401, detail="Authentication required.")
        return await _checked(app.access(subject))

    access_dependency = Depends(access)

    @router.post("/sessions/query")
    async def sessions(query: SessionQuery, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.sessions.list_sessions(query))

    @router.get("/sessions/{session_id}")
    async def session(session_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.sessions.load(session_id))

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
                    session_id,
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

    @router.get("/graphs/{graph_id}")
    async def graph(graph_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.graph(graph_id))

    @router.get("/groups/{group_id}")
    async def group(group_id: str, handle: ScopedCayuAccess = access_dependency):
        return await _checked(handle.tasks.group(group_id))

    @router.post("/runs")
    async def run(body: RunRequest, handle: ScopedCayuAccess = access_dependency):
        return await _stream(handle.run(body), handle.revalidate_delivery)

    @router.post("/resume")
    async def resume(body: ResumeRequest, handle: ScopedCayuAccess = access_dependency):
        return await _stream(handle.resume(body), handle.revalidate_delivery)

    @router.post("/fork")
    async def fork(body: ForkSessionRequest, handle: ScopedCayuAccess = access_dependency):
        return await _stream(handle.fork(body), handle.revalidate_delivery)

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

    return router
