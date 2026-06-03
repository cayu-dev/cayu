"""API routes for the cayu server."""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StringConstraints
from sse_starlette.sse import EventSourceResponse

from cayu.core.messages import Message
from cayu.runtime.sessions import ResumeRequest, RunRequest, SessionQuery
from cayu.runtime.tasks import TaskCreate, TaskQuery
from cayu.server.sse import event_to_sse_data

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RunBody(BaseModel):
    prompt: NonBlankString
    agent: NonBlankString = "assistant"


class ResumeBody(BaseModel):
    session_id: NonBlankString
    prompt: NonBlankString


def create_router(
    *,
    cayu_app,
    session_store,
    task_store,
) -> APIRouter:
    """Create an APIRouter with standard cayu endpoints."""

    router = APIRouter(prefix="/api")

    @router.post("/run")
    async def run_agent(body: RunBody):
        session_id = f"session-{uuid4().hex[:8]}"

        if task_store is not None:
            task = await task_store.create_task(
                TaskCreate(
                    type="run",
                    title=body.prompt[:80],
                    assigned_agent_name=body.agent,
                    input={"prompt": body.prompt},
                )
            )
            task_id = task.id
        else:
            task_id = None

        request = RunRequest(
            agent_name=body.agent,
            session_id=session_id,
            task_id=task_id,
            messages=[Message.text("user", body.prompt)],
            max_steps=20,
        )

        async def generate():
            async for event in cayu_app.run(request):
                yield event_to_sse_data(event)

        return EventSourceResponse(generate())

    @router.post("/resume")
    async def resume_agent(body: ResumeBody):
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ResumeRequest(
            session_id=body.session_id,
            messages=[Message.text("user", body.prompt)],
            max_steps=20,
        )

        async def generate():
            async for event in cayu_app.resume(request):
                yield event_to_sse_data(event)

        return EventSourceResponse(generate())

    @router.get("/sessions")
    async def list_sessions(limit: int = 50):
        sessions = await session_store.list_sessions(SessionQuery(limit=limit))
        return [
            {
                "id": s.id,
                "status": s.status.value,
                "agent_name": s.agent_name,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        events = await session_store.load_events(session_id)
        transcript = await session_store.load_transcript(session_id)

        return {
            "session": {
                "id": session.id,
                "status": session.status.value,
                "agent_name": session.agent_name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
            },
            "events": [
                {
                    "id": e.id,
                    "type": str(e.type),
                    "agent_name": e.agent_name,
                    "tool_name": e.tool_name,
                    "payload": e.payload,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in events
            ],
            "transcript": [
                {
                    "role": str(m.role),
                    "content": [p.model_dump() for p in m.content],
                }
                for m in transcript
            ],
        }

    @router.get("/tasks")
    async def list_tasks(limit: int = 50):
        if task_store is None:
            return []
        tasks = await task_store.list_tasks(TaskQuery(limit=limit))
        return [
            {
                "id": t.id,
                "type": t.type,
                "title": t.title,
                "status": t.status.value,
                "session_id": t.session_id,
                "created_at": t.created_at.isoformat(),
                "completed_at": (t.completed_at.isoformat() if t.completed_at else None),
            }
            for t in tasks
        ]

    @router.get("/health")
    async def health():
        return {"ok": True}

    return router
