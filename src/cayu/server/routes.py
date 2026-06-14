"""API routes for the cayu server."""

from __future__ import annotations

import contextlib
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints, ValidationError
from sse_starlette.sse import EventSourceResponse

from cayu.core.messages import Message, MessageRole
from cayu.runtime.approvals import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.runtime.costs import CostBudget, PricingCatalog
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    EventQuery,
    EventRecord,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    SessionQuery,
    SessionStatus,
    TranscriptQuery,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.tasks import TaskCreate, TaskQuery
from cayu.server.sse import event_to_sse_data

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_EVENT_PAGE_LIMIT_MAX = 1000
_TRANSCRIPT_PAGE_LIMIT_MAX = 1000
_SERVER_INTERRUPTIBLE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
}


class RunBody(BaseModel):
    prompt: NonBlankString
    agent: NonBlankString = "assistant"
    limits: RunLimits = Field(default_factory=RunLimits)
    cost_budget: CostBudget | None = None
    retry_policy: RetryPolicy | None = None


class ResumeBody(BaseModel):
    session_id: NonBlankString
    prompt: NonBlankString
    limits: RunLimits = Field(default_factory=RunLimits)
    cost_budget: CostBudget | None = None
    retry_policy: RetryPolicy | None = None


class InterruptSessionBody(BaseModel):
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionCostBody(BaseModel):
    pricing: PricingCatalog
    currency: NonBlankString = "USD"


class ToolApprovalBody(BaseModel):
    session_id: NonBlankString
    approval_id: NonBlankString
    decision: ToolApprovalDecision
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    limits: RunLimits = Field(default_factory=RunLimits)
    cost_budget: CostBudget | None = None
    retry_policy: RetryPolicy | None = None


class ToolApprovalRecoveryBody(BaseModel):
    session_id: NonBlankString
    approval_id: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    limits: RunLimits = Field(default_factory=RunLimits)
    cost_budget: CostBudget | None = None
    retry_policy: RetryPolicy | None = None


def _serialize_event_record(record: EventRecord) -> dict[str, Any]:
    event = record.event
    return {
        "sequence": record.sequence,
        "id": event.id,
        "type": str(event.type),
        "session_id": event.session_id,
        "agent_name": event.agent_name,
        "environment_name": event.environment_name,
        "workflow_name": event.workflow_name,
        "tool_name": event.tool_name,
        "payload": event.payload,
        "timestamp": event.timestamp.isoformat(),
    }


def _serialize_transcript_message(index: int, message: Message) -> dict[str, Any]:
    return {
        "index": index,
        "role": str(message.role),
        "content": [part.model_dump(mode="json") for part in message.content],
    }


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
            limits=body.limits,
            cost_budget=body.cost_budget,
            retry_policy=body.retry_policy,
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
            limits=body.limits,
            cost_budget=body.cost_budget,
            retry_policy=body.retry_policy,
        )

        async def generate():
            async for event in cayu_app.resume(request):
                yield event_to_sse_data(event)

        return EventSourceResponse(generate())

    @router.post("/sessions/{session_id}/interrupt")
    async def interrupt_session(
        session_id: NonBlankString,
        body: InterruptSessionBody | None = None,
    ):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )
        if session.status not in _SERVER_INTERRUPTIBLE_SESSION_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Session cannot be interrupted from status: {session.status.value}",
            )

        request = InterruptSessionRequest(
            session_id=session_id,
            reason=body.reason if body is not None else None,
            metadata=body.metadata if body is not None else {},
        )
        event_stream = cayu_app.interrupt_session(request)
        try:
            first_event = await anext(event_stream)
        except StopAsyncIteration as exc:
            await event_stream.aclose()
            raise HTTPException(
                status_code=500,
                detail="Session interruption produced no events.",
            ) from exc
        except ValueError as exc:
            await event_stream.aclose()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TimeoutError as exc:
            await event_stream.aclose()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            await event_stream.aclose()
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception:
            with contextlib.suppress(Exception):
                await event_stream.aclose()
            raise

        async def generate():
            yield event_to_sse_data(first_event)
            async for event in event_stream:
                yield event_to_sse_data(event)

        return EventSourceResponse(generate())

    @router.post("/tool-approvals/resolve")
    async def resolve_tool_approval(body: ToolApprovalBody):
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolApprovalRequest(
            session_id=body.session_id,
            approval_id=body.approval_id,
            decision=body.decision,
            reason=body.reason,
            metadata=body.metadata,
            max_steps=20,
            limits=body.limits,
            cost_budget=body.cost_budget,
            retry_policy=body.retry_policy,
        )

        async def generate():
            async for event in cayu_app.resolve_tool_approval(request):
                yield event_to_sse_data(event)

        return EventSourceResponse(generate())

    @router.post("/tool-approvals/recover")
    async def recover_tool_approval(body: ToolApprovalRecoveryBody):
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolApprovalRecoveryRequest(
            session_id=body.session_id,
            approval_id=body.approval_id,
            tool_call_id=body.tool_call_id,
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            max_steps=20,
            limits=body.limits,
            cost_budget=body.cost_budget,
            retry_policy=body.retry_policy,
        )

        async def generate():
            async for event in cayu_app.recover_tool_approval(request):
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
                "provider_name": s.provider_name,
                "model": s.model,
                "parent_session_id": s.parent_session_id,
                "runtime_name": s.runtime_name,
                "runtime_version": s.runtime_version,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]

    @router.get("/sessions/{session_id}/usage")
    async def get_session_usage(session_id: NonBlankString):
        try:
            summary = await cayu_app.get_session_usage(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary.model_dump()

    @router.post("/sessions/{session_id}/cost")
    async def estimate_session_cost(session_id: NonBlankString, body: SessionCostBody):
        try:
            summary = await cayu_app.get_session_cost(
                session_id,
                body.pricing,
                currency=body.currency,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary.model_dump(mode="json")

    @router.get("/sessions/{session_id}/events")
    async def list_session_events(
        session_id: NonBlankString,
        event_type: str | None = None,
        tool_name: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        workflow_name: str | None = None,
        after_sequence: int | None = Query(default=None, ge=0),
        limit: int = Query(default=100, ge=1, le=_EVENT_PAGE_LIMIT_MAX),
    ):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            query = EventQuery(
                session_id=session_id,
                event_type=event_type,
                tool_name=tool_name,
                agent_name=agent_name,
                environment_name=environment_name,
                workflow_name=workflow_name,
                after_sequence=after_sequence,
                limit=limit + 1,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_context=False, include_url=False),
            ) from exc

        records = await session_store.query_events(query)
        page = records[:limit]
        has_more = len(records) > limit
        next_sequence = page[-1].sequence if page else after_sequence

        return {
            "session_id": session_id,
            "events": [_serialize_event_record(record) for record in page],
            "next_sequence": next_sequence,
            "has_more": has_more,
        }

    @router.get("/sessions/{session_id}/transcript")
    async def get_session_transcript(
        session_id: NonBlankString,
        role: MessageRole | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=_TRANSCRIPT_PAGE_LIMIT_MAX),
    ):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        transcript_page = await session_store.query_transcript(
            TranscriptQuery(
                session_id=session_id,
                role=role,
                offset=offset,
                limit=limit,
            )
        )
        next_offset = offset + len(transcript_page.records)

        return {
            "session_id": session_id,
            "messages": [
                _serialize_transcript_message(record.index, record.message)
                for record in transcript_page.records
            ],
            "offset": offset,
            "next_offset": next_offset,
            "has_more": next_offset < transcript_page.total_records,
            "total_messages": transcript_page.total_records,
        }

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
                "provider_name": session.provider_name,
                "model": session.model,
                "parent_session_id": session.parent_session_id,
                "runtime_name": session.runtime_name,
                "runtime_version": session.runtime_version,
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
                    "content": [p.model_dump(mode="json") for p in m.content],
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
