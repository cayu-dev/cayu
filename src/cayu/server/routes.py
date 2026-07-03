"""API routes for the cayu server."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
)
from sse_starlette.sse import EventSourceResponse

from cayu._validation import copy_label_map, require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.approvals import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.costs import PricingCatalog
from cayu.runtime.costs import (
    estimate_causal_budget_cost as build_causal_budget_cost_summary,
)
from cayu.runtime.costs import (
    estimate_session_cost as build_session_cost_summary,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    EventQuery,
    EventRecord,
    InterruptSessionRequest,
    LabelSelectorOperator,
    LabelSelectorRequirement,
    ResumeRequest,
    RunRequest,
    Session,
    SessionOrder,
    SessionOutcome,
    SessionQuery,
    SessionStatus,
    TranscriptQuery,
    event_summary_from_records,
    session_outcome_from_records,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.structured_output import StructuredOutputSpec
from cayu.runtime.tasks import Task, TaskCreate, TaskQuery, TaskStatus
from cayu.runtime.usage import causal_budget_usage_summary
from cayu.server.sse import (
    error_to_sse_message,
    event_to_sse_message,
    parse_last_event_id,
)
from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListItem,
    KnowledgeReviewWorkflow,
    KnowledgeVisibility,
)

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# Server-entrypoint step budget. The default preserves the historical value while the
# ceiling matches the runtime's own ``max_steps`` bound (RunRequest/ResumeRequest and the
# tool-approval bodies all cap at 256) so a request cannot ask for an unbounded run.
_DEFAULT_RUN_MAX_STEPS = 20
_MAX_RUN_STEPS = 256
_EVENT_PAGE_LIMIT_MAX = 1000
_TRANSCRIPT_PAGE_LIMIT_MAX = 1000
_KNOWLEDGE_REVIEW_PREVIEW_CHARS = 1200
_KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS = 50
_KNOWLEDGE_PENDING_DETAIL_MAX_BYTES = 128_000
_SERVER_INTERRUPTIBLE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
}
_REPLAY_ACTIVE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
}
_REPLAY_POLL_INTERVAL_S = 0.05

# Detached event pumps must outlive their SSE consumer (a client disconnect must not
# cancel agent work), so hold strong references until each pump finishes — the event
# loop only keeps weak references to tasks.
_detached_event_pumps: set[asyncio.Task[None]] = set()


def _detached_event_stream_response(event_stream: AsyncIterator[Event]) -> EventSourceResponse:
    """Run ``event_stream`` to completion in a detached task; stream it as an observer.

    The run is driven by the pump task, not by the SSE consumer: a client disconnect
    stops the observer while the session still runs to a terminal state (finalized,
    not stranded RUNNING). Each frame carries a resumable ``id:`` field, and a runtime
    failure surfaces as a terminal structured ``error`` frame instead of an aborted
    connection.
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in event_stream:
                queue.put_nowait(("event", event))
        except BaseException as exc:
            queue.put_nowait(("error", exc))
            if not isinstance(exc, Exception):
                raise
        else:
            queue.put_nowait(("done", None))

    pump_task = asyncio.create_task(pump())
    _detached_event_pumps.add(pump_task)
    pump_task.add_done_callback(_detached_event_pumps.discard)

    async def observe() -> AsyncIterator[dict[str, str]]:
        while True:
            kind, item = await queue.get()
            if kind == "event":
                yield event_to_sse_message(item)
                continue
            if kind == "error":
                yield error_to_sse_message(item)
            return

    return EventSourceResponse(observe())


class RunBody(BaseModel):
    prompt: NonBlankString
    agent: NonBlankString = "assistant"
    causal_budget_id: NonBlankString | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=_DEFAULT_RUN_MAX_STEPS, ge=1, le=_MAX_RUN_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels", allow_reserved=False)


class ResumeBody(BaseModel):
    session_id: NonBlankString
    prompt: NonBlankString
    max_steps: StrictInt = Field(default=_DEFAULT_RUN_MAX_STEPS, ge=1, le=_MAX_RUN_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)


class InterruptSessionBody(BaseModel):
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateSessionLabelsBody(BaseModel):
    # Required + extra="forbid": a missing/typo'd key must 422, never silently replace
    # all labels with {} (these are full-replacement mutations).
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str]


class UpdateSessionMetadataBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any]


class SessionCostBody(BaseModel):
    pricing: PricingCatalog
    currency: NonBlankString = "USD"


class SessionsSummaryBody(BaseModel):
    pricing: PricingCatalog | None = None
    currency: NonBlankString = "USD"


class TaskHoldBody(BaseModel):
    reason: NonBlankString | None = None
    payload: dict[str, Any] | None = None


class ToolApprovalBody(BaseModel):
    """Body for resolving a pending tool approval.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run inherits the original run's configuration
    persisted on the pending approval. Explicit values override it.
    """

    session_id: NonBlankString
    approval_id: NonBlankString
    decision: ToolApprovalDecision
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class ToolApprovalRecoveryBody(BaseModel):
    """Body for recovering an approved tool call with an unknown result.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run inherits the original run's configuration
    persisted on the pending approval. Explicit values override it.
    """

    session_id: NonBlankString
    approval_id: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


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


def _serialize_session_outcome(outcome: SessionOutcome) -> dict[str, Any]:
    return {
        "session_id": outcome.session_id,
        "status": outcome.status.value,
        "reason": outcome.reason,
        "details": outcome.details,
        "retry": outcome.retry,
        "terminal_event": (
            None
            if outcome.terminal_event is None
            else _serialize_event_record(outcome.terminal_event)
        ),
        "latest_retry_event": (
            None
            if outcome.latest_retry_event is None
            else _serialize_event_record(outcome.latest_retry_event)
        ),
    }


def _serialize_session_base(session: Session) -> dict[str, Any]:
    # Shared list-view fields. The list endpoint omits the (potentially large,
    # unbounded) per-session metadata; callers fetch a single session to get it.
    return {
        "id": session.id,
        "status": session.status.value,
        "agent_name": session.agent_name,
        "provider_name": session.provider_name,
        "model": session.model,
        "parent_session_id": session.parent_session_id,
        "causal_budget_id": session.causal_budget_id,
        "runtime_name": session.runtime_name,
        "runtime_version": session.runtime_version,
        "environment_name": session.environment_name,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "labels": session.labels,
    }


def _serialize_session(session: Session) -> dict[str, Any]:
    return {**_serialize_session_base(session), "metadata": session.metadata}


def _parse_session_label_filters(values: list[str] | None) -> dict[str, str]:
    if values is None:
        return {}
    labels: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise HTTPException(
                status_code=422,
                detail="Session label filters must use `key=value`.",
            )
        key, value = raw.split("=", 1)
        try:
            parsed = copy_label_map({key: value}, "label")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_key, parsed_value = next(iter(parsed.items()))
        if parsed_key in labels:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate session label filter: {parsed_key}",
            )
        labels[parsed_key] = parsed_value
    return labels


def _parse_session_label_selectors(
    values: list[str] | None,
) -> tuple[LabelSelectorRequirement, ...]:
    if values is None:
        return ()
    selectors: list[LabelSelectorRequirement] = []
    for raw in values:
        if type(raw) is not str:
            raise HTTPException(status_code=422, detail="Label selector must be a string.")
        for expression in _split_label_selector(raw):
            selectors.append(_parse_label_selector_expression(expression))
    return tuple(selectors)


def _split_label_selector(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise HTTPException(status_code=422, detail="Invalid label selector.")
        elif char == "," and depth == 0:
            part = value[start:index].strip()
            if not part:
                raise HTTPException(status_code=422, detail="Invalid label selector.")
            parts.append(part)
            start = index + 1
    if depth != 0:
        raise HTTPException(status_code=422, detail="Invalid label selector.")
    part = value[start:].strip()
    if not part:
        raise HTTPException(status_code=422, detail="Invalid label selector.")
    parts.append(part)
    return parts


def _parse_label_selector_expression(expression: str) -> LabelSelectorRequirement:
    try:
        if expression.startswith("!"):
            key = expression[1:].strip()
            return LabelSelectorRequirement(
                key=key,
                operator=LabelSelectorOperator.NOT_EXISTS,
            )
        if " notin " in expression:
            key, raw_values = expression.split(" notin ", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.NOT_IN,
                values=_parse_label_selector_values(raw_values),
            )
        if " in " in expression:
            key, raw_values = expression.split(" in ", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=_parse_label_selector_values(raw_values),
            )
        if "!=" in expression:
            key, value = expression.split("!=", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.NOT_IN,
                values=(value.strip(),),
            )
        if "==" in expression:
            key, value = expression.split("==", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=(value.strip(),),
            )
        if "=" in expression:
            key, value = expression.split("=", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=(value.strip(),),
            )
        return LabelSelectorRequirement(
            key=expression.strip(),
            operator=LabelSelectorOperator.EXISTS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _parse_label_selector_values(raw_values: str) -> tuple[str, ...]:
    raw_values = raw_values.strip()
    if not raw_values.startswith("(") or not raw_values.endswith(")"):
        raise HTTPException(status_code=422, detail="Label selector values must use `(a,b)`.")
    values = tuple(value.strip() for value in raw_values[1:-1].split(","))
    if any(not value for value in values):
        raise HTTPException(status_code=422, detail="Label selector values cannot be blank.")
    return values


def _clean_optional_query_value(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        return require_clean_nonblank(value, field_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _trace_context_metadata(http_request: Request) -> dict[str, Any]:
    # Carry an inbound W3C trace context into the session metadata so an
    # OpenTelemetryEventSink can root the session span under the caller's trace.
    # Used as a shared dependency by every route that starts a traced session.
    metadata: dict[str, Any] = {}
    traceparent = http_request.headers.get("traceparent")
    if traceparent:
        metadata["traceparent"] = traceparent
        tracestate = http_request.headers.get("tracestate")
        if tracestate:
            metadata["tracestate"] = tracestate
    return metadata


TraceContextMetadata = Annotated[dict[str, Any], Depends(_trace_context_metadata)]


def _serialize_message_part(part: Any) -> dict[str, Any]:
    if part.type == "thinking":
        # The opaque round-trip state (Anthropic signatures / redacted blobs) is
        # provider-internal and must not be exposed to transcript API consumers.
        return part.model_dump(mode="json", exclude={"provider_state"})
    return part.model_dump(mode="json")


def _serialize_transcript_message(index: int, message: Message) -> dict[str, Any]:
    return {
        "index": index,
        "role": str(message.role),
        "content": [_serialize_message_part(part) for part in message.content],
    }


def _serialize_task_list_item(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "type": task.type,
        "title": task.title,
        "status": task.status.value,
        "status_reason": task.status_reason,
        "status_payload": task.status_payload,
        "session_id": task.session_id,
        "worker_id": task.worker_id,
        "lease_expires_at": (task.lease_expires_at.isoformat() if task.lease_expires_at else None),
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


def _serialize_task_detail(task: Task) -> dict[str, Any]:
    return {
        **_serialize_task_list_item(task),
        "description": task.description,
        "parent_task_id": task.parent_task_id,
        "assigned_agent_name": task.assigned_agent_name,
        "input": task.input,
        "result": task.result,
        "error": task.error,
        "metadata": task.metadata,
        "updated_at": task.updated_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
    }


def _serialize_knowledge_entry_base(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.id,
        "namespace": entry.namespace,
        "kind": entry.kind,
        "visibility": entry.visibility.value,
        "status": entry.status.value,
        "title": entry.title,
        "labels": dict(entry.labels),
        "aspects": list(entry.aspects),
        "impact_targets": list(entry.impact_targets),
        "source_type": entry.source_type,
        "source_uri": entry.source_uri,
        "source_id": entry.source_id,
        "created_by_type": entry.created_by_type.value,
        "created_by": entry.created_by,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "importance": entry.importance,
        "importance_source": entry.importance_source,
        "confidence": entry.confidence,
    }


def _serialize_knowledge_list_item(item: KnowledgeListItem) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(item.entry),
        "chunk_count": item.chunk_count,
        "text_preview": item.text_preview,
    }


def _serialize_reviewed_knowledge_entry(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(entry),
        "text_preview": _knowledge_text_preview(entry.text),
    }


def _serialize_knowledge_detail(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(entry),
        "text": entry.text,
        "metadata": entry.metadata,
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
    }


def _serialize_knowledge_chunk(chunk: KnowledgeChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "entry_id": chunk.entry_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "source_uri": chunk.source_uri,
        "metadata": dict(chunk.metadata),
    }


def _knowledge_text_preview(text: str) -> str:
    if len(text) <= _KNOWLEDGE_REVIEW_PREVIEW_CHARS:
        return text
    return f"{text[:_KNOWLEDGE_REVIEW_PREVIEW_CHARS]}..."


def _parse_knowledge_label_filters(values: list[str] | None) -> dict[str, str]:
    if values is None:
        return {}
    labels: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise HTTPException(
                status_code=422,
                detail="Knowledge label filters must use `key=value`.",
            )
        key, value = raw.split("=", 1)
        try:
            parsed = copy_label_map({key: value}, "label")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_key, parsed_value = next(iter(parsed.items()))
        if parsed_key in labels:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate knowledge label filter: {parsed_key}",
            )
        labels[parsed_key] = parsed_value
    return labels


def _parse_knowledge_string_filters(values: list[str] | None, field_name: str) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        try:
            result.append(require_clean_nonblank(value, field_name))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return list(dict.fromkeys(result))


def create_router(
    *,
    cayu_app,
    session_store,
    task_store,
    knowledge_store=None,
    knowledge_review_namespace: str | None = None,
    knowledge_review_labels: dict[str, str] | None = None,
    auth: Callable[..., Any] | None = None,
) -> APIRouter:
    """Create an APIRouter with standard cayu endpoints.

    Args:
        auth: Optional FastAPI dependency guarding every state-mutating route
            (run/resume, session interrupt/delete/label/metadata mutation, task
            lifecycle actions, tool-approval resolution/recovery, and knowledge
            review decisions). It is resolved like any FastAPI dependency (it may
            declare ``Request``/header parameters) and must raise
            ``HTTPException`` (401/403) to deny a request; its return value is
            ignored. Read-only routes and ``/health`` stay open so dashboards
            keep working. ``None`` (default) leaves all routes unguarded.
    """

    router = APIRouter(prefix="/api")

    # Shared dependency list for sensitive (state-mutating) routes. FastAPI treats
    # an empty sequence like no dependencies, so `auth=None` keeps current behavior.
    protected: list[Any] = [Depends(auth)] if auth is not None else []

    async def _marker_sequence(session_id: str, event_id: str) -> int | None:
        """Sequence of the persisted event named by a ``Last-Event-ID`` marker.

        Returns ``None`` when the marker event is unknown, so the caller replays the
        full history (at-least-once delivery beats silently dropping events).
        """
        after_sequence: int | None = None
        while True:
            page = await session_store.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    limit=_EVENT_PAGE_LIMIT_MAX,
                )
            )
            for record in page:
                if record.event.id == event_id:
                    return record.sequence
            if len(page) < _EVENT_PAGE_LIMIT_MAX:
                return None
            after_sequence = page[-1].sequence

    async def _replay_events_response(http_request: Request) -> EventSourceResponse | None:
        """SSE resume for reconnecting clients (``Last-Event-ID`` header).

        Instead of starting new work, replay the session's persisted events after the
        last one the client saw and keep following until the session reaches a
        terminal status (the detached pump finishes the run even after a disconnect).
        """
        last_event_id = http_request.headers.get("last-event-id")
        if last_event_id is None:
            return None
        marker = parse_last_event_id(last_event_id)
        if marker is None:
            raise HTTPException(
                status_code=422,
                detail="Last-Event-ID must use `session_id:event_id`.",
            )
        session_id, last_seen_event_id = marker
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        async def replay() -> AsyncIterator[dict[str, str]]:
            after_sequence = await _marker_sequence(session_id, last_seen_event_id)
            while True:
                page = await session_store.query_events(
                    EventQuery(
                        session_id=session_id,
                        after_sequence=after_sequence,
                        limit=_EVENT_PAGE_LIMIT_MAX,
                    )
                )
                for record in page:
                    after_sequence = record.sequence
                    yield event_to_sse_message(record.event)
                if len(page) == _EVENT_PAGE_LIMIT_MAX:
                    continue
                current = await session_store.load(session_id)
                if current is None or current.status not in _REPLAY_ACTIVE_SESSION_STATUSES:
                    return
                await asyncio.sleep(_REPLAY_POLL_INTERVAL_S)

        return EventSourceResponse(replay())

    @router.post("/run", dependencies=protected)
    async def run_agent(
        body: RunBody,
        http_request: Request,
        trace_metadata: TraceContextMetadata,
    ):
        replay = await _replay_events_response(http_request)
        if replay is not None:
            return replay
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
            causal_budget_id=body.causal_budget_id,
            task_id=task_id,
            labels=body.labels,
            messages=[Message.text("user", body.prompt)],
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            metadata=trace_metadata,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.run(request))

    @router.post("/resume", dependencies=protected)
    async def resume_agent(
        body: ResumeBody,
        http_request: Request,
        trace_metadata: TraceContextMetadata,
    ):
        replay = await _replay_events_response(http_request)
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ResumeRequest(
            session_id=body.session_id,
            messages=[Message.text("user", body.prompt)],
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            metadata=trace_metadata,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.resume(request))

    @router.post("/sessions/{session_id}/interrupt", dependencies=protected)
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
            yield event_to_sse_message(first_event)
            async for event in event_stream:
                yield event_to_sse_message(event)

        return EventSourceResponse(generate())

    @router.post("/tool-approvals/resolve", dependencies=protected)
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
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.resolve_tool_approval(request))

    @router.post("/tool-approvals/recover", dependencies=protected)
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
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.recover_tool_approval(request))

    @router.get("/sessions")
    async def list_sessions(
        limit: Annotated[int, Query(ge=1, le=1000)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        cursor: Annotated[str | None, Query()] = None,
        status: SessionStatus | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC,
        label: Annotated[list[str] | None, Query()] = None,
        label_selector: Annotated[list[str] | None, Query()] = None,
    ):
        labels = _parse_session_label_filters(label)
        label_selectors = _parse_session_label_selectors(label_selector)
        try:
            result = await session_store.list_sessions(
                SessionQuery(
                    status=status,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    environment_name=_clean_optional_query_value(
                        environment_name,
                        "environment_name",
                    ),
                    parent_session_id=_clean_optional_query_value(
                        parent_session_id,
                        "parent_session_id",
                    ),
                    causal_budget_id=_clean_optional_query_value(
                        causal_budget_id,
                        "causal_budget_id",
                    ),
                    labels=labels,
                    label_selectors=label_selectors,
                    limit=limit,
                    offset=offset,
                    cursor=cursor,
                    include_total_count=True,
                    order_by=order_by,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "sessions": [_serialize_session_base(session) for session in result.sessions],
            "next_cursor": result.next_cursor,
            "total_count": result.total_count,
        }

    @router.post("/sessions/summary")
    async def get_sessions_summary(
        body: SessionsSummaryBody | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
        offset: Annotated[int, Query(ge=0)] = 0,
        status: SessionStatus | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC,
        label: Annotated[list[str] | None, Query()] = None,
        label_selector: Annotated[list[str] | None, Query()] = None,
    ):
        body = body or SessionsSummaryBody()
        labels = _parse_session_label_filters(label)
        label_selectors = _parse_session_label_selectors(label_selector)
        sessions = (
            await session_store.list_sessions(
                SessionQuery(
                    status=status,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    environment_name=_clean_optional_query_value(
                        environment_name,
                        "environment_name",
                    ),
                    parent_session_id=_clean_optional_query_value(
                        parent_session_id,
                        "parent_session_id",
                    ),
                    causal_budget_id=_clean_optional_query_value(
                        causal_budget_id,
                        "causal_budget_id",
                    ),
                    labels=labels,
                    label_selectors=label_selectors,
                    limit=limit,
                    offset=offset,
                    order_by=order_by,
                )
            )
        ).sessions
        session_event_records_by_id: dict[str, list[EventRecord]] = {}
        session_ids = [session.id for session in sessions]
        all_event_records = await _query_all_session_event_records(session_ids)
        for record in all_event_records:
            session_event_records_by_id.setdefault(record.event.session_id, []).append(record)
        for session in sessions:
            session_event_records_by_id.setdefault(session.id, [])

        usage_event_records = [
            record
            for record in all_event_records
            if record.event.type in {EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED}
        ]
        usage_events = [
            record.event
            for record in sorted(usage_event_records, key=lambda record: record.sequence)
        ]
        usage_summary = causal_budget_usage_summary(
            causal_budget_id="session-query",
            session_ids=session_ids,
            events=usage_events,
        ).model_dump()
        usage_summary.pop("causal_budget_id", None)

        cost_summary = None
        if body.pricing is not None:
            model_events = [
                record.event
                for record in sorted(all_event_records, key=lambda record: record.sequence)
                if record.event.type == EventType.MODEL_COMPLETED
            ]
            aggregate_cost = build_session_cost_summary(
                session_id="session-query",
                events=model_events,
                pricing=body.pricing,
                currency=body.currency,
            ).model_dump(mode="json")
            aggregate_cost.pop("session_id", None)
            aggregate_cost["session_ids"] = session_ids
            aggregate_cost["session_count"] = len(session_ids)
            aggregate_cost["session_costs"] = [
                build_session_cost_summary(
                    session_id=session.id,
                    events=[
                        record.event
                        for record in session_event_records_by_id[session.id]
                        if record.event.type == EventType.MODEL_COMPLETED
                    ],
                    pricing=body.pricing,
                    currency=body.currency,
                ).model_dump(mode="json")
                for session in sessions
            ]
            cost_summary = aggregate_cost

        session_items = []
        for session in sessions:
            records = session_event_records_by_id[session.id]
            outcome = session_outcome_from_records(session, records)
            event_summary = event_summary_from_records(session.id, records)
            session_items.append(
                {
                    "session": _serialize_session(session),
                    "outcome": _serialize_session_outcome(outcome),
                    "events": {
                        "total_events": event_summary.total_events,
                        "counts_by_type": event_summary.counts_by_type,
                        "latest_event": (
                            None
                            if event_summary.latest_event is None
                            else _serialize_event_record(event_summary.latest_event)
                        ),
                    },
                }
            )

        return {
            "session_count": len(sessions),
            "sessions": session_items,
            "usage": usage_summary,
            "cost": cost_summary,
        }

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

    @router.get("/causal-budgets/{causal_budget_id}/usage")
    async def get_causal_budget_usage(causal_budget_id: NonBlankString):
        try:
            summary = await cayu_app.get_causal_budget_usage(causal_budget_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Causal budget not found") from exc
        return summary.model_dump()

    @router.post("/causal-budgets/{causal_budget_id}/cost")
    async def estimate_causal_budget_cost(
        causal_budget_id: NonBlankString,
        body: SessionCostBody,
    ):
        try:
            summary = await cayu_app.get_causal_budget_cost(
                causal_budget_id,
                body.pricing,
                currency=body.currency,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Causal budget not found") from exc
        return summary.model_dump(mode="json")

    @router.post("/causal-budgets/{causal_budget_id}/summary")
    async def get_causal_budget_summary(
        causal_budget_id: NonBlankString,
        body: SessionCostBody,
    ):
        sessions = await _list_all_causal_sessions(causal_budget_id)
        if not sessions:
            raise HTTPException(status_code=404, detail="Causal budget not found")

        session_ids = [session.id for session in sessions]
        causal_event_records = await _query_all_causal_event_records(causal_budget_id)
        usage_event_records = [
            record
            for record in causal_event_records
            if record.event.type == EventType.MODEL_COMPLETED
        ]
        tool_event_records = [
            record
            for record in causal_event_records
            if record.event.type == EventType.TOOL_CALL_STARTED
        ]
        usage_events = [
            record.event
            for record in sorted(
                [*usage_event_records, *tool_event_records],
                key=lambda record: record.sequence,
            )
        ]
        usage_summary = causal_budget_usage_summary(
            causal_budget_id=causal_budget_id,
            session_ids=session_ids,
            events=usage_events,
        )
        cost_summary = build_causal_budget_cost_summary(
            causal_budget_id=causal_budget_id,
            session_ids=session_ids,
            events=[record.event for record in usage_event_records],
            pricing=body.pricing,
            currency=body.currency,
        )
        session_items = []
        for session in sessions:
            session_event_records = [
                record for record in causal_event_records if record.event.session_id == session.id
            ]
            outcome = session_outcome_from_records(
                session,
                session_event_records,
            )
            event_summary = event_summary_from_records(
                session.id,
                session_event_records,
            )
            session_items.append(
                {
                    "session": _serialize_session(session),
                    "outcome": _serialize_session_outcome(outcome),
                    "events": {
                        "total_events": event_summary.total_events,
                        "counts_by_type": event_summary.counts_by_type,
                        "latest_event": (
                            None
                            if event_summary.latest_event is None
                            else _serialize_event_record(event_summary.latest_event)
                        ),
                    },
                }
            )

        return {
            "causal_budget_id": causal_budget_id,
            "session_count": len(sessions),
            "sessions": session_items,
            "usage": usage_summary.model_dump(),
            "cost": cost_summary.model_dump(mode="json"),
        }

    @router.get("/sessions/{session_id}/summary")
    async def get_session_summary(session_id: NonBlankString):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        event_summary = await session_store.summarize_events(session_id)
        outcome = await session_store.summarize_outcome(session_id)
        transcript_page = await session_store.query_transcript(
            TranscriptQuery(session_id=session_id, limit=1)
        )
        usage_summary = await cayu_app.get_session_usage(session_id)

        return {
            "session": _serialize_session(session),
            "events": {
                "total_events": event_summary.total_events,
                "counts_by_type": event_summary.counts_by_type,
                "latest_event": (
                    None
                    if event_summary.latest_event is None
                    else _serialize_event_record(event_summary.latest_event)
                ),
            },
            "transcript": {
                "total_messages": transcript_page.total_records,
            },
            "outcome": _serialize_session_outcome(outcome),
            "usage": usage_summary.model_dump(),
        }

    async def _list_all_causal_sessions(causal_budget_id: str) -> list[Session]:
        sessions: list[Session] = []
        offset = 0
        while True:
            page = (
                await session_store.list_sessions(
                    SessionQuery(
                        causal_budget_id=causal_budget_id,
                        limit=1000,
                        offset=offset,
                        order_by=SessionOrder.CREATED_AT_ASC,
                    )
                )
            ).sessions
            if not page:
                return sessions
            sessions.extend(page)
            if len(page) < 1000:
                return sessions
            offset += len(page)

    async def _query_all_session_event_records(session_ids: list[str]) -> list[EventRecord]:
        if not session_ids:
            return []
        records: list[EventRecord] = []
        after_sequence = None
        while True:
            page = await session_store.query_events(
                EventQuery(
                    session_ids=tuple(session_ids),
                    after_sequence=after_sequence,
                    limit=5000,
                )
            )
            if not page:
                return records
            records.extend(page)
            if len(page) < 5000:
                return records
            after_sequence = page[-1].sequence

    async def _query_all_causal_event_records(causal_budget_id: str) -> list[EventRecord]:
        records: list[EventRecord] = []
        after_sequence = None
        while True:
            page = await session_store.query_events(
                EventQuery(
                    causal_budget_id=causal_budget_id,
                    after_sequence=after_sequence,
                    limit=5000,
                )
            )
            if not page:
                return records
            records.extend(page)
            if len(page) < 5000:
                return records
            after_sequence = page[-1].sequence

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
        include_thinking: bool = Query(default=True),
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
                include_thinking=include_thinking,
            )
        )
        # Advance by the queried window size, not the returned record count: the
        # include_thinking filter can drop thinking-only records from a page, so
        # len(records) under-counts the messages consumed and would stall pagination.
        consumed = min(limit, max(0, transcript_page.total_records - offset))
        next_offset = offset + consumed

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
                "causal_budget_id": session.causal_budget_id,
                "runtime_name": session.runtime_name,
                "runtime_version": session.runtime_version,
                "labels": session.labels,
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
                    "content": [_serialize_message_part(p) for p in m.content],
                }
                for m in transcript
            ],
        }

    @router.delete("/sessions/{session_id}", status_code=204, dependencies=protected)
    async def delete_session(session_id: NonBlankString):
        try:
            await session_store.delete_session(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return None

    @router.patch("/sessions/{session_id}/labels", dependencies=protected)
    async def update_session_labels(
        session_id: NonBlankString,
        body: UpdateSessionLabelsBody,
    ):
        try:
            session = await session_store.update_labels(session_id, body.labels)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_session(session)

    @router.patch("/sessions/{session_id}/metadata", dependencies=protected)
    async def update_session_metadata(
        session_id: NonBlankString,
        body: UpdateSessionMetadataBody,
    ):
        try:
            session = await session_store.update_metadata(session_id, body.metadata)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_session(session)

    @router.get("/tasks")
    async def list_tasks(
        status: TaskStatus | None = None,
        task_type: str | None = Query(default=None, alias="type"),
        session_id: str | None = None,
        parent_task_id: str | None = None,
        assigned_agent_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        if task_store is None:
            return []
        try:
            query = TaskQuery(
                status=status,
                type=task_type,
                session_id=session_id,
                parent_task_id=parent_task_id,
                assigned_agent_name=assigned_agent_name,
                limit=limit,
                offset=offset,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        tasks = await task_store.list_tasks(query)
        return [_serialize_task_list_item(t) for t in tasks]

    async def _require_task_store():
        if task_store is None:
            raise HTTPException(status_code=404, detail="Task store is not configured.")
        return task_store

    async def _apply_task_action(action, task_id: str):
        try:
            task = await action(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _serialize_task_detail(task)

    @router.post("/tasks/{task_id}/pause", dependencies=protected)
    async def pause_task(task_id: NonBlankString, body: TaskHoldBody | None = None):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.pause_task(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post("/tasks/{task_id}/block", dependencies=protected)
    async def block_task(task_id: NonBlankString, body: TaskHoldBody | None = None):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.block_task(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post("/tasks/{task_id}/needs-attention", dependencies=protected)
    async def mark_task_needs_attention(
        task_id: NonBlankString,
        body: TaskHoldBody | None = None,
    ):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.mark_task_needs_attention(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post("/tasks/{task_id}/resume", dependencies=protected)
    async def resume_task(task_id: NonBlankString):
        store = await _require_task_store()
        return await _apply_task_action(store.resume_task, task_id)

    def _knowledge_review_workflow() -> KnowledgeReviewWorkflow:
        if knowledge_store is None:
            raise HTTPException(status_code=404, detail="Knowledge store is not configured.")
        return KnowledgeReviewWorkflow(
            knowledge_store,
            namespace=knowledge_review_namespace,
            labels=knowledge_review_labels,
            default_limit=50,
        )

    async def _apply_knowledge_review_action(action, entry_id: str):
        try:
            entry = await action(entry_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _serialize_reviewed_knowledge_entry(entry)

    @router.get("/knowledge/pending")
    async def list_pending_knowledge(
        namespace: str | None = None,
        label: Annotated[list[str] | None, Query()] = None,
        kind: Annotated[list[str] | None, Query()] = None,
        aspect: Annotated[list[str] | None, Query()] = None,
        visibility: Annotated[list[KnowledgeVisibility] | None, Query()] = None,
        source_type: str | None = None,
        source_id: str | None = None,
        limit: int = 50,
        max_bytes: int = 20_000,
    ):
        workflow = _knowledge_review_workflow()
        try:
            result = await workflow.list_pending(
                namespace=_clean_optional_query_value(namespace, "namespace"),
                labels=_parse_knowledge_label_filters(label),
                kinds=_parse_knowledge_string_filters(kind, "kind") if kind is not None else None,
                visibilities=visibility,
                aspects=_parse_knowledge_string_filters(aspect, "aspect"),
                source_type=_clean_optional_query_value(source_type, "source_type"),
                source_id=_clean_optional_query_value(source_id, "source_id"),
                limit=limit,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "entries": [_serialize_knowledge_list_item(item) for item in result.entries],
            "truncated": result.truncated,
            "limit": result.limit,
            "max_bytes": result.max_bytes,
            "total_entries_known": result.total_entries_known,
        }

    @router.get("/knowledge/pending/{entry_id}")
    async def get_pending_knowledge(
        entry_id: NonBlankString,
        max_chunks: Annotated[
            int,
            Query(ge=1, le=_KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS),
        ] = _KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS,
        max_bytes: Annotated[
            int,
            Query(ge=1, le=_KNOWLEDGE_PENDING_DETAIL_MAX_BYTES),
        ] = _KNOWLEDGE_PENDING_DETAIL_MAX_BYTES,
    ):
        workflow = _knowledge_review_workflow()
        assert knowledge_store is not None
        try:
            entry = await workflow.get_pending(entry_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            chunks = await knowledge_store.read_chunks(
                entry.id,
                max_chunks=max_chunks,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            **_serialize_knowledge_detail(entry),
            "chunks": [_serialize_knowledge_chunk(chunk) for chunk in chunks],
            "chunk_limit": max_chunks,
            "chunk_max_bytes": max_bytes,
        }

    @router.post("/knowledge/{entry_id}/approve", dependencies=protected)
    async def approve_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.approve, entry_id)

    @router.post("/knowledge/{entry_id}/reject", dependencies=protected)
    async def reject_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.reject, entry_id)

    @router.get("/health")
    async def health():
        return {"ok": True}

    return router
