"""API routes for the cayu server."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, Literal
from unicodedata import category as unicode_category
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
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

from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts import (
    ArtifactListResult,
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    copy_artifact_read_result,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole
from cayu.core.thinking import ThinkingConfig
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.approvals import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.costs import CausalBudgetCostSummary, PricingCatalog, SessionCostSummary
from cayu.runtime.costs import (
    estimate_causal_budget_cost as build_causal_budget_cost_summary,
)
from cayu.runtime.costs import (
    estimate_session_cost as build_session_cost_summary,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    EventRecord,
    InterruptSessionRequest,
    LabelSelectorOperator,
    LabelSelectorRequirement,
    ResumeRequest,
    RunRequest,
    Session,
    SessionDebugState,
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
from cayu.runtime.tasks import Task, TaskCreate, TaskOrder, TaskQuery, TaskStatus
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest
from cayu.runtime.usage import (
    CacheUsageMetrics,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    UsageMetrics,
    causal_budget_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.runtime.user_input import (
    UserInputRecoveryRequest,
    UserInputResponse,
    pending_user_input_from_checkpoint,
)
from cayu.server.auth import AuthContext, AuthDependency, server_auth_dependency
from cayu.server.contracts import (
    ARTIFACT_CONTENT_ENDPOINT_RESPONSES,
    ARTIFACT_ENDPOINT_ERROR_RESPONSES,
    SERVER_API_PREFIX,
    STREAMING_ENDPOINT_RESPONSES,
    AgentsResponse,
    ApiReviewedKnowledgeEntry,
    ApiSession,
    ApiTaskDetail,
    ApiTaskListItem,
    ArtifactReadResponse,
    ArtifactsResponse,
    CausalBudgetSummaryResponse,
    ClientGenerationContract,
    EnvironmentsResponse,
    HealthResponse,
    ListSessionEventsResponse,
    ListSessionsResponse,
    PendingActionsResponse,
    PendingKnowledgeDetailResponse,
    PendingKnowledgeListResponse,
    ServerContractResponse,
    SessionDetailResponse,
    SessionsSummaryResponse,
    SessionStateResponse,
    SessionSummaryResponse,
    SessionTranscriptResponse,
    UsageBreakdownItem,
)
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
ArtifactIdPath = Annotated[str, StringConstraints(min_length=1)]
# Server-entrypoint step budget. The default preserves the historical value while the
# ceiling matches the runtime's own ``max_steps`` bound (RunRequest/ResumeRequest and the
# tool-approval bodies all cap at 256) so a request cannot ask for an unbounded run.
_DEFAULT_RUN_MAX_STEPS = 20
_MAX_RUN_STEPS = 256
_EVENT_PAGE_LIMIT_MAX = 1000
_TRANSCRIPT_PAGE_LIMIT_MAX = 1000
_ARTIFACT_PAGE_LIMIT_MAX = 500
_ARTIFACT_PAGE_OFFSET_MAX = 10_000
_ARTIFACT_CONTENT_BYTES_MAX = 64 * 1024 * 1024
_ARTIFACT_FILENAME_HEADER_UTF8_MAX_BYTES = 512
_ARTIFACT_FILENAME_HEADER_ASCII_MAX_CHARS = 255
_ARTIFACT_ID_HEADER_MAX_CHARS = 512
_ARTIFACT_UNSAFE_FILENAME_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
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
_PENDING_ACTION_EVENT_TYPES = (
    EventType.SESSION_RESUMED,
    EventType.SESSION_COMPLETED,
    EventType.SESSION_FAILED,
    EventType.SESSION_INTERRUPTED,
    EventType.SESSION_AWAITING_USER_INPUT,
    EventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_BLOCKED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_REQUESTED,
)
_PENDING_ACTION_SESSION_STATUSES = frozenset(
    {
        SessionStatus.INTERRUPTED,
        SessionStatus.FAILED,
    }
)
# Replays check quickly after reconnect, then back off while a live session is
# quiet so idle streams do not continuously hammer the durable stores.
_REPLAY_POLL_INTERVAL_MIN_S = 0.05
_REPLAY_POLL_INTERVAL_MAX_S = 1.0


def _next_replay_poll_interval(current: float, *, received_events: bool) -> float:
    if received_events:
        return _REPLAY_POLL_INTERVAL_MIN_S
    return min(current * 2, _REPLAY_POLL_INTERVAL_MAX_S)


# Detached event pumps must outlive their SSE consumer (a client disconnect must not
# cancel agent work), so hold strong references until each pump finishes — the event
# loop only keeps weak references to tasks.
_detached_event_pumps: set[asyncio.Task[None]] = set()
_ARTIFACT_SAFE_INLINE_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/pdf",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
        "text/csv",
        "text/markdown",
        "text/plain",
    }
)


async def _fail_task_on_prestream_error(
    event_stream: AsyncIterator[Event],
    *,
    task_store: Any,
    task_id: str,
) -> AsyncIterator[Event]:
    """Fail the route-created task when the run dies before its first event.

    A pre-session failure (request validation, unknown agent, unsupported
    native structured output) would otherwise strand the task as ``pending``
    with no session ever attached. Once a first event exists, the session owns
    the task lifecycle and failures are recorded through it.
    """
    emitted = False
    try:
        async for event in event_stream:
            emitted = True
            yield event
    except BaseException as exc:
        if not emitted:
            with contextlib.suppress(Exception):
                await task_store.fail_task(
                    task_id,
                    {"error": str(exc), "error_type": type(exc).__name__},
                )
        raise


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
    model: NonBlankString | None = None
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
    requested_by: ResolutionActor | None = None


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
    resolved_by: ResolutionActor | None = None
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
    resolved_by: ResolutionActor | None = None
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


class ToolRoundRecoveryBody(BaseModel):
    """Body for recovering a crashed ordinary tool call with an operator outcome.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run applies the runtime defaults (a pending tool
    round persists no run configuration). Explicit values override them.
    """

    session_id: NonBlankString
    round_id: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
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


class UserInputResolveBody(BaseModel):
    """Body for answering a session paused by ``ask_user``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``: the
    resumed run inherits the original run's configuration persisted on the pending user input.
    """

    session_id: NonBlankString
    input_id: NonBlankString
    answer: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
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


class UserInputRecoveryBody(BaseModel):
    """Body for recovering a user-input round stuck on ``manual_recovery_required``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``: the
    resumed run inherits the original run's configuration persisted on the pending user input.
    """

    session_id: NonBlankString
    input_id: NonBlankString
    answer: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
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


def _request_actor(
    auth_context: AuthContext | None,
    body_actor: ResolutionActor | None,
    *,
    field_name: Literal["requested_by", "resolved_by"],
) -> ResolutionActor | None:
    """Derive a typed operator actor for an authenticated control-plane route.

    With auth configured, provenance comes from the verified caller and a
    body-supplied actor is rejected loudly (mirroring the reserved ``cayu:``
    label rejection) — a silent override would let clients believe they
    recorded an actor the audit trail replaced. Dev-mode bodies are accepted
    but re-stamped ``source="request"``, so a request can never claim
    server-verified (``http_auth``) or system provenance.
    """
    if auth_context is not None:
        if body_actor is not None:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} is derived from the authenticated caller and "
                "cannot be supplied in the request body.",
            )
        try:
            return ResolutionActor(
                subject=auth_context.subject,
                tenant=auth_context.tenant,
                source=ResolutionActorSource.HTTP_AUTH,
                claims=auth_context.claims,
            )
        except ValueError as exc:
            # AuthContext.subject is unconstrained, so an auth backend can hand
            # back a reserved ``cayu:``-prefixed subject; surface that as a
            # clean 400 instead of an unhandled 500.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body_actor is None:
        return None
    try:
        return ResolutionActor(
            subject=body_actor.subject,
            tenant=body_actor.tenant,
            source=ResolutionActorSource.REQUEST,
            claims=body_actor.claims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _request_resolution_actor(
    auth_context: AuthContext | None,
    body_resolved_by: ResolutionActor | None,
) -> ResolutionActor | None:
    return _request_actor(auth_context, body_resolved_by, field_name="resolved_by")


def _request_interruption_actor(
    auth_context: AuthContext | None,
    body_requested_by: ResolutionActor | None,
) -> ResolutionActor | None:
    return _request_actor(auth_context, body_requested_by, field_name="requested_by")


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


def _redact_control_plane_json(cayu_app: Any, value: Any, field_name: str) -> Any:
    copied = copy_json_value(value, field_name)
    redactor = getattr(cayu_app, "redact_json", None)
    if callable(redactor):
        return redactor(copied)
    return copied


def _serialize_tool(cayu_app: Any, tool: Any) -> dict[str, Any]:
    effect = getattr(tool.effect, "value", str(tool.effect))
    return {
        "name": tool.name,
        "description": _redact_control_plane_json(cayu_app, tool.description, "description"),
        "input_schema": _redact_control_plane_json(cayu_app, tool.schema, "input_schema"),
        "parallel_safe": tool.parallel_safe,
        "effect": effect,
    }


def _serialize_agent(cayu_app: Any, agent: Any) -> dict[str, Any]:
    spec = agent.spec
    thinking = (
        None
        if spec.thinking is None
        else _redact_control_plane_json(cayu_app, spec.thinking.model_dump(mode="json"), "thinking")
    )
    tools = [_serialize_tool(cayu_app, tool) for tool in agent.tools.values()]
    return {
        "name": spec.name,
        "provider_name": spec.provider_name,
        "model": spec.model,
        "tool_count": len(tools),
        "tools": sorted(tools, key=lambda item: item["name"]),
        "metadata": _redact_control_plane_json(cayu_app, spec.metadata, "metadata"),
        "provider_options": _redact_control_plane_json(
            cayu_app,
            spec.provider_options,
            "provider_options",
        ),
        "thinking": thinking,
        "has_system_prompt": spec.system_prompt is not None and bool(spec.system_prompt.strip()),
    }


def _object_type_name(value: Any) -> str | None:
    if value is None:
        return None
    return type(value).__name__


def _object_id(value: Any) -> str | None:
    if value is None:
        return None
    object_id = getattr(value, "id", None)
    return object_id if isinstance(object_id, str) and object_id.strip() else None


def _workspace_instruction_summary(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return "inline"
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return "inline"
    mode = getattr(value, "mode", None)
    if isinstance(mode, str):
        return mode
    return type(value).__name__


def _serialize_environment(cayu_app: Any, record: Any) -> dict[str, Any]:
    environment = record.environment
    workspace = environment.workspace
    artifact_store = environment.artifact_store
    bound_workspace = record.bound_workspace
    bound_payload = None
    if bound_workspace is not None:
        bound_payload = {
            "source_workspace_id": _object_id(bound_workspace.source_workspace),
            "bound_workspace_id": _object_id(bound_workspace.workspace),
            "runner_type": _object_type_name(bound_workspace.runner),
            "path": bound_workspace.path,
            "metadata": _redact_control_plane_json(
                cayu_app,
                bound_workspace.metadata,
                "metadata",
            ),
        }
    return {
        "name": record.spec.name,
        "metadata": _redact_control_plane_json(cayu_app, record.spec.metadata, "metadata"),
        "is_factory": record.factory is not None,
        "workspace_id": _object_id(workspace),
        "artifact_store_id": _object_id(artifact_store),
        "runner_type": _object_type_name(environment.runner),
        "binding_type": _object_type_name(environment.binding),
        "vault_type": _object_type_name(environment.vault),
        "proxy_type": _object_type_name(environment.proxy),
        "knowledge_store_type": _object_type_name(environment.knowledge_store),
        "mcp_server_count": len(environment.mcp_servers),
        "workspace_instructions": _workspace_instruction_summary(
            environment.workspace_instructions
        ),
        "bound_workspace": bound_payload,
    }


def _artifact_stores_by_id(cayu_app: Any) -> dict[str, ArtifactStore]:
    stores: dict[str, ArtifactStore] = {}
    for record in cayu_app.list_environment_registrations():
        store = record.environment.artifact_store
        if isinstance(store, ArtifactStore):
            try:
                store_id = require_clean_nonblank(store.id, "artifact_store.id")
                store_id = require_unicode_scalar_text(store_id, "artifact_store.id")
            except ValueError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="An artifact store has an invalid id configuration.",
                ) from exc
            existing = stores.get(store_id)
            if existing is not None and existing is not store:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Multiple registered environments use the same artifact_store_id: "
                        f"{store_id}. Configure unique artifact store ids."
                    ),
                )
            stores[store_id] = store
    return stores


def _serialize_artifact(cayu_app: Any, metadata: Any, *, artifact_store_id: str) -> dict[str, Any]:
    return {
        "id": metadata.id,
        "artifact_store_id": artifact_store_id,
        "filename": metadata.filename,
        "content_type": metadata.content_type,
        "size_bytes": metadata.size_bytes,
        "scope": metadata.scope.value,
        "session_id": metadata.session_id,
        "agent_name": metadata.agent_name,
        "environment_name": metadata.environment_name,
        "created_at": metadata.created_at.isoformat(),
        "metadata": _redact_control_plane_json(cayu_app, metadata.metadata, "metadata"),
    }


def _artifact_sort_key(artifact: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(artifact["created_at"]),
        str(artifact["artifact_store_id"]),
        str(artifact["id"]),
    )


def _decode_artifact_text(content: bytes, content_type: str) -> str | None:
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type.startswith("text/") or normalized_content_type in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    }:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="replace")
    return None


def _artifact_read_preview(cayu_app: Any, read: Any) -> tuple[str, str | None]:
    text_preview = _decode_artifact_text(read.content, read.metadata.content_type)
    if text_preview is None:
        return base64.b64encode(read.content).decode("ascii"), None
    redacted_preview = _redact_control_plane_json(cayu_app, text_preview, "artifact.content")
    if not isinstance(redacted_preview, str):
        raise TypeError("Artifact text preview redaction must return a string.")
    return base64.b64encode(redacted_preview.encode("utf-8")).decode("ascii"), redacted_preview


def _artifact_content_disposition(filename: str, disposition: str) -> str:
    safe_filename = "".join(
        "_"
        if char in {"/", "\\"}
        or unicode_category(char) in _ARTIFACT_UNSAFE_FILENAME_UNICODE_CATEGORIES
        else char
        for char in filename
    ).strip()
    if not safe_filename:
        safe_filename = "artifact"
    safe_filename = _truncate_utf8_filename(
        safe_filename,
        max_bytes=_ARTIFACT_FILENAME_HEADER_UTF8_MAX_BYTES,
    )
    ascii_filename = "".join(
        char if 0x20 <= ord(char) < 0x7F and char not in {'"', "/", "\\"} else "_"
        for char in safe_filename
    ).strip()
    if not ascii_filename:
        ascii_filename = "artifact"
    ascii_filename = ascii_filename[:_ARTIFACT_FILENAME_HEADER_ASCII_MAX_CHARS]
    encoded_filename = quote(safe_filename, safe="", encoding="utf-8", errors="replace")
    return f"{disposition}; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"


def _truncate_utf8_filename(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    stem, separator, extension = value.rpartition(".")
    if not separator:
        stem = value
    suffix = f".{extension}" if separator and 0 < len(extension) <= 32 else ""
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        suffix = ""
        suffix_bytes = b""
        stem = value
    prefix_bytes = stem.encode("utf-8")[: max_bytes - len(suffix_bytes)]
    prefix = prefix_bytes.decode("utf-8", errors="ignore")
    return f"{prefix}{suffix}" or "artifact"


def _artifact_content_disposition_kind(content_type: str, requested: str) -> str:
    if requested != "inline":
        return "attachment"
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type in _ARTIFACT_SAFE_INLINE_CONTENT_TYPES:
        return "inline"
    return "attachment"


def _artifact_header_value(value: str, fallback: str) -> str:
    for candidate in (value, fallback, "unknown"):
        stripped = candidate.strip()
        if stripped and all(0x20 <= ord(char) < 0x7F for char in stripped):
            return stripped[:_ARTIFACT_ID_HEADER_MAX_CHARS]
    return "unknown"


def _object_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def _optional_payload_string(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _payload_string_list(payload: dict[str, Any] | None, key: str) -> list[str]:
    if payload is None:
        return []
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _pending_action_matches_query(action: dict[str, Any], q: str | None) -> bool:
    if q is None:
        return True
    needle = q.lower()
    session = _object_payload(action.get("session")) or {}
    values = [
        action.get("id"),
        action.get("kind"),
        action.get("title"),
        action.get("detail"),
        action.get("tool_name"),
        action.get("approval_id"),
        action.get("input_id"),
        action.get("round_id"),
        action.get("tool_call_id"),
        action.get("question"),
        session.get("id"),
        session.get("agent_name"),
        session.get("provider_name"),
        session.get("model"),
        session.get("environment_name"),
    ]
    return any(isinstance(value, str) and needle in value.lower() for value in values)


def _pending_action_from_event_record(
    *,
    session: Session,
    record: EventRecord,
    action_kind: str,
    title: str,
    detail: str | None = None,
    tool_name: str | None = None,
    approval_id: str | None = None,
    input_id: str | None = None,
    round_id: str | None = None,
    tool_call_id: str | None = None,
    question: str | None = None,
    options: list[str] | None = None,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    discriminator = approval_id or input_id or tool_call_id or record.event.id
    return {
        "id": f"{session.id}:{record.sequence}:{action_kind}:{discriminator}",
        "kind": action_kind,
        "session": _serialize_session_base(session),
        "event": _serialize_event_record(record),
        "title": title,
        "detail": detail,
        "tool_name": tool_name,
        "approval_id": approval_id,
        "input_id": input_id,
        "round_id": round_id,
        "tool_call_id": tool_call_id,
        "question": question,
        "options": options or [],
        "arguments": arguments,
    }


def _pending_approval_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    approval_id: str,
    tool_call_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        pending = approval_support.pending_approval_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.approval_id != approval_id:
        return None
    if tool_call_id is None:
        return {
            "tool_name": pending.tool_name,
            "arguments": copy_json_value(pending.arguments, "arguments"),
        }
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": copy_json_value(call.arguments, "arguments"),
            }
    if pending.tool_call_id == tool_call_id:
        return {
            "tool_name": pending.tool_name,
            "arguments": copy_json_value(pending.arguments, "arguments"),
        }
    return None


def _pending_user_input_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    input_id: str,
    tool_call_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        pending = pending_user_input_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.input_id != input_id:
        return None
    if tool_call_id is None:
        return {
            "tool_name": pending.tool_name,
            "arguments": copy_json_value(pending.arguments, "arguments"),
        }
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": copy_json_value(call.arguments, "arguments"),
            }
    if pending.tool_call_id == tool_call_id:
        return {
            "tool_name": pending.tool_name,
            "arguments": copy_json_value(pending.arguments, "arguments"),
        }
    return None


def _pending_tool_round_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    round_id: str,
    tool_call_id: str,
) -> dict[str, Any] | None:
    try:
        pending = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.round_id != round_id:
        return None
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": copy_json_value(call.arguments, "arguments"),
            }
    return None


def _pending_tool_round_manual_recovery_action(
    session: Session,
    records_desc: list[EventRecord],
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending_round is None:
        return None

    events = [record.event for record in reversed(records_desc)]
    recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=events,
        pending_round=pending_round,
    )
    unresolved_calls = [
        call
        for call in pending_round.tool_calls
        if call.tool_call_id in started_ids and call.tool_call_id not in recorded_outcomes
    ]
    if not unresolved_calls:
        return None

    pending_call = unresolved_calls[0]
    source_record = next(
        (
            record
            for record in records_desc
            if record.event.type in {EventType.SESSION_INTERRUPTED, EventType.SESSION_FAILED}
            and (
                record.event.payload.get("tool_round_id") in {None, pending_round.round_id}
                or record.event.payload.get("manual_recovery_required") is True
            )
        ),
        None,
    )
    if source_record is None:
        source_record = next(
            (
                record
                for record in records_desc
                if record.event.type == EventType.TOOL_CALL_STARTED
                and record.event.payload.get("tool_round_id") == pending_round.round_id
                and record.event.payload.get("tool_call_id") == pending_call.tool_call_id
            ),
            None,
        )
    if source_record is None:
        return None

    detail = (
        "Tool started but no terminal result was recorded before the session failed."
        if session.status == SessionStatus.FAILED
        else "Tool started but no terminal result was recorded."
    )
    return _pending_action_from_event_record(
        session=session,
        record=source_record,
        action_kind="manual_recovery",
        title="Manual recovery required",
        detail=detail,
        tool_name=pending_call.tool_name,
        round_id=pending_round.round_id,
        tool_call_id=pending_call.tool_call_id,
        arguments=copy_json_value(pending_call.arguments, "arguments"),
    )


def _pending_action_from_records(
    session: Session,
    records_desc: list[EventRecord],
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if session.status != SessionStatus.INTERRUPTED:
        return None

    for record in records_desc:
        event = record.event
        event_type = str(event.type)
        if event_type in {"session.resumed", "session.completed", "session.failed"}:
            return None

        payload = event.payload
        interruption_type = _optional_payload_string(payload, "interruption_type")
        manual_recovery_required = payload.get("manual_recovery_required") is True

        if event_type == "tool.call.approval_requested":
            approval = _object_payload(payload.get("approval"))
            approval_id = _optional_payload_string(approval, "approval_id")
            if approval is not None and approval_id is not None:
                checkpoint_call = _pending_approval_checkpoint_call(
                    checkpoint,
                    approval_id=approval_id,
                )
                if checkpoint_call is None:
                    continue
                tool_name = (
                    _optional_payload_string(approval, "tool_name")
                    or _optional_payload_string(checkpoint_call, "tool_name")
                    or event.tool_name
                )
                return _pending_action_from_event_record(
                    session=session,
                    record=record,
                    action_kind="tool_approval",
                    title="Tool approval required",
                    detail=_optional_payload_string(approval, "reason"),
                    tool_name=tool_name,
                    approval_id=approval_id,
                    arguments=_object_payload(approval.get("arguments"))
                    or _object_payload(checkpoint_call.get("arguments"))
                    or {},
                )

        if event_type == "session.awaiting_user_input":
            input_id = _optional_payload_string(payload, "input_id")
            if input_id is not None:
                tool_call_id = _optional_payload_string(payload, "tool_call_id")
                checkpoint_call = _pending_user_input_checkpoint_call(
                    checkpoint,
                    input_id=input_id,
                    tool_call_id=tool_call_id,
                )
                if checkpoint_call is None:
                    continue
                question = _optional_payload_string(payload, "question") or "Input required"
                return _pending_action_from_event_record(
                    session=session,
                    record=record,
                    action_kind="user_input",
                    title="User input required",
                    detail=question,
                    tool_name=event.tool_name
                    or _optional_payload_string(checkpoint_call, "tool_name"),
                    input_id=input_id,
                    tool_call_id=tool_call_id,
                    question=question,
                    options=_payload_string_list(payload, "options"),
                    arguments=_object_payload(checkpoint_call.get("arguments")),
                )

        if event_type != "session.interrupted":
            continue

        if manual_recovery_required:
            approval = _object_payload(payload.get("approval"))
            user_input = _object_payload(payload.get("user_input"))
            approval_id = _optional_payload_string(
                payload, "approval_id"
            ) or _optional_payload_string(approval, "approval_id")
            input_id = _optional_payload_string(user_input, "input_id")
            tool_name = (
                _optional_payload_string(payload, "tool_name")
                or _optional_payload_string(approval, "tool_name")
                or event.tool_name
            )
            tool_call_id = _optional_payload_string(payload, "tool_call_id") or (
                _optional_payload_string(user_input, "tool_call_id")
            )
            round_id = _optional_payload_string(payload, "tool_round_id")
            if tool_call_id is None or (
                approval_id is None and input_id is None and round_id is None
            ):
                continue
            checkpoint_call: dict[str, Any] | None
            if input_id is not None:
                checkpoint_call = _pending_user_input_checkpoint_call(
                    checkpoint,
                    input_id=input_id,
                    tool_call_id=tool_call_id,
                )
                if checkpoint_call is None:
                    continue
            elif approval_id is not None:
                checkpoint_call = _pending_approval_checkpoint_call(
                    checkpoint,
                    approval_id=approval_id,
                    tool_call_id=tool_call_id,
                )
                if checkpoint_call is None:
                    continue
            elif round_id is not None:
                checkpoint_call = _pending_tool_round_checkpoint_call(
                    checkpoint,
                    round_id=round_id,
                    tool_call_id=tool_call_id,
                )
                if checkpoint_call is None:
                    continue
            else:
                continue
            arguments = _object_payload(approval.get("arguments")) if approval else None
            if arguments is None:
                arguments = _object_payload(checkpoint_call.get("arguments"))
            return _pending_action_from_event_record(
                session=session,
                record=record,
                action_kind="manual_recovery",
                title="Manual recovery required",
                detail=_optional_payload_string(payload, "error")
                or _optional_payload_string(payload, "message")
                or "A previously started tool result must be reconciled before the session can continue.",
                tool_name=tool_name or _optional_payload_string(checkpoint_call, "tool_name"),
                approval_id=approval_id,
                input_id=input_id,
                round_id=round_id,
                tool_call_id=tool_call_id,
                question=_optional_payload_string(user_input, "question"),
                options=_payload_string_list(user_input, "options"),
                arguments=arguments,
            )

        if interruption_type == "tool_approval_required":
            approval = _object_payload(payload.get("approval"))
            approval_id = _optional_payload_string(approval, "approval_id")
            if approval is not None and approval_id is not None:
                checkpoint_call = _pending_approval_checkpoint_call(
                    checkpoint,
                    approval_id=approval_id,
                )
                if checkpoint_call is None:
                    continue
                tool_name = (
                    _optional_payload_string(approval, "tool_name")
                    or _optional_payload_string(checkpoint_call, "tool_name")
                    or event.tool_name
                )
                return _pending_action_from_event_record(
                    session=session,
                    record=record,
                    action_kind="tool_approval",
                    title="Tool approval required",
                    detail=_optional_payload_string(approval, "reason"),
                    tool_name=tool_name,
                    approval_id=approval_id,
                    arguments=_object_payload(approval.get("arguments"))
                    or _object_payload(checkpoint_call.get("arguments"))
                    or {},
                )

        if interruption_type == "user_input_required":
            user_input = _object_payload(payload.get("user_input"))
            input_id = _optional_payload_string(user_input, "input_id")
            if user_input is not None and input_id is not None:
                tool_call_id = _optional_payload_string(user_input, "tool_call_id")
                checkpoint_call = _pending_user_input_checkpoint_call(
                    checkpoint,
                    input_id=input_id,
                    tool_call_id=tool_call_id,
                )
                if checkpoint_call is None:
                    continue
                question = _optional_payload_string(user_input, "question") or "Input required"
                return _pending_action_from_event_record(
                    session=session,
                    record=record,
                    action_kind="user_input",
                    title="User input required",
                    detail=question,
                    tool_name=event.tool_name
                    or _optional_payload_string(checkpoint_call, "tool_name"),
                    input_id=input_id,
                    tool_call_id=tool_call_id,
                    question=question,
                    options=_payload_string_list(user_input, "options"),
                    arguments=_object_payload(checkpoint_call.get("arguments")),
                )

    return None


def _usage_breakdown(
    events: list[Event],
    *,
    key_fn: Callable[[UsageMetrics], tuple[str | None, str | None]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for event in events:
        if event.type != EventType.MODEL_COMPLETED:
            continue
        metrics = usage_metrics_from_event_payload(event.payload)
        if metrics is None:
            continue
        provider_name, model = key_fn(metrics)
        key = (provider_name, model)
        bucket = buckets.setdefault(
            key,
            {
                "provider_name": provider_name,
                "model": model,
                "session_ids": set(),
                "model_steps": 0,
                "usage": UsageMetrics(provider_name=provider_name, model=model),
            },
        )
        bucket["session_ids"].add(event.session_id)
        bucket["model_steps"] += 1
        bucket["usage"] = _add_usage_metrics(bucket["usage"], metrics)

    items = [
        UsageBreakdownItem(
            provider_name=provider_name,
            model=model,
            session_count=len(bucket["session_ids"]),
            model_steps=bucket["model_steps"],
            usage=bucket["usage"],
        ).model_dump()
        for (provider_name, model), bucket in buckets.items()
    ]
    return sorted(
        items,
        key=lambda item: (
            -item["usage"]["total_tokens"],
            item["provider_name"] or "",
            item["model"] or "",
        ),
    )


def _add_usage_metrics(left: UsageMetrics, right: UsageMetrics) -> UsageMetrics:
    return UsageMetrics(
        provider_name=left.provider_name,
        requested_model=left.requested_model or right.requested_model,
        model=left.model,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        reasoning_output_tokens=left.reasoning_output_tokens + right.reasoning_output_tokens,
        cache=CacheUsageMetrics(
            read_tokens=left.cache.read_tokens + right.cache.read_tokens,
            write_tokens=left.cache.write_tokens + right.cache.write_tokens,
            cached_input_tokens=left.cache.cached_input_tokens + right.cache.cached_input_tokens,
            uncached_input_tokens=left.cache.uncached_input_tokens
            + right.cache.uncached_input_tokens,
        ),
    )


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
        "description": task.description,
        "status": task.status.value,
        "status_reason": task.status_reason,
        "status_payload": task.status_payload,
        "session_id": task.session_id,
        "parent_task_id": task.parent_task_id,
        "assigned_agent_name": task.assigned_agent_name,
        "worker_id": task.worker_id,
        "lease_expires_at": (task.lease_expires_at.isoformat() if task.lease_expires_at else None),
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


def _serialize_task_detail(task: Task) -> dict[str, Any]:
    return {
        **_serialize_task_list_item(task),
        "input": task.input,
        "result": task.result,
        "error": task.error,
        "metadata": task.metadata,
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
    auth: AuthDependency | None = None,
    api_path: str = SERVER_API_PREFIX,
    openapi_url: str | None = "/openapi.json",
    replay_idle_timeout_s: float = 300.0,
) -> APIRouter:
    """Create an APIRouter with standard cayu endpoints.

    Args:
        auth: FastAPI-compatible dependency guarding the CAYU control plane.
            Production callers should pass this through ``create_server``; only
            explicit dev-mode callers should leave it unset. It protects every
            control-plane route that can start, change, inspect, or reveal runtime
            state; only the health route stays open for load balancers. It must
            return ``AuthContext`` or a compatible mapping and raise
            ``HTTPException`` (401/403) to deny a request.
        api_path: URL path prefix for the CAYU control plane. Defaults to
            ``/api``.
        openapi_url: Public OpenAPI schema URL advertised by ``/contract`` for
            client generation. Pass ``None`` when generated OpenAPI is disabled.
        replay_idle_timeout_s: Maximum time an active replay stream may wait
            without seeing a new persisted event before emitting an error and closing.
    """

    if (
        isinstance(replay_idle_timeout_s, bool)
        or not isinstance(replay_idle_timeout_s, (int, float))
        or replay_idle_timeout_s <= 0
    ):
        raise ValueError("replay_idle_timeout_s must be a positive number.")
    replay_idle_timeout_s = float(replay_idle_timeout_s)

    api_prefix = _normalize_api_path(api_path)
    router = APIRouter(prefix=api_prefix)

    # Shared dependency list for control-plane routes. FastAPI treats an empty
    # sequence like no dependencies, so `auth=None` keeps current dev behavior.
    auth_dependency = server_auth_dependency(auth) if auth is not None else None
    protected: list[Any] = [Depends(auth_dependency)] if auth_dependency is not None else []

    async def _optional_auth_context(request: Request) -> AuthContext | None:
        # The interruption and approval/user-input resolution routes take this as a
        # handler parameter INSTEAD of `dependencies=protected`: one callable
        # both guards the route (its 401/403 raises before the handler body)
        # and yields the verified caller identity for typed operator
        # provenance. Splitting guard and extraction into two differently
        # wrapped callables would invoke the user's auth dependency twice per
        # request (FastAPI caches per-callable, not per-underlying-auth).
        # Handlers must take this via the `= Depends(...)` default form: with
        # `from __future__ import annotations`, a function-local Annotated
        # alias is an unresolvable string annotation that FastAPI silently
        # degrades to a required query parameter.
        if auth_dependency is None:
            return None
        return await auth_dependency(request)

    optional_auth_context = Depends(_optional_auth_context)

    @router.get("/contract", response_model=ServerContractResponse, dependencies=protected)
    async def get_contract():
        return ServerContractResponse(
            api_prefix=api_prefix,
            client_generation=ClientGenerationContract(openapi_url=openapi_url),
        )

    async def _marker_sequence(session_id: str, event_id: str) -> int | None:
        """Sequence of the persisted event named by a ``Last-Event-ID`` marker.

        Returns ``None`` when the marker event is unknown, so the caller replays the
        full history (at-least-once delivery beats silently dropping events).
        """
        records = await session_store.query_events(
            EventQuery(session_id=session_id, event_id=event_id, limit=1)
        )
        return records[0].sequence if records else None

    async def _replay_events_response(
        http_request: Request,
        *,
        expected_session_id: str | None = None,
    ) -> EventSourceResponse | None:
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
        if expected_session_id is not None and session_id != expected_session_id:
            raise HTTPException(
                status_code=422,
                detail="Last-Event-ID session does not match the request session_id.",
            )
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        async def replay() -> AsyncIterator[dict[str, str]]:
            after_sequence = await _marker_sequence(session_id, last_seen_event_id)
            loop = asyncio.get_running_loop()
            idle_deadline = loop.time() + replay_idle_timeout_s
            poll_interval = _REPLAY_POLL_INTERVAL_MIN_S
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
                if page:
                    idle_deadline = loop.time() + replay_idle_timeout_s
                    poll_interval = _REPLAY_POLL_INTERVAL_MIN_S
                if len(page) == _EVENT_PAGE_LIMIT_MAX:
                    continue
                current = await session_store.load_state(session_id)
                if current is None or current.status not in _REPLAY_ACTIVE_SESSION_STATUSES:
                    return
                remaining = idle_deadline - loop.time()
                if remaining <= 0:
                    yield error_to_sse_message(
                        TimeoutError(
                            f"Replay for session {session_id} received no events for "
                            f"{replay_idle_timeout_s:g} seconds."
                        )
                    )
                    return
                await asyncio.sleep(min(poll_interval, remaining))
                poll_interval = _next_replay_poll_interval(
                    poll_interval,
                    received_events=bool(page),
                )

        return EventSourceResponse(replay())

    @router.post(
        "/run",
        dependencies=protected,
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
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
            model=body.model,
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

        event_stream: AsyncIterator[Event] = cayu_app.run(request)
        if task_id is not None:
            event_stream = _fail_task_on_prestream_error(
                event_stream, task_store=task_store, task_id=task_id
            )
        return _detached_event_stream_response(event_stream)

    @router.post(
        "/resume",
        dependencies=protected,
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def resume_agent(
        body: ResumeBody,
        http_request: Request,
        trace_metadata: TraceContextMetadata,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
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

    @router.post(
        "/sessions/{session_id}/interrupt",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def interrupt_session(
        session_id: NonBlankString,
        http_request: Request,
        body: InterruptSessionBody | None = None,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=session_id,
        )
        if replay is not None:
            return replay
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
            requested_by=_request_interruption_actor(
                auth_context,
                body.requested_by if body is not None else None,
            ),
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

    @router.post(
        "/tool-approvals/resolve",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def resolve_tool_approval(
        body: ToolApprovalBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
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
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.resolve_tool_approval(request))

    @router.post(
        "/tool-approvals/recover",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_tool_approval(
        body: ToolApprovalRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
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
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.recover_tool_approval(request))

    @router.post(
        "/tool-rounds/recover",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_tool_round(
        body: ToolRoundRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolRoundRecoveryRequest(
            session_id=body.session_id,
            round_id=body.round_id,
            tool_call_id=body.tool_call_id,
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.recover_tool_round(request))

    @router.post(
        "/user-input/resolve",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def resolve_user_input(
        body: UserInputResolveBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        response = UserInputResponse(
            session_id=body.session_id,
            input_id=body.input_id,
            answer=body.answer,
            structured=body.structured,
            artifacts=body.artifacts,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.resolve_user_input(response))

    @router.post(
        "/user-input/recover",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_user_input(
        body: UserInputRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session = await session_store.load(body.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = UserInputRecoveryRequest(
            session_id=body.session_id,
            input_id=body.input_id,
            answer=body.answer,
            tool_call_id=body.tool_call_id,
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
        )

        return _detached_event_stream_response(cayu_app.recover_user_input(request))

    @router.get("/agents", response_model=AgentsResponse, dependencies=protected)
    async def list_agents():
        agents = [
            _serialize_agent(cayu_app, cayu_app.get_agent(name)) for name in cayu_app.list_agents()
        ]
        return {"agents": agents, "total_count": len(agents)}

    @router.get("/agents/{agent_name}", response_model=AgentsResponse, dependencies=protected)
    async def get_agent(agent_name: NonBlankString):
        try:
            agent = cayu_app.get_agent(agent_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent not found") from exc
        serialized = _serialize_agent(cayu_app, agent)
        return {"agents": [serialized], "total_count": 1}

    @router.get(
        "/environments",
        response_model=EnvironmentsResponse,
        dependencies=protected,
    )
    async def list_environments():
        records = cayu_app.list_environment_registrations()
        environments = [_serialize_environment(cayu_app, record) for record in records]
        return {"environments": environments, "total_count": len(environments)}

    @router.get(
        "/environments/{environment_name}",
        response_model=EnvironmentsResponse,
        dependencies=protected,
    )
    async def get_environment(environment_name: NonBlankString):
        record = next(
            (
                item
                for item in cayu_app.list_environment_registrations()
                if item.spec.name == environment_name
            ),
            None,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return {"environments": [_serialize_environment(cayu_app, record)], "total_count": 1}

    @router.get(
        "/artifacts",
        response_model=ArtifactsResponse,
        responses=ARTIFACT_ENDPOINT_ERROR_RESPONSES,
        dependencies=protected,
    )
    async def list_artifacts(
        limit: Annotated[int, Query(ge=1, le=_ARTIFACT_PAGE_LIMIT_MAX)] = 100,
        offset: Annotated[int, Query(ge=0, le=_ARTIFACT_PAGE_OFFSET_MAX)] = 0,
        artifact_store_id: Annotated[str | None, Query()] = None,
        scope: ArtifactScope | None = None,
        session_id: Annotated[str | None, Query()] = None,
        agent_name: Annotated[str | None, Query()] = None,
        environment_name: Annotated[str | None, Query()] = None,
    ):
        requested_store_id = _clean_optional_query_value(
            artifact_store_id,
            "artifact_store_id",
        )
        requested_session_id = _clean_optional_query_value(session_id, "session_id")
        requested_agent_name = _clean_optional_query_value(agent_name, "agent_name")
        requested_environment_name = _clean_optional_query_value(
            environment_name,
            "environment_name",
        )
        stores = _artifact_stores_by_id(cayu_app)
        if requested_store_id is not None:
            store = stores.get(requested_store_id)
            if store is None:
                raise HTTPException(status_code=404, detail="Artifact store not found")
            selected_stores = {requested_store_id: store}
        else:
            selected_stores = stores

        artifacts: list[dict[str, Any]] = []
        total_count: int | None = 0
        truncated = False
        per_store_limit = offset + limit
        for store_id, store in selected_stores.items():
            try:
                result = await store.list(
                    scope=scope,
                    session_id=requested_session_id,
                    agent_name=requested_agent_name,
                    environment_name=requested_environment_name,
                    limit=per_store_limit,
                )
                if type(result) is not ArtifactListResult:
                    raise TypeError("Artifact stores must return ArtifactListResult from list().")
                page = ArtifactListResult(
                    artifacts=result.artifacts,
                    total_count=result.total_count,
                    truncated=result.truncated,
                )
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc
            artifacts.extend(
                _serialize_artifact(cayu_app, artifact, artifact_store_id=store_id)
                for artifact in page.artifacts
            )
            if page.total_count is None:
                total_count = None
                truncated = True
            elif total_count is not None:
                total_count += page.total_count
            truncated = truncated or page.truncated

        artifacts.sort(key=_artifact_sort_key, reverse=True)
        page_artifacts = artifacts[offset : offset + limit]
        next_offset = None
        has_more = False
        if total_count is None:
            if len(artifacts) > offset + limit or truncated:
                has_more = True
                candidate_next_offset = offset + limit
                if candidate_next_offset <= _ARTIFACT_PAGE_OFFSET_MAX:
                    next_offset = candidate_next_offset
        elif offset + limit < total_count:
            has_more = True
            candidate_next_offset = offset + limit
            if candidate_next_offset <= _ARTIFACT_PAGE_OFFSET_MAX:
                next_offset = candidate_next_offset
        truncated = has_more
        return {
            "artifacts": page_artifacts,
            "total_count": total_count,
            "truncated": truncated,
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset,
        }

    async def _read_artifact_from_request(
        artifact_id: str,
        artifact_store_id: str | None,
        *,
        max_bytes: int | None = None,
    ):
        try:
            artifact_id = require_clean_nonblank(artifact_id, "artifact_id")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        stores = _artifact_stores_by_id(cayu_app)
        requested_store_id = _clean_optional_query_value(
            artifact_store_id,
            "artifact_store_id",
        )
        if requested_store_id is not None:
            store = stores.get(requested_store_id)
            if store is None:
                raise HTTPException(status_code=404, detail="Artifact store not found")
            try:
                read = await store.read_bytes(artifact_id, max_bytes=max_bytes)
                return requested_store_id, copy_artifact_read_result(
                    read,
                    expected_artifact_id=artifact_id,
                    max_content_bytes=max_bytes,
                )
            except InvalidArtifactIdError as exc:
                raise HTTPException(status_code=404, detail="Artifact not found") from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Artifact not found") from exc
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc

        matches = []
        for store_id, store in stores.items():
            try:
                read = await store.read_bytes(artifact_id, max_bytes=max_bytes)
                matches.append(
                    (
                        store_id,
                        copy_artifact_read_result(
                            read,
                            expected_artifact_id=artifact_id,
                            max_content_bytes=max_bytes,
                        ),
                    )
                )
            except (FileNotFoundError, InvalidArtifactIdError):
                continue
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc
        if not matches:
            raise HTTPException(status_code=404, detail="Artifact not found")
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail="Artifact id exists in multiple stores; pass artifact_store_id.",
            )
        return matches[0]

    @router.get(
        "/artifacts/{artifact_id}",
        response_model=ArtifactReadResponse,
        responses=ARTIFACT_ENDPOINT_ERROR_RESPONSES,
        dependencies=protected,
    )
    async def get_artifact(
        artifact_id: ArtifactIdPath,
        artifact_store_id: Annotated[str | None, Query()] = None,
        max_bytes: Annotated[int, Query(ge=1, le=262_144)] = 64_000,
    ):
        store_id, read = await _read_artifact_from_request(
            artifact_id,
            artifact_store_id,
            max_bytes=max_bytes,
        )
        preview_base64, text_preview = _artifact_read_preview(cayu_app, read)
        return {
            "artifact": _serialize_artifact(cayu_app, read.metadata, artifact_store_id=store_id),
            "preview_base64": preview_base64,
            "text_preview": text_preview,
            "total_bytes": read.total_bytes,
            "truncated": read.truncated,
        }

    @router.get(
        "/artifacts/{artifact_id}/content",
        response_class=Response,
        responses=ARTIFACT_CONTENT_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_artifact_content(
        artifact_id: ArtifactIdPath,
        artifact_store_id: Annotated[str, Query(min_length=1)],
        disposition: Annotated[Literal["attachment", "inline"], Query()] = "attachment",
        max_bytes: Annotated[int, Query(ge=1, le=_ARTIFACT_CONTENT_BYTES_MAX)] = (
            _ARTIFACT_CONTENT_BYTES_MAX
        ),
    ):
        store_id, read = await _read_artifact_from_request(
            artifact_id,
            artifact_store_id,
            max_bytes=max_bytes,
        )
        if read.truncated or read.total_bytes > max_bytes or len(read.content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Artifact exceeds the requested max_bytes for direct content "
                    "response. Use the bounded artifact preview, increase max_bytes "
                    "up to the server maximum, or use a store-native streaming/range "
                    "reader for artifacts above that maximum."
                ),
            )
        response_disposition = _artifact_content_disposition_kind(
            read.metadata.content_type,
            disposition,
        )
        return Response(
            content=read.content,
            media_type=read.metadata.content_type,
            headers={
                "Content-Disposition": _artifact_content_disposition(
                    read.metadata.filename,
                    response_disposition,
                ),
                "X-Content-Type-Options": "nosniff",
                "X-Cayu-Artifact-Id": _artifact_header_value(read.metadata.id, artifact_id),
                "X-Cayu-Artifact-Store-Id": _artifact_header_value(store_id, "artifact-store"),
                "Cache-Control": "private, no-store",
            },
        )

    @router.get(
        "/pending-actions",
        response_model=PendingActionsResponse,
        dependencies=protected,
    )
    async def list_pending_actions(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        session_limit: Annotated[int, Query(ge=1, le=1000)] = 250,
        session_id: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        kind: Annotated[
            str | None,
            Query(pattern="^(tool_approval|user_input|manual_recovery)$"),
        ] = None,
    ):
        requested_session_id = _clean_optional_query_value(session_id, "session_id")
        search = _clean_optional_query_value(q, "q")
        if requested_session_id is not None:
            session = await session_store.load(requested_session_id)
            sessions = (
                [session]
                if session is not None and session.status in _PENDING_ACTION_SESSION_STATUSES
                else []
            )
        else:
            sessions_by_id: dict[str, Session] = {}
            for status in _PENDING_ACTION_SESSION_STATUSES:
                page = await session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        limit=session_limit,
                        order_by=SessionOrder.UPDATED_AT_DESC,
                    )
                )
                for session in page.sessions:
                    sessions_by_id[session.id] = session
            sessions = sorted(
                sessions_by_id.values(),
                key=lambda session: session.updated_at,
                reverse=True,
            )[:session_limit]

        actions: list[dict[str, Any]] = []
        for session in sessions:
            records = await session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_types=_PENDING_ACTION_EVENT_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1000,
                )
            )
            checkpoint = await session_store.load_checkpoint(session.id)
            action = _pending_action_from_records(session, records, checkpoint)
            if action is None:
                action = _pending_tool_round_manual_recovery_action(
                    session,
                    records,
                    checkpoint,
                )
            if action is None:
                continue
            if kind is not None and action["kind"] != kind:
                continue
            if not _pending_action_matches_query(action, search):
                continue
            actions.append(action)

        return {
            "actions": actions[:limit],
            "total_count": len(actions),
            "inspected_session_count": len(sessions),
        }

    @router.get("/sessions", response_model=ListSessionsResponse, dependencies=protected)
    async def list_sessions(
        limit: Annotated[int, Query(ge=1, le=1000)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        cursor: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        status: SessionStatus | None = None,
        debug_state: SessionDebugState | None = None,
        agent_name: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
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
                    q=_clean_optional_query_value(q, "q"),
                    status=status,
                    debug_state=debug_state,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    provider_name=_clean_optional_query_value(provider_name, "provider_name"),
                    model=_clean_optional_query_value(model, "model"),
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

    @router.post(
        "/sessions/summary",
        response_model=SessionsSummaryResponse,
        dependencies=protected,
    )
    async def get_sessions_summary(
        body: SessionsSummaryBody | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
        offset: Annotated[int, Query(ge=0)] = 0,
        cursor: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        status: SessionStatus | None = None,
        debug_state: SessionDebugState | None = None,
        agent_name: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
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
        try:
            result = await session_store.list_sessions(
                SessionQuery(
                    q=_clean_optional_query_value(q, "q"),
                    status=status,
                    debug_state=debug_state,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    provider_name=_clean_optional_query_value(provider_name, "provider_name"),
                    model=_clean_optional_query_value(model, "model"),
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
        sessions = result.sessions
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
        model_events = [
            record.event
            for record in sorted(all_event_records, key=lambda record: record.sequence)
            if record.event.type == EventType.MODEL_COMPLETED
        ]
        provider_breakdown = _usage_breakdown(
            model_events,
            key_fn=lambda metrics: (metrics.provider_name, None),
        )
        model_breakdown = _usage_breakdown(
            model_events,
            key_fn=lambda metrics: (metrics.provider_name, metrics.model),
        )

        cost_summary = None
        if body.pricing is not None:
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
            "next_cursor": result.next_cursor,
            "total_count": result.total_count,
            "usage": usage_summary,
            "provider_breakdown": provider_breakdown,
            "model_breakdown": model_breakdown,
            "cost": cost_summary,
        }

    @router.get(
        "/sessions/{session_id}/usage",
        response_model=SessionUsageSummary,
        dependencies=protected,
    )
    async def get_session_usage(session_id: NonBlankString):
        try:
            summary = await cayu_app.get_session_usage(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary.model_dump()

    @router.post(
        "/sessions/{session_id}/cost",
        response_model=SessionCostSummary,
        dependencies=protected,
    )
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

    @router.get(
        "/causal-budgets/{causal_budget_id}/usage",
        response_model=CausalBudgetUsageSummary,
        dependencies=protected,
    )
    async def get_causal_budget_usage(causal_budget_id: NonBlankString):
        try:
            summary = await cayu_app.get_causal_budget_usage(causal_budget_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Causal budget not found") from exc
        return summary.model_dump()

    @router.post(
        "/causal-budgets/{causal_budget_id}/cost",
        response_model=CausalBudgetCostSummary,
        dependencies=protected,
    )
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

    @router.post(
        "/causal-budgets/{causal_budget_id}/summary",
        response_model=CausalBudgetSummaryResponse,
        dependencies=protected,
    )
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

    @router.get(
        "/sessions/{session_id}/state",
        response_model=SessionStateResponse,
        dependencies=protected,
    )
    async def get_session_state(session_id: NonBlankString):
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        interruption_cascade = await cayu_app.interruption_cascade_status(session_id)
        return {
            "session_id": state.id,
            "status": state.status,
            "updated_at": state.updated_at.isoformat(),
            "last_activity_at": state.last_activity_at.isoformat(),
            "interruption_cascade": interruption_cascade,
        }

    @router.get(
        "/sessions/{session_id}/summary",
        response_model=SessionSummaryResponse,
        dependencies=protected,
    )
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

    @router.get(
        "/sessions/{session_id}/events",
        response_model=ListSessionEventsResponse,
        dependencies=protected,
    )
    async def list_session_events(
        session_id: NonBlankString,
        event_type: str | None = None,
        tool_name: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        workflow_name: str | None = None,
        after_sequence: int | None = Query(
            default=None,
            ge=0,
            description="Return only events with a greater durable sequence.",
        ),
        before_sequence: int | None = Query(
            default=None,
            ge=1,
            description="Return only events with a smaller durable sequence.",
        ),
        order_by: Annotated[
            EventOrder,
            Query(description="Return events in durable sequence order."),
        ] = EventOrder.SEQUENCE_ASC,
        limit: int = Query(default=100, ge=1, le=_EVENT_PAGE_LIMIT_MAX),
    ):
        state = await session_store.load_state(session_id)
        if state is None:
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
                before_sequence=before_sequence,
                limit=limit + 1,
                order_by=order_by,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_context=False, include_url=False),
            ) from exc

        records = await session_store.query_events(query)
        page = records[:limit]
        has_more = len(records) > limit
        cursor = after_sequence if order_by == EventOrder.SEQUENCE_ASC else before_sequence
        next_sequence = page[-1].sequence if page else cursor

        return {
            "session_id": session_id,
            "events": [_serialize_event_record(record) for record in page],
            "order_by": order_by,
            "next_sequence": next_sequence,
            "has_more": has_more,
        }

    @router.get(
        "/sessions/{session_id}/transcript",
        response_model=SessionTranscriptResponse,
        dependencies=protected,
    )
    async def get_session_transcript(
        session_id: NonBlankString,
        role: MessageRole | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=_TRANSCRIPT_PAGE_LIMIT_MAX),
        include_thinking: bool = Query(default=True),
    ):
        state = await session_store.load_state(session_id)
        if state is None:
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

    @router.get(
        "/sessions/{session_id}",
        response_model=SessionDetailResponse,
        dependencies=protected,
    )
    async def get_session(session_id: str):
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        events = await session_store.load_events(session_id)
        transcript = await session_store.load_transcript(session_id)
        interruption_cascade = await cayu_app.interruption_cascade_status(session_id)

        return {
            "session": _serialize_session(session),
            "interruption_cascade": interruption_cascade,
            "events": [
                {
                    "id": e.id,
                    "type": str(e.type),
                    "agent_name": e.agent_name,
                    "environment_name": e.environment_name,
                    "workflow_name": e.workflow_name,
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

    @router.patch(
        "/sessions/{session_id}/labels",
        dependencies=protected,
        response_model=ApiSession,
    )
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

    @router.patch(
        "/sessions/{session_id}/metadata",
        dependencies=protected,
        response_model=ApiSession,
    )
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

    @router.get("/tasks", response_model=list[ApiTaskListItem], dependencies=protected)
    async def list_tasks(
        q: str | None = None,
        status: TaskStatus | None = None,
        task_type: str | None = Query(default=None, alias="type"),
        session_id: str | None = None,
        parent_task_id: str | None = None,
        assigned_agent_name: str | None = None,
        order_by: TaskOrder = TaskOrder.UPDATED_AT_DESC,
        limit: int = 50,
        offset: int = 0,
    ):
        if task_store is None:
            return []
        try:
            query = TaskQuery(
                q=q,
                status=status,
                type=task_type,
                session_id=session_id,
                parent_task_id=parent_task_id,
                assigned_agent_name=assigned_agent_name,
                order_by=order_by,
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

    @router.get(
        "/tasks/{task_id}",
        response_model=ApiTaskDetail,
        dependencies=protected,
    )
    async def get_task(task_id: NonBlankString):
        store = await _require_task_store()
        task = await store.load_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return _serialize_task_detail(task)

    async def _apply_task_action(action, task_id: str):
        try:
            task = await action(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _serialize_task_detail(task)

    @router.post(
        "/tasks/{task_id}/pause",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
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

    @router.post(
        "/tasks/{task_id}/block",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
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

    @router.post(
        "/tasks/{task_id}/needs-attention",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
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

    @router.post(
        "/tasks/{task_id}/resume",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
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

    @router.get(
        "/knowledge/pending",
        response_model=PendingKnowledgeListResponse,
        dependencies=protected,
    )
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

    @router.get(
        "/knowledge/pending/{entry_id}",
        response_model=PendingKnowledgeDetailResponse,
        dependencies=protected,
    )
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

    @router.post(
        "/knowledge/{entry_id}/approve",
        dependencies=protected,
        response_model=ApiReviewedKnowledgeEntry,
    )
    async def approve_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.approve, entry_id)

    @router.post(
        "/knowledge/{entry_id}/reject",
        dependencies=protected,
        response_model=ApiReviewedKnowledgeEntry,
    )
    async def reject_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.reject, entry_id)

    @router.get("/health", response_model=HealthResponse)
    async def health():
        return {"ok": True}

    return router


def _normalize_api_path(path: str) -> str:
    value = path.strip()
    if not value:
        raise ValueError("api_path must not be blank.")
    if "?" in value or "#" in value or "://" in value:
        raise ValueError("api_path must be a URL path, not a URL.")
    value = "/" + value.strip("/")
    if value == "/":
        raise ValueError("api_path must not be the site root.")
    return value
