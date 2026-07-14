"""Public server API contract models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    CostLineItem,
    SessionCostSummary,
)
from cayu.runtime.usage import CausalBudgetUsageSummary, SessionUsageSummary, UsageMetrics
from cayu.server.sse import (
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_EVENT_DATA_MAX_BYTES,
    SseErrorCode,
    SseErrorKind,
)

SERVER_API_PREFIX = "/api"
SERVER_CONTRACT_VERSION = "1"
SSE_CONTENT_TYPE = "text/event-stream"
SSE_LAST_EVENT_ID_FORMAT = "session_id:event_id"


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiBaseModel):
    ok: StrictBool


class ApiErrorResponse(ApiBaseModel):
    detail: str


class SseEventEnvelope(ApiBaseModel):
    """JSON payload in each runtime event SSE ``data:`` frame."""

    id: str
    type: str
    session_id: str
    agent_name: str | None
    environment_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None
    payload: dict[str, Any]
    timestamp: str


class SseErrorEnvelope(ApiBaseModel):
    """JSON payload in classified terminal SSE ``event: error`` frames."""

    type: Literal["stream.error"]
    kind: SseErrorKind
    code: SseErrorCode
    error: str
    error_type: str
    retryable: StrictBool
    session_id: str | None


def _sse_event_example() -> SseEventEnvelope:
    return SseEventEnvelope(
        id="event_123",
        type="session.started",
        session_id="session-123",
        agent_name="assistant",
        environment_name="production",
        workflow_name=None,
        tool_name=None,
        payload={"status": "running"},
        timestamp="2026-07-06T00:00:00+00:00",
    )


def _sse_error_example() -> SseErrorEnvelope:
    return SseErrorEnvelope(
        type="stream.error",
        kind="runtime",
        code="runtime_failed",
        error="Runtime stream failed.",
        error_type="RuntimeError",
        retryable=False,
        session_id="session-123",
    )


class SseFrameExamples(ApiBaseModel):
    event_data: SseEventEnvelope = Field(default_factory=_sse_event_example)
    error_data: SseErrorEnvelope = Field(default_factory=_sse_error_example)


class SseContract(ApiBaseModel):
    content_type: Literal["text/event-stream"] = SSE_CONTENT_TYPE
    event_id_format: Literal["session_id:event_id"] = SSE_LAST_EVENT_ID_FORMAT
    replay_header: Literal["Last-Event-ID"] = "Last-Event-ID"
    event_data_schema: Literal["SseEventEnvelope"] = "SseEventEnvelope"
    error_event_name: Literal["error"] = "error"
    error_data_schema: Literal["SseErrorEnvelope"] = "SseErrorEnvelope"
    max_event_data_bytes: StrictInt = Field(
        default=SSE_EVENT_DATA_MAX_BYTES,
        ge=1,
        description="Maximum UTF-8 bytes in one live SSE event data value.",
    )
    max_error_text_bytes: StrictInt = Field(
        default=SSE_ERROR_TEXT_MAX_BYTES,
        ge=1,
        description="Maximum UTF-8 bytes in the redacted error field.",
    )
    examples: SseFrameExamples = Field(default_factory=SseFrameExamples)


class ClientGenerationContract(ApiBaseModel):
    openapi_url: str | None = "/openapi.json"
    supported_targets: tuple[Literal["typescript", "python"], ...] = ("typescript", "python")
    source_of_truth: Literal["openapi"] = "openapi"


class VersioningContract(ApiBaseModel):
    contract_version: str = SERVER_CONTRACT_VERSION
    compatibility: Literal["additive-with-explicit-breaking-review"] = (
        "additive-with-explicit-breaking-review"
    )
    breaking_change_requires: tuple[
        Literal["openapi_snapshot_update", "client_regeneration", "migration_note"],
        ...,
    ] = ("openapi_snapshot_update", "client_regeneration", "migration_note")


class ServerContractResponse(ApiBaseModel):
    api_prefix: str = SERVER_API_PREFIX
    contract_version: str = SERVER_CONTRACT_VERSION
    versioning: VersioningContract = Field(default_factory=VersioningContract)
    sse: SseContract = Field(default_factory=SseContract)
    client_generation: ClientGenerationContract = Field(default_factory=ClientGenerationContract)


class ApiEventRecord(ApiBaseModel):
    sequence: StrictInt = Field(ge=0)
    id: str
    type: str
    session_id: str
    agent_name: str | None
    environment_name: str | None
    workflow_name: str | None
    tool_name: str | None
    payload: dict[str, Any]
    timestamp: str


class ApiSessionOutcome(ApiBaseModel):
    session_id: str
    status: str
    reason: str | None
    details: dict[str, Any]
    retry: dict[str, Any] | None
    terminal_event: ApiEventRecord | None
    latest_retry_event: ApiEventRecord | None


class ApiSessionBase(ApiBaseModel):
    id: str
    status: str
    agent_name: str
    provider_name: str | None
    model: str | None
    parent_session_id: str | None
    causal_budget_id: str | None
    runtime_name: str
    runtime_version: str | None
    environment_name: str | None
    created_at: str
    updated_at: str
    labels: dict[str, str]


class ApiSession(ApiSessionBase):
    metadata: dict[str, Any]


class ListSessionsResponse(ApiBaseModel):
    sessions: list[ApiSessionBase]
    next_cursor: str | None
    total_count: StrictInt | None = Field(default=None, ge=0)


class ApiEventSummary(ApiBaseModel):
    total_events: StrictInt = Field(ge=0)
    counts_by_type: dict[str, StrictInt]
    latest_event: ApiEventRecord | None


class ApiSessionSummaryItem(ApiBaseModel):
    session: ApiSession
    outcome: ApiSessionOutcome
    events: ApiEventSummary


class AggregateUsageSummary(ApiBaseModel):
    session_ids: list[str]
    session_count: StrictInt = Field(ge=0)
    model_steps: StrictInt = Field(ge=0)
    tool_calls: StrictInt = Field(ge=0)
    provider_names: list[str]
    models: list[str]
    usage: UsageMetrics
    session_summaries: tuple[SessionUsageSummary, ...]


class UsageBreakdownItem(ApiBaseModel):
    provider_name: str | None
    model: str | None
    session_count: StrictInt = Field(ge=0)
    model_steps: StrictInt = Field(ge=0)
    usage: UsageMetrics


class AggregateCostSummary(ApiBaseModel):
    session_ids: list[str]
    session_count: StrictInt = Field(ge=0)
    currency: str
    model_steps: StrictInt = Field(ge=0)
    priced_model_steps: StrictInt = Field(ge=0)
    unpriced_model_steps: StrictInt = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    line_items: tuple[CostLineItem, ...]
    session_costs: tuple[SessionCostSummary, ...]


class SessionsSummaryResponse(ApiBaseModel):
    session_count: StrictInt = Field(ge=0)
    sessions: list[ApiSessionSummaryItem]
    next_cursor: str | None
    total_count: StrictInt | None = Field(ge=0)
    usage: AggregateUsageSummary
    provider_breakdown: tuple[UsageBreakdownItem, ...] = Field(default_factory=tuple)
    model_breakdown: tuple[UsageBreakdownItem, ...] = Field(default_factory=tuple)
    cost: AggregateCostSummary | None


class ApiPendingAction(ApiBaseModel):
    id: str
    kind: Literal["tool_approval", "user_input", "manual_recovery"]
    session: ApiSessionBase
    event: ApiEventRecord
    title: str
    detail: str | None = None
    tool_name: str | None = None
    approval_id: str | None = None
    input_id: str | None = None
    round_id: str | None = None
    tool_call_id: str | None = None
    question: str | None = None
    options: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] | None = None


class ApiPendingActionIssue(ApiBaseModel):
    code: Literal["source_too_large", "source_too_complex", "source_invalid"]
    session_id: str
    agent_name: str
    status: Literal["interrupted", "failed", "completed"]
    updated_at: datetime
    detail: str


class PendingActionsResponse(ApiBaseModel):
    actions: list[ApiPendingAction]
    issues: list[ApiPendingActionIssue]
    next_cursor: str | None
    has_more: StrictBool
    total_count: StrictInt | None = Field(ge=0)
    inspected_candidate_count: StrictInt = Field(ge=0)


PENDING_ACTION_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    413: {
        "description": "The pending-action page exceeds the bounded response size.",
        "model": ApiErrorResponse,
    }
}


class ApiToolSummary(ApiBaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    parallel_safe: StrictBool
    effect: str


class ApiAgentSummary(ApiBaseModel):
    name: str
    provider_name: str | None
    model: str
    tool_count: StrictInt = Field(ge=0)
    tools: list[ApiToolSummary]
    metadata: dict[str, Any]
    provider_options: dict[str, Any]
    thinking: dict[str, Any] | None
    has_system_prompt: StrictBool


class AgentsResponse(ApiBaseModel):
    agents: list[ApiAgentSummary]
    total_count: StrictInt = Field(ge=0)


class ApiEnvironmentSummary(ApiBaseModel):
    name: str
    metadata: dict[str, Any]
    is_factory: StrictBool
    workspace_id: str | None
    artifact_store_id: str | None
    runner_type: str | None
    binding_type: str | None
    vault_type: str | None
    proxy_type: str | None
    knowledge_store_type: str | None
    mcp_server_count: StrictInt = Field(ge=0)
    workspace_instructions: str | None
    bound_workspace: dict[str, Any] | None = None


class EnvironmentsResponse(ApiBaseModel):
    environments: list[ApiEnvironmentSummary]
    total_count: StrictInt = Field(ge=0)


class ApiArtifactSummary(ApiBaseModel):
    id: str
    artifact_store_id: str
    filename: str
    content_type: str
    size_bytes: StrictInt = Field(ge=0)
    scope: str
    session_id: str | None
    agent_name: str | None
    environment_name: str | None
    created_at: str
    metadata: dict[str, Any]


class ArtifactsResponse(ApiBaseModel):
    artifacts: list[ApiArtifactSummary]
    total_count: StrictInt | None = Field(default=None, ge=0)
    truncated: StrictBool
    limit: StrictInt = Field(ge=1)
    offset: StrictInt = Field(ge=0)
    next_offset: StrictInt | None = Field(default=None, ge=0)


class ArtifactReadResponse(ApiBaseModel):
    artifact: ApiArtifactSummary
    preview_base64: str
    text_preview: str | None
    total_bytes: StrictInt = Field(ge=0)
    truncated: StrictBool


class TranscriptSummary(ApiBaseModel):
    total_messages: StrictInt = Field(ge=0)


class SessionSummaryResponse(ApiBaseModel):
    session: ApiSession
    events: ApiEventSummary
    transcript: TranscriptSummary
    outcome: ApiSessionOutcome
    usage: SessionUsageSummary


class SessionStateResponse(ApiBaseModel):
    session_id: str
    status: Literal["pending", "running", "interrupting", "completed", "failed", "interrupted"]
    updated_at: str
    last_activity_at: str
    interruption_cascade: Literal["none", "pending", "failed"]


class CausalBudgetSummaryResponse(ApiBaseModel):
    causal_budget_id: str
    session_count: StrictInt = Field(ge=0)
    sessions: list[ApiSessionSummaryItem]
    usage: CausalBudgetUsageSummary
    cost: CausalBudgetCostSummary


class ListSessionEventsResponse(ApiBaseModel):
    session_id: str
    events: list[ApiEventRecord]
    order_by: Literal["sequence_asc", "sequence_desc"] = Field(
        description="Ordering applied to the returned event page."
    )
    next_sequence: StrictInt | None = Field(
        default=None,
        ge=0,
        description=(
            "Exclusive sequence cursor for the next page in the returned order: pass it as "
            "after_sequence for ascending pages or before_sequence for descending pages."
        ),
    )
    scan_through_sequence: StrictInt | None = Field(
        ge=0,
        description=(
            "Highest durable sequence that a forward tail reader can safely pass as "
            "after_sequence. This can be newer than next_sequence when filters exclude events."
        ),
    )
    has_more: StrictBool


class ApiTranscriptMessage(ApiBaseModel):
    index: StrictInt = Field(ge=0)
    role: str
    content: list[dict[str, Any]]


class SessionTranscriptResponse(ApiBaseModel):
    session_id: str
    messages: list[ApiTranscriptMessage]
    offset: StrictInt = Field(ge=0)
    next_offset: StrictInt = Field(ge=0)
    has_more: StrictBool
    total_messages: StrictInt = Field(ge=0)


class ApiTaskListItem(ApiBaseModel):
    id: str
    type: str
    title: str | None
    description: str | None
    status: str
    status_reason: str | None
    status_payload: dict[str, Any] | None
    session_id: str | None
    parent_task_id: str | None
    assigned_agent_name: str | None
    worker_id: str | None
    lease_expires_at: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


class ApiTaskDetail(ApiTaskListItem):
    input: dict[str, Any]
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    metadata: dict[str, Any]
    started_at: str | None


class ApiKnowledgeEntryBase(ApiBaseModel):
    entry_id: str
    namespace: str
    kind: str
    visibility: str
    status: str
    title: str | None
    labels: dict[str, str]
    aspects: list[str]
    impact_targets: list[str]
    source_type: str | None
    source_uri: str | None
    source_id: str | None
    created_by_type: str
    created_by: str | None
    created_at: str
    updated_at: str
    importance: Decimal | None
    importance_source: str | None
    confidence: Decimal | None


class ApiKnowledgeListItem(ApiKnowledgeEntryBase):
    chunk_count: StrictInt = Field(ge=0)
    text_preview: str


class ApiReviewedKnowledgeEntry(ApiKnowledgeEntryBase):
    text_preview: str


class PendingKnowledgeListResponse(ApiBaseModel):
    entries: list[ApiKnowledgeListItem]
    truncated: StrictBool
    limit: StrictInt = Field(ge=1)
    max_bytes: StrictInt = Field(ge=1)
    total_entries_known: StrictInt = Field(ge=0)


class ApiKnowledgeChunk(ApiBaseModel):
    chunk_id: str
    entry_id: str
    chunk_index: StrictInt = Field(ge=0)
    text: str
    content_hash: str | None
    source_uri: str | None
    metadata: dict[str, Any]


class PendingKnowledgeDetailResponse(ApiKnowledgeEntryBase):
    text: str
    metadata: dict[str, Any]
    expires_at: str | None
    chunks: list[ApiKnowledgeChunk]
    chunk_limit: StrictInt = Field(ge=1)
    chunk_max_bytes: StrictInt = Field(ge=1)


STREAMING_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "Server-Sent Events stream. Each runtime event is emitted as an SSE frame "
            "whose `data:` value is a JSON SseEventEnvelope. A classified terminal "
            "runtime or observer condition is emitted as `event: error` with a "
            "SseErrorEnvelope payload."
        ),
        "content": {
            SSE_CONTENT_TYPE: {
                "schema": {
                    "type": "string",
                    "description": (
                        "SSE stream. Runtime `data:` frames contain SseEventEnvelope JSON; "
                        "`event: error` frames contain SseErrorEnvelope JSON."
                    ),
                }
            }
        },
    }
}


ARTIFACT_ENDPOINT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {
        "description": "The requested artifact store or artifact does not exist.",
        "model": ApiErrorResponse,
    },
    409: {
        "description": "Registered artifact-store identifiers are not unique.",
        "model": ApiErrorResponse,
    },
    500: {
        "description": "An artifact store is misconfigured or returned invalid data.",
        "model": ApiErrorResponse,
    },
    503: {
        "description": "An artifact store is unavailable.",
        "model": ApiErrorResponse,
    },
}


ARTIFACT_CONTENT_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "Complete artifact bytes. The response Content-Type reflects validated stored "
            "artifact metadata."
        ),
        "content": {
            "application/octet-stream": {
                "schema": {
                    "type": "string",
                    "format": "binary",
                }
            }
        },
        "headers": {
            "Content-Disposition": {
                "description": "Sanitized inline or attachment disposition and filename.",
                "schema": {"type": "string"},
            },
            "X-Content-Type-Options": {
                "description": "Always nosniff.",
                "schema": {"type": "string", "enum": ["nosniff"]},
            },
            "X-Cayu-Artifact-Id": {
                "description": "Sanitized artifact identifier.",
                "schema": {"type": "string"},
            },
            "X-Cayu-Artifact-Store-Id": {
                "description": "Sanitized artifact-store identifier.",
                "schema": {"type": "string"},
            },
            "Cache-Control": {
                "description": "Prevents authenticated artifact bytes from being cached.",
                "schema": {"type": "string", "enum": ["private, no-store"]},
            },
        },
    },
    413: {
        "description": "Artifact exceeds the direct content response limit.",
        "model": ApiErrorResponse,
    },
    **ARTIFACT_ENDPOINT_ERROR_RESPONSES,
}
