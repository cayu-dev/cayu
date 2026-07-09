"""Public server API contract models."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    CostLineItem,
    SessionCostSummary,
)
from cayu.runtime.usage import CausalBudgetUsageSummary, SessionUsageSummary, UsageMetrics

SERVER_API_PREFIX = "/api"
SERVER_CONTRACT_VERSION = "1"
SSE_CONTENT_TYPE = "text/event-stream"
SSE_LAST_EVENT_ID_FORMAT = "session_id:event_id"


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiBaseModel):
    ok: StrictBool


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
    """JSON payload in terminal SSE ``event: error`` frames."""

    type: Literal["stream.error"]
    error: str
    error_type: str


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
        error="Runtime stream failed.",
        error_type="RuntimeError",
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


class TranscriptSummary(ApiBaseModel):
    total_messages: StrictInt = Field(ge=0)


class SessionSummaryResponse(ApiBaseModel):
    session: ApiSession
    events: ApiEventSummary
    transcript: TranscriptSummary
    outcome: ApiSessionOutcome
    usage: SessionUsageSummary


class CausalBudgetSummaryResponse(ApiBaseModel):
    causal_budget_id: str
    session_count: StrictInt = Field(ge=0)
    sessions: list[ApiSessionSummaryItem]
    usage: CausalBudgetUsageSummary
    cost: CausalBudgetCostSummary


class ListSessionEventsResponse(ApiBaseModel):
    session_id: str
    events: list[ApiEventRecord]
    next_sequence: StrictInt | None = Field(default=None, ge=0)
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


class ApiSessionDetailEvent(ApiBaseModel):
    id: str
    type: str
    agent_name: str | None
    environment_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None
    payload: dict[str, Any]
    timestamp: str


class ApiSessionDetailTranscriptMessage(ApiBaseModel):
    role: str
    content: list[dict[str, Any]]


class SessionDetailResponse(ApiBaseModel):
    session: ApiSession
    events: list[ApiSessionDetailEvent]
    transcript: list[ApiSessionDetailTranscriptMessage]


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
            "whose `data:` value is a JSON SseEventEnvelope. A terminal runtime "
            "failure is emitted as `event: error` with a SseErrorEnvelope payload."
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
