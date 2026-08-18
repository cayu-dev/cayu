"""Bounded, serializable runtime evidence reconstructed from durable stores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import IntEnum, StrEnum
from heapq import heappop, heappush
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    compact_json_utf8_size,
    require_clean_nonblank,
)
from cayu.core.events import EventType
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook, estimate_model_step_cost
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    EventQueryResultTooLarge,
    EventRecord,
    Session,
    SessionLineageQuery,
    SessionOrder,
    SessionQuery,
    SessionStatus,
)
from cayu.runtime.tasks import TaskTopologyQuery
from cayu.runtime.tool_policy import taint_labels_from_metadata
from cayu.runtime.usage import AggregateCount, UsageMetrics, usage_metrics_from_event_payload

RUNTIME_EVIDENCE_SCHEMA_VERSION = 1

_HARD_MAX_SESSIONS = 500
_HARD_MAX_EVENTS = 100_000
_DEFAULT_MAX_CAUSAL_BUDGET_SESSIONS = 100
_EVENT_PAGE_SIZE = 5_000
_LINEAGE_PAGE_SIZE = 100
_MAX_EVENT_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_TASKS = 500
_TASK_SESSION_BATCH_SIZE = 50
_TASK_PAGE_SIZE = 100
_MAX_IDENTITY_CHARS = 1_024
_MAX_TAINT_LABELS = 64
_MAX_TAINT_LABEL_CHARS = 256
_MAX_COST_DECIMAL_DIGITS = 64
_MAX_COST_DECIMAL_PLACES = 64
_MAX_TOTAL_COST_DECIMAL_DIGITS = 128
_MAX_PRICING_CATALOG_ROWS = 512
_MAX_PRICING_CATALOG_BYTES = 1024 * 1024
_EVIDENCE_DECIMAL_CONTEXT = Context(
    prec=_MAX_TOTAL_COST_DECIMAL_DIGITS,
    rounding=ROUND_HALF_EVEN,
)
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

_ToolCallStatus = Literal["started", "completed", "failed", "blocked"]
_ApprovalDecision = Literal["pending", "approved", "denied", "expired", "blocked"]
_PolicyDecision = Literal["allow", "deny", "require_approval", "ambiguous"]


class _OperationPrecedence(IntEnum):
    UNKNOWN = 0
    ORDINARY_STEP = 1
    RUNTIME_PROTOCOL = 3
    SESSION_DECLARATION = 4
    EVENT_DECLARATION = 5


class RuntimeEvidenceOperation(StrEnum):
    """Why a retained provider attempt was dispatched."""

    AGENT_STEP = "agent_step"
    COMPACTION = "compaction"
    STRUCTURED_OUTPUT_REPAIR = "structured_output_repair"
    EVALUATION = "evaluation"
    REPAIR = "repair"
    COMPARISON_CONTROL = "comparison_control"
    UNKNOWN = "unknown"


class RuntimeEvidenceAttemptStatus(StrEnum):
    """Strongest terminal state observed for one provider attempt."""

    STARTED = "started"
    RETRY_SCHEDULED = "retry_scheduled"
    DISCARDED = "discarded"
    FAILED = "failed"
    COMPLETED = "completed"


class RuntimeEvidenceUsageStatus(StrEnum):
    """Availability of provider-reported usage for one attempt."""

    REPORTED = "reported"
    MISSING = "missing"
    MALFORMED = "malformed"


class RuntimeEvidenceCostStatus(StrEnum):
    """Availability of optional application-priced cost for one attempt."""

    NOT_REQUESTED = "not_requested"
    MISSING_USAGE = "missing_usage"
    PRICED = "priced"
    UNPRICED = "unpriced"


class RuntimeEvidenceWarningCode(StrEnum):
    """Stable reason some requested evidence is partial or unavailable."""

    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    UNKNOWN_OPERATION = "unknown_operation"
    LEGACY_ATTEMPT_IDENTITY = "legacy_attempt_identity"
    MISSING_USAGE = "missing_usage"
    MALFORMED_USAGE = "malformed_usage"
    UNPRICED_USAGE = "unpriced_usage"
    MALFORMED_CHECKPOINT = "malformed_checkpoint"
    MALFORMED_COMPACTION = "malformed_compaction"
    MALFORMED_TOOL_CALL = "malformed_tool_call"
    MALFORMED_APPROVAL = "malformed_approval"
    MALFORMED_RECEIPT = "malformed_receipt"
    MALFORMED_POLICY_DECISION = "malformed_policy_decision"
    MALFORMED_TAINT_LABELS = "malformed_taint_labels"
    ORIGIN_EVIDENCE_UNAVAILABLE = "origin_evidence_unavailable"
    TASK_EVIDENCE_UNAVAILABLE = "task_evidence_unavailable"
    TASK_EVIDENCE_PARTIAL = "task_evidence_partial"


class RuntimeEvidenceErrorCode(StrEnum):
    """Stable fail-closed error from :func:`runtime_evidence`."""

    ROOT_NOT_FOUND = "root_not_found"
    STORE_UNSUPPORTED = "store_unsupported"
    EVIDENCE_READ_FAILED = "evidence_read_failed"
    PARENT_CONTRADICTION = "parent_contradiction"
    CYCLE_DETECTED = "cycle_detected"
    SESSION_LIMIT_EXCEEDED = "session_limit_exceeded"
    CAUSAL_BUDGET_LIMIT_EXCEEDED = "causal_budget_limit_exceeded"
    EVENT_LIMIT_EXCEEDED = "event_limit_exceeded"
    EVENT_SOURCE_BYTES_EXCEEDED = "event_source_bytes_exceeded"


_ERROR_MESSAGES = {
    RuntimeEvidenceErrorCode.ROOT_NOT_FOUND: "The runtime-evidence root session does not exist.",
    RuntimeEvidenceErrorCode.STORE_UNSUPPORTED: (
        "The configured session store lacks a required bounded evidence query."
    ),
    RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED: (
        "The durable runtime-evidence snapshot could not be read completely."
    ),
    RuntimeEvidenceErrorCode.PARENT_CONTRADICTION: (
        "Durable session records disagree about a parent relationship."
    ),
    RuntimeEvidenceErrorCode.CYCLE_DETECTED: "The durable session lineage contains a cycle.",
    RuntimeEvidenceErrorCode.SESSION_LIMIT_EXCEEDED: (
        "The descendant lineage exceeds the configured session bound."
    ),
    RuntimeEvidenceErrorCode.CAUSAL_BUDGET_LIMIT_EXCEEDED: (
        "The optional causal-budget expansion exceeds its separate session bound."
    ),
    RuntimeEvidenceErrorCode.EVENT_LIMIT_EXCEEDED: (
        "The selected evidence scope exceeds the configured event bound."
    ),
    RuntimeEvidenceErrorCode.EVENT_SOURCE_BYTES_EXCEEDED: (
        "The bounded source-event read exceeds its internal byte ceiling."
    ),
}


class RuntimeEvidenceError(RuntimeError):
    """Typed runtime-evidence traversal or bounds failure."""

    def __init__(
        self,
        code: RuntimeEvidenceErrorCode,
        *,
        root_session_id: str,
        session_id: str | None = None,
        limit: int | None = None,
        observed: int | None = None,
    ) -> None:
        if not isinstance(code, RuntimeEvidenceErrorCode):
            raise TypeError("code must be a RuntimeEvidenceErrorCode.")
        self.code = code
        self.root_session_id = require_clean_nonblank(root_session_id, "root_session_id")
        self.session_id = (
            None if session_id is None else require_clean_nonblank(session_id, "session_id")
        )
        self.limit = limit
        self.observed = observed
        detail = _ERROR_MESSAGES[code]
        if limit is not None:
            detail = f"{detail} Limit: {limit}."
        if observed is not None:
            detail = f"{detail} Observed: {observed}."
        super().__init__(detail)


class RuntimeEvidenceRequest(BaseModel):
    """Bounded v1 request for a durable runtime-evidence projection."""

    model_config = _MODEL_CONFIG

    root_session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    max_sessions: StrictInt = Field(ge=1, le=_HARD_MAX_SESSIONS)
    max_events: StrictInt = Field(ge=1, le=_HARD_MAX_EVENTS)
    include_causal_budget: StrictBool = False
    max_causal_budget_sessions: StrictInt = Field(
        default=_DEFAULT_MAX_CAUSAL_BUDGET_SESSIONS,
        ge=1,
        le=_HARD_MAX_SESSIONS,
    )
    pricing: PriceBook | None = None

    @field_validator("root_session_id")
    @classmethod
    def validate_root_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "root_session_id")

    @field_validator("pricing")
    @classmethod
    def copy_pricing(cls, value: PriceBook | None) -> PriceBook | None:
        if value is None:
            return None
        if type(value) is not PriceBook:
            raise TypeError("pricing must be a PriceBook.")
        copied = PriceBook.model_validate(value.model_dump(mode="python"))
        copied_json = copied.model_dump(mode="json")
        row_count = (
            len(copied.prices)
            + len(copied.contextual_pricing_requirements)
            + len(copied.resource_mappings)
        )
        if row_count > _MAX_PRICING_CATALOG_ROWS:
            raise ValueError(
                f"pricing cannot contain more than {_MAX_PRICING_CATALOG_ROWS} catalog rows."
            )
        if compact_json_utf8_size(copied_json) > _MAX_PRICING_CATALOG_BYTES:
            raise ValueError(
                f"pricing cannot exceed {_MAX_PRICING_CATALOG_BYTES} canonical JSON bytes."
            )
        pending: list[object] = [copied_json]
        while pending:
            item = pending.pop()
            if type(item) is str and len(item) > _MAX_IDENTITY_CHARS:
                raise ValueError(
                    f"pricing text values cannot exceed {_MAX_IDENTITY_CHARS} characters."
                )
            if type(item) is dict:
                pending.extend(item.keys())
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
        return copied


class RuntimeEvidenceSourceRef(BaseModel):
    """Stable source event identity for one derived fact."""

    model_config = _MODEL_CONFIG

    event_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)


class RuntimeEvidenceEventCursor(RuntimeEvidenceSourceRef):
    """Last durable event retained for one session."""


class RuntimeEvidenceWarning(BaseModel):
    """One bounded, content-free evidence degradation marker."""

    model_config = _MODEL_CONFIG

    code: RuntimeEvidenceWarningCode
    session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    event_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    sequence: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)


class RuntimeEvidenceCacheUsage(BaseModel):
    """Identity-free cache counters safe for report serialization."""

    model_config = _MODEL_CONFIG

    read_tokens: AggregateCount = Field(ge=0)
    write_tokens: AggregateCount = Field(ge=0)
    write_5m_tokens: AggregateCount = Field(ge=0)
    write_1h_tokens: AggregateCount = Field(ge=0)
    write_unknown_ttl_tokens: AggregateCount = Field(ge=0)
    cached_input_tokens: AggregateCount = Field(ge=0)
    uncached_input_tokens: AggregateCount = Field(ge=0)


class RuntimeEvidenceUsage(BaseModel):
    """Provider-reported token counters with commercial identity removed."""

    model_config = _MODEL_CONFIG

    input_tokens: AggregateCount = Field(ge=0)
    output_tokens: AggregateCount = Field(ge=0)
    total_tokens: AggregateCount = Field(ge=0)
    reasoning_output_tokens: AggregateCount = Field(ge=0)
    cache: RuntimeEvidenceCacheUsage


class RuntimeEvidenceCost(BaseModel):
    """Bounded optional pricing result for one provider attempt."""

    model_config = _MODEL_CONFIG

    status: RuntimeEvidenceCostStatus
    currency: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    total_cost: Decimal | None = Field(default=None, ge=0)
    pricing_provider_name: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    pricing_model: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)

    @field_validator("total_cost")
    @classmethod
    def validate_total_cost(cls, value: Decimal | None) -> Decimal | None:
        return _bounded_decimal(
            value,
            "total_cost",
            max_digits=_MAX_COST_DECIMAL_DIGITS,
        )

    @model_validator(mode="after")
    def validate_status(self) -> RuntimeEvidenceCost:
        fields = (
            self.currency,
            self.total_cost,
            self.pricing_provider_name,
            self.pricing_model,
        )
        if self.status is RuntimeEvidenceCostStatus.PRICED:
            if self.currency is None or self.total_cost is None:
                raise ValueError("Priced attempts require currency and total_cost.")
        elif any(value is not None for value in fields):
            raise ValueError("Unpriced attempt states cannot retain priced cost fields.")
        return self


class RuntimeEvidenceCurrencyCost(BaseModel):
    """One currency-local priced subtotal."""

    model_config = _MODEL_CONFIG

    currency: str = Field(max_length=_MAX_IDENTITY_CHARS)
    total_cost: Decimal = Field(ge=0)

    @field_validator("total_cost")
    @classmethod
    def validate_total_cost(cls, value: Decimal) -> Decimal:
        bounded = _bounded_decimal(
            value,
            "total_cost",
            max_digits=_MAX_TOTAL_COST_DECIMAL_DIGITS,
        )
        assert bounded is not None
        return bounded


class RuntimeEvidenceOperationTotals(BaseModel):
    """Usage attributed to one typed operation."""

    model_config = _MODEL_CONFIG

    operation: RuntimeEvidenceOperation
    attempt_count: AggregateCount = Field(ge=0)
    usage: RuntimeEvidenceUsage


class RuntimeEvidenceTotals(BaseModel):
    """Recomputed accounting over one explicit evidence scope."""

    model_config = _MODEL_CONFIG

    session_count: AggregateCount = Field(ge=0)
    model_step_count: AggregateCount = Field(ge=0)
    attempt_count: AggregateCount = Field(ge=0)
    first_attempt_count: AggregateCount = Field(ge=0)
    provider_retry_attempt_count: AggregateCount = Field(ge=0)
    structured_output_repair_attempt_count: AggregateCount = Field(ge=0)
    tool_call_count: AggregateCount = Field(ge=0)
    missing_usage_attempt_count: AggregateCount = Field(ge=0)
    usage: RuntimeEvidenceUsage
    first_attempt_usage: RuntimeEvidenceUsage
    provider_retry_usage: RuntimeEvidenceUsage
    structured_output_repair_usage: RuntimeEvidenceUsage
    compaction_usage: RuntimeEvidenceUsage
    evaluation_usage: RuntimeEvidenceUsage
    repair_usage: RuntimeEvidenceUsage
    comparison_control_usage: RuntimeEvidenceUsage
    operations: tuple[RuntimeEvidenceOperationTotals, ...] = ()
    priced_costs: tuple[RuntimeEvidenceCurrencyCost, ...] = ()
    priced_attempt_count: AggregateCount = Field(default=0, ge=0)
    unpriced_attempt_count: AggregateCount = Field(default=0, ge=0)


class RuntimeEvidenceAttempt(BaseModel):
    """One durable provider dispatch and its allowlisted accounting evidence."""

    model_config = _MODEL_CONFIG

    attempt_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    model_step_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    operation: RuntimeEvidenceOperation
    attempt_ordinal: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    provider_name: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    requested_model: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    model: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    status: RuntimeEvidenceAttemptStatus
    usage_status: RuntimeEvidenceUsageStatus
    usage: RuntimeEvidenceUsage | None
    cost: RuntimeEvidenceCost
    source_refs: tuple[RuntimeEvidenceSourceRef, ...]


class RuntimeEvidenceToolCall(BaseModel):
    """Payload-free logical tool-call evidence."""

    model_config = _MODEL_CONFIG

    tool_call_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    tool_round_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    tool_name: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    idempotency_key: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    status: _ToolCallStatus
    source_refs: tuple[RuntimeEvidenceSourceRef, ...]


class RuntimeEvidenceApproval(BaseModel):
    """Secret-free approval identity and terminal decision."""

    model_config = _MODEL_CONFIG

    approval_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    tool_call_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    tool_round_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    policy_decision: _PolicyDecision | None = None
    decision: _ApprovalDecision
    source_refs: tuple[RuntimeEvidenceSourceRef, ...]


class RuntimeEvidenceCheckpoint(BaseModel):
    """Checkpoint publication identity without checkpoint contents."""

    model_config = _MODEL_CONFIG

    checkpoint_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    kind: Literal[
        "context_compaction",
        "pending_tool_approval",
        "pending_user_input",
        "manual_recovery",
        "custom",
        "unknown",
    ]
    compacted_transcript_cursor: StrictInt | None = Field(default=None, ge=0)
    source_ref: RuntimeEvidenceSourceRef


class RuntimeEvidenceCompaction(BaseModel):
    """Payload-free compaction identity, outcome, and durable provenance."""

    model_config = _MODEL_CONFIG

    compaction_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    status: Literal["started", "completed", "failed"]
    source_refs: tuple[RuntimeEvidenceSourceRef, ...]


class RuntimeEvidenceTask(BaseModel):
    """Payload-free durable task identity linked to a retained session."""

    model_config = _MODEL_CONFIG

    task_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    parent_task_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    status: Literal[
        "pending",
        "claimed",
        "running",
        "paused",
        "blocked",
        "needs_attention",
        "completed",
        "failed",
        "cancelled",
    ]


class RuntimeEvidenceReceipt(BaseModel):
    """External-effect receipt identity without body or tool result."""

    model_config = _MODEL_CONFIG

    receipt_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    tool_call_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    reconciliation_state: Literal["recorded", "reconciled", "unknown"]
    source_ref: RuntimeEvidenceSourceRef


class RuntimeEvidencePolicyDecision(BaseModel):
    """Allowlisted policy outcome and the event that recorded it."""

    model_config = _MODEL_CONFIG

    decision: Literal["allow", "deny", "require_approval", "ambiguous", "blocked"]
    tool_call_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    tool_name: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    matched_taint_labels: tuple[str, ...] = ()
    source_ref: RuntimeEvidenceSourceRef


class RuntimeEvidenceRecoverySummary(BaseModel):
    """Content-free interruption and recovery counts."""

    model_config = _MODEL_CONFIG

    interruption_count: AggregateCount = Field(ge=0)
    resume_count: AggregateCount = Field(ge=0)
    manual_recovery_required_count: AggregateCount = Field(ge=0)
    manual_reconciliation_count: AggregateCount = Field(ge=0)
    source_refs: tuple[RuntimeEvidenceSourceRef, ...] = ()


class RuntimeEvidenceSession(BaseModel):
    """Stable, safe evidence projection for one durable session."""

    model_config = _MODEL_CONFIG

    session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    parent_session_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    agent_name: str = Field(max_length=_MAX_IDENTITY_CHARS)
    task_ids: tuple[str, ...] = ()
    causal_budget_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    provider_name: str = Field(max_length=_MAX_IDENTITY_CHARS)
    model: str = Field(max_length=_MAX_IDENTITY_CHARS)
    status: SessionStatus
    last_event_cursor: RuntimeEvidenceEventCursor | None
    origin_refs: tuple[RuntimeEvidenceSourceRef, ...] = ()
    checkpoints: tuple[RuntimeEvidenceCheckpoint, ...] = ()
    compactions: tuple[RuntimeEvidenceCompaction, ...] = ()
    compaction_count: AggregateCount = Field(default=0, ge=0)
    attempts: tuple[RuntimeEvidenceAttempt, ...] = ()
    tool_calls: tuple[RuntimeEvidenceToolCall, ...] = ()
    approvals: tuple[RuntimeEvidenceApproval, ...] = ()
    effective_taint_labels: tuple[str, ...] = ()
    policy_decisions: tuple[RuntimeEvidencePolicyDecision, ...] = ()
    recovery: RuntimeEvidenceRecoverySummary
    receipts: tuple[RuntimeEvidenceReceipt, ...] = ()
    totals: RuntimeEvidenceTotals


class RuntimeEvidenceBranchTotals(BaseModel):
    """Totals for one direct child subtree of the requested root."""

    model_config = _MODEL_CONFIG

    branch_root_session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    session_ids: tuple[str, ...] = Field(max_length=_HARD_MAX_SESSIONS)
    totals: RuntimeEvidenceTotals


class RuntimeEvidenceScope(BaseModel):
    """Exact retained descendant and optional causal-budget scope."""

    model_config = _MODEL_CONFIG

    root_session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    causal_budget_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    descendant_session_ids: tuple[str, ...] = Field(max_length=_HARD_MAX_SESSIONS)
    causal_budget_session_ids: tuple[str, ...] | None = Field(
        default=None,
        max_length=_HARD_MAX_SESSIONS,
    )


class RuntimeEvidenceReport(BaseModel):
    """Versioned, deterministic runtime-evidence read model."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[1] = RUNTIME_EVIDENCE_SCHEMA_VERSION
    root_session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    scope: RuntimeEvidenceScope
    sessions: tuple[RuntimeEvidenceSession, ...] = Field(max_length=_HARD_MAX_SESSIONS)
    tasks: tuple[RuntimeEvidenceTask, ...] = Field(default=(), max_length=_MAX_TASKS)
    lineage_totals: RuntimeEvidenceTotals
    branch_totals: tuple[RuntimeEvidenceBranchTotals, ...]
    causal_budget_totals: RuntimeEvidenceTotals | None = None
    whole_workflow_totals: RuntimeEvidenceTotals | None = None
    pricing_catalog_version: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    pricing_catalog_generated_at: str | None = Field(
        default=None,
        max_length=_MAX_IDENTITY_CHARS,
    )
    warnings: tuple[RuntimeEvidenceWarning, ...] = ()


@dataclass(slots=True)
class _Capture:
    request: RuntimeEvidenceRequest
    sessions: dict[str, Session] = field(default_factory=dict)
    descendant_ids: set[str] = field(default_factory=set)
    causal_budget_ids: set[str] = field(default_factory=set)
    origin_refs: dict[str, tuple[RuntimeEvidenceSourceRef, ...]] = field(default_factory=dict)
    records: dict[str, tuple[EventRecord, ...]] = field(default_factory=dict)
    event_count: int = 0
    event_source_bytes: int = 0
    warnings: list[RuntimeEvidenceWarning] = field(default_factory=list)
    tasks: list[RuntimeEvidenceTask] = field(default_factory=list)
    task_ids_by_session: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


@dataclass(slots=True)
class _AttemptRecord:
    attempt_id: str
    model_step_id: str | None
    operation: RuntimeEvidenceOperation
    attempt_ordinal: int
    provider_name: str | None = None
    requested_model: str | None = None
    model: str | None = None
    status: RuntimeEvidenceAttemptStatus = RuntimeEvidenceAttemptStatus.STARTED
    usage_status: RuntimeEvidenceUsageStatus = RuntimeEvidenceUsageStatus.MISSING
    usage: RuntimeEvidenceUsage | None = None
    usage_warning_recorded: bool = False
    metrics: UsageMetrics | None = None
    completed_at: datetime | None = None
    source_refs: list[RuntimeEvidenceSourceRef] = field(default_factory=list)


@dataclass(slots=True)
class _ToolCallRecord:
    tool_call_id: str
    tool_round_id: str | None = None
    tool_name: str | None = None
    idempotency_key: str | None = None
    status: _ToolCallStatus = "started"
    source_refs: list[RuntimeEvidenceSourceRef] = field(default_factory=list)


@dataclass(slots=True)
class _ApprovalRecord:
    approval_id: str
    tool_call_id: str
    tool_round_id: str
    policy_decision: _PolicyDecision | None = None
    decision: _ApprovalDecision = "pending"
    source_refs: list[RuntimeEvidenceSourceRef] = field(default_factory=list)


@dataclass(slots=True)
class _CompactionRecord:
    compaction_id: str
    status: Literal["started", "completed", "failed"] = "started"
    source_refs: list[RuntimeEvidenceSourceRef] = field(default_factory=list)


async def runtime_evidence(
    app: CayuApp,
    request: RuntimeEvidenceRequest,
) -> RuntimeEvidenceReport:
    """Reconstruct one bounded, deterministic runtime-evidence report."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    if type(request) is not RuntimeEvidenceRequest:
        raise TypeError("request must be a RuntimeEvidenceRequest.")
    selected = RuntimeEvidenceRequest.model_validate(request.model_dump(mode="python"))
    store = app.session_store
    if not store.supports_session_lineage:
        raise RuntimeEvidenceError(
            RuntimeEvidenceErrorCode.STORE_UNSUPPORTED,
            root_session_id=selected.root_session_id,
        )
    try:
        root = await store.load(selected.root_session_id)
    except Exception as exc:
        raise RuntimeEvidenceError(
            RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
            root_session_id=selected.root_session_id,
        ) from exc
    if root is None:
        raise RuntimeEvidenceError(
            RuntimeEvidenceErrorCode.ROOT_NOT_FOUND,
            root_session_id=selected.root_session_id,
        )

    capture = _Capture(request=selected)
    capture.sessions[root.id] = root
    capture.descendant_ids.add(root.id)
    await _capture_descendants(app, root, capture)
    if selected.include_causal_budget:
        await _capture_causal_budget(app, root, capture)
    ordered_sessions = _stable_parent_first(capture.sessions, root_session_id=root.id)
    for session in ordered_sessions:
        capture.records[session.id] = await _load_event_records(app, session.id, capture)
    await _capture_tasks(app, ordered_sessions, capture)

    projected_sessions = tuple(
        _project_session(
            session,
            capture.records[session.id],
            capture,
            pricing=selected.pricing,
        )
        for session in ordered_sessions
    )
    descendant_sessions = tuple(
        session for session in projected_sessions if session.session_id in capture.descendant_ids
    )
    lineage_totals = _totals(descendant_sessions)
    branches = _branch_totals(root.id, descendant_sessions)
    descendant_ids = tuple(
        session.session_id
        for session in projected_sessions
        if session.session_id in capture.descendant_ids
    )
    causal_ids = (
        tuple(
            session.session_id
            for session in projected_sessions
            if session.session_id in capture.causal_budget_ids
        )
        if selected.include_causal_budget
        else None
    )
    causal_sessions = tuple(
        session for session in projected_sessions if session.session_id in capture.causal_budget_ids
    )
    causal_totals = _totals(causal_sessions) if selected.include_causal_budget else None
    return RuntimeEvidenceReport(
        root_session_id=root.id,
        scope=RuntimeEvidenceScope(
            root_session_id=root.id,
            causal_budget_id=root.causal_budget_id,
            descendant_session_ids=descendant_ids,
            causal_budget_session_ids=causal_ids,
        ),
        sessions=projected_sessions,
        tasks=tuple(capture.tasks),
        lineage_totals=lineage_totals,
        branch_totals=branches,
        causal_budget_totals=causal_totals,
        whole_workflow_totals=causal_totals,
        pricing_catalog_version=(
            None if selected.pricing is None else selected.pricing.price_book_version
        ),
        pricing_catalog_generated_at=(
            None if selected.pricing is None else selected.pricing.generated_at
        ),
        warnings=tuple(capture.warnings),
    )


async def _capture_descendants(app: CayuApp, root: Session, capture: _Capture) -> None:
    queue = [root.id]
    while queue:
        parent_id = queue.pop(0)
        cursor: str | None = None
        while True:
            remaining = capture.request.max_sessions - len(capture.descendant_ids)
            page_limit = min(_LINEAGE_PAGE_SIZE, max(1, remaining + 1))
            try:
                page = await app.session_store.query_session_lineage(
                    SessionLineageQuery(
                        parent_session_id=parent_id,
                        cursor=cursor,
                        limit=page_limit,
                    )
                )
            except RuntimeEvidenceError:
                raise
            except Exception as exc:
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                    root_session_id=root.id,
                    session_id=parent_id,
                ) from exc
            for node in page.children:
                if node.id in capture.descendant_ids:
                    raise RuntimeEvidenceError(
                        RuntimeEvidenceErrorCode.CYCLE_DETECTED,
                        root_session_id=root.id,
                        session_id=node.id,
                    )
                observed = len(capture.descendant_ids) + 1
                if observed > capture.request.max_sessions:
                    raise RuntimeEvidenceError(
                        RuntimeEvidenceErrorCode.SESSION_LIMIT_EXCEEDED,
                        root_session_id=root.id,
                        session_id=node.id,
                        limit=capture.request.max_sessions,
                        observed=observed,
                    )
                try:
                    child = await app.session_store.load(node.id)
                except Exception as exc:
                    raise RuntimeEvidenceError(
                        RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                        root_session_id=root.id,
                        session_id=node.id,
                    ) from exc
                if (
                    child is None
                    or child.parent_session_id != parent_id
                    or child.created_at != node.created_at
                ):
                    raise RuntimeEvidenceError(
                        RuntimeEvidenceErrorCode.PARENT_CONTRADICTION,
                        root_session_id=root.id,
                        session_id=node.id,
                    )
                refs = tuple(
                    RuntimeEvidenceSourceRef(
                        event_id=origin.event_id,
                        sequence=origin.sequence,
                    )
                    for origin in node.origin_events
                )
                if not refs:
                    capture.warnings.append(
                        RuntimeEvidenceWarning(
                            code=RuntimeEvidenceWarningCode.ORIGIN_EVIDENCE_UNAVAILABLE,
                            session_id=node.id,
                        )
                    )
                capture.sessions[child.id] = child
                capture.descendant_ids.add(child.id)
                capture.origin_refs[child.id] = refs
                queue.append(child.id)
            if not page.has_more:
                break
            if page.next_cursor is None or page.next_cursor == cursor:
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                    root_session_id=root.id,
                    session_id=parent_id,
                )
            cursor = page.next_cursor


async def _capture_causal_budget(app: CayuApp, root: Session, capture: _Capture) -> None:
    cursor: str | None = None
    observed_ids: set[str] = set()
    while True:
        remaining = capture.request.max_causal_budget_sessions - len(observed_ids)
        page_limit = min(1_000, max(1, remaining + 1))
        try:
            page = await app.session_store.list_sessions(
                SessionQuery(
                    causal_budget_id=root.causal_budget_id,
                    cursor=cursor,
                    limit=page_limit,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        except Exception as exc:
            raise RuntimeEvidenceError(
                RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                root_session_id=root.id,
            ) from exc
        for session in page.sessions:
            if session.causal_budget_id != root.causal_budget_id:
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                    root_session_id=root.id,
                    session_id=session.id,
                )
            if session.id in observed_ids:
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                    root_session_id=root.id,
                    session_id=session.id,
                )
            observed = len(observed_ids) + 1
            if observed > capture.request.max_causal_budget_sessions:
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.CAUSAL_BUDGET_LIMIT_EXCEEDED,
                    root_session_id=root.id,
                    session_id=session.id,
                    limit=capture.request.max_causal_budget_sessions,
                    observed=observed,
                )
            prior = capture.sessions.get(session.id)
            if prior is not None and not _same_session_identity(prior, session):
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.PARENT_CONTRADICTION,
                    root_session_id=root.id,
                    session_id=session.id,
                )
            if prior is None and len(capture.sessions) >= _HARD_MAX_SESSIONS:
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.SESSION_LIMIT_EXCEEDED,
                    root_session_id=root.id,
                    session_id=session.id,
                    limit=_HARD_MAX_SESSIONS,
                    observed=len(capture.sessions) + 1,
                )
            capture.sessions[session.id] = session
            capture.causal_budget_ids.add(session.id)
            observed_ids.add(session.id)
        if page.next_cursor is None:
            break
        if page.next_cursor == cursor:
            raise RuntimeEvidenceError(
                RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                root_session_id=root.id,
            )
        cursor = page.next_cursor


async def _capture_tasks(
    app: CayuApp,
    sessions: tuple[Session, ...],
    capture: _Capture,
) -> None:
    task_store = app.task_store
    if task_store is None or not task_store.supports_task_topology:
        capture.warnings.append(
            RuntimeEvidenceWarning(
                code=RuntimeEvidenceWarningCode.TASK_EVIDENCE_UNAVAILABLE,
                session_id=capture.request.root_session_id,
            )
        )
        return
    seen: dict[str, RuntimeEvidenceTask] = {}

    def retain_seen() -> None:
        capture.tasks.extend(seen.values())

    session_ids = tuple(session.id for session in sessions)
    for offset in range(0, len(session_ids), _TASK_SESSION_BATCH_SIZE):
        pending = session_ids[offset : offset + _TASK_SESSION_BATCH_SIZE]
        cursors: dict[str, str] = {}
        while pending:
            try:
                result = await task_store.query_task_topology(
                    TaskTopologyQuery(
                        linked_session_ids=pending,
                        session_cursors=cursors,
                        session_task_limit=_TASK_PAGE_SIZE,
                    )
                )
            except Exception:
                capture.warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.TASK_EVIDENCE_UNAVAILABLE,
                        session_id=capture.request.root_session_id,
                    )
                )
                retain_seen()
                return
            next_pending: list[str] = []
            next_cursors: dict[str, str] = {}
            for branch in result.session_branches:
                for node in branch.tasks:
                    if len(seen) >= _MAX_TASKS and node.id not in seen:
                        capture.warnings.append(
                            RuntimeEvidenceWarning(
                                code=RuntimeEvidenceWarningCode.TASK_EVIDENCE_PARTIAL,
                                session_id=capture.request.root_session_id,
                            )
                        )
                        retain_seen()
                        return
                    projected = RuntimeEvidenceTask(
                        task_id=node.id,
                        session_id=branch.session_id,
                        parent_task_id=node.parent_task_id,
                        status=node.status.value,
                    )
                    prior = seen.setdefault(node.id, projected)
                    if prior != projected:
                        capture.warnings.append(
                            RuntimeEvidenceWarning(
                                code=RuntimeEvidenceWarningCode.TASK_EVIDENCE_PARTIAL,
                                session_id=branch.session_id,
                            )
                        )
                        retain_seen()
                        return
                    if node.id not in capture.task_ids_by_session[branch.session_id]:
                        capture.task_ids_by_session[branch.session_id].append(node.id)
                if branch.has_more:
                    if branch.next_cursor is None:
                        capture.warnings.append(
                            RuntimeEvidenceWarning(
                                code=RuntimeEvidenceWarningCode.TASK_EVIDENCE_PARTIAL,
                                session_id=branch.session_id,
                            )
                        )
                        retain_seen()
                        return
                    next_pending.append(branch.session_id)
                    next_cursors[branch.session_id] = branch.next_cursor
            pending = tuple(next_pending)
            cursors = next_cursors
    retain_seen()


def _stable_parent_first(
    sessions: dict[str, Session],
    *,
    root_session_id: str,
) -> tuple[Session, ...]:
    children: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {session_id: 0 for session_id in sessions}
    for session in sessions.values():
        parent_id = session.parent_session_id
        if parent_id is None or parent_id not in sessions:
            continue
        children[parent_id].append(session.id)
        indegree[session.id] += 1
    ready: list[tuple[datetime, str]] = []
    for session_id, degree in indegree.items():
        if degree == 0:
            session = sessions[session_id]
            heappush(ready, (session.created_at, session.id))
    ordered: list[Session] = []
    while ready:
        _, session_id = heappop(ready)
        ordered.append(sessions[session_id])
        for child_id in children.get(session_id, ()):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                child = sessions[child_id]
                heappush(ready, (child.created_at, child.id))
    if len(ordered) != len(sessions):
        raise RuntimeEvidenceError(
            RuntimeEvidenceErrorCode.CYCLE_DETECTED,
            root_session_id=root_session_id,
        )
    return tuple(ordered)


async def _load_event_records(
    app: CayuApp,
    session_id: str,
    capture: _Capture,
) -> tuple[EventRecord, ...]:
    records: list[EventRecord] = []
    after_sequence = 0
    while True:
        remaining_events = capture.request.max_events - capture.event_count
        page_limit = min(_EVENT_PAGE_SIZE, max(1, remaining_events + 1))
        remaining_bytes = _MAX_EVENT_SOURCE_BYTES - capture.event_source_bytes
        if remaining_bytes <= 0:
            raise RuntimeEvidenceError(
                RuntimeEvidenceErrorCode.EVENT_SOURCE_BYTES_EXCEEDED,
                root_session_id=capture.request.root_session_id,
                session_id=session_id,
                limit=_MAX_EVENT_SOURCE_BYTES,
                observed=capture.event_source_bytes,
            )
        try:
            page = await app.session_store.query_events_bounded(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    limit=page_limit,
                    order_by=EventOrder.SEQUENCE_ASC,
                ),
                max_bytes=remaining_bytes,
            )
        except EventQueryResultTooLarge as exc:
            raise RuntimeEvidenceError(
                RuntimeEvidenceErrorCode.EVENT_SOURCE_BYTES_EXCEEDED,
                root_session_id=capture.request.root_session_id,
                session_id=session_id,
                limit=_MAX_EVENT_SOURCE_BYTES,
                observed=_MAX_EVENT_SOURCE_BYTES + 1,
            ) from exc
        except NotImplementedError as exc:
            raise RuntimeEvidenceError(
                RuntimeEvidenceErrorCode.STORE_UNSUPPORTED,
                root_session_id=capture.request.root_session_id,
                session_id=session_id,
            ) from exc
        except Exception as exc:
            raise RuntimeEvidenceError(
                RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                root_session_id=capture.request.root_session_id,
                session_id=session_id,
            ) from exc
        if len(page) > remaining_events:
            raise RuntimeEvidenceError(
                RuntimeEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                root_session_id=capture.request.root_session_id,
                session_id=session_id,
                limit=capture.request.max_events,
                observed=capture.event_count + len(page),
            )
        if not page:
            break
        for record in page:
            if record.event.session_id != session_id or record.sequence <= after_sequence:
                raise RuntimeEvidenceError(
                    RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED,
                    root_session_id=capture.request.root_session_id,
                    session_id=session_id,
                )
            capture.event_source_bytes += compact_json_utf8_size(record.model_dump(mode="json"))
            records.append(record.model_copy(deep=True))
        capture.event_count += len(page)
        after_sequence = page[-1].sequence
        if len(page) < page_limit:
            break
    return tuple(records)


def _project_session(
    session: Session,
    records: tuple[EventRecord, ...],
    capture: _Capture,
    *,
    pricing: PriceBook | None,
) -> RuntimeEvidenceSession:
    attempts = _project_attempts(session, records, capture.warnings, pricing=pricing)
    checkpoints = _project_checkpoints(session.id, records, capture.warnings)
    compactions = _project_compactions(session.id, records, capture.warnings)
    tool_calls = _project_tool_calls(session.id, records, capture.warnings)
    approvals = _project_approvals(session.id, records, capture.warnings)
    policy_decisions = _project_policy_decisions(session.id, records, capture.warnings)
    receipts = _project_receipts(session.id, records, capture.warnings)
    last_cursor = (
        None
        if not records
        else RuntimeEvidenceEventCursor(
            event_id=records[-1].event.id,
            sequence=records[-1].sequence,
        )
    )
    recovery_refs = tuple(
        _source_ref(record)
        for record in records
        if record.event.type
        in {
            EventType.SESSION_INTERRUPTED,
            EventType.SESSION_RESUMED,
        }
        or _is_manual_reconciliation(record)
    )
    interruptions = sum(record.event.type == EventType.SESSION_INTERRUPTED for record in records)
    resumes = sum(record.event.type == EventType.SESSION_RESUMED for record in records)
    manual_recovery_required = sum(
        record.event.type == EventType.SESSION_INTERRUPTED
        and record.event.payload.get("manual_recovery_required") is True
        for record in records
    )
    reconciliations = sum(_is_manual_reconciliation(record) for record in records)
    totals = _totals_from_attempts(
        attempts,
        session_count=1,
        tool_call_count=len(tool_calls),
    )
    try:
        taint_labels = _safe_taint_labels(taint_labels_from_metadata(session.metadata))
    except (TypeError, ValueError):
        taint_labels = None
    if taint_labels is None:
        taint_labels = ()
        capture.warnings.append(
            RuntimeEvidenceWarning(
                code=RuntimeEvidenceWarningCode.MALFORMED_TAINT_LABELS,
                session_id=session.id,
            )
        )
    return RuntimeEvidenceSession(
        session_id=session.id,
        parent_session_id=session.parent_session_id,
        agent_name=session.agent_name,
        causal_budget_id=session.causal_budget_id,
        provider_name=session.provider_name,
        model=session.model,
        status=session.status,
        last_event_cursor=last_cursor,
        origin_refs=capture.origin_refs.get(session.id, ()),
        task_ids=tuple(capture.task_ids_by_session.get(session.id, ())),
        checkpoints=checkpoints,
        compactions=compactions,
        compaction_count=len(compactions),
        attempts=attempts,
        tool_calls=tool_calls,
        approvals=approvals,
        effective_taint_labels=taint_labels,
        policy_decisions=policy_decisions,
        recovery=RuntimeEvidenceRecoverySummary(
            interruption_count=interruptions,
            resume_count=resumes,
            manual_recovery_required_count=manual_recovery_required,
            manual_reconciliation_count=reconciliations,
            source_refs=recovery_refs,
        ),
        receipts=receipts,
        totals=totals,
    )


def _project_checkpoints(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> tuple[RuntimeEvidenceCheckpoint, ...]:
    allowed = {
        "context_compaction",
        "pending_tool_approval",
        "pending_user_input",
        "manual_recovery",
    }
    projected: list[RuntimeEvidenceCheckpoint] = []
    for record in records:
        if record.event.type != EventType.SESSION_CHECKPOINTED:
            continue
        raw_kind = record.event.payload.get("checkpoint")
        if raw_kind is None:
            kind = "unknown"
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_CHECKPOINT,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
        elif type(raw_kind) is str and raw_kind in allowed:
            kind = raw_kind
        elif type(raw_kind) is str:
            kind = "custom"
        else:
            kind = "unknown"
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_CHECKPOINT,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
        projected.append(
            RuntimeEvidenceCheckpoint(
                checkpoint_id=record.event.id,
                kind=kind,
                compacted_transcript_cursor=_nonnegative_int(
                    record.event.payload.get("compacted_transcript_cursor")
                ),
                source_ref=_source_ref(record),
            )
        )
    return tuple(projected)


def _project_compactions(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> tuple[RuntimeEvidenceCompaction, ...]:
    builders: dict[str, _CompactionRecord] = {}
    active: list[str] = []
    terminal_types: dict[EventType, Literal["completed", "failed"]] = {
        EventType.CONTEXT_COMPACTION_COMPLETED: "completed",
        EventType.CONTEXT_COMPACTION_FAILED: "failed",
    }
    for record in records:
        event_type = record.event.type
        if event_type not in {EventType.CONTEXT_COMPACTION_STARTED, *terminal_types}:
            continue
        payload = record.event.payload
        explicit_id = _optional_text(payload.get("operation_id") or payload.get("compaction_id"))
        if event_type == EventType.CONTEXT_COMPACTION_STARTED:
            compaction_id = explicit_id or record.event.id
            builder = builders.get(compaction_id)
            if builder is None:
                builder = _CompactionRecord(compaction_id=compaction_id)
                builders[compaction_id] = builder
                active.append(compaction_id)
            elif builder.status == "started":
                if compaction_id not in active:
                    active.append(compaction_id)
            else:
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.MALFORMED_COMPACTION,
                        session_id=session_id,
                        event_id=record.event.id,
                        sequence=record.sequence,
                    )
                )
                continue
        else:
            terminal_event_type = (
                event_type if isinstance(event_type, EventType) else EventType(event_type)
            )
            compaction_id = explicit_id or (active[-1] if active else record.event.id)
            builder = builders.get(compaction_id)
            if builder is None:
                builder = _CompactionRecord(compaction_id=compaction_id)
                builders[compaction_id] = builder
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.MALFORMED_COMPACTION,
                        session_id=session_id,
                        event_id=record.event.id,
                        sequence=record.sequence,
                    )
                )
            elif builder.status != "started":
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.MALFORMED_COMPACTION,
                        session_id=session_id,
                        event_id=record.event.id,
                        sequence=record.sequence,
                    )
                )
            builder.status = terminal_types[terminal_event_type]
            if compaction_id in active:
                active.remove(compaction_id)
        builder.source_refs.append(_source_ref(record))
    return tuple(
        RuntimeEvidenceCompaction(
            compaction_id=builder.compaction_id,
            status=builder.status,
            source_refs=tuple(builder.source_refs),
        )
        for builder in builders.values()
    )


def _project_tool_calls(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> tuple[RuntimeEvidenceToolCall, ...]:
    statuses: dict[EventType, _ToolCallStatus] = {
        EventType.TOOL_CALL_STARTED: "started",
        EventType.TOOL_CALL_COMPLETED: "completed",
        EventType.TOOL_CALL_FAILED: "failed",
        EventType.TOOL_CALL_BLOCKED: "blocked",
    }
    builders: dict[str, _ToolCallRecord] = {}
    for record in records:
        if record.event.type not in statuses:
            continue
        tool_call_id = _optional_text(record.event.payload.get("tool_call_id"))
        if tool_call_id is None:
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_TOOL_CALL,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
            continue
        builder = builders.setdefault(tool_call_id, _ToolCallRecord(tool_call_id=tool_call_id))
        builder.tool_round_id = builder.tool_round_id or _optional_text(
            record.event.payload.get("tool_round_id")
        )
        builder.tool_name = builder.tool_name or _optional_text(
            record.event.tool_name or record.event.payload.get("tool_name")
        )
        builder.idempotency_key = builder.idempotency_key or _optional_text(
            record.event.payload.get("idempotency_key")
        )
        builder.status = statuses[EventType(record.event.type)]
        builder.source_refs.append(_source_ref(record))
    return tuple(
        RuntimeEvidenceToolCall(
            tool_call_id=builder.tool_call_id,
            tool_round_id=builder.tool_round_id,
            tool_name=builder.tool_name,
            idempotency_key=builder.idempotency_key,
            status=builder.status,
            source_refs=tuple(builder.source_refs),
        )
        for builder in builders.values()
    )


def _project_approvals(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> tuple[RuntimeEvidenceApproval, ...]:
    decisions: dict[EventType, _ApprovalDecision] = {
        EventType.TOOL_CALL_APPROVAL_REQUESTED: "pending",
        EventType.TOOL_CALL_APPROVED: "approved",
        EventType.TOOL_CALL_APPROVAL_DENIED: "denied",
        EventType.TOOL_CALL_APPROVAL_EXPIRED: "expired",
    }
    builders: dict[str, _ApprovalRecord] = {}
    for record in records:
        event_type = record.event.type
        if event_type not in decisions and event_type != EventType.TOOL_CALL_BLOCKED:
            continue
        payload = record.event.payload
        nested = payload.get("approval")
        nested_payload = nested if type(nested) is dict else {}
        approval_id = _optional_text(
            payload.get("approval_id") or nested_payload.get("approval_id")
        )
        if approval_id is None:
            if event_type in decisions:
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.MALFORMED_APPROVAL,
                        session_id=session_id,
                        event_id=record.event.id,
                        sequence=record.sequence,
                    )
                )
            continue
        tool_call_id = _optional_text(
            payload.get("tool_call_id") or nested_payload.get("tool_call_id")
        )
        tool_round_id = _optional_text(
            payload.get("tool_round_id") or nested_payload.get("tool_round_id")
        )
        existing = builders.get(approval_id)
        if existing is None:
            if tool_call_id is None or tool_round_id is None:
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.MALFORMED_APPROVAL,
                        session_id=session_id,
                        event_id=record.event.id,
                        sequence=record.sequence,
                    )
                )
                continue
            existing = _ApprovalRecord(
                approval_id=approval_id,
                tool_call_id=tool_call_id,
                tool_round_id=tool_round_id,
            )
            builders[approval_id] = existing
        elif (tool_call_id is not None and tool_call_id != existing.tool_call_id) or (
            tool_round_id is not None and tool_round_id != existing.tool_round_id
        ):
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_APPROVAL,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
            continue
        raw_policy = payload.get("policy_decision") or nested_payload.get("policy_decision")
        if raw_policy in {"allow", "deny", "require_approval", "ambiguous"}:
            existing.policy_decision = raw_policy
        existing.decision = (
            decisions[EventType(event_type)] if event_type in decisions else "blocked"
        )
        existing.source_refs.append(_source_ref(record))
    return tuple(
        RuntimeEvidenceApproval(
            approval_id=builder.approval_id,
            tool_call_id=builder.tool_call_id,
            tool_round_id=builder.tool_round_id,
            policy_decision=builder.policy_decision,
            decision=builder.decision,
            source_refs=tuple(builder.source_refs),
        )
        for builder in builders.values()
    )


def _project_policy_decisions(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> tuple[RuntimeEvidencePolicyDecision, ...]:
    projected: list[RuntimeEvidencePolicyDecision] = []
    allowed = {"allow", "deny", "require_approval", "ambiguous", "blocked"}
    for record in records:
        if record.event.type != EventType.TOOL_CALL_BLOCKED:
            continue
        raw_decision = record.event.payload.get("decision")
        if raw_decision is None and record.event.payload.get("denied_by") is not None:
            raw_decision = "deny"
        if raw_decision not in allowed:
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_POLICY_DECISION,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
            continue
        metadata = record.event.payload.get("metadata")
        metadata = metadata if type(metadata) is dict else {}
        labels = _safe_taint_labels(
            record.event.payload.get("matched_taint_labels") or metadata.get("matched_taint_labels")
        )
        if labels is None:
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_TAINT_LABELS,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
            labels = ()
        projected.append(
            RuntimeEvidencePolicyDecision(
                decision=raw_decision,
                tool_call_id=_optional_text(record.event.payload.get("tool_call_id")),
                tool_name=_optional_text(
                    record.event.tool_name or record.event.payload.get("tool_name")
                ),
                matched_taint_labels=labels,
                source_ref=_source_ref(record),
            )
        )
    return tuple(projected)


def _project_receipts(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> tuple[RuntimeEvidenceReceipt, ...]:
    projected: dict[str, RuntimeEvidenceReceipt] = {}
    for record in records:
        if record.event.type not in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.PROVIDER_OPERATION_RECONCILED,
        }:
            continue
        payload = record.event.payload
        result = payload.get("result")
        result = result if type(result) is dict else {}
        structured = result.get("structured")
        structured = structured if type(structured) is dict else {}
        portable = structured.get("portable_result_evidence")
        portable = portable if type(portable) is dict else {}
        portable_structured = portable.get("structured")
        portable_structured = portable_structured if type(portable_structured) is dict else {}
        raw_receipt_id = (
            payload.get("receipt_id")
            or structured.get("receipt_id")
            or portable_structured.get("receipt_id")
        )
        if raw_receipt_id is None:
            continue
        receipt_id = _optional_text(raw_receipt_id)
        if receipt_id is None:
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_RECEIPT,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
            continue
        raw_state = (
            payload.get("reconciliation_state")
            or structured.get("reconciliation_state")
            or portable_structured.get("reconciliation_state")
        )
        state = raw_state if raw_state in {"recorded", "reconciled", "unknown"} else "recorded"
        receipt = RuntimeEvidenceReceipt(
            receipt_id=receipt_id,
            tool_call_id=_optional_text(payload.get("tool_call_id")),
            reconciliation_state=state,
            source_ref=_source_ref(record),
        )
        prior = projected.get(receipt_id)
        if prior is None:
            projected[receipt_id] = receipt
            continue
        if (
            prior.tool_call_id is not None
            and receipt.tool_call_id is not None
            and prior.tool_call_id != receipt.tool_call_id
        ):
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_RECEIPT,
                    session_id=session_id,
                    event_id=record.event.id,
                    sequence=record.sequence,
                )
            )
            continue
        if (
            prior.reconciliation_state == "reconciled"
            and receipt.reconciliation_state != "reconciled"
        ):
            continue
        projected[receipt_id] = RuntimeEvidenceReceipt(
            receipt_id=receipt_id,
            tool_call_id=prior.tool_call_id or receipt.tool_call_id,
            reconciliation_state=receipt.reconciliation_state,
            source_ref=receipt.source_ref,
        )
    return tuple(projected.values())


def _project_attempts(
    session: Session,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
    *,
    pricing: PriceBook | None,
) -> tuple[RuntimeEvidenceAttempt, ...]:
    relevant = {
        EventType.MODEL_STARTED,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_ERROR,
        EventType.MODEL_COMPLETED,
    }
    builders: dict[str, _AttemptRecord] = {}
    operation_by_model_step: dict[
        str,
        tuple[RuntimeEvidenceOperation, _OperationPrecedence],
    ] = {}
    builders_by_model_step: dict[str, list[_AttemptRecord]] = defaultdict(list)
    repair_pending = False
    for record in records:
        event = record.event
        if event.type == EventType.STRUCTURED_OUTPUT_RETRY:
            repair_pending = True
            continue
        if event.type not in relevant:
            if not isinstance(event.type, EventType):
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.UNKNOWN_EVENT_TYPE,
                        session_id=session.id,
                        event_id=event.id,
                        sequence=record.sequence,
                    )
                )
            continue
        event_type = EventType(event.type)
        payload = event.payload
        attempt_id = _optional_text(payload.get("model_attempt_id"))
        model_step_id = _optional_text(payload.get("model_step_id"))
        ordinal = _positive_int(payload.get("attempt"))
        if attempt_id is None:
            if model_step_id is not None and ordinal is not None:
                attempt_id = f"{model_step_id}:{ordinal}"
            else:
                attempt_id = event.id
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.LEGACY_ATTEMPT_IDENTITY,
                        session_id=session.id,
                        event_id=event.id,
                        sequence=record.sequence,
                    )
                )
        operation, operation_rank = _attempt_operation(
            payload=payload,
            session=session,
            repair_pending=repair_pending,
            has_stable_identity=model_step_id is not None,
        )
        if model_step_id is not None:
            prior = operation_by_model_step.get(model_step_id)
            if prior is not None and operation_rank < prior[1]:
                operation, operation_rank = prior
            elif prior is not None and operation_rank == prior[1] and operation is not prior[0]:
                operation = RuntimeEvidenceOperation.UNKNOWN
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.UNKNOWN_OPERATION,
                        session_id=session.id,
                        event_id=event.id,
                        sequence=record.sequence,
                    )
                )
            if prior != (operation, operation_rank):
                operation_by_model_step[model_step_id] = (operation, operation_rank)
                if prior is not None:
                    for existing in builders_by_model_step[model_step_id]:
                        existing.operation = operation
        if operation is RuntimeEvidenceOperation.STRUCTURED_OUTPUT_REPAIR:
            repair_pending = False
        if operation is RuntimeEvidenceOperation.UNKNOWN:
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.UNKNOWN_OPERATION,
                    session_id=session.id,
                    event_id=event.id,
                    sequence=record.sequence,
                )
            )
        builder = builders.get(attempt_id)
        if builder is None:
            builder = _AttemptRecord(
                attempt_id=attempt_id,
                model_step_id=model_step_id,
                operation=operation,
                attempt_ordinal=ordinal or 1,
            )
            builders[attempt_id] = builder
            if model_step_id is not None:
                builders_by_model_step[model_step_id].append(builder)
        else:
            builder.operation = operation
        builder.source_refs.append(_source_ref(record))
        builder.provider_name = builder.provider_name or _optional_text(
            payload.get("provider_name") or payload.get("provider")
        )
        builder.requested_model = builder.requested_model or _optional_text(
            payload.get("requested_model")
        )
        builder.model = builder.model or _optional_text(payload.get("model"))
        builder.status = {
            EventType.MODEL_STARTED: RuntimeEvidenceAttemptStatus.STARTED,
            EventType.MODEL_RETRY: RuntimeEvidenceAttemptStatus.RETRY_SCHEDULED,
            EventType.MODEL_ATTEMPT_DISCARDED: RuntimeEvidenceAttemptStatus.DISCARDED,
            EventType.MODEL_ERROR: RuntimeEvidenceAttemptStatus.FAILED,
            EventType.MODEL_COMPLETED: RuntimeEvidenceAttemptStatus.COMPLETED,
        }[event_type]
        if event_type == EventType.MODEL_COMPLETED:
            builder.completed_at = event.timestamp
            try:
                metrics = usage_metrics_from_event_payload(payload)
            except (TypeError, ValueError):
                metrics = None
                warning_code = RuntimeEvidenceWarningCode.MALFORMED_USAGE
                builder.usage_status = RuntimeEvidenceUsageStatus.MALFORMED
            else:
                warning_code = RuntimeEvidenceWarningCode.MISSING_USAGE
                builder.usage_status = RuntimeEvidenceUsageStatus.MISSING
            if metrics is None:
                builder.usage_warning_recorded = True
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=warning_code,
                        session_id=session.id,
                        event_id=event.id,
                        sequence=record.sequence,
                    )
                )
            else:
                builder.provider_name = metrics.provider_name or builder.provider_name
                builder.requested_model = metrics.requested_model or builder.requested_model
                builder.model = metrics.model or builder.model
                builder.metrics = metrics
                builder.usage = _usage_from_metrics(metrics)
                builder.usage_status = RuntimeEvidenceUsageStatus.REPORTED

    projected: list[RuntimeEvidenceAttempt] = []
    for builder in builders.values():
        if builder.usage is None and not builder.usage_warning_recorded:
            source = builder.source_refs[-1] if builder.source_refs else None
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MISSING_USAGE,
                    session_id=session.id,
                    event_id=None if source is None else source.event_id,
                    sequence=None if source is None else source.sequence,
                )
            )
        cost = _attempt_cost(builder, pricing)
        if cost.status is RuntimeEvidenceCostStatus.UNPRICED:
            source = builder.source_refs[-1] if builder.source_refs else None
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.UNPRICED_USAGE,
                    session_id=session.id,
                    event_id=None if source is None else source.event_id,
                    sequence=None if source is None else source.sequence,
                )
            )
        projected.append(
            RuntimeEvidenceAttempt(
                attempt_id=builder.attempt_id,
                model_step_id=builder.model_step_id,
                operation=builder.operation,
                attempt_ordinal=builder.attempt_ordinal,
                provider_name=builder.provider_name,
                requested_model=builder.requested_model,
                model=builder.model,
                status=builder.status,
                usage_status=builder.usage_status,
                usage=builder.usage,
                cost=cost,
                source_refs=tuple(builder.source_refs),
            )
        )
    return tuple(projected)


def _attempt_operation(
    *,
    payload: dict[str, object],
    session: Session,
    repair_pending: bool,
    has_stable_identity: bool,
) -> tuple[RuntimeEvidenceOperation, _OperationPrecedence]:
    explicit = payload.get("operation")
    if type(explicit) is str:
        try:
            return RuntimeEvidenceOperation(explicit), _OperationPrecedence.EVENT_DECLARATION
        except ValueError:
            return RuntimeEvidenceOperation.UNKNOWN, _OperationPrecedence.EVENT_DECLARATION
    configured = session.labels.get("runtime_evidence_operation")
    if configured is not None:
        try:
            return (
                RuntimeEvidenceOperation(configured),
                _OperationPrecedence.SESSION_DECLARATION,
            )
        except ValueError:
            return RuntimeEvidenceOperation.UNKNOWN, _OperationPrecedence.SESSION_DECLARATION
    if payload.get("purpose") == "context_compaction":
        return RuntimeEvidenceOperation.COMPACTION, _OperationPrecedence.RUNTIME_PROTOCOL
    if repair_pending:
        return (
            RuntimeEvidenceOperation.STRUCTURED_OUTPUT_REPAIR,
            _OperationPrecedence.RUNTIME_PROTOCOL,
        )
    if has_stable_identity:
        return RuntimeEvidenceOperation.AGENT_STEP, _OperationPrecedence.ORDINARY_STEP
    return RuntimeEvidenceOperation.UNKNOWN, _OperationPrecedence.UNKNOWN


def _is_manual_reconciliation(record: EventRecord) -> bool:
    return (
        record.event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
        }
        and record.event.payload.get("manual_recovery") is True
    )


def _source_ref(record: EventRecord) -> RuntimeEvidenceSourceRef:
    return RuntimeEvidenceSourceRef(event_id=record.event.id, sequence=record.sequence)


def _attempt_cost(
    attempt: _AttemptRecord,
    pricing: PriceBook | None,
) -> RuntimeEvidenceCost:
    if pricing is None:
        return RuntimeEvidenceCost(status=RuntimeEvidenceCostStatus.NOT_REQUESTED)
    if attempt.metrics is None or attempt.completed_at is None:
        return RuntimeEvidenceCost(status=RuntimeEvidenceCostStatus.MISSING_USAGE)
    try:
        with localcontext(_EVIDENCE_DECIMAL_CONTEXT):
            estimate = estimate_model_step_cost(
                metrics=attempt.metrics,
                pricing=pricing,
                effective_on=attempt.completed_at.date(),
            )
    except (ArithmeticError, ValueError):
        return RuntimeEvidenceCost(status=RuntimeEvidenceCostStatus.UNPRICED)
    if not estimate.priced:
        return RuntimeEvidenceCost(status=RuntimeEvidenceCostStatus.UNPRICED)
    try:
        return RuntimeEvidenceCost(
            status=RuntimeEvidenceCostStatus.PRICED,
            currency=estimate.currency,
            total_cost=estimate.total_cost,
            pricing_provider_name=estimate.pricing_provider_name,
            pricing_model=estimate.pricing_model,
        )
    except ValueError:
        return RuntimeEvidenceCost(status=RuntimeEvidenceCostStatus.UNPRICED)


def _usage_from_metrics(metrics: UsageMetrics) -> RuntimeEvidenceUsage:
    return RuntimeEvidenceUsage(
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        total_tokens=metrics.total_tokens,
        reasoning_output_tokens=metrics.reasoning_output_tokens,
        cache=RuntimeEvidenceCacheUsage(
            read_tokens=metrics.cache.read_tokens,
            write_tokens=metrics.cache.write_tokens,
            write_5m_tokens=metrics.cache.write_5m_tokens,
            write_1h_tokens=metrics.cache.write_1h_tokens,
            write_unknown_ttl_tokens=metrics.cache.write_unknown_ttl_tokens,
            cached_input_tokens=metrics.cache.cached_input_tokens,
            uncached_input_tokens=metrics.cache.uncached_input_tokens,
        ),
    )


def _empty_usage() -> RuntimeEvidenceUsage:
    return RuntimeEvidenceUsage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        reasoning_output_tokens=0,
        cache=RuntimeEvidenceCacheUsage(
            read_tokens=0,
            write_tokens=0,
            write_5m_tokens=0,
            write_1h_tokens=0,
            write_unknown_ttl_tokens=0,
            cached_input_tokens=0,
            uncached_input_tokens=0,
        ),
    )


def _sum_usage(values: tuple[RuntimeEvidenceUsage, ...]) -> RuntimeEvidenceUsage:
    return RuntimeEvidenceUsage(
        input_tokens=sum(value.input_tokens for value in values),
        output_tokens=sum(value.output_tokens for value in values),
        total_tokens=sum(value.total_tokens for value in values),
        reasoning_output_tokens=sum(value.reasoning_output_tokens for value in values),
        cache=RuntimeEvidenceCacheUsage(
            read_tokens=sum(value.cache.read_tokens for value in values),
            write_tokens=sum(value.cache.write_tokens for value in values),
            write_5m_tokens=sum(value.cache.write_5m_tokens for value in values),
            write_1h_tokens=sum(value.cache.write_1h_tokens for value in values),
            write_unknown_ttl_tokens=sum(value.cache.write_unknown_ttl_tokens for value in values),
            cached_input_tokens=sum(value.cache.cached_input_tokens for value in values),
            uncached_input_tokens=sum(value.cache.uncached_input_tokens for value in values),
        ),
    )


def _totals_from_attempts(
    attempts: tuple[RuntimeEvidenceAttempt, ...],
    *,
    session_count: int,
    tool_call_count: int,
) -> RuntimeEvidenceTotals:
    usages = tuple(attempt.usage for attempt in attempts if attempt.usage is not None)
    by_operation: dict[RuntimeEvidenceOperation, list[RuntimeEvidenceAttempt]] = defaultdict(list)
    for attempt in attempts:
        by_operation[attempt.operation].append(attempt)
    priced_costs: dict[str, Decimal] = defaultdict(Decimal)
    with localcontext(_EVIDENCE_DECIMAL_CONTEXT):
        for attempt in attempts:
            if (
                attempt.cost.status is RuntimeEvidenceCostStatus.PRICED
                and attempt.cost.currency is not None
                and attempt.cost.total_cost is not None
            ):
                priced_costs[attempt.cost.currency] += attempt.cost.total_cost

    def usage_where(
        selected_attempts: tuple[RuntimeEvidenceAttempt, ...],
    ) -> RuntimeEvidenceUsage:
        selected_usages = tuple(
            attempt.usage for attempt in selected_attempts if attempt.usage is not None
        )
        return _sum_usage(selected_usages) if selected_usages else _empty_usage()

    return RuntimeEvidenceTotals(
        session_count=session_count,
        model_step_count=len(
            {attempt.model_step_id for attempt in attempts if attempt.model_step_id is not None}
        ),
        attempt_count=len(attempts),
        first_attempt_count=sum(attempt.attempt_ordinal == 1 for attempt in attempts),
        provider_retry_attempt_count=sum(attempt.attempt_ordinal > 1 for attempt in attempts),
        structured_output_repair_attempt_count=sum(
            attempt.operation is RuntimeEvidenceOperation.STRUCTURED_OUTPUT_REPAIR
            for attempt in attempts
        ),
        tool_call_count=tool_call_count,
        missing_usage_attempt_count=sum(attempt.usage is None for attempt in attempts),
        usage=_sum_usage(usages) if usages else _empty_usage(),
        first_attempt_usage=usage_where(
            tuple(attempt for attempt in attempts if attempt.attempt_ordinal == 1)
        ),
        provider_retry_usage=usage_where(
            tuple(attempt for attempt in attempts if attempt.attempt_ordinal > 1)
        ),
        structured_output_repair_usage=usage_where(
            tuple(
                attempt
                for attempt in attempts
                if attempt.operation is RuntimeEvidenceOperation.STRUCTURED_OUTPUT_REPAIR
            )
        ),
        compaction_usage=usage_where(
            tuple(
                attempt
                for attempt in attempts
                if attempt.operation is RuntimeEvidenceOperation.COMPACTION
            )
        ),
        evaluation_usage=usage_where(
            tuple(
                attempt
                for attempt in attempts
                if attempt.operation is RuntimeEvidenceOperation.EVALUATION
            )
        ),
        repair_usage=usage_where(
            tuple(
                attempt
                for attempt in attempts
                if attempt.operation is RuntimeEvidenceOperation.REPAIR
            )
        ),
        comparison_control_usage=usage_where(
            tuple(
                attempt
                for attempt in attempts
                if attempt.operation is RuntimeEvidenceOperation.COMPARISON_CONTROL
            )
        ),
        operations=tuple(
            RuntimeEvidenceOperationTotals(
                operation=operation,
                attempt_count=len(by_operation[operation]),
                usage=_sum_usage(
                    tuple(
                        attempt.usage
                        for attempt in by_operation[operation]
                        if attempt.usage is not None
                    )
                )
                if any(attempt.usage is not None for attempt in by_operation[operation])
                else _empty_usage(),
            )
            for operation in sorted(by_operation, key=lambda value: value.value)
        ),
        priced_costs=tuple(
            RuntimeEvidenceCurrencyCost(currency=currency, total_cost=priced_costs[currency])
            for currency in sorted(priced_costs)
        ),
        priced_attempt_count=sum(
            attempt.cost.status is RuntimeEvidenceCostStatus.PRICED for attempt in attempts
        ),
        unpriced_attempt_count=sum(
            attempt.cost.status is RuntimeEvidenceCostStatus.UNPRICED for attempt in attempts
        ),
    )


def _totals(sessions: tuple[RuntimeEvidenceSession, ...]) -> RuntimeEvidenceTotals:
    attempts = tuple(attempt for session in sessions for attempt in session.attempts)
    return _totals_from_attempts(
        attempts,
        session_count=len(sessions),
        tool_call_count=sum(session.totals.tool_call_count for session in sessions),
    )


def _branch_totals(
    root_session_id: str,
    sessions: tuple[RuntimeEvidenceSession, ...],
) -> tuple[RuntimeEvidenceBranchTotals, ...]:
    by_id = {session.session_id: session for session in sessions}
    children: dict[str, list[str]] = defaultdict(list)
    for session in sessions:
        if session.parent_session_id in by_id:
            assert session.parent_session_id is not None
            children[session.parent_session_id].append(session.session_id)

    def subtree(root_id: str) -> tuple[str, ...]:
        values: list[str] = []
        queue = [root_id]
        while queue:
            session_id = queue.pop(0)
            values.append(session_id)
            queue.extend(children.get(session_id, ()))
        return tuple(values)

    branches: list[RuntimeEvidenceBranchTotals] = []
    for branch_root_id in children.get(root_session_id, ()):
        session_ids = subtree(branch_root_id)
        branches.append(
            RuntimeEvidenceBranchTotals(
                branch_root_session_id=branch_root_id,
                session_ids=session_ids,
                totals=_totals(tuple(by_id[session_id] for session_id in session_ids)),
            )
        )
    return tuple(branches)


def _optional_text(value: object) -> str | None:
    if type(value) is not str or not value.strip() or len(value) > _MAX_IDENTITY_CHARS:
        return None
    try:
        return require_clean_nonblank(value, "evidence identity")
    except ValueError:
        return None


def _same_session_identity(left: Session, right: Session) -> bool:
    return (
        left.id == right.id
        and left.agent_name == right.agent_name
        and left.provider_name == right.provider_name
        and left.model == right.model
        and left.parent_session_id == right.parent_session_id
        and left.causal_budget_id == right.causal_budget_id
        and left.created_at == right.created_at
    )


def _positive_int(value: object) -> int | None:
    if type(value) is not int or value < 1 or value > MAX_DURABLE_JSON_INTEGER:
        return None
    return value


def _nonnegative_int(value: object) -> int | None:
    if type(value) is not int or value < 0 or value > MAX_DURABLE_JSON_INTEGER:
        return None
    return value


def _safe_taint_labels(value: object) -> tuple[str, ...] | None:
    if value is None:
        return ()
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return None
    items = tuple(value)
    if len(items) > _MAX_TAINT_LABELS:
        return None
    labels: list[str] = []
    for item in items:
        if type(item) is not str or len(item) > _MAX_TAINT_LABEL_CHARS:
            return None
        try:
            label = require_clean_nonblank(item, "taint label")
        except ValueError:
            return None
        labels.append(label)
    return tuple(sorted(set(labels)))


def _bounded_decimal(
    value: Decimal | None,
    field_name: str,
    *,
    max_digits: int,
) -> Decimal | None:
    if value is None:
        return None
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be finite and nonnegative.")
    if len(value.as_tuple().digits) > max_digits:
        raise ValueError(f"{field_name} exceeds its decimal digit limit.")
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int) or exponent < -_MAX_COST_DECIMAL_PLACES:
        raise ValueError(f"{field_name} exceeds its decimal scale limit.")
    return value
