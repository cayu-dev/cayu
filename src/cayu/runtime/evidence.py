"""Bounded, serializable runtime evidence reconstructed from durable stores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import IntEnum, StrEnum
from heapq import heappop, heappush
from typing import Literal, cast

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
from cayu.runtime.workspace_observation_recovery import (
    WORKSPACE_OBSERVATION_TERMINAL_CONTROLS,
    workspace_observation_terminal_from_delta_status,
)

RUNTIME_EVIDENCE_SCHEMA_VERSION = 3

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

_WORKSPACE_REVISION_DETAIL_CODES = frozenset(
    {
        "final_revision_observer_failed",
        "final_revision_observer_capacity_exhausted",
        "final_revision_observer_limit_exceeded",
        "final_revision_secret_scope_unavailable",
        "final_revision_observer_timeout",
        "file_byte_limit_exceeded",
        "manifest_artifact_reference_invalid",
        "manifest_artifact_store_inside_workspace",
        "manifest_artifact_store_unavailable",
        "manifest_artifact_write_failed",
        "manifest_artifact_write_unsettled",
        "manifest_byte_limit_exceeded",
        "manifest_redaction_failed",
        "path_byte_limit_exceeded",
        "path_count_limit_exceeded",
        "revision_observation_unsupported",
        "revision_observer_capacity_exhausted",
        "revision_observer_failed",
        "revision_observer_limit_exceeded",
        "revision_observer_timeout",
        "total_file_byte_limit_exceeded",
        "unsafe_workspace_path",
        "workspace_file_read_failed",
        "workspace_evidence_quarantined",
        "workspace_list_failed",
    }
)
_WORKSPACE_DELTA_DETAIL_CODES = _WORKSPACE_REVISION_DETAIL_CODES | frozenset(
    {
        "finalization_baseline_evidence_incomplete",
        "finalization_baseline_unavailable",
        "finalization_delta_secret_scope_unavailable",
        "manifest_artifact_reference_invalid",
        "manifest_artifact_store_inside_workspace",
        "manifest_artifact_store_unavailable",
        "manifest_artifact_write_unsettled",
        "manifest_redaction_failed",
        "observation_not_complete",
        "observation_path_scope_mismatch",
        "publication_storage_exhausted",
        "revision_comparison_failed",
        "workspace_evidence_quarantined",
    }
)
_WORKSPACE_ATTRIBUTION_DETAIL_CODES = frozenset(
    {
        "direct_and_observed_workspace_evidence_conflict",
        "exclusive_writer_isolation_unproven",
        "exclusive_writer_isolation_verified",
        "overlapping_workspace_mutation_windows",
        "workspace_attribution_recovery_incomplete",
        "workspace_evidence_quarantined",
        "workspace_resource_identity_unavailable",
    }
)
_WORKSPACE_TERMINAL_DETAIL_CODES = frozenset(
    {
        "durable_tool_outcome_evidence_missing",
        "mutation_settlement_unproven",
        "receipt_publication_failed",
        "receipt_publication_interrupted",
        "referenced_workspace_artifact_missing",
        "worker_lost_before_tool_outcome_was_durable",
        "worker_lost_before_workspace_observation_completed",
        "workspace_artifact_verification_failed",
        "workspace_delta_evidence_conflict",
        "workspace_delta_evidence_missing",
        "workspace_revision_comparison_failed",
        "workspace_revision_evidence_incomplete",
    }
)


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
    MALFORMED_EXECUTION_PROFILE = "malformed_execution_profile"
    MALFORMED_WORKSPACE_EVIDENCE = "malformed_workspace_evidence"
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
    """Bounded request for a durable runtime-evidence projection."""

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
    execution_profile_fingerprint: str | None = Field(
        default=None,
        max_length=64,
        exclude_if=lambda value: value is None,
    )
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

    @field_validator("execution_profile_fingerprint")
    @classmethod
    def validate_execution_profile_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("execution_profile_fingerprint must be a lowercase SHA-256 digest.")
        return value


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


class RuntimeEvidenceWorkspaceArtifact(BaseModel):
    """Content-free workspace evidence artifact reference."""

    model_config = _MODEL_CONFIG

    kind: Literal["revision-before", "revision-after", "revision-delta"]
    artifact_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: StrictInt = Field(ge=1, le=16 * 1024 * 1024)
    state: Literal["intent", "published", "referenced", "failed", "orphaned", "missing"]

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "artifact_id")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest.")
        return value


class RuntimeEvidenceWorkspaceRevision(BaseModel):
    """Path-free summary of one workspace revision observation."""

    model_config = _MODEL_CONFIG

    phase: Literal["before", "after"]
    status: Literal["supported", "unsupported", "failed", "incomplete", "truncated"]
    revision: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    path_scope: Literal["complete", "changed"]
    total_paths: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    detail_code: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    source_ref: RuntimeEvidenceSourceRef

    @field_validator("revision")
    @classmethod
    def validate_optional_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "revision")

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str | None) -> str | None:
        if value is not None and value not in _WORKSPACE_REVISION_DETAIL_CODES:
            raise ValueError("detail_code is not a fixed workspace revision value.")
        return value

    @model_validator(mode="after")
    def validate_revision_shape(self) -> RuntimeEvidenceWorkspaceRevision:
        if (self.status == "supported") != (self.revision is not None):
            raise ValueError("Only supported workspace observations may carry a revision.")
        return self


class RuntimeEvidenceWorkspaceDelta(BaseModel):
    """Path-free summary of one workspace revision delta."""

    model_config = _MODEL_CONFIG

    status: Literal["changed", "no_change", "unsupported", "failed", "incomplete", "truncated"]
    before_revision: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    after_revision: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    total_paths: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    head_changed: StrictBool
    branch_changed: StrictBool
    detail_code: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    attribution_confidence: (
        Literal[
            "exclusive_tool",
            "concurrent_ambiguity",
            "external_or_unknown",
            "unattributed_finalization_change",
        ]
        | None
    ) = None
    source_ref: RuntimeEvidenceSourceRef

    @field_validator("before_revision", "after_revision")
    @classmethod
    def validate_optional_revision(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str | None) -> str | None:
        if value is not None and value not in _WORKSPACE_DELTA_DETAIL_CODES:
            raise ValueError("detail_code is not a fixed workspace delta value.")
        return value

    @model_validator(mode="after")
    def validate_delta_shape(self) -> RuntimeEvidenceWorkspaceDelta:
        complete = self.status in {"changed", "no_change"}
        has_revisions = self.before_revision is not None and self.after_revision is not None
        if complete and not has_revisions:
            raise ValueError("Complete workspace deltas require both revisions.")
        if self.status == "no_change" and (
            self.before_revision != self.after_revision
            or self.total_paths != 0
            or self.head_changed
            or self.branch_changed
        ):
            raise ValueError("A no-change workspace delta cannot contain change evidence.")
        return self


class RuntimeEvidenceWorkspaceAttribution(BaseModel):
    """Safe classification of responsibility for one workspace mutation."""

    model_config = _MODEL_CONFIG

    confidence: Literal[
        "exclusive_tool",
        "concurrent_ambiguity",
        "external_or_unknown",
        "unattributed_finalization_change",
    ]
    writer_isolation: Literal["exclusive", "shared", "unknown"]
    overlap_detected: StrictBool
    direct_reconciliation: Literal[
        "not_observed",
        "consistent",
        "incomplete",
        "contradictory",
        "truncated",
    ]
    detail_code: str = Field(max_length=_MAX_IDENTITY_CHARS)

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str) -> str:
        value = require_clean_nonblank(value, "detail_code")
        if value not in _WORKSPACE_ATTRIBUTION_DETAIL_CODES:
            raise ValueError("detail_code is not a fixed workspace attribution value.")
        return value

    @model_validator(mode="after")
    def validate_attribution_shape(self) -> RuntimeEvidenceWorkspaceAttribution:
        if self.confidence == "exclusive_tool" and (
            self.writer_isolation != "exclusive"
            or self.overlap_detected
            or self.direct_reconciliation == "contradictory"
        ):
            raise ValueError("Exclusive workspace attribution requires exclusive evidence.")
        if (
            self.confidence == "unattributed_finalization_change"
            and self.writer_isolation == "exclusive"
        ):
            raise ValueError("Finalization-only change cannot claim exclusive isolation.")
        return self


class RuntimeEvidenceWorkspaceTerminal(BaseModel):
    """Final durable state of one workspace-observation window."""

    model_config = _MODEL_CONFIG

    status: Literal["complete", "incomplete", "ambiguous", "failed"]
    detail_code: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    session_run_epoch: StrictInt | None = Field(default=None, ge=1)
    recovery_run_epoch: StrictInt | None = Field(default=None, ge=1)
    binding_generation_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    source_ref: RuntimeEvidenceSourceRef

    @field_validator("binding_generation_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "binding_generation_id")

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str | None) -> str | None:
        if value is not None and value not in _WORKSPACE_TERMINAL_DETAIL_CODES:
            raise ValueError("detail_code is not a fixed workspace terminal value.")
        return value

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> RuntimeEvidenceWorkspaceTerminal:
        if (self.status, self.detail_code) not in WORKSPACE_OBSERVATION_TERMINAL_CONTROLS:
            raise ValueError("Workspace terminal status and detail code are inconsistent.")
        if (self.session_run_epoch is None) != (self.recovery_run_epoch is None):
            raise ValueError("Workspace terminal run epochs must be retained together.")
        if (
            self.session_run_epoch is not None
            and self.recovery_run_epoch is not None
            and self.recovery_run_epoch < self.session_run_epoch
        ):
            raise ValueError("Workspace terminal recovery cannot precede its source run.")
        return self


class RuntimeEvidenceWorkspaceMutation(BaseModel):
    """Correlated, content-free evidence for one workspace mutation window."""

    model_config = _MODEL_CONFIG

    window_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    workspace_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    tool_call_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    tool_round_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    before: RuntimeEvidenceWorkspaceRevision | None = None
    after: RuntimeEvidenceWorkspaceRevision | None = None
    delta: RuntimeEvidenceWorkspaceDelta | None = None
    attribution: RuntimeEvidenceWorkspaceAttribution | None = None
    terminal: RuntimeEvidenceWorkspaceTerminal | None = None
    artifacts: tuple[RuntimeEvidenceWorkspaceArtifact, ...] = ()
    source_refs: tuple[RuntimeEvidenceSourceRef, ...] = ()

    @field_validator("window_id", "workspace_id", "tool_call_id", "tool_round_id")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_correlated_evidence(self) -> RuntimeEvidenceWorkspaceMutation:
        if self.delta is not None:
            if self.before is not None and self.before.revision != self.delta.before_revision:
                raise ValueError("Workspace before revision contradicts the retained delta.")
            if self.after is not None and self.after.revision != self.delta.after_revision:
                raise ValueError("Workspace after revision contradicts the retained delta.")
        if (
            self.terminal is not None
            and self.terminal.status == "complete"
            and (self.delta is None or self.delta.status not in {"changed", "no_change"})
        ):
            raise ValueError("Complete workspace terminal evidence requires a complete delta.")
        if (
            self.delta is not None
            and self.terminal is not None
            and self.terminal.session_run_epoch is not None
            and self.terminal.recovery_run_epoch == self.terminal.session_run_epoch
        ):
            expected_status, expected_detail = workspace_observation_terminal_from_delta_status(
                self.delta.status,
                detail_code=self.delta.detail_code,
            )
            if (self.terminal.status, self.terminal.detail_code) != (
                expected_status.value,
                expected_detail,
            ):
                raise ValueError("Workspace terminal evidence contradicts its direct-run delta.")
        return self


class RuntimeEvidenceWorkspaceFinalization(BaseModel):
    """Last safely observable workspace state at environment finalization."""

    model_config = _MODEL_CONFIG

    workspace_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    binding_generation_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    status: Literal["supported", "unsupported", "failed", "incomplete", "truncated"]
    revision: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    path_scope: Literal["complete", "changed"]
    total_paths: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    detail_code: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    delta: RuntimeEvidenceWorkspaceDelta | None = None
    source_refs: tuple[RuntimeEvidenceSourceRef, ...] = ()

    @field_validator("workspace_id", "binding_generation_id")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_optional_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "revision")

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str | None) -> str | None:
        if value is not None and value not in _WORKSPACE_REVISION_DETAIL_CODES:
            raise ValueError("detail_code is not a fixed workspace finalization value.")
        return value

    @model_validator(mode="after")
    def validate_final_revision_shape(self) -> RuntimeEvidenceWorkspaceFinalization:
        if (self.status == "supported") != (self.revision is not None):
            raise ValueError("Only supported workspace finalization may carry a revision.")
        if self.delta is not None:
            if self.delta.after_revision != self.revision:
                raise ValueError("Finalization delta must end at the retained final revision.")
            if self.delta.attribution_confidence != "unattributed_finalization_change":
                raise ValueError("Finalization delta cannot claim tool attribution.")
        return self


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
    workspace_mutations: tuple[RuntimeEvidenceWorkspaceMutation, ...] = ()
    workspace_finalization: RuntimeEvidenceWorkspaceFinalization | None = None
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

    schema_version: Literal[3] = RUNTIME_EVIDENCE_SCHEMA_VERSION
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
    execution_profile_fingerprint: str | None = None
    execution_profile_conflict: bool = False
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


@dataclass(slots=True)
class _WorkspaceMutationRecord:
    window_id: str
    workspace_id: str
    tool_call_id: str
    tool_round_id: str | None
    before: RuntimeEvidenceWorkspaceRevision | None = None
    after: RuntimeEvidenceWorkspaceRevision | None = None
    delta: RuntimeEvidenceWorkspaceDelta | None = None
    attribution: RuntimeEvidenceWorkspaceAttribution | None = None
    terminal: RuntimeEvidenceWorkspaceTerminal | None = None
    artifacts: dict[str, RuntimeEvidenceWorkspaceArtifact] = field(default_factory=dict)
    terminal_artifact_kinds: set[str] = field(default_factory=set)
    source_refs: list[RuntimeEvidenceSourceRef] = field(default_factory=list)
    last_record: EventRecord | None = None
    conflicted: bool = False


@dataclass(slots=True)
class _WorkspaceFinalizationRecord:
    candidate: RuntimeEvidenceWorkspaceFinalization
    generation_sequence: int
    conflicted: bool = False


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
    workspace_mutations = _project_workspace_mutations(
        session.id,
        records,
        capture.warnings,
    )
    workspace_finalization = _project_workspace_finalization(
        session.id,
        records,
        capture.warnings,
    )
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
        workspace_mutations=workspace_mutations,
        workspace_finalization=workspace_finalization,
        totals=totals,
    )


def _project_workspace_mutations(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> tuple[RuntimeEvidenceWorkspaceMutation, ...]:
    allowed_types = {
        EventType.WORKSPACE_REVISION_OBSERVED,
        EventType.WORKSPACE_MUTATION_RECORDED,
        EventType.WORKSPACE_OBSERVATION_FINALIZED,
    }
    by_window: dict[str, _WorkspaceMutationRecord] = {}
    order: list[str] = []
    for record in records:
        if record.event.type not in allowed_types:
            continue
        payload = record.event.payload
        try:
            window_id = _workspace_required_text(payload, "window_id")
            workspace_id = _workspace_required_text(payload, "workspace_id")
            tool_call_id = _workspace_required_text(payload, "tool_call_id")
            tool_round_id = _workspace_optional_text(payload, "tool_round_id")
        except (TypeError, ValueError):
            _warn_malformed_workspace(session_id, record, warnings)
            continue

        mutation = by_window.get(window_id)
        if mutation is None:
            mutation = _WorkspaceMutationRecord(
                window_id=window_id,
                workspace_id=workspace_id,
                tool_call_id=tool_call_id,
                tool_round_id=tool_round_id,
            )
            by_window[window_id] = mutation
            order.append(window_id)
        elif (
            mutation.workspace_id != workspace_id
            or mutation.tool_call_id != tool_call_id
            or (
                mutation.tool_round_id is not None
                and tool_round_id is not None
                and mutation.tool_round_id != tool_round_id
            )
        ):
            mutation.conflicted = True
            _warn_malformed_workspace(session_id, record, warnings)
            continue
        elif mutation.tool_round_id is None:
            mutation.tool_round_id = tool_round_id

        source_ref = _source_ref(record)
        try:
            artifacts: tuple[RuntimeEvidenceWorkspaceArtifact, ...]
            if record.event.type is EventType.WORKSPACE_REVISION_OBSERVED:
                revision = _workspace_revision_from_record(record)
                phase = revision.phase
                previous = mutation.before if phase == "before" else mutation.after
                if previous is not None and not _same_workspace_fact(previous, revision):
                    raise ValueError("Workspace revision phase conflicts with prior evidence.")
                if previous is None:
                    if phase == "before":
                        mutation.before = revision
                    else:
                        mutation.after = revision
                artifacts = _workspace_manifest_artifacts(
                    payload,
                    kind=f"revision-{phase}",
                )
            elif record.event.type is EventType.WORKSPACE_MUTATION_RECORDED:
                delta = _workspace_delta_from_payload(payload, source_ref=source_ref)
                if mutation.delta is not None and not _same_workspace_fact(mutation.delta, delta):
                    raise ValueError("Workspace delta conflicts with prior evidence.")
                if mutation.delta is None:
                    mutation.delta = delta
                attribution = _workspace_attribution_from_payload(payload)
                if attribution is not None:
                    if mutation.attribution is not None and mutation.attribution != attribution:
                        raise ValueError("Workspace attribution conflicts with prior evidence.")
                    mutation.attribution = attribution
                artifacts = _workspace_manifest_artifacts(
                    payload,
                    kind="revision-delta",
                )
            else:
                terminal = _workspace_terminal_from_record(record)
                if mutation.terminal is not None and not _same_workspace_fact(
                    mutation.terminal, terminal
                ):
                    raise ValueError("Workspace terminal state conflicts with prior evidence.")
                if mutation.terminal is None:
                    mutation.terminal = terminal
                attribution = _workspace_attribution_from_payload(payload)
                if attribution is not None:
                    if mutation.attribution is not None and mutation.attribution != attribution:
                        raise ValueError("Workspace attribution conflicts with prior evidence.")
                    mutation.attribution = attribution
                artifacts = _workspace_terminal_artifacts(payload)
            for artifact in artifacts:
                previous_artifact = mutation.artifacts.get(artifact.kind)
                terminal_artifact = record.event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
                if previous_artifact is not None:
                    if not _same_workspace_fact(
                        previous_artifact,
                        artifact,
                        excluded={"state"},
                    ):
                        raise ValueError("Workspace artifact conflicts with prior evidence.")
                    if (
                        terminal_artifact
                        and artifact.kind in mutation.terminal_artifact_kinds
                        and previous_artifact != artifact
                    ):
                        raise ValueError("Workspace terminal artifact state conflicts.")
                mutation.artifacts[artifact.kind] = artifact
                if terminal_artifact:
                    mutation.terminal_artifact_kinds.add(artifact.kind)
        except (TypeError, ValueError):
            mutation.conflicted = True
            _warn_malformed_workspace(session_id, record, warnings)
            continue
        mutation.source_refs.append(source_ref)
        mutation.last_record = record

    projected: list[RuntimeEvidenceWorkspaceMutation] = []
    for window_id in order:
        mutation = by_window[window_id]
        if mutation.conflicted or not mutation.source_refs:
            continue
        try:
            candidate = RuntimeEvidenceWorkspaceMutation(
                window_id=mutation.window_id,
                workspace_id=mutation.workspace_id,
                tool_call_id=mutation.tool_call_id,
                tool_round_id=mutation.tool_round_id,
                before=mutation.before,
                after=mutation.after,
                delta=mutation.delta,
                attribution=mutation.attribution,
                terminal=mutation.terminal,
                artifacts=tuple(
                    mutation.artifacts[kind]
                    for kind in ("revision-before", "revision-after", "revision-delta")
                    if kind in mutation.artifacts
                ),
                source_refs=tuple(mutation.source_refs),
            )
        except (TypeError, ValueError):
            if mutation.last_record is not None:
                _warn_malformed_workspace(session_id, mutation.last_record, warnings)
            continue
        projected.append(candidate)
    return tuple(projected)


def _project_workspace_finalization(
    session_id: str,
    records: tuple[EventRecord, ...],
    warnings: list[RuntimeEvidenceWarning],
) -> RuntimeEvidenceWorkspaceFinalization | None:
    finalization_types = {
        EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }
    by_generation: dict[str | None, _WorkspaceFinalizationRecord] = {}
    generation_sequences: dict[str, int] = {}
    current_generation_id: str | None = None
    current_generation_sequence = 0
    for record in records:
        payload = record.event.payload
        record_generation_id: str | None = None
        if record.event.type in {
            EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        }:
            try:
                generation_id = _workspace_optional_text(payload, "binding_generation_id")
            except (TypeError, ValueError):
                if payload.get("final_revision") is not None:
                    _warn_malformed_workspace(session_id, record, warnings)
                continue
            if generation_id is not None:
                record_generation_id = generation_id
                generation_sequence = generation_sequences.setdefault(
                    generation_id,
                    record.sequence,
                )
                if generation_sequence > current_generation_sequence:
                    current_generation_id = generation_id
                    current_generation_sequence = generation_sequence
        if record.event.type not in finalization_types:
            continue
        raw_final = payload.get("final_revision")
        if raw_final is None:
            continue
        try:
            if type(raw_final) is not dict:
                raise TypeError("final_revision must be an object.")
            final_payload = cast("dict[str, object]", raw_final)
            source_ref = _source_ref(record)
            status = _workspace_enum(
                final_payload,
                "status",
                {"supported", "unsupported", "failed", "incomplete", "truncated"},
            )
            raw_delta = final_payload.get("finalization_delta")
            if raw_delta is not None and type(raw_delta) is not dict:
                raise TypeError("finalization_delta must be an object.")
            delta = (
                None
                if raw_delta is None
                else _workspace_delta_from_payload(
                    cast("dict[str, object]", raw_delta),
                    source_ref=source_ref,
                    include_attribution_confidence=True,
                )
            )
            candidate = RuntimeEvidenceWorkspaceFinalization.model_validate(
                {
                    "workspace_id": _workspace_required_text(final_payload, "workspace_id"),
                    "binding_generation_id": record_generation_id,
                    "status": status,
                    "revision": _workspace_optional_text(final_payload, "revision"),
                    "path_scope": _workspace_enum(
                        final_payload,
                        "path_scope",
                        {"complete", "changed"},
                    ),
                    "total_paths": _workspace_required_nonnegative_int(
                        final_payload,
                        "total_paths",
                    ),
                    "detail_code": _workspace_optional_detail_code(
                        final_payload,
                        _WORKSPACE_REVISION_DETAIL_CODES,
                    ),
                    "delta": delta,
                    "source_refs": (source_ref,),
                }
            )
        except (TypeError, ValueError):
            _warn_malformed_workspace(session_id, record, warnings)
            continue
        effective_generation_id = record_generation_id
        if effective_generation_id is None:
            matching_generations = tuple(
                (
                    generation_id,
                    generation,
                )
                for generation_id, generation in by_generation.items()
                if generation_id is not None
                and _same_workspace_fact(
                    generation.candidate,
                    candidate,
                    excluded={
                        "binding_generation_id",
                        "source_ref",
                        "source_refs",
                    },
                )
            )
            if matching_generations:
                effective_generation_id = max(
                    matching_generations,
                    key=lambda item: item[1].generation_sequence,
                )[0]
            else:
                effective_generation_id = current_generation_id
            if effective_generation_id is not None:
                candidate = candidate.model_copy(
                    update={"binding_generation_id": effective_generation_id}
                )
        generation = by_generation.get(effective_generation_id)
        if generation is None:
            by_generation[effective_generation_id] = _WorkspaceFinalizationRecord(
                candidate=candidate,
                generation_sequence=(
                    record.sequence
                    if effective_generation_id is None
                    else generation_sequences.get(effective_generation_id, record.sequence)
                ),
            )
        elif not _same_workspace_fact(
            generation.candidate,
            candidate,
            excluded={"source_ref", "source_refs"},
        ):
            generation.conflicted = True
            _warn_malformed_workspace(session_id, record, warnings)
        else:
            generation.candidate = generation.candidate.model_copy(
                update={"source_refs": (*generation.candidate.source_refs, source_ref)}
            )
    valid = tuple(record for record in by_generation.values() if not record.conflicted)
    if not valid:
        return None
    return max(valid, key=lambda record: record.generation_sequence).candidate


def _workspace_revision_from_record(
    record: EventRecord,
) -> RuntimeEvidenceWorkspaceRevision:
    payload = record.event.payload
    return RuntimeEvidenceWorkspaceRevision.model_validate(
        {
            "phase": _workspace_enum(payload, "phase", {"before", "after"}),
            "status": _workspace_enum(
                payload,
                "status",
                {"supported", "unsupported", "failed", "incomplete", "truncated"},
            ),
            "revision": _workspace_optional_text(payload, "revision"),
            "path_scope": _workspace_enum(payload, "path_scope", {"complete", "changed"}),
            "total_paths": _workspace_required_nonnegative_int(payload, "total_paths"),
            "detail_code": _workspace_optional_detail_code(
                payload,
                _WORKSPACE_REVISION_DETAIL_CODES,
            ),
            "source_ref": _source_ref(record),
        }
    )


def _workspace_delta_from_payload(
    payload: dict[str, object],
    *,
    source_ref: RuntimeEvidenceSourceRef,
    include_attribution_confidence: bool = False,
) -> RuntimeEvidenceWorkspaceDelta:
    confidence = None
    if include_attribution_confidence:
        confidence = _workspace_enum(
            payload,
            "attribution_confidence",
            {
                "exclusive_tool",
                "concurrent_ambiguity",
                "external_or_unknown",
                "unattributed_finalization_change",
            },
        )
    return RuntimeEvidenceWorkspaceDelta.model_validate(
        {
            "status": _workspace_enum(
                payload,
                "status",
                {"changed", "no_change", "unsupported", "failed", "incomplete", "truncated"},
            ),
            "before_revision": _workspace_optional_text(payload, "before_revision"),
            "after_revision": _workspace_optional_text(payload, "after_revision"),
            "total_paths": _workspace_required_nonnegative_int(payload, "total_paths"),
            "head_changed": _workspace_required_bool(payload, "head_changed"),
            "branch_changed": _workspace_required_bool(payload, "branch_changed"),
            "detail_code": _workspace_optional_detail_code(
                payload,
                _WORKSPACE_DELTA_DETAIL_CODES,
            ),
            "attribution_confidence": confidence,
            "source_ref": source_ref,
        }
    )


def _workspace_attribution_from_payload(
    payload: dict[str, object],
) -> RuntimeEvidenceWorkspaceAttribution | None:
    raw = payload.get("attribution")
    if raw is None:
        return None
    if type(raw) is not dict:
        raise TypeError("attribution must be an object.")
    attribution = cast("dict[str, object]", raw)
    return RuntimeEvidenceWorkspaceAttribution.model_validate(
        {
            "confidence": _workspace_enum(
                attribution,
                "confidence",
                {
                    "exclusive_tool",
                    "concurrent_ambiguity",
                    "external_or_unknown",
                    "unattributed_finalization_change",
                },
            ),
            "writer_isolation": _workspace_enum(
                attribution,
                "writer_isolation",
                {"exclusive", "shared", "unknown"},
            ),
            "overlap_detected": _workspace_required_bool(
                attribution,
                "overlap_detected",
            ),
            "direct_reconciliation": _workspace_enum(
                attribution,
                "direct_reconciliation",
                {"not_observed", "consistent", "incomplete", "contradictory", "truncated"},
            ),
            "detail_code": _workspace_required_detail_code(
                attribution,
                _WORKSPACE_ATTRIBUTION_DETAIL_CODES,
            ),
        }
    )


def _workspace_terminal_from_record(
    record: EventRecord,
) -> RuntimeEvidenceWorkspaceTerminal:
    payload = record.event.payload
    return RuntimeEvidenceWorkspaceTerminal.model_validate(
        {
            "status": _workspace_enum(
                payload,
                "status",
                {"complete", "incomplete", "ambiguous", "failed"},
            ),
            "detail_code": _workspace_optional_detail_code(
                payload,
                _WORKSPACE_TERMINAL_DETAIL_CODES,
            ),
            "session_run_epoch": _workspace_optional_positive_int(
                payload,
                "session_run_epoch",
            ),
            "recovery_run_epoch": _workspace_optional_positive_int(
                payload,
                "recovery_run_epoch",
            ),
            "binding_generation_id": _workspace_optional_text(
                payload,
                "binding_generation_id",
            ),
            "source_ref": _source_ref(record),
        }
    )


def _workspace_manifest_artifacts(
    payload: dict[str, object],
    *,
    kind: str,
) -> tuple[RuntimeEvidenceWorkspaceArtifact, ...]:
    field_names = {
        "artifact_id": "manifest_artifact_id",
        "sha256": "manifest_artifact_sha256",
        "size_bytes": "manifest_artifact_size_bytes",
    }
    if not any(payload.get(field_name) is not None for field_name in field_names.values()):
        return ()
    return (
        RuntimeEvidenceWorkspaceArtifact.model_validate(
            {
                "kind": kind,
                "artifact_id": _workspace_required_text(payload, field_names["artifact_id"]),
                "sha256": _workspace_required_text(payload, field_names["sha256"]),
                "size_bytes": _workspace_required_positive_int(
                    payload,
                    field_names["size_bytes"],
                ),
                "state": "referenced",
            }
        ),
    )


def _workspace_terminal_artifacts(
    payload: dict[str, object],
) -> tuple[RuntimeEvidenceWorkspaceArtifact, ...]:
    projected: list[RuntimeEvidenceWorkspaceArtifact] = []
    for field_prefix, kind in (
        ("revision_before", "revision-before"),
        ("revision_after", "revision-after"),
        ("revision_delta", "revision-delta"),
    ):
        names = {
            "artifact_id": f"{field_prefix}_artifact_id",
            "sha256": f"{field_prefix}_artifact_sha256",
            "size_bytes": f"{field_prefix}_artifact_size_bytes",
            "state": f"{field_prefix}_artifact_state",
        }
        if not any(payload.get(field_name) is not None for field_name in names.values()):
            continue
        projected.append(
            RuntimeEvidenceWorkspaceArtifact.model_validate(
                {
                    "kind": kind,
                    "artifact_id": _workspace_required_text(payload, names["artifact_id"]),
                    "sha256": _workspace_required_text(payload, names["sha256"]),
                    "size_bytes": _workspace_required_positive_int(
                        payload,
                        names["size_bytes"],
                    ),
                    "state": _workspace_enum(
                        payload,
                        names["state"],
                        {"intent", "published", "referenced", "failed", "orphaned", "missing"},
                    ),
                }
            )
        )
    return tuple(projected)


def _workspace_required_text(payload: Mapping[str, object], field_name: str) -> str:
    value = _optional_text(payload.get(field_name))
    if value is None:
        raise ValueError(f"{field_name} must be bounded nonblank text.")
    return value


def _workspace_optional_text(payload: Mapping[str, object], field_name: str) -> str | None:
    raw = payload.get(field_name)
    if raw is None:
        return None
    value = _optional_text(raw)
    if value is None:
        raise ValueError(f"{field_name} must be bounded nonblank text or null.")
    return value


def _workspace_optional_detail_code(
    payload: Mapping[str, object],
    allowed: frozenset[str],
) -> str | None:
    raw = payload.get("detail_code")
    if raw is None:
        return None
    if type(raw) is not str or raw not in allowed:
        raise ValueError("detail_code is not a fixed workspace evidence value.")
    return raw


def _workspace_required_detail_code(
    payload: Mapping[str, object],
    allowed: frozenset[str],
) -> str:
    value = _workspace_optional_detail_code(payload, allowed)
    if value is None:
        raise ValueError("detail_code is required for workspace attribution.")
    return value


def _workspace_enum(
    payload: Mapping[str, object],
    field_name: str,
    allowed: set[str],
) -> str:
    value = payload.get(field_name)
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{field_name} is not a recognized workspace evidence value.")
    return value


def _workspace_required_nonnegative_int(payload: Mapping[str, object], field_name: str) -> int:
    value = _nonnegative_int(payload.get(field_name))
    if value is None:
        raise ValueError(f"{field_name} must be a bounded nonnegative integer.")
    return value


def _workspace_required_positive_int(payload: Mapping[str, object], field_name: str) -> int:
    value = _positive_int(payload.get(field_name))
    if value is None:
        raise ValueError(f"{field_name} must be a bounded positive integer.")
    return value


def _workspace_optional_positive_int(
    payload: Mapping[str, object],
    field_name: str,
) -> int | None:
    if payload.get(field_name) is None:
        return None
    return _workspace_required_positive_int(payload, field_name)


def _workspace_required_bool(payload: Mapping[str, object], field_name: str) -> bool:
    value = payload.get(field_name)
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _same_workspace_fact(
    left: BaseModel,
    right: BaseModel,
    *,
    excluded: set[str] | None = None,
) -> bool:
    excluded_fields = {"source_ref"} if excluded is None else excluded

    def normalized(value: object) -> object:
        if isinstance(value, BaseModel):
            return normalized(value.model_dump())
        if type(value) is dict:
            return {
                key: normalized(item) for key, item in value.items() if key not in excluded_fields
            }
        if isinstance(value, list | tuple):
            return tuple(normalized(item) for item in value)
        return value

    return normalized(left) == normalized(right)


def _warn_malformed_workspace(
    session_id: str,
    record: EventRecord,
    warnings: list[RuntimeEvidenceWarning],
) -> None:
    warnings.append(
        RuntimeEvidenceWarning(
            code=RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE,
            session_id=session_id,
            event_id=record.event.id,
            sequence=record.sequence,
        )
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
        raw_profile_fingerprint = payload.get("execution_profile_fingerprint")
        profile_fingerprint = _optional_execution_profile_fingerprint(raw_profile_fingerprint)
        if raw_profile_fingerprint is not None and profile_fingerprint is None:
            builder.execution_profile_conflict = True
            builder.execution_profile_fingerprint = None
            warnings.append(
                RuntimeEvidenceWarning(
                    code=RuntimeEvidenceWarningCode.MALFORMED_EXECUTION_PROFILE,
                    session_id=session.id,
                    event_id=event.id,
                    sequence=record.sequence,
                )
            )
        elif profile_fingerprint is not None and not builder.execution_profile_conflict:
            if builder.execution_profile_fingerprint is None:
                builder.execution_profile_fingerprint = profile_fingerprint
            elif builder.execution_profile_fingerprint != profile_fingerprint:
                builder.execution_profile_conflict = True
                builder.execution_profile_fingerprint = None
                warnings.append(
                    RuntimeEvidenceWarning(
                        code=RuntimeEvidenceWarningCode.MALFORMED_EXECUTION_PROFILE,
                        session_id=session.id,
                        event_id=event.id,
                        sequence=record.sequence,
                    )
                )
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
                execution_profile_fingerprint=builder.execution_profile_fingerprint,
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


def _optional_execution_profile_fingerprint(value: object) -> str | None:
    if (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
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
