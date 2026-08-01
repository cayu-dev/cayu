"""Public server API contract models."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from cayu._validation import json_utf8_size_within_limit, require_unicode_scalar_text
from cayu.core.events import EVENT_ID_MAX_CHARS
from cayu.runtime.aggregates import (
    AggregateAccuracy,
    AggregateCount,
    UsageAggregateBreakdown,
    UsageAggregateTotals,
    UsageCostRollup,
    UsageSessionAggregateBreakdown,
    UsageSessionCostBreakdown,
)
from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    CostLineItem,
    PriceBook,
    SessionCostSummary,
)
from cayu.runtime.interactions import InteractionSummaryEvidence
from cayu.runtime.sessions import (
    MAX_USAGE_ROLLUP_WINDOW,
    SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT,
    SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
    SESSION_TOPOLOGY_MAX_CHILD_LIMIT,
    SESSION_TOPOLOGY_MAX_CURSOR_BYTES,
    SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    SESSION_TOPOLOGY_MAX_NODES,
    SessionAggregateFilter,
    SessionOperationalSnapshot,
)
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
    TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    TASK_TOPOLOGY_MAX_CURSOR_BYTES,
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
    TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    TASK_TOPOLOGY_MAX_NODES,
    TaskAggregateFilter,
    TaskOperationalSnapshot,
    TaskTopologyTruncatedField,
)
from cayu.runtime.usage import (
    AggregateUsageMetrics,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
)
from cayu.server.sse import (
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_EVENT_DATA_MAX_BYTES,
    SSE_REPLAY_START_MARKER_FORMAT,
    SseErrorCode,
    SseErrorKind,
)

SERVER_API_PREFIX = "/api"
SERVER_CONTRACT_VERSION = "6"
SSE_CONTENT_TYPE = "text/event-stream"
SSE_LAST_EVENT_ID_FORMAT = "session_id:cayu_event_<sequence>"
MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS = 64
MAX_SYSTEM_DEPLOYMENT_NAME_CHARS = 128
MAX_SYSTEM_PRICING_METADATA_CHARS = 256
DEFAULT_SESSION_TOPOLOGY_RESULT_BYTES = 1024 * 1024
MAX_SESSION_TOPOLOGY_RESULT_BYTES = 4 * 1024 * 1024
MAX_SESSION_TOPOLOGY_REQUEST_BYTES = 256 * 1024
MAX_EXECUTION_TOPOLOGY_EDGES = 1500

SessionTopologyIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    ),
]
SessionTopologyCursor = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=SESSION_TOPOLOGY_MAX_CURSOR_BYTES,
    ),
]
TaskTopologyIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    ),
]
TaskTopologyCursor = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=TASK_TOPOLOGY_MAX_CURSOR_BYTES,
    ),
]


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiBaseModel):
    ok: StrictBool


SystemAccessKind = Literal["open", "authenticated"]
SystemDiagnosticTextStatus = Literal["available", "not_provided", "omitted"]


class SystemDeploymentDiagnostics(ApiBaseModel):
    """Bounded resolved server identity and effective access posture."""

    name: str | None = Field(default=None, max_length=MAX_SYSTEM_DEPLOYMENT_NAME_CHARS)
    name_status: SystemDiagnosticTextStatus
    api_access: SystemAccessKind
    dashboard_access: SystemAccessKind | None
    dashboard_enabled: StrictBool
    docs_enabled: StrictBool | None

    @model_validator(mode="after")
    def validate_availability(self) -> SystemDeploymentDiagnostics:
        if self.name_status == "available" and self.name is None:
            raise ValueError("Available deployment names require a value.")
        if self.name_status != "available" and self.name is not None:
            raise ValueError("Unavailable deployment names cannot include a value.")
        if self.dashboard_enabled and self.dashboard_access is None:
            raise ValueError("Enabled dashboards require an access posture.")
        if not self.dashboard_enabled and self.dashboard_access is not None:
            raise ValueError("Disabled dashboards cannot include an access posture.")
        return self


class SystemVersionDiagnostics(ApiBaseModel):
    cayu: str | None = Field(max_length=128)
    server_contract: str = Field(max_length=32)


class ArtifactStoreDiagnostic(ApiBaseModel):
    """Path-safe registration identity and the required ArtifactStore contract."""

    fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$", max_length=71)
    store_contract_operations: tuple[
        Literal["list"],
        Literal["read"],
        Literal["write"],
        Literal["delete"],
    ] = (
        "list",
        "read",
        "write",
        "delete",
    )


class ArtifactStoreDiagnostics(ApiBaseModel):
    registrations: tuple[ArtifactStoreDiagnostic, ...] = Field(
        max_length=MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS
    )
    total_count: StrictInt = Field(ge=0)
    truncated: StrictBool

    @model_validator(mode="after")
    def validate_count(self) -> ArtifactStoreDiagnostics:
        if self.total_count < len(self.registrations):
            raise ValueError("Artifact store total cannot be smaller than registrations.")
        if self.truncated is not (self.total_count > len(self.registrations)):
            raise ValueError("Artifact store truncation must match the returned count.")
        return self


class PricingCatalogDiagnostics(ApiBaseModel):
    configured: StrictBool
    metadata_status: Literal["available", "not_configured", "omitted"]
    price_book_version: str | None = Field(
        default=None,
        max_length=MAX_SYSTEM_PRICING_METADATA_CHARS,
    )
    generated_at: str | None = Field(
        default=None,
        max_length=MAX_SYSTEM_PRICING_METADATA_CHARS,
    )

    @model_validator(mode="after")
    def validate_metadata(self) -> PricingCatalogDiagnostics:
        metadata_present = self.price_book_version is not None and self.generated_at is not None
        if self.metadata_status == "available" and not metadata_present:
            raise ValueError("Available pricing metadata requires both identity fields.")
        if self.metadata_status != "available" and (
            self.price_book_version is not None or self.generated_at is not None
        ):
            raise ValueError("Unavailable pricing metadata cannot include identity fields.")
        if self.configured and self.metadata_status == "not_configured":
            raise ValueError("Configured pricing cannot be marked not configured.")
        if not self.configured and self.metadata_status != "not_configured":
            raise ValueError("Unconfigured pricing must be marked not configured.")
        return self


class OperationalSnapshotRequest(ApiBaseModel):
    session_filter: SessionAggregateFilter = Field(default_factory=SessionAggregateFilter)
    task_filter: TaskAggregateFilter = Field(default_factory=TaskAggregateFilter)
    include_tasks: StrictBool = True


class OperationalSnapshotResponse(ApiBaseModel):
    scope: Literal["configured_stores"]
    cross_store_atomic: Literal[False]
    sessions: SessionOperationalSnapshot
    task_snapshot_status: Literal[
        "available",
        "not_requested",
        "not_configured",
        "unsupported",
    ]
    tasks: TaskOperationalSnapshot | None

    @model_validator(mode="after")
    def validate_task_availability(self) -> OperationalSnapshotResponse:
        if self.task_snapshot_status == "available" and self.tasks is None:
            raise ValueError("Available task snapshots must include tasks.")
        if self.task_snapshot_status != "available" and self.tasks is not None:
            raise ValueError("Unavailable task snapshots cannot include tasks.")
        return self


MAX_USAGE_ROLLUP_PRICE_BOOK_BYTES = 2 * 1024 * 1024
MAX_USAGE_ROLLUP_PRICES = 500
MAX_USAGE_ROLLUP_PRICE_MATCH_RULES = 2000
MAX_USAGE_ROLLUP_RESOURCE_MAPPINGS = 1000
MAX_USAGE_ROLLUP_CONTEXT_REQUIREMENTS = 100
MAX_USAGE_ROLLUP_CONTEXT_SELECTOR_VALUES = 2000
MAX_USAGE_ROLLUP_PRICE_RESOLUTION_WORK = 500_000
MAX_USAGE_ROLLUP_REQUEST_BYTES = 3 * 1024 * 1024
MAX_USAGE_ROLLUP_RESULT_BYTES = 4 * 1024 * 1024
MAX_USAGE_ROLLUP_CURRENCY_BYTES = 64


class UsageRollupRequest(ApiBaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    start_at: datetime
    end_at: datetime
    session_filter: SessionAggregateFilter = Field(default_factory=SessionAggregateFilter)
    group_limit: StrictInt = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Maximum returned groups, applied independently to provider, model, and billing "
            "identity breakdowns. Omitted groups are represented by an explicit remainder."
        ),
    )
    session_group_limit: StrictInt | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Opt-in maximum number of per-session usage groups. Omitted matching sessions "
            "are represented by an exact aggregate remainder. When omitted, stores perform "
            "no per-session breakdown or session-aware pricing work."
        ),
    )
    pricing_input_limit: StrictInt = Field(
        default=1000,
        ge=1,
        le=5000,
        description=(
            "Maximum canonical price-input groups, applied independently to the shared "
            "projection and, when session_group_limit is present, the session-aware "
            "projection."
        ),
    )
    pricing: PriceBook | None = Field(
        default=None,
        description=(
            "Optional bounded price book. The serialized value may contain at most "
            "2 MiB, 500 prices, 2,000 price match rules, 1,000 resource mappings, "
            "100 contextual requirements, and 2,000 contextual selector values. "
            "Each currency identity may contain at most 64 UTF-8 bytes. "
            "The price-input limit multiplied by the resolution candidates may not "
            "exceed 500,000."
        ),
    )

    @field_validator("pricing", mode="before")
    @classmethod
    def bound_raw_pricing(cls, value: object) -> object:
        if value is None or isinstance(value, PriceBook):
            return value
        if type(value) is not dict:
            return value
        raw_pricing = cast("dict[str, object]", value)
        if not json_utf8_size_within_limit(value, MAX_USAGE_ROLLUP_PRICE_BOOK_BYTES):
            raise ValueError(
                f"pricing cannot exceed {MAX_USAGE_ROLLUP_PRICE_BOOK_BYTES} JSON bytes."
            )
        prices = _bounded_raw_sequence(
            raw_pricing.get("prices"),
            field_name="pricing.prices",
            limit=MAX_USAGE_ROLLUP_PRICES,
        )
        _bounded_raw_sequence(
            raw_pricing.get("resource_mappings", ()),
            field_name="pricing.resource_mappings",
            limit=MAX_USAGE_ROLLUP_RESOURCE_MAPPINGS,
        )
        _bounded_raw_sequence(
            raw_pricing.get("contextual_pricing_requirements", ()),
            field_name="pricing.contextual_pricing_requirements",
            limit=MAX_USAGE_ROLLUP_CONTEXT_REQUIREMENTS,
        )
        match_rule_count = 0
        context_selector_value_count = 0
        for raw_price in prices:
            match_rule_count += 1
            if type(raw_price) is dict:
                price = cast("dict[str, object]", raw_price)
                match_rule_count += len(
                    _bounded_raw_sequence(
                        price.get("aliases", ()),
                        field_name="pricing.prices[].aliases",
                        limit=MAX_USAGE_ROLLUP_PRICE_MATCH_RULES,
                    )
                )
                match_rule_count += len(
                    _bounded_raw_sequence(
                        price.get("match_prefixes", ()),
                        field_name="pricing.prices[].match_prefixes",
                        limit=MAX_USAGE_ROLLUP_PRICE_MATCH_RULES,
                    )
                )
                raw_context = price.get("pricing_context")
                if type(raw_context) is dict:
                    raw_dimensions = cast("dict[str, object]", raw_context).get("dimensions")
                    if type(raw_dimensions) is dict:
                        for raw_values in cast("dict[object, object]", raw_dimensions).values():
                            context_selector_value_count += len(
                                _bounded_raw_sequence(
                                    raw_values,
                                    field_name=("pricing.prices[].pricing_context.dimensions[]"),
                                    limit=MAX_USAGE_ROLLUP_CONTEXT_SELECTOR_VALUES,
                                )
                            )
                            if (
                                context_selector_value_count
                                > MAX_USAGE_ROLLUP_CONTEXT_SELECTOR_VALUES
                            ):
                                raise ValueError(
                                    "pricing cannot contain more than "
                                    f"{MAX_USAGE_ROLLUP_CONTEXT_SELECTOR_VALUES} total "
                                    "contextual selector values."
                                )
            if match_rule_count > MAX_USAGE_ROLLUP_PRICE_MATCH_RULES:
                raise ValueError(
                    "pricing cannot contain more than "
                    f"{MAX_USAGE_ROLLUP_PRICE_MATCH_RULES} total match rules."
                )
        return value

    @field_validator("start_at", "end_at")
    @classmethod
    def normalize_window_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def bound_pricing_resolution_work(self) -> UsageRollupRequest:
        if self.start_at >= self.end_at:
            raise ValueError("Usage rollup start_at must be before end_at.")
        if self.end_at - self.start_at > MAX_USAGE_ROLLUP_WINDOW:
            raise ValueError(
                f"Usage rollup window cannot exceed {MAX_USAGE_ROLLUP_WINDOW.days} days."
            )
        if self.pricing is None:
            return self
        for price in self.pricing.prices:
            for schedule in price.schedules:
                currency = require_unicode_scalar_text(
                    schedule.pricing.currency,
                    "pricing currency",
                )
                if len(currency.encode("utf-8")) > MAX_USAGE_ROLLUP_CURRENCY_BYTES:
                    raise ValueError(
                        "pricing currencies cannot exceed "
                        f"{MAX_USAGE_ROLLUP_CURRENCY_BYTES} UTF-8 bytes."
                    )
        if not json_utf8_size_within_limit(
            self.pricing.model_dump(mode="json"),
            MAX_USAGE_ROLLUP_PRICE_BOOK_BYTES,
        ):
            raise ValueError(
                f"pricing cannot exceed {MAX_USAGE_ROLLUP_PRICE_BOOK_BYTES} JSON bytes."
            )
        if len(self.pricing.prices) > MAX_USAGE_ROLLUP_PRICES:
            raise ValueError(
                f"pricing.prices cannot contain more than {MAX_USAGE_ROLLUP_PRICES} items."
            )
        if len(self.pricing.resource_mappings) > MAX_USAGE_ROLLUP_RESOURCE_MAPPINGS:
            raise ValueError(
                "pricing.resource_mappings cannot contain more than "
                f"{MAX_USAGE_ROLLUP_RESOURCE_MAPPINGS} items."
            )
        if (
            len(self.pricing.contextual_pricing_requirements)
            > MAX_USAGE_ROLLUP_CONTEXT_REQUIREMENTS
        ):
            raise ValueError(
                "pricing.contextual_pricing_requirements cannot contain more than "
                f"{MAX_USAGE_ROLLUP_CONTEXT_REQUIREMENTS} items."
            )
        rule_counts = tuple(
            1 + len(price.aliases) + len(price.match_prefixes) for price in self.pricing.prices
        )
        match_rules = sum(rule_counts)
        if match_rules > MAX_USAGE_ROLLUP_PRICE_MATCH_RULES:
            raise ValueError(
                "pricing cannot contain more than "
                f"{MAX_USAGE_ROLLUP_PRICE_MATCH_RULES} total match rules."
            )
        context_selector_values = sum(
            len(values)
            for price in self.pricing.prices
            if price.pricing_context is not None
            for values in price.pricing_context.dimensions.values()
        )
        if context_selector_values > MAX_USAGE_ROLLUP_CONTEXT_SELECTOR_VALUES:
            raise ValueError(
                "pricing cannot contain more than "
                f"{MAX_USAGE_ROLLUP_CONTEXT_SELECTOR_VALUES} total contextual selector values."
            )
        match_work = sum(
            rule_count
            * (
                1
                + (
                    0
                    if price.pricing_context is None
                    else len(price.pricing_context.dimensions)
                    + sum(len(values) for values in price.pricing_context.dimensions.values())
                )
            )
            for price, rule_count in zip(self.pricing.prices, rule_counts, strict=True)
        )
        schedule_work = max(
            (
                len(price.schedules)
                + max(len(schedule.pricing.standard) for schedule in price.schedules)
                for price in self.pricing.prices
            ),
            default=0,
        )
        resolution_work = (
            1
            + match_work
            + len(self.pricing.resource_mappings)
            + len(self.pricing.contextual_pricing_requirements)
            + schedule_work
        )
        pricing_projection_count = 2 if self.session_group_limit is not None else 1
        if (
            self.pricing_input_limit * resolution_work * pricing_projection_count
            > MAX_USAGE_ROLLUP_PRICE_RESOLUTION_WORK
        ):
            raise ValueError(
                "pricing_input_limit, requested projections, and pricing candidates exceed the "
                f"{MAX_USAGE_ROLLUP_PRICE_RESOLUTION_WORK}-candidate resolution bound."
            )
        return self


def validate_usage_rollup_price_book(value: object) -> PriceBook:
    """Validate a default catalog against the usage endpoint's exact bounds."""

    validation_request = UsageRollupRequest.model_validate(
        {
            "start_at": datetime(2000, 1, 1, tzinfo=UTC),
            "end_at": datetime(2000, 1, 2, tzinfo=UTC),
            "pricing": value,
        }
    )
    if validation_request.pricing is None:
        raise ValueError("A default usage price book cannot be null.")
    return validation_request.pricing


def _bounded_raw_sequence(
    value: object,
    *,
    field_name: str,
    limit: int,
) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if len(value) > limit:
            raise ValueError(f"{field_name} cannot contain more than {limit} items.")
        return value
    return ()


class UsageRollupResponse(ApiBaseModel):
    scope: Literal["configured_session_store"]
    time_basis: Literal["event.timestamp"]
    session_filter_basis: Literal["current_session_attributes"]
    as_of: datetime
    start_at: datetime
    end_at: datetime
    accuracy: AggregateAccuracy
    matching_session_count: AggregateCount = Field(ge=0)
    active_session_count: AggregateCount = Field(ge=0)
    includes_active_sessions: StrictBool
    totals: UsageAggregateTotals
    provider_breakdown: UsageAggregateBreakdown
    model_breakdown: UsageAggregateBreakdown
    cost: UsageCostRollup | None
    session_breakdown: UsageSessionAggregateBreakdown | None = None
    session_cost_breakdown: UsageSessionCostBreakdown | None = None

    @model_validator(mode="after")
    def validate_session_breakdown_alignment(self) -> UsageRollupResponse:
        if self.session_cost_breakdown is None:
            return self
        if self.session_breakdown is None or self.cost is None:
            raise ValueError("Session costs require session usage and shared cost results.")
        usage_ids = tuple(group.session_id for group in self.session_breakdown.groups)
        cost_ids = tuple(group.session_id for group in self.session_cost_breakdown.groups)
        if cost_ids != usage_ids:
            raise ValueError("Session cost groups must match retained session usage groups.")
        for usage_group, cost_group in zip(
            self.session_breakdown.groups,
            self.session_cost_breakdown.groups,
            strict=True,
        ):
            represented_steps = (
                cost_group.cost.evaluated_model_steps + cost_group.cost.unevaluated_model_steps
            )
            if represented_steps != usage_group.totals.model_steps:
                raise ValueError("Session cost groups must account for session model steps.")
        usage_remainder_count = (
            None
            if self.session_breakdown.remainder is None
            else self.session_breakdown.remainder.group_count
        )
        cost_remainder_count = (
            None
            if self.session_cost_breakdown.remainder is None
            else self.session_cost_breakdown.remainder.group_count
        )
        if cost_remainder_count != usage_remainder_count:
            raise ValueError("Session cost remainder must match the session usage remainder.")
        if (
            self.session_breakdown.remainder is not None
            and self.session_cost_breakdown.remainder is not None
            and (
                self.session_cost_breakdown.remainder.cost.evaluated_model_steps
                + self.session_cost_breakdown.remainder.cost.unevaluated_model_steps
                != self.session_breakdown.remainder.totals.model_steps
            )
        ):
            raise ValueError("Session cost remainder must account for omitted model steps.")
        return self


class ApiErrorResponse(ApiBaseModel):
    detail: str


CheckpointCompatibilityReason = Literal[
    "invalid_root_checkpoint",
    "invalid_checkpoint_schema_version",
    "checkpoint_schema_version_too_new",
    "checkpoint_schema_version_too_old",
]


class CheckpointCompatibilityEvidence(ApiBaseModel):
    checkpoint_kind: Literal["root"]
    observed_version: StrictInt | None = Field(ge=1)
    reason: CheckpointCompatibilityReason
    recovery_disposition: Literal["cannot_migrate"]
    resumable_in_place: Literal[False]
    session_id: str
    supported_max_version: StrictInt = Field(ge=1)
    supported_min_version: StrictInt = Field(ge=1)


class CheckpointCompatibilityErrorResponse(ApiBaseModel):
    detail: CheckpointCompatibilityEvidence


AGGREGATE_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    501: {
        "description": "The configured store does not implement this aggregate read.",
        "model": ApiErrorResponse,
    }
}
USAGE_ROLLUP_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    **AGGREGATE_ENDPOINT_RESPONSES,
    409: {
        "description": ("A per-session identity cannot cross the configured redaction boundary."),
        "model": ApiErrorResponse,
    },
    413: {
        "description": "The usage-rollup request or serialized response exceeds its byte limit.",
        "model": ApiErrorResponse,
    },
    500: {
        "description": "The configured store returned an inconsistent usage projection.",
        "model": ApiErrorResponse,
    },
}


class SseEventEnvelope(ApiBaseModel):
    """JSON payload in each runtime event SSE ``data:`` frame."""

    id: str = Field(max_length=EVENT_ID_MAX_CHARS)
    type: str
    session_id: str
    interaction_id: str | None
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
        interaction_id="interaction-123",
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
    event_id_format: Literal["session_id:cayu_event_<sequence>"] = SSE_LAST_EVENT_ID_FORMAT
    replay_header: Literal["Last-Event-ID"] = "Last-Event-ID"
    max_event_id_chars: StrictInt = Field(default=EVENT_ID_MAX_CHARS, ge=1)
    mutation_id_header: Literal["Cayu-Mutation-ID"] = "Cayu-Mutation-ID"
    mutation_acceptance_event_type: Literal["server.mutation.accepted"] = "server.mutation.accepted"
    replay_start_marker_format: Literal["session_id:"] = SSE_REPLAY_START_MARKER_FORMAT
    unknown_event_marker_behavior: Literal["reject"] = "reject"
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


CapabilityUnavailableReason = Literal["not_configured", "unsupported"]
ConfiguredStoreRole = Literal["session", "task", "knowledge", "artifact"]


class CapabilityOperation(ApiBaseModel):
    """Availability of one control-plane read or mutation operation."""

    enabled: StrictBool
    unavailable_reason: CapabilityUnavailableReason | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> CapabilityOperation:
        if self.enabled and self.unavailable_reason is not None:
            raise ValueError("Enabled capability operations cannot have an unavailable reason.")
        if not self.enabled and self.unavailable_reason is None:
            raise ValueError("Disabled capability operations require an unavailable reason.")
        return self


class OptionalSurfaceCapability(ApiBaseModel):
    """Configuration and operation availability for one optional surface."""

    configured: StrictBool
    read: CapabilityOperation
    mutate: CapabilityOperation

    @model_validator(mode="after")
    def validate_configuration(self) -> OptionalSurfaceCapability:
        if not self.configured and (self.read.enabled or self.mutate.enabled):
            raise ValueError("Unconfigured surfaces cannot expose enabled operations.")
        return self


class ControlPlaneSurfaceCapabilities(ApiBaseModel):
    dashboard: OptionalSurfaceCapability
    # Added within control-plane contract v4. Keep the field optional so a
    # dashboard can fail closed against an incomplete capability response
    # instead of dereferencing a capability that did not exist yet.
    workflow: OptionalSurfaceCapability | None = None
    tasks: OptionalSurfaceCapability
    reviewed_knowledge: OptionalSurfaceCapability
    artifacts: OptionalSurfaceCapability
    usage: OptionalSurfaceCapability
    pricing: OptionalSurfaceCapability


class ControlPlaneMutationCapabilities(ApiBaseModel):
    session_execution: CapabilityOperation
    session_interruption: CapabilityOperation
    pending_action_resolution: CapabilityOperation
    session_annotations: CapabilityOperation
    task_lifecycle: CapabilityOperation
    knowledge_review: CapabilityOperation


class ServerContractActor(ApiBaseModel):
    """Bounded actor projection that deliberately excludes arbitrary claims."""

    subject: str = Field(max_length=512)
    tenant: str | None = Field(default=None, max_length=512)


class ControlPlaneCapabilities(ApiBaseModel):
    """Server-authoritative discovery metadata for the Cayu control plane.

    This projection is presentation metadata rather than an authorization
    token. Every underlying route continues to enforce its configured access
    dependency and runtime preconditions.
    """

    cayu_version: str | None = Field(max_length=128)
    configured_store_roles: tuple[ConfiguredStoreRole, ...] = Field(max_length=4)
    actor: ServerContractActor | None
    surfaces: ControlPlaneSurfaceCapabilities
    mutations: ControlPlaneMutationCapabilities


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
    capabilities: ControlPlaneCapabilities


class SystemDiagnosticsResponse(ApiBaseModel):
    """Protected bounded Cayu configuration and registration diagnostics."""

    observed_at: datetime
    deployment: SystemDeploymentDiagnostics
    versions: SystemVersionDiagnostics
    capabilities: ControlPlaneCapabilities
    artifact_stores: ArtifactStoreDiagnostics
    pricing_catalog: PricingCatalogDiagnostics

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        return value.astimezone(UTC)


class ApiEventRecord(ApiBaseModel):
    sequence: StrictInt = Field(ge=0)
    id: str = Field(max_length=EVENT_ID_MAX_CHARS)
    type: str
    session_id: str
    interaction_id: str | None
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


class SessionTopologyRequest(ApiBaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expanded_parent_ids: tuple[SessionTopologyIdentifier, ...] = Field(
        default_factory=tuple,
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    child_cursors: dict[SessionTopologyIdentifier, SessionTopologyCursor] = Field(
        default_factory=dict,
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    ancestor_depth_limit: StrictInt = Field(
        default=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
        ge=1,
        le=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
    )
    child_limit: StrictInt = Field(
        default=SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT,
        ge=1,
        le=SESSION_TOPOLOGY_MAX_CHILD_LIMIT,
    )
    linked_task_session_ids: tuple[TaskTopologyIdentifier, ...] = Field(
        default_factory=tuple,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    task_session_cursors: dict[TaskTopologyIdentifier, TaskTopologyCursor] = Field(
        default_factory=dict,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    expanded_task_parent_ids: tuple[TaskTopologyIdentifier, ...] = Field(
        default_factory=tuple,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    task_child_cursors: dict[TaskTopologyIdentifier, TaskTopologyCursor] = Field(
        default_factory=dict,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    task_session_limit: StrictInt = Field(
        default=TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
        ge=1,
        le=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    task_child_limit: StrictInt = Field(
        default=TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
        ge=1,
        le=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    max_result_bytes: StrictInt = Field(
        default=DEFAULT_SESSION_TOPOLOGY_RESULT_BYTES,
        ge=1024,
        le=MAX_SESSION_TOPOLOGY_RESULT_BYTES,
    )

    @field_validator("expanded_parent_ids")
    @classmethod
    def validate_expanded_parent_id_bytes(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        for value in values:
            try:
                encoded = value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "Session topology identifiers must contain portable Unicode text."
                ) from exc
            if len(encoded) > SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES:
                raise ValueError("A session topology identifier exceeds its byte limit.")
        return values

    @field_validator("linked_task_session_ids", "expanded_task_parent_ids")
    @classmethod
    def validate_task_topology_identifier_bytes(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        for value in values:
            try:
                encoded = value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "Task topology identifiers must contain portable Unicode text."
                ) from exc
            if len(encoded) > TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES:
                raise ValueError("A task topology identifier exceeds its byte limit.")
        return values

    @field_validator("task_session_cursors", "task_child_cursors")
    @classmethod
    def validate_task_topology_cursor_bytes(
        cls,
        values: dict[str, str],
    ) -> dict[str, str]:
        for scope_id, cursor in values.items():
            try:
                encoded_scope_id = scope_id.encode("utf-8")
                encoded_cursor = cursor.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "Task topology identifiers and cursors must contain portable Unicode text."
                ) from exc
            if len(encoded_scope_id) > TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES:
                raise ValueError("A task topology identifier exceeds its byte limit.")
            if len(encoded_cursor) > TASK_TOPOLOGY_MAX_CURSOR_BYTES:
                raise ValueError("A task topology cursor exceeds its byte limit.")
        return values

    @field_validator("child_cursors")
    @classmethod
    def validate_child_cursor_bytes(cls, values: dict[str, str]) -> dict[str, str]:
        for parent_id, cursor in values.items():
            try:
                encoded_parent_id = parent_id.encode("utf-8")
                encoded_cursor = cursor.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "Session topology identifiers and cursors must contain portable Unicode text."
                ) from exc
            if len(encoded_parent_id) > SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES:
                raise ValueError("A session topology identifier exceeds its byte limit.")
            if len(encoded_cursor) > SESSION_TOPOLOGY_MAX_CURSOR_BYTES:
                raise ValueError("A session topology cursor exceeds its byte limit.")
        return values


class ApiSessionTopologyNode(ApiBaseModel):
    id: SessionTopologyIdentifier
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: SessionTopologyIdentifier | None
    causal_budget_id: SessionTopologyIdentifier
    runtime_name: str
    runtime_version: str | None
    environment_name: str | None
    status: str
    created_at: str
    updated_at: str
    last_activity_at: str


class ApiSessionTopologyBranch(ApiBaseModel):
    parent_session_id: SessionTopologyIdentifier
    children: list[ApiSessionTopologyNode] = Field(max_length=SESSION_TOPOLOGY_MAX_CHILD_LIMIT)
    next_cursor: SessionTopologyCursor | None
    has_more: StrictBool


class ApiTaskTopologyNode(ApiBaseModel):
    id: TaskTopologyIdentifier
    type: str | None = Field(max_length=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES)
    title: str | None = Field(max_length=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES)
    status: str
    status_reason: str | None = Field(max_length=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES)
    session_id: TaskTopologyIdentifier | None
    parent_task_id: TaskTopologyIdentifier | None
    assigned_agent_name: str | None = Field(max_length=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES)
    created_at: str
    updated_at: str
    truncated_fields: list[TaskTopologyTruncatedField] = Field(max_length=4)


class ApiTaskTopologySessionBranch(ApiBaseModel):
    session_id: TaskTopologyIdentifier
    tasks: list[ApiTaskTopologyNode] = Field(max_length=TASK_TOPOLOGY_MAX_BRANCH_LIMIT)
    next_cursor: TaskTopologyCursor | None
    has_more: StrictBool


class ApiTaskTopologyChildBranch(ApiBaseModel):
    parent_task_id: TaskTopologyIdentifier
    children: list[ApiTaskTopologyNode] = Field(max_length=TASK_TOPOLOGY_MAX_BRANCH_LIMIT)
    next_cursor: TaskTopologyCursor | None
    has_more: StrictBool


class ApiTaskTopologyProjection(ApiBaseModel):
    status: Literal["available", "not_configured", "unsupported"]
    observed_at: datetime | None
    session_branches: list[ApiTaskTopologySessionBranch] = Field(
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS
    )
    expanded_parents: list[ApiTaskTopologyNode] = Field(
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS
    )
    child_branches: list[ApiTaskTopologyChildBranch] = Field(
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS
    )
    unique_node_count: StrictInt = Field(ge=0, le=TASK_TOPOLOGY_MAX_NODES)

    @model_validator(mode="after")
    def validate_availability(self) -> ApiTaskTopologyProjection:
        has_projection = bool(
            self.observed_at is not None
            or self.session_branches
            or self.expanded_parents
            or self.child_branches
            or self.unique_node_count
        )
        if self.status == "available" and self.observed_at is None:
            raise ValueError("Available task topology requires an observation timestamp.")
        if self.status != "available" and has_projection:
            raise ValueError("Unavailable task topology cannot include projected task data.")
        return self


class ApiExecutionTopologyEdge(ApiBaseModel):
    kind: Literal["session_parent", "task_parent", "task_session"]
    source_id: SessionTopologyIdentifier
    target_id: SessionTopologyIdentifier
    target_loaded: StrictBool


class SessionTopologyResponse(ApiBaseModel):
    scope: Literal["session_focus"] = "session_focus"
    observed_at: datetime
    cross_store_atomic: Literal[False]
    focus: ApiSessionTopologyNode
    ancestors: list[ApiSessionTopologyNode] = Field(max_length=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH)
    expanded_parents: list[ApiSessionTopologyNode] = Field(
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS
    )
    branches: list[ApiSessionTopologyBranch] = Field(
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS
    )
    unique_node_count: StrictInt = Field(ge=1, le=SESSION_TOPOLOGY_MAX_NODES)
    task_projection: ApiTaskTopologyProjection
    edges: list[ApiExecutionTopologyEdge] = Field(max_length=MAX_EXECUTION_TOPOLOGY_EDGES)


SESSION_TOPOLOGY_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {
        "description": (
            "The focus session or one requested expanded session/task parent does not exist."
        ),
        "model": ApiErrorResponse,
    },
    409: {
        "description": (
            "Durable session/task lineage is inconsistent, or continuation authority cannot "
            "cross the configured redaction boundary."
        ),
        "model": ApiErrorResponse,
    },
    413: {
        "description": (
            "The request bytes, session/task ancestry, or serialized response exceed "
            "a safety bound."
        ),
        "model": ApiErrorResponse,
    },
    501: {
        "description": "The configured session store does not implement topology reads.",
        "model": ApiErrorResponse,
    },
}


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
    usage: AggregateUsageMetrics
    session_summaries: tuple[SessionUsageSummary, ...]


class UsageBreakdownItem(ApiBaseModel):
    provider_name: str | None
    model: str | None
    session_count: StrictInt = Field(ge=0)
    model_steps: StrictInt = Field(ge=0)
    usage: AggregateUsageMetrics


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
    policy_evidence: (
        Literal[
            "unplanned",
            "authoritative",
            "unregistered",
            "ambiguous",
        ]
        | None
    ) = None
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
    409: {
        "description": "A checkpoint requires a different Cayu runtime version.",
        "model": CheckpointCompatibilityErrorResponse,
    },
    413: {
        "description": "The pending-action page exceeds the bounded response size.",
        "model": ApiErrorResponse,
    },
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


CAUSAL_BUDGET_SUMMARY_ENDPOINT_RESPONSES: dict[int | str, dict[str, Any]] = {
    413: {
        "description": (
            "The causal-budget summary exceeds its session, event-count, event-input-byte, "
            "or serialized response safety bound."
        ),
        "model": ApiErrorResponse,
    },
    501: {
        "description": (
            "The configured session store cannot enforce byte-bounded event reads "
            "for this legacy summary."
        ),
        "model": ApiErrorResponse,
    },
}


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


class ApiInteractionSummary(InteractionSummaryEvidence):
    interaction_id: str
    session_id: str
    terminal_event_id: str | None
    terminal_event_sequence: StrictInt | None = Field(default=None, ge=1)
    updated_at: datetime


class ListSessionInteractionsResponse(ApiBaseModel):
    session_id: str
    interactions: list[ApiInteractionSummary]
    next_sequence: StrictInt | None = Field(ge=0)
    has_more: StrictBool


class ApiTranscriptMessage(ApiBaseModel):
    index: StrictInt = Field(ge=0)
    interaction_id: str | None
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
    },
    404: {
        "description": "The replay session or mutation target does not exist.",
        "model": ApiErrorResponse,
    },
    409: {
        "description": (
            "The replay event marker is unknown or the mutation conflicts with "
            "the current session state."
        ),
        "model": ApiErrorResponse,
    },
    500: {
        "description": "The mutation could not open an accepted durable stream.",
        "model": ApiErrorResponse,
    },
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
