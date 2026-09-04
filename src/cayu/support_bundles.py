"""Bounded, redacted diagnostic support-bundle primitives."""

from __future__ import annotations

import asyncio
import io
import json
import os
import platform
import re
import secrets
import stat
import sys
import threading
import time
import zipfile
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal, Protocol, TypeAlias, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from cayu._version import package_version
from cayu.core.events import EventType
from cayu.environments.lifecycle import (
    EnvironmentLifecyclePolicy,
    EnvironmentLifecycleProgress,
    environment_lifecycle_progress_from_event,
)
from cayu.evals.store import EvalStore
from cayu.runtime.app import CayuApp
from cayu.runtime.checks import ProjectCheckReport
from cayu.runtime.manifest import AppManifest
from cayu.runtime.recovery_cleanup import (
    RECOVERY_CLEANUP_MAX_TIMEOUT_SECONDS,
    RecoveryCleanupDeadlineScope,
    RecoveryCleanupSupervisorSnapshot,
)
from cayu.runtime.service_manifest import PublicServiceManifest
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    EventQueryResultTooLarge,
    SessionOperationalSnapshot,
    SessionStore,
)
from cayu.runtime.system_diagnostics import SystemDiagnosticsResponse
from cayu.runtime.tasks import TaskOperationalSnapshot
from cayu.workspaces.branches import (
    WorkspaceBranchCapabilities,
    WorkspaceBranchLifecycleSummary,
)

SUPPORT_BUNDLE_SCHEMA_VERSION = "1"
SUPPORT_BUNDLE_COMMAND_VERSION = "1"
SUPPORT_BUNDLE_REPORT_MEMBER = "report.json"
SUPPORT_BUNDLE_SUMMARY_MEMBER = "summary.txt"
SUPPORT_BUNDLE_ALLOWED_MEMBERS = frozenset(
    {SUPPORT_BUNDLE_REPORT_MEMBER, SUPPORT_BUNDLE_SUMMARY_MEMBER}
)
_KNOWN_EVENT_TYPE_VALUES = frozenset(item.value for item in EventType)
_REDACTED_CUSTOM_EVENT_TYPE = "custom.redacted"
_SyncStepResult = TypeVar("_SyncStepResult")

DEFAULT_COLLECTOR_TIMEOUT_SECONDS = 2.0
DEFAULT_COLLECTION_TIMEOUT_SECONDS = 15.0
DEFAULT_WORKER_TIMEOUT_SECONDS = 20.0
DEFAULT_PUBLICATION_TIMEOUT_SECONDS = 10.0
DEFAULT_RECONCILIATION_TIMEOUT_SECONDS = 5.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 40.0
DEFAULT_MAX_ITEMS = 100
DEFAULT_MAX_SESSIONS = 10
DEFAULT_EVENT_LIMIT = 50
_MAX_EVENT_LIMIT = 4999
DEFAULT_EVENT_QUERY_BYTES = 256 * 1024
DEFAULT_MAX_COLLECTOR_BYTES = 256 * 1024
DEFAULT_MAX_EVIDENCE_BYTES = 1024 * 1024
DEFAULT_MAX_BUNDLE_BYTES = 2 * 1024 * 1024

_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COLLECTOR_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_EMBEDDED_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\)[^\s\"'<>]+"
    r"|(?<![A-Za-z0-9_./-])/[^\s\"'<>]+"
)
_FORBIDDEN_KEYS = frozenset(
    {
        "arguments",
        "artifact_content",
        "credentials",
        "dsn",
        "env",
        "environ",
        "environment_map",
        "exception",
        "messages",
        "metadata",
        "model_output",
        "prompt",
        "raw_exception",
        "system_prompt",
        "tool_arguments",
        "tool_result",
        "tool_results",
        "traceback",
        "transcript",
    }
)
_OPTIONAL_DISTRIBUTIONS = (
    "boto3",
    "cryptography",
    "e2b",
    "fastapi",
    "google-auth",
    "ipython",
    "microsandbox",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "pillow",
    "playwright",
    "psycopg",
    "psycopg-pool",
    "pydantic-settings",
    "pypdf",
    "sse-starlette",
    "uvicorn",
)
_SQLITE_SCHEMA_STORE_IDENTITIES = frozenset(
    {
        ("cayu.storage.budget_ledger", "SQLiteBudgetLedger"),
        ("cayu.storage.evals_sqlite", "SQLiteEvalStore"),
        ("cayu.storage.event_watchers", "SQLiteEventWatcherStore"),
        ("cayu.storage.knowledge_sqlite", "SQLiteKnowledgeStore"),
        ("cayu.storage.sqlite", "SQLiteSessionStore"),
        ("cayu.storage.sqlite", "SQLiteTaskStore"),
    }
)
_POSTGRES_SCHEMA_STORE_IDENTITIES = frozenset(
    {
        ("cayu.storage.evals_postgres", "PostgresEvalStore"),
        ("cayu.storage.postgres", "PostgresBudgetLedger"),
        ("cayu.storage.postgres", "PostgresEmbeddingKnowledgeStore"),
        ("cayu.storage.postgres", "PostgresEventWatcherStore"),
        ("cayu.storage.postgres", "PostgresKnowledgeStore"),
        ("cayu.storage.postgres", "PostgresSessionStore"),
        ("cayu.storage.postgres", "PostgresTaskStore"),
    }
)
_SCHEMALESS_STORE_IDENTITIES = frozenset(
    {
        ("cayu.runtime.budgets", "InMemoryBudgetLedger"),
        ("cayu.runtime.budgets", "InMemoryBudgetStore"),
        ("cayu.runtime.budgets", "SessionBudgetStore"),
        ("cayu.runtime.event_watchers", "InMemoryEventWatcherStore"),
        ("cayu.runtime.sessions", "InMemorySessionStore"),
        ("cayu.runtime.tasks", "InMemoryTaskStore"),
        ("cayu.storage.memory", "InMemoryEmbeddingKnowledgeStore"),
        ("cayu.storage.memory", "InMemoryKnowledgeStore"),
    }
)

StoreSchemaReadiness: TypeAlias = Literal[
    "validated_compatible",
    "validation_failed",
    "not_applicable",
    "unavailable",
]


class _SupportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class CollectorDisposition(StrEnum):
    COLLECTED = "collected"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    REDACTED = "redacted"


class SupportBundleOutcome(StrEnum):
    CLEAN = "clean"
    PARTIAL = "partial"
    BOOT_FAILED = "boot_failed"
    VALIDATION_FAILED = "validation_failed"


class SupportBundleLimits(_SupportModel):
    collector_timeout_seconds: float = Field(gt=0, le=60)
    collection_timeout_seconds: float = Field(gt=0, le=120)
    worker_timeout_seconds: float = Field(gt=0, le=180)
    publication_timeout_seconds: float = Field(gt=0, le=60)
    reconciliation_timeout_seconds: float = Field(gt=0, le=60)
    command_timeout_seconds: float = Field(gt=0, le=300)
    max_items: StrictInt = Field(ge=1, le=1000)
    max_sessions: StrictInt = Field(ge=1, le=100)
    event_limit: StrictInt = Field(ge=1, le=_MAX_EVENT_LIMIT)
    event_query_bytes: StrictInt = Field(ge=1024, le=8 * 1024 * 1024)
    max_collector_bytes: StrictInt = Field(ge=1024, le=2 * 1024 * 1024)
    max_evidence_bytes: StrictInt = Field(ge=1024, le=8 * 1024 * 1024)
    max_bundle_bytes: StrictInt = Field(ge=4096, le=16 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_byte_limits(self) -> SupportBundleLimits:
        if self.max_evidence_bytes > self.max_bundle_bytes:
            raise ValueError("evidence byte limit cannot exceed bundle byte limit.")
        if self.worker_timeout_seconds >= self.command_timeout_seconds:
            raise ValueError("worker timeout must leave time for bundle publication.")
        if self.publication_timeout_seconds >= self.command_timeout_seconds:
            raise ValueError("publication timeout must fit inside the command timeout.")
        bounded_phases = (
            self.worker_timeout_seconds
            + self.publication_timeout_seconds
            + self.reconciliation_timeout_seconds
        )
        if bounded_phases >= self.command_timeout_seconds:
            raise ValueError("bounded phases must leave command teardown time.")
        return self


DEFAULT_SUPPORT_BUNDLE_LIMITS = SupportBundleLimits(
    collector_timeout_seconds=DEFAULT_COLLECTOR_TIMEOUT_SECONDS,
    collection_timeout_seconds=DEFAULT_COLLECTION_TIMEOUT_SECONDS,
    worker_timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
    publication_timeout_seconds=DEFAULT_PUBLICATION_TIMEOUT_SECONDS,
    reconciliation_timeout_seconds=DEFAULT_RECONCILIATION_TIMEOUT_SECONDS,
    command_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    max_items=DEFAULT_MAX_ITEMS,
    max_sessions=DEFAULT_MAX_SESSIONS,
    event_limit=DEFAULT_EVENT_LIMIT,
    event_query_bytes=DEFAULT_EVENT_QUERY_BYTES,
    max_collector_bytes=DEFAULT_MAX_COLLECTOR_BYTES,
    max_evidence_bytes=DEFAULT_MAX_EVIDENCE_BYTES,
    max_bundle_bytes=DEFAULT_MAX_BUNDLE_BYTES,
)


class RuntimeIdentityEvidence(_SupportModel):
    kind: Literal["runtime_identity"] = "runtime_identity"
    cayu_version: str
    python_version: str
    python_implementation: str
    operating_system: str
    machine: str


class ProjectIdentityEvidence(_SupportModel):
    kind: Literal["project_identity"] = "project_identity"
    project_id: str | None
    application_release_id: str
    manifest_fingerprint: str
    manifest_schema_version: str
    profile: Literal["production"] = "production"
    service_declared: bool


class CheckReportEvidence(_SupportModel):
    kind: Literal["check_report"] = "check_report"
    report: ProjectCheckReport


class ControlPlaneDiagnosticsEvidence(_SupportModel):
    kind: Literal["control_plane_diagnostics"] = "control_plane_diagnostics"
    report: SystemDiagnosticsResponse

    @field_validator("report")
    @classmethod
    def own_report(cls, value: SystemDiagnosticsResponse) -> SystemDiagnosticsResponse:
        if type(value) is not SystemDiagnosticsResponse:
            raise TypeError("report must be a SystemDiagnosticsResponse.")
        if value.capabilities.actor is not None:
            raise ValueError("support-bundle diagnostics cannot contain a request actor.")
        if value.artifact_stores.registrations:
            raise ValueError("support-bundle diagnostics cannot contain artifact store identities.")
        return SystemDiagnosticsResponse.model_validate_json(value.model_dump_json(warnings=False))


class BoundedInventory(_SupportModel):
    total_count: StrictInt = Field(ge=0)
    included_count: StrictInt = Field(ge=0)
    truncated: bool

    @model_validator(mode="after")
    def validate_counts(self) -> BoundedInventory:
        if self.included_count > self.total_count:
            raise ValueError("included_count cannot exceed total_count.")
        if self.truncated != (self.included_count < self.total_count):
            raise ValueError("truncated must match the inventory counts.")
        return self


class RecoveryCleanupPolicyEvidence(_SupportModel):
    step_timeout_seconds: StrictFloat = Field(
        gt=0,
        le=RECOVERY_CLEANUP_MAX_TIMEOUT_SECONDS,
    )
    overall_timeout_seconds: StrictFloat = Field(
        gt=0,
        le=RECOVERY_CLEANUP_MAX_TIMEOUT_SECONDS,
    )
    max_supervised_tasks: StrictInt = Field(ge=1, le=4096)


class RecoveryCleanupRetainedEvidence(_SupportModel):
    operation: str = Field(min_length=1, max_length=160)
    scope: RecoveryCleanupDeadlineScope
    timeout_seconds: StrictFloat = Field(
        gt=0,
        le=RECOVERY_CLEANUP_MAX_TIMEOUT_SECONDS,
    )
    outcome_unknown: Literal[True] = True
    caller_cancellation_observed: StrictBool


class RecoveryCleanupSnapshotEvidence(_SupportModel):
    active_tasks: StrictInt = Field(ge=0)
    retained_tasks: StrictInt = Field(ge=0)
    timed_out_steps: StrictInt = Field(ge=0)
    completed_after_timeout: StrictInt = Field(ge=0)
    failed_after_timeout: StrictInt = Field(ge=0)
    retained_after_cancellation: StrictInt = Field(ge=0)
    capacity_exhausted_steps: StrictInt = Field(ge=0)
    retained: tuple[RecoveryCleanupRetainedEvidence, ...]
    retained_inventory: BoundedInventory

    @model_validator(mode="after")
    def validate_retained_inventory(self) -> RecoveryCleanupSnapshotEvidence:
        if (
            self.retained_inventory.total_count != self.retained_tasks
            or self.retained_inventory.included_count != len(self.retained)
        ):
            raise ValueError("recovery cleanup retained inventory is inconsistent.")
        return self


class RecoveryCleanupEvidence(_SupportModel):
    kind: Literal["recovery_cleanup"] = "recovery_cleanup"
    policy: RecoveryCleanupPolicyEvidence
    snapshot: RecoveryCleanupSnapshotEvidence


class CapabilityEvidence(_SupportModel):
    name: str
    declared: bool
    resolved: bool


class ProviderRegistrationEvidence(_SupportModel):
    name: str
    implementation: str
    source: Literal["project", "built_in", "external", "dynamic", "unavailable"]
    is_default: bool


class AgentProviderResolutionEvidence(_SupportModel):
    agent_name: str
    configured_provider: str | None
    resolved_provider: str | None
    resolution: Literal["explicit", "model_pattern", "default", "missing", "ambiguous"]


class EnvironmentComponentEvidence(_SupportModel):
    name: str
    factory_backed: bool
    workspace: str | None
    workspace_branch_capabilities: WorkspaceBranchCapabilities
    workspace_branch_lifecycle: WorkspaceBranchLifecycleSummary
    runner: str | None
    artifact_store: str | None
    vault: str | None
    credential_proxy: str | None
    knowledge_store: str | None
    mcp_server_count: StrictInt = Field(ge=0)
    lifecycle_policy: EnvironmentLifecyclePolicy | None = None


class ManifestSummaryEvidence(_SupportModel):
    kind: Literal["manifest_summary"] = "manifest_summary"
    fingerprint: str
    schema_version: str
    agents: tuple[AgentProviderResolutionEvidence, ...]
    agent_inventory: BoundedInventory
    providers: tuple[ProviderRegistrationEvidence, ...]
    provider_inventory: BoundedInventory
    environments: tuple[EnvironmentComponentEvidence, ...]
    environment_inventory: BoundedInventory
    capabilities: tuple[CapabilityEvidence, ...]
    capability_inventory: BoundedInventory
    mcp_manifest_policy_configured: bool


class StoreDescriptorEvidence(_SupportModel):
    role: Literal[
        "session",
        "task",
        "knowledge",
        "budget",
        "budget_ledger",
        "event_watcher",
        "eval",
    ]
    implementation: str | None
    durability: Literal["development", "durable", "read_only", "unverified", "missing"]
    schema_readiness: StoreSchemaReadiness
    bounded_event_reads: bool | None = None


class StoreSummaryEvidence(_SupportModel):
    kind: Literal["store_summary"] = "store_summary"
    stores: tuple[StoreDescriptorEvidence, ...]
    eval_backend: Literal["sqlite", "postgres"] | None
    eval_source: str | None


class SessionOperationalEvidence(_SupportModel):
    kind: Literal["session_operational"] = "session_operational"
    snapshot: SessionOperationalSnapshot


class TaskOperationalEvidence(_SupportModel):
    kind: Literal["task_operational"] = "task_operational"
    snapshot: TaskOperationalSnapshot


class ArtifactAvailabilityEvidence(_SupportModel):
    kind: Literal["artifact_availability"] = "artifact_availability"
    registered: bool
    registration_count: StrictInt = Field(ge=0)
    availability: Literal["configured_only_not_live_verified"] = "configured_only_not_live_verified"

    @model_validator(mode="after")
    def validate_registration_count(self) -> ArtifactAvailabilityEvidence:
        if self.registered is not (self.registration_count > 0):
            raise ValueError("artifact registration state must match its count.")
        return self


class OptionalPackageEvidence(_SupportModel):
    distribution: str
    availability: Literal["installed", "not_installed"]
    version: str | None


class OptionalPackagesEvidence(_SupportModel):
    kind: Literal["optional_packages"] = "optional_packages"
    packages: tuple[OptionalPackageEvidence, ...]


class EventEnvelopeEvidence(_SupportModel):
    sequence: StrictInt = Field(ge=1)
    type: str
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware.")
        return value.astimezone(UTC)


class EnvironmentLifecycleSummaryEvidence(_SupportModel):
    sequence: StrictInt = Field(ge=1)
    timestamp: datetime
    progress: EnvironmentLifecycleProgress

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lifecycle progress timestamp must be timezone-aware.")
        return value.astimezone(UTC)


class SessionEventTailEvidence(_SupportModel):
    kind: Literal["session_event_tail"] = "session_event_tail"
    projection: Literal["redacted_envelope_only"] = "redacted_envelope_only"
    session_ordinal: StrictInt = Field(ge=1)
    returned_count: StrictInt = Field(ge=0)
    omitted_count_lower_bound: StrictInt = Field(ge=0)
    omitted_count_exact: bool
    tail_complete: bool
    first_sequence: StrictInt | None = Field(default=None, ge=1)
    last_sequence: StrictInt | None = Field(default=None, ge=1)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    events: tuple[EventEnvelopeEvidence, ...]
    lifecycle_progress: tuple[EnvironmentLifecycleSummaryEvidence, ...] = ()
    lifecycle_progress_inventory: BoundedInventory = Field(
        default_factory=lambda: BoundedInventory(
            total_count=0,
            included_count=0,
            truncated=False,
        )
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> SessionEventTailEvidence:
        if self.returned_count != len(self.events):
            raise ValueError("event tail returned_count must match events.")
        sequences = tuple(item.sequence for item in self.events)
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError("event tail sequences must be unique and ascending.")
        expected_first_sequence = None if not self.events else self.events[0].sequence
        expected_last_sequence = None if not self.events else self.events[-1].sequence
        expected_first_timestamp = None if not self.events else self.events[0].timestamp
        expected_last_timestamp = None if not self.events else self.events[-1].timestamp
        if (
            self.first_sequence != expected_first_sequence
            or self.last_sequence != expected_last_sequence
            or self.first_timestamp != expected_first_timestamp
            or self.last_timestamp != expected_last_timestamp
        ):
            raise ValueError("event tail bounds must match returned events.")
        if self.tail_complete:
            if self.omitted_count_lower_bound != 0 or not self.omitted_count_exact:
                raise ValueError("complete event tails cannot report omissions.")
        elif self.omitted_count_lower_bound < 1 or self.omitted_count_exact:
            raise ValueError("incomplete event tails require a non-exact omission lower bound.")
        lifecycle_sequences = tuple(item.sequence for item in self.lifecycle_progress)
        if lifecycle_sequences != tuple(sorted(set(lifecycle_sequences))):
            raise ValueError("lifecycle progress sequences must be unique and ascending.")
        if any(sequence not in sequences for sequence in lifecycle_sequences):
            raise ValueError("lifecycle progress must come from the returned event tail.")
        if self.lifecycle_progress_inventory.included_count != len(self.lifecycle_progress):
            raise ValueError("lifecycle progress inventory must match included summaries.")
        return self


SupportEvidence: TypeAlias = Annotated[
    RuntimeIdentityEvidence
    | ProjectIdentityEvidence
    | CheckReportEvidence
    | ControlPlaneDiagnosticsEvidence
    | RecoveryCleanupEvidence
    | ManifestSummaryEvidence
    | StoreSummaryEvidence
    | SessionOperationalEvidence
    | TaskOperationalEvidence
    | ArtifactAvailabilityEvidence
    | OptionalPackagesEvidence
    | SessionEventTailEvidence,
    Field(discriminator="kind"),
]
_SUPPORT_EVIDENCE_ADAPTER = TypeAdapter(SupportEvidence)


class SupportCollectorOutput(_SupportModel):
    disposition: Literal[
        CollectorDisposition.COLLECTED,
        CollectorDisposition.UNAVAILABLE,
        CollectorDisposition.SKIPPED,
    ] = CollectorDisposition.COLLECTED
    reason_code: str | None = None
    evidence: SupportEvidence | None = None

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        if value is not None and _REASON_CODE_PATTERN.fullmatch(value) is None:
            raise ValueError("reason_code is invalid.")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> SupportCollectorOutput:
        if self.disposition is CollectorDisposition.COLLECTED:
            if self.evidence is None or self.reason_code is not None:
                raise ValueError("collected output requires only evidence.")
        elif self.evidence is not None or self.reason_code is None:
            raise ValueError("non-collected output requires only a reason_code.")
        return self


class SupportCollectorResult(_SupportModel):
    name: str
    disposition: CollectorDisposition
    duration_ms: StrictInt = Field(ge=0, le=180_000)
    evidence_bytes: StrictInt = Field(ge=0, le=8 * 1024 * 1024)
    reason_code: str | None = None
    evidence: SupportEvidence | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if _COLLECTOR_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("collector name is invalid.")
        return value

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        if value is not None and _REASON_CODE_PATTERN.fullmatch(value) is None:
            raise ValueError("reason_code is invalid.")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> SupportCollectorResult:
        if self.disposition is CollectorDisposition.COLLECTED:
            if self.evidence is None or self.reason_code is not None:
                raise ValueError("collected result requires only evidence.")
            expected_bytes = len(
                _canonical_json_bytes(self.evidence.model_dump(mode="json", warnings=False))
            )
            if self.evidence_bytes != expected_bytes:
                raise ValueError("collected result byte evidence is inconsistent.")
        elif self.evidence is not None or self.reason_code is None or self.evidence_bytes != 0:
            raise ValueError("non-collected result requires only timing and a reason_code.")
        return self


def _results_require_partial(results: Sequence[SupportCollectorResult]) -> bool:
    return any(item.disposition is not CollectorDisposition.COLLECTED for item in results)


class SupportBundleReport(_SupportModel):
    schema_version: Literal["1"] = SUPPORT_BUNDLE_SCHEMA_VERSION
    command_version: Literal["1"] = SUPPORT_BUNDLE_COMMAND_VERSION
    bundle_id: str = Field(pattern=r"^bundle_[0-9a-f]{32}$")
    generated_at: datetime
    outcome: SupportBundleOutcome
    limits: SupportBundleLimits
    collection_duration_ms: StrictInt = Field(ge=0, le=180_000)
    collector_count: StrictInt = Field(ge=0, le=1100)
    collected_count: StrictInt = Field(ge=0, le=1100)
    omitted_count: StrictInt = Field(ge=0, le=1100)
    evidence_complete: bool
    total_evidence_bytes: StrictInt = Field(ge=0, le=8 * 1024 * 1024)
    collectors: tuple[SupportCollectorResult, ...]

    @classmethod
    def from_results(
        cls,
        *,
        generated_at: datetime,
        outcome: SupportBundleOutcome,
        limits: SupportBundleLimits,
        collectors: Sequence[SupportCollectorResult],
        collection_duration_ms: int,
        bundle_id: str | None = None,
    ) -> SupportBundleReport:
        results = tuple(
            SupportCollectorResult.model_validate_json(item.model_dump_json(warnings=False))
            for item in collectors
        )
        collected_count = sum(
            item.disposition is CollectorDisposition.COLLECTED for item in results
        )
        omitted_count = len(results) - collected_count
        return cls(
            bundle_id=bundle_id or f"bundle_{secrets.token_hex(16)}",
            generated_at=generated_at,
            outcome=outcome,
            limits=limits,
            collection_duration_ms=collection_duration_ms,
            collector_count=len(results),
            collected_count=collected_count,
            omitted_count=omitted_count,
            evidence_complete=omitted_count == 0,
            total_evidence_bytes=sum(item.evidence_bytes for item in results),
            collectors=results,
        )

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_outcome(self) -> SupportBundleReport:
        if len({item.name for item in self.collectors}) != len(self.collectors):
            raise ValueError("collector result names must be unique.")
        collected_count = sum(
            item.disposition is CollectorDisposition.COLLECTED for item in self.collectors
        )
        omitted_count = len(self.collectors) - collected_count
        if (
            self.collector_count != len(self.collectors)
            or self.collected_count != collected_count
            or self.omitted_count != omitted_count
            or self.evidence_complete is not (omitted_count == 0)
            or self.total_evidence_bytes != sum(item.evidence_bytes for item in self.collectors)
        ):
            raise ValueError("support bundle aggregate evidence is inconsistent.")
        if self.total_evidence_bytes > self.limits.max_evidence_bytes:
            raise ValueError("support bundle evidence exceeds its collection byte limit.")
        requires_partial = _results_require_partial(self.collectors)
        if self.outcome is SupportBundleOutcome.CLEAN and requires_partial:
            raise ValueError("clean outcome cannot contain non-collected results.")
        if self.outcome is SupportBundleOutcome.PARTIAL and not requires_partial:
            raise ValueError("partial outcome requires a non-collected result.")
        return self


@dataclass(frozen=True, slots=True)
class SupportBundleContext:
    app: CayuApp
    manifest: AppManifest
    check_report: ProjectCheckReport
    service_manifest: PublicServiceManifest | None
    project_id: str | None
    application_release_id: str
    eval_backend: Literal["sqlite", "postgres"] | None
    eval_source: str | None
    eval_store: EvalStore | None = None
    control_plane_diagnostics: SystemDiagnosticsResponse | None = None
    limits: SupportBundleLimits = DEFAULT_SUPPORT_BUNDLE_LIMITS


class SupportBundleCollector(Protocol):
    name: str

    async def collect(self, context: SupportBundleContext) -> SupportCollectorOutput:
        """Collect one bounded, read-only diagnostic projection."""


@dataclass(frozen=True, slots=True)
class FunctionalSupportBundleCollector:
    name: str
    function: Callable[[SupportBundleContext], Awaitable[SupportCollectorOutput]]

    def __post_init__(self) -> None:
        if _COLLECTOR_NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError("collector name is invalid.")

    async def collect(self, context: SupportBundleContext) -> SupportCollectorOutput:
        return await self.function(context)


async def _run_disposable_sync_step(
    function: Callable[[], _SyncStepResult],
) -> _SyncStepResult:
    """Run one read-only synchronous step without letting it pin the event loop."""

    loop = asyncio.get_running_loop()
    completed: asyncio.Future[tuple[bool, object]] = loop.create_future()

    def publish(outcome: tuple[bool, object]) -> None:
        if not completed.done():
            completed.set_result(outcome)

    def run() -> None:
        try:
            outcome: tuple[bool, object] = (True, function())
        except BaseException as exc:
            outcome = (False, exc)
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(publish, outcome)

    worker = threading.Thread(
        target=run,
        name="cayu-support-bundle-sync-step",
        daemon=True,
    )
    worker.start()
    succeeded, value = await completed
    if succeeded:
        return cast("_SyncStepResult", value)
    if not isinstance(value, BaseException):
        raise RuntimeError("synchronous collector step returned an invalid failure.")
    raise value


class _CollectorDeadlineElapsed(TimeoutError):
    """Raised when a framework-owned synchronous collector step exceeds its owner."""


async def _run_sync_step_before(
    deadline: float,
    function: Callable[[], _SyncStepResult],
) -> _SyncStepResult:
    timeout_scope = asyncio.timeout_at(deadline)
    try:
        async with timeout_scope:
            return await _run_disposable_sync_step(function)
    except TimeoutError:
        if timeout_scope.expired():
            raise _CollectorDeadlineElapsed from None
        raise


@dataclass(frozen=True, slots=True)
class _PreparedCollectorResult:
    result: SupportCollectorResult
    accepted_evidence_bytes: int = 0
    byte_limit_reached: bool = False


def _prepare_collected_result(
    *,
    context: SupportBundleContext,
    name: str,
    evidence: SupportEvidence,
    collector_started: float,
    total_evidence_bytes: int,
) -> _PreparedCollectorResult:
    def failed(
        disposition: CollectorDisposition,
        reason_code: str,
        *,
        byte_limit_reached: bool = False,
    ) -> _PreparedCollectorResult:
        return _PreparedCollectorResult(
            result=_failed_result(
                name,
                disposition,
                reason_code,
                duration_ms=_elapsed_milliseconds(time.monotonic() - collector_started),
            ),
            byte_limit_reached=byte_limit_reached,
        )

    try:
        dumped = evidence.model_dump(mode="json", warnings=False)
        encoded = _canonical_json_bytes(dumped)
    except Exception:
        return failed(CollectorDisposition.FAILED, "invalid_collector_evidence")
    try:
        redacted = context.app.redact_json(dumped)
        redacted_encoded = _canonical_json_bytes(redacted)
    except Exception:
        return failed(CollectorDisposition.REDACTED, "application_redaction_failed")
    if redacted_encoded != encoded:
        return failed(
            CollectorDisposition.REDACTED,
            "application_redaction_changed_evidence",
        )
    try:
        _validate_forbidden_content(redacted)
    except Exception:
        return failed(
            CollectorDisposition.REDACTED,
            "collector_evidence_forbidden_content",
        )
    evidence_bytes = len(encoded)
    if evidence_bytes > context.limits.max_collector_bytes:
        return failed(CollectorDisposition.FAILED, "collector_result_too_large")
    if total_evidence_bytes + evidence_bytes > context.limits.max_evidence_bytes:
        return failed(
            CollectorDisposition.SKIPPED,
            "bundle_evidence_byte_limit_reached",
            byte_limit_reached=True,
        )
    try:
        owned_evidence = _SUPPORT_EVIDENCE_ADAPTER.validate_json(encoded)
        result = SupportCollectorResult(
            name=name,
            disposition=CollectorDisposition.COLLECTED,
            duration_ms=_elapsed_milliseconds(time.monotonic() - collector_started),
            evidence_bytes=evidence_bytes,
            evidence=owned_evidence,
        )
    except Exception:
        return failed(CollectorDisposition.FAILED, "invalid_collector_evidence")
    return _PreparedCollectorResult(
        result=result,
        accepted_evidence_bytes=evidence_bytes,
    )


def collected(evidence: SupportEvidence) -> SupportCollectorOutput:
    return SupportCollectorOutput(evidence=evidence)


def unavailable(reason_code: str) -> SupportCollectorOutput:
    return SupportCollectorOutput(
        disposition=CollectorDisposition.UNAVAILABLE,
        reason_code=reason_code,
    )


def skipped(reason_code: str) -> SupportCollectorOutput:
    return SupportCollectorOutput(
        disposition=CollectorDisposition.SKIPPED,
        reason_code=reason_code,
    )


async def collect_support_bundle(
    context: SupportBundleContext,
    collectors: Sequence[SupportBundleCollector],
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SupportBundleReport:
    """Run collectors sequentially with independent and aggregate deadlines."""

    loop = asyncio.get_running_loop()
    collection_started = loop.time()
    deadline = collection_started + context.limits.collection_timeout_seconds
    results: list[SupportCollectorResult] = []
    seen: set[str] = set()
    total_evidence_bytes = 0
    byte_limit_reached = False
    for collector in tuple(collectors):
        name = collector.name
        if _COLLECTOR_NAME_PATTERN.fullmatch(name) is None or name in seen:
            raise ValueError("collector names must be valid and unique.")
        seen.add(name)
        if byte_limit_reached:
            results.append(
                _failed_result(
                    name,
                    CollectorDisposition.SKIPPED,
                    "bundle_evidence_byte_limit_reached",
                    duration_ms=0,
                )
            )
            continue
        remaining = deadline - loop.time()
        if remaining <= 0:
            results.append(
                _failed_result(
                    name,
                    CollectorDisposition.SKIPPED,
                    "collection_deadline_elapsed",
                    duration_ms=0,
                )
            )
            continue
        collector_started = loop.time()
        collector_started_monotonic = time.monotonic()
        collector_deadline = min(
            deadline,
            collector_started + context.limits.collector_timeout_seconds,
        )
        owner_task = asyncio.current_task()
        cancellation_count_at_start = 0 if owner_task is None else owner_task.cancelling()
        timeout_scope = asyncio.timeout_at(collector_deadline)

        def collector_deadline_elapsed(
            owned_timeout: asyncio.Timeout = timeout_scope,
            deadline_at: float = collector_deadline,
        ) -> bool:
            return owned_timeout.expired() or loop.time() >= deadline_at

        def record_collector_result(
            disposition: CollectorDisposition,
            reason_code: str,
            collector_name: str = name,
            started_at: float = collector_started,
        ) -> CollectorDisposition:
            if collector_deadline_elapsed():
                disposition = CollectorDisposition.TIMED_OUT
                reason_code = "collector_deadline_elapsed"
            results.append(
                _failed_result(
                    collector_name,
                    disposition,
                    reason_code,
                    duration_ms=_elapsed_milliseconds(loop.time() - started_at),
                )
            )
            return disposition

        try:
            async with timeout_scope:
                output = await collector.collect(context)
        except TimeoutError:
            record_collector_result(CollectorDisposition.FAILED, "collector_failed")
            continue
        except asyncio.CancelledError:
            current_cancellation_count = 0 if owner_task is None else owner_task.cancelling()
            if current_cancellation_count > cancellation_count_at_start:
                raise
            record_collector_result(
                CollectorDisposition.FAILED,
                "collector_cancelled_without_task_cancellation",
            )
            continue
        except NotImplementedError:
            record_collector_result(
                CollectorDisposition.UNAVAILABLE,
                "collector_not_supported",
            )
            continue
        except Exception:
            record_collector_result(CollectorDisposition.FAILED, "collector_failed")
            continue
        if collector_deadline_elapsed():
            record_collector_result(
                CollectorDisposition.TIMED_OUT,
                "collector_deadline_elapsed",
            )
            continue
        if type(output) is not SupportCollectorOutput:
            record_collector_result(CollectorDisposition.FAILED, "invalid_collector_output")
            continue
        if output.disposition is not CollectorDisposition.COLLECTED:
            record_collector_result(
                output.disposition,
                output.reason_code or "collector_failed",
            )
            continue
        evidence = output.evidence
        if evidence is None:
            record_collector_result(
                CollectorDisposition.FAILED,
                "missing_collector_evidence",
            )
            continue
        try:
            prepared = await _run_sync_step_before(
                collector_deadline,
                partial(
                    _prepare_collected_result,
                    context=context,
                    name=name,
                    evidence=evidence,
                    collector_started=collector_started_monotonic,
                    total_evidence_bytes=total_evidence_bytes,
                ),
            )
        except _CollectorDeadlineElapsed:
            record_collector_result(
                CollectorDisposition.TIMED_OUT,
                "collector_deadline_elapsed",
            )
            continue
        except asyncio.CancelledError:
            current_cancellation_count = 0 if owner_task is None else owner_task.cancelling()
            if current_cancellation_count > cancellation_count_at_start:
                raise
            record_collector_result(
                CollectorDisposition.FAILED,
                "collector_cancelled_without_task_cancellation",
            )
            continue
        except Exception:
            record_collector_result(
                CollectorDisposition.FAILED,
                "invalid_collector_evidence",
            )
            continue
        if collector_deadline_elapsed():
            record_collector_result(
                CollectorDisposition.TIMED_OUT,
                "collector_deadline_elapsed",
            )
            continue
        results.append(prepared.result)
        total_evidence_bytes += prepared.accepted_evidence_bytes
        byte_limit_reached = prepared.byte_limit_reached
    requires_partial = _results_require_partial(results)
    return SupportBundleReport.from_results(
        generated_at=_normalized_now(now),
        outcome=(SupportBundleOutcome.PARTIAL if requires_partial else SupportBundleOutcome.CLEAN),
        limits=context.limits,
        collection_duration_ms=_elapsed_milliseconds(loop.time() - collection_started),
        collectors=results,
    )


def _failed_result(
    name: str,
    disposition: CollectorDisposition,
    reason_code: str,
    *,
    duration_ms: int,
) -> SupportCollectorResult:
    return SupportCollectorResult(
        name=name,
        disposition=disposition,
        duration_ms=duration_ms,
        evidence_bytes=0,
        reason_code=reason_code,
    )


def _elapsed_milliseconds(elapsed_seconds: float) -> int:
    return min(180_000, max(0, round(elapsed_seconds * 1000)))


def _normalized_now(now: Callable[[], datetime]) -> datetime:
    value = now()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("support-bundle clock must return a timezone-aware datetime.")
    return value.astimezone(UTC)


def builtin_support_collectors(
    *,
    session_selectors: Sequence[str] = (),
) -> tuple[SupportBundleCollector, ...]:
    collectors: list[SupportBundleCollector] = [
        FunctionalSupportBundleCollector("runtime_identity", _collect_runtime_identity),
        FunctionalSupportBundleCollector("project_identity", _collect_project_identity),
        FunctionalSupportBundleCollector("check", _collect_check),
        FunctionalSupportBundleCollector("control_plane", _collect_control_plane),
        FunctionalSupportBundleCollector("manifest", _collect_manifest),
        FunctionalSupportBundleCollector("recovery_cleanup", _collect_recovery_cleanup),
        FunctionalSupportBundleCollector("stores", _collect_stores),
        FunctionalSupportBundleCollector("sessions", _collect_session_operational),
        FunctionalSupportBundleCollector("tasks", _collect_task_operational),
        FunctionalSupportBundleCollector("artifacts", _collect_artifacts),
        FunctionalSupportBundleCollector("optional_packages", _collect_optional_packages),
    ]
    for index, selector in enumerate(tuple(session_selectors), start=1):
        collectors.append(
            FunctionalSupportBundleCollector(
                f"session_events.{index}",
                _event_tail_function(selector, index),
            )
        )
    return tuple(collectors)


async def _collect_runtime_identity(_context: SupportBundleContext) -> SupportCollectorOutput:
    return await _run_disposable_sync_step(
        lambda: collected(
            RuntimeIdentityEvidence(
                cayu_version=package_version(),
                python_version=platform.python_version(),
                python_implementation=sys.implementation.name,
                operating_system=platform.system(),
                machine=platform.machine(),
            )
        )
    )


async def _collect_project_identity(context: SupportBundleContext) -> SupportCollectorOutput:
    return collected(
        ProjectIdentityEvidence(
            project_id=context.project_id,
            application_release_id=context.application_release_id,
            manifest_fingerprint=context.manifest.fingerprint,
            manifest_schema_version=context.manifest.schema_version,
            service_declared=context.service_manifest is not None,
        )
    )


async def _collect_check(context: SupportBundleContext) -> SupportCollectorOutput:
    return collected(CheckReportEvidence(report=context.check_report))


async def _collect_control_plane(context: SupportBundleContext) -> SupportCollectorOutput:
    if context.control_plane_diagnostics is None:
        return unavailable("maintained_service_not_selected")
    return collected(ControlPlaneDiagnosticsEvidence(report=context.control_plane_diagnostics))


async def _collect_recovery_cleanup(
    context: SupportBundleContext,
) -> SupportCollectorOutput:
    snapshot = context.app.recovery_cleanup_status()
    if type(snapshot) is not RecoveryCleanupSupervisorSnapshot:
        raise TypeError(
            "recovery_cleanup_status() must return a RecoveryCleanupSupervisorSnapshot."
        )
    policy = context.manifest.runtime.recovery_cleanup_policy
    retained = tuple(
        RecoveryCleanupRetainedEvidence(
            operation=item.operation,
            scope=item.scope,
            timeout_seconds=item.timeout_seconds,
            outcome_unknown=item.outcome_unknown,
            caller_cancellation_observed=item.caller_cancellation_observed,
        )
        for item in snapshot.retained[: context.limits.max_items]
    )
    return collected(
        RecoveryCleanupEvidence(
            policy=RecoveryCleanupPolicyEvidence(
                step_timeout_seconds=policy.step_timeout_seconds,
                overall_timeout_seconds=policy.overall_timeout_seconds,
                max_supervised_tasks=policy.max_supervised_tasks,
            ),
            snapshot=RecoveryCleanupSnapshotEvidence(
                active_tasks=snapshot.active_tasks,
                retained_tasks=snapshot.retained_tasks,
                timed_out_steps=snapshot.timed_out_steps,
                completed_after_timeout=snapshot.completed_after_timeout,
                failed_after_timeout=snapshot.failed_after_timeout,
                retained_after_cancellation=snapshot.retained_after_cancellation,
                capacity_exhausted_steps=snapshot.capacity_exhausted_steps,
                retained=retained,
                retained_inventory=_inventory(snapshot.retained_tasks, len(retained)),
            ),
        )
    )


def _inventory(total: int, included: int) -> BoundedInventory:
    return BoundedInventory(
        total_count=total,
        included_count=included,
        truncated=included < total,
    )


def _has_exact_builtin_type(
    value: object,
    identities: frozenset[tuple[str, str]],
) -> bool:
    cls = type(value)
    identity = (cls.__module__, cls.__name__)
    if identity not in identities:
        return False
    module = sys.modules.get(cls.__module__)
    return module is not None and getattr(module, cls.__name__, None) is cls


async def _store_schema_readiness(value: object | None) -> StoreSchemaReadiness:
    if value is None:
        return "not_applicable"
    if _has_exact_builtin_type(value, _SCHEMALESS_STORE_IDENTITIES):
        return "not_applicable"
    if _has_exact_builtin_type(value, _SQLITE_SCHEMA_STORE_IDENTITIES):
        if getattr(value, "_diagnostic_source_missing", None) is True:
            return "unavailable"
        path = getattr(value, "path", None)
        if type(path) is not type(Path()):
            return "unavailable"
        if path == Path(":memory:"):
            return "not_applicable"
        # Built-in SQLite stores reconcile and validate synchronously before
        # their constructors return. Diagnostic inspection forces file-backed
        # stores through the non-mutating VALIDATE path.
        return "validated_compatible"
    if not _has_exact_builtin_type(value, _POSTGRES_SCHEMA_STORE_IDENTITIES):
        return "unavailable"
    # PostgreSQL schema setup is lazy. Only invoke it when the diagnostic owner
    # has positively forced this exact built-in store into read-only validation.
    if getattr(value, "_read_only", None) is not True:
        return "unavailable"
    ensure_schema = getattr(type(value), "ensure_schema", None)
    if not callable(ensure_schema):
        return "unavailable"
    try:
        await ensure_schema(value)
    except asyncio.CancelledError:
        raise
    except (GeneratorExit, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return "validation_failed"
    return "validated_compatible"


async def _collect_manifest(context: SupportBundleContext) -> SupportCollectorOutput:
    manifest = context.manifest
    limit = context.limits.max_items
    agents = tuple(
        AgentProviderResolutionEvidence(
            agent_name=item.name,
            configured_provider=item.configured_provider,
            resolved_provider=item.resolved_provider,
            resolution=item.provider_resolution,
        )
        for item in manifest.agents[:limit]
    )
    providers = tuple(
        ProviderRegistrationEvidence(
            name=item.name,
            implementation=item.implementation,
            source=item.registration_provenance.kind,
            is_default=item.is_default,
        )
        for item in manifest.providers[:limit]
    )
    environments = tuple(
        EnvironmentComponentEvidence(
            name=item.name,
            factory_backed=item.factory_backed,
            workspace=item.workspace,
            workspace_branch_capabilities=WorkspaceBranchCapabilities.model_validate(
                dict(item.workspace_branch_capabilities)
            ),
            workspace_branch_lifecycle=WorkspaceBranchLifecycleSummary.model_validate(
                dict(item.workspace_branch_lifecycle)
            ),
            runner=item.runner,
            artifact_store=item.artifact_store,
            vault=item.vault,
            credential_proxy=item.credential_proxy,
            knowledge_store=item.knowledge_store,
            mcp_server_count=len(item.mcp_servers),
            lifecycle_policy=(
                None
                if item.lifecycle_policy is None
                else EnvironmentLifecyclePolicy.model_validate(dict(item.lifecycle_policy))
            ),
        )
        for item in manifest.environments[:limit]
    )
    capabilities = tuple(
        CapabilityEvidence(
            name=item.name,
            declared=item.declared,
            resolved=item.resolved,
        )
        for item in manifest.capabilities[:limit]
    )
    return collected(
        ManifestSummaryEvidence(
            fingerprint=manifest.fingerprint,
            schema_version=manifest.schema_version,
            agents=agents,
            agent_inventory=_inventory(len(manifest.agents), len(agents)),
            providers=providers,
            provider_inventory=_inventory(len(manifest.providers), len(providers)),
            environments=environments,
            environment_inventory=_inventory(len(manifest.environments), len(environments)),
            capabilities=capabilities,
            capability_inventory=_inventory(len(manifest.capabilities), len(capabilities)),
            mcp_manifest_policy_configured=manifest.runtime.mcp_manifest_policy is not None,
        )
    )


async def _collect_stores(context: SupportBundleContext) -> SupportCollectorOutput:
    app = context.app
    manifest = context.manifest
    session_store = app.session_store
    task_store = app.task_store
    store_values = (
        session_store,
        task_store,
        app.knowledge_store,
        app.budget_store,
        app.budget_ledger,
        app.event_watcher_store,
        context.eval_store,
    )
    readiness = await asyncio.gather(*(_store_schema_readiness(value) for value in store_values))
    descriptors = (
        StoreDescriptorEvidence(
            role="session",
            implementation=manifest.stores.session,
            durability=session_store.service_durability.value,
            schema_readiness=readiness[0],
            bounded_event_reads=(
                type(session_store).query_events_bounded is not SessionStore.query_events_bounded
            ),
        ),
        StoreDescriptorEvidence(
            role="task",
            implementation=manifest.stores.task,
            durability=("missing" if task_store is None else task_store.service_durability.value),
            schema_readiness=readiness[1],
        ),
        StoreDescriptorEvidence(
            role="knowledge",
            implementation=manifest.stores.knowledge,
            durability="missing" if app.knowledge_store is None else "unverified",
            schema_readiness=readiness[2],
        ),
        StoreDescriptorEvidence(
            role="budget",
            implementation=manifest.stores.budget,
            durability="unverified",
            schema_readiness=readiness[3],
        ),
        StoreDescriptorEvidence(
            role="budget_ledger",
            implementation=manifest.stores.budget_ledger,
            durability="unverified",
            schema_readiness=readiness[4],
        ),
        StoreDescriptorEvidence(
            role="event_watcher",
            implementation=manifest.stores.event_watcher,
            durability="unverified",
            schema_readiness=readiness[5],
        ),
        StoreDescriptorEvidence(
            role="eval",
            implementation=context.eval_backend,
            durability="missing" if context.eval_backend is None else "durable",
            schema_readiness=readiness[6],
        ),
    )
    return collected(
        StoreSummaryEvidence(
            stores=descriptors,
            eval_backend=context.eval_backend,
            eval_source=context.eval_source,
        )
    )


async def _collect_session_operational(
    context: SupportBundleContext,
) -> SupportCollectorOutput:
    if (
        _has_exact_builtin_type(context.app.session_store, _SQLITE_SCHEMA_STORE_IDENTITIES)
        and getattr(context.app.session_store, "_diagnostic_source_missing", None) is True
    ):
        return unavailable("store_source_not_available")
    snapshot = await context.app.session_store.aggregate_operational_snapshot()
    return collected(SessionOperationalEvidence(snapshot=snapshot))


async def _collect_task_operational(context: SupportBundleContext) -> SupportCollectorOutput:
    if context.app.task_store is None:
        return unavailable("task_store_not_configured")
    if (
        _has_exact_builtin_type(context.app.task_store, _SQLITE_SCHEMA_STORE_IDENTITIES)
        and getattr(context.app.task_store, "_diagnostic_source_missing", None) is True
    ):
        return unavailable("store_source_not_available")
    snapshot = await context.app.task_store.aggregate_operational_snapshot()
    return collected(TaskOperationalEvidence(snapshot=snapshot))


async def _collect_artifacts(context: SupportBundleContext) -> SupportCollectorOutput:
    total = context.app.artifact_store_registration_count()
    return collected(
        ArtifactAvailabilityEvidence(
            registered=total > 0,
            registration_count=total,
        )
    )


async def _collect_optional_packages(
    _context: SupportBundleContext,
) -> SupportCollectorOutput:
    packages: list[OptionalPackageEvidence] = []
    for distribution in _OPTIONAL_DISTRIBUTIONS:
        try:
            installed_version = await _run_disposable_sync_step(
                lambda distribution=distribution: metadata.version(distribution)
            )
        except metadata.PackageNotFoundError:
            packages.append(
                OptionalPackageEvidence(
                    distribution=distribution,
                    availability="not_installed",
                    version=None,
                )
            )
        else:
            packages.append(
                OptionalPackageEvidence(
                    distribution=distribution,
                    availability="installed",
                    version=installed_version,
                )
            )
    return collected(OptionalPackagesEvidence(packages=tuple(packages)))


def _event_tail_function(
    selector: str,
    ordinal: int,
) -> Callable[[SupportBundleContext], Awaitable[SupportCollectorOutput]]:
    async def collect_event_tail(context: SupportBundleContext) -> SupportCollectorOutput:
        return await _collect_event_tail(context, selector=selector, ordinal=ordinal)

    return collect_event_tail


async def _collect_event_tail(
    context: SupportBundleContext,
    *,
    selector: str,
    ordinal: int,
) -> SupportCollectorOutput:
    private_session_id = await context.app._resolve_public_session_id(selector)
    state = await context.app.session_store.load_state(private_session_id)
    if state is None:
        return unavailable("session_not_found")
    query_limit = context.limits.event_limit + 1
    try:
        records = await context.app.session_store.query_events_bounded(
            EventQuery(
                session_id=private_session_id,
                limit=query_limit,
                order_by=EventOrder.SEQUENCE_DESC,
            ),
            max_bytes=context.limits.event_query_bytes,
        )
    except EventQueryResultTooLarge:
        return unavailable("event_tail_exceeds_byte_limit")
    truncated = len(records) > context.limits.event_limit
    selected = records[: context.limits.event_limit]
    envelopes: list[EventEnvelopeEvidence] = []
    lifecycle_progress: list[EnvironmentLifecycleSummaryEvidence] = []
    for record in reversed(selected):
        projected = context.app._project_persisted_event_record_for_exposure(record)
        event_type = str(projected.event.type)
        if event_type not in _KNOWN_EVENT_TYPE_VALUES:
            event_type = _REDACTED_CUSTOM_EVENT_TYPE
        envelopes.append(
            EventEnvelopeEvidence(
                sequence=projected.sequence,
                type=event_type,
                timestamp=projected.event.timestamp,
            )
        )
        if projected.event.type == EventType.ENVIRONMENT_LIFECYCLE_PROGRESS:
            lifecycle_progress.append(
                EnvironmentLifecycleSummaryEvidence(
                    sequence=projected.sequence,
                    timestamp=projected.event.timestamp,
                    progress=environment_lifecycle_progress_from_event(projected.event),
                )
            )
    lifecycle_progress_total = len(lifecycle_progress)
    lifecycle_progress = lifecycle_progress[-context.limits.max_items :]
    return collected(
        SessionEventTailEvidence(
            session_ordinal=ordinal,
            returned_count=len(envelopes),
            omitted_count_lower_bound=1 if truncated else 0,
            omitted_count_exact=not truncated,
            tail_complete=not truncated,
            first_sequence=None if not envelopes else envelopes[0].sequence,
            last_sequence=None if not envelopes else envelopes[-1].sequence,
            first_timestamp=None if not envelopes else envelopes[0].timestamp,
            last_timestamp=None if not envelopes else envelopes[-1].timestamp,
            events=tuple(envelopes),
            lifecycle_progress=tuple(lifecycle_progress),
            lifecycle_progress_inventory=_inventory(
                lifecycle_progress_total,
                len(lifecycle_progress),
            ),
        )
    )


def minimal_support_bundle_report(
    *,
    outcome: Literal[
        SupportBundleOutcome.BOOT_FAILED,
        SupportBundleOutcome.VALIDATION_FAILED,
    ],
    reason_code: str,
    limits: SupportBundleLimits = DEFAULT_SUPPORT_BUNDLE_LIMITS,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SupportBundleReport:
    disposition = (
        CollectorDisposition.FAILED
        if outcome is SupportBundleOutcome.BOOT_FAILED
        else CollectorDisposition.REDACTED
    )
    return SupportBundleReport.from_results(
        generated_at=_normalized_now(now),
        outcome=outcome,
        limits=limits,
        collection_duration_ms=0,
        collectors=(
            SupportCollectorResult(
                name="bootstrap" if outcome is SupportBundleOutcome.BOOT_FAILED else "validation",
                disposition=disposition,
                duration_ms=0,
                evidence_bytes=0,
                reason_code=reason_code,
            ),
        ),
    )


def encode_support_bundle(report: SupportBundleReport) -> bytes:
    report_bytes = _canonical_json_bytes(
        report.model_dump(mode="json", warnings=False),
        pretty=True,
    )
    summary_bytes = render_support_bundle_summary(report).encode("utf-8")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        _write_archive_member(archive, SUPPORT_BUNDLE_REPORT_MEMBER, report_bytes)
        _write_archive_member(archive, SUPPORT_BUNDLE_SUMMARY_MEMBER, summary_bytes)
    encoded = stream.getvalue()
    validate_support_bundle_archive(encoded, max_bytes=report.limits.max_bundle_bytes)
    return encoded


def _write_archive_member(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
) -> None:
    member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = zipfile.ZIP_STORED
    member.create_system = 3
    member.external_attr = (stat.S_IFREG | 0o600) << 16
    archive.writestr(member, payload)


def validate_support_bundle_archive(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
) -> SupportBundleReport:
    if type(payload) is not bytes:
        raise TypeError("support bundle payload must be bytes.")
    if len(payload) > max_bytes:
        raise ValueError("support bundle exceeds its byte limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            if archive.comment:
                raise ValueError("support bundle archive metadata is unsafe.")
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                len(names) != len(SUPPORT_BUNDLE_ALLOWED_MEMBERS)
                or set(names) != SUPPORT_BUNDLE_ALLOWED_MEMBERS
                or len(set(names)) != len(names)
            ):
                raise ValueError("support bundle member set is invalid.")
            for member in members:
                member_mode = member.external_attr >> 16
                if (
                    member.is_dir()
                    or member.flag_bits != 0
                    or member.compress_type != zipfile.ZIP_STORED
                    or member.create_system != 3
                    or not stat.S_ISREG(member_mode)
                    or stat.S_IMODE(member_mode) != 0o600
                    or bool(member.comment)
                    or bool(member.extra)
                    or member.file_size > max_bytes
                    or member.compress_size > max_bytes
                    or PurePosixPath(member.filename).is_absolute()
                    or ".." in PurePosixPath(member.filename).parts
                ):
                    raise ValueError("support bundle member is unsafe.")
            report_payload = archive.read(SUPPORT_BUNDLE_REPORT_MEMBER)
            summary_payload = archive.read(SUPPORT_BUNDLE_SUMMARY_MEMBER)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("support bundle archive is invalid.") from exc
    if len(report_payload) + len(summary_payload) > max_bytes:
        raise ValueError("support bundle contents exceed their byte limit.")
    try:
        document = json.loads(report_payload)
        report = SupportBundleReport.model_validate_json(report_payload)
    except Exception as exc:
        raise ValueError("support bundle report is invalid.") from exc
    expected_report_payload = _canonical_json_bytes(
        report.model_dump(mode="json", warnings=False),
        pretty=True,
    )
    if report_payload != expected_report_payload:
        raise ValueError("support bundle report is not canonical.")
    if report.limits.max_bundle_bytes != max_bytes:
        raise ValueError("support bundle limit evidence does not match validation.")
    _validate_forbidden_content(document)
    expected_summary = render_support_bundle_summary(report).encode("utf-8")
    if summary_payload != expected_summary:
        raise ValueError("support bundle summary does not match its report.")
    return report


def render_support_bundle_summary(report: SupportBundleReport) -> str:
    lines = [
        "Cayu diagnostic support bundle",
        f"schema: {report.schema_version}",
        f"command: {report.command_version}",
        f"bundle: {report.bundle_id}",
        f"outcome: {report.outcome.value}",
        f"collection duration: {report.collection_duration_ms} ms",
        (
            "collector evidence: "
            f"{report.collected_count}/{report.collector_count} collected, "
            f"{report.omitted_count} omitted, {report.total_evidence_bytes} bytes"
        ),
        "collectors:",
    ]
    for item in report.collectors:
        suffix = "" if item.reason_code is None else f" ({item.reason_code})"
        lines.append(
            f"- {item.name}: {item.disposition.value}{suffix}; "
            f"{item.duration_ms} ms, {item.evidence_bytes} bytes"
        )
    return "\n".join(lines) + "\n"


def _validate_forbidden_content(value: object, *, path: tuple[str, ...] = ()) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("support bundle object keys must be strings.")
            normalized_key = key.lower().replace("-", "_")
            if normalized_key in _FORBIDDEN_KEYS:
                raise ValueError("support bundle contains a forbidden field.")
            _validate_forbidden_content(item, path=(*path, key))
        return
    if type(value) is list:
        for item in value:
            _validate_forbidden_content(item, path=path)
        return
    if type(value) is str:
        if "\x00" in value:
            raise ValueError("support bundle contains NUL text.")
        if (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or _EMBEDDED_ABSOLUTE_PATH_PATTERN.search(value)
        ):
            raise ValueError("support bundle contains an absolute path.")
        lowered = value.lower()
        if "://" in lowered or lowered.startswith(("postgresql:", "sqlite:", "file:")):
            raise ValueError("support bundle contains a URI or DSN.")


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        return (text + "\n").encode("utf-8")
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def support_bundle_staging_name(path: str | os.PathLike[str]) -> str:
    """Return one opaque publication name for an owning supervisor."""

    destination = Path(path)
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError("support bundle output must name a file.")
    return f".{destination.name}.cayu-doctor-{secrets.token_hex(12)}.tmp"


def _validated_staging_name(value: str, *, destination_name: str) -> str:
    prefix = f".{destination_name}.cayu-doctor-"
    token = value[len(prefix) : -len(".tmp")] if type(value) is str else ""
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or not value.endswith(".tmp")
        or re.fullmatch(r"[0-9a-f]{24}", token) is None
        or PurePosixPath(value).name != value
        or PureWindowsPath(value).name != value
    ):
        raise ValueError("support bundle staging name is invalid.")
    return value


def write_support_bundle_atomic(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    _temporary_name: str | None = None,
) -> None:
    """Publish an already-validated bundle atomically with private permissions."""

    report = validate_support_bundle_archive(payload)
    del report
    destination = Path(path)
    if not destination.name or destination.name in {".", ".."}:
        raise OSError("support bundle output must name a file.")
    absolute = destination.absolute()
    parent = absolute.parent
    temporary_name = (
        None
        if _temporary_name is None
        else _validated_staging_name(_temporary_name, destination_name=absolute.name)
    )
    if os.name == "posix":
        _write_support_bundle_posix(
            parent,
            absolute.name,
            payload,
            requested_temporary_name=temporary_name,
        )
    elif os.name == "nt":
        _write_support_bundle_windows(
            parent,
            absolute.name,
            payload,
            requested_temporary_name=temporary_name,
        )
    else:
        raise OSError("secure support bundle publication is unavailable.")


def _validate_existing_leaf(name: str, *, directory_fd: int) -> None:
    try:
        evidence = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(evidence.st_mode) or not stat.S_ISREG(evidence.st_mode):
        raise OSError("support bundle output must be a regular file or absent.")


def _validate_published_leaf(
    name: str,
    *,
    expected_payload: bytes,
    expected_identity: os.stat_result,
    directory_fd: int,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        path_evidence = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        opened_evidence = os.fstat(descriptor)
        if (
            stat.S_ISLNK(path_evidence.st_mode)
            or not stat.S_ISREG(path_evidence.st_mode)
            or not stat.S_ISREG(opened_evidence.st_mode)
            or stat.S_IMODE(path_evidence.st_mode) != 0o600
            or stat.S_IMODE(opened_evidence.st_mode) != 0o600
            or path_evidence.st_size != len(expected_payload)
            or opened_evidence.st_size != len(expected_payload)
            or not os.path.samestat(expected_identity, path_evidence)
            or not os.path.samestat(expected_identity, opened_evidence)
        ):
            raise OSError(
                "published support bundle failed identity, regular-file, mode, or size validation."
            )
        chunks: list[bytes] = []
        remaining = len(expected_payload) + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        published_payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if published_payload != expected_payload:
        raise OSError("published support bundle content changed before validation.")
    validate_support_bundle_archive(published_payload)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("support bundle write made no progress.")
        offset += written


def _open_parent_directory_posix(parent: Path) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("secure support bundle directory traversal is unavailable.")
    resolved_parent = parent.resolve(strict=True)
    flags = os.O_RDONLY | directory | getattr(os, "O_CLOEXEC", 0) | no_follow
    directory_fd = os.open(resolved_parent.anchor, flags)
    try:
        for part in resolved_parent.parts[1:]:
            child_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _write_support_bundle_posix(
    parent: Path,
    name: str,
    payload: bytes,
    *,
    requested_temporary_name: str | None,
) -> None:
    directory_fd = _open_parent_directory_posix(parent)
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        _validate_existing_leaf(name, directory_fd=directory_fd)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for _attempt in range(1 if requested_temporary_name is not None else 32):
            candidate = requested_temporary_name or support_bundle_staging_name(name)
            try:
                descriptor = os.open(candidate, flags, 0o600, dir_fd=directory_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        else:
            raise OSError("could not allocate support bundle staging file.")
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        staging_identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staging_identity.st_mode)
            or stat.S_IMODE(staging_identity.st_mode) != 0o600
            or staging_identity.st_size != len(payload)
        ):
            raise OSError("support bundle staging identity, mode, or size changed.")
        _validate_existing_leaf(name, directory_fd=directory_fd)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        _validate_published_leaf(
            name,
            directory_fd=directory_fd,
            expected_payload=payload,
            expected_identity=staging_identity,
        )
        os.fsync(directory_fd)
        verification_fd = _open_parent_directory_posix(parent)
        try:
            if not os.path.samestat(os.fstat(directory_fd), os.fstat(verification_fd)):
                raise OSError("support bundle parent path changed during publication.")
        finally:
            os.close(verification_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def _write_support_bundle_windows(
    parent: Path,
    name: str,
    payload: bytes,
    *,
    requested_temporary_name: str | None,
) -> None:
    from cayu.cli import _guarded_tree_publication as guarded_publication
    from cayu.cli._cloud_private_state import _move_file_ex_windows

    try:
        parent_identity = guarded_publication._capture_parent(parent)
        with guarded_publication._pinned_parent(
            parent,
            expected=parent_identity,
        ) as parent_guard:
            parent_guard.assert_unchanged()
            _validate_existing_windows_leaf(parent / name)
            temporary: Path | None = None
            temporary_identity: os.stat_result | None = None
            descriptor: int | None = None
            publication_error: BaseException | None = None
            try:
                for _attempt in range(1 if requested_temporary_name is not None else 32):
                    candidate = parent / (
                        requested_temporary_name or support_bundle_staging_name(name)
                    )
                    try:
                        descriptor = _open_private_windows_file(candidate)
                    except FileExistsError:
                        continue
                    temporary = candidate
                    temporary_identity = os.fstat(descriptor)
                    break
                else:
                    raise OSError("could not allocate support bundle staging file.")
                if descriptor is None or temporary is None or temporary_identity is None:
                    raise OSError("support bundle staging ownership was not established.")

                _write_all(descriptor, payload)
                os.fsync(descriptor)
                staging_evidence = os.lstat(temporary)
                opened_evidence = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(staging_evidence.st_mode)
                    or not stat.S_ISREG(opened_evidence.st_mode)
                    or staging_evidence.st_size != len(payload)
                    or opened_evidence.st_size != len(payload)
                    or not os.path.samestat(temporary_identity, staging_evidence)
                    or not os.path.samestat(temporary_identity, opened_evidence)
                ):
                    raise OSError("support bundle staging identity or size changed.")
                dacl_present, dacl_protected = guarded_publication._windows_directory_dacl_state(
                    temporary
                )
                if not dacl_present or not dacl_protected:
                    raise OSError(
                        "support bundle staging file does not have a protected private DACL."
                    )

                parent_guard.assert_unchanged()
                _validate_existing_windows_leaf(parent / name)
                _move_file_ex_windows(temporary, parent / name, flags=0x1 | 0x8)
                temporary = None
                _validate_published_windows_leaf(
                    parent / name,
                    expected_payload=payload,
                    expected_identity=temporary_identity,
                    descriptor=descriptor,
                )
                parent_guard.assert_unchanged()
            except BaseException as exc:
                publication_error = exc
                raise
            finally:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError as cleanup_error:
                        if publication_error is not None:
                            publication_error.add_note(str(cleanup_error))
                        else:
                            raise
                if temporary is not None and temporary_identity is not None:
                    try:
                        current = os.lstat(temporary)
                        if not stat.S_ISREG(current.st_mode) or not os.path.samestat(
                            temporary_identity, current
                        ):
                            raise OSError(
                                "support bundle staging ownership changed during cleanup."
                            )
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        if publication_error is not None:
                            publication_error.add_note(str(cleanup_error))
                        else:
                            raise
    except OSError:
        raise
    except Exception as exc:
        raise OSError("secure Windows support bundle publication failed.") from exc


def cleanup_support_bundle_staging(
    path: str | os.PathLike[str],
    temporary_name: str,
) -> None:
    """Remove one supervisor-owned staging leaf after failed publication."""

    destination = Path(path)
    absolute = destination.absolute()
    if not absolute.name or absolute.name in {".", ".."}:
        raise OSError("support bundle output must name a file.")
    temporary_name = _validated_staging_name(
        temporary_name,
        destination_name=absolute.name,
    )
    parent = absolute.parent
    if os.name == "posix":
        directory_fd = _open_parent_directory_posix(parent)
        try:
            try:
                evidence = os.stat(
                    temporary_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            if stat.S_ISLNK(evidence.st_mode) or not stat.S_ISREG(evidence.st_mode):
                raise OSError("support bundle staging leaf changed before cleanup.")
            os.unlink(temporary_name, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
        return
    if os.name == "nt":
        from cayu.cli import _guarded_tree_publication as guarded_publication

        parent_identity = guarded_publication._capture_parent(parent)
        with guarded_publication._pinned_parent(
            parent,
            expected=parent_identity,
        ) as guard:
            guard.assert_unchanged()
            temporary = parent / temporary_name
            try:
                _validate_existing_windows_leaf(temporary)
                temporary.unlink()
            except FileNotFoundError:
                return
            guard.assert_unchanged()
        return
    raise OSError("secure support bundle staging cleanup is unavailable.")


def _validate_existing_windows_leaf(path: Path) -> None:
    from cayu.cli import _guarded_tree_publication as guarded_publication

    try:
        evidence = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(evidence.st_mode)
        or guarded_publication._is_windows_reparse_point(evidence)
        or not stat.S_ISREG(evidence.st_mode)
    ):
        raise OSError("support bundle output must be a regular file or absent.")


def _validate_published_windows_leaf(
    path: Path,
    *,
    expected_payload: bytes,
    expected_identity: os.stat_result,
    descriptor: int,
) -> None:
    from cayu.cli import _guarded_tree_publication as guarded_publication

    evidence = os.lstat(path)
    opened_evidence = os.fstat(descriptor)
    dacl_present, dacl_protected = guarded_publication._windows_directory_dacl_state(path)
    if (
        stat.S_ISLNK(evidence.st_mode)
        or guarded_publication._is_windows_reparse_point(evidence)
        or not stat.S_ISREG(evidence.st_mode)
        or not stat.S_ISREG(opened_evidence.st_mode)
        or evidence.st_size != len(expected_payload)
        or opened_evidence.st_size != len(expected_payload)
        or not os.path.samestat(expected_identity, evidence)
        or not os.path.samestat(expected_identity, opened_evidence)
        or not dacl_present
        or not dacl_protected
    ):
        raise OSError(
            "published support bundle failed identity, regular-file, private-DACL, "
            "or size validation."
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = len(expected_payload) + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    published_payload = b"".join(chunks)
    if published_payload != expected_payload:
        raise OSError("published support bundle content changed before validation.")
    validate_support_bundle_archive(published_payload)
    final_evidence = os.lstat(path)
    if (
        stat.S_ISLNK(final_evidence.st_mode)
        or guarded_publication._is_windows_reparse_point(final_evidence)
        or not stat.S_ISREG(final_evidence.st_mode)
        or not os.path.samestat(expected_identity, final_evidence)
    ):
        raise OSError("published support bundle identity changed during validation.")


def _open_private_windows_file(path: Path) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = (
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        )

    windows_ctypes: Any = ctypes
    advapi32 = windows_ctypes.WinDLL("advapi32", use_last_error=True)
    convert_security_descriptor = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert_security_descriptor.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    )
    convert_security_descriptor.restype = wintypes.BOOL
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_SecurityAttributes),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = (wintypes.LPVOID,)
    local_free.restype = wintypes.LPVOID

    security_descriptor = wintypes.LPVOID()
    if not convert_security_descriptor(
        "D:P(A;;FA;;;OW)(A;;FA;;;SY)(A;;FA;;;BA)",
        1,
        ctypes.byref(security_descriptor),
        None,
    ):
        error_code = windows_ctypes.get_last_error()
        raise OSError(error_code, "could not construct a private support-bundle DACL.")
    descriptor = -1
    operation_error: BaseException | None = None
    try:
        attributes = _SecurityAttributes(
            ctypes.sizeof(_SecurityAttributes),
            security_descriptor,
            False,
        )
        handle = create_file(
            os.fspath(path),
            0x80000000 | 0x40000000,
            0x4,
            ctypes.byref(attributes),
            1,
            0x80,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            error_code = windows_ctypes.get_last_error()
            if error_code in {80, 183}:
                raise FileExistsError(error_code, "support bundle staging file exists.")
            raise OSError(error_code, "could not create private support bundle staging file.")
        try:
            open_osfhandle = getattr(msvcrt, "open_osfhandle", None)
            if not callable(open_osfhandle):
                raise OSError("Windows file-descriptor conversion is unavailable.")
            descriptor = open_osfhandle(
                int(handle),
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
        except BaseException:
            close_handle(handle)
            raise
        return descriptor
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if local_free(security_descriptor):
            free_error = OSError("could not release the support-bundle security descriptor.")
            if operation_error is not None:
                operation_error.add_note(str(free_error))
            else:
                if descriptor >= 0:
                    os.close(descriptor)
                raise free_error
