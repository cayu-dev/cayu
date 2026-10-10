"""Typed retention policies and reports for runtime-owned storage.

Retention is off by default: nothing is removed unless an operator or an
application explicitly applies a policy to a store. Each store that supports
retention decides for itself what is still referenced and never removes it.
The contract is store-neutral so later stores (eval results, artifacts,
workspaces) can implement the same policy, report and audit shapes; the session
store is the first implementation.

A policy selects by age, status and an optional per-run budget, and chooses one
of two modes. ``compact`` removes bulky, reconstructible payloads and keeps the
record itself; ``delete`` removes the whole record. A dry run reports exactly
what an apply with the same inputs would act on; every apply writes a durable
audit record.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import copy_durable_json_object, require_durable_clean_nonblank
from cayu.sessions.records import SessionStatus

#: The hard ceiling on items one retention run may act on. It also bounds the
#: size of the report and the audit record.
MAX_RETENTION_ITEMS_PER_RUN = 10_000
DEFAULT_RETENTION_ITEMS_PER_RUN = 1_000
#: Report detail strings are diagnostics, never payload copies.
MAX_RETENTION_DETAIL_CHARS = 512
#: Default size above which compaction replaces a stored tool-output body.
DEFAULT_COMPACT_TOOL_OUTPUT_MIN_BYTES = 16 * 1024
MIN_COMPACT_TOOL_OUTPUT_BYTES = 1024
MAX_RETENTION_AUDIT_LIST_LIMIT = 200

#: Session statuses retention may select. Pending, running, interrupting and
#: interrupted sessions can still run, resume or await recovery, so no policy
#: can select them.
RETENTION_TERMINAL_SESSION_STATUSES = frozenset({SessionStatus.COMPLETED, SessionStatus.FAILED})


def _terminal_session_statuses(value: object) -> frozenset[SessionStatus]:
    if isinstance(value, str) or not isinstance(value, Collection):
        raise ValueError("statuses must be a collection of SessionStatus values.")
    statuses = frozenset(SessionStatus(status) for status in value)
    if not statuses:
        raise ValueError("statuses must not be empty.")
    unsupported = statuses - RETENTION_TERMINAL_SESSION_STATUSES
    if unsupported:
        names = ", ".join(sorted(status.value for status in unsupported))
        raise ValueError(
            f"Retention selects only terminal sessions (completed, failed); got: {names}."
        )
    return statuses


class RetentionMode(StrEnum):
    """What a retention run does to each selected item."""

    #: Remove bulky payloads and keep the record, its audit trail and usage.
    COMPACT = "compact"
    #: Remove the whole record.
    DELETE = "delete"


class RetentionDisposition(StrEnum):
    """The outcome of one item in a retention report."""

    #: Dry run: an apply with the same inputs would act on this item.
    SELECTED = "selected"
    #: Apply: the store compacted or deleted this item.
    APPLIED = "applied"
    #: Kept because a protection applies.
    PROTECTED = "protected"
    #: Eligible, but the run's budget was exhausted before it.
    DEFERRED = "deferred"


class RetentionProtection(StrEnum):
    """Why a store kept an item that matched the policy's age and status."""

    #: A session in the same parent/fork lineage must be kept, so the lineage
    #: is kept whole.
    LINEAGE = "lineage"
    #: A task that is not terminal references the session.
    LIVE_TASK = "live_task"
    #: An unexpired execution lease is held on the session.
    EXECUTION_LEASE = "execution_lease"
    #: The session's checkpoint carries a pending approval, user input or
    #: delegated action, or its pending-action metadata is not yet backfilled.
    PENDING_ACTION = "pending_action"
    #: A collaboration participant bound to the session has an open request,
    #: clarification question, or unsettled clarification delivery or service.
    PENDING_CLARIFICATION = "pending_clarification"
    #: Another session holds an active context-view selection on this
    #: session's transcript and checkpoint.
    CHECKPOINT_DEPENDENCY = "checkpoint_dependency"
    #: An unreleased agent-snapshot pin or protection mentions the session.
    SNAPSHOT_PIN = "snapshot_pin"
    #: A stored eval result, captured result or trial checkpoint names the
    #: session.
    EVAL_REFERENCE = "eval_reference"
    #: Knowledge evidence that is not detached points at the session or one
    #: of its events.
    KNOWLEDGE_EVIDENCE = "knowledge_evidence"
    #: A pending application product operation is bound to the session.
    PRODUCT_OPERATION = "product_operation"
    #: An event watcher or persisted event side effect has not yet consumed
    #: the session's events.
    EVENT_DELIVERY_BACKLOG = "event_delivery_backlog"
    #: Compaction only: the session has session-export records whose
    #: commitments cover its stored events.
    SESSION_EXPORT = "session_export"
    #: A session or task closure owns the session's lineage.
    CLOSURE_IN_PROGRESS = "closure_in_progress"
    #: The store's own deletion guard refused the session, for example an
    #: incomplete terminal publication or a pending budget settlement.
    ERASURE_GUARD = "erasure_guard"
    #: The caller named the session as protected.
    CALLER_PROTECTED = "caller_protected"
    #: The item changed between planning and its write transaction.
    CHANGED = "changed"
    #: Evals: an eval baseline or baseline history selects the run's result.
    BASELINE = "baseline"
    #: Evals: the run retains trial checkpoints for a benchmark campaign
    #: (resume, retry, rescore and inspection read them).
    CAMPAIGN_EVIDENCE = "campaign_evidence"
    #: Artifacts: a durable pin, including resource and workspace-checkpoint
    #: pins, retains the artifact.
    PINNED = "pinned"
    #: Artifacts: a session that still exists owns or references the artifact,
    #: or a session closure claim is in progress for its session.
    SESSION_REFERENCE = "session_reference"
    #: Workspaces: a workspace checkpoint is mutating or checkpointing.
    WORKSPACE_CHECKPOINT = "workspace_checkpoint"
    #: Workspaces: the environment's workspace keeps durable branch state that
    #: retention does not settle.
    BRANCH_RETENTION = "branch_retention"


class RetentionPolicy(BaseModel):
    """Store-neutral retention selection: age, mode, per-run budget and dry run.

    A store applies a policy only when asked; there is no implicit retention.
    ``max_items`` and ``max_bytes`` bound one run: items are taken oldest first
    and the run stops after ``max_items`` items, or after the first item that
    brings the selected bytes to at least ``max_bytes``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    older_than: timedelta
    mode: RetentionMode = RetentionMode.COMPACT
    max_items: StrictInt = Field(
        default=DEFAULT_RETENTION_ITEMS_PER_RUN,
        ge=1,
        le=MAX_RETENTION_ITEMS_PER_RUN,
    )
    max_bytes: StrictInt | None = Field(default=None, ge=1)
    dry_run: StrictBool = True

    @field_validator("older_than")
    @classmethod
    def validate_older_than(cls, value: timedelta) -> timedelta:
        if value <= timedelta(0):
            raise ValueError("older_than must be a positive duration.")
        return value

    def policy_document(self) -> dict[str, Any]:
        """Return the JSON document recorded in reports and audits."""

        document = self.model_dump(mode="json")
        document["older_than_seconds"] = self.older_than.total_seconds()
        document.pop("older_than", None)
        return document


class SessionRetentionPolicy(RetentionPolicy):
    """Retention policy for a session store.

    ``statuses`` defaults to the terminal statuses and may only narrow them.
    ``compact`` removes model text and thinking delta events and replaces
    tool-output bodies larger than ``compact_tool_output_min_bytes`` in stored
    tool events with a short marker naming their size and digest. It keeps the
    session record, transcript, checkpoint, terminal events, usage-bearing
    events and every other audit-relevant event. ``delete`` removes the whole
    session through the store's own deletion path.
    """

    statuses: frozenset[SessionStatus] = RETENTION_TERMINAL_SESSION_STATUSES
    compact_tool_output_min_bytes: StrictInt = Field(
        default=DEFAULT_COMPACT_TOOL_OUTPUT_MIN_BYTES,
        ge=MIN_COMPACT_TOOL_OUTPUT_BYTES,
    )

    @field_validator("statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value: object) -> frozenset[SessionStatus]:
        return _terminal_session_statuses(value)

    def policy_document(self) -> dict[str, Any]:
        document = super().policy_document()
        document["statuses"] = sorted(status.value for status in self.statuses)
        return document


#: Eval run statuses retention may select; queued, running and cancelling runs
#: are never selected.
RETENTION_TERMINAL_EVAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
RETENTION_ARTIFACT_SCOPES = frozenset({"session", "environment"})


def _require_delete_mode(mode: RetentionMode, target: str) -> RetentionMode:
    if mode is not RetentionMode.DELETE:
        raise ValueError(f"{target} retention supports only mode='delete'.")
    return mode


class EvalRetentionPolicy(RetentionPolicy):
    """Retention policy for an eval store.

    Selects terminal eval runs whose ``finished_at`` is older than
    ``older_than`` and deletes each run with its result, its fresh result
    record and its trial checkpoints. Captured result records, corpora,
    scenarios, authored suites, baselines and judge calibrations are never
    removed.
    """

    mode: RetentionMode = RetentionMode.DELETE
    statuses: frozenset[str] = RETENTION_TERMINAL_EVAL_RUN_STATUSES

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: RetentionMode) -> RetentionMode:
        return _require_delete_mode(value, "Eval")

    @field_validator("statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value: object) -> frozenset[str]:
        if isinstance(value, str) or not isinstance(value, Collection):
            raise ValueError("statuses must be a collection of eval run statuses.")
        statuses = frozenset(str(getattr(status, "value", status)) for status in value)
        if not statuses:
            raise ValueError("statuses must not be empty.")
        unsupported = statuses - RETENTION_TERMINAL_EVAL_RUN_STATUSES
        if unsupported:
            raise ValueError(
                "Retention selects only terminal eval runs (completed, failed, cancelled); "
                f"got: {', '.join(sorted(unsupported))}."
            )
        return statuses

    def policy_document(self) -> dict[str, Any]:
        document = super().policy_document()
        document["statuses"] = sorted(self.statuses)
        return document


class ArtifactRetentionPolicy(RetentionPolicy):
    """Age and size retention for an artifact store.

    Selects artifacts created before ``now - older_than`` in the given scopes
    and at least ``min_size_bytes`` large, oldest first. With
    ``target_total_bytes`` a run stops once the store's listed total would be
    at or below that size.
    """

    mode: RetentionMode = RetentionMode.DELETE
    scopes: frozenset[str] = RETENTION_ARTIFACT_SCOPES
    min_size_bytes: StrictInt = Field(default=0, ge=0)
    target_total_bytes: StrictInt | None = Field(default=None, ge=0)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: RetentionMode) -> RetentionMode:
        return _require_delete_mode(value, "Artifact")

    @field_validator("scopes", mode="before")
    @classmethod
    def copy_scopes(cls, value: object) -> frozenset[str]:
        if isinstance(value, str) or not isinstance(value, Collection):
            raise ValueError("scopes must be a collection of artifact scopes.")
        scopes = frozenset(str(getattr(scope, "value", scope)) for scope in value)
        if not scopes or not scopes <= RETENTION_ARTIFACT_SCOPES:
            raise ValueError("scopes must be a non-empty subset of session, environment.")
        return scopes

    def policy_document(self) -> dict[str, Any]:
        document = super().policy_document()
        document["scopes"] = sorted(self.scopes)
        return document


class WorkspaceRetentionPolicy(RetentionPolicy):
    """Retention for leftover runner and coding workspaces.

    Selects terminal sessions last updated before ``now - older_than`` that
    still own unsettled environment allocations, and disposes them through the
    runtime's incomplete-session recovery, which reaps each allocation through
    its environment factory under a recovery claim.
    """

    mode: RetentionMode = RetentionMode.DELETE
    statuses: frozenset[SessionStatus] = RETENTION_TERMINAL_SESSION_STATUSES

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: RetentionMode) -> RetentionMode:
        return _require_delete_mode(value, "Workspace")

    @field_validator("statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value: object) -> frozenset[SessionStatus]:
        return _terminal_session_statuses(value)

    def policy_document(self) -> dict[str, Any]:
        document = super().policy_document()
        document["statuses"] = sorted(status.value for status in self.statuses)
        return document


class StorageRetentionTarget(StrEnum):
    """A runtime-owned store kind an application-level policy can prune."""

    SESSIONS = "sessions"
    EVALS = "evals"
    ARTIFACTS = "artifacts"
    WORKSPACES = "workspaces"


class StorageRetentionPolicy(BaseModel):
    """One application-level policy that combines the per-store policies.

    Only the targets given are pruned. ``dry_run`` applies to every target and
    overrides the sub-policies' own flags. The protected identifier sets cover
    references outside the application; references held by the application's
    configured stores are collected automatically.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    sessions: SessionRetentionPolicy | None = None
    evals: EvalRetentionPolicy | None = None
    artifacts: ArtifactRetentionPolicy | None = None
    workspaces: WorkspaceRetentionPolicy | None = None
    dry_run: StrictBool = True
    protected_session_ids: frozenset[str] = frozenset()
    protected_eval_ids: frozenset[str] = frozenset()
    protected_artifact_ids: frozenset[str] = frozenset()

    @field_validator(
        "protected_session_ids", "protected_eval_ids", "protected_artifact_ids", mode="before"
    )
    @classmethod
    def copy_identifiers(cls, value: object, info) -> frozenset[str]:
        return copy_protected_ids(value, info.field_name)

    @model_validator(mode="after")
    def validate_targets(self) -> StorageRetentionPolicy:
        if not self.targets:
            raise ValueError("A storage retention policy needs at least one target policy.")
        return self

    @property
    def targets(self) -> tuple[StorageRetentionTarget, ...]:
        return tuple(
            target for target in StorageRetentionTarget if getattr(self, target.value) is not None
        )

    def target_policy(self, target: StorageRetentionTarget) -> RetentionPolicy | None:
        """The target's policy with this policy's ``dry_run``."""

        policy = getattr(self, StorageRetentionTarget(target).value)
        if policy is None:
            return None
        return policy.model_copy(update={"dry_run": self.dry_run})

    @classmethod
    def for_targets(
        cls,
        targets: Collection[StorageRetentionTarget | str],
        *,
        older_than: timedelta,
        session_mode: RetentionMode = RetentionMode.COMPACT,
        dry_run: bool = True,
        max_items: int = DEFAULT_RETENTION_ITEMS_PER_RUN,
        max_bytes: int | None = None,
        protected_session_ids: Collection[str] = (),
        protected_eval_ids: Collection[str] = (),
        protected_artifact_ids: Collection[str] = (),
    ) -> StorageRetentionPolicy:
        """Build one policy that applies the same age and budget to each target."""

        selected = {StorageRetentionTarget(target) for target in targets}
        common: dict[str, Any] = {
            "older_than": older_than,
            "max_items": max_items,
            "max_bytes": max_bytes,
            "dry_run": dry_run,
        }
        return cls(
            sessions=(
                SessionRetentionPolicy(mode=session_mode, **common)
                if StorageRetentionTarget.SESSIONS in selected
                else None
            ),
            evals=(
                EvalRetentionPolicy(**common) if StorageRetentionTarget.EVALS in selected else None
            ),
            artifacts=(
                ArtifactRetentionPolicy(**common)
                if StorageRetentionTarget.ARTIFACTS in selected
                else None
            ),
            workspaces=(
                WorkspaceRetentionPolicy(**common)
                if StorageRetentionTarget.WORKSPACES in selected
                else None
            ),
            dry_run=dry_run,
            protected_session_ids=copy_protected_ids(
                protected_session_ids, "protected_session_ids"
            ),
            protected_eval_ids=copy_protected_ids(protected_eval_ids, "protected_eval_ids"),
            protected_artifact_ids=copy_protected_ids(
                protected_artifact_ids, "protected_artifact_ids"
            ),
        )


class RetentionItem(BaseModel):
    """One evaluated item. Counts and bytes are logical stored-value sizes."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    item_id: str
    status: str | None = None
    last_updated_at: datetime | None = None
    disposition: RetentionDisposition
    protections: tuple[RetentionProtection, ...] = ()
    detail: str | None = Field(default=None, max_length=MAX_RETENTION_DETAIL_CHARS)
    counts: dict[str, StrictInt] = Field(default_factory=dict)
    bytes: StrictInt = Field(default=0, ge=0)

    @field_validator("item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "item_id")

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("Retention counts must be non-negative.")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_disposition(self) -> RetentionItem:
        if (self.disposition is RetentionDisposition.PROTECTED) != bool(self.protections):
            raise ValueError("Protected items, and only protected items, carry protections.")
        return self


class RetentionReport(BaseModel):
    """The result of one dry run or apply against one store."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    store_kind: str
    mode: RetentionMode
    dry_run: StrictBool
    policy: dict[str, Any]
    started_at: datetime
    completed_at: datetime
    cutoff: datetime
    #: The durable audit record for an apply; ``None`` for a dry run.
    audit_id: str | None = None
    #: Selected (dry run) or applied items, oldest first, in action order.
    items: tuple[RetentionItem, ...] = ()
    #: Items kept by a protection. Lists at most ``MAX_RETENTION_ITEMS_PER_RUN``.
    protected: tuple[RetentionItem, ...] = ()
    protected_truncated: StrictBool = False
    #: Eligible items left for a later run by the budget.
    deferred_count: StrictInt = Field(default=0, ge=0)

    @field_validator("policy", mode="before")
    @classmethod
    def copy_policy(cls, value: object) -> dict[str, Any]:
        return copy_durable_json_object(value, "policy")

    @model_validator(mode="after")
    def validate_report(self) -> RetentionReport:
        acted = RetentionDisposition.SELECTED if self.dry_run else RetentionDisposition.APPLIED
        if any(item.disposition is not acted for item in self.items):
            raise ValueError(f"Report items must all be {acted.value}.")
        if any(item.disposition is not RetentionDisposition.PROTECTED for item in self.protected):
            raise ValueError("Report protections must all be protected items.")
        if self.dry_run != (self.audit_id is None):
            raise ValueError("An apply has an audit record and a dry run has none.")
        return self

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes for item in self.items)

    def totals(self) -> dict[str, int]:
        """Summed per-item counts plus items and bytes."""

        totals: dict[str, int] = {}
        for item in self.items:
            for key, count in item.counts.items():
                totals[key] = totals.get(key, 0) + count
        totals["items"] = len(self.items)
        totals["bytes"] = self.total_bytes
        totals["protected"] = len(self.protected)
        totals["deferred"] = self.deferred_count
        return dict(sorted(totals.items()))


class StorageRetentionReport(BaseModel):
    """Reports of one application-level retention run, in the order applied."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    dry_run: StrictBool
    started_at: datetime
    completed_at: datetime
    reports: tuple[RetentionReport, ...] = ()
    #: Targets the policy named but the application could not prune, with why.
    skipped: dict[str, str] = Field(default_factory=dict)
    #: Targets that failed, with a bounded error message.
    errors: dict[str, str] = Field(default_factory=dict)

    def totals(self) -> dict[str, dict[str, int]]:
        totals: dict[str, dict[str, int]] = {}
        for report in self.reports:
            target = totals.setdefault(report.store_kind, {})
            for key, value in report.totals().items():
                target[key] = target.get(key, 0) + value
        return totals


class RetentionAuditEntry(BaseModel):
    """One item an apply acted on, written in the same transaction as the change."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    item_id: str
    action: RetentionMode
    counts: dict[str, StrictInt] = Field(default_factory=dict)
    bytes: StrictInt = Field(default=0, ge=0)
    recorded_at: datetime


class RetentionAuditState(StrEnum):
    #: The run started; entries record every item it changed so far.
    STARTED = "started"
    COMPLETED = "completed"


class RetentionAuditRecord(BaseModel):
    """The durable record of one retention apply."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    audit_id: str
    store_kind: str
    mode: RetentionMode
    state: RetentionAuditState
    started_at: datetime
    completed_at: datetime | None = None
    policy: dict[str, Any]
    summary: dict[str, Any] = Field(default_factory=dict)
    entries: tuple[RetentionAuditEntry, ...] = ()

    @field_validator("policy", "summary", mode="before")
    @classmethod
    def copy_documents(cls, value: object, info) -> dict[str, Any]:
        return copy_durable_json_object(value, info.field_name)


class RetentionAuditUnavailable(RuntimeError):
    """The database predates the retention audit tables (storage revision 118)."""


class RetentionPhase(StrEnum):
    PLANNED = "planned"
    BATCH = "batch"


class RetentionProgress(BaseModel):
    """Reported after planning and after each committed apply batch.

    A store releases its write lock before reporting, so a callback runs while
    other work on the store can proceed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    store_kind: str
    phase: RetentionPhase
    dry_run: StrictBool
    audit_id: str | None = None
    batch_index: StrictInt = Field(default=0, ge=0)
    planned_items: StrictInt = Field(default=0, ge=0)
    applied: tuple[RetentionItem, ...] = ()
    protected: tuple[RetentionItem, ...] = ()


RetentionProgressCallback = Callable[[RetentionProgress], Awaitable[None]]


@runtime_checkable
class RetentionAuditSink(Protocol):
    """Durable audit storage for retention runs of any store kind.

    The SQLite and Postgres session stores implement it with the same tables
    they use for their own runs, so stores without a database (artifacts,
    workspaces) record their applies there.
    """

    async def begin_retention_audit(
        self,
        *,
        store_kind: str,
        mode: RetentionMode,
        policy: Mapping[str, Any],
        started_at: datetime,
    ) -> str: ...

    async def record_retention_entry(self, audit_id: str, entry: RetentionAuditEntry) -> None: ...

    async def complete_retention_audit(
        self, audit_id: str, *, completed_at: datetime, summary: Mapping[str, Any]
    ) -> None: ...


@runtime_checkable
class SessionRetentionTarget(Protocol):
    """A store that applies a :class:`SessionRetentionPolicy` to its sessions."""

    supports_storage_retention: bool

    async def apply_retention_policy(
        self,
        policy: SessionRetentionPolicy,
        *,
        protected_session_ids: Collection[str] = (),
        references: Mapping[RetentionProtection, Collection[str]] | None = None,
        progress: RetentionProgressCallback | None = None,
    ) -> RetentionReport: ...

    async def list_retention_audits(
        self,
        *,
        limit: int = 20,
        item_id: str | None = None,
        store_kind: str | None = None,
    ) -> tuple[RetentionAuditRecord, ...]: ...

    async def load_retention_audit(self, audit_id: str) -> RetentionAuditRecord | None: ...


def copy_protected_ids(value: object, field_name: str) -> frozenset[str]:
    """Validate caller-supplied protected identifiers."""

    if isinstance(value, str | bytes) or not isinstance(value, Collection):
        raise TypeError(f"{field_name} must be a collection of identifiers.")
    copied: set[str] = set()
    for item in value:
        if type(item) is not str:
            raise TypeError(f"{field_name} must contain only strings.")
        copied.add(require_durable_clean_nonblank(item, field_name))
    return frozenset(copied)


def bounded_retention_detail(value: object) -> str | None:
    """Return a short single-line diagnostic for a report item."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) > MAX_RETENTION_DETAIL_CHARS:
        text = text[: MAX_RETENTION_DETAIL_CHARS - 3] + "..."
    return text


def retention_audit_summary(report: RetentionReport) -> Mapping[str, Any]:
    """The run summary stored with a completed audit record."""

    return {
        "totals": report.totals(),
        "cutoff": report.cutoff.isoformat(),
        "protected_by_reason": _protected_by_reason(report.protected),
        "protected_truncated": report.protected_truncated,
    }


def _protected_by_reason(items: tuple[RetentionItem, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for protection in item.protections:
            counts[protection.value] = counts.get(protection.value, 0) + 1
    return dict(sorted(counts.items()))
