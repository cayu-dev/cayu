"""Maintained trusted-repository coding product contracts and evidence."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._coding_product_authority import (
    CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA,
    CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY,
    CodingProductSourceCopyAuthority,
    is_final_git_result_envelope,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.artifacts import (
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    copy_artifact_read_result,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import Message
from cayu.runtime.app import CayuApp
from cayu.runtime.completion_result_resolvers import (
    CompletionResultResolver,
    CompletionResultResolverRequest,
)
from cayu.runtime.completion_verifiers import (
    CompletionVerifierRequest,
    DeterministicCompletionVerifier,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.runtime.sessions import RunRequest, session_input_messages_sha256
from cayu.runtime.work_contracts import (
    CompletionConstraintOutcome,
    CompletionContinuationPolicy,
    CompletionCriterionOutcome,
    CompletionGap,
    CompletionRejectionAction,
    CompletionResultReference,
    CompletionResultResolverRef,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerifierDecision,
    CompletionVerifierKind,
    CompletionVerifierRef,
    CriterionOutcomeStatus,
    WorkConstraint,
    WorkContract,
    WorkContractDraft,
    WorkCriterion,
    WorkEvidenceReference,
    WorkEvidenceRequirement,
    work_contract_from_draft,
)
from cayu.vaults import REDACTED_SECRET
from cayu.workspaces import Workspace, WorkspaceRevisionObservation
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    observe_deterministic_workspace,
)

CODING_PRODUCT_SCHEMA_VERSION = "cayu.coding_product.v1"
CODING_PRODUCT_RESULT_KIND = "coding_product_result"
CODING_PRODUCT_EVIDENCE_KIND = "coding_product_evidence"
CODING_PRODUCT_MAX_EVENTS = 20_000
CODING_PRODUCT_MAX_EVENT_BYTES = 16 * 1024 * 1024
CODING_PRODUCT_MAX_RESULT_BYTES = 1024 * 1024
CODING_PRODUCT_MAX_LIFECYCLE_RECEIPTS = 64
CODING_PRODUCT_MAX_SOURCE_ARTIFACT_BYTES = 16 * 1024 * 1024
CODING_PRODUCT_MAX_MUTATION_ARTIFACT_BYTES = 3 * 1024 * 1024
CODING_PRODUCT_MAX_TOOL_OUTPUT_ARTIFACT_BYTES = 1024 * 1024
CODING_PRODUCT_MAX_GIT_DIFF_BYTES = 1024 * 1024

_FINGERPRINT_PATTERN_LENGTHS = frozenset({64, 71})
_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_AWAITING_USER_INPUT,
    }
)
_MUTATION_TOOL_NAMES = frozenset({"apply_patch", "write_file", "edit_file", "delete_file"})
_TRUSTED_DOCKER_WARNING = (
    "Docker execution is for application-classified trusted repositories; it is not "
    "hostile-code isolation."
)


class CodingProductAdmissionError(ValueError):
    """The application-supplied product authority cannot be admitted."""


class CodingProductEvidenceError(ValueError):
    """Runtime evidence cannot support an authoritative product result."""


class CodingProductReconstructionRequiredError(RuntimeError):
    """Durable evidence must be reconciled before execution can continue."""


def _validated_identifier(value: str, field_name: str, *, maximum: int = 512) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > maximum or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{field_name} must not exceed {maximum} UTF-8 bytes.")
    return value


def _validated_fingerprint(value: str, field_name: str) -> str:
    value = _validated_identifier(value, field_name)
    raw = value.removeprefix("sha256:")
    if len(value) not in _FINGERPRINT_PATTERN_LENGTHS or len(raw) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 identity.")
    if any(character not in "0123456789abcdef" for character in raw):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 identity.")
    return value


def _canonical_model_bytes(value: BaseModel, field_name: str) -> bytes:
    return canonical_durable_json_bytes(
        value.model_dump(mode="json", warnings=False),
        field_name,
    )


def _model_fingerprint(value: BaseModel, field_name: str) -> str:
    return "sha256:" + sha256(_canonical_model_bytes(value, field_name)).hexdigest()


def _artifact_id(*parts: str) -> str:
    material = "\0".join(parts).encode("utf-8")
    return "art_" + sha256(material).hexdigest()[:32]


class _FrozenCodingModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class CodingProductState(StrEnum):
    """Durable product states, including every non-success settlement class."""

    ADMITTED = "admitted"
    PREPARING_WORKSPACE = "preparing_workspace"
    ACTIVE = "active"
    CHECKS_NOT_RUN = "checks_not_run"
    CHECKS_FAILED = "checks_failed"
    SOURCE_CONFLICT = "source_conflict"
    TOOLCHAIN_REBUILD_REQUIRED = "toolchain_rebuild_required"
    REVIEW_REQUIRED = "review_required"
    HUMAN_INPUT_REQUIRED = "human_input_required"
    READY_TO_PUBLISH = "ready_to_publish"
    PUBLISHING = "publishing"
    PATCH_READY_FOR_DELIVERY = "patch_ready_for_delivery"
    BLOCKED = "blocked"
    DENIED = "denied"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    RECONSTRUCTION_REQUIRED = "reconstruction_required"


class CodingGitBaselineAuthority(_FrozenCodingModel):
    """Application-observed clean Git state for the admitted source tree."""

    head_revision: str
    staged_entries_sha256: str
    tracked_flags_sha256: str
    status_sha256: str
    diff_sha256: str
    clean: Literal[True] = True

    @field_validator("head_revision")
    @classmethod
    def validate_head_revision(cls, value: str) -> str:
        value = _validated_identifier(value, "head_revision", maximum=128)
        if len(value) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("head_revision must be a lowercase Git object identity.")
        return value

    @field_validator(
        "staged_entries_sha256",
        "tracked_flags_sha256",
        "status_sha256",
        "diff_sha256",
    )
    @classmethod
    def validate_git_digest(cls, value: str, info) -> str:
        return _validated_fingerprint(value, info.field_name)


class CodingSourceAuthority(_FrozenCodingModel):
    """Application-owned source origin, immutable baseline, and destination."""

    origin_id: str
    workspace_id: str
    baseline_revision: str
    destination_id: str
    path_scope: Literal["complete"] = "complete"
    git_baseline: CodingGitBaselineAuthority
    observation_limits: WorkspaceRevisionObservationLimits

    @field_validator("origin_id", "workspace_id", "destination_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name)

    @field_validator("baseline_revision")
    @classmethod
    def validate_baseline(cls, value: str) -> str:
        return _validated_fingerprint(value, "baseline_revision")


class CodingTaskAuthority(_FrozenCodingModel):
    """Immutable application task identity without copying task prose into evidence."""

    task_id: str
    instruction_sha256: str

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _validated_identifier(value, "task_id")

    @field_validator("instruction_sha256")
    @classmethod
    def validate_instruction_sha256(cls, value: str) -> str:
        return _validated_fingerprint(value, "instruction_sha256")


class CodingRuntimeAuthority(_FrozenCodingModel):
    """Exact admitted Docker, tool, policy, approval, and redaction identity."""

    execution_kind: Literal["trusted_repository_docker"] = "trusted_repository_docker"
    environment_name: str = "coding"
    network: Literal["none"] = "none"
    credential_visibility: Literal["non_possession"] = "non_possession"
    toolchain_profile_id: str
    toolchain_profile_revision: str
    toolchain_profile_fingerprint: str
    image_fingerprint: str
    dependency_identity: str
    execution_profile_fingerprint: str
    tool_manifest_fingerprint: str
    tool_policy_fingerprint: str
    approval_policy_fingerprint: str
    redaction_profile_fingerprint: str

    @field_validator(
        "environment_name",
        "toolchain_profile_id",
        "toolchain_profile_revision",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name)

    @field_validator(
        "toolchain_profile_fingerprint",
        "image_fingerprint",
        "dependency_identity",
        "execution_profile_fingerprint",
        "tool_manifest_fingerprint",
        "tool_policy_fingerprint",
        "approval_policy_fingerprint",
        "redaction_profile_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _validated_fingerprint(value, info.field_name)


class CodingSettlementPolicy(_FrozenCodingModel):
    """Finite application-owned completion gates."""

    required_checks: tuple[str, ...] = ("format", "lint", "test")
    reviewer_required: StrictBool = True
    human_approval_required: StrictBool = False
    max_events: StrictInt = Field(default=CODING_PRODUCT_MAX_EVENTS, ge=1, le=100_000)
    max_event_bytes: StrictInt = Field(
        default=CODING_PRODUCT_MAX_EVENT_BYTES,
        ge=1024,
        le=128 * 1024 * 1024,
    )
    max_result_bytes: StrictInt = Field(
        default=CODING_PRODUCT_MAX_RESULT_BYTES,
        ge=4096,
        le=CODING_PRODUCT_MAX_RESULT_BYTES,
    )

    @field_validator("required_checks")
    @classmethod
    def validate_required_checks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if type(value) is not tuple:
            raise TypeError("required_checks must be a tuple.")
        checks = tuple(_validated_identifier(item, "required_checks item") for item in value)
        if not checks or checks != tuple(sorted(set(checks))):
            raise ValueError("required_checks must contain unique canonical values.")
        return checks


class CodingProductRequest(_FrozenCodingModel):
    """Complete immutable application authority for one coding-product run."""

    schema_version: Literal["cayu.coding_product.v1"] = CODING_PRODUCT_SCHEMA_VERSION
    product_run_id: str
    session_id: str
    agent_name: str
    source: CodingSourceAuthority
    task: CodingTaskAuthority
    runtime: CodingRuntimeAuthority
    settlement: CodingSettlementPolicy = Field(default_factory=CodingSettlementPolicy)

    @field_validator("product_run_id", "session_id", "agent_name")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name)

    @model_validator(mode="after")
    def validate_distinct_authority(self) -> CodingProductRequest:
        if self.task.task_id == self.product_run_id:
            raise ValueError("task_id and product_run_id must be distinct authorities.")
        return self

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "coding_product_request")


class CodingLifecycleReceipt(_FrozenCodingModel):
    """One append-only, idempotently addressable product-state transition."""

    schema_version: Literal["cayu.coding_product_lifecycle.v1"] = "cayu.coding_product_lifecycle.v1"
    product_run_id: str
    session_id: str
    request_fingerprint: str
    ordinal: StrictInt = Field(ge=1, le=CODING_PRODUCT_MAX_LIFECYCLE_RECEIPTS)
    prior_state: CodingProductState | None = None
    state: CodingProductState
    evidence_sha256: str | None = None
    reason_code: str | None = None

    @field_validator("product_run_id", "session_id", "reason_code")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, info.field_name)

    @field_validator("request_fingerprint")
    @classmethod
    def validate_request_fingerprint(cls, value: str) -> str:
        return _validated_fingerprint(value, "request_fingerprint")

    @field_validator("evidence_sha256")
    @classmethod
    def validate_evidence_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validated_fingerprint(value, "evidence_sha256")

    @model_validator(mode="after")
    def validate_transition(self) -> CodingLifecycleReceipt:
        if self.ordinal == 1:
            if self.prior_state is not None or self.state is not CodingProductState.ADMITTED:
                raise ValueError("The first lifecycle receipt must admit the product run.")
        elif self.prior_state is None:
            raise ValueError("Later lifecycle receipts require prior_state.")
        if self.prior_state is self.state:
            raise ValueError("Lifecycle transitions must change state.")
        return self


class CodingArtifactReference(_FrozenCodingModel):
    artifact_id: str
    sha256: str
    size_bytes: StrictInt = Field(ge=0)
    content_type: str = "application/json"

    @field_validator("artifact_id", "content_type")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name, maximum=1024)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validated_fingerprint(value, "sha256")


class CodingSourceObservationEvidence(_FrozenCodingModel):
    """Bounded source-manifest evidence retained outside the product result."""

    observer: str
    workspace_id: str
    status: WorkspaceRevisionObservationStatus
    path_scope: Literal["complete", "changed"]
    revision: str | None = None
    total_paths: StrictInt = Field(ge=0)
    artifact: CodingArtifactReference

    @field_validator("observer", "workspace_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validated_fingerprint(value, "revision")

    @model_validator(mode="after")
    def validate_supported_observation(self) -> CodingSourceObservationEvidence:
        if self.status is WorkspaceRevisionObservationStatus.SUPPORTED and (
            self.path_scope != "complete" or self.revision is None
        ):
            raise ValueError("Supported source evidence must be a complete revision.")
        return self


class CodingCheckEvidence(_FrozenCodingModel):
    check: str
    event_id: str
    profile_fingerprint: str
    status: str
    exit_code: StrictInt | None = None
    output_sha256: str | None = None
    output_artifact: CodingArtifactReference | None = None
    duration_ms: StrictInt | None = Field(default=None, ge=0)
    stdout_truncated: StrictBool = False
    stderr_truncated: StrictBool = False
    stdout_runner_truncated: StrictBool = False
    stderr_runner_truncated: StrictBool = False
    stdout_projection_truncated: StrictBool = False
    stderr_projection_truncated: StrictBool = False
    output_artifact_status: str | None = None
    timed_out: StrictBool = False
    cancelled: StrictBool = False
    workspace_mutation_settlement: str | None = None
    workspace_revision: str | None = None

    @field_validator(
        "check",
        "event_id",
        "status",
        "output_artifact_status",
        "workspace_mutation_settlement",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, info.field_name, maximum=1024)

    @field_validator("profile_fingerprint", "output_sha256", "workspace_revision")
    @classmethod
    def validate_optional_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_fingerprint(value, info.field_name)

    @property
    def settled_pass(self) -> bool:
        return (
            self.status == "passed"
            and self.exit_code == 0
            and not self.stdout_runner_truncated
            and not self.stderr_runner_truncated
            and (
                not (self.stdout_truncated or self.stderr_truncated)
                or (
                    (self.stdout_projection_truncated or self.stderr_projection_truncated)
                    and self.output_artifact_status == "stored"
                    and self.output_artifact is not None
                )
            )
            and not self.timed_out
            and not self.cancelled
            and self.duration_ms is not None
            and self.workspace_mutation_settlement in {"complete", "runner_quiescent"}
        )


class CodingCommandEvidence(_FrozenCodingModel):
    selector: str
    event_id: str
    selector_fingerprint: str
    argv_sha256: str | None = None
    status: str
    exit_code: StrictInt | None = None
    duration_ms: StrictInt | None = Field(default=None, ge=0)
    output_sha256: str | None = None
    output_artifact: CodingArtifactReference | None = None
    output_truncated: StrictBool = False
    output_runner_truncated: StrictBool = False
    output_projection_truncated: StrictBool = False
    output_artifact_status: str | None = None
    exit_code_admitted: StrictBool = False
    output_collection_complete: StrictBool = False
    output_publication_complete: StrictBool = False
    timed_out: StrictBool = False
    cancelled: StrictBool = False
    workspace_mutation_settlement: str | None = None

    @field_validator(
        "selector",
        "event_id",
        "status",
        "output_artifact_status",
        "workspace_mutation_settlement",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, info.field_name, maximum=1024)

    @field_validator("selector_fingerprint", "argv_sha256", "output_sha256")
    @classmethod
    def validate_optional_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_fingerprint(value, info.field_name)

    @property
    def settled(self) -> bool:
        return (
            self.status in {"succeeded", "nonzero"}
            and self.argv_sha256 is not None
            and self.exit_code is not None
            and self.duration_ms is not None
            and self.output_sha256 is not None
            and self.exit_code_admitted
            and self.output_collection_complete
            and self.output_publication_complete
            and not self.output_runner_truncated
            and (
                not self.output_projection_truncated
                or (self.output_artifact_status == "stored" and self.output_artifact is not None)
            )
            and not self.timed_out
            and not self.cancelled
            and self.workspace_mutation_settlement in {"complete", "runner_quiescent"}
        )


class CodingMutationEvidence(_FrozenCodingModel):
    tool_name: str
    event_id: str
    outcome: str
    operation_identity: str | None = None
    artifact: CodingArtifactReference | None = None
    evidence_sha256: str
    requires_fresh_read: StrictBool = False

    @field_validator("tool_name", "event_id", "outcome", "operation_identity")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, info.field_name, maximum=1024)

    @field_validator("evidence_sha256")
    @classmethod
    def validate_evidence_sha256(cls, value: str) -> str:
        return _validated_fingerprint(value, "evidence_sha256")

    @property
    def settled(self) -> bool:
        return self.outcome in {"applied", "completed"} and not self.requires_fresh_read


class CodingGitEntry(_FrozenCodingModel):
    path: str
    index: str
    worktree: str
    renamed_from: str | None = None

    @field_validator("path", "renamed_from")
    @classmethod
    def validate_entry_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, info.field_name, maximum=4096)

    @field_validator("index", "worktree")
    @classmethod
    def validate_status_code(cls, value: str, info) -> str:
        if type(value) is not str or len(value) != 1 or value not in " MTADRCU?!":
            raise ValueError(f"{info.field_name} must be one Git status code.")
        return value


class CodingGitEvidence(_FrozenCodingModel):
    event_id: str
    mode: Literal["diff"] = "diff"
    scope: str
    artifact: CodingArtifactReference
    entries: tuple[CodingGitEntry, ...] = ()
    entry_count: StrictInt = Field(ge=0)
    truncated: StrictBool = False
    binary_omitted: StrictBool = False

    @field_validator("event_id", "scope")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name, maximum=1024)

    @model_validator(mode="after")
    def validate_entry_count(self) -> CodingGitEvidence:
        if self.entry_count != len(self.entries):
            raise ValueError("Git entry_count must match the retained status entries.")
        return self


class CodingGitStatusEvidence(_FrozenCodingModel):
    event_id: str
    mode: Literal["status"] = "status"
    scope: str
    entries: tuple[CodingGitEntry, ...] = ()
    truncated: StrictBool = False

    @field_validator("event_id", "scope")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name, maximum=1024)


class CodingGitSummaryEntry(CodingGitEntry):
    additions: StrictInt | None = Field(default=None, ge=0)
    deletions: StrictInt | None = Field(default=None, ge=0)
    count_kind: Literal["text", "binary", "untracked", "unknown"]


class CodingGitSummaryEvidence(_FrozenCodingModel):
    event_id: str
    mode: Literal["summary"] = "summary"
    scope: str
    entries: tuple[CodingGitSummaryEntry, ...] = ()
    truncated: StrictBool = False

    @field_validator("event_id", "scope")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name, maximum=1024)


def _git_evidence_matches_source_revisions(
    *,
    initial_revision: str,
    final_revision: str | None,
    status: CodingGitStatusEvidence,
    summary: CodingGitSummaryEvidence,
    diff: CodingGitEvidence,
) -> bool:
    if final_revision is None:
        return False
    status_entries = tuple(
        (entry.path, entry.index, entry.worktree, entry.renamed_from) for entry in status.entries
    )
    summary_entries = tuple(
        (entry.path, entry.index, entry.worktree, entry.renamed_from) for entry in summary.entries
    )
    diff_entries = tuple(
        (entry.path, entry.index, entry.worktree, entry.renamed_from) for entry in diff.entries
    )
    source_changed = initial_revision != final_revision
    return (
        status_entries == summary_entries == diff_entries and bool(status_entries) is source_changed
    )


class CodingPublicationEvidence(_FrozenCodingModel):
    event_id: str
    destination_id: str
    outcome: Literal[
        "copied",
        "unchanged",
        "conflicted",
        "partial",
        "failed",
        "cancelled",
        "ambiguous",
    ]
    baseline_revision: str
    final_revision: str | None = None
    snapshot_id: str | None = None
    snapshot_sha256: str | None = None
    receipt_sha256: str | None = None
    destination_workspace_id: str | None = None
    workload_workspace_id: str | None = None
    source_conflict_policy: Literal["require_revision"] | None = None
    sync_back: Literal["always"] | None = None
    delete_missing: StrictBool | None = None
    copied_files: StrictInt | None = Field(default=None, ge=0)
    copied_bytes: StrictInt | None = Field(default=None, ge=0)
    deleted_files: StrictInt | None = Field(default=None, ge=0)
    detail_code: str | None = None

    @field_validator(
        "event_id",
        "destination_id",
        "snapshot_id",
        "destination_workspace_id",
        "workload_workspace_id",
        "detail_code",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, info.field_name, maximum=1024)

    @field_validator(
        "baseline_revision",
        "final_revision",
        "snapshot_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_optional_revision(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_fingerprint(value, info.field_name)

    @property
    def settled(self) -> bool:
        return (
            self.outcome in {"copied", "unchanged"}
            and self.final_revision is not None
            and self.receipt_sha256 is not None
            and self.snapshot_sha256 is not None
            and self.destination_workspace_id is not None
            and self.workload_workspace_id is not None
            and self.destination_workspace_id != "workspace-authority-unavailable"
            and self.workload_workspace_id != "workspace-authority-unavailable"
            and self.workload_workspace_id != self.destination_workspace_id
            and self.source_conflict_policy == "require_revision"
            and self.sync_back == "always"
            and self.delete_missing is True
            and self.copied_files is not None
            and self.copied_bytes is not None
            and self.deleted_files is not None
        )

    @model_validator(mode="after")
    def validate_settled_receipt_integrity(self) -> CodingPublicationEvidence:
        if self.outcome not in {"copied", "unchanged"}:
            return self
        if (
            self.snapshot_sha256 is None
            or self.receipt_sha256 is None
            or self.destination_workspace_id
            in {
                None,
                "workspace-authority-unavailable",
            }
            or self.workload_workspace_id
            in {
                None,
                "workspace-authority-unavailable",
            }
            or self.workload_workspace_id == self.destination_workspace_id
            or self.source_conflict_policy != "require_revision"
            or self.sync_back != "always"
            or self.delete_missing is not True
            or self.copied_files is None
            or self.copied_bytes is None
            or self.deleted_files is None
        ):
            raise ValueError("Settled publication evidence is incomplete.")
        snapshot_material = {
            "schema": "cayu.source_publication_snapshot.v1",
            "destination_workspace_id": self.destination_workspace_id,
            "workload_workspace_id": self.workload_workspace_id,
            "source": "sync",
            "outcome": "completed",
            "source_conflict_policy": self.source_conflict_policy,
            "sync_back": self.sync_back,
            "delete_missing": self.delete_missing,
            "copied_files": self.copied_files,
            "copied_bytes": self.copied_bytes,
            "deleted_files": self.deleted_files,
        }
        expected_snapshot_sha256 = (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    snapshot_material,
                    "source_publication_snapshot",
                )
            ).hexdigest()
        )
        receipt_material = {
            "schema": "cayu.source_publication_receipt.v1",
            "snapshot_sha256": self.snapshot_sha256,
            "destination_workspace_id": self.destination_workspace_id,
            "workload_workspace_id": self.workload_workspace_id,
            "outcome": "completed",
            "source_conflict_policy": self.source_conflict_policy,
            "sync_back": self.sync_back,
            "delete_missing": self.delete_missing,
            "copied_files": self.copied_files,
            "copied_bytes": self.copied_bytes,
            "deleted_files": self.deleted_files,
        }
        expected_receipt_sha256 = (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    receipt_material,
                    "source_publication_receipt",
                )
            ).hexdigest()
        )
        if (
            self.snapshot_sha256 != expected_snapshot_sha256
            or self.receipt_sha256 != expected_receipt_sha256
        ):
            raise ValueError("Settled publication receipt integrity is invalid.")
        return self


class CodingReviewSettlement(_FrozenCodingModel):
    reviewer: Literal["not_required", "passed", "required", "failed", "ambiguous"]
    human: Literal["not_required", "approved", "required", "denied", "ambiguous"]
    reviewer_event_id: str | None = None
    human_event_id: str | None = None

    @field_validator("reviewer_event_id", "human_event_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, info.field_name, maximum=1024)


class CodingProductCandidate(_FrozenCodingModel):
    """Bounded candidate evidence evaluated before patch-ready success."""

    schema_version: Literal["cayu.coding_product_result.v1"] = "cayu.coding_product_result.v1"
    product_run_id: str
    request_fingerprint: str
    session_id: str
    agent_name: str
    interaction_id: str | None = None
    task_id: str
    state: CodingProductState
    source_origin_id: str
    source_workspace_id: str
    source_destination_id: str
    initial_revision: str
    initial_git: CodingGitBaselineAuthority
    final_revision: str | None = None
    initial_source: CodingSourceObservationEvidence
    final_source: CodingSourceObservationEvidence
    runtime: CodingRuntimeAuthority
    checks: tuple[CodingCheckEvidence, ...] = ()
    commands: tuple[CodingCommandEvidence, ...] = ()
    mutations: tuple[CodingMutationEvidence, ...] = ()
    git_status: CodingGitStatusEvidence | None = None
    git_summary: CodingGitSummaryEvidence | None = None
    git: CodingGitEvidence | None = None
    review: CodingReviewSettlement
    publication: CodingPublicationEvidence
    lifecycle_receipt_ids: tuple[str, ...] = ()
    external_delivery_performed: StrictBool = False
    remaining_delivery_action: Literal["commit_push_or_provider_delivery"] = (
        "commit_push_or_provider_delivery"
    )
    warnings: tuple[str, ...] = (_TRUSTED_DOCKER_WARNING,)

    @field_validator(
        "product_run_id",
        "session_id",
        "agent_name",
        "task_id",
        "source_origin_id",
        "source_workspace_id",
        "source_destination_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validated_identifier(value, info.field_name)

    @field_validator("interaction_id")
    @classmethod
    def validate_optional_interaction_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validated_identifier(value, "interaction_id")

    @field_validator("request_fingerprint", "initial_revision", "final_revision")
    @classmethod
    def validate_optional_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validated_fingerprint(value, info.field_name)

    @field_validator("checks")
    @classmethod
    def validate_check_order(cls, value: tuple[CodingCheckEvidence, ...]):
        names = tuple(item.check for item in value)
        if names != tuple(sorted(set(names))):
            raise ValueError("checks must contain one canonical result per check.")
        return value

    @field_validator("lifecycle_receipt_ids")
    @classmethod
    def validate_lifecycle_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > CODING_PRODUCT_MAX_LIFECYCLE_RECEIPTS:
            raise ValueError("lifecycle_receipt_ids exceeds the durable lifecycle bound.")
        return tuple(_validated_identifier(item, "lifecycle_receipt_ids item") for item in value)

    @model_validator(mode="after")
    def validate_patch_ready_claim(self) -> CodingProductCandidate:
        if self.external_delivery_performed:
            raise ValueError("The core coding product cannot claim an external delivery effect.")
        if self.warnings != (_TRUSTED_DOCKER_WARNING,):
            raise ValueError("Coding-product results require the exact trusted-Docker warning.")
        if self.publication.baseline_revision != self.initial_revision:
            raise ValueError("Publication evidence conflicts with the admitted source baseline.")
        if self.publication.destination_id != self.source_destination_id:
            raise ValueError("Publication evidence conflicts with the admitted destination.")
        if not self.initial_git.clean:
            raise ValueError("Coding-product source Git authority must be clean.")
        if (
            self.publication.final_revision is not None
            and self.publication.final_revision != self.final_revision
        ):
            raise ValueError("Publication evidence conflicts with the final source observation.")
        if (
            self.initial_source.workspace_id != self.source_workspace_id
            or self.initial_source.revision != self.initial_revision
        ):
            raise ValueError("Initial source evidence conflicts with admitted source authority.")
        if (
            self.final_source.workspace_id != self.source_workspace_id
            or self.final_source.revision != self.final_revision
        ):
            raise ValueError("Final source evidence conflicts with observed source authority.")
        if self.state is CodingProductState.PATCH_READY_FOR_DELIVERY:
            if (
                self.interaction_id is None
                or self.final_revision is None
                or not self.publication.settled
            ):
                raise ValueError("Patch-ready success requires settled source publication.")
            if (
                self.git_status is None
                or self.git_status.scope != "all"
                or self.git_status.truncated
                or self.git_summary is None
                or self.git_summary.scope != "all"
                or self.git_summary.truncated
                or self.git is None
                or self.git.scope != "all"
                or self.git.truncated
                or self.git.binary_omitted
            ):
                raise ValueError(
                    "Patch-ready success requires complete retained Git status, summary, and "
                    "diff evidence."
                )
            if not _git_evidence_matches_source_revisions(
                initial_revision=self.initial_revision,
                final_revision=self.final_revision,
                status=self.git_status,
                summary=self.git_summary,
                diff=self.git,
            ):
                raise ValueError(
                    "Patch-ready Git evidence conflicts with the observed source revisions."
                )
            if any(
                not check.settled_pass or check.workspace_revision != self.final_revision
                for check in self.checks
            ):
                raise ValueError(
                    "Patch-ready success requires every retained check to pass against the "
                    "final source revision."
                )
            if any(not command.settled for command in self.commands):
                raise ValueError("Patch-ready success cannot contain unsettled command evidence.")
            if any(not mutation.settled for mutation in self.mutations):
                raise ValueError("Patch-ready success cannot contain unsettled mutation evidence.")
            if self.review.reviewer not in {"not_required", "passed"}:
                raise ValueError("Patch-ready success requires reviewer settlement.")
            if self.review.human not in {"not_required", "approved"}:
                raise ValueError("Patch-ready success requires human settlement.")
        encoded = _canonical_model_bytes(self, "coding_product_candidate")
        if len(encoded) > CODING_PRODUCT_MAX_RESULT_BYTES:
            raise ValueError("Coding-product candidate exceeds the result byte bound.")
        return self

    @property
    def digest(self) -> str:
        return sha256(_canonical_model_bytes(self, "coding_product_candidate")).hexdigest()


@dataclass(frozen=True)
class CodingProductPublication:
    candidate: CodingProductCandidate
    result_reference: CompletionResultReference
    evidence_reference: WorkEvidenceReference
    artifact: CodingArtifactReference


class CodingProductArtifactRepository:
    """Durable append-only lifecycle and result storage over an ArtifactStore."""

    def __init__(self, store: ArtifactStore) -> None:
        if not isinstance(store, ArtifactStore):
            raise TypeError("store must implement ArtifactStore.")
        self.store = store

    @property
    def configuration_fingerprint(self) -> str:
        return (
            "sha256:"
            + sha256(f"cayu.coding_product.artifacts.v1\0{self.store.id}".encode()).hexdigest()
        )

    def lifecycle_artifact_id(self, product_run_id: str, ordinal: int) -> str:
        return _artifact_id("coding-product-lifecycle-v1", product_run_id, str(ordinal))

    def request_artifact_id(self, product_run_id: str) -> str:
        return _artifact_id("coding-product-request-v1", product_run_id)

    def execution_claim_artifact_id(self, product_run_id: str) -> str:
        return _artifact_id("coding-product-execution-claim-v1", product_run_id)

    async def acquire_execution_claim(
        self,
        request: CodingProductRequest,
        *,
        claim_id: str,
    ) -> CodingArtifactReference:
        """Atomically fence one stable product identity before session dispatch."""

        if type(request) is not CodingProductRequest:
            raise TypeError("Execution claims require CodingProductRequest.")
        claim_id = _validated_identifier(claim_id, "claim_id")
        material = {
            "schema": "cayu.coding_product_execution_claim.v1",
            "product_run_id": request.product_run_id,
            "session_id": request.session_id,
            "request_fingerprint": request.fingerprint,
            "claim_id": claim_id,
        }
        content = canonical_durable_json_bytes(material, "coding_product_execution_claim")
        artifact_id = self.execution_claim_artifact_id(request.product_run_id)
        try:
            await self.store.put_bytes(
                content,
                artifact_id=artifact_id,
                filename="coding-product-execution-claim.json",
                content_type="application/json",
                scope=ArtifactScope.SESSION,
                session_id=request.session_id,
                metadata={
                    "schema_version": material["schema"],
                    "product_run_id": request.product_run_id,
                    "request_fingerprint": request.fingerprint,
                    "claim_id": claim_id,
                    "content_sha256": "sha256:" + sha256(content).hexdigest(),
                },
            )
        except ValueError:
            raise CodingProductReconstructionRequiredError(
                "Coding-product execution is already durably claimed."
            ) from None
        stored = self._validate_session_json_artifact(
            await self.store.read_bytes(
                artifact_id,
                max_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
            ),
            artifact_id=artifact_id,
            session_id=request.session_id,
            filename="coding-product-execution-claim.json",
        )
        if stored.content != content:
            raise CodingProductReconstructionRequiredError(
                "Coding-product execution claim acknowledgement is ambiguous."
            )
        return CodingArtifactReference(
            artifact_id=artifact_id,
            sha256="sha256:" + sha256(content).hexdigest(),
            size_bytes=len(content),
            content_type="application/json",
        )

    @staticmethod
    def _validate_session_json_artifact(
        result: ArtifactReadResult,
        *,
        artifact_id: str,
        session_id: str,
        filename: str,
    ) -> ArtifactReadResult:
        copied = copy_artifact_read_result(
            result,
            expected_artifact_id=artifact_id,
            max_content_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
        )
        content_digest = "sha256:" + sha256(copied.content).hexdigest()
        if (
            copied.truncated
            or copied.redaction_truncated
            or copied.source_bytes_read != copied.total_bytes
            or len(copied.content) != copied.total_bytes
            or copied.metadata.scope is not ArtifactScope.SESSION
            or copied.metadata.session_id != session_id
            or copied.metadata.agent_name is not None
            or copied.metadata.environment_name is not None
            or copied.metadata.filename != filename
            or copied.metadata.content_type != "application/json"
            or copied.metadata.metadata.get("content_sha256") != content_digest
        ):
            raise ValueError("Coding-product artifact authority is inconsistent.")
        return copied

    async def _read_evidence_artifact(
        self,
        reference: CodingArtifactReference,
        *,
        session_id: str,
        agent_name: str,
        environment_name: str,
        filename: str | None,
        content_type: str,
        max_bytes: int,
        expected_metadata: dict[str, object],
    ) -> bytes:
        if type(reference) is not CodingArtifactReference:
            raise TypeError("Evidence reads require CodingArtifactReference.")
        if reference.size_bytes > max_bytes:
            raise ValueError("Coding-product evidence exceeds its independent byte bound.")
        result = copy_artifact_read_result(
            await self.store.read_bytes(
                reference.artifact_id,
                max_bytes=max(1, reference.size_bytes),
            ),
            expected_artifact_id=reference.artifact_id,
            max_content_bytes=max_bytes,
        )
        metadata = result.metadata
        actual_digest = sha256(result.content).hexdigest()
        metadata_digest = metadata.metadata.get("content_sha256")
        if (
            result.truncated
            or result.redaction_truncated
            or result.source_bytes_read != result.total_bytes
            or len(result.content) != result.total_bytes
            or result.total_bytes != reference.size_bytes
            or metadata.scope is not ArtifactScope.SESSION
            or metadata.session_id != session_id
            or metadata.agent_name != agent_name
            or metadata.environment_name != environment_name
            or (filename is not None and metadata.filename != filename)
            or metadata.content_type != content_type
            or reference.content_type != content_type
            or actual_digest != reference.sha256.removeprefix("sha256:")
            or type(metadata_digest) is not str
            or metadata_digest.removeprefix("sha256:") != actual_digest
            or any(metadata.metadata.get(key) != value for key, value in expected_metadata.items())
        ):
            raise ValueError("Coding-product evidence artifact authority is inconsistent.")
        return result.content

    async def validate_candidate_artifacts(self, candidate: CodingProductCandidate) -> None:
        """Reconstruct every artifact capable of supporting a successful candidate."""

        if type(candidate) is not CodingProductCandidate:
            raise TypeError("candidate must be CodingProductCandidate.")
        for phase, source in (
            ("initial", candidate.initial_source),
            ("final", candidate.final_source),
        ):
            content = await self._read_evidence_artifact(
                source.artifact,
                session_id=candidate.session_id,
                agent_name=candidate.agent_name,
                environment_name=candidate.runtime.environment_name,
                filename=f"coding-product-source-{phase}.json",
                content_type="application/json",
                max_bytes=CODING_PRODUCT_MAX_SOURCE_ARTIFACT_BYTES,
                expected_metadata={
                    "schema_version": "cayu.coding_product_source_observation.v1",
                    "phase": phase,
                    "workspace_id": source.workspace_id,
                    "observer": source.observer,
                    "status": source.status.value,
                },
            )
            observation = WorkspaceRevisionObservation.model_validate_json(content)
            if (
                observation.identity.observer != source.observer
                or observation.identity.workspace_id != source.workspace_id
                or observation.status is not source.status
                or observation.path_scope != source.path_scope
                or observation.revision != source.revision
                or observation.total_paths != source.total_paths
            ):
                raise ValueError("Coding-product source artifact conflicts with its result.")
        if candidate.git is not None:
            await self._read_evidence_artifact(
                candidate.git.artifact,
                session_id=candidate.session_id,
                agent_name=candidate.agent_name,
                environment_name=candidate.runtime.environment_name,
                filename="coding-product.diff",
                content_type="text/x-diff",
                max_bytes=CODING_PRODUCT_MAX_GIT_DIFF_BYTES,
                expected_metadata={
                    "schema_version": "cayu.coding_product_git_diff.v1",
                    "event_id": candidate.git.event_id,
                },
            )
        for check in candidate.checks:
            if check.output_artifact is None:
                continue
            await self._read_evidence_artifact(
                check.output_artifact,
                session_id=candidate.session_id,
                agent_name=candidate.agent_name,
                environment_name=candidate.runtime.environment_name,
                filename=f"check-{check.check}-output.json",
                content_type="application/json",
                max_bytes=CODING_PRODUCT_MAX_TOOL_OUTPUT_ARTIFACT_BYTES,
                expected_metadata={
                    "operation": "run_check",
                    "check": check.check,
                    "check_profile_fingerprint": check.profile_fingerprint,
                },
            )
            if check.output_sha256 != check.output_artifact.sha256:
                raise ValueError("Coding-product check output digest is inconsistent.")
        for command in candidate.commands:
            if command.output_artifact is None:
                continue
            await self._read_evidence_artifact(
                command.output_artifact,
                session_id=candidate.session_id,
                agent_name=candidate.agent_name,
                environment_name=candidate.runtime.environment_name,
                filename=f"command-{command.selector}-output.json",
                content_type="application/json",
                max_bytes=CODING_PRODUCT_MAX_TOOL_OUTPUT_ARTIFACT_BYTES,
                expected_metadata={
                    "operation": "run_command",
                    "selector": command.selector,
                    "selector_fingerprint": command.selector_fingerprint,
                },
            )
            if command.output_sha256 != command.output_artifact.sha256:
                raise ValueError("Coding-product command output digest is inconsistent.")
        for mutation in candidate.mutations:
            if mutation.artifact is None:
                continue
            await self._read_evidence_artifact(
                mutation.artifact,
                session_id=candidate.session_id,
                agent_name=candidate.agent_name,
                environment_name=candidate.runtime.environment_name,
                filename=(
                    f"{mutation.operation_identity}-manifest.json"
                    if mutation.tool_name == "apply_patch"
                    and mutation.operation_identity is not None
                    else None
                ),
                content_type=mutation.artifact.content_type,
                max_bytes=CODING_PRODUCT_MAX_MUTATION_ARTIFACT_BYTES,
                expected_metadata={"operation": mutation.tool_name},
            )

    async def load_request(
        self,
        product_run_id: str,
        *,
        session_id: str,
    ) -> CodingProductRequest:
        """Load the exact admitted request used to recover a stable product run."""

        product_run_id = _validated_identifier(product_run_id, "product_run_id")
        session_id = _validated_identifier(session_id, "session_id")
        artifact_id = self.request_artifact_id(product_run_id)
        result = self._validate_session_json_artifact(
            await self.store.read_bytes(
                artifact_id,
                max_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
            ),
            artifact_id=artifact_id,
            session_id=session_id,
            filename="coding-product-request.json",
        )
        request = CodingProductRequest.model_validate_json(result.content)
        if request.product_run_id != product_run_id or request.session_id != session_id:
            raise ValueError("Coding-product request reconstruction is inconsistent.")
        return request

    async def ensure_request(
        self,
        request: CodingProductRequest,
    ) -> CodingArtifactReference:
        """Publish once or verify the already-admitted immutable request authority."""

        if type(request) is not CodingProductRequest:
            raise TypeError("request must be CodingProductRequest.")
        try:
            existing = await self.load_request(
                request.product_run_id,
                session_id=request.session_id,
            )
        except FileNotFoundError:
            pass
        else:
            if existing != request:
                raise CodingProductAdmissionError(
                    "Stable product identity is already bound to different authority."
                )
            content = _canonical_model_bytes(existing, "coding_product_request")
            return CodingArtifactReference(
                artifact_id=self.request_artifact_id(request.product_run_id),
                sha256="sha256:" + sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        content = _canonical_model_bytes(request, "coding_product_request")
        artifact_id = self.request_artifact_id(request.product_run_id)
        await self.store.put_bytes(
            content,
            artifact_id=artifact_id,
            filename="coding-product-request.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=request.session_id,
            metadata={
                "schema_version": request.schema_version,
                "product_run_id": request.product_run_id,
                "request_fingerprint": request.fingerprint,
                "content_sha256": "sha256:" + sha256(content).hexdigest(),
            },
        )
        stored = await self.load_request(
            request.product_run_id,
            session_id=request.session_id,
        )
        if stored != request:
            raise CodingProductAdmissionError(
                "Coding-product request publication did not preserve authority."
            )
        return CodingArtifactReference(
            artifact_id=artifact_id,
            sha256="sha256:" + sha256(content).hexdigest(),
            size_bytes=len(content),
            content_type="application/json",
        )

    async def append_lifecycle(self, receipt: CodingLifecycleReceipt) -> CodingArtifactReference:
        if type(receipt) is not CodingLifecycleReceipt:
            raise TypeError("Lifecycle publication requires CodingLifecycleReceipt.")
        content = _canonical_model_bytes(receipt, "coding_product_lifecycle_receipt")
        artifact_id = self.lifecycle_artifact_id(receipt.product_run_id, receipt.ordinal)
        await self.store.put_bytes(
            content,
            artifact_id=artifact_id,
            filename=f"coding-product-lifecycle-{receipt.ordinal}.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=receipt.session_id,
            metadata={
                "schema_version": receipt.schema_version,
                "request_fingerprint": receipt.request_fingerprint,
                "ordinal": receipt.ordinal,
                "state": receipt.state.value,
                "content_sha256": "sha256:" + sha256(content).hexdigest(),
            },
        )
        stored = self._validate_session_json_artifact(
            await self.store.read_bytes(
                artifact_id,
                max_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
            ),
            artifact_id=artifact_id,
            session_id=receipt.session_id,
            filename=f"coding-product-lifecycle-{receipt.ordinal}.json",
        )
        if stored.content != content:
            raise ValueError("Coding-product lifecycle publication is inconsistent.")
        return CodingArtifactReference(
            artifact_id=stored.metadata.id,
            sha256="sha256:" + sha256(content).hexdigest(),
            size_bytes=stored.metadata.size_bytes,
            content_type=stored.metadata.content_type,
        )

    async def load_lifecycle(
        self,
        product_run_id: str,
        *,
        session_id: str,
        request_fingerprint: str,
    ) -> tuple[tuple[CodingLifecycleReceipt, ...], tuple[str, ...]]:
        product_run_id = _validated_identifier(product_run_id, "product_run_id")
        request_fingerprint = _validated_fingerprint(
            request_fingerprint,
            "request_fingerprint",
        )
        session_id = _validated_identifier(session_id, "session_id")
        receipts: list[CodingLifecycleReceipt] = []
        artifact_ids: list[str] = []
        for ordinal in range(1, CODING_PRODUCT_MAX_LIFECYCLE_RECEIPTS + 1):
            artifact_id = self.lifecycle_artifact_id(product_run_id, ordinal)
            try:
                result = self._validate_session_json_artifact(
                    await self.store.read_bytes(
                        artifact_id,
                        max_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
                    ),
                    artifact_id=artifact_id,
                    session_id=session_id,
                    filename=f"coding-product-lifecycle-{ordinal}.json",
                )
            except FileNotFoundError:

                async def later_artifact_exists(later_ordinal: int) -> bool:
                    try:
                        await self.store.read_bytes(
                            self.lifecycle_artifact_id(product_run_id, later_ordinal),
                            max_bytes=1,
                        )
                    except FileNotFoundError:
                        return False
                    return True

                later_artifacts = await asyncio.gather(
                    *(
                        later_artifact_exists(later_ordinal)
                        for later_ordinal in range(
                            ordinal + 1,
                            CODING_PRODUCT_MAX_LIFECYCLE_RECEIPTS + 1,
                        )
                    )
                )
                if any(later_artifacts):
                    raise ValueError(
                        "Coding-product lifecycle contains a reconstruction gap."
                    ) from None
                break
            receipt = CodingLifecycleReceipt.model_validate_json(result.content)
            if (
                receipt.product_run_id != product_run_id
                or receipt.session_id != session_id
                or receipt.request_fingerprint != request_fingerprint
                or receipt.ordinal != ordinal
                or (receipts and receipt.prior_state is not receipts[-1].state)
            ):
                raise ValueError("Coding-product lifecycle reconstruction is inconsistent.")
            receipts.append(receipt)
            artifact_ids.append(artifact_id)
        return tuple(receipts), tuple(artifact_ids)

    async def publish_source_observation(
        self,
        *,
        session_id: str,
        agent_name: str,
        environment_name: str,
        phase: Literal["initial", "final"],
        observation: WorkspaceRevisionObservation,
    ) -> CodingSourceObservationEvidence:
        """Retain one bounded complete source manifest by immutable content identity."""

        if type(observation) is not WorkspaceRevisionObservation:
            raise TypeError("observation must be WorkspaceRevisionObservation.")
        content = canonical_durable_json_bytes(
            observation.model_dump(mode="json", warnings=False),
            "coding_product_source_observation",
        )
        digest = sha256(content).hexdigest()
        artifact_id = _artifact_id(
            "coding-product-source-observation-v1",
            session_id,
            phase,
            digest,
        )
        await self.store.put_bytes(
            content,
            artifact_id=artifact_id,
            filename=f"coding-product-source-{phase}.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata={
                "schema_version": "cayu.coding_product_source_observation.v1",
                "phase": phase,
                "workspace_id": observation.identity.workspace_id,
                "observer": observation.identity.observer,
                "status": observation.status.value,
                "content_sha256": "sha256:" + digest,
            },
        )
        evidence = CodingSourceObservationEvidence(
            observer=observation.identity.observer,
            workspace_id=observation.identity.workspace_id,
            status=observation.status,
            path_scope=observation.path_scope,
            revision=observation.revision,
            total_paths=observation.total_paths,
            artifact=CodingArtifactReference(
                artifact_id=artifact_id,
                sha256="sha256:" + digest,
                size_bytes=len(content),
                content_type="application/json",
            ),
        )
        await self._read_evidence_artifact(
            evidence.artifact,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            filename=f"coding-product-source-{phase}.json",
            content_type="application/json",
            max_bytes=CODING_PRODUCT_MAX_SOURCE_ARTIFACT_BYTES,
            expected_metadata={
                "schema_version": "cayu.coding_product_source_observation.v1",
                "phase": phase,
                "workspace_id": observation.identity.workspace_id,
                "observer": observation.identity.observer,
                "status": observation.status.value,
            },
        )
        return evidence

    async def publish_candidate(
        self,
        candidate: CodingProductCandidate,
    ) -> CodingProductPublication:
        if type(candidate) is not CodingProductCandidate:
            raise TypeError("Result publication requires CodingProductCandidate.")
        candidate = CodingProductCandidate.model_validate(candidate.model_dump(mode="python"))
        content = _canonical_model_bytes(candidate, "coding_product_candidate")
        if len(content) > CODING_PRODUCT_MAX_RESULT_BYTES:
            raise ValueError("Coding-product candidate exceeds the publication bound.")
        digest = sha256(content).hexdigest()
        artifact_id = _artifact_id(
            "coding-product-result-v1",
            candidate.request_fingerprint,
            digest,
        )
        await self.store.put_bytes(
            content,
            artifact_id=artifact_id,
            filename="coding-product-result.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=candidate.session_id,
            metadata={
                "schema_version": candidate.schema_version,
                "product_run_id": candidate.product_run_id,
                "request_fingerprint": candidate.request_fingerprint,
                "state": candidate.state.value,
                "content_sha256": "sha256:" + digest,
            },
        )
        stored_result = self._validate_session_json_artifact(
            await self.store.read_bytes(
                artifact_id,
                max_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
            ),
            artifact_id=artifact_id,
            session_id=candidate.session_id,
            filename="coding-product-result.json",
        )
        if stored_result.content != content:
            raise ValueError("Coding-product result publication is inconsistent.")
        artifact = CodingArtifactReference(
            artifact_id=stored_result.metadata.id,
            sha256="sha256:" + digest,
            size_bytes=stored_result.metadata.size_bytes,
            content_type=stored_result.metadata.content_type,
        )
        publication = CodingProductPublication(
            candidate=candidate,
            result_reference=CompletionResultReference(
                kind=CODING_PRODUCT_RESULT_KIND,
                reference_id=artifact_id,
                digest=digest,
            ),
            evidence_reference=WorkEvidenceReference(
                kind=CODING_PRODUCT_EVIDENCE_KIND,
                reference_id=artifact_id,
                version="v1",
                digest=digest,
                available=True,
            ),
            artifact=artifact,
        )
        stored = await self.read_candidate(publication.result_reference)
        if stored != candidate:
            raise ValueError("Coding-product result publication is inconsistent.")
        return publication

    async def read_candidate(
        self,
        reference: CompletionResultReference,
    ) -> CodingProductCandidate:
        if type(reference) is not CompletionResultReference:
            raise TypeError("Candidate reads require CompletionResultReference.")
        if reference.kind != CODING_PRODUCT_RESULT_KIND:
            raise ValueError("Completion result does not reference a coding-product result.")
        result = copy_artifact_read_result(
            await self.store.read_bytes(
                reference.reference_id,
                max_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
            ),
            expected_artifact_id=reference.reference_id,
            max_content_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
        )
        if result.truncated or sha256(result.content).hexdigest() != reference.digest:
            raise ValueError("Coding-product result content does not match its durable reference.")
        candidate = CodingProductCandidate.model_validate_json(result.content)
        expected_artifact_id = _artifact_id(
            "coding-product-result-v1",
            candidate.request_fingerprint,
            reference.digest,
        )
        if expected_artifact_id != reference.reference_id:
            raise ValueError("Coding-product result reference is not content-addressed.")
        self._validate_session_json_artifact(
            result,
            artifact_id=expected_artifact_id,
            session_id=candidate.session_id,
            filename="coding-product-result.json",
        )
        await self.validate_candidate_artifacts(candidate)
        return candidate

    async def load_publication(
        self,
        *,
        request_fingerprint: str,
        digest: str,
    ) -> CodingProductPublication:
        """Reconstruct one content-addressed result after acknowledgement loss."""

        request_fingerprint = _validated_fingerprint(
            request_fingerprint,
            "request_fingerprint",
        )
        digest = _validated_fingerprint(digest, "digest").removeprefix("sha256:")
        artifact_id = _artifact_id(
            "coding-product-result-v1",
            request_fingerprint,
            digest,
        )
        reference = CompletionResultReference(
            kind=CODING_PRODUCT_RESULT_KIND,
            reference_id=artifact_id,
            digest=digest,
        )
        candidate = await self.read_candidate(reference)
        if candidate.request_fingerprint != request_fingerprint:
            raise ValueError("Coding-product result conflicts with reconstruction authority.")
        result = self._validate_session_json_artifact(
            await self.store.read_bytes(
                artifact_id,
                max_bytes=CODING_PRODUCT_MAX_RESULT_BYTES,
            ),
            artifact_id=artifact_id,
            session_id=candidate.session_id,
            filename="coding-product-result.json",
        )
        artifact = CodingArtifactReference(
            artifact_id=artifact_id,
            sha256="sha256:" + digest,
            size_bytes=result.metadata.size_bytes,
            content_type=result.metadata.content_type,
        )
        return CodingProductPublication(
            candidate=candidate,
            result_reference=reference,
            evidence_reference=WorkEvidenceReference(
                kind=CODING_PRODUCT_EVIDENCE_KIND,
                reference_id=artifact_id,
                version="v1",
                digest=digest,
                available=True,
            ),
            artifact=artifact,
        )

    async def publish_git_diff(
        self,
        *,
        session_id: str,
        agent_name: str,
        environment_name: str,
        event_id: str,
        content: str,
    ) -> CodingArtifactReference:
        encoded = content.encode("utf-8")
        if len(encoded) > CODING_PRODUCT_MAX_GIT_DIFF_BYTES:
            raise ValueError("Coding-product Git diff exceeds its independent byte bound.")
        digest = sha256(encoded).hexdigest()
        artifact_id = _artifact_id("coding-product-git-diff-v1", session_id, event_id, digest)
        metadata = await self.store.put_bytes(
            encoded,
            artifact_id=artifact_id,
            filename="coding-product.diff",
            content_type="text/x-diff",
            scope=ArtifactScope.SESSION,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata={
                "schema_version": "cayu.coding_product_git_diff.v1",
                "event_id": event_id,
                "content_sha256": "sha256:" + digest,
            },
        )
        reference = CodingArtifactReference(
            artifact_id=metadata.id,
            sha256="sha256:" + digest,
            size_bytes=metadata.size_bytes,
            content_type=metadata.content_type,
        )
        await self._read_evidence_artifact(
            reference,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            filename="coding-product.diff",
            content_type="text/x-diff",
            max_bytes=CODING_PRODUCT_MAX_GIT_DIFF_BYTES,
            expected_metadata={
                "schema_version": "cayu.coding_product_git_diff.v1",
                "event_id": event_id,
            },
        )
        return reference


def coding_product_work_contract(
    request: CodingProductRequest,
    repository: CodingProductArtifactRepository,
) -> WorkContract:
    """Build the exact WorkContract used by the product verifier and resolver."""

    if type(request) is not CodingProductRequest:
        raise TypeError("Coding-product contracts require CodingProductRequest.")
    if type(repository) is not CodingProductArtifactRepository:
        raise TypeError("repository must be CodingProductArtifactRepository.")
    configuration = (
        "sha256:"
        + sha256(
            (
                "cayu.coding_product.contract.v1\0"
                + request.fingerprint
                + "\0"
                + repository.configuration_fingerprint
            ).encode("utf-8")
        ).hexdigest()
    )
    verifier = CompletionVerifierRef(
        verifier_id="cayu-coding-product",
        version="v1",
        kind=CompletionVerifierKind.DETERMINISTIC,
        configuration_fingerprint=configuration.removeprefix("sha256:"),
    )
    resolver = CompletionResultResolverRef(
        resolver_id="cayu-coding-product-result",
        version="v1",
        configuration_fingerprint=configuration.removeprefix("sha256:"),
    )
    requirements = (
        WorkEvidenceRequirement(
            requirement_id="checks",
            kind="check_receipts",
            description="Every application-required check settled successfully.",
        ),
        WorkEvidenceRequirement(
            requirement_id="diff",
            kind="git_diff",
            description="A complete bounded final Git diff was retained.",
        ),
        WorkEvidenceRequirement(
            requirement_id="publication",
            kind="source_publication",
            description="Revision-aware source publication settled authoritatively.",
        ),
        WorkEvidenceRequirement(
            requirement_id="review",
            kind="review_settlement",
            description="Configured reviewer and human gates settled.",
        ),
        WorkEvidenceRequirement(
            requirement_id="source",
            kind="source_manifest",
            description="Initial and final source identities remain application-bound.",
        ),
    )
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id=f"coding-product-{sha256(request.product_run_id.encode()).hexdigest()[:24]}",
            version=1,
            objective="Produce an authoritative patch_ready_for_delivery result.",
            criteria=(
                WorkCriterion(
                    criterion_id="patch_ready_for_delivery",
                    ordinal=1,
                    description=(
                        "The bounded trusted-repository coding result is ready for optional "
                        "external delivery."
                    ),
                    evidence_requirement_ids=(
                        "checks",
                        "diff",
                        "publication",
                        "review",
                        "source",
                    ),
                ),
            ),
            constraints=(
                WorkConstraint(
                    constraint_id="delivery_boundary",
                    description="Core coding performs no commit, push, PR, CI, or merge effect.",
                ),
                WorkConstraint(
                    constraint_id="trusted_docker_boundary",
                    description=(
                        "Execution remains in the admitted no-network trusted-repository Docker "
                        "profile without credentials."
                    ),
                ),
            ),
            evidence_requirements=requirements,
            verifier=verifier,
            result_resolver=resolver,
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=3,
                max_repeated_gap_count=2,
            ),
        )
    )


class CodingProductCompletionVerifier(DeterministicCompletionVerifier):
    """Deterministically verify one exact candidate artifact."""

    def __init__(
        self,
        request: CodingProductRequest,
        repository: CodingProductArtifactRepository,
        *,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    ) -> None:
        if type(request) is not CodingProductRequest:
            raise TypeError("request must be CodingProductRequest.")
        if type(repository) is not CodingProductArtifactRepository:
            raise TypeError("repository must be CodingProductArtifactRepository.")
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError(
                "public_authority_alias_codec must be PublicAuthorityAliasCodec or None."
            )
        self.request = request
        self.repository = repository
        self.public_authority_alias_codec = public_authority_alias_codec

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="cayu.coding_product.completion_verifier",
            behavior_version="2",
            implementation_version="1",
        )

    async def verify(self, request: CompletionVerifierRequest) -> CompletionVerifierDecision:
        candidate = await self.repository.read_candidate(request.proposal.result)
        contract = coding_product_work_contract(self.request, self.repository)
        if request.contract != contract:
            raise ValueError("Coding-product verifier received another work contract.")
        return coding_product_completion_decision(
            self.request,
            candidate,
            public_authority_alias_codec=self.public_authority_alias_codec,
            evidence=WorkEvidenceReference(
                kind=CODING_PRODUCT_EVIDENCE_KIND,
                reference_id=request.proposal.result.reference_id,
                version="v1",
                digest=request.proposal.result.digest,
                available=True,
            ),
        )


class CodingProductResultResolver(CompletionResultResolver):
    """Resolve only the accepted immutable coding-product result artifact."""

    def __init__(
        self,
        request: CodingProductRequest,
        repository: CodingProductArtifactRepository,
    ) -> None:
        if type(request) is not CodingProductRequest:
            raise TypeError("request must be CodingProductRequest.")
        if type(repository) is not CodingProductArtifactRepository:
            raise TypeError("repository must be CodingProductArtifactRepository.")
        self.request = request
        self.repository = repository

    async def resolve(self, request: CompletionResultResolverRequest) -> dict[str, object]:
        candidate = await self.repository.read_candidate(request.result_reference)
        if (
            candidate.request_fingerprint != self.request.fingerprint
            or candidate.state is not CodingProductState.PATCH_READY_FOR_DELIVERY
        ):
            raise ValueError("Accepted coding-product result is no longer reconstructable.")
        return cast("dict[str, object]", candidate.model_dump(mode="json", warnings=False))


def coding_product_completion_decision(
    request: CodingProductRequest,
    candidate: CodingProductCandidate,
    *,
    evidence: WorkEvidenceReference,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
) -> CompletionVerifierDecision:
    """Evaluate exact candidate evidence without granting application authority."""

    if type(request) is not CodingProductRequest:
        raise TypeError("request must be CodingProductRequest.")
    if type(candidate) is not CodingProductCandidate:
        raise TypeError("candidate must be CodingProductCandidate.")
    if type(evidence) is not WorkEvidenceReference:
        raise TypeError("evidence must be WorkEvidenceReference.")
    if public_authority_alias_codec is not None and not isinstance(
        public_authority_alias_codec,
        PublicAuthorityAliasCodec,
    ):
        raise TypeError("public_authority_alias_codec must be PublicAuthorityAliasCodec or None.")
    authority_matches = (
        candidate.product_run_id == request.product_run_id
        and candidate.request_fingerprint == request.fingerprint
        and candidate.session_id == request.session_id
        and candidate.agent_name == request.agent_name
        and candidate.task_id == request.task.task_id
        and candidate.source_origin_id == request.source.origin_id
        and candidate.source_workspace_id == request.source.workspace_id
        and candidate.source_destination_id == request.source.destination_id
        and candidate.initial_revision == request.source.baseline_revision
        and candidate.initial_git == request.source.git_baseline
        and candidate.runtime == request.runtime
        and _workspace_public_authority_matches(
            candidate.publication.destination_workspace_id,
            request.source.workspace_id,
            session_id=request.session_id,
            public_authority_alias_codec=public_authority_alias_codec,
        )
    )
    required_checks = {check.check: check for check in candidate.checks}
    checks_pass = set(required_checks) == set(request.settlement.required_checks) and all(
        required_checks[name].settled_pass
        and required_checks[name].workspace_revision == candidate.final_revision
        for name in request.settlement.required_checks
    )
    reviewer_pass = (
        candidate.review.reviewer == "passed"
        if request.settlement.reviewer_required
        else candidate.review.reviewer in {"not_required", "passed"}
    )
    human_pass = (
        candidate.review.human == "approved"
        if request.settlement.human_approval_required
        else candidate.review.human in {"not_required", "approved"}
    )
    patch_ready = (
        authority_matches
        and candidate.interaction_id is not None
        and checks_pass
        and reviewer_pass
        and human_pass
        and candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
        and candidate.publication.settled
        and candidate.git_status is not None
        and not candidate.git_status.truncated
        and candidate.git_summary is not None
        and not candidate.git_summary.truncated
        and candidate.git is not None
        and not candidate.git.truncated
        and not candidate.git.binary_omitted
        and all(command.settled for command in candidate.commands)
        and all(mutation.settled for mutation in candidate.mutations)
    )
    trusted_boundary = (
        candidate.runtime.execution_kind == "trusted_repository_docker"
        and candidate.runtime.network == "none"
        and candidate.runtime.credential_visibility == "non_possession"
    )
    delivery_boundary = not candidate.external_delivery_performed

    outcome_values = (
        (
            "criterion",
            "patch_ready_for_delivery",
            patch_ready,
            "coding.patch_ready" if patch_ready else "coding.patch_not_ready",
        ),
        (
            "constraint",
            "delivery_boundary",
            delivery_boundary,
            "coding.delivery_absent" if delivery_boundary else "coding.delivery_performed",
        ),
        (
            "constraint",
            "trusted_docker_boundary",
            trusted_boundary,
            "coding.trusted_docker" if trusted_boundary else "coding.runtime_drift",
        ),
    )
    criterion_outcomes = tuple(
        CompletionCriterionOutcome(
            criterion_id=subject_id,
            status=(
                CriterionOutcomeStatus.SATISFIED
                if satisfied
                else CriterionOutcomeStatus.UNSATISFIED
            ),
            reason_code=reason,
            satisfaction_basis=(CompletionSatisfactionBasis.EVIDENCE if satisfied else None),
            evidence_references=(evidence,),
        )
        for kind, subject_id, satisfied, reason in outcome_values
        if kind == "criterion"
    )
    constraint_outcomes = tuple(
        CompletionConstraintOutcome(
            constraint_id=subject_id,
            status=(
                CriterionOutcomeStatus.SATISFIED
                if satisfied
                else CriterionOutcomeStatus.UNSATISFIED
            ),
            reason_code=reason,
            satisfaction_basis=(CompletionSatisfactionBasis.EVIDENCE if satisfied else None),
            evidence_references=(evidence,),
        )
        for kind, subject_id, satisfied, reason in outcome_values
        if kind == "constraint"
    )
    gaps = tuple(
        CompletionGap(
            criterion_id=subject_id if kind == "criterion" else None,
            constraint_id=subject_id if kind == "constraint" else None,
            code=reason,
            evidence_requirement_ids=(
                ("checks", "diff", "publication", "review", "source") if kind == "criterion" else ()
            ),
        )
        for kind, subject_id, satisfied, reason in outcome_values
        if not satisfied
    )
    verdict = (
        CompletionVerdict.ACCEPTED
        if patch_ready and trusted_boundary and delivery_boundary
        else CompletionVerdict.NEEDS_REVIEW
        if candidate.state
        in {CodingProductState.REVIEW_REQUIRED, CodingProductState.HUMAN_INPUT_REQUIRED}
        else CompletionVerdict.BLOCKED
        if candidate.state
        in {
            CodingProductState.BLOCKED,
            CodingProductState.DENIED,
            CodingProductState.SOURCE_CONFLICT,
            CodingProductState.TOOLCHAIN_REBUILD_REQUIRED,
            CodingProductState.RECONSTRUCTION_REQUIRED,
        }
        else CompletionVerdict.REJECTED
    )
    return CompletionVerifierDecision(
        verdict=verdict,
        criterion_outcomes=criterion_outcomes,
        constraint_outcomes=constraint_outcomes,
        gaps=gaps,
        evidence_references=(evidence,),
    )


async def register_coding_product_contract(
    app: CayuApp,
    request: CodingProductRequest,
    repository: CodingProductArtifactRepository,
) -> WorkContract:
    """Publish and reconstruct the exact verifier/resolver during app startup."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be CayuApp.")
    contract = coding_product_work_contract(request, repository)
    app.register_completion_verifier(
        contract.verifier,
        CodingProductCompletionVerifier(
            request,
            repository,
            public_authority_alias_codec=app.session_store.public_authority_alias_codec,
        ),
    )
    app.register_completion_result_resolver(
        contract.result_resolver,
        CodingProductResultResolver(request, repository),
    )
    published = await app.create_work_contract(
        WorkContractDraft.model_validate(
            contract.model_dump(mode="python", exclude={"fingerprint"})
        )
    )
    if published != contract:
        raise ValueError("Task store published a different coding-product contract.")
    return published


def _event_bytes(event: Event) -> int:
    return len(event.model_dump_json().encode("utf-8"))


def _structured_result(event: Event) -> dict[str, Any] | None:
    result = event.payload.get("result")
    if type(result) is not dict:
        return None
    structured = result.get("structured")
    if type(structured) is not dict:
        return None
    return copy_durable_json_object(structured, "coding_product_tool_result")


def _optional_text(value: object) -> str | None:
    return value if type(value) is str and value.strip() else None


def _optional_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _optional_bool(value: object) -> bool:
    return value if type(value) is bool else False


def _output_artifact_from_structured(
    structured: dict[str, Any],
    *,
    operation: Literal["run_check", "run_command"],
) -> CodingArtifactReference | None:
    artifacts = structured.get("artifacts")
    if type(artifacts) is not list:
        return None
    for artifact in reversed(artifacts):
        if type(artifact) is dict:
            artifact = cast("dict[str, Any]", artifact)
            metadata = artifact.get("metadata")
            if type(metadata) is not dict or metadata.get("operation") != operation:
                continue
            artifact_id = _optional_text(artifact.get("id") or artifact.get("artifact_id"))
            digest = _optional_text(metadata.get("content_sha256"))
            size_bytes = artifact.get("size_bytes")
            content_type = _optional_text(artifact.get("content_type"))
            if (
                artifact_id is not None
                and digest is not None
                and type(size_bytes) is int
                and size_bytes >= 0
                and content_type is not None
            ):
                return CodingArtifactReference(
                    artifact_id=artifact_id,
                    sha256=digest,
                    size_bytes=size_bytes,
                    content_type=content_type,
                )
    return None


def _check_evidence(event: Event, structured: dict[str, Any]) -> CodingCheckEvidence | None:
    check = _optional_text(structured.get("check"))
    profile = _optional_text(structured.get("check_profile_fingerprint"))
    status = _optional_text(structured.get("status"))
    if check is None or profile is None or status is None:
        return None
    return CodingCheckEvidence(
        check=check,
        event_id=event.id,
        profile_fingerprint=profile,
        status=status,
        exit_code=(structured["exit_code"] if type(structured.get("exit_code")) is int else None),
        output_sha256=_optional_text(structured.get("output_sha256")),
        output_artifact=_output_artifact_from_structured(
            structured,
            operation="run_check",
        ),
        duration_ms=_optional_int(structured.get("duration_ms")),
        stdout_truncated=_optional_bool(structured.get("stdout_truncated")),
        stderr_truncated=_optional_bool(structured.get("stderr_truncated")),
        stdout_runner_truncated=_optional_bool(structured.get("stdout_runner_truncated")),
        stderr_runner_truncated=_optional_bool(structured.get("stderr_runner_truncated")),
        stdout_projection_truncated=_optional_bool(structured.get("stdout_projection_truncated")),
        stderr_projection_truncated=_optional_bool(structured.get("stderr_projection_truncated")),
        output_artifact_status=_optional_text(structured.get("output_artifact_status")),
        timed_out=_optional_bool(structured.get("timed_out")),
        cancelled=_optional_bool(structured.get("cancelled")),
        workspace_mutation_settlement=_optional_text(
            structured.get("workspace_mutation_settlement")
        ),
        workspace_revision=_optional_text(structured.get("workspace_revision")),
    )


def _command_evidence(
    event: Event,
    structured: dict[str, Any],
) -> CodingCommandEvidence | None:
    selector = _optional_text(structured.get("selector"))
    profile = _optional_text(structured.get("selector_fingerprint"))
    status = _optional_text(structured.get("status"))
    if selector is None or profile is None or status is None:
        return None
    return CodingCommandEvidence(
        selector=selector,
        event_id=event.id,
        selector_fingerprint=profile,
        argv_sha256=_optional_text(structured.get("argv_sha256")),
        status=status,
        exit_code=(structured["exit_code"] if type(structured.get("exit_code")) is int else None),
        duration_ms=_optional_int(structured.get("duration_ms")),
        output_sha256=_optional_text(structured.get("output_sha256")),
        output_artifact=_output_artifact_from_structured(
            structured,
            operation="run_command",
        ),
        output_truncated=(
            _optional_bool(structured.get("stdout_truncated"))
            or _optional_bool(structured.get("stderr_truncated"))
        ),
        output_runner_truncated=(
            _optional_bool(structured.get("stdout_runner_truncated"))
            or _optional_bool(structured.get("stderr_runner_truncated"))
        ),
        output_projection_truncated=(
            _optional_bool(structured.get("stdout_projection_truncated"))
            or _optional_bool(structured.get("stderr_projection_truncated"))
        ),
        output_artifact_status=_optional_text(structured.get("output_artifact_status")),
        exit_code_admitted=_optional_bool(structured.get("exit_code_admitted")),
        output_collection_complete=_optional_bool(structured.get("output_collection_complete")),
        output_publication_complete=_optional_bool(structured.get("output_publication_complete")),
        timed_out=_optional_bool(structured.get("timed_out")),
        cancelled=_optional_bool(structured.get("cancelled")),
        workspace_mutation_settlement=_optional_text(
            structured.get("workspace_mutation_settlement")
        ),
    )


def _structured_artifact(structured: dict[str, Any]) -> CodingArtifactReference | None:
    artifact = structured.get("artifact")
    if type(artifact) is not dict:
        return None
    artifact_id = _optional_text(artifact.get("artifact_id") or artifact.get("id"))
    digest = _optional_text(artifact.get("sha256") or artifact.get("content_sha256"))
    size = artifact.get("size_bytes")
    if artifact_id is None or digest is None or type(size) is not int or size < 0:
        return None
    return CodingArtifactReference(
        artifact_id=artifact_id,
        sha256=digest,
        size_bytes=size,
        content_type=_optional_text(artifact.get("content_type")) or "application/json",
    )


def _git_entries(value: object) -> tuple[CodingGitEntry, ...] | None:
    if type(value) is not list:
        return None
    entries: list[CodingGitEntry] = []
    for raw in value:
        if type(raw) is not dict:
            return None
        raw = cast("dict[str, Any]", raw)
        path = _optional_text(raw.get("path"))
        index = raw.get("index")
        worktree = raw.get("worktree")
        renamed_from = raw.get("original_path")
        if (
            path is None
            or type(index) is not str
            or type(worktree) is not str
            or (renamed_from is not None and _optional_text(renamed_from) is None)
        ):
            return None
        entries.append(
            CodingGitEntry(
                path=path,
                index=index,
                worktree=worktree,
                renamed_from=renamed_from,
            )
        )
    return tuple(entries)


def _git_summary_entries(value: object) -> tuple[CodingGitSummaryEntry, ...] | None:
    status_entries = _git_entries(value)
    if status_entries is None or type(value) is not list:
        return None
    summaries: list[CodingGitSummaryEntry] = []
    for raw, entry in zip(value, status_entries, strict=True):
        if type(raw) is not dict:
            return None
        raw = cast("dict[str, Any]", raw)
        additions = raw.get("additions")
        deletions = raw.get("deletions")
        count_kind = raw.get("count_kind")
        if (
            (additions is not None and (type(additions) is not int or additions < 0))
            or (deletions is not None and (type(deletions) is not int or deletions < 0))
            or count_kind not in {"text", "binary", "untracked", "unknown"}
        ):
            return None
        summaries.append(
            CodingGitSummaryEntry(
                **entry.model_dump(mode="python"),
                additions=additions,
                deletions=deletions,
                count_kind=count_kind,
            )
        )
    return tuple(summaries)


def _git_result_truncated(structured: dict[str, Any]) -> bool:
    reasons = structured.get("truncation_reasons")
    offset = structured.get("offset")
    if (
        structured.get("truncated") is not False
        or type(reasons) is not list
        or bool(reasons)
        or type(offset) is not int
        or offset != 0
        or "next_offset" not in structured
        or structured["next_offset"] is not None
    ):
        return True
    if structured.get("mode") != "diff":
        return False
    diff_offset = structured.get("diff_offset")
    return (
        type(diff_offset) is not int
        or diff_offset != 0
        or "next_diff_offset" not in structured
        or structured["next_diff_offset"] is not None
    )


def _final_git_receipt(
    events: Sequence[Event],
    request: CodingProductRequest,
    *,
    final_revision: str | None,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> tuple[int, Event, dict[str, Any], dict[str, Any], dict[str, Any], str] | None:
    finalizations = tuple(
        (index, event)
        for index, event in enumerate(events)
        if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    )
    if len(finalizations) != 1:
        return None
    event_index, event = finalizations[0]
    receipt = event.payload.get("final_git_receipt")
    if type(receipt) is not dict or set(receipt) != {
        "schema",
        "receipt_sha256",
        "request_fingerprint",
        "destination_workspace_id",
        "workload_workspace_id",
        "baseline_revision",
        "workspace_revision",
        "status",
        "summary",
        "diff",
    }:
        return None
    receipt_sha256 = _optional_text(receipt.get("receipt_sha256"))
    material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_digest = (
        "sha256:" + sha256(canonical_durable_json_bytes(material, "final_git_receipt")).hexdigest()
    )
    status = receipt.get("status")
    summary = receipt.get("summary")
    diff = receipt.get("diff")
    status_structured = status.get("structured") if type(status) is dict else None
    summary_structured = summary.get("structured") if type(summary) is dict else None
    diff_structured = diff.get("structured") if type(diff) is dict else None
    diff_content = diff.get("content") if type(diff) is dict else None
    destination_workspace_id = _optional_text(receipt.get("destination_workspace_id"))
    workload_workspace_id = _optional_text(receipt.get("workload_workspace_id"))
    publication_receipt = event.payload.get("source_publication_receipt")
    publication_destination_workspace_id = (
        _optional_text(publication_receipt.get("destination_workspace_id"))
        if type(publication_receipt) is dict
        else None
    )
    publication_workload_workspace_id = (
        _optional_text(publication_receipt.get("workload_workspace_id"))
        if type(publication_receipt) is dict
        else None
    )
    if (
        receipt.get("schema") != CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA
        or receipt_sha256 != expected_digest
        or receipt.get("request_fingerprint") != request.fingerprint
        or receipt.get("baseline_revision") != request.source.baseline_revision
        or receipt.get("workspace_revision") != final_revision
        or destination_workspace_id is None
        or not _workspace_public_authority_matches(
            destination_workspace_id,
            request.source.workspace_id,
            session_id=request.session_id,
            public_authority_alias_codec=public_authority_alias_codec,
        )
        or workload_workspace_id is None
        or workload_workspace_id == destination_workspace_id
        or destination_workspace_id != publication_destination_workspace_id
        or workload_workspace_id != publication_workload_workspace_id
        or type(status) is not dict
        or set(status) != {"structured"}
        or type(status_structured) is not dict
        or not is_final_git_result_envelope(status_structured, mode="status")
        or type(summary) is not dict
        or set(summary) != {"structured"}
        or type(summary_structured) is not dict
        or not is_final_git_result_envelope(summary_structured, mode="summary")
        or type(diff) is not dict
        or set(diff) != {"content", "structured"}
        or type(diff_structured) is not dict
        or not is_final_git_result_envelope(diff_structured, mode="diff")
        or type(diff_content) is not str
        or REDACTED_SECRET in diff_content
    ):
        return None
    return (
        event_index,
        event,
        status_structured,
        summary_structured,
        diff_structured,
        diff_content,
    )


def _mutation_evidence(event: Event, structured: dict[str, Any]) -> CodingMutationEvidence:
    outcome = _optional_text(structured.get("outcome"))
    if outcome is None:
        outcome = "completed" if event.type == EventType.TOOL_CALL_COMPLETED else "failed"
    operation_identity = _optional_text(
        structured.get("patch_id")
        or structured.get("tool_call_identity")
        or event.payload.get("idempotency_key")
    )
    evidence_sha256 = (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(structured, "coding_product_mutation_evidence")
        ).hexdigest()
    )
    artifact = _structured_artifact(structured)
    artifact_required = _optional_bool(structured.get("manifest_truncated")) or _optional_bool(
        structured.get("diff_truncated")
    )
    return CodingMutationEvidence(
        tool_name=event.tool_name or "unknown_mutation",
        event_id=event.id,
        outcome=outcome,
        operation_identity=operation_identity,
        artifact=artifact,
        evidence_sha256=evidence_sha256,
        requires_fresh_read=(
            _optional_bool(structured.get("requires_fresh_read"))
            or outcome in {"partial", "ambiguous", "cancelled", "failed"}
            or (
                artifact_required
                and (structured.get("artifact_status") != "stored" or artifact is None)
            )
        ),
    )


def _review_settlement(
    events: Sequence[Event],
    policy: CodingSettlementPolicy,
    recorded: CodingReviewSettlement | None,
) -> CodingReviewSettlement:
    by_id = {event.id: event for event in events}
    if recorded is not None:
        if type(recorded) is not CodingReviewSettlement:
            raise TypeError("review_settlement must be CodingReviewSettlement.")
        if recorded.reviewer == "passed":
            reviewer_event = by_id.get(recorded.reviewer_event_id or "")
            if (
                reviewer_event is None
                or reviewer_event.tool_name != "subagent_result"
                or reviewer_event.type != EventType.TOOL_CALL_COMPLETED
            ):
                raise CodingProductEvidenceError(
                    "Positive reviewer settlement requires its completed result event."
                )
        if recorded.human == "approved":
            human_event = by_id.get(recorded.human_event_id or "")
            if (
                human_event is None
                or human_event.tool_name != "ask_user"
                or human_event.type != EventType.TOOL_CALL_COMPLETED
            ):
                raise CodingProductEvidenceError(
                    "Positive human settlement requires its completed input event."
                )
        if policy.reviewer_required and recorded.reviewer == "not_required":
            raise CodingProductEvidenceError("Required reviewer settlement cannot be omitted.")
        if policy.human_approval_required and recorded.human == "not_required":
            raise CodingProductEvidenceError("Required human settlement cannot be omitted.")
        return recorded
    reviewer_failed = any(
        event.tool_name in {"subagent", "subagent_result"}
        and event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED}
        for event in events
    )
    human_denied = any(
        event.tool_name == "ask_user"
        and event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED}
        for event in events
    )
    return CodingReviewSettlement(
        reviewer=(
            "not_required"
            if not policy.reviewer_required
            else "failed"
            if reviewer_failed
            else "required"
        ),
        human=(
            "not_required"
            if not policy.human_approval_required
            else "denied"
            if human_denied
            else "required"
        ),
    )


def _workspace_public_authority_matches(
    public_or_private_workspace_id: str | None,
    expected_private_workspace_id: str,
    *,
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> bool:
    if public_or_private_workspace_id == expected_private_workspace_id:
        return True
    if public_or_private_workspace_id is None or public_authority_alias_codec is None:
        return False
    return public_authority_alias_codec.matches(
        public_or_private_workspace_id,
        expected_private_workspace_id,
        field_name="workspace_observation_workspace_id",
        session_id=session_id,
    )


def _publication_evidence(
    events: Sequence[Event],
    *,
    destination_id: str,
    destination_workspace_id: str,
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    initial_revision: str,
    final_revision: str | None,
) -> CodingPublicationEvidence:
    completed = next(
        (
            event
            for event in reversed(events)
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
        ),
        None,
    )
    if completed is not None:
        receipt = completed.payload.get("source_publication_receipt")
        receipt = receipt if type(receipt) is dict else {}
        copied = _optional_int(receipt.get("copied_files"))
        copied_bytes = _optional_int(receipt.get("copied_bytes"))
        deleted = _optional_int(receipt.get("deleted_files"))
        receipt_sha256 = _optional_text(receipt.get("receipt_sha256"))
        snapshot_sha256 = _optional_text(receipt.get("snapshot_sha256"))
        receipt_destination_workspace_id = _optional_text(receipt.get("destination_workspace_id"))
        workload_workspace_id = _optional_text(receipt.get("workload_workspace_id"))
        source_conflict_policy = _optional_text(receipt.get("source_conflict_policy"))
        sync_back = _optional_text(receipt.get("sync_back"))
        delete_missing = receipt.get("delete_missing")
        snapshot_material = {
            "schema": "cayu.source_publication_snapshot.v1",
            "destination_workspace_id": receipt_destination_workspace_id,
            "workload_workspace_id": workload_workspace_id,
            "source": "sync",
            "outcome": receipt.get("outcome"),
            "source_conflict_policy": source_conflict_policy,
            "sync_back": sync_back,
            "delete_missing": delete_missing,
            "copied_files": copied,
            "copied_bytes": copied_bytes,
            "deleted_files": deleted,
        }
        expected_snapshot_sha256 = (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    snapshot_material,
                    "source_publication_snapshot",
                )
            ).hexdigest()
        )
        receipt_material = {
            "schema": receipt.get("schema"),
            "snapshot_sha256": snapshot_sha256,
            "destination_workspace_id": receipt_destination_workspace_id,
            "workload_workspace_id": workload_workspace_id,
            "outcome": receipt.get("outcome"),
            "source_conflict_policy": source_conflict_policy,
            "sync_back": sync_back,
            "delete_missing": delete_missing,
            "copied_files": copied,
            "copied_bytes": copied_bytes,
            "deleted_files": deleted,
        }
        expected_receipt_sha256 = (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    receipt_material,
                    "source_publication_receipt",
                )
            ).hexdigest()
        )
        receipt_valid = (
            set(receipt) == {*receipt_material, "receipt_sha256"}
            and receipt.get("schema") == "cayu.source_publication_receipt.v1"
            and receipt.get("outcome") == "completed"
            and snapshot_sha256 == expected_snapshot_sha256
            and receipt_sha256 == expected_receipt_sha256
            and _workspace_public_authority_matches(
                receipt_destination_workspace_id,
                destination_workspace_id,
                session_id=session_id,
                public_authority_alias_codec=public_authority_alias_codec,
            )
            and workload_workspace_id not in {None, "workspace-authority-unavailable"}
            and workload_workspace_id != receipt_destination_workspace_id
            and source_conflict_policy == "require_revision"
            and sync_back == "always"
            and delete_missing is True
            and copied is not None
            and copied_bytes is not None
            and deleted is not None
        )
        if not receipt_valid:
            return CodingPublicationEvidence(
                event_id=completed.id,
                destination_id=destination_id,
                outcome="ambiguous",
                baseline_revision=initial_revision,
                final_revision=final_revision,
                detail_code="source_publication_receipt_invalid",
            )
        outcome: Literal["copied", "unchanged"] = (
            "unchanged" if final_revision == initial_revision else "copied"
        )
        return CodingPublicationEvidence(
            event_id=completed.id,
            destination_id=destination_id,
            outcome=outcome,
            baseline_revision=initial_revision,
            final_revision=final_revision,
            snapshot_sha256=snapshot_sha256,
            receipt_sha256=receipt_sha256,
            destination_workspace_id=receipt_destination_workspace_id,
            workload_workspace_id=workload_workspace_id,
            source_conflict_policy="require_revision",
            sync_back="always",
            delete_missing=True,
            copied_files=copied,
            copied_bytes=copied_bytes,
            deleted_files=deleted,
        )
    failed = next(
        (
            event
            for event in reversed(events)
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
        ),
        None,
    )
    if failed is not None and failed.payload.get("error_type") == (
        "SyncBindingSourceConflictError"
    ):
        return CodingPublicationEvidence(
            event_id=failed.id,
            destination_id=destination_id,
            outcome=(
                "partial"
                if final_revision is not None and final_revision != initial_revision
                else "conflicted"
            ),
            baseline_revision=initial_revision,
            final_revision=final_revision,
            detail_code="source_revision_conflict",
        )
    terminal = next(
        (event for event in reversed(events) if event.type in _TERMINAL_EVENT_TYPES), None
    )
    event = failed or terminal
    if event is None:
        return CodingPublicationEvidence(
            event_id="coding-product-publication-unobserved",
            destination_id=destination_id,
            outcome="ambiguous",
            baseline_revision=initial_revision,
            detail_code="publication_terminal_unobserved",
        )
    outcome = (
        "cancelled"
        if event.type == EventType.SESSION_INTERRUPTED
        else "failed"
        if event.type in {EventType.SESSION_FAILED, EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED}
        else "ambiguous"
    )
    return CodingPublicationEvidence(
        event_id=event.id,
        destination_id=destination_id,
        outcome=outcome,
        baseline_revision=initial_revision,
        detail_code="source_publication_not_settled",
    )


def _candidate_state(
    *,
    request: CodingProductRequest,
    events: Sequence[Event],
    checks: Sequence[CodingCheckEvidence],
    commands: Sequence[CodingCommandEvidence],
    mutations: Sequence[CodingMutationEvidence],
    git_status: CodingGitStatusEvidence | None,
    git_summary: CodingGitSummaryEvidence | None,
    git: CodingGitEvidence | None,
    review: CodingReviewSettlement,
    publication: CodingPublicationEvidence,
    final_revision: str | None,
) -> CodingProductState:
    terminal = next(
        (event for event in reversed(events) if event.type in _TERMINAL_EVENT_TYPES), None
    )
    if terminal is None:
        return CodingProductState.RECONSTRUCTION_REQUIRED
    if terminal.type == EventType.SESSION_AWAITING_USER_INPUT:
        return CodingProductState.HUMAN_INPUT_REQUIRED
    if terminal.type == EventType.SESSION_INTERRUPTED:
        return CodingProductState.CANCELLED
    if any(event.type == EventType.TOOL_CALL_APPROVAL_DENIED for event in events):
        return CodingProductState.DENIED
    if any(event.type == EventType.TOOL_CALL_BLOCKED for event in events):
        return CodingProductState.BLOCKED
    if any(
        event.tool_name in {"run_check", "run_command"}
        and (
            (structured := _structured_result(event)) is not None
            and (
                structured.get("status") in {"rebuild_required", "stale_toolchain"}
                or structured.get("error")
                in {
                    "dependency_inputs_changed",
                    "dependency_rebuild_required",
                    "toolchain_rebuild_required",
                    "toolchain_profile_drift",
                }
            )
        )
        for event in events
    ):
        return CodingProductState.TOOLCHAIN_REBUILD_REQUIRED
    if publication.outcome == "conflicted":
        return CodingProductState.SOURCE_CONFLICT
    if publication.outcome == "partial":
        return CodingProductState.PARTIAL
    if publication.outcome == "ambiguous":
        return CodingProductState.RECONSTRUCTION_REQUIRED
    if terminal.type in {EventType.SESSION_FAILED, EventType.SESSION_LIMIT_REACHED}:
        return CodingProductState.FAILED
    if not publication.settled or final_revision is None:
        return CodingProductState.FAILED
    if any(not mutation.settled for mutation in mutations):
        if any(
            mutation.outcome in {"conflict", "stale", "stale_revision"} for mutation in mutations
        ):
            return CodingProductState.SOURCE_CONFLICT
        return (
            CodingProductState.AMBIGUOUS
            if any(mutation.outcome == "ambiguous" for mutation in mutations)
            else CodingProductState.PARTIAL
        )
    if any(not command.settled for command in commands):
        if any(command.status == "ambiguous" for command in commands):
            return CodingProductState.AMBIGUOUS
        if any(command.status == "partial" for command in commands):
            return CodingProductState.PARTIAL
        return CodingProductState.FAILED
    checks_by_name = {check.check: check for check in checks}
    if not set(request.settlement.required_checks).issubset(checks_by_name):
        return CodingProductState.CHECKS_NOT_RUN
    if not all(checks_by_name[name].settled_pass for name in request.settlement.required_checks):
        return CodingProductState.CHECKS_FAILED
    if request.settlement.reviewer_required and review.reviewer != "passed":
        return CodingProductState.REVIEW_REQUIRED
    if request.settlement.human_approval_required and review.human != "approved":
        return CodingProductState.HUMAN_INPUT_REQUIRED
    if (
        git_status is None
        or git_status.scope != "all"
        or git_status.truncated
        or git_summary is None
        or git_summary.scope != "all"
        or git_summary.truncated
        or git is None
        or git.scope != "all"
        or git.truncated
        or git.binary_omitted
    ):
        return CodingProductState.PARTIAL
    if not _git_evidence_matches_source_revisions(
        initial_revision=request.source.baseline_revision,
        final_revision=final_revision,
        status=git_status,
        summary=git_summary,
        diff=git,
    ):
        return CodingProductState.RECONSTRUCTION_REQUIRED
    return CodingProductState.PATCH_READY_FOR_DELIVERY


async def compile_coding_product_candidate(
    request: CodingProductRequest,
    events: Sequence[Event],
    *,
    initial_observation: WorkspaceRevisionObservation,
    final_observation: WorkspaceRevisionObservation,
    repository: CodingProductArtifactRepository,
    lifecycle_receipt_ids: tuple[str, ...] = (),
    review_settlement: CodingReviewSettlement | None = None,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
) -> CodingProductCandidate:
    """Compile bounded public runtime events into verifier-ready product evidence."""

    if type(request) is not CodingProductRequest:
        raise TypeError("request must be CodingProductRequest.")
    if type(initial_observation) is not WorkspaceRevisionObservation:
        raise TypeError("initial_observation must be WorkspaceRevisionObservation.")
    if type(final_observation) is not WorkspaceRevisionObservation:
        raise TypeError("final_observation must be WorkspaceRevisionObservation.")
    if public_authority_alias_codec is not None and not isinstance(
        public_authority_alias_codec,
        PublicAuthorityAliasCodec,
    ):
        raise TypeError("public_authority_alias_codec must be PublicAuthorityAliasCodec or None.")
    if (
        initial_observation.identity.workspace_id != request.source.workspace_id
        or initial_observation.status is not WorkspaceRevisionObservationStatus.SUPPORTED
        or initial_observation.path_scope != "complete"
        or initial_observation.revision != request.source.baseline_revision
    ):
        raise CodingProductAdmissionError(
            "Initial source observation conflicts with admitted source authority."
        )
    if final_observation.identity.workspace_id != request.source.workspace_id:
        raise CodingProductEvidenceError(
            "Final source observation conflicts with admitted workspace authority."
        )
    copied_events = tuple(copy_event(event) for event in events)
    if len(copied_events) > request.settlement.max_events:
        raise ValueError("Coding-product event count exceeds the admitted bound.")
    if sum(_event_bytes(event) for event in copied_events) > request.settlement.max_event_bytes:
        raise ValueError("Coding-product event bytes exceed the admitted bound.")
    if any(event.session_id != request.session_id for event in copied_events):
        raise CodingProductEvidenceError(
            "Coding-product events conflict with admitted session authority."
        )
    if any(event.agent_name not in {None, request.agent_name} for event in copied_events):
        raise CodingProductEvidenceError(
            "Coding-product events conflict with admitted agent authority."
        )
    if any(
        event.agent_name != request.agent_name
        for event in copied_events
        if event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_BLOCKED,
        }
    ):
        raise CodingProductEvidenceError(
            "Coding-product tool evidence lacks admitted agent authority."
        )
    if any(
        event.environment_name != request.runtime.environment_name
        for event in copied_events
        if event.type in _TERMINAL_EVENT_TYPES
        or event.type
        in {
            EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_BLOCKED,
        }
    ):
        raise CodingProductEvidenceError(
            "Coding-product events conflict with admitted environment authority."
        )
    event_ids = tuple(event.id for event in copied_events)
    evidence_sequence_complete = len(event_ids) == len(set(event_ids))
    terminal_indexes = tuple(
        index for index, event in enumerate(copied_events) if event.type in _TERMINAL_EVENT_TYPES
    )
    evidence_sequence_complete = evidence_sequence_complete and terminal_indexes == (
        len(copied_events) - 1,
    )
    finalization_indexes = tuple(
        index
        for index, event in enumerate(copied_events)
        if event.type
        in {
            EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        }
    )
    evidence_sequence_complete = (
        evidence_sequence_complete
        and len(finalization_indexes) == 1
        and finalization_indexes[0] < terminal_indexes[0]
    )

    check_map: dict[str, CodingCheckEvidence] = {}
    commands: list[CodingCommandEvidence] = []
    mutations: list[CodingMutationEvidence] = []
    last_workspace_effect_index = -1
    for event_index, event in enumerate(copied_events):
        if event.type not in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_BLOCKED,
        }:
            continue
        if event.tool_name in _MUTATION_TOOL_NAMES | {"run_check", "run_command"}:
            last_workspace_effect_index = event_index
        structured = _structured_result(event)
        if structured is None:
            if event.tool_name in _MUTATION_TOOL_NAMES | {"run_check", "run_command"}:
                evidence_sequence_complete = False
            continue
        completed = event.type is EventType.TOOL_CALL_COMPLETED
        if event.tool_name == "run_check":
            evidence = _check_evidence(event, structured)
            if evidence is not None:
                if not completed:
                    evidence = CodingCheckEvidence.model_validate(
                        {**evidence.model_dump(mode="python"), "status": "failed"}
                    )
                check_map[evidence.check] = evidence
            else:
                evidence_sequence_complete = False
        elif event.tool_name == "run_command":
            evidence = _command_evidence(event, structured)
            if evidence is not None:
                if not completed:
                    evidence = CodingCommandEvidence.model_validate(
                        {**evidence.model_dump(mode="python"), "status": "failed"}
                    )
                commands.append(evidence)
            else:
                evidence_sequence_complete = False
        elif event.tool_name in _MUTATION_TOOL_NAMES:
            evidence = _mutation_evidence(event, structured)
            if not completed:
                evidence = CodingMutationEvidence.model_validate(
                    {
                        **evidence.model_dump(mode="python"),
                        "outcome": "failed",
                        "requires_fresh_read": True,
                    }
                )
            mutations.append(evidence)

    final_revision = (
        final_observation.revision
        if final_observation.status is WorkspaceRevisionObservationStatus.SUPPORTED
        and final_observation.path_scope == "complete"
        else None
    )
    final_git = _final_git_receipt(
        copied_events,
        request,
        final_revision=final_revision,
        public_authority_alias_codec=public_authority_alias_codec,
    )
    final_git_event_index = -1
    git_status: CodingGitStatusEvidence | None = None
    git_summary: CodingGitSummaryEvidence | None = None
    git: CodingGitEvidence | None = None
    if final_git is not None:
        (
            final_git_event_index,
            event,
            status_structured,
            summary_structured,
            diff_structured,
            diff_content,
        ) = final_git
        status_entries = _git_entries(status_structured.get("changes"))
        git_status = CodingGitStatusEvidence(
            event_id=event.id,
            scope=_optional_text(status_structured.get("scope")) or "unknown",
            entries=() if status_entries is None else status_entries,
            truncated=(status_entries is None or _git_result_truncated(status_structured)),
        )
        summary_entries = _git_summary_entries(summary_structured.get("changes"))
        git_summary = CodingGitSummaryEvidence(
            event_id=event.id,
            scope=_optional_text(summary_structured.get("scope")) or "unknown",
            entries=() if summary_entries is None else summary_entries,
            truncated=(summary_entries is None or _git_result_truncated(summary_structured)),
        )
        diff_entries = _git_entries(diff_structured.get("changes"))
        artifact = await repository.publish_git_diff(
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name=request.runtime.environment_name,
            event_id=event.id,
            content=diff_content,
        )
        git = CodingGitEvidence(
            event_id=event.id,
            scope=_optional_text(diff_structured.get("scope")) or "unknown",
            artifact=artifact,
            entries=() if diff_entries is None else diff_entries,
            entry_count=0 if diff_entries is None else len(diff_entries),
            truncated=diff_entries is None or _git_result_truncated(diff_structured),
            binary_omitted=_optional_bool(diff_structured.get("binary_omitted")),
        )
    checks = tuple(
        check_map[name] for name in request.settlement.required_checks if name in check_map
    )
    review = _review_settlement(
        copied_events,
        request.settlement,
        review_settlement,
    )
    publication = _publication_evidence(
        copied_events,
        destination_id=request.source.destination_id,
        destination_workspace_id=request.source.workspace_id,
        session_id=request.session_id,
        public_authority_alias_codec=public_authority_alias_codec,
        initial_revision=request.source.baseline_revision,
        final_revision=final_revision,
    )
    state = _candidate_state(
        request=request,
        events=copied_events,
        checks=checks,
        commands=commands,
        mutations=mutations,
        git_status=git_status,
        git_summary=git_summary,
        git=git,
        review=review,
        publication=publication,
        final_revision=final_revision,
    )
    if state is CodingProductState.PARTIAL and final_git is None:
        state = CodingProductState.RECONSTRUCTION_REQUIRED
    if state is CodingProductState.PATCH_READY_FOR_DELIVERY and (
        not evidence_sequence_complete
        or any(check.workspace_revision != final_revision for check in checks)
        or final_git_event_index <= last_workspace_effect_index
    ):
        state = CodingProductState.RECONSTRUCTION_REQUIRED
    evidence_event_ids = {
        *(check.event_id for check in checks),
        *(command.event_id for command in commands),
        *(mutation.event_id for mutation in mutations),
        publication.event_id,
        *(event.id for event in copied_events if event.type in _TERMINAL_EVENT_TYPES),
    }
    if review.reviewer_event_id is not None:
        evidence_event_ids.add(review.reviewer_event_id)
    if review.human_event_id is not None:
        evidence_event_ids.add(review.human_event_id)
    authority_events = tuple(event for event in copied_events if event.id in evidence_event_ids)
    expected_execution_profile = request.runtime.execution_profile_fingerprint.removeprefix(
        "sha256:"
    )
    execution_profile_events = tuple(
        event for event in authority_events if event.type not in _TERMINAL_EVENT_TYPES
    )
    execution_profiles_match = all(
        (value := _optional_text(event.payload.get("execution_profile_fingerprint"))) is not None
        and value.removeprefix("sha256:") == expected_execution_profile
        for event in execution_profile_events
    )
    if state is CodingProductState.PATCH_READY_FOR_DELIVERY and not execution_profiles_match:
        state = CodingProductState.RECONSTRUCTION_REQUIRED
    initial_source, final_source = await asyncio.gather(
        repository.publish_source_observation(
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name=request.runtime.environment_name,
            phase="initial",
            observation=initial_observation,
        ),
        repository.publish_source_observation(
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name=request.runtime.environment_name,
            phase="final",
            observation=final_observation,
        ),
    )
    interaction_events = tuple(
        event
        for event in authority_events
        if event.type
        not in {
            EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
            *_TERMINAL_EVENT_TYPES,
        }
    )
    interaction_ids = {event.interaction_id for event in interaction_events}
    interaction_id = (
        next(iter(interaction_ids))
        if len(interaction_ids) == 1 and None not in interaction_ids
        else None
    )
    if state is CodingProductState.PATCH_READY_FOR_DELIVERY and interaction_id is None:
        state = CodingProductState.RECONSTRUCTION_REQUIRED
    candidate = CodingProductCandidate(
        product_run_id=request.product_run_id,
        request_fingerprint=request.fingerprint,
        session_id=request.session_id,
        agent_name=request.agent_name,
        interaction_id=interaction_id,
        task_id=request.task.task_id,
        state=state,
        source_origin_id=request.source.origin_id,
        source_workspace_id=request.source.workspace_id,
        source_destination_id=request.source.destination_id,
        initial_revision=request.source.baseline_revision,
        initial_git=request.source.git_baseline,
        final_revision=final_revision,
        initial_source=initial_source,
        final_source=final_source,
        runtime=request.runtime,
        checks=checks,
        commands=tuple(commands),
        mutations=tuple(mutations),
        git_status=git_status,
        git_summary=git_summary,
        git=git,
        review=review,
        publication=publication,
        lifecycle_receipt_ids=lifecycle_receipt_ids,
    )
    if len(_canonical_model_bytes(candidate, "coding_product_candidate")) > (
        request.settlement.max_result_bytes
    ):
        raise CodingProductEvidenceError(
            "Coding-product candidate exceeds the admitted result byte bound."
        )
    await repository.validate_candidate_artifacts(candidate)
    return candidate


class CodingProductRunner:
    """Run one ordinary Cayu session and retain authoritative product evidence."""

    def __init__(
        self,
        app: CayuApp,
        *,
        source_workspace: Workspace,
        repository: CodingProductArtifactRepository,
        source_git_authority_validator: Callable[[CodingGitBaselineAuthority], Awaitable[None]],
        observation_limits: WorkspaceRevisionObservationLimits | None = None,
    ) -> None:
        if not isinstance(app, CayuApp):
            raise TypeError("app must be CayuApp.")
        if not isinstance(source_workspace, Workspace):
            raise TypeError("source_workspace must implement Workspace.")
        if type(repository) is not CodingProductArtifactRepository:
            raise TypeError("repository must be CodingProductArtifactRepository.")
        if not callable(source_git_authority_validator):
            raise TypeError("source_git_authority_validator must be callable.")
        self.app = app
        self.source_workspace = source_workspace
        self.repository = repository
        self.source_git_authority_validator = source_git_authority_validator
        self.observation_limits = observation_limits or WorkspaceRevisionObservationLimits()

    async def _require_source_git_authority(
        self,
        request: CodingProductRequest,
        receipts: list[CodingLifecycleReceipt],
        artifact_ids: list[str],
    ) -> None:
        """Fail closed unless the application still owns the admitted Git control state."""

        try:
            result = await self.source_git_authority_validator(request.source.git_baseline)
            if result is not None:
                raise TypeError("source_git_authority_validator must return None.")
        except asyncio.CancelledError:
            await self._append_state(
                request,
                receipts,
                artifact_ids,
                CodingProductState.CANCELLED,
                reason_code="caller_cancelled",
            )
            raise
        except Exception:
            await self._append_state(
                request,
                receipts,
                artifact_ids,
                CodingProductState.SOURCE_CONFLICT,
                reason_code="source_git_authority_mismatch",
            )
            raise CodingProductAdmissionError(
                "Coding-product source Git authority changed after admission."
            ) from None

    async def _append_state(
        self,
        request: CodingProductRequest,
        receipts: list[CodingLifecycleReceipt],
        artifact_ids: list[str],
        state: CodingProductState,
        *,
        evidence_sha256: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        if receipts and receipts[-1].state is state:
            return
        receipt = CodingLifecycleReceipt(
            product_run_id=request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
            ordinal=len(receipts) + 1,
            prior_state=None if not receipts else receipts[-1].state,
            state=state,
            evidence_sha256=evidence_sha256,
            reason_code=reason_code,
        )
        artifact = await self.repository.append_lifecycle(receipt)
        receipts.append(receipt)
        artifact_ids.append(artifact.artifact_id)

    async def _require_exact_source_revision(
        self,
        request: CodingProductRequest,
        receipts: list[CodingLifecycleReceipt],
        artifact_ids: list[str],
        *,
        expected_revision: str,
    ) -> None:
        """Reject publication unless the source still matches the compiled candidate."""

        try:
            observed = await observe_deterministic_workspace(
                self.source_workspace,
                observer="cayu-coding-product-source",
                limits=self.observation_limits,
            )
        except asyncio.CancelledError:
            await self._append_state(
                request,
                receipts,
                artifact_ids,
                CodingProductState.CANCELLED,
                reason_code="caller_cancelled",
            )
            raise
        if (
            observed.status is not WorkspaceRevisionObservationStatus.SUPPORTED
            or observed.path_scope != "complete"
            or observed.revision != expected_revision
        ):
            await self._append_state(
                request,
                receipts,
                artifact_ids,
                CodingProductState.SOURCE_CONFLICT,
                reason_code="source_changed_before_publication",
            )
            raise CodingProductAdmissionError(
                "Coding-product source changed after final evidence compilation."
            ) from None

    async def run(
        self,
        request: CodingProductRequest,
        run_request: RunRequest,
        *,
        review_settlement: CodingReviewSettlement | None = None,
    ) -> CodingProductPublication:
        """Execute, compile, and durably publish one bounded coding-product result."""

        if type(request) is not CodingProductRequest:
            raise TypeError("request must be CodingProductRequest.")
        if type(run_request) is not RunRequest:
            raise TypeError("run_request must be RunRequest.")
        if (
            run_request.session_id != request.session_id
            or run_request.agent_name != request.agent_name
            or run_request.environment_name != request.runtime.environment_name
            or run_request.task_id not in {None, request.task.task_id}
            or session_input_messages_sha256(run_request.messages)
            != request.task.instruction_sha256.removeprefix("sha256:")
        ):
            raise ValueError("RunRequest identities conflict with CodingProductRequest.")
        if self.observation_limits != request.source.observation_limits:
            raise CodingProductAdmissionError(
                "Source observation bounds conflict with admitted authority."
            )
        if CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY in run_request.metadata:
            raise ValueError("RunRequest metadata contains reserved coding-product authority.")
        source_copy_authority = CodingProductSourceCopyAuthority(
            request_fingerprint=request.fingerprint,
            source_workspace_id=request.source.workspace_id,
            baseline_revision=request.source.baseline_revision,
            observation_limits=request.source.observation_limits,
        )
        admitted_run_request = run_request.model_copy(
            deep=True,
            update={
                "metadata": {
                    **run_request.metadata,
                    CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY: (
                        source_copy_authority.model_dump(mode="json")
                    ),
                }
            },
        )
        await self.repository.ensure_request(request)
        receipts, artifact_ids = await self.repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
        receipt_list = list(receipts)
        artifact_list = list(artifact_ids)
        if receipt_list:
            latest = receipt_list[-1]
            recoverable_result_states = {
                CodingProductState.CHECKS_NOT_RUN,
                CodingProductState.CHECKS_FAILED,
                CodingProductState.SOURCE_CONFLICT,
                CodingProductState.TOOLCHAIN_REBUILD_REQUIRED,
                CodingProductState.REVIEW_REQUIRED,
                CodingProductState.HUMAN_INPUT_REQUIRED,
                CodingProductState.PATCH_READY_FOR_DELIVERY,
                CodingProductState.BLOCKED,
                CodingProductState.DENIED,
                CodingProductState.CANCELLED,
                CodingProductState.FAILED,
                CodingProductState.PARTIAL,
                CodingProductState.AMBIGUOUS,
                CodingProductState.RECONSTRUCTION_REQUIRED,
            }
            if latest.evidence_sha256 is not None and latest.state in (
                recoverable_result_states
                | {
                    CodingProductState.READY_TO_PUBLISH,
                    CodingProductState.PUBLISHING,
                }
            ):
                try:
                    recovered = await self.repository.load_publication(
                        request_fingerprint=request.fingerprint,
                        digest=latest.evidence_sha256,
                    )
                except (FileNotFoundError, ValueError):
                    if latest.state is not CodingProductState.RECONSTRUCTION_REQUIRED:
                        await self._append_state(
                            request,
                            receipt_list,
                            artifact_list,
                            CodingProductState.RECONSTRUCTION_REQUIRED,
                            reason_code="result_publication_unsettled",
                        )
                    raise CodingProductReconstructionRequiredError(
                        "Coding-product result publication requires reconciliation."
                    ) from None
                if recovered.candidate.request_fingerprint != request.fingerprint:
                    raise CodingProductReconstructionRequiredError(
                        "Recovered coding-product result conflicts with current authority."
                    )
                if latest.state in {
                    CodingProductState.READY_TO_PUBLISH,
                    CodingProductState.PUBLISHING,
                }:
                    await self._append_state(
                        request,
                        receipt_list,
                        artifact_list,
                        recovered.candidate.state,
                        evidence_sha256="sha256:" + recovered.candidate.digest,
                    )
                return recovered
        await self.repository.acquire_execution_claim(
            request,
            claim_id=uuid4().hex,
        )
        if receipt_list and receipt_list[-1].state not in {
            CodingProductState.ADMITTED,
            CodingProductState.PREPARING_WORKSPACE,
        }:
            latest = receipt_list[-1]
            if latest.state is not CodingProductState.RECONSTRUCTION_REQUIRED:
                await self._append_state(
                    request,
                    receipt_list,
                    artifact_list,
                    CodingProductState.RECONSTRUCTION_REQUIRED,
                    reason_code="unsettled_prior_execution",
                )
            raise CodingProductReconstructionRequiredError(
                "A prior coding-product execution requires evidence reconciliation."
            )
        await self._append_state(
            request,
            receipt_list,
            artifact_list,
            CodingProductState.ADMITTED,
        )
        await self._append_state(
            request,
            receipt_list,
            artifact_list,
            CodingProductState.PREPARING_WORKSPACE,
        )
        initial = await observe_deterministic_workspace(
            self.source_workspace,
            observer="cayu-coding-product-source",
            limits=self.observation_limits,
        )
        if (
            initial.status is not WorkspaceRevisionObservationStatus.SUPPORTED
            or initial.path_scope != "complete"
            or initial.revision != request.source.baseline_revision
        ):
            await self._append_state(
                request,
                receipt_list,
                artifact_list,
                CodingProductState.SOURCE_CONFLICT,
                reason_code="source_baseline_mismatch",
            )
            raise CodingProductAdmissionError(
                "Coding-product source does not match its admitted baseline."
            )
        await self._require_source_git_authority(
            request,
            receipt_list,
            artifact_list,
        )
        await self._append_state(
            request,
            receipt_list,
            artifact_list,
            CodingProductState.ACTIVE,
            evidence_sha256=initial.revision,
        )

        events: list[Event] = []
        event_bytes = 0
        try:
            async for event in self.app.run(admitted_run_request):
                copied = copy_event(event)
                events.append(copied)
                event_bytes += _event_bytes(copied)
                if len(events) > request.settlement.max_events:
                    raise ValueError("Coding-product event count exceeds the admitted bound.")
                if event_bytes > request.settlement.max_event_bytes:
                    raise ValueError("Coding-product event bytes exceed the admitted bound.")
        except asyncio.CancelledError:
            await self._append_state(
                request,
                receipt_list,
                artifact_list,
                CodingProductState.CANCELLED,
                reason_code="caller_cancelled",
            )
            raise
        except BaseException:
            await self._append_state(
                request,
                receipt_list,
                artifact_list,
                CodingProductState.FAILED,
                reason_code="session_execution_failed",
            )
            raise

        await self._require_source_git_authority(
            request,
            receipt_list,
            artifact_list,
        )

        final = await observe_deterministic_workspace(
            self.source_workspace,
            observer="cayu-coding-product-source",
            limits=self.observation_limits,
        )
        candidate = await compile_coding_product_candidate(
            request,
            events,
            initial_observation=initial,
            final_observation=final,
            repository=self.repository,
            lifecycle_receipt_ids=tuple(artifact_list),
            review_settlement=review_settlement,
            public_authority_alias_codec=(self.app.session_store.public_authority_alias_codec),
        )
        decision = coding_product_completion_decision(
            request,
            candidate,
            public_authority_alias_codec=(self.app.session_store.public_authority_alias_codec),
            evidence=WorkEvidenceReference(
                kind=CODING_PRODUCT_EVIDENCE_KIND,
                reference_id=_artifact_id(
                    "coding-product-result-v1",
                    candidate.request_fingerprint,
                    candidate.digest,
                ),
                version="v1",
                digest=candidate.digest,
                available=True,
            ),
        )
        if candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY and (
            decision.verdict is not CompletionVerdict.ACCEPTED
        ):
            raise CodingProductEvidenceError(
                "Patch-ready candidate did not satisfy its deterministic verifier."
            )
        await self._append_state(
            request,
            receipt_list,
            artifact_list,
            CodingProductState.READY_TO_PUBLISH,
            evidence_sha256="sha256:" + candidate.digest,
        )
        await self._append_state(
            request,
            receipt_list,
            artifact_list,
            CodingProductState.PUBLISHING,
            evidence_sha256="sha256:" + candidate.digest,
        )
        if candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY:
            final_revision = candidate.final_revision
            if final_revision is None:
                raise CodingProductEvidenceError(
                    "Patch-ready candidate has no final source revision."
                )
            await self._require_source_git_authority(
                request,
                receipt_list,
                artifact_list,
            )
            await self._require_exact_source_revision(
                request,
                receipt_list,
                artifact_list,
                expected_revision=final_revision,
            )
        publication = await self.repository.publish_candidate(candidate)
        await self._append_state(
            request,
            receipt_list,
            artifact_list,
            candidate.state,
            evidence_sha256="sha256:" + candidate.digest,
        )
        return publication


async def admit_coding_product_request(
    *,
    product_run_id: str,
    session_id: str,
    agent_name: str,
    task_id: str,
    messages: Sequence[Message],
    source_workspace: Workspace,
    source_origin_id: str,
    source_destination_id: str,
    source_git_baseline: CodingGitBaselineAuthority,
    runtime: CodingRuntimeAuthority,
    settlement: CodingSettlementPolicy | None = None,
    observation_limits: WorkspaceRevisionObservationLimits | None = None,
) -> CodingProductRequest:
    """Observe exact source authority and construct one immutable product request."""

    if not isinstance(source_workspace, Workspace):
        raise TypeError("source_workspace must implement Workspace.")
    if type(runtime) is not CodingRuntimeAuthority:
        raise TypeError("runtime must be CodingRuntimeAuthority.")
    if type(source_git_baseline) is not CodingGitBaselineAuthority:
        raise TypeError("source_git_baseline must be CodingGitBaselineAuthority.")
    limits = observation_limits or WorkspaceRevisionObservationLimits()
    if type(limits) is not WorkspaceRevisionObservationLimits:
        raise TypeError("observation_limits must be WorkspaceRevisionObservationLimits.")
    try:
        instruction_sha256 = session_input_messages_sha256(messages)
    except (TypeError, ValueError):
        raise CodingProductAdmissionError(
            "Coding-product messages do not form a durable session instruction."
        ) from None
    observed = await observe_deterministic_workspace(
        source_workspace,
        observer="cayu-coding-product-source",
        limits=limits,
    )
    if (
        observed.status is not WorkspaceRevisionObservationStatus.SUPPORTED
        or observed.path_scope != "complete"
        or observed.revision is None
    ):
        raise CodingProductAdmissionError(
            "Coding-product source could not produce a complete bounded baseline."
        )
    return CodingProductRequest(
        product_run_id=product_run_id,
        session_id=session_id,
        agent_name=agent_name,
        source=CodingSourceAuthority(
            origin_id=source_origin_id,
            workspace_id=source_workspace.id,
            baseline_revision=observed.revision,
            destination_id=source_destination_id,
            git_baseline=source_git_baseline,
            observation_limits=limits,
        ),
        task=CodingTaskAuthority(
            task_id=task_id,
            instruction_sha256=instruction_sha256,
        ),
        runtime=runtime,
        settlement=settlement or CodingSettlementPolicy(),
    )


async def admit_or_recover_coding_product_request(
    *,
    repository: CodingProductArtifactRepository,
    product_run_id: str,
    session_id: str,
    agent_name: str,
    task_id: str,
    messages: Sequence[Message],
    source_workspace: Workspace,
    source_origin_id: str,
    source_destination_id: str,
    source_git_baseline: CodingGitBaselineAuthority,
    runtime: CodingRuntimeAuthority,
    settlement: CodingSettlementPolicy | None = None,
    observation_limits: WorkspaceRevisionObservationLimits | None = None,
) -> CodingProductRequest:
    """Recover immutable admission authority or create and retain it exactly once."""

    if type(repository) is not CodingProductArtifactRepository:
        raise TypeError("repository must be CodingProductArtifactRepository.")
    selected_settlement = settlement or CodingSettlementPolicy()
    selected_limits = observation_limits or WorkspaceRevisionObservationLimits()
    try:
        recovered = await repository.load_request(
            product_run_id,
            session_id=session_id,
        )
    except FileNotFoundError:
        request = await admit_coding_product_request(
            product_run_id=product_run_id,
            session_id=session_id,
            agent_name=agent_name,
            task_id=task_id,
            messages=messages,
            source_workspace=source_workspace,
            source_origin_id=source_origin_id,
            source_destination_id=source_destination_id,
            source_git_baseline=source_git_baseline,
            runtime=runtime,
            settlement=selected_settlement,
            observation_limits=selected_limits,
        )
        await repository.ensure_request(request)
        return request
    try:
        instruction_sha256 = session_input_messages_sha256(messages)
    except (TypeError, ValueError):
        raise CodingProductAdmissionError(
            "Coding-product messages do not form a durable session instruction."
        ) from None
    if (
        recovered.agent_name != agent_name
        or recovered.task.task_id != task_id
        or recovered.task.instruction_sha256.removeprefix("sha256:") != instruction_sha256
        or recovered.source.workspace_id != source_workspace.id
        or recovered.source.origin_id != source_origin_id
        or recovered.source.destination_id != source_destination_id
        or recovered.source.git_baseline != source_git_baseline
        or recovered.source.observation_limits != selected_limits
        or recovered.runtime != runtime
        or recovered.settlement != selected_settlement
    ):
        raise CodingProductAdmissionError(
            "Stable product identity is already bound to different caller authority."
        )
    return recovered


async def collect_coding_product_events(events: AsyncIterator[Event]) -> tuple[Event, ...]:
    """Collect an already bounded event stream for application-level composition tests."""

    collected: list[Event] = []
    total_bytes = 0
    async for event in events:
        copied = copy_event(event)
        collected.append(copied)
        total_bytes += _event_bytes(copied)
        if len(collected) > CODING_PRODUCT_MAX_EVENTS:
            raise ValueError("Coding-product event stream exceeds the default bound.")
        if total_bytes > CODING_PRODUCT_MAX_EVENT_BYTES:
            raise ValueError("Coding-product event bytes exceed the default bound.")
    return tuple(collected)


__all__ = [
    "CODING_PRODUCT_EVIDENCE_KIND",
    "CODING_PRODUCT_MAX_EVENTS",
    "CODING_PRODUCT_MAX_EVENT_BYTES",
    "CODING_PRODUCT_MAX_GIT_DIFF_BYTES",
    "CODING_PRODUCT_MAX_LIFECYCLE_RECEIPTS",
    "CODING_PRODUCT_MAX_MUTATION_ARTIFACT_BYTES",
    "CODING_PRODUCT_MAX_RESULT_BYTES",
    "CODING_PRODUCT_MAX_SOURCE_ARTIFACT_BYTES",
    "CODING_PRODUCT_MAX_TOOL_OUTPUT_ARTIFACT_BYTES",
    "CODING_PRODUCT_RESULT_KIND",
    "CODING_PRODUCT_SCHEMA_VERSION",
    "CodingArtifactReference",
    "CodingCheckEvidence",
    "CodingCommandEvidence",
    "CodingGitBaselineAuthority",
    "CodingGitEntry",
    "CodingGitEvidence",
    "CodingGitStatusEvidence",
    "CodingGitSummaryEntry",
    "CodingGitSummaryEvidence",
    "CodingLifecycleReceipt",
    "CodingMutationEvidence",
    "CodingProductAdmissionError",
    "CodingProductArtifactRepository",
    "CodingProductCandidate",
    "CodingProductCompletionVerifier",
    "CodingProductEvidenceError",
    "CodingProductPublication",
    "CodingProductReconstructionRequiredError",
    "CodingProductRequest",
    "CodingProductResultResolver",
    "CodingProductRunner",
    "CodingProductState",
    "CodingPublicationEvidence",
    "CodingReviewSettlement",
    "CodingRuntimeAuthority",
    "CodingSettlementPolicy",
    "CodingSourceAuthority",
    "CodingSourceObservationEvidence",
    "CodingTaskAuthority",
    "admit_coding_product_request",
    "admit_or_recover_coding_product_request",
    "coding_product_completion_decision",
    "coding_product_work_contract",
    "collect_coding_product_events",
    "compile_coding_product_candidate",
    "register_coding_product_contract",
]
