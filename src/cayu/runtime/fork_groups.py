"""Bounded, durable, in-process fork-group execution."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any
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

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_json_value,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.runtime._fork_source_snapshot import fork_source_checkpoint_sha256
from cayu.runtime.budgets import BudgetLimit
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    execution_profile_baseline_from_session_metadata,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    FORK_GROUP_SOURCE_SNAPSHOT_METADATA_KEY,
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    ResumeRequest,
    RunRequest,
    Session,
    SessionOperationPublication,
    SessionStatus,
    SessionStore,
    _bind_fork_execution_profile_fingerprint_capture,
    _bind_fork_expected_source_snapshot,
    _bind_fork_group_initial_invocation,
    session_fork_profile_relationship,
    session_input_messages_sha256,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.structured_output import StructuredOutputSpec
from cayu.runtime.usage import (
    SessionUsageSummary,
    aggregate_usage_metrics_from_durable_payload,
)
from cayu.vaults import SecretRedactor

FORK_GROUP_MIN_BRANCHES = 2
FORK_GROUP_MAX_BRANCHES = 16
FORK_GROUP_MAX_PARALLELISM = 16
FORK_GROUP_MAX_ARTIFACT_REFERENCES = 16
FORK_GROUP_MAX_GATE_RESULTS = 32
FORK_GROUP_MAX_REPLACEMENT_ATTEMPTS = 64
FORK_GROUP_MAX_EVIDENCE_BYTES = 131_072
FORK_GROUP_MAX_BRANCH_OUTPUT_BYTES = 16_384
FORK_GROUP_MAX_REASON_CHARS = 2_048
_FORK_GROUP_RECORD_TYPE = "cayu.fork-group"
_FORK_GROUP_SCHEMA_VERSION = 2
_FORK_GROUP_EXECUTION_CLAIM_SECONDS = 30
_FORK_GROUP_EXECUTION_HEARTBEAT_SECONDS = 10
_FORK_GROUP_TRANSITIONS: frozenset[tuple[ForkGroupState, ForkGroupState]]


class ForkGroupState(StrEnum):
    CREATED = "created"
    BRANCHES_RUNNING = "branches-running"
    AWAITING_EVALUATION = "awaiting-evaluation"
    COMPLETED = "completed"
    FAILED = "failed"


class ForkGroupBranchStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    INVALID = "invalid"


class ForkGroupFailureMode(StrEnum):
    """How one fork group treats an ineligible candidate attempt."""

    FAIL_GROUP = "fail-group"
    EVALUATE_VIABLE = "evaluate-viable"


class ForkGroupDisposition(StrEnum):
    SELECTED = "selected"
    REJECTED = "rejected"
    ADVISORY = "advisory"
    ARCHIVED = "archived"


class ForkGroupFailureCode(StrEnum):
    SOURCE_CHANGED = "source_changed"
    BRANCH_FAILED = "branch_failed"
    BRANCH_INVALID = "branch_invalid"
    GATE_FAILED = "gate_failed"
    EVALUATOR_FAILED = "evaluator_failed"
    JUDGMENT_INVALID = "judgment_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPLACEMENT_FAILED = "replacement_failed"
    REPLACEMENTS_EXHAUSTED = "replacements_exhausted"
    INTERNAL_ERROR = "internal_error"


_TERMINAL_SESSION_STATUSES = frozenset(
    {
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    }
)


_FORK_GROUP_TRANSITIONS = frozenset(
    {
        (ForkGroupState.CREATED, ForkGroupState.BRANCHES_RUNNING),
        (ForkGroupState.BRANCHES_RUNNING, ForkGroupState.AWAITING_EVALUATION),
        (ForkGroupState.BRANCHES_RUNNING, ForkGroupState.FAILED),
        (ForkGroupState.AWAITING_EVALUATION, ForkGroupState.COMPLETED),
        (ForkGroupState.AWAITING_EVALUATION, ForkGroupState.FAILED),
    }
)


class ForkGroupConflict(RuntimeError):
    """The durable group identity is already bound to a different request."""


class _ForkGroupPublicationSuperseded(RuntimeError):
    """Another coordinator already advanced this exact durable group."""


class _ForkGroupPublicationFailure(RuntimeError):
    """A proposed durable transition failed before its revision was committed."""

    def __init__(
        self,
        expected_record: _ForkGroupRecord,
        proposed_record: _ForkGroupRecord,
        original_error: BaseException,
    ) -> None:
        super().__init__("Fork-group publication failed before its durable transition.")
        self.expected_record = expected_record
        self.proposed_record = proposed_record
        self.original_error = original_error


class _ForkGroupContinuationPending(RuntimeError):
    """An exact child operation remains active and eligible for later recovery."""


class _ForkGroupExecutionClaim(BaseModel):
    """One leased durable owner of a nonterminal fork-group continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    claim_id: str = Field(min_length=32, max_length=32)
    expires_at: datetime

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("claim_id must be lowercase hexadecimal authority.")
        return value

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware.")
        return value.astimezone(UTC)


class ForkGroupCheckpointSelector(BaseModel):
    """Optional caller assertions resolved to one exact durable source snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    expected_run_epoch: StrictInt | None = Field(default=None, ge=0)
    expected_transcript_cursor: StrictInt | None = Field(default=None, ge=0)
    expected_profile_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)

    @field_validator("expected_profile_fingerprint")
    @classmethod
    def validate_profile_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and any(character not in "0123456789abcdef" for character in value):
            raise ValueError("expected_profile_fingerprint must be a lowercase SHA-256 digest.")
        return value


class ForkGroupArtifactReference(BaseModel):
    """A bounded application-declared artifact reference exposed to the evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    artifact_id: str = Field(max_length=512)
    description: str | None = Field(default=None, max_length=1_024)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "artifact_id")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_text(value, "description")


class ForkGroupGateResult(BaseModel):
    """One application-supplied deterministic gate result."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    gate_id: str = Field(max_length=256)
    passed: StrictBool
    summary: str | None = Field(default=None, max_length=1_024)

    @field_validator("gate_id")
    @classmethod
    def validate_gate_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "gate_id")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_text(value, "summary")


class ForkGroupBranchSpec(BaseModel):
    """One caller-named sibling and its bounded run configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str = Field(max_length=256)
    session_id: str = Field(max_length=512)
    agent_name: str | None = Field(default=None, max_length=256)
    profile_adoption: ExecutionProfileAdoptionIntent | None = None
    messages: tuple[Message, ...]
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    artifact_references: tuple[ForkGroupArtifactReference, ...] = Field(
        default_factory=tuple,
        max_length=FORK_GROUP_MAX_ARTIFACT_REFERENCES,
    )

    @field_validator("branch_id", "session_id", "agent_name")
    @classmethod
    def validate_names(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("messages", mode="before")
    @classmethod
    def copy_messages(cls, value: object) -> tuple[Message, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Iterable):
            raise TypeError("messages must be a sequence of Message instances.")
        copied = tuple(
            detach_message(message if type(message) is Message else Message.model_validate(message))
            for message in value
        )
        if not copied:
            raise ValueError("Fork-group branch messages cannot be empty.")
        return copied

    @model_validator(mode="after")
    def validate_profile_and_evidence(self) -> ForkGroupBranchSpec:
        if (self.agent_name is None) != (self.profile_adoption is None):
            raise ValueError("agent_name and profile_adoption must be supplied together.")
        if self.structured_output is None and not self.artifact_references:
            raise ValueError(
                "Each fork-group branch must declare structured_output or artifact_references."
            )
        artifact_ids = [artifact.artifact_id for artifact in self.artifact_references]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Fork-group artifact_ids must be unique within a branch.")
        return self


class ForkGroupEvaluatorSpec(BaseModel):
    """Tool-free evaluator session configuration; evidence input is runtime-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str = Field(max_length=512)
    agent_name: str = Field(max_length=256)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("session_id", "agent_name")
    @classmethod
    def validate_names(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class ForkGroupGateSelection(BaseModel):
    """One registered gate and the exact implementation identity the request expects."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    gate_id: str = Field(max_length=256)
    gate_identity: str = Field(max_length=256)

    @field_validator("gate_id", "gate_identity")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class ForkGroupReplacementPlannerSelection(BaseModel):
    """One registered replacement planner and its stable implementation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    planner_id: str = Field(max_length=256)
    planner_identity: str = Field(max_length=256)

    @field_validator("planner_id", "planner_identity")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class ForkGroupFailurePolicy(BaseModel):
    """Bounded durable policy for failed or deterministically ineligible attempts."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    mode: ForkGroupFailureMode = ForkGroupFailureMode.FAIL_GROUP
    minimum_viable_branches: StrictInt | None = Field(default=None, ge=1, le=16)
    max_replacement_attempts: StrictInt = Field(
        default=0,
        ge=0,
        le=FORK_GROUP_MAX_REPLACEMENT_ATTEMPTS,
    )
    replacement_parallelism: StrictInt = Field(
        default=1,
        ge=1,
        le=FORK_GROUP_MAX_PARALLELISM,
    )
    replacement_planner: ForkGroupReplacementPlannerSelection | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> ForkGroupFailurePolicy:
        if self.mode is ForkGroupFailureMode.FAIL_GROUP:
            if self.minimum_viable_branches is not None:
                raise ValueError("fail-group policy cannot declare a viable-branch minimum.")
            if self.max_replacement_attempts != 0:
                raise ValueError("fail-group policy cannot declare replacement attempts.")
            if self.replacement_planner is not None:
                raise ValueError("fail-group policy cannot declare a replacement planner.")
            return self
        if self.minimum_viable_branches is None:
            raise ValueError("evaluate-viable policy requires minimum_viable_branches.")
        if self.max_replacement_attempts < 1:
            raise ValueError("evaluate-viable policy requires replacement attempts.")
        if self.replacement_planner is None:
            raise ValueError("evaluate-viable policy requires a replacement planner.")
        return self


class ForkGroupReplacementSpec(BaseModel):
    """Application-owned content for one runtime-identified replacement attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    agent_name: str | None = Field(default=None, max_length=256)
    profile_adoption: ExecutionProfileAdoptionIntent | None = None
    messages: tuple[Message, ...]
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    artifact_references: tuple[ForkGroupArtifactReference, ...] = Field(
        default_factory=tuple,
        max_length=FORK_GROUP_MAX_ARTIFACT_REFERENCES,
    )

    @field_validator("agent_name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "agent_name")

    @field_validator("messages", mode="before")
    @classmethod
    def copy_messages(cls, value: object) -> tuple[Message, ...]:
        return ForkGroupBranchSpec.copy_messages(value)

    @model_validator(mode="after")
    def validate_profile_and_evidence(self) -> ForkGroupReplacementSpec:
        if (self.agent_name is None) != (self.profile_adoption is None):
            raise ValueError("agent_name and profile_adoption must be supplied together.")
        if self.structured_output is None and not self.artifact_references:
            raise ValueError(
                "Each replacement must declare structured_output or artifact_references."
            )
        artifact_ids = [artifact.artifact_id for artifact in self.artifact_references]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Replacement artifact_ids must be unique.")
        return self


class ForkGroupRequest(BaseModel):
    """A normalized, idempotent request for 2-16 sibling candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: str = Field(max_length=256)
    source_session_id: str = Field(max_length=512)
    source_checkpoint: ForkGroupCheckpointSelector = Field(
        default_factory=ForkGroupCheckpointSelector
    )
    causal_budget_id: str = Field(max_length=512)
    max_parallelism: StrictInt = Field(default=4, ge=1, le=FORK_GROUP_MAX_PARALLELISM)
    branches: tuple[ForkGroupBranchSpec, ...] = Field(
        min_length=FORK_GROUP_MIN_BRANCHES,
        max_length=FORK_GROUP_MAX_BRANCHES,
    )
    gates: tuple[ForkGroupGateSelection, ...] = Field(
        default_factory=tuple,
        max_length=FORK_GROUP_MAX_GATE_RESULTS,
    )
    failure_policy: ForkGroupFailurePolicy = Field(default_factory=ForkGroupFailurePolicy)
    evaluator: ForkGroupEvaluatorSpec

    @field_validator("group_id", "source_session_id", "causal_budget_id")
    @classmethod
    def validate_names(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_unique_identities(self) -> ForkGroupRequest:
        branch_ids = [branch.branch_id for branch in self.branches]
        session_ids = [branch.session_id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("Fork-group branch_ids must be unique.")
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Fork-group branch session_ids must be unique.")
        if self.evaluator.session_id in set(session_ids):
            raise ValueError("The evaluator session_id must differ from every branch session_id.")
        if self.source_session_id in {*session_ids, self.evaluator.session_id}:
            raise ValueError("Fork-group descendants must differ from source_session_id.")
        gate_ids = [gate.gate_id for gate in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("Fork-group gates must have unique gate_ids.")
        policy = self.failure_policy
        if policy.mode is ForkGroupFailureMode.EVALUATE_VIABLE:
            if not self.gates:
                raise ValueError("evaluate-viable policy requires deterministic gates.")
            if policy.minimum_viable_branches is None or policy.minimum_viable_branches > len(
                self.branches
            ):
                raise ValueError("minimum_viable_branches cannot exceed candidate slots.")
            if policy.replacement_parallelism > self.max_parallelism:
                raise ValueError("replacement_parallelism cannot exceed max_parallelism.")
        return self


class ForkGroupSourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source_session_id: str
    status: SessionStatus
    run_epoch: StrictInt = Field(ge=0)
    transcript_cursor: StrictInt = Field(ge=0)
    transcript_sha256: str
    checkpoint_sha256: str
    execution_profile_fingerprint: str
    causal_budget_id: str


class ForkGroupBranchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str
    attempt_id: str = Field(max_length=256)
    attempt_request_sha256: str = Field(min_length=64, max_length=64)
    attempt_index: StrictInt = Field(ge=0, le=FORK_GROUP_MAX_REPLACEMENT_ATTEMPTS)
    replaced_attempt_id: str | None = Field(default=None, max_length=256)
    superseded_by_attempt_id: str | None = Field(default=None, max_length=256)
    session_id: str
    status: ForkGroupBranchStatus
    failure_code: ForkGroupFailureCode | None = None
    eligible: StrictBool = False
    source_checkpoint_sha256: str = Field(min_length=64, max_length=64)
    causal_budget_id: str = Field(max_length=512)
    execution_profile_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    has_structured_output: StrictBool = False
    structured_output: Any = None
    artifact_references: tuple[ForkGroupArtifactReference, ...] = ()
    gate_results: tuple[ForkGroupGateResult, ...] = ()
    usage: SessionUsageSummary
    error: str | None = Field(default=None, max_length=2_048)

    @field_validator(
        "branch_id",
        "attempt_id",
        "replaced_attempt_id",
        "superseded_by_attempt_id",
        "session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_attempt_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "attempt_request_sha256",
        "source_checkpoint_sha256",
        "execution_profile_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is not None and any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("structured_output", mode="before")
    @classmethod
    def copy_output(cls, value: Any) -> Any:
        return copy_json_value(value, "structured_output")

    @field_validator("usage", mode="before")
    @classmethod
    def copy_usage(cls, value: object) -> SessionUsageSummary:
        if type(value) is SessionUsageSummary:
            return value.model_copy(deep=True)
        copied = copy_durable_json_object(value, "usage")
        copied["usage"] = aggregate_usage_metrics_from_durable_payload(copied["usage"])
        return SessionUsageSummary.model_validate(copied)


class ForkGroupGateRequest(BaseModel):
    """Bounded candidate evidence passed to an application-owned deterministic gate."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: str
    source: ForkGroupSourceSnapshot
    branch: ForkGroupBranchResult


class ForkGroupReplacementPlannerRequest(BaseModel):
    """Stable application request for the next attempt in one candidate slot."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: str = Field(max_length=256)
    source: ForkGroupSourceSnapshot
    branch_id: str = Field(max_length=256)
    attempt_id: str = Field(max_length=256)
    attempt_index: StrictInt = Field(ge=1, le=FORK_GROUP_MAX_REPLACEMENT_ATTEMPTS)
    idempotency_key: str = Field(max_length=256)
    replaced_attempt: ForkGroupBranchResult

    @field_validator("group_id", "branch_id", "attempt_id", "idempotency_key")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class ForkGroupGateDecision(BaseModel):
    """A deterministic application's pass/fail decision for one branch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    passed: StrictBool
    summary: str | None = Field(default=None, max_length=1_024)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_text(value, "summary")


class ForkGroupGate(ABC):
    """Application-owned deterministic gate registered under a stable request ID."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable implementation identity bound into normalized group requests."""

    @abstractmethod
    async def evaluate(self, request: ForkGroupGateRequest) -> ForkGroupGateDecision:
        """Evaluate one completed branch without model authority."""


class ForkGroupReplacementPlanner(ABC):
    """Application-owned mutation strategy behind a stable idempotent request."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable implementation identity bound into the normalized group request."""

    @abstractmethod
    async def plan(
        self,
        request: ForkGroupReplacementPlannerRequest,
    ) -> ForkGroupReplacementSpec:
        """Return replacement content without choosing runtime attempt or session identity."""


@dataclass(frozen=True, slots=True)
class _ForkGroupAttempt:
    branch: ForkGroupBranchSpec
    attempt_id: str
    attempt_index: int
    replaced_attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ForkGroupAttemptIdentity:
    branch_id: str
    attempt_id: str


class ForkGroupCoordinator:
    """Own durable fork-group orchestration through narrow runtime callbacks."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        resolve_public_session_authority: Callable[[str], Awaitable[tuple[str, str | None]]],
        fork_session: Callable[[ForkSessionRequest], AsyncIterator[Event]],
        run_session: Callable[[RunRequest], AsyncIterator[Event]],
        resume_session: Callable[[ResumeRequest], AsyncIterator[Event]],
        recover_incomplete_session: Callable[
            [IncompleteSessionRecoveryRequest], Awaitable[IncompleteSessionRecoveryResult]
        ],
        get_session_usage: Callable[[str], Awaitable[SessionUsageSummary]],
        preflight_evaluator_agent: Callable[[RunRequest, str, str], Awaitable[tuple[str, str]]],
        prepare_evaluator_agent: Callable[
            [RunRequest, str, str, str | None], Awaitable[tuple[str, str]]
        ],
        preflight_fork_source: Callable[[str], Awaitable[None]],
        preflight_fork_source_state: Callable[[Session, dict[str, Any] | None], Awaitable[None]],
        admit_fork_source: Callable[[str], Awaitable[None]],
    ) -> None:
        if not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("ForkGroupCoordinator requires a SecretRedactor.")
        self.session_store = session_store
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._resolve_public_session_authority_callback = resolve_public_session_authority
        self._fork_session_callback = fork_session
        self._run_session_callback = run_session
        self._resume_session_callback = resume_session
        self._recover_incomplete_session_callback = recover_incomplete_session
        self._get_session_usage_callback = get_session_usage
        self._preflight_evaluator_agent_callback = preflight_evaluator_agent
        self._prepare_evaluator_agent_callback = prepare_evaluator_agent
        self._preflight_fork_source_callback = preflight_fork_source
        self._preflight_fork_source_state_callback = preflight_fork_source_state
        self._admit_fork_source_callback = admit_fork_source
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._gates: dict[str, ForkGroupGate] = {}
        self._replacement_planners: dict[str, ForkGroupReplacementPlanner] = {}

    @property
    def secret_redactor(self) -> SecretRedactor:
        """Return the immutable redaction authority for fork-group diagnostics."""

        return self._secret_redactor

    def now(self) -> datetime:
        """Return the coordinator clock normalized to UTC."""

        return self._clock().astimezone(UTC)

    def gate(self, gate_id: str) -> ForkGroupGate | None:
        """Resolve one application-registered deterministic gate."""

        return self._gates.get(gate_id)

    def replacement_planner(self, planner_id: str) -> ForkGroupReplacementPlanner | None:
        """Resolve one application-registered replacement planner."""

        return self._replacement_planners.get(planner_id)

    async def resolve_public_session_authority(self, session_id: str) -> tuple[str, str | None]:
        return await self._resolve_public_session_authority_callback(session_id)

    def fork_session(self, request: ForkSessionRequest) -> AsyncIterator[Event]:
        return self._fork_session_callback(request)

    def run(self, request: RunRequest) -> AsyncIterator[Event]:
        return self._run_session_callback(request)

    def resume(self, request: ResumeRequest) -> AsyncIterator[Event]:
        return self._resume_session_callback(request)

    async def recover_incomplete_session(
        self, request: IncompleteSessionRecoveryRequest
    ) -> IncompleteSessionRecoveryResult:
        return await self._recover_incomplete_session_callback(request)

    async def get_session_usage(self, session_id: str) -> SessionUsageSummary:
        return await self._get_session_usage_callback(session_id)

    async def prepare_evaluator_agent(
        self,
        request: RunRequest,
        *,
        source_agent_name: str,
        request_sha256: str,
        store_resolved_existing_session_id: str | None,
    ) -> tuple[str, str]:
        return await self._prepare_evaluator_agent_callback(
            request,
            source_agent_name,
            request_sha256,
            store_resolved_existing_session_id,
        )

    async def preflight_evaluator_agent(
        self,
        request: RunRequest,
        *,
        source_agent_name: str,
        request_sha256: str,
    ) -> tuple[str, str]:
        return await self._preflight_evaluator_agent_callback(
            request,
            source_agent_name,
            request_sha256,
        )

    async def preflight_fork_source(self, session_id: str) -> None:
        await self._preflight_fork_source_callback(session_id)

    async def preflight_fork_source_state(
        self,
        source: Session,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        await self._preflight_fork_source_state_callback(source, checkpoint)

    async def admit_fork_source(self, session_id: str) -> None:
        await self._admit_fork_source_callback(session_id)

    def _validate_extension_authority(
        self,
        extension_id: str,
        extension_identity: str,
        *,
        id_field_name: str,
        identity_field_name: str,
        authority_kind: str,
    ) -> tuple[str, str]:
        extension_id = require_durable_clean_nonblank(extension_id, id_field_name)
        if len(extension_id.encode("utf-8")) > 256:
            raise ValueError(f"{id_field_name} must be at most 256 UTF-8 bytes.")
        extension_identity = require_durable_clean_nonblank(
            extension_identity,
            identity_field_name,
        )
        if len(extension_identity.encode("utf-8")) > 256:
            raise ValueError(f"{identity_field_name} must be at most 256 UTF-8 bytes.")
        for field_name, value in (
            (id_field_name, extension_id),
            (identity_field_name, extension_identity),
        ):
            session_request_boundary.require_secret_free_session_authority(
                value,
                field_name=field_name,
                redactor=self._secret_redactor,
                authority_kind=authority_kind,
            )
        return extension_id, extension_identity

    def register_gate(self, gate_id: str, gate: ForkGroupGate) -> ForkGroupGateSelection:
        if not isinstance(gate, ForkGroupGate):
            raise TypeError("gate must be a ForkGroupGate.")
        gate_id, gate_identity = self._validate_extension_authority(
            gate_id,
            gate.identity,
            id_field_name="gate_id",
            identity_field_name="gate.identity",
            authority_kind="durable fork-group gate authority",
        )
        if gate_id in self._gates:
            raise ValueError(f"Fork-group gate already registered: {gate_id}")
        self._gates[gate_id] = gate
        return ForkGroupGateSelection(gate_id=gate_id, gate_identity=gate_identity)

    def register_replacement_planner(
        self,
        planner_id: str,
        planner: ForkGroupReplacementPlanner,
    ) -> ForkGroupReplacementPlannerSelection:
        if not isinstance(planner, ForkGroupReplacementPlanner):
            raise TypeError("planner must be a ForkGroupReplacementPlanner.")
        planner_id, planner_identity = self._validate_extension_authority(
            planner_id,
            planner.identity,
            id_field_name="planner_id",
            identity_field_name="planner.identity",
            authority_kind="durable fork-group replacement authority",
        )
        if planner_id in self._replacement_planners:
            raise ValueError(f"Fork-group replacement planner already registered: {planner_id}")
        self._replacement_planners[planner_id] = planner
        return ForkGroupReplacementPlannerSelection(
            planner_id=planner_id,
            planner_identity=planner_identity,
        )

    async def run_group(self, request: ForkGroupRequest) -> ForkGroupResult:
        """Run or reconstruct one bounded durable fork group."""

        if type(request) is not ForkGroupRequest:
            raise TypeError("Runtime fork group requires a ForkGroupRequest.")
        public_source_session_id = request.source_session_id
        copied = ForkGroupRequest.model_validate(request.model_dump(mode="python", warnings=False))
        (
            source_id,
            store_resolved_source_session_id,
        ) = await self.resolve_public_session_authority(copied.source_session_id)
        session_request_boundary.require_store_resolved_or_secret_free_session_authority(
            source_id,
            store_resolved_value=store_resolved_source_session_id,
            field_name="source_session_id",
            redactor=self.secret_redactor,
        )
        await self.preflight_fork_source(source_id)
        copied = _prepare_request(
            self,
            copied,
            source_session_id=source_id,
            store_resolved_source_session_id=store_resolved_source_session_id,
        )
        lock_key = (source_id, copied.group_id)
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            record = await _load_record(self, source_id, copied)
            replayed = record is not None
            if record is None:
                record = await _create_record(self, copied)
            else:
                # A durable legacy/current record does not itself authorize a
                # new ordinary replay. Establish the source admission before a
                # claim update or result publication.
                await self.admit_fork_source(source_id)
            if record.result.state not in {ForkGroupState.COMPLETED, ForkGroupState.FAILED}:
                record, claim_id = await _claim_execution(
                    self,
                    source_id,
                    record,
                )
                if claim_id is None:
                    public_source = record.result.source.model_copy(
                        update={"source_session_id": public_source_session_id},
                        deep=True,
                    )
                    return record.result.model_copy(
                        update={"source": public_source, "replayed": True},
                        deep=True,
                    )
                try:
                    record = await _execute_with_claim(self, record, claim_id)
                except asyncio.CancelledError:
                    with suppress(Exception):
                        await asyncio.shield(
                            _update_execution_claim(
                                self,
                                source_id,
                                record,
                                claim_id,
                                release=True,
                            )
                        )
                    raise
                except _ForkGroupPublicationSuperseded:
                    latest = await _load_record(self, source_id, copied)
                    if latest is None:
                        raise RuntimeError(
                            "Superseded fork-group execution has no durable winner."
                        ) from None
                    record = latest
                except ForkGroupConflict:
                    raise
                except Exception as exc:
                    publication_failure = (
                        exc if isinstance(exc, _ForkGroupPublicationFailure) else None
                    )
                    proposed_record = (
                        None if publication_failure is None else publication_failure.proposed_record
                    )
                    diagnostic_error = (
                        exc if publication_failure is None else publication_failure.original_error
                    )
                    latest = await _load_record(self, source_id, copied)
                    if latest is not None:
                        record = latest
                    owns_failed_transition = (
                        record.execution_claim is not None
                        and record.execution_claim.claim_id == claim_id
                        and (
                            publication_failure is None
                            or _transition_material(record)
                            == _transition_material(publication_failure.expected_record)
                        )
                    )
                    if (
                        owns_failed_transition
                        and proposed_record is not None
                        and proposed_record.result.state
                        in {ForkGroupState.COMPLETED, ForkGroupState.FAILED}
                    ):
                        record = await _publish_record(
                            self,
                            source_id,
                            proposed_record,
                            (
                                EventType.FORK_GROUP_COMPLETED
                                if proposed_record.result.state is ForkGroupState.COMPLETED
                                else EventType.FORK_GROUP_FAILED
                            ),
                            expected_record=record,
                        )
                    elif owns_failed_transition and record.result.state not in {
                        ForkGroupState.COMPLETED,
                        ForkGroupState.FAILED,
                    }:
                        diagnostic_message = _exception_message(
                            diagnostic_error,
                            redactor=self.secret_redactor,
                        )
                        failure = _failure(
                            ForkGroupFailureCode.SOURCE_CHANGED
                            if any(
                                marker in diagnostic_message.lower()
                                for marker in ("epoch", "stale", "source", "snapshot")
                            )
                            else ForkGroupFailureCode.INTERNAL_ERROR,
                            diagnostic_message,
                        )
                        failed = (
                            _result_with(
                                record,
                                state=ForkGroupState.FAILED,
                                failure=failure,
                            )
                            if proposed_record is None or not proposed_record.result.branches
                            else proposed_record.model_copy(
                                update={
                                    "result": proposed_record.result.model_copy(
                                        update={
                                            "state": ForkGroupState.FAILED,
                                            "dispositions": (),
                                            "failure": failure,
                                            "replayed": False,
                                        },
                                        deep=True,
                                    )
                                },
                                deep=True,
                            )
                        )
                        record = await _publish_record(
                            self,
                            source_id,
                            failed,
                            EventType.FORK_GROUP_FAILED,
                            expected_record=record,
                        )
                finally:
                    if record.result.state not in {
                        ForkGroupState.COMPLETED,
                        ForkGroupState.FAILED,
                    }:
                        with suppress(_ForkGroupPublicationSuperseded):
                            record = await _update_execution_claim(
                                self,
                                source_id,
                                record,
                                claim_id,
                                release=True,
                            )
            public_source = record.result.source.model_copy(
                update={"source_session_id": public_source_session_id},
                deep=True,
            )
            return record.result.model_copy(
                update={"source": public_source, "replayed": replayed},
                deep=True,
            )


class ForkGroupDispositionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str
    attempt_id: str
    disposition: ForkGroupDisposition
    reason: str = Field(max_length=FORK_GROUP_MAX_REASON_CHARS)

    @field_validator("branch_id", "attempt_id", "reason")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if info.field_name in {"branch_id", "attempt_id"}:
            return require_durable_clean_nonblank(value, info.field_name)
        return require_durable_text(value, info.field_name)


class ForkGroupFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    code: ForkGroupFailureCode
    message: str = Field(max_length=2_048)
    branch_id: str | None = Field(default=None, max_length=256)


class ForkGroupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: str
    state: ForkGroupState
    source: ForkGroupSourceSnapshot
    branches: tuple[ForkGroupBranchResult, ...] = ()
    evaluator_session_id: str | None = None
    dispositions: tuple[ForkGroupDispositionRecord, ...] = ()
    failure: ForkGroupFailure | None = None
    replayed: StrictBool = False


class _ForkGroupRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: str = _FORK_GROUP_RECORD_TYPE
    schema_version: StrictInt = _FORK_GROUP_SCHEMA_VERSION
    revision: StrictInt = Field(ge=0)
    request_sha256: str
    request: ForkGroupRequest
    result: ForkGroupResult
    evaluator_execution_profile_fingerprint: str | None = None
    execution_claim: _ForkGroupExecutionClaim | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> _ForkGroupRecord:
        if self.record_type != _FORK_GROUP_RECORD_TYPE or (
            self.schema_version != _FORK_GROUP_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported fork-group operation record.")
        if _request_sha256(self.request) != self.request_sha256:
            raise ValueError("Fork-group operation record request digest is invalid.")
        if self.result.group_id != self.request.group_id:
            raise ValueError("Fork-group operation record group identity is invalid.")
        evaluator_fingerprint = self.evaluator_execution_profile_fingerprint
        if evaluator_fingerprint is not None and (
            len(evaluator_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in evaluator_fingerprint)
        ):
            raise ValueError("Fork-group evaluator profile fingerprint is invalid.")
        if (self.result.evaluator_session_id is None) != (evaluator_fingerprint is None):
            raise ValueError("Fork-group evaluator session and profile authority must agree.")
        if self.result.state in {ForkGroupState.AWAITING_EVALUATION, ForkGroupState.COMPLETED} and (
            evaluator_fingerprint is None
        ):
            raise ValueError("Fork-group evaluator authority is missing from an evaluable group.")
        minimum_revision = {
            ForkGroupState.CREATED: 0,
            ForkGroupState.BRANCHES_RUNNING: 1,
            ForkGroupState.AWAITING_EVALUATION: 2,
            ForkGroupState.COMPLETED: 3,
            ForkGroupState.FAILED: 2,
        }[self.result.state]
        if self.revision < minimum_revision:
            raise ValueError("Fork-group operation record revision is inconsistent with state.")
        _validate_attempt_graph(self)
        return self


def _request_sha256(request: ForkGroupRequest) -> str:
    return sha256(
        canonical_durable_json_bytes(
            request.model_dump(mode="json", warnings=False),
            "fork_group.request",
        )
    ).hexdigest()


def _storage_key(group_id: str) -> str:
    return "fork-group:" + sha256(group_id.encode("utf-8")).hexdigest()


def _attempt_id(request: ForkGroupRequest, branch_id: str, attempt_index: int) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema_version": 2,
            "group_id": request.group_id,
            "source_session_id": request.source_session_id,
            "source_checkpoint": request.source_checkpoint.model_dump(mode="json"),
            "branch_id": branch_id,
            "attempt_index": attempt_index,
        },
        "fork_group.attempt_identity",
    )
    return "fork-attempt:" + sha256(material).hexdigest()


def _replacement_session_id(attempt_id: str) -> str:
    return "fork-replacement:" + sha256(attempt_id.encode("utf-8")).hexdigest()


def _attempt_request_sha256(attempt: _ForkGroupAttempt) -> str:
    """Bind one logical attempt to its complete prepared application request."""

    return sha256(
        canonical_durable_json_bytes(
            {
                "schema_version": 1,
                "attempt_id": attempt.attempt_id,
                "attempt_index": attempt.attempt_index,
                "replaced_attempt_id": attempt.replaced_attempt_id,
                "branch": attempt.branch.model_dump(mode="json", warnings=False),
            },
            "fork_group.attempt_request",
        )
    ).hexdigest()


def _initial_attempt(request: ForkGroupRequest, branch: ForkGroupBranchSpec) -> _ForkGroupAttempt:
    return _ForkGroupAttempt(
        branch=branch,
        attempt_id=_attempt_id(request, branch.branch_id, 0),
        attempt_index=0,
    )


def _validate_attempt_graph(record: _ForkGroupRecord) -> None:
    """Reject durable attempt graphs that could widen evaluator or replay authority."""

    request = record.request
    result = record.result
    slots = {branch.branch_id: branch for branch in request.branches}
    attempts_by_id: dict[str, ForkGroupBranchResult] = {}
    attempts_by_slot: dict[str, list[ForkGroupBranchResult]] = {
        branch_id: [] for branch_id in slots
    }
    session_ids: set[str] = set()
    for attempt in result.branches:
        slot = slots.get(attempt.branch_id)
        if slot is None:
            raise ValueError("Fork-group result refers to an unknown candidate slot.")
        if attempt.attempt_id in attempts_by_id or attempt.session_id in session_ids:
            raise ValueError("Fork-group attempt and session identities must be unique.")
        if attempt.attempt_id != _attempt_id(
            request,
            attempt.branch_id,
            attempt.attempt_index,
        ):
            raise ValueError("Fork-group attempt identity is not runtime-derived.")
        if attempt.attempt_index == 0 and attempt.attempt_request_sha256 != (
            _attempt_request_sha256(_initial_attempt(request, slot))
        ):
            raise ValueError("Fork-group initial attempt request authority changed.")
        expected_session_id = (
            slot.session_id
            if attempt.attempt_index == 0
            else _replacement_session_id(attempt.attempt_id)
        )
        if attempt.session_id != expected_session_id:
            raise ValueError("Fork-group attempt session identity is not authoritative.")
        if attempt.source_checkpoint_sha256 != result.source.checkpoint_sha256:
            raise ValueError("Fork-group attempt source checkpoint changed.")
        if attempt.causal_budget_id != request.causal_budget_id:
            raise ValueError("Fork-group attempt causal budget changed.")
        if attempt.attempt_index > 0 and attempt.execution_profile_fingerprint is None:
            raise ValueError(
                "Fork-group replacement attempt has no durable execution profile authority."
            )
        if attempt.eligible and (
            attempt.status is not ForkGroupBranchStatus.COMPLETED
            or attempt.superseded_by_attempt_id is not None
        ):
            raise ValueError("Only completed, current attempts may be evaluator eligible.")
        attempts_by_id[attempt.attempt_id] = attempt
        attempts_by_slot[attempt.branch_id].append(attempt)
        session_ids.add(attempt.session_id)

    for slot_attempts in attempts_by_slot.values():
        if not slot_attempts:
            continue
        ordered = sorted(slot_attempts, key=lambda item: item.attempt_index)
        if [attempt.attempt_index for attempt in ordered] != list(range(len(ordered))):
            raise ValueError("Fork-group replacement attempt indexes must be contiguous.")
        for index, attempt in enumerate(ordered):
            expected_replaced = None if index == 0 else ordered[index - 1].attempt_id
            expected_superseded = (
                None if index + 1 == len(ordered) else ordered[index + 1].attempt_id
            )
            if attempt.replaced_attempt_id != expected_replaced or (
                attempt.superseded_by_attempt_id != expected_superseded
            ):
                raise ValueError("Fork-group replacement lineage is inconsistent.")

    maximum_attempts = len(request.branches) + request.failure_policy.max_replacement_attempts
    if len(result.branches) > maximum_attempts:
        raise ValueError("Fork-group attempt graph exceeds the declared replacement limit.")
    disposition_identities = {(item.branch_id, item.attempt_id) for item in result.dispositions}
    if len(disposition_identities) != len(result.dispositions):
        raise ValueError("Fork-group dispositions must have unique attempt identities.")
    eligible_identities = {
        (attempt.branch_id, attempt.attempt_id) for attempt in result.branches if attempt.eligible
    }
    if result.state is ForkGroupState.COMPLETED:
        if (
            disposition_identities != eligible_identities
            or sum(
                item.disposition is ForkGroupDisposition.SELECTED for item in result.dispositions
            )
            != 1
        ):
            raise ValueError(
                "Completed fork groups must dispose every eligible attempt and select one."
            )
    elif result.dispositions:
        raise ValueError("Only completed fork groups may publish evaluator dispositions.")


def _coerce_attempt(
    request: ForkGroupRequest,
    attempt: _ForkGroupAttempt | ForkGroupBranchSpec,
) -> _ForkGroupAttempt:
    """Keep the initial-attempt helpers convenient for focused contract tests."""

    if isinstance(attempt, ForkGroupBranchSpec):
        return _initial_attempt(request, attempt)
    return attempt


def _branch_metadata(
    request: ForkGroupRequest,
    attempt: _ForkGroupAttempt,
) -> dict[str, Any]:
    return {
        "fork_group_id": request.group_id,
        "fork_group_branch_id": attempt.branch.branch_id,
        "fork_group_attempt_id": attempt.attempt_id,
        "fork_group_attempt_request_sha256": _attempt_request_sha256(attempt),
        "fork_group_attempt_index": attempt.attempt_index,
        "fork_group_replaced_attempt_id": attempt.replaced_attempt_id,
    }


def _branch_fork_request(
    request: ForkGroupRequest,
    attempt: _ForkGroupAttempt | ForkGroupBranchSpec,
) -> ForkSessionRequest:
    attempt = _coerce_attempt(request, attempt)
    branch = attempt.branch
    selection = (
        ForkExecutionProfileSelection.INHERIT_PARENT
        if branch.agent_name is None
        else ForkExecutionProfileSelection.CURRENT_CHILD
    )
    fork_request = ForkSessionRequest(
        source_session_id=request.source_session_id,
        session_id=branch.session_id,
        agent_name=branch.agent_name,
        execution_profile_selection=selection,
        profile_adoption=branch.profile_adoption,
        metadata=_branch_metadata(request, attempt),
    )
    return _bind_fork_group_initial_invocation(
        fork_request,
        _branch_resume_request(request, attempt),
    )


def _branch_resume_request(
    request: ForkGroupRequest,
    attempt: _ForkGroupAttempt | ForkGroupBranchSpec,
) -> ResumeRequest:
    attempt = _coerce_attempt(request, attempt)
    branch = attempt.branch
    return ResumeRequest(
        session_id=branch.session_id,
        messages=list(branch.messages),
        max_steps=branch.max_steps,
        limits=branch.limits,
        budget_limits=branch.budget_limits,
        retry_policy=branch.retry_policy,
        structured_output=branch.structured_output,
        thinking=branch.thinking,
        metadata=_branch_metadata(request, attempt),
    )


def _branch_matches_frozen_source(
    child: Session,
    source: ForkGroupSourceSnapshot,
) -> bool:
    try:
        relationship = session_fork_profile_relationship(child)
    except Exception:
        return False
    return (
        child.metadata.get(FORK_GROUP_SOURCE_SNAPSHOT_METADATA_KEY)
        == source.model_dump(mode="json")
        and relationship is not None
        and relationship.source_session_id == source.source_session_id
        and relationship.source_status is source.status
        and relationship.source_run_epoch == source.run_epoch
        and relationship.source_profile.fingerprint == source.execution_profile_fingerprint
    )


async def _source_matches_frozen_snapshot(
    coordinator: ForkGroupCoordinator,
    source: ForkGroupSourceSnapshot,
) -> bool:
    current = await coordinator.session_store.load(source.source_session_id)
    if current is None:
        return False
    transcript = await coordinator.session_store.load_transcript_snapshot(current.id)
    checkpoint = await coordinator.session_store.load_checkpoint(current.id)
    effective_checkpoint = {} if checkpoint is None else checkpoint
    try:
        _, profile, _ = session_request_boundary.prepare_fork_source_execution_profile(
            current,
            effective_checkpoint,
        )
    except Exception:
        return False
    return (
        current.status is source.status
        and current.run_epoch == source.run_epoch
        and current.causal_budget_id == source.causal_budget_id
        and transcript.cursor == source.transcript_cursor
        and session_input_messages_sha256([record.message for record in transcript.records])
        == source.transcript_sha256
        and fork_source_checkpoint_sha256(effective_checkpoint) == source.checkpoint_sha256
        and profile.fingerprint == source.execution_profile_fingerprint
    )


def _require_branch_authority(
    child: Session,
    *,
    request: ForkGroupRequest,
    source: ForkGroupSourceSnapshot,
    attempt: _ForkGroupAttempt,
) -> None:
    branch_session_id = attempt.branch.session_id
    if (
        child.parent_session_id != source.source_session_id
        or child.causal_budget_id != request.causal_budget_id
        or not _branch_matches_frozen_source(child, source)
        or child.metadata.get("fork_group_id") != request.group_id
        or child.metadata.get("fork_group_branch_id") != attempt.branch.branch_id
        or child.metadata.get("fork_group_attempt_id") != attempt.attempt_id
        or child.metadata.get("fork_group_attempt_request_sha256")
        != _attempt_request_sha256(attempt)
        or child.metadata.get("fork_group_attempt_index") != attempt.attempt_index
        or child.metadata.get("fork_group_replaced_attempt_id") != attempt.replaced_attempt_id
    ):
        raise ForkGroupConflict(
            f"Branch session {branch_session_id!r} conflicts with the fork-group request."
        )


def _prepare_request(
    coordinator: ForkGroupCoordinator,
    request: ForkGroupRequest,
    *,
    source_session_id: str,
    store_resolved_source_session_id: str | None = None,
) -> ForkGroupRequest:
    """Apply ordinary fork/resume durability boundaries before identity is bound."""

    redactor = coordinator.secret_redactor
    session_request_boundary.require_store_resolved_or_secret_free_session_authority(
        source_session_id,
        store_resolved_value=store_resolved_source_session_id,
        field_name="source_session_id",
        redactor=redactor,
    )
    request = request.model_copy(
        update={"source_session_id": source_session_id},
        deep=True,
    )
    for field_name, value in (
        ("group_id", request.group_id),
        ("causal_budget_id", request.causal_budget_id),
        ("evaluator.session_id", request.evaluator.session_id),
        ("evaluator.agent_name", request.evaluator.agent_name),
    ):
        session_request_boundary.require_secret_free_session_authority(
            value,
            field_name=field_name,
            redactor=redactor,
            authority_kind="durable fork-group authority",
        )

    prepared_branches: list[ForkGroupBranchSpec] = []
    for index, branch in enumerate(request.branches):
        attempt = _initial_attempt(request, branch)
        prepared_fork = session_request_boundary.prepare_fork_session_request(
            _branch_fork_request(request, attempt),
            redactor=redactor,
            store_resolved_source_session_id=store_resolved_source_session_id,
        )
        prepared_resume = session_request_boundary.prepare_resume_request(
            _branch_resume_request(request, attempt),
            redactor=redactor,
        )
        session_request_boundary.require_secret_free_session_authority(
            branch.branch_id,
            field_name=f"branches[{index}].branch_id",
            redactor=redactor,
            authority_kind="durable fork-group authority",
        )
        artifacts: list[ForkGroupArtifactReference] = []
        for artifact_index, artifact in enumerate(branch.artifact_references):
            session_request_boundary.require_secret_free_session_authority(
                artifact.artifact_id,
                field_name=(f"branches[{index}].artifact_references[{artifact_index}].artifact_id"),
                redactor=redactor,
                authority_kind="durable fork-group artifact authority",
            )
            artifacts.append(
                artifact.model_copy(
                    update={
                        "description": (
                            None
                            if artifact.description is None
                            else redactor.redact_text(artifact.description)
                        )
                    },
                    deep=True,
                )
            )
        prepared_branches.append(
            branch.model_copy(
                update={
                    "agent_name": prepared_fork.agent_name,
                    "profile_adoption": prepared_fork.profile_adoption,
                    "messages": tuple(prepared_resume.messages),
                    "structured_output": prepared_resume.structured_output,
                    "artifact_references": tuple(artifacts),
                },
                deep=True,
            )
        )

    for index, gate in enumerate(request.gates):
        for field_name, value in (
            (f"gates[{index}].gate_id", gate.gate_id),
            (f"gates[{index}].gate_identity", gate.gate_identity),
        ):
            session_request_boundary.require_secret_free_session_authority(
                value,
                field_name=field_name,
                redactor=redactor,
                authority_kind="durable fork-group gate authority",
            )
    planner_selection = request.failure_policy.replacement_planner
    if planner_selection is not None:
        for field_name, value in (
            ("failure_policy.replacement_planner.planner_id", planner_selection.planner_id),
            (
                "failure_policy.replacement_planner.planner_identity",
                planner_selection.planner_identity,
            ),
        ):
            session_request_boundary.require_secret_free_session_authority(
                value,
                field_name=field_name,
                redactor=redactor,
                authority_kind="durable fork-group replacement authority",
            )
    return request.model_copy(
        update={
            "source_session_id": source_session_id,
            "branches": tuple(prepared_branches),
        },
        deep=True,
    )


def _event(group: _ForkGroupRecord, event_type: EventType) -> Event:
    result = group.result
    dispositions = [item.model_dump(mode="json") for item in result.dispositions]
    selected = next(
        (
            item.branch_id
            for item in result.dispositions
            if item.disposition is ForkGroupDisposition.SELECTED
        ),
        None,
    )
    selected_attempt = next(
        (
            item.attempt_id
            for item in result.dispositions
            if item.disposition is ForkGroupDisposition.SELECTED
        ),
        None,
    )
    branch_payload = (
        [
            {
                "branch_id": branch.branch_id,
                "attempt_id": branch.attempt_id,
                "attempt_request_sha256": branch.attempt_request_sha256,
                "attempt_index": branch.attempt_index,
                "replaced_attempt_id": branch.replaced_attempt_id,
                "superseded_by_attempt_id": branch.superseded_by_attempt_id,
                "session_id": branch.session_id,
                "status": branch.status.value,
                "eligible": branch.eligible,
            }
            for branch in result.branches
        ]
        if result.branches
        else [
            {
                "branch_id": branch.branch_id,
                "attempt_id": _initial_attempt(group.request, branch).attempt_id,
                "attempt_request_sha256": _attempt_request_sha256(
                    _initial_attempt(group.request, branch)
                ),
                "attempt_index": 0,
                "replaced_attempt_id": None,
                "superseded_by_attempt_id": None,
                "session_id": branch.session_id,
                "status": None,
                "eligible": False,
            }
            for branch in group.request.branches
        ]
    )
    return Event(
        id=(
            "fork_group_"
            + sha256(
                canonical_durable_json_bytes(
                    {
                        "group_id": result.group_id,
                        "request_sha256": group.request_sha256,
                        "revision": group.revision,
                        "event_type": event_type.value,
                    },
                    "fork_group.event_identity",
                )
            ).hexdigest()
        ),
        type=event_type,
        session_id=result.source.source_session_id,
        payload={
            "schema_version": _FORK_GROUP_SCHEMA_VERSION,
            "revision": group.revision,
            "group_id": result.group_id,
            "state": result.state.value,
            "request_sha256": group.request_sha256,
            "source_run_epoch": result.source.run_epoch,
            "source_transcript_cursor": result.source.transcript_cursor,
            "source_checkpoint_sha256": result.source.checkpoint_sha256,
            "source_execution_profile_fingerprint": (result.source.execution_profile_fingerprint),
            "branches": branch_payload,
            "evaluator_session_id": group.request.evaluator.session_id,
            "dispositions": dispositions,
            "selected_branch_id": selected,
            "selected_attempt_id": selected_attempt,
            "failure_code": None if result.failure is None else result.failure.code.value,
            "failure_branch_id": (None if result.failure is None else result.failure.branch_id),
        },
    )


async def _load_record(
    coordinator: ForkGroupCoordinator,
    source_session_id: str,
    request: ForkGroupRequest,
) -> _ForkGroupRecord | None:
    raw = await coordinator.session_store.load_session_operation(
        source_session_id,
        _storage_key(request.group_id),
    )
    if raw is None:
        return None
    try:
        record = _ForkGroupRecord.model_validate(raw)
    except Exception as exc:
        raise RuntimeError("Durable fork-group operation record is malformed.") from exc
    if record.request_sha256 != _request_sha256(request):
        raise ForkGroupConflict(
            f"Fork group {request.group_id!r} is already bound to a different request."
        )
    return record


def _transition_material(record: _ForkGroupRecord) -> dict[str, Any]:
    """Return immutable group state while excluding its renewable lease."""

    material = record.model_dump(mode="json", warnings=False)
    material.pop("execution_claim", None)
    return material


def _claim_is_live(claim: _ForkGroupExecutionClaim, now: datetime) -> bool:
    return claim.expires_at > now.astimezone(UTC)


async def _claim_execution(
    coordinator: ForkGroupCoordinator,
    source_session_id: str,
    record: _ForkGroupRecord,
) -> tuple[_ForkGroupRecord, str | None]:
    """Atomically lease one exact nonterminal group to this coordinator."""

    key = _storage_key(record.request.group_id)
    claim_id = uuid4().hex
    now = coordinator.now()

    def transform(
        source: Session,
        checkpoint: dict[str, Any] | None,
        current: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        del source
        if current is None:
            raise ForkGroupConflict("Durable fork-group record disappeared before execution.")
        parsed = _ForkGroupRecord.model_validate(current)
        if parsed.request_sha256 != record.request_sha256:
            raise ForkGroupConflict(
                f"Fork group {record.request.group_id!r} is already bound to a different request."
            )
        if parsed.result.state in {ForkGroupState.COMPLETED, ForkGroupState.FAILED}:
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: parsed.model_dump(mode="json", warnings=False)},
            )
        if parsed.execution_claim is not None and _claim_is_live(parsed.execution_claim, now):
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: parsed.model_dump(mode="json", warnings=False)},
            )
        claimed = parsed.model_copy(
            update={
                "execution_claim": _ForkGroupExecutionClaim(
                    claim_id=claim_id,
                    expires_at=now + timedelta(seconds=_FORK_GROUP_EXECUTION_CLAIM_SECONDS),
                )
            }
        )
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else checkpoint,
            operation_records={key: claimed.model_dump(mode="json", warnings=False)},
        )

    try:
        await coordinator.session_store.publish_session_operation(
            source_session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
    except Exception:
        loaded = await _load_record(coordinator, source_session_id, record.request)
        if (
            loaded is None
            or loaded.execution_claim is None
            or loaded.execution_claim.claim_id != claim_id
        ):
            raise
        return loaded, claim_id
    loaded = await _load_record(coordinator, source_session_id, record.request)
    if loaded is None:
        raise RuntimeError("Fork-group execution claim committed without a durable record.")
    owned = loaded.execution_claim
    return loaded, claim_id if owned is not None and owned.claim_id == claim_id else None


async def _update_execution_claim(
    coordinator: ForkGroupCoordinator,
    source_session_id: str,
    record: _ForkGroupRecord,
    claim_id: str,
    *,
    release: bool,
) -> _ForkGroupRecord:
    key = _storage_key(record.request.group_id)
    now = coordinator.now()

    def transform(
        source: Session,
        checkpoint: dict[str, Any] | None,
        current: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        del source
        if current is None:
            raise ForkGroupConflict("Durable fork-group record disappeared during execution.")
        parsed = _ForkGroupRecord.model_validate(current)
        if parsed.request_sha256 != record.request_sha256:
            raise ForkGroupConflict("Fork-group execution claim changed request identity.")
        current_claim = parsed.execution_claim
        if current_claim is None or current_claim.claim_id != claim_id:
            raise _ForkGroupPublicationSuperseded
        updated = parsed.model_copy(
            update={
                "execution_claim": (
                    None
                    if release
                    else current_claim.model_copy(
                        update={
                            "expires_at": now
                            + timedelta(seconds=_FORK_GROUP_EXECUTION_CLAIM_SECONDS)
                        }
                    )
                )
            }
        )
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else checkpoint,
            operation_records={key: updated.model_dump(mode="json", warnings=False)},
        )

    try:
        await coordinator.session_store.publish_session_operation(
            source_session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
    except _ForkGroupPublicationSuperseded:
        raise
    except Exception:
        loaded = await _load_record(coordinator, source_session_id, record.request)
        if loaded is None:
            raise
        loaded_claim = loaded.execution_claim
        if release and loaded_claim is None:
            return loaded
        if (
            not release
            and loaded_claim is not None
            and loaded_claim.claim_id == claim_id
            and loaded_claim.expires_at
            >= now + timedelta(seconds=_FORK_GROUP_EXECUTION_CLAIM_SECONDS)
        ):
            return loaded
        raise
    loaded = await _load_record(coordinator, source_session_id, record.request)
    if loaded is None:
        raise RuntimeError("Fork-group claim update committed without a durable record.")
    return loaded


async def _heartbeat_execution_claim(
    coordinator: ForkGroupCoordinator,
    source_session_id: str,
    record: _ForkGroupRecord,
    claim_id: str,
) -> None:
    while True:
        await asyncio.sleep(_FORK_GROUP_EXECUTION_HEARTBEAT_SECONDS)
        record = await _update_execution_claim(
            coordinator,
            source_session_id,
            record,
            claim_id,
            release=False,
        )


async def _publish_record(
    coordinator: ForkGroupCoordinator,
    source_session_id: str,
    record: _ForkGroupRecord,
    event_type: EventType,
    *,
    expected_run_epoch: int | None = None,
    expected_transcript_cursor: int | None = None,
    expected_record: _ForkGroupRecord | None = None,
) -> _ForkGroupRecord:
    key = _storage_key(record.request.group_id)

    def transform(
        source: Session,
        checkpoint: dict[str, Any] | None,
        current: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        record_to_store = record
        if current is not None:
            parsed = _ForkGroupRecord.model_validate(current)
            if parsed.request_sha256 != record.request_sha256:
                raise ForkGroupConflict(
                    f"Fork group {record.request.group_id!r} is already bound to a "
                    "different request."
                )
            if expected_record is None:
                raise _ForkGroupPublicationSuperseded
            if parsed.revision != expected_record.revision or parsed.result.state is not (
                expected_record.result.state
            ):
                raise _ForkGroupPublicationSuperseded
            if _transition_material(parsed) != _transition_material(expected_record):
                raise ForkGroupConflict(
                    f"Fork group {record.request.group_id!r} durable state changed at the "
                    "expected revision."
                )
            expected_claim = expected_record.execution_claim
            if expected_claim is not None and (
                parsed.execution_claim is None
                or parsed.execution_claim.claim_id != expected_claim.claim_id
            ):
                raise _ForkGroupPublicationSuperseded
            transition = (parsed.result.state, record.result.state)
            if transition not in _FORK_GROUP_TRANSITIONS or record.revision != (
                parsed.revision + 1
            ):
                raise ForkGroupConflict(
                    f"Fork group {record.request.group_id!r} rejected an invalid durable "
                    "state transition."
                )
            record_to_store = record.model_copy(
                update={
                    "execution_claim": (
                        None
                        if record.result.state in {ForkGroupState.COMPLETED, ForkGroupState.FAILED}
                        else parsed.execution_claim
                    )
                }
            )
        elif expected_record is not None:
            raise ForkGroupConflict(
                f"Fork group {record.request.group_id!r} durable state disappeared."
            )
        effective_checkpoint = {} if checkpoint is None else checkpoint
        if expected_run_epoch is not None:
            if source.status != record.result.source.status:
                raise ValueError("Fork-group source status changed before snapshot publication.")
            if source.causal_budget_id != record.result.source.causal_budget_id:
                raise ValueError("Fork-group source causal budget changed before publication.")
            if fork_source_checkpoint_sha256(effective_checkpoint) != (
                record.result.source.checkpoint_sha256
            ):
                raise ValueError("Fork-group source checkpoint changed before publication.")
            _, profile, _ = session_request_boundary.prepare_fork_source_execution_profile(
                source,
                effective_checkpoint,
            )
            if profile.fingerprint != record.result.source.execution_profile_fingerprint:
                raise ValueError("Fork-group source execution profile changed before publication.")
        return SessionOperationPublication(
            checkpoint=effective_checkpoint,
            operation_records={key: record_to_store.model_dump(mode="json", warnings=False)},
        )

    try:
        await coordinator.session_store.publish_session_operation(
            source_session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[_event(record, event_type)],
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
    except _ForkGroupPublicationSuperseded:
        loaded = await _load_record(coordinator, source_session_id, record.request)
        if loaded is None:
            raise RuntimeError("Superseded fork-group publication has no durable winner.") from None
        if (
            expected_record is not None
            and _transition_material(loaded) == _transition_material(expected_record)
            and expected_record.execution_claim is not None
            and (
                loaded.execution_claim is None
                or loaded.execution_claim.claim_id != expected_record.execution_claim.claim_id
            )
        ):
            raise
        return loaded
    except Exception as exc:
        # A backend may commit the transaction and then lose its acknowledgement.
        # Accept only this owner's exact nonterminal transition or an authoritative
        # terminal winner. A successor's same-revision transition is not this
        # owner's lost acknowledgement.
        loaded = await _load_record(coordinator, source_session_id, record.request)
        if loaded is not None and loaded.revision >= record.revision:
            if loaded.result.state in {ForkGroupState.COMPLETED, ForkGroupState.FAILED}:
                return loaded
            expected_claim = None if expected_record is None else expected_record.execution_claim
            if (
                loaded.revision == record.revision
                and _transition_material(loaded) == _transition_material(record)
                and (
                    expected_record is None
                    or (
                        expected_claim is not None
                        and loaded.execution_claim is not None
                        and loaded.execution_claim.claim_id == expected_claim.claim_id
                    )
                )
            ):
                return loaded
            raise _ForkGroupPublicationSuperseded from exc
        if expected_record is not None and expected_record.execution_claim is not None:
            if loaded is not None and (
                loaded.execution_claim is None
                or loaded.execution_claim.claim_id != expected_record.execution_claim.claim_id
            ):
                raise _ForkGroupPublicationSuperseded from exc
            if record.result.branches:
                raise _ForkGroupPublicationFailure(expected_record, record, exc) from exc
        raise
    loaded = await _load_record(coordinator, source_session_id, record.request)
    if loaded is None:
        raise RuntimeError("Fork-group publication committed without a durable record.")
    return loaded


def _result_with(
    record: _ForkGroupRecord,
    *,
    state: ForkGroupState,
    branches: tuple[ForkGroupBranchResult, ...] | None = None,
    evaluator_session_id: str | None = None,
    dispositions: tuple[ForkGroupDispositionRecord, ...] = (),
    failure: ForkGroupFailure | None = None,
) -> _ForkGroupRecord:
    current = record.result
    return record.model_copy(
        update={
            "revision": record.revision + 1,
            "result": current.model_copy(
                update={
                    "state": state,
                    "branches": current.branches if branches is None else branches,
                    "evaluator_session_id": (
                        current.evaluator_session_id
                        if evaluator_session_id is None
                        else evaluator_session_id
                    ),
                    "dispositions": dispositions,
                    "failure": failure,
                    "replayed": False,
                },
                deep=True,
            ),
        },
        deep=True,
    )


def _failure(
    code: ForkGroupFailureCode,
    message: str,
    *,
    branch_id: str | None = None,
) -> ForkGroupFailure:
    return ForkGroupFailure(
        code=code,
        message=message[:2_048],
        branch_id=branch_id,
    )


def _exception_message(
    exc: BaseException,
    *,
    redactor: SecretRedactor,
    prefix: str = "",
) -> str:
    diagnostic = exception_diagnostic(
        exc,
        empty_message="fork-group operation failed",
        nonportable_message="Fork-group operation failed with a non-portable diagnostic.",
        redactor=redactor,
    )
    return redactor.redact_text_bounded(
        f"{prefix}{diagnostic.error_type}: {diagnostic.message}",
        max_bytes=2_048,
    )


async def _run_outcome(
    stream: AsyncIterator[Event],
    *,
    redactor: SecretRedactor,
) -> tuple[
    SessionStatus,
    Any,
    str | None,
    ForkGroupFailureCode | None,
]:
    status = SessionStatus.INTERRUPTED
    output: Any = None
    has_output = False
    error: str | None = None
    failure_code: ForkGroupFailureCode | None = None
    try:
        async for event in stream:
            if event.type == EventType.STRUCTURED_OUTPUT_VALIDATED:
                output = copy_json_value(event.payload.get("output"), "structured_output")
                has_output = True
            elif event.type == EventType.SESSION_COMPLETED:
                status = SessionStatus.COMPLETED
            elif event.type == EventType.SESSION_FAILED:
                status = SessionStatus.FAILED
                raw_error = event.payload.get("error")
                error = raw_error if isinstance(raw_error, str) else "Session failed."
                if failure_code is None:
                    failure_code = ForkGroupFailureCode.BRANCH_FAILED
            elif event.type == EventType.SESSION_INTERRUPTED:
                status = SessionStatus.INTERRUPTED
                if error is None:
                    error = "Session was interrupted."
            elif event.type in {
                EventType.BUDGET_LIMIT_REACHED,
                EventType.BUDGET_RESERVATION_FAILED,
            } or (
                event.type == EventType.SESSION_LIMIT_REACHED
                and (event.payload.get("limit") == "estimated_cost")
            ):
                error = "Fork-group causal budget was exhausted."
                failure_code = ForkGroupFailureCode.BUDGET_EXHAUSTED
    except Exception as exc:
        status = SessionStatus.FAILED
        error = _exception_message(exc, redactor=redactor)
        failure_code = ForkGroupFailureCode.BRANCH_FAILED
    return status, output if has_output else _MISSING, error, failure_code


_MISSING = object()


async def _reconstruct_session_output(coordinator: ForkGroupCoordinator, session_id: str) -> Any:
    from cayu.runtime.sessions import EventOrder, EventQuery

    records = await coordinator.session_store.query_events(
        EventQuery(
            session_id=session_id,
            event_types=(EventType.STRUCTURED_OUTPUT_VALIDATED,),
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not records:
        return _MISSING
    return copy_json_value(records[0].event.payload.get("output"), "structured_output")


async def _reconstruct_session_failure_code(
    coordinator: ForkGroupCoordinator,
    session_id: str,
) -> ForkGroupFailureCode | None:
    from cayu.runtime.sessions import EventOrder, EventQuery

    records = await coordinator.session_store.query_events(
        EventQuery(
            session_id=session_id,
            event_types=(
                EventType.BUDGET_LIMIT_REACHED,
                EventType.BUDGET_RESERVATION_FAILED,
                EventType.SESSION_LIMIT_REACHED,
            ),
            order_by=EventOrder.SEQUENCE_DESC,
            limit=100,
        )
    )
    if any(
        record.event.type in {EventType.BUDGET_LIMIT_REACHED, EventType.BUDGET_RESERVATION_FAILED}
        or (
            record.event.type == EventType.SESSION_LIMIT_REACHED
            and record.event.payload.get("limit") == "estimated_cost"
        )
        for record in records
    ):
        return ForkGroupFailureCode.BUDGET_EXHAUSTED
    return None


async def _recover_child_for_continuation(
    coordinator: ForkGroupCoordinator,
    child: Session,
    request: IncompleteSessionRecoveryRequest,
    *,
    role: str,
    pending_message: str,
) -> tuple[Session | None, str | None]:
    if child.run_epoch == 0 or child.status in _TERMINAL_SESSION_STATUSES:
        return child, None

    recovery = None
    error: str | None = None
    try:
        recovery = await coordinator.recover_incomplete_session(request)
    except Exception as exc:
        error = _exception_message(
            exc,
            redactor=coordinator.secret_redactor,
            prefix=f"Incomplete {role} recovery failed: ",
        )
    if recovery is not None and (
        IncompleteSessionRecoveryAction.SKIPPED_ACTIVE in recovery.actions
        and recovery.status not in _TERMINAL_SESSION_STATUSES
    ):
        raise _ForkGroupContinuationPending(pending_message)
    return await coordinator.session_store.load(request.session_id), error


async def _fork_exact_source(
    coordinator: ForkGroupCoordinator,
    request: ForkSessionRequest,
    *,
    source: ForkGroupSourceSnapshot,
) -> tuple[SessionStatus, str | None, ForkGroupFailureCode | None, str | None]:
    selected_profile_fingerprints: list[str] = []
    bound = _bind_fork_expected_source_snapshot(
        request,
        expected_source_run_epoch=source.run_epoch,
        expected_source_transcript_cursor=source.transcript_cursor,
        expected_source_transcript_sha256=source.transcript_sha256,
        expected_source_checkpoint_sha256=source.checkpoint_sha256,
        expected_source_profile_fingerprint=source.execution_profile_fingerprint,
    )
    bound = _bind_fork_execution_profile_fingerprint_capture(
        bound,
        selected_profile_fingerprints.append,
    )
    try:
        async for _ in coordinator.fork_session(bound):
            pass
    except Exception as exc:
        return (
            SessionStatus.FAILED,
            _exception_message(
                exc,
                redactor=coordinator.secret_redactor,
            ),
            (
                ForkGroupFailureCode.SOURCE_CHANGED
                if not await _source_matches_frozen_snapshot(coordinator, source)
                else ForkGroupFailureCode.BRANCH_FAILED
            ),
            (
                selected_profile_fingerprints[-1]
                if selected_profile_fingerprints
                else (
                    source.execution_profile_fingerprint
                    if bound.execution_profile_selection
                    is ForkExecutionProfileSelection.INHERIT_PARENT
                    else None
                )
            ),
        )
    child = await coordinator.session_store.load(request.session_id or "")
    if child is None:
        return (
            SessionStatus.FAILED,
            "Fork completed without a durable child session.",
            ForkGroupFailureCode.BRANCH_FAILED,
            (selected_profile_fingerprints[-1] if selected_profile_fingerprints else None),
        )
    profile = execution_profile_baseline_from_session_metadata(child.metadata)
    return child.status, None, None, None if profile is None else profile.fingerprint


async def _prepare_branch_fork(
    coordinator: ForkGroupCoordinator,
    request: ForkGroupRequest,
    source: ForkGroupSourceSnapshot,
    attempt: _ForkGroupAttempt | ForkGroupBranchSpec,
) -> ForkGroupBranchResult | None:
    attempt = _coerce_attempt(request, attempt)
    branch = attempt.branch
    existing = await coordinator.session_store.load(branch.session_id)
    if existing is not None and not _branch_matches_frozen_source(existing, source):
        return ForkGroupBranchResult(
            branch_id=branch.branch_id,
            attempt_id=attempt.attempt_id,
            attempt_request_sha256=_attempt_request_sha256(attempt),
            attempt_index=attempt.attempt_index,
            replaced_attempt_id=attempt.replaced_attempt_id,
            session_id=branch.session_id,
            status=ForkGroupBranchStatus.FAILED,
            failure_code=ForkGroupFailureCode.SOURCE_CHANGED,
            source_checkpoint_sha256=source.checkpoint_sha256,
            causal_budget_id=request.causal_budget_id,
            artifact_references=branch.artifact_references,
            usage=await coordinator.get_session_usage(branch.session_id),
            error="Branch session source snapshot changed from the frozen group authority.",
        )
    (
        fork_status,
        fork_error,
        fork_failure_code,
        prepared_profile_fingerprint,
    ) = await _fork_exact_source(
        coordinator,
        _branch_fork_request(request, attempt),
        source=source,
    )
    if fork_error is not None:
        child = await coordinator.session_store.load(branch.session_id)
        if child is not None:
            _require_branch_authority(
                child,
                request=request,
                source=source,
                attempt=attempt,
            )
            return None
        return ForkGroupBranchResult(
            branch_id=branch.branch_id,
            attempt_id=attempt.attempt_id,
            attempt_request_sha256=_attempt_request_sha256(attempt),
            attempt_index=attempt.attempt_index,
            replaced_attempt_id=attempt.replaced_attempt_id,
            session_id=branch.session_id,
            status=ForkGroupBranchStatus.FAILED,
            failure_code=fork_failure_code or ForkGroupFailureCode.BRANCH_FAILED,
            source_checkpoint_sha256=source.checkpoint_sha256,
            causal_budget_id=request.causal_budget_id,
            execution_profile_fingerprint=prepared_profile_fingerprint,
            artifact_references=branch.artifact_references,
            usage=SessionUsageSummary(session_id=branch.session_id),
            error=fork_error,
        )
    child = await coordinator.session_store.load(branch.session_id)
    if child is None:
        raise RuntimeError("Forked branch could not be reconstructed.")
    _require_branch_authority(
        child,
        request=request,
        source=source,
        attempt=attempt,
    )
    if fork_status not in _TERMINAL_SESSION_STATUSES and child.run_epoch == 0:
        raise RuntimeError("Newly forked branch did not inherit a terminal source status.")
    return None


async def _run_branch(
    coordinator: ForkGroupCoordinator,
    request: ForkGroupRequest,
    source: ForkGroupSourceSnapshot,
    attempt: _ForkGroupAttempt | ForkGroupBranchSpec,
) -> ForkGroupBranchResult:
    attempt = _coerce_attempt(request, attempt)
    branch = attempt.branch
    child = await coordinator.session_store.load(branch.session_id)
    if child is None:
        raise RuntimeError("Prepared fork-group branch disappeared before execution.")
    _require_branch_authority(
        child,
        request=request,
        source=source,
        attempt=attempt,
    )

    output = _MISSING
    error: str | None = None
    failure_code: ForkGroupFailureCode | None = None
    child, recovery_error = await _recover_child_for_continuation(
        coordinator,
        child,
        IncompleteSessionRecoveryRequest(
            session_id=branch.session_id,
            reason="fork_group_recovered_incomplete_branch",
            metadata={
                "fork_group_id": request.group_id,
                "fork_group_branch_id": branch.branch_id,
                "fork_group_attempt_id": attempt.attempt_id,
            },
        ),
        role="branch",
        pending_message=(
            f"Branch session {branch.session_id!r} still has an active provider operation."
        ),
    )
    if recovery_error is not None:
        error = recovery_error
        failure_code = ForkGroupFailureCode.BRANCH_FAILED
    if child is None:
        raise RuntimeError("Recovered fork-group branch disappeared.")
    if child.run_epoch == 0:
        status, output, error, failure_code = await _run_outcome(
            coordinator.resume(_branch_resume_request(request, attempt)),
            redactor=coordinator.secret_redactor,
        )
    else:
        status = child.status
        output = await _reconstruct_session_output(coordinator, branch.session_id)
        failure_code = await _reconstruct_session_failure_code(
            coordinator,
            branch.session_id,
        )
        if status is SessionStatus.FAILED:
            error = (
                "Fork-group causal budget was exhausted."
                if failure_code is ForkGroupFailureCode.BUDGET_EXHAUSTED
                else "Branch session failed before fork-group reconstruction."
            )
        elif status is SessionStatus.INTERRUPTED:
            error = (
                "Fork-group causal budget was exhausted."
                if failure_code is ForkGroupFailureCode.BUDGET_EXHAUSTED
                else "Branch session was interrupted before fork-group reconstruction."
            )

    usage = await coordinator.get_session_usage(branch.session_id)
    branch_status = {
        SessionStatus.COMPLETED: ForkGroupBranchStatus.COMPLETED,
        SessionStatus.FAILED: ForkGroupBranchStatus.FAILED,
        SessionStatus.INTERRUPTED: ForkGroupBranchStatus.INTERRUPTED,
    }.get(status, ForkGroupBranchStatus.INVALID)
    if branch_status is ForkGroupBranchStatus.COMPLETED:
        if branch.structured_output is not None and output is _MISSING:
            branch_status = ForkGroupBranchStatus.INVALID
            error = "Completed branch has no validated structured output."
        elif output is not _MISSING:
            output_bytes = canonical_durable_json_bytes(output, "fork_group.branch_output")
            if len(output_bytes) > FORK_GROUP_MAX_BRANCH_OUTPUT_BYTES:
                branch_status = ForkGroupBranchStatus.INVALID
                error = "Validated branch output exceeds the fork-group evidence limit."
                output = _MISSING
    profile = execution_profile_baseline_from_session_metadata(child.metadata)
    return ForkGroupBranchResult(
        branch_id=branch.branch_id,
        attempt_id=attempt.attempt_id,
        attempt_request_sha256=_attempt_request_sha256(attempt),
        attempt_index=attempt.attempt_index,
        replaced_attempt_id=attempt.replaced_attempt_id,
        session_id=branch.session_id,
        status=branch_status,
        failure_code=failure_code,
        source_checkpoint_sha256=source.checkpoint_sha256,
        causal_budget_id=request.causal_budget_id,
        execution_profile_fingerprint=None if profile is None else profile.fingerprint,
        structured_output=None if output is _MISSING else output,
        has_structured_output=output is not _MISSING,
        artifact_references=branch.artifact_references,
        usage=usage,
        error=error,
    )


def _evidence_payload(record: _ForkGroupRecord) -> dict[str, Any]:
    source = record.result.source
    viable_mode = record.request.failure_policy.mode is ForkGroupFailureMode.EVALUATE_VIABLE
    branches: list[dict[str, Any]] = []
    excluded_attempts: list[dict[str, Any]] = []
    for branch in record.result.branches:
        if viable_mode and not branch.eligible:
            excluded_attempts.append(
                {
                    "attempt_id": branch.attempt_id,
                    "attempt_index": branch.attempt_index,
                    "branch_id": branch.branch_id,
                    "replaced_attempt_id": branch.replaced_attempt_id,
                    "session_id": branch.session_id,
                    "status": branch.status.value,
                    "superseded_by_attempt_id": branch.superseded_by_attempt_id,
                    "gate_results": [gate.model_dump(mode="json") for gate in branch.gate_results],
                    "error": branch.error,
                }
            )
            continue
        item: dict[str, Any] = {
            "branch_id": branch.branch_id,
            "session_id": branch.session_id,
            "status": branch.status.value,
            "usage": branch.usage.model_dump(mode="json"),
            "artifact_references": [
                artifact.model_dump(mode="json") for artifact in branch.artifact_references
            ],
            "gate_results": [gate.model_dump(mode="json") for gate in branch.gate_results],
        }
        if viable_mode:
            item.update(
                {
                    "attempt_id": branch.attempt_id,
                    "attempt_index": branch.attempt_index,
                    "replaced_attempt_id": branch.replaced_attempt_id,
                }
            )
        if branch.has_structured_output:
            item["structured_output"] = copy_json_value(
                branch.structured_output,
                "structured_output",
            )
        branches.append(item)
    evidence = {
        "schema": (
            "cayu.fork-group-evidence.v2"
            if record.request.failure_policy.mode is ForkGroupFailureMode.EVALUATE_VIABLE
            else "cayu.fork-group-evidence.v1"
        ),
        "group_id": record.request.group_id,
        "source": {
            "session_id": source.source_session_id,
            "run_epoch": source.run_epoch,
            "transcript_cursor": source.transcript_cursor,
            "transcript_sha256": source.transcript_sha256,
            "checkpoint_sha256": source.checkpoint_sha256,
            "execution_profile_fingerprint": source.execution_profile_fingerprint,
            "causal_budget_id": source.causal_budget_id,
        },
        "branches": branches,
    }
    if viable_mode:
        evidence["excluded_attempts"] = excluded_attempts
    if len(canonical_durable_json_bytes(evidence, "fork_group.evidence")) > (
        FORK_GROUP_MAX_EVIDENCE_BYTES
    ):
        raise ValueError("Fork-group evaluator evidence exceeds the bounded evidence limit.")
    return evidence


def _judgment_spec(
    record: _ForkGroupRecord,
    branch_identities: tuple[_ForkGroupAttemptIdentity, ...],
) -> StructuredOutputSpec:
    branch_ids = tuple(identity.branch_id for identity in branch_identities)
    minimum_items = len(branch_ids)
    maximum_items = len(branch_ids)
    disposition_properties: dict[str, Any] = {
        "branch_id": {"type": "string", "enum": list(branch_ids)},
        "disposition": {
            "type": "string",
            "enum": [item.value for item in ForkGroupDisposition],
        },
        "reason": {
            "type": "string",
            "maxLength": FORK_GROUP_MAX_REASON_CHARS,
        },
    }
    required = ["branch_id", "disposition", "reason"]
    return StructuredOutputSpec(
        name="fork-group-judgment",
        json_schema={
            "type": "object",
            "properties": {
                "dispositions": {
                    "type": "array",
                    "minItems": minimum_items,
                    "maxItems": maximum_items,
                    "items": {
                        "type": "object",
                        "properties": disposition_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["dispositions"],
            "additionalProperties": False,
        },
    )


def _evaluator_metadata(record: _ForkGroupRecord) -> dict[str, str]:
    return {
        "fork_group_id": record.request.group_id,
        "fork_group_role": "evaluator",
    }


def _evaluator_messages(record: _ForkGroupRecord) -> list[Message]:
    evidence = _evidence_payload(record)
    return [
        Message.text(
            "user",
            canonical_durable_json_bytes(evidence, "fork_group.evidence").decode("utf-8"),
        )
    ]


def _evaluator_execution_material(
    record: _ForkGroupRecord,
    *,
    branch_identities: tuple[_ForkGroupAttemptIdentity, ...],
) -> dict[str, Any]:
    """Return the one evaluator continuation contract shared by run and resume."""

    evaluator = record.request.evaluator
    return {
        "messages": _evaluator_messages(record),
        "max_steps": evaluator.max_steps,
        "limits": evaluator.limits,
        "budget_limits": evaluator.budget_limits,
        "retry_policy": evaluator.retry_policy,
        "structured_output": _judgment_spec(record, branch_identities),
        "thinking": evaluator.thinking,
        "metadata": _evaluator_metadata(record),
    }


def _evaluator_run_request(
    record: _ForkGroupRecord,
    *,
    synthetic_agent_name: str,
    branch_identities: tuple[_ForkGroupAttemptIdentity, ...],
) -> RunRequest:
    evaluator = record.request.evaluator
    return RunRequest(
        session_id=evaluator.session_id,
        agent_name=synthetic_agent_name,
        causal_budget_id=record.request.causal_budget_id,
        **_evaluator_execution_material(record, branch_identities=branch_identities),
    )


def _evaluator_resume_request(
    record: _ForkGroupRecord,
    *,
    branch_identities: tuple[_ForkGroupAttemptIdentity, ...],
) -> ResumeRequest:
    evaluator = record.request.evaluator
    return ResumeRequest(
        session_id=evaluator.session_id,
        **_evaluator_execution_material(record, branch_identities=branch_identities),
    )


def _synthetic_evaluator_name(record: _ForkGroupRecord) -> str:
    evaluator = record.request.evaluator
    return (
        "__cayu_fork_group_evaluator_"
        + sha256(
            canonical_durable_json_bytes(
                {
                    "group_id": record.request.group_id,
                    "agent_name": evaluator.agent_name,
                    "request_sha256": record.request_sha256,
                },
                "fork_group.evaluator_agent",
            )
        ).hexdigest()[:24]
    )


async def _tool_free_evaluator_authority(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
    *,
    branch_identities: tuple[_ForkGroupAttemptIdentity, ...],
) -> tuple[str, str]:
    evaluator = record.request.evaluator
    synthetic_name = _synthetic_evaluator_name(record)
    return await coordinator.prepare_evaluator_agent(
        _evaluator_run_request(
            record,
            synthetic_agent_name=synthetic_name,
            branch_identities=branch_identities,
        ),
        source_agent_name=evaluator.agent_name,
        request_sha256=record.request_sha256,
        store_resolved_existing_session_id=record.result.evaluator_session_id,
    )


async def _preflight_tool_free_evaluator_authority(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
) -> tuple[str, str]:
    evaluator = record.request.evaluator
    synthetic_name = _synthetic_evaluator_name(record)
    branch_identities = tuple(
        _ForkGroupAttemptIdentity(
            branch_id=branch.branch_id,
            attempt_id=_initial_attempt(record.request, branch).attempt_id,
        )
        for branch in record.request.branches
    )
    return await coordinator.preflight_evaluator_agent(
        _evaluator_run_request(
            record,
            synthetic_agent_name=synthetic_name,
            branch_identities=branch_identities,
        ),
        source_agent_name=evaluator.agent_name,
        request_sha256=record.request_sha256,
    )


def _validate_judgment(
    output: Any,
    branch_identities: tuple[_ForkGroupAttemptIdentity, ...],
) -> tuple[ForkGroupDispositionRecord, ...]:
    if type(output) is not dict or type(output.get("dispositions")) is not list:
        raise ValueError("Evaluator judgment has no dispositions array.")
    attempts_by_branch = {identity.branch_id: identity.attempt_id for identity in branch_identities}
    normalized: list[Any] = []
    for item in output["dispositions"]:
        if type(item) is not dict:
            normalized.append(item)
            continue
        branch_id = item.get("branch_id")
        normalized.append({**item, "attempt_id": attempts_by_branch.get(branch_id)})
    dispositions = tuple(ForkGroupDispositionRecord.model_validate(item) for item in normalized)
    judged_identities = [
        _ForkGroupAttemptIdentity(branch_id=item.branch_id, attempt_id=item.attempt_id)
        for item in dispositions
    ]
    if len(judged_identities) != len(set(judged_identities)) or set(judged_identities) != set(
        branch_identities
    ):
        raise ValueError("Evaluator must cover every eligible attempt exactly once.")
    if sum(item.disposition is ForkGroupDisposition.SELECTED for item in dispositions) != 1:
        raise ValueError("Evaluator must select exactly one successful branch.")
    by_id = {
        _ForkGroupAttemptIdentity(branch_id=item.branch_id, attempt_id=item.attempt_id): item
        for item in dispositions
    }
    return tuple(by_id[identity] for identity in branch_identities)


def _evaluator_branch_identities(
    record: _ForkGroupRecord,
) -> tuple[_ForkGroupAttemptIdentity, ...]:
    attempts = (
        tuple(branch for branch in record.result.branches if branch.eligible)
        if record.request.failure_policy.mode is ForkGroupFailureMode.EVALUATE_VIABLE
        else record.result.branches
    )
    return tuple(
        _ForkGroupAttemptIdentity(branch_id=branch.branch_id, attempt_id=branch.attempt_id)
        for branch in attempts
    )


async def _bind_tool_free_evaluator_authority(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
) -> _ForkGroupRecord:
    _, profile_fingerprint = await _tool_free_evaluator_authority(
        coordinator,
        record,
        branch_identities=_evaluator_branch_identities(record),
    )
    return record.model_copy(
        update={"evaluator_execution_profile_fingerprint": profile_fingerprint}
    )


async def _run_evaluator(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
) -> tuple[tuple[ForkGroupDispositionRecord, ...] | None, ForkGroupFailure | None]:
    evaluator = record.request.evaluator
    branch_identities = _evaluator_branch_identities(record)
    synthetic_name, profile_fingerprint = await _tool_free_evaluator_authority(
        coordinator,
        record,
        branch_identities=branch_identities,
    )
    if profile_fingerprint != record.evaluator_execution_profile_fingerprint:
        raise ForkGroupConflict(
            "Evaluator execution profile changed from the frozen fork-group authority."
        )
    child = await coordinator.session_store.load(evaluator.session_id)
    ran_new = False
    output: Any = _MISSING
    error: str | None = None
    if child is None:
        status, output, error, _ = await _run_outcome(
            coordinator.run(
                _evaluator_run_request(
                    record,
                    synthetic_agent_name=synthetic_name,
                    branch_identities=branch_identities,
                )
            ),
            redactor=coordinator.secret_redactor,
        )
        ran_new = True
        child = await coordinator.session_store.load(evaluator.session_id)
    if child is None:
        return None, _failure(
            ForkGroupFailureCode.EVALUATOR_FAILED,
            "Evaluator completed without a durable session.",
        )
    child_profile = execution_profile_baseline_from_session_metadata(child.metadata)
    if (
        child.parent_session_id is not None
        or child.agent_name != synthetic_name
        or child_profile is None
        or child_profile.fingerprint != record.evaluator_execution_profile_fingerprint
        or child.metadata.get("fork_group_id") != record.request.group_id
        or child.metadata.get("fork_group_role") != "evaluator"
    ):
        raise ForkGroupConflict("Evaluator session conflicts with the fork-group request.")
    if child.causal_budget_id != record.request.causal_budget_id:
        raise ForkGroupConflict("Evaluator session does not share the fork-group causal budget.")

    if child.run_epoch == 0:
        status, output, error, _ = await _run_outcome(
            coordinator.resume(
                _evaluator_resume_request(
                    record,
                    branch_identities=branch_identities,
                )
            ),
            redactor=coordinator.secret_redactor,
        )
    elif not ran_new:
        child, recovery_error = await _recover_child_for_continuation(
            coordinator,
            child,
            IncompleteSessionRecoveryRequest(
                session_id=evaluator.session_id,
                reason="fork_group_recovered_incomplete_evaluator",
                metadata=_evaluator_metadata(record),
            ),
            role="evaluator",
            pending_message="Evaluator still has an active provider operation.",
        )
        if recovery_error is not None:
            error = recovery_error
        if child is None:
            return None, _failure(
                ForkGroupFailureCode.EVALUATOR_FAILED,
                "Recovered evaluator session disappeared.",
            )
        status = child.status
        output = await _reconstruct_session_output(coordinator, evaluator.session_id)
    if status is not SessionStatus.COMPLETED:
        return None, _failure(
            ForkGroupFailureCode.EVALUATOR_FAILED,
            error or f"Evaluator ended with status {status.value}.",
        )
    if output is _MISSING:
        return None, _failure(
            ForkGroupFailureCode.JUDGMENT_INVALID,
            "Evaluator completed without a validated judgment.",
        )
    try:
        dispositions = _validate_judgment(output, branch_identities)
        return (
            tuple(
                item.model_copy(
                    update={"reason": coordinator.secret_redactor.redact_text(item.reason)},
                    deep=True,
                )
                for item in dispositions
            ),
            None,
        )
    except Exception as exc:
        return None, _failure(
            ForkGroupFailureCode.JUDGMENT_INVALID,
            _exception_message(exc, redactor=coordinator.secret_redactor),
        )


async def _plan_replacement(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
    replaced_attempt: ForkGroupBranchResult,
) -> _ForkGroupAttempt:
    policy = record.request.failure_policy
    selection = policy.replacement_planner
    if selection is None:
        raise RuntimeError("Evaluate-viable policy has no replacement planner authority.")
    planner = coordinator.replacement_planner(selection.planner_id)
    if planner is None:
        raise RuntimeError(
            f"Fork-group replacement planner {selection.planner_id!r} is not registered."
        )
    if planner.identity != selection.planner_identity:
        raise RuntimeError(
            f"Fork-group replacement planner {selection.planner_id!r} changed identity."
        )
    attempt_index = replaced_attempt.attempt_index + 1
    attempt_id = _attempt_id(record.request, replaced_attempt.branch_id, attempt_index)
    planner_request = ForkGroupReplacementPlannerRequest(
        group_id=record.request.group_id,
        source=record.result.source,
        branch_id=replaced_attempt.branch_id,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        idempotency_key=attempt_id,
        replaced_attempt=replaced_attempt,
    )
    replacement = await planner.plan(planner_request)
    if type(replacement) is not ForkGroupReplacementSpec:
        raise TypeError("Fork-group replacement planner must return ForkGroupReplacementSpec.")
    branch = ForkGroupBranchSpec(
        branch_id=replaced_attempt.branch_id,
        session_id=_replacement_session_id(attempt_id),
        agent_name=replacement.agent_name,
        profile_adoption=replacement.profile_adoption,
        messages=replacement.messages,
        max_steps=replacement.max_steps,
        limits=replacement.limits,
        budget_limits=replacement.budget_limits,
        retry_policy=replacement.retry_policy,
        structured_output=replacement.structured_output,
        thinking=replacement.thinking,
        artifact_references=replacement.artifact_references,
    )
    attempt = _ForkGroupAttempt(
        branch=branch,
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        replaced_attempt_id=replaced_attempt.attempt_id,
    )
    redactor = coordinator.secret_redactor
    prepared_fork = session_request_boundary.prepare_fork_session_request(
        _branch_fork_request(record.request, attempt),
        redactor=redactor,
        store_resolved_source_session_id=record.request.source_session_id,
    )
    prepared_resume = session_request_boundary.prepare_resume_request(
        _branch_resume_request(record.request, attempt),
        redactor=redactor,
    )
    artifacts: list[ForkGroupArtifactReference] = []
    for artifact_index, artifact in enumerate(branch.artifact_references):
        session_request_boundary.require_secret_free_session_authority(
            artifact.artifact_id,
            field_name=(
                f"replacement[{attempt.attempt_id}].artifact_references["
                f"{artifact_index}].artifact_id"
            ),
            redactor=redactor,
            authority_kind="durable fork-group artifact authority",
        )
        artifacts.append(
            artifact.model_copy(
                update={
                    "description": (
                        None
                        if artifact.description is None
                        else redactor.redact_text(artifact.description)
                    )
                },
                deep=True,
            )
        )
    prepared_branch = branch.model_copy(
        update={
            "agent_name": prepared_fork.agent_name,
            "profile_adoption": prepared_fork.profile_adoption,
            "messages": tuple(prepared_resume.messages),
            "structured_output": prepared_resume.structured_output,
            "artifact_references": tuple(artifacts),
        },
        deep=True,
    )
    return _ForkGroupAttempt(
        branch=prepared_branch,
        attempt_id=attempt.attempt_id,
        attempt_index=attempt.attempt_index,
        replaced_attempt_id=attempt.replaced_attempt_id,
    )


def _branch_failure(branches: tuple[ForkGroupBranchResult, ...]) -> ForkGroupFailure | None:
    for branch in branches:
        if branch.failure_code is ForkGroupFailureCode.BUDGET_EXHAUSTED:
            return _failure(
                ForkGroupFailureCode.BUDGET_EXHAUSTED,
                branch.error or "Fork-group causal budget was exhausted.",
                branch_id=branch.branch_id,
            )
        if branch.status is ForkGroupBranchStatus.FAILED:
            code = branch.failure_code or ForkGroupFailureCode.BRANCH_FAILED
            return _failure(code, branch.error or "Branch failed.", branch_id=branch.branch_id)
        if branch.status is not ForkGroupBranchStatus.COMPLETED:
            return _failure(
                ForkGroupFailureCode.BRANCH_INVALID,
                branch.error or f"Branch ended with status {branch.status.value}.",
                branch_id=branch.branch_id,
            )
        failed_gate = next((gate for gate in branch.gate_results if not gate.passed), None)
        if failed_gate is not None:
            return _failure(
                ForkGroupFailureCode.GATE_FAILED,
                f"Deterministic gate {failed_gate.gate_id!r} failed.",
                branch_id=branch.branch_id,
            )
    return None


async def _apply_gates(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
    branches: tuple[ForkGroupBranchResult, ...],
) -> tuple[tuple[ForkGroupBranchResult, ...], ForkGroupFailure | None]:
    if not record.request.gates:
        return (
            tuple(
                branch.model_copy(
                    update={"eligible": branch.status is ForkGroupBranchStatus.COMPLETED},
                    deep=True,
                )
                for branch in branches
            ),
            None,
        )
    fail_on_rejection = record.request.failure_policy.mode is ForkGroupFailureMode.FAIL_GROUP
    gated: list[ForkGroupBranchResult] = []
    for branch in branches:
        if branch.status is not ForkGroupBranchStatus.COMPLETED:
            gated.append(branch)
            continue
        results: list[ForkGroupGateResult] = []
        rejected = False
        for selection in record.request.gates:
            gate_id = selection.gate_id
            gate = coordinator.gate(gate_id)
            if gate is None:
                failure = _failure(
                    ForkGroupFailureCode.GATE_FAILED,
                    f"Deterministic fork-group gate {gate_id!r} is not registered.",
                    branch_id=branch.branch_id,
                )
                gated.append(branch.model_copy(update={"gate_results": tuple(results)}, deep=True))
                gated.extend(branches[len(gated) :])
                return tuple(gated), failure
            if gate.identity != selection.gate_identity:
                failure = _failure(
                    ForkGroupFailureCode.GATE_FAILED,
                    f"Deterministic fork-group gate {gate_id!r} has a different identity.",
                    branch_id=branch.branch_id,
                )
                gated.append(branch.model_copy(update={"gate_results": tuple(results)}, deep=True))
                gated.extend(branches[len(gated) :])
                return tuple(gated), failure
            try:
                decision = await gate.evaluate(
                    ForkGroupGateRequest(
                        group_id=record.request.group_id,
                        source=record.result.source,
                        branch=branch,
                    )
                )
                if type(decision) is not ForkGroupGateDecision:
                    raise TypeError("Fork-group gate must return ForkGroupGateDecision.")
            except Exception as exc:
                failure = _failure(
                    ForkGroupFailureCode.GATE_FAILED,
                    _exception_message(
                        exc,
                        redactor=coordinator.secret_redactor,
                        prefix=f"Deterministic gate {gate_id!r} failed: ",
                    ),
                    branch_id=branch.branch_id,
                )
                gated.append(branch.model_copy(update={"gate_results": tuple(results)}, deep=True))
                gated.extend(branches[len(gated) :])
                return tuple(gated), failure
            result = ForkGroupGateResult(
                gate_id=gate_id,
                passed=decision.passed,
                summary=(
                    None
                    if decision.summary is None
                    else coordinator.secret_redactor.redact_text(decision.summary)
                ),
            )
            results.append(result)
            if not result.passed:
                rejected = True
                if fail_on_rejection:
                    gated.append(
                        branch.model_copy(
                            update={"gate_results": tuple(results), "eligible": False},
                            deep=True,
                        )
                    )
                    gated.extend(branches[len(gated) :])
                    return tuple(gated), _failure(
                        ForkGroupFailureCode.GATE_FAILED,
                        f"Deterministic gate {gate_id!r} failed.",
                        branch_id=branch.branch_id,
                    )
                break
        gated.append(
            branch.model_copy(
                update={"gate_results": tuple(results), "eligible": not rejected},
                deep=True,
            )
        )
    return tuple(gated), None


async def _create_record(
    coordinator: ForkGroupCoordinator,
    request: ForkGroupRequest,
) -> _ForkGroupRecord:
    source = await coordinator.session_store.load(request.source_session_id)
    if source is None:
        raise KeyError("Fork-group source session was not found.")
    if source.status not in _TERMINAL_SESSION_STATUSES:
        raise ValueError("Fork-group source must be terminal before its snapshot is frozen.")
    if source.causal_budget_id != request.causal_budget_id:
        raise ValueError("Fork-group causal_budget_id must match the source session.")
    snapshot = await coordinator.session_store.load_transcript_snapshot(source.id)
    checkpoint = await coordinator.session_store.load_checkpoint(source.id)
    await coordinator.preflight_fork_source_state(source, checkpoint)
    effective_checkpoint = {} if checkpoint is None else checkpoint
    _, profile, _ = session_request_boundary.prepare_fork_source_execution_profile(
        source,
        effective_checkpoint,
    )
    selector = request.source_checkpoint
    if selector.expected_run_epoch is not None and selector.expected_run_epoch != source.run_epoch:
        raise ValueError("Fork-group source run epoch does not match the checkpoint selector.")
    if selector.expected_transcript_cursor is not None and (
        selector.expected_transcript_cursor != snapshot.cursor
    ):
        raise ValueError("Fork-group transcript cursor does not match the checkpoint selector.")
    if selector.expected_profile_fingerprint is not None and (
        selector.expected_profile_fingerprint != profile.fingerprint
    ):
        raise ValueError("Fork-group profile does not match the checkpoint selector.")
    transcript = [record.message for record in snapshot.records]
    if not session_request_boundary.fork_transcript_is_secret_free(
        tuple(transcript),
        redactor=coordinator.secret_redactor,
    ):
        raise ValueError(
            "Fork-group source transcript contains a workload secret and cannot be copied."
        )
    if not session_request_boundary.fork_checkpoint_is_secret_free(
        effective_checkpoint,
        redactor=coordinator.secret_redactor,
    ):
        raise ValueError(
            "Fork-group source checkpoint contains a workload secret and cannot be copied."
        )
    source_snapshot = ForkGroupSourceSnapshot(
        source_session_id=source.id,
        status=source.status,
        run_epoch=source.run_epoch,
        transcript_cursor=snapshot.cursor,
        transcript_sha256=session_input_messages_sha256(transcript),
        checkpoint_sha256=fork_source_checkpoint_sha256(effective_checkpoint),
        execution_profile_fingerprint=profile.fingerprint,
        causal_budget_id=source.causal_budget_id,
    )
    record = _ForkGroupRecord(
        revision=0,
        request_sha256=_request_sha256(request),
        request=request,
        result=ForkGroupResult(
            group_id=request.group_id,
            state=ForkGroupState.CREATED,
            source=source_snapshot,
        ),
    )
    # Resolve every deterministic evaluator dependency without registering its
    # synthetic agent or admitting either session. Known-invalid groups must
    # not consume the source's future contract-attachment authority.
    await _preflight_tool_free_evaluator_authority(coordinator, record)
    # The exact eligible-only evaluator schema is not known until candidate
    # execution finishes. Close the source attachment race now, but defer
    # evaluator registration and admission until that cohort is durable.
    await coordinator.admit_fork_source(source.id)
    return await _publish_record(
        coordinator,
        source.id,
        record,
        EventType.FORK_GROUP_CREATED,
        expected_run_epoch=source.run_epoch,
        expected_transcript_cursor=snapshot.cursor,
    )


async def _run_attempt_batch(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
    attempts: tuple[_ForkGroupAttempt, ...],
    *,
    parallelism: int,
) -> tuple[
    tuple[ForkGroupBranchResult, ...],
    ForkGroupFailure | None,
    frozenset[str],
]:
    fork_failures: dict[str, ForkGroupBranchResult] = {}
    runnable: list[_ForkGroupAttempt] = []
    for attempt in attempts:
        outcome = await _prepare_branch_fork(
            coordinator,
            record.request,
            record.result.source,
            attempt,
        )
        if outcome is None:
            runnable.append(attempt)
        else:
            fork_failures[attempt.attempt_id] = outcome

    semaphore = asyncio.Semaphore(parallelism)

    async def run_one(attempt: _ForkGroupAttempt) -> ForkGroupBranchResult:
        async with semaphore:
            return await _run_branch(
                coordinator,
                record.request,
                record.result.source,
                attempt,
            )

    outcomes = await asyncio.gather(
        *(run_one(attempt) for attempt in runnable),
        return_exceptions=True,
    )
    if any(isinstance(outcome, _ForkGroupContinuationPending) for outcome in outcomes):
        raise _ForkGroupContinuationPending(
            "A fork-group candidate attempt still has an active provider operation."
        )
    by_id = dict(fork_failures)
    for attempt, outcome in zip(runnable, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            if isinstance(outcome, ForkGroupConflict):
                raise outcome
            child = await coordinator.session_store.load(attempt.branch.session_id)
            if child is None:
                fork_failures[attempt.attempt_id] = ForkGroupBranchResult(
                    branch_id=attempt.branch.branch_id,
                    attempt_id=attempt.attempt_id,
                    attempt_request_sha256=_attempt_request_sha256(attempt),
                    attempt_index=attempt.attempt_index,
                    replaced_attempt_id=attempt.replaced_attempt_id,
                    session_id=attempt.branch.session_id,
                    status=ForkGroupBranchStatus.FAILED,
                    failure_code=ForkGroupFailureCode.BRANCH_FAILED,
                    source_checkpoint_sha256=record.result.source.checkpoint_sha256,
                    causal_budget_id=record.request.causal_budget_id,
                    artifact_references=attempt.branch.artifact_references,
                    usage=SessionUsageSummary(session_id=attempt.branch.session_id),
                    error=_exception_message(
                        outcome,
                        redactor=coordinator.secret_redactor,
                    ),
                )
                by_id[attempt.attempt_id] = fork_failures[attempt.attempt_id]
                continue
            _require_branch_authority(
                child,
                request=record.request,
                source=record.result.source,
                attempt=attempt,
            )
            profile = execution_profile_baseline_from_session_metadata(child.metadata)
            by_id[attempt.attempt_id] = ForkGroupBranchResult(
                branch_id=attempt.branch.branch_id,
                attempt_id=attempt.attempt_id,
                attempt_request_sha256=_attempt_request_sha256(attempt),
                attempt_index=attempt.attempt_index,
                replaced_attempt_id=attempt.replaced_attempt_id,
                session_id=attempt.branch.session_id,
                status=ForkGroupBranchStatus.FAILED,
                failure_code=ForkGroupFailureCode.BRANCH_FAILED,
                source_checkpoint_sha256=record.result.source.checkpoint_sha256,
                causal_budget_id=record.request.causal_budget_id,
                execution_profile_fingerprint=(None if profile is None else profile.fingerprint),
                artifact_references=attempt.branch.artifact_references,
                usage=SessionUsageSummary(session_id=attempt.branch.session_id),
                error=_exception_message(
                    outcome,
                    redactor=coordinator.secret_redactor,
                ),
            )
        else:
            by_id[attempt.attempt_id] = outcome
    ordered = tuple(by_id[attempt.attempt_id] for attempt in attempts)
    gated, failure = await _apply_gates(coordinator, record, ordered)
    return gated, failure, frozenset(fork_failures)


def _fatal_viable_attempt_failure(
    branches: tuple[ForkGroupBranchResult, ...],
) -> ForkGroupFailure | None:
    for branch in branches:
        if branch.failure_code is ForkGroupFailureCode.BUDGET_EXHAUSTED:
            return _failure(
                ForkGroupFailureCode.BUDGET_EXHAUSTED,
                branch.error or "Fork-group causal budget was exhausted.",
                branch_id=branch.branch_id,
            )
        if branch.failure_code is ForkGroupFailureCode.SOURCE_CHANGED:
            return _failure(
                ForkGroupFailureCode.SOURCE_CHANGED,
                branch.error or "Fork-group source authority changed.",
                branch_id=branch.branch_id,
            )
    return None


def _supersede_attempt(
    branches: list[ForkGroupBranchResult],
    *,
    replaced_attempt_id: str,
    replacement_attempt_id: str,
) -> None:
    for index, branch in enumerate(branches):
        if branch.attempt_id == replaced_attempt_id:
            branches[index] = branch.model_copy(
                update={
                    "eligible": False,
                    "superseded_by_attempt_id": replacement_attempt_id,
                },
                deep=True,
            )
            return
    raise RuntimeError("Replacement refers to an unknown fork-group attempt.")


async def _run_viable_attempts(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
) -> tuple[
    tuple[ForkGroupBranchResult, ...],
    ForkGroupFailure | None,
    _ForkGroupRecord | None,
]:
    policy = record.request.failure_policy
    minimum = policy.minimum_viable_branches
    if minimum is None:  # pragma: no cover - public model validation owns this
        raise AssertionError("Evaluate-viable policy lost its minimum.")
    initial_attempts = tuple(
        _initial_attempt(record.request, branch) for branch in record.request.branches
    )
    try:
        initial_results, gate_failure, _ = await _run_attempt_batch(
            coordinator,
            record,
            initial_attempts,
            parallelism=record.request.max_parallelism,
        )
    except _ForkGroupContinuationPending:
        raise
    attempt_results = list(initial_results)
    if gate_failure is not None:
        return tuple(attempt_results), gate_failure, None
    if failure := _fatal_viable_attempt_failure(tuple(attempt_results)):
        return tuple(attempt_results), failure, None

    replacement_count = 0
    while sum(attempt.eligible for attempt in attempt_results) < minimum:
        remaining = policy.max_replacement_attempts - replacement_count
        if remaining <= 0:
            break
        latest_by_branch: dict[str, ForkGroupBranchResult] = {}
        for attempt in attempt_results:
            latest_by_branch[attempt.branch_id] = attempt
        candidates = [
            latest_by_branch[branch.branch_id]
            for branch in record.request.branches
            if not latest_by_branch[branch.branch_id].eligible
        ]
        if not candidates:
            break
        deficit = minimum - sum(attempt.eligible for attempt in attempt_results)
        selected = tuple(candidates[: min(deficit, remaining, policy.replacement_parallelism)])
        try:
            planned_attempts = await asyncio.gather(
                *(_plan_replacement(coordinator, record, candidate) for candidate in selected)
            )
        except Exception as exc:
            return (
                tuple(attempt_results),
                _failure(
                    ForkGroupFailureCode.REPLACEMENT_FAILED,
                    _exception_message(
                        exc,
                        redactor=coordinator.secret_redactor,
                        prefix="Fork-group replacement planning failed: ",
                    ),
                ),
                None,
            )
        try:
            replacement_results, gate_failure, uncreated_attempt_ids = await _run_attempt_batch(
                coordinator,
                record,
                tuple(planned_attempts),
                parallelism=policy.replacement_parallelism,
            )
        except _ForkGroupContinuationPending:
            raise
        unresolved_profile_attempts = tuple(
            result
            for result in replacement_results
            if result.attempt_id in uncreated_attempt_ids
            and result.execution_profile_fingerprint is None
        )
        unresolved_profile_ids = {result.attempt_id for result in unresolved_profile_attempts}
        for candidate, replacement, result in zip(
            selected,
            planned_attempts,
            replacement_results,
            strict=True,
        ):
            if result.attempt_id in unresolved_profile_ids:
                continue
            _supersede_attempt(
                attempt_results,
                replaced_attempt_id=candidate.attempt_id,
                replacement_attempt_id=replacement.attempt_id,
            )
            attempt_results.append(result)
        replacement_count += len(planned_attempts)
        if failure := _fatal_viable_attempt_failure(
            tuple(
                result
                for result in replacement_results
                if result.attempt_id not in unresolved_profile_ids
            )
        ):
            return tuple(attempt_results), failure, None
        if unresolved_profile_attempts:
            return (
                tuple(attempt_results),
                _failure(
                    ForkGroupFailureCode.REPLACEMENT_FAILED,
                    "Fork-group replacement execution-profile authority was not accepted.",
                    branch_id=unresolved_profile_attempts[0].branch_id,
                ),
                None,
            )
        if uncreated_attempt_ids:
            failure = _failure(
                ForkGroupFailureCode.REPLACEMENT_FAILED,
                "Fork-group replacement could not create every durable child session.",
            )
            failed = _result_with(
                record,
                state=ForkGroupState.FAILED,
                branches=tuple(attempt_results),
                failure=failure,
            )
            published = await _publish_record(
                coordinator,
                record.result.source.source_session_id,
                failed,
                EventType.FORK_GROUP_FAILED,
                expected_record=record,
            )
            return tuple(attempt_results), failure, published
        if gate_failure is not None:
            return tuple(attempt_results), gate_failure, None

    if sum(attempt.eligible for attempt in attempt_results) < minimum:
        return (
            tuple(attempt_results),
            _failure(
                ForkGroupFailureCode.REPLACEMENTS_EXHAUSTED,
                "Fork-group replacement limits were exhausted before the minimum viable "
                "candidate count was reached.",
            ),
            None,
        )
    return tuple(attempt_results), None, None


async def _execute(coordinator: ForkGroupCoordinator, record: _ForkGroupRecord) -> _ForkGroupRecord:
    if record.result.state is ForkGroupState.CREATED:
        running = _result_with(record, state=ForkGroupState.BRANCHES_RUNNING)
        record = await _publish_record(
            coordinator,
            record.result.source.source_session_id,
            running,
            EventType.FORK_GROUP_BRANCHES_RUNNING,
            expected_record=record,
        )
    if record.result.state is ForkGroupState.BRANCHES_RUNNING:
        if record.request.failure_policy.mode is ForkGroupFailureMode.EVALUATE_VIABLE:
            try:
                branch_tuple, failure, published = await _run_viable_attempts(
                    coordinator,
                    record,
                )
            except _ForkGroupContinuationPending:
                return record
            if published is not None:
                return published
        else:
            initial_attempts = tuple(
                _initial_attempt(record.request, branch) for branch in record.request.branches
            )
            fork_failures: dict[str, ForkGroupBranchResult] = {}
            for attempt in initial_attempts:
                outcome = await _prepare_branch_fork(
                    coordinator,
                    record.request,
                    record.result.source,
                    attempt,
                )
                if outcome is not None:
                    fork_failures[attempt.attempt_id] = outcome
            if fork_failures:
                prepared_results: list[ForkGroupBranchResult] = []
                for attempt in initial_attempts:
                    failed_branch = fork_failures.get(attempt.attempt_id)
                    if failed_branch is not None:
                        prepared_results.append(failed_branch)
                        continue
                    prepared_results.append(
                        ForkGroupBranchResult(
                            branch_id=attempt.branch.branch_id,
                            attempt_id=attempt.attempt_id,
                            attempt_request_sha256=_attempt_request_sha256(attempt),
                            attempt_index=attempt.attempt_index,
                            session_id=attempt.branch.session_id,
                            status=ForkGroupBranchStatus.INTERRUPTED,
                            source_checkpoint_sha256=record.result.source.checkpoint_sha256,
                            causal_budget_id=record.request.causal_budget_id,
                            artifact_references=attempt.branch.artifact_references,
                            usage=await coordinator.get_session_usage(attempt.branch.session_id),
                            error=("Branch was not run because sibling fork preparation failed."),
                        )
                    )
                branch_tuple = tuple(prepared_results)
                failure = _branch_failure(tuple(fork_failures.values())) or _failure(
                    ForkGroupFailureCode.INTERNAL_ERROR,
                    "Fork-group branch preparation failed without a classified error.",
                )
            else:
                semaphore = asyncio.Semaphore(record.request.max_parallelism)

                async def run_one(attempt: _ForkGroupAttempt) -> ForkGroupBranchResult:
                    async with semaphore:
                        return await _run_branch(
                            coordinator,
                            record.request,
                            record.result.source,
                            attempt,
                        )

                outcomes = await asyncio.gather(
                    *(run_one(attempt) for attempt in initial_attempts),
                    return_exceptions=True,
                )
                if any(isinstance(outcome, _ForkGroupContinuationPending) for outcome in outcomes):
                    return record
                branches: list[ForkGroupBranchResult] = []
                for attempt, outcome in zip(initial_attempts, outcomes, strict=True):
                    if isinstance(outcome, BaseException):
                        if isinstance(outcome, ForkGroupConflict):
                            raise outcome
                        branches.append(
                            ForkGroupBranchResult(
                                branch_id=attempt.branch.branch_id,
                                attempt_id=attempt.attempt_id,
                                attempt_request_sha256=_attempt_request_sha256(attempt),
                                attempt_index=attempt.attempt_index,
                                session_id=attempt.branch.session_id,
                                status=ForkGroupBranchStatus.FAILED,
                                source_checkpoint_sha256=(record.result.source.checkpoint_sha256),
                                causal_budget_id=record.request.causal_budget_id,
                                artifact_references=attempt.branch.artifact_references,
                                usage=SessionUsageSummary(session_id=attempt.branch.session_id),
                                error=_exception_message(
                                    outcome,
                                    redactor=coordinator.secret_redactor,
                                ),
                            )
                        )
                    else:
                        branches.append(outcome)
                branch_tuple = tuple(branches)
                failure = _branch_failure(branch_tuple)
                if failure is None:
                    branch_tuple, failure = await _apply_gates(
                        coordinator,
                        record,
                        branch_tuple,
                    )
        if failure is not None:
            failed = _result_with(
                record,
                state=ForkGroupState.FAILED,
                branches=branch_tuple,
                failure=failure,
            )
            return await _publish_record(
                coordinator,
                record.result.source.source_session_id,
                failed,
                EventType.FORK_GROUP_FAILED,
                expected_record=record,
            )
        awaiting = _result_with(
            record,
            state=ForkGroupState.AWAITING_EVALUATION,
            branches=branch_tuple,
            evaluator_session_id=record.request.evaluator.session_id,
        )
        try:
            awaiting = await _bind_tool_free_evaluator_authority(coordinator, awaiting)
        except Exception as exc:
            failed = _result_with(
                record,
                state=ForkGroupState.FAILED,
                branches=branch_tuple,
                failure=_failure(
                    ForkGroupFailureCode.EVALUATOR_FAILED,
                    _exception_message(
                        exc,
                        redactor=coordinator.secret_redactor,
                        prefix="Fork-group evaluator authority failed: ",
                    ),
                ),
            )
            return await _publish_record(
                coordinator,
                record.result.source.source_session_id,
                failed,
                EventType.FORK_GROUP_FAILED,
                expected_record=record,
            )
        record = await _publish_record(
            coordinator,
            record.result.source.source_session_id,
            awaiting,
            EventType.FORK_GROUP_AWAITING_EVALUATION,
            expected_record=record,
        )
    if record.result.state is ForkGroupState.AWAITING_EVALUATION:
        try:
            dispositions, failure = await _run_evaluator(coordinator, record)
        except _ForkGroupContinuationPending:
            return record
        if failure is not None or dispositions is None:
            failed = _result_with(
                record,
                state=ForkGroupState.FAILED,
                branches=record.result.branches,
                evaluator_session_id=record.request.evaluator.session_id,
                failure=failure
                or _failure(ForkGroupFailureCode.INTERNAL_ERROR, "Evaluator returned no result."),
            )
            return await _publish_record(
                coordinator,
                record.result.source.source_session_id,
                failed,
                EventType.FORK_GROUP_FAILED,
                expected_record=record,
            )
        completed = _result_with(
            record,
            state=ForkGroupState.COMPLETED,
            branches=record.result.branches,
            evaluator_session_id=record.request.evaluator.session_id,
            dispositions=dispositions,
        )
        return await _publish_record(
            coordinator,
            record.result.source.source_session_id,
            completed,
            EventType.FORK_GROUP_COMPLETED,
            expected_record=record,
        )
    return record


async def _execute_with_claim(
    coordinator: ForkGroupCoordinator,
    record: _ForkGroupRecord,
    claim_id: str,
) -> _ForkGroupRecord:
    """Execute while a renewable durable owner fences competing applications."""

    source_session_id = record.result.source.source_session_id
    execution = asyncio.create_task(_execute(coordinator, record))
    heartbeat = asyncio.create_task(
        _heartbeat_execution_claim(
            coordinator,
            source_session_id,
            record,
            claim_id,
        )
    )
    try:
        done, _ = await asyncio.wait(
            {execution, heartbeat},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if execution in done:
            result = await execution
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            except _ForkGroupPublicationSuperseded:
                if result.result.state not in {
                    ForkGroupState.COMPLETED,
                    ForkGroupState.FAILED,
                }:
                    raise
            return result
        if heartbeat in done:
            execution.cancel()
            with suppress(asyncio.CancelledError):
                await execution
            await heartbeat
            raise RuntimeError("Fork-group execution claim ended unexpectedly.")
        raise AssertionError("Fork-group execution supervisor woke without a completed task.")
    except BaseException:
        execution.cancel()
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await execution
        with suppress(asyncio.CancelledError):
            await heartbeat
        raise


__all__ = [
    "FORK_GROUP_MAX_BRANCHES",
    "FORK_GROUP_MIN_BRANCHES",
    "ForkGroupArtifactReference",
    "ForkGroupBranchResult",
    "ForkGroupBranchSpec",
    "ForkGroupBranchStatus",
    "ForkGroupCheckpointSelector",
    "ForkGroupConflict",
    "ForkGroupDisposition",
    "ForkGroupDispositionRecord",
    "ForkGroupEvaluatorSpec",
    "ForkGroupFailure",
    "ForkGroupFailureCode",
    "ForkGroupFailureMode",
    "ForkGroupFailurePolicy",
    "ForkGroupGate",
    "ForkGroupGateDecision",
    "ForkGroupGateRequest",
    "ForkGroupGateResult",
    "ForkGroupGateSelection",
    "ForkGroupReplacementPlanner",
    "ForkGroupReplacementPlannerRequest",
    "ForkGroupReplacementPlannerSelection",
    "ForkGroupReplacementSpec",
    "ForkGroupRequest",
    "ForkGroupResult",
    "ForkGroupSourceSnapshot",
    "ForkGroupState",
]
