from __future__ import annotations

import hashlib
import json
import threading
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank
from cayu.workspaces.base import Workspace
from cayu.workspaces.revisions import (
    WorkspaceIdentity,
    WorkspacePathRevision,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimitExceeded,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    copy_bounded_workspace_revision_observation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


WorkspaceBranchRecordTransform = Callable[
    [dict[str, Any] | None],
    dict[str, Any],
]


@runtime_checkable
class WorkspaceBranchStore(Protocol):
    """Narrow durable journal required by local workspace branches."""

    async def load_workspace_branch_record(
        self,
        session_id: str,
        storage_key: str,
    ) -> dict[str, Any] | None:
        """Load one branch-owned durable record."""

    async def publish_workspace_branch_record(
        self,
        session_id: str,
        storage_key: str,
        *,
        record_transform: WorkspaceBranchRecordTransform,
        expected_run_epoch: int,
        commit_guard: Callable[[], None] | None = None,
    ) -> None:
        """Atomically transform one record, optionally beside an owned guarded mutation.

        A supplied guard may perform bounded synchronous filesystem work. The
        implementation must not abandon it on cancellation or run it on an
        asyncio event-loop thread.
        """


class WorkspaceBranchOutcomeStatus(StrEnum):
    CREATED = "created"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    CONFLICTED = "conflicted"
    UNSUPPORTED = "unsupported"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    EXPIRED = "expired"


class WorkspaceBranchLifecycleStatus(StrEnum):
    ACTIVE = "active"
    PUBLISHING = "publishing"
    ROLLING_BACK = "rolling_back"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FENCED = "fenced"


class WorkspaceBranchDurableState(StrEnum):
    """Reconstructible lifecycle state for one durable workspace branch."""

    CREATING = "creating"
    OPEN = "open"
    PUBLICATION_INTENT = "publication_intent"
    PUBLICATION_PROGRESS = "publication_progress"
    COMMITTED = "committed"
    ROLLBACK_INTENT = "rollback_intent"
    ROLLED_BACK = "rolled_back"
    CONFLICTED = "conflicted"
    FAILED = "failed"
    EXPIRED = "expired"
    AMBIGUOUS = "ambiguous"


class WorkspaceBranchAuthority(BaseModel):
    """Durable creation and fencing authority supplied by the owning runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str = Field(max_length=512)
    expected_run_epoch: StrictInt = Field(ge=0)
    environment_name: str = Field(max_length=256)
    binding_generation: str = Field(max_length=512)
    binding_identity: str = Field(max_length=512)
    creating_authority: str = Field(max_length=512)
    resource_policy: str = Field(max_length=512)

    @field_validator(
        "session_id",
        "environment_name",
        "binding_generation",
        "binding_identity",
        "creating_authority",
        "resource_policy",
    )
    @classmethod
    def validate_authority(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class WorkspaceBranchBindingAuthority(BaseModel):
    """The binding identity that is live for a local workspace right now."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    environment_name: str = Field(max_length=256)
    binding_generation: str = Field(max_length=512)
    binding_identity: str = Field(max_length=512)

    @field_validator("environment_name", "binding_generation", "binding_identity")
    @classmethod
    def validate_authority(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


@runtime_checkable
class WorkspaceBranchBindingAuthorityClaim(Protocol):
    """One transferable claim on an exact live workspace binding generation.

    Cayu may retain the claim after its initiating task returns and release it
    from a later settlement task. A failed release remains owned and may be
    retried.
    """

    def release(self) -> None:
        """Release the claim, or raise without abandoning its ownership."""


@runtime_checkable
class WorkspaceBranchBindingAuthorityProvider(Protocol):
    """Own one live binding generation and its in-flight branch claims.

    A successful ``claim()`` is the positive authority evidence for the whole
    guarded operation. That operation does not call the provider again,
    acquire a nested claim, or invoke it from an off-thread commit guard before
    the claim is released.
    """

    def __call__(self) -> WorkspaceBranchBindingAuthority:
        """Return the current binding authority."""

    def claim(
        self,
        expected: WorkspaceBranchBindingAuthority,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        """Keep ``expected`` current until the guarded operation settles.

        The claim must be transferable across task contexts. Its ``release``
        method may be called by a later settlement task and must retain
        ownership when it raises so Cayu can retry it.
        """


class WorkspaceBranchLimits(BaseModel):
    """Hard limits for one local workspace branch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    lifetime_ms: StrictInt = Field(default=15 * 60 * 1000, ge=1, le=24 * 60 * 60 * 1000)
    max_paths: StrictInt = Field(default=4096, ge=1, le=100_000)
    max_files: StrictInt = Field(default=4096, ge=1, le=100_000)
    max_path_bytes: StrictInt = Field(default=4096, ge=1, le=65_536)
    max_file_bytes: StrictInt = Field(default=8 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    max_baseline_bytes: StrictInt = Field(
        default=64 * 1024 * 1024,
        ge=1,
        le=1024 * 1024 * 1024,
    )
    max_overlay_bytes: StrictInt = Field(
        default=64 * 1024 * 1024,
        ge=1,
        le=1024 * 1024 * 1024,
    )
    max_changed_paths: StrictInt = Field(default=4096, ge=1, le=100_000)
    max_publication_attempts: StrictInt = Field(default=64, ge=1, le=4096)
    max_evidence_bytes: StrictInt = Field(default=1024 * 1024, ge=1024, le=16 * 1024 * 1024)
    max_active_branches: StrictInt = Field(default=16, ge=1, le=256)


class WorkspaceBranchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    baseline: WorkspaceRevisionObservation
    limits: WorkspaceBranchLimits = Field(default_factory=WorkspaceBranchLimits)
    branch_id: str | None = Field(default=None, max_length=256)
    idempotency_key: str | None = Field(default=None, max_length=512)
    authority: WorkspaceBranchAuthority | None = None

    @field_validator("branch_id", "idempotency_key")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_baseline(self) -> WorkspaceBranchRequest:
        if self.baseline.status is not WorkspaceRevisionObservationStatus.SUPPORTED:
            raise ValueError("Workspace branch baseline must be a supported observation.")
        if self.baseline.path_scope != "complete":
            raise ValueError("Workspace branch baseline must contain a complete path inventory.")
        if self.baseline.revision is None:
            raise ValueError("Workspace branch baseline must define a revision.")
        durable_fields = (self.branch_id, self.idempotency_key, self.authority)
        if any(value is not None for value in durable_fields) and any(
            value is None for value in durable_fields
        ):
            raise ValueError(
                "Durable workspace branches require branch_id, idempotency_key, and authority."
            )
        return self


class WorkspaceBranchContentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    sha256: str
    bytes: StrictInt = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("Workspace branch content identity requires lowercase SHA-256.")
        return value


WorkspaceBranchChangeOperation = Literal["created", "modified", "deleted"]


class WorkspaceBranchChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str
    operation: WorkspaceBranchChangeOperation
    before: WorkspaceBranchContentIdentity | None = None
    after: WorkspaceBranchContentIdentity | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        from cayu.workspaces.base import _validate_workspace_relative_path

        path = _validate_workspace_relative_path(value)
        if path != value:
            raise ValueError("Workspace branch change paths must be canonical POSIX paths.")
        return path

    @model_validator(mode="after")
    def validate_change(self) -> WorkspaceBranchChange:
        if self.operation == "created" and (self.before is not None or self.after is None):
            raise ValueError("A created branch path requires only an after identity.")
        if self.operation == "modified" and (self.before is None or self.after is None):
            raise ValueError("A modified branch path requires before and after identities.")
        if self.operation == "deleted" and (self.before is None or self.after is not None):
            raise ValueError("A deleted branch path requires only a before identity.")
        return self


class WorkspaceBranchChangeSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str
    source: WorkspaceIdentity
    baseline_revision: str
    changes: tuple[WorkspaceBranchChange, ...]
    digest: str

    @field_validator("branch_id", "baseline_revision")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not value.startswith("sha256:") or len(value) != 71:
            raise ValueError("Workspace branch change-set digest is invalid.")
        if any(character not in "0123456789abcdef" for character in value[7:]):
            raise ValueError("Workspace branch change-set digest is invalid.")
        return value

    @model_validator(mode="after")
    def validate_order_and_digest(self) -> WorkspaceBranchChangeSet:
        if tuple(sorted(change.path for change in self.changes)) != tuple(
            change.path for change in self.changes
        ):
            raise ValueError("Workspace branch changes must be path ordered.")
        if len({change.path for change in self.changes}) != len(self.changes):
            raise ValueError("Workspace branch changes must have unique paths.")
        expected = workspace_branch_change_set_digest(
            branch_id=self.branch_id,
            source=self.source,
            baseline_revision=self.baseline_revision,
            changes=self.changes,
        )
        if self.digest != expected:
            raise ValueError("Workspace branch change-set digest does not match its contents.")
        return self


class WorkspaceBranchConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    path: str
    expected: WorkspaceBranchContentIdentity | None = None
    actual: WorkspaceBranchContentIdentity | None = None
    actual_kind: Literal["missing", "file", "directory", "symlink", "special"]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        from cayu.workspaces.base import _validate_workspace_relative_path

        return _validate_workspace_relative_path(value)


class WorkspaceBranchEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str | None = None
    source: WorkspaceIdentity
    baseline_revision: str | None = None
    outcome: WorkspaceBranchOutcomeStatus
    change_set_digest: str | None = None
    affected_path_sha256: tuple[str, ...] = ()
    detail_code: str | None = None

    @field_validator("branch_id", "baseline_revision", "change_set_digest", "detail_code")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("affected_path_sha256")
    @classmethod
    def validate_path_digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if type(value) is not tuple:
            raise TypeError("Workspace branch affected path identities must be a tuple.")
        for digest in value:
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Workspace branch affected path identity is invalid.")
        if tuple(sorted(value)) != value:
            raise ValueError("Workspace branch affected path identities must be ordered.")
        return value


class WorkspaceBranchPublicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str
    baseline_revision: str
    change_set_digest: str
    idempotency_key: str | None = Field(default=None, max_length=512)
    expected_run_epoch: StrictInt | None = Field(default=None, ge=0)
    binding_generation: str | None = Field(default=None, max_length=512)

    @field_validator(
        "branch_id",
        "baseline_revision",
        "change_set_digest",
        "idempotency_key",
        "binding_generation",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_durable_authority(self) -> WorkspaceBranchPublicationRequest:
        durable_fields = (
            self.idempotency_key,
            self.expected_run_epoch,
            self.binding_generation,
        )
        if any(value is not None for value in durable_fields) and any(
            value is None for value in durable_fields
        ):
            raise ValueError(
                "Durable publication requires idempotency_key, expected_run_epoch, and "
                "binding_generation."
            )
        return self


class WorkspaceBranchRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str = Field(max_length=256)
    idempotency_key: str = Field(max_length=512)
    expected_run_epoch: StrictInt = Field(ge=0)
    binding_generation: str = Field(max_length=512)
    reason: Literal["explicit", "expired"] = "explicit"

    @field_validator("branch_id", "idempotency_key", "binding_generation")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class WorkspaceBranchRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    branch_id: str = Field(max_length=256)
    session_id: str = Field(max_length=512)
    expected_run_epoch: StrictInt = Field(ge=0)
    binding_generation: str = Field(max_length=512)
    binding_identity: str = Field(max_length=512)
    recovery_id: str = Field(max_length=512)

    @field_validator(
        "branch_id",
        "session_id",
        "binding_generation",
        "binding_identity",
        "recovery_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


@dataclass(frozen=True)
class WorkspaceBranchCreationResult:
    status: WorkspaceBranchOutcomeStatus
    branch: WorkspaceBranch | None
    evidence: WorkspaceBranchEvidence
    conflicts: tuple[WorkspaceBranchConflict, ...] = ()

    def __post_init__(self) -> None:
        if type(self.evidence) is not WorkspaceBranchEvidence:
            raise TypeError("Workspace branch creation evidence is invalid.")
        if self.status not in {
            WorkspaceBranchOutcomeStatus.CREATED,
            WorkspaceBranchOutcomeStatus.CONFLICTED,
            WorkspaceBranchOutcomeStatus.UNSUPPORTED,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            WorkspaceBranchOutcomeStatus.FAILED,
        }:
            raise ValueError("Workspace branch creation status is invalid.")
        if self.status is WorkspaceBranchOutcomeStatus.CREATED:
            if not isinstance(self.branch, WorkspaceBranch):
                raise ValueError("A created workspace branch result requires a branch.")
            if self.evidence.branch_id is None or self.evidence.baseline_revision is None:
                raise ValueError("A created workspace branch requires complete identity evidence.")
        elif self.branch is not None:
            raise ValueError("An unsuccessful workspace branch result cannot expose a branch.")
        _validate_result_evidence(self.status, self.evidence, self.conflicts)


@dataclass(frozen=True)
class WorkspaceBranchPublicationResult:
    status: WorkspaceBranchOutcomeStatus
    evidence: WorkspaceBranchEvidence
    conflicts: tuple[WorkspaceBranchConflict, ...] = ()

    def __post_init__(self) -> None:
        if type(self.evidence) is not WorkspaceBranchEvidence:
            raise TypeError("Workspace branch publication evidence is invalid.")
        if self.status not in {
            WorkspaceBranchOutcomeStatus.COMMITTED,
            WorkspaceBranchOutcomeStatus.CONFLICTED,
            WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            WorkspaceBranchOutcomeStatus.FAILED,
            WorkspaceBranchOutcomeStatus.AMBIGUOUS,
        }:
            raise ValueError("Workspace branch publication status is invalid.")
        if (
            self.evidence.branch_id is None
            or self.evidence.baseline_revision is None
            or self.evidence.change_set_digest is None
        ):
            raise ValueError("Workspace branch publication requires complete authority evidence.")
        _validate_result_evidence(self.status, self.evidence, self.conflicts)


@dataclass(frozen=True)
class WorkspaceBranchRollbackResult:
    status: WorkspaceBranchOutcomeStatus
    evidence: WorkspaceBranchEvidence

    def __post_init__(self) -> None:
        if type(self.evidence) is not WorkspaceBranchEvidence:
            raise TypeError("Workspace branch rollback evidence is invalid.")
        if self.status not in {
            WorkspaceBranchOutcomeStatus.ROLLED_BACK,
            WorkspaceBranchOutcomeStatus.EXPIRED,
        }:
            raise ValueError("Workspace branch rollback result has an invalid terminal status.")
        if self.evidence.branch_id is None or self.evidence.baseline_revision is None:
            raise ValueError("Workspace branch rollback requires complete identity evidence.")
        _validate_result_evidence(self.status, self.evidence, ())


@dataclass(frozen=True)
class WorkspaceBranchRecoveryResult:
    state: WorkspaceBranchDurableState
    evidence: WorkspaceBranchEvidence
    branch: WorkspaceBranch | None = None
    publication: WorkspaceBranchPublicationResult | None = None
    rollback: WorkspaceBranchRollbackResult | None = None

    def __post_init__(self) -> None:
        open_state = self.state in {
            WorkspaceBranchDurableState.OPEN,
            WorkspaceBranchDurableState.CONFLICTED,
        }
        if open_state != isinstance(self.branch, WorkspaceBranch):
            raise ValueError("Only recoverable open workspace branches expose a branch.")
        if self.state is WorkspaceBranchDurableState.COMMITTED:
            if self.publication is None or self.rollback is not None:
                raise ValueError("Committed recovery requires only publication evidence.")
        elif self.state in {
            WorkspaceBranchDurableState.ROLLED_BACK,
            WorkspaceBranchDurableState.EXPIRED,
        }:
            if self.rollback is None or self.publication is not None:
                raise ValueError("Rollback recovery requires only rollback evidence.")
        elif self.publication is not None or self.rollback is not None:
            raise ValueError("Nonterminal recovery cannot expose a terminal result.")


class WorkspaceBranchClosedError(RuntimeError):
    """A terminal or expired workspace branch cannot accept this operation."""


class WorkspaceBranchFencedError(RuntimeError):
    """Workspace reuse is unsafe after publication rollback failed."""


class WorkspaceBranchOperationConflict(ValueError):
    """A stable branch operation identity was reused with different authority."""


def _copy_workspace_branch_binding_authority(
    value: WorkspaceBranchBindingAuthority,
) -> WorkspaceBranchBindingAuthority:
    if type(value) is not WorkspaceBranchBindingAuthority:
        raise TypeError("Workspace branch binding authority must use the exact public type.")
    return WorkspaceBranchBindingAuthority(
        environment_name=value.environment_name,
        binding_generation=value.binding_generation,
        binding_identity=value.binding_identity,
    )


class _WorkspaceBranchBindingAuthorityRegistryClaim:
    def __init__(self, registry: WorkspaceBranchBindingAuthorityRegistry) -> None:
        self._registry = registry
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._registry._release_claim()
            self._released = True


class WorkspaceBranchBindingAuthorityRegistry:
    """Process-local owner for one replaceable workspace binding generation."""

    def __init__(self, authority: WorkspaceBranchBindingAuthority) -> None:
        self._lock = threading.Lock()
        self._authority = _copy_workspace_branch_binding_authority(authority)
        self._active_claims = 0

    def __call__(self) -> WorkspaceBranchBindingAuthority:
        with self._lock:
            return _copy_workspace_branch_binding_authority(self._authority)

    def claim(
        self,
        expected: WorkspaceBranchBindingAuthority,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        expected = _copy_workspace_branch_binding_authority(expected)
        with self._lock:
            if self._authority != expected:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch creation binding authority is no longer current."
                )
            self._active_claims += 1
        return _WorkspaceBranchBindingAuthorityRegistryClaim(self)

    def _release_claim(self) -> None:
        with self._lock:
            if self._active_claims <= 0:
                raise RuntimeError("Workspace branch binding claim is not active.")
            self._active_claims -= 1

    def replace(self, authority: WorkspaceBranchBindingAuthority) -> None:
        replacement = _copy_workspace_branch_binding_authority(authority)
        with self._lock:
            if replacement == self._authority:
                return
            if self._active_claims:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch binding replacement conflicts with an active generation "
                    "claim."
                )
            self._authority = replacement


class WorkspaceBranchResourceExhaustedError(RuntimeError):
    def __init__(self, detail_code: str) -> None:
        self.detail_code = require_durable_clean_nonblank(detail_code, "detail_code")
        super().__init__(f"Workspace branch resource limit reached: {self.detail_code}.")


class WorkspaceBranchPublicationError(RuntimeError):
    """Publication failed after mutation began; source rollback was attempted."""


class WorkspaceBranch(Workspace):
    """An isolated workspace view with explicit publication."""

    @property
    @abstractmethod
    def branch_id(self) -> str:
        """Stable identity for this branch."""

    @property
    @abstractmethod
    def lifecycle_status(self) -> WorkspaceBranchLifecycleStatus:
        """Current attached lifecycle state."""

    @abstractmethod
    async def changes(self) -> WorkspaceBranchChangeSet:
        """Return the exact deterministic net change set."""

    @abstractmethod
    async def publish(
        self,
        request: WorkspaceBranchPublicationRequest,
    ) -> WorkspaceBranchPublicationResult:
        """Publish the exact requested change set or leave the source unchanged."""

    @abstractmethod
    async def rollback(
        self,
        request: WorkspaceBranchRollbackRequest | None = None,
    ) -> WorkspaceBranchRollbackResult:
        """Discard the complete private branch without modifying its source."""


@dataclass(frozen=True, slots=True)
class _WorkspaceBranchRequestEnvelope:
    source: WorkspaceIdentity
    baseline_revision: str
    limits: WorkspaceBranchLimits
    branch_id: str | None
    idempotency_key: str | None
    authority: WorkspaceBranchAuthority | None


def _copy_workspace_branch_request_envelope(
    request: object,
) -> _WorkspaceBranchRequestEnvelope:
    """Detach the bounded authority needed before copying baseline path evidence."""

    if type(request) is not WorkspaceBranchRequest:
        raise TypeError("Workspace branch request must be WorkspaceBranchRequest.")
    if type(request.limits) is not WorkspaceBranchLimits:
        raise TypeError("Workspace branch request limits are invalid.")
    if type(request.baseline) is not WorkspaceRevisionObservation:
        raise TypeError("Workspace branch baseline observation is invalid.")
    if type(request.baseline.identity) is not WorkspaceIdentity:
        raise TypeError("Workspace branch baseline identity is invalid.")
    if (
        type(request.baseline.identity.workspace_id) is not str
        or type(request.baseline.identity.observer) is not str
    ):
        raise TypeError("Workspace branch baseline identity fields are invalid.")
    if request.baseline.status is not WorkspaceRevisionObservationStatus.SUPPORTED:
        raise ValueError("Workspace branch baseline must be a supported observation.")
    if request.baseline.path_scope != "complete":
        raise ValueError("Workspace branch baseline must contain a complete path inventory.")
    if type(request.baseline.revision) is not str:
        raise ValueError("Workspace branch baseline must define a revision.")
    return _WorkspaceBranchRequestEnvelope(
        source=WorkspaceIdentity(
            workspace_id=request.baseline.identity.workspace_id,
            observer=request.baseline.identity.observer,
        ),
        baseline_revision=request.baseline.revision,
        limits=WorkspaceBranchLimits(
            **{name: getattr(request.limits, name) for name in WorkspaceBranchLimits.model_fields}
        ),
        branch_id=request.branch_id,
        idempotency_key=request.idempotency_key,
        authority=(
            None
            if request.authority is None
            else WorkspaceBranchAuthority.model_validate(
                request.authority.model_dump(mode="python", warnings=False)
            )
        ),
    )


def copy_workspace_branch_request(request: object) -> WorkspaceBranchRequest:
    envelope = _copy_workspace_branch_request_envelope(request)
    baseline_input = cast("WorkspaceBranchRequest", request).baseline
    if type(baseline_input.paths) is not tuple:
        raise TypeError("Workspace observation paths must be a tuple.")
    if len(baseline_input.paths) > envelope.limits.max_files:
        raise WorkspaceBranchResourceExhaustedError("file_count_limit_exceeded")
    if len(baseline_input.paths) > envelope.limits.max_paths:
        raise WorkspaceBranchResourceExhaustedError("path_count_limit_exceeded")
    for entry in baseline_input.paths:
        if type(entry) is not WorkspacePathRevision:
            raise TypeError("Workspace branch baseline path evidence is invalid.")
        if type(entry.path) is not str:
            raise TypeError("Workspace branch baseline paths must be strings.")
        if len(entry.path.encode("utf-8")) > envelope.limits.max_path_bytes:
            raise WorkspaceBranchResourceExhaustedError("path_byte_limit_exceeded")
    try:
        baseline = copy_bounded_workspace_revision_observation(
            baseline_input,
            expected_identity=envelope.source,
            limits=WorkspaceRevisionObservationLimits(
                max_paths=100_000,
                max_path_bytes=65_536,
                max_file_bytes=64 * 1024 * 1024,
                max_total_file_bytes=1024 * 1024 * 1024,
                max_manifest_bytes=16 * 1024 * 1024,
            ),
        )
    except WorkspaceRevisionObservationLimitExceeded as exc:
        raise WorkspaceBranchResourceExhaustedError("baseline_observation_limit_exceeded") from exc
    if baseline.path_scope != "complete":
        raise ValueError("Workspace branch baseline must contain a complete path inventory.")
    for entry in baseline.paths:
        if type(entry) is not WorkspacePathRevision:
            raise TypeError("Workspace branch baseline path evidence is invalid.")
        if entry.kind != "file" or entry.present is not True or entry.content_sha256 is None:
            raise ValueError(
                "Workspace branch baseline must contain only present regular-file evidence."
            )
    return WorkspaceBranchRequest(
        baseline=baseline,
        limits=envelope.limits,
        branch_id=envelope.branch_id,
        idempotency_key=envelope.idempotency_key,
        authority=envelope.authority,
    )


def workspace_branch_change_set_digest(
    *,
    branch_id: str,
    source: WorkspaceIdentity,
    baseline_revision: str,
    changes: Sequence[WorkspaceBranchChange],
) -> str:
    payload = {
        "baseline_revision": baseline_revision,
        "branch_id": branch_id,
        "changes": [
            {
                "after": (
                    None
                    if change.after is None
                    else {"bytes": change.after.bytes, "sha256": change.after.sha256}
                ),
                "before": (
                    None
                    if change.before is None
                    else {"bytes": change.before.bytes, "sha256": change.before.sha256}
                ),
                "operation": change.operation,
                "path": change.path,
            }
            for change in changes
        ],
        "source": {
            "observer": source.observer,
            "workspace_id": source.workspace_id,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def workspace_branch_evidence(
    *,
    source: WorkspaceIdentity,
    outcome: WorkspaceBranchOutcomeStatus,
    baseline_revision: str | None,
    branch_id: str | None = None,
    change_set_digest: str | None = None,
    paths: Sequence[str] = (),
    detail_code: str | None = None,
) -> WorkspaceBranchEvidence:
    return WorkspaceBranchEvidence(
        branch_id=branch_id,
        source=WorkspaceIdentity(
            workspace_id=source.workspace_id,
            observer=source.observer,
        ),
        baseline_revision=baseline_revision,
        outcome=outcome,
        change_set_digest=change_set_digest,
        affected_path_sha256=tuple(
            sorted(hashlib.sha256(path.encode("utf-8")).hexdigest() for path in paths)
        ),
        detail_code=detail_code,
    )


def _bounded_workspace_branch_evidence(
    *,
    source: WorkspaceIdentity,
    outcome: WorkspaceBranchOutcomeStatus,
    baseline_revision: str | None,
    max_bytes: int,
    branch_id: str | None = None,
    change_set_digest: str | None = None,
    paths: Sequence[str] = (),
    detail_code: str | None = None,
    hash_fixed_identity_on_overflow: bool = False,
) -> WorkspaceBranchEvidence:
    bounded_source = source
    bounded_baseline_revision = baseline_revision
    bounded_branch_id = branch_id
    bounded_change_set_digest = change_set_digest
    projected_size = _workspace_branch_evidence_json_size(
        source=bounded_source,
        outcome=outcome,
        baseline_revision=bounded_baseline_revision,
        branch_id=bounded_branch_id,
        change_set_digest=bounded_change_set_digest,
        affected_path_count=len(paths),
        detail_code=detail_code,
    )
    if projected_size > max_bytes:
        if not hash_fixed_identity_on_overflow:
            raise WorkspaceBranchResourceExhaustedError("result_evidence_limit_exceeded")
        # Choose the bounded representation before constructing a Pydantic
        # model or a complete JSON document containing caller-controlled
        # authority. This makes ``max_evidence_bytes`` an allocation boundary,
        # rather than only a limit on the value eventually returned.
        bounded_source = WorkspaceIdentity(
            workspace_id=_stable_evidence_token(source.workspace_id),
            observer=_stable_evidence_token(source.observer),
        )
        bounded_baseline_revision = (
            None if baseline_revision is None else _stable_evidence_token(baseline_revision)
        )
        bounded_branch_id = None if branch_id is None else _stable_evidence_token(branch_id)
        bounded_change_set_digest = (
            None if change_set_digest is None else _stable_evidence_token(change_set_digest)
        )
        projected_size = _workspace_branch_evidence_json_size(
            source=bounded_source,
            outcome=outcome,
            baseline_revision=bounded_baseline_revision,
            branch_id=bounded_branch_id,
            change_set_digest=bounded_change_set_digest,
            affected_path_count=len(paths),
            detail_code=detail_code,
        )
    if projected_size > max_bytes:
        raise WorkspaceBranchResourceExhaustedError("result_evidence_limit_exceeded")
    evidence = workspace_branch_evidence(
        source=bounded_source,
        outcome=outcome,
        baseline_revision=bounded_baseline_revision,
        branch_id=bounded_branch_id,
        change_set_digest=bounded_change_set_digest,
        paths=paths,
        detail_code=detail_code,
    )
    # Keep the serializer as a final compatibility assertion. The incremental
    # projection above ensures this allocation is already bounded even if a
    # future serializer changes its escaping rules.
    if len(evidence.model_dump_json().encode("utf-8")) > max_bytes:
        raise WorkspaceBranchResourceExhaustedError("result_evidence_limit_exceeded")
    return evidence


def _stable_evidence_token(value: str) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(value), 4096):
        digest.update(value[start : start + 4096].encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _workspace_branch_evidence_json_size(
    *,
    source: WorkspaceIdentity,
    outcome: WorkspaceBranchOutcomeStatus,
    baseline_revision: str | None,
    branch_id: str | None,
    change_set_digest: str | None,
    affected_path_count: int,
    detail_code: str | None,
) -> int:
    """Project Pydantic's compact JSON size without materializing the document."""

    affected_paths_size = 2
    if affected_path_count:
        # Each path is represented by one quoted lowercase SHA-256 plus commas.
        affected_paths_size += affected_path_count * 66 + affected_path_count - 1
    return sum(
        (
            len(b'{"branch_id":'),
            _optional_json_text_size(branch_id),
            len(b',"source":{"workspace_id":'),
            _json_text_size(source.workspace_id),
            len(b',"observer":'),
            _json_text_size(source.observer),
            len(b'},"baseline_revision":'),
            _optional_json_text_size(baseline_revision),
            len(b',"outcome":'),
            _json_text_size(outcome.value),
            len(b',"change_set_digest":'),
            _optional_json_text_size(change_set_digest),
            len(b',"affected_path_sha256":'),
            affected_paths_size,
            len(b',"detail_code":'),
            _optional_json_text_size(detail_code),
            1,
        )
    )


def _workspace_branch_empty_change_set_json_size(
    *,
    branch_id: str,
    source: WorkspaceIdentity,
    baseline_revision: str,
) -> int:
    """Project the empty change-set JSON size without copying authority text."""

    return sum(
        (
            len(b'{"branch_id":'),
            _json_text_size(branch_id),
            len(b',"source":{"workspace_id":'),
            _json_text_size(source.workspace_id),
            len(b',"observer":'),
            _json_text_size(source.observer),
            len(b'},"baseline_revision":'),
            _json_text_size(baseline_revision),
            len(b',"changes":[],"digest":'),
            _json_text_size("sha256:" + "0" * 64),
            1,
        )
    )


def _optional_json_text_size(value: str | None) -> int:
    return 4 if value is None else _json_text_size(value)


def _json_text_size(value: str) -> int:
    """Return compact UTF-8 JSON string size without allocating encoded input."""

    if type(value) is not str:
        raise TypeError("Workspace branch evidence text must be a string.")
    size = 2
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in {"\b", "\t", "\n", "\f", "\r"}:
            size += 2
        elif codepoint < 0x20:
            size += 6
        elif codepoint < 0x80:
            size += 1
        elif codepoint < 0x800:
            size += 2
        elif codepoint < 0x10000:
            size += 3
        else:
            size += 4
    return size


def _validate_result_evidence(
    status: WorkspaceBranchOutcomeStatus,
    evidence: WorkspaceBranchEvidence,
    conflicts: tuple[WorkspaceBranchConflict, ...],
) -> None:
    if type(evidence) is not WorkspaceBranchEvidence or evidence.outcome is not status:
        raise ValueError("Workspace branch result evidence must match its status.")
    if type(conflicts) is not tuple:
        raise TypeError("Workspace branch conflicts must be a tuple.")
    if any(type(conflict) is not WorkspaceBranchConflict for conflict in conflicts):
        raise TypeError("Workspace branch conflicts contain an invalid entry.")
    conflict_paths = tuple(conflict.path for conflict in conflicts)
    if conflict_paths != tuple(sorted(conflict_paths)) or len(set(conflict_paths)) != len(
        conflict_paths
    ):
        raise ValueError("Workspace branch conflicts must be uniquely path ordered.")
    if status is WorkspaceBranchOutcomeStatus.CONFLICTED and not conflicts:
        raise ValueError("A conflicted workspace branch result requires conflict evidence.")
    if status is not WorkspaceBranchOutcomeStatus.CONFLICTED and conflicts:
        raise ValueError("Only a conflicted workspace branch result can contain conflicts.")
    if conflicts and evidence.affected_path_sha256 != tuple(
        sorted(hashlib.sha256(path.encode("utf-8")).hexdigest() for path in conflict_paths)
    ):
        raise ValueError("Workspace branch conflict paths must match result evidence.")
