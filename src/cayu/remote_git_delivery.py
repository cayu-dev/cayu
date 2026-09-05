"""Approved host-side Git delivery for patch-ready coding products."""

from __future__ import annotations

import asyncio
import os
import re
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum
from hashlib import new as hashlib_new
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._remote_git_configuration import executable_digest, require_inert_repository_config
from cayu._remote_git_ownership import (
    RemoteGitOwnershipUnavailable,
    delivery_write_pending,
    exclusive_delivery,
)
from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.artifacts import (
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    copy_artifact_read_result,
)
from cayu.coding_products import (
    CODING_PRODUCT_EVIDENCE_KIND,
    CodingArtifactReference,
    CodingCheckEvidence,
    CodingProductArtifactRepository,
    CodingProductPublication,
    CodingProductState,
    coding_product_completion_decision,
)
from cayu.runners import ExecCommand, ExecResult, LocalRunner
from cayu.runtime.work_contracts import CompletionVerdict, WorkEvidenceReference
from cayu.vaults import SecretEnv, SecretRef, SecretResolver, validate_secret_resolver
from cayu.workspaces import Workspace, WorkspaceReadResult, WorkspaceRevisionObservation
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    observe_deterministic_workspace,
)

REMOTE_GIT_DELIVERY_SCHEMA_VERSION = "cayu.remote_git_delivery.v1"
REMOTE_GIT_DELIVERY_RESULT_KIND = "remote_git_delivery_result"
REMOTE_GIT_MAX_RESULT_BYTES = 1024 * 1024
REMOTE_GIT_MAX_LIFECYCLE_RECEIPTS = 48

_SHA256_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_REF_RE = re.compile(r"refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]{0,240}\Z")
_EMAIL_RE = re.compile(r"[^\s<>@]+@[^\s<>@]+\Z")
_PROTECTED_SOURCE_ROOTS = frozenset({".git", ".cayu", ".runtime"})
_REMOTE_GIT_ENV_REMOVE = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_EXTERNAL_DIFF",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_WORK_TREE",
    "SSH_AGENT_PID",
    "SSH_AUTH_SOCK",
)


class RemoteGitDeliveryError(RuntimeError):
    """Base error for the optional remote Git delivery product."""


class RemoteGitDeliveryAdmissionError(RemoteGitDeliveryError):
    """Delivery authority or its patch-ready source failed admission."""


class RemoteGitDeliveryConflictError(RemoteGitDeliveryAdmissionError):
    """The observed remote state conflicts with immutable delivery intent."""

    def __init__(
        self,
        reason_code: str,
        *,
        observed_base: str | None,
        observed_destination: str | None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = _identifier(reason_code, "reason_code")
        self.observed_base = observed_base
        self.observed_destination = observed_destination


class RemoteGitDeliveryReconstructionRequiredError(RemoteGitDeliveryError):
    """Durable evidence must be reconciled before delivery can continue."""


def _identifier(value: str, field_name: str, *, maximum: int = 512) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} bytes.")
    return value


def _fingerprint(value: str, field_name: str) -> str:
    value = _identifier(value, field_name, maximum=80)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a SHA-256 fingerprint.")
    return "sha256:" + value.removeprefix("sha256:")


def _object_id(value: str, field_name: str) -> str:
    value = _identifier(value, field_name, maximum=64)
    if _OBJECT_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a full Git object ID.")
    return value


def _git_ref(value: str, field_name: str) -> str:
    value = _identifier(value, field_name, maximum=255)
    components = value.split("/")
    if (
        _REF_RE.fullmatch(value) is None
        or ".." in value
        or "@{" in value
        or "//" in value
        or value.endswith((".", "/", ".lock"))
        or any(component.startswith(".") or component.endswith(".lock") for component in components)
    ):
        raise ValueError(f"{field_name} must be one canonical branch ref.")
    return value


def _canonical_model_bytes(value: BaseModel, field_name: str) -> bytes:
    return canonical_durable_json_bytes(
        value.model_dump(mode="json", warnings=False),
        field_name,
    )


def _model_fingerprint(value: BaseModel, field_name: str) -> str:
    return "sha256:" + sha256(_canonical_model_bytes(value, field_name)).hexdigest()


def _artifact_id(*parts: str) -> str:
    digest = sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return "art_" + digest[:32]


def _coding_check_evidence_sha256(checks: Sequence[CodingCheckEvidence]) -> str:
    material: list[dict[str, str | None]] = []
    for check in checks:
        material.append(
            {
                "check": check.check,
                "profile": check.profile_fingerprint,
                "event_id": check.event_id,
                "output_sha256": check.output_sha256,
                "artifact_id": (
                    None if check.output_artifact is None else check.output_artifact.artifact_id
                ),
            }
        )
    return (
        "sha256:"
        + sha256(canonical_durable_json_bytes(material, "remote_git_check_evidence")).hexdigest()
    )


def _secure_owned_directory(path: Path, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RemoteGitDeliveryAdmissionError(f"Remote Git {label} is unsafe.") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or (
            hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()
        ):
            raise RemoteGitDeliveryAdmissionError(f"Remote Git {label} is unsafe.")
        os.fchmod(descriptor, stat.S_IRWXU)
    finally:
        os.close(descriptor)


class _FrozenDeliveryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class RemoteGitDeliveryState(StrEnum):
    PREPARING = "preparing"
    PREPARED = "prepared"
    APPROVAL_REQUIRED = "approval_required"
    COMMITTING = "committing"
    COMMITTED_LOCALLY = "committed_locally"
    PUSHING = "pushing"
    PUSHED = "pushed"
    CONFLICT = "conflict"
    DENIED = "denied"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    RECONSTRUCTION_REQUIRED = "reconstruction_required"


class _RemoteGitConfigurationAuthority(_FrozenDeliveryModel):
    request_fingerprint: str
    configuration_sha256: str


class _BaseObservationDefault(Enum):
    PREPARED = "prepared"


class RemoteGitSourceAuthority(_FrozenDeliveryModel):
    product_result_artifact_id: str
    product_result_sha256: str
    product_request_fingerprint: str
    product_run_id: str
    source_workspace_id: str
    final_source_revision: str
    diff_artifact_id: str
    diff_sha256: str
    check_evidence_sha256: str

    @field_validator(
        "product_result_artifact_id",
        "product_run_id",
        "source_workspace_id",
        "diff_artifact_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "product_result_sha256",
        "product_request_fingerprint",
        "final_source_revision",
        "diff_sha256",
        "check_evidence_sha256",
    )
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)


class RemoteGitRepositoryAuthority(_FrozenDeliveryModel):
    repository_id: str
    broker_repository_id: str
    remote_alias: str
    remote_identity: str
    base_ref: str
    expected_base_commit: str
    destination_ref: str

    @field_validator(
        "repository_id",
        "broker_repository_id",
        "remote_alias",
        "remote_identity",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("base_ref", "destination_ref")
    @classmethod
    def validate_ref(cls, value: str, info) -> str:
        return _git_ref(value, info.field_name)

    @field_validator("expected_base_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        return _object_id(value, "expected_base_commit")

    @model_validator(mode="after")
    def validate_distinct_refs(self) -> RemoteGitRepositoryAuthority:
        if self.base_ref == self.destination_ref:
            raise ValueError("Remote Git delivery cannot update its admitted base ref.")
        return self


class RemoteGitCommitAuthority(_FrozenDeliveryModel):
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    authored_at: str
    title: str
    body: str = ""

    @field_validator("author_name", "committer_name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        value = _identifier(value, info.field_name, maximum=256)
        if "<" in value or ">" in value:
            raise ValueError(f"{info.field_name} cannot contain angle brackets.")
        return value

    @field_validator("author_email", "committer_email")
    @classmethod
    def validate_email(cls, value: str, info) -> str:
        value = _identifier(value, info.field_name, maximum=320)
        if _EMAIL_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a bounded email address.")
        return value

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = _identifier(value, "title", maximum=256)
        if "\n" in value or "\r" in value:
            raise ValueError("Commit title must be one line.")
        return value

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        if type(value) is not str or "\x00" in value or len(value.encode("utf-8")) > 8_192:
            raise ValueError("Commit body must be bounded UTF-8 text without NUL bytes.")
        return value

    @field_validator("authored_at")
    @classmethod
    def validate_authored_at(cls, value: str) -> str:
        value = _identifier(value, "authored_at", maximum=64)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("authored_at must be an ISO-8601 timestamp.") from None
        if parsed.tzinfo is None:
            raise ValueError("authored_at must include a timezone.")
        return value

    @property
    def message(self) -> str:
        return self.title if not self.body else f"{self.title}\n\n{self.body}"

    @property
    def message_sha256(self) -> str:
        return "sha256:" + sha256(self.message.encode("utf-8")).hexdigest()


class RemoteGitSecurityAuthority(_FrozenDeliveryModel):
    broker_behavior_fingerprint: str
    credential_profile_id: str
    egress_profile_id: str
    policy_fingerprint: str
    approval_policy_fingerprint: str
    redaction_profile_fingerprint: str

    @field_validator("credential_profile_id", "egress_profile_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "broker_behavior_fingerprint",
        "policy_fingerprint",
        "approval_policy_fingerprint",
        "redaction_profile_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)


class RemoteGitDeliveryLimits(_FrozenDeliveryModel):
    timeout_seconds: StrictInt = Field(default=120, ge=1, le=1800)
    output_limit_bytes: StrictInt = Field(default=64 * 1024, ge=1024, le=1024 * 1024)
    max_paths: StrictInt = Field(default=20_000, ge=1, le=100_000)
    max_path_bytes: StrictInt = Field(default=4096, ge=1, le=16_384)
    max_file_bytes: StrictInt = Field(default=16 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    max_total_file_bytes: StrictInt = Field(
        default=512 * 1024 * 1024,
        ge=1,
        le=1024 * 1024 * 1024,
    )
    max_manifest_bytes: StrictInt = Field(default=4 * 1024 * 1024, ge=1024, le=16 * 1024 * 1024)
    max_git_objects: StrictInt = Field(default=100_000, ge=1, le=1_000_000)
    max_git_storage_bytes: StrictInt = Field(
        default=1024 * 1024 * 1024,
        ge=1024,
        le=2 * 1024 * 1024 * 1024,
    )
    retry_limit: StrictInt = Field(default=1, ge=0, le=1)


class RemoteGitDeliveryRequest(_FrozenDeliveryModel):
    schema_version: Literal["cayu.remote_git_delivery.v1"] = REMOTE_GIT_DELIVERY_SCHEMA_VERSION
    delivery_id: str
    session_id: str
    idempotency_key: str
    source: RemoteGitSourceAuthority
    repository: RemoteGitRepositoryAuthority
    commit: RemoteGitCommitAuthority
    security: RemoteGitSecurityAuthority
    limits: RemoteGitDeliveryLimits = Field(default_factory=RemoteGitDeliveryLimits)
    force_update: Literal[False] = False
    delete_ref: Literal[False] = False

    @field_validator("delivery_id", "session_id", "idempotency_key")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "remote_git_delivery_request")


class RemoteGitDeliveryApproval(_FrozenDeliveryModel):
    approval_id: str
    request_fingerprint: str
    prepared_tree: str
    policy_fingerprint: str
    commit_approved: StrictBool
    push_approved: StrictBool

    @field_validator("approval_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _identifier(value, "approval_id")

    @field_validator("request_fingerprint", "policy_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @field_validator("prepared_tree")
    @classmethod
    def validate_tree(cls, value: str) -> str:
        return _object_id(value, "prepared_tree")

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "remote_git_delivery_approval")


class RemoteGitStepEvidence(_FrozenDeliveryModel):
    operation: str
    argv_sha256: str
    status: Literal["passed", "nonzero", "timed_out", "cancelled", "failed"]
    exit_code: StrictInt | None = None
    duration_ms: StrictInt = Field(ge=0)
    output_sha256: str
    output_truncated: StrictBool = False

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        return _identifier(value, "operation")

    @field_validator("argv_sha256", "output_sha256")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)


class RemoteGitPreparedIntent(_FrozenDeliveryModel):
    delivery_id: str
    request_fingerprint: str
    repository_id: str
    remote_identity: str
    observed_base_commit: str
    observed_destination_commit: str | None = None
    tree: str
    changed_paths: tuple[str, ...]
    source_revision: str
    source_manifest_sha256: str
    commit_message_sha256: str
    steps: tuple[RemoteGitStepEvidence, ...]

    @field_validator("delivery_id", "repository_id", "remote_identity")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "request_fingerprint",
        "source_revision",
        "source_manifest_sha256",
        "commit_message_sha256",
    )
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @field_validator("observed_base_commit", "observed_destination_commit", "tree")
    @classmethod
    def validate_object(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _object_id(value, info.field_name)

    @field_validator("changed_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("changed_paths must be a canonical path set.")
        return tuple(_source_path(item) for item in value)

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "remote_git_prepared_intent")


class RemoteGitDeliveryResult(_FrozenDeliveryModel):
    schema_version: Literal["cayu.remote_git_delivery_result.v1"] = (
        "cayu.remote_git_delivery_result.v1"
    )
    delivery_id: str
    session_id: str
    request_fingerprint: str
    state: RemoteGitDeliveryState
    product_result_artifact_id: str
    product_result_sha256: str
    product_run_id: str
    source_workspace_id: str
    final_source_revision: str
    diff_artifact_id: str
    diff_sha256: str
    check_evidence_sha256: str
    repository_id: str
    broker_repository_id: str
    remote_identity: str
    base_ref: str
    expected_base_commit: str
    observed_base_commit: str | None = None
    destination_ref: str
    destination_before: str | None = None
    destination_after: str | None = None
    local_commit: str | None = None
    parent_commit: str | None = None
    tree: str | None = None
    commit_message_sha256: str
    policy_fingerprint: str
    approval_policy_fingerprint: str
    redaction_profile_fingerprint: str
    approval_id: str | None = None
    approval_fingerprint: str | None = None
    credential_profile_id: str
    egress_profile_id: str
    broker_behavior_fingerprint: str
    limits: RemoteGitDeliveryLimits
    steps: tuple[RemoteGitStepEvidence, ...] = ()
    cleanup_settled: StrictBool = False
    reason_code: str | None = None
    next_commit: str | None = None
    next_ref: str | None = None

    @field_validator(
        "delivery_id",
        "session_id",
        "product_result_artifact_id",
        "product_run_id",
        "source_workspace_id",
        "diff_artifact_id",
        "repository_id",
        "broker_repository_id",
        "remote_identity",
        "approval_id",
        "credential_profile_id",
        "egress_profile_id",
        "reason_code",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _identifier(value, info.field_name, maximum=1024)

    @field_validator(
        "request_fingerprint",
        "product_result_sha256",
        "final_source_revision",
        "diff_sha256",
        "check_evidence_sha256",
        "commit_message_sha256",
        "policy_fingerprint",
        "approval_policy_fingerprint",
        "redaction_profile_fingerprint",
        "approval_fingerprint",
        "broker_behavior_fingerprint",
    )
    @classmethod
    def validate_optional_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _fingerprint(value, info.field_name)

    @field_validator("base_ref", "destination_ref", "next_ref")
    @classmethod
    def validate_optional_ref(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _git_ref(value, info.field_name)

    @field_validator(
        "expected_base_commit",
        "observed_base_commit",
        "destination_before",
        "destination_after",
        "local_commit",
        "parent_commit",
        "tree",
        "next_commit",
    )
    @classmethod
    def validate_optional_object(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _object_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_pushed_claim(self) -> RemoteGitDeliveryResult:
        if (self.approval_id is None) != (self.approval_fingerprint is None):
            raise ValueError("Delivery approval identity and fingerprint must settle together.")
        if self.local_commit is None:
            if self.parent_commit is not None:
                raise ValueError("A delivery without a local commit cannot claim its parent.")
        elif self.parent_commit != self.expected_base_commit or self.tree is None:
            raise ValueError("A local delivery commit requires its exact parent and tree.")
        if self.state is RemoteGitDeliveryState.PUSHED and (
            self.local_commit is None
            or self.destination_after != self.local_commit
            or self.next_commit != self.local_commit
            or self.next_ref != self.destination_ref
            or not self.cleanup_settled
        ):
            raise ValueError("Pushed delivery requires exact remote observation and cleanup.")
        if self.state is not RemoteGitDeliveryState.PUSHED and (
            self.next_commit is not None or self.next_ref is not None
        ):
            raise ValueError("Only a pushed delivery can expose settled next-step authority.")
        if self.cleanup_settled and self.state not in {
            RemoteGitDeliveryState.PUSHED,
            RemoteGitDeliveryState.DENIED,
            RemoteGitDeliveryState.CONFLICT,
            RemoteGitDeliveryState.FAILED,
        }:
            raise ValueError("Unsettled delivery cannot claim terminal cleanup.")
        if self.state is RemoteGitDeliveryState.PARTIAL and (
            self.local_commit is None or self.destination_after != self.local_commit
        ):
            raise ValueError("Partial delivery requires an exact remote commit observation.")
        return self

    @property
    def digest(self) -> str:
        return sha256(_canonical_model_bytes(self, "remote_git_delivery_result")).hexdigest()


class RemoteGitLifecycleReceipt(_FrozenDeliveryModel):
    delivery_id: str
    request_fingerprint: str
    ordinal: StrictInt = Field(ge=1, le=REMOTE_GIT_MAX_LIFECYCLE_RECEIPTS)
    state: RemoteGitDeliveryState
    prior_state: RemoteGitDeliveryState | None = None
    evidence_sha256: str | None = None
    tree: str | None = None
    commit: str | None = None
    reason_code: str | None = None

    @field_validator("delivery_id", "reason_code")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _identifier(value, info.field_name, maximum=1024)

    @field_validator("request_fingerprint", "evidence_sha256")
    @classmethod
    def validate_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _fingerprint(value, info.field_name)

    @field_validator("tree", "commit")
    @classmethod
    def validate_object(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _object_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_transition(self) -> RemoteGitLifecycleReceipt:
        if self.ordinal == 1:
            if self.prior_state is not None:
                raise ValueError("The first Remote Git lifecycle receipt has no prior state.")
        elif self.prior_state is None:
            raise ValueError("Later Remote Git lifecycle receipts require prior_state.")
        if self.prior_state is self.state and not (
            self.state
            in {
                RemoteGitDeliveryState.DENIED,
                RemoteGitDeliveryState.CONFLICT,
                RemoteGitDeliveryState.FAILED,
            }
            and self.evidence_sha256 is not None
        ):
            raise ValueError("Remote Git lifecycle transitions must change state.")
        return self


@dataclass(frozen=True)
class RemoteGitDeliveryPublication:
    result: RemoteGitDeliveryResult
    artifact: CodingArtifactReference


@dataclass(frozen=True)
class RemoteGitHttpCredentials:
    """Vault-backed HTTPS credential refs resolved only by remote Git calls."""

    credential_profile_id: str
    username: SecretRef
    password: SecretRef
    resolver: SecretResolver

    def __post_init__(self) -> None:
        _identifier(self.credential_profile_id, "credential_profile_id")
        if type(self.username) is not SecretRef or type(self.password) is not SecretRef:
            raise TypeError("Remote Git HTTPS credentials require SecretRef values.")
        validate_secret_resolver(self.resolver)


@dataclass(frozen=True)
class RemoteGitRemoteConfig:
    alias: str
    remote_identity: str
    url: str
    default_branch_ref: str
    credential_profile_id: str = "none"
    egress_profile_id: str = "application-local"
    credentials: RemoteGitHttpCredentials | None = None

    def __post_init__(self) -> None:
        _identifier(self.alias, "alias")
        _identifier(self.remote_identity, "remote_identity")
        url = _identifier(self.url, "url", maximum=4096)
        if url.startswith("-") or "::" in url:
            raise ValueError("Remote Git URLs cannot invoke Git options or remote helpers.")
        parsed = urlsplit(url) if "://" in url else None
        if parsed is not None and (parsed.username is not None or parsed.password is not None):
            raise ValueError("Remote Git URLs cannot embed credentials.")
        if self.credentials is not None and (parsed is None or parsed.scheme != "https"):
            raise ValueError("Vault-backed Remote Git credentials require HTTPS.")
        if parsed is None:
            if not Path(url).is_absolute():
                raise ValueError("Remote Git requires HTTPS or an absolute local fixture path.")
        elif (
            parsed.scheme not in {"https", "file"}
            or parsed.query
            or parsed.fragment
            or (parsed.scheme == "https" and not parsed.hostname)
            or (parsed.scheme == "file" and (parsed.netloc or not parsed.path.startswith("/")))
        ):
            raise ValueError("Remote Git transport is not an admitted HTTPS or local path.")
        _git_ref(self.default_branch_ref, "default_branch_ref")
        _identifier(self.credential_profile_id, "credential_profile_id")
        _identifier(self.egress_profile_id, "egress_profile_id")
        if self.credentials is not None:
            if self.credentials.credential_profile_id != self.credential_profile_id:
                raise ValueError("Credential configuration conflicts with its public profile ID.")
            if parsed is None or parsed.scheme != "https":
                raise ValueError("Vault-backed Remote Git credentials require HTTPS.")
        elif self.credential_profile_id != "none":
            raise ValueError("A non-none credential profile requires credential configuration.")


@dataclass(frozen=True)
class RemoteGitBrokerProfile:
    broker_id: str
    repository_id: str
    broker_repository_id: str
    root: Path
    git_executable: str
    behavior_fingerprint: str
    remotes: Mapping[str, RemoteGitRemoteConfig]
    destination_prefix: str = "refs/heads/cayu/"

    def __post_init__(self) -> None:
        _identifier(self.broker_id, "broker_id")
        if os.name != "posix":
            raise NotImplementedError(
                "Remote Git requires a POSIX host with process resource limits."
            )
        _identifier(self.repository_id, "repository_id")
        _identifier(self.broker_repository_id, "broker_repository_id")
        configured_root = Path(self.root).expanduser()
        if not configured_root.is_absolute() or configured_root == Path(configured_root.anchor):
            raise ValueError("Remote Git broker root must be a bounded absolute directory.")
        if configured_root.is_symlink():
            raise ValueError("Remote Git broker root cannot be a symbolic link.")
        configured_root.mkdir(parents=True, exist_ok=True)
        root = configured_root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Remote Git broker root is not a directory: {root}")
        _secure_owned_directory(root, "broker root")
        object.__setattr__(self, "root", root)
        git = Path(self.git_executable).expanduser()
        if not git.is_absolute():
            raise ValueError("git_executable must be an exact executable file.")
        git = git.resolve()
        if not git.is_file() or not os.access(git, os.X_OK):
            raise ValueError("git_executable must be an exact executable file.")
        object.__setattr__(self, "git_executable", str(git))
        object.__setattr__(
            self,
            "behavior_fingerprint",
            _fingerprint(self.behavior_fingerprint, "behavior_fingerprint"),
        )
        copied = dict(self.remotes)
        if not copied or any(alias != remote.alias for alias, remote in copied.items()):
            raise ValueError("Remote Git remotes must be a non-empty alias-keyed mapping.")
        object.__setattr__(self, "remotes", MappingProxyType(copied))
        prefix = _identifier(self.destination_prefix, "destination_prefix", maximum=240)
        if not prefix.startswith("refs/heads/") or not prefix.endswith("/"):
            raise ValueError("destination_prefix must be a branch-ref namespace ending in '/'.")


def _source_path(value: str) -> str:
    value = _identifier(value, "source path", maximum=16_384)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or value != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0].casefold() in _PROTECTED_SOURCE_ROOTS
    ):
        raise RemoteGitDeliveryAdmissionError("Source manifest contains a protected path.")
    return value


class RemoteGitDeliveryRepository:
    """Append-only delivery authority, lifecycle, and result storage."""

    def __init__(self, store: ArtifactStore) -> None:
        if not isinstance(store, ArtifactStore):
            raise TypeError("store must implement ArtifactStore.")
        self.store = store

    def request_artifact_id(self, delivery_id: str) -> str:
        return _artifact_id("remote-git-request-v1", delivery_id)

    def prepared_artifact_id(self, delivery_id: str) -> str:
        return _artifact_id("remote-git-prepared-v1", delivery_id)

    def approval_artifact_id(self, delivery_id: str) -> str:
        return _artifact_id("remote-git-approval-v1", delivery_id)

    def lifecycle_artifact_id(self, delivery_id: str, ordinal: int) -> str:
        return _artifact_id("remote-git-lifecycle-v1", delivery_id, str(ordinal))

    @staticmethod
    def _validate_session_json_artifact(
        result: ArtifactReadResult,
        *,
        artifact_id: str,
        session_id: str,
        filename: str,
    ) -> ArtifactReadResult:
        try:
            copied = copy_artifact_read_result(
                result,
                expected_artifact_id=artifact_id,
                max_content_bytes=REMOTE_GIT_MAX_RESULT_BYTES,
            )
        except (TypeError, ValueError) as exc:
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Remote Git artifact store returned inconsistent evidence."
            ) from exc
        content_digest = "sha256:" + sha256(copied.content).hexdigest()
        if (
            copied.truncated
            or copied.redaction_truncated
            or copied.metadata.scope is not ArtifactScope.SESSION
            or copied.metadata.session_id != session_id
            or copied.metadata.agent_name is not None
            or copied.metadata.environment_name is not None
            or copied.metadata.filename != filename
            or copied.metadata.content_type != "application/json"
            or copied.metadata.metadata.get("content_sha256") != content_digest
        ):
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Remote Git artifact authority is inconsistent."
            )
        return copied

    @staticmethod
    def _validate_prepared_authority(
        request: RemoteGitDeliveryRequest,
        prepared: RemoteGitPreparedIntent,
    ) -> None:
        if (
            prepared.delivery_id != request.delivery_id
            or prepared.request_fingerprint != request.fingerprint
            or prepared.repository_id != request.repository.repository_id
            or prepared.remote_identity != request.repository.remote_identity
            or prepared.observed_base_commit != request.repository.expected_base_commit
            or prepared.observed_destination_commit is not None
            or prepared.source_revision != request.source.final_source_revision
            or prepared.commit_message_sha256 != request.commit.message_sha256
        ):
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Prepared Git intent conflicts with durable request authority."
            )

    @staticmethod
    def _validate_result_authority(
        request: RemoteGitDeliveryRequest,
        result: RemoteGitDeliveryResult,
    ) -> None:
        if (
            result.delivery_id != request.delivery_id
            or result.session_id != request.session_id
            or result.request_fingerprint != request.fingerprint
            or result.product_result_artifact_id != request.source.product_result_artifact_id
            or result.product_result_sha256 != request.source.product_result_sha256
            or result.product_run_id != request.source.product_run_id
            or result.source_workspace_id != request.source.source_workspace_id
            or result.final_source_revision != request.source.final_source_revision
            or result.diff_artifact_id != request.source.diff_artifact_id
            or result.diff_sha256 != request.source.diff_sha256
            or result.check_evidence_sha256 != request.source.check_evidence_sha256
            or result.repository_id != request.repository.repository_id
            or result.broker_repository_id != request.repository.broker_repository_id
            or result.remote_identity != request.repository.remote_identity
            or result.base_ref != request.repository.base_ref
            or result.expected_base_commit != request.repository.expected_base_commit
            or result.destination_ref != request.repository.destination_ref
            or result.commit_message_sha256 != request.commit.message_sha256
            or result.policy_fingerprint != request.security.policy_fingerprint
            or result.approval_policy_fingerprint != request.security.approval_policy_fingerprint
            or result.redaction_profile_fingerprint
            != request.security.redaction_profile_fingerprint
            or result.credential_profile_id != request.security.credential_profile_id
            or result.egress_profile_id != request.security.egress_profile_id
            or result.broker_behavior_fingerprint != request.security.broker_behavior_fingerprint
            or result.limits != request.limits
        ):
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Remote Git result conflicts with durable request authority."
            )

    async def _ensure_model(
        self,
        value: BaseModel,
        *,
        artifact_id: str,
        filename: str,
        session_id: str,
        field_name: str,
    ) -> CodingArtifactReference:
        content = _canonical_model_bytes(value, field_name)
        if len(content) > REMOTE_GIT_MAX_RESULT_BYTES:
            raise ValueError("Remote Git durable evidence exceeds its publication bound.")
        try:
            existing = self._validate_session_json_artifact(
                await self.store.read_bytes(
                    artifact_id,
                    max_bytes=REMOTE_GIT_MAX_RESULT_BYTES,
                ),
                artifact_id=artifact_id,
                session_id=session_id,
                filename=filename,
            )
        except FileNotFoundError:
            await self.store.put_bytes(
                content,
                artifact_id=artifact_id,
                filename=filename,
                content_type="application/json",
                scope=ArtifactScope.SESSION,
                session_id=session_id,
                metadata={"content_sha256": "sha256:" + sha256(content).hexdigest()},
            )
            existing = self._validate_session_json_artifact(
                await self.store.read_bytes(
                    artifact_id,
                    max_bytes=REMOTE_GIT_MAX_RESULT_BYTES,
                ),
                artifact_id=artifact_id,
                session_id=session_id,
                filename=filename,
            )
        if existing.content != content:
            raise RemoteGitDeliveryAdmissionError(
                "Stable remote Git delivery identity is bound to different authority."
            )
        return CodingArtifactReference(
            artifact_id=artifact_id,
            sha256="sha256:" + sha256(content).hexdigest(),
            size_bytes=existing.metadata.size_bytes,
            content_type=existing.metadata.content_type,
        )

    async def ensure_request(self, request: RemoteGitDeliveryRequest) -> CodingArtifactReference:
        return await self._ensure_model(
            request,
            artifact_id=self.request_artifact_id(request.delivery_id),
            filename="remote-git-delivery-request.json",
            session_id=request.session_id,
            field_name="remote_git_delivery_request",
        )

    async def ensure_prepared(
        self,
        request: RemoteGitDeliveryRequest,
        prepared: RemoteGitPreparedIntent,
    ) -> CodingArtifactReference:
        try:
            self._validate_prepared_authority(request, prepared)
        except RemoteGitDeliveryReconstructionRequiredError as exc:
            raise RemoteGitDeliveryAdmissionError(str(exc)) from exc
        return await self._ensure_model(
            prepared,
            artifact_id=self.prepared_artifact_id(request.delivery_id),
            filename="remote-git-prepared-intent.json",
            session_id=request.session_id,
            field_name="remote_git_prepared_intent",
        )

    async def ensure_approval(
        self,
        request: RemoteGitDeliveryRequest,
        approval: RemoteGitDeliveryApproval,
    ) -> CodingArtifactReference:
        if (
            approval.request_fingerprint != request.fingerprint
            or approval.policy_fingerprint != request.security.policy_fingerprint
            or not approval.commit_approved
            or not approval.push_approved
        ):
            raise RemoteGitDeliveryAdmissionError("Delivery approval conflicts with request.")
        return await self._ensure_model(
            approval,
            artifact_id=self.approval_artifact_id(request.delivery_id),
            filename="remote-git-delivery-approval.json",
            session_id=request.session_id,
            field_name="remote_git_delivery_approval",
        )

    async def append_lifecycle(
        self,
        request: RemoteGitDeliveryRequest,
        receipt: RemoteGitLifecycleReceipt,
    ) -> CodingArtifactReference:
        if (
            receipt.delivery_id != request.delivery_id
            or receipt.request_fingerprint != request.fingerprint
        ):
            raise RemoteGitDeliveryAdmissionError(
                "Remote Git lifecycle receipt conflicts with request."
            )
        return await self._ensure_model(
            receipt,
            artifact_id=self.lifecycle_artifact_id(request.delivery_id, receipt.ordinal),
            filename=f"remote-git-lifecycle-{receipt.ordinal}.json",
            session_id=request.session_id,
            field_name="remote_git_lifecycle_receipt",
        )

    async def load_prepared(
        self,
        request: RemoteGitDeliveryRequest,
    ) -> RemoteGitPreparedIntent | None:
        try:
            artifact_id = self.prepared_artifact_id(request.delivery_id)
            result = self._validate_session_json_artifact(
                await self.store.read_bytes(
                    artifact_id,
                    max_bytes=REMOTE_GIT_MAX_RESULT_BYTES,
                ),
                artifact_id=artifact_id,
                session_id=request.session_id,
                filename="remote-git-prepared-intent.json",
            )
        except FileNotFoundError:
            return None
        try:
            prepared = RemoteGitPreparedIntent.model_validate_json(result.content)
        except Exception as exc:
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Prepared Git intent cannot be reconstructed."
            ) from exc
        self._validate_prepared_authority(request, prepared)
        return prepared

    async def load_lifecycle(
        self,
        request: RemoteGitDeliveryRequest,
    ) -> tuple[RemoteGitLifecycleReceipt, ...]:
        receipts: list[RemoteGitLifecycleReceipt] = []
        missing_ordinal = False
        for ordinal in range(1, REMOTE_GIT_MAX_LIFECYCLE_RECEIPTS + 1):
            artifact_id = self.lifecycle_artifact_id(request.delivery_id, ordinal)
            try:
                result = self._validate_session_json_artifact(
                    await self.store.read_bytes(
                        artifact_id,
                        max_bytes=REMOTE_GIT_MAX_RESULT_BYTES,
                    ),
                    artifact_id=artifact_id,
                    session_id=request.session_id,
                    filename=f"remote-git-lifecycle-{ordinal}.json",
                )
            except FileNotFoundError:
                missing_ordinal = True
                continue
            if missing_ordinal:
                raise RemoteGitDeliveryReconstructionRequiredError(
                    "Remote Git lifecycle contains a reconstruction gap."
                )
            try:
                receipt = RemoteGitLifecycleReceipt.model_validate_json(result.content)
            except Exception as exc:
                raise RemoteGitDeliveryReconstructionRequiredError(
                    "Remote Git lifecycle receipt cannot be reconstructed."
                ) from exc
            if (
                receipt.delivery_id != request.delivery_id
                or receipt.request_fingerprint != request.fingerprint
                or receipt.ordinal != ordinal
                or (receipts and receipt.prior_state is not receipts[-1].state)
            ):
                raise RemoteGitDeliveryReconstructionRequiredError(
                    "Remote Git lifecycle reconstruction is inconsistent."
                )
            receipts.append(receipt)
        return tuple(receipts)

    async def publish_result(
        self,
        request: RemoteGitDeliveryRequest,
        result: RemoteGitDeliveryResult,
    ) -> RemoteGitDeliveryPublication:
        try:
            self._validate_result_authority(request, result)
        except RemoteGitDeliveryReconstructionRequiredError as exc:
            raise RemoteGitDeliveryAdmissionError(str(exc)) from exc
        content = _canonical_model_bytes(result, "remote_git_delivery_result")
        if len(content) > REMOTE_GIT_MAX_RESULT_BYTES:
            raise ValueError("Remote Git result exceeds its publication bound.")
        digest = sha256(content).hexdigest()
        artifact_id = _artifact_id("remote-git-result-v1", request.fingerprint, digest)
        artifact = await self._ensure_model(
            result,
            artifact_id=artifact_id,
            filename="remote-git-delivery-result.json",
            session_id=request.session_id,
            field_name="remote_git_delivery_result",
        )
        return RemoteGitDeliveryPublication(result=result, artifact=artifact)

    async def load_result(
        self,
        request: RemoteGitDeliveryRequest,
        digest: str,
    ) -> RemoteGitDeliveryPublication:
        digest = _fingerprint(digest, "digest").removeprefix("sha256:")
        artifact_id = _artifact_id("remote-git-result-v1", request.fingerprint, digest)
        stored = self._validate_session_json_artifact(
            await self.store.read_bytes(
                artifact_id,
                max_bytes=REMOTE_GIT_MAX_RESULT_BYTES,
            ),
            artifact_id=artifact_id,
            session_id=request.session_id,
            filename="remote-git-delivery-result.json",
        )
        if sha256(stored.content).hexdigest() != digest:
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Remote Git result does not match its durable identity."
            )
        try:
            result = RemoteGitDeliveryResult.model_validate_json(stored.content)
        except Exception as exc:
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Remote Git result cannot be reconstructed."
            ) from exc
        self._validate_result_authority(request, result)
        return RemoteGitDeliveryPublication(
            result=result,
            artifact=CodingArtifactReference(
                artifact_id=artifact_id,
                sha256="sha256:" + digest,
                size_bytes=stored.metadata.size_bytes,
                content_type=stored.metadata.content_type,
            ),
        )


def remote_git_delivery_request(
    publication: CodingProductPublication,
    *,
    delivery_id: str,
    session_id: str,
    idempotency_key: str,
    repository: RemoteGitRepositoryAuthority,
    commit: RemoteGitCommitAuthority,
    security: RemoteGitSecurityAuthority,
    limits: RemoteGitDeliveryLimits | None = None,
) -> RemoteGitDeliveryRequest:
    """Bind one immutable delivery request to an accepted patch-ready result."""

    if type(publication) is not CodingProductPublication:
        raise TypeError("publication must be CodingProductPublication.")
    candidate = publication.candidate
    if candidate.state is not CodingProductState.PATCH_READY_FOR_DELIVERY:
        raise RemoteGitDeliveryAdmissionError(
            "Remote Git delivery requires patch_ready_for_delivery evidence."
        )
    if candidate.git is None:
        raise RemoteGitDeliveryAdmissionError("Patch-ready result is missing Git evidence.")
    return RemoteGitDeliveryRequest(
        delivery_id=delivery_id,
        session_id=session_id,
        idempotency_key=idempotency_key,
        source=RemoteGitSourceAuthority(
            product_result_artifact_id=publication.artifact.artifact_id,
            product_result_sha256=publication.artifact.sha256,
            product_request_fingerprint=candidate.request_fingerprint,
            product_run_id=candidate.product_run_id,
            source_workspace_id=candidate.source_workspace_id,
            final_source_revision=candidate.final_revision or "",
            diff_artifact_id=candidate.git.artifact.artifact_id,
            diff_sha256=candidate.git.artifact.sha256,
            check_evidence_sha256=_coding_check_evidence_sha256(candidate.checks),
        ),
        repository=repository,
        commit=commit,
        security=security,
        limits=limits or RemoteGitDeliveryLimits(),
    )


def approve_remote_git_delivery(
    request: RemoteGitDeliveryRequest,
    prepared: RemoteGitPreparedIntent,
    *,
    approval_id: str,
) -> RemoteGitDeliveryApproval:
    """Create exact application-owned approval for commit and push effects."""

    if prepared.request_fingerprint != request.fingerprint:
        raise RemoteGitDeliveryAdmissionError("Prepared Git intent conflicts with request.")
    return RemoteGitDeliveryApproval(
        approval_id=approval_id,
        request_fingerprint=request.fingerprint,
        prepared_tree=prepared.tree,
        policy_fingerprint=request.security.policy_fingerprint,
        commit_approved=True,
        push_approved=True,
    )


class RemoteGitDeliveryBroker:
    """Prepare, approve, commit, push, and reconcile one bounded Git delivery."""

    def __init__(
        self,
        profile: RemoteGitBrokerProfile,
        *,
        repository: RemoteGitDeliveryRepository,
        coding_repository: CodingProductArtifactRepository,
    ) -> None:
        if type(profile) is not RemoteGitBrokerProfile:
            raise TypeError("profile must be RemoteGitBrokerProfile.")
        if type(repository) is not RemoteGitDeliveryRepository:
            raise TypeError("repository must be RemoteGitDeliveryRepository.")
        if type(coding_repository) is not CodingProductArtifactRepository:
            raise TypeError("coding_repository must be CodingProductArtifactRepository.")
        self.profile = profile
        self.repository = repository
        self.coding_repository = coding_repository
        self._local_runner = LocalRunner(profile.root, inherit_env=False)

    async def _ensure_configuration(
        self, request: RemoteGitDeliveryRequest, remote: RemoteGitRemoteConfig
    ) -> None:
        credentials = remote.credentials
        material = {
            "broker_id": self.profile.broker_id,
            "root": str(self.profile.root),
            "git_executable": self.profile.git_executable,
            "git_executable_sha256": executable_digest(Path(self.profile.git_executable)),
            "destination_prefix": self.profile.destination_prefix,
            "url": remote.url,
            "default_branch_ref": remote.default_branch_ref,
            "username_ref": (
                None
                if credentials is None
                else credentials.username.model_dump(mode="json", warnings=False)
            ),
            "password_ref": (
                None
                if credentials is None
                else credentials.password.model_dump(mode="json", warnings=False)
            ),
        }
        authority = _RemoteGitConfigurationAuthority(
            request_fingerprint=request.fingerprint,
            configuration_sha256="sha256:"
            + sha256(
                canonical_durable_json_bytes(material, "remote_git_configuration")
            ).hexdigest(),
        )
        await self.repository._ensure_model(
            authority,
            artifact_id=_artifact_id("remote-git-configuration-v1", request.delivery_id),
            filename="remote-git-configuration.json",
            session_id=request.session_id,
            field_name="remote_git_configuration_authority",
        )

    def _remote(self, request: RemoteGitDeliveryRequest) -> RemoteGitRemoteConfig:
        try:
            remote = self.profile.remotes[request.repository.remote_alias]
        except KeyError:
            raise RemoteGitDeliveryAdmissionError(
                "Delivery request names an unconfigured remote alias."
            ) from None
        if (
            request.repository.repository_id != self.profile.repository_id
            or request.repository.broker_repository_id != self.profile.broker_repository_id
            or remote.remote_identity != request.repository.remote_identity
            or remote.credential_profile_id != request.security.credential_profile_id
            or remote.egress_profile_id != request.security.egress_profile_id
        ):
            raise RemoteGitDeliveryAdmissionError(
                "Configured remote authority conflicts with delivery request."
            )
        if request.repository.destination_ref == remote.default_branch_ref:
            raise RemoteGitDeliveryAdmissionError("Default-branch delivery is forbidden.")
        if not request.repository.destination_ref.startswith(self.profile.destination_prefix):
            raise RemoteGitDeliveryAdmissionError(
                "Destination ref is outside the implementation-owned namespace."
            )
        if request.security.broker_behavior_fingerprint != self.profile.behavior_fingerprint:
            raise RemoteGitDeliveryAdmissionError("Broker behavior authority has drifted.")
        return remote

    def _delivery_root(self, request: RemoteGitDeliveryRequest) -> Path:
        name = sha256(request.delivery_id.encode("utf-8")).hexdigest()[:32]
        return self.profile.root / f"delivery-{name}"

    async def _cleanup_delivery_root(self, request: RemoteGitDeliveryRequest, root: Path) -> bool:
        if root.parent != self.profile.root:
            return False
        result = await self._local_runner.exec(
            ExecCommand.process(
                sys.executable,
                "-I",
                str(Path(__file__).with_name("_remote_git_cleanup.py")),
                str(root),
            ),
            timeout_s=request.limits.timeout_seconds,
            output_limit_bytes=request.limits.output_limit_bytes,
        )
        return (
            result.exit_code == 0
            and not result.timed_out
            and not result.cancelled
            and not root.exists()
            and not root.is_symlink()
        )

    def _control_root(self) -> Path:
        root = self.profile.root / ".cayu-remote-git-control"
        root.mkdir(mode=0o700, exist_ok=True)
        _secure_owned_directory(root, "control root")
        home = root / "home"
        home.mkdir(mode=0o700, exist_ok=True)
        _secure_owned_directory(home, "control home")
        return root

    def _askpass_path(self) -> Path:
        control = self._control_root()
        helper = control / "git-askpass"
        content = (
            b"#!/bin/sh\n"
            b'case "$1" in\n'
            b"  *[Uu]sername*) printf '%s\\n' \"$CAYU_GIT_USERNAME\" ;;\n"
            b"  *[Pp]assword*) printf '%s\\n' \"$CAYU_GIT_PASSWORD\" ;;\n"
            b"  *) exit 1 ;;\n"
            b"esac\n"
        )
        descriptor: int | None = None
        created = False
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for _ in range(2):
            try:
                descriptor = os.open(helper, flags)
                break
            except FileNotFoundError:
                try:
                    descriptor = os.open(
                        helper,
                        flags | os.O_CREAT | os.O_EXCL,
                        stat.S_IRWXU,
                    )
                except FileExistsError:
                    continue
                created = True
                break
            except OSError as exc:
                raise RemoteGitDeliveryAdmissionError(
                    "Remote Git credential helper identity is inconsistent."
                ) from exc
        if descriptor is None:
            raise RemoteGitDeliveryAdmissionError(
                "Remote Git credential helper identity is inconsistent."
            )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise RemoteGitDeliveryAdmissionError(
                    "Remote Git credential helper identity is inconsistent."
                )
            if created:
                offset = 0
                while offset < len(content):
                    written = os.write(descriptor, content[offset:])
                    if written <= 0:
                        raise OSError("credential helper write made no progress")
                    offset += written
            elif os.read(descriptor, len(content) + 1) != content:
                raise RemoteGitDeliveryAdmissionError(
                    "Remote Git credential helper identity is inconsistent."
                )
            os.fchmod(descriptor, stat.S_IRWXU)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return helper

    def _remote_runner(self, remote: RemoteGitRemoteConfig) -> LocalRunner:
        credentials = remote.credentials
        if credentials is None:
            return self._local_runner
        return LocalRunner(
            self.profile.root,
            inherit_env=False,
            secret_env=(
                SecretEnv(name="CAYU_GIT_USERNAME", ref=credentials.username),
                SecretEnv(name="CAYU_GIT_PASSWORD", ref=credentials.password),
            ),
            secret_resolver=credentials.resolver,
        )

    def _git_environment(
        self,
        remote: RemoteGitRemoteConfig | None = None,
        *,
        commit: RemoteGitCommitAuthority | None = None,
    ) -> dict[str, str]:
        control = self._control_root()
        environment = {
            "HOME": str(control / "home"),
            "XDG_CONFIG_HOME": str(control / "home"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "GIT_EDITOR": "false",
            "GIT_SEQUENCE_EDITOR": "false",
            "GIT_ATTR_NOSYSTEM": "1",
            "LC_ALL": "C",
        }
        if remote is not None and remote.credentials is not None:
            environment.update(
                {
                    "GIT_ASKPASS": str(self._askpass_path()),
                    "SSH_ASKPASS": str(self._askpass_path()),
                }
            )
        if commit is not None:
            environment.update(
                {
                    "GIT_AUTHOR_NAME": commit.author_name,
                    "GIT_AUTHOR_EMAIL": commit.author_email,
                    "GIT_AUTHOR_DATE": commit.authored_at,
                    "GIT_COMMITTER_NAME": commit.committer_name,
                    "GIT_COMMITTER_EMAIL": commit.committer_email,
                    "GIT_COMMITTER_DATE": commit.authored_at,
                }
            )
        return environment

    async def _git(
        self,
        request: RemoteGitDeliveryRequest,
        operation: str,
        arguments: Sequence[str],
        *,
        cwd: Path,
        remote: RemoteGitRemoteConfig | None = None,
        stdin: str | None = None,
        commit: RemoteGitCommitAuthority | None = None,
    ) -> tuple[ExecResult, RemoteGitStepEvidence]:
        try:
            require_inert_repository_config(cwd)
        except (OSError, ValueError):
            raise RemoteGitDeliveryAdmissionError(
                "Broker repository contains unsafe Git configuration."
            ) from None
        argv = (
            self.profile.git_executable,
            "--no-pager",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "commit.gpgSign=false",
            "-c",
            "tag.gpgSign=false",
            "-c",
            "credential.helper=",
            "-c",
            "http.followRedirects=false",
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.https.allow=always",
            "-c",
            "protocol.file.allow=always",
            "-c",
            "fetch.unpackLimit=0",
            "-c",
            "pack.threads=1",
            *arguments,
        )
        public_argv = ["[GIT]"]
        for item in argv[1:]:
            public_argv.append("[REMOTE]" if remote is not None and item == remote.url else item)
        started = time.monotonic_ns()
        runner = self._remote_runner(remote) if remote is not None else self._local_runner
        try:
            result = await runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("_remote_git_process.py")),
                    str(request.limits.max_git_storage_bytes),
                    *argv,
                ),
                cwd=str(cwd),
                env={
                    **self._git_environment(remote, commit=commit),
                    # Inspect cwd itself, never an ancestor repository/config.
                    "GIT_CEILING_DIRECTORIES": str(cwd.parent.resolve()),
                },
                env_remove=_REMOTE_GIT_ENV_REMOVE,
                timeout_s=request.limits.timeout_seconds,
                stdin=stdin,
                output_limit_bytes=request.limits.output_limit_bytes,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise RemoteGitDeliveryError(
                f"Structured Git operation failed before returning evidence: {operation}"
            ) from None
        elapsed = max(0, (time.monotonic_ns() - started) // 1_000_000)
        output = f"{result.stdout}\n{result.stderr}".encode()
        status: Literal["passed", "nonzero", "timed_out", "cancelled", "failed"] = (
            "timed_out"
            if result.timed_out
            else "cancelled"
            if result.cancelled
            else "passed"
            if result.exit_code == 0
            else "nonzero"
        )
        evidence = RemoteGitStepEvidence(
            operation=operation,
            argv_sha256="sha256:"
            + sha256(canonical_durable_json_bytes(public_argv, "remote_git_argv")).hexdigest(),
            status=status,
            exit_code=result.exit_code,
            duration_ms=elapsed,
            output_sha256="sha256:" + sha256(output).hexdigest(),
            output_truncated=result.stdout_truncated or result.stderr_truncated,
        )
        return result, evidence

    @staticmethod
    def _validated_product_artifact(
        result: ArtifactReadResult,
        *,
        artifact: CodingArtifactReference,
        session_id: str,
        agent_name: str,
        environment_name: str,
        filename: str,
        content_type: str,
        expected_metadata: Mapping[str, object],
    ) -> ArtifactReadResult:
        try:
            copied = copy_artifact_read_result(
                result,
                expected_artifact_id=artifact.artifact_id,
                max_content_bytes=max(1, artifact.size_bytes),
            )
        except (TypeError, ValueError) as exc:
            raise RemoteGitDeliveryAdmissionError(
                "Artifact store returned inconsistent coding-product evidence."
            ) from exc
        content_digest = "sha256:" + sha256(copied.content).hexdigest()
        if (
            copied.truncated
            or copied.redaction_truncated
            or copied.source_bytes_read != copied.total_bytes
            or len(copied.content) != copied.total_bytes
            or copied.metadata.scope is not ArtifactScope.SESSION
            or copied.metadata.session_id != session_id
            or copied.metadata.agent_name != agent_name
            or copied.metadata.environment_name != environment_name
            or copied.metadata.filename != filename
            or copied.metadata.content_type != content_type
            or artifact.content_type != content_type
            or copied.metadata.size_bytes != artifact.size_bytes
            or content_digest != artifact.sha256
            or copied.metadata.metadata.get("content_sha256") != content_digest
            or any(
                copied.metadata.metadata.get(key) != value
                for key, value in expected_metadata.items()
            )
        ):
            raise RemoteGitDeliveryAdmissionError(
                "Retained coding-product artifact authority is inconsistent."
            )
        return copied

    async def _observe_remote(
        self,
        request: RemoteGitDeliveryRequest,
        remote: RemoteGitRemoteConfig,
        *,
        cwd: Path,
    ) -> tuple[dict[str, str], RemoteGitStepEvidence]:
        result, evidence = await self._git(
            request,
            "observe_remote_refs",
            (
                "ls-remote",
                "--refs",
                remote.url,
                request.repository.base_ref,
                request.repository.destination_ref,
            ),
            cwd=cwd,
            remote=remote,
        )
        _require_git_success(result, "observe remote refs")
        refs: dict[str, str] = {}
        expected_refs = {
            request.repository.base_ref,
            request.repository.destination_ref,
        }
        for line in result.stdout.splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise RemoteGitDeliveryError("Remote ref observation returned malformed output.")
            object_id, ref = parts
            observed_ref = _git_ref(ref, "observed ref")
            if observed_ref not in expected_refs or observed_ref in refs:
                raise RemoteGitDeliveryError("Remote ref observation returned unexpected output.")
            refs[observed_ref] = _object_id(
                object_id,
                "observed commit",
            )
        return refs, evidence

    async def _append_state(
        self,
        request: RemoteGitDeliveryRequest,
        receipts: list[RemoteGitLifecycleReceipt],
        state: RemoteGitDeliveryState,
        *,
        evidence_sha256: str | None = None,
        tree: str | None = None,
        commit: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        if delivery_write_pending():
            # Do not write cancellation diagnostics over an unsettled write.
            # The outer owner retains the store observation and fences reuse.
            return
        if (
            receipts
            and receipts[-1].state is state
            and (
                state
                not in {
                    RemoteGitDeliveryState.DENIED,
                    RemoteGitDeliveryState.CONFLICT,
                    RemoteGitDeliveryState.FAILED,
                }
                or receipts[-1].evidence_sha256 == evidence_sha256
            )
        ):
            return
        receipt = RemoteGitLifecycleReceipt(
            delivery_id=request.delivery_id,
            request_fingerprint=request.fingerprint,
            ordinal=len(receipts) + 1,
            prior_state=None if not receipts else receipts[-1].state,
            state=state,
            evidence_sha256=evidence_sha256,
            tree=tree,
            commit=commit,
            reason_code=reason_code,
        )
        await self.repository.append_lifecycle(request, receipt)
        receipts.append(receipt)

    async def _validated_product(
        self,
        request: RemoteGitDeliveryRequest,
        publication: CodingProductPublication,
        source_workspace: Workspace,
    ) -> WorkspaceRevisionObservation:
        if type(publication) is not CodingProductPublication:
            raise TypeError("publication must be CodingProductPublication.")
        if not isinstance(source_workspace, Workspace):
            raise TypeError("source_workspace must implement Workspace.")
        candidate = publication.candidate
        try:
            durable = await self.coding_repository.read_candidate(publication.result_reference)
        except Exception as exc:
            raise RemoteGitDeliveryAdmissionError(
                "Coding-product artifact authority is inconsistent."
            ) from exc
        if durable != candidate:
            raise RemoteGitDeliveryAdmissionError(
                "Coding-product publication conflicts with durable result evidence."
            )
        if candidate.initial_git.head_revision != request.repository.expected_base_commit:
            raise RemoteGitDeliveryAdmissionError(
                "Delivery parent conflicts with the retained coding baseline."
            )
        if (
            candidate.state is not CodingProductState.PATCH_READY_FOR_DELIVERY
            or candidate.git is None
            or publication.artifact.artifact_id != request.source.product_result_artifact_id
            or publication.artifact.sha256 != request.source.product_result_sha256
            or candidate.request_fingerprint != request.source.product_request_fingerprint
            or candidate.product_run_id != request.source.product_run_id
            or candidate.source_workspace_id != request.source.source_workspace_id
            or source_workspace.id != request.source.source_workspace_id
            or candidate.final_revision != request.source.final_source_revision
            or candidate.git.artifact.artifact_id != request.source.diff_artifact_id
            or candidate.git.artifact.sha256 != request.source.diff_sha256
        ):
            raise RemoteGitDeliveryAdmissionError(
                "Delivery source authority conflicts with patch-ready evidence."
            )
        check_digest = _coding_check_evidence_sha256(candidate.checks)
        if check_digest != request.source.check_evidence_sha256:
            raise RemoteGitDeliveryAdmissionError("Required-check evidence has drifted.")
        decision = coding_product_completion_decision(
            await self.coding_repository.load_request(
                candidate.product_run_id,
                session_id=candidate.session_id,
            ),
            candidate,
            evidence=WorkEvidenceReference(
                kind=CODING_PRODUCT_EVIDENCE_KIND,
                reference_id=publication.artifact.artifact_id,
                version="v1",
                digest=candidate.digest,
                available=True,
            ),
        )
        if decision.verdict is not CompletionVerdict.ACCEPTED:
            raise RemoteGitDeliveryAdmissionError(
                "Coding-product evidence no longer satisfies its deterministic verifier."
            )
        expected_diff_id = _artifact_id(
            "coding-product-git-diff-v1",
            candidate.session_id,
            candidate.git.event_id,
            request.source.diff_sha256.removeprefix("sha256:"),
        )
        if expected_diff_id != request.source.diff_artifact_id:
            raise RemoteGitDeliveryAdmissionError(
                "Retained Git diff evidence is not content-addressed."
            )
        diff = self._validated_product_artifact(
            await self.coding_repository.store.read_bytes(
                request.source.diff_artifact_id,
                max_bytes=max(1, candidate.git.artifact.size_bytes),
            ),
            artifact=candidate.git.artifact,
            session_id=candidate.session_id,
            agent_name=candidate.agent_name,
            environment_name=candidate.runtime.environment_name,
            filename="coding-product.diff",
            content_type="text/x-diff",
            expected_metadata={
                "schema_version": "cayu.coding_product_git_diff.v1",
                "event_id": candidate.git.event_id,
            },
        )
        if "sha256:" + sha256(diff.content).hexdigest() != request.source.diff_sha256:
            raise RemoteGitDeliveryAdmissionError("Retained Git diff evidence has drifted.")
        final_artifact = candidate.final_source.artifact
        expected_final_source_id = _artifact_id(
            "coding-product-source-observation-v1",
            candidate.session_id,
            "final",
            final_artifact.sha256.removeprefix("sha256:"),
        )
        if expected_final_source_id != final_artifact.artifact_id:
            raise RemoteGitDeliveryAdmissionError(
                "Final source manifest evidence is not content-addressed."
            )
        observed_artifact = self._validated_product_artifact(
            await self.coding_repository.store.read_bytes(
                final_artifact.artifact_id,
                max_bytes=max(1, final_artifact.size_bytes),
            ),
            artifact=final_artifact,
            session_id=candidate.session_id,
            agent_name=candidate.agent_name,
            environment_name=candidate.runtime.environment_name,
            filename="coding-product-source-final.json",
            content_type="application/json",
            expected_metadata={
                "schema_version": "cayu.coding_product_source_observation.v1",
                "phase": "final",
                "workspace_id": candidate.final_source.workspace_id,
                "observer": candidate.final_source.observer,
                "status": candidate.final_source.status.value,
            },
        )
        recorded = WorkspaceRevisionObservation.model_validate_json(observed_artifact.content)
        limits = WorkspaceRevisionObservationLimits(
            max_paths=request.limits.max_paths,
            max_path_bytes=request.limits.max_path_bytes,
            max_file_bytes=request.limits.max_file_bytes,
            max_total_file_bytes=request.limits.max_total_file_bytes,
            max_manifest_bytes=request.limits.max_manifest_bytes,
        )
        fresh = await observe_deterministic_workspace(
            source_workspace,
            observer="cayu-remote-git-delivery-source",
            limits=limits,
        )
        if (
            recorded.status is not WorkspaceRevisionObservationStatus.SUPPORTED
            or fresh.status is not WorkspaceRevisionObservationStatus.SUPPORTED
            or recorded.path_scope != "complete"
            or fresh.path_scope != "complete"
            or recorded.revision != request.source.final_source_revision
            or fresh.revision != request.source.final_source_revision
            or recorded.paths != fresh.paths
        ):
            raise RemoteGitDeliveryAdmissionError(
                "Application source no longer matches its patch-ready final manifest."
            )
        return recorded

    async def _record_cancellation(
        self,
        cancellation: asyncio.CancelledError,
        request: RemoteGitDeliveryRequest,
        state: RemoteGitDeliveryState,
        *,
        tree: str | None = None,
        commit: str | None = None,
        reason_code: str,
    ) -> None:
        """Keep caller cancellation authoritative if its diagnostic write fails."""
        if delivery_write_pending():
            return
        try:
            receipts = list(await self.repository.load_lifecycle(request))
            await self._append_state(
                request, receipts, state, tree=tree, commit=commit, reason_code=reason_code
            )
        except BaseException as secondary:
            if secondary is cancellation:
                raise
            causes = [] if cancellation.__cause__ is None else [cancellation.__cause__]
            causes.append(secondary)
            raise cancellation from BaseExceptionGroup(
                "Remote Git cancellation diagnostic settlement failed", causes
            )

    async def _materialize_source(
        self,
        request: RemoteGitDeliveryRequest,
        source_workspace: Workspace,
        observation: WorkspaceRevisionObservation,
        root: Path,
    ) -> str:
        total = 0
        manifest: list[dict[str, object]] = []
        for entry in observation.paths:
            path = _source_path(entry.path)
            if len(path.encode("utf-8")) > request.limits.max_path_bytes:
                raise RemoteGitDeliveryAdmissionError("Source path exceeds delivery bounds.")
            if not entry.present or entry.kind != "file" or entry.content_sha256 is None:
                raise RemoteGitDeliveryAdmissionError(
                    "Remote Git v1 supports regular source files only."
                )
            read = await source_workspace.read_bytes(
                path,
                max_bytes=request.limits.max_file_bytes,
            )
            if (
                type(read) is not WorkspaceReadResult
                or read.truncated
                or len(read.content) > request.limits.max_file_bytes
                or read.git_mode != entry.worktree_mode
                or entry.worktree_mode not in {"100644", "100755"}
            ):
                raise RemoteGitDeliveryAdmissionError("Source file exceeds delivery bounds.")
            digest = sha256(read.content).hexdigest()
            if digest != entry.content_sha256:
                raise RemoteGitDeliveryAdmissionError("Source file changed during materialization.")
            total += len(read.content)
            if total > request.limits.max_total_file_bytes:
                raise RemoteGitDeliveryAdmissionError("Source tree exceeds delivery bounds.")
            destination = (root / Path(*PurePosixPath(path).parts)).resolve()
            try:
                destination.relative_to(root)
            except ValueError:
                raise RemoteGitDeliveryAdmissionError(
                    "Source path escaped the broker repository."
                ) from None
            destination.parent.mkdir(parents=True, exist_ok=True)
            cursor = destination.parent
            while cursor != root:
                if cursor.is_symlink():
                    raise RemoteGitDeliveryAdmissionError("Source path crosses a symbolic link.")
                cursor = cursor.parent
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                remaining = memoryview(read.content)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("Source materialization made no progress.")
                    remaining = remaining[written:]
                os.fchmod(descriptor, 0o700 if entry.worktree_mode == "100755" else 0o600)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            manifest.append({"path": path, "sha256": digest, "bytes": len(read.content)})
        return (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(manifest, "remote_git_source_manifest")
            ).hexdigest()
        )

    @staticmethod
    def _changed_paths(output: str) -> tuple[str, ...]:
        fields = output.split("\0")
        if fields and fields[-1] == "":
            fields.pop()
        paths: set[str] = set()
        index = 0
        while index < len(fields):
            status_value = fields[index]
            index += 1
            if not status_value:
                raise RemoteGitDeliveryError("Git changed-path evidence is malformed.")
            if status_value[0] in {"R", "C"}:
                if index + 1 >= len(fields):
                    raise RemoteGitDeliveryError("Git rename evidence is incomplete.")
                paths.add(_source_path(fields[index]))
                paths.add(_source_path(fields[index + 1]))
                index += 2
            else:
                if index >= len(fields):
                    raise RemoteGitDeliveryError("Git changed-path evidence is incomplete.")
                paths.add(_source_path(fields[index]))
                index += 1
        return tuple(sorted(paths))

    async def _verify_staged_blobs(
        self,
        request: RemoteGitDeliveryRequest,
        observation: WorkspaceRevisionObservation,
        *,
        root: Path,
    ) -> tuple[RemoteGitStepEvidence, RemoteGitStepEvidence]:
        format_result, format_evidence = await self._git(
            request,
            "observe_object_format",
            ("rev-parse", "--show-object-format"),
            cwd=root,
        )
        _require_git_success(format_result, "observe Git object format")
        algorithm = format_result.stdout.strip()
        if algorithm not in {"sha1", "sha256"}:
            raise RemoteGitDeliveryAdmissionError("Git object format is unsupported.")
        index_result, index_evidence = await self._git(
            request,
            "verify_staged_blobs",
            ("ls-files", "--stage", "-z"),
            cwd=root,
        )
        _require_git_success(index_result, "verify staged Git blobs")
        observed_entries = {entry.path: entry for entry in observation.paths}
        observed_paths = set(observed_entries)
        indexed_paths: set[str] = set()
        for record in index_result.stdout.split("\0"):
            if not record:
                continue
            metadata, separator, raw_path = record.partition("\t")
            fields = metadata.split(" ")
            if not separator or len(fields) != 3:
                raise RemoteGitDeliveryAdmissionError("Git index evidence is malformed.")
            mode, object_id, stage = fields
            path = _source_path(raw_path)
            if mode not in {"100644", "100755"} or stage != "0":
                raise RemoteGitDeliveryAdmissionError(
                    "Remote Git v1 rejects links, submodules, and unresolved index stages."
                )
            entry = observed_entries.get(path)
            if entry is None or mode != entry.worktree_mode:
                raise RemoteGitDeliveryAdmissionError(
                    "Git file mode conflicts with source authority."
                )
            content = (root / Path(*PurePosixPath(path).parts)).read_bytes()
            header = f"blob {len(content)}\0".encode("ascii")
            digest = hashlib_new(algorithm, header + content).hexdigest()
            if object_id != digest:
                raise RemoteGitDeliveryAdmissionError(
                    "Git attributes or filters changed staged source bytes."
                )
            indexed_paths.add(path)
        if indexed_paths != observed_paths:
            raise RemoteGitDeliveryAdmissionError(
                "Git index paths conflict with the final source manifest."
            )
        return format_evidence, index_evidence

    async def prepare(
        self,
        request: RemoteGitDeliveryRequest,
        publication: CodingProductPublication,
        *,
        source_workspace: Workspace,
    ) -> RemoteGitPreparedIntent:
        """Materialize and verify an exact tree without creating a commit or push."""

        if type(request) is not RemoteGitDeliveryRequest:
            raise TypeError("request must be RemoteGitDeliveryRequest.")
        try:
            with exclusive_delivery(
                self._control_root(), sha256(request.delivery_id.encode()).hexdigest()
            ):
                prepared = await self._prepare_owned(
                    request, publication, source_workspace=source_workspace
                )
                await self._validated_product(request, publication, source_workspace)
                return prepared
        except RemoteGitOwnershipUnavailable as error:
            raise RemoteGitDeliveryReconstructionRequiredError(str(error)) from None

    async def _prepare_owned(
        self,
        request: RemoteGitDeliveryRequest,
        publication: CodingProductPublication,
        *,
        source_workspace: Workspace,
    ) -> RemoteGitPreparedIntent:

        if type(request) is not RemoteGitDeliveryRequest:
            raise TypeError("request must be RemoteGitDeliveryRequest.")
        remote = self._remote(request)
        await self.repository.ensure_request(request)
        await self._ensure_configuration(request, remote)
        existing = await self.repository.load_prepared(request)
        if existing is not None:
            return existing
        recorded = await self._validated_product(request, publication, source_workspace)
        receipts = list(await self.repository.load_lifecycle(request))
        if receipts and receipts[-1].state not in {RemoteGitDeliveryState.PREPARING}:
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Prior Git delivery evidence cannot be replayed as preparation."
            )
        await self._append_state(request, receipts, RemoteGitDeliveryState.PREPARING)
        root = self._delivery_root(request)
        if root.is_symlink():
            raise RemoteGitDeliveryAdmissionError("Remote Git delivery root is unsafe.")
        if root.exists():
            if not root.is_dir():
                raise RemoteGitDeliveryAdmissionError("Remote Git delivery root is unsafe.")
            if not await self._cleanup_delivery_root(request, root):
                raise RemoteGitDeliveryReconstructionRequiredError(
                    "Prior preparation cleanup did not settle."
                )
        root.mkdir(mode=0o700, parents=True)
        _secure_owned_directory(root, "delivery root")
        steps: list[RemoteGitStepEvidence] = []
        refs, evidence = await self._observe_remote(request, remote, cwd=root)
        steps.append(evidence)
        observed_base = refs.get(request.repository.base_ref)
        observed_destination = refs.get(request.repository.destination_ref)
        if observed_base != request.repository.expected_base_commit:
            raise RemoteGitDeliveryConflictError(
                "remote_base_changed_before_preparation",
                observed_base=observed_base,
                observed_destination=observed_destination,
            )
        observed_base = request.repository.expected_base_commit
        if observed_destination is not None:
            raise RemoteGitDeliveryConflictError(
                "destination_ref_already_exists",
                observed_base=observed_base,
                observed_destination=observed_destination,
            )
        result, evidence = await self._git(
            request,
            "initialize_repository",
            (
                "init",
                "--quiet",
                "--object-format="
                + ("sha256" if len(request.repository.expected_base_commit) == 64 else "sha1"),
            ),
            cwd=root,
        )
        steps.append(evidence)
        _require_git_success(result, "initialize broker repository")
        result, evidence = await self._git(
            request,
            "fetch_base",
            (
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-recurse-submodules",
                "--depth=1",
                remote.url,
                request.repository.base_ref,
            ),
            cwd=root,
            remote=remote,
        )
        steps.append(evidence)
        _require_git_success(result, "fetch admitted base")
        result, evidence = await self._git(
            request,
            "verify_fetched_base",
            ("rev-parse", "--verify", "FETCH_HEAD"),
            cwd=root,
        )
        steps.append(evidence)
        _require_git_success(result, "verify fetched base")
        if result.stdout.strip() != request.repository.expected_base_commit:
            raise RemoteGitDeliveryAdmissionError("Fetched base conflicts with remote observation.")
        result, evidence = await self._git(
            request,
            "initialize_empty_index",
            ("read-tree", "--empty"),
            cwd=root,
        )
        steps.append(evidence)
        _require_git_success(result, "initialize empty index")
        manifest_digest = await self._materialize_source(
            request,
            source_workspace,
            recorded,
            root,
        )
        result, evidence = await self._git(
            request,
            "stage_exact_source",
            (
                "--literal-pathspecs",
                "add",
                "--force",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
            ),
            cwd=root,
            stdin="".join(entry.path + "\0" for entry in recorded.paths),
        )
        steps.append(evidence)
        _require_git_success(result, "stage exact source")
        format_evidence, index_evidence = await self._verify_staged_blobs(
            request,
            recorded,
            root=root,
        )
        steps.extend((format_evidence, index_evidence))
        result, evidence = await self._git(
            request,
            "write_tree",
            ("write-tree",),
            cwd=root,
        )
        steps.append(evidence)
        _require_git_success(result, "write exact tree")
        tree = _object_id(result.stdout.strip(), "prepared tree")
        # Independently reconstruct the reviewed delta on its exact parent.
        # Equal filename sets alone cannot authenticate the delivered contents.
        assert publication.candidate.git is not None
        retained_diff = await self.coding_repository.store.read_bytes(
            request.source.diff_artifact_id,
            max_bytes=max(1, publication.candidate.git.artifact.size_bytes),
        )
        if (
            retained_diff.truncated
            or "sha256:" + sha256(retained_diff.content).hexdigest() != request.source.diff_sha256
        ):
            raise RemoteGitDeliveryAdmissionError("Retained reviewed delta changed before staging.")
        candidate = publication.candidate
        no_change = (
            retained_diff.content == b"No textual diff for the selected changes."
            and candidate.git_status is not None
            and not candidate.git_status.entries
            and candidate.git_summary is not None
            and not candidate.git_summary.entries
            and candidate.git is not None
            and not candidate.git.entries
            and candidate.git.entry_count == 0
        )
        for operation, arguments, input_text in (
            ("load_reviewed_parent", ("read-tree", request.repository.expected_base_commit), None),
            (
                "apply_reviewed_delta",
                ("apply", "--cached", "--whitespace=nowarn", "-"),
                retained_diff.content.decode("utf-8"),
            ),
            ("verify_reviewed_tree", ("write-tree",), None),
        ):
            if operation == "apply_reviewed_delta" and no_change:
                # Empty production evidence is a diagnostic, not patch text.
                # Still write and compare the exact parent tree below.
                continue
            result, evidence = await self._git(
                request, operation, arguments, cwd=root, stdin=input_text
            )
            steps.append(evidence)
            _require_git_success(result, operation)
        if result.stdout.strip() != tree:
            raise RemoteGitDeliveryAdmissionError(
                "Prepared Git tree conflicts with the retained reviewed delta."
            )
        result, evidence = await self._git(
            request,
            "verify_changed_paths",
            (
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-z",
                "-r",
                "-M",
                request.repository.expected_base_commit,
                tree,
            ),
            cwd=root,
        )
        steps.append(evidence)
        _require_git_success(result, "verify changed paths")
        changed_paths = self._changed_paths(result.stdout)
        candidate = publication.candidate
        assert candidate.git_status is not None
        expected_paths = tuple(
            sorted(
                {
                    path
                    for entry in candidate.git_status.entries
                    for path in (entry.path, entry.renamed_from)
                    if path is not None
                }
            )
        )
        if changed_paths != expected_paths:
            raise RemoteGitDeliveryAdmissionError(
                "Prepared Git tree conflicts with retained patch path evidence."
            )
        result, evidence = await self._git(
            request,
            "bound_object_count",
            ("count-objects", "-v"),
            cwd=root,
        )
        steps.append(evidence)
        _require_git_success(result, "bound Git object count")
        object_count, storage_bytes = _git_object_usage(result.stdout)
        if object_count > request.limits.max_git_objects:
            raise RemoteGitDeliveryAdmissionError("Git object count exceeds delivery bounds.")
        if storage_bytes > request.limits.max_git_storage_bytes:
            raise RemoteGitDeliveryAdmissionError("Git storage exceeds delivery bounds.")
        prepared = RemoteGitPreparedIntent(
            delivery_id=request.delivery_id,
            request_fingerprint=request.fingerprint,
            repository_id=request.repository.repository_id,
            remote_identity=request.repository.remote_identity,
            observed_base_commit=observed_base,
            observed_destination_commit=None,
            tree=tree,
            changed_paths=changed_paths,
            source_revision=request.source.final_source_revision,
            source_manifest_sha256=manifest_digest,
            commit_message_sha256=request.commit.message_sha256,
            steps=tuple(steps),
        )
        await self.repository.ensure_prepared(request, prepared)
        await self._append_state(
            request,
            receipts,
            RemoteGitDeliveryState.PREPARED,
            evidence_sha256=prepared.fingerprint,
            tree=tree,
        )
        return prepared

    def _result(
        self,
        request: RemoteGitDeliveryRequest,
        state: RemoteGitDeliveryState,
        *,
        prepared: RemoteGitPreparedIntent | None,
        steps: Sequence[RemoteGitStepEvidence],
        approval: RemoteGitDeliveryApproval | None = None,
        commit: str | None = None,
        destination_after: str | None = None,
        cleanup_settled: bool = False,
        reason_code: str | None = None,
        observed_base_override: str | None | _BaseObservationDefault = (
            _BaseObservationDefault.PREPARED
        ),
        destination_before_override: str | None = None,
    ) -> RemoteGitDeliveryResult:
        return RemoteGitDeliveryResult(
            delivery_id=request.delivery_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
            state=state,
            product_result_artifact_id=request.source.product_result_artifact_id,
            product_result_sha256=request.source.product_result_sha256,
            product_run_id=request.source.product_run_id,
            source_workspace_id=request.source.source_workspace_id,
            final_source_revision=request.source.final_source_revision,
            diff_artifact_id=request.source.diff_artifact_id,
            diff_sha256=request.source.diff_sha256,
            check_evidence_sha256=request.source.check_evidence_sha256,
            repository_id=request.repository.repository_id,
            broker_repository_id=request.repository.broker_repository_id,
            remote_identity=request.repository.remote_identity,
            base_ref=request.repository.base_ref,
            expected_base_commit=request.repository.expected_base_commit,
            observed_base_commit=(
                (None if prepared is None else prepared.observed_base_commit)
                if isinstance(observed_base_override, _BaseObservationDefault)
                else observed_base_override
            ),
            destination_ref=request.repository.destination_ref,
            destination_before=(
                destination_before_override
                if prepared is None
                else prepared.observed_destination_commit
            ),
            destination_after=destination_after,
            local_commit=commit,
            parent_commit=request.repository.expected_base_commit if commit is not None else None,
            tree=None if prepared is None else prepared.tree,
            commit_message_sha256=request.commit.message_sha256,
            policy_fingerprint=request.security.policy_fingerprint,
            approval_policy_fingerprint=request.security.approval_policy_fingerprint,
            redaction_profile_fingerprint=request.security.redaction_profile_fingerprint,
            approval_id=None if approval is None else approval.approval_id,
            approval_fingerprint=None if approval is None else approval.fingerprint,
            credential_profile_id=request.security.credential_profile_id,
            egress_profile_id=request.security.egress_profile_id,
            broker_behavior_fingerprint=request.security.broker_behavior_fingerprint,
            limits=request.limits,
            steps=tuple(steps),
            cleanup_settled=cleanup_settled,
            reason_code=reason_code,
            next_commit=commit if state is RemoteGitDeliveryState.PUSHED else None,
            next_ref=(
                request.repository.destination_ref
                if state is RemoteGitDeliveryState.PUSHED
                else None
            ),
        )

    async def _publish_state(
        self,
        request: RemoteGitDeliveryRequest,
        receipts: list[RemoteGitLifecycleReceipt],
        result: RemoteGitDeliveryResult,
        *,
        settle_cleanup: bool = True,
    ) -> RemoteGitDeliveryPublication:
        publication = await self.repository.publish_result(request, result)
        await self._append_state(
            request,
            receipts,
            result.state,
            evidence_sha256="sha256:" + result.digest,
            tree=result.tree,
            commit=result.local_commit,
            reason_code=result.reason_code,
        )
        if settle_cleanup:
            return await self._settle_terminal_cleanup(request, receipts, publication)
        return publication

    async def _settle_terminal_cleanup(
        self,
        request: RemoteGitDeliveryRequest,
        receipts: list[RemoteGitLifecycleReceipt],
        publication: RemoteGitDeliveryPublication,
    ) -> RemoteGitDeliveryPublication:
        # Persist the definitive outcome before deleting private recovery files.
        # An interrupted cleanup is retried from that same durable outcome, not
        # by restarting preparation or repeating a remote mutation.
        result = publication.result
        if (
            result.state
            not in {
                RemoteGitDeliveryState.DENIED,
                RemoteGitDeliveryState.CONFLICT,
                RemoteGitDeliveryState.FAILED,
            }
            or result.cleanup_settled
        ):
            return publication
        root = self._delivery_root(request)
        try:
            settled = not root.exists() and not root.is_symlink()
            if not settled:
                settled = await self._cleanup_delivery_root(request, root)
        except Exception:
            # The durable outcome remains cleanup-unsettled and owns retry.
            return publication
        if not settled:
            return publication
        return await self._publish_state(
            request,
            receipts,
            result.model_copy(update={"cleanup_settled": True}),
            settle_cleanup=False,
        )

    async def _settle_remote_commit(
        self,
        request: RemoteGitDeliveryRequest,
        receipts: list[RemoteGitLifecycleReceipt],
        *,
        prepared: RemoteGitPreparedIntent,
        steps: Sequence[RemoteGitStepEvidence],
        approval: RemoteGitDeliveryApproval,
        commit: str,
        root: Path,
        observed_base: str | None,
    ) -> RemoteGitDeliveryPublication:
        # Persist remote success before removing its local recovery material.
        # If either acknowledgement is lost, the retained commit plus a fresh
        # remote read permits cleanup and result publication to converge.
        pending = self._result(
            request,
            RemoteGitDeliveryState.PARTIAL,
            prepared=prepared,
            steps=steps,
            approval=approval,
            commit=commit,
            destination_after=commit,
            observed_base_override=observed_base,
            reason_code="cleanup_unsettled",
        )
        pending_publication = await self._publish_state(request, receipts, pending)
        cleanup = not root.exists() and not root.is_symlink()
        if not cleanup:
            cleanup = await self._cleanup_delivery_root(request, root)
        if not cleanup:
            return pending_publication
        return await self._publish_state(
            request,
            receipts,
            self._result(
                request,
                RemoteGitDeliveryState.PUSHED,
                prepared=prepared,
                steps=steps,
                approval=approval,
                commit=commit,
                destination_after=commit,
                observed_base_override=observed_base,
                cleanup_settled=True,
            ),
        )

    @staticmethod
    def _approval_valid(
        request: RemoteGitDeliveryRequest,
        prepared: RemoteGitPreparedIntent,
        approval: RemoteGitDeliveryApproval,
    ) -> bool:
        return (
            approval.request_fingerprint == request.fingerprint
            and approval.prepared_tree == prepared.tree
            and approval.policy_fingerprint == request.security.policy_fingerprint
            and approval.commit_approved
            and approval.push_approved
        )

    async def _commit_tree(
        self,
        request: RemoteGitDeliveryRequest,
        prepared: RemoteGitPreparedIntent,
        *,
        root: Path,
    ) -> tuple[str, RemoteGitStepEvidence]:
        result, evidence = await self._git(
            request,
            "create_commit",
            (
                "commit-tree",
                prepared.tree,
                "-p",
                request.repository.expected_base_commit,
            ),
            cwd=root,
            stdin=request.commit.message,
            commit=request.commit,
        )
        _require_git_success(result, "create exact commit")
        commit = _object_id(result.stdout.strip(), "local commit")
        if commit != self._expected_commit(request, prepared):
            raise RemoteGitDeliveryError("Created commit conflicts with exact approved authority.")
        result, verify = await self._git(
            request,
            "verify_commit",
            ("cat-file", "-p", commit),
            cwd=root,
        )
        _require_git_success(result, "verify exact commit")
        lines = result.stdout.splitlines()
        authored_at = datetime.fromisoformat(request.commit.authored_at.replace("Z", "+00:00"))
        timestamp = int(authored_at.timestamp())
        timezone = authored_at.strftime("%z")
        expected_author = (
            f"author {request.commit.author_name} <{request.commit.author_email}> "
            f"{timestamp} {timezone}"
        )
        expected_committer = (
            f"committer {request.commit.committer_name} <{request.commit.committer_email}> "
            f"{timestamp} {timezone}"
        )
        headers, separator, message = result.stdout.partition("\n\n")
        if (
            not lines
            or lines[0] != f"tree {prepared.tree}"
            or (f"parent {request.repository.expected_base_commit}" not in lines[:4])
        ):
            raise RemoteGitDeliveryError("Created commit tree or parent conflicts with authority.")
        header_lines = headers.splitlines()
        if expected_author not in header_lines or expected_committer not in header_lines:
            raise RemoteGitDeliveryError("Created commit identity conflicts with authority.")
        if not separator or message != request.commit.message:
            raise RemoteGitDeliveryError("Created commit message conflicts with authority.")
        return commit, RemoteGitStepEvidence(
            operation="create_and_verify_commit",
            argv_sha256="sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    [evidence.argv_sha256, verify.argv_sha256],
                    "remote_git_commit_steps",
                )
            ).hexdigest(),
            status=("passed" if verify.status == "passed" else verify.status),
            exit_code=verify.exit_code,
            duration_ms=evidence.duration_ms + verify.duration_ms,
            output_sha256="sha256:"
            + sha256(f"{evidence.output_sha256}\0{verify.output_sha256}".encode()).hexdigest(),
            output_truncated=evidence.output_truncated or verify.output_truncated,
        )

    @staticmethod
    def _expected_commit(
        request: RemoteGitDeliveryRequest, prepared: RemoteGitPreparedIntent
    ) -> str:
        instant = datetime.fromisoformat(request.commit.authored_at.replace("Z", "+00:00"))
        timestamp = int(instant.timestamp())
        timezone = instant.strftime("%z")
        material = (
            f"tree {prepared.tree}\nparent {request.repository.expected_base_commit}\n"
            f"author {request.commit.author_name} <{request.commit.author_email}> "
            f"{timestamp} {timezone}\n"
            f"committer {request.commit.committer_name} <{request.commit.committer_email}> "
            f"{timestamp} {timezone}\n\n{request.commit.message}"
        ).encode()
        algorithm = "sha256" if len(request.repository.expected_base_commit) == 64 else "sha1"
        return hashlib_new(algorithm, f"commit {len(material)}\0".encode() + material).hexdigest()

    async def _push(
        self,
        request: RemoteGitDeliveryRequest,
        remote: RemoteGitRemoteConfig,
        commit: str,
        *,
        root: Path,
    ) -> tuple[ExecResult, RemoteGitStepEvidence]:
        return await self._git(
            request,
            "push_exact_commit",
            (
                "push",
                "--porcelain",
                "--no-verify",
                f"--force-with-lease={request.repository.destination_ref}:",
                remote.url,
                f"{commit}:{request.repository.destination_ref}",
            ),
            cwd=root,
            remote=remote,
        )

    async def run(
        self,
        request: RemoteGitDeliveryRequest,
        publication: CodingProductPublication,
        *,
        source_workspace: Workspace,
        approval: RemoteGitDeliveryApproval | None = None,
    ) -> RemoteGitDeliveryPublication:
        """Settle an approved exact commit/ref or return a truthful non-success state."""

        if type(request) is not RemoteGitDeliveryRequest:
            raise TypeError("request must be RemoteGitDeliveryRequest.")
        try:
            with exclusive_delivery(
                self._control_root(), sha256(request.delivery_id.encode()).hexdigest()
            ):
                return await self._run_owned(
                    request, publication, source_workspace=source_workspace, approval=approval
                )
        except RemoteGitOwnershipUnavailable as error:
            raise RemoteGitDeliveryReconstructionRequiredError(str(error)) from None

    async def _run_owned(
        self,
        request: RemoteGitDeliveryRequest,
        publication: CodingProductPublication,
        *,
        source_workspace: Workspace,
        approval: RemoteGitDeliveryApproval | None,
    ) -> RemoteGitDeliveryPublication:

        if type(request) is not RemoteGitDeliveryRequest:
            raise TypeError("request must be RemoteGitDeliveryRequest.")
        remote = self._remote(request)
        await self.repository.ensure_request(request)
        await self._ensure_configuration(request, remote)
        receipts = list(await self.repository.load_lifecycle(request))
        if receipts:
            latest = receipts[-1]
            terminal = {
                RemoteGitDeliveryState.PUSHED,
                RemoteGitDeliveryState.CONFLICT,
                RemoteGitDeliveryState.DENIED,
                RemoteGitDeliveryState.FAILED,
                RemoteGitDeliveryState.CANCELLED,
                RemoteGitDeliveryState.RECONSTRUCTION_REQUIRED,
            }
            if latest.state in terminal and latest.evidence_sha256 is not None:
                result = await self.repository.load_result(request, latest.evidence_sha256)
                if result.result.local_commit is not None:
                    prepared = await self.repository.load_prepared(request)
                    if (
                        prepared is None
                        or result.result.tree != prepared.tree
                        or result.result.local_commit != self._expected_commit(request, prepared)
                    ):
                        raise RemoteGitDeliveryReconstructionRequiredError(
                            "Terminal commit conflicts with exact prepared authority."
                        )
                if approval is not None and (
                    type(approval) is not RemoteGitDeliveryApproval
                    or result.result.approval_fingerprint != approval.fingerprint
                ):
                    raise RemoteGitDeliveryAdmissionError("Terminal delivery approval conflicts.")
                result = await self._settle_terminal_cleanup(request, receipts, result)
                await self._validated_product(request, publication, source_workspace)
                if result.result.state is RemoteGitDeliveryState.PUSHED:
                    refs, _evidence = await self._observe_remote(
                        request, remote, cwd=self._control_root()
                    )
                    if refs.get(request.repository.destination_ref) != result.result.local_commit:
                        raise RemoteGitDeliveryConflictError(
                            "terminal_remote_destination_changed",
                            observed_base=refs.get(request.repository.base_ref),
                            observed_destination=refs.get(request.repository.destination_ref),
                        )
                return result
        try:
            prepared = await self._prepare_owned(
                request,
                publication,
                source_workspace=source_workspace,
            )
        except asyncio.CancelledError as cancellation:
            await self._record_cancellation(
                cancellation,
                request,
                RemoteGitDeliveryState.CANCELLED,
                reason_code="caller_cancelled_before_commit",
            )
            raise
        except RemoteGitDeliveryConflictError as conflict:
            receipts = list(await self.repository.load_lifecycle(request))
            result = self._result(
                request,
                RemoteGitDeliveryState.CONFLICT,
                prepared=None,
                steps=(),
                reason_code=conflict.reason_code,
                observed_base_override=conflict.observed_base,
                destination_before_override=conflict.observed_destination,
                destination_after=conflict.observed_destination,
            )
            return await self._publish_state(request, receipts, result)
        receipts = list(await self.repository.load_lifecycle(request))
        steps = list(prepared.steps)
        if approval is None:
            result = self._result(
                request,
                RemoteGitDeliveryState.APPROVAL_REQUIRED,
                prepared=prepared,
                steps=steps,
                reason_code="durable_delivery_approval_required",
            )
            return await self._publish_state(request, receipts, result)
        if type(approval) is not RemoteGitDeliveryApproval or not self._approval_valid(
            request,
            prepared,
            approval,
        ):
            result = self._result(
                request,
                RemoteGitDeliveryState.DENIED,
                prepared=prepared,
                steps=steps,
                approval=(approval if type(approval) is RemoteGitDeliveryApproval else None),
                reason_code="delivery_approval_conflict",
            )
            return await self._publish_state(request, receipts, result)
        await self.repository.ensure_approval(request, approval)
        try:
            await self._validated_product(request, publication, source_workspace)
        except RemoteGitDeliveryAdmissionError:
            result = self._result(
                request,
                RemoteGitDeliveryState.CONFLICT,
                prepared=prepared,
                steps=steps,
                approval=approval,
                reason_code="source_evidence_changed_before_delivery",
            )
            return await self._publish_state(request, receipts, result)
        root = self._delivery_root(request)
        refs, evidence = await self._observe_remote(request, remote, cwd=self._control_root())
        steps.append(evidence)
        observed_base = refs.get(request.repository.base_ref)
        destination = refs.get(request.repository.destination_ref)
        latest_commit = next(
            (receipt.commit for receipt in reversed(receipts) if receipt.commit is not None),
            None,
        )
        if latest_commit is not None and latest_commit != self._expected_commit(request, prepared):
            raise RemoteGitDeliveryReconstructionRequiredError(
                "Retained commit conflicts with exact prepared authority."
            )
        if destination is not None:
            if latest_commit is not None and destination == latest_commit:
                return await self._settle_remote_commit(
                    request,
                    receipts,
                    prepared=prepared,
                    steps=steps,
                    approval=approval,
                    commit=latest_commit,
                    root=root,
                    observed_base=observed_base,
                )
            result = self._result(
                request,
                RemoteGitDeliveryState.CONFLICT,
                prepared=prepared,
                steps=steps,
                approval=approval,
                destination_after=destination,
                reason_code="destination_ref_changed_before_delivery",
            )
            return await self._publish_state(request, receipts, result)
        if observed_base != request.repository.expected_base_commit:
            result = self._result(
                request,
                RemoteGitDeliveryState.CONFLICT,
                prepared=prepared,
                steps=steps,
                approval=approval,
                destination_after=destination,
                reason_code="remote_base_changed_before_delivery",
                observed_base_override=observed_base,
            )
            return await self._publish_state(request, receipts, result)
        if root.is_symlink() or not root.is_dir() or not (root / ".git").is_dir():
            result = self._result(
                request,
                RemoteGitDeliveryState.RECONSTRUCTION_REQUIRED,
                prepared=prepared,
                steps=steps,
                approval=approval,
                reason_code="broker_repository_unavailable",
            )
            return await self._publish_state(request, receipts, result)
        commit = latest_commit
        if commit is None:
            try:
                await self._append_state(
                    request,
                    receipts,
                    RemoteGitDeliveryState.COMMITTING,
                    tree=prepared.tree,
                )
                commit, evidence = await self._commit_tree(request, prepared, root=root)
                steps.append(evidence)
                await self._append_state(
                    request,
                    receipts,
                    RemoteGitDeliveryState.COMMITTED_LOCALLY,
                    tree=prepared.tree,
                    commit=commit,
                )
            except asyncio.CancelledError as cancellation:
                await self._record_cancellation(
                    cancellation,
                    request,
                    (
                        RemoteGitDeliveryState.COMMITTED_LOCALLY
                        if commit is not None
                        else RemoteGitDeliveryState.COMMITTING
                    ),
                    tree=prepared.tree,
                    commit=commit,
                    reason_code="caller_cancelled_during_commit_settlement",
                )
                raise
            except RemoteGitDeliveryError:
                result = self._result(
                    request,
                    RemoteGitDeliveryState.FAILED,
                    prepared=prepared,
                    steps=steps,
                    approval=approval,
                    reason_code="local_commit_failed",
                )
                return await self._publish_state(request, receipts, result)
        try:
            await self._validated_product(request, publication, source_workspace)
        except RemoteGitDeliveryAdmissionError:
            result = self._result(
                request,
                RemoteGitDeliveryState.CONFLICT,
                prepared=prepared,
                steps=steps,
                approval=approval,
                commit=commit,
                reason_code="source_evidence_changed_before_push",
            )
            return await self._publish_state(request, receipts, result)
        try:
            push_attempts = sum(
                receipt.state is RemoteGitDeliveryState.PUSHING for receipt in receipts
            )
            if push_attempts >= request.limits.retry_limit + 1:
                raise RemoteGitDeliveryReconstructionRequiredError(
                    "Remote Git push retry authority is exhausted."
                )
            if receipts and receipts[-1].state is RemoteGitDeliveryState.PUSHING:
                await self._append_state(
                    request,
                    receipts,
                    RemoteGitDeliveryState.AMBIGUOUS,
                    tree=prepared.tree,
                    commit=commit,
                    reason_code="prior_push_reconciled_before_retry",
                )
            await self._append_state(
                request,
                receipts,
                RemoteGitDeliveryState.PUSHING,
                tree=prepared.tree,
                commit=commit,
            )
            push_result, evidence = await self._push(request, remote, commit, root=root)
            steps.append(evidence)
        except asyncio.CancelledError as cancellation:
            await self._record_cancellation(
                cancellation,
                request,
                RemoteGitDeliveryState.AMBIGUOUS,
                tree=prepared.tree,
                commit=commit,
                reason_code="caller_cancelled",
            )
            raise
        except RemoteGitDeliveryReconstructionRequiredError:
            result = self._result(
                request,
                RemoteGitDeliveryState.RECONSTRUCTION_REQUIRED,
                prepared=prepared,
                steps=steps,
                approval=approval,
                commit=commit,
                reason_code="push_retry_authority_exhausted",
            )
            return await self._publish_state(request, receipts, result)
        except RemoteGitDeliveryError:
            refs, evidence = await self._observe_remote(request, remote, cwd=root)
            steps.append(evidence)
            destination = refs.get(request.repository.destination_ref)
            remote_settled = commit is not None and destination == commit
            if remote_settled:
                return await self._settle_remote_commit(
                    request,
                    receipts,
                    prepared=prepared,
                    steps=steps,
                    approval=approval,
                    commit=commit,
                    root=root,
                    observed_base=refs.get(request.repository.base_ref),
                )
            state = (
                RemoteGitDeliveryState.CONFLICT
                if destination is not None
                else RemoteGitDeliveryState.AMBIGUOUS
            )
            result = self._result(
                request,
                state,
                prepared=prepared,
                steps=steps,
                approval=approval,
                commit=commit,
                destination_after=destination,
                cleanup_settled=False,
                reason_code=(
                    None
                    if state is RemoteGitDeliveryState.PUSHED
                    else "cleanup_unsettled"
                    if state is RemoteGitDeliveryState.PARTIAL
                    else "push_unsettled"
                ),
            )
            return await self._publish_state(request, receipts, result)
        refs, evidence = await self._observe_remote(request, remote, cwd=root)
        steps.append(evidence)
        destination = refs.get(request.repository.destination_ref)
        if destination == commit:
            return await self._settle_remote_commit(
                request,
                receipts,
                prepared=prepared,
                steps=steps,
                approval=approval,
                commit=commit,
                root=root,
                observed_base=refs.get(request.repository.base_ref),
            )
        state = (
            RemoteGitDeliveryState.CONFLICT
            if destination is not None
            else RemoteGitDeliveryState.FAILED
            if push_result.exit_code != 0 and not push_result.timed_out
            else RemoteGitDeliveryState.AMBIGUOUS
        )
        result = self._result(
            request,
            state,
            prepared=prepared,
            steps=steps,
            approval=approval,
            commit=commit,
            destination_after=destination,
            reason_code=(
                "push_rejected"
                if state is RemoteGitDeliveryState.FAILED
                else "remote_destination_conflict"
                if state is RemoteGitDeliveryState.CONFLICT
                else "push_acknowledgement_ambiguous"
            ),
        )
        return await self._publish_state(request, receipts, result)


def _require_git_success(result: ExecResult, operation: str) -> None:
    if result.timed_out:
        raise RemoteGitDeliveryError(f"Git operation timed out: {operation}")
    if result.cancelled:
        raise RemoteGitDeliveryError(f"Git operation was cancelled: {operation}")
    if result.stdout_truncated or result.stderr_truncated:
        raise RemoteGitDeliveryError(f"Git operation output was truncated: {operation}")
    if result.exit_code != 0:
        raise RemoteGitDeliveryError(f"Git operation failed: {operation}")


def _git_object_usage(output: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in output.splitlines():
        key, separator, raw = line.partition(": ")
        if separator and raw.isdigit():
            values[key] = int(raw)
    if not {"count", "in-pack", "size", "size-pack"}.issubset(values):
        raise RemoteGitDeliveryError("Git object-count evidence is incomplete.")
    return (
        values["count"] + values["in-pack"],
        (values["size"] + values["size-pack"]) * 1024,
    )


def remote_git_broker_behavior_fingerprint() -> str:
    """Return the package behavior identity bound by delivery requests."""

    material = {
        "schema": REMOTE_GIT_DELIVERY_SCHEMA_VERSION,
        "implementation": "host-broker-v2",
        "transport": "structured-git-posix-process-limits-v2",
        "destination": "new-branch-only-with-empty-lease-v1",
        "credentials": "vault-backed-askpass-v1",
        "recovery": "owned-remote-observation-before-cleanup-v2",
        "configuration": "sealed-host-configuration-v1",
    }
    return (
        "sha256:"
        + sha256(canonical_durable_json_bytes(material, "remote_git_broker_behavior")).hexdigest()
    )


__all__ = [
    "REMOTE_GIT_DELIVERY_RESULT_KIND",
    "REMOTE_GIT_DELIVERY_SCHEMA_VERSION",
    "RemoteGitBrokerProfile",
    "RemoteGitCommitAuthority",
    "RemoteGitDeliveryAdmissionError",
    "RemoteGitDeliveryApproval",
    "RemoteGitDeliveryBroker",
    "RemoteGitDeliveryConflictError",
    "RemoteGitDeliveryError",
    "RemoteGitDeliveryLimits",
    "RemoteGitDeliveryPublication",
    "RemoteGitDeliveryReconstructionRequiredError",
    "RemoteGitDeliveryRepository",
    "RemoteGitDeliveryRequest",
    "RemoteGitDeliveryResult",
    "RemoteGitDeliveryState",
    "RemoteGitHttpCredentials",
    "RemoteGitLifecycleReceipt",
    "RemoteGitPreparedIntent",
    "RemoteGitRemoteConfig",
    "RemoteGitRepositoryAuthority",
    "RemoteGitSecurityAuthority",
    "RemoteGitSourceAuthority",
    "RemoteGitStepEvidence",
    "approve_remote_git_delivery",
    "remote_git_broker_behavior_fingerprint",
    "remote_git_delivery_request",
]
