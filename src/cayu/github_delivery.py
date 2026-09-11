"""Approved GitHub pull-request, check, and review delivery."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import quote, urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._exception_groups import exception_cause, exception_context, exception_suppresses_context
from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.artifacts import (
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    copy_artifact_read_result,
)
from cayu.artifacts.settlement import ArtifactWriteSettlementObserver
from cayu.coding_products import (
    CodingArtifactReference,
    CodingProductArtifactRepository,
    CodingProductPublication,
    CodingProductReconstructionRequiredError,
    CodingProductState,
)
from cayu.remote_git_delivery import (
    RemoteGitDeliveryPublication,
    RemoteGitDeliveryResult,
    RemoteGitDeliveryState,
)
from cayu.vaults import SecretRedactor, SecretRef, SecretResolver, validate_secret_resolver

GITHUB_DELIVERY_SCHEMA_VERSION = "cayu.github_delivery.v1"
GITHUB_DELIVERY_RESULT_KIND = "github_pull_request_delivery_result"
GITHUB_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
GITHUB_MAX_RECEIPTS = 256

_SHA256_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_REF_RE = re.compile(r"refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]{0,240}\Z")
_LOGIN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_REPOSITORY_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
_CHECK_STATUSES = frozenset(
    {"queued", "in_progress", "completed", "waiting", "requested", "pending"}
)
_CHECK_CONCLUSIONS = frozenset(
    {
        "action_required",
        "cancelled",
        "failure",
        "neutral",
        "success",
        "skipped",
        "stale",
        "timed_out",
        "startup_failure",
    }
)


class _GitHubOwnerStopped(Exception):
    """The public waiter left; no further provider effects may start."""


class GitHubDeliveryError(RuntimeError):
    """Base error for GitHub delivery."""


class GitHubDeliveryAdmissionError(GitHubDeliveryError):
    """Immutable provider or delivery authority failed admission."""


class GitHubDeliveryReconstructionRequiredError(GitHubDeliveryError):
    """Durable GitHub evidence cannot be reconstructed authoritatively."""


class GitHubProviderError(GitHubDeliveryError):
    """Safe classified provider failure without response content."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        ambiguous: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = _identifier(code, "code")
        self.status_code = status_code
        self.retryable = retryable
        self.ambiguous = ambiguous
        self.request_id = (
            None if request_id is None else _identifier(request_id, "request_id", maximum=1024)
        )


def _safe_github_failure(error: BaseException, *, ambiguous: bool = False) -> BaseException:
    """Copy a bounded, content-free failure graph without extension formatting."""
    seen: set[int] = set()
    codes = {
        "rate_limited",
        "permission_denied",
        "not_found",
        "provider_unavailable",
        "provider_timeout",
        "provider_transport_failure",
        "provider_rejected",
        "provider_response_truncated",
        "provider_redirect_forbidden",
        "malformed_provider_json",
        "malformed_provider_pagination",
        "malformed_provider_ref",
        "malformed_pull_request",
        "malformed_pull_request_list",
        "malformed_check_runs",
        "malformed_commit_statuses",
        "malformed_review_feedback",
        "mark_ready_rejected",
        "pull_request_search_truncated",
        "pull_request_binding_changed",
        "pull_request_identity_mismatch",
        "pull_request_repository_mismatch",
        "provider_acknowledgement_ambiguous",
        "provider_extension_failure",
        "provider_failure_graph_bounded",
    }

    def copy(current: BaseException, depth: int = 0) -> BaseException:
        if id(current) in seen or len(seen) >= 64 or depth >= 16:
            return GitHubProviderError("provider_failure_graph_bounded", ambiguous=ambiguous)
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            children = BaseExceptionGroup.exceptions.__get__(current)
            result = BaseExceptionGroup(
                "GitHub dependency failures", [copy(child, depth + 1) for child in children[:64]]
            )
        elif type(current) in {
            asyncio.CancelledError,
            SystemExit,
            KeyboardInterrupt,
            GeneratorExit,
        }:
            result = type(current)()
        elif isinstance(current, httpx.TimeoutException):
            result = GitHubProviderError("provider_timeout", retryable=True, ambiguous=ambiguous)
        elif isinstance(current, httpx.TransportError):
            result = GitHubProviderError(
                "provider_transport_failure", retryable=True, ambiguous=ambiguous
            )
        elif type(current) is GitHubProviderError:
            code = (
                current.code
                if type(current.code) is str and current.code in codes
                else "provider_acknowledgement_ambiguous"
                if current.ambiguous is True
                else "provider_extension_failure"
            )
            result = GitHubProviderError(
                code,
                retryable=current.retryable is True,
                ambiguous=ambiguous or current.ambiguous is True,
            )
        elif type(current) in {
            GitHubDeliveryAdmissionError,
            GitHubDeliveryReconstructionRequiredError,
        }:
            result = type(current)("GitHub dependency authority could not be validated.")
        else:
            result = GitHubProviderError("provider_extension_failure", ambiguous=ambiguous)
        cause = exception_cause(current)
        if cause is None and not exception_suppresses_context(current):
            cause = exception_context(current)
        if cause is not None:
            result.__cause__ = copy(cause, depth + 1)
        return result

    safe = copy(error)

    def extract_signal(current):
        if type(current) in {asyncio.CancelledError, SystemExit, KeyboardInterrupt, GeneratorExit}:
            return current, None
        if isinstance(current, BaseExceptionGroup):
            for index, child in enumerate(current.exceptions):
                signal, remainder = extract_signal(child)
                if signal is not None:
                    others = list(current.exceptions[:index])
                    if remainder is not None:
                        others.append(remainder)
                    others.extend(current.exceptions[index + 1 :])
                    residual = (
                        BaseExceptionGroup("GitHub secondary failures", others) if others else None
                    )
                    return signal, residual
        return None, current

    signal, residual = extract_signal(safe)
    if signal is not None:
        if residual is not None:
            causes = [] if signal.__cause__ is None else [signal.__cause__]
            signal.__cause__ = BaseExceptionGroup(
                "GitHub signal cleanup evidence", [*causes, residual]
            )
        return signal
    return safe


@dataclass(frozen=True)
class _GitHubOwnedOutcome:
    value: Any = None
    failure: BaseException | None = None


class _GitHubTransportBoundary:
    def __init__(self, transport: GitHubConnectorTransport) -> None:
        self._transport = transport

    def __getattr__(self, name: str):
        if name not in {
            "observe_ref",
            "find_pull_requests",
            "get_pull_request",
            "create_pull_request",
            "update_pull_request",
            "set_labels",
            "request_reviewers",
            "mark_ready",
            "observe_checks",
            "observe_reviews",
        }:
            raise AttributeError(name)

        async def invoke(*args, **kwargs):
            failure = None
            try:
                return await getattr(self._transport, name)(*args, **kwargs)
            except BaseException as error:
                failure = _safe_github_failure(
                    error,
                    ambiguous=name
                    in {
                        "create_pull_request",
                        "update_pull_request",
                        "set_labels",
                        "request_reviewers",
                        "mark_ready",
                    }
                    and type(error) is not GitHubProviderError,
                )
            raise failure from failure.__cause__

        return invoke


def _identifier(value: str, field: str, *, maximum: int = 512) -> str:
    value = require_durable_clean_nonblank(value, field)
    if len(value.encode()) > maximum:
        raise ValueError(f"{field} exceeds {maximum} bytes.")
    return value


def _fingerprint(value: str, field: str) -> str:
    value = _identifier(value, field, maximum=80)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256 fingerprint.")
    return "sha256:" + value.removeprefix("sha256:")


def _object_id(value: str, field: str) -> str:
    value = _identifier(value, field, maximum=64)
    if _OBJECT_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full Git object ID.")
    return value


def _ref(value: str, field: str) -> str:
    value = _identifier(value, field, maximum=255)
    components = value.split("/")
    if (
        _REF_RE.fullmatch(value) is None
        or ".." in value
        or "@{" in value
        or "//" in value
        or value.endswith((".", "/", ".lock"))
        or any(component.startswith(".") or component.endswith(".lock") for component in components)
    ):
        raise ValueError(f"{field} must be one canonical branch ref.")
    return value


def _model_bytes(value: BaseModel, field: str) -> bytes:
    return canonical_durable_json_bytes(value.model_dump(mode="json", warnings=False), field)


def _model_fingerprint(value: BaseModel, field: str) -> str:
    return "sha256:" + sha256(_model_bytes(value, field)).hexdigest()


def _artifact_id(*parts: str) -> str:
    return "art_" + sha256("\0".join(parts).encode()).hexdigest()[:32]


def _bounded_text(value: str, field: str, maximum: int) -> str:
    if type(value) is not str or "\x00" in value or len(value.encode()) > maximum:
        raise ValueError(f"{field} must be bounded text without NUL bytes.")
    return value


def _timestamp(value: str, field: str) -> str:
    value = _identifier(value, field, maximum=64)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} must be an ISO-8601 timestamp.") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone.")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _truncate_provider_text(value: Any, maximum: int) -> tuple[str, bool]:
    text = str(value or "")
    encoded = text.encode()
    if len(encoded) <= maximum:
        return text, False
    return encoded[:maximum].decode(errors="ignore"), True


def _provider_integer_id(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise GitHubProviderError(f"malformed_{field}")
    return value


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)


class GitHubOperation(StrEnum):
    CREATE_PULL_REQUEST = "create_pull_request"
    UPDATE_PULL_REQUEST = "update_pull_request"
    SET_LABELS = "set_labels"
    REQUEST_REVIEWERS = "request_reviewers"
    MARK_READY = "mark_ready"


_OPERATIONS = frozenset(
    {
        GitHubOperation.CREATE_PULL_REQUEST,
        GitHubOperation.UPDATE_PULL_REQUEST,
        GitHubOperation.SET_LABELS,
        GitHubOperation.REQUEST_REVIEWERS,
        GitHubOperation.MARK_READY,
    }
)


class GitHubDeliveryState(StrEnum):
    APPROVAL_REQUIRED = "approval_required"
    DENIED = "denied"
    PR_CREATED = "pr_created"
    PR_UPDATED = "pr_updated"
    CHECKS_PENDING = "checks_pending"
    CHECKS_PASSED = "checks_passed"
    CHECKS_FAILED = "checks_failed"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    CLOSED = "closed"
    SUPERSEDED = "superseded"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PERMISSION_DENIED = "permission_denied"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    RECONSTRUCTION_REQUIRED = "reconstruction_required"


class GitHubCheckState(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    MISSING = "missing"
    SUPERSEDED = "superseded"
    PROVIDER_AMBIGUOUS = "provider_ambiguous"


class GitHubReviewState(StrEnum):
    NONE = "none"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"


class GitHubSourceAuthority(_FrozenModel):
    product_result_artifact_id: str
    product_result_sha256: str
    product_request_fingerprint: str
    product_run_id: str
    source_workspace_id: str
    source_revision: str
    diff_artifact_id: str
    diff_sha256: str
    remote_result_artifact_id: str
    remote_result_sha256: str
    remote_request_fingerprint: str
    remote_delivery_id: str

    @field_validator(
        "product_result_artifact_id",
        "product_run_id",
        "source_workspace_id",
        "diff_artifact_id",
        "remote_result_artifact_id",
        "remote_delivery_id",
    )
    @classmethod
    def identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "product_result_sha256",
        "product_request_fingerprint",
        "source_revision",
        "diff_sha256",
        "remote_result_sha256",
        "remote_request_fingerprint",
    )
    @classmethod
    def digest(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)


class GitHubRepositoryAuthority(_FrozenModel):
    repository_id: str
    repository_alias: str
    installation_id: str
    account_id: str
    base_ref: str
    expected_base_commit: str
    head_ref: str
    head_commit: str

    @field_validator("repository_id", "repository_alias", "installation_id", "account_id")
    @classmethod
    def identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("base_ref", "head_ref")
    @classmethod
    def branch_ref(cls, value: str, info) -> str:
        return _ref(value, info.field_name)

    @field_validator("expected_base_commit", "head_commit")
    @classmethod
    def commit(cls, value: str, info) -> str:
        return _object_id(value, info.field_name)


class GitHubPullRequestMetadata(_FrozenModel):
    title: str
    body: str = ""
    labels: tuple[str, ...] = ()
    reviewers: tuple[str, ...] = ()
    teams: tuple[str, ...] = ()
    draft: StrictBool = True

    @field_validator("title")
    @classmethod
    def title_text(cls, value: str) -> str:
        value = _identifier(value, "title", maximum=256)
        if "\n" in value or "\r" in value:
            raise ValueError("title must be one line.")
        return value

    @field_validator("body")
    @classmethod
    def body_text(cls, value: str) -> str:
        return _bounded_text(value, "body", 32 * 1024)

    @field_validator("labels", "teams")
    @classmethod
    def names(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        normalized = tuple(_identifier(item, info.field_name, maximum=100) for item in value)
        if len(normalized) > 100 or normalized != tuple(sorted(set(normalized))):
            raise ValueError(f"{info.field_name} must be a bounded canonical sorted set.")
        return normalized

    @field_validator("reviewers")
    @classmethod
    def logins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            len(value) > 100
            or value != tuple(sorted(set(value)))
            or len({item.casefold() for item in value}) != len(value)
            or any(_LOGIN_RE.fullmatch(item) is None for item in value)
        ):
            raise ValueError(
                "reviewers must be a bounded canonical case-insensitive GitHub-login set."
            )
        return value

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "github_pull_request_metadata")


class GitHubCheckPolicy(_FrozenModel):
    required_checks: tuple[str, ...]
    allow_neutral: StrictBool = False
    allow_skipped: StrictBool = False

    @field_validator("required_checks")
    @classmethod
    def checks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_identifier(item, "required_check", maximum=256) for item in value)
        if not normalized or len(normalized) > 100 or normalized != tuple(sorted(set(normalized))):
            raise ValueError("required_checks must be a bounded non-empty canonical set.")
        return normalized

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "github_check_policy")


class GitHubReviewPolicy(_FrozenModel):
    approval_required: StrictBool = False
    required_approvers: tuple[str, ...] = ()
    minimum_approvals: StrictInt = Field(default=1, ge=1, le=100)
    allow_follow_up: StrictBool = False
    max_follow_up_items: StrictInt = Field(default=8, ge=1, le=32)
    max_follow_up_iterations: StrictInt = Field(default=3, ge=1, le=20)

    @field_validator("required_approvers")
    @classmethod
    def approvers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            len(value) > 100
            or value != tuple(sorted(set(value)))
            or len({item.casefold() for item in value}) != len(value)
            or any(_LOGIN_RE.fullmatch(item) is None for item in value)
        ):
            raise ValueError(
                "required_approvers must be a bounded canonical case-insensitive login set."
            )
        return value

    @model_validator(mode="after")
    def coherent_approval_policy(self) -> GitHubReviewPolicy:
        if self.approval_required and not self.required_approvers:
            raise ValueError("Required review approval needs explicit approver authority.")
        if self.required_approvers and self.minimum_approvals > len(self.required_approvers):
            raise ValueError("minimum_approvals exceeds the explicit approver set.")
        return self

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "github_review_policy")


class GitHubDeliveryLimits(_FrozenModel):
    timeout_seconds: StrictInt = Field(default=30, ge=1, le=300)
    max_response_bytes: StrictInt = Field(default=1024 * 1024, ge=1024, le=4 * 1024 * 1024)
    max_polls: StrictInt = Field(default=40, ge=1, le=200)
    poll_interval_seconds: StrictInt = Field(default=15, ge=1, le=900)
    max_elapsed_seconds: StrictInt = Field(default=3600, ge=1, le=86400)
    max_checks: StrictInt = Field(default=100, ge=1, le=100)
    max_reviews: StrictInt = Field(default=100, ge=1, le=100)
    max_comments: StrictInt = Field(default=100, ge=1, le=100)
    max_feedback_bytes: StrictInt = Field(default=4096, ge=256, le=16 * 1024)


class GitHubSecurityAuthority(_FrozenModel):
    connector_id: str
    connector_behavior_fingerprint: str
    credential_profile_id: str
    egress_profile_id: str
    policy_fingerprint: str
    approval_policy_fingerprint: str
    redaction_profile_fingerprint: str
    allowed_operations: tuple[GitHubOperation, ...]

    @field_validator("connector_id", "credential_profile_id", "egress_profile_id")
    @classmethod
    def identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "connector_behavior_fingerprint",
        "policy_fingerprint",
        "approval_policy_fingerprint",
        "redaction_profile_fingerprint",
    )
    @classmethod
    def digest(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @field_validator("allowed_operations")
    @classmethod
    def operations(cls, value: tuple[GitHubOperation, ...]) -> tuple[GitHubOperation, ...]:
        if not value or value != tuple(sorted(set(value))) or not set(value) <= _OPERATIONS:
            raise ValueError("allowed_operations must be a non-empty canonical supported set.")
        return value


class GitHubPullRequestDeliveryRequest(_FrozenModel):
    schema_version: Literal["cayu.github_delivery.v1"] = GITHUB_DELIVERY_SCHEMA_VERSION
    connector_run_id: str
    session_id: str
    idempotency_key: str
    requested_at: str
    source: GitHubSourceAuthority
    repository: GitHubRepositoryAuthority
    mode: Literal["create", "update"]
    existing_pull_request_number: StrictInt | None = Field(default=None, ge=1)
    metadata: GitHubPullRequestMetadata
    checks: GitHubCheckPolicy
    reviews: GitHubReviewPolicy = Field(default_factory=GitHubReviewPolicy)
    security: GitHubSecurityAuthority
    limits: GitHubDeliveryLimits = Field(default_factory=GitHubDeliveryLimits)
    merge: Literal[False] = False

    @field_validator("connector_run_id", "session_id", "idempotency_key")
    @classmethod
    def identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("requested_at")
    @classmethod
    def timestamp(cls, value: str) -> str:
        return _timestamp(value, "requested_at")

    @model_validator(mode="after")
    def coherent_mode(self):
        required = (
            GitHubOperation.CREATE_PULL_REQUEST
            if self.mode == "create"
            else GitHubOperation.UPDATE_PULL_REQUEST
        )
        if (self.mode == "create") != (self.existing_pull_request_number is None):
            raise ValueError("create/update mode conflicts with pull-request identity.")
        if required not in self.security.allowed_operations:
            raise ValueError("mode requires its explicit provider operation.")
        if (
            self.metadata.labels
            and GitHubOperation.SET_LABELS not in self.security.allowed_operations
        ):
            raise ValueError("labels require set_labels authority.")
        if (
            self.metadata.reviewers or self.metadata.teams
        ) and GitHubOperation.REQUEST_REVIEWERS not in self.security.allowed_operations:
            raise ValueError("reviewers require request_reviewers authority.")
        if (
            self.mode == "update"
            and not self.metadata.draft
            and GitHubOperation.MARK_READY not in self.security.allowed_operations
        ):
            raise ValueError("ready updates require mark_ready authority.")
        return self

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "github_delivery_request")


class GitHubDeliveryApproval(_FrozenModel):
    approval_id: str
    request_fingerprint: str
    metadata_fingerprint: str
    policy_fingerprint: str
    approved_operations: tuple[GitHubOperation, ...]

    @field_validator("approval_id")
    @classmethod
    def identity(cls, value: str) -> str:
        return _identifier(value, "approval_id")

    @field_validator("request_fingerprint", "metadata_fingerprint", "policy_fingerprint")
    @classmethod
    def digest(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @field_validator("approved_operations")
    @classmethod
    def operations(cls, value: tuple[GitHubOperation, ...]) -> tuple[GitHubOperation, ...]:
        if not value or value != tuple(sorted(set(value))) or not set(value) <= _OPERATIONS:
            raise ValueError("approved_operations must be canonical and supported.")
        return value

    @property
    def fingerprint(self) -> str:
        return _model_fingerprint(self, "github_delivery_approval")


class GitHubPullRequestSnapshot(_FrozenModel):
    number: StrictInt = Field(ge=1)
    node_id: str
    url: str
    state: Literal["open", "closed"]
    draft: StrictBool
    merged: StrictBool = False
    mergeable: StrictBool | None = None
    mergeable_state: str | None = None
    base_ref: str
    base_commit: str
    head_ref: str
    head_commit: str
    title: str
    body: str = ""
    labels: tuple[str, ...] = ()

    @field_validator("node_id", "url", "mergeable_state")
    @classmethod
    def identity(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, info.field_name, maximum=2048)

    @field_validator("base_ref", "head_ref")
    @classmethod
    def branch(cls, value: str, info) -> str:
        return _ref(value, info.field_name)

    @field_validator("base_commit", "head_commit")
    @classmethod
    def commit(cls, value: str, info) -> str:
        return _object_id(value, info.field_name)

    @field_validator("title", "body")
    @classmethod
    def text(cls, value: str, info) -> str:
        return _bounded_text(value, info.field_name, 32 * 1024)

    @field_validator("labels")
    @classmethod
    def canonical_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_identifier(item, "label", maximum=100) for item in value)
        if len(normalized) > 100 or normalized != tuple(sorted(set(normalized))):
            raise ValueError("Pull-request labels must be a bounded canonical set.")
        return normalized

    @model_validator(mode="after")
    def coherent_state(self) -> GitHubPullRequestSnapshot:
        if self.merged and self.state != "closed":
            raise ValueError("A merged pull request must be closed.")
        return self


class GitHubCheckObservation(_FrozenModel):
    provider_id: str
    name: str
    head_commit: str
    status: str
    conclusion: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    url: str | None = None

    @field_validator(
        "provider_id", "name", "status", "conclusion", "started_at", "completed_at", "url"
    )
    @classmethod
    def identity(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, info.field_name, maximum=2048)

    @field_validator("head_commit")
    @classmethod
    def commit(cls, value: str) -> str:
        return _object_id(value, "head_commit")

    @field_validator("status")
    @classmethod
    def check_status(cls, value: str) -> str:
        if value not in _CHECK_STATUSES:
            raise ValueError("status is not a supported GitHub check status.")
        return value

    @field_validator("conclusion")
    @classmethod
    def check_conclusion(cls, value: str | None) -> str | None:
        if value is not None and value not in _CHECK_CONCLUSIONS:
            raise ValueError("conclusion is not a supported GitHub check conclusion.")
        return value

    @field_validator("started_at", "completed_at")
    @classmethod
    def timestamp(cls, value: str | None, info) -> str | None:
        return None if value is None else _timestamp(value, info.field_name)


class GitHubFeedbackObservation(_FrozenModel):
    provider_id: str
    kind: Literal["review", "review_comment", "issue_comment"]
    author_login: str
    author_type: str
    state: str
    created_at: str
    body: str
    head_commit: str | None = None
    url: str | None = None
    thread_id: str | None = None
    in_reply_to_id: str | None = None
    path: str | None = None
    line: StrictInt | None = Field(default=None, ge=1)
    resolved: StrictBool | None = None

    @field_validator("head_commit")
    @classmethod
    def commit(cls, value: str | None) -> str | None:
        return None if value is None else _object_id(value, "head_commit")

    @field_validator(
        "provider_id",
        "author_login",
        "author_type",
        "state",
        "created_at",
        "url",
        "thread_id",
        "in_reply_to_id",
        "path",
    )
    @classmethod
    def identity(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, info.field_name, maximum=2048)

    @field_validator("body")
    @classmethod
    def text(cls, value: str) -> str:
        return _bounded_text(value, "body", 16 * 1024)

    @field_validator("created_at")
    @classmethod
    def timestamp(cls, value: str) -> str:
        return _timestamp(value, "created_at")

    @model_validator(mode="after")
    def coherent_feedback_state(self) -> GitHubFeedbackObservation:
        if self.kind == "review" and self.state not in {
            "approved",
            "changes_requested",
            "commented",
            "dismissed",
            "pending",
        }:
            raise ValueError("Review evidence has an unsupported state.")
        return self


class GitHubCheckBundle(_FrozenModel):
    head_commit: str
    checks: tuple[GitHubCheckObservation, ...]
    truncated: StrictBool = False

    @field_validator("head_commit")
    @classmethod
    def commit(cls, value: str) -> str:
        return _object_id(value, "head_commit")


class GitHubReviewBundle(_FrozenModel):
    feedback: tuple[GitHubFeedbackObservation, ...]
    truncated: StrictBool = False


class GitHubOperationEvidence(_FrozenModel):
    operation: GitHubOperation
    status: Literal["reconciled", "succeeded", "failed", "ambiguous"]
    request_id: str
    provider_id: str | None = None

    @field_validator("request_id", "provider_id")
    @classmethod
    def identity(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, info.field_name, maximum=1024)


class GitHubFollowUpCodingInput(_FrozenModel):
    prior_connector_run_id: str
    prior_product_run_id: str
    prior_delivery_id: str
    head_commit: str
    iteration: StrictInt = Field(ge=1, le=100)
    product_run_id: str
    session_id: str
    task_id: str
    feedback_provider_ids: tuple[str, ...]
    messages: tuple[str, ...]

    @field_validator(
        "prior_connector_run_id",
        "prior_product_run_id",
        "prior_delivery_id",
        "product_run_id",
        "session_id",
        "task_id",
    )
    @classmethod
    def identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("head_commit")
    @classmethod
    def commit(cls, value: str) -> str:
        return _object_id(value, "head_commit")

    @field_validator("feedback_provider_ids")
    @classmethod
    def provider_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_identifier(item, "feedback_provider_id") for item in value)
        if not normalized or normalized != tuple(sorted(set(normalized))):
            raise ValueError("feedback_provider_ids must be a non-empty canonical set.")
        return normalized

    @field_validator("messages")
    @classmethod
    def bounded_messages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 32:
            raise ValueError("messages must be a bounded non-empty sequence.")
        return tuple(_bounded_text(item, "message", 20 * 1024) for item in value)

    @model_validator(mode="after")
    def matching_evidence(self):
        if len(self.feedback_provider_ids) != len(self.messages):
            raise ValueError("Each selected provider item requires one provenance message.")
        return self


class GitHubDeliveryResult(_FrozenModel):
    schema_version: Literal["cayu.github_delivery_result.v1"] = "cayu.github_delivery_result.v1"
    connector_run_id: str
    session_id: str
    idempotency_key: str
    request_fingerprint: str
    state: GitHubDeliveryState
    repository_id: str
    connector_id: str
    installation_id: str
    account_id: str
    credential_profile_id: str
    egress_profile_id: str
    policy_fingerprint: str
    connector_behavior_fingerprint: str
    approval_policy_fingerprint: str
    redaction_profile_fingerprint: str
    metadata_fingerprint: str
    check_policy_fingerprint: str
    review_policy_fingerprint: str
    approval_fingerprint: str | None = None
    approval_id: str | None = None
    product_result_artifact_id: str
    product_run_id: str
    source_workspace_id: str
    diff_artifact_id: str
    remote_result_artifact_id: str
    remote_delivery_id: str
    product_request_fingerprint: str
    remote_request_fingerprint: str
    product_result_sha256: str
    remote_result_sha256: str
    source_revision: str
    diff_sha256: str
    repository_alias: str
    base_ref: str
    base_commit: str
    head_ref: str
    head_commit: str
    pull_request: GitHubPullRequestSnapshot | None = None
    checks_state: GitHubCheckState | None = None
    review_state: GitHubReviewState | None = None
    checks: tuple[GitHubCheckObservation, ...] = ()
    feedback: tuple[GitHubFeedbackObservation, ...] = ()
    checks_truncated: StrictBool = False
    feedback_truncated: StrictBool = False
    required_checks: tuple[str, ...]
    allowed_operations: tuple[GitHubOperation, ...]
    review_approval_required: StrictBool = False
    required_approvers: tuple[str, ...] = ()
    minimum_approvals: StrictInt = Field(default=1, ge=1, le=100)
    follow_up_allowed: StrictBool = False
    max_follow_up_items: StrictInt = Field(default=8, ge=1, le=32)
    max_follow_up_iterations: StrictInt = Field(default=3, ge=1, le=20)
    limits: GitHubDeliveryLimits
    operations: tuple[GitHubOperationEvidence, ...] = ()
    poll_count: StrictInt = Field(default=0, ge=0, le=200)
    next_poll_after_seconds: StrictInt | None = Field(default=None, ge=1, le=900)
    next_poll_at: str | None = None
    reason_code: str | None = None

    @field_validator("next_poll_at")
    @classmethod
    def poll_timestamp(cls, value: str | None) -> str | None:
        return None if value is None else _timestamp(value, "next_poll_at")

    @field_validator(
        "connector_run_id",
        "session_id",
        "idempotency_key",
        "product_result_artifact_id",
        "product_run_id",
        "source_workspace_id",
        "diff_artifact_id",
        "remote_result_artifact_id",
        "remote_delivery_id",
        "repository_id",
        "connector_id",
        "repository_alias",
        "installation_id",
        "account_id",
        "credential_profile_id",
        "egress_profile_id",
        "reason_code",
        "approval_id",
    )
    @classmethod
    def identity(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, info.field_name, maximum=1024)

    @field_validator(
        "request_fingerprint",
        "policy_fingerprint",
        "connector_behavior_fingerprint",
        "approval_policy_fingerprint",
        "redaction_profile_fingerprint",
        "metadata_fingerprint",
        "check_policy_fingerprint",
        "review_policy_fingerprint",
        "approval_fingerprint",
        "product_request_fingerprint",
        "remote_request_fingerprint",
        "product_result_sha256",
        "remote_result_sha256",
        "source_revision",
        "diff_sha256",
    )
    @classmethod
    def digest(cls, value: str | None, info) -> str | None:
        return None if value is None else _fingerprint(value, info.field_name)

    @field_validator("base_ref", "head_ref")
    @classmethod
    def branch(cls, value: str, info) -> str:
        return _ref(value, info.field_name)

    @field_validator("base_commit", "head_commit")
    @classmethod
    def commit(cls, value: str, info) -> str:
        return _object_id(value, info.field_name)

    @field_validator("required_approvers")
    @classmethod
    def approvers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            len(value) > 100
            or value != tuple(sorted(set(value)))
            or len({item.casefold() for item in value}) != len(value)
            or any(_LOGIN_RE.fullmatch(item) is None for item in value)
        ):
            raise ValueError(
                "required_approvers must be a bounded canonical case-insensitive login set."
            )
        return value

    @field_validator("required_checks")
    @classmethod
    def checks_policy(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_identifier(item, "required_check", maximum=256) for item in value)
        if not normalized or len(normalized) > 100 or normalized != tuple(sorted(set(normalized))):
            raise ValueError("required_checks must be a bounded non-empty canonical set.")
        return normalized

    @field_validator("allowed_operations")
    @classmethod
    def operations_policy(cls, value: tuple[GitHubOperation, ...]) -> tuple[GitHubOperation, ...]:
        if not value or value != tuple(sorted(set(value))) or not set(value) <= _OPERATIONS:
            raise ValueError("allowed_operations must be a non-empty canonical supported set.")
        return value

    @model_validator(mode="after")
    def coherent_evidence(self) -> GitHubDeliveryResult:
        if (self.next_poll_after_seconds is None) != (self.next_poll_at is None):
            raise ValueError("Retry delay and durable next-poll timestamp must settle together.")
        if (self.approval_id is None) != (self.approval_fingerprint is None):
            raise ValueError("GitHub approval identity and fingerprint must settle together.")
        if self.poll_count > self.limits.max_polls:
            raise ValueError("GitHub result poll count exceeds admitted limits.")
        if len(self.checks) > self.limits.max_checks:
            raise ValueError("GitHub result checks exceed admitted limits.")
        if len(self.feedback) > self.limits.max_reviews + 2 * self.limits.max_comments:
            raise ValueError("GitHub result feedback exceeds admitted limits.")
        if len(self.operations) > GITHUB_MAX_RECEIPTS * len(_OPERATIONS):
            raise ValueError("GitHub result operation evidence exceeds its lifecycle bound.")
        if self.review_approval_required and not self.required_approvers:
            raise ValueError("Required review approval needs explicit approver authority.")
        if self.required_approvers and self.minimum_approvals > len(self.required_approvers):
            raise ValueError("minimum_approvals exceeds the explicit approver set.")
        if (
            self.pull_request is not None
            and self.state not in {GitHubDeliveryState.CONFLICT, GitHubDeliveryState.SUPERSEDED}
            and (
                self.pull_request.base_ref != self.base_ref
                or self.pull_request.base_commit != self.base_commit
                or self.pull_request.head_ref != self.head_ref
                or self.pull_request.head_commit != self.head_commit
            )
        ):
            raise ValueError("GitHub pull request conflicts with result authority.")
        if self.state is GitHubDeliveryState.CHECKS_PASSED and (
            self.checks_state is not GitHubCheckState.PASSED
        ):
            raise ValueError("checks_passed requires exact passed check evidence.")
        if self.state is GitHubDeliveryState.CHANGES_REQUESTED and (
            self.review_state is not GitHubReviewState.CHANGES_REQUESTED
        ):
            raise ValueError("changes_requested requires exact review evidence.")
        if self.state is GitHubDeliveryState.APPROVED and (
            self.review_state is not GitHubReviewState.APPROVED
        ):
            raise ValueError("approved requires exact review evidence.")
        if self.state is GitHubDeliveryState.PARTIAL and not (
            self.checks_truncated or self.feedback_truncated
        ):
            raise ValueError("partial requires explicit truncated provider evidence.")
        if self.state is GitHubDeliveryState.CLOSED and (
            self.pull_request is None or self.pull_request.state != "closed"
        ):
            raise ValueError("closed requires an observed closed pull request.")
        return self

    @property
    def digest_value(self) -> str:
        return sha256(_model_bytes(self, "github_delivery_result")).hexdigest()


class GitHubLifecycleReceipt(_FrozenModel):
    connector_run_id: str
    request_fingerprint: str
    ordinal: StrictInt = Field(ge=1, le=GITHUB_MAX_RECEIPTS)
    state: GitHubDeliveryState
    result_sha256: str
    poll_count: StrictInt = Field(ge=0, le=200)

    @field_validator("connector_run_id")
    @classmethod
    def identity(cls, value: str) -> str:
        return _identifier(value, "identity")

    @field_validator("request_fingerprint", "result_sha256")
    @classmethod
    def digest(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)


@dataclass(frozen=True)
class GitHubDeliveryPublication:
    result: GitHubDeliveryResult
    artifact: CodingArtifactReference


@dataclass(frozen=True)
class GitHubCredentials:
    credential_profile_id: str
    token: SecretRef
    resolver: SecretResolver

    def __post_init__(self) -> None:
        _identifier(self.credential_profile_id, "credential_profile_id")
        if type(self.token) is not SecretRef:
            raise TypeError("GitHub token must be a SecretRef.")
        validate_secret_resolver(self.resolver)


@dataclass(frozen=True)
class GitHubRepositoryConfig:
    alias: str
    repository_id: str
    installation_id: str
    account_id: str
    api_base_url: str
    owner: str
    name: str
    credential_profile_id: str
    egress_profile_id: str
    credentials: GitHubCredentials

    def __post_init__(self) -> None:
        for field, value in (
            ("alias", self.alias),
            ("repository_id", self.repository_id),
            ("installation_id", self.installation_id),
            ("account_id", self.account_id),
            ("owner", self.owner),
            ("name", self.name),
            ("credential_profile_id", self.credential_profile_id),
            ("egress_profile_id", self.egress_profile_id),
        ):
            _identifier(value, field)
        if type(self.credentials) is not GitHubCredentials:
            raise TypeError("credentials must be GitHubCredentials.")
        api_base_url = _identifier(self.api_base_url, "api_base_url", maximum=4096).rstrip("/")
        parsed = urlsplit(api_base_url)
        try:
            port = parsed.port
        except ValueError:
            raise ValueError("GitHub API base has an invalid port.") from None
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or ".." in parsed.path.split("/")
            or "//" in parsed.path
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError("GitHub API base must be one credential-free HTTPS origin/path.")
        object.__setattr__(self, "api_base_url", api_base_url)
        if self.credentials.credential_profile_id != self.credential_profile_id:
            raise ValueError("GitHub credential profile identity conflicts with credentials.")
        if _LOGIN_RE.fullmatch(self.owner) is None:
            raise ValueError("GitHub owner must be one canonical account login.")
        if self.name in {".", ".."} or _REPOSITORY_NAME_RE.fullmatch(self.name) is None:
            raise ValueError("GitHub repository name is not canonical.")


@dataclass(frozen=True)
class GitHubConnectorProfile:
    connector_id: str
    behavior_fingerprint: str
    repositories: Mapping[str, GitHubRepositoryConfig]

    def __post_init__(self) -> None:
        _identifier(self.connector_id, "connector_id")
        object.__setattr__(
            self,
            "behavior_fingerprint",
            _fingerprint(self.behavior_fingerprint, "behavior_fingerprint"),
        )
        copied = dict(self.repositories)
        if not copied or any(
            type(config) is not GitHubRepositoryConfig or alias != config.alias
            for alias, config in copied.items()
        ):
            raise ValueError("GitHub repositories must be a non-empty alias-keyed mapping.")
        object.__setattr__(self, "repositories", MappingProxyType(copied))


class _GitHubConfigurationBinding(_FrozenModel):
    request_fingerprint: str
    api_base_url: str
    owner: str
    name: str
    credential_reference_sha256: str


@runtime_checkable
class GitHubConnectorTransport(Protocol):
    async def observe_ref(
        self, config: GitHubRepositoryConfig, ref: str, limits: GitHubDeliveryLimits
    ) -> str | None: ...
    async def find_pull_requests(
        self,
        config: GitHubRepositoryConfig,
        *,
        base_ref: str,
        head_ref: str,
        limits: GitHubDeliveryLimits,
    ) -> tuple[GitHubPullRequestSnapshot, ...]: ...
    async def get_pull_request(
        self, config: GitHubRepositoryConfig, number: int, limits: GitHubDeliveryLimits
    ) -> GitHubPullRequestSnapshot: ...
    async def create_pull_request(
        self, config: GitHubRepositoryConfig, request: GitHubPullRequestDeliveryRequest
    ) -> tuple[GitHubPullRequestSnapshot, str]: ...
    async def update_pull_request(
        self, config: GitHubRepositoryConfig, request: GitHubPullRequestDeliveryRequest, number: int
    ) -> tuple[GitHubPullRequestSnapshot, str]: ...
    async def set_labels(
        self,
        config: GitHubRepositoryConfig,
        number: int,
        labels: tuple[str, ...],
        limits: GitHubDeliveryLimits,
    ) -> str: ...
    async def request_reviewers(
        self,
        config: GitHubRepositoryConfig,
        number: int,
        reviewers: tuple[str, ...],
        teams: tuple[str, ...],
        limits: GitHubDeliveryLimits,
    ) -> str: ...
    async def mark_ready(
        self, config: GitHubRepositoryConfig, number: int, limits: GitHubDeliveryLimits
    ) -> str: ...
    async def observe_checks(
        self, config: GitHubRepositoryConfig, head_commit: str, limits: GitHubDeliveryLimits
    ) -> GitHubCheckBundle: ...
    async def observe_reviews(
        self, config: GitHubRepositoryConfig, number: int, limits: GitHubDeliveryLimits
    ) -> GitHubReviewBundle: ...


@asynccontextmanager
async def _github_response(client, *, close_client: bool, method: str, url: str, **kwargs):
    """Keep the active operation signal authoritative across both close phases."""
    manager = None
    entered = False
    primary = None
    failures = []
    try:
        manager = client.stream(method, url, **kwargs)
        response = await manager.__aenter__()
        entered = True
        yield response
    except BaseException as error:
        primary = error
        failures.append(error)
    if entered and manager is not None:
        try:
            await manager.__aexit__(
                None if primary is None else type(primary),
                primary,
                None if primary is None else primary.__traceback__,
            )
        except BaseException as error:
            if all(error is not prior for prior in failures):
                failures.append(error)
    if close_client:
        try:
            await client.aclose()
        except BaseException as error:
            if all(error is not prior for prior in failures):
                failures.append(error)
    if failures:
        failure = _safe_github_failure(
            failures[0]
            if len(failures) == 1
            else BaseExceptionGroup("GitHub operation and cleanup failures", failures),
            ambiguous=method != "GET",
        )
        raise failure from failure.__cause__


class GitHubRestTransport:
    """Fixed-operation, bounded GitHub REST transport with vault-backed auth."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def _request(self, config, method, path, limits, **kwargs):
        failure = None
        try:
            async with asyncio.timeout(limits.timeout_seconds):
                return await self._request_impl(config, method, path, limits, **kwargs)
        except TimeoutError as error:
            failure = GitHubProviderError(
                "provider_timeout", retryable=True, ambiguous=method != "GET"
            )
            # asyncio.timeout translates the active cancellation. Its safe
            # cleanup evidence remains attached to that cancellation's cause.
            cancelled = exception_cause(error)
            if type(cancelled) is asyncio.CancelledError:
                failure.__cause__ = exception_cause(cancelled)
        except BaseException as error:
            failure = _safe_github_failure(
                error, ambiguous=method != "GET" and type(error) is not GitHubProviderError
            )
        raise failure from failure.__cause__

    async def _request_impl(
        self,
        config: GitHubRepositoryConfig,
        method: str,
        path: str,
        limits: GitHubDeliveryLimits,
        *,
        params: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
        absolute_url: str | None = None,
    ) -> tuple[Any, str, bool]:
        resolved = await config.credentials.resolver.resolve(
            config.credentials.token, scope={"repository_id": config.repository_id}
        )
        token = resolved.value.get_secret_value()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        client = self._client or httpx.AsyncClient(follow_redirects=False)
        url = absolute_url or config.api_base_url.rstrip("/") + path
        try:
            async with _github_response(
                client,
                close_client=self._client is None,
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=payload,
                timeout=limits.timeout_seconds,
                follow_redirects=False,
            ) as response:
                request_id = SecretRedactor(resolved).redact_text_bounded(
                    response.headers.get(
                        "x-github-request-id",
                        "github-request-unavailable",
                    ),
                    max_bytes=1024,
                )
                try:
                    request_id = _identifier(request_id, "request_id", maximum=1024)
                except ValueError:
                    request_id = "github-request-unavailable"
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(chunk) > limits.max_response_bytes - len(content):
                        raise GitHubProviderError(
                            "provider_response_truncated",
                            request_id=request_id,
                        )
                    content.extend(chunk)
                if 300 <= response.status_code < 400:
                    raise GitHubProviderError(
                        "provider_redirect_forbidden",
                        status_code=response.status_code,
                        request_id=request_id,
                    )
                if response.status_code in {401, 403}:
                    rate = (
                        response.headers.get("x-ratelimit-remaining") == "0"
                        or "retry-after" in response.headers
                    )
                    raise GitHubProviderError(
                        "rate_limited" if rate else "permission_denied",
                        status_code=response.status_code,
                        retryable=rate,
                        request_id=request_id,
                    )
                if response.status_code == 429:
                    raise GitHubProviderError(
                        "rate_limited",
                        status_code=429,
                        retryable=True,
                        request_id=request_id,
                    )
                if response.status_code == 404:
                    raise GitHubProviderError(
                        "not_found",
                        status_code=404,
                        request_id=request_id,
                    )
                if response.status_code >= 500:
                    raise GitHubProviderError(
                        "provider_unavailable",
                        status_code=response.status_code,
                        retryable=True,
                        ambiguous=method != "GET",
                        request_id=request_id,
                    )
                if response.status_code >= 400:
                    raise GitHubProviderError(
                        "provider_rejected",
                        status_code=response.status_code,
                        request_id=request_id,
                    )
                try:
                    parsed = json.loads(content) if content else None
                    parsed = SecretRedactor(resolved).redact_json_values(parsed)
                except (UnicodeDecodeError, ValueError, RecursionError):
                    raise GitHubProviderError(
                        "malformed_provider_json",
                        ambiguous=method != "GET",
                        request_id=request_id,
                    ) from None
                try:
                    has_next_page = "next" in response.links
                except (KeyError, TypeError, ValueError):
                    raise GitHubProviderError(
                        "malformed_provider_pagination",
                        ambiguous=method != "GET",
                        request_id=request_id,
                    ) from None
                return parsed, request_id, has_next_page
        except httpx.TimeoutException:
            raise GitHubProviderError(
                "provider_timeout", retryable=True, ambiguous=method != "GET"
            ) from None
        except httpx.TransportError:
            raise GitHubProviderError(
                "provider_transport_failure", retryable=True, ambiguous=method != "GET"
            ) from None

    @staticmethod
    def _pr(data: Mapping[str, Any], config: GitHubRepositoryConfig) -> GitHubPullRequestSnapshot:
        base, head = data.get("base"), data.get("head")
        if not isinstance(base, Mapping) or not isinstance(head, Mapping):
            raise GitHubProviderError("malformed_pull_request")
        expected_repository = f"{config.owner}/{config.name}".casefold()
        for branch in (base, head):
            repository = branch.get("repo")
            if (
                not isinstance(repository, Mapping)
                or type(repository.get("full_name")) is not str
                or repository["full_name"].casefold() != expected_repository
            ):
                raise GitHubProviderError("pull_request_repository_mismatch")
        number = data.get("number")
        state = data.get("state")
        draft = data.get("draft", False)
        merged = data.get("merged", False)
        mergeable = data.get("mergeable")
        node_id = data.get("node_id")
        html_url = data.get("html_url")
        title = data.get("title")
        body = data.get("body")
        if (
            type(number) is not int
            or number < 1
            or state not in {"open", "closed"}
            or type(draft) is not bool
            or type(merged) is not bool
            or (mergeable is not None and type(mergeable) is not bool)
            or type(node_id) is not str
            or type(html_url) is not str
            or type(title) is not str
            or (body is not None and type(body) is not str)
            or type(base.get("ref")) is not str
            or type(base.get("sha")) is not str
            or type(head.get("ref")) is not str
            or type(head.get("sha")) is not str
        ):
            raise GitHubProviderError("malformed_pull_request")
        labels = data.get("labels", [])
        if not isinstance(labels, list) or any(
            not isinstance(item, Mapping)
            or type(item.get("name")) is not str
            or not item.get("name")
            for item in labels
        ):
            raise GitHubProviderError("malformed_pull_request")
        try:
            return GitHubPullRequestSnapshot(
                number=number,
                node_id=node_id,
                url=html_url,
                state=state,
                draft=draft,
                merged=merged,
                mergeable=mergeable,
                mergeable_state=data.get("mergeable_state"),
                base_ref="refs/heads/" + base["ref"],
                base_commit=base["sha"],
                head_ref="refs/heads/" + head["ref"],
                head_commit=head["sha"],
                title=title,
                body=body or "",
                labels=tuple(sorted({item["name"] for item in labels})),
            )
        except (TypeError, ValueError):
            raise GitHubProviderError("malformed_pull_request") from None

    async def observe_ref(self, config, ref, limits):
        try:
            data, _, _ = await self._request(
                config,
                "GET",
                f"/repos/{quote(config.owner)}/{quote(config.name)}/git/ref/{quote(ref.removeprefix('refs/'), safe='/')}",
                limits,
            )
        except GitHubProviderError as exc:
            if exc.code == "not_found":
                return None
            raise
        obj = data.get("object") if isinstance(data, Mapping) else None
        if not isinstance(obj, Mapping):
            raise GitHubProviderError("malformed_provider_ref")
        try:
            return _object_id(str(obj.get("sha", "")), "provider ref")
        except ValueError:
            raise GitHubProviderError("malformed_provider_ref") from None

    async def find_pull_requests(self, config, *, base_ref, head_ref, limits):
        data, _, has_next_page = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls",
            limits,
            params={
                "state": "all",
                "base": base_ref.removeprefix("refs/heads/"),
                "head": f"{config.owner}:{head_ref.removeprefix('refs/heads/')}",
                "per_page": "100",
            },
        )
        if not isinstance(data, list):
            raise GitHubProviderError("malformed_pull_request_list")
        if has_next_page:
            raise GitHubProviderError("pull_request_search_truncated", ambiguous=True)
        if any(not isinstance(item, Mapping) for item in data):
            raise GitHubProviderError("malformed_pull_request_list")
        return tuple(self._pr(item, config) for item in data)

    async def get_pull_request(self, config, number, limits):
        data, _, _ = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls/{number}",
            limits,
        )
        if not isinstance(data, Mapping):
            raise GitHubProviderError("malformed_pull_request")
        result = self._pr(data, config)
        if result.number != number:
            raise GitHubProviderError("pull_request_identity_mismatch")
        return result

    async def create_pull_request(self, config, request):
        data, request_id, _ = await self._request(
            config,
            "POST",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls",
            request.limits,
            payload={
                "title": request.metadata.title,
                "body": request.metadata.body,
                "base": request.repository.base_ref.removeprefix("refs/heads/"),
                "head": request.repository.head_ref.removeprefix("refs/heads/"),
                "draft": request.metadata.draft,
            },
        )
        if not isinstance(data, Mapping):
            raise GitHubProviderError("malformed_pull_request", ambiguous=True)
        return self._pr(data, config), request_id

    async def update_pull_request(self, config, request, number):
        data, request_id, _ = await self._request(
            config,
            "PATCH",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls/{number}",
            request.limits,
            payload={"title": request.metadata.title, "body": request.metadata.body},
        )
        if not isinstance(data, Mapping):
            raise GitHubProviderError("malformed_pull_request", ambiguous=True)
        result = self._pr(data, config)
        if result.number != number:
            raise GitHubProviderError("pull_request_identity_mismatch", ambiguous=True)
        return result, request_id

    async def set_labels(self, config, number, labels, limits):
        _, request_id, _ = await self._request(
            config,
            "PUT",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/issues/{number}/labels",
            limits,
            payload={"labels": list(labels)},
        )
        return request_id

    async def request_reviewers(self, config, number, reviewers, teams, limits):
        _, request_id, _ = await self._request(
            config,
            "POST",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls/{number}/requested_reviewers",
            limits,
            payload={"reviewers": list(reviewers), "team_reviewers": list(teams)},
        )
        return request_id

    async def mark_ready(self, config, number, limits):
        current, _, _ = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls/{number}",
            limits,
        )
        if not isinstance(current, Mapping):
            raise GitHubProviderError("malformed_pull_request")
        pull_request = self._pr(current, config)
        if pull_request.number != number:
            raise GitHubProviderError("pull_request_identity_mismatch")
        parsed = urlsplit(config.api_base_url)
        graphql_path = (
            parsed.path.rstrip("/").removesuffix("/api/v3") + "/api/graphql"
            if parsed.path.rstrip("/").endswith("/api/v3")
            else parsed.path.rstrip("/") + "/graphql"
        )
        graphql_url = f"{parsed.scheme}://{parsed.netloc}{graphql_path}"
        data, request_id, _ = await self._request(
            config,
            "POST",
            "",
            limits,
            absolute_url=graphql_url,
            payload={
                "query": (
                    "mutation MarkPullRequestReadyForReview($pullRequestId: ID!) { "
                    "markPullRequestReadyForReview(input: {pullRequestId: $pullRequestId}) { "
                    "pullRequest { id isDraft } } }"
                ),
                "variables": {"pullRequestId": pull_request.node_id},
            },
        )
        mutation = data.get("data") if isinstance(data, Mapping) else None
        ready = (
            mutation.get("markPullRequestReadyForReview") if isinstance(mutation, Mapping) else None
        )
        observed = ready.get("pullRequest") if isinstance(ready, Mapping) else None
        if (
            not isinstance(observed, Mapping)
            or observed.get("id") != pull_request.node_id
            or observed.get("isDraft") is not False
        ):
            raise GitHubProviderError("mark_ready_rejected", ambiguous=True)
        return request_id

    async def observe_checks(self, config, head_commit, limits):
        data, _, checks_have_next_page = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/commits/{head_commit}/check-runs",
            limits,
            params={"filter": "latest", "per_page": str(limits.max_checks)},
        )
        runs = data.get("check_runs", []) if isinstance(data, Mapping) else []
        if not isinstance(runs, list) or any(not isinstance(item, Mapping) for item in runs):
            raise GitHubProviderError("malformed_check_runs")
        for item in runs:
            _provider_integer_id(item.get("id"), "check_runs")
        try:
            observations = tuple(
                GitHubCheckObservation(
                    provider_id=f"check-run:{item['id']}",
                    name=item.get("name"),
                    head_commit=item.get("head_sha"),
                    status=item.get("status"),
                    conclusion=item.get("conclusion"),
                    started_at=item.get("started_at"),
                    completed_at=item.get("completed_at"),
                    url=item.get("html_url"),
                )
                for item in runs[: limits.max_checks]
            )
        except (TypeError, ValueError):
            raise GitHubProviderError("malformed_check_runs") from None
        statuses, _, statuses_have_next_page = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/commits/{head_commit}/statuses",
            limits,
            params={"per_page": str(limits.max_checks)},
        )
        if not isinstance(statuses, list) or any(
            not isinstance(item, Mapping) for item in statuses
        ):
            raise GitHubProviderError("malformed_commit_statuses")
        legacy: list[GitHubCheckObservation] = []
        legacy_names: set[str] = set()
        try:
            for item in statuses[: limits.max_checks]:
                _provider_integer_id(item.get("id"), "commit_statuses")
                context, status = item.get("context"), item.get("state")
                if type(context) is not str or status not in {
                    "pending",
                    "success",
                    "failure",
                    "error",
                }:
                    raise GitHubProviderError("malformed_commit_statuses")
                if context in legacy_names:
                    continue
                legacy_names.add(context)
                legacy.append(
                    GitHubCheckObservation(
                        provider_id=f"commit-status:{item['id']}",
                        name=context,
                        head_commit=head_commit,
                        status=(
                            "completed"
                            if status in {"success", "failure", "error"}
                            else "in_progress"
                        ),
                        conclusion=(
                            "success"
                            if status == "success"
                            else "failure"
                            if status in {"failure", "error"}
                            else None
                        ),
                        started_at=item.get("created_at"),
                        completed_at=item.get("updated_at"),
                        url=item.get("target_url"),
                    )
                )
        except GitHubProviderError:
            raise
        except (TypeError, ValueError):
            raise GitHubProviderError("malformed_commit_statuses") from None
        combined = (*observations, *legacy)
        total_count = data.get("total_count", len(observations)) if isinstance(data, Mapping) else 0
        if type(total_count) is not int or total_count < len(observations):
            raise GitHubProviderError("malformed_check_runs")
        return GitHubCheckBundle(
            head_commit=head_commit,
            checks=combined[: limits.max_checks],
            truncated=bool(isinstance(data, Mapping) and total_count > len(observations))
            or len(combined) > limits.max_checks
            or checks_have_next_page
            or statuses_have_next_page,
        )

    async def observe_reviews(self, config, number, limits):
        reviews, _, reviews_have_next_page = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls/{number}/reviews",
            limits,
            params={"per_page": str(limits.max_reviews)},
        )
        comments, _, comments_have_next_page = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/pulls/{number}/comments",
            limits,
            params={"per_page": str(limits.max_comments)},
        )
        issue_comments, _, issue_comments_have_next_page = await self._request(
            config,
            "GET",
            f"/repos/{quote(config.owner)}/{quote(config.name)}/issues/{number}/comments",
            limits,
            params={"per_page": str(limits.max_comments)},
        )
        feedback: list[GitHubFeedbackObservation] = []
        truncated = False
        for kind, items, has_next_page in (
            ("review", reviews, reviews_have_next_page),
            ("review_comment", comments, comments_have_next_page),
            ("issue_comment", issue_comments, issue_comments_have_next_page),
        ):
            if not isinstance(items, list):
                raise GitHubProviderError("malformed_review_feedback")
            if any(not isinstance(item, Mapping) for item in items):
                raise GitHubProviderError("malformed_review_feedback")
            truncated = truncated or has_next_page
            for item in items:
                user = item.get("user") if isinstance(item, Mapping) else None
                _provider_integer_id(item.get("id"), "review_feedback")
                created_at = item.get("submitted_at") or item.get("created_at")
                state = item.get("state", "commented")
                if (
                    not isinstance(user, Mapping)
                    or type(user.get("login")) is not str
                    or type(user.get("type")) is not str
                    or type(created_at) is not str
                    or type(state) is not str
                ):
                    raise GitHubProviderError("malformed_review_feedback")
                body, body_truncated = _truncate_provider_text(item.get("body"), 16 * 1024)
                truncated = truncated or body_truncated
                try:
                    feedback.append(
                        GitHubFeedbackObservation(
                            provider_id=f"{kind}:{item['id']}",
                            kind=kind,
                            head_commit=item.get("commit_id"),
                            author_login=user["login"],
                            author_type=user["type"],
                            state=state.lower(),
                            created_at=created_at,
                            body=body,
                            url=item.get("html_url"),
                            thread_id=(
                                str(item.get("in_reply_to_id") or item["id"])
                                if kind == "review_comment"
                                else None
                            ),
                            in_reply_to_id=(
                                None
                                if item.get("in_reply_to_id") is None
                                else str(item.get("in_reply_to_id"))
                            ),
                            path=item.get("path"),
                            line=item.get("line") or item.get("original_line"),
                        )
                    )
                except (TypeError, ValueError):
                    raise GitHubProviderError("malformed_review_feedback") from None
        feedback.sort(key=lambda item: (item.created_at, item.provider_id))
        bound = limits.max_reviews + 2 * limits.max_comments
        return GitHubReviewBundle(
            feedback=tuple(feedback[:bound]),
            truncated=truncated or len(feedback) > bound,
        )


class GitHubDeliveryRepository:
    """Immutable requests, approvals, results, and ordered progress receipts."""

    def __init__(self, store: ArtifactStore) -> None:
        if not isinstance(store, ArtifactStore):
            raise TypeError("store must implement ArtifactStore.")
        self.store = store

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
                max_content_bytes=GITHUB_MAX_ARTIFACT_BYTES,
            )
        except (TypeError, ValueError) as exc:
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub artifact store returned inconsistent evidence."
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
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub artifact authority is inconsistent."
            )
        return copied

    @staticmethod
    def _validate_result_authority(
        request: GitHubPullRequestDeliveryRequest,
        result: GitHubDeliveryResult,
    ) -> None:
        if (
            result.connector_run_id != request.connector_run_id
            or result.session_id != request.session_id
            or result.idempotency_key != request.idempotency_key
            or result.request_fingerprint != request.fingerprint
            or result.product_result_artifact_id != request.source.product_result_artifact_id
            or result.product_run_id != request.source.product_run_id
            or result.source_workspace_id != request.source.source_workspace_id
            or result.diff_artifact_id != request.source.diff_artifact_id
            or result.remote_result_artifact_id != request.source.remote_result_artifact_id
            or result.remote_delivery_id != request.source.remote_delivery_id
            or result.product_request_fingerprint != request.source.product_request_fingerprint
            or result.remote_request_fingerprint != request.source.remote_request_fingerprint
            or result.product_result_sha256 != request.source.product_result_sha256
            or result.remote_result_sha256 != request.source.remote_result_sha256
            or result.source_revision != request.source.source_revision
            or result.diff_sha256 != request.source.diff_sha256
            or result.repository_id != request.repository.repository_id
            or result.connector_id != request.security.connector_id
            or result.repository_alias != request.repository.repository_alias
            or result.installation_id != request.repository.installation_id
            or result.account_id != request.repository.account_id
            or result.base_ref != request.repository.base_ref
            or result.base_commit != request.repository.expected_base_commit
            or result.head_ref != request.repository.head_ref
            or result.head_commit != request.repository.head_commit
            or result.credential_profile_id != request.security.credential_profile_id
            or result.egress_profile_id != request.security.egress_profile_id
            or result.policy_fingerprint != request.security.policy_fingerprint
            or result.connector_behavior_fingerprint
            != request.security.connector_behavior_fingerprint
            or result.approval_policy_fingerprint != request.security.approval_policy_fingerprint
            or result.redaction_profile_fingerprint
            != request.security.redaction_profile_fingerprint
            or result.metadata_fingerprint != request.metadata.fingerprint
            or result.check_policy_fingerprint != request.checks.fingerprint
            or result.review_policy_fingerprint != request.reviews.fingerprint
            or result.required_checks != request.checks.required_checks
            or result.allowed_operations != request.security.allowed_operations
            or result.review_approval_required != request.reviews.approval_required
            or result.required_approvers != request.reviews.required_approvers
            or result.minimum_approvals != request.reviews.minimum_approvals
            or result.follow_up_allowed != request.reviews.allow_follow_up
            or result.max_follow_up_items != request.reviews.max_follow_up_items
            or result.max_follow_up_iterations != request.reviews.max_follow_up_iterations
            or result.limits != request.limits
        ):
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub result conflicts with durable request authority."
            )

    async def _ensure(
        self, value: BaseModel, artifact_id: str, filename: str, session_id: str, field: str
    ) -> CodingArtifactReference:
        content = _model_bytes(value, field)
        if len(content) > GITHUB_MAX_ARTIFACT_BYTES:
            raise GitHubDeliveryAdmissionError("GitHub delivery artifact exceeds its bound.")
        try:
            existing = self._validate_session_json_artifact(
                await self.store.read_bytes(
                    artifact_id,
                    max_bytes=GITHUB_MAX_ARTIFACT_BYTES,
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
                    max_bytes=GITHUB_MAX_ARTIFACT_BYTES,
                ),
                artifact_id=artifact_id,
                session_id=session_id,
                filename=filename,
            )
        if existing.content != content:
            raise GitHubDeliveryAdmissionError(
                "Stable GitHub identity conflicts with durable authority."
            )
        return CodingArtifactReference(
            artifact_id=artifact_id,
            sha256="sha256:" + sha256(content).hexdigest(),
            size_bytes=existing.metadata.size_bytes,
            content_type=existing.metadata.content_type,
        )

    async def ensure_request(self, request: GitHubPullRequestDeliveryRequest):
        if type(request) is not GitHubPullRequestDeliveryRequest:
            raise TypeError("request must be GitHubPullRequestDeliveryRequest.")
        return await self._ensure(
            request,
            _artifact_id("github-request", request.connector_run_id),
            "github-delivery-request.json",
            request.session_id,
            "github_delivery_request",
        )

    async def ensure_approval(self, request, approval):
        if not self._approval_matches(request, approval):
            raise GitHubDeliveryAdmissionError("GitHub approval conflicts with request.")
        return await self._ensure(
            approval,
            _artifact_id("github-approval", request.connector_run_id),
            "github-delivery-approval.json",
            request.session_id,
            "github_delivery_approval",
        )

    async def approval(
        self, request: GitHubPullRequestDeliveryRequest
    ) -> GitHubDeliveryApproval | None:
        artifact_id = _artifact_id("github-approval", request.connector_run_id)
        try:
            stored = self._validate_session_json_artifact(
                await self.store.read_bytes(
                    artifact_id,
                    max_bytes=GITHUB_MAX_ARTIFACT_BYTES,
                ),
                artifact_id=artifact_id,
                session_id=request.session_id,
                filename="github-delivery-approval.json",
            )
        except FileNotFoundError:
            return None
        try:
            approval = GitHubDeliveryApproval.model_validate_json(stored.content)
        except Exception as exc:
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub approval cannot be reconstructed."
            ) from exc
        if not self._approval_matches(request, approval):
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub approval conflicts with durable request authority."
            )
        return approval

    @staticmethod
    def _approval_matches(
        request: GitHubPullRequestDeliveryRequest,
        approval: GitHubDeliveryApproval,
    ) -> bool:
        return (
            type(approval) is GitHubDeliveryApproval
            and approval.request_fingerprint == request.fingerprint
            and approval.metadata_fingerprint == request.metadata.fingerprint
            and approval.policy_fingerprint == request.security.policy_fingerprint
            and set(request.security.allowed_operations) <= set(approval.approved_operations)
        )

    async def receipts(self, request) -> tuple[GitHubLifecycleReceipt, ...]:
        found: list[GitHubLifecycleReceipt] = []
        missing = False
        for ordinal in range(1, GITHUB_MAX_RECEIPTS + 1):
            artifact_id = _artifact_id(
                "github-receipt",
                request.connector_run_id,
                str(ordinal),
            )
            try:
                stored = self._validate_session_json_artifact(
                    await self.store.read_bytes(
                        artifact_id,
                        max_bytes=GITHUB_MAX_ARTIFACT_BYTES,
                    ),
                    artifact_id=artifact_id,
                    session_id=request.session_id,
                    filename=f"github-lifecycle-{ordinal}.json",
                )
            except FileNotFoundError:
                missing = True
                continue
            if missing:
                raise GitHubDeliveryReconstructionRequiredError(
                    "GitHub lifecycle contains a reconstruction gap."
                )
            try:
                receipt = GitHubLifecycleReceipt.model_validate_json(stored.content)
            except Exception as exc:
                raise GitHubDeliveryReconstructionRequiredError(
                    "GitHub lifecycle receipt cannot be reconstructed."
                ) from exc
            if (
                receipt.connector_run_id != request.connector_run_id
                or receipt.request_fingerprint != request.fingerprint
                or receipt.ordinal != ordinal
                or (found and receipt.poll_count < found[-1].poll_count)
            ):
                raise GitHubDeliveryReconstructionRequiredError(
                    "GitHub lifecycle reconstruction is inconsistent."
                )
            found.append(receipt)
        return tuple(found)

    async def publish(self, request, result) -> GitHubDeliveryPublication:
        if type(result) is not GitHubDeliveryResult:
            raise TypeError("result must be GitHubDeliveryResult.")
        try:
            self._validate_result_authority(request, result)
        except GitHubDeliveryReconstructionRequiredError as exc:
            raise GitHubDeliveryAdmissionError(str(exc)) from exc
        digest = result.digest_value
        artifact = await self._ensure(
            result,
            _artifact_id("github-result", request.fingerprint, digest),
            "github-delivery-result.json",
            request.session_id,
            "github_delivery_result",
        )
        receipts = await self.receipts(request)
        if receipts and receipts[-1].result_sha256 == "sha256:" + digest:
            return GitHubDeliveryPublication(result=result, artifact=artifact)
        if len(receipts) >= GITHUB_MAX_RECEIPTS:
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub lifecycle receipt capacity is exhausted."
            )
        receipt = GitHubLifecycleReceipt(
            connector_run_id=request.connector_run_id,
            request_fingerprint=request.fingerprint,
            ordinal=len(receipts) + 1,
            state=result.state,
            result_sha256="sha256:" + digest,
            poll_count=result.poll_count,
        )
        await self._ensure(
            receipt,
            _artifact_id("github-receipt", request.connector_run_id, str(receipt.ordinal)),
            f"github-lifecycle-{receipt.ordinal}.json",
            request.session_id,
            "github_lifecycle",
        )
        return GitHubDeliveryPublication(result=result, artifact=artifact)

    async def latest(self, request) -> GitHubDeliveryPublication | None:
        receipts = await self.receipts(request)
        if not receipts:
            return None
        digest = receipts[-1].result_sha256.removeprefix("sha256:")
        artifact_id = _artifact_id("github-result", request.fingerprint, digest)
        stored = self._validate_session_json_artifact(
            await self.store.read_bytes(
                artifact_id,
                max_bytes=GITHUB_MAX_ARTIFACT_BYTES,
            ),
            artifact_id=artifact_id,
            session_id=request.session_id,
            filename="github-delivery-result.json",
        )
        if "sha256:" + sha256(stored.content).hexdigest() != receipts[-1].result_sha256:
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub result reconstruction is inconsistent."
            )
        try:
            result = GitHubDeliveryResult.model_validate_json(stored.content)
        except Exception as exc:
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub result cannot be reconstructed."
            ) from exc
        if (
            result.digest_value != digest
            or result.state != receipts[-1].state
            or result.poll_count != receipts[-1].poll_count
        ):
            raise GitHubDeliveryReconstructionRequiredError(
                "GitHub result authority conflicts with its lifecycle receipt."
            )
        self._validate_result_authority(request, result)
        return GitHubDeliveryPublication(
            result=result,
            artifact=CodingArtifactReference(
                artifact_id=artifact_id,
                sha256="sha256:" + digest,
                size_bytes=stored.metadata.size_bytes,
                content_type=stored.metadata.content_type,
            ),
        )


def github_connector_behavior_fingerprint() -> str:
    material = {
        "schema": GITHUB_DELIVERY_SCHEMA_VERSION,
        "transport": "fixed-github-operations-v1",
        "polling": "one-durable-observation-per-call-v1",
        "recovery": "durable-per-effect-intent-and-settlement-v3",
        "concurrency": "bounded-connector-run-single-flight-v1",
        "lifecycle": "sealed-noncancelling-owner-and-artifact-drain-v1",
        "reviews": "exact-head-explicit-approver-settlement-v2",
        "artifacts": "consumer-validated-session-authority-v1",
        "merge": "forbidden-v1",
    }
    return (
        "sha256:"
        + sha256(canonical_durable_json_bytes(material, "github_connector_behavior")).hexdigest()
    )


def github_pull_request_delivery_request(
    product: CodingProductPublication,
    remote: RemoteGitDeliveryPublication,
    *,
    connector_run_id: str,
    session_id: str,
    idempotency_key: str,
    requested_at: str,
    repository_alias: str,
    installation_id: str,
    account_id: str,
    mode: Literal["create", "update"],
    existing_pull_request_number: int | None,
    metadata: GitHubPullRequestMetadata,
    checks: GitHubCheckPolicy,
    reviews: GitHubReviewPolicy,
    security: GitHubSecurityAuthority,
    limits: GitHubDeliveryLimits | None = None,
) -> GitHubPullRequestDeliveryRequest:
    if type(product) is not CodingProductPublication:
        raise TypeError("product must be CodingProductPublication.")
    if type(remote) is not RemoteGitDeliveryPublication:
        raise TypeError("remote must be RemoteGitDeliveryPublication.")
    candidate, delivered = product.candidate, remote.result
    if (
        candidate.state is not CodingProductState.PATCH_READY_FOR_DELIVERY
        or candidate.git is None
        or candidate.final_revision is None
    ):
        raise GitHubDeliveryAdmissionError("GitHub delivery requires patch-ready coding evidence.")
    if (
        delivered.state != RemoteGitDeliveryState.PUSHED
        or delivered.next_commit is None
        or delivered.next_ref is None
    ):
        raise GitHubDeliveryAdmissionError(
            "GitHub delivery requires authoritative pushed Git evidence."
        )
    if (
        product.artifact.sha256 != "sha256:" + candidate.digest
        or remote.artifact.sha256 != "sha256:" + delivered.digest
        or delivered.product_result_artifact_id != product.artifact.artifact_id
        or delivered.product_result_sha256 != product.artifact.sha256
        or delivered.product_run_id != candidate.product_run_id
        or delivered.source_workspace_id != candidate.source_workspace_id
        or delivered.final_source_revision != candidate.final_revision
        or delivered.diff_artifact_id != candidate.git.artifact.artifact_id
        or delivered.diff_sha256 != candidate.git.artifact.sha256
        or delivered.tree is None
    ):
        raise GitHubDeliveryAdmissionError("Git and coding delivery evidence conflict.")
    return GitHubPullRequestDeliveryRequest(
        connector_run_id=connector_run_id,
        session_id=session_id,
        idempotency_key=idempotency_key,
        requested_at=requested_at,
        source=GitHubSourceAuthority(
            product_result_artifact_id=product.artifact.artifact_id,
            product_result_sha256=product.artifact.sha256,
            product_request_fingerprint=candidate.request_fingerprint,
            product_run_id=candidate.product_run_id,
            source_workspace_id=candidate.source_workspace_id,
            source_revision=candidate.final_revision,
            diff_artifact_id=candidate.git.artifact.artifact_id,
            diff_sha256=candidate.git.artifact.sha256,
            remote_result_artifact_id=remote.artifact.artifact_id,
            remote_result_sha256=remote.artifact.sha256,
            remote_request_fingerprint=delivered.request_fingerprint,
            remote_delivery_id=delivered.delivery_id,
        ),
        repository=GitHubRepositoryAuthority(
            repository_id=delivered.repository_id,
            repository_alias=repository_alias,
            installation_id=installation_id,
            account_id=account_id,
            base_ref=delivered.base_ref,
            expected_base_commit=delivered.expected_base_commit,
            head_ref=delivered.next_ref,
            head_commit=delivered.next_commit,
        ),
        mode=mode,
        existing_pull_request_number=existing_pull_request_number,
        metadata=metadata,
        checks=checks,
        reviews=reviews,
        security=security,
        limits=limits or GitHubDeliveryLimits(),
    )


def approve_github_delivery(
    request: GitHubPullRequestDeliveryRequest, *, approval_id: str
) -> GitHubDeliveryApproval:
    return GitHubDeliveryApproval(
        approval_id=approval_id,
        request_fingerprint=request.fingerprint,
        metadata_fingerprint=request.metadata.fingerprint,
        policy_fingerprint=request.security.policy_fingerprint,
        approved_operations=request.security.allowed_operations,
    )


def _check_state(
    request: GitHubPullRequestDeliveryRequest, bundle: GitHubCheckBundle
) -> GitHubCheckState:
    if bundle.head_commit != request.repository.head_commit:
        return GitHubCheckState.SUPERSEDED
    if bundle.truncated:
        return GitHubCheckState.PROVIDER_AMBIGUOUS
    exact = [item for item in bundle.checks if item.head_commit == request.repository.head_commit]
    grouped: dict[str, list[GitHubCheckObservation]] = {}
    for item in exact:
        grouped.setdefault(item.name, []).append(item)
    families: dict[tuple[str, str], list[GitHubCheckObservation]] = {}
    for item in exact:
        family = "commit-status" if item.provider_id.startswith("commit-status:") else "check-run"
        families.setdefault((item.name, family), []).append(item)
    if any(
        len({(item.status, item.conclusion) for item in observations}) > 1
        for observations in families.values()
    ):
        return GitHubCheckState.PROVIDER_AMBIGUOUS
    if any(name not in grouped for name in request.checks.required_checks):
        return GitHubCheckState.MISSING
    required = [item for name in request.checks.required_checks for item in grouped[name]]
    conclusions = [item.conclusion for item in required]
    statuses = [item.status for item in required]
    if any(status != "completed" for status in statuses):
        return GitHubCheckState.PENDING
    allowed = (
        {"success"}
        | ({"neutral"} if request.checks.allow_neutral else set())
        | ({"skipped"} if request.checks.allow_skipped else set())
    )
    if all(conclusion in allowed for conclusion in conclusions):
        return GitHubCheckState.PASSED
    if any(conclusion in {"cancelled", "stale"} for conclusion in conclusions):
        return GitHubCheckState.CANCELLED
    if any(conclusion == "timed_out" for conclusion in conclusions):
        return GitHubCheckState.TIMED_OUT
    return GitHubCheckState.FAILED


def _review_state(
    bundle: GitHubReviewBundle,
    policy: GitHubReviewPolicy,
    head_commit: str,
) -> GitHubReviewState:
    latest: dict[str, GitHubFeedbackObservation] = {}
    for item in bundle.feedback:
        if (
            item.kind == "review"
            and item.head_commit == head_commit
            and item.state in {"approved", "changes_requested", "dismissed"}
        ):
            latest[item.author_login.casefold()] = item
    relevant = (
        [
            latest[login.casefold()]
            for login in policy.required_approvers
            if login.casefold() in latest
        ]
        if policy.required_approvers
        else list(latest.values())
    )
    states = {item.state for item in relevant}
    if "changes_requested" in states:
        return GitHubReviewState.CHANGES_REQUESTED
    approvals = sum(item.state == "approved" for item in relevant)
    if approvals >= policy.minimum_approvals:
        return GitHubReviewState.APPROVED
    return GitHubReviewState.COMMENTED if bundle.feedback else GitHubReviewState.NONE


class GitHubPullRequestConnector:
    """Mutate one bound PR and durably observe one exact-head poll per call."""

    def __init__(
        self,
        profile: GitHubConnectorProfile,
        *,
        repository: GitHubDeliveryRepository,
        transport: GitHubConnectorTransport,
        clock=None,
    ) -> None:
        if type(profile) is not GitHubConnectorProfile:
            raise TypeError("profile must be GitHubConnectorProfile.")
        if type(repository) is not GitHubDeliveryRepository:
            raise TypeError("repository must be GitHubDeliveryRepository.")
        if not isinstance(transport, GitHubConnectorTransport):
            raise TypeError("transport must implement GitHubConnectorTransport.")
        self.profile, self.repository = profile, repository
        self.transport = _GitHubTransportBoundary(transport)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._owners: dict[
            str, tuple[str, asyncio.Task, asyncio.Event, ArtifactWriteSettlementObserver]
        ] = {}
        self._sealed = False

    def seal(self) -> None:
        """Reject new calls and stop subsequent effects in retained calls."""

        self._sealed = True
        for _, _, stopped, _ in self._owners.values():
            stopped.set()

    async def aclose(self, *, timeout_s: float = 30.0) -> bool:
        """Seal and observe local quiescence without cancelling owned mutations.

        False (or cancellation) requires retaining this connector and its loop;
        a later call may observe settlement. True is not proof of remote success.
        Shared transports and artifact stores are not closed by this method.
        """

        if type(timeout_s) not in (int, float):
            raise ValueError("timeout_s must be a finite non-negative number.")
        try:
            timeout = float(timeout_s)
        except OverflowError:
            raise ValueError("timeout_s must be a finite non-negative number.") from None
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_s must be a finite non-negative number.")
        self.seal()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            pending = []
            for run_id, owned in tuple(self._owners.items()):
                _, owner, _, observer = owned
                if not owner.done():
                    pending.append(owner)
                elif not observer.record_active_candidates() and self._owners.get(run_id) is owned:
                    del self._owners[run_id]
            if not self._owners:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            if pending:
                await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            else:
                # Artifact callbacks may finish after their coroutine. They are
                # thread-owned; never mistake a done task for their settlement.
                await asyncio.sleep(min(remaining, 0.01))

    def _config(self, request):
        try:
            config = self.profile.repositories[request.repository.repository_alias]
        except KeyError:
            raise GitHubDeliveryAdmissionError(
                "GitHub repository alias is not configured."
            ) from None
        if (
            config.repository_id,
            config.installation_id,
            config.account_id,
            config.credential_profile_id,
            config.egress_profile_id,
            self.profile.connector_id,
            self.profile.behavior_fingerprint,
        ) != (
            request.repository.repository_id,
            request.repository.installation_id,
            request.repository.account_id,
            request.security.credential_profile_id,
            request.security.egress_profile_id,
            request.security.connector_id,
            request.security.connector_behavior_fingerprint,
        ):
            raise GitHubDeliveryAdmissionError(
                "Configured GitHub authority conflicts with request."
            )
        return config

    async def _validate_inputs(self, request, product, remote):
        if type(product) is not CodingProductPublication:
            raise TypeError("product must be CodingProductPublication.")
        if type(remote) is not RemoteGitDeliveryPublication:
            raise TypeError("remote must be RemoteGitDeliveryPublication.")
        candidate, delivered = product.candidate, remote.result
        if (
            candidate.state is not CodingProductState.PATCH_READY_FOR_DELIVERY
            or delivered.state != RemoteGitDeliveryState.PUSHED
            or candidate.git is None
            or candidate.final_revision is None
            or delivered.next_commit is None
            or delivered.next_ref is None
        ):
            raise GitHubDeliveryAdmissionError("GitHub inputs are not settled for delivery.")
        if (
            product.artifact.sha256 != "sha256:" + candidate.digest
            or remote.artifact.sha256 != "sha256:" + delivered.digest
            or delivered.product_result_artifact_id != product.artifact.artifact_id
            or delivered.product_result_sha256 != product.artifact.sha256
            or delivered.product_run_id != candidate.product_run_id
            or delivered.source_workspace_id != candidate.source_workspace_id
            or delivered.final_source_revision != candidate.final_revision
            or delivered.diff_artifact_id != candidate.git.artifact.artifact_id
            or delivered.diff_sha256 != candidate.git.artifact.sha256
        ):
            raise GitHubDeliveryAdmissionError(
                "Remote Git result conflicts with exact coding-product evidence."
            )
        expected = (
            product.artifact.artifact_id,
            product.artifact.sha256,
            candidate.request_fingerprint,
            candidate.product_run_id,
            candidate.source_workspace_id,
            candidate.final_revision,
            candidate.git.artifact.artifact_id if candidate.git else None,
            candidate.git.artifact.sha256 if candidate.git else None,
            remote.artifact.artifact_id,
            remote.artifact.sha256,
            delivered.request_fingerprint,
            delivered.delivery_id,
            delivered.repository_id,
            delivered.base_ref,
            delivered.expected_base_commit,
            delivered.next_ref,
            delivered.next_commit,
        )
        actual = (
            request.source.product_result_artifact_id,
            request.source.product_result_sha256,
            request.source.product_request_fingerprint,
            request.source.product_run_id,
            request.source.source_workspace_id,
            request.source.source_revision,
            request.source.diff_artifact_id,
            request.source.diff_sha256,
            request.source.remote_result_artifact_id,
            request.source.remote_result_sha256,
            request.source.remote_request_fingerprint,
            request.source.remote_delivery_id,
            request.repository.repository_id,
            request.repository.base_ref,
            request.repository.expected_base_commit,
            request.repository.head_ref,
            request.repository.head_commit,
        )
        if expected != actual:
            raise GitHubDeliveryAdmissionError(
                "GitHub request conflicts with exact core delivery evidence."
            )
        try:
            durable_product = await CodingProductArtifactRepository(
                self.repository.store
            ).load_publication(
                request_fingerprint=candidate.request_fingerprint,
                digest=product.artifact.sha256,
            )
        except (
            FileNotFoundError,
            TypeError,
            ValueError,
            CodingProductReconstructionRequiredError,
        ) as exc:
            raise GitHubDeliveryAdmissionError(
                "Coding-product publication cannot be reconstructed."
            ) from exc
        if durable_product != product:
            raise GitHubDeliveryAdmissionError(
                "Coding-product publication conflicts with durable authority."
            )
        expected_remote_artifact_id = _artifact_id(
            "remote-git-result-v1",
            delivered.request_fingerprint,
            remote.artifact.sha256.removeprefix("sha256:"),
        )
        if (
            remote.artifact.artifact_id != expected_remote_artifact_id
            or remote.artifact.sha256 != "sha256:" + delivered.digest
        ):
            raise GitHubDeliveryAdmissionError("Remote Git publication is not content-addressed.")
        try:
            stored = copy_artifact_read_result(
                await self.repository.store.read_bytes(
                    expected_remote_artifact_id,
                    max_bytes=GITHUB_MAX_ARTIFACT_BYTES,
                ),
                expected_artifact_id=expected_remote_artifact_id,
                max_content_bytes=GITHUB_MAX_ARTIFACT_BYTES,
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            raise GitHubDeliveryAdmissionError(
                "Remote Git artifact store returned inconsistent evidence."
            ) from exc
        content_digest = "sha256:" + sha256(stored.content).hexdigest()
        if (
            stored.truncated
            or stored.redaction_truncated
            or stored.metadata.scope is not ArtifactScope.SESSION
            or stored.metadata.session_id != delivered.session_id
            or stored.metadata.agent_name is not None
            or stored.metadata.environment_name is not None
            or stored.metadata.filename != "remote-git-delivery-result.json"
            or stored.metadata.content_type != "application/json"
            or stored.metadata.size_bytes != remote.artifact.size_bytes
            or remote.artifact.content_type != "application/json"
            or content_digest != remote.artifact.sha256
            or stored.metadata.metadata.get("content_sha256") != content_digest
        ):
            raise GitHubDeliveryAdmissionError("Remote Git artifact authority is inconsistent.")
        try:
            durable_remote = RemoteGitDeliveryResult.model_validate_json(stored.content)
        except Exception as exc:
            raise GitHubDeliveryAdmissionError(
                "Remote Git publication cannot be reconstructed."
            ) from exc
        if durable_remote != delivered:
            raise GitHubDeliveryAdmissionError(
                "Remote Git publication conflicts with durable authority."
            )

    async def _redactor(self, config):
        resolved = await config.credentials.resolver.resolve(
            config.credentials.token, scope={"repository_id": config.repository_id}
        )
        return SecretRedactor(resolved)

    async def _publish_preserving_signal(self, request, result, signal: BaseException):
        failure = None
        try:
            await self.repository.publish(request, result)
        except BaseException as cleanup:
            failure = _safe_github_failure(
                signal
                if cleanup is signal
                else BaseExceptionGroup("GitHub signal publication failed", [signal, cleanup])
            )
        if failure is not None:
            raise failure from failure.__cause__

    @staticmethod
    def _require_secret_free_model(value: BaseModel, redactor: SecretRedactor, field: str) -> None:
        original = value.model_dump(mode="json", warnings=False)
        if redactor.redact_json_values(original) != original:
            raise GitHubDeliveryAdmissionError(
                f"{field} contains configured GitHub credential material."
            )

    @staticmethod
    def _pr_matches(request, pr):
        return (
            (
                request.existing_pull_request_number is None
                or pr.number == request.existing_pull_request_number
            )
            and pr.base_ref == request.repository.base_ref
            and pr.base_commit == request.repository.expected_base_commit
            and pr.head_ref == request.repository.head_ref
            and pr.head_commit == request.repository.head_commit
        )

    @classmethod
    def _require_mutation_target(cls, request, pr, *, number=None):
        if not cls._pr_matches(request, pr) or (number is not None and pr.number != number):
            raise GitHubProviderError("pull_request_binding_changed", ambiguous=True)

    async def _read_bound_pr(self, config, request, number, redactor):
        pr = self._redacted_pr(
            await self.transport.get_pull_request(config, number, request.limits), redactor
        )
        self._require_mutation_target(request, pr, number=number)
        return pr

    @staticmethod
    def _operation_states(
        operations: Sequence[GitHubOperationEvidence],
    ) -> dict[GitHubOperation, str]:
        # Diagnostics may change, but only later evidence for this exact
        # operation can settle its durable uncertainty.
        return {item.operation: item.status for item in operations}

    @staticmethod
    def _mutation_observed(
        operation: GitHubOperation,
        request: GitHubPullRequestDeliveryRequest,
        pr: GitHubPullRequestSnapshot | None,
    ) -> bool:
        if pr is None:
            return False
        if operation is GitHubOperation.CREATE_PULL_REQUEST:
            return True
        if operation is GitHubOperation.UPDATE_PULL_REQUEST:
            return pr.title == request.metadata.title and pr.body == request.metadata.body
        if operation is GitHubOperation.SET_LABELS:
            return tuple(sorted(pr.labels)) == request.metadata.labels
        if operation is GitHubOperation.MARK_READY:
            return not pr.draft
        return False

    @staticmethod
    def _approval_valid(
        request: GitHubPullRequestDeliveryRequest,
        approval: GitHubDeliveryApproval,
    ) -> bool:
        return GitHubDeliveryRepository._approval_matches(request, approval)

    @staticmethod
    def _redacted_pr(
        pr: GitHubPullRequestSnapshot, redactor: SecretRedactor
    ) -> GitHubPullRequestSnapshot:
        if type(pr) is not GitHubPullRequestSnapshot:
            raise GitHubProviderError("malformed_pull_request")
        return GitHubPullRequestSnapshot.model_validate_json(
            canonical_durable_json_bytes(
                redactor.redact_json_values(pr.model_dump(mode="json", warnings=False)),
                "github_redacted_pull_request",
            )
        )

    @classmethod
    def _redacted_prs(
        cls,
        pull_requests: tuple[GitHubPullRequestSnapshot, ...],
        redactor: SecretRedactor,
    ) -> tuple[GitHubPullRequestSnapshot, ...]:
        if type(pull_requests) is not tuple:
            raise GitHubProviderError("malformed_pull_request_list")
        return tuple(cls._redacted_pr(item, redactor) for item in pull_requests[:2])

    @staticmethod
    def _redacted_checks(
        bundle: GitHubCheckBundle,
        redactor: SecretRedactor,
        limits: GitHubDeliveryLimits,
    ) -> GitHubCheckBundle:
        if type(bundle) is not GitHubCheckBundle:
            raise GitHubProviderError("malformed_check_runs")
        redacted = GitHubCheckBundle.model_validate_json(
            canonical_durable_json_bytes(
                redactor.redact_json_values(bundle.model_dump(mode="json", warnings=False)),
                "github_redacted_checks",
            )
        )
        deduplicated: dict[str, GitHubCheckObservation] = {}
        truncated = redacted.truncated or len(redacted.checks) > limits.max_checks
        for item in redacted.checks:
            existing = deduplicated.get(item.provider_id)
            if existing is not None and existing != item:
                truncated = True
                continue
            deduplicated[item.provider_id] = item
        checks = tuple(
            sorted(
                deduplicated.values(),
                key=lambda item: (item.name, item.provider_id),
            )
        )
        return redacted.model_copy(
            update={
                "checks": checks[: limits.max_checks],
                "truncated": truncated or len(checks) > limits.max_checks,
            }
        )

    @staticmethod
    def _redacted_reviews(
        bundle: GitHubReviewBundle,
        redactor: SecretRedactor,
        limits: GitHubDeliveryLimits,
    ) -> GitHubReviewBundle:
        if type(bundle) is not GitHubReviewBundle:
            raise GitHubProviderError("malformed_review_feedback")
        redacted = GitHubReviewBundle.model_validate_json(
            canonical_durable_json_bytes(
                redactor.redact_json_values(bundle.model_dump(mode="json", warnings=False)),
                "github_redacted_reviews",
            )
        )
        bound = limits.max_reviews + 2 * limits.max_comments
        deduplicated: dict[str, GitHubFeedbackObservation] = {}
        truncated = redacted.truncated or len(redacted.feedback) > bound
        for item in redacted.feedback:
            existing = deduplicated.get(item.provider_id)
            if existing is not None and existing != item:
                truncated = True
                continue
            deduplicated[item.provider_id] = item
        feedback = tuple(
            sorted(
                deduplicated.values(),
                key=lambda item: (item.created_at, item.provider_id),
            )
        )
        review_versions: dict[tuple[str, str], str] = {}
        for item in feedback:
            if item.kind != "review":
                continue
            version = (item.author_login.casefold(), item.created_at)
            prior_state = review_versions.setdefault(version, item.state)
            if prior_state != item.state:
                truncated = True
        return redacted.model_copy(
            update={
                "feedback": feedback[:bound],
                "truncated": truncated or len(feedback) > bound,
            }
        )

    @staticmethod
    def _operation(
        operation: GitHubOperation,
        status: Literal["reconciled", "succeeded", "failed", "ambiguous"],
        request_id: str,
        provider_id: str | None,
        redactor: SecretRedactor,
    ) -> GitHubOperationEvidence:
        return GitHubOperationEvidence(
            operation=operation,
            status=status,
            request_id=redactor.redact_text_bounded(str(request_id), max_bytes=1024),
            provider_id=(
                None
                if provider_id is None
                else redactor.redact_text_bounded(str(provider_id), max_bytes=1024)
            ),
        )

    def _result(
        self,
        request,
        state,
        *,
        approval=None,
        pr=None,
        checks_state=None,
        review_state=None,
        checks=(),
        feedback=(),
        checks_truncated=False,
        feedback_truncated=False,
        operations=(),
        poll_count=0,
        reason=None,
    ):
        retryable = (
            state
            in {
                GitHubDeliveryState.CHECKS_PENDING,
                GitHubDeliveryState.RATE_LIMITED,
                GitHubDeliveryState.PROVIDER_UNAVAILABLE,
                GitHubDeliveryState.PR_UPDATED,
            }
            or (
                state is GitHubDeliveryState.AMBIGUOUS
                and reason
                in {
                    "provider_mutation_in_flight",
                    "provider_mutation_cancelled_ambiguous",
                    "provider_mutation_reconciliation_unavailable",
                }
            )
            or checks_state in {GitHubCheckState.PENDING, GitHubCheckState.MISSING}
        )
        return GitHubDeliveryResult(
            connector_run_id=request.connector_run_id,
            session_id=request.session_id,
            idempotency_key=request.idempotency_key,
            request_fingerprint=request.fingerprint,
            state=state,
            repository_id=request.repository.repository_id,
            connector_id=request.security.connector_id,
            installation_id=request.repository.installation_id,
            account_id=request.repository.account_id,
            credential_profile_id=request.security.credential_profile_id,
            egress_profile_id=request.security.egress_profile_id,
            policy_fingerprint=request.security.policy_fingerprint,
            connector_behavior_fingerprint=request.security.connector_behavior_fingerprint,
            approval_policy_fingerprint=request.security.approval_policy_fingerprint,
            redaction_profile_fingerprint=request.security.redaction_profile_fingerprint,
            metadata_fingerprint=request.metadata.fingerprint,
            check_policy_fingerprint=request.checks.fingerprint,
            review_policy_fingerprint=request.reviews.fingerprint,
            approval_fingerprint=None if approval is None else approval.fingerprint,
            approval_id=None if approval is None else approval.approval_id,
            product_result_artifact_id=request.source.product_result_artifact_id,
            product_run_id=request.source.product_run_id,
            source_workspace_id=request.source.source_workspace_id,
            diff_artifact_id=request.source.diff_artifact_id,
            remote_result_artifact_id=request.source.remote_result_artifact_id,
            remote_delivery_id=request.source.remote_delivery_id,
            product_request_fingerprint=request.source.product_request_fingerprint,
            remote_request_fingerprint=request.source.remote_request_fingerprint,
            product_result_sha256=request.source.product_result_sha256,
            remote_result_sha256=request.source.remote_result_sha256,
            source_revision=request.source.source_revision,
            diff_sha256=request.source.diff_sha256,
            repository_alias=request.repository.repository_alias,
            base_ref=request.repository.base_ref,
            base_commit=request.repository.expected_base_commit,
            head_ref=request.repository.head_ref,
            head_commit=request.repository.head_commit,
            pull_request=pr,
            checks_state=checks_state,
            review_state=review_state,
            checks=tuple(checks),
            feedback=tuple(feedback),
            checks_truncated=checks_truncated,
            feedback_truncated=feedback_truncated,
            required_checks=request.checks.required_checks,
            allowed_operations=request.security.allowed_operations,
            review_approval_required=request.reviews.approval_required,
            required_approvers=request.reviews.required_approvers,
            minimum_approvals=request.reviews.minimum_approvals,
            follow_up_allowed=request.reviews.allow_follow_up,
            max_follow_up_items=request.reviews.max_follow_up_items,
            max_follow_up_iterations=request.reviews.max_follow_up_iterations,
            limits=request.limits,
            operations=tuple(operations),
            poll_count=poll_count,
            next_poll_after_seconds=request.limits.poll_interval_seconds if retryable else None,
            next_poll_at=(
                self.clock() + timedelta(seconds=request.limits.poll_interval_seconds)
            ).isoformat()
            if retryable
            else None,
            reason_code=reason,
        )

    async def run(
        self,
        request: GitHubPullRequestDeliveryRequest,
        product: CodingProductPublication,
        remote: RemoteGitDeliveryPublication,
        *,
        approval: GitHubDeliveryApproval | None = None,
    ) -> GitHubDeliveryPublication:
        if self._sealed:
            raise GitHubDeliveryAdmissionError("GitHub connector is sealed.")
        if type(request) is not GitHubPullRequestDeliveryRequest:
            raise TypeError("request must be GitHubPullRequestDeliveryRequest.")
        identity = request.fingerprint + ("" if approval is None else approval.fingerprint)
        prior = self._owners.get(request.connector_run_id)
        if prior is not None and prior[1].done():
            if prior[3].record_active_candidates():
                raise GitHubDeliveryAdmissionError("GitHub artifact settlement is still active.")
            del self._owners[request.connector_run_id]
            prior = None
        if prior is None:
            if len(self._owners) >= 64:
                raise GitHubDeliveryAdmissionError("GitHub execution-owner capacity is exhausted.")
            stopped = asyncio.Event()
            observer = ArtifactWriteSettlementObserver(max_operations=1)

            async def execute_owned():
                try:
                    with observer:
                        value = await self._run_once(
                            request,
                            product,
                            remote,
                            approval=approval,
                            stop_requested=stopped,
                        )
                    return _GitHubOwnedOutcome(value=value)
                except BaseException as error:
                    return _GitHubOwnedOutcome(failure=_safe_github_failure(error))

            owner = asyncio.create_task(execute_owned())
            self._owners[request.connector_run_id] = (identity, owner, stopped, observer)

            def consume(completed: asyncio.Task) -> None:
                # Keep the exact owner reachable until settlement, and retrieve
                # failures even if the public waiter has already left.
                if not completed.cancelled():
                    completed.exception()
                current = self._owners.get(request.connector_run_id)
                if (
                    current is not None
                    and current[1] is completed
                    and not observer.record_active_candidates()
                ):
                    del self._owners[request.connector_run_id]

            owner.add_done_callback(consume)
        else:
            raise GitHubDeliveryAdmissionError(
                "GitHub execution is still owned by an unsettled call."
            )
        try:
            done, _ = await asyncio.wait({owner}, timeout=request.limits.timeout_seconds)
            if not done:
                stopped.set()
                raise GitHubProviderError("provider_timeout", ambiguous=True)
            outcome = owner.result()
            if outcome.failure is not None:
                raise outcome.failure from outcome.failure.__cause__
            return outcome.value
        except asyncio.CancelledError:
            stopped.set()
            # Do not cancel opaque provider or artifact operations. The owner
            # retains their lifetime and prevents another effect from starting.
            raise

    async def _run_once(
        self,
        request: GitHubPullRequestDeliveryRequest,
        product: CodingProductPublication,
        remote: RemoteGitDeliveryPublication,
        *,
        approval: GitHubDeliveryApproval | None = None,
        stop_requested: asyncio.Event | None = None,
    ) -> GitHubDeliveryPublication:
        config = self._config(request)
        await self._validate_inputs(request, product, remote)
        redactor = await self._redactor(config)
        self._require_secret_free_model(request, redactor, "GitHub request")
        await self.repository.ensure_request(request)
        binding = _GitHubConfigurationBinding(
            request_fingerprint=request.fingerprint,
            api_base_url=config.api_base_url,
            owner=config.owner,
            name=config.name,
            credential_reference_sha256="sha256:"
            + sha256(
                _model_bytes(config.credentials.token, "github_credential_reference")
            ).hexdigest(),
        )
        self._require_secret_free_model(binding, redactor, "GitHub configuration")
        await self.repository._ensure(
            binding,
            _artifact_id("github-configuration", request.connector_run_id),
            "github-configuration.json",
            request.session_id,
            "github_configuration",
        )
        latest = await self.repository.latest(request)
        durable_approval: GitHubDeliveryApproval | None = None
        if latest is not None and latest.result.approval_id is not None:
            durable_approval = await self.repository.approval(request)
            if (
                durable_approval is None
                or durable_approval.approval_id != latest.result.approval_id
                or durable_approval.fingerprint != latest.result.approval_fingerprint
            ):
                raise GitHubDeliveryReconstructionRequiredError(
                    "GitHub result conflicts with durable approval authority."
                )
        terminal = {
            GitHubDeliveryState.DENIED,
            GitHubDeliveryState.CHECKS_PASSED,
            GitHubDeliveryState.CHECKS_FAILED,
            GitHubDeliveryState.CHANGES_REQUESTED,
            GitHubDeliveryState.CLOSED,
            GitHubDeliveryState.SUPERSEDED,
            GitHubDeliveryState.CONFLICT,
            GitHubDeliveryState.PERMISSION_DENIED,
            GitHubDeliveryState.CANCELLED,
            GitHubDeliveryState.FAILED,
            GitHubDeliveryState.PARTIAL,
            GitHubDeliveryState.AMBIGUOUS,
            GitHubDeliveryState.RECONSTRUCTION_REQUIRED,
        }
        unresolved_operations = tuple(
            operation
            for operation, status in self._operation_states(
                () if latest is None else latest.result.operations
            ).items()
            if status == "ambiguous"
        )
        retryable_ambiguous_result = bool(
            latest is not None
            and latest.result.state is GitHubDeliveryState.AMBIGUOUS
            and latest.result.next_poll_at is not None
        )
        if latest is not None and (
            (latest.result.state in terminal and not retryable_ambiguous_result)
            or (
                latest.result.state is GitHubDeliveryState.APPROVED
                and latest.result.checks_state is GitHubCheckState.PASSED
            )
        ):
            return latest
        if (
            latest is not None
            and latest.result.next_poll_at is not None
            and self.clock() < datetime.fromisoformat(latest.result.next_poll_at)
        ):
            return latest
        poll_count = 0 if latest is None else latest.result.poll_count
        operations = [] if latest is None else list(latest.result.operations)
        if approval is None:
            approval = durable_approval or await self.repository.approval(request)
        else:
            if type(approval) is not GitHubDeliveryApproval or not self._approval_valid(
                request,
                approval,
            ):
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.DENIED,
                        poll_count=poll_count,
                        operations=operations,
                        reason="provider_approval_conflict",
                    ),
                )
            self._require_secret_free_model(approval, redactor, "GitHub approval")
            await self.repository.ensure_approval(request, approval)
        requested = datetime.fromisoformat(request.requested_at.replace("Z", "+00:00"))
        if (
            self.clock() > requested + timedelta(seconds=request.limits.max_elapsed_seconds)
            or poll_count >= request.limits.max_polls
        ):
            return await self.repository.publish(
                request,
                self._result(
                    request,
                    GitHubDeliveryState.CHECKS_FAILED,
                    approval=approval,
                    pr=None if latest is None else latest.result.pull_request,
                    poll_count=poll_count,
                    operations=operations,
                    checks_state=GitHubCheckState.TIMED_OUT,
                    reason="poll_bounds_exhausted",
                ),
            )
        active_operation: GitHubOperation | None = None
        try:
            if stop_requested is not None and stop_requested.is_set():
                raise _GitHubOwnerStopped()
            base = await self.transport.observe_ref(
                config, request.repository.base_ref, request.limits
            )
            head = await self.transport.observe_ref(
                config, request.repository.head_ref, request.limits
            )
            if (
                base != request.repository.expected_base_commit
                or head != request.repository.head_commit
            ):
                state = (
                    GitHubDeliveryState.SUPERSEDED
                    if head not in {None, request.repository.head_commit}
                    else GitHubDeliveryState.CONFLICT
                )
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        state,
                        approval=approval,
                        poll_count=poll_count,
                        operations=operations,
                        reason="provider_ref_mismatch",
                    ),
                )
            if request.mode == "create":
                matches = await self.transport.find_pull_requests(
                    config,
                    base_ref=request.repository.base_ref,
                    head_ref=request.repository.head_ref,
                    limits=request.limits,
                )
                matches = self._redacted_prs(matches, redactor)
                if len(matches) > 1:
                    return await self.repository.publish(
                        request,
                        self._result(
                            request,
                            GitHubDeliveryState.CONFLICT,
                            approval=approval,
                            poll_count=poll_count,
                            operations=operations,
                            reason="multiple_matching_pull_requests",
                        ),
                    )
                pr = matches[0] if matches else None
            else:
                assert request.existing_pull_request_number is not None
                pr = await self.transport.get_pull_request(
                    config, request.existing_pull_request_number, request.limits
                )
                pr = self._redacted_pr(pr, redactor)
            if pr is not None and not self._pr_matches(request, pr):
                state = (
                    GitHubDeliveryState.SUPERSEDED
                    if (
                        pr.head_ref != request.repository.head_ref
                        or pr.head_commit != request.repository.head_commit
                    )
                    else GitHubDeliveryState.CONFLICT
                )
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        state,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count,
                        operations=operations,
                        reason="pull_request_binding_changed_before_mutation",
                    ),
                )
            for recovery_operation in unresolved_operations:
                if self._mutation_observed(recovery_operation, request, pr):
                    assert pr is not None
                    operations.append(
                        self._operation(
                            recovery_operation,
                            "reconciled",
                            "durable-provider-reconciliation",
                            str(pr.number),
                            redactor,
                        )
                    )
                else:
                    # Absence is not proof that an interrupted provider write
                    # cannot still commit. Never repeat an unobserved effect.
                    return await self.repository.publish(
                        request,
                        self._result(
                            request,
                            GitHubDeliveryState.AMBIGUOUS,
                            approval=approval,
                            pr=pr,
                            poll_count=min(request.limits.max_polls, poll_count + 1),
                            operations=operations,
                            reason="provider_mutation_reconciliation_unavailable",
                        ),
                    )
            if pr is not None and pr.state == "closed":
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.CLOSED,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count,
                        operations=operations,
                        reason="pull_request_closed",
                    ),
                )
            if pr is not None and request.metadata.draft and not pr.draft:
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.DENIED,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count,
                        operations=operations,
                        reason="ready_to_draft_not_supported",
                    ),
                )
            completed_operations = {
                operation
                for operation, status in self._operation_states(operations).items()
                if status in {"succeeded", "reconciled"}
            }
            needed_operations: set[GitHubOperation] = set()
            if pr is None:
                needed_operations.add(GitHubOperation.CREATE_PULL_REQUEST)
            elif pr.title != request.metadata.title or pr.body != request.metadata.body:
                needed_operations.add(GitHubOperation.UPDATE_PULL_REQUEST)
            if (
                pr is not None
                and request.metadata.labels
                and tuple(sorted(pr.labels)) != request.metadata.labels
            ):
                needed_operations.add(GitHubOperation.SET_LABELS)
            if (
                request.metadata.reviewers or request.metadata.teams
            ) and GitHubOperation.REQUEST_REVIEWERS not in completed_operations:
                needed_operations.add(GitHubOperation.REQUEST_REVIEWERS)
            if pr is not None and pr.draft and not request.metadata.draft:
                needed_operations.add(GitHubOperation.MARK_READY)
            mutation_needed = bool(needed_operations)

            async def persist_effect_intent(operation: GitHubOperation) -> None:
                if stop_requested is not None and stop_requested.is_set():
                    raise _GitHubOwnerStopped()
                operations.append(
                    self._operation(
                        operation,
                        "ambiguous",
                        "durable-provider-intent",
                        None if pr is None else str(pr.number),
                        redactor,
                    )
                )
                await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.AMBIGUOUS,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count,
                        operations=operations,
                        reason="provider_mutation_in_flight",
                    ),
                )
                if stop_requested is not None and stop_requested.is_set():
                    raise _GitHubOwnerStopped()

            async def persist_effect_settlement() -> None:
                await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.PR_UPDATED,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count,
                        operations=operations,
                    ),
                )

            if mutation_needed:
                if not needed_operations <= set(request.security.allowed_operations):
                    return await self.repository.publish(
                        request,
                        self._result(
                            request,
                            GitHubDeliveryState.DENIED,
                            approval=approval,
                            pr=pr,
                            poll_count=poll_count,
                            operations=operations,
                            reason="provider_operation_not_allowed",
                        ),
                    )
                if approval is None:
                    return await self.repository.publish(
                        request,
                        self._result(
                            request,
                            GitHubDeliveryState.APPROVAL_REQUIRED,
                            pr=pr,
                            poll_count=poll_count,
                            operations=operations,
                            reason="durable_provider_approval_required",
                        ),
                    )
                try:
                    if pr is None:
                        await persist_effect_intent(GitHubOperation.CREATE_PULL_REQUEST)
                        active_operation = GitHubOperation.CREATE_PULL_REQUEST
                        pr, request_id = await self.transport.create_pull_request(config, request)
                        pr = self._redacted_pr(pr, redactor)
                        self._require_mutation_target(request, pr)
                        operations.append(
                            self._operation(
                                GitHubOperation.CREATE_PULL_REQUEST,
                                "succeeded",
                                request_id,
                                str(pr.number),
                                redactor,
                            )
                        )
                        active_operation = None
                        await persist_effect_settlement()
                    elif pr.title != request.metadata.title or pr.body != request.metadata.body:
                        await persist_effect_intent(GitHubOperation.UPDATE_PULL_REQUEST)
                        active_operation = GitHubOperation.UPDATE_PULL_REQUEST
                        target_number = pr.number
                        pr, request_id = await self.transport.update_pull_request(
                            config, request, pr.number
                        )
                        pr = self._redacted_pr(pr, redactor)
                        self._require_mutation_target(request, pr, number=target_number)
                        operations.append(
                            self._operation(
                                GitHubOperation.UPDATE_PULL_REQUEST,
                                "succeeded",
                                request_id,
                                str(pr.number),
                                redactor,
                            )
                        )
                        active_operation = None
                        await persist_effect_settlement()
                    if (
                        request.metadata.labels
                        and tuple(sorted(pr.labels)) != request.metadata.labels
                    ):
                        await persist_effect_intent(GitHubOperation.SET_LABELS)
                        active_operation = GitHubOperation.SET_LABELS
                        request_id = await self.transport.set_labels(
                            config, pr.number, request.metadata.labels, request.limits
                        )
                        operations.append(
                            self._operation(
                                GitHubOperation.SET_LABELS,
                                "succeeded",
                                request_id,
                                str(pr.number),
                                redactor,
                            )
                        )
                        active_operation = None
                        await persist_effect_settlement()
                        pr = await self._read_bound_pr(config, request, pr.number, redactor)
                    if (
                        request.metadata.reviewers or request.metadata.teams
                    ) and GitHubOperation.REQUEST_REVIEWERS not in completed_operations:
                        await persist_effect_intent(GitHubOperation.REQUEST_REVIEWERS)
                        active_operation = GitHubOperation.REQUEST_REVIEWERS
                        request_id = await self.transport.request_reviewers(
                            config,
                            pr.number,
                            request.metadata.reviewers,
                            request.metadata.teams,
                            request.limits,
                        )
                        operations.append(
                            self._operation(
                                GitHubOperation.REQUEST_REVIEWERS,
                                "succeeded",
                                request_id,
                                str(pr.number),
                                redactor,
                            )
                        )
                        active_operation = None
                        await persist_effect_settlement()
                    if pr.draft and not request.metadata.draft:
                        if GitHubOperation.MARK_READY not in request.security.allowed_operations:
                            return await self.repository.publish(
                                request,
                                self._result(
                                    request,
                                    GitHubDeliveryState.DENIED,
                                    approval=approval,
                                    pr=pr,
                                    poll_count=poll_count,
                                    operations=operations,
                                    reason="mark_ready_not_allowed",
                                ),
                            )
                        await persist_effect_intent(GitHubOperation.MARK_READY)
                        active_operation = GitHubOperation.MARK_READY
                        request_id = await self.transport.mark_ready(
                            config, pr.number, request.limits
                        )
                        operations.append(
                            self._operation(
                                GitHubOperation.MARK_READY,
                                "succeeded",
                                request_id,
                                str(pr.number),
                                redactor,
                            )
                        )
                        active_operation = None
                        await persist_effect_settlement()
                        pr = await self._read_bound_pr(config, request, pr.number, redactor)
                except GitHubProviderError as exc:
                    if not exc.ambiguous:
                        if active_operation is not None:
                            operations.append(
                                self._operation(
                                    active_operation,
                                    "failed",
                                    exc.request_id or "provider-request-failed",
                                    None if pr is None else str(pr.number),
                                    redactor,
                                )
                            )
                        raise
                    try:
                        matches = await self.transport.find_pull_requests(
                            config,
                            base_ref=request.repository.base_ref,
                            head_ref=request.repository.head_ref,
                            limits=request.limits,
                        )
                        matches = self._redacted_prs(matches, redactor)
                    except GitHubProviderError:
                        if active_operation is not None:
                            operations.append(
                                self._operation(
                                    active_operation,
                                    "ambiguous",
                                    exc.request_id or "provider-ack-ambiguous",
                                    None if pr is None else str(pr.number),
                                    redactor,
                                )
                            )
                        return await self.repository.publish(
                            request,
                            self._result(
                                request,
                                GitHubDeliveryState.AMBIGUOUS,
                                approval=approval,
                                pr=pr,
                                poll_count=min(request.limits.max_polls, poll_count + 1),
                                operations=operations,
                                reason="provider_mutation_reconciliation_unavailable",
                            ),
                        )
                    candidate = matches[0] if len(matches) == 1 else None
                    reconciled = (
                        candidate is not None
                        and self._pr_matches(request, candidate)
                        and exc.code != "provider_extension_failure"
                    )
                    if active_operation is GitHubOperation.UPDATE_PULL_REQUEST:
                        reconciled = bool(
                            reconciled
                            and candidate.title == request.metadata.title
                            and candidate.body == request.metadata.body
                        )
                    elif active_operation is GitHubOperation.SET_LABELS:
                        reconciled = bool(
                            reconciled
                            and candidate is not None
                            and tuple(sorted(candidate.labels)) == request.metadata.labels
                        )
                    elif active_operation is GitHubOperation.MARK_READY:
                        reconciled = bool(
                            reconciled and candidate is not None and not candidate.draft
                        )
                    elif active_operation is GitHubOperation.REQUEST_REVIEWERS:
                        reconciled = False
                    if not reconciled or active_operation is None:
                        if active_operation is not None:
                            operations.append(
                                self._operation(
                                    active_operation,
                                    "ambiguous",
                                    exc.request_id or "provider-ack-ambiguous",
                                    None if candidate is None else str(candidate.number),
                                    redactor,
                                )
                            )
                        return await self.repository.publish(
                            request,
                            self._result(
                                request,
                                GitHubDeliveryState.AMBIGUOUS,
                                approval=approval,
                                pr=candidate,
                                poll_count=poll_count,
                                operations=operations,
                                reason="provider_mutation_acknowledgement_ambiguous",
                            ),
                        )
                    pr = matches[0]
                    operations.append(
                        self._operation(
                            active_operation,
                            "reconciled",
                            exc.request_id or "provider-ack-reconciled",
                            str(pr.number),
                            redactor,
                        )
                    )
                    active_operation = None
                    # A later requested effect may not have run. Publish progress
                    # and resume the remaining effects on the next bounded call.
                    return await self.repository.publish(
                        request,
                        self._result(
                            request,
                            GitHubDeliveryState.PR_UPDATED,
                            approval=approval,
                            pr=pr,
                            poll_count=poll_count,
                            operations=operations,
                            reason="provider_mutation_reconciled",
                        ),
                    )
            assert pr is not None
            expected_number = pr.number
            pr = await self.transport.get_pull_request(config, pr.number, request.limits)
            pr = self._redacted_pr(pr, redactor)
            if pr.number != expected_number:
                raise GitHubProviderError("pull_request_identity_mismatch")
            if not self._pr_matches(request, pr):
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.SUPERSEDED,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count,
                        operations=operations,
                        reason="pull_request_binding_changed",
                    ),
                )
            if pr.state == "closed":
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.CLOSED,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count,
                        operations=operations,
                        reason="pull_request_closed",
                    ),
                )
            if stop_requested is not None and stop_requested.is_set():
                raise _GitHubOwnerStopped()
            check_bundle = await self.transport.observe_checks(
                config, request.repository.head_commit, request.limits
            )
            review_bundle = await self.transport.observe_reviews(config, pr.number, request.limits)
            check_bundle = self._redacted_checks(check_bundle, redactor, request.limits)
            review_bundle = self._redacted_reviews(review_bundle, redactor, request.limits)
            observed_pr = self._redacted_pr(
                await self.transport.get_pull_request(config, pr.number, request.limits),
                redactor,
            )
            if observed_pr.number != pr.number or not self._pr_matches(request, observed_pr):
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.SUPERSEDED,
                        approval=approval,
                        pr=observed_pr,
                        poll_count=poll_count + 1,
                        operations=operations,
                        reason="pull_request_changed_during_observation",
                    ),
                )
            pr = observed_pr
            if pr.state == "closed":
                return await self.repository.publish(
                    request,
                    self._result(
                        request,
                        GitHubDeliveryState.CLOSED,
                        approval=approval,
                        pr=pr,
                        poll_count=poll_count + 1,
                        operations=operations,
                        reason="pull_request_closed_during_observation",
                    ),
                )
            bounded_feedback: list[GitHubFeedbackObservation] = []
            remaining_feedback_bytes = request.limits.max_feedback_bytes
            feedback_truncated = review_bundle.truncated
            for item in review_bundle.feedback:
                if remaining_feedback_bytes <= 0:
                    feedback_truncated = True
                    continue
                feedback_truncated = (
                    feedback_truncated or len(item.body.encode()) > remaining_feedback_bytes
                )
                bounded = redactor.redact_text_bounded(
                    item.body,
                    max_bytes=remaining_feedback_bytes,
                )
                remaining_feedback_bytes -= len(bounded.encode())
                bounded_feedback.append(
                    item.model_copy(
                        update={
                            "body": bounded,
                        }
                    )
                )
            review_bundle = review_bundle.model_copy(
                update={
                    "feedback": tuple(bounded_feedback),
                    "truncated": feedback_truncated,
                }
            )
            checks_state, review_state = (
                _check_state(request, check_bundle),
                _review_state(review_bundle, request.reviews, request.repository.head_commit),
            )
            poll_count += 1
            state = (
                GitHubDeliveryState.CHANGES_REQUESTED
                if review_state == GitHubReviewState.CHANGES_REQUESTED
                else GitHubDeliveryState.SUPERSEDED
                if checks_state == GitHubCheckState.SUPERSEDED
                else GitHubDeliveryState.CHECKS_FAILED
                if checks_state
                in {
                    GitHubCheckState.FAILED,
                    GitHubCheckState.CANCELLED,
                    GitHubCheckState.TIMED_OUT,
                    GitHubCheckState.PROVIDER_AMBIGUOUS,
                }
                else GitHubDeliveryState.APPROVED
                if review_state == GitHubReviewState.APPROVED
                else GitHubDeliveryState.CHECKS_PENDING
                if checks_state in {GitHubCheckState.PENDING, GitHubCheckState.MISSING}
                else GitHubDeliveryState.CHECKS_PASSED
            )
            if (
                request.reviews.approval_required
                and review_state != GitHubReviewState.APPROVED
                and state == GitHubDeliveryState.CHECKS_PASSED
            ):
                state = GitHubDeliveryState.CHECKS_PENDING
            if check_bundle.truncated or review_bundle.truncated:
                state = GitHubDeliveryState.PARTIAL
            return await self.repository.publish(
                request,
                self._result(
                    request,
                    state,
                    approval=approval,
                    pr=pr,
                    checks_state=checks_state,
                    review_state=review_state,
                    checks=check_bundle.checks,
                    feedback=review_bundle.feedback,
                    checks_truncated=check_bundle.truncated,
                    feedback_truncated=review_bundle.truncated,
                    operations=operations,
                    poll_count=poll_count,
                ),
            )
        except _GitHubOwnerStopped:
            return await self.repository.publish(
                request,
                self._result(
                    request,
                    GitHubDeliveryState.CANCELLED,
                    approval=approval,
                    pr=None if latest is None else latest.result.pull_request,
                    poll_count=poll_count,
                    operations=operations,
                    reason="caller_stopped_waiting",
                ),
            )
        except asyncio.CancelledError as signal:
            if active_operation is not None:
                operations.append(
                    self._operation(
                        active_operation,
                        "ambiguous",
                        "provider-cancelled-ambiguous",
                        None,
                        redactor,
                    )
                )
            state = (
                GitHubDeliveryState.AMBIGUOUS
                if active_operation is not None
                else GitHubDeliveryState.CANCELLED
            )
            await self._publish_preserving_signal(
                request,
                self._result(
                    request,
                    state,
                    approval=approval,
                    pr=None if latest is None else latest.result.pull_request,
                    poll_count=poll_count,
                    operations=operations,
                    reason=(
                        "provider_mutation_cancelled_ambiguous"
                        if active_operation is not None
                        else "caller_cancelled"
                    ),
                ),
                signal,
            )
            raise
        except GitHubProviderError as exc:
            state = (
                GitHubDeliveryState.RATE_LIMITED
                if exc.code == "rate_limited"
                else GitHubDeliveryState.PERMISSION_DENIED
                if exc.code == "permission_denied"
                else GitHubDeliveryState.AMBIGUOUS
                if exc.ambiguous
                else GitHubDeliveryState.PROVIDER_UNAVAILABLE
                if exc.retryable
                else GitHubDeliveryState.FAILED
            )
            return await self.repository.publish(
                request,
                self._result(
                    request,
                    state,
                    approval=approval,
                    poll_count=min(request.limits.max_polls, poll_count + 1),
                    operations=operations,
                    reason=exc.code,
                ),
            )


def github_follow_up_coding_input(
    result: GitHubDeliveryResult,
    *,
    provider_ids: Sequence[str],
    iteration: int,
    product_run_id: str,
    session_id: str,
    task_id: str,
) -> GitHubFollowUpCodingInput:
    if type(result) is not GitHubDeliveryResult:
        raise TypeError("result must be GitHubDeliveryResult.")
    if type(iteration) is not int:
        raise TypeError("iteration must be an integer.")
    if isinstance(provider_ids, str | bytes):
        raise TypeError("provider_ids must be a sequence of provider identities.")
    if not result.follow_up_allowed:
        raise GitHubDeliveryAdmissionError("Follow-up is disabled by application policy.")
    if iteration > result.max_follow_up_iterations:
        raise GitHubDeliveryAdmissionError("Follow-up iteration limit is exhausted.")
    if result.state not in {
        GitHubDeliveryState.CHECKS_FAILED,
        GitHubDeliveryState.CHANGES_REQUESTED,
    }:
        raise GitHubDeliveryAdmissionError("Follow-up requires failed checks or requested changes.")
    selected = tuple(sorted(set(provider_ids)))
    if len(selected) > result.max_follow_up_items:
        raise GitHubDeliveryAdmissionError("Follow-up evidence selection exceeds policy.")
    available_feedback = {
        item.provider_id: item for item in result.feedback if item.resolved is not True
    }
    available_checks = {
        item.provider_id: item
        for item in result.checks
        if item.conclusion not in {None, "success", "neutral", "skipped"}
    }
    if not selected or not set(selected) <= (set(available_feedback) | set(available_checks)):
        raise GitHubDeliveryAdmissionError(
            "Follow-up selection is not present in retained feedback."
        )
    messages = []
    for item in selected:
        if item in available_feedback:
            feedback = available_feedback[item]
            messages.append(
                f"Untrusted GitHub {feedback.kind} evidence from "
                f"{feedback.author_login} [{feedback.author_type}] ({item}), "
                f"recorded {feedback.created_at}, reviewed commit "
                f"{feedback.head_commit or 'unspecified'}, observed for {result.head_commit}:\n"
                f"{feedback.body}"
            )
        else:
            check = available_checks[item]
            messages.append(
                f"Untrusted GitHub check evidence ({item}) for exact head "
                f"{check.head_commit}: {check.name} concluded {check.conclusion}."
            )
    return GitHubFollowUpCodingInput(
        prior_connector_run_id=result.connector_run_id,
        prior_product_run_id=result.product_run_id,
        prior_delivery_id=result.remote_delivery_id,
        head_commit=result.head_commit,
        iteration=iteration,
        product_run_id=product_run_id,
        session_id=session_id,
        task_id=task_id,
        feedback_provider_ids=selected,
        messages=tuple(messages),
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("GitHub")
    or name
    in {
        "approve_github_delivery",
        "github_connector_behavior_fingerprint",
        "github_follow_up_coding_input",
        "github_pull_request_delivery_request",
        "GITHUB_DELIVERY_RESULT_KIND",
        "GITHUB_DELIVERY_SCHEMA_VERSION",
    }
]
