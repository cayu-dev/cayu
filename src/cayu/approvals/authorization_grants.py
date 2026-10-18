"""Durable reusable authorization grants for tool policies.

A grant is policy state recorded alongside one approved pending tool approval.
It never changes the binary approve/deny primitive: the originating call is
still approved by its own approval, and a later call is authorized only when a
:class:`~cayu.approvals.ToolAuthorizationGrantPolicy` finds an active grant
whose class of decision, scope, boundary, and matcher all apply.

The parts are independent. Records, matchers, and the store contract can back
an application-owned policy; the bundled policy only reads the store; the
service creates grants from the runtime's pending-approval snapshot; and the
server routes wrap the service.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.tools.policy import APPROVAL_GATE_METADATA_KEY
from cayu.vaults.redaction import REDACTED_SECRET

TOOL_AUTHORIZATION_GRANT_SCHEMA_VERSION = 1
#: Policy metadata naming the approval gate (class of decision) that required
#: approval. Gating policies set it; grants match it exactly.
TOOL_APPROVAL_GATE_METADATA_KEY = APPROVAL_GATE_METADATA_KEY
#: Policy metadata marking a mandatory control. Grants never apply to it.
TOOL_MANDATORY_APPROVAL_METADATA_KEY = "cayu:mandatory_approval"
# Marks the read-only re-authorization of arguments a person changed before
# approving: a grant policy never selects, uses or samples a grant for it.
APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY = "cayu:approved_arguments_reauthorization"
# The non-grantable offer reason a grant policy records for a mandatory control.
MANDATORY_CONTROL_OFFER_REASON = "mandatory_control"
#: Evidence on a policy result authorized (or sampled) by a grant, and the
#: marker on the approval request that created a grant.
TOOL_AUTHORIZATION_GRANT_METADATA_KEY = "cayu:authorization_grant"
#: Grantability recorded on a REQUIRE_APPROVAL result, and therefore on the
#: runtime's immutable pending-approval snapshot.
TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY = "cayu:authorization_grant_offer"

TOOL_AUTHORIZATION_GRANT_ID_PREFIX = "tag_"
TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS = 10_000
TOOL_AUTHORIZATION_GRANT_MAX_PINNED_ARGUMENTS = 16
TOOL_AUTHORIZATION_GRANT_MAX_REQUEST_DIGESTS = 8
TOOL_AUTHORIZATION_GRANT_MAX_BOUNDARY_LABELS = 16
TOOL_AUTHORIZATION_GRANT_MAX_LIST = 500
TOOL_AUTHORIZATION_GRANT_MAX_EVIDENCE_BYTES = 16_384
TOOL_AUTHORIZATION_GRANT_MAX_TEXT_BYTES = 2_048
TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS = 366 * 86_400
#: Shown instead of argument values copied from a call whose arguments were
#: not publishable.
REDACTED_ARGUMENT = REDACTED_SECRET

_PROCESS_COMMAND_KEYS = frozenset(
    {"kind", "argv", "cwd", "env", "stdin", "timeout_s", "max_output_bytes"}
)


class ToolAuthorizationGrantError(Exception):
    """Base class for authorization-grant failures."""


class ToolAuthorizationGrantConflict(ToolAuthorizationGrantError):
    """A request conflicts with the grant already recorded for its approval."""


class ToolAuthorizationGrantNotFound(ToolAuthorizationGrantError, KeyError):
    """No grant exists with the requested identity."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "Tool authorization grant not found."


class ToolAuthorizationGrantDenied(ToolAuthorizationGrantError):
    """The actor is not authorized to create or revoke this grant."""


class ToolAuthorizationGrantUnavailable(ToolAuthorizationGrantError):
    """The pending approval cannot back a grant (not grantable, resolved, or expired)."""


class ToolAuthorizationGrantScopeKind(StrEnum):
    SESSION = "session"
    WORKSPACE = "workspace"


class ToolAuthorizationGrantStatus(StrEnum):
    """Grant lifecycle. ``EXPIRED`` is observed, never stored."""

    STAGED = "staged"
    ACTIVE = "active"
    EXPIRED = "expired"
    ABANDONED = "abandoned"
    REVOKED = "revoked"


class ToolAuthorizationGrantMatcherKind(StrEnum):
    PROCESS_COMMAND = "process_command"
    TOOL_CALL = "tool_call"


class ToolAuthorizationGrantUseDecision(StrEnum):
    ALLOWED = "allowed"
    SAMPLED_REVIEW = "sampled_review"


class _GrantModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


def _bounded_text(value: str, field_name: str) -> str:
    value = require_durable_nonblank(value, field_name)
    if len(value.encode("utf-8")) > TOOL_AUTHORIZATION_GRANT_MAX_TEXT_BYTES:
        raise ValueError(
            f"{field_name} cannot exceed {TOOL_AUTHORIZATION_GRANT_MAX_TEXT_BYTES} UTF-8 bytes."
        )
    return value


def _identifier(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > TOOL_AUTHORIZATION_GRANT_MAX_TEXT_BYTES:
        raise ValueError(
            f"{field_name} cannot exceed {TOOL_AUTHORIZATION_GRANT_MAX_TEXT_BYTES} UTF-8 bytes."
        )
    return value


def _optional_identifier(value: str | None, field_name: str) -> str | None:
    return None if value is None else _identifier(value, field_name)


def normalize_grant_timestamp(value: datetime, field_name: str = "timestamp") -> datetime:
    """UTC with millisecond precision, so every store round-trips it exactly."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    value = value.astimezone(UTC)
    return value.replace(microsecond=value.microsecond - value.microsecond % 1000)


def _optional_timestamp(value: datetime | None, field_name: str) -> datetime | None:
    return None if value is None else normalize_grant_timestamp(value, field_name)


def _canonical(value: Any, field_name: str) -> bytes:
    return canonical_durable_json_bytes(value, field_name)


def _digest(*parts: str) -> str:
    return sha256("\0".join(parts).encode("utf-8")).hexdigest()


def tool_authorization_grant_id(*, session_id: str, approval_id: str) -> str:
    """Deterministic grant identity: one grant per originating approval."""

    session_id = require_durable_clean_nonblank(session_id, "session_id")
    approval_id = require_durable_clean_nonblank(approval_id, "approval_id")
    return (
        TOOL_AUTHORIZATION_GRANT_ID_PREFIX
        + _digest("cayu.tool_authorization_grant.v1", session_id, approval_id)[:40]
    )


def tool_authorization_grant_sample_bucket(
    *, grant_id: str, session_id: str, tool_call_id: str, tool_round_id: str | None = None
) -> int:
    """Reconstructable sampling bucket in ``[0, 10000)`` for one granted call.

    A call is sent to sampled review when its bucket is below the grant's
    ``sampled_review_rate_bps``. The bucket depends only on durable identity, so
    replay, recovery, and audit always reach the same decision.
    """

    material = _digest(
        "cayu.tool_authorization_grant.sample.v1",
        require_durable_clean_nonblank(grant_id, "grant_id"),
        require_durable_clean_nonblank(session_id, "session_id"),
        require_durable_clean_nonblank(tool_call_id, "tool_call_id"),
        ""
        if tool_round_id is None
        else require_durable_clean_nonblank(tool_round_id, "tool_round_id"),
    )
    return int(material[:16], 16) % TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS


def tool_arguments_digest(arguments: Mapping[str, Any]) -> str:
    """Canonical SHA-256 of a call's arguments, recorded as use evidence."""

    return "sha256:" + sha256(_canonical(dict(arguments), "arguments")).hexdigest()


class ToolAuthorizationGrantActor(_GrantModel):
    """Claims-free actor provenance recorded on grants."""

    subject: str
    tenant: str | None = None
    source: ResolutionActorSource | None = None

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        return _identifier(value, "subject")

    @field_validator("tenant")
    @classmethod
    def validate_tenant(cls, value: str | None) -> str | None:
        return _optional_identifier(value, "tenant")

    @classmethod
    def from_resolution_actor(
        cls, actor: ResolutionActor | None
    ) -> ToolAuthorizationGrantActor | None:
        if actor is None:
            return None
        if type(actor) is not ResolutionActor:
            raise TypeError("actor must be a ResolutionActor.")
        return cls(subject=actor.subject, tenant=actor.tenant, source=actor.source)


def _normalized_process_command(arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the exact process-command authority, or ``None`` when not one.

    Only an explicit argument vector qualifies. Shell text, unknown argument
    keys, and malformed values never normalize, so they cannot match a grant.
    ``timeout_s`` and ``max_output_bytes`` bound execution and are excluded.
    """

    if not isinstance(arguments, Mapping) or not set(arguments) <= _PROCESS_COMMAND_KEYS:
        return None
    if arguments.get("kind", "process") != "process":
        return None
    argv = arguments.get("argv")
    if type(argv) is not list or not argv:
        return None
    if any(type(item) is not str or not item.strip() for item in argv):
        return None
    cwd = arguments.get("cwd")
    if cwd is not None and type(cwd) is not str:
        return None
    env = arguments.get("env")
    if env is None:
        env = {}
    if type(env) is not dict or any(
        type(key) is not str or type(value) is not str for key, value in env.items()
    ):
        return None
    stdin = arguments.get("stdin")
    if stdin is not None and type(stdin) is not str:
        return None
    return {"argv": list(argv), "cwd": cwd, "env": dict(env), "stdin": stdin}


def _private_inputs_digest(env: Mapping[str, str], stdin: str | None) -> str:
    material = {"env": dict(env), "stdin": stdin}
    return "sha256:" + sha256(_canonical(material, "process inputs")).hexdigest()


class ProcessCommandGrantMatcher(_GrantModel):
    """Authorize one exact process command: argv, cwd, env, and stdin.

    There are no prefixes, globs, substrings, or shell parsing. A request that
    is not an explicit process command never matches. Environment and stdin
    often carry credentials, so the grant keeps only their digest.
    """

    kind: Literal["process_command"] = "process_command"
    version: Literal[1] = 1
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str | None = None
    private_inputs_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_command(self) -> ProcessCommandGrantMatcher:
        if _normalized_process_command({"argv": list(self.argv), "cwd": self.cwd}) is None:
            raise ValueError("Process command matcher must describe an explicit argv command.")
        _canonical({"argv": list(self.argv), "cwd": self.cwd}, "matcher")
        return self

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> ProcessCommandGrantMatcher:
        normalized = _normalized_process_command(arguments)
        if normalized is None:
            raise ToolAuthorizationGrantUnavailable(
                "Only an explicit process command (argv) can back a process-command grant."
            )
        return cls(
            argv=tuple(normalized["argv"]),
            cwd=normalized["cwd"],
            private_inputs_digest=_private_inputs_digest(normalized["env"], normalized["stdin"]),
        )

    def matches(self, arguments: Mapping[str, Any]) -> bool:
        normalized = _normalized_process_command(arguments)
        return (
            normalized is not None
            and normalized["argv"] == list(self.argv)
            and normalized["cwd"] == self.cwd
            and _private_inputs_digest(normalized["env"], normalized["stdin"])
            == self.private_inputs_digest
        )

    def redacted(self) -> ProcessCommandGrantMatcher:
        return self.model_copy(
            update={
                "argv": tuple(REDACTED_ARGUMENT for _ in self.argv),
                "cwd": None if self.cwd is None else REDACTED_ARGUMENT,
            }
        )


def _argument_path(path: str) -> tuple[str, ...]:
    path = require_durable_clean_nonblank(path, "argument path")
    parts = tuple(path.split("."))
    if any(not part or part != part.strip() for part in parts):
        raise ValueError("Argument paths must be dotted names without empty parts.")
    return parts


_MISSING = object()


def _argument_value(arguments: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = arguments
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


class ToolCallGrantMatcher(_GrantModel):
    """Authorize a class of calls to one tool, optionally pinning exact values.

    Each pinned argument path must equal the value it had on the originating
    approval (type-strict JSON equality). Values come from the runtime's pending
    snapshot, never from the client that creates the grant.
    """

    kind: Literal["tool_call"] = "tool_call"
    version: Literal[1] = 1
    pinned_arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pinned_arguments", mode="before")
    @classmethod
    def validate_pinned_arguments(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("pinned_arguments must be a mapping of argument paths to values.")
        if len(value) > TOOL_AUTHORIZATION_GRANT_MAX_PINNED_ARGUMENTS:
            raise ValueError(
                "pinned_arguments cannot contain more than "
                f"{TOOL_AUTHORIZATION_GRANT_MAX_PINNED_ARGUMENTS} paths."
            )
        copied: dict[str, Any] = {}
        for path, pinned in value.items():
            _argument_path(path)
            copied[path] = copy_durable_json_value(pinned, f"pinned_arguments[{path!r}]")
        return copied

    @classmethod
    def from_arguments(
        cls, arguments: Mapping[str, Any], paths: Iterable[str] = ()
    ) -> ToolCallGrantMatcher:
        if isinstance(paths, str):
            raise TypeError("pinned argument paths must be an iterable of strings.")
        pinned: dict[str, Any] = {}
        for path in paths:
            value = _argument_value(arguments, _argument_path(path))
            if value is _MISSING:
                raise ToolAuthorizationGrantUnavailable(
                    f"Pinned argument path is absent from the approved call: {path}"
                )
            pinned[path] = value
        return cls(pinned_arguments=pinned)

    def redacted(self) -> ToolCallGrantMatcher:
        return self.model_copy(
            update={"pinned_arguments": dict.fromkeys(self.pinned_arguments, REDACTED_ARGUMENT)}
        )

    def matches(self, arguments: Mapping[str, Any]) -> bool:
        for path, pinned in self.pinned_arguments.items():
            value = _argument_value(arguments, _argument_path(path))
            if value is _MISSING:
                return False
            try:
                if _canonical(value, "argument") != _canonical(pinned, "pinned"):
                    return False
            except Exception:
                return False
        return True


ToolAuthorizationGrantMatcher = Annotated[
    ProcessCommandGrantMatcher | ToolCallGrantMatcher,
    Field(discriminator="kind"),
]


def grant_matcher_from_arguments(
    kind: ToolAuthorizationGrantMatcherKind,
    arguments: Mapping[str, Any],
    *,
    pinned_argument_paths: Iterable[str] = (),
) -> ProcessCommandGrantMatcher | ToolCallGrantMatcher:
    """Build the matcher a policy offered from snapshot arguments."""

    kind = ToolAuthorizationGrantMatcherKind(kind)
    pinned = tuple(pinned_argument_paths)
    if kind is ToolAuthorizationGrantMatcherKind.PROCESS_COMMAND:
        if pinned:
            raise ValueError("Process-command grants already match the exact command.")
        return ProcessCommandGrantMatcher.from_arguments(arguments)
    return ToolCallGrantMatcher.from_arguments(arguments, pinned)


class ToolAuthorizationGrantControl(_GrantModel):
    """A mandatory control: a tool, optionally narrowed to one approval gate.

    ``approval_gate`` is the identity a gating policy records under
    ``cayu:approval_gate``; ``None`` covers every gate on the tool. Calls that
    match a mandatory control always go to a person and can never be granted.
    A project objectives document is the expected source of these controls.
    """

    tool_name: str
    approval_gate: str | None = None

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _identifier(value, "tool_name")

    @field_validator("approval_gate")
    @classmethod
    def validate_gate(cls, value: str | None) -> str | None:
        return _optional_identifier(value, "approval_gate")

    def covers(self, *, tool_name: str, approval_gate: str | None) -> bool:
        return self.tool_name == tool_name and (
            self.approval_gate is None or self.approval_gate == approval_gate
        )


def copy_mandatory_controls(value: Iterable[Any]) -> tuple[ToolAuthorizationGrantControl, ...]:
    """Validate mandatory controls given as models or ``{tool_name, approval_gate}`` mappings."""

    if isinstance(value, str | bytes | Mapping | BaseModel):
        raise TypeError("mandatory_controls must be an iterable of controls.")
    controls = tuple(
        ToolAuthorizationGrantControl.model_validate(
            item.model_dump() if isinstance(item, ToolAuthorizationGrantControl) else item
        )
        for item in value
    )
    return tuple(dict.fromkeys(controls))


def is_mandatory_control(
    controls: Iterable[ToolAuthorizationGrantControl],
    *,
    tool_name: str,
    approval_gate: str | None,
) -> bool:
    return any(
        control.covers(tool_name=tool_name, approval_gate=approval_gate) for control in controls
    )


class ToolAuthorizationGrantScope(_GrantModel):
    """Where a grant applies: one session, or later sessions in one workspace."""

    kind: ToolAuthorizationGrantScopeKind
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "scope.id")


class ToolAuthorizationGrantPrincipal(_GrantModel):
    """The application authorization identity a session runs under.

    Taken from the session's resource-access binding, which only trusted
    application access control produces.
    """

    authority: str
    subject: str

    @field_validator("authority", "subject")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)


class ToolAuthorizationGrantBoundary(_GrantModel):
    """Authorization boundary captured when the originating call paused.

    A grant applies only to a request with exactly the same agent, environment,
    workspace, principal, and boundary-label values. ``principal`` is the
    session's resource-access identity (absent matches only absent). Boundary
    labels are session labels the application chose to require; clients can
    set labels, so they narrow a grant but are not an authority boundary.
    ``workspace_explicit`` records whether the application named the workspace;
    only such a workspace can carry workspace-scoped grants.
    """

    agent_name: str
    environment_name: str | None = None
    workspace_id: str | None = None
    workspace_explicit: bool = False
    principal: ToolAuthorizationGrantPrincipal | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, value: str) -> str:
        return _identifier(value, "agent_name")

    @field_validator("environment_name", "workspace_id")
    @classmethod
    def validate_optional(cls, value: str | None, info) -> str | None:
        return _optional_identifier(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise TypeError("labels must be a mapping of strings.")
        if len(value) > TOOL_AUTHORIZATION_GRANT_MAX_BOUNDARY_LABELS:
            raise ValueError("Too many boundary labels.")
        return {
            _identifier(key, "label key"): _identifier(label, "label value")
            for key, label in value.items()
        }


class ToolAuthorizationGrantOrigin(_GrantModel):
    """Identity of the pending approval that created the grant.

    ``session_instance_id`` is the originating session's incarnation. Session
    ids can be reused after deletion, so a session grant applies only to the
    incarnation that approved it.
    """

    session_id: str
    session_instance_id: str
    approval_id: str
    tool_round_id: str
    tool_call_id: str

    @field_validator(
        "session_id", "session_instance_id", "approval_id", "tool_round_id", "tool_call_id"
    )
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)


class ToolAuthorizationGrantOffer(_GrantModel):
    """Grantability a grant policy recorded on a REQUIRE_APPROVAL decision.

    The runtime persists policy metadata on the pending-approval checkpoint,
    so grant creation reads this offer from the immutable snapshot instead of
    trusting a client's description of the call.
    """

    schema_version: Literal[1] = TOOL_AUTHORIZATION_GRANT_SCHEMA_VERSION
    policy_id: str
    grantable: bool
    reason: str | None = None
    tool_name: str | None = None
    approval_gate: str | None = None
    matcher_kind: ToolAuthorizationGrantMatcherKind | None = None
    session_id: str | None = None
    boundary: ToolAuthorizationGrantBoundary | None = None

    @field_validator("policy_id")
    @classmethod
    def validate_policy_id(cls, value: str) -> str:
        return _identifier(value, "policy_id")

    @field_validator("reason", "tool_name", "approval_gate", "session_id")
    @classmethod
    def validate_optional(cls, value: str | None, info) -> str | None:
        return _optional_identifier(value, info.field_name)

    @model_validator(mode="after")
    def validate_offer(self) -> ToolAuthorizationGrantOffer:
        complete = (
            self.tool_name is not None
            and self.matcher_kind is not None
            and self.session_id is not None
            and self.boundary is not None
        )
        if self.grantable and not complete:
            raise ValueError("A grantable offer must identify its tool, matcher, and boundary.")
        if not self.grantable and self.reason is None:
            raise ValueError("A non-grantable offer must state its reason.")
        return self


class ToolAuthorizationGrant(_GrantModel):
    """Typed, versioned durable grant record.

    ``status`` is stored lifecycle; :meth:`effective_status` adds deterministic
    expiry. Only ``ACTIVE`` grants that have not reached ``expires_at`` can
    authorize a call. ``STAGED`` grants are written before their approval is
    resolved and stay inert until that approval is durably approved by one of
    ``resolution_request_digests``.
    """

    schema_version: Literal[1] = TOOL_AUTHORIZATION_GRANT_SCHEMA_VERSION
    grant_id: str
    policy_id: str
    tool_name: str
    approval_gate: str | None = None
    matcher: ToolAuthorizationGrantMatcher
    #: Whether the originating approval allowed its arguments to be published.
    #: When ``False``, public views redact argument values copied into ``matcher``.
    arguments_public: bool = False
    scope: ToolAuthorizationGrantScope
    boundary: ToolAuthorizationGrantBoundary
    origin: ToolAuthorizationGrantOrigin
    resolution_request_digests: tuple[str, ...] = Field(
        min_length=1, max_length=TOOL_AUTHORIZATION_GRANT_MAX_REQUEST_DIGESTS
    )
    created_by: ToolAuthorizationGrantActor | None = None
    created_at: datetime
    lifetime_seconds: StrictInt | None = Field(
        default=None, ge=1, le=TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS
    )
    expires_at: datetime | None = None
    sampled_review_rate_bps: StrictInt = Field(
        default=0, ge=0, le=TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS
    )
    justification: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: ToolAuthorizationGrantStatus = ToolAuthorizationGrantStatus.STAGED
    activated_at: datetime | None = None
    resolved_by: ToolAuthorizationGrantActor | None = None
    revoked_at: datetime | None = None
    revoked_by: ToolAuthorizationGrantActor | None = None
    revocation_reason: str | None = None
    abandoned_at: datetime | None = None
    abandonment_reason: str | None = None

    @field_validator("grant_id", "policy_id", "tool_name")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("approval_gate")
    @classmethod
    def validate_gate(cls, value: str | None) -> str | None:
        return _optional_identifier(value, "approval_gate")

    @field_validator("resolution_request_digests")
    @classmethod
    def validate_digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        copied = tuple(_identifier(item, "resolution_request_digest") for item in value)
        if len(set(copied)) != len(copied):
            raise ValueError("resolution_request_digests must be unique.")
        return copied

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return normalize_grant_timestamp(value, "created_at")

    @field_validator("expires_at", "activated_at", "revoked_at", "abandoned_at")
    @classmethod
    def validate_optional_timestamps(cls, value: datetime | None, info) -> datetime | None:
        return _optional_timestamp(value, info.field_name)

    @field_validator("justification", "revocation_reason", "abandonment_reason")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _bounded_text(value, info.field_name)

    @field_validator("evidence", mode="before")
    @classmethod
    def validate_evidence(cls, value: Any) -> dict[str, Any]:
        copied = copy_durable_json_value(value, "evidence")
        if type(copied) is not dict:
            raise TypeError("evidence must be a JSON object.")
        if len(_canonical(copied, "evidence")) > TOOL_AUTHORIZATION_GRANT_MAX_EVIDENCE_BYTES:
            raise ValueError(
                f"evidence cannot exceed {TOOL_AUTHORIZATION_GRANT_MAX_EVIDENCE_BYTES} bytes."
            )
        return copied

    @model_validator(mode="after")
    def validate_record(self) -> ToolAuthorizationGrant:
        if self.grant_id != tool_authorization_grant_id(
            session_id=self.origin.session_id, approval_id=self.origin.approval_id
        ):
            raise ValueError("grant_id must be derived from the originating approval.")
        if self.status is ToolAuthorizationGrantStatus.EXPIRED:
            raise ValueError("Expiry is observed from expires_at and is never stored.")
        expected_expiry = (
            None
            if self.lifetime_seconds is None
            else self.created_at + timedelta(seconds=self.lifetime_seconds)
        )
        if self.expires_at != expected_expiry:
            raise ValueError("expires_at must equal created_at plus lifetime_seconds.")
        if (
            self.scope.kind is ToolAuthorizationGrantScopeKind.SESSION
            and self.scope.id != self.origin.session_id
        ):
            raise ValueError("A session grant must be scoped to its originating session.")
        if self.scope.kind is ToolAuthorizationGrantScopeKind.WORKSPACE and (
            self.scope.id != self.boundary.workspace_id or not self.boundary.workspace_explicit
        ):
            raise ValueError(
                "A workspace grant must be scoped to its explicitly identified workspace."
            )
        activated = self.activated_at is not None
        revoked = self.revoked_at is not None
        abandoned = self.abandoned_at is not None
        if (self.status is ToolAuthorizationGrantStatus.ACTIVE) != (activated and not revoked):
            raise ValueError("Only an activated, unrevoked grant can be active.")
        if (self.status is ToolAuthorizationGrantStatus.REVOKED) != revoked:
            raise ValueError("A revoked grant must record its revocation.")
        if (self.status is ToolAuthorizationGrantStatus.ABANDONED) != abandoned:
            raise ValueError("An abandoned grant must record why.")
        if abandoned and activated:
            raise ValueError("An activated grant cannot be abandoned.")
        if self.resolved_by is not None and not activated:
            raise ValueError("resolved_by is recorded on activation.")
        if (self.revocation_reason is not None or self.revoked_by is not None) and not revoked:
            raise ValueError("Revocation details require revoked_at.")
        if (self.abandonment_reason is None) == abandoned:
            raise ValueError("abandonment_reason is required exactly when abandoned.")
        return self

    def effective_status(self, now: datetime) -> ToolAuthorizationGrantStatus:
        """Stored status, with ``EXPIRED`` once an active grant reaches its expiry."""

        now = normalize_grant_timestamp(now, "now")
        if (
            self.status is ToolAuthorizationGrantStatus.ACTIVE
            and self.expires_at is not None
            and now >= self.expires_at
        ):
            return ToolAuthorizationGrantStatus.EXPIRED
        return self.status

    def is_active(self, now: datetime) -> bool:
        return self.effective_status(now) is ToolAuthorizationGrantStatus.ACTIVE

    def public_view(self) -> ToolAuthorizationGrant:
        """This grant with non-publishable argument values redacted."""

        if self.arguments_public:
            return self
        return self.model_copy(update={"matcher": self.matcher.redacted()})

    def evidence_payload(self) -> dict[str, Any]:
        """The grant identity and scope recorded wherever the grant decides a call."""

        return {
            "grant_id": self.grant_id,
            "policy_id": self.policy_id,
            "scope_kind": self.scope.kind.value,
            "scope_id": self.scope.id,
            "origin_session_id": self.origin.session_id,
            "origin_approval_id": self.origin.approval_id,
        }


def grant_terms(grant: ToolAuthorizationGrant) -> bytes:
    """Fields that define what a grant authorizes and who asked for it.

    Two requests for the same approval are idempotent exactly when these agree.
    """

    return _canonical(
        grant.model_dump(
            mode="json",
            exclude={
                "created_at",
                "expires_at",
                "resolution_request_digests",
                "status",
                "activated_at",
                "resolved_by",
                "revoked_at",
                "revoked_by",
                "revocation_reason",
                "abandoned_at",
                "abandonment_reason",
            },
        ),
        "grant",
    )


class ToolAuthorizationGrantUse(_GrantModel):
    """Durable evidence that a grant decided one tool call.

    Written before the policy returns, so the decision, its grant, scope, and
    sampling bucket can be reconstructed after a restart.
    """

    schema_version: Literal[1] = TOOL_AUTHORIZATION_GRANT_SCHEMA_VERSION
    grant_id: str
    policy_id: str
    scope: ToolAuthorizationGrantScope
    session_id: str
    #: The session incarnation and tool round that executed the call. With the
    #: call id they identify one logical invocation; a provider may reuse a call
    #: id in a later round.
    session_instance_id: str
    tool_round_id: str | None = None
    tool_call_id: str
    tool_name: str
    arguments_digest: str
    decision: ToolAuthorizationGrantUseDecision
    sample_bucket: StrictInt = Field(ge=0, lt=TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS)
    sampled_review_rate_bps: StrictInt = Field(ge=0, le=TOOL_AUTHORIZATION_GRANT_SAMPLE_BUCKETS)
    recorded_at: datetime

    @field_validator(
        "grant_id",
        "policy_id",
        "session_id",
        "session_instance_id",
        "tool_call_id",
        "tool_name",
        "arguments_digest",
    )
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("tool_round_id")
    @classmethod
    def validate_round(cls, value: str | None) -> str | None:
        return _optional_identifier(value, "tool_round_id")

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        return normalize_grant_timestamp(value, "recorded_at")

    @model_validator(mode="after")
    def validate_sampling(self) -> ToolAuthorizationGrantUse:
        sampled = self.sample_bucket < self.sampled_review_rate_bps
        if sampled != (self.decision is ToolAuthorizationGrantUseDecision.SAMPLED_REVIEW):
            raise ValueError("Use decision must follow its sampling bucket and rate.")
        return self

    @property
    def use_key(self) -> tuple[str, str, str, str, str, str]:
        """One logical invocation; retries of it share the key."""

        return (
            self.grant_id,
            self.session_id,
            self.session_instance_id,
            self.tool_round_id or "",
            self.tool_call_id,
            self.arguments_digest,
        )


class ToolAuthorizationGrantQuery(_GrantModel):
    """Exact-filter listing. Results are ordered by ``(created_at, grant_id)``."""

    policy_id: str | None = None
    tool_name: str | None = None
    scope_kind: ToolAuthorizationGrantScopeKind | None = None
    scope_id: str | None = None
    origin_session_id: str | None = None
    statuses: tuple[ToolAuthorizationGrantStatus, ...] | None = None
    limit: StrictInt = Field(default=100, ge=1, le=TOOL_AUTHORIZATION_GRANT_MAX_LIST)
    #: Exclusive ``(created_at, grant_id)`` position to continue a listing after.
    after_created_at: datetime | None = None
    after_grant_id: str | None = None

    @field_validator("policy_id", "tool_name", "scope_id", "origin_session_id", "after_grant_id")
    @classmethod
    def validate_optional(cls, value: str | None, info) -> str | None:
        return _optional_identifier(value, info.field_name)

    @field_validator("after_created_at")
    @classmethod
    def validate_after(cls, value: datetime | None) -> datetime | None:
        return _optional_timestamp(value, "after_created_at")

    @model_validator(mode="after")
    def validate_cursor(self) -> ToolAuthorizationGrantQuery:
        if (self.after_created_at is None) != (self.after_grant_id is None):
            raise ValueError("after_created_at and after_grant_id are set together.")
        return self

    def after(self) -> tuple[datetime, str] | None:
        if self.after_created_at is None or self.after_grant_id is None:
            return None
        return self.after_created_at, self.after_grant_id

    def next_page(
        self, page: tuple[ToolAuthorizationGrant, ...]
    ) -> ToolAuthorizationGrantQuery | None:
        """The query for the following page, or ``None`` when ``page`` was the last."""

        after = self.after()
        if after is not None and any(grant_order_key(grant) <= after for grant in page):
            raise RuntimeError("The grant store returned records before the listing cursor.")
        if len(page) < self.limit:
            return None
        last = page[-1]
        return self.model_copy(
            update={"after_created_at": last.created_at, "after_grant_id": last.grant_id}
        )

    @field_validator("statuses")
    @classmethod
    def validate_statuses(
        cls, value: tuple[ToolAuthorizationGrantStatus, ...] | None
    ) -> tuple[ToolAuthorizationGrantStatus, ...] | None:
        if value is not None and ToolAuthorizationGrantStatus.EXPIRED in value:
            raise ValueError("Filter on ACTIVE and observe expiry with effective_status().")
        return value

    def matches(self, grant: ToolAuthorizationGrant) -> bool:
        after = self.after()
        return (
            (self.policy_id is None or grant.policy_id == self.policy_id)
            and (self.tool_name is None or grant.tool_name == self.tool_name)
            and (self.scope_kind is None or grant.scope.kind is self.scope_kind)
            and (self.scope_id is None or grant.scope.id == self.scope_id)
            and (
                self.origin_session_id is None or grant.origin.session_id == self.origin_session_id
            )
            and (self.statuses is None or grant.status in self.statuses)
            and (after is None or grant_order_key(grant) > after)
        )


def grant_order_key(grant: ToolAuthorizationGrant) -> tuple[datetime, str]:
    return (grant.created_at, grant.grant_id)


class ToolAuthorizationGrantStore(ABC):
    """Durable grant state. Every implementation shares the transition rules in
    :mod:`cayu.approvals.authorization_grants` so matching, idempotency, expiry,
    revocation, and ordering agree across backends.
    """

    @abstractmethod
    async def stage(self, grant: ToolAuthorizationGrant) -> ToolAuthorizationGrant:
        """Insert a staged grant, or return the one already recorded for its approval.

        Matching terms are idempotent: a new resolution-request digest is added
        to a still-staged grant. Different terms raise
        :class:`ToolAuthorizationGrantConflict` and change nothing.
        """

    @abstractmethod
    async def get(self, grant_id: str) -> ToolAuthorizationGrant | None: ...

    @abstractmethod
    async def list(self, query: ToolAuthorizationGrantQuery) -> tuple[ToolAuthorizationGrant, ...]:
        """Return matching grants ordered by ``(created_at, grant_id)``.

        At most ``query.limit`` records, all strictly after the exclusive
        ``(after_created_at, after_grant_id)`` cursor when it is set. Callers
        page with :meth:`ToolAuthorizationGrantQuery.next_page`, which refuses a
        page that does not advance past the cursor.
        """

    @abstractmethod
    async def activate(
        self,
        grant_id: str,
        *,
        resolved_by: ToolAuthorizationGrantActor | None,
        activated_at: datetime,
    ) -> ToolAuthorizationGrant:
        """Make a staged grant active. Idempotent for an already-active grant."""

    @abstractmethod
    async def abandon(
        self,
        grant_id: str,
        *,
        reason: str,
        abandoned_at: datetime,
        expected_request_digests: tuple[str, ...] | None = None,
    ) -> ToolAuthorizationGrant:
        """Retire a staged grant whose approval was not approved with it.

        With ``expected_request_digests``, the grant must still carry exactly those
        approval requests; otherwise :class:`ToolAuthorizationGrantConflict` is raised
        and nothing changes.
        """

    @abstractmethod
    async def revoke(
        self,
        grant_id: str,
        *,
        revoked_by: ToolAuthorizationGrantActor | None,
        reason: str | None,
        revoked_at: datetime,
    ) -> ToolAuthorizationGrant:
        """Revoke a staged or active grant. Revoking again returns the first revocation."""

    @abstractmethod
    async def record_use(self, use: ToolAuthorizationGrantUse) -> ToolAuthorizationGrantUse:
        """Insert use evidence; the first record for a use key wins."""

    @abstractmethod
    async def list_uses(
        self,
        grant_id: str,
        *,
        limit: int = 100,
        after: ToolAuthorizationGrantUseCursor | None = None,
    ) -> tuple[ToolAuthorizationGrantUse, ...]:
        """Return at most ``limit`` uses ordered by :func:`use_order_key`.

        When ``after`` is set, only uses strictly after that position are returned;
        continue with ``ToolAuthorizationGrantUseCursor.after(page[-1])``.
        """

    async def close(self) -> None:
        return None


# Shared transition rules -----------------------------------------------------


def staged_grant_transition(
    existing: ToolAuthorizationGrant | None, requested: ToolAuthorizationGrant
) -> tuple[ToolAuthorizationGrant, bool]:
    """Return ``(record, changed)`` for a stage request."""

    if requested.status is not ToolAuthorizationGrantStatus.STAGED:
        raise ValueError("Only staged grants can be written by stage().")
    if existing is None:
        return requested, True
    if grant_terms(existing) != grant_terms(requested):
        raise ToolAuthorizationGrantConflict(
            "A different grant was already requested for this approval."
        )
    if existing.status is not ToolAuthorizationGrantStatus.STAGED:
        return existing, False
    added = tuple(
        digest
        for digest in requested.resolution_request_digests
        if digest not in existing.resolution_request_digests
    )
    if not added:
        return existing, False
    digests = existing.resolution_request_digests + added
    if len(digests) > TOOL_AUTHORIZATION_GRANT_MAX_REQUEST_DIGESTS:
        raise ToolAuthorizationGrantConflict("Too many approval attempts for this grant.")
    return existing.model_copy(update={"resolution_request_digests": digests}), True


def activated_grant_transition(
    existing: ToolAuthorizationGrant | None,
    *,
    grant_id: str,
    resolved_by: ToolAuthorizationGrantActor | None,
    activated_at: datetime,
) -> tuple[ToolAuthorizationGrant, bool]:
    grant = _require_grant(existing, grant_id)
    if grant.status is ToolAuthorizationGrantStatus.ACTIVE:
        return grant, False
    if grant.status is not ToolAuthorizationGrantStatus.STAGED:
        raise ToolAuthorizationGrantConflict(f"A {grant.status.value} grant cannot be activated.")
    return _revalidated(
        grant,
        status=ToolAuthorizationGrantStatus.ACTIVE,
        activated_at=normalize_grant_timestamp(activated_at, "activated_at"),
        resolved_by=resolved_by,
    ), True


def abandoned_grant_transition(
    existing: ToolAuthorizationGrant | None,
    *,
    grant_id: str,
    reason: str,
    abandoned_at: datetime,
    expected_request_digests: tuple[str, ...] | None = None,
) -> tuple[ToolAuthorizationGrant, bool]:
    grant = _require_grant(existing, grant_id)
    if grant.status is ToolAuthorizationGrantStatus.ABANDONED:
        return grant, False
    if grant.status is not ToolAuthorizationGrantStatus.STAGED:
        raise ToolAuthorizationGrantConflict(f"A {grant.status.value} grant cannot be abandoned.")
    if expected_request_digests is not None and grant.resolution_request_digests != tuple(
        expected_request_digests
    ):
        # Another approval request was added since the caller looked; reclassify.
        raise ToolAuthorizationGrantConflict(
            "The grant's approval requests changed; reconcile it again."
        )
    return _revalidated(
        grant,
        status=ToolAuthorizationGrantStatus.ABANDONED,
        abandoned_at=normalize_grant_timestamp(abandoned_at, "abandoned_at"),
        abandonment_reason=reason,
    ), True


def revoked_grant_transition(
    existing: ToolAuthorizationGrant | None,
    *,
    grant_id: str,
    revoked_by: ToolAuthorizationGrantActor | None,
    reason: str | None,
    revoked_at: datetime,
) -> tuple[ToolAuthorizationGrant, bool]:
    grant = _require_grant(existing, grant_id)
    if grant.status is ToolAuthorizationGrantStatus.REVOKED:
        return grant, False
    if grant.status is ToolAuthorizationGrantStatus.ABANDONED:
        raise ToolAuthorizationGrantConflict("An abandoned grant cannot be revoked.")
    return _revalidated(
        grant,
        status=ToolAuthorizationGrantStatus.REVOKED,
        revoked_at=normalize_grant_timestamp(revoked_at, "revoked_at"),
        revoked_by=revoked_by,
        revocation_reason=reason,
    ), True


def _require_grant(grant: ToolAuthorizationGrant | None, grant_id: str) -> ToolAuthorizationGrant:
    if grant is None:
        raise ToolAuthorizationGrantNotFound(f"Tool authorization grant not found: {grant_id}")
    return grant


def _revalidated(grant: ToolAuthorizationGrant, **update: Any) -> ToolAuthorizationGrant:
    payload = grant.model_dump(mode="python")
    payload.update(update)
    return ToolAuthorizationGrant.model_validate(payload)


def use_order_key(use: ToolAuthorizationGrantUse) -> tuple[datetime, str, str, str, str, str]:
    return (use.recorded_at, *use.use_key[1:])


class ToolAuthorizationGrantUseCursor(_GrantModel):
    """The exclusive position to continue a use listing after.

    Uses are ordered by ``recorded_at`` and then their complete identity, so
    records with equal timestamps are paged exactly once.
    """

    recorded_at: datetime
    session_id: str
    session_instance_id: str
    tool_round_id: str
    tool_call_id: str
    arguments_digest: str

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        return normalize_grant_timestamp(value, "recorded_at")

    @field_validator("session_id", "session_instance_id", "tool_call_id", "arguments_digest")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @classmethod
    def after(cls, use: ToolAuthorizationGrantUse) -> ToolAuthorizationGrantUseCursor:
        _, session_id, session_instance_id, tool_round_id, tool_call_id, digest = use.use_key
        return cls(
            recorded_at=use.recorded_at,
            session_id=session_id,
            session_instance_id=session_instance_id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            arguments_digest=digest,
        )

    def key(self) -> tuple[datetime, str, str, str, str, str]:
        return (
            self.recorded_at,
            self.session_id,
            self.session_instance_id,
            self.tool_round_id,
            self.tool_call_id,
            self.arguments_digest,
        )


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def grant_timestamp_ms(value: datetime) -> int:
    """Exact UTC milliseconds for indexed store columns."""

    return (normalize_grant_timestamp(value) - _EPOCH) // timedelta(milliseconds=1)


def grant_record_json(grant: ToolAuthorizationGrant) -> str:
    return grant.model_dump_json()


def grant_from_record_json(value: str | bytes) -> ToolAuthorizationGrant:
    return ToolAuthorizationGrant.model_validate_json(value)


def use_record_json(use: ToolAuthorizationGrantUse) -> str:
    return use.model_dump_json()


def use_from_record_json(value: str | bytes) -> ToolAuthorizationGrantUse:
    return ToolAuthorizationGrantUse.model_validate_json(value)


def require_grant_list_limit(limit: int) -> int:
    if type(limit) is not int or not 1 <= limit <= TOOL_AUTHORIZATION_GRANT_MAX_LIST:
        raise ValueError(
            f"limit must be an integer between 1 and {TOOL_AUTHORIZATION_GRANT_MAX_LIST}."
        )
    return limit


def _detached_grant(grant: ToolAuthorizationGrant) -> ToolAuthorizationGrant:
    return grant_from_record_json(grant_record_json(grant))


def _detached_use(use: ToolAuthorizationGrantUse) -> ToolAuthorizationGrantUse:
    return use_from_record_json(use_record_json(use))


class InMemoryToolAuthorizationGrantStore(ToolAuthorizationGrantStore):
    """Process-local grant store with the same rules as the durable stores.

    Like the durable stores, it never shares a record with its callers: every
    write keeps its own copy and every read returns a fresh one.
    """

    def __init__(self) -> None:
        self._grants: dict[str, ToolAuthorizationGrant] = {}
        self._uses: dict[tuple[str, str, str, str, str, str], ToolAuthorizationGrantUse] = {}

    async def stage(self, grant: ToolAuthorizationGrant) -> ToolAuthorizationGrant:
        record, changed = staged_grant_transition(
            self._grants.get(grant.grant_id), _detached_grant(grant)
        )
        if changed:
            self._grants[record.grant_id] = record
        return _detached_grant(record)

    async def get(self, grant_id: str) -> ToolAuthorizationGrant | None:
        grant = self._grants.get(_identifier(grant_id, "grant_id"))
        return None if grant is None else _detached_grant(grant)

    async def list(self, query: ToolAuthorizationGrantQuery) -> tuple[ToolAuthorizationGrant, ...]:
        matching = sorted(
            (grant for grant in self._grants.values() if query.matches(grant)),
            key=grant_order_key,
        )
        return tuple(_detached_grant(grant) for grant in matching[: query.limit])

    async def activate(
        self,
        grant_id: str,
        *,
        resolved_by: ToolAuthorizationGrantActor | None,
        activated_at: datetime,
    ) -> ToolAuthorizationGrant:
        record, changed = activated_grant_transition(
            self._grants.get(grant_id),
            grant_id=grant_id,
            resolved_by=resolved_by,
            activated_at=activated_at,
        )
        if changed:
            self._grants[grant_id] = record
        return _detached_grant(record)

    async def abandon(
        self,
        grant_id: str,
        *,
        reason: str,
        abandoned_at: datetime,
        expected_request_digests: tuple[str, ...] | None = None,
    ) -> ToolAuthorizationGrant:
        record, changed = abandoned_grant_transition(
            self._grants.get(grant_id),
            grant_id=grant_id,
            reason=reason,
            abandoned_at=abandoned_at,
            expected_request_digests=expected_request_digests,
        )
        if changed:
            self._grants[grant_id] = record
        return _detached_grant(record)

    async def revoke(
        self,
        grant_id: str,
        *,
        revoked_by: ToolAuthorizationGrantActor | None,
        reason: str | None,
        revoked_at: datetime,
    ) -> ToolAuthorizationGrant:
        record, changed = revoked_grant_transition(
            self._grants.get(grant_id),
            grant_id=grant_id,
            revoked_by=revoked_by,
            reason=reason,
            revoked_at=revoked_at,
        )
        if changed:
            self._grants[grant_id] = record
        return _detached_grant(record)

    async def record_use(self, use: ToolAuthorizationGrantUse) -> ToolAuthorizationGrantUse:
        use = _detached_use(use)
        return _detached_use(self._uses.setdefault(use.use_key, use))

    async def list_uses(
        self,
        grant_id: str,
        *,
        limit: int = 100,
        after: ToolAuthorizationGrantUseCursor | None = None,
    ) -> tuple[ToolAuthorizationGrantUse, ...]:
        limit = require_grant_list_limit(limit)
        uses = sorted(
            (
                use
                for use in self._uses.values()
                if use.grant_id == grant_id and (after is None or use_order_key(use) > after.key())
            ),
            key=use_order_key,
        )
        return tuple(_detached_use(use) for use in uses[:limit])


class ToolAuthorizationGrantCreation(_GrantModel):
    """What a grant authority sees before a grant is written."""

    actor: ResolutionActor | None
    grant: ToolAuthorizationGrant
    offer: ToolAuthorizationGrantOffer


class ToolAuthorizationGrantAuthority(ABC):
    """Decides who may create and revoke grants.

    It runs inside the runtime boundary with the trusted actor and the grant
    derived from the pending-approval snapshot. Returning anything but ``True``
    denies and leaves both approval and grant state unchanged.
    """

    @abstractmethod
    async def authorize_creation(self, creation: ToolAuthorizationGrantCreation) -> bool: ...

    @abstractmethod
    async def authorize_revocation(
        self, actor: ResolutionActor | None, grant: ToolAuthorizationGrant
    ) -> bool: ...


class AllowedActorsGrantAuthority(ToolAuthorizationGrantAuthority):
    """Allow exact actor subjects (optionally one tenant and actor sources).

    ``revokers`` defaults to ``creators``. An absent actor is always denied.
    """

    def __init__(
        self,
        *,
        creators: Iterable[str],
        revokers: Iterable[str] | None = None,
        tenant: str | None = None,
        sources: Iterable[ResolutionActorSource] | None = None,
    ) -> None:
        if isinstance(creators, str) or isinstance(revokers, str):
            raise TypeError("creators and revokers must be iterables of subjects.")
        self._creators = frozenset(_identifier(item, "creator") for item in creators)
        self._revokers = (
            self._creators
            if revokers is None
            else frozenset(_identifier(item, "revoker") for item in revokers)
        )
        self._tenant = _optional_identifier(tenant, "tenant")
        self._sources = (
            None if sources is None else frozenset(ResolutionActorSource(s) for s in sources)
        )

    def _allowed(self, actor: ResolutionActor | None, subjects: frozenset[str]) -> bool:
        return (
            actor is not None
            and actor.subject in subjects
            and (self._tenant is None or actor.tenant == self._tenant)
            and (self._sources is None or actor.source in self._sources)
        )

    async def authorize_creation(self, creation: ToolAuthorizationGrantCreation) -> bool:
        return self._allowed(creation.actor, self._creators)

    async def authorize_revocation(
        self, actor: ResolutionActor | None, grant: ToolAuthorizationGrant
    ) -> bool:
        return self._allowed(actor, self._revokers)
