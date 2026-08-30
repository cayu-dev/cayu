"""Lineage-scoped handoff of one durable artifact between isolated workspaces."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import posixpath
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    computed_field,
    field_validator,
    model_validator,
)

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_clean_nonblank,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.artifacts import ArtifactMetadata, ArtifactScope, ArtifactStore, copy_artifact_read_result
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import (
    DurableToolOperationConflict,
    DurableToolRecoveryAuthority,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _runtime_tool_invocation_authority,
)
from cayu.runtime.invocation import InvocationOrigin, SessionExecutionSource, SessionInvocation
from cayu.runtime.sessions import SessionOperationPublication, SessionStore
from cayu.tools._redaction import (
    active_secret_redactor_snapshot,
    await_revision_stable_secret_output,
)
from cayu.workspaces import Workspace, WorkspaceRevisionMismatchError

SHARED_ARTIFACT_SCHEMA_VERSION = 1
SHARED_ARTIFACT_REFERENCE_PREFIX = "cayu-shared-artifact-v1."
PUBLISH_WORKSPACE_ARTIFACT_TOOL_NAME = "publish_workspace_artifact"
MATERIALIZE_SHARED_ARTIFACT_TOOL_NAME = "materialize_shared_artifact"
DEFAULT_SHARED_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
MAX_SHARED_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_SHARED_ARTIFACT_MAX_PUBLICATIONS = 64
MAX_SHARED_ARTIFACT_PUBLICATIONS = 1024
DEFAULT_SHARED_ARTIFACT_GRANT_TTL_SECONDS = 24 * 60 * 60
MAX_SHARED_ARTIFACT_GRANT_TTL_SECONDS = 30 * 24 * 60 * 60
DEFAULT_SHARED_ARTIFACT_MAX_LINEAGE_DEPTH = 64
MAX_SHARED_ARTIFACT_REFERENCE_BYTES = 4096
MAX_SHARED_ARTIFACT_PATH_BYTES = 4096
MAX_SHARED_ARTIFACT_POLICY_PREFIXES = 64
MAX_SHARED_ARTIFACT_CONTENT_TYPES = 64
MAX_SHARED_ARTIFACT_WRITE_ATTEMPTS = 16

_SHA256_IDENTIFIER_PATTERN = r"^sha256:[0-9a-f]{64}$"
_RAW_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ARTIFACT_ID_PATTERN = r"^art_[0-9a-f]{32}$"
_GRANT_ID_PATTERN = r"^sag_[0-9a-f]{32}$"
_OPERATION_ID_PATTERN = r"^sao_[0-9a-f]{32}$"
_PUBLICATION_INDEX_KEY = "cayu.shared-artifact.publication-index.v1"
_PUBLICATION_STATE_KEY_PREFIX = "cayu.shared-artifact.publication.v1:"
_MATERIALIZATION_STATE_KEY_PREFIX = "cayu.shared-artifact.materialization.v1:"
_GRANT_KEY_PREFIX = "cayu.shared-artifact.grant.v1:"
_CALL_LOCATOR_KEY_PREFIX = "cayu.shared-artifact.call-locator.v1:"
_PUBLICATION_INDEX_RECORD_TYPE = "cayu.shared-artifact-publication-index"
_PUBLICATION_PREPARATION_RECORD_TYPE = "cayu.shared-artifact-publication-preparation"
_PUBLICATION_RECEIPT_RECORD_TYPE = "cayu.shared-artifact-publication-receipt"
_MATERIALIZATION_PREPARATION_RECORD_TYPE = "cayu.shared-artifact-materialization-preparation"
_MATERIALIZATION_RECEIPT_RECORD_TYPE = "cayu.shared-artifact-materialization-receipt"
_GRANT_RECORD_TYPE = "cayu.shared-artifact-grant"
_CALL_LOCATOR_RECORD_TYPE = "cayu.shared-artifact-call-locator"
_ALLOWED_DESCENDANT_SOURCES = frozenset(
    {SessionExecutionSource.FORK, SessionExecutionSource.SUBAGENT}
)


class SharedArtifactAudience(StrEnum):
    """Closed lineage audiences supported by the first handoff protocol."""

    DESCENDANT_FORK_OR_SUBAGENT = "descendant_fork_or_subagent"


class SharedArtifactGrantStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class SharedArtifactAuthorizationError(PermissionError):
    """The caller could not prove one active lineage-scoped grant."""


class SharedArtifactPolicy(BaseModel):
    """Application-sealed publication and materialization policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    publish_path_prefixes: tuple[str, ...]
    materialize_path_prefixes: tuple[str, ...]
    allowed_content_types: tuple[str, ...]
    max_bytes: StrictInt = Field(
        default=DEFAULT_SHARED_ARTIFACT_MAX_BYTES,
        ge=1,
        le=MAX_SHARED_ARTIFACT_BYTES,
    )
    max_publications_per_session: StrictInt = Field(
        default=DEFAULT_SHARED_ARTIFACT_MAX_PUBLICATIONS,
        ge=1,
        le=MAX_SHARED_ARTIFACT_PUBLICATIONS,
    )
    grant_ttl_seconds: StrictInt = Field(
        default=DEFAULT_SHARED_ARTIFACT_GRANT_TTL_SECONDS,
        ge=1,
        le=MAX_SHARED_ARTIFACT_GRANT_TTL_SECONDS,
    )
    max_lineage_depth: StrictInt = Field(
        default=DEFAULT_SHARED_ARTIFACT_MAX_LINEAGE_DEPTH,
        ge=1,
        le=DEFAULT_SHARED_ARTIFACT_MAX_LINEAGE_DEPTH,
    )
    audience: SharedArtifactAudience = SharedArtifactAudience.DESCENDANT_FORK_OR_SUBAGENT
    retention_class: str = Field(default="lineage_handoff", max_length=128)
    allow_overwrite: StrictBool = False

    @field_validator("publish_path_prefixes", "materialize_path_prefixes", mode="before")
    @classmethod
    def copy_path_prefixes(cls, value: object, info) -> tuple[str, ...]:
        return _validate_policy_tuple(
            value,
            field_name=info.field_name,
            max_items=MAX_SHARED_ARTIFACT_POLICY_PREFIXES,
            item_validator=_normalize_policy_prefix,
        )

    @field_validator("allowed_content_types", mode="before")
    @classmethod
    def copy_content_types(cls, value: object) -> tuple[str, ...]:
        return _validate_policy_tuple(
            value,
            field_name="allowed_content_types",
            max_items=MAX_SHARED_ARTIFACT_CONTENT_TYPES,
            item_validator=_validate_content_type,
        )

    @field_validator("retention_class")
    @classmethod
    def validate_retention_class(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "retention_class")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
            raise ValueError(
                "retention_class must contain lowercase ASCII letters, digits, underscores, or hyphens."
            )
        return value

    @computed_field
    @property
    def fingerprint(self) -> str:
        material = self.model_dump(mode="json", exclude={"fingerprint"})
        return sha256(canonical_durable_json_bytes(material, "shared_artifact_policy")).hexdigest()


class SharedArtifactRef(BaseModel):
    """Opaque routing identity for one grant; possession is not authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    artifact_store_id: str = Field(max_length=1024)
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    content_digest: str = Field(pattern=_SHA256_IDENTIFIER_PATTERN)
    size_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_session_id: str = Field(max_length=1024)
    access_grant_id: str = Field(pattern=_GRANT_ID_PATTERN)

    @field_validator("artifact_store_id", "source_session_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    def to_opaque_ref(self) -> str:
        payload = canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "shared_artifact_ref",
        )
        encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        reference = SHARED_ARTIFACT_REFERENCE_PREFIX + encoded
        if len(reference.encode("utf-8")) > MAX_SHARED_ARTIFACT_REFERENCE_BYTES:
            raise ValueError("Shared artifact reference exceeds its public byte bound.")
        return reference

    @classmethod
    def from_opaque_ref(cls, value: str) -> SharedArtifactRef:
        value = require_durable_clean_nonblank(value, "ref")
        if len(value.encode("utf-8")) > MAX_SHARED_ARTIFACT_REFERENCE_BYTES or not value.startswith(
            SHARED_ARTIFACT_REFERENCE_PREFIX
        ):
            raise ValueError("Shared artifact reference is malformed.")
        encoded = value.removeprefix(SHARED_ARTIFACT_REFERENCE_PREFIX)
        if not encoded or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in encoded
        ):
            raise ValueError("Shared artifact reference is malformed.")
        padding = "=" * (-len(encoded) % 4)
        try:
            decoded = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
            payload = json.loads(decoded)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Shared artifact reference is malformed.") from exc
        reference = cls.model_validate(payload)
        if reference.to_opaque_ref() != value:
            raise ValueError("Shared artifact reference is not canonical.")
        return reference


class SharedArtifactGrant(BaseModel):
    """Durable read/materialize-only authority owned by the source session."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.shared-artifact-grant"] = _GRANT_RECORD_TYPE
    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    reference: SharedArtifactRef
    source_session_instance_id: str = Field(max_length=64)
    source_workspace_id: str = Field(max_length=1024)
    source_path_sha256: str = Field(pattern=_RAW_SHA256_PATTERN)
    root_invocation_id: str = Field(max_length=64)
    root_session_id: str = Field(max_length=1024)
    invocation_origin_sha256: str = Field(pattern=_RAW_SHA256_PATTERN)
    causal_budget_id: str = Field(max_length=1024)
    content_type: str = Field(max_length=1024)
    policy_fingerprint: str = Field(pattern=_RAW_SHA256_PATTERN)
    audience: SharedArtifactAudience
    max_lineage_depth: StrictInt = Field(
        ge=1,
        le=DEFAULT_SHARED_ARTIFACT_MAX_LINEAGE_DEPTH,
    )
    retention_class: str = Field(max_length=128)
    published_at: datetime
    expires_at: datetime
    status: SharedArtifactGrantStatus = SharedArtifactGrantStatus.ACTIVE
    revoked_at: datetime | None = None
    revocation_reason: str | None = Field(default=None, max_length=1024)

    @field_validator(
        "source_session_instance_id",
        "source_workspace_id",
        "root_invocation_id",
        "root_session_id",
        "causal_budget_id",
        "content_type",
        "policy_fingerprint",
        "retention_class",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("published_at", "expires_at", "revoked_at")
    @classmethod
    def normalize_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Shared artifact timestamps must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> SharedArtifactGrant:
        if self.expires_at <= self.published_at:
            raise ValueError("Shared artifact grant expiry must follow publication.")
        if self.status is SharedArtifactGrantStatus.ACTIVE:
            if self.revoked_at is not None or self.revocation_reason is not None:
                raise ValueError("Active shared artifact grants cannot carry revocation fields.")
        elif self.revoked_at is None or self.revocation_reason is None:
            raise ValueError("Revoked shared artifact grants require time and reason.")
        return self


class SharedArtifactPublicationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.shared-artifact-publication-receipt"] = (
        _PUBLICATION_RECEIPT_RECORD_TYPE
    )
    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    operation_id: str = Field(pattern=_OPERATION_ID_PATTERN)
    reference: SharedArtifactRef
    source_workspace_id: str = Field(max_length=1024)
    source_path_sha256: str = Field(pattern=_RAW_SHA256_PATTERN)
    content_type: str = Field(max_length=1024)
    policy_fingerprint: str = Field(pattern=_RAW_SHA256_PATTERN)
    retention_class: str = Field(max_length=128)
    terminal_disposition: Literal["published"] = "published"
    published_at: datetime

    @field_validator("source_workspace_id", "content_type", "policy_fingerprint", "retention_class")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("published_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value, "published_at")


class SharedArtifactMaterializationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.shared-artifact-materialization-receipt"] = (
        _MATERIALIZATION_RECEIPT_RECORD_TYPE
    )
    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    operation_id: str = Field(pattern=_OPERATION_ID_PATTERN)
    reference: SharedArtifactRef
    source_workspace_id: str = Field(max_length=1024)
    destination_session_id: str = Field(max_length=1024)
    destination_workspace_id: str = Field(max_length=1024)
    destination_path_sha256: str = Field(pattern=_RAW_SHA256_PATTERN)
    policy_fingerprint: str = Field(pattern=_RAW_SHA256_PATTERN)
    bytes_written: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    terminal_disposition: Literal["materialized"] = "materialized"
    materialized_at: datetime

    @field_validator(
        "source_workspace_id",
        "destination_session_id",
        "destination_workspace_id",
        "policy_fingerprint",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("materialized_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value, "materialized_at")


class _SessionLineageNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    session_instance_id: str
    parent_session_id: str | None = None
    causal_budget_id: str
    invocation: SessionInvocation

    @field_validator("session_id", "session_instance_id", "parent_session_id", "causal_budget_id")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)


async def authorize_shared_artifact_materialization(
    *,
    session_store: SessionStore,
    caller_session_id: str,
    caller_session_instance_id: str,
    reference: dict[str, Any],
    policy_fingerprint: str,
    observed_at: str,
) -> dict[str, Any]:
    """Resolve one exact grant only after bounded durable ancestry validation."""

    _require_session_store_operations(
        session_store,
        "load",
        "load_session_operation",
    )
    caller_session_id = require_durable_clean_nonblank(
        caller_session_id,
        "caller_session_id",
    )
    caller_session_instance_id = require_durable_clean_nonblank(
        caller_session_instance_id,
        "caller_session_instance_id",
    )
    ref = SharedArtifactRef.model_validate(reference)
    policy_fingerprint = _require_raw_sha256(policy_fingerprint, "policy_fingerprint")
    observed = _parse_timestamp(observed_at, "observed_at")
    raw_grant = await session_store.load_session_operation(
        ref.source_session_id,
        _grant_storage_key(ref.access_grant_id),
    )
    try:
        grant = SharedArtifactGrant.model_validate(raw_grant)
    except Exception as exc:
        raise SharedArtifactAuthorizationError("Shared artifact authorization denied.") from exc
    if (
        grant.reference != ref
        or grant.policy_fingerprint != policy_fingerprint
        or grant.status is not SharedArtifactGrantStatus.ACTIVE
        or observed >= grant.expires_at
        or grant.audience is not SharedArtifactAudience.DESCENDANT_FORK_OR_SUBAGENT
    ):
        raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")

    caller = await session_store.load(caller_session_id)
    source = await session_store.load(ref.source_session_id)
    if caller is None or source is None or caller.instance_id != caller_session_instance_id:
        raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
    source_node = _lineage_node_from_session(source)
    if (
        source_node.session_instance_id != grant.source_session_instance_id
        or source_node.causal_budget_id != grant.causal_budget_id
        or source_node.invocation.root_invocation_id != grant.root_invocation_id
        or source_node.invocation.root_session_id != grant.root_session_id
        or _origin_sha256(source_node.invocation.origin) != grant.invocation_origin_sha256
    ):
        raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
    await _require_supported_descendant(
        session_store=session_store,
        caller=_lineage_node_from_session(caller),
        source=source_node,
        grant=grant,
    )
    return grant.model_dump(mode="json")


async def revoke_shared_artifact_grant(
    session_store: SessionStore,
    reference: SharedArtifactRef | str,
    *,
    reason: str,
    revoked_at: datetime | None = None,
) -> SharedArtifactGrant:
    """Programmatically revoke one exact grant without deleting its evidence."""

    _require_session_store_operations(
        session_store,
        "load_session_operation",
        "publish_session_operation",
    )
    ref = (
        SharedArtifactRef.from_opaque_ref(reference)
        if type(reference) is str
        else SharedArtifactRef.model_validate(reference)
    )
    reason = require_durable_clean_nonblank(reason, "reason")
    if len(reason) > 1024:
        raise ValueError("reason must not exceed 1024 characters.")
    timestamp = _normalize_timestamp(revoked_at or datetime.now(UTC), "revoked_at")
    key = _grant_storage_key(ref.access_grant_id)

    def revoke(
        _session: Any,
        checkpoint: dict[str, Any] | None,
        current: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        try:
            grant = SharedArtifactGrant.model_validate(current)
        except Exception as exc:
            raise SharedArtifactAuthorizationError("Shared artifact grant was not found.") from exc
        if grant.reference != ref:
            raise SharedArtifactAuthorizationError("Shared artifact grant was not found.")
        if grant.status is SharedArtifactGrantStatus.REVOKED:
            if grant.revocation_reason != reason:
                raise ValueError("Shared artifact grant is already revoked for another reason.")
            desired = grant
        else:
            desired = grant.model_copy(
                update={
                    "status": SharedArtifactGrantStatus.REVOKED,
                    "revoked_at": timestamp,
                    "revocation_reason": reason,
                }
            )
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else checkpoint,
            operation_records={key: desired.model_dump(mode="json")},
        )

    await session_store.publish_session_operation(
        ref.source_session_id,
        idempotency_key=key,
        operation_transform=revoke,
        events=[],
    )
    persisted = await session_store.load_session_operation(ref.source_session_id, key)
    return SharedArtifactGrant.model_validate(persisted)


def _lineage_node_from_session(session: Any) -> _SessionLineageNode:
    return _SessionLineageNode(
        session_id=session.id,
        session_instance_id=session.instance_id,
        parent_session_id=session.parent_session_id,
        causal_budget_id=session.causal_budget_id,
        invocation=session.invocation,
    )


def _require_session_store_operations(session_store: object, *names: str) -> None:
    if not names or any(
        type(name) is not str or not name or not callable(getattr(session_store, name, None))
        for name in names
    ):
        raise TypeError("session_store does not provide the required durable operations.")


async def _require_supported_descendant(
    *,
    session_store: SessionStore,
    caller: _SessionLineageNode,
    source: _SessionLineageNode,
    grant: SharedArtifactGrant,
) -> None:
    if caller.session_id == source.session_id:
        raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
    expected_invocation = source.invocation
    if (
        caller.invocation.root_invocation_id != expected_invocation.root_invocation_id
        or caller.invocation.root_session_id != expected_invocation.root_session_id
        or caller.causal_budget_id != source.causal_budget_id
        or caller.invocation.origin != expected_invocation.origin
    ):
        raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
    current = caller
    seen = {caller.session_id}
    for _depth in range(grant.max_lineage_depth):
        if (
            current.invocation.source not in _ALLOWED_DESCENDANT_SOURCES
            or current.parent_session_id is None
        ):
            raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
        if current.parent_session_id == source.session_id:
            return
        if current.parent_session_id in seen:
            raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
        seen.add(current.parent_session_id)
        parent = await session_store.load(current.parent_session_id)
        if parent is None:
            raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
        current = _lineage_node_from_session(parent)
        if (
            current.invocation.root_invocation_id != expected_invocation.root_invocation_id
            or current.invocation.root_session_id != expected_invocation.root_session_id
            or current.causal_budget_id != source.causal_budget_id
            or current.invocation.origin != expected_invocation.origin
        ):
            raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")
    raise SharedArtifactAuthorizationError("Shared artifact authorization denied.")


def _origin_sha256(origin: InvocationOrigin) -> str:
    return sha256(
        canonical_durable_json_bytes(
            origin.model_dump(mode="json"),
            "shared_artifact_invocation_origin",
        )
    ).hexdigest()


def _validate_policy_tuple(
    value: object,
    *,
    field_name: str,
    max_items: int,
    item_validator: Callable[[str, str], str],
) -> tuple[str, ...]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of strings.")
    if not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be an iterable of strings.")
    values = tuple(value)
    if not values or len(values) > max_items:
        raise ValueError(f"{field_name} must contain from 1 to {max_items} entries.")
    normalized: list[str] = []
    for index, item in enumerate(values):
        if type(item) is not str:
            raise TypeError(f"{field_name}[{index}] must be a string.")
        normalized.append(item_validator(item, f"{field_name}[{index}]"))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} entries must be unique after normalization.")
    return tuple(sorted(normalized))


def _normalize_policy_prefix(value: str, field_name: str) -> str:
    if value == ".":
        return value
    return _normalize_workspace_path(value, field_name)


def _normalize_workspace_path(value: str, field_name: str) -> str:
    value = require_durable_text(value, field_name)
    if not value.strip():
        raise ValueError(f"{field_name} must be nonblank.")
    if len(value.encode("utf-8")) > MAX_SHARED_ARTIFACT_PATH_BYTES:
        raise ValueError(f"{field_name} exceeds the path byte bound.")
    if posixpath.isabs(value):
        raise ValueError(f"{field_name} must be workspace-relative.")
    parts = tuple(part for part in value.split("/") if part)
    if ".." in parts:
        raise ValueError(f"{field_name} cannot contain parent traversal.")
    normalized = posixpath.normpath(value)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ValueError(f"{field_name} must reference a workspace file.")
    return normalized


def _path_allowed(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        prefix == "." or path == prefix or path.startswith(prefix + "/") for prefix in prefixes
    )


def _validate_content_type(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value) > 1024 or any(
        ord(character) < 0x20 or ord(character) > 0x7E for character in value
    ):
        raise ValueError(f"{field_name} must be bounded printable ASCII.")
    return value.lower()


def _require_raw_sha256(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _normalize_timestamp(value: datetime, field_name: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime.")
    return value.astimezone(UTC)


def _parse_timestamp(value: str, field_name: str) -> datetime:
    value = require_durable_clean_nonblank(value, field_name)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime.") from exc
    return _normalize_timestamp(parsed, field_name)


def _grant_storage_key(grant_id: str) -> str:
    if type(grant_id) is not str or not grant_id.startswith("sag_"):
        raise ValueError("Shared artifact grant id is malformed.")
    return _GRANT_KEY_PREFIX + grant_id


class _SharedArtifactPublicationIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.shared-artifact-publication-index"] = _PUBLICATION_INDEX_RECORD_TYPE
    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    publication_ids: tuple[str, ...] = ()

    @field_validator("publication_ids", mode="before")
    @classmethod
    def copy_publication_ids(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str | bytes):
            raise TypeError("publication_ids must be an iterable of strings.")
        if not isinstance(value, Iterable):
            raise TypeError("publication_ids must be an iterable of strings.")
        raw_values = tuple(value)
        if len(raw_values) > MAX_SHARED_ARTIFACT_PUBLICATIONS:
            raise ValueError("Shared artifact publication index exceeds its bound.")
        values: list[str] = []
        for item in raw_values:
            if type(item) is not str or not item.startswith("sao_"):
                raise ValueError("Shared artifact publication index is malformed.")
            values.append(item)
        result = tuple(values)
        if result != tuple(sorted(set(result))):
            raise ValueError("Shared artifact publication index must be sorted and unique.")
        return result


class _SharedArtifactPublicationPreparation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.shared-artifact-publication-preparation"] = (
        _PUBLICATION_PREPARATION_RECORD_TYPE
    )
    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    receipt: SharedArtifactPublicationReceipt
    grant: SharedArtifactGrant
    artifact_filename: str = Field(max_length=256)

    @field_validator("artifact_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "artifact_filename")

    @model_validator(mode="after")
    def validate_correlated_state(self) -> _SharedArtifactPublicationPreparation:
        if (
            self.receipt.reference != self.grant.reference
            or self.receipt.operation_id != _operation_id_from_grant(self.grant.reference)
            or self.receipt.source_workspace_id != self.grant.source_workspace_id
            or self.receipt.source_path_sha256 != self.grant.source_path_sha256
            or self.receipt.content_type != self.grant.content_type
            or self.receipt.policy_fingerprint != self.grant.policy_fingerprint
            or self.receipt.retention_class != self.grant.retention_class
            or self.receipt.published_at != self.grant.published_at
        ):
            raise ValueError("Shared artifact publication preparation is inconsistent.")
        return self


class _SharedArtifactMaterializationPreparation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.shared-artifact-materialization-preparation"] = (
        _MATERIALIZATION_PREPARATION_RECORD_TYPE
    )
    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    receipt: SharedArtifactMaterializationReceipt
    destination_was_missing: StrictBool
    expected_destination_revision: str | None = Field(default=None, max_length=1024)
    expected_destination_sha256: str | None = Field(
        default=None,
        pattern=_RAW_SHA256_PATTERN,
    )
    expected_destination_bytes: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )

    @model_validator(mode="after")
    def validate_before_state(self) -> _SharedArtifactMaterializationPreparation:
        before = (
            self.expected_destination_revision,
            self.expected_destination_sha256,
            self.expected_destination_bytes,
        )
        if self.destination_was_missing != all(item is None for item in before):
            raise ValueError("Shared artifact destination preparation is inconsistent.")
        if not self.destination_was_missing and any(item is None for item in before):
            raise ValueError("Shared artifact destination preparation is incomplete.")
        return self


class _SharedArtifactCallLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.shared-artifact-call-locator"] = _CALL_LOCATOR_RECORD_TYPE
    schema_version: Literal[1] = SHARED_ARTIFACT_SCHEMA_VERSION
    kind: Literal["publication", "materialization"]
    session_id: str
    parent_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    model_step_id: str
    model_attempt_id: str
    tool_round_id: str
    tool_call_id: str
    tool_name: str
    idempotency_key: str
    effective_arguments_sha256: str = Field(pattern=_RAW_SHA256_PATTERN)
    execution_profile_fingerprint: str = Field(pattern=_RAW_SHA256_PATTERN)
    operation_id: str = Field(pattern=_OPERATION_ID_PATTERN)
    state_storage_key: str

    @field_validator(
        "session_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "state_storage_key",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class _SharedArtifactRefusal(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        super().__init__(code)


class PublishWorkspaceArtifactTool(Tool):
    """Publish one exact workspace file for explicit descendant handoff."""

    def __init__(
        self,
        policy: SharedArtifactPolicy,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = SharedArtifactPolicy.model_validate(
            policy.model_dump(mode="python", exclude={"fingerprint"})
        )
        self._clock = datetime_now_utc if clock is None else clock
        if not callable(self._clock):
            raise TypeError("clock must be callable.")
        super().__init__(
            ToolSpec(
                name=PUBLISH_WORKSPACE_ARTIFACT_TOOL_NAME,
                description=(
                    "Publish one policy-approved workspace file and return an opaque reference "
                    "that only an authorized descendant fork or subagent can materialize."
                ),
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_SHARED_ARTIFACT_PATH_BYTES,
                            "description": "Workspace-relative file path to publish.",
                        }
                    },
                    "required": ["path"],
                },
                parallel_safe=False,
                effect=ToolEffect.IDEMPOTENT,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="cayu:publish-workspace-artifact",
                    behavior_version=f"1:{self.policy.fingerprint}",
                    implementation_version="1",
                ),
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            return await self._run(ctx, args)
        except _SharedArtifactRefusal as exc:
            return _shared_artifact_error_result(exc.code)

    async def _run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = _tool_path_argument(args, "path")
        if not _path_allowed(path, self.policy.publish_path_prefixes):
            raise _SharedArtifactRefusal("path_not_allowed")
        workspace, artifact_store, authority = _required_runtime_resources(ctx)
        lineage = _SessionLineageNode.model_validate(authority.current_session_lineage)
        if lineage.session_id != ctx.session_id:
            raise RuntimeError("Shared artifact invocation lost its source session identity.")

        source = await _read_secret_free_workspace_source(
            ctx=ctx,
            workspace=workspace,
            path=path,
            max_bytes=self.policy.max_bytes,
        )
        content = source.content
        _freeze_and_require_secret_free_content(
            ctx=ctx,
            authority=authority,
            content=content,
            refusal_code="source_contains_secret",
        )
        content_type = (
            mimetypes.guess_type(posixpath.basename(path))[0] or "application/octet-stream"
        ).lower()
        if content_type not in self.policy.allowed_content_types:
            raise _SharedArtifactRefusal("content_type_not_allowed")

        observed_at = _normalize_timestamp(self._clock(), "clock result")
        preparation = _publication_preparation(
            ctx=ctx,
            lineage=lineage,
            workspace=workspace,
            artifact_store=artifact_store,
            path=path,
            content=content,
            content_type=content_type,
            policy=self.policy,
            observed_at=observed_at,
        )
        state_key = _publication_state_storage_key(preparation.receipt.operation_id)
        locator = _call_locator(
            kind="publication",
            ctx=ctx,
            authority=authority,
            operation_id=preparation.receipt.operation_id,
            state_storage_key=state_key,
        )

        async def settle() -> tuple[SharedArtifactPublicationReceipt, bool]:
            reserved, recovered = await _reserve_publication(
                authority=authority,
                preparation=preparation,
                locator=locator,
                max_publications=self.policy.max_publications_per_session,
            )
            if isinstance(reserved, SharedArtifactPublicationReceipt):
                return reserved, True
            await _publish_artifact_bytes(
                artifact_store=artifact_store,
                ctx=ctx,
                content=content,
                preparation=reserved,
            )
            receipt = await _finalize_publication(
                authority=authority,
                preparation=reserved,
            )
            return receipt, recovered

        receipt, recovered = await _settle_despite_cancellation(settle())
        return _publication_result(receipt, recovered=recovered)

    async def reconcile_durable_tool_call(
        self,
        *,
        parent_session_id: str,
        parent_run_epoch: int,
        execution_profile_fingerprint: str | None,
        environment_name: str | None,
        environment_allocation_fingerprint: str | None,
        model_step_id: str,
        model_attempt_id: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
        started: bool,
        load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
        recovery_authority: DurableToolRecoveryAuthority | None = None,
    ) -> ToolResult | None:
        del environment_allocation_fingerprint, started
        return await _recover_shared_artifact_call(
            kind="publication",
            tool_name=self.name,
            policy=self.policy,
            parent_session_id=parent_session_id,
            parent_run_epoch=parent_run_epoch,
            execution_profile_fingerprint=execution_profile_fingerprint,
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            arguments=arguments,
            environment_name=environment_name,
            load_operation=load_operation,
            recovery_authority=recovery_authority,
        )


class MaterializeSharedArtifactTool(Tool):
    """Materialize one authorized ancestor artifact into this isolated workspace."""

    def __init__(
        self,
        policy: SharedArtifactPolicy,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = SharedArtifactPolicy.model_validate(
            policy.model_dump(mode="python", exclude={"fingerprint"})
        )
        self._clock = datetime_now_utc if clock is None else clock
        if not callable(self._clock):
            raise TypeError("clock must be callable.")
        super().__init__(
            ToolSpec(
                name=MATERIALIZE_SHARED_ARTIFACT_TOOL_NAME,
                description=(
                    "Materialize an explicitly passed shared-artifact reference after Runtime "
                    "validates its grant and supported descendant lineage."
                ),
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ref": {
                            "type": "string",
                            "minLength": len(SHARED_ARTIFACT_REFERENCE_PREFIX) + 1,
                            "maxLength": MAX_SHARED_ARTIFACT_REFERENCE_BYTES,
                        },
                        "destination": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_SHARED_ARTIFACT_PATH_BYTES,
                        },
                    },
                    "required": ["ref", "destination"],
                },
                parallel_safe=False,
                effect=ToolEffect.IDEMPOTENT,
                workspace_mutation=True,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="cayu:materialize-shared-artifact",
                    behavior_version=f"1:{self.policy.fingerprint}",
                    implementation_version="1",
                ),
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            return await self._run(ctx, args)
        except _SharedArtifactRefusal as exc:
            return _shared_artifact_error_result(exc.code)

    async def _run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict or set(args) != {"ref", "destination"}:
            raise _SharedArtifactRefusal("invalid_arguments")
        try:
            reference = SharedArtifactRef.from_opaque_ref(args["ref"])
            destination = _normalize_workspace_path(args["destination"], "destination")
        except (TypeError, ValueError):
            raise _SharedArtifactRefusal("invalid_arguments") from None
        if not _path_allowed(destination, self.policy.materialize_path_prefixes):
            raise _SharedArtifactRefusal("path_not_allowed")
        workspace, artifact_store, authority = _required_runtime_resources(ctx)
        lineage = _SessionLineageNode.model_validate(authority.current_session_lineage)
        if lineage.session_id != ctx.session_id:
            raise RuntimeError("Shared artifact invocation lost its destination session identity.")
        if artifact_store.id != reference.artifact_store_id:
            raise _SharedArtifactRefusal("artifact_store_mismatch")
        observed_at = _normalize_timestamp(self._clock(), "clock result")
        try:
            raw_grant = await authority.authorize_shared_artifact(
                reference.model_dump(mode="json"),
                self.policy.fingerprint,
                observed_at.isoformat(),
            )
            grant = SharedArtifactGrant.model_validate(raw_grant)
        except SharedArtifactAuthorizationError:
            raise _SharedArtifactRefusal("authorization_denied") from None
        if grant.reference != reference:
            raise RuntimeError("Shared artifact authorizer returned a different grant.")
        _seal_exact_durable_record(
            authority,
            grant.model_dump(mode="json"),
            "shared artifact grant",
        )

        artifact = await _read_secret_free_artifact(
            ctx=ctx,
            artifact_store=artifact_store,
            reference=reference,
            grant=grant,
            max_bytes=self.policy.max_bytes,
        )
        _freeze_and_require_secret_free_content(
            ctx=ctx,
            authority=authority,
            content=artifact.content,
            refusal_code="artifact_contains_secret",
        )
        preparation = await _materialization_preparation(
            ctx=ctx,
            lineage=lineage,
            workspace=workspace,
            authority=authority,
            destination=destination,
            reference=reference,
            grant=grant,
            policy=self.policy,
            observed_at=observed_at,
        )
        state_key = _materialization_state_storage_key(preparation.receipt.operation_id)
        locator = _call_locator(
            kind="materialization",
            ctx=ctx,
            authority=authority,
            operation_id=preparation.receipt.operation_id,
            state_storage_key=state_key,
        )

        async def settle() -> tuple[SharedArtifactMaterializationReceipt, bool]:
            reserved, recovered = await _reserve_materialization(
                authority=authority,
                preparation=preparation,
                locator=locator,
            )
            if isinstance(reserved, SharedArtifactMaterializationReceipt):
                return reserved, True
            await _materialize_bytes(
                workspace=workspace,
                destination=destination,
                content=artifact.content,
                preparation=reserved,
            )
            receipt = await _finalize_materialization(
                authority=authority,
                preparation=reserved,
            )
            return receipt, recovered

        receipt, recovered = await _settle_despite_cancellation(settle())
        return _materialization_result(receipt, recovered=recovered)

    async def reconcile_durable_tool_call(
        self,
        *,
        parent_session_id: str,
        parent_run_epoch: int,
        execution_profile_fingerprint: str | None,
        environment_name: str | None,
        environment_allocation_fingerprint: str | None,
        model_step_id: str,
        model_attempt_id: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
        started: bool,
        load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
        recovery_authority: DurableToolRecoveryAuthority | None = None,
    ) -> ToolResult | None:
        del environment_allocation_fingerprint, started
        return await _recover_shared_artifact_call(
            kind="materialization",
            tool_name=self.name,
            policy=self.policy,
            parent_session_id=parent_session_id,
            parent_run_epoch=parent_run_epoch,
            execution_profile_fingerprint=execution_profile_fingerprint,
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            arguments=arguments,
            environment_name=environment_name,
            load_operation=load_operation,
            recovery_authority=recovery_authority,
        )


def datetime_now_utc() -> datetime:
    return datetime.now(UTC)


def _tool_path_argument(args: object, field_name: str) -> str:
    if not isinstance(args, Mapping) or set(args) != {field_name}:
        raise _SharedArtifactRefusal("invalid_arguments")
    arguments = cast("Mapping[str, object]", args)
    value = arguments.get(field_name)
    if type(value) is not str:
        raise _SharedArtifactRefusal("invalid_arguments")
    try:
        return _normalize_workspace_path(value, field_name)
    except (TypeError, ValueError):
        raise _SharedArtifactRefusal("invalid_arguments") from None


def _required_runtime_resources(ctx: ToolContext) -> tuple[Workspace, ArtifactStore, Any]:
    if type(ctx) is not ToolContext:
        raise TypeError("Shared artifact tools require a ToolContext.")
    workspace = ctx._authoritative_workspace_for_builtin()
    artifact_store = ctx._authoritative_artifact_store_for_builtin()
    authority = _runtime_tool_invocation_authority(ctx)
    if (
        not isinstance(workspace, Workspace)
        or not isinstance(artifact_store, ArtifactStore)
        or authority is None
    ):
        raise _SharedArtifactRefusal("capability_unavailable")
    if ctx.workspace_id != workspace.id or ctx.artifact_store_id != artifact_store.id:
        raise RuntimeError("Shared artifact resource identity changed before dispatch.")
    return workspace, artifact_store, authority


async def _read_secret_free_workspace_source(
    *,
    ctx: ToolContext,
    workspace: Workspace,
    path: str,
    max_bytes: int,
) -> Any:
    bounded_limit = workspace.bounded_read_limit(max_bytes)

    async def capture(redactor):
        try:
            source = await workspace.read_bytes(path, max_bytes=bounded_limit)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError):
            raise _SharedArtifactRefusal("source_file_invalid") from None
        if source.truncated or source.total_bytes > max_bytes:
            raise _SharedArtifactRefusal("source_file_oversize")
        if len(source.content) != source.total_bytes:
            raise RuntimeError("Workspace returned an incomplete shared artifact snapshot.")
        if redactor.contains_secret_bytes(source.content):
            raise _SharedArtifactRefusal("source_contains_secret")
        return source

    captured = await await_revision_stable_secret_output(ctx, capture)
    if captured is None:
        raise _SharedArtifactRefusal("secret_redaction_scope_unstable")
    return captured[0]


def _freeze_and_require_secret_free_content(
    *,
    ctx: ToolContext,
    authority: Any,
    content: bytes,
    refusal_code: str,
) -> None:
    # Freeze the invocation's publication scope before any durable reservation or
    # raw-byte copy. A secret discovered between the stable read and this seal is
    # included by the final snapshot and cannot be smuggled through the handoff.
    _seal_exact_durable_record(
        authority,
        {
            "record_type": "cayu.shared-artifact-secret-scope-seal",
            "schema_version": SHARED_ARTIFACT_SCHEMA_VERSION,
        },
        "shared artifact secret-scope seal",
    )
    if active_secret_redactor_snapshot(ctx).redactor.contains_secret_bytes(content):
        raise _SharedArtifactRefusal(refusal_code)


def _seal_exact_durable_record(
    authority: Any,
    record: dict[str, Any],
    field_name: str,
) -> dict[str, Any]:
    desired = copy_durable_json_object(record, field_name)
    sealed = authority.seal_durable_output(desired)
    if type(sealed) is not dict:
        raise RuntimeError("Shared artifact durable output sealing returned invalid evidence.")
    if sealed != desired:
        raise _SharedArtifactRefusal("durable_identity_contains_secret")
    return sealed


def _publication_preparation(
    *,
    ctx: ToolContext,
    lineage: _SessionLineageNode,
    workspace: Workspace,
    artifact_store: ArtifactStore,
    path: str,
    content: bytes,
    content_type: str,
    policy: SharedArtifactPolicy,
    observed_at: datetime,
) -> _SharedArtifactPublicationPreparation:
    content_digest = "sha256:" + sha256(content).hexdigest()
    path_digest = sha256(path.encode("utf-8")).hexdigest()
    logical_digest = sha256(
        canonical_durable_json_bytes(
            {
                "protocol": "cayu.shared-artifact-publication.v1",
                "source_session_id": lineage.session_id,
                "source_session_instance_id": lineage.session_instance_id,
                "source_workspace_id": workspace.id,
                "source_path_sha256": path_digest,
                "artifact_store_id": artifact_store.id,
                "content_digest": content_digest,
                "size_bytes": len(content),
                "content_type": content_type,
                "policy_fingerprint": policy.fingerprint,
            },
            "shared_artifact_publication_identity",
        )
    ).hexdigest()
    identity = logical_digest[:32]
    operation_id = f"sao_{identity}"
    reference = SharedArtifactRef(
        artifact_store_id=artifact_store.id,
        artifact_id=f"art_{identity}",
        content_digest=content_digest,
        size_bytes=len(content),
        source_session_id=lineage.session_id,
        access_grant_id=f"sag_{identity}",
    )
    grant = SharedArtifactGrant(
        reference=reference,
        source_session_instance_id=lineage.session_instance_id,
        source_workspace_id=workspace.id,
        source_path_sha256=path_digest,
        root_invocation_id=lineage.invocation.root_invocation_id,
        root_session_id=lineage.invocation.root_session_id,
        invocation_origin_sha256=_origin_sha256(lineage.invocation.origin),
        causal_budget_id=lineage.causal_budget_id,
        content_type=content_type,
        policy_fingerprint=policy.fingerprint,
        audience=policy.audience,
        max_lineage_depth=policy.max_lineage_depth,
        retention_class=policy.retention_class,
        published_at=observed_at,
        expires_at=observed_at + timedelta(seconds=policy.grant_ttl_seconds),
    )
    receipt = SharedArtifactPublicationReceipt(
        operation_id=operation_id,
        reference=reference,
        source_workspace_id=workspace.id,
        source_path_sha256=path_digest,
        content_type=content_type,
        policy_fingerprint=policy.fingerprint,
        retention_class=policy.retention_class,
        published_at=observed_at,
    )
    return _SharedArtifactPublicationPreparation(
        receipt=receipt,
        grant=grant,
        artifact_filename=f"shared-artifact-{identity}",
    )


async def _reserve_publication(
    *,
    authority: Any,
    preparation: _SharedArtifactPublicationPreparation,
    locator: _SharedArtifactCallLocator,
    max_publications: int,
) -> tuple[_SharedArtifactPublicationPreparation | SharedArtifactPublicationReceipt, bool]:
    state_key = _publication_state_storage_key(preparation.receipt.operation_id)
    for attempt in range(MAX_SHARED_ARTIFACT_WRITE_ATTEMPTS):
        raw_state = await authority.load_durable_operation(state_key)
        if raw_state is not None:
            _seal_exact_durable_record(
                authority,
                raw_state,
                "persisted shared artifact publication state",
            )
            if raw_state.get("record_type") == _PUBLICATION_RECEIPT_RECORD_TYPE:
                receipt = SharedArtifactPublicationReceipt.model_validate(raw_state)
                _require_same_publication(receipt, preparation.receipt)
                await _ensure_call_locator(authority, locator)
                return receipt, True
            existing = _SharedArtifactPublicationPreparation.model_validate(raw_state)
            _require_same_publication(existing.receipt, preparation.receipt)
            await _ensure_call_locator(authority, locator)
            return existing, True

        raw_index = await authority.load_durable_operation(_PUBLICATION_INDEX_KEY)
        index = (
            _SharedArtifactPublicationIndex()
            if raw_index is None
            else _SharedArtifactPublicationIndex.model_validate(raw_index)
        )
        operation_id = preparation.receipt.operation_id
        if operation_id in index.publication_ids:
            raise RuntimeError("Shared artifact index references missing publication state.")
        if len(index.publication_ids) >= max_publications:
            raise _SharedArtifactRefusal("publication_limit_reached")
        desired_index = index.model_copy(
            update={"publication_ids": tuple(sorted((*index.publication_ids, operation_id)))}
        )
        sealed_index = _seal_exact_durable_record(
            authority,
            desired_index.model_dump(mode="json"),
            "shared artifact publication index",
        )
        sealed_preparation = _seal_exact_durable_record(
            authority,
            preparation.model_dump(mode="json"),
            "shared artifact publication preparation",
        )
        try:
            await authority.compare_and_set_durable_operation(
                _PUBLICATION_INDEX_KEY,
                raw_index,
                sealed_index,
                {state_key: sealed_preparation},
            )
        except DurableToolOperationConflict:
            if attempt + 1 == MAX_SHARED_ARTIFACT_WRITE_ATTEMPTS:
                raise
            continue
        await _ensure_call_locator(authority, locator)
        return preparation, False
    raise RuntimeError("Shared artifact publication reservation did not converge.")


async def _publish_artifact_bytes(
    *,
    artifact_store: ArtifactStore,
    ctx: ToolContext,
    content: bytes,
    preparation: _SharedArtifactPublicationPreparation,
) -> None:
    grant = preparation.grant
    reference = grant.reference
    metadata = _publication_artifact_metadata(preparation)
    artifact = await artifact_store.put_bytes(
        content,
        artifact_id=reference.artifact_id,
        filename=preparation.artifact_filename,
        content_type=grant.content_type,
        scope=ArtifactScope.SESSION,
        session_id=reference.source_session_id,
        agent_name=ctx.agent_name,
        environment_name=ctx.environment_name,
        metadata=metadata,
    )
    _require_artifact_metadata(
        artifact,
        reference=reference,
        grant=grant,
        expected_filename=preparation.artifact_filename,
        expected_agent_name=ctx.agent_name,
        expected_environment_name=ctx.environment_name,
        expected_metadata=metadata,
    )


def _publication_artifact_metadata(
    preparation: _SharedArtifactPublicationPreparation,
) -> dict[str, str | int]:
    grant = preparation.grant
    reference = grant.reference
    return {
        "source": "workspace",
        "operation": PUBLISH_WORKSPACE_ARTIFACT_TOOL_NAME,
        "shared_artifact_schema_version": SHARED_ARTIFACT_SCHEMA_VERSION,
        "shared_artifact_operation_id": preparation.receipt.operation_id,
        "shared_artifact_grant_id": reference.access_grant_id,
        "shared_artifact_content_digest": reference.content_digest,
        "source_workspace_id": grant.source_workspace_id,
        "source_workspace_path_sha256": grant.source_path_sha256,
        "policy_fingerprint": grant.policy_fingerprint,
        "retention_class": grant.retention_class,
    }


async def _finalize_publication(
    *,
    authority: Any,
    preparation: _SharedArtifactPublicationPreparation,
) -> SharedArtifactPublicationReceipt:
    state_key = _publication_state_storage_key(preparation.receipt.operation_id)
    expected = _seal_exact_durable_record(
        authority,
        preparation.model_dump(mode="json"),
        "shared artifact publication preparation",
    )
    desired = _seal_exact_durable_record(
        authority,
        preparation.receipt.model_dump(mode="json"),
        "shared artifact publication receipt",
    )
    grant = _seal_exact_durable_record(
        authority,
        preparation.grant.model_dump(mode="json"),
        "shared artifact grant",
    )
    try:
        await authority.compare_and_set_durable_operation(
            state_key,
            expected,
            desired,
            {_grant_storage_key(preparation.grant.reference.access_grant_id): grant},
        )
    except DurableToolOperationConflict:
        persisted = await authority.load_durable_operation(state_key)
        if persisted is None:
            raise RuntimeError("Shared artifact publication receipt was not persisted.") from None
        _seal_exact_durable_record(
            authority,
            persisted,
            "persisted shared artifact publication receipt",
        )
        receipt = SharedArtifactPublicationReceipt.model_validate(persisted)
        _require_same_publication(receipt, preparation.receipt)
        return receipt
    persisted = await authority.load_durable_operation(state_key)
    if persisted is None:
        raise RuntimeError("Shared artifact publication receipt was not persisted.")
    _seal_exact_durable_record(
        authority,
        persisted,
        "persisted shared artifact publication receipt",
    )
    receipt = SharedArtifactPublicationReceipt.model_validate(persisted)
    _require_same_publication(receipt, preparation.receipt)
    return receipt


async def _read_secret_free_artifact(
    *,
    ctx: ToolContext,
    artifact_store: ArtifactStore,
    reference: SharedArtifactRef,
    grant: SharedArtifactGrant,
    max_bytes: int,
) -> Any:
    if reference.size_bytes > max_bytes:
        raise _SharedArtifactRefusal("artifact_oversize")

    async def capture(redactor):
        try:
            result = copy_artifact_read_result(
                await artifact_store.read_bytes(
                    reference.artifact_id,
                    max_bytes=max_bytes,
                ),
                expected_artifact_id=reference.artifact_id,
                max_content_bytes=max_bytes,
            )
        except FileNotFoundError:
            raise _SharedArtifactRefusal("artifact_missing") from None
        if result.truncated or result.total_bytes != reference.size_bytes:
            raise _SharedArtifactRefusal("artifact_mismatch")
        if "sha256:" + sha256(result.content).hexdigest() != reference.content_digest:
            raise _SharedArtifactRefusal("artifact_mismatch")
        metadata = result.metadata
        if (
            metadata.scope is not ArtifactScope.SESSION
            or metadata.session_id != reference.source_session_id
            or metadata.size_bytes != reference.size_bytes
            or metadata.content_type != grant.content_type
            or metadata.metadata.get("shared_artifact_grant_id") != reference.access_grant_id
            or metadata.metadata.get("shared_artifact_content_digest") != reference.content_digest
            or metadata.metadata.get("source_workspace_id") != grant.source_workspace_id
            or metadata.metadata.get("source_workspace_path_sha256") != grant.source_path_sha256
            or metadata.metadata.get("policy_fingerprint") != grant.policy_fingerprint
        ):
            raise _SharedArtifactRefusal("artifact_mismatch")
        if redactor.contains_secret_bytes(result.content):
            raise _SharedArtifactRefusal("artifact_contains_secret")
        return result

    captured = await await_revision_stable_secret_output(ctx, capture)
    if captured is None:
        raise _SharedArtifactRefusal("secret_redaction_scope_unstable")
    return captured[0]


async def _materialization_preparation(
    *,
    ctx: ToolContext,
    lineage: _SessionLineageNode,
    workspace: Workspace,
    authority: Any,
    destination: str,
    reference: SharedArtifactRef,
    grant: SharedArtifactGrant,
    policy: SharedArtifactPolicy,
    observed_at: datetime,
) -> _SharedArtifactMaterializationPreparation:
    destination_digest = sha256(destination.encode("utf-8")).hexdigest()
    operation_id = _materialization_operation_id(
        lineage=lineage,
        workspace_id=workspace.id,
        destination_path_sha256=destination_digest,
        reference=reference,
        policy_fingerprint=policy.fingerprint,
    )
    desired_receipt = SharedArtifactMaterializationReceipt(
        operation_id=operation_id,
        reference=reference,
        source_workspace_id=grant.source_workspace_id,
        destination_session_id=lineage.session_id,
        destination_workspace_id=workspace.id,
        destination_path_sha256=destination_digest,
        policy_fingerprint=policy.fingerprint,
        bytes_written=reference.size_bytes,
        materialized_at=observed_at,
    )
    state_key = _materialization_state_storage_key(operation_id)
    raw_state = await authority.load_durable_operation(state_key)
    if raw_state is not None:
        _seal_exact_durable_record(
            authority,
            raw_state,
            "persisted shared artifact materialization state",
        )
        if raw_state.get("record_type") == _MATERIALIZATION_RECEIPT_RECORD_TYPE:
            existing_receipt = SharedArtifactMaterializationReceipt.model_validate(raw_state)
            _require_same_materialization(existing_receipt, desired_receipt)
            return _SharedArtifactMaterializationPreparation(
                receipt=existing_receipt,
                destination_was_missing=True,
            )
        existing = _SharedArtifactMaterializationPreparation.model_validate(raw_state)
        _require_same_materialization(existing.receipt, desired_receipt)
        return existing

    destination_was_missing = False
    expected_revision: str | None = None
    expected_sha256: str | None = None
    expected_bytes: int | None = None
    try:
        current = await workspace.read_bytes(destination, max_bytes=policy.max_bytes)
    except FileNotFoundError:
        destination_was_missing = True
    except (IsADirectoryError, NotADirectoryError, ValueError):
        raise _SharedArtifactRefusal("destination_invalid") from None
    else:
        if not policy.allow_overwrite:
            # Another identical invocation can reserve, write, and finalize between
            # the first durable-state read above and this workspace observation.
            # Rejoin that exact state instead of misclassifying its file as an
            # unrelated overwrite.
            concurrent_state = await authority.load_durable_operation(state_key)
            if concurrent_state is not None:
                _seal_exact_durable_record(
                    authority,
                    concurrent_state,
                    "persisted shared artifact materialization state",
                )
                if concurrent_state.get("record_type") == _MATERIALIZATION_RECEIPT_RECORD_TYPE:
                    concurrent_receipt = SharedArtifactMaterializationReceipt.model_validate(
                        concurrent_state
                    )
                    _require_same_materialization(concurrent_receipt, desired_receipt)
                    return _SharedArtifactMaterializationPreparation(
                        receipt=concurrent_receipt,
                        destination_was_missing=True,
                    )
                concurrent_preparation = _SharedArtifactMaterializationPreparation.model_validate(
                    concurrent_state
                )
                _require_same_materialization(
                    concurrent_preparation.receipt,
                    desired_receipt,
                )
                return concurrent_preparation
            raise _SharedArtifactRefusal("overwrite_denied")
        if current.truncated or current.revision is None:
            raise _SharedArtifactRefusal("destination_oversize")
        expected_revision = current.revision
        expected_sha256 = sha256(current.content).hexdigest()
        expected_bytes = current.total_bytes
    del ctx
    return _SharedArtifactMaterializationPreparation(
        receipt=desired_receipt,
        destination_was_missing=destination_was_missing,
        expected_destination_revision=expected_revision,
        expected_destination_sha256=expected_sha256,
        expected_destination_bytes=expected_bytes,
    )


async def _reserve_materialization(
    *,
    authority: Any,
    preparation: _SharedArtifactMaterializationPreparation,
    locator: _SharedArtifactCallLocator,
) -> tuple[_SharedArtifactMaterializationPreparation | SharedArtifactMaterializationReceipt, bool]:
    state_key = _materialization_state_storage_key(preparation.receipt.operation_id)
    for attempt in range(MAX_SHARED_ARTIFACT_WRITE_ATTEMPTS):
        raw = await authority.load_durable_operation(state_key)
        if raw is not None:
            _seal_exact_durable_record(
                authority,
                raw,
                "persisted shared artifact materialization state",
            )
            if raw.get("record_type") == _MATERIALIZATION_RECEIPT_RECORD_TYPE:
                receipt = SharedArtifactMaterializationReceipt.model_validate(raw)
                _require_same_materialization(receipt, preparation.receipt)
                await _ensure_call_locator(authority, locator)
                return receipt, True
            existing = _SharedArtifactMaterializationPreparation.model_validate(raw)
            _require_same_materialization(existing.receipt, preparation.receipt)
            await _ensure_call_locator(authority, locator)
            return existing, True
        try:
            desired = _seal_exact_durable_record(
                authority,
                preparation.model_dump(mode="json"),
                "shared artifact materialization preparation",
            )
            await authority.compare_and_set_durable_operation(
                state_key,
                None,
                desired,
                {},
            )
        except DurableToolOperationConflict:
            if attempt + 1 == MAX_SHARED_ARTIFACT_WRITE_ATTEMPTS:
                raise
            continue
        await _ensure_call_locator(authority, locator)
        return preparation, False
    raise RuntimeError("Shared artifact materialization reservation did not converge.")


async def _materialize_bytes(
    *,
    workspace: Workspace,
    destination: str,
    content: bytes,
    preparation: _SharedArtifactMaterializationPreparation,
) -> None:
    target_digest = sha256(content).hexdigest()
    if preparation.destination_was_missing:
        try:
            mutation = await workspace.create_bytes(destination, content)
        except FileExistsError:
            await _require_workspace_content(
                workspace,
                destination,
                expected_content=content,
            )
        else:
            if mutation.after_sha256 != target_digest or mutation.after_bytes != len(content):
                raise RuntimeError(
                    "Workspace create returned inconsistent materialization evidence."
                )
    else:
        current = await workspace.read_bytes(destination, max_bytes=len(content) + 1)
        if (
            not current.truncated
            and current.total_bytes == len(content)
            and sha256(current.content).hexdigest() == target_digest
        ):
            return
        expected_revision = preparation.expected_destination_revision
        if expected_revision is None:
            raise RuntimeError("Shared artifact overwrite preparation lost its revision.")
        try:
            mutation = await workspace.replace_bytes(
                destination,
                content,
                expected_revision=expected_revision,
            )
        except WorkspaceRevisionMismatchError:
            await _require_workspace_content(
                workspace,
                destination,
                expected_content=content,
            )
        else:
            if mutation.after_sha256 != target_digest or mutation.after_bytes != len(content):
                raise RuntimeError(
                    "Workspace replace returned inconsistent materialization evidence."
                )
    await _require_workspace_content(workspace, destination, expected_content=content)


async def _finalize_materialization(
    *,
    authority: Any,
    preparation: _SharedArtifactMaterializationPreparation,
) -> SharedArtifactMaterializationReceipt:
    state_key = _materialization_state_storage_key(preparation.receipt.operation_id)
    expected = _seal_exact_durable_record(
        authority,
        preparation.model_dump(mode="json"),
        "shared artifact materialization preparation",
    )
    desired = _seal_exact_durable_record(
        authority,
        preparation.receipt.model_dump(mode="json"),
        "shared artifact materialization receipt",
    )
    with suppress(DurableToolOperationConflict):
        await authority.compare_and_set_durable_operation(
            state_key,
            expected,
            desired,
            {},
        )
    persisted = await authority.load_durable_operation(state_key)
    if persisted is None:
        raise RuntimeError("Shared artifact materialization receipt was not persisted.")
    _seal_exact_durable_record(
        authority,
        persisted,
        "persisted shared artifact materialization receipt",
    )
    receipt = SharedArtifactMaterializationReceipt.model_validate(persisted)
    _require_same_materialization(receipt, preparation.receipt)
    return receipt


async def _require_workspace_content(
    workspace: Workspace,
    path: str,
    *,
    expected_content: bytes,
) -> None:
    try:
        observed = await workspace.read_bytes(path, max_bytes=len(expected_content) + 1)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError):
        raise _SharedArtifactRefusal("destination_invalid") from None
    if (
        observed.truncated
        or observed.total_bytes != len(expected_content)
        or observed.content != expected_content
    ):
        raise _SharedArtifactRefusal("destination_conflict")


async def _settle_despite_cancellation(awaitable: Coroutine[Any, Any, Any]) -> Any:
    task = asyncio.create_task(awaitable)
    outcome = await await_shielded_task_outcome(task)
    if outcome.cancellation is not None:
        if outcome.error is not None:
            outcome.cancellation.add_note(
                f"Shared artifact settlement also failed: {type(outcome.error).__name__}."
            )
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=outcome.cancellation,
        )
        if outcome.error is not None:
            raise outcome.cancellation from outcome.error
        raise outcome.cancellation
    if outcome.error is not None:
        raise outcome.error
    return outcome.result


def _call_locator(
    *,
    kind: Literal["publication", "materialization"],
    ctx: ToolContext,
    authority: Any,
    operation_id: str,
    state_storage_key: str,
) -> _SharedArtifactCallLocator:
    return _SharedArtifactCallLocator(
        kind=kind,
        session_id=ctx.session_id,
        parent_run_epoch=authority.parent_run_epoch,
        model_step_id=authority.model_step_id,
        model_attempt_id=authority.model_attempt_id,
        tool_round_id=authority.tool_round_id,
        tool_call_id=authority.tool_call_id,
        tool_name=authority.tool_name,
        idempotency_key=authority.idempotency_key,
        effective_arguments_sha256=authority.effective_arguments_sha256,
        execution_profile_fingerprint=authority.execution_profile_fingerprint,
        operation_id=operation_id,
        state_storage_key=state_storage_key,
    )


async def _ensure_call_locator(authority: Any, locator: _SharedArtifactCallLocator) -> None:
    key = _call_locator_storage_key(
        session_id=locator.session_id,
        parent_run_epoch=locator.parent_run_epoch,
        model_step_id=locator.model_step_id,
        model_attempt_id=locator.model_attempt_id,
        tool_round_id=locator.tool_round_id,
        tool_call_id=locator.tool_call_id,
        tool_name=locator.tool_name,
        idempotency_key=locator.idempotency_key,
    )
    desired = _seal_exact_durable_record(
        authority,
        locator.model_dump(mode="json"),
        "shared artifact call locator",
    )
    raw = await authority.load_durable_operation(key)
    if raw == desired:
        return
    if raw is not None:
        raise RuntimeError("Shared artifact call locator conflicts with durable evidence.")
    try:
        await authority.compare_and_set_durable_operation(key, None, desired, {})
    except DurableToolOperationConflict:
        raw = await authority.load_durable_operation(key)
        if raw != desired:
            raise RuntimeError("Shared artifact call locator did not converge.") from None


async def _recover_shared_artifact_call(
    *,
    kind: Literal["publication", "materialization"],
    tool_name: str,
    policy: SharedArtifactPolicy,
    parent_session_id: str,
    parent_run_epoch: int,
    execution_profile_fingerprint: str | None,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    idempotency_key: str,
    arguments: dict[str, Any],
    environment_name: str | None,
    load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
    recovery_authority: DurableToolRecoveryAuthority | None,
) -> ToolResult | None:
    if execution_profile_fingerprint is None:
        return _shared_artifact_error_result("recovery_authority_unavailable")
    key = _call_locator_storage_key(
        session_id=parent_session_id,
        parent_run_epoch=parent_run_epoch,
        model_step_id=model_step_id,
        model_attempt_id=model_attempt_id,
        tool_round_id=tool_round_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        idempotency_key=idempotency_key,
    )
    raw_locator = await load_operation(key)
    if raw_locator is None:
        return None
    try:
        locator = _SharedArtifactCallLocator.model_validate(raw_locator)
        arguments_sha256 = sha256(
            canonical_durable_json_bytes(arguments, "effective_arguments")
        ).hexdigest()
    except Exception:
        return _shared_artifact_error_result("recovery_evidence_invalid")
    if (
        locator.kind != kind
        or locator.session_id != parent_session_id
        or locator.parent_run_epoch != parent_run_epoch
        or locator.model_step_id != model_step_id
        or locator.model_attempt_id != model_attempt_id
        or locator.tool_round_id != tool_round_id
        or locator.tool_call_id != tool_call_id
        or locator.tool_name != tool_name
        or locator.idempotency_key != idempotency_key
        or locator.effective_arguments_sha256 != arguments_sha256
        or locator.execution_profile_fingerprint != execution_profile_fingerprint
    ):
        return _shared_artifact_error_result("recovery_evidence_invalid")
    raw_state = await load_operation(locator.state_storage_key)
    if raw_state is None:
        return None
    try:
        if kind == "publication":
            path = _tool_path_argument(arguments, "path")
            if not _path_allowed(path, policy.publish_path_prefixes):
                raise ValueError("publication path is outside the recovered policy")
            path_sha256 = sha256(path.encode("utf-8")).hexdigest()
            if raw_state.get("record_type") == _PUBLICATION_RECEIPT_RECORD_TYPE:
                receipt = SharedArtifactPublicationReceipt.model_validate(raw_state)
            else:
                preparation = _SharedArtifactPublicationPreparation.model_validate(raw_state)
                receipt = preparation.receipt
            if receipt.operation_id != locator.operation_id:
                raise ValueError("publication operation mismatch")
            if (
                receipt.source_path_sha256 != path_sha256
                or receipt.policy_fingerprint != policy.fingerprint
            ):
                raise ValueError("publication recovery arguments mismatch")
            if raw_state.get("record_type") != _PUBLICATION_RECEIPT_RECORD_TYPE:
                if recovery_authority is None:
                    return None
                recovered = await _recover_prepared_publication(
                    preparation=preparation,
                    environment_name=environment_name,
                    authority=recovery_authority,
                    load_operation=load_operation,
                    state_storage_key=locator.state_storage_key,
                )
                if recovered is None:
                    return None
                receipt = recovered
            return _publication_result(receipt, recovered=True)
        if type(arguments) is not dict or set(arguments) != {"ref", "destination"}:
            raise ValueError("materialization recovery arguments are invalid")
        reference = SharedArtifactRef.from_opaque_ref(arguments["ref"])
        destination = _normalize_workspace_path(arguments["destination"], "destination")
        if not _path_allowed(destination, policy.materialize_path_prefixes):
            raise ValueError("materialization path is outside the recovered policy")
        destination_sha256 = sha256(destination.encode("utf-8")).hexdigest()
        if raw_state.get("record_type") == _MATERIALIZATION_RECEIPT_RECORD_TYPE:
            receipt = SharedArtifactMaterializationReceipt.model_validate(raw_state)
        else:
            preparation = _SharedArtifactMaterializationPreparation.model_validate(raw_state)
            receipt = preparation.receipt
        if receipt.operation_id != locator.operation_id:
            raise ValueError("materialization operation mismatch")
        if (
            receipt.reference != reference
            or receipt.destination_path_sha256 != destination_sha256
            or receipt.policy_fingerprint != policy.fingerprint
        ):
            raise ValueError("materialization recovery arguments mismatch")
        if raw_state.get("record_type") != _MATERIALIZATION_RECEIPT_RECORD_TYPE:
            if recovery_authority is None:
                return None
            recovered = await _recover_prepared_materialization(
                preparation=preparation,
                destination=destination,
                authority=recovery_authority,
                load_operation=load_operation,
                state_storage_key=locator.state_storage_key,
            )
            if recovered is None:
                return None
            receipt = recovered
        return _materialization_result(receipt, recovered=True)
    except Exception:
        return _shared_artifact_error_result("recovery_evidence_invalid")


async def _recover_prepared_publication(
    *,
    preparation: _SharedArtifactPublicationPreparation,
    environment_name: str | None,
    authority: DurableToolRecoveryAuthority,
    load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
    state_storage_key: str,
) -> SharedArtifactPublicationReceipt | None:
    artifact_reader = authority.artifact_reader
    reference = preparation.receipt.reference
    if (
        artifact_reader is None
        or artifact_reader.id != reference.artifact_store_id
        or authority.environment_name != environment_name
    ):
        return None
    try:
        artifact = copy_artifact_read_result(
            await artifact_reader.read_bytes(
                reference.artifact_id,
                max_bytes=reference.size_bytes,
            ),
            expected_artifact_id=reference.artifact_id,
            max_content_bytes=reference.size_bytes,
        )
    except FileNotFoundError:
        return None
    if (
        artifact.truncated
        or artifact.total_bytes != reference.size_bytes
        or "sha256:" + sha256(artifact.content).hexdigest() != reference.content_digest
    ):
        raise ValueError("prepared publication artifact content does not match")
    _require_artifact_metadata(
        artifact.metadata,
        reference=reference,
        grant=preparation.grant,
        expected_filename=preparation.artifact_filename,
        expected_agent_name=authority.agent_name,
        expected_environment_name=environment_name,
        expected_metadata=_publication_artifact_metadata(preparation),
    )
    desired = preparation.receipt.model_dump(mode="json")
    try:
        await authority.compare_and_set_operation(
            state_storage_key,
            preparation.model_dump(mode="json"),
            desired,
            {
                _grant_storage_key(reference.access_grant_id): preparation.grant.model_dump(
                    mode="json"
                )
            },
        )
    except Exception:
        if await load_operation(state_storage_key) != desired:
            raise
    persisted = await load_operation(state_storage_key)
    receipt = SharedArtifactPublicationReceipt.model_validate(persisted)
    _require_same_publication(receipt, preparation.receipt)
    return receipt


async def _recover_prepared_materialization(
    *,
    preparation: _SharedArtifactMaterializationPreparation,
    destination: str,
    authority: DurableToolRecoveryAuthority,
    load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
    state_storage_key: str,
) -> SharedArtifactMaterializationReceipt | None:
    workspace = authority.workspace
    receipt = preparation.receipt
    if workspace is None or getattr(workspace, "id", None) != receipt.destination_workspace_id:
        return None
    try:
        observed = await workspace.read_bytes(
            destination,
            max_bytes=receipt.reference.size_bytes + 1,
        )
    except FileNotFoundError:
        return None
    if (
        observed.truncated
        or observed.total_bytes != receipt.reference.size_bytes
        or "sha256:" + sha256(observed.content).hexdigest() != receipt.reference.content_digest
    ):
        raise ValueError("prepared materialization workspace content does not match")
    desired = receipt.model_dump(mode="json")
    try:
        await authority.compare_and_set_operation(
            state_storage_key,
            preparation.model_dump(mode="json"),
            desired,
            {},
        )
    except Exception:
        if await load_operation(state_storage_key) != desired:
            raise
    persisted = await load_operation(state_storage_key)
    recovered = SharedArtifactMaterializationReceipt.model_validate(persisted)
    _require_same_materialization(recovered, receipt)
    return recovered


def _publication_result(
    receipt: SharedArtifactPublicationReceipt,
    *,
    recovered: bool,
) -> ToolResult:
    opaque_ref = receipt.reference.to_opaque_ref()
    structured = {
        "shared_artifact_kind": "publication",
        "opaque_ref": opaque_ref,
        "shared_artifact_ref": receipt.reference.model_dump(mode="json"),
        "publication_receipt": receipt.model_dump(mode="json"),
        "recovered_from_durable_receipt": recovered,
    }
    return ToolResult(
        content=opaque_ref,
        structured=structured,
    )


def _materialization_result(
    receipt: SharedArtifactMaterializationReceipt,
    *,
    recovered: bool,
) -> ToolResult:
    structured = {
        "shared_artifact_kind": "materialization",
        "opaque_ref": receipt.reference.to_opaque_ref(),
        "shared_artifact_ref": receipt.reference.model_dump(mode="json"),
        "materialization_receipt": receipt.model_dump(mode="json"),
        "recovered_from_durable_receipt": recovered,
    }
    return ToolResult(
        content=(
            f"Materialized {receipt.bytes_written} bytes into the current governed workspace."
        ),
        structured=structured,
    )


def _shared_artifact_error_result(code: str) -> ToolResult:
    code = require_durable_clean_nonblank(code, "shared artifact error code")
    return ToolResult(
        content=f"Shared artifact operation refused: {code}.",
        structured={"error": code},
        is_error=True,
    )


def _require_artifact_metadata(
    artifact: ArtifactMetadata,
    *,
    reference: SharedArtifactRef,
    grant: SharedArtifactGrant,
    expected_filename: str,
    expected_agent_name: str | None,
    expected_environment_name: str | None,
    expected_metadata: Mapping[str, Any],
) -> None:
    if type(artifact) is not ArtifactMetadata or (
        artifact.id != reference.artifact_id
        or artifact.filename != expected_filename
        or artifact.content_type != grant.content_type
        or artifact.size_bytes != reference.size_bytes
        or artifact.scope is not ArtifactScope.SESSION
        or artifact.session_id != reference.source_session_id
        or artifact.agent_name != expected_agent_name
        or artifact.environment_name != expected_environment_name
        or dict(artifact.metadata) != dict(expected_metadata)
    ):
        raise RuntimeError("Artifact store returned inconsistent shared artifact metadata.")


def _operation_id_from_grant(reference: SharedArtifactRef) -> str:
    return "sao_" + reference.access_grant_id.removeprefix("sag_")


def _materialization_operation_id(
    *,
    lineage: _SessionLineageNode,
    workspace_id: str,
    destination_path_sha256: str,
    reference: SharedArtifactRef,
    policy_fingerprint: str,
) -> str:
    digest = sha256(
        canonical_durable_json_bytes(
            {
                "protocol": "cayu.shared-artifact-materialization.v1",
                "destination_session_id": lineage.session_id,
                "destination_session_instance_id": lineage.session_instance_id,
                "destination_workspace_id": workspace_id,
                "destination_path_sha256": destination_path_sha256,
                "reference": reference.model_dump(mode="json"),
                "policy_fingerprint": policy_fingerprint,
            },
            "shared_artifact_materialization_identity",
        )
    ).hexdigest()
    return "sao_" + digest[:32]


def _publication_state_storage_key(operation_id: str) -> str:
    return _PUBLICATION_STATE_KEY_PREFIX + operation_id


def _materialization_state_storage_key(operation_id: str) -> str:
    return _MATERIALIZATION_STATE_KEY_PREFIX + operation_id


def _call_locator_storage_key(
    *,
    session_id: str,
    parent_run_epoch: int,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    tool_name: str,
    idempotency_key: str,
) -> str:
    digest = sha256(
        canonical_durable_json_bytes(
            {
                "protocol": "cayu.shared-artifact-call-locator.v1",
                "session_id": session_id,
                "parent_run_epoch": parent_run_epoch,
                "model_step_id": model_step_id,
                "model_attempt_id": model_attempt_id,
                "tool_round_id": tool_round_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "idempotency_key": idempotency_key,
            },
            "shared_artifact_call_locator_identity",
        )
    ).hexdigest()
    return _CALL_LOCATOR_KEY_PREFIX + digest


def _require_same_publication(
    observed: SharedArtifactPublicationReceipt,
    expected: SharedArtifactPublicationReceipt,
) -> None:
    if observed.model_dump(mode="json", exclude={"published_at"}) != expected.model_dump(
        mode="json",
        exclude={"published_at"},
    ):
        raise RuntimeError("Shared artifact publication identity conflicts with durable evidence.")


def _require_same_materialization(
    observed: SharedArtifactMaterializationReceipt,
    expected: SharedArtifactMaterializationReceipt,
) -> None:
    if observed.model_dump(mode="json", exclude={"materialized_at"}) != expected.model_dump(
        mode="json",
        exclude={"materialized_at"},
    ):
        raise RuntimeError(
            "Shared artifact materialization identity conflicts with durable evidence."
        )


__all__ = [
    "DEFAULT_SHARED_ARTIFACT_GRANT_TTL_SECONDS",
    "DEFAULT_SHARED_ARTIFACT_MAX_BYTES",
    "DEFAULT_SHARED_ARTIFACT_MAX_LINEAGE_DEPTH",
    "DEFAULT_SHARED_ARTIFACT_MAX_PUBLICATIONS",
    "MATERIALIZE_SHARED_ARTIFACT_TOOL_NAME",
    "MAX_SHARED_ARTIFACT_BYTES",
    "MAX_SHARED_ARTIFACT_GRANT_TTL_SECONDS",
    "MAX_SHARED_ARTIFACT_PUBLICATIONS",
    "PUBLISH_WORKSPACE_ARTIFACT_TOOL_NAME",
    "SHARED_ARTIFACT_REFERENCE_PREFIX",
    "SHARED_ARTIFACT_SCHEMA_VERSION",
    "MaterializeSharedArtifactTool",
    "PublishWorkspaceArtifactTool",
    "SharedArtifactAudience",
    "SharedArtifactAuthorizationError",
    "SharedArtifactGrant",
    "SharedArtifactGrantStatus",
    "SharedArtifactMaterializationReceipt",
    "SharedArtifactPolicy",
    "SharedArtifactPublicationReceipt",
    "SharedArtifactRef",
    "authorize_shared_artifact_materialization",
    "revoke_shared_artifact_grant",
]
