"""Durable provider-independent targeted tool grant contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.core.events import Event, EventType, copy_event, event_with_runtime_payload_authority
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.runtime.tool_catalogue import (
    ToolCatalogSnapshot,
    ToolDescriptor,
    validate_canonical_tool_id,
    validate_tool_catalogue_revision,
    validate_tool_descriptor_version,
)
from cayu.runtime.tool_exposure import ToolCapabilityCeiling
from cayu.vaults.redaction import REDACTED_SECRET

TARGETED_TOOL_GRANT_SCHEMA_VERSION = 1
TARGETED_TOOL_GRANT_MAX_REQUESTS = 32
TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS = 256
TARGETED_TOOL_GRANT_MAX_CALLS = 32
TARGETED_TOOL_GRANT_DEFAULT_LIFETIME_SECONDS = 300
TARGETED_TOOL_GRANT_MAX_LIFETIME_SECONDS = 86_400
TARGETED_TOOL_GRANT_REQUEST_ID_MAX_CHARS = 256
TARGETED_TOOL_GRANT_REQUEST_ID_MAX_BYTES = 512
TARGETED_TOOL_GRANT_ORIGIN_MAX_BYTES = 512
TARGETED_TOOL_GRANT_REVOCATION_REASON_MAX_BYTES = 512
TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES = 2048
TARGETED_TOOL_GRANT_EXECUTION_ID_MAX_BYTES = 2048
TARGETED_TOOL_REFERENCE_MAX_BYTES = 256
TARGETED_TOOL_REFERENCE_FIELD_NAME = "tool_ref"
TARGETED_TOOL_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
TARGETED_TOOL_TRANSCRIPT_REFERENCE = REDACTED_SECRET


def _bounded_utf8_text(value: str, field_name: str, *, max_bytes: int) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} cannot exceed {max_bytes} UTF-8 bytes.")
    return value


def validate_targeted_tool_digest(value: str, field_name: str) -> str:
    """Validate one canonical targeted-tool SHA-256 identity."""

    value = require_durable_clean_nonblank(value, field_name)
    prefix = "sha256:"
    digest = value[len(prefix) :] if value.startswith(prefix) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a canonical SHA-256 identity.")
    return value


def validate_targeted_tool_grant_revocation_reason(
    value: str,
    field_name: str = "reason",
) -> str:
    """Validate one bounded durable operator-supplied revocation reason."""

    return _bounded_utf8_text(
        value,
        field_name,
        max_bytes=TARGETED_TOOL_GRANT_REVOCATION_REASON_MAX_BYTES,
    )


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _sha256_identity(value: object, field_name: str) -> str:
    return f"sha256:{sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()}"


class TargetedToolGrant(BaseModel):
    """Application request for bounded addressability to one registered tool.

    A request contains no executable, schema, policy, or authorization material.
    The runtime resolves all such identity from the registered catalogue.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TARGETED_TOOL_GRANT_SCHEMA_VERSION
    request_id: str = Field(max_length=TARGETED_TOOL_GRANT_REQUEST_ID_MAX_CHARS)
    tool_id: str
    max_calls: StrictInt = Field(default=1, ge=1, le=TARGETED_TOOL_GRANT_MAX_CALLS)
    lifetime_seconds: StrictInt = Field(
        default=TARGETED_TOOL_GRANT_DEFAULT_LIFETIME_SECONDS,
        ge=1,
        le=TARGETED_TOOL_GRANT_MAX_LIFETIME_SECONDS,
    )
    expires_after_interaction: Literal[True] = True
    origin: str | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TARGETED_TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Targeted tool grant schema_version must be the integer 1.")
        return value

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "request_id",
            max_bytes=TARGETED_TOOL_GRANT_REQUEST_ID_MAX_BYTES,
        )

    @field_validator("expires_after_interaction", mode="before")
    @classmethod
    def validate_interaction_expiry(cls, value: object) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError("Targeted grants must expire after the issuing interaction.")
        return value

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_utf8_text(
            value,
            "origin",
            max_bytes=TARGETED_TOOL_GRANT_ORIGIN_MAX_BYTES,
        )


def copy_targeted_tool_grant(value: TargetedToolGrant) -> TargetedToolGrant:
    """Return a detached, fully revalidated targeted grant request."""

    if type(value) is not TargetedToolGrant:
        raise TypeError("value must be a TargetedToolGrant.")
    return TargetedToolGrant.model_validate(value.model_dump(mode="python"))


def validate_targeted_tool_grants(value: object) -> tuple[TargetedToolGrant, ...]:
    """Copy and validate one bounded, unambiguous grant-request collection."""

    if isinstance(value, str | bytes | bytearray | dict | BaseModel):
        raise TypeError("tool_grants must be an iterable of TargetedToolGrant values.")
    if not isinstance(value, Iterable):
        raise TypeError("tool_grants must be an iterable of TargetedToolGrant values.")
    iterator = iter(value)
    grants: list[TargetedToolGrant] = []
    for index, item in enumerate(iterator):
        if index >= TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError(
                f"tool_grants cannot contain more than {TARGETED_TOOL_GRANT_MAX_REQUESTS} requests."
            )
        grant = (
            copy_targeted_tool_grant(item)
            if isinstance(item, TargetedToolGrant)
            else TargetedToolGrant.model_validate(item)
        )
        grants.append(grant)
    request_ids = tuple(grant.request_id for grant in grants)
    tool_ids = tuple(grant.tool_id for grant in grants)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("tool_grants must contain unique request_id values.")
    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError("tool_grants must not target the same tool more than once.")
    return tuple(grants)


class PreparedTargetedToolGrant(BaseModel):
    """Callable-free grant material resolved from one catalogue snapshot."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    request: TargetedToolGrant
    tool_name: str
    catalogue_revision: str
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> object:
        if isinstance(value, TargetedToolGrant):
            return value.model_dump(mode="python")
        return value

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_name",
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)


def prepare_targeted_tool_grants(
    grants: tuple[TargetedToolGrant, ...],
    *,
    catalogue: ToolCatalogSnapshot,
    capability_ceiling: ToolCapabilityCeiling,
) -> tuple[PreparedTargetedToolGrant, ...]:
    """Resolve requests against one exact registered catalogue and durable ceiling."""

    grants = validate_targeted_tool_grants(grants)
    if type(catalogue) is not ToolCatalogSnapshot:
        raise TypeError("catalogue must be a ToolCatalogSnapshot.")
    if type(capability_ceiling) is not ToolCapabilityCeiling:
        raise TypeError("capability_ceiling must be a ToolCapabilityCeiling.")
    admitted_names = frozenset(capability_ceiling.tool_names)
    prepared: list[PreparedTargetedToolGrant] = []
    for grant in grants:
        try:
            descriptor = catalogue.descriptor_for_id(grant.tool_id)
        except KeyError:
            raise ValueError("A targeted tool grant names an unregistered tool_id.") from None
        if descriptor.name not in admitted_names:
            raise ValueError("A targeted tool grant exceeds the session capability ceiling.")
        prepared.append(
            PreparedTargetedToolGrant(
                request=grant,
                tool_name=descriptor.name,
                catalogue_revision=catalogue.revision,
                descriptor_version=descriptor.version,
                schema_fingerprint=descriptor.schema_fingerprint,
            )
        )
    return tuple(prepared)


def _targeted_tool_grant_batch_fingerprint(material: list[dict[str, object]]) -> str:
    if not 1 <= len(material) <= TARGETED_TOOL_GRANT_MAX_REQUESTS:
        raise ValueError("Targeted grant batch fingerprint requires a bounded non-empty batch.")
    return _sha256_identity(
        {
            "record_type": "cayu.targeted-tool-grant-batch",
            "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
            "grants": sorted(material, key=lambda item: str(item["request_id"])),
        },
        "targeted_tool_grant_batch",
    )


def prepared_targeted_tool_grant_batch_fingerprint(
    prepared: tuple[PreparedTargetedToolGrant, ...],
) -> str:
    """Bind one requested interaction batch before its durable admission."""

    if type(prepared) is not tuple or any(
        type(grant) is not PreparedTargetedToolGrant for grant in prepared
    ):
        raise TypeError("prepared must be a tuple of PreparedTargetedToolGrant values.")
    return _targeted_tool_grant_batch_fingerprint(
        [
            {
                "request_id": grant.request.request_id,
                "tool_id": grant.request.tool_id,
                "tool_name": grant.tool_name,
                "catalogue_revision": grant.catalogue_revision,
                "descriptor_version": grant.descriptor_version,
                "schema_fingerprint": grant.schema_fingerprint,
                "max_calls": grant.request.max_calls,
                "lifetime_seconds": grant.request.lifetime_seconds,
                "expires_after_interaction": grant.request.expires_after_interaction,
                "origin": grant.request.origin,
            }
            for grant in prepared
        ]
    )


def persisted_targeted_tool_grant_batch_fingerprint(
    records: tuple[TargetedToolGrantRecord, ...],
) -> str:
    """Rebuild the admitted batch fingerprint from immutable durable records."""

    if type(records) is not tuple or any(
        type(record) is not TargetedToolGrantRecord for record in records
    ):
        raise TypeError("records must be a tuple of TargetedToolGrantRecord values.")
    copied = tuple(copy_targeted_tool_grant_record(record) for record in records)
    return _targeted_tool_grant_batch_fingerprint(
        [
            {
                "request_id": record.request_id,
                "tool_id": record.tool_id,
                "tool_name": record.tool_name,
                "catalogue_revision": record.catalogue_revision,
                "descriptor_version": record.descriptor_version,
                "schema_fingerprint": record.schema_fingerprint,
                "max_calls": record.max_calls,
                "lifetime_seconds": int((record.expires_at - record.issued_at).total_seconds()),
                "expires_after_interaction": record.expires_after_interaction,
                "origin": record.origin,
            }
            for record in copied
        ]
    )


def validate_targeted_tool_grant_batch_evidence(
    records: tuple[TargetedToolGrantRecord, ...],
    interaction_started_event: Event,
) -> None:
    """Require durable records to match the atomically admitted interaction intent."""

    if type(records) is not tuple:
        raise TypeError("records must be a tuple of TargetedToolGrantRecord values.")
    copied = tuple(copy_targeted_tool_grant_record(record) for record in records)
    event = copy_event(interaction_started_event)
    if event.type is not EventType.INTERACTION_STARTED or event.interaction_id is None:
        raise ValueError("Targeted grant batch evidence requires an interaction start event.")
    if any(
        record.session_id != event.session_id or record.interaction_id != event.interaction_id
        for record in copied
    ):
        raise ValueError("Targeted grant batch conflicts with its interaction scope.")
    count = event.payload.get("targeted_tool_grant_count")
    fingerprint = event.payload.get("targeted_tool_grant_batch_fingerprint")
    if count is None and fingerprint is None and not copied:
        return
    if (
        type(count) is not int
        or not 1 <= count <= TARGETED_TOOL_GRANT_MAX_REQUESTS
        or type(fingerprint) is not str
        or len(fingerprint) != 71
        or not fingerprint.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in fingerprint[7:])
    ):
        raise ValueError("Targeted grant batch admission evidence is malformed.")
    if count != len(copied) or fingerprint != persisted_targeted_tool_grant_batch_fingerprint(
        copied
    ):
        raise ValueError("Targeted grant batch conflicts with admitted interaction authority.")


def targeted_tool_view_generation_id(*, session_id: str, root_invocation_id: str) -> str:
    """Return a branch-local generation identity that is never inherited by forks."""

    session_id = require_durable_clean_nonblank(session_id, "session_id")
    root_invocation_id = require_durable_clean_nonblank(
        root_invocation_id,
        "root_invocation_id",
    )
    return _sha256_identity(
        {
            "record_type": "cayu.targeted-tool-view-generation",
            "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
            "session_id": session_id,
            "root_invocation_id": root_invocation_id,
        },
        "targeted_tool_view_generation",
    )


class TargetedToolGrantRecord(BaseModel):
    """Immutable durable state for one issued targeted reference."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TARGETED_TOOL_GRANT_SCHEMA_VERSION
    grant_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    tool_ref: str
    request_id: str = Field(max_length=TARGETED_TOOL_GRANT_REQUEST_ID_MAX_CHARS)
    session_id: str
    interaction_id: str
    generation_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    agent_name: str
    task_id: str | None = None
    environment_name: str | None = None
    principal: str | None = None
    tenant: str | None = None
    tool_id: str
    tool_name: str
    catalogue_revision: str
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    issued_at: datetime
    expires_at: datetime
    expires_after_interaction: StrictBool = True
    max_calls: StrictInt = Field(ge=1, le=TARGETED_TOOL_GRANT_MAX_CALLS)
    used_calls: StrictInt = Field(default=0, ge=0, le=TARGETED_TOOL_GRANT_MAX_CALLS)
    origin: str | None = None
    revoked_at: datetime | None = None
    revocation_reason: str | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TARGETED_TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Targeted grant record schema_version must be the integer 1.")
        return value

    @field_validator("grant_id", "generation_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("request_id")
    @classmethod
    def validate_request_identity(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "request_id",
            max_bytes=TARGETED_TOOL_GRANT_REQUEST_ID_MAX_BYTES,
        )

    @field_validator("tool_ref")
    @classmethod
    def validate_tool_ref(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_ref",
            max_bytes=TARGETED_TOOL_REFERENCE_MAX_BYTES,
        )

    @field_validator("session_id", "interaction_id", "agent_name", "tool_name")
    @classmethod
    def validate_scope_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)

    @field_validator("task_id", "environment_name", "principal", "tenant")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_utf8_text(
            value,
            "origin",
            max_bytes=TARGETED_TOOL_GRANT_ORIGIN_MAX_BYTES,
        )

    @field_validator("revocation_reason")
    @classmethod
    def validate_revocation_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_targeted_tool_grant_revocation_reason(value, "revocation_reason")

    @field_validator("issued_at", "expires_at", "revoked_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> TargetedToolGrantRecord:
        if not self.expires_after_interaction:
            raise ValueError("Targeted grants must expire after their issuing interaction.")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at.")
        lifetime = self.expires_at - self.issued_at
        lifetime_seconds = float(lifetime.total_seconds())
        if not lifetime_seconds.is_integer():
            raise ValueError("Targeted grant lifetime must contain whole seconds.")
        if lifetime > timedelta(seconds=TARGETED_TOOL_GRANT_MAX_LIFETIME_SECONDS):
            raise ValueError("Targeted grant lifetime exceeds the supported maximum.")
        if self.used_calls > self.max_calls:
            raise ValueError("used_calls cannot exceed max_calls.")
        if (self.revoked_at is None) != (self.revocation_reason is None):
            raise ValueError("Revocation timestamp and reason must be present together.")
        if self.revoked_at is not None and self.revoked_at < self.issued_at:
            raise ValueError("revoked_at cannot precede issued_at.")
        return self

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.used_calls


class TargetedToolGrantInspection(BaseModel):
    """Public grant state with aliased scope and no principal authority."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TARGETED_TOOL_GRANT_SCHEMA_VERSION
    grant_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    tool_ref: str
    request_id: str = Field(max_length=TARGETED_TOOL_GRANT_REQUEST_ID_MAX_CHARS)
    session_id: str
    interaction_id: str
    generation_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    agent_name: str
    task_id: str | None = None
    environment_name: str | None = None
    tool_id: str
    tool_name: str
    catalogue_revision: str
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    issued_at: datetime
    expires_at: datetime
    expires_after_interaction: Literal[True] = True
    max_calls: StrictInt = Field(ge=1, le=TARGETED_TOOL_GRANT_MAX_CALLS)
    used_calls: StrictInt = Field(ge=0, le=TARGETED_TOOL_GRANT_MAX_CALLS)
    origin: str | None = None
    revoked_at: datetime | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TARGETED_TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Targeted grant inspection schema_version must be the integer 1.")
        return value

    @field_validator("expires_after_interaction", mode="before")
    @classmethod
    def validate_interaction_expiry(cls, value: object) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError("Targeted grants must expire after their issuing interaction.")
        return value

    @field_validator("tool_ref")
    @classmethod
    def validate_tool_ref(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_ref",
            max_bytes=TARGETED_TOOL_REFERENCE_MAX_BYTES,
        )

    @field_validator("request_id")
    @classmethod
    def validate_request_identity(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "request_id",
            max_bytes=TARGETED_TOOL_GRANT_REQUEST_ID_MAX_BYTES,
        )

    @field_validator("session_id", "interaction_id", "agent_name", "tool_name")
    @classmethod
    def validate_scope_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("task_id", "environment_name")
    @classmethod
    def validate_optional_scope_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_utf8_text(
            value,
            "origin",
            max_bytes=TARGETED_TOOL_GRANT_ORIGIN_MAX_BYTES,
        )

    @field_validator("issued_at", "expires_at", "revoked_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> TargetedToolGrantInspection:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at.")
        lifetime = self.expires_at - self.issued_at
        if not float(lifetime.total_seconds()).is_integer():
            raise ValueError("Targeted grant lifetime must contain whole seconds.")
        if lifetime > timedelta(seconds=TARGETED_TOOL_GRANT_MAX_LIFETIME_SECONDS):
            raise ValueError("Targeted grant lifetime exceeds the supported maximum.")
        if self.used_calls > self.max_calls:
            raise ValueError("used_calls cannot exceed max_calls.")
        if self.revoked_at is not None and self.revoked_at < self.issued_at:
            raise ValueError("revoked_at cannot precede issued_at.")
        return self

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.used_calls


def copy_targeted_tool_grant_record(
    value: TargetedToolGrantRecord,
) -> TargetedToolGrantRecord:
    if type(value) is not TargetedToolGrantRecord:
        raise TypeError("value must be a TargetedToolGrantRecord.")
    copied = TargetedToolGrantRecord.model_validate(value.model_dump(mode="python"))
    return validate_targeted_tool_grant_record_identity(copied)


def validate_targeted_tool_grant_record_identity(
    value: TargetedToolGrantRecord,
) -> TargetedToolGrantRecord:
    """Verify the private durable scope bound into one deterministic grant id."""

    if type(value) is not TargetedToolGrantRecord:
        raise TypeError("value must be a TargetedToolGrantRecord.")
    lifetime_seconds = int((value.expires_at - value.issued_at).total_seconds())
    expected_grant_id = _sha256_identity(
        {
            "record_type": "cayu.targeted-tool-grant",
            "schema_version": value.schema_version,
            "request_id": value.request_id,
            "session_id": value.session_id,
            "interaction_id": value.interaction_id,
            "generation_id": value.generation_id,
            "agent_name": value.agent_name,
            "task_id": value.task_id,
            "environment_name": value.environment_name,
            "principal": value.principal,
            "tenant": value.tenant,
            "tool_id": value.tool_id,
            "tool_name": value.tool_name,
            "catalogue_revision": value.catalogue_revision,
            "descriptor_version": value.descriptor_version,
            "schema_fingerprint": value.schema_fingerprint,
            "max_calls": value.max_calls,
            "lifetime_seconds": lifetime_seconds,
            "expires_after_interaction": value.expires_after_interaction,
            "origin": value.origin,
        },
        "targeted_tool_grant",
    )
    if value.grant_id != expected_grant_id:
        raise ValueError("grant_id conflicts with immutable targeted grant authority.")
    return value


def targeted_tool_grant_with_active_reference(
    record: TargetedToolGrantRecord,
    codec: PublicAuthorityAliasCodec,
) -> TargetedToolGrantRecord:
    """Project a durable grant through the keyring's current signing key."""

    record = copy_targeted_tool_grant_record(record)
    if type(codec) is not PublicAuthorityAliasCodec:
        raise TypeError("codec must be a PublicAuthorityAliasCodec.")
    return TargetedToolGrantRecord.model_validate(
        record.model_copy(
            update={
                "tool_ref": codec.encode(
                    record.grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=record.session_id,
                )
            }
        ).model_dump(mode="python")
    )


def validate_targeted_tool_grant_reference(
    record: TargetedToolGrantRecord,
    codec: PublicAuthorityAliasCodec,
) -> TargetedToolGrantRecord:
    """Require a grant record to carry an authenticated reference to itself."""

    record = copy_targeted_tool_grant_record(record)
    if type(codec) is not PublicAuthorityAliasCodec:
        raise TypeError("codec must be a PublicAuthorityAliasCodec.")
    if not codec.matches(
        record.tool_ref,
        record.grant_id,
        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
        session_id=record.session_id,
    ):
        raise ValueError("tool_ref conflicts with immutable targeted grant authority.")
    return record


def targeted_tool_grant_inspection(
    record: TargetedToolGrantRecord,
    codec: PublicAuthorityAliasCodec,
) -> TargetedToolGrantInspection:
    """Project private durable authority into a bounded public inspection view."""

    record = targeted_tool_grant_with_active_reference(record, codec)
    public_values = record.model_dump(
        mode="python",
        exclude={"principal", "tenant", "revocation_reason"},
    )
    public_values.update(
        {
            "session_id": codec.encode(record.session_id, field_name="session_id"),
            "interaction_id": codec.encode(
                record.interaction_id,
                field_name="interaction_id",
                session_id=record.session_id,
            ),
            "task_id": (
                None
                if record.task_id is None
                else codec.encode(
                    record.task_id,
                    field_name="task_id",
                    session_id=record.session_id,
                )
            ),
        }
    )
    return TargetedToolGrantInspection.model_validate(public_values)


def build_targeted_tool_grant_record(
    prepared: PreparedTargetedToolGrant,
    *,
    session_id: str,
    interaction_id: str,
    generation_id: str,
    agent_name: str,
    task_id: str | None,
    environment_name: str | None,
    principal: str | None,
    tenant: str | None,
    issued_at: datetime,
    codec: PublicAuthorityAliasCodec,
) -> TargetedToolGrantRecord:
    """Bind prepared catalogue authority to one session interaction."""

    if type(prepared) is not PreparedTargetedToolGrant:
        raise TypeError("prepared must be a PreparedTargetedToolGrant.")
    if type(codec) is not PublicAuthorityAliasCodec:
        raise TypeError("codec must be a PublicAuthorityAliasCodec.")
    issued_at = _utc_datetime(issued_at, "issued_at")
    immutable_identity = {
        "record_type": "cayu.targeted-tool-grant",
        "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
        "request_id": prepared.request.request_id,
        "session_id": session_id,
        "interaction_id": interaction_id,
        "generation_id": generation_id,
        "agent_name": agent_name,
        "task_id": task_id,
        "environment_name": environment_name,
        "principal": principal,
        "tenant": tenant,
        "tool_id": prepared.request.tool_id,
        "tool_name": prepared.tool_name,
        "catalogue_revision": prepared.catalogue_revision,
        "descriptor_version": prepared.descriptor_version,
        "schema_fingerprint": prepared.schema_fingerprint,
        "max_calls": prepared.request.max_calls,
        "lifetime_seconds": prepared.request.lifetime_seconds,
        "expires_after_interaction": prepared.request.expires_after_interaction,
        "origin": prepared.request.origin,
    }
    grant_id = _sha256_identity(immutable_identity, "targeted_tool_grant")
    return TargetedToolGrantRecord(
        grant_id=grant_id,
        tool_ref=codec.encode(
            grant_id,
            field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
            session_id=session_id,
        ),
        request_id=prepared.request.request_id,
        session_id=session_id,
        interaction_id=interaction_id,
        generation_id=generation_id,
        agent_name=agent_name,
        task_id=task_id,
        environment_name=environment_name,
        principal=principal,
        tenant=tenant,
        tool_id=prepared.request.tool_id,
        tool_name=prepared.tool_name,
        catalogue_revision=prepared.catalogue_revision,
        descriptor_version=prepared.descriptor_version,
        schema_fingerprint=prepared.schema_fingerprint,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(seconds=prepared.request.lifetime_seconds),
        expires_after_interaction=prepared.request.expires_after_interaction,
        max_calls=prepared.request.max_calls,
        origin=prepared.request.origin,
    )


class TargetedToolGrantIssueOutcome(StrEnum):
    ISSUED = "issued"
    REUSED = "reused"


class TargetedToolGrantIssueResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    records: tuple[TargetedToolGrantRecord, ...]
    outcomes: tuple[TargetedToolGrantIssueOutcome, ...]
    events: tuple[Event, ...]

    @field_validator("records", mode="before")
    @classmethod
    def copy_records(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Grant issue records must be a sequence.")
        if len(value) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Grant issue records exceed the bounded request count.")
        return tuple(
            copy_targeted_tool_grant_record(record)
            if isinstance(record, TargetedToolGrantRecord)
            else record
            for record in value
        )

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Grant issue evidence must be a sequence.")
        if len(value) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Grant issue evidence exceeds the bounded request count.")
        return tuple(copy_event(event) if isinstance(event, Event) else event for event in value)

    @field_validator("outcomes", mode="before")
    @classmethod
    def bound_outcomes(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Grant issue outcomes must be a sequence.")
        if len(value) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Grant issue outcomes exceed the bounded request count.")
        return value

    @model_validator(mode="after")
    def validate_lengths(self) -> TargetedToolGrantIssueResult:
        if len(self.records) != len(self.outcomes) or len(self.records) != len(self.events):
            raise ValueError("Grant issue records, outcomes, and events must align.")
        for record, outcome, event in zip(
            self.records,
            self.outcomes,
            self.events,
            strict=True,
        ):
            event_type = (
                EventType.TARGETED_TOOL_GRANT_ISSUED
                if outcome is TargetedToolGrantIssueOutcome.ISSUED
                else EventType.TARGETED_TOOL_GRANT_REUSED
            )
            if (
                outcome is TargetedToolGrantIssueOutcome.ISSUED
                and event.timestamp != record.issued_at
            ):
                raise ValueError("Grant issuance time conflicts with its durable record.")
            evidence_record = (
                record
                if outcome is TargetedToolGrantIssueOutcome.ISSUED
                else _targeted_tool_grant_record_at_evidence_count(
                    record,
                    event,
                    require_current=False,
                )
            )
            expected = targeted_tool_grant_event(
                evidence_record,
                event_type=event_type,
                timestamp=event.timestamp,
                outcome=outcome.value,
                event_id_suffix=outcome.value,
            )
            if event != expected:
                raise ValueError("Grant issue evidence conflicts with its durable record.")
        return self


class TargetedToolUseDisposition(StrEnum):
    BOUND = "bound"
    REJOINED = "rejoined"
    REJECTED = "rejected"


class TargetedToolUseRejectionReason(StrEnum):
    MALFORMED = "malformed"
    UNKNOWN = "unknown"
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    EXHAUSTED = "exhausted"
    CROSS_SESSION = "cross_session"
    CROSS_INTERACTION = "cross_interaction"
    CROSS_GENERATION = "cross_generation"
    PRINCIPAL_MISMATCH = "principal_mismatch"
    TENANT_MISMATCH = "tenant_mismatch"
    AGENT_MISMATCH = "agent_mismatch"
    TASK_MISMATCH = "task_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    OUT_OF_CEILING = "out_of_ceiling"
    CATALOGUE_DRIFT = "catalogue_drift"
    DESCRIPTOR_DRIFT = "descriptor_drift"
    INVALID_ARGUMENTS = "invalid_arguments"
    ALTERED_REPLAY = "altered_replay"


class TargetedToolUseRequest(BaseModel):
    """Bounded identities required to reserve one future effective invocation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    tool_ref: str
    session_id: str
    interaction_id: str
    generation_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    agent_name: str
    task_id: str | None = None
    environment_name: str | None = None
    principal: str | None = None
    tenant: str | None = None
    catalogue_revision: str
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    tool_id: str
    tool_name: str
    model_step_id: str
    outer_tool_call_id: str
    arguments_sha256: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    invocation_id: str
    expected_run_epoch: StrictInt = Field(ge=0)

    @field_validator("tool_ref")
    @classmethod
    def validate_tool_ref(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_ref",
            max_bytes=TARGETED_TOOL_REFERENCE_MAX_BYTES,
        )

    @field_validator("session_id", "interaction_id", "agent_name", "tool_name")
    @classmethod
    def validate_scope_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("model_step_id", "outer_tool_call_id", "invocation_id")
    @classmethod
    def validate_execution_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_EXECUTION_ID_MAX_BYTES,
        )

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)

    @field_validator("task_id", "environment_name", "principal", "tenant")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )


class TargetedToolUseBinding(BaseModel):
    """Permanent digest-only evidence for one accepted grant use."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TARGETED_TOOL_GRANT_SCHEMA_VERSION
    grant_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    use_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    session_id: str
    interaction_id: str
    model_step_id: str
    outer_tool_call_id: str
    arguments_sha256: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    invocation_id: str
    bound_at: datetime

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TARGETED_TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Targeted tool use schema_version must be the integer 1.")
        return value

    @field_validator("grant_id", "use_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("session_id", "interaction_id")
    @classmethod
    def validate_scope_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("model_step_id", "outer_tool_call_id", "invocation_id")
    @classmethod
    def validate_execution_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_EXECUTION_ID_MAX_BYTES,
        )

    @field_validator("bound_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _utc_datetime(value, "bound_at")

    @model_validator(mode="after")
    def validate_use_id(self) -> TargetedToolUseBinding:
        expected_use_id = _sha256_identity(
            {
                "record_type": "cayu.targeted-tool-use",
                "schema_version": self.schema_version,
                "grant_id": self.grant_id,
                "session_id": self.session_id,
                "interaction_id": self.interaction_id,
                "model_step_id": self.model_step_id,
                "outer_tool_call_id": self.outer_tool_call_id,
                "arguments_sha256": self.arguments_sha256,
                "invocation_id": self.invocation_id,
            },
            "targeted_tool_use",
        )
        if self.use_id != expected_use_id:
            raise ValueError("use_id conflicts with immutable targeted tool use evidence.")
        return self


class ResolvedTargetedToolInvocation(BaseModel):
    """Durable dual identity for one accepted ``call_tool`` envelope.

    The provider transcript retains the outer ``call_tool`` call and identifier.
    Policy and execution use the canonical effective target recorded here.  The
    opaque reference is private checkpoint material required to rejoin the
    already-bound consumption after process loss; it is never public evidence.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TARGETED_TOOL_GRANT_SCHEMA_VERSION
    dispatch_kind: Literal["gateway"] = "gateway"
    model_tool_name: Literal["call_tool"] = "call_tool"
    tool_ref: str
    grant_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    use_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    session_id: str
    interaction_id: str
    tool_id: str
    effective_tool_name: str
    catalogue_revision: str
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    model_step_id: str
    outer_tool_call_id: str
    arguments_sha256: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    invocation_id: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TARGETED_TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Resolved targeted invocation schema_version must be 1.")
        return value

    @field_validator("tool_ref")
    @classmethod
    def validate_tool_ref(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_ref",
            max_bytes=TARGETED_TOOL_REFERENCE_MAX_BYTES,
        )

    @field_validator("grant_id", "use_id")
    @classmethod
    def validate_digest_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("session_id", "interaction_id")
    @classmethod
    def validate_scope_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("effective_tool_name")
    @classmethod
    def validate_effective_tool_name(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "effective_tool_name",
            max_bytes=TARGETED_TOOL_GRANT_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)

    @field_validator("model_step_id", "outer_tool_call_id", "invocation_id")
    @classmethod
    def validate_execution_identity(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TARGETED_TOOL_GRANT_EXECUTION_ID_MAX_BYTES,
        )

    @model_validator(mode="after")
    def validate_use_id(self) -> ResolvedTargetedToolInvocation:
        expected_use_id = _sha256_identity(
            {
                "record_type": "cayu.targeted-tool-use",
                "schema_version": self.schema_version,
                "grant_id": self.grant_id,
                "session_id": self.session_id,
                "interaction_id": self.interaction_id,
                "model_step_id": self.model_step_id,
                "outer_tool_call_id": self.outer_tool_call_id,
                "arguments_sha256": self.arguments_sha256,
                "invocation_id": self.invocation_id,
            },
            "targeted_tool_use",
        )
        if self.use_id != expected_use_id:
            raise ValueError("use_id conflicts with resolved targeted tool authority.")
        return self


class RejectedTargetedToolInvocation(BaseModel):
    """Durable fail-closed outcome for a rejected ``call_tool`` envelope."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TARGETED_TOOL_GRANT_SCHEMA_VERSION
    dispatch_kind: Literal["gateway"] = "gateway"
    model_tool_name: Literal["call_tool"] = "call_tool"
    reason: TargetedToolUseRejectionReason
    rejection_event_id: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TARGETED_TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Rejected targeted invocation schema_version must be 1.")
        return value

    @field_validator("rejection_event_id")
    @classmethod
    def validate_rejection_event_id(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "rejection_event_id",
            max_bytes=TARGETED_TOOL_GRANT_EXECUTION_ID_MAX_BYTES,
        )


def targeted_tool_use_binding(
    grant_id: str,
    request: TargetedToolUseRequest,
    *,
    bound_at: datetime,
) -> TargetedToolUseBinding:
    """Build deterministic identity for an accepted digest-only binding."""

    if type(request) is not TargetedToolUseRequest:
        raise TypeError("request must be a TargetedToolUseRequest.")
    grant_id = require_durable_clean_nonblank(grant_id, "grant_id")
    use_id = _sha256_identity(
        {
            "record_type": "cayu.targeted-tool-use",
            "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
            "grant_id": grant_id,
            "session_id": request.session_id,
            "interaction_id": request.interaction_id,
            "model_step_id": request.model_step_id,
            "outer_tool_call_id": request.outer_tool_call_id,
            "arguments_sha256": request.arguments_sha256,
            "invocation_id": request.invocation_id,
        },
        "targeted_tool_use",
    )
    return TargetedToolUseBinding(
        grant_id=grant_id,
        use_id=use_id,
        session_id=request.session_id,
        interaction_id=request.interaction_id,
        model_step_id=request.model_step_id,
        outer_tool_call_id=request.outer_tool_call_id,
        arguments_sha256=request.arguments_sha256,
        invocation_id=request.invocation_id,
        bound_at=bound_at,
    )


def targeted_tool_use_scope_rejection_reason(
    record: TargetedToolGrantRecord,
    request: TargetedToolUseRequest,
) -> TargetedToolUseRejectionReason | None:
    """Validate execution identity against immutable grant authority.

    Exact retries reuse an existing consumption binding, but they must still
    present the same scope and descriptor authority as the original admission.
    Availability checks are intentionally separate because expiry, revocation,
    and exhaustion cannot undo a use that was already durably accepted.
    """

    if type(record) is not TargetedToolGrantRecord:
        raise TypeError("record must be a TargetedToolGrantRecord.")
    if type(request) is not TargetedToolUseRequest:
        raise TypeError("request must be a TargetedToolUseRequest.")
    if request.session_id != record.session_id:
        return TargetedToolUseRejectionReason.CROSS_SESSION
    if request.interaction_id != record.interaction_id:
        return TargetedToolUseRejectionReason.CROSS_INTERACTION
    if request.generation_id != record.generation_id:
        return TargetedToolUseRejectionReason.CROSS_GENERATION
    if request.principal != record.principal:
        return TargetedToolUseRejectionReason.PRINCIPAL_MISMATCH
    if request.tenant != record.tenant:
        return TargetedToolUseRejectionReason.TENANT_MISMATCH
    if request.agent_name != record.agent_name:
        return TargetedToolUseRejectionReason.AGENT_MISMATCH
    if request.task_id != record.task_id:
        return TargetedToolUseRejectionReason.TASK_MISMATCH
    if request.environment_name != record.environment_name:
        return TargetedToolUseRejectionReason.ENVIRONMENT_MISMATCH
    if request.tool_id != record.tool_id or request.tool_name != record.tool_name:
        return TargetedToolUseRejectionReason.OUT_OF_CEILING
    if request.catalogue_revision != record.catalogue_revision:
        return TargetedToolUseRejectionReason.CATALOGUE_DRIFT
    if request.descriptor_version != record.descriptor_version:
        return TargetedToolUseRejectionReason.DESCRIPTOR_DRIFT
    if request.schema_fingerprint != record.schema_fingerprint:
        return TargetedToolUseRejectionReason.DESCRIPTOR_DRIFT
    return None


def targeted_tool_use_rejection_reason(
    record: TargetedToolGrantRecord,
    request: TargetedToolUseRequest,
    *,
    observed_at: datetime,
) -> TargetedToolUseRejectionReason | None:
    """Validate current execution identity and grant availability."""

    scope_rejection = targeted_tool_use_scope_rejection_reason(record, request)
    if scope_rejection is not None:
        return scope_rejection
    observed_at = _utc_datetime(observed_at, "observed_at")
    if observed_at < record.issued_at:
        return TargetedToolUseRejectionReason.NOT_YET_VALID
    if record.revoked_at is not None:
        return TargetedToolUseRejectionReason.REVOKED
    if observed_at >= record.expires_at:
        return TargetedToolUseRejectionReason.EXPIRED
    if record.used_calls >= record.max_calls:
        return TargetedToolUseRejectionReason.EXHAUSTED
    return None


def targeted_tool_grant_reconstruction_rejection_reason(
    record: TargetedToolGrantRecord,
    *,
    generation_id: str,
    agent_name: str,
    task_id: str | None,
    environment_name: str | None,
    principal: str | None,
    tenant: str | None,
    catalogue_revision: str,
    descriptors_by_id: Mapping[str, tuple[str, str, str]],
    capability_ceiling_names: frozenset[str],
    observed_at: datetime,
    interaction_ended: bool,
) -> TargetedToolUseRejectionReason | None:
    """Validate one recovered record against current registered authority."""

    if type(record) is not TargetedToolGrantRecord:
        raise TypeError("record must be a TargetedToolGrantRecord.")
    observed_at = _utc_datetime(observed_at, "observed_at")
    if record.generation_id != generation_id:
        return TargetedToolUseRejectionReason.CROSS_GENERATION
    if record.agent_name != agent_name:
        return TargetedToolUseRejectionReason.AGENT_MISMATCH
    if record.task_id != task_id:
        return TargetedToolUseRejectionReason.TASK_MISMATCH
    if record.environment_name != environment_name:
        return TargetedToolUseRejectionReason.ENVIRONMENT_MISMATCH
    if record.principal != principal:
        return TargetedToolUseRejectionReason.PRINCIPAL_MISMATCH
    if record.tenant != tenant:
        return TargetedToolUseRejectionReason.TENANT_MISMATCH
    if record.catalogue_revision != catalogue_revision:
        return TargetedToolUseRejectionReason.CATALOGUE_DRIFT
    if record.tool_name not in capability_ceiling_names:
        return TargetedToolUseRejectionReason.OUT_OF_CEILING
    if observed_at < record.issued_at:
        return TargetedToolUseRejectionReason.NOT_YET_VALID
    if record.revoked_at is not None:
        return TargetedToolUseRejectionReason.REVOKED
    if interaction_ended or observed_at >= record.expires_at:
        return TargetedToolUseRejectionReason.EXPIRED
    descriptor = descriptors_by_id.get(record.tool_id)
    if descriptor != (
        record.tool_name,
        record.descriptor_version,
        record.schema_fingerprint,
    ):
        return TargetedToolUseRejectionReason.DESCRIPTOR_DRIFT
    return None


def _targeted_tool_grant_record_at_evidence_count(
    record: TargetedToolGrantRecord,
    event: Event,
    *,
    require_current: bool,
) -> TargetedToolGrantRecord:
    """Rebuild immutable grant authority at a valid event-time call count."""

    used_calls = event.payload.get("used_calls")
    remaining_calls = event.payload.get("remaining_calls")
    if (
        type(used_calls) is not int
        or used_calls < 0
        or used_calls > record.used_calls
        or (require_current and used_calls != record.used_calls)
        or type(remaining_calls) is not int
        or remaining_calls != record.max_calls - used_calls
    ):
        raise ValueError("Targeted tool grant event call counters are inconsistent.")
    return TargetedToolGrantRecord.model_validate(
        record.model_copy(update={"used_calls": used_calls}).model_dump(mode="python")
    )


class TargetedToolUseResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    disposition: TargetedToolUseDisposition
    reason: TargetedToolUseRejectionReason | None = None
    grant: TargetedToolGrantRecord | None = None
    binding: TargetedToolUseBinding | None = None
    event: Event | None = None

    @field_validator("grant", mode="before")
    @classmethod
    def copy_grant(cls, value: object) -> object:
        if isinstance(value, TargetedToolGrantRecord):
            return copy_targeted_tool_grant_record(value)
        return value

    @field_validator("event", mode="before")
    @classmethod
    def copy_event_evidence(cls, value: object) -> object:
        if isinstance(value, Event):
            return copy_event(value)
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> TargetedToolUseResult:
        if self.disposition is TargetedToolUseDisposition.REJECTED:
            if (
                self.reason is None
                or self.binding is not None
                or self.event is None
                or self.event.type != EventType.TARGETED_TOOL_REFERENCE_REJECTED
            ):
                raise ValueError("Rejected targeted tool uses require reason and event evidence.")
            if (
                self.event.payload.get("outcome") != TargetedToolUseDisposition.REJECTED.value
                or self.event.payload.get("rejection_reason") != self.reason.value
                or "tool_ref" in self.event.payload
            ):
                raise ValueError("Rejected targeted tool use evidence is inconsistent.")
            if self.grant is None:
                if (
                    self.reason
                    not in {
                        TargetedToolUseRejectionReason.MALFORMED,
                        TargetedToolUseRejectionReason.UNKNOWN,
                        TargetedToolUseRejectionReason.CROSS_SESSION,
                    }
                    or "grant_id" in self.event.payload
                ):
                    raise ValueError("Unresolved targeted tool use evidence is inconsistent.")
            elif (
                self.event.session_id != self.grant.session_id
                or self.event.interaction_id != self.grant.interaction_id
                or self.event.payload.get("grant_id") != self.grant.grant_id
            ):
                raise ValueError("Rejected targeted tool use conflicts with its grant.")
        elif (
            self.reason is not None
            or self.grant is None
            or self.binding is None
            or self.event is None
            or self.event.type
            != (
                EventType.TARGETED_TOOL_REFERENCE_CONSUMED
                if self.disposition is TargetedToolUseDisposition.BOUND
                else EventType.TARGETED_TOOL_REFERENCE_REJOINED
            )
        ):
            raise ValueError("Accepted targeted tool uses require complete durable evidence.")
        else:
            assert self.grant is not None
            assert self.binding is not None
            assert self.event is not None
            if (
                self.binding.grant_id != self.grant.grant_id
                or self.binding.session_id != self.grant.session_id
                or self.binding.interaction_id != self.grant.interaction_id
            ):
                raise ValueError("Targeted tool use binding conflicts with its grant.")
            evidence_grant = _targeted_tool_grant_record_at_evidence_count(
                self.grant,
                self.event,
                require_current=self.disposition is TargetedToolUseDisposition.BOUND,
            )
            expected = targeted_tool_grant_event(
                evidence_grant,
                event_type=EventType(self.event.type),
                timestamp=self.event.timestamp,
                outcome=self.disposition.value,
                event_id_suffix=(
                    f"use:{self.binding.use_id}"
                    if self.disposition is TargetedToolUseDisposition.BOUND
                    else f"rejoined:{self.binding.use_id}"
                ),
                binding=self.binding,
            )
            if self.event != expected:
                raise ValueError("Accepted targeted tool use evidence is inconsistent.")
        return self


class TargetedToolGrantReconstructionResult(BaseModel):
    """Validated grants remaining callable within one recovered interaction."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    valid: tuple[TargetedToolGrantRecord, ...] = ()
    rejected: tuple[tuple[str, TargetedToolUseRejectionReason], ...] = ()
    events: tuple[Event, ...] = ()

    @field_validator("valid", mode="before")
    @classmethod
    def copy_valid_records(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Valid reconstructed grants must be a sequence.")
        if len(value) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Valid reconstructed grants exceed the bounded count.")
        return tuple(
            copy_targeted_tool_grant_record(record)
            if isinstance(record, TargetedToolGrantRecord)
            else record
            for record in value
        )

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Grant reconstruction evidence must be a sequence.")
        if len(value) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Grant reconstruction evidence exceeds its bounded count.")
        return tuple(copy_event(event) if isinstance(event, Event) else event for event in value)

    @field_validator("rejected", mode="before")
    @classmethod
    def bound_rejections(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Rejected reconstructed grants must be a sequence.")
        if len(value) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Rejected reconstructed grants exceed the bounded count.")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> TargetedToolGrantReconstructionResult:
        if len(self.valid) + len(self.rejected) > TARGETED_TOOL_GRANT_MAX_REQUESTS or len(
            self.events
        ) != len(self.valid) + len(self.rejected):
            raise ValueError("Grant reconstruction records and events must align.")
        grant_ids = [record.grant_id for record in self.valid] + [
            grant_id for grant_id, _reason in self.rejected
        ]
        if len(grant_ids) != len(set(grant_ids)):
            raise ValueError("Grant reconstruction contains duplicate grant identities.")
        valid_by_id = {record.grant_id: record for record in self.valid}
        rejected_by_id = dict(self.rejected)
        event_grant_ids = []
        for event in self.events:
            grant_id = event.payload.get("grant_id")
            if (
                event.type != EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED
                or type(grant_id) is not str
            ):
                raise ValueError("Grant reconstruction contains malformed event evidence.")
            if grant_id in valid_by_id:
                evidence_record = _targeted_tool_grant_record_at_evidence_count(
                    valid_by_id[grant_id],
                    event,
                    require_current=False,
                )
                expected = targeted_tool_grant_event(
                    evidence_record,
                    event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                    timestamp=event.timestamp,
                    outcome="reconstructed",
                    event_id_suffix="reconstructed",
                )
                if event != expected:
                    raise ValueError("Valid grant reconstruction evidence is inconsistent.")
            else:
                reason = rejected_by_id.get(grant_id)
                if (
                    reason is None
                    or event.id != f"{grant_id}:reconstruction-rejected:{reason.value}"
                    or event.payload.get("outcome") != "rejected"
                    or event.payload.get("rejection_reason") != reason.value
                    or "tool_ref" in event.payload
                ):
                    raise ValueError("Rejected grant reconstruction evidence is inconsistent.")
            event_grant_ids.append(grant_id)
        if set(event_grant_ids) != set(grant_ids):
            raise ValueError("Grant reconstruction event identities do not match its records.")
        return self


class TargetedToolGrantStateSnapshot(BaseModel):
    """Complete typed grant state for trusted durable export and validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TARGETED_TOOL_GRANT_SCHEMA_VERSION
    records: tuple[TargetedToolGrantRecord, ...] = ()
    uses: tuple[TargetedToolUseBinding, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TARGETED_TOOL_GRANT_SCHEMA_VERSION:
            raise ValueError("Targeted grant state schema_version must be the integer 1.")
        return value

    @field_validator("records", "uses", mode="before")
    @classmethod
    def require_materialized_sequences(cls, value: object, info) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError(f"Targeted grant state {info.field_name} must be a sequence.")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> TargetedToolGrantStateSnapshot:
        if (
            tuple(sorted(self.records, key=lambda record: (record.issued_at, record.grant_id)))
            != self.records
        ):
            raise ValueError("Targeted grant state records must be canonically ordered.")
        if tuple(sorted(self.uses, key=lambda use: (use.bound_at, use.use_id))) != self.uses:
            raise ValueError("Targeted grant state uses must be canonically ordered.")
        records_by_id = {record.grant_id: record for record in self.records}
        if len(records_by_id) != len(self.records):
            raise ValueError("Targeted grant state contains duplicate grant identities.")
        if len({record.session_id for record in self.records}) > 1:
            raise ValueError("Targeted grant state must belong to one session.")
        interaction_counts: dict[tuple[str, str], int] = {}
        for record in self.records:
            interaction_key = (record.session_id, record.interaction_id)
            interaction_counts[interaction_key] = interaction_counts.get(interaction_key, 0) + 1
        if any(count > TARGETED_TOOL_GRANT_MAX_REQUESTS for count in interaction_counts.values()):
            raise ValueError("Targeted grant interaction exceeds its bounded count.")
        if len({use.use_id for use in self.uses}) != len(self.uses):
            raise ValueError("Targeted grant state contains duplicate use identities.")
        if len(
            {(use.session_id, use.interaction_id, use.invocation_id) for use in self.uses}
        ) != len(self.uses):
            raise ValueError("Targeted grant state contains duplicate invocation identities.")
        if len(
            {(use.session_id, use.interaction_id, use.outer_tool_call_id) for use in self.uses}
        ) != len(self.uses):
            raise ValueError("Targeted grant state contains duplicate outer-call identities.")
        use_counts = {grant_id: 0 for grant_id in records_by_id}
        for use in self.uses:
            record = records_by_id.get(use.grant_id)
            if record is None:
                raise ValueError("Targeted grant use has no owning grant record.")
            if use.session_id != record.session_id or use.interaction_id != record.interaction_id:
                raise ValueError("Targeted grant use conflicts with its owning scope.")
            if not record.issued_at <= use.bound_at < record.expires_at:
                raise ValueError("Targeted grant use falls outside its grant lifetime.")
            if record.revoked_at is not None and use.bound_at > record.revoked_at:
                raise ValueError("Targeted grant use follows its revocation timestamp.")
            use_counts[record.grant_id] += 1
        for record in self.records:
            validate_targeted_tool_grant_record_identity(record)
        if any(record.used_calls != use_counts[record.grant_id] for record in self.records):
            raise ValueError("Targeted grant call counters conflict with durable uses.")
        return self


def copy_targeted_tool_grant_state_snapshot(
    value: TargetedToolGrantStateSnapshot,
) -> TargetedToolGrantStateSnapshot:
    if type(value) is not TargetedToolGrantStateSnapshot:
        raise TypeError("value must be a TargetedToolGrantStateSnapshot.")
    return TargetedToolGrantStateSnapshot.model_validate(value.model_dump(mode="python"))


def targeted_tool_grant_event(
    record: TargetedToolGrantRecord,
    *,
    event_type: EventType,
    timestamp: datetime,
    outcome: str,
    event_id_suffix: str,
    rejection_reason: TargetedToolUseRejectionReason | None = None,
    binding: TargetedToolUseBinding | None = None,
) -> Event:
    """Build bounded privacy-safe lifecycle evidence without exposing a tool ref."""

    record = copy_targeted_tool_grant_record(record)
    timestamp = _utc_datetime(timestamp, "timestamp")
    payload: dict[str, object] = {
        "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
        "grant_id": record.grant_id,
        "request_id": record.request_id,
        "tool_id": record.tool_id,
        "catalogue_revision": record.catalogue_revision,
        "descriptor_version": record.descriptor_version,
        "schema_fingerprint": record.schema_fingerprint,
        "generation_id": record.generation_id,
        "outcome": require_durable_clean_nonblank(outcome, "outcome"),
        "max_calls": record.max_calls,
        "used_calls": record.used_calls,
        "remaining_calls": record.remaining_calls,
        "issued_at": record.issued_at.isoformat(),
        "expires_at": record.expires_at.isoformat(),
    }
    if record.origin is not None:
        payload["origin"] = record.origin
    if rejection_reason is not None:
        payload["rejection_reason"] = rejection_reason.value
    if binding is not None:
        payload.update(
            {
                "use_id": binding.use_id,
                "model_step_id": binding.model_step_id,
                "outer_tool_call_id": binding.outer_tool_call_id,
                "arguments_sha256": binding.arguments_sha256,
                "invocation_id": binding.invocation_id,
            }
        )
    event = Event(
        id=f"{record.grant_id}:{event_id_suffix}",
        type=event_type,
        session_id=record.session_id,
        interaction_id=record.interaction_id,
        timestamp=timestamp,
        agent_name=record.agent_name,
        environment_name=record.environment_name,
        tool_name=record.tool_name,
        payload=payload,
    )
    public_authority_fields = [
        "grant_id",
        "request_id",
        "tool_id",
        "catalogue_revision",
        "descriptor_version",
        "schema_fingerprint",
        "generation_id",
    ]
    if binding is not None:
        public_authority_fields.extend(
            [
                "use_id",
                "arguments_sha256",
            ]
        )
    return event_with_runtime_payload_authority(event, *public_authority_fields)


def validate_targeted_tool_grant_issuance_evidence(
    record: TargetedToolGrantRecord,
    event: Event,
) -> None:
    """Reject issuance input that is not exact runtime-authored record evidence."""

    record = copy_targeted_tool_grant_record(record)
    event = copy_event(event)
    if event.timestamp != record.issued_at:
        raise ValueError("Targeted grant issuance time conflicts with its durable record.")
    if (
        event.payload.get("used_calls") != 0
        or event.payload.get("remaining_calls") != record.max_calls
    ):
        raise ValueError("Targeted grant issuance must record an unused call budget.")
    validate_targeted_tool_grant_lifecycle_event(
        record,
        event,
        event_type=EventType.TARGETED_TOOL_GRANT_ISSUED,
        outcome=TargetedToolGrantIssueOutcome.ISSUED.value,
        event_id_suffix=TargetedToolGrantIssueOutcome.ISSUED.value,
        require_current_call_count=False,
    )


def validate_targeted_tool_grant_lifecycle_event(
    record: TargetedToolGrantRecord,
    event: Event,
    *,
    event_type: EventType,
    outcome: str,
    event_id_suffix: str,
    rejection_reason: TargetedToolUseRejectionReason | None = None,
    binding: TargetedToolUseBinding | None = None,
    require_current_call_count: bool,
) -> None:
    """Validate exact lifecycle evidence, including a historical call-count snapshot."""

    record = copy_targeted_tool_grant_record(record)
    event = copy_event(event)
    evidence_record = _targeted_tool_grant_record_at_evidence_count(
        record,
        event,
        require_current=require_current_call_count,
    )
    expected = targeted_tool_grant_event(
        evidence_record,
        event_type=event_type,
        timestamp=event.timestamp,
        outcome=outcome,
        event_id_suffix=event_id_suffix,
        rejection_reason=rejection_reason,
        binding=binding,
    )
    if event != expected:
        raise ValueError("Targeted grant lifecycle evidence conflicts with durable authority.")


def validate_targeted_tool_grant_revocation_evidence(
    record: TargetedToolGrantRecord,
    event: Event,
) -> None:
    """Validate the one durable event bound to a revoked grant record."""

    record = copy_targeted_tool_grant_record(record)
    if record.revoked_at is None or record.revocation_reason is None:
        raise ValueError("Targeted grant revocation evidence requires a revoked record.")
    event = copy_event(event)
    if event.timestamp != record.revoked_at:
        raise ValueError("Targeted grant revocation time conflicts with durable authority.")
    validate_targeted_tool_grant_lifecycle_event(
        record,
        event,
        event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
        outcome="revoked",
        event_id_suffix="revoked",
        require_current_call_count=True,
    )


def targeted_tool_use_rejection_event(
    record: TargetedToolGrantRecord,
    request: TargetedToolUseRequest,
    *,
    reason: TargetedToolUseRejectionReason,
    timestamp: datetime,
) -> Event:
    """Build deterministic digest-only evidence for one rejected admission."""

    record = copy_targeted_tool_grant_record(record)
    if type(request) is not TargetedToolUseRequest:
        raise TypeError("request must be a TargetedToolUseRequest.")
    if type(reason) is not TargetedToolUseRejectionReason:
        raise TypeError("reason must be a TargetedToolUseRejectionReason.")
    rejection_id = _sha256_identity(
        {
            "record_type": "cayu.targeted-tool-use-rejection",
            "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
            "grant_id": record.grant_id,
            "interaction_id": request.interaction_id,
            "model_step_id": request.model_step_id,
            "outer_tool_call_id": request.outer_tool_call_id,
            "arguments_sha256": request.arguments_sha256,
            "invocation_id": request.invocation_id,
            "reason": reason.value,
        },
        "targeted_tool_use_rejection",
    )
    event = targeted_tool_grant_event(
        record,
        event_type=EventType.TARGETED_TOOL_REFERENCE_REJECTED,
        timestamp=timestamp,
        outcome=TargetedToolUseDisposition.REJECTED.value,
        event_id_suffix=f"rejected:{rejection_id.removeprefix('sha256:')}",
        rejection_reason=reason,
    )
    return event_with_runtime_payload_authority(
        event.model_copy(
            update={
                "payload": {
                    **event.payload,
                    "rejection_id": rejection_id,
                    "model_step_id": request.model_step_id,
                    "outer_tool_call_id": request.outer_tool_call_id,
                    "arguments_sha256": request.arguments_sha256,
                    "invocation_id": request.invocation_id,
                }
            }
        ),
        "arguments_sha256",
        "rejection_id",
    )


def validate_targeted_tool_use_rejection_evidence(
    record: TargetedToolGrantRecord,
    request: TargetedToolUseRequest,
    *,
    reason: TargetedToolUseRejectionReason,
    event: Event,
) -> None:
    """Validate exact persisted evidence for a resolved rejected use."""

    event = copy_event(event)
    expected = targeted_tool_use_rejection_event(
        record,
        request,
        reason=reason,
        timestamp=event.timestamp,
    )
    if event != expected:
        raise ValueError("Targeted tool rejection evidence conflicts with durable authority.")


def targeted_tool_unresolved_rejection_event(
    request: TargetedToolUseRequest,
    *,
    reason: TargetedToolUseRejectionReason,
    timestamp: datetime,
    agent_name: str,
    environment_name: str | None,
) -> Event:
    """Build rejection evidence when no authenticated grant can be disclosed."""

    if type(request) is not TargetedToolUseRequest:
        raise TypeError("request must be a TargetedToolUseRequest.")
    if reason not in {
        TargetedToolUseRejectionReason.MALFORMED,
        TargetedToolUseRejectionReason.UNKNOWN,
        TargetedToolUseRejectionReason.CROSS_SESSION,
    }:
        raise ValueError("Unresolved rejection evidence requires an unresolved reason.")
    timestamp = _utc_datetime(timestamp, "timestamp")
    rejection_id = _sha256_identity(
        {
            "record_type": "cayu.targeted-tool-unresolved-rejection",
            "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
            "session_id": request.session_id,
            "interaction_id": request.interaction_id,
            "model_step_id": request.model_step_id,
            "outer_tool_call_id": request.outer_tool_call_id,
            "arguments_sha256": request.arguments_sha256,
            "invocation_id": request.invocation_id,
            "reason": reason.value,
        },
        "targeted_tool_unresolved_rejection",
    )
    event = Event(
        id=f"{rejection_id}:targeted-tool-rejected",
        type=EventType.TARGETED_TOOL_REFERENCE_REJECTED,
        session_id=request.session_id,
        interaction_id=request.interaction_id,
        timestamp=timestamp,
        agent_name=require_durable_clean_nonblank(agent_name, "agent_name"),
        environment_name=environment_name,
        payload={
            "schema_version": TARGETED_TOOL_GRANT_SCHEMA_VERSION,
            "outcome": TargetedToolUseDisposition.REJECTED.value,
            "rejection_reason": reason.value,
            "rejection_id": rejection_id,
            "model_step_id": request.model_step_id,
            "outer_tool_call_id": request.outer_tool_call_id,
            "arguments_sha256": request.arguments_sha256,
            "invocation_id": request.invocation_id,
        },
    )
    return event_with_runtime_payload_authority(
        event,
        "arguments_sha256",
        "rejection_id",
    )


def validate_targeted_tool_unresolved_rejection_evidence(
    request: TargetedToolUseRequest,
    *,
    reason: TargetedToolUseRejectionReason,
    event: Event,
    agent_name: str,
    environment_name: str | None,
) -> None:
    """Validate exact persisted evidence for an unauthenticated reference."""

    event = copy_event(event)
    expected = targeted_tool_unresolved_rejection_event(
        request,
        reason=reason,
        timestamp=event.timestamp,
        agent_name=agent_name,
        environment_name=environment_name,
    )
    if event != expected:
        raise ValueError("Unresolved targeted tool rejection evidence is inconsistent.")


def targeted_tool_descriptor_matches_record(
    descriptor: ToolDescriptor,
    record: TargetedToolGrantRecord,
) -> bool:
    """Return whether current registered identity exactly reconstructs a grant."""

    if type(descriptor) is not ToolDescriptor or type(record) is not TargetedToolGrantRecord:
        return False
    return (
        descriptor.tool_id == record.tool_id
        and descriptor.name == record.tool_name
        and descriptor.version == record.descriptor_version
        and descriptor.schema_fingerprint == record.schema_fingerprint
    )


__all__ = [
    "TARGETED_TOOL_GRANT_DEFAULT_LIFETIME_SECONDS",
    "TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS",
    "TARGETED_TOOL_GRANT_MAX_CALLS",
    "TARGETED_TOOL_GRANT_MAX_LIFETIME_SECONDS",
    "TARGETED_TOOL_GRANT_MAX_REQUESTS",
    "TARGETED_TOOL_GRANT_SCHEMA_VERSION",
    "TARGETED_TOOL_REFERENCE_FIELD_NAME",
    "PreparedTargetedToolGrant",
    "TargetedToolGrant",
    "TargetedToolGrantInspection",
    "TargetedToolGrantIssueOutcome",
    "TargetedToolGrantIssueResult",
    "TargetedToolGrantReconstructionResult",
    "TargetedToolGrantRecord",
    "TargetedToolGrantStateSnapshot",
    "TargetedToolUseBinding",
    "TargetedToolUseDisposition",
    "TargetedToolUseRejectionReason",
    "TargetedToolUseRequest",
    "TargetedToolUseResult",
    "build_targeted_tool_grant_record",
    "copy_targeted_tool_grant",
    "copy_targeted_tool_grant_record",
    "copy_targeted_tool_grant_state_snapshot",
    "persisted_targeted_tool_grant_batch_fingerprint",
    "prepare_targeted_tool_grants",
    "prepared_targeted_tool_grant_batch_fingerprint",
    "targeted_tool_descriptor_matches_record",
    "targeted_tool_grant_event",
    "targeted_tool_grant_inspection",
    "targeted_tool_grant_reconstruction_rejection_reason",
    "targeted_tool_grant_with_active_reference",
    "targeted_tool_unresolved_rejection_event",
    "targeted_tool_use_binding",
    "targeted_tool_use_rejection_event",
    "targeted_tool_use_rejection_reason",
    "targeted_tool_use_scope_rejection_reason",
    "targeted_tool_view_generation_id",
    "validate_targeted_tool_grant_batch_evidence",
    "validate_targeted_tool_grant_issuance_evidence",
    "validate_targeted_tool_grant_lifecycle_event",
    "validate_targeted_tool_grant_record_identity",
    "validate_targeted_tool_grant_reference",
    "validate_targeted_tool_grant_revocation_evidence",
    "validate_targeted_tool_grants",
    "validate_targeted_tool_unresolved_rejection_evidence",
    "validate_targeted_tool_use_rejection_evidence",
]
