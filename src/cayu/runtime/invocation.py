"""Bounded durable invocation provenance for session trees."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank

INVOCATION_PROVENANCE_SCHEMA_VERSION = 1
INVOCATION_IDENTITY_MAX_CHARS = 512
_CANONICAL_UUID4_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


class InvocationOriginTrust(StrEnum):
    """How Cayu obtained the root invocation identity."""

    SERVER_VERIFIED = "server_verified"
    HOST_ASSERTED = "host_asserted"
    UNATTRIBUTED = "unattributed"


class SessionExecutionSource(StrEnum):
    """The runtime boundary that created one session."""

    HTTP_RUN = "http_run"
    SDK_RUN = "sdk_run"
    FORK = "fork"
    SUBAGENT = "subagent"
    WORKFLOW_STEP = "workflow_step"


class InvocationOriginClaim(BaseModel):
    """Bounded identity asserted by a trusted direct-SDK host application."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    subject: str = Field(max_length=INVOCATION_IDENTITY_MAX_CHARS)
    tenant: str | None = Field(default=None, max_length=INVOCATION_IDENTITY_MAX_CHARS)

    @field_validator("subject", "tenant")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)


class InvocationOrigin(BaseModel):
    """Immutable root identity retained across a complete session tree."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    trust: InvocationOriginTrust
    subject: str | None = Field(default=None, max_length=INVOCATION_IDENTITY_MAX_CHARS)
    tenant: str | None = Field(default=None, max_length=INVOCATION_IDENTITY_MAX_CHARS)

    @field_validator("subject", "tenant")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_trust_shape(self) -> InvocationOrigin:
        if self.trust is InvocationOriginTrust.UNATTRIBUTED:
            if self.subject is not None or self.tenant is not None:
                raise ValueError("Unattributed invocation origins cannot carry identity fields.")
        elif self.subject is None:
            raise ValueError("Attributed invocation origins require subject.")
        return self


class SessionInvocation(BaseModel):
    """Versioned provenance attached atomically to one durable session."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = INVOCATION_PROVENANCE_SCHEMA_VERSION
    origin: InvocationOrigin
    root_invocation_id: str = Field(
        min_length=36,
        max_length=36,
        pattern=_CANONICAL_UUID4_PATTERN,
        json_schema_extra={"format": "uuid4"},
    )
    root_session_id: str
    source: SessionExecutionSource

    @field_validator("root_invocation_id")
    @classmethod
    def validate_root_invocation_id(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "root_invocation_id")
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("root_invocation_id must be a canonical UUIDv4 string.") from exc
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("root_invocation_id must be a canonical UUIDv4 string.")
        return value

    @field_validator("root_session_id")
    @classmethod
    def validate_root_session_id(cls, value: str) -> str:
        # Like Session.id, this is also a read-side value-object field. New run
        # creation enforces MAX_SESSION_ID_BYTES before this value is minted, but
        # external stores must remain inspectable if they return an older or
        # otherwise oversized identifier.
        return require_durable_clean_nonblank(value, "root_session_id")


def copy_invocation_origin_claim(
    value: InvocationOriginClaim | None,
) -> InvocationOriginClaim | None:
    if value is None:
        return None
    if type(value) is not InvocationOriginClaim:
        raise TypeError("Invocation origin claims must be InvocationOriginClaim instances.")
    return InvocationOriginClaim(subject=value.subject, tenant=value.tenant)


def copy_invocation_origin(value: InvocationOrigin) -> InvocationOrigin:
    if type(value) is not InvocationOrigin:
        raise TypeError("Invocation origins must be InvocationOrigin instances.")
    return InvocationOrigin(trust=value.trust, subject=value.subject, tenant=value.tenant)


def copy_session_invocation(value: SessionInvocation) -> SessionInvocation:
    if type(value) is not SessionInvocation:
        raise TypeError("Session invocation provenance must be a SessionInvocation instance.")
    return SessionInvocation(
        schema_version=value.schema_version,
        origin=copy_invocation_origin(value.origin),
        root_invocation_id=value.root_invocation_id,
        root_session_id=value.root_session_id,
        source=value.source,
    )


def inherited_session_invocation(
    parent: SessionInvocation,
    *,
    source: SessionExecutionSource,
) -> SessionInvocation:
    """Derive the exact immutable provenance for one child session."""

    if type(parent) is not SessionInvocation:
        raise TypeError("Parent invocation provenance must be a SessionInvocation.")
    if type(source) is not SessionExecutionSource:
        raise TypeError("source must be a SessionExecutionSource.")
    if source is SessionExecutionSource.HTTP_RUN:
        raise ValueError("Derived sessions cannot claim a root HTTP run source.")
    return SessionInvocation(
        origin=copy_invocation_origin(parent.origin),
        root_invocation_id=parent.root_invocation_id,
        root_session_id=parent.root_session_id,
        source=source,
    )
