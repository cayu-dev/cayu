"""Session creation claims and exact durable authentication."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.sessions.authority import _require_raw_sha256_digest
from cayu.sessions.records import SessionStatus

SESSION_CREATE_CLAIM_METADATA_KEY = "cayu:session_create_claim"
SESSION_CREATE_CLAIM_RECORD_TYPE = "cayu.session-create-claim"
SESSION_CREATE_CLAIM_SCHEMA_VERSION = 1
RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_RECORD_TYPE = "cayu.runtime-session-create-claim-reference"
RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_SCHEMA_VERSION = 1
RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_KEY_ID_CHARS = 256
RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_OPERATION_ID_CHARS = 256


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeSessionCreateClaimReferenceKey:
    """Caller-scoped secret used only to blind request-authority identities."""

    key_id: str
    secret: bytes

    def __post_init__(self) -> None:
        key_id = require_clean_nonblank(self.key_id, "key_id")
        if len(key_id) > RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_KEY_ID_CHARS:
            raise ValueError("Runtime session create reference key_id exceeds its character bound.")
        if type(self.secret) is not bytes or len(self.secret) < 32:
            raise ValueError("Runtime session create reference keys require at least 32 bytes.")
        object.__setattr__(self, "key_id", key_id)

    def __repr__(self) -> str:
        return f"RuntimeSessionCreateClaimReferenceKey(key_id={self.key_id!r})"


class RuntimeSessionCreateClaimReference(BaseModel):
    """Secret-free keyed identity for reconstructing one private create claim."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )

    record_type: Literal["cayu.runtime-session-create-claim-reference"] = (
        RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_RECORD_TYPE
    )
    schema_version: Literal[1] = RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_SCHEMA_VERSION
    session_id: StrictStr = Field(max_length=512)
    operation_id: StrictStr = Field(
        max_length=RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_OPERATION_ID_CHARS
    )
    request_authority_key_id: StrictStr = Field(
        max_length=RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_KEY_ID_CHARS
    )
    request_authority_hmac_sha256: StrictStr = Field(min_length=64, max_length=64)
    claim_id: StrictStr = Field(min_length=64, max_length=64)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be a JSON integer.")
        return value

    @field_validator("session_id", "operation_id", "request_authority_key_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("request_authority_hmac_sha256", "claim_id")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        _require_raw_sha256_digest(value)
        return value


class RuntimeSessionCreateClaimAuthenticationDisposition(StrEnum):
    MISSING_SESSION = "missing_session"
    MATCHING_SESSION = "matching_session"
    FOREIGN_SESSION = "foreign_session"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    MALFORMED_EVIDENCE = "malformed_evidence"
    TAMPERED_EVIDENCE = "tampered_evidence"
    IDENTITY_CONFLICT = "identity_conflict"


class RuntimeSessionCreateClaimAuthentication(BaseModel):
    """Content-free classification of existing-session create authority."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    disposition: RuntimeSessionCreateClaimAuthenticationDisposition
    session_status: SessionStatus | None = None
    transient_input_authenticated: StrictBool = False

    @field_validator("transient_input_authenticated", mode="before")
    @classmethod
    def validate_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("transient_input_authenticated must be a JSON boolean.")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        missing = (
            self.disposition is RuntimeSessionCreateClaimAuthenticationDisposition.MISSING_SESSION
        )
        if missing != (self.session_status is None):
            raise ValueError("Only a missing-session result can omit session status.")
        if (
            self.transient_input_authenticated
            and self.disposition
            is not RuntimeSessionCreateClaimAuthenticationDisposition.MATCHING_SESSION
        ):
            raise ValueError("Only a matching session can authenticate transient input.")
        return self

    @property
    def matches(self) -> bool:
        return (
            self.disposition is RuntimeSessionCreateClaimAuthenticationDisposition.MATCHING_SESSION
        )
