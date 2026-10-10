"""Session creation claims and exact durable authentication."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
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

from cayu._validation import canonical_durable_json_bytes, copy_durable_metadata
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import requests as session_request_contracts
from cayu.sessions import summaries as session_summary_rules
from cayu.sessions.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
    inherited_session_invocation,
    session_invocation_from_task,
)
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import (
    RunRequest,
    _run_request_invocation_lifecycle_authority_sha256,
    copy_run_request,
    run_request_authority_is_runtime_generated,
)
from cayu.sessions.transcript_input import DeferredInteractionInput, session_input_messages_sha256

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
        session_authority_rules._require_raw_sha256_digest(value)
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


def _runtime_session_create_reference_operation_id(value: str) -> str:
    operation_id = require_clean_nonblank(value, "operation_id")
    if len(operation_id) > RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_OPERATION_ID_CHARS:
        raise ValueError("Runtime session create operation_id exceeds its bound.")
    return operation_id


def _runtime_session_create_reference_request_hmac_sha256(
    request: RunRequest,
    *,
    operation_id: str,
    key: RuntimeSessionCreateClaimReferenceKey,
) -> str:
    """Blind exact request and operation authority with a stable key."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime session create reference requires a RunRequest.")
    if type(key) is not RuntimeSessionCreateClaimReferenceKey:
        raise TypeError("key must be an exact RuntimeSessionCreateClaimReferenceKey.")
    operation_id = _runtime_session_create_reference_operation_id(operation_id)
    copied = copy_run_request(request)
    if SESSION_CREATE_CLAIM_METADATA_KEY in copied.metadata:
        raise ValueError("Session create claim metadata is runtime-owned.")
    claim = copied._runtime_session_create_claim
    if claim is not None and (
        type(claim) is not session_request_contracts._RuntimeSessionCreateClaim
        or claim.token is not session_request_contracts._RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        or claim.session_id != copied.session_id
    ):
        raise ValueError("Runtime session create authority is invalid.")
    copied._runtime_session_create_claim = None
    material = canonical_durable_json_bytes(
        {
            "request_authority_key_id": key.key_id,
            "operation_id": operation_id,
            "request": copied.model_dump(mode="json", warnings=False),
            "lifecycle_authority_sha256": (
                _run_request_invocation_lifecycle_authority_sha256(copied)
            ),
        },
        "runtime session create reference request",
    )
    return hmac.new(
        key.secret,
        b"cayu.runtime-session-create-reference.request-authority.v1\0" + material,
        sha256,
    ).hexdigest()


def _runtime_session_create_reference_claim_id(
    *,
    session_id: str,
    operation_id: str,
) -> str:
    """Derive a stable claim across key rotation and fresh-process reconstruction."""

    return sha256(
        canonical_durable_json_bytes(
            {
                "record_type": RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_RECORD_TYPE,
                "schema_version": RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_SCHEMA_VERSION,
                "session_id": session_id,
                "operation_id": operation_id,
            },
            "runtime session create reference identity",
        )
    ).hexdigest()


def runtime_session_create_claim_reference(
    request: RunRequest,
    *,
    operation_id: str,
    key: RuntimeSessionCreateClaimReferenceKey,
) -> RuntimeSessionCreateClaimReference:
    """Create one bounded keyed reference before session admission."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime session create reference requires a RunRequest.")
    if type(key) is not RuntimeSessionCreateClaimReferenceKey:
        raise TypeError("key must be an exact RuntimeSessionCreateClaimReferenceKey.")
    session_id = request.session_id
    if type(session_id) is not str or not run_request_authority_is_runtime_generated(
        request,
        field_name="session_id",
        value=session_id,
    ):
        raise ValueError("Runtime session create reference requires a generated session_id.")
    operation_id = _runtime_session_create_reference_operation_id(operation_id)
    if request._runtime_session_create_claim is not None:
        raise ValueError("Runtime session create reference must precede claim attachment.")
    request_hmac = _runtime_session_create_reference_request_hmac_sha256(
        request,
        operation_id=operation_id,
        key=key,
    )
    return RuntimeSessionCreateClaimReference(
        session_id=session_id,
        operation_id=operation_id,
        request_authority_key_id=key.key_id,
        request_authority_hmac_sha256=request_hmac,
        claim_id=_runtime_session_create_reference_claim_id(
            session_id=session_id,
            operation_id=operation_id,
        ),
    )


def run_request_with_runtime_session_create_claim_reference(
    request: RunRequest,
    reference: RuntimeSessionCreateClaimReference,
    *,
    operation_id: str,
    key: RuntimeSessionCreateClaimReferenceKey,
) -> tuple[RunRequest, object]:
    """Reconstruct private create authority from an exact durable reference."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime session create reference requires a RunRequest.")
    if type(reference) is not RuntimeSessionCreateClaimReference:
        raise TypeError("reference must be an exact RuntimeSessionCreateClaimReference.")
    if type(key) is not RuntimeSessionCreateClaimReferenceKey:
        raise TypeError("key must be an exact RuntimeSessionCreateClaimReferenceKey.")
    operation_id = _runtime_session_create_reference_operation_id(operation_id)
    request_hmac = _runtime_session_create_reference_request_hmac_sha256(
        request,
        operation_id=reference.operation_id,
        key=key,
    )
    expected_claim_id = _runtime_session_create_reference_claim_id(
        session_id=reference.session_id,
        operation_id=reference.operation_id,
    )
    if (
        request.session_id != reference.session_id
        or operation_id != reference.operation_id
        or key.key_id != reference.request_authority_key_id
        or not hmac.compare_digest(
            request_hmac,
            reference.request_authority_hmac_sha256,
        )
        or not hmac.compare_digest(reference.claim_id, expected_claim_id)
    ):
        raise ValueError("Runtime session create reference conflicts with request authority.")
    if request._runtime_session_create_claim is not None:
        raise ValueError("Runtime session create authority is already attached.")
    return run_request_with_runtime_session_create_claim(
        request,
        claim_id=reference.claim_id,
    )


def run_request_with_runtime_session_create_claim(
    request: RunRequest,
    *,
    claim_id: str,
) -> tuple[RunRequest, object]:
    """Attach authenticated authority for exact generated-session readback."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime session create authority requires a RunRequest.")
    session_id = request.session_id
    if type(session_id) is not str or not run_request_authority_is_runtime_generated(
        request,
        field_name="session_id",
        value=session_id,
    ):
        raise ValueError("Runtime session create authority requires a generated session_id.")
    if SESSION_CREATE_CLAIM_METADATA_KEY in request.metadata:
        raise ValueError("Session create claim metadata is runtime-owned.")
    copied = copy_run_request(request)
    claim = session_request_contracts._RuntimeSessionCreateClaim(
        session_id=session_id, claim_id=claim_id
    )
    copied._runtime_session_create_claim = claim
    return copied, claim


def _session_create_claim_record(
    *,
    claim_id: str,
    request_sha256: str,
    interaction_id: str | None = None,
    messages_sha256: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": SESSION_CREATE_CLAIM_RECORD_TYPE,
        "schema_version": SESSION_CREATE_CLAIM_SCHEMA_VERSION,
        "claim_id": claim_id,
        "request_sha256": request_sha256,
    }
    if interaction_id is not None or messages_sha256 is not None:
        if interaction_id is None or messages_sha256 is None:
            raise ValueError("Final session create claims require complete interaction evidence.")
        record["interaction_id"] = interaction_id
        record["messages_sha256"] = messages_sha256
    return record


def strip_runtime_session_create_claim_before_redaction(request: RunRequest) -> RunRequest:
    """Remove authenticated claim metadata before generic value redaction."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime session create claim preparation requires a RunRequest.")
    record = request.metadata.get(SESSION_CREATE_CLAIM_METADATA_KEY)
    if record is None:
        return request
    claim = request._runtime_session_create_claim
    if (
        type(claim) is not session_request_contracts._RuntimeSessionCreateClaim
        or claim.token is not session_request_contracts._RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        or request.session_id != claim.session_id
        or type(record) is not dict
        or record.get("record_type") != SESSION_CREATE_CLAIM_RECORD_TYPE
        or record.get("schema_version") != SESSION_CREATE_CLAIM_SCHEMA_VERSION
        or record.get("claim_id") != claim.claim_id
    ):
        raise ValueError("Session create claim metadata is runtime-owned.")
    metadata = copy_durable_metadata(request.metadata)
    metadata.pop(SESSION_CREATE_CLAIM_METADATA_KEY)
    prepared = request.model_copy(update={"metadata": metadata})
    prepared._runtime_session_create_claim = claim
    return prepared


def apply_runtime_session_create_claim(request: RunRequest) -> RunRequest:
    """Materialize an authenticated create claim after public metadata preparation."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime session create claim requires a RunRequest.")
    claim = request._runtime_session_create_claim
    if (
        type(claim) is not session_request_contracts._RuntimeSessionCreateClaim
        or claim.token is not session_request_contracts._RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        or request.session_id != claim.session_id
    ):
        if SESSION_CREATE_CLAIM_METADATA_KEY in request.metadata:
            raise ValueError("Session create claim metadata is runtime-owned.")
        copied = copy_run_request(request)
        copied._runtime_session_create_claim = None
        return copied
    copied = copy_run_request(request)
    metadata = copy_durable_metadata(copied.metadata)
    previous_record = metadata.pop(SESSION_CREATE_CLAIM_METADATA_KEY, None)
    request_without_claim = copied.model_copy(update={"metadata": metadata})
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            request_without_claim.model_dump(mode="json", warnings=False),
            "runtime session create request",
        )
    ).hexdigest()
    if previous_record is not None and (
        type(previous_record) is not dict
        or previous_record.get("record_type") != SESSION_CREATE_CLAIM_RECORD_TYPE
        or previous_record.get("schema_version") != SESSION_CREATE_CLAIM_SCHEMA_VERSION
        or previous_record.get("claim_id") != claim.claim_id
        or previous_record.get("request_sha256") != request_sha256
    ):
        raise ValueError("Session create claim metadata is runtime-owned.")
    claim.request_sha256 = request_sha256
    claim.messages_sha256 = session_input_messages_sha256(request_without_claim.messages)
    metadata[SESSION_CREATE_CLAIM_METADATA_KEY] = _session_create_claim_record(
        claim_id=claim.claim_id,
        request_sha256=claim.request_sha256,
    )
    prepared = request_without_claim.model_copy(update={"metadata": metadata})
    prepared._runtime_session_create_claim = claim
    return prepared


def bind_runtime_session_create_claim(
    request: RunRequest,
    *,
    identity: SessionIdentity,
    interaction_started_event: Event,
) -> None:
    """Bind a claim to the exact prepared request crossing the store boundary."""

    if (
        type(request) is not RunRequest
        or type(identity) is not SessionIdentity
        or type(interaction_started_event) is not Event
    ):
        raise TypeError("Runtime session create claim binding received invalid input.")
    claim = request._runtime_session_create_claim
    if claim is None:
        return
    if (
        type(claim) is not session_request_contracts._RuntimeSessionCreateClaim
        or claim.token is not session_request_contracts._RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        or request.session_id != claim.session_id
        or claim.request_sha256 is None
        or request.metadata.get(SESSION_CREATE_CLAIM_METADATA_KEY)
        != _session_create_claim_record(
            claim_id=claim.claim_id,
            request_sha256=claim.request_sha256,
        )
    ):
        request._runtime_session_create_claim = None
        return
    claim.expected_session_material = session_request_contracts._SessionCreateMaterial.from_request(
        request,
        identity=identity,
        session_id=claim.session_id,
    )
    if (
        interaction_started_event.session_id != claim.session_id
        or interaction_started_event.interaction_id is None
    ):
        request._runtime_session_create_claim = None
        claim.expected_session_material = None
        return
    claim.messages_sha256 = session_input_messages_sha256(request.messages)
    claim.interaction_id = interaction_started_event.interaction_id
    request.metadata[SESSION_CREATE_CLAIM_METADATA_KEY] = _session_create_claim_record(
        claim_id=claim.claim_id,
        request_sha256=claim.request_sha256,
        interaction_id=interaction_started_event.interaction_id,
        messages_sha256=claim.messages_sha256,
    )


def session_has_runtime_create_claim(session: Session | None, claim: object) -> bool:
    """Return whether a session carries one authenticated logical create claim."""

    if (
        type(session) is not Session
        or type(claim) is not session_request_contracts._RuntimeSessionCreateClaim
        or claim.token is not session_request_contracts._RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        or session.id != claim.session_id
        or claim.request_sha256 is None
    ):
        return False
    record = session.metadata.get(SESSION_CREATE_CLAIM_METADATA_KEY)
    return (
        type(record) is dict
        and set(record)
        == {
            "record_type",
            "schema_version",
            "claim_id",
            "request_sha256",
            "interaction_id",
            "messages_sha256",
        }
        and record.get("record_type") == SESSION_CREATE_CLAIM_RECORD_TYPE
        and record.get("schema_version") == SESSION_CREATE_CLAIM_SCHEMA_VERSION
        and record.get("claim_id") == claim.claim_id
        and record.get("request_sha256") == claim.request_sha256
        and type(record.get("interaction_id")) is str
        and type(record.get("messages_sha256")) is str
    )


def _runtime_session_create_authentication(
    disposition: RuntimeSessionCreateClaimAuthenticationDisposition,
    *,
    session: Session | None,
    transient_input_authenticated: bool = False,
) -> RuntimeSessionCreateClaimAuthentication:
    return RuntimeSessionCreateClaimAuthentication(
        disposition=disposition,
        session_status=None if session is None else session.status,
        transient_input_authenticated=transient_input_authenticated,
    )


def authenticate_runtime_session_create_claim_reference(
    session: Session | None,
    deferred_input: DeferredInteractionInput | None,
    claim: object,
    reference: RuntimeSessionCreateClaimReference,
    *,
    request: RunRequest,
    operation_id: str,
    parent_session: Session | None,
    key: RuntimeSessionCreateClaimReferenceKey,
) -> RuntimeSessionCreateClaimAuthentication:
    """Authenticate one reconstructed claim, including after terminal cleanup.

    For a live session, the final claim record, exact invocation/session
    material, and transient source-input digest must all agree.  A terminal
    session may have cleaned its transient input; in that case the final claim
    record's request and message digests plus exact invocation/session material
    are the complete proof boundary.
    """

    if session is None:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.MISSING_SESSION,
            session=None,
        )
    if type(session) is not Session:
        raise TypeError("session must be an exact Session or None.")
    if type(request) is not RunRequest:
        raise TypeError("request must be an exact RunRequest.")
    if type(reference) is not RuntimeSessionCreateClaimReference:
        raise TypeError("reference must be an exact RuntimeSessionCreateClaimReference.")
    if type(key) is not RuntimeSessionCreateClaimReferenceKey:
        raise TypeError("key must be an exact RuntimeSessionCreateClaimReferenceKey.")
    operation_id = _runtime_session_create_reference_operation_id(operation_id)
    if session.id != reference.session_id:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.FOREIGN_SESSION,
            session=session,
        )
    try:
        request_authority_hmac_sha256 = _runtime_session_create_reference_request_hmac_sha256(
            request,
            operation_id=reference.operation_id,
            key=key,
        )
        expected_reference_claim_id = _runtime_session_create_reference_claim_id(
            session_id=reference.session_id,
            operation_id=reference.operation_id,
        )
    except (TypeError, ValueError):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.IDENTITY_CONFLICT,
            session=session,
        )
    if (
        request.session_id != reference.session_id
        or operation_id != reference.operation_id
        or key.key_id != reference.request_authority_key_id
        or not hmac.compare_digest(
            request_authority_hmac_sha256,
            reference.request_authority_hmac_sha256,
        )
        or not hmac.compare_digest(reference.claim_id, expected_reference_claim_id)
        or type(claim) is not session_request_contracts._RuntimeSessionCreateClaim
        or claim.token is not session_request_contracts._RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        or request._runtime_session_create_claim is not claim
        or claim.session_id != reference.session_id
        or claim.claim_id != reference.claim_id
    ):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.IDENTITY_CONFLICT,
            session=session,
        )

    raw_record = session.metadata.get(SESSION_CREATE_CLAIM_METADATA_KEY)
    if raw_record is None:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.FOREIGN_SESSION,
            session=session,
        )
    if type(raw_record) is not dict:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE,
            session=session,
        )
    record = raw_record
    base_fields = {
        "record_type",
        "schema_version",
        "claim_id",
        "request_sha256",
    }
    final_fields = base_fields | {"interaction_id", "messages_sha256"}
    if not base_fields.issubset(record) or not set(record).issubset(final_fields):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE,
            session=session,
        )
    if (
        record.get("record_type") != SESSION_CREATE_CLAIM_RECORD_TYPE
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != SESSION_CREATE_CLAIM_SCHEMA_VERSION
        or type(record.get("claim_id")) is not str
        or type(record.get("request_sha256")) is not str
    ):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE,
            session=session,
        )
    try:
        session_authority_rules._require_raw_sha256_digest(record["claim_id"])
        session_authority_rules._require_raw_sha256_digest(record["request_sha256"])
    except (TypeError, ValueError):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE,
            session=session,
        )
    if record["claim_id"] != reference.claim_id:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.FOREIGN_SESSION,
            session=session,
        )
    if claim.request_sha256 is None:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.INCOMPLETE_EVIDENCE,
            session=session,
        )
    if record["request_sha256"] != claim.request_sha256:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.TAMPERED_EVIDENCE,
            session=session,
        )
    if set(record) != final_fields:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.INCOMPLETE_EVIDENCE,
            session=session,
        )
    interaction_id = record.get("interaction_id")
    messages_sha256 = record.get("messages_sha256")
    if type(interaction_id) is not str or type(messages_sha256) is not str:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE,
            session=session,
        )
    try:
        require_clean_nonblank(interaction_id, "interaction_id")
        session_authority_rules._require_raw_sha256_digest(messages_sha256)
    except (TypeError, ValueError):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE,
            session=session,
        )
    if claim.messages_sha256 is None:
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.INCOMPLETE_EVIDENCE,
            session=session,
        )
    if not hmac.compare_digest(messages_sha256, claim.messages_sha256):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.TAMPERED_EVIDENCE,
            session=session,
        )
    expected_causal_budget_id = request.causal_budget_id or request.task_id or session.id
    if (
        session.agent_name != request.agent_name
        or session.parent_session_id != request.parent_session_id
        or session.causal_budget_id != expected_causal_budget_id
        or (
            request.environment_name is not None
            and session.environment_name != request.environment_name
        )
        or (
            request.target is not None
            and (
                session.provider_name != request.target.provider_name
                or session.model != request.target.model
            )
        )
        or (
            claim.expected_session_material is not None
            and session_request_contracts._SessionCreateMaterial.from_session(session)
            != claim.expected_session_material
        )
        or not session_invocation_matches_run_request(
            session,
            request=request,
            parent_session=parent_session,
        )
    ):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.FOREIGN_SESSION,
            session=session,
        )

    transient_matches = False
    if deferred_input is not None:
        if type(deferred_input) is not DeferredInteractionInput:
            return _runtime_session_create_authentication(
                RuntimeSessionCreateClaimAuthenticationDisposition.MALFORMED_EVIDENCE,
                session=session,
            )
        transient_matches = (
            deferred_input.interaction_id == interaction_id
            and session_input_messages_sha256(deferred_input.source_messages) == messages_sha256
        )
        if not transient_matches:
            return _runtime_session_create_authentication(
                RuntimeSessionCreateClaimAuthenticationDisposition.TAMPERED_EVIDENCE,
                session=session,
            )
    if (
        session.status not in session_summary_rules._OUTCOME_TERMINAL_STATUSES
        and not transient_matches
    ):
        return _runtime_session_create_authentication(
            RuntimeSessionCreateClaimAuthenticationDisposition.INCOMPLETE_EVIDENCE,
            session=session,
        )
    return _runtime_session_create_authentication(
        RuntimeSessionCreateClaimAuthenticationDisposition.MATCHING_SESSION,
        session=session,
        transient_input_authenticated=transient_matches,
    )


def session_matches_reconstructed_runtime_create_claim(
    session: Session | None,
    deferred_input: DeferredInteractionInput | None,
    claim: object,
    *,
    request: RunRequest,
    parent_session: Session | None,
) -> bool:
    """Authenticate a deterministic create claim after process reconstruction."""

    if (
        not session_has_runtime_create_claim(session, claim)
        or type(deferred_input) is not DeferredInteractionInput
    ):
        return False
    assert isinstance(session, Session)  # narrowed by the authenticated helper
    record = session.metadata[SESSION_CREATE_CLAIM_METADATA_KEY]
    assert isinstance(record, dict)
    return (
        session_invocation_matches_run_request(
            session,
            request=request,
            parent_session=parent_session,
        )
        and deferred_input.interaction_id == record["interaction_id"]
        and session_input_messages_sha256(deferred_input.source_messages)
        == record["messages_sha256"]
    )


def session_matches_runtime_create_claim(
    session: Session | None,
    deferred_input: DeferredInteractionInput | None,
    claim: object,
    *,
    request: RunRequest,
    parent_session: Session | None,
) -> bool:
    """Authenticate exact durable evidence for one runtime-owned create claim."""

    if (
        type(session) is not Session
        or type(deferred_input) is not DeferredInteractionInput
        or type(claim) is not session_request_contracts._RuntimeSessionCreateClaim
        or claim.token is not session_request_contracts._RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        or session.id != claim.session_id
        or claim.expected_session_material is None
        or claim.interaction_id is None
        or claim.messages_sha256 is None
        or claim.request_sha256 is None
    ):
        return False
    return (
        session_request_contracts._SessionCreateMaterial.from_session(session)
        == claim.expected_session_material
        and session_invocation_matches_run_request(
            session,
            request=request,
            parent_session=parent_session,
        )
        and session.metadata.get(SESSION_CREATE_CLAIM_METADATA_KEY)
        == _session_create_claim_record(
            claim_id=claim.claim_id,
            request_sha256=claim.request_sha256,
            interaction_id=claim.interaction_id,
            messages_sha256=claim.messages_sha256,
        )
        and deferred_input.interaction_id == claim.interaction_id
        and session_input_messages_sha256(deferred_input.source_messages) == claim.messages_sha256
    )


def session_invocation_matches_run_request(
    session: Session,
    *,
    request: RunRequest,
    parent_session: Session | None,
) -> bool:
    """Authenticate stored invocation provenance against one create request."""

    if type(session) is not Session or type(request) is not RunRequest:
        return False
    source = request._runtime_invocation_source or SessionExecutionSource.SDK_RUN
    task_invocation = request._runtime_task_invocation
    if parent_session is not None:
        if (
            request.parent_session_id != parent_session.id
            or session.parent_session_id != parent_session.id
        ):
            return False
        if request.invocation_origin is not None or request._verified_invocation_origin is not None:
            return False
        try:
            expected = inherited_session_invocation(parent_session.invocation, source=source)
        except (TypeError, ValueError):
            return False
        if task_invocation is not None and (
            task_invocation.id != request.task_id
            or (
                task_invocation.session_instance_id is not None
                and task_invocation.session_instance_id != session.instance_id
            )
            or task_invocation.invocation.origin != expected.origin
            or task_invocation.invocation.root_invocation_id != expected.root_invocation_id
            or (
                task_invocation.invocation.root_session_id is not None
                and task_invocation.invocation.root_session_id != expected.root_session_id
            )
        ):
            return False
        return session.invocation == expected
    if request.parent_session_id is not None or session.parent_session_id is not None:
        return False
    if source in {SessionExecutionSource.FORK, SessionExecutionSource.SUBAGENT}:
        return False
    verified_origin = request._verified_invocation_origin
    if request.invocation_origin is not None and verified_origin is not None:
        return False
    if task_invocation is not None:
        if (
            request.task_id is None
            or task_invocation.id != request.task_id
            or task_invocation.session_id not in {None, session.id}
            or (
                task_invocation.session_instance_id is not None
                and task_invocation.session_instance_id != session.instance_id
            )
            or request.invocation_origin is not None
            or verified_origin is not None
            or source is not SessionExecutionSource.TASK
            or (
                task_invocation.invocation.root_session_id is not None
                and task_invocation.invocation.root_session_id != session.id
            )
        ):
            return False
        try:
            expected = session_invocation_from_task(
                task_invocation.invocation,
                session_id=session.id,
                source=source,
            )
        except (TypeError, ValueError):
            return False
        return session.invocation == expected
    if source is SessionExecutionSource.TASK:
        return False
    if verified_origin is not None:
        if (
            source is not SessionExecutionSource.HTTP_RUN
            or verified_origin.trust is not InvocationOriginTrust.SERVER_VERIFIED
        ):
            return False
        expected_origin = verified_origin
    elif request.invocation_origin is not None:
        if source is not SessionExecutionSource.SDK_RUN:
            return False
        expected_origin = InvocationOrigin(
            trust=InvocationOriginTrust.HOST_ASSERTED,
            subject=request.invocation_origin.subject,
            tenant=request.invocation_origin.tenant,
        )
    else:
        expected_origin = InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED)
    return (
        session.invocation.origin == expected_origin
        and session.invocation.root_session_id == session.id
        and session.invocation.source is source
    )
