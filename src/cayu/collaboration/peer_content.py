"""Typed, bounded peer-content delivery contracts.

These values describe historical collaboration evidence.  They are not
execution permits, recipient-session handles, or consent to expose content.
The collaboration and session owners must still authenticate every operation
and persist its result transactionally.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, asynccontextmanager
from hashlib import sha256
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    canonical_bounded_durable_json_bytes,
    inspect_bounded_durable_json,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_ENVELOPE_BYTES,
    MAX_NODES,
    ContractValue,
    Generation,
    Identifier,
    OwnerRef,
)
from cayu.messages import PeerContentPart
from cayu.sessions.creation_fence import SessionCreationDecision, SessionCreationTarget, decide

if TYPE_CHECKING:
    from cayu.collaboration.exports import (
        SessionExportAccessContext,
        SessionExportRuntimeOrigin,
    )

PEER_CONTENT_CONTRACT_VERSION = 1
PEER_CONTENT_MAX_BYTES = 256 * 1024
PEER_CONTENT_MAX_TEXT_BYTES = 128 * 1024
PEER_CONTENT_MAX_ARTIFACTS = 32
PEER_CONTENT_MAX_ATTEMPTS = 64
PEER_CONTENT_MAX_OUTSTANDING_PER_CONSUMER = 64


def validate_peer_discovery(after_operation_key: str | None, limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError("Peer discovery limit must be between one and 64.")
    if after_operation_key is not None:
        require_durable_clean_nonblank(after_operation_key, "after_operation_key")


def _commitment(value: object, field_name: str) -> str:
    return sha256(
        canonical_bounded_durable_json_bytes(
            value,
            field_name,
            max_bytes=MAX_ENVELOPE_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
    ).hexdigest()


class PeerContentConflict(ValueError):
    """An immutable peer append key was reused with different evidence."""


class PeerContentUnavailable(PermissionError):
    """Peer content cannot be exposed under current authority."""


@runtime_checkable
class PeerContentExposureReceiver(Protocol):
    """Registered #1763 owner for current, revocable provider disclosure."""

    def acquire_peer_exposures(
        self,
        context: SessionExportAccessContext | None,
        *,
        items: tuple[PeerContentExposureItem, ...],
    ) -> AbstractAsyncContextManager[tuple[PeerContentPayload, ...]]:
        """Authorize all items under one guard; yield projections in item order.

        Retain current authority through serialization. Do not recursively enter
        a non-reentrant per-item guard. Unsupported batches must fail closed.
        """
        ...

    def acquire_peer_exposure(
        self,
        context: SessionExportAccessContext | None,
        *,
        origin: SessionExportRuntimeOrigin | PeerModelAttemptOrigin,
        append_key: PeerAppendKey,
        occurrence: PeerContentOccurrence,
        audience: OwnerRef,
        provider_name: Identifier,
        model: Identifier,
        model_attempt_id: Identifier,
        capability_version: Generation,
    ) -> AbstractAsyncContextManager[PeerContentPayload]:
        """Yield the bounded projection while holding disclosure authority.

        Provider adapters must serialize only this returned projection.  The
        occurrence is evidence used for authorization, never the projection
        source passed directly to an external provider.
        """
        ...


class PeerContentAppendAuthorization(ContractValue):
    """Typed evidence returned by the registered #1763 export owner."""

    source_export_receipt_id: Identifier
    producer_receipt_id: Identifier
    source_session_id: Identifier
    source_session_instance_id: Identifier
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    audience: tuple[Identifier, ...] = Field(min_length=1, max_length=64)


class RegisteredPeerContentExposureReceiver:
    """Adapter from the registered #1763 policy to peer exposure.

    This adapter has no authorization of its own. It delegates to the
    registered policy hook, whose default implementation denies the operation.
    """

    def __init__(self, policy: object) -> None:
        self._policy = policy

    def acquire_peer_exposures(self, context, *, items):
        acquire = getattr(self._policy, "acquire_peer_exposures", None)
        if not callable(acquire):
            raise PeerContentUnavailable()
        return acquire(context, items=items)

    @asynccontextmanager
    async def acquire_peer_exposure(self, context, **kwargs):
        acquire = getattr(self._policy, "acquire_peer_exposure", None)
        if not callable(acquire):
            raise PeerContentUnavailable()
        async with acquire(context, **kwargs) as projection:
            if type(projection) is not PeerContentPayload:
                raise PeerContentUnavailable()
            yield projection


class PeerContentPayload(ContractValue):
    """Bounded historical content selected by the authenticated producer."""

    schema_version: Literal[1] = 1
    text: StrictStr
    # Only independently qualified artifact commitments cross this boundary;
    # raw resource references and executable handles are intentionally absent.
    artifact_commitments: tuple[StrictStr, ...] = Field(
        default=(), max_length=PEER_CONTENT_MAX_ARTIFACTS
    )
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = require_durable_text(value, "peer_content.text")
        if len(value.encode("utf-8")) > PEER_CONTENT_MAX_TEXT_BYTES:
            raise ValueError("Peer content text exceeds the byte limit.")
        return value

    @field_validator("artifact_commitments", mode="before")
    @classmethod
    def copy_artifacts(cls, value):
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("Peer artifact commitments must be a sequence.")
        result = tuple(value)
        if any(type(item) is not str for item in result):
            raise ValueError("Peer artifact commitments must be strings.")
        result = tuple(
            require_durable_clean_nonblank(item, "peer_content.artifact_commitment")
            for item in result
        )
        inspect_bounded_durable_json(
            list(result),
            "peer_content.artifact_commitments",
            max_bytes=PEER_CONTENT_MAX_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
        return result

    @model_validator(mode="after")
    def validate_commitment(self) -> PeerContentPayload:
        material = {"text": self.text, "artifact_commitments": list(self.artifact_commitments)}
        expected = _commitment(material, "peer_content.payload")
        if self.content_sha256 != expected:
            raise ValueError("Peer content commitment does not match its payload.")
        return self


class PeerAppendKey(ContractValue):
    """Stable identity of one logical source occurrence delivered to a target."""

    schema_version: Literal[1] = 1
    collaboration_namespace: Identifier
    collaboration_generation: Generation
    occurrence_id: Identifier
    consumer_id: Identifier
    consumer_participant_incarnation: Identifier
    projection_id: Identifier
    projection_schema: Identifier
    target_session_id: Identifier | None = None
    target_session_instance_id: Identifier | None = None
    creation_target: SessionCreationTarget | None = None

    @model_validator(mode="after")
    def validate_target(self) -> PeerAppendKey:
        if self.creation_target is None:
            if self.target_session_id is None or self.target_session_instance_id is None:
                raise ValueError("Peer delivery requires an exact existing or creation target.")
        else:
            if self.target_session_id is not None or self.target_session_instance_id is not None:
                raise ValueError("Creation targets cannot claim a future session incarnation.")
            participant = self.creation_target.permit.intent.request.participant
            if (participant.participant_id, participant.incarnation) != (
                self.consumer_id,
                self.consumer_participant_incarnation,
            ):
                raise ValueError("Peer consumer conflicts with recipient creation authority.")
        return self


def resolve_peer_target(
    key: PeerAppendKey, decision: SessionCreationDecision | None
) -> tuple[str | None, str | None, bool]:
    """Resolve only inside the receiving-store mutation transaction.

    The boolean denotes whole-creation exclusion, not delivery withdrawal.
    Missing creation evidence is unavailable, never proof of non-creation.
    """
    if key.creation_target is None:
        return key.target_session_id, key.target_session_instance_id, False
    if decision is None:
        raise PeerContentUnavailable("Exact recipient creation evidence is unavailable.")
    decide(key.creation_target, decision)
    return decision.session_id, decision.session_instance_id, decision.state == "excluded"


def peer_queue_id(key: PeerAppendKey, session_id: str, instance_id: str) -> str:
    """One queue identity per successful occurrence/consumer/projection/target."""
    return "peer:" + _commitment(
        {
            "namespace": key.collaboration_namespace,
            "generation": key.collaboration_generation,
            "occurrence": key.occurrence_id,
            "consumer": key.consumer_id,
            "consumer_incarnation": key.consumer_participant_incarnation,
            "projection": key.projection_id,
            "schema": key.projection_schema,
            "session": session_id,
            "instance": instance_id,
        },
        "peer_queue_identity",
    )


class PeerModelAttemptOrigin(ContractValue):
    """Runtime-minted origin for one model-attempt disclosure."""

    schema_version: Literal[1] = 1
    target_session_id: Identifier
    target_session_instance_id: Identifier
    run_epoch: StrictInt = Field(ge=0, le=MAX_PORTABLE_JSON_INTEGER)
    root_invocation_id: Identifier
    requester_principal: Identifier
    interaction_id: Identifier
    model_step_id: Identifier
    model_attempt_id: Identifier
    append_key: PeerAppendKey
    provider_name: Identifier
    model: Identifier
    capability_version: Generation
    exposure_generation: Generation


class PeerDeliveryAttemptKey(ContractValue):
    """Identity of one retryable delivery attempt for an append key."""

    schema_version: Literal[1] = 1
    append_key: PeerAppendKey
    interest_id: Identifier
    attempt_generation: StrictInt = Field(ge=1, le=PEER_CONTENT_MAX_ATTEMPTS)
    target_run_epoch: StrictInt = Field(ge=0, le=MAX_PORTABLE_JSON_INTEGER)
    target_transcript_cursor: StrictInt = Field(ge=0, le=MAX_PORTABLE_JSON_INTEGER)
    withdrawal_generation: Generation
    deadline_at_ms: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)


class PeerContentOccurrence(ContractValue):
    """Authenticated producer evidence referenced by a delivery request."""

    schema_version: Literal[1] = 1
    occurrence_id: Identifier
    sender_participant_id: Identifier
    sender_participant_incarnation: Identifier
    sender_session_id: Identifier
    sender_session_instance_id: Identifier
    producer_receipt_id: Identifier
    source_export_receipt_id: Identifier
    payload: PeerContentPayload
    audience: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    provenance_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> PeerContentOccurrence:
        material = {
            "occurrence_id": self.occurrence_id,
            "sender_participant_id": self.sender_participant_id,
            "sender_participant_incarnation": self.sender_participant_incarnation,
            "sender_session_id": self.sender_session_id,
            "sender_session_instance_id": self.sender_session_instance_id,
            "producer_receipt_id": self.producer_receipt_id,
            "source_export_receipt_id": self.source_export_receipt_id,
            "payload": self.payload.model_dump(mode="json"),
            "audience": list(self.audience),
        }
        expected = _commitment(material, "peer_content.occurrence")
        if self.provenance_sha256 != expected:
            raise ValueError("Peer occurrence provenance does not match its evidence.")
        return self

    def to_message_part(
        self, *, append_key: PeerAppendKey, projection_id: str, operation_key: str
    ) -> PeerContentPart:
        return PeerContentPart(
            text=self.payload.text,
            sender_participant_id=self.sender_participant_id,
            sender_participant_incarnation=self.sender_participant_incarnation,
            sender_session_id=self.sender_session_id,
            sender_session_instance_id=self.sender_session_instance_id,
            occurrence_id=self.occurrence_id,
            append_key_json=append_key.model_dump_json(),
            operation_key=operation_key,
            projection_id=projection_id,
            provenance_sha256=self.provenance_sha256,
        )


class PeerContentExposureItem(ContractValue):
    """One runtime-owned member of an atomic disclosure authorization batch."""

    origin: PeerModelAttemptOrigin
    occurrence: PeerContentOccurrence
    audience: OwnerRef

    @model_validator(mode="after")
    def validate_source(self) -> PeerContentExposureItem:
        key = self.origin.append_key
        if (
            self.occurrence.occurrence_id != key.occurrence_id
            or key.consumer_id not in self.occurrence.audience
        ):
            raise ValueError("Peer exposure source conflicts with its append identity.")
        if key.creation_target is None and (
            self.origin.target_session_id != key.target_session_id
            or self.origin.target_session_instance_id != key.target_session_instance_id
        ):
            raise ValueError("Peer exposure target conflicts with its append identity.")
        return self

    def single_arguments(self):
        return {
            "origin": self.origin,
            "append_key": self.origin.append_key,
            "occurrence": self.occurrence,
            "audience": self.audience,
            "provider_name": self.origin.provider_name,
            "model": self.origin.model,
            "model_attempt_id": self.origin.model_attempt_id,
            "capability_version": self.origin.capability_version,
        }


class PeerContentAppendRequest(ContractValue):
    """Authenticated append intent; IDs alone never authenticate the source."""

    schema_version: Literal[1] = 1
    operation_key: Identifier
    append_key: PeerAppendKey
    attempt_key: PeerDeliveryAttemptKey
    occurrence: PeerContentOccurrence
    wake_policy: Literal["none", "ordinary_continuation"] = "none"
    replaces_operation_key: Identifier | None = None

    @model_validator(mode="after")
    def validate_keys(self) -> PeerContentAppendRequest:
        if self.attempt_key.append_key != self.append_key:
            raise ValueError("Peer attempt and append keys must match.")
        if self.occurrence.occurrence_id != self.append_key.occurrence_id:
            raise ValueError("Peer occurrence does not match append key.")
        if self.append_key.consumer_id not in self.occurrence.audience:
            raise ValueError("Peer consumer is outside the authenticated audience.")
        return self


class PeerContentReceipt(ContractValue):
    """Durable append/exclusion outcome; queue, transcript, and exposure remain separate."""

    schema_version: Literal[1] = 1
    operation_key: Identifier
    append_key: PeerAppendKey
    attempt_generation: Generation
    status: Literal["appended", "excluded", "pending", "not_exposed"]
    disclosure: Literal["available", "withheld"] = "available"
    occurrence: PeerContentOccurrence | None = None
    queue_id: Identifier | None = None
    transcript_event_id: Identifier | None = None
    target_session_id: Identifier | None = None
    target_session_instance_id: Identifier | None = None
    reason: Identifier | None = None
    replayed: StrictBool = False

    @model_validator(mode="after")
    def validate_status(self) -> PeerContentReceipt:
        if self.status == "appended" and (
            self.target_session_id is None or self.target_session_instance_id is None
        ):
            raise ValueError("An appended peer receipt requires resolved target identity.")
        if self.status == "appended" and self.queue_id is None:
            raise ValueError("An appended peer receipt requires queue evidence.")
        if self.status == "appended" and self.occurrence is None and self.disclosure != "withheld":
            raise ValueError("An appended peer receipt requires occurrence evidence.")
        if self.status in {"excluded", "not_exposed"} and self.reason is None:
            raise ValueError("A non-append peer receipt requires a reason.")
        return self


class PeerContentExposureRequest(ContractValue):
    """Authenticated record of one named model-attempt exposure decision."""

    schema_version: Literal[1] = 1
    operation_key: Identifier
    append_key: PeerAppendKey
    exposure_id: Identifier
    model_attempt_id: Identifier
    provider_name: Identifier
    capability_version: Generation
    outcome: Literal["exposed", "not_exposed"]
    reason: Identifier | None = None

    @classmethod
    def for_model_attempt(
        cls,
        *,
        append_key: PeerAppendKey,
        append_operation_key: str,
        model_attempt_id: str,
        provider_name: str,
        capability_version: int,
    ) -> PeerContentExposureRequest:
        identity = _commitment(
            {
                "domain": "cayu.peer-model-exposure.v1",
                "append_key": append_key.model_dump(mode="json"),
                "append_operation_key": append_operation_key,
                "model_attempt_id": model_attempt_id,
                "provider_name": provider_name,
                "capability_version": capability_version,
            },
            "peer_content.exposure_identity",
        )
        return cls(
            operation_key=f"peer-exposure:{identity}",
            exposure_id=f"peer-exposure:{identity}",
            append_key=append_key,
            model_attempt_id=model_attempt_id,
            provider_name=provider_name,
            capability_version=capability_version,
            outcome="exposed",
        )

    @model_validator(mode="after")
    def validate_outcome(self) -> PeerContentExposureRequest:
        if self.outcome == "not_exposed" and self.reason is None:
            raise ValueError("A non-exposure receipt requires a reason.")
        if self.outcome == "exposed" and self.reason is not None:
            raise ValueError("An exposure receipt cannot carry a failure reason.")
        return self

    def identity_commitment(self) -> str:
        return _commitment(
            {
                "operation_key": self.operation_key,
                "append_key": self.append_key.model_dump(mode="json"),
                "exposure_id": self.exposure_id,
                "model_attempt_id": self.model_attempt_id,
                "provider_name": self.provider_name,
                "capability_version": self.capability_version,
            },
            "peer_content.exposure_identity",
        )


class PeerContentExposureReceipt(ContractValue):
    """Durable, replayable exposure outcome separate from queue/append ownership."""

    schema_version: Literal[1] = 1
    operation_key: Identifier
    append_key: PeerAppendKey
    exposure_id: Identifier
    model_attempt_id: Identifier
    outcome: Literal["pending", "exposed", "not_exposed"]
    reason: Identifier | None = None
    replayed: StrictBool = False


__all__ = [
    "PEER_CONTENT_CONTRACT_VERSION",
    "PEER_CONTENT_MAX_ATTEMPTS",
    "PEER_CONTENT_MAX_BYTES",
    "PEER_CONTENT_MAX_TEXT_BYTES",
    "PeerAppendKey",
    "PeerContentAppendAuthorization",
    "PeerContentAppendRequest",
    "PeerContentConflict",
    "PeerContentExposureItem",
    "PeerContentExposureReceipt",
    "PeerContentExposureReceiver",
    "PeerContentExposureRequest",
    "PeerContentOccurrence",
    "PeerContentPayload",
    "PeerContentReceipt",
    "PeerContentUnavailable",
    "PeerDeliveryAttemptKey",
    "PeerModelAttemptOrigin",
    "RegisteredPeerContentExposureReceiver",
]
