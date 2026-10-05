"""Durable session continuation tickets and early-result latches.

This module owns the session side of a wait.  It deliberately does not elect
finite predicates or subscribe to another owner; those responsibilities belong
to the collaboration wait layer.  The values below are immutable evidence and
must be authenticated by the SessionStore before they are accepted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import MAX_PORTABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    ContractValue,
    ExpectedOperation,
    HandoffIntent,
    Identifier,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._preparation import contract_bytes
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.sessions._invocation_lifecycle import (
        AdmitInvocationCommand,
    )

CONTINUATION_OPERATION_PREFIX = "session-continuation:"
CONTINUATION_NAMESPACE_KEY = CONTINUATION_OPERATION_PREFIX + "namespace"
CONTINUATION_SCHEMA_VERSION = 1
CONTINUATION_MAX_TARGETS = 64
CONTINUATION_MAX_RELEASED_RETIREMENT_BYTES = 512
CONTINUATION_MAX_DIGEST_BYTES = 256
CONTINUATION_MAX_LATCH_EVIDENCE_BYTES = 8192
CONTINUATION_MAX_CONSUMPTION_EVIDENCE_BYTES = 12288
CONTINUATION_MAX_RETIREMENT_EVIDENCE_BYTES = 4096
CONTINUATION_MAX_EVENTS = 8
CONTINUATION_MAX_RECOVERY_WRITER_BYTES = 2048
CONTINUATION_MAX_EVENT_BYTES = 512
CONTINUATION_MAX_SERVICES = 32
# Fixed hash/key encodings, closed enums and portable integer bounds make the
# largest current reference 352 bytes. Keep explicit headroom without reserving
# enough redundant slack to exclude ordinary generated continuation identities.
CONTINUATION_MAX_SERVICE_REFERENCE_BYTES = 384
CONTINUATION_SERVICE_PREFIX = CONTINUATION_OPERATION_PREFIX + "service:"


class ContinuationConflict(ValueError):
    """A continuation key was reused with different authority or state."""


class ContinuationUnavailable(RuntimeError):
    """The exact continuation cannot be positively reconstructed."""


class ContinuationLatchReceiver(Protocol):
    """Qualified receiving boundary for wait evidence."""

    async def authenticate_continuation_latch(
        self,
        latch: ContinuationLatch,
    ) -> ContinuationLatch: ...


class RetainedContinuationLatchReceiver(ABC):
    """Explicit registration contract for source-owned latch recovery.

    Implementations authenticate the complete latch against their durable source
    owner. The retained entrance must observe that owner's operation through
    settlement; stopping a public observer must not abandon a dispatched write.
    This is application configuration, never authority supplied in a latch.
    """

    @abstractmethod
    async def authenticate_continuation_latch(self, latch: ContinuationLatch) -> ContinuationLatch:
        """Authenticate through the source owner's bounded public observer."""

    @abstractmethod
    async def _authenticate_latch_owned(self, latch: ContinuationLatch) -> ContinuationLatch:
        """Authenticate while retaining the source operation until it settles."""


def _bounded_digest(value: str, field_name: str) -> str:
    if type(value) is not str or not value or len(value) > CONTINUATION_MAX_DIGEST_BYTES:
        raise ValueError(f"{field_name} must be a bounded non-empty digest.")
    return value


def _aware_time(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


class ContinuationNamespace(ContractValue):
    """Store-owned namespace binding for one session incarnation."""

    session_id: Identifier
    session_instance_id: Identifier
    owner: OwnerRef
    namespace_id: Identifier
    generation: Literal[1] = 1

    @model_validator(mode="after")
    def exact_namespace(self) -> ContinuationNamespace:
        if self.namespace_id != continuation_namespace_id(
            self.session_id, self.session_instance_id, self.owner
        ):
            raise ValueError("Continuation namespace is not runtime-derived.")
        return self


class ContinuationWait(ContractValue):
    """Caller wait intent, excluding runtime-generated session and namespace identity."""

    # The durable ticket uses the same bounded key.  Keep the public intent
    # equally strict so a valid-looking wait cannot fail only after the owner
    # has derived and attempted to persist its ticket.
    registration_key: Identifier = Field(max_length=256)
    targets: tuple[ObjectRef, ...] = Field(min_length=1, max_length=CONTINUATION_MAX_TARGETS)
    predicate_kind: Literal["ALL_SUCCESS", "ALL_SETTLED", "ANY_SUCCESS", "QUORUM_SUCCESS"]
    predicate_version: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    threshold: StrictInt | None = Field(default=None, ge=1, le=CONTINUATION_MAX_TARGETS)
    deadline: StrictStr | None
    failure_policy: StrictStr = Field(min_length=1, max_length=128)
    service_policy: StrictStr = Field(min_length=1, max_length=128)
    wait_edge_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    purpose: StrictStr = Field(min_length=1, max_length=256)
    execution_admission_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    collaboration_wait_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    unordered_fields = frozenset({"targets"})

    @model_validator(mode="after")
    def validate_execution_commitments(self) -> ContinuationWait:
        if (self.execution_admission_sha256 is None) != (self.collaboration_wait_sha256 is None):
            raise ValueError("Execution and collaboration wait commitments must be paired.")
        return self


class ContinuationTicket(ContractValue):
    """Immutable wait identity plus its mutable lifecycle revision."""

    schema_version: Literal[1] = CONTINUATION_SCHEMA_VERSION
    namespace: ContinuationNamespace
    session_id: StrictStr
    session_instance_id: StrictStr
    owner: OwnerRef
    registration_key: StrictStr = Field(min_length=1, max_length=256)
    targets: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=CONTINUATION_MAX_TARGETS,
    )
    predicate_kind: Literal["ALL_SUCCESS", "ALL_SETTLED", "ANY_SUCCESS", "QUORUM_SUCCESS"]
    predicate_version: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    threshold: StrictInt | None = Field(default=None, ge=1, le=CONTINUATION_MAX_TARGETS)
    deadline: StrictStr | None
    failure_policy: StrictStr = Field(min_length=1, max_length=128)
    service_policy: StrictStr = Field(min_length=1, max_length=128)
    wait_edge_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    interaction_id: StrictStr
    writer_generation: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    purpose: StrictStr = Field(min_length=1, max_length=256)
    execution_admission_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    collaboration_wait_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    state: Literal["ARMING", "WAITING", "SERVICING", "CONSUMED", "RETIRED"]
    revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)

    unordered_fields = frozenset({"targets"})

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _aware_time(datetime.fromisoformat(value), "deadline").isoformat()
        )

    @field_validator("session_id", "session_instance_id", "registration_key", "interaction_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value or len(value.encode("utf-8")) > 512:
            raise ValueError(f"{info.field_name} is too large.")
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> ContinuationTicket:
        if (self.execution_admission_sha256 is None) != (self.collaboration_wait_sha256 is None):
            raise ValueError("Execution and collaboration wait commitments must be paired.")
        if (
            self.namespace.session_id != self.session_id
            or self.namespace.session_instance_id != self.session_instance_id
            or self.namespace.owner != self.owner
        ):
            raise ValueError("Continuation ticket ownership conflicts with its namespace.")
        if self.predicate_kind == "QUORUM_SUCCESS":
            if self.threshold is None or self.threshold > len(self.targets):
                raise ValueError("Quorum threshold must fit the distinct target set.")
        elif self.threshold is not None:
            raise ValueError("Only quorum predicates may carry a threshold.")
        if self.state == "ARMING" and self.revision != 1:
            raise ValueError("A newly armed ticket must have revision one.")
        return self


class ContinuationLatch(ContractValue):
    """Immutable readiness proof, independent of ticket lifecycle revision."""

    schema_version: Literal[1] = CONTINUATION_SCHEMA_VERSION
    ticket: ContinuationTicket
    wait_receipt_digest: StrictStr
    outcome_kind: Literal["success", "settled", "failure", "unavailable"]
    selected_manifest: tuple[ObjectRef, ...] = Field(max_length=CONTINUATION_MAX_TARGETS)
    disclosure_digest: StrictStr
    latch_key: StrictStr = Field(min_length=1, max_length=256)
    outcome_digest: StrictStr
    accepted_at: StrictStr
    # Optional source identity used by registered collaboration receivers.
    # Existing continuation producers leave it unset; a collaboration wait
    # supplies it so the receiver can perform durable exact readback after a
    # process restart instead of relying on an in-memory callback map.
    wait_operation: OperationRef | None = None

    unordered_fields = frozenset({"selected_manifest"})

    @field_validator(
        "wait_receipt_digest",
        "disclosure_digest",
        "outcome_digest",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _bounded_digest(value, info.field_name)

    @field_validator("accepted_at")
    @classmethod
    def validate_accepted_at(cls, value: str) -> str:
        return _aware_time(datetime.fromisoformat(value), "accepted_at").isoformat()

    @model_validator(mode="after")
    def validate_ticket_identity(self) -> ContinuationLatch:
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json", exclude={"ticket"}), "latch evidence"
                )
            )
            > CONTINUATION_MAX_LATCH_EVIDENCE_BYTES
        ):
            raise ValueError("Continuation latch evidence exceeds its reserved capacity.")
        if self.ticket.state in {"CONSUMED", "RETIRED"}:
            raise ValueError("A terminal ticket cannot accept a new latch.")
        return self


def continuation_registration_operation(parent: OperationRef) -> OperationRef:
    """Derive a retained child key from the stable parent/slot, never kind."""

    material = {"parent": parent.model_dump(mode="json"), "slot": "wait-registration"}
    key = sha256(canonical_durable_json_bytes(material, "continuation registration")).hexdigest()
    return parent.model_copy(update={"caller_key": "wait-registration:" + key})


class ContinuationPreparation(ExpectedOperation[ContinuationTicket]):
    """Exact receiving command; structural validity does not grant authority."""

    kind: Literal["session.continuation.prepare"] = "session.continuation.prepare"
    schema_version: Literal[1] = 1
    mode: Literal["wait"] = "wait"
    receipt_stage: Literal["prepared"] = "prepared"
    registration: HandoffIntent[ContinuationTicket]

    @model_validator(mode="after")
    def consistent_ticket(self) -> ContinuationPreparation:
        ticket = self.intent
        if (
            self.source != ticket.owner
            or self.destination != ticket.owner
            or self.operation.namespace_incarnation != ticket.namespace.namespace_id
            or self.operation.generation != ticket.namespace.generation
            or self.operation.caller_key != ticket.registration_key
            or self.initiator.issuer != ticket.owner
            or self.initiator.interaction_id != ticket.interaction_id
            or self.initiator.invocation_id is None
            or self.initiator.participant is not None
            or self.initiator.mandate is not None
            or ticket.state != "ARMING"
            or ticket.revision != 1
            or self.registration.slot.source != self.source
            or self.registration.slot.parent != self.operation
            or self.registration.slot.slot != "wait-registration"
            or self.registration.child.operation
            != continuation_registration_operation(self.operation)
            or self.registration.child.source != self.source
            or self.registration.child.initiator != self.initiator
            or self.registration.child.kind != "wait.register"
            or self.registration.child.schema_version != 1
            or self.registration.child.mode != "wait"
            or self.registration.child.receipt_stage != "registered"
            or self.registration.child.intent != ticket
        ):
            raise ValueError("Continuation preparation authority conflicts with its ticket.")
        return self


class ContinuationConsumption(ContractValue):
    """One durable handoff from a ready latch to one invocation."""

    schema_version: Literal[1] = CONTINUATION_SCHEMA_VERSION
    ticket: ContinuationTicket
    latch: ContinuationLatch
    continuation_id: StrictStr = Field(min_length=1, max_length=256)
    mode: Literal["inline", "queued"]
    input_digest: StrictStr
    profile_digest: StrictStr
    budget_digest: StrictStr
    service_digest: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    admission_command_digest: StrictStr
    admission_expected_run_epoch: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    admission_claimed: StrictBool = False
    admission_claim_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    receipt_stage: Literal["prepared", "admitted", "excluded"]
    accepted_at: StrictStr

    @field_validator(
        "input_digest",
        "profile_digest",
        "budget_digest",
        "admission_command_digest",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _bounded_digest(value, info.field_name)

    @field_validator("accepted_at")
    @classmethod
    def validate_accepted_at(cls, value: str) -> str:
        return _aware_time(datetime.fromisoformat(value), "accepted_at").isoformat()

    @model_validator(mode="after")
    def validate_match(self) -> ContinuationConsumption:
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json", exclude={"ticket", "latch"}),
                    "consumption evidence",
                )
            )
            > CONTINUATION_MAX_CONSUMPTION_EVIDENCE_BYTES
        ):
            raise ValueError("Continuation consumption evidence exceeds its reserved capacity.")
        if self.latch.ticket.namespace != self.ticket.namespace:
            raise ValueError("Continuation consumption has a foreign latch.")
        if self.admission_claimed != (self.admission_claim_id is not None):
            raise ValueError("Continuation admission claim identity does not match its state.")
        if self.receipt_stage == "excluded":
            if self.ticket.state != "RETIRED":
                raise ValueError("Excluded continuation must be retired.")
        elif self.ticket.state not in {"WAITING", "CONSUMED"}:
            raise ValueError("Only a waiting or consumed continuation can carry a receipt.")
        return self


class ContinuationService(ContractValue):
    """Exact host-selected continuation, independent of automatic scheduling."""

    ticket: ContinuationTicket
    latch: ContinuationLatch
    continuation_id: StrictStr = Field(min_length=1, max_length=256)
    mode: Literal["inline", "queued"]
    accepted_at: StrictStr

    @field_validator("accepted_at")
    @classmethod
    def validate_accepted_at(cls, value: str) -> str:
        return _aware_time(datetime.fromisoformat(value), "accepted_at").isoformat()

    @model_validator(mode="after")
    def validate_wait(self) -> ContinuationService:
        require_ticket_identity(self.ticket, self.latch.ticket)
        if self.ticket.state != "WAITING":
            raise ValueError("Continuation service requires a waiting ticket.")
        return self


class ContinuationRetirement(ContractValue):
    """Explicit, separately keyed retirement authority."""

    schema_version: Literal[1] = CONTINUATION_SCHEMA_VERSION
    ticket: ContinuationTicket
    control_id: StrictStr = Field(min_length=1, max_length=256)
    reason: Literal["cancelled", "expired", "failed", "superseded", "unavailable"]
    retired_at: StrictStr

    @field_validator("retired_at")
    @classmethod
    def validate_retired_at(cls, value: str) -> str:
        return _aware_time(datetime.fromisoformat(value), "retired_at").isoformat()

    @model_validator(mode="after")
    def validate_reserved_capacity(self) -> ContinuationRetirement:
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json", exclude={"ticket"}), "retirement evidence"
                )
            )
            > CONTINUATION_MAX_RETIREMENT_EVIDENCE_BYTES
        ):
            raise ValueError("Continuation retirement exceeds its reserved capacity.")
        return self


class ContinuationReleasedExecution(ContractValue):
    """Bounded native proof retained after the invocation ledger can be pruned."""

    permit_operation: StrictStr | None = Field(pattern=r"^participant-execution:[0-9a-f]{64}$")
    permit_commitment: StrictStr | None = Field(pattern=r"^[0-9a-f]{64}$")
    admission_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    release_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bounded_evidence(self) -> ContinuationReleasedExecution:
        if (self.permit_operation is None) != (self.permit_commitment is None):
            raise ValueError("Released execution has incomplete permit evidence.")
        if (
            len(canonical_durable_json_bytes(self.model_dump(mode="json"), "released execution"))
            > CONTINUATION_MAX_RELEASED_RETIREMENT_BYTES
        ):
            raise ValueError("Released execution evidence exceeds its reserved capacity.")
        return self


class ContinuationReleasedRetirement(ContractValue):
    """Exact original execution expected by released-writer cleanup.

    This is data, not invocation authority. The receiving transaction must prove
    native permit consumption, writer release and exact settled service history.
    """

    retirement: ContinuationRetirement
    permit_operation: Identifier | None
    permit_commitment: StrictStr | None = Field(pattern=r"^[0-9a-f]{64}$")
    settled_services: tuple[tuple[StrictStr, StrictStr], ...] = Field(
        default=(), max_length=CONTINUATION_MAX_SERVICES
    )

    @model_validator(mode="after")
    def complete_execution_identity(self) -> ContinuationReleasedRetirement:
        if (self.permit_operation is None) != (self.permit_commitment is None):
            raise ValueError("Released retirement has incomplete permit evidence.")
        if self.permit_operation is None and (
            self.retirement.ticket.purpose != "external-event-v1"
            or self.retirement.ticket.execution_admission_sha256 is not None
            or self.retirement.ticket.collaboration_wait_sha256 is not None
        ):
            raise ValueError("Ordinary released retirement requires an external-wait ticket.")
        return self


class ContinuationEvent(ContractValue):
    """Bounded receiving-owner history committed atomically with its aggregate."""

    sequence: StrictInt = Field(ge=1, le=CONTINUATION_MAX_EVENTS)
    ticket_key: StrictStr = Field(max_length=128)
    kind: Literal[
        "prepared",
        "parked",
        "latched",
        "consumption_prepared",
        "admission_claimed",
        "admission_claim_released",
        "admitted",
        "excluded",
        "retired",
        "retirement_acknowledged",
    ]
    record_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_size(self) -> ContinuationEvent:
        if (
            len(canonical_durable_json_bytes(self.model_dump(mode="json"), "continuation event"))
            > CONTINUATION_MAX_EVENT_BYTES
        ):
            raise ValueError("Continuation event exceeds its reserved capacity.")
        return self


class ContinuationServiceReference(ContractValue):
    """Bounded child-history index; the full service receipt is stored separately."""

    key: StrictStr = Field(pattern=r"^session-continuation:service:[0-9a-f]{64}$")
    record_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    generation: StrictInt = Field(ge=1, le=CONTINUATION_MAX_SERVICES)
    parent_generation: StrictInt | None = Field(default=None, ge=1, le=CONTINUATION_MAX_SERVICES)
    state: Literal["prepared", "reserved", "admitted", "returned", "excluded"]
    mode: Literal["same_session", "side_session"]
    expected_run_epoch: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    returned_writer_generation: StrictInt | None = Field(
        default=None, ge=1, le=MAX_PORTABLE_JSON_INTEGER
    )

    @model_validator(mode="after")
    def validate_reference(self) -> ContinuationServiceReference:
        if (self.state == "returned") != (self.returned_writer_generation is not None):
            raise ValueError("Temporary service reference lacks exact return evidence.")
        if (
            len(canonical_durable_json_bytes(self.model_dump(mode="json"), "service reference"))
            > CONTINUATION_MAX_SERVICE_REFERENCE_BYTES
        ):
            raise ValueError("Temporary service reference exceeds its reserved capacity.")
        return self


class ContinuationRecoveryWriter(ContractValue):
    """Current native recovery owner; never a replacement for the original ticket."""

    run_epoch: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    recovery_claim_id: StrictStr = Field(min_length=1, max_length=256)
    profile_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bounded_evidence(self) -> ContinuationRecoveryWriter:
        if (
            len(canonical_durable_json_bytes(self.model_dump(mode="json"), "recovery writer"))
            > CONTINUATION_MAX_RECOVERY_WRITER_BYTES
        ):
            raise ValueError("Continuation recovery writer exceeds reserved capacity.")
        return self


class ContinuationRecord(ContractValue):
    """One atomic aggregate for ticket, latch, consumption and retirement."""

    schema_version: Literal[1] = CONTINUATION_SCHEMA_VERSION
    namespace: ContinuationNamespace
    preparation: ContinuationPreparation
    ticket: ContinuationTicket
    latch: ContinuationLatch | None = None
    consumption: ContinuationConsumption | None = None
    retirement: ContinuationRetirement | None = None
    released_retirement: ContinuationReleasedExecution | None = None
    retirement_acknowledged: StrictBool = False
    # Preserve the canonical bytes of pre-external-wait receipts. Their history
    # and foreign acknowledgements commit the record without this optional field.
    recovery_writer: ContinuationRecoveryWriter | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    events: tuple[ContinuationEvent, ...] = Field(default=(), max_length=CONTINUATION_MAX_EVENTS)
    services: tuple[ContinuationServiceReference, ...] = Field(
        default=(), max_length=CONTINUATION_MAX_SERVICES
    )

    @model_validator(mode="after")
    def validate_aggregate(self) -> ContinuationRecord:
        if self.recovery_writer is not None and (
            self.ticket.purpose != "external-event-v1"
            or self.recovery_writer.run_epoch <= self.ticket.writer_generation
        ):
            raise ValueError("Continuation recovery writer conflicts with its origin.")
        keys = {item.key for item in self.services}
        if len(keys) != len(self.services) or any(
            item.generation != index for index, item in enumerate(self.services, 1)
        ):
            raise ValueError("Temporary service history is incomplete or duplicated.")
        active: list[int] = []
        prepared = False
        for item in self.services:
            if item.parent_generation is not None and item.parent_generation >= item.generation:
                raise ValueError("Temporary service parent is not retained.")
            if item.state in {"reserved", "admitted"}:
                if item.parent_generation != (active[-1] if active else None):
                    raise ValueError("Temporary services do not form one active stack.")
                active.append(item.generation)
            if item.state == "prepared":
                if prepared or item.generation != len(self.services):
                    raise ValueError("Only one next service may reserve preparation capacity.")
                if item.parent_generation != (active[-1] if active else None):
                    raise ValueError("Temporary preparation has a different service parent.")
                prepared = True
        if bool(active) != (self.ticket.state == "SERVICING"):
            raise ValueError("Temporary service ownership conflicts with ticket state.")
        if active and (self.consumption is not None or self.retirement is not None):
            raise ValueError("Temporary service must settle before final continuation.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "continuation record",
                )
            )
            > MAX_ENVELOPE_BYTES
        ):
            raise ValueError("Continuation record exceeds the durable envelope bound.")
        if self.ticket.namespace != self.namespace:
            raise ValueError("Continuation aggregate namespace mismatch.")
        require_ticket_identity(self.preparation.intent, self.ticket)
        if self.latch is not None:
            if self.latch.ticket.namespace != self.namespace:
                raise ValueError("Continuation latch namespace mismatch.")
            require_ticket_identity(self.ticket, self.latch.ticket)
        if self.consumption is not None:
            if self.latch is None:
                raise ValueError("Continuation consumption lacks its exact latch.")
            require_ticket_identity(self.ticket, self.consumption.ticket)
            require_latch_identity(self.latch, self.consumption.latch)
            if self.consumption.receipt_stage == "prepared" and self.ticket.state != "WAITING":
                raise ValueError("Prepared continuation must still be waiting.")
            if self.consumption.receipt_stage == "admitted" and self.ticket.state != "CONSUMED":
                raise ValueError("Admitted continuation must have CONSUMED state.")
            if self.consumption.receipt_stage == "excluded" and self.ticket.state != "RETIRED":
                raise ValueError("Excluded continuation must have RETIRED state.")
        if self.retirement is not None:
            if self.ticket.state != "RETIRED":
                raise ValueError("Retired continuation must have RETIRED state.")
            require_ticket_identity(self.ticket, self.retirement.ticket)
            if self.consumption is not None and self.consumption.receipt_stage != "excluded":
                raise ValueError("A continuation cannot be consumed and retired together.")
        if self.released_retirement is not None and self.retirement is None:
            raise ValueError("Released retirement evidence requires a terminal retirement.")
        if (
            self.released_retirement is not None
            and self.released_retirement.permit_operation is None
            and (
                self.ticket.purpose != "external-event-v1"
                or self.ticket.execution_admission_sha256 is not None
                or self.ticket.collaboration_wait_sha256 is not None
            )
        ):
            raise ValueError("Ordinary release evidence cannot authorize a participant wait.")
        if self.retirement_acknowledged and self.released_retirement is None:
            raise ValueError("Retirement acknowledgement requires released execution evidence.")
        if self.ticket.state == "CONSUMED" and self.consumption is None:
            raise ValueError("Consumed continuation lacks its receipt.")
        if self.ticket.state == "RETIRED" and self.retirement is None:
            raise ValueError("Retired continuation lacks its receipt.")
        return self


def continuation_operation_key(ticket: ContinuationTicket) -> str:
    return continuation_operation_key_for_registration(
        session_id=ticket.session_id,
        session_instance_id=ticket.session_instance_id,
        namespace_id=ticket.namespace.namespace_id,
        registration_key=ticket.registration_key,
    )


def continuation_operation_key_for_registration(
    *,
    session_id: str,
    session_instance_id: str,
    namespace_id: str,
    registration_key: str,
) -> str:
    material = {
        "schema": CONTINUATION_SCHEMA_VERSION,
        "session_id": session_id,
        "session_instance_id": session_instance_id,
        "namespace_id": namespace_id,
        "registration_key": registration_key,
    }
    digest = sha256(
        canonical_durable_json_bytes(material, "continuation operation key")
    ).hexdigest()
    return CONTINUATION_OPERATION_PREFIX + digest


def continuation_namespace_id(session_id: str, session_instance_id: str, owner: OwnerRef) -> str:
    return (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(
                {
                    "schema": "cayu.session-continuation-namespace.v1",
                    "session_id": session_id,
                    "session_instance_id": session_instance_id,
                    "owner": owner.model_dump(mode="json"),
                },
                "continuation namespace",
            )
        ).hexdigest()
    )


def continuation_digest(value: ContractValue) -> str:
    return sha256(contract_bytes(value, redactor=SecretRedactor())).hexdigest()


def continuation_admission_digest(command: Any) -> str:
    """Digest every typed admission field used by a continuation handoff."""

    from cayu.sessions._invocation_lifecycle import (
        invocation_admission_command_sha256,
    )

    return invocation_admission_command_sha256(command)


def continuation_admission_inputs(command: AdmitInvocationCommand) -> tuple[str, str, str]:
    """Derive input/profile/budget attribution from the exact receiving command."""
    from cayu.execution_profiles import (
        ExecutionProfileComponentClass,
    )

    continuation_admission_digest(command)  # Reject untyped lookalikes before serialization.
    input_digest = sha256(
        canonical_durable_json_bytes(
            [
                message.model_dump(mode="json", warnings=False)
                for message in command.interaction_source_messages
            ],
            "continuation admission input",
        )
    ).hexdigest()
    profile = command.target_active_profile.profile
    budget_components = sorted(
        (
            component.model_dump(mode="json", warnings=False)
            for component in profile.components
            if component.component_class
            in {
                ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
            }
        ),
        key=lambda item: item["component_class"],
    )
    budget_digest = sha256(
        canonical_durable_json_bytes(budget_components, "continuation budget authority")
    ).hexdigest()
    return input_digest, profile.fingerprint, budget_digest


def ticket_identity(ticket: ContinuationTicket) -> dict[str, Any]:
    """Return the immutable portion of a ticket for lifecycle CAS checks."""

    value = ticket.model_dump(mode="json")
    value.pop("state", None)
    value.pop("revision", None)
    return value


def require_ticket_identity(expected: ContinuationTicket, observed: ContinuationTicket) -> None:
    if ticket_identity(expected) != ticket_identity(observed):
        raise ContinuationConflict("Continuation ticket identity changed.")


def latch_identity(latch: ContinuationLatch) -> dict[str, Any]:
    """Return latch evidence with mutable ticket revision excluded."""

    value = latch.model_dump(mode="json")
    value["ticket"] = ticket_identity(latch.ticket)
    return value


def require_latch_identity(expected: ContinuationLatch, observed: ContinuationLatch) -> None:
    if latch_identity(expected) != latch_identity(observed):
        raise ContinuationConflict("Continuation latch evidence changed.")


def require_writer_generation(
    ticket: ContinuationTicket,
    current_run_epoch: int,
    *,
    allow_post_admission: bool = False,
    allow_released_next_generation: bool = False,
) -> None:
    """Fence a ticket to its owning session writer generation."""

    allowed = {ticket.writer_generation}
    if allow_post_admission or allow_released_next_generation:
        allowed.add(ticket.writer_generation + 1)
    if current_run_epoch not in allowed:
        raise ContinuationConflict("Continuation ticket belongs to a stale writer generation.")


def continuation_writer_frontier(record: ContinuationRecord) -> tuple[int, bool]:
    """Return the authenticated writer and whether its release is already included."""
    generation = (
        record.ticket.writer_generation
        if record.recovery_writer is None
        else record.recovery_writer.run_epoch
    )
    already_released = False
    for service in record.services:
        if service.mode != "same_session":
            continue
        if service.state == "returned":
            assert service.returned_writer_generation is not None
            generation = service.returned_writer_generation
            already_released = True
        elif service.state == "admitted":
            generation = service.expected_run_epoch + 1
            already_released = False
    return generation, already_released


def require_record_writer_generation(
    record: ContinuationRecord,
    current_run_epoch: int,
    *,
    allow_post_admission: bool = False,
    allow_released_next_generation: bool = False,
) -> None:
    """Use receiving-owner service succession without rewriting the original ticket."""
    generation, already_released = continuation_writer_frontier(record)
    allowed = {generation}
    if allow_post_admission or (allow_released_next_generation and not already_released):
        allowed.add(generation + 1)
    if current_run_epoch not in allowed:
        raise ContinuationConflict(
            "Continuation belongs to a stale writer generation without receiving-owner succession."
        )


def record_from_json(raw: object) -> ContinuationRecord:
    return ContinuationRecord.model_validate(raw)


__all__ = [
    "CONTINUATION_NAMESPACE_KEY",
    "CONTINUATION_OPERATION_PREFIX",
    "ContinuationConflict",
    "ContinuationConsumption",
    "ContinuationLatch",
    "ContinuationLatchReceiver",
    "ContinuationNamespace",
    "ContinuationPreparation",
    "ContinuationRecord",
    "ContinuationRetirement",
    "ContinuationService",
    "ContinuationTicket",
    "ContinuationUnavailable",
    "ContinuationWait",
    "continuation_admission_digest",
    "continuation_digest",
    "continuation_namespace_id",
    "continuation_operation_key",
    "continuation_operation_key_for_registration",
    "continuation_registration_operation",
    "latch_identity",
    "record_from_json",
    "require_latch_identity",
    "require_ticket_identity",
    "require_writer_generation",
]
