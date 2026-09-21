"""Durable session continuation tickets and early-result latches.

This module owns the session side of a wait.  It deliberately does not elect
finite predicates or subscribe to another owner; those responsibilities belong
to the collaboration wait layer.  The values below are immutable evidence and
must be authenticated by the SessionStore before they are accepted.
"""

from __future__ import annotations

from contextlib import suppress
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
    from cayu.runtime._invocation_lifecycle import AdmitInvocationCommand, InvocationMutationResult
    from cayu.sessions.base import SessionStore

CONTINUATION_OPERATION_PREFIX = "session-continuation:"
CONTINUATION_NAMESPACE_KEY = CONTINUATION_OPERATION_PREFIX + "namespace"
CONTINUATION_SCHEMA_VERSION = 1
CONTINUATION_MAX_TARGETS = 64
CONTINUATION_MAX_DIGEST_BYTES = 256
CONTINUATION_MAX_LATCH_EVIDENCE_BYTES = 8192
CONTINUATION_MAX_CONSUMPTION_EVIDENCE_BYTES = 12288
CONTINUATION_MAX_RETIREMENT_EVIDENCE_BYTES = 4096
CONTINUATION_MAX_EVENTS = 8
CONTINUATION_MAX_EVENT_BYTES = 512


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
    deadline: StrictStr
    failure_policy: StrictStr = Field(min_length=1, max_length=128)
    service_policy: StrictStr = Field(min_length=1, max_length=128)
    wait_edge_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    purpose: StrictStr = Field(min_length=1, max_length=256)

    unordered_fields = frozenset({"targets"})


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
    deadline: StrictStr
    failure_policy: StrictStr = Field(min_length=1, max_length=128)
    service_policy: StrictStr = Field(min_length=1, max_length=128)
    wait_edge_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    interaction_id: StrictStr
    writer_generation: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    purpose: StrictStr = Field(min_length=1, max_length=256)
    state: Literal["ARMING", "WAITING", "SERVICING", "CONSUMED", "RETIRED"]
    revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)

    unordered_fields = frozenset({"targets"})

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, value: str) -> str:
        return _aware_time(datetime.fromisoformat(value), "deadline").isoformat()

    @field_validator("session_id", "session_instance_id", "registration_key", "interaction_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value or len(value.encode("utf-8")) > 512:
            raise ValueError(f"{info.field_name} is too large.")
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> ContinuationTicket:
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
        elif self.ticket.state not in {"WAITING", "SERVICING", "CONSUMED"}:
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


class ContinuationRecord(ContractValue):
    """One atomic aggregate for ticket, latch, consumption and retirement."""

    schema_version: Literal[1] = CONTINUATION_SCHEMA_VERSION
    namespace: ContinuationNamespace
    preparation: ContinuationPreparation
    ticket: ContinuationTicket
    latch: ContinuationLatch | None = None
    consumption: ContinuationConsumption | None = None
    retirement: ContinuationRetirement | None = None
    events: tuple[ContinuationEvent, ...] = Field(default=(), max_length=CONTINUATION_MAX_EVENTS)

    @model_validator(mode="after")
    def validate_aggregate(self) -> ContinuationRecord:
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
            if self.consumption.receipt_stage == "prepared" and self.ticket.state not in {
                "WAITING",
                "SERVICING",
            }:
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

    from cayu.runtime._invocation_lifecycle import invocation_admission_command_sha256

    return invocation_admission_command_sha256(command)


def continuation_admission_inputs(command: AdmitInvocationCommand) -> tuple[str, str, str]:
    """Derive input/profile/budget attribution from the exact receiving command."""
    from cayu.runtime.execution_profiles import ExecutionProfileComponentClass

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


def record_from_json(raw: object) -> ContinuationRecord:
    return ContinuationRecord.model_validate(raw)


def require_operation_record_owner(key: str, record: object) -> None:
    """Validate the reserved continuation key without granting read authority."""

    if not key.startswith(CONTINUATION_OPERATION_PREFIX):
        return
    from cayu.runtime._session_continuation_scope import require_publication

    require_publication(key)
    if type(record) is not dict:
        raise ContinuationConflict("Continuation operation record is not an object.")
    if key == CONTINUATION_NAMESPACE_KEY:
        ContinuationNamespace.model_validate(record)
        return
    parsed = record_from_json(record)
    if continuation_operation_key(parsed.ticket) != key:
        raise ContinuationConflict("Continuation operation key is not content-bound.")


async def admit_continuation(
    store: SessionStore,
    consumption: ContinuationConsumption,
    command: AdmitInvocationCommand,
) -> tuple[InvocationMutationResult, ContinuationRecord]:
    """Admit exactly one retained continuation through the typed lifecycle command.

    The prepared receipt is committed before admission.  If admission fails or
    its acknowledgement is lost, the prepared responsibility remains durable
    and a retry can reconcile it without creating a second invocation.
    """

    if consumption.receipt_stage != "prepared":
        raise ContinuationConflict("Continuation admission requires a prepared receipt.")
    if (
        getattr(command, "session_id", None) != consumption.ticket.session_id
        or getattr(command, "expected_session_instance_id", None)
        != consumption.ticket.session_instance_id
    ):
        raise ContinuationConflict("Invocation admission belongs to another session authority.")
    if continuation_admission_digest(command) != consumption.admission_command_digest:
        raise ContinuationConflict("Invocation admission does not match its retained receipt.")
    if getattr(command, "expected_run_epoch", None) != consumption.admission_expected_run_epoch:
        raise ContinuationConflict(
            "Invocation admission epoch does not match its retained receipt."
        )
    prepared = await store.consume_continuation(consumption)
    retained = prepared.consumption
    if retained is None:
        raise ContinuationConflict("Prepared continuation readback changed before admission.")
    comparable = retained.model_copy(
        update={
            "ticket": consumption.ticket,
            "receipt_stage": "prepared",
            "admission_claimed": False,
            "admission_claim_id": None,
        }
    )
    if comparable != consumption:
        raise ContinuationConflict("Prepared continuation readback changed before admission.")
    if retained.receipt_stage == "admitted":
        from cayu.runtime._invocation_lifecycle import reconcile_invocation_admission_from_state

        session = await store.load(consumption.ticket.session_id)
        checkpoint = await store.load_checkpoint(consumption.ticket.session_id)
        result = (
            None
            if session is None
            else reconcile_invocation_admission_from_state(
                session,
                checkpoint,
                session_id=consumption.ticket.session_id,
                session_instance_id=consumption.ticket.session_instance_id,
                expected_run_epoch=consumption.admission_expected_run_epoch,
                command_sha256=consumption.admission_command_digest,
                profile_sha256=consumption.profile_digest,
            )
        )
        if result is None:
            raise ContinuationUnavailable("Continuation admission is pending reconciliation.")
        return result, prepared
    if retained.receipt_stage != "prepared":
        raise ContinuationConflict("Continuation responsibility is no longer admissible.")
    from cayu.runtime._session_continuation_scope import admission_claim_scope

    claimed, _ = await store._claim_continuation_admission(consumption)
    claim_consumption = claimed.consumption
    if claim_consumption is None or not claim_consumption.admission_claimed:
        raise ContinuationConflict("Continuation admission claim was not retained.")
    # Exact retries share the claim. The lifecycle transaction verifies its
    # ID atomically; release fences every still-pending dispatch with that ID.
    admitted = claim_consumption.model_copy(
        update={"receipt_stage": "admitted", "admission_claimed": True}
    )
    try:
        with admission_claim_scope(claim_consumption):
            result = await store.apply_invocation_lifecycle_command(command)
    except Exception:
        # Failed reconciliation retains the claim and the original error.
        with suppress(Exception):
            # The transaction distinguishes exact commitment, positive
            # supersession, and a still-uncommitted claim. No pre-read grants
            # exclusion authority or permits a late dispatch after release.
            await store._release_continuation_admission_claim(claim_consumption)
        raise
    settled = await store._finalize_continuation_admission(admitted, command)
    return result, settled


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
    "require_operation_record_owner",
    "require_ticket_identity",
    "require_writer_generation",
]
