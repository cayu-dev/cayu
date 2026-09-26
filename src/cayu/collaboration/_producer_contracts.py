"""Immutable output-registration proposals; construction grants no launch rights.

Admission remains owned by RequestReceivingOwner. These values bind subsequent
production and finite delivery to that exact admission, rather than replacing it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from cayu.collaboration._contracts import (
    CollaborationContractError,
    ContractValue,
    Generation,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration._producer_bounds import (
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_DESTINATIONS,
    MAX_OUTPUT_PROGRESS,
)
from cayu.collaboration.exports import SessionExportReceipt, SessionExportRequest
from cayu.collaboration.participants import ParticipantRef, VersionOne
from cayu.collaboration.peer_content import (
    PeerContentAppendRequest,
    PeerContentReceipt,
    PeerDeliveryAttemptKey,
)
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget, NativeCommitment
from cayu.collaboration.requests import Counter, Millis, RequestAdmissionCommand


class ProducerOutputLimits(ContractValue):
    """Ceilings reserved before native execution; actual allocation is store-owned."""

    output_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_OUTPUT_BYTES)]
    progress_occurrences: Annotated[StrictInt, Field(ge=0, le=MAX_OUTPUT_PROGRESS)]
    destinations: Annotated[StrictInt, Field(ge=1, le=MAX_OUTPUT_DESTINATIONS)]
    deadline_at_ms: Millis


class ProducerDeliveryDestination(ContractValue):
    """One finite intended delivery, not a queue entry or append authorization."""

    operation: OperationRef
    recipient: ParticipantRef
    attempt: PeerDeliveryAttemptKey
    projector: ObjectRef
    validator: ObjectRef
    disclosure_policy: ObjectRef
    mandate: ObjectRef

    @model_validator(mode="after")
    def exact_destination(self) -> ProducerDeliveryDestination:
        key = self.attempt.append_key
        if (
            self.operation.application_scope != self.recipient.owner.application_scope
            or key.collaboration_namespace != self.operation.namespace_incarnation
            or key.collaboration_generation != self.operation.generation
            or key.consumer_id != self.recipient.participant_id
            or key.consumer_participant_incarnation != self.recipient.incarnation
            or any(
                ref.revision is None
                for ref in (self.projector, self.validator, self.disclosure_policy, self.mandate)
            )
            # Qualified deterministic export owns project() and validate() under
            # one registered version. A separate opaque validator is not qualified.
            or self.validator != self.projector
            or (
                key.creation_target is not None
                and key.creation_target.permit.intent.request.participant != self.recipient
            )
        ):
            raise CollaborationContractError("Output destination identity conflicts.")
        return self


class ProducerRegistrationEvent(ContractValue):
    """Content-free durable registration event, separate from request outcome."""

    mode: Literal["producer_output_registered"] = "producer_output_registered"
    operation: OperationRef
    registration: OperationRef
    sequence: Generation
    registered_at_ms: Millis


class ProducerRequestIndex(ContractValue):
    """Unique request-to-production attachment, not a launch authorization."""

    mode: Literal["producer_request_index"] = "producer_request_index"
    operation: OperationRef
    registration: OperationRef
    command_commitment: NativeCommitment


class ProducerOutputProposal(ContractValue):
    """Caller choices for inert preparation; no caller-supplied native authority."""

    schema_version: VersionOne = 1
    operation: OperationRef
    admission: RequestAdmissionCommand
    binding_incarnation: Identifier
    limits: ProducerOutputLimits
    destinations: tuple[ProducerDeliveryDestination, ...] = Field(
        min_length=1, max_length=MAX_OUTPUT_DESTINATIONS
    )


class ProducerOutputRegistration(ContractValue):
    """Complete proposed binding for one producer and one admitted request.

    The registered owner must re-read admission, validate current launch authority,
    authenticate execution_commitment from native preparation, and reserve durable
    capacity. Neither this object nor an equal durable value authenticates callers.
    """

    schema_version: VersionOne = 1
    mode: Literal["producer_output_registration"] = "producer_output_registration"
    operation: OperationRef
    admission: RequestAdmissionCommand
    initiator: InitiatorBinding
    receiver: ObjectRef
    binding_incarnation: Identifier
    execution_key: Identifier
    execution_commitment: NativeCommitment
    publisher_generation: Generation
    disposition: Literal["stop"]
    limits: ProducerOutputLimits
    destinations: tuple[ProducerDeliveryDestination, ...] = Field(
        min_length=1, max_length=MAX_OUTPUT_DESTINATIONS
    )

    @model_validator(mode="after")
    def exact_admitted_binding(self) -> ProducerOutputRegistration:
        admitted = self.admission
        prepared = admitted.prepared
        if (
            prepared is None
            or admitted.decision != "fresh"
            or not isinstance(prepared.target, FreshRecipientAdmissionTarget)
            or prepared.target.resources
        ):
            raise CollaborationContractError(
                "Output requires resource-free prepared FRESH admission."
            )
        request = admitted.expected.intent.request
        namespace = (
            self.operation.application_scope,
            self.operation.namespace_incarnation,
            self.operation.generation,
        )
        operations = (
            admitted.operation,
            admitted.expected.operation,
            *(item.operation for item in self.destinations),
        )
        if (
            any(
                (item.application_scope, item.namespace_incarnation, item.generation) != namespace
                for item in operations
            )
            or self.operation in operations
            or self.initiator != admitted.initiator
            or self.receiver.owner != prepared.recipient.owner
            or self.receiver.revision is None
            or self.publisher_generation != admitted.generation
            or self.disposition != request.cancellation
            or self.limits.deadline_at_ms > admitted.expected.intent.selection.expires_at_ms
            or len(self.destinations) > self.limits.destinations
        ):
            raise CollaborationContractError("Output registration conflicts with admission.")
        identities = []
        for destination in self.destinations:
            if (
                destination.operation in (admitted.operation, admitted.expected.operation)
                or destination.attempt.deadline_at_ms > self.limits.deadline_at_ms
                or destination.projector != request.output_contract
                or destination.disclosure_policy != request.disclosure_policy
            ):
                raise CollaborationContractError("Output delivery conflicts with its contract.")
            identities.append(destination.attempt.append_key.model_dump_json())
        if len(set(identities)) != len(identities) or len(
            {item.operation for item in self.destinations}
        ) != len(self.destinations):
            raise CollaborationContractError("Output deliveries must have distinct identities.")
        return self


class ProducerNativeOutput(ContractValue):
    """Native-owner evidence; reconstruction alone supplies no disclosure authority."""

    registration: ProducerOutputRegistration
    stage_id: Identifier
    publication_id: Identifier
    publication_commitment: NativeCommitment
    interaction_id: Identifier
    run_epoch: Generation
    source_indices: tuple[Annotated[StrictInt, Field(ge=0, lt=2**53)], ...] = Field(max_length=16)
    source_commitment: NativeCommitment | None
    disposition: Literal["answer", "oversized", "unsupported", "empty"]

    @model_validator(mode="after")
    def exact_selection(self):
        if tuple(sorted(set(self.source_indices))) != self.source_indices:
            raise CollaborationContractError("Producer output selection is not canonical.")
        if self.disposition == "answer":
            if not self.source_indices or self.source_commitment is None:
                raise CollaborationContractError("Producer answer lacks exact output evidence.")
        elif self.source_indices or self.source_commitment is not None:
            raise CollaborationContractError("Producer failure cannot claim an answer selection.")
        return self


class ProducerNativeFailure(ContractValue):
    """Content-free native failure receipt, not a model output or quiescence proof."""

    registration: ProducerOutputRegistration
    interaction_id: Identifier
    run_epoch: Generation
    event_id: Identifier
    settlement_commitment: NativeCommitment
    disposition: Literal["failed", "stopped"] = "failed"
    source_indices: tuple[()] = ()
    source_commitment: None = None


def prepare_native_production(value, *, redactor):
    from cayu.collaboration._preparation import prepare_contract

    failure = type(value) is ProducerNativeFailure or (
        type(value) is dict
        and type(value.get("disposition")) is str
        and value["disposition"] in ("failed", "stopped")
    )
    return prepare_contract(
        ProducerNativeFailure if failure else ProducerNativeOutput, value, redactor=redactor
    )


class ProducerCompletionRecord(ContractValue):
    """Accepted native production, separate from election, delivery and quiescence."""

    mode: Literal["producer_completion"] = "producer_completion"
    operation: OperationRef
    output: ProducerNativeOutput | ProducerNativeFailure
    native_commitment: NativeCommitment
    sequence: Generation
    recorded_at_ms: Millis


class ProducerExportRecord(ContractValue):
    """Frozen per-destination export intent, never a delivery or disclosure grant."""

    mode: Literal["producer_export"] = "producer_export"
    operation: OperationRef
    registration: OperationRef
    completion: OperationRef
    destination: OperationRef
    request: SessionExportRequest
    initiator: InitiatorBinding
    sequence: Generation
    state: Literal["prepared", "published"] = "prepared"
    publication: OperationRef | None = None
    receipt_commitment: NativeCommitment | None = None
    output_commitment: NativeCommitment | None = None
    rejection: OperationRef | None = None

    @model_validator(mode="after")
    def exact_publication(self):
        if self.rejection is not None and self.state != "prepared":
            raise CollaborationContractError("Rejected output cannot be published.")
        if any(
            (value is not None) != (self.state == "published")
            for value in (self.publication, self.receipt_commitment, self.output_commitment)
        ):
            raise CollaborationContractError("Producer export publication is incomplete.")
        return self


class ProducerDeliveryRecord(ContractValue):
    """Source obligation around one existing peer queue; not exposure or quiescence."""

    mode: Literal["producer_delivery"] = "producer_delivery"
    operation: OperationRef
    registration: OperationRef
    destination: OperationRef
    export: OperationRef
    outcome: OperationRef
    source_receipt: SessionExportReceipt
    append: PeerContentAppendRequest
    sequence: Generation
    receipt: PeerContentReceipt | None = None
    acceptance: OperationRef | None = None

    @model_validator(mode="after")
    def exact_acceptance(self):
        if (self.receipt is None) != (self.acceptance is None):
            raise CollaborationContractError("Producer delivery acceptance is incomplete.")
        receipt = self.receipt
        request = self.append
        if receipt is not None and (
            receipt.operation_key != request.operation_key
            or receipt.append_key != request.append_key
            or receipt.attempt_generation != request.attempt_key.attempt_generation
            or receipt.status not in {"appended", "excluded"}
            or receipt.disclosure != "available"
            or receipt.replayed
            or (receipt.occurrence is not None and receipt.occurrence != request.occurrence)
            or (
                request.append_key.creation_target is None
                and receipt.status == "appended"
                and (receipt.target_session_id, receipt.target_session_instance_id)
                != (
                    request.append_key.target_session_id,
                    request.append_key.target_session_instance_id,
                )
            )
        ):
            raise CollaborationContractError("Producer delivery receiving evidence conflicts.")
        return self


class ProducerDeliveryAccepted(ContractValue):
    mode: Literal["producer_delivery_accepted"] = "producer_delivery_accepted"
    operation: OperationRef
    delivery: OperationRef
    receipt_commitment: NativeCommitment
    sequence: Generation


class ProducerDeliveryIndex(ContractValue):
    """Exact export-to-delivery address for a registered receiving reader."""

    mode: Literal["producer_delivery_index"] = "producer_delivery_index"
    operation: OperationRef
    registration: OperationRef
    destination: OperationRef
    delivery: OperationRef
    export_receipt_commitment: NativeCommitment


class ProducerExportPublished(ContractValue):
    mode: Literal["producer_export_published"] = "producer_export_published"
    operation: OperationRef
    export: OperationRef
    receipt_commitment: NativeCommitment
    sequence: Generation


class ProducerLaunchDecision(ContractValue):
    """Source-side launch election; not native admission or quiescence evidence."""

    mode: Literal["producer_launch_decision"] = "producer_launch_decision"
    operation: OperationRef
    registration: OperationRef
    command_commitment: NativeCommitment
    sequence: Generation
    request_revision: Generation
    elected_at_ms: Millis
    authority_expires_at_ms: Millis


class ProducerNativeExclusion(ContractValue):
    """Native exact no-start receipt; data requires its registered store reader."""

    registration: OperationRef
    control_operation: OperationRef
    control_commitment: NativeCommitment
    attachment_commitment: NativeCommitment
    session_id: Identifier
    session_instance_id: Identifier
    execution_commitment: NativeCommitment
    state: Literal["excluded"] = "excluded"


class ProducerCleanupRecord(ContractValue):
    mode: Literal["producer_cleanup"] = "producer_cleanup"
    operation: OperationRef
    registration: OperationRef
    sequence: Generation
    exclusion: ProducerNativeExclusion


class ProducerDestinationSettlement(ContractValue):
    destination: OperationRef
    kind: Literal["delivery", "export_retirement", "destination_exclusion"] = "delivery"
    delivery: OperationRef | None = None
    receipt_commitment: NativeCommitment | None = None
    export_settlement_commitment: NativeCommitment
    export_state: Literal["excluded", "retired", "released"]

    @model_validator(mode="after")
    def exact_kind(self):
        if any(
            (value is not None) != (self.kind == "delivery")
            for value in (self.delivery, self.receipt_commitment)
        ):
            raise ValueError(
                "Producer settlement requires exact delivery or export-retirement evidence."
            )
        if self.kind == "delivery" and self.export_state == "excluded":
            raise ValueError("A delivery requires a previously published export.")
        return self


class ProducerSettlementEvidence(ContractValue):
    """Historical owner facts, not execution or erasure authority."""

    registration: OperationRef
    registration_commitment: NativeCommitment
    completion: OperationRef
    completion_commitment: NativeCommitment
    terminal: OperationRef
    terminal_kind: Literal["outcome", "closure"]
    native_release_commitment: NativeCommitment
    native_release_run_epoch: Generation
    budget_settlement_commitment: NativeCommitment
    destinations: tuple[ProducerDestinationSettlement, ...] = Field(
        max_length=MAX_OUTPUT_DESTINATIONS
    )


class ProducerAdmittedCleanup(ContractValue):
    """Accepted cleanup evidence; not final release or permit settlement."""

    mode: Literal["producer_admitted_cleanup"] = "producer_admitted_cleanup"
    operation: OperationRef
    registration: OperationRef
    sequence: Generation
    evidence: ProducerSettlementEvidence

    @model_validator(mode="after")
    def exact_evidence(self):
        if self.registration != self.evidence.registration:
            raise ValueError("Producer cleanup evidence belongs to another registration.")
        return self


class ProducerOutputRecord(ContractValue):
    """Initial durable responsibility; no native launch has yet been accepted."""

    mode: Literal["producer_output_record"] = "producer_output_record"
    command: ProducerOutputRegistration
    event: ProducerRegistrationEvent
    permit: PermitCommand
    state: Literal["registered", "launch_claimed", "excluded"] = "registered"
    launch: ProducerLaunchDecision | None = None
    cleanup: ProducerCleanupRecord | ProducerAdmittedCleanup | None = None
    cleanup_ack: OperationRef | None = None
    completion: OperationRef | None = None
    exports: tuple[OperationRef, ...] = Field(default=(), max_length=MAX_OUTPUT_DESTINATIONS)
    deliveries: tuple[OperationRef, ...] = Field(default=(), max_length=MAX_OUTPUT_DESTINATIONS)
    exclusions: tuple[OperationRef, ...] = Field(default=(), max_length=MAX_OUTPUT_DESTINATIONS)
    reserved_operations: Counter
    reserved_events: Counter
    reserved_bytes: Counter

    @model_validator(mode="after")
    def exact_registration(self) -> ProducerOutputRecord:
        if (
            self.event.registration != self.command.operation
            or self.permit.intent.request.source_operation != self.command.operation
            or (self.state == "registered" and self.launch is not None)
            or (self.state == "launch_claimed" and self.launch is None)
            or (self.state == "excluded") != isinstance(self.cleanup, ProducerCleanupRecord)
            or (self.cleanup_ack is not None and self.cleanup is None)
            or (
                isinstance(self.cleanup, ProducerAdmittedCleanup)
                and (
                    self.state != "launch_claimed"
                    or self.completion != self.cleanup.evidence.completion
                )
            )
            or (self.completion is not None and self.state != "launch_claimed")
            or (self.exports and self.completion is None)
            or len(set(self.exports)) != len(self.exports)
            or (self.deliveries and not self.exports)
            or len(set(self.deliveries)) != len(self.deliveries)
            or len(set(self.exclusions)) != len(self.exclusions)
            or (
                self.cleanup is not None
                and (
                    self.cleanup.registration != self.command.operation
                    or (
                        isinstance(self.cleanup, ProducerCleanupRecord)
                        and self.cleanup.exclusion.registration != self.command.operation
                    )
                    or (
                        self.cleanup_ack is not None
                        and (
                            self.reserved_operations or self.reserved_events or self.reserved_bytes
                        )
                    )
                )
            )
            or (
                self.cleanup_ack is None
                and not (self.reserved_operations and self.reserved_events and self.reserved_bytes)
            )
            or (
                self.launch is not None
                and (
                    self.launch.registration != self.command.operation
                    or self.launch.sequence <= self.event.sequence
                    or self.launch.authority_expires_at_ms > self.command.limits.deadline_at_ms
                    or self.launch.authority_expires_at_ms <= self.launch.elected_at_ms
                )
            )
        ):
            raise CollaborationContractError("Producer registration event conflicts.")
        return self
