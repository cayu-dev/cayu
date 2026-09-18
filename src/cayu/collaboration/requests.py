"""Durable request-owner values; acceptance never grants execution authority."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_NODES,
    ContractValue,
    ExpectedOperation,
    Generation,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
    snapshot_input,
)
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration.exports import SessionExportReceipt, SessionExportRef
from cayu.collaboration.mandates import MandateResolution
from cayu.collaboration.participants import (
    CollaborationLimits,
    Counter,
    ParticipantRef,
    ParticipantSnapshot,
    VersionOne,
)

Millis = Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]
MAX_CONTROL_INITIATOR_BYTES = 8 * 1024


class RequestAlias(ContractValue):
    """Original addressing intent, resolved only during first acceptance."""

    owner: OwnerRef
    alias: Identifier


class CollaborationRequest(ContractValue):
    operation: OperationRef
    kind: Literal["question", "contribution"]
    sender: ParticipantRef
    target: ParticipantRef | RequestAlias
    content: str = Field(max_length=8192)
    inputs: tuple[ObjectRef, ...] = Field(max_length=32)
    context_hint: ObjectRef | None
    delivery_contract: ObjectRef
    output_contract: ObjectRef | None
    independence_policy: ObjectRef
    disclosure_policy: ObjectRef
    ttl_ms: Millis
    cancellation: Literal["detach", "stop"]

    @model_validator(mode="after")
    def exact_intent(self) -> CollaborationRequest:
        if self.sender.owner != self.target.owner:
            raise ValueError("Request participants must belong to the same collaboration owner.")
        if self.operation.application_scope != self.sender.owner.application_scope:
            raise ValueError("Request namespace belongs to another scope.")
        if (self.kind == "question") != (self.output_contract is not None):
            raise ValueError("Only questions require an output contract.")
        refs = (
            *self.inputs,
            self.delivery_contract,
            self.independence_policy,
            self.disclosure_policy,
            *((self.context_hint,) if self.context_hint is not None else ()),
            *((self.output_contract,) if self.output_contract is not None else ()),
        )
        if any(ref.revision is None for ref in refs):
            raise ValueError("Request inputs and contracts require pinned revisions.")
        return self


class RequestRef(ContractValue):
    owner: OwnerRef
    request_id: Identifier
    incarnation: Identifier


class RequestSelection(ContractValue):
    reference: RequestRef
    sender: ParticipantSnapshot
    recipient: ParticipantSnapshot
    accepted_at_ms: Millis
    expires_at_ms: Millis


class RequestIntent(ContractValue):
    request: CollaborationRequest
    selection: RequestSelection
    limits: CollaborationLimits
    authority: MandateResolution


class RequestCommand(ExpectedOperation[RequestIntent]):
    kind: Literal["question", "contribution"]
    mode: Literal["request"] = "request"
    schema_version: VersionOne = 1
    receipt_stage: Literal["accepted"] = "accepted"

    @model_validator(mode="after")
    def exact_selection(self) -> RequestCommand:
        original = self.intent.request
        chosen = self.intent.selection
        authority = self.intent.authority
        leaf = authority.chain.entries[-1]
        if (
            self.operation != original.operation
            or self.kind != original.kind
            or self.source != self.destination
            or self.destination != chosen.reference.owner
            or original.sender != chosen.sender.reference
            or original.sender.owner != self.source
            or chosen.recipient.reference.owner != self.destination
            or (
                isinstance(original.target, ParticipantRef)
                and original.target != chosen.recipient.reference
            )
            or chosen.expires_at_ms - chosen.accepted_at_ms != original.ttl_ms
            or authority.principal.issuer != self.initiator.issuer
            or authority.principal.principal != self.initiator.principal
            or leaf.reference != self.initiator.mandate
            or leaf.participant != original.sender
            or self.initiator.participant
            != ObjectRef(
                owner=original.sender.owner,
                kind="participant",
                object_id=original.sender.participant_id,
                incarnation=original.sender.incarnation,
            )
        ):
            raise ValueError("Request selection conflicts with its original authority.")
        return self


class RequestEvent(ContractValue):
    id: Identifier
    sequence: Generation
    operation: OperationRef
    request: RequestRef
    type: Literal[
        "request_accepted",
        "request_admission",
        "request_progress",
        "request_answered",
        "request_failed",
        "request_declined",
        "request_cancelled",
        "request_expired",
        "request_observation_registered",
    ]
    participants: tuple[ParticipantRef, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def scope_matches(self) -> RequestEvent:
        if (
            self.operation.application_scope != self.request.owner.application_scope
            or any(ref.owner != self.request.owner for ref in self.participants)
            or len(set(self.participants)) != len(self.participants)
        ):
            raise ValueError("Request event authority conflicts.")
        return self


class RequestReceipt(ContractValue):
    expected: RequestCommand
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> RequestReceipt:
        selected = self.expected.intent.selection
        refs = tuple(dict.fromkeys((selected.sender.reference, selected.recipient.reference)))
        if (
            self.event.operation != self.expected.operation
            or self.event.request != selected.reference
            or self.event.type != "request_accepted"
            or self.event.participants != refs
        ):
            raise ValueError("Request receipt conflicts with acceptance.")
        return self


class RequestControl(ContractValue):
    operation: OperationRef
    expected: RequestCommand
    expected_revision: Generation
    kind: Literal["cancel", "expire"]
    source_receipt: SessionExportReceipt | None = None

    @model_validator(mode="after")
    def separate_control(self) -> RequestControl:
        if (
            self.operation.application_scope != self.expected.operation.application_scope
            or self.operation.namespace_incarnation != self.expected.operation.namespace_incarnation
            or self.operation.generation != self.expected.operation.generation
            or self.operation == self.expected.operation
        ):
            raise ValueError(
                "Control requires a separate key in the accepted namespace generation."
            )
        return self


class RequestControlCommand(ExpectedOperation[RequestControl]):
    kind: Literal["cancel", "expire"]
    mode: Literal["request_control"] = "request_control"
    schema_version: VersionOne = 1
    receipt_stage: Literal["elected"] = "elected"

    @model_validator(mode="after")
    def exact_control(self) -> RequestControlCommand:
        # Bound the combined initiating identity, including JSON escaping, so
        # acceptance can reserve future controls independently of their author.
        canonical_bounded_durable_json_bytes(
            snapshot_input(self.initiator),
            "control initiator",
            max_bytes=MAX_CONTROL_INITIATOR_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
        if (
            self.operation != self.intent.operation
            or self.kind != self.intent.kind
            or self.source != self.destination
            or self.destination != self.intent.expected.destination
        ):
            raise ValueError("Control command authority conflicts.")
        return self


class RequestControlReceipt(ContractValue):
    expected: RequestControlCommand
    state: Literal["cancelled", "expired"]
    revision: Generation
    elected_at_ms: Millis
    event: RequestEvent

    @model_validator(mode="after")
    def exact_election(self) -> RequestControlReceipt:
        command = self.expected
        selected = command.intent.expected.intent.selection
        if (
            self.revision != command.intent.expected_revision + 1
            or self.event.operation != command.operation
            or self.event.request != selected.reference
            or self.event.type != "request_" + self.state
            or self.event.participants
            != tuple(dict.fromkeys((selected.sender.reference, selected.recipient.reference)))
            or self.elected_at_ms < selected.accepted_at_ms
            or (self.state == "expired") != (self.elected_at_ms >= selected.expires_at_ms)
            or (command.kind == "expire" and self.state != "expired")
        ):
            raise ValueError("Control receipt conflicts with its election.")
        return self


AdmissionDecision = Literal["continue", "fork", "fresh", "defer", "decline", "clarify"]
AdmissionState = Literal[
    "undecided", "planning", "preparing", "deferred", "clarifying", "admitted", "closed"
]
RequestOutcomeKind = Literal["answered", "failed", "declined"]


class RequestAdmissionCommand(ContractValue):
    """Authenticated receiving-owner admission decision."""

    operation: OperationRef
    mode: Literal["request_admission"] = "request_admission"
    expected: RequestCommand
    expected_revision: Generation
    generation: Generation
    decision: AdmissionDecision
    proposal_commitment: Identifier | None = None
    source_export: SessionExportRef | None = None
    source_receipt: SessionExportReceipt | None = None
    evidence: tuple[ObjectRef, ...] = Field(max_length=32)
    initiator: InitiatorBinding

    @model_validator(mode="after")
    def exact_admission(self) -> RequestAdmissionCommand:
        source_required = self.decision in {"continue", "fork", "fresh"}
        if (
            self.operation.application_scope != self.expected.operation.application_scope
            or self.operation.namespace_incarnation != self.expected.operation.namespace_incarnation
            or self.operation.generation != self.expected.operation.generation
            or self.operation == self.expected.operation
            or self.generation < 1
            or (
                self.source_receipt is not None
                and self.source_receipt.expected.intent.request.ref != self.source_export
            )
            or (self.source_export is not None and self.source_receipt is None)
            or (source_required and self.source_receipt is None)
            or (
                self.source_receipt is not None
                and not source_export_matches_request(
                    self.source_receipt,
                    self.expected.intent.request,
                    self.expected.initiator,
                )
            )
        ):
            raise ValueError("Admission operation conflicts with its request.")
        return self


def source_export_matches_request(
    source: SessionExportReceipt,
    request: CollaborationRequest,
    initiator: InitiatorBinding,
) -> bool:
    """Check the complete source evidence tuple for a collaboration request.

    The source reference and output commitment are necessary but not
    sufficient: a receipt from another producer, session policy, or projector
    must not satisfy this request merely because it is valid on its own.
    """
    source_request = source.expected.intent.request
    recipient_owner = request.target.owner
    return (
        source.expected.initiator == initiator
        and source.expected.source == request.sender.owner
        and source.expected.destination == recipient_owner
        and source_request.audience == recipient_owner
        and request.output_contract is not None
        and source_request.projector == request.output_contract
        and source_request.policy == request.disclosure_policy
    )


class RequestAdmissionReceipt(ContractValue):
    command: RequestAdmissionCommand
    state: AdmissionState
    revision: Generation
    decided_at_ms: Millis
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> RequestAdmissionReceipt:
        selected = self.command.expected.intent.selection
        expected_state = {
            "continue": "admitted" if self.command.evidence else "preparing",
            "fork": "admitted" if self.command.evidence else "preparing",
            "fresh": "admitted" if self.command.evidence else "preparing",
            "defer": "deferred",
            "clarify": "clarifying",
            "decline": "closed",
        }[self.command.decision]
        if (
            self.event.operation != self.command.operation
            or self.event.request != selected.reference
            or self.event.type != "request_admission"
            or self.event.participants
            != tuple(dict.fromkeys((selected.sender.reference, selected.recipient.reference)))
            or self.state != expected_state
            or self.revision != self.command.expected_revision + 1
            or self.decided_at_ms < selected.accepted_at_ms
        ):
            raise ValueError("Admission receipt conflicts with its decision.")
        return self


class RequestProgressCommand(ContractValue):
    """One authenticated, idempotent producer progress occurrence."""

    operation: OperationRef
    mode: Literal["request_progress"] = "request_progress"
    expected: RequestCommand
    expected_revision: Generation
    admission_generation: Generation
    publisher: InitiatorBinding
    sequence: Generation
    kind: Literal["started", "prepared", "producing", "published"]
    commitment: Identifier
    source_receipt: SessionExportReceipt | None = None

    @model_validator(mode="after")
    def exact_namespace(self) -> RequestProgressCommand:
        original = self.expected.operation
        source = self.source_receipt
        if (
            self.operation.application_scope != original.application_scope
            or self.operation.namespace_incarnation != original.namespace_incarnation
            or self.operation.generation != original.generation
            or self.operation == original
            or self.expected_revision < 1
            or (
                source is not None
                and (
                    source.expected.intent.output_commitment != self.commitment
                    or source.expected.intent.request.audience
                    != self.expected.intent.selection.recipient.reference.owner
                )
            )
        ):
            raise ValueError("Progress operation conflicts with its request namespace.")
        return self


class RequestProgressReceipt(ContractValue):
    command: RequestProgressCommand
    revision: Generation
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> RequestProgressReceipt:
        selected = self.command.expected.intent.selection
        if (
            self.event.operation != self.command.operation
            or self.event.request != selected.reference
            or self.event.type != "request_progress"
            or self.event.participants
            != tuple(dict.fromkeys((selected.sender.reference, selected.recipient.reference)))
            or self.revision != self.command.expected_revision + 1
        ):
            raise ValueError("Progress receipt conflicts with its event.")
        return self


class RequestOutcomeCommand(ContractValue):
    """Authenticated answer/failure/decline election request."""

    operation: OperationRef
    mode: Literal["request_outcome"] = "request_outcome"
    expected: RequestCommand
    expected_revision: Generation
    outcome: RequestOutcomeKind
    commitment: Identifier | None = None
    source_receipt: SessionExportReceipt | None = None
    initiator: InitiatorBinding

    @model_validator(mode="after")
    def exact_outcome(self) -> RequestOutcomeCommand:
        source = self.source_receipt
        source_audience = None if source is None else source.expected.intent.request.audience
        if (
            self.operation.application_scope != self.expected.operation.application_scope
            or self.operation.namespace_incarnation != self.expected.operation.namespace_incarnation
            or self.operation.generation != self.expected.operation.generation
            or self.operation == self.expected.operation
            or self.expected_revision < 1
            or (self.outcome == "answered" and self.source_receipt is None)
            or (
                self.outcome == "answered"
                and (
                    self.commitment is None
                    or source is None
                    or source.expected.intent.output_commitment != self.commitment
                    or source_audience != self.expected.intent.selection.recipient.reference.owner
                    or not source_export_matches_request(
                        source,
                        self.expected.intent.request,
                        self.expected.initiator,
                    )
                )
            )
        ):
            raise ValueError("Outcome operation conflicts with its request.")
        return self


class RequestOutcomeReceipt(ContractValue):
    command: RequestOutcomeCommand
    revision: Generation
    elected_at_ms: Millis
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> RequestOutcomeReceipt:
        selected = self.command.expected.intent.selection
        if (
            self.event.operation != self.command.operation
            or self.event.request != selected.reference
            or self.event.type != "request_" + self.command.outcome
            or self.event.participants
            != tuple(dict.fromkeys((selected.sender.reference, selected.recipient.reference)))
            or self.revision != self.command.expected_revision + 1
            or self.elected_at_ms < selected.accepted_at_ms
        ):
            raise ValueError("Outcome receipt conflicts with its election.")
        return self


class RequestObservation(ContractValue):
    """Finite pull observation registration and its durable coverage frontier."""

    key: Identifier
    filter_commitment: Identifier
    projection_commitment: Identifier
    after_sequence: Counter
    coverage_sequence: Counter
    revision: Generation
    retention_until_ms: Millis | None = None

    @model_validator(mode="after")
    def valid_frontier(self) -> RequestObservation:
        if self.coverage_sequence < self.after_sequence:
            raise ValueError("Observation coverage cannot precede its cursor.")
        if self.retention_until_ms is not None and self.retention_until_ms < 0:
            raise ValueError("Observation retention is invalid.")
        return self


class RequestObservationReceipt(ContractValue):
    mode: Literal["request_observation"] = "request_observation"
    operation: OperationRef
    expected: RequestCommand
    intent: RequestObservation
    initiator: InitiatorBinding
    request: RequestRef
    observation: RequestObservation
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> RequestObservationReceipt:
        original = self.intent
        current = self.observation
        if (
            self.event.operation != self.operation
            or self.event.request != self.request
            or self.event.type != "request_observation_registered"
            or self.request != self.expected.intent.selection.reference
            or current.key != original.key
            or current.filter_commitment != original.filter_commitment
            or current.projection_commitment != original.projection_commitment
            or current.after_sequence != original.after_sequence
            or current.retention_until_ms != original.retention_until_ms
            or current.coverage_sequence < current.after_sequence
            or current.revision < original.revision
        ):
            raise ValueError("Observation receipt conflicts with its event.")
        return self


class RequestObservationPage(ContractValue):
    """Bounded source-owned readback at a current transactional frontier.

    Registration is immutable; each read reconciles coverage with publication.
    """

    registration: RequestObservationReceipt
    events: tuple[RequestEvent, ...] = Field(max_length=64)
    coverage_sequence: Counter
    complete: bool

    @model_validator(mode="after")
    def exact_page(self) -> RequestObservationPage:
        observation = self.registration.observation
        if (
            self.coverage_sequence < observation.coverage_sequence
            or any(event.request != self.registration.request for event in self.events)
            or any(event.sequence > self.coverage_sequence for event in self.events)
            or any(event.sequence <= observation.after_sequence for event in self.events)
            or any(
                a.sequence >= b.sequence for a, b in zip(self.events, self.events[1:], strict=False)
            )
        ):
            raise ValueError("Observation page conflicts with its registered frontier.")
        return self


class RequestSnapshot(ContractValue):
    """Current owner state, independently authenticated against immutable receipts."""

    receipt: RequestReceipt
    permit: PermitCommand
    revision: Generation
    state: Literal["open", "answered", "failed", "declined", "cancelled", "expired"]
    admission: AdmissionState
    delivery: Literal["pending", "excluded", "published"]
    terminal: RequestControlReceipt | None
    next_due_at_ms: Counter
    admission_generation: Annotated[Generation, Field(ge=0)] = 0
    admission_decision: AdmissionDecision | None = None
    admission_operation: OperationRef | None = None
    progress: tuple[RequestProgressReceipt, ...] = Field(default=(), max_length=64)
    outcome: RequestOutcomeReceipt | None = None
    observations: tuple[RequestObservation, ...] = Field(default=(), max_length=32)
    observation_revision: Annotated[Generation, Field(ge=0)] = 0
    event_sequences: tuple[Generation, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def coherent_state(self) -> RequestSnapshot:
        reference = self.receipt.expected.intent.selection.reference
        if (
            not self.event_sequences
            or self.event_sequences[0] != self.receipt.event.sequence
            or any(
                a >= b for a, b in zip(self.event_sequences, self.event_sequences[1:], strict=False)
            )
        ):
            raise ValueError("Request event frontier is inconsistent.")
        responsibility = self.permit.intent.request
        if (
            responsibility.source_operation != self.receipt.expected.operation
            or responsibility.participant
            != self.receipt.expected.intent.selection.recipient.reference
            or responsibility.target
            != ObjectRef(
                owner=reference.owner,
                kind="collaboration_request",
                object_id=reference.request_id,
                incarnation=reference.incarnation,
            )
        ):
            raise ValueError("Request responsibility conflicts with its acceptance.")
        if self.state == "open":
            if (
                self.terminal is not None
                or self.outcome is not None
                or self.delivery != "pending"
                or (
                    self.admission == "undecided"
                    and (
                        self.revision != 1
                        or self.admission_generation != 0
                        or self.next_due_at_ms
                        != self.receipt.expected.intent.selection.accepted_at_ms
                    )
                )
                or (self.admission != "undecided" and self.admission_generation < 1)
            ):
                raise ValueError("Open request responsibility is inconsistent.")
        elif self.state in {"cancelled", "expired"} and (
            self.terminal is None
            or self.terminal.expected.intent.expected != self.receipt.expected
            or self.terminal.state != self.state
            or self.terminal.revision != self.revision
            or self.admission != "closed"
            or self.delivery != "excluded"
            or self.next_due_at_ms != 0
        ):
            raise ValueError("Terminal request evidence is inconsistent.")
        elif self.state in {"answered", "failed", "declined"} and (
            self.terminal is not None
            or self.outcome is None
            or self.outcome.command.expected != self.receipt.expected
            or self.outcome.command.outcome != self.state
            or self.outcome.revision != self.revision
            or self.admission not in {"admitted", "closed"}
            or self.next_due_at_ms != 0
        ):
            raise ValueError("Request outcome evidence is inconsistent.")
        return self


class RequestDueCursor(ContractValue):
    """Position for current due inspection, not an observation or history pin."""

    after: Counter


class RequestDuePage(ContractValue):
    items: tuple[RequestSnapshot, ...] = Field(max_length=64)
    next_cursor: RequestDueCursor | None
    observed_at_ms: Millis
