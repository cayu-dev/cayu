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
    ObjectRef,
    OperationRef,
    OwnerRef,
    snapshot_input,
)
from cayu.collaboration._permits import PermitCommand
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
    type: Literal["request_accepted", "request_cancelled", "request_expired"]
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


class RequestSnapshot(ContractValue):
    """First-slice states: no admission or external delivery has been dispatched."""

    receipt: RequestReceipt
    permit: PermitCommand
    revision: Generation
    state: Literal["open", "cancelled", "expired"]
    admission: Literal["undecided", "closed"]
    delivery: Literal["pending", "excluded"]
    terminal: RequestControlReceipt | None
    next_due_at_ms: Counter

    @model_validator(mode="after")
    def coherent_state(self) -> RequestSnapshot:
        reference = self.receipt.expected.intent.selection.reference
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
                self.revision != 1
                or self.terminal is not None
                or self.admission != "undecided"
                or self.delivery != "pending"
                or self.next_due_at_ms != self.receipt.expected.intent.selection.accepted_at_ms
            ):
                raise ValueError("Open request responsibility is inconsistent.")
        elif (
            self.terminal is None
            or self.terminal.expected.intent.expected != self.receipt.expected
            or self.terminal.state != self.state
            or self.terminal.revision != self.revision
            or self.admission != "closed"
            or self.delivery != "excluded"
            or self.next_due_at_ms != 0
        ):
            raise ValueError("Terminal request evidence is inconsistent.")
        return self


class RequestDueCursor(ContractValue):
    """Position for current due inspection, not an observation or history pin."""

    after: Counter


class RequestDuePage(ContractValue):
    items: tuple[RequestSnapshot, ...] = Field(max_length=64)
    next_cursor: RequestDueCursor | None
    observed_at_ms: Millis
