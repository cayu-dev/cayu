"""Private retained-permit contracts for qualified owner integrations.

Registration records responsibility; it does not mint a public execution grant.
Only a registered receiving-owner reader can provide settlement evidence. Reader
I/O happens outside the collaboration transaction, followed by exact revalidation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import (
    Code,
    ContractValue,
    ExactLookup,
    ExpectedOperation,
    Generation,
    Identifier,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration.participants import (
    CollaborationLimits,
    ParticipantEvent,
    ParticipantRef,
    VersionOne,
)


class PermitRegistration(ContractValue):
    operation: OperationRef
    participant: ParticipantRef
    expected_lifecycle_revision: Generation
    admission_generation: Generation
    source_operation: OperationRef
    target: ObjectRef
    target_state: Literal["existing", "future"]
    effect_scope: Code
    required_settlement: Literal["exclusion", "quiescence"]
    settlement_operation: OperationRef
    kind: Literal["permit_register"] = "permit_register"

    @model_validator(mode="after")
    def bound_source_and_reserved_key(self) -> PermitRegistration:
        scope = self.participant.owner.application_scope
        if any(
            operation.application_scope != scope
            for operation in (self.operation, self.source_operation, self.settlement_operation)
        ):
            raise ValueError("Permit source operations belong to another scope.")
        if (
            self.operation.namespace_incarnation != self.settlement_operation.namespace_incarnation
            or self.operation.generation != self.settlement_operation.generation
            or self.operation == self.settlement_operation
        ):
            raise ValueError("Settlement needs a distinct key in the admitted generation.")
        return self


class PermitIntent(ContractValue):
    request: PermitRegistration
    limits: CollaborationLimits


class PermitCommand(ExpectedOperation[PermitIntent]):
    kind: Literal["permit_register"] = "permit_register"
    schema_version: VersionOne = 1
    mode: Literal["permit"] = "permit"
    receipt_stage: Literal["registered"] = "registered"

    @model_validator(mode="after")
    def exact_registration(self) -> PermitCommand:
        request = self.intent.request
        if (
            self.operation != request.operation
            or self.source != self.destination
            or request.participant.owner != self.source
            or self.initiator.issuer != self.source
        ):
            raise ValueError("Permit registration authority conflicts.")
        return self


class ReceivingSettlementReceipt(ContractValue):
    """Content returned by an authenticated reader, not self-authenticating proof."""

    expected: PermitCommand
    receiving_owner: OwnerRef
    receipt_id: Identifier
    outcome: Literal["excluded", "quiescent"]

    @model_validator(mode="after")
    def matches_receiving_responsibility(self) -> ReceivingSettlementReceipt:
        request = self.expected.intent.request
        if self.receiving_owner != request.target.owner:
            raise ValueError("Settlement was issued by another receiving owner.")
        if request.required_settlement == "quiescence" and self.outcome != "quiescent":
            raise ValueError("Exclusion alone does not prove required quiescence.")
        return self


class PermitSettlementReader(ABC):
    """Trusted integration registration; no raw receipt-taking public SDK method.

    Implementations authenticate the exact receiver's durable receipt. Missing
    targets, stop acknowledgements and expired claims must not become MATCH.
    Implementations must not dispatch new effects while reading settlement.
    """

    @property
    @abstractmethod
    def owner(self) -> OwnerRef: ...

    @abstractmethod
    async def lookup(self, expected: PermitCommand) -> ExactLookup[ReceivingSettlementReceipt]: ...


class PermitSnapshot(ContractValue):
    expected: PermitCommand
    position: Generation
    state: Literal["pending", "settled"]
    settlement: ReceivingSettlementReceipt | None

    @model_validator(mode="after")
    def exact_settlement(self) -> PermitSnapshot:
        if (self.state == "settled") != (self.settlement is not None):
            raise ValueError("Permit state conflicts with settlement evidence.")
        if self.settlement is not None and self.settlement.expected != self.expected:
            raise ValueError("Settlement describes another permit.")
        return self


class PermitReceipt(ContractValue):
    record_type: Literal["permit_registered"] = "permit_registered"
    expected: PermitCommand
    position: Generation
    event: ParticipantEvent

    @model_validator(mode="after")
    def exact_registration_event(self) -> PermitReceipt:
        if (
            self.event.operation != self.expected.operation
            or self.event.type != "permit_registered"
            or self.event.participants != (self.expected.intent.request.participant,)
        ):
            raise ValueError("Permit registration event conflicts.")
        return self


class ReservedPermitSettlement(ContractValue):
    record_type: Literal["permit_settlement_reserved"] = "permit_settlement_reserved"
    expected: PermitCommand


class PermitSettlement(ContractValue):
    record_type: Literal["permit_settled"] = "permit_settled"
    expected: PermitCommand
    receiving_receipt: ReceivingSettlementReceipt
    event: ParticipantEvent

    @model_validator(mode="after")
    def exact_settlement_event(self) -> PermitSettlement:
        if (
            self.receiving_receipt.expected != self.expected
            or self.event.operation != self.expected.intent.request.settlement_operation
            or self.event.type != "permit_settled"
            or self.event.participants != (self.expected.intent.request.participant,)
        ):
            raise ValueError("Permit settlement event conflicts.")
        return self
