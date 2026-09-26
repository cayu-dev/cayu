"""Content-free progress identity; values alone confer no publication authority."""

from hashlib import sha256
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue, Generation, Identifier, OperationRef
from cayu.collaboration.prepared_admission import NativeCommitment

ProducerProgressKind = Literal["prepared", "started", "producing", "published"]


class ProducerProgressOccurrence(ContractValue):
    operation: OperationRef
    expected_revision: Generation
    sequence: Generation
    kind: ProducerProgressKind


class ProducerProgressEvidence(ContractValue):
    registration: OperationRef
    registration_commitment: NativeCommitment
    kind: ProducerProgressKind
    session_id: Identifier
    session_instance_id: Identifier
    interaction_id: Identifier | None
    run_epoch: Generation | None
    profile_commitment: NativeCommitment
    native_commitment: NativeCommitment

    @model_validator(mode="after")
    def coherent(self):
        if (self.interaction_id is None) != (self.run_epoch is None) or (
            self.kind != "prepared" and self.interaction_id is None
        ):
            raise ValueError("Native progress invocation evidence is incomplete.")
        return self


class ProducerProgressReference(ContractValue):
    """Exact frontier pointer; the complete receipt remains in the operation store."""

    mode: Literal["producer_progress_reference"] = "producer_progress_reference"
    operation: OperationRef
    sequence: Generation
    revision: Generation
    kind: ProducerProgressKind
    event_sequence: Generation
    receipt_commitment: NativeCommitment

    @classmethod
    def from_receipt(cls, receipt, *, redactor):
        from cayu.collaboration._preparation import contract_bytes

        return cls(
            operation=receipt.command.operation,
            sequence=receipt.command.sequence,
            revision=receipt.revision,
            kind=receipt.command.kind,
            event_sequence=receipt.event.sequence,
            receipt_commitment="sha256:"
            + sha256(contract_bytes(receipt, redactor=redactor)).hexdigest(),
        )
