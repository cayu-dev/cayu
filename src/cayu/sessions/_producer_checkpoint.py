"""Producer checkpoint records, visibility and exact mutation authority.

Runtime publishers and session-store guards share this scope. Native admission,
publication, cleanup and recovery execution remain with their runtime owners.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue, Generation, Identifier
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerLaunchDecision,
    ProducerOutputRecord,
    ProducerOutputRegistration,
    ProducerRegistrationEvent,
)
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget, NativeCommitment
from cayu.sessions._producer_cleanup_contract import NativeProducerCleanupReceipt
from cayu.vaults.redaction import SecretRedactor

ROOT_KEY = "producer_output_owner"

OPERATION_PREFIX = "producer-output:"


class NativeProducerAttachment(ContractValue):
    command: ProducerOutputRegistration
    permit: PermitCommand
    registration_event: ProducerRegistrationEvent
    state: Literal["prepared"] = "prepared"

    @classmethod
    def from_registration(cls, registration: ProducerOutputRecord) -> NativeProducerAttachment:
        registration = prepare_contract(
            ProducerOutputRecord, registration, redactor=SecretRedactor()
        )
        # Mutable source accounting and disposition do not replace the immutable
        # native attachment when recovery rereads the same registration later.
        return cls(
            command=registration.command,
            permit=registration.permit,
            registration_event=registration.event,
        )


class NativeProducerInvocation(ContractValue):
    launch: ProducerLaunchDecision
    interaction_id: Identifier
    run_epoch: Generation
    profile_commitment: NativeCommitment
    participant_permit_operation: Identifier
    participant_permit_commitment: NativeCommitment


class NativeProducerPausedStop(ContractValue):
    """An exact released human pause sealed against every later rebind."""

    run_epoch: Generation
    closure_commitment: NativeCommitment
    release_commitment: NativeCommitment


class NativeProducerIndex(ContractValue):
    session_id: Identifier
    session_instance_id: Identifier
    operation_key: Identifier
    record_commitment: NativeCommitment
    state: Literal["prepared", "excluded", "admitted"] = "prepared"
    exclusion_commitment: NativeCommitment | None = None
    invocation: NativeProducerInvocation | None = None
    cleanup_commitment: NativeCommitment | None = None
    cleanup_receipt: NativeProducerCleanupReceipt | None = None
    output_commitment: NativeCommitment | None = None
    paused_stop: NativeProducerPausedStop | None = None

    @model_validator(mode="after")
    def exact_exclusion(self):
        if (self.state == "excluded") != (self.exclusion_commitment is not None):
            raise ValueError("Producer exclusion index is inconsistent.")
        if (self.state == "admitted") != (self.invocation is not None):
            raise ValueError("Producer invocation index is inconsistent.")
        if self.cleanup_commitment is not None and self.state not in ("excluded", "admitted"):
            raise ValueError("Native cleanup lacks producer admission or exclusion.")
        receipt = self.cleanup_receipt
        if (receipt is None) != (self.cleanup_commitment is None):
            raise ValueError("Producer ownership release requires its native cleanup receipt.")
        if receipt is not None and (
            self.cleanup_commitment
            != "sha256:" + sha256(contract_bytes(receipt, redactor=SecretRedactor())).hexdigest()
            or (receipt.session_id, receipt.session_instance_id)
            != (self.session_id, self.session_instance_id)
            or (receipt.mode == "invocation") != (self.state == "admitted")
            or (
                self.invocation is not None
                and (
                    receipt.interaction_id != self.invocation.interaction_id
                    or receipt.run_epoch is None
                    or receipt.run_epoch < self.invocation.run_epoch
                )
            )
        ):
            raise ValueError("Producer ownership release conflicts with its native identity.")
        if self.output_commitment is not None and self.state != "admitted":
            raise ValueError("Native output lacks producer admission.")
        if self.paused_stop is not None and (
            self.state != "admitted" or self.output_commitment is not None
        ):
            raise ValueError("Paused producer stop conflicts with native output.")
        return self


_PUBLICATION: ContextVar[NativeProducerIndex | None] = ContextVar(
    "native_producer_publication", default=None
)


def attachment_operation_key(command: ProducerOutputRegistration) -> str:
    return (
        OPERATION_PREFIX
        + sha256(contract_bytes(command.operation, redactor=SecretRedactor())).hexdigest()
    )


def attachment_index(attachment: NativeProducerAttachment) -> NativeProducerIndex:
    command = attachment.command
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    redactor = SecretRedactor()
    return NativeProducerIndex(
        session_id=prepared.target.session_id,
        session_instance_id=prepared.target.session_instance_id,
        operation_key=attachment_operation_key(command),
        record_commitment="sha256:"
        + sha256(contract_bytes(attachment, redactor=redactor)).hexdigest(),
    )


@contextmanager
def _publication_scope(index: NativeProducerIndex):
    token = _PUBLICATION.set(index)
    try:
        yield
    finally:
        _PUBLICATION.reset(token)


def checkpoint_visible(*, session_id: str) -> bool:
    current = _PUBLICATION.get()
    return current is not None and current.session_id == session_id


def require_operation_key_access(key: str, *, read: bool) -> None:
    if not read and key.startswith(OPERATION_PREFIX):
        current = _PUBLICATION.get()
        if current is None or key not in (
            current.operation_key,
            current.operation_key + ":excluded",
            current.operation_key + ":cleanup",
            current.operation_key + ":output",
        ):
            raise PermissionError("Producer operation requires its native owner.")


def project_checkpoint_root(current, replacement, *, session_id: str):
    redactor = SecretRedactor()
    before = None if current is None else current.get(ROOT_KEY)
    previous = (
        None if before is None else prepare_contract(NativeProducerIndex, before, redactor=redactor)
    )
    if previous is not None and previous.session_id != session_id:
        raise ValueError("Producer index belongs to another session.")
    after = replacement.get(ROOT_KEY)
    publication = _PUBLICATION.get()
    if publication is not None and publication.session_id == session_id:
        proposed = prepare_contract(NativeProducerIndex, after, redactor=redactor)
        if (
            previous is not None
            and previous.cleanup_commitment is not None
            and proposed == previous
            and publication
            == previous.model_copy(update={"cleanup_commitment": None, "cleanup_receipt": None})
        ):
            return previous.model_dump(mode="json")
        require_exact_contract(publication, proposed, redactor=redactor)
        if previous is not None:
            if previous.state == "prepared" and proposed.state in ("excluded", "admitted"):
                require_exact_contract(
                    previous,
                    proposed.model_copy(
                        update={
                            "state": "prepared",
                            "exclusion_commitment": None,
                            "invocation": None,
                        }
                    ),
                    redactor=redactor,
                )
            else:
                if previous.paused_stop is None and proposed.paused_stop is not None:
                    require_exact_contract(
                        previous,
                        proposed.model_copy(update={"paused_stop": None}),
                        redactor=redactor,
                    )
                    return proposed.model_dump(mode="json")
                if (
                    previous.state == "admitted"
                    and previous.cleanup_commitment is None
                    and proposed.cleanup_commitment is not None
                ):
                    require_exact_contract(
                        previous,
                        proposed.model_copy(
                            update={"cleanup_commitment": None, "cleanup_receipt": None}
                        ),
                        redactor=redactor,
                    )
                    return proposed.model_dump(mode="json")
                if previous.state == "admitted" and previous.output_commitment is None:
                    require_exact_contract(
                        previous,
                        proposed.model_copy(update={"output_commitment": None}),
                        redactor=redactor,
                    )
                    return proposed.model_dump(mode="json")
                require_exact_contract(
                    previous,
                    proposed.model_copy(
                        update={"cleanup_commitment": None, "cleanup_receipt": None}
                    )
                    if previous.state == "excluded" and previous.cleanup_commitment is None
                    else proposed,
                    redactor=redactor,
                )
        return proposed.model_dump(mode="json")
    if after is not None:
        proposed = prepare_contract(NativeProducerIndex, after, redactor=redactor)
        if previous is None:
            raise PermissionError("Generic checkpoint mutation cannot create producer authority.")
        require_exact_contract(previous, proposed, redactor=redactor)
    return None if previous is None else previous.model_dump(mode="json")
