"""Exact source-owner responsibility for the existing receiving peer queue.

These records are not a second content queue or disclosure grant. A coordinator
must acquire current export authority before registering or servicing an intent,
and reconcile the complete request through the receiving SessionStore. Neither
cancellation nor question closure constitutes receiving exclusion or exposure.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import (
    ContractValue,
    Identifier,
    InitiatorBinding,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration.clarifications import ClarificationQuestion, ClarificationReply
from cayu.collaboration.exports import SessionExportRequest
from cayu.collaboration.participants import ParticipantRef
from cayu.collaboration.peer_content import PeerContentAppendRequest, PeerContentReceipt


class ClarificationDeliveryIntent(ContractValue):
    """Immutable retry tuple; receiving authority is established separately."""

    operation: OperationRef
    initiator: InitiatorBinding
    question: ClarificationQuestion
    reply: ClarificationReply | None = None
    sender: ParticipantRef
    recipient: ParticipantRef
    export: SessionExportRequest
    append: PeerContentAppendRequest

    @model_validator(mode="after")
    def exact_source_and_target(self) -> ClarificationDeliveryIntent:
        source = self.question.source if self.reply is None else self.reply.source
        scope = self.operation.application_scope
        occurrence = self.append.occurrence
        key = self.append.append_key
        audience = OwnerRef(
            application_scope=scope,
            owner_id=self.recipient.participant_id,
            incarnation=self.recipient.incarnation,
        )
        if (
            self.operation in (self.question.operation, self.question.admission)
            or self.question.operation.application_scope != scope
            or self.sender.owner.application_scope != scope
            or self.recipient.owner.application_scope != scope
            or self.initiator.issuer.application_scope != scope
            or self.export.ref != source.export
            or self.export.source_selection != source.selection
            or self.export.projector != source.projector
            or self.export.policy != source.policy
            or self.export.audience != audience
            or source.audience.owner != self.recipient.owner
            or source.audience.kind != "participant"
            or source.audience.object_id != self.recipient.participant_id
            or source.audience.incarnation != self.recipient.incarnation
            or occurrence.sender_participant_id != self.sender.participant_id
            or occurrence.sender_participant_incarnation != self.sender.incarnation
            or occurrence.sender_session_id != source.export.session_id
            or occurrence.sender_session_instance_id != source.export.session_instance_id
            or occurrence.source_export_receipt_id != source.producer.object_id
            or source.producer.kind != "session_export"
            or source.producer.incarnation != source.export.session_instance_id
            or key.consumer_id != self.recipient.participant_id
            or key.consumer_participant_incarnation != self.recipient.incarnation
            or occurrence.audience != (self.recipient.participant_id,)
            or key.creation_target is not None
            or key.collaboration_namespace != self.operation.namespace_incarnation
            or key.collaboration_generation != self.operation.generation
            or self.append.wake_policy != "none"
            or self.append.attempt_key.deadline_at_ms > self.question.deadline_at_ms
            or occurrence.payload.artifact_commitments
        ):
            raise ValueError("Clarification delivery conflicts with its source or target.")
        encoded = occurrence.payload.text.encode("utf-8")
        if len(encoded) != source.content_bytes or sha256(encoded).hexdigest() != (
            source.content_sha256
        ):
            raise ValueError("Clarification delivery content conflicts with its export.")
        if self.reply is None:
            if self.recipient != self.question.responder:
                raise ValueError("Question delivery requires its exact responder.")
        else:
            from cayu.collaboration._clarification_state import clarification_commitment
            from cayu.vaults.redaction import SecretRedactor

            reply = self.reply
            if (
                self.operation in (reply.operation, reply.service)
                or reply.question != self.question.operation
                or reply.question_generation != self.question.generation
                or reply.question_input_revision != self.question.input_revision
                or reply.question_sha256
                != clarification_commitment(self.question, SecretRedactor())
                or reply.responder != self.question.responder
                or self.sender != reply.responder
            ):
                raise ValueError("Reply delivery conflicts with its exact question.")
        return self


class ClarificationDeliveryRecord(ContractValue):
    """Pending until positive append/exclusion readback, never exposure evidence."""

    mode: Literal["clarification_delivery"] = "clarification_delivery"
    intent: ClarificationDeliveryIntent
    state: Literal["pending", "settled"] = "pending"
    receipt: PeerContentReceipt | None = None

    @model_validator(mode="after")
    def exact_receiving_decision(self) -> ClarificationDeliveryRecord:
        if (self.state == "settled") != (self.receipt is not None):
            raise ValueError("Delivery settlement requires a receiving decision.")
        receipt = self.receipt
        if receipt is None:
            return self
        request = self.intent.append
        key = request.append_key
        if (
            receipt.operation_key != request.operation_key
            or receipt.append_key != key
            or receipt.attempt_generation != request.attempt_key.attempt_generation
            or receipt.status not in ("appended", "excluded")
            or receipt.disclosure != "available"
            or (receipt.occurrence is not None and receipt.occurrence != request.occurrence)
            or (
                receipt.target_session_id is not None
                and receipt.target_session_id != key.target_session_id
            )
            or (
                receipt.target_session_instance_id is not None
                and receipt.target_session_instance_id != key.target_session_instance_id
            )
        ):
            raise ValueError("Delivery receipt conflicts with the exact append attempt.")
        return self


class ClarificationDeliveryReceipt(ContractValue):
    """Public handoff status, without source payload or private receiving intent."""

    operation: OperationRef
    question: OperationRef
    kind: Literal["question", "reply"]
    status: Literal["pending", "appended", "excluded"]
    queue_id: Identifier | None = None
    reason: Identifier | None = None

    @model_validator(mode="after")
    def coherent(self) -> ClarificationDeliveryReceipt:
        if (
            self.operation.application_scope != self.question.application_scope
            or (self.status == "appended") != (self.queue_id is not None)
            or (self.status == "excluded") != (self.reason is not None)
        ):
            raise ValueError("Clarification delivery status contradicts its evidence.")
        return self

    @classmethod
    def from_record(cls, record: ClarificationDeliveryRecord) -> ClarificationDeliveryReceipt:
        receipt = record.receipt
        status: Literal["pending", "appended", "excluded"] = "pending"
        if receipt is not None:
            if receipt.status == "appended":
                status = "appended"
            elif receipt.status == "excluded":
                status = "excluded"
            else:
                raise ValueError("Delivery projection requires terminal receiving evidence.")
        return cls(
            operation=record.intent.operation,
            question=record.intent.question.operation,
            kind="question" if record.intent.reply is None else "reply",
            status=status,
            queue_id=None if receipt is None else receipt.queue_id,
            reason=None if receipt is None else receipt.reason,
        )
