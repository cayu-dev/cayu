"""Exact request-owner clarification commands; values do not grant authority."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from cayu.collaboration._clarification_state import (
    ClarificationQuestionState,
    clarification_commitment,
)
from cayu.collaboration._contracts import ContractValue, Generation, InitiatorBinding, OperationRef
from cayu.collaboration.clarifications import ClarificationQuestion, ClarificationReply, Commitment
from cayu.collaboration.requests import RequestCommand, RequestEvent
from cayu.vaults.redaction import SecretRedactor


class ClarificationOpenCommand(ContractValue):
    mode: Literal["clarification_open"] = "clarification_open"
    operation: OperationRef
    expected: RequestCommand
    expected_revision: Generation
    question: ClarificationQuestion

    @model_validator(mode="after")
    def exact_parent(self) -> ClarificationOpenCommand:
        selection = self.expected.intent.selection
        parent = self.expected.operation
        if (
            self.operation != self.question.operation
            or self.operation == parent
            or self.operation.application_scope != parent.application_scope
            or self.operation.namespace_incarnation != parent.namespace_incarnation
            or self.operation.generation != parent.generation
            or self.question.request != selection.reference
            or self.question.responder != selection.sender.reference
            or self.question.deadline_at_ms > selection.expires_at_ms
            or self.question.request_sha256
            != clarification_commitment(self.expected, SecretRedactor())
        ):
            raise ValueError("Clarification opening conflicts with its original request.")
        return self


class ClarificationReplyCommand(ContractValue):
    mode: Literal["clarification_reply"] = "clarification_reply"
    operation: OperationRef
    expected: RequestCommand
    reply: ClarificationReply

    @model_validator(mode="after")
    def exact_parent(self) -> ClarificationReplyCommand:
        parent = self.expected.operation
        if (
            self.operation != self.reply.operation
            or self.operation == parent
            or self.operation.application_scope != parent.application_scope
            or self.operation.namespace_incarnation != parent.namespace_incarnation
            or self.operation.generation != parent.generation
            or self.reply.responder != self.expected.intent.selection.sender.reference
        ):
            raise ValueError("Clarification reply conflicts with its original request.")
        return self


class ClarificationOpenReceipt(ContractValue):
    command: ClarificationOpenCommand
    decision: ClarificationQuestionState
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> ClarificationOpenReceipt:
        selection = self.command.expected.intent.selection
        if (
            self.decision.question != self.command.question
            or self.decision.state != "open"
            or self.decision.opened_at_ms < selection.accepted_at_ms
            or self.event.operation != self.command.operation
            or self.event.request != selection.reference
            or self.event.type != "clarification_opened"
            or self.event.participants
            != tuple(dict.fromkeys((selection.sender.reference, selection.recipient.reference)))
        ):
            raise ValueError("Clarification opening receipt conflicts with its command.")
        return self


class ClarificationReplyReceipt(ContractValue):
    command: ClarificationReplyCommand
    decision: ClarificationQuestionState
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> ClarificationReplyReceipt:
        selection = self.command.expected.intent.selection
        if (
            self.decision.reply != self.command.reply
            or self.decision.state != "answered"
            or self.decision.question.request != selection.reference
            or self.decision.question.request_sha256
            != clarification_commitment(self.command.expected, SecretRedactor())
            or self.event.operation != self.command.operation
            or self.event.request != selection.reference
            or self.event.type != "clarification_replied"
            or self.event.participants
            != tuple(dict.fromkeys((selection.sender.reference, selection.recipient.reference)))
        ):
            raise ValueError("Clarification reply receipt conflicts with its command.")
        return self


class ClarificationCloseCommand(ContractValue):
    mode: Literal["clarification_close"] = "clarification_close"
    operation: OperationRef
    expected: RequestCommand
    question: OperationRef
    question_sha256: Commitment
    kind: Literal["cancelled", "expired", "superseded", "request_terminal"]
    initiator: InitiatorBinding

    @model_validator(mode="after")
    def exact_parent(self) -> ClarificationCloseCommand:
        parent = self.expected.operation
        if (
            self.operation in (parent, self.question)
            or self.initiator.issuer.application_scope != parent.application_scope
            or any(
                (ref.application_scope, ref.namespace_incarnation, ref.generation)
                != (parent.application_scope, parent.namespace_incarnation, parent.generation)
                for ref in (self.operation, self.question)
            )
        ):
            raise ValueError("Clarification closure conflicts with its original request.")
        return self


class ClarificationCloseReceipt(ContractValue):
    command: ClarificationCloseCommand
    decision: ClarificationQuestionState
    event: RequestEvent

    @model_validator(mode="after")
    def exact_receipt(self) -> ClarificationCloseReceipt:
        selection = self.command.expected.intent.selection
        if (
            self.decision.question.operation != self.command.question
            or clarification_commitment(self.decision.question, SecretRedactor())
            != self.command.question_sha256
            or self.decision.question.request_sha256
            != clarification_commitment(self.command.expected, SecretRedactor())
            or self.decision.question.request != selection.reference
            or self.decision.state != self.command.kind
            or self.decision.closure_operation != self.command.operation
            or self.event.operation != self.command.operation
            or self.event.request != selection.reference
            or self.event.type != "clarification_closed"
            or self.event.participants
            != tuple(dict.fromkeys((selection.sender.reference, selection.recipient.reference)))
        ):
            raise ValueError("Clarification closure receipt conflicts with its command.")
        return self
