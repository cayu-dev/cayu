"""Pure request-owner clarification election, never an authorization entrance.

The transaction owner supplies authenticated commands, current parent state and
its own clock. The returned immutable change must be committed together with
the parent's input revision, capacity, receipt and event. No foreign callbacks
or runtime dispatch are performed here.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import CollaborationConflict, ContractValue, OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration.clarifications import (
    ClarificationInputRevision,
    ClarificationQuestion,
    ClarificationReply,
)
from cayu.collaboration.requests import Millis
from cayu.vaults.redaction import SecretRedactor


def clarification_commitment(value: ContractValue, redactor: SecretRedactor) -> str:
    return sha256(contract_bytes(value, redactor=redactor)).hexdigest()


class ClarificationQuestionState(ContractValue):
    """Question decision, deliberately independent of delivery and service return."""

    question: ClarificationQuestion
    opened_at_ms: Millis
    state: Literal["open", "answered", "superseded", "cancelled", "expired", "request_terminal"]
    reply: ClarificationReply | None = None
    input: ClarificationInputRevision | None = None
    closed_at_ms: Millis | None = None
    closure_operation: OperationRef | None = None

    @model_validator(mode="after")
    def coherent(self) -> ClarificationQuestionState:
        terminal_without_reply = self.state not in ("open", "answered")
        if terminal_without_reply != (self.closure_operation is not None):
            raise ValueError("Clarification closure requires its exact operation.")
        if self.closure_operation is not None and (
            self.closure_operation.application_scope != self.question.operation.application_scope
            or self.closure_operation == self.question.operation
        ):
            raise ValueError("Clarification closure operation conflicts.")
        if self.opened_at_ms >= self.question.deadline_at_ms:
            raise ValueError("Question opening must precede its deadline.")
        if self.state == "open":
            if self.reply is not None or self.input is not None or self.closed_at_ms is not None:
                raise ValueError("Open question cannot carry terminal evidence.")
        elif self.closed_at_ms is None or self.closed_at_ms < self.opened_at_ms:
            raise ValueError("Closed question requires ordered owner-time evidence.")
        if self.state != "answered":
            if self.reply is not None or self.input is not None:
                raise ValueError("Unanswered question cannot publish an accepted input revision.")
            if self.state == "expired" and (
                self.closed_at_ms is None or self.closed_at_ms < self.question.deadline_at_ms
            ):
                raise ValueError("Question cannot expire before its deadline.")
            return self
        if self.reply is None or self.input is None:
            raise ValueError("Answered question requires both reply and input evidence.")
        reply, revision = self.reply, self.input
        if (
            self.closed_at_ms != revision.accepted_at_ms
            or revision.accepted_at_ms >= self.question.deadline_at_ms
            or reply.question != self.question.operation
            or reply.question_generation != self.question.generation
            or reply.question_input_revision != self.question.input_revision
            or reply.responder != self.question.responder
            or revision.request != self.question.request
            or revision.question != self.question.operation
            or revision.reply != reply.operation
            or revision.revision != reply.expected_input_revision + 1
            or revision.previous_sha256 != reply.expected_input_sha256
            or revision.content_sha256 != reply.source.content_sha256
            or reply.source.content_bytes > self.question.policy.max_reply_bytes
        ):
            raise ValueError("Question answer disagrees with its input ancestry.")
        redactor = SecretRedactor()
        if reply.question_sha256 != clarification_commitment(
            self.question, redactor
        ) or revision.reply_sha256 != clarification_commitment(reply, redactor):
            raise ValueError("Question answer commitment is inconsistent.")
        return self


def accept_clarification_reply(
    retained: ClarificationQuestionState,
    candidate: ClarificationReply,
    *,
    request_is_open: bool,
    current_input_revision: int,
    current_input_sha256: str,
    now_ms: int,
    redactor: SecretRedactor,
) -> ClarificationQuestionState:
    """Elect one reply or reject unchanged, inside the request owner's transaction.

    Exact replay precedes new-effect deadline/input checks. Authentication and
    current receipt read permission must already have succeeded at the receiver.
    Replaying an answer never authorizes another export or exposure.
    """
    state = prepare_contract(ClarificationQuestionState, retained, redactor=redactor)
    reply = prepare_contract(ClarificationReply, candidate, redactor=redactor)
    if type(request_is_open) is not bool:
        raise ValueError("Parent terminal evidence must be explicit.")
    if type(now_ms) is not int or not 0 < now_ms <= 2**53 - 1:
        raise ValueError("Reply election requires valid owner time.")
    if type(current_input_revision) is not int or not 0 <= current_input_revision < 2**53:
        raise ValueError("Reply election requires a valid input revision.")
    if (
        type(current_input_sha256) is not str
        or len(current_input_sha256) != 64
        or any(char not in "0123456789abcdef" for char in current_input_sha256)
    ):
        raise ValueError("Reply election requires the exact input commitment.")
    if state.reply is not None:
        require_exact_contract(reply, state.reply, redactor=redactor)
        return state
    question = state.question
    if state.state != "open" or not request_is_open:
        raise CollaborationConflict("Clarification question is no longer open.")
    if now_ms < state.opened_at_ms or now_ms >= question.deadline_at_ms:
        raise CollaborationConflict("Reply is outside the question admission interval.")
    if (
        reply.question != question.operation
        or reply.question_sha256 != clarification_commitment(question, redactor)
        or reply.question_generation != question.generation
        or reply.question_input_revision != question.input_revision
        or reply.responder != question.responder
        or reply.expected_input_revision != current_input_revision
        or reply.expected_input_sha256 != current_input_sha256
        or reply.source.content_bytes > question.policy.max_reply_bytes
    ):
        raise CollaborationConflict("Reply conflicts with the exact open question or input.")
    revision = ClarificationInputRevision(
        request=question.request,
        revision=current_input_revision + 1,
        previous_sha256=current_input_sha256,
        question=question.operation,
        reply=reply.operation,
        reply_sha256=clarification_commitment(reply, redactor),
        content_sha256=reply.source.content_sha256,
        accepted_at_ms=now_ms,
    )
    return prepare_contract(
        ClarificationQuestionState,
        state.model_copy(
            update={
                "state": "answered",
                "reply": reply,
                "input": revision,
                "closed_at_ms": now_ms,
            }
        ),
        redactor=redactor,
    )
