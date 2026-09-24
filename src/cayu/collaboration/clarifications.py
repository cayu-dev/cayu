"""Finite clarification values. Construction never grants service authority.

Source references and finite policy registrations are public values. Question,
reply and service operations require their registered receiving owners; neither
construction nor a historical source reference grants execution or disclosure.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from cayu.collaboration._contracts import (
    ContractValue,
    Generation,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration.exports import SessionExportRef
from cayu.collaboration.participants import ParticipantRef, VersionOne
from cayu.collaboration.requests import MAX_REQUEST_CLARIFICATIONS, Millis, RequestRef

MAX_CLARIFICATION_QUESTIONS = MAX_REQUEST_CLARIFICATIONS
MAX_CLARIFICATION_DEPTH = 4
MAX_CLARIFICATION_TURNS = 32
MAX_CLARIFICATION_TEXT_BYTES = 16 * 1024
MAX_CLARIFICATION_CONTENT_BYTES = 1024 * 1024
MAX_CLARIFICATION_PENDING = 64
MAX_CLARIFICATION_POLICIES = 32

Commitment = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
InputRevision = Annotated[StrictInt, Field(ge=0, le=2**53 - 1)]


class ClarificationDueCursor(ContractValue):
    """Stable deadline/operation ordering; not a claim or execution permission."""

    deadline_at_ms: Millis
    operation: OperationRef


class ClarificationPolicy(ContractValue):
    """Explicit registered policy; every ceiling is required and finite."""

    reference: ObjectRef
    max_questions: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_QUESTIONS)
    max_depth: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_DEPTH)
    max_service_turns: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_TURNS)
    max_question_bytes: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_TEXT_BYTES)
    max_reply_bytes: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_TEXT_BYTES)
    max_content_bytes: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_CONTENT_BYTES)
    max_pending: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_PENDING)
    service_timeout_ms: Millis
    max_spend_usd: StrictStr = Field(min_length=1, max_length=64)
    failure_policy: Literal["return_and_report"]
    return_policy: Literal["original_wait_or_terminal_exclusion"]

    @field_validator("max_spend_usd")
    @classmethod
    def finite_spend(cls, value: str) -> str:
        # No decimal arithmetic/normalization: preserve exact policy intent and
        # avoid context precision or exponent expansion changing its commitment.
        if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None:
            raise ValueError("Clarification spending requires a finite positive amount.")
        try:
            amount = Decimal(value)
        except InvalidOperation:
            raise ValueError("Clarification spending requires a finite positive amount.") from None
        if not amount.is_finite() or amount <= 0:
            raise ValueError("Clarification spending requires a finite positive amount.")
        return value

    @model_validator(mode="after")
    def usable_capacity(self) -> ClarificationPolicy:
        if self.max_question_bytes > self.max_content_bytes:
            raise ValueError("Question allowance exceeds the lineage content ceiling.")
        if self.max_reply_bytes > self.max_content_bytes:
            raise ValueError("Reply allowance exceeds the lineage content ceiling.")
        return self


class ClarificationSource(ContractValue):
    """Exact export selection to authenticate through its registered owner."""

    export: SessionExportRef
    export_receipt_sha256: Commitment
    producer: ObjectRef
    content_sha256: Commitment
    content_bytes: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_TEXT_BYTES)
    selection: Literal["whole_records", "assistant_visible_text_v1"]
    projector: ObjectRef
    policy: ObjectRef
    audience: ObjectRef

    @model_validator(mode="after")
    def consistent_scope(self) -> ClarificationSource:
        if any(
            value.owner.application_scope != self.export.operation.application_scope
            for value in (self.producer, self.projector, self.policy, self.audience)
        ):
            raise ValueError("Clarification source belongs to another application scope.")
        return self


class ClarificationQuestion(ContractValue):
    """Immutable question intent; foreign receipts must still be authenticated."""

    schema_version: VersionOne = 1
    operation: OperationRef
    request: RequestRef
    request_sha256: Commitment
    admission: OperationRef
    initiator: InitiatorBinding
    responder: ParticipantRef
    receiver: ObjectRef
    generation: Generation
    input_revision: InputRevision
    input_sha256: Commitment
    lineage: OperationRef
    parent_question: OperationRef | None
    depth: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_DEPTH)
    source: ClarificationSource
    policy: ClarificationPolicy
    budget_binding: ObjectRef
    budget_authority_sha256: Commitment
    deadline_at_ms: Millis

    @model_validator(mode="after")
    def consistent_scope(self) -> ClarificationQuestion:
        operation = self.operation
        scope = operation.application_scope
        if (
            self.request.owner.application_scope != scope
            or self.responder.owner.application_scope != scope
            or self.receiver.owner.application_scope != scope
            or self.admission.application_scope != scope
            or self.lineage.application_scope != scope
            or self.budget_binding.owner.application_scope != scope
            or self.policy.reference.owner.application_scope != scope
            or self.source.export.operation.application_scope != scope
            or self.initiator.issuer.application_scope != scope
        ):
            raise ValueError("Clarification authority belongs to another application scope.")
        if (self.parent_question is None) != (self.depth == 1):
            raise ValueError("Clarification depth requires its exact parent question.")
        if self.parent_question is not None and (
            self.parent_question == operation or self.parent_question.application_scope != scope
        ):
            raise ValueError("Clarification parent identity is invalid.")
        if self.operation in (self.admission, self.parent_question):
            raise ValueError("Question requires a distinct operation identity.")
        if self.depth > self.policy.max_depth:
            raise ValueError("Clarification nesting exceeds the admitted policy.")
        if self.source.content_bytes > self.policy.max_question_bytes:
            raise ValueError("Clarification question exceeds the admitted byte ceiling.")
        return self


class ClarificationReply(ContractValue):
    """A reply proposal, not proof of production, disclosure, or acceptance."""

    schema_version: VersionOne = 1
    operation: OperationRef
    question: OperationRef
    question_sha256: Commitment
    question_generation: Generation
    question_input_revision: InputRevision
    expected_input_revision: InputRevision
    expected_input_sha256: Commitment
    initiator: InitiatorBinding
    responder: ParticipantRef
    service: OperationRef
    service_generation: Generation
    service_session: ObjectRef
    invocation: ObjectRef
    admission_sha256: Commitment
    production_stage_id: Identifier
    production_sha256: Commitment
    source: ClarificationSource

    @model_validator(mode="after")
    def distinct_operation(self) -> ClarificationReply:
        scope = self.operation.application_scope
        if self.operation in {self.question, self.service}:
            raise ValueError("Reply requires a distinct operation identity.")
        if any(value.application_scope != scope for value in (self.question, self.service)) or any(
            value.owner.application_scope != scope
            for value in (self.responder, self.service_session, self.invocation)
        ):
            raise ValueError("Reply authority belongs to another application scope.")
        if (
            self.source.export.operation.application_scope != scope
            or self.initiator.issuer.application_scope != scope
        ):
            raise ValueError("Reply source belongs to another application scope.")
        if self.expected_input_revision < self.question_input_revision:
            raise ValueError("Reply cannot precede the question input revision.")
        return self


class ClarificationInputRevision(ContractValue):
    """Append-only input ancestry; the original request is never rewritten."""

    request: RequestRef
    revision: Generation
    previous_sha256: Commitment
    question: OperationRef
    reply: OperationRef
    reply_sha256: Commitment
    content_sha256: Commitment
    accepted_at_ms: Millis


class ClarificationLineageUsage(ContractValue):
    """Owner counters include pending reservations; they are not a money ledger."""

    questions: StrictInt = Field(default=0, ge=0, le=MAX_CLARIFICATION_QUESTIONS)
    service_turns: StrictInt = Field(default=0, ge=0, le=MAX_CLARIFICATION_TURNS)
    content_bytes: StrictInt = Field(default=0, ge=0, le=MAX_CLARIFICATION_CONTENT_BYTES)
    pending: StrictInt = Field(default=0, ge=0, le=MAX_CLARIFICATION_PENDING)

    def require_within(self, policy: ClarificationPolicy) -> None:
        if (
            self.questions > policy.max_questions
            or self.service_turns > policy.max_service_turns
            or self.content_bytes > policy.max_content_bytes
            or self.pending > policy.max_pending
        ):
            raise ValueError("Clarification lineage capacity is exhausted.")
