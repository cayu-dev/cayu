"""Bounded maintenance selectors; none grants execution or source disclosure."""

from typing import Literal

from pydantic import Field, StrictInt, model_validator

from cayu.collaboration._contracts import ContractValue, Identifier, OperationRef
from cayu.collaboration.clarifications import ClarificationDueCursor, Commitment


class ClarificationServiceRecovery(ContractValue):
    """Exact retained dispatch commitment, not a reconstructed service instruction.

    Recovery compares both commitments against the native and collaboration
    owners' full immutable tuples. This selector only permits reconciliation;
    it cannot be submitted as a service request or used for admission.
    """

    kind: Literal["clarification_service_recovery"] = "clarification_service_recovery"
    operation: OperationRef
    session_id: Identifier
    session_instance_id: Identifier
    selection_sha256: Commitment
    dispatch_sha256: Commitment


class ClarificationPendingService(ContractValue):
    recovery: ClarificationServiceRecovery
    question: OperationRef
    deadline_at_ms: StrictInt = Field(ge=1, le=2**53 - 1)


class ClarificationPendingServicePage(ContractValue):
    items: tuple[ClarificationPendingService, ...] = Field(max_length=64)
    next_cursor: ClarificationDueCursor | None


class ClarificationServiceInspection(ClarificationPendingService):
    state: Literal["prepared", "reserved", "admitted", "returned", "excluded"]


class ClarificationServiceInspectionPage(ContractValue):
    items: tuple[ClarificationServiceInspection, ...] = Field(max_length=64)
    next_cursor: ClarificationDueCursor | None


class ClarificationPendingServiceQuery(ContractValue):
    cursor: ClarificationDueCursor | None = None
    limit: StrictInt = Field(default=32, ge=1, le=64)


class ClarificationDeliveryRecovery(ContractValue):
    """Expected complete delivery identity; never an append or disclosure grant."""

    kind: Literal["clarification_delivery_recovery"] = "clarification_delivery_recovery"
    operation: OperationRef
    intent_sha256: Commitment


class ClarificationPendingDelivery(ContractValue):
    recovery: ClarificationDeliveryRecovery
    question: OperationRef
    deadline_at_ms: StrictInt = Field(ge=1, le=2**53 - 1)


class ClarificationPendingDeliveryPage(ContractValue):
    items: tuple[ClarificationPendingDelivery, ...] = Field(max_length=64)
    next_cursor: ClarificationDueCursor | None


class ClarificationQuestionRecovery(ContractValue):
    """Exact historical question identity, never source disclosure authority."""

    kind: Literal["clarification_question_recovery"] = "clarification_question_recovery"
    operation: OperationRef
    question_sha256: Commitment
    request_sha256: Commitment


class ClarificationDueQuestion(ContractValue):
    recovery: ClarificationQuestionRecovery
    deadline_at_ms: StrictInt = Field(ge=1, le=2**53 - 1)


class ClarificationDueQuestionPage(ContractValue):
    items: tuple[ClarificationDueQuestion, ...] = Field(max_length=64)
    next_cursor: ClarificationDueCursor | None


class ClarificationExpiryRequest(ContractValue):
    """Stable maintenance operation; retries must preserve both exact identities."""

    operation: OperationRef
    recovery: ClarificationQuestionRecovery

    @model_validator(mode="after")
    def coherent(self):
        question = self.recovery.operation
        if self.operation == question or (
            self.operation.application_scope,
            self.operation.namespace_incarnation,
            self.operation.generation,
        ) != (question.application_scope, question.namespace_incarnation, question.generation):
            raise ValueError("Clarification expiry requires a distinct operation in its namespace.")
        return self


class ClarificationExpiryReceipt(ContractValue):
    """Decision settlement only, not provider/tool quiescence or delivery exclusion."""

    expected: ClarificationExpiryRequest
    status: Literal["expired"] = "expired"
    event_sequence: StrictInt = Field(ge=1, le=2**53 - 1)
    expired_at_ms: StrictInt = Field(ge=1, le=2**53 - 1)
