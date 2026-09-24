"""Durable service responsibility, using the existing participant permit owner."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue
from cayu.collaboration._permits import PermitReceipt, ReceivingSettlementReceipt
from cayu.runtime._session_continuation import continuation_digest
from cayu.runtime._temporary_continuation import TemporaryServiceDispatch


class ClarificationServiceRecord(ContractValue):
    """Discoverable handoff even if no receiving session acknowledgement arrives."""

    dispatch: TemporaryServiceDispatch
    permit: PermitReceipt
    state: Literal["pending", "settled"] = "pending"
    settlement: ReceivingSettlementReceipt | None = None

    @model_validator(mode="after")
    def exact_responsibility(self) -> ClarificationServiceRecord:
        intent = self.dispatch.intent
        expected = self.permit.expected
        registration = expected.intent.request
        if (
            registration.source_operation != intent.operation
            or registration.target != intent.target
            or registration.target_state != "existing"
            or registration.participant != intent.question.responder
            or registration.expected_configuration_revision is None
            or expected.initiator != intent.initiator
            or registration.admission_commitment != continuation_digest(self.dispatch)
            or registration.effect_scope != "clarification_service"
            or registration.required_settlement != "quiescence"
            or (self.state == "settled") != (self.settlement is not None)
            or (self.settlement is not None and self.settlement.expected != expected)
        ):
            raise ValueError("Clarification service responsibility conflicts.")
        return self
