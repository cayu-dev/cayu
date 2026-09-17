"""Read-only retained responsibility views; none of these values grant execution."""

from typing import Literal

from pydantic import Field, StrictBool, model_validator

from cayu.collaboration._contracts import (
    Code,
    ContractValue,
    Generation,
    Identifier,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration.participants import ParticipantRef


class ParticipantObligation(ContractValue):
    participant: ParticipantRef
    position: Generation
    operation: OperationRef
    source_operation: OperationRef
    settlement_operation: OperationRef
    admission_generation: Generation
    target: ObjectRef
    target_state: Literal["existing", "future"]
    effect_scope: Code
    required_settlement: Literal["exclusion", "quiescence"]
    state: Literal["pending", "settled"]
    outcome: Literal["excluded", "quiescent"] | None

    @model_validator(mode="after")
    def exact_scope_and_settlement(self):
        if (
            any(
                op.application_scope != self.participant.owner.application_scope
                for op in (
                    self.operation,
                    self.source_operation,
                    self.settlement_operation,
                )
            )
            or (self.state == "settled") != (self.outcome is not None)
            or (self.required_settlement == "quiescence" and self.outcome == "excluded")
        ):
            raise ValueError("Obligation projection authority conflicts.")
        return self


class ParticipantObligationCursor(ContractValue):
    scope: Identifier
    principal: Identifier
    participant: ParticipantRef
    pending_only: StrictBool
    retention_revision: Generation
    after_position: Generation


class ParticipantObligationPage(ContractValue):
    obligations: tuple[ParticipantObligation, ...] = Field(max_length=64)
    next_cursor: ParticipantObligationCursor | None
