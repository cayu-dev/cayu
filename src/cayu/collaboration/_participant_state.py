"""Authoritative mutable permit accounting, distinct from historical receipts."""

from __future__ import annotations

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue
from cayu.collaboration.participants import Counter, ParticipantRef


class ParticipantPermitState(ContractValue):
    reference: ParticipantRef
    issued_frontier: Counter
    outstanding: Counter

    @model_validator(mode="after")
    def bounded_outstanding(self) -> ParticipantPermitState:
        if self.outstanding > self.issued_frontier:
            raise ValueError("Permit accounting exceeds issued authority.")
        return self
