"""Private handoff at the existing resume admission boundary, after its gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from cayu._validation import canonical_durable_json_bytes
from cayu.runtime._invocation_lifecycle import (
    AdmitInvocationCommand,
    InvocationContext,
    InvocationMutationResult,
)
from cayu.runtime._session_continuation import (
    ContinuationConsumption,
    ContinuationService,
    continuation_admission_digest,
    continuation_admission_inputs,
)

if TYPE_CHECKING:
    from cayu.runtime._session_continuation_owner import SessionContinuationOwner


@dataclass(frozen=True)
class _ContinuationResumeHandoff:
    owner: SessionContinuationOwner
    service: ContinuationService
    service_digest: str

    @property
    def interaction_id(self) -> str:
        return str(
            uuid5(
                NAMESPACE_URL,
                "cayu:continuation-interaction:" + continuation_admission_key(self),
            )
        )

    @property
    def interaction_started_event_id(self) -> str:
        return str(
            uuid5(
                NAMESPACE_URL,
                "cayu:continuation-interaction-start:" + continuation_admission_key(self),
            )
        )

    @property
    def interaction_started_at(self) -> datetime:
        return datetime.fromisoformat(self.service.accepted_at)

    @property
    def run_operation_id(self) -> str:
        return str(
            uuid5(NAMESPACE_URL, "cayu:continuation-run:" + continuation_admission_key(self))
        )

    @property
    def model_transition_event_id(self) -> str:
        return str(
            uuid5(NAMESPACE_URL, "cayu:continuation-model:" + continuation_admission_key(self))
        )

    async def admit(
        self, command: AdmitInvocationCommand, invocation: InvocationContext
    ) -> InvocationMutationResult:
        input_digest, profile_digest, budget_digest = continuation_admission_inputs(command)
        consumption = ContinuationConsumption(
            ticket=self.service.ticket,
            latch=self.service.latch,
            continuation_id=self.service.continuation_id,
            mode=self.service.mode,
            accepted_at=self.service.accepted_at,
            service_digest=self.service_digest,
            input_digest=input_digest,
            profile_digest=profile_digest,
            budget_digest=budget_digest,
            admission_command_digest=continuation_admission_digest(command),
            admission_expected_run_epoch=command.expected_run_epoch,
            receipt_stage="prepared",
        )
        result, _ = await self.owner.admit(consumption, command, invocation=invocation)
        return result


def continuation_admission_key(handoff: _ContinuationResumeHandoff) -> str:
    ticket = handoff.service.ticket
    return sha256(
        canonical_durable_json_bytes(
            [
                ticket.namespace.namespace_id,
                ticket.registration_key,
                handoff.service.continuation_id,
                handoff.service.mode,
            ],
            "continuation admission identity",
        )
    ).hexdigest()
