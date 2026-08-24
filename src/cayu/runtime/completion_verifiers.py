"""Application-owned deterministic completion-verifier adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank, revalidate_model_input
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.completion_verifier_profiles import (
    CompletionVerifierProfileComponentDeclaration,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    copy_execution_profile_adoption_intent,
)
from cayu.runtime.work_contracts import (
    WORK_CONTRACT_IDENTIFIER_MAX_BYTES,
    WORK_VERIFICATION_LEASE_MAX_SECONDS,
    CompletionProposal,
    CompletionVerifierDecision,
    FrozenWorkContractModel,
    WorkAttempt,
    WorkContract,
)


class CompletionVerifierUnavailable(RuntimeError):
    """The exact deterministic verifier required by a contract is unavailable."""


class CompletionVerifierExecutionError(RuntimeError):
    """A deterministic verifier failed without publishing a completion decision."""


def _bounded_execution_identifier(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) > WORK_CONTRACT_IDENTIFIER_MAX_BYTES
        or len(value.encode("utf-8")) > WORK_CONTRACT_IDENTIFIER_MAX_BYTES
    ):
        raise ValueError(
            f"{field_name} must not exceed {WORK_CONTRACT_IDENTIFIER_MAX_BYTES} UTF-8 bytes."
        )
    return value


class CompletionVerifierRequest(FrozenWorkContractModel):
    """Immutable bounded evidence presented to one deterministic verifier."""

    contract: WorkContract
    attempt: WorkAttempt
    proposal: CompletionProposal

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContract)

    @field_validator("attempt", mode="before")
    @classmethod
    def copy_attempt(cls, value: object) -> object:
        return revalidate_model_input(value, WorkAttempt)

    @field_validator("proposal", mode="before")
    @classmethod
    def copy_proposal(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionProposal)

    @model_validator(mode="after")
    def validate_authority_chain(self) -> CompletionVerifierRequest:
        reference = self.contract.reference()
        if self.attempt.contract != reference or self.proposal.contract != reference:
            raise ValueError("Verifier context conflicts with its work contract.")
        if self.proposal.attempt_id != self.attempt.attempt_id:
            raise ValueError("Verifier proposal belongs to another work attempt.")
        if self.proposal.task_id != self.attempt.task_id:
            raise ValueError("Verifier proposal belongs to another task.")
        return self


class CompletionVerifierExecutionRequest(FrozenWorkContractModel):
    """Serializable authority for one bounded deterministic verifier execution."""

    proposal_id: str
    claim_id: str
    decision_id: str
    worker_id: str
    lease_seconds: StrictInt = Field(
        default=300,
        ge=1,
        le=WORK_VERIFICATION_LEASE_MAX_SECONDS,
    )
    execution_timeout_seconds: StrictFloat = Field(
        default=30.0,
        gt=0,
        le=WORK_VERIFICATION_LEASE_MAX_SECONDS,
    )
    profile_adoption: ExecutionProfileAdoptionIntent | None = None

    @field_validator("proposal_id", "claim_id", "decision_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_execution_identifier(value, info.field_name)

    @model_validator(mode="after")
    def validate_timeout_within_lease(self) -> CompletionVerifierExecutionRequest:
        if self.execution_timeout_seconds >= self.lease_seconds:
            raise ValueError("execution_timeout_seconds must be shorter than lease_seconds.")
        return self

    @field_validator("profile_adoption", mode="before")
    @classmethod
    def copy_profile_adoption(cls, value: object) -> ExecutionProfileAdoptionIntent | None:
        if value is None:
            return None
        if type(value) is not ExecutionProfileAdoptionIntent:
            raise TypeError("profile_adoption must be an ExecutionProfileAdoptionIntent or None.")
        return copy_execution_profile_adoption_intent(value)


class DeterministicCompletionVerifier(ABC):
    """Read-only application policy resolved from a durable verifier reference.

    The runtime may repeat ``verify`` after a crash before decision publication.
    Implementations must therefore be deterministic and side-effect-free. Any
    necessary external mutation belongs behind Cayu's ordinary effect contracts.
    """

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return stable application-versioned identity for verifier behavior."""

        return None

    @property
    def execution_profile_components(
        self,
    ) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
        """Return stable identities for other decision-bearing dependencies."""

        return ()

    @abstractmethod
    async def verify(self, request: CompletionVerifierRequest) -> CompletionVerifierDecision:
        """Evaluate one completion proposal without choosing durable authority."""


def copy_completion_verifier_request(
    value: CompletionVerifierRequest,
) -> CompletionVerifierRequest:
    if type(value) is not CompletionVerifierRequest:
        raise TypeError("Verifier evaluation requires a CompletionVerifierRequest.")
    return cast(
        "CompletionVerifierRequest",
        revalidate_model_input(value, CompletionVerifierRequest),
    )


def copy_completion_verifier_execution_request(
    value: CompletionVerifierExecutionRequest,
) -> CompletionVerifierExecutionRequest:
    if type(value) is not CompletionVerifierExecutionRequest:
        raise TypeError("Verifier execution requires a CompletionVerifierExecutionRequest.")
    return CompletionVerifierExecutionRequest(
        proposal_id=value.proposal_id,
        claim_id=value.claim_id,
        decision_id=value.decision_id,
        worker_id=value.worker_id,
        lease_seconds=value.lease_seconds,
        execution_timeout_seconds=value.execution_timeout_seconds,
        profile_adoption=(
            None
            if value.profile_adoption is None
            else copy_execution_profile_adoption_intent(value.profile_adoption)
        ),
    )


__all__ = [
    "CompletionVerifierExecutionError",
    "CompletionVerifierExecutionRequest",
    "CompletionVerifierRequest",
    "CompletionVerifierUnavailable",
    "DeterministicCompletionVerifier",
]
