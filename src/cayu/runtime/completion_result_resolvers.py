"""Application-owned adapters for reconstructing accepted completion results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

from pydantic import Field, StrictFloat, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank, revalidate_model_input
from cayu.runtime.work_contracts import (
    WORK_CONTRACT_IDENTIFIER_MAX_BYTES,
    CompletionDecision,
    CompletionProposal,
    CompletionResultReference,
    CompletionVerdict,
    FrozenWorkContractModel,
    WorkAttempt,
    WorkContract,
    copy_completion_decision,
    copy_completion_proposal,
    copy_work_attempt,
    copy_work_contract,
    validate_work_completion_idempotency_key,
    validate_work_completion_linked_id,
)

COMPLETION_RESULT_RESOLUTION_MAX_SECONDS = 300.0


def _bounded_resolution_identifier(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) > WORK_CONTRACT_IDENTIFIER_MAX_BYTES
        or len(value.encode("utf-8")) > WORK_CONTRACT_IDENTIFIER_MAX_BYTES
    ):
        raise ValueError(
            f"{field_name} must not exceed {WORK_CONTRACT_IDENTIFIER_MAX_BYTES} UTF-8 bytes."
        )
    return value


class CompletionResultResolverUnavailable(RuntimeError):
    """The exact result resolver required by a work contract is unavailable."""


class CompletionResultUnavailable(RuntimeError):
    """The accepted result can no longer be reconstructed by its resolver."""


class CompletionResultResolverExecutionError(RuntimeError):
    """A result resolver failed before producing validated result content."""


class CompletionResultResolverRequest(FrozenWorkContractModel):
    """Detached immutable authority presented to one exact result resolver."""

    contract: WorkContract
    attempt: WorkAttempt
    proposal: CompletionProposal
    decision: CompletionDecision
    result_reference: CompletionResultReference

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        if type(value) is not WorkContract:
            return revalidate_model_input(value, WorkContract)
        return copy_work_contract(value)

    @field_validator("attempt", mode="before")
    @classmethod
    def copy_attempt(cls, value: object) -> object:
        if type(value) is not WorkAttempt:
            return revalidate_model_input(value, WorkAttempt)
        return copy_work_attempt(value)

    @field_validator("proposal", mode="before")
    @classmethod
    def copy_proposal(cls, value: object) -> object:
        if type(value) is not CompletionProposal:
            return revalidate_model_input(value, CompletionProposal)
        return copy_completion_proposal(value)

    @field_validator("decision", mode="before")
    @classmethod
    def copy_decision(cls, value: object) -> object:
        if type(value) is not CompletionDecision:
            return revalidate_model_input(value, CompletionDecision)
        return copy_completion_decision(value)

    @field_validator("result_reference", mode="before")
    @classmethod
    def copy_result_reference(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionResultReference)

    @model_validator(mode="after")
    def validate_authority_chain(self) -> CompletionResultResolverRequest:
        contract_ref = self.contract.reference()
        if (
            self.attempt.contract != contract_ref
            or self.proposal.contract != contract_ref
            or self.decision.contract != contract_ref
        ):
            raise ValueError("Result-resolution context conflicts with its work contract.")
        if (
            self.proposal.attempt_id != self.attempt.attempt_id
            or self.decision.attempt_id != self.attempt.attempt_id
        ):
            raise ValueError("Result-resolution evidence belongs to another work attempt.")
        if (
            self.proposal.task_id != self.attempt.task_id
            or self.decision.task_id != self.attempt.task_id
        ):
            raise ValueError("Result-resolution evidence belongs to another task.")
        if self.decision.proposal_id != self.proposal.proposal_id:
            raise ValueError("Result-resolution decision belongs to another proposal.")
        if self.decision.verifier != self.contract.verifier:
            raise ValueError("Result-resolution decision used another completion verifier.")
        if self.decision.verdict is not CompletionVerdict.ACCEPTED:
            raise ValueError("Only an accepted completion decision can resolve a result.")
        if self.proposal.result != self.result_reference:
            raise ValueError("Result-resolution reference conflicts with the accepted proposal.")
        return self


class CompletionResultResolutionRequest(FrozenWorkContractModel):
    """Serializable authority for one bounded accepted-result resolution."""

    task_id: str
    decision_id: str
    idempotency_key: str
    execution_timeout_seconds: StrictFloat = Field(
        default=30.0,
        gt=0,
        le=COMPLETION_RESULT_RESOLUTION_MAX_SECONDS,
    )

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return validate_work_completion_linked_id(value, "task_id")

    @field_validator("decision_id")
    @classmethod
    def validate_decision_id(cls, value: str) -> str:
        return _bounded_resolution_identifier(value, "decision_id")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return validate_work_completion_idempotency_key(value)


class CompletionResultResolver(ABC):
    """Application-owned resolver selected by an exact durable contract identity.

    Implementations may read external result storage. Cayu may repeat ``resolve``
    after process loss before application acknowledgement, so adapters must bind
    reads to the supplied immutable reference and avoid unrelated side effects.
    """

    @abstractmethod
    async def resolve(self, request: CompletionResultResolverRequest) -> dict[str, object]:
        """Return the complete result whose digest matches ``result_reference``."""


def copy_completion_result_resolver_request(
    value: CompletionResultResolverRequest,
) -> CompletionResultResolverRequest:
    if type(value) is not CompletionResultResolverRequest:
        raise TypeError("Result resolution requires a CompletionResultResolverRequest.")
    return cast(
        "CompletionResultResolverRequest",
        revalidate_model_input(value, CompletionResultResolverRequest),
    )


def copy_completion_result_resolution_request(
    value: CompletionResultResolutionRequest,
) -> CompletionResultResolutionRequest:
    if type(value) is not CompletionResultResolutionRequest:
        raise TypeError("Result resolution requires a CompletionResultResolutionRequest.")
    return cast(
        "CompletionResultResolutionRequest",
        revalidate_model_input(value, CompletionResultResolutionRequest),
    )


__all__ = [
    "COMPLETION_RESULT_RESOLUTION_MAX_SECONDS",
    "CompletionResultResolutionRequest",
    "CompletionResultResolver",
    "CompletionResultResolverExecutionError",
    "CompletionResultResolverRequest",
    "CompletionResultResolverUnavailable",
    "CompletionResultUnavailable",
]
