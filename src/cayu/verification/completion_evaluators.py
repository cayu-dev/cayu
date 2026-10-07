"""Application-owned independent evaluators for verified-work acceptance.

An evaluator performs an expensive, effectful, possibly non-deterministic
evaluation of a candidate (a benchmark run, a test suite, a canary probe or an
external grader) independently of the agent that produced it. Cayu runs it once
per durable effect identity under the verification claim, records an immutable
receipt, and gives that receipt to the verifier as trusted evidence. The
evaluator produces evidence only: it cannot change the contract, the candidate,
the verifier or the task result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field, StrictInt, field_validator, model_validator

from cayu._validation import require_durable_nonblank, revalidate_model_input
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tasks.completion_evaluations import (
    COMPLETION_EVALUATION_EVIDENCE_MAX_BYTES,
    COMPLETION_EVALUATION_SUMMARY_MAX_BYTES,
    COMPLETION_EVALUATION_USAGE_MAX_BYTES,
    COMPLETION_EVALUATION_USAGE_MAX_ITEMS,
    _bounded_document,
)
from cayu.tasks.contracts import (
    WORK_EVALUATION_MAX_RUNS,
    CompletionProposal,
    FrozenWorkContractModel,
    WorkAttempt,
    WorkContract,
)
from cayu.verification.completion_verifiers import (
    CompletionVerifierExecutionError,
    CompletionVerifierUnavailable,
)


class CompletionEvaluatorUnavailable(CompletionVerifierUnavailable):
    """The exact evaluator required by a work contract is unavailable."""


class CompletionEvaluationExecutionError(CompletionVerifierExecutionError):
    """An evaluation run failed, timed out or ended without a receipt.

    This is an evaluation-execution failure, never a rejected candidate.
    """


class CompletionEvaluatorBudgetExhausted(CompletionEvaluationExecutionError):
    """The contract's evaluator-run budget for this proposal is exhausted."""


class CompletionEvaluationRequest(FrozenWorkContractModel):
    """Detached authority presented to one evaluator run.

    ``effect_id`` is stable for this run across processes and claims. Use it as
    the idempotency key of the external effect so ``reconcile`` can find a run
    an earlier owner started.
    """

    effect_id: str
    run_ordinal: StrictInt = Field(ge=1, le=WORK_EVALUATION_MAX_RUNS)
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
    def validate_authority_chain(self) -> CompletionEvaluationRequest:
        reference = self.contract.reference()
        if self.attempt.contract != reference or self.proposal.contract != reference:
            raise ValueError("Evaluation context conflicts with its work contract.")
        if self.proposal.attempt_id != self.attempt.attempt_id:
            raise ValueError("Evaluation proposal belongs to another work attempt.")
        if self.contract.evaluation is None:
            raise ValueError("The work contract declares no evaluation.")
        return self


class CompletionEvaluationResult(FrozenWorkContractModel):
    """Evidence from one evaluator run.

    ``evidence`` is a bounded JSON object the verifier will judge, for example
    ``{"pass_k": 0.82, "tasks": 50}``. ``reported_usage`` is the evaluator's own
    accounting (cost, task or token counts) and is recorded as reported.
    """

    evidence: dict[str, object]
    summary: str | None = None
    reported_usage: dict[str, object] | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        return _bounded_document(
            value,
            field_name="evidence",
            max_bytes=COMPLETION_EVALUATION_EVIDENCE_MAX_BYTES,
            max_items=8_192,
        )

    @field_validator("reported_usage", mode="before")
    @classmethod
    def copy_reported_usage(cls, value: object) -> object:
        if value is None:
            return None
        return _bounded_document(
            value,
            field_name="reported_usage",
            max_bytes=COMPLETION_EVALUATION_USAGE_MAX_BYTES,
            max_items=COMPLETION_EVALUATION_USAGE_MAX_ITEMS,
        )

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_durable_nonblank(value, "summary")
        if len(value.encode("utf-8")) > COMPLETION_EVALUATION_SUMMARY_MAX_BYTES:
            raise ValueError(
                f"summary must not exceed {COMPLETION_EVALUATION_SUMMARY_MAX_BYTES} bytes."
            )
        return value


class CompletionEvaluator(ABC):
    """Application-owned independent evaluator resolved from a durable reference."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return stable application-versioned identity for evaluator behavior."""

        return None

    @abstractmethod
    async def evaluate(self, request: CompletionEvaluationRequest) -> CompletionEvaluationResult:
        """Run the evaluation for ``request.effect_id`` and return its evidence."""

    async def reconcile(
        self, request: CompletionEvaluationRequest
    ) -> CompletionEvaluationResult | None:
        """Return the outcome of an effect an earlier owner started, if known.

        Cayu calls this for a run whose intent is durable but whose outcome was
        never recorded (process loss, lost acknowledgement). Return ``None``
        when there is no evidence the effect completed; Cayu then records the
        run as ``outcome_unknown`` and starts the next run if the budget allows.
        """

        del request
        return None


def copy_completion_evaluation_result(value: object) -> CompletionEvaluationResult:
    if type(value) is not CompletionEvaluationResult:
        raise TypeError("Evaluators must return a CompletionEvaluationResult.")
    return CompletionEvaluationResult.model_validate(
        {
            "evidence": value.evidence,
            "summary": value.summary,
            "reported_usage": value.reported_usage,
        }
    )


__all__ = [
    "CompletionEvaluationExecutionError",
    "CompletionEvaluationRequest",
    "CompletionEvaluationResult",
    "CompletionEvaluator",
    "CompletionEvaluatorBudgetExhausted",
    "CompletionEvaluatorUnavailable",
]
