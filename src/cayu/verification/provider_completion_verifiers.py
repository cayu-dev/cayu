"""Application-owned provider-backed (model-judge) completion verifiers.

An application declares the provider target, bounded limits and verifier budget,
and builds the model-facing material for one ``CompletionVerifierRequest``. The
runtime owns everything else: the decision contract shown to the model, the
durable dispatch ledger, the provider call, retries, accounting and the strict
decoding of the response. The verifier has no tools and no external effects.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.budgets.pricing import PriceBook, estimate_model_step_cost
from cayu.messages import Message, MessageRole, TextPart
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tasks.completion_verifier_dispatches import (
    COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS_PER_EXECUTION,
    CompletionVerifierDispatch,
    CompletionVerifierDispatchBudget,
    CompletionVerifierDispatchOutcome,
)
from cayu.tasks.completion_verifier_profiles import (
    CompletionVerifierProfileComponentDeclaration,
)
from cayu.tasks.contracts import (
    WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES,
    WORK_CONTRACT_IDENTIFIER_MAX_BYTES,
    WORK_VERIFICATION_LEASE_MAX_SECONDS,
    CompletionConstraintOutcome,
    CompletionCriterionOutcome,
    CompletionDecisionCreate,
    CompletionGap,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerifierDecision,
    CriterionOutcomeStatus,
    FrozenWorkContractModel,
    WorkCompletionConflict,
    WorkEvidenceReference,
    validate_completion_decision_contract,
)
from cayu.tools.inference import MAX_INFERENCE_BYTES
from cayu.verification.completion_verifiers import (
    CompletionVerifierExecutionError,
    CompletionVerifierRequest,
)

PROVIDER_COMPLETION_VERIFIER_DECISION_CONTRACT_VERSION = 1
_MAX_RENDERED_CONTRACT_BYTES = 1024 * 1024


class ProviderCompletionVerifierDispatchError(CompletionVerifierExecutionError):
    """The provider attempt failed, timed out or ended without a usable outcome.

    This is a verifier-execution failure, never a rejected candidate decision.
    """


class ProviderCompletionVerifierDecodingError(CompletionVerifierExecutionError):
    """The provider response did not satisfy the strict decision contract.

    This is a verifier-execution failure, never a rejected candidate decision.
    """


class ProviderCompletionVerifierBudgetExhausted(CompletionVerifierExecutionError):
    """The durable verifier budget for this proposal does not admit another attempt."""


def _identifier(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > WORK_CONTRACT_IDENTIFIER_MAX_BYTES:
        raise ValueError(
            f"{field_name} must not exceed {WORK_CONTRACT_IDENTIFIER_MAX_BYTES} UTF-8 bytes."
        )
    return value


class ProviderCompletionVerifierTarget(FrozenWorkContractModel):
    """Provider target, per-attempt limits and the durable verifier budget.

    ``retry_policy`` bounds provider retries inside one verifier execution.
    ``budget`` bounds every provider attempt made for one proposal, across
    verifier executions, crashes and recoveries. Both are part of the verifier
    execution profile, so changing them is a profile change.
    """

    provider_name: str
    model: str
    max_input_tokens: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    max_output_tokens: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    attempt_timeout_seconds: StrictFloat = Field(gt=0, le=WORK_VERIFICATION_LEASE_MAX_SECONDS)
    max_request_bytes: StrictInt = Field(default=1024 * 1024, gt=0, le=MAX_INFERENCE_BYTES)
    max_response_bytes: StrictInt = Field(default=256 * 1024, gt=0, le=MAX_INFERENCE_BYTES)
    retry_policy: RetryPolicy = Field(
        default_factory=lambda: RetryPolicy(max_attempts=2, max_unknown_attempts=1)
    )
    budget: CompletionVerifierDispatchBudget = Field(
        default_factory=lambda: CompletionVerifierDispatchBudget(max_attempts=4)
    )

    @field_validator("provider_name", "model")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("attempt_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout(cls, value: object) -> object:
        if type(value) is int:
            return float(value)
        return value

    @field_validator("retry_policy", mode="before")
    @classmethod
    def copy_retry_policy(cls, value: object) -> object:
        return revalidate_model_input(value, RetryPolicy)

    @field_validator("budget", mode="before")
    @classmethod
    def copy_budget(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierDispatchBudget)

    @model_validator(mode="after")
    def validate_target(self) -> ProviderCompletionVerifierTarget:
        if self.retry_policy.max_attempts > COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS_PER_EXECUTION:
            raise ValueError(
                "retry_policy.max_attempts must not exceed "
                f"{COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS_PER_EXECUTION}."
            )
        if self.retry_policy.max_attempts > self.budget.max_attempts:
            raise ValueError("retry_policy.max_attempts cannot exceed budget.max_attempts.")
        if (
            self.budget.max_input_tokens is not None
            and self.max_input_tokens > self.budget.max_input_tokens
        ) or (
            self.budget.max_output_tokens is not None
            and self.max_output_tokens > self.budget.max_output_tokens
        ):
            raise ValueError("Per-attempt token limits cannot exceed the verifier token budget.")
        return self


def copy_provider_completion_verifier_target(
    value: ProviderCompletionVerifierTarget,
) -> ProviderCompletionVerifierTarget:
    if type(value) is not ProviderCompletionVerifierTarget:
        raise TypeError("Provider verifier target must be a ProviderCompletionVerifierTarget.")
    return ProviderCompletionVerifierTarget.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


class ProviderCompletionVerifier(ABC):
    """Application policy for one provider-backed completion verifier.

    ``build_messages`` must be a read-only, deterministic function of its
    request: the runtime may call it again after a crash before a provider
    attempt and fingerprints the resulting request. It may read
    application-owned state but must not mutate anything. Messages carry text
    only; the runtime appends the decision contract and rejects tools, raw
    provider options and non-text content.
    """

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return stable application-versioned identity for prompt and policy behavior."""

        return None

    @property
    def execution_profile_components(
        self,
    ) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
        """Return stable identities for other decision-bearing dependencies."""

        return ()

    @property
    @abstractmethod
    def target(self) -> ProviderCompletionVerifierTarget:
        """Return the provider target, limits and verifier budget."""

    @abstractmethod
    async def build_messages(self, request: CompletionVerifierRequest) -> Sequence[Message]:
        """Return the model-facing evaluation material for one proposal."""


def _evidence_key(value: WorkEvidenceReference) -> tuple[object, ...]:
    return (
        value.kind,
        value.reference_id,
        value.requirement_id or "",
        value.version or "",
        value.digest or "",
        value.available,
        value.unavailable_reason or "",
    )


def _citation(index: int) -> str:
    return f"E{index + 1}"


def render_provider_completion_verifier_contract(request: CompletionVerifierRequest) -> str:
    """Render the runtime-owned decision contract appended to the model request."""

    contract = request.contract
    proposal = request.proposal
    document = {
        "decision_contract_version": PROVIDER_COMPLETION_VERIFIER_DECISION_CONTRACT_VERSION,
        "objective": contract.objective,
        "criteria": [
            {
                "id": item.criterion_id,
                "description": item.description,
                "required_evidence": list(item.evidence_requirement_ids),
            }
            for item in contract.criteria
        ],
        "constraints": [
            {
                "id": item.constraint_id,
                "description": item.description,
                "required_evidence": list(item.evidence_requirement_ids),
            }
            for item in contract.constraints
        ],
        "evidence_requirements": [
            {"id": item.requirement_id, "kind": item.kind, "description": item.description}
            for item in contract.evidence_requirements
        ],
        "candidate": {
            "result": {
                "kind": proposal.result.kind,
                "reference_id": proposal.result.reference_id,
                "digest": proposal.result.digest,
            },
            "evidence": [
                {
                    "citation": _citation(index),
                    "kind": item.kind,
                    "reference_id": item.reference_id,
                    "requirement_id": item.requirement_id,
                    "version": item.version,
                    "digest": item.digest,
                    "available": item.available,
                    "unavailable_reason": item.unavailable_reason,
                }
                for index, item in enumerate(proposal.evidence_references)
            ],
        },
    }
    rendered = canonical_durable_json_bytes(document, "provider_verifier_contract").decode()
    if len(rendered.encode("utf-8")) > _MAX_RENDERED_CONTRACT_BYTES:
        raise ValueError("The rendered decision contract exceeds its bound.")
    return (
        "You are an independent completion verifier. Decide whether the candidate "
        "satisfies the work contract below. The candidate result and its evidence "
        "references are claims made by the worker, not proof. Cite evidence only by "
        "its citation token, and only when it supports your judgement.\n\n"
        "Respond with exactly one JSON object and nothing else, with these keys:\n"
        '- "verdict": one of "accepted", "rejected", "blocked", "needs_review".\n'
        '- "criteria": one object per contract criterion, in contract order.\n'
        '- "constraints": one object per contract constraint, in contract order '
        "(an empty list when there are none).\n"
        'Each outcome object has "id", "status" (one of "satisfied", "unsatisfied", '
        '"unverifiable"), "reason_code" (lowercase code such as "tests.passed"), and '
        'optionally "summary" (short text) and "evidence" (a list of citation tokens). '
        'Every outcome that is not "satisfied" must also have "gap_code" (lowercase '
        'code), and a satisfied outcome must not. Use "accepted" only when every '
        "outcome is satisfied; otherwise use another verdict. A satisfied outcome "
        "whose contract entry lists required evidence must cite available evidence "
        "for each listed requirement.\n\n"
        f"Work contract:\n{rendered}"
    )


def compose_provider_completion_verifier_messages(
    messages: Sequence[Message],
    request: CompletionVerifierRequest,
) -> list[Message]:
    """Append the runtime decision contract to application evaluation material."""

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("build_messages must return a sequence of Message values.")
    copied: list[Message] = []
    for message in messages:
        if type(message) is not Message:
            raise TypeError("build_messages must return Message values.")
        if message.role not in {MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT}:
            raise ValueError("Provider verifier messages cannot carry tool turns.")
        if not message.content or any(type(part) is not TextPart for part in message.content):
            raise ValueError("Provider verifier messages must carry text only.")
        copied.append(Message(role=message.role, content=message.content))
    contract_part = TextPart(text=render_provider_completion_verifier_contract(request))
    if copied and copied[-1].role is MessageRole.USER:
        last = copied.pop()
        copied.append(Message(role=MessageRole.USER, content=(*last.content, contract_part)))
    else:
        copied.append(Message(role=MessageRole.USER, content=(contract_part,)))
    return copied


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Decision JSON contains a duplicate key.")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"Decision JSON contains a non-finite number: {value}.")


def _json_document(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if len(lines) < 3 or lines[0] not in {"```", "```json"} or lines[-1] != "```":
            raise ValueError("Decision response must be one JSON object.")
        stripped = "\n".join(lines[1:-1]).strip()
    if not stripped.startswith("{"):
        raise ValueError("Decision response must be one JSON object.")
    return json.loads(
        stripped,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )


_OUTCOME_KEYS = frozenset({"id", "status", "reason_code", "summary", "evidence", "gap_code"})
_REQUIRED_OUTCOME_KEYS = frozenset({"id", "status", "reason_code"})


@dataclass(frozen=True, slots=True)
class _DecodedOutcome:
    subject_id: str
    status: CriterionOutcomeStatus
    reason_code: str
    summary: str | None
    evidence: tuple[WorkEvidenceReference, ...]
    gap_code: str | None


def _decode_outcome(
    value: object,
    *,
    expected_id: str,
    evidence: tuple[WorkEvidenceReference, ...],
) -> _DecodedOutcome:
    if type(value) is not dict:
        raise ValueError("Each decision outcome must be a JSON object.")
    value = cast("dict[str, object]", value)
    keys = frozenset(value)
    if not keys >= _REQUIRED_OUTCOME_KEYS or not keys <= _OUTCOME_KEYS:
        raise ValueError("Decision outcome has missing or unknown keys.")
    if value["id"] != expected_id:
        raise ValueError("Decision outcomes must cover the contract in order.")
    status_value = value["status"]
    if type(status_value) is not str:
        raise ValueError("Decision outcome status must be a string.")
    status = CriterionOutcomeStatus(status_value)
    reason_code = value["reason_code"]
    if type(reason_code) is not str:
        raise ValueError("Decision outcome reason_code must be a string.")
    summary = value.get("summary")
    if summary is not None and type(summary) is not str:
        raise ValueError("Decision outcome summary must be a string.")
    gap_code = value.get("gap_code")
    if gap_code is not None and type(gap_code) is not str:
        raise ValueError("Decision outcome gap_code must be a string.")
    if (status is CriterionOutcomeStatus.SATISFIED) == (gap_code is not None):
        raise ValueError("Exactly the unresolved outcomes must carry a gap_code.")
    citations = value.get("evidence", [])
    if type(citations) is not list:
        raise ValueError("Decision outcome evidence must be a list of citations.")
    if len(citations) > WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES:
        raise ValueError("Decision outcome cites too much evidence.")
    cited: list[WorkEvidenceReference] = []
    seen: set[str] = set()
    for citation in citations:
        if type(citation) is not str or citation in seen:
            raise ValueError("Decision outcome evidence citations must be unique strings.")
        seen.add(citation)
        if (
            not citation.startswith("E")
            or not citation[1:].isascii()
            or not citation[1:].isdigit()
            or citation[1] == "0"
        ):
            raise ValueError("Decision outcome cites unknown evidence.")
        index = int(citation[1:]) - 1
        if index >= len(evidence):
            raise ValueError("Decision outcome cites unknown evidence.")
        cited.append(evidence[index])
    return _DecodedOutcome(
        subject_id=expected_id,
        status=status,
        reason_code=reason_code,
        summary=summary,
        evidence=tuple(sorted(cited, key=_evidence_key)),
        gap_code=gap_code,
    )


def _basis(outcome: _DecodedOutcome) -> CompletionSatisfactionBasis | None:
    if outcome.status is not CriterionOutcomeStatus.SATISFIED:
        return None
    if any(reference.available for reference in outcome.evidence):
        return CompletionSatisfactionBasis.EVIDENCE
    return CompletionSatisfactionBasis.VERIFIER_ASSERTION


def _gap(
    outcome: _DecodedOutcome,
    *,
    kind: str,
    required: tuple[str, ...],
) -> CompletionGap:
    covered = {
        reference.requirement_id
        for reference in outcome.evidence
        if reference.available and reference.requirement_id is not None
    }
    missing = tuple(sorted(item for item in required if item not in covered))
    subject = (
        {"criterion_id": outcome.subject_id}
        if kind == "criterion"
        else {"constraint_id": outcome.subject_id}
    )
    if outcome.gap_code is None:  # pragma: no cover - only unresolved outcomes have gaps
        raise ValueError("Unresolved outcome has no gap code.")
    return CompletionGap(
        **subject,
        code=outcome.gap_code,
        evidence_requirement_ids=missing,
        summary=outcome.summary,
    )


def decode_provider_completion_verifier_decision(
    text: str,
    request: CompletionVerifierRequest,
) -> CompletionVerifierDecision:
    """Strictly decode one model response into a contract-complete decision.

    Every failure raises ``ProviderCompletionVerifierDecodingError``. A response
    that cannot be decoded is never interpreted as a rejected candidate.
    """

    try:
        document = _json_document(text)
        if type(document) is not dict:
            raise ValueError("Decision response must be one JSON object.")
        document = cast("dict[str, object]", document)
        if frozenset(document) != frozenset({"verdict", "criteria", "constraints"}):
            raise ValueError("Decision JSON must contain exactly verdict, criteria, constraints.")
        verdict_value = document["verdict"]
        if type(verdict_value) is not str:
            raise ValueError("Decision verdict must be a string.")
        verdict = CompletionVerdict(verdict_value)
        contract = request.contract
        evidence = request.proposal.evidence_references
        criteria = document["criteria"]
        constraints = document["constraints"]
        if type(criteria) is not list or len(criteria) != len(contract.criteria):
            raise ValueError("Decision must cover every contract criterion exactly once.")
        if type(constraints) is not list or len(constraints) != len(contract.constraints):
            raise ValueError("Decision must cover every contract constraint exactly once.")
        criterion_outcomes = tuple(
            _decode_outcome(value, expected_id=item.criterion_id, evidence=evidence)
            for value, item in zip(criteria, contract.criteria, strict=True)
        )
        constraint_outcomes = tuple(
            _decode_outcome(value, expected_id=item.constraint_id, evidence=evidence)
            for value, item in zip(constraints, contract.constraints, strict=True)
        )
        gaps = tuple(
            sorted(
                (
                    *(
                        _gap(outcome, kind="criterion", required=item.evidence_requirement_ids)
                        for outcome, item in zip(criterion_outcomes, contract.criteria, strict=True)
                        if outcome.gap_code is not None
                    ),
                    *(
                        _gap(outcome, kind="constraint", required=item.evidence_requirement_ids)
                        for outcome, item in zip(
                            constraint_outcomes, contract.constraints, strict=True
                        )
                        if outcome.gap_code is not None
                    ),
                ),
                key=lambda gap: (
                    0 if gap.criterion_id is not None else 1,
                    gap.criterion_id or gap.constraint_id,
                    gap.code,
                    gap.evidence_requirement_ids,
                ),
            )
        )
        decision = CompletionVerifierDecision(
            verdict=verdict,
            criterion_outcomes=tuple(
                CompletionCriterionOutcome(
                    criterion_id=outcome.subject_id,
                    status=outcome.status,
                    reason_code=outcome.reason_code,
                    satisfaction_basis=_basis(outcome),
                    evidence_references=outcome.evidence,
                    summary=outcome.summary,
                )
                for outcome in criterion_outcomes
            ),
            constraint_outcomes=tuple(
                CompletionConstraintOutcome(
                    constraint_id=outcome.subject_id,
                    status=outcome.status,
                    reason_code=outcome.reason_code,
                    satisfaction_basis=_basis(outcome),
                    evidence_references=outcome.evidence,
                    summary=outcome.summary,
                )
                for outcome in constraint_outcomes
            ),
            gaps=gaps,
        )
        validate_completion_decision_contract(
            contract,
            CompletionDecisionCreate(
                decision_id="provider-decision-decoding",
                proposal_id=request.proposal.proposal_id,
                claim_id="provider-decision-decoding",
                worker_id="provider-decision-decoding",
                verifier=contract.verifier,
                verifier_profile_fingerprint="0" * 64,
                verdict=decision.verdict,
                criterion_outcomes=decision.criterion_outcomes,
                constraint_outcomes=decision.constraint_outcomes,
                gaps=decision.gaps,
                evidence_references=decision.evidence_references,
            ),
        )
        return decision
    except (ValueError, TypeError, KeyError, WorkCompletionConflict, RecursionError):
        # Model output can echo prompt material; never retain it in diagnostics.
        raise ProviderCompletionVerifierDecodingError(
            "The provider verifier response does not satisfy the decision contract."
        ) from None


@dataclass(frozen=True, slots=True)
class CompletionVerifierUsageSummary:
    """Verifier-only usage and cost, separate from worker invocation accounting.

    ``unknown_usage_attempts`` counts attempts that are unsettled or settled
    without observed usage; their tokens are not included in the totals.
    ``cost`` is ``None`` when no price book was supplied or any observed attempt
    could not be priced; ``unpriced_attempts`` says how many.
    """

    attempts: int
    settled_attempts: int
    outcomes: dict[str, int]
    input_tokens: int
    output_tokens: int
    unknown_usage_attempts: int
    cost: Decimal | None
    currency: str | None
    unpriced_attempts: int


def summarize_completion_verifier_dispatches(
    dispatches: Sequence[CompletionVerifierDispatch],
    *,
    pricing: PriceBook | None = None,
) -> CompletionVerifierUsageSummary:
    """Summarize one proposal's provider-verifier attempts.

    Pricing is applied per attempt with the price book effective on the day the
    attempt was dispatched, matching how Cayu prices model steps elsewhere.
    """

    outcomes: dict[str, int] = {}
    input_tokens = 0
    output_tokens = 0
    unknown = 0
    settled = 0
    total_cost = Decimal(0)
    currency: str | None = None
    unpriced = 0
    for dispatch in dispatches:
        if type(dispatch) is not CompletionVerifierDispatch:
            raise TypeError("Dispatch summaries require CompletionVerifierDispatch values.")
        settlement = dispatch.settlement
        outcome = (
            "unsettled"
            if settlement is None
            else CompletionVerifierDispatchOutcome(settlement.outcome).value
        )
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if settlement is not None:
            settled += 1
        usage = None if settlement is None else settlement.usage
        if usage is None:
            unknown += 1
            continue
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        if pricing is None:
            continue
        estimate = estimate_model_step_cost(
            metrics=usage,
            pricing=pricing,
            effective_on=dispatch.dispatched_at.date(),
        )
        if not estimate.priced or (currency is not None and estimate.currency != currency):
            unpriced += 1
            continue
        currency = estimate.currency
        total_cost += estimate.total_cost
    return CompletionVerifierUsageSummary(
        attempts=len(dispatches),
        settled_attempts=settled,
        outcomes=dict(sorted(outcomes.items())),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        unknown_usage_attempts=unknown,
        cost=None if pricing is None or unpriced else total_cost,
        currency=None if pricing is None or unpriced else currency,
        unpriced_attempts=unpriced,
    )


__all__ = [
    "PROVIDER_COMPLETION_VERIFIER_DECISION_CONTRACT_VERSION",
    "CompletionVerifierUsageSummary",
    "ProviderCompletionVerifier",
    "ProviderCompletionVerifierBudgetExhausted",
    "ProviderCompletionVerifierDecodingError",
    "ProviderCompletionVerifierDispatchError",
    "ProviderCompletionVerifierTarget",
    "compose_provider_completion_verifier_messages",
    "copy_provider_completion_verifier_target",
    "decode_provider_completion_verifier_decision",
    "render_provider_completion_verifier_contract",
    "summarize_completion_verifier_dispatches",
]
