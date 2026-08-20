from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu import (
    ComparableGenerationSettings,
    ComparableOutputBudget,
    ComparisonCostLineItem,
    ComparisonPricingCatalog,
    CostAccountingTotals,
    CostDirection,
    CostQualityAttemptOperation,
    CostQualityComparisonStatus,
    CostQualityFindingCode,
    PairedCostAttempt,
    PairedCostQualityComparisonRequest,
    PairedCostQualityPair,
    PairedCostQualitySide,
    PairedQualityEvidence,
    QualityEvidenceReference,
    QualityEvidenceStatus,
    SavingsPercentageState,
    compare_paired_cost_quality,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core.billing import BillingIdentity
from cayu.core.events import Event, EventType
from cayu.runtime.costs import (
    ModelPrice,
    PriceBook,
    PriceSchedule,
    PriceTier,
    Provenance,
    TieredPricing,
    estimate_session_cost,
)


def _price_book(*, model: str = "model", currency: str = "USD") -> PriceBook:
    return PriceBook(
        price_book_version="fixture-v1",
        generated_at="2026-08-17T00:00:00Z",
        prices=(
            ModelPrice(
                provider_name="provider",
                model=model,
                match="exact",
                schedules=(
                    PriceSchedule(
                        pricing=TieredPricing(
                            currency=currency,
                            standard=(
                                PriceTier(
                                    input_per_million=Decimal("1000000"),
                                    output_per_million=Decimal("0"),
                                ),
                            ),
                        ),
                        provenance=Provenance(
                            source="fixture price",
                            url="https://example.invalid/pricing",
                            as_of="2026-08-17",
                        ),
                    ),
                ),
            ),
        ),
    )


def _cost(*, session_id: str, input_tokens: int, price_book: PriceBook | None = None):
    selected = price_book or _price_book()
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        payload={
            "usage_metrics": {
                "provider_name": "provider",
                "model": "model",
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": input_tokens,
                },
            }
        },
    )
    return estimate_session_cost(
        session_id=session_id,
        events=[event],
        pricing=selected,
        currency=selected.prices[0].schedules[0].pricing.currency,
    ).line_items[0]


def _quality(
    *,
    status: QualityEvidenceStatus = QualityEvidenceStatus.PASSED,
    score: Decimal | None = Decimal("0.9"),
    threshold: Decimal = Decimal("0.8"),
    version: str = "v1",
) -> PairedQualityEvidence:
    return PairedQualityEvidence(
        contract_name="semantic-equivalence",
        contract_version=version,
        threshold=threshold,
        role="final-answer",
        status=status,
        score=score,
        references=(
            QualityEvidenceReference(
                kind="artifact",
                reference="sha256:" + "1" * 64,
            ),
        ),
    )


def _attempt(
    *,
    attempt_id: str,
    session_id: str,
    input_tokens: int,
    operation: CostQualityAttemptOperation = CostQualityAttemptOperation.AGENT_STEP,
    operation_id: str = "answer",
    attempt_ordinal: int = 1,
    branch_id: str | None = "answer-branch",
    cost=None,
) -> PairedCostAttempt:
    return PairedCostAttempt(
        attempt_id=attempt_id,
        operation_id=operation_id,
        session_id=session_id,
        branch_id=branch_id,
        role="final-answer",
        operation=operation,
        attempt_ordinal=attempt_ordinal,
        provider_name="provider",
        model="model",
        source_checkpoint_id="checkpoint-1",
        cost=_cost(session_id=session_id, input_tokens=input_tokens) if cost is None else cost,
    )


def _side(
    *,
    strategy_id: str,
    attempts: tuple[PairedCostAttempt, ...],
    quality: PairedQualityEvidence | None = None,
    workload_id: str = "workload-1",
    task_id: str = "task-1",
    source_id: str = "source-1",
    output_tokens: int = 100,
    generation_settings: ComparableGenerationSettings | None = None,
    pricing_catalog: ComparisonPricingCatalog | None = None,
) -> PairedCostQualitySide:
    return PairedCostQualitySide(
        strategy_id=strategy_id,
        workload_id=workload_id,
        task_id=task_id,
        source_id=source_id,
        role="final-answer",
        output_budget=ComparableOutputBudget(max_output_tokens=output_tokens),
        generation_settings=generation_settings
        or ComparableGenerationSettings(revision="sha256:" + "2" * 64),
        pricing_catalog=pricing_catalog
        or ComparisonPricingCatalog(
            price_book_version="fixture-v1",
            generated_at="2026-08-17T00:00:00Z",
        ),
        quality=quality if quality is not None else _quality(),
        attempts=attempts,
    )


def _assert_totals_recomputed(
    attempts: tuple[PairedCostAttempt, ...],
    totals: CostAccountingTotals,
) -> None:
    costs = [attempt.cost for attempt in attempts if attempt.cost is not None]
    priced = [cost for cost in costs if cost.priced]
    unpriced = [cost for cost in costs if not cost.priced]
    assert totals.attempt_count == len(attempts)
    assert totals.first_attempt_count == sum(attempt.attempt_ordinal == 1 for attempt in attempts)
    assert totals.retry_attempt_count == sum(attempt.attempt_ordinal > 1 for attempt in attempts)
    assert totals.priced_attempt_count == len(priced)
    assert totals.unpriced_attempt_count == len(unpriced)
    assert totals.missing_usage_attempt_count == len(attempts) - len(costs)
    for field_name in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "uncached_input_tokens",
    ):
        assert getattr(totals, field_name) == sum(getattr(cost, field_name) for cost in costs)
    expected_costs: dict[str, Decimal] = {}
    for cost in priced:
        expected_costs[cost.currency] = (
            expected_costs.get(cost.currency, Decimal("0")) + cost.total_cost
        )
    assert {item.currency: item.priced_cost for item in totals.currencies} == expected_costs


def _pair(
    *,
    pair_id: str = "pair-1",
    baseline_tokens: int = 100,
    candidate_tokens: int = 50,
    baseline_quality: PairedQualityEvidence | None = None,
    candidate_quality: PairedQualityEvidence | None = None,
) -> PairedCostQualityPair:
    return PairedCostQualityPair(
        pair_id=pair_id,
        baseline=_side(
            strategy_id="baseline",
            attempts=(
                _attempt(
                    attempt_id=f"{pair_id}-baseline",
                    session_id=f"{pair_id}-baseline-session",
                    input_tokens=baseline_tokens,
                ),
            ),
            quality=baseline_quality,
        ),
        candidate=_side(
            strategy_id="candidate",
            attempts=(
                _attempt(
                    attempt_id=f"{pair_id}-candidate",
                    session_id=f"{pair_id}-candidate-session",
                    input_tokens=candidate_tokens,
                ),
            ),
            quality=candidate_quality,
        ),
    )


def _fully_attributed_pair() -> PairedCostQualityPair:
    return PairedCostQualityPair(
        pair_id="paired",
        baseline=_side(
            strategy_id="baseline",
            attempts=(
                _attempt(
                    attempt_id="baseline-first",
                    session_id="baseline-answer",
                    input_tokens=100,
                ),
                _attempt(
                    attempt_id="baseline-retry",
                    session_id="baseline-answer",
                    input_tokens=20,
                    attempt_ordinal=2,
                ),
                _attempt(
                    attempt_id="baseline-eval",
                    session_id="baseline-eval",
                    input_tokens=30,
                    operation=CostQualityAttemptOperation.EVALUATION,
                    operation_id="evaluation",
                    branch_id=None,
                ),
            ),
        ),
        candidate=_side(
            strategy_id="candidate",
            attempts=(
                _attempt(
                    attempt_id="candidate-first",
                    session_id="candidate-answer",
                    input_tokens=60,
                ),
                _attempt(
                    attempt_id="candidate-repair",
                    session_id="candidate-repair",
                    input_tokens=15,
                    operation=CostQualityAttemptOperation.STRUCTURED_OUTPUT_REPAIR,
                    operation_id="structured-repair",
                    branch_id=None,
                ),
            ),
        ),
    )


def test_verified_pair_recomputes_attempt_operation_session_and_branch_totals() -> None:
    pair = _fully_attributed_pair()

    report = compare_paired_cost_quality(PairedCostQualityComparisonRequest(pairs=(pair,)))

    assert report.status is CostQualityComparisonStatus.VERIFIED
    result = report.pairs[0]
    assert result.baseline_cost == Decimal("150")
    assert result.candidate_cost == Decimal("75")
    assert result.savings == Decimal("75")
    assert result.savings_percentage == Decimal("50.00")
    assert result.cost_direction is CostDirection.SAVINGS
    assert result.baseline is not None
    assert result.baseline.whole_harness.attempt_count == 3
    assert result.baseline.whole_harness.first_attempt_count == 2
    assert result.baseline.whole_harness.retry_attempt_count == 1
    assert [attempt.attempt_id for attempt in result.baseline.attempts] == [
        "baseline-first",
        "baseline-retry",
        "baseline-eval",
    ]
    assert {attempt.model for attempt in result.baseline.attempts} == {"model"}
    assert {attempt.source_checkpoint_id for attempt in result.baseline.attempts} == {
        "checkpoint-1"
    }
    assert [item.operation for item in result.baseline.operations] == [
        CostQualityAttemptOperation.AGENT_STEP,
        CostQualityAttemptOperation.EVALUATION,
    ]
    assert [item.session_id for item in result.baseline.sessions] == [
        "baseline-answer",
        "baseline-eval",
    ]
    assert [item.branch_id for item in result.baseline.branches] == ["answer-branch"]
    assert result.baseline.branches[0].totals.currencies[0].priced_cost == Decimal("120")
    assert [item.source for item in result.baseline.pricing_provenance] == ["fixture price"]
    assert report.aggregate.eligible_pair_ids == ("paired",)
    _assert_totals_recomputed(result.baseline.attempts, result.baseline.whole_harness)
    assert result.candidate is not None
    _assert_totals_recomputed(result.candidate.attempts, result.candidate.whole_harness)
    for operation in result.baseline.operations:
        _assert_totals_recomputed(
            tuple(
                attempt
                for attempt in result.baseline.attempts
                if attempt.operation is operation.operation
            ),
            operation.totals,
        )
    for session in result.baseline.sessions:
        _assert_totals_recomputed(
            tuple(
                attempt
                for attempt in result.baseline.attempts
                if attempt.session_id == session.session_id
            ),
            session.totals,
        )
    for branch in result.baseline.branches:
        _assert_totals_recomputed(
            tuple(
                attempt
                for attempt in result.baseline.attempts
                if attempt.branch_id == branch.branch_id
            ),
            branch.totals,
        )


def test_every_operation_kind_has_a_separate_accounting_bucket() -> None:
    operations = tuple(CostQualityAttemptOperation)
    baseline_attempts = tuple(
        _attempt(
            attempt_id=f"baseline-{operation.value}",
            session_id=f"baseline-{operation.value}",
            input_tokens=index,
            operation=operation,
            operation_id=f"baseline-{operation.value}",
        )
        for index, operation in enumerate(operations, start=1)
    )
    candidate_attempts = tuple(
        _attempt(
            attempt_id=f"candidate-{operation.value}",
            session_id=f"candidate-{operation.value}",
            input_tokens=index,
            operation=operation,
            operation_id=f"candidate-{operation.value}",
        )
        for index, operation in enumerate(operations, start=1)
    )
    report = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(
                PairedCostQualityPair(
                    pair_id="operations",
                    baseline=_side(strategy_id="baseline", attempts=baseline_attempts),
                    candidate=_side(strategy_id="candidate", attempts=candidate_attempts),
                ),
            )
        )
    )

    pair = report.pairs[0]
    assert pair.status is CostQualityComparisonStatus.VERIFIED
    assert pair.candidate is not None
    assert {item.operation for item in pair.candidate.operations} == set(operations)
    assert all(item.totals.attempt_count == 1 for item in pair.candidate.operations)


def test_status_precedence_and_exclusions_keep_raw_measured_costs() -> None:
    verified = _pair(pair_id="verified")
    measured = _pair(
        pair_id="measured",
        candidate_quality=_quality(
            status=QualityEvidenceStatus.FAILED,
            score=Decimal("0.4"),
        ),
    )
    unpriced_cost = _cost(
        session_id="unpriced-candidate",
        input_tokens=50,
        price_book=_price_book(model="different-model"),
    )
    unpriced = PairedCostQualityPair(
        pair_id="unpriced",
        baseline=_side(
            strategy_id="baseline",
            attempts=(
                _attempt(
                    attempt_id="unpriced-baseline",
                    session_id="unpriced-baseline",
                    input_tokens=100,
                ),
            ),
        ),
        candidate=_side(
            strategy_id="candidate",
            attempts=(
                _attempt(
                    attempt_id="unpriced-candidate",
                    session_id="unpriced-candidate",
                    input_tokens=50,
                    cost=unpriced_cost,
                ),
            ),
        ),
    )
    unavailable = PairedCostQualityPair(
        pair_id="unavailable",
        baseline=_pair().baseline,
        candidate=None,
    )

    report = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(pairs=(unpriced, verified, unavailable, measured))
    )

    assert [pair.pair_id for pair in report.pairs] == [
        "measured",
        "unavailable",
        "unpriced",
        "verified",
    ]
    assert [pair.status for pair in report.pairs] == [
        CostQualityComparisonStatus.MEASURED_UNMATCHED,
        CostQualityComparisonStatus.UNAVAILABLE,
        CostQualityComparisonStatus.UNPRICED,
        CostQualityComparisonStatus.VERIFIED,
    ]
    measured_result = report.pairs[0]
    assert measured_result.baseline_cost == Decimal("100")
    assert measured_result.candidate_cost == Decimal("50")
    assert measured_result.savings_percentage == Decimal("50.00")
    assert report.status is CostQualityComparisonStatus.UNAVAILABLE
    assert report.aggregate.eligible_pair_ids == ("verified",)
    assert [item.pair_id for item in report.aggregate.exclusions] == [
        "measured",
        "unavailable",
        "unpriced",
    ]
    assert report.aggregate.baseline_cost == Decimal("100")
    assert report.aggregate.candidate_cost == Decimal("50")


@pytest.mark.parametrize(
    ("baseline", "candidate", "savings", "percentage", "state", "direction"),
    [
        (
            0,
            0,
            Decimal("0"),
            None,
            SavingsPercentageState.ZERO_BASELINE,
            CostDirection.BREAK_EVEN,
        ),
        (
            0,
            5,
            Decimal("-5"),
            None,
            SavingsPercentageState.ZERO_BASELINE,
            CostDirection.INCREASED_COST,
        ),
        (
            3,
            1,
            Decimal("2"),
            Decimal("66.67"),
            SavingsPercentageState.COMPUTED,
            CostDirection.SAVINGS,
        ),
        (
            5,
            8,
            Decimal("-3"),
            Decimal("-60.00"),
            SavingsPercentageState.COMPUTED,
            CostDirection.INCREASED_COST,
        ),
    ],
)
def test_decimal_rounding_zero_denominators_and_negative_savings_are_explicit(
    baseline: int,
    candidate: int,
    savings: Decimal,
    percentage: Decimal | None,
    state: SavingsPercentageState,
    direction: CostDirection,
) -> None:
    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(
                _pair(
                    baseline_tokens=baseline,
                    candidate_tokens=candidate,
                ),
            )
        )
    ).pairs[0]

    assert result.savings == savings
    assert result.savings_percentage == percentage
    assert result.savings_percentage_state is state
    assert result.cost_direction is direction


def test_mismatched_pair_contracts_fail_closed_with_typed_findings() -> None:
    pair = _pair()
    assert pair.candidate is not None
    mismatched = pair.model_copy(
        update={
            "candidate": pair.candidate.model_copy(
                update={
                    "source_id": "different-source",
                    "output_budget": ComparableOutputBudget(max_output_tokens=200),
                    "generation_settings": ComparableGenerationSettings(
                        revision="sha256:" + "3" * 64
                    ),
                    "quality": _quality(version="v2"),
                }
            )
        }
    )

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(pairs=(mismatched,))
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.UNAVAILABLE
    assert [finding.code for finding in result.findings] == [
        CostQualityFindingCode.SOURCE_MISMATCH,
        CostQualityFindingCode.OUTPUT_BUDGET_MISMATCH,
        CostQualityFindingCode.GENERATION_SETTINGS_MISMATCH,
        CostQualityFindingCode.QUALITY_CONTRACT_MISMATCH,
    ]
    assert result.eligible_for_verified_aggregate is False


def test_quality_evidence_references_require_opaque_content_ids() -> None:
    with pytest.raises(ValidationError, match="sha256 content identifier"):
        QualityEvidenceReference(
            kind="artifact",
            reference="The candidate output was: sensitive answer",
        )

    reference = QualityEvidenceReference(
        kind="artifact",
        reference="sha256:" + "4" * 64,
    )

    assert reference.model_dump(mode="json") == {
        "kind": "artifact",
        "reference": "sha256:" + "4" * 64,
    }


def test_available_source_checkpoint_identity_must_match() -> None:
    pair = _pair()
    assert pair.candidate is not None
    candidate_attempt = pair.candidate.attempts[0].model_copy(
        update={"source_checkpoint_id": "different-checkpoint"}
    )
    candidate = pair.candidate.model_copy(update={"attempts": (candidate_attempt,)})

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.UNAVAILABLE
    assert [finding.code for finding in result.findings] == [
        CostQualityFindingCode.SOURCE_CHECKPOINT_MISMATCH
    ]


def test_aggregate_rejects_verified_pairs_from_different_workload_cohorts() -> None:
    first = _pair(pair_id="first")
    second = _pair(pair_id="second")
    assert second.baseline is not None
    assert second.candidate is not None
    second = second.model_copy(
        update={
            "baseline": second.baseline.model_copy(update={"workload_id": "other-workload"}),
            "candidate": second.candidate.model_copy(update={"workload_id": "other-workload"}),
        }
    )

    report = compare_paired_cost_quality(PairedCostQualityComparisonRequest(pairs=(first, second)))

    assert [pair.status for pair in report.pairs] == [
        CostQualityComparisonStatus.VERIFIED,
        CostQualityComparisonStatus.VERIFIED,
    ]
    assert report.aggregate.status is CostQualityComparisonStatus.UNAVAILABLE
    assert report.aggregate.eligible_pair_ids == ()
    assert [finding.code for finding in report.aggregate.findings] == [
        CostQualityFindingCode.AGGREGATE_COHORT_MISMATCH
    ]


def test_aggregate_rejects_verified_pairs_from_different_pricing_catalogs() -> None:
    first = _pair(pair_id="first")
    second = _pair(pair_id="second")
    assert second.baseline is not None
    assert second.candidate is not None
    other_catalog = ComparisonPricingCatalog(
        price_book_version="fixture-v2",
        generated_at="2026-08-18T00:00:00Z",
    )
    second = second.model_copy(
        update={
            "baseline": second.baseline.model_copy(update={"pricing_catalog": other_catalog}),
            "candidate": second.candidate.model_copy(update={"pricing_catalog": other_catalog}),
        }
    )

    report = compare_paired_cost_quality(PairedCostQualityComparisonRequest(pairs=(first, second)))

    assert [pair.status for pair in report.pairs] == [
        CostQualityComparisonStatus.VERIFIED,
        CostQualityComparisonStatus.VERIFIED,
    ]
    assert report.aggregate.status is CostQualityComparisonStatus.UNAVAILABLE
    assert report.aggregate.eligible_pair_ids == ()
    assert [finding.code for finding in report.aggregate.findings] == [
        CostQualityFindingCode.AGGREGATE_COHORT_MISMATCH
    ]


def test_missing_and_contradictory_attempt_evidence_is_unavailable() -> None:
    missing_attempt = PairedCostAttempt(
        attempt_id="missing",
        operation_id="answer",
        session_id="candidate",
        role="final-answer",
        operation=CostQualityAttemptOperation.AGENT_STEP,
        attempt_ordinal=1,
        provider_name="provider",
        model="model",
        cost=None,
        usage_unavailable_reason="provider returned no counters",
    )
    contradictory_cost = _cost(session_id="baseline", input_tokens=100).model_copy(
        update={"total_cost": Decimal("999")}
    )
    pair = PairedCostQualityPair(
        pair_id="bad-evidence",
        baseline=_side(
            strategy_id="baseline",
            attempts=(
                _attempt(
                    attempt_id="contradictory",
                    session_id="baseline",
                    input_tokens=100,
                    cost=contradictory_cost,
                ),
            ),
        ),
        candidate=_side(strategy_id="candidate", attempts=(missing_attempt,)),
    )

    result = compare_paired_cost_quality(PairedCostQualityComparisonRequest(pairs=(pair,))).pairs[0]

    assert result.status is CostQualityComparisonStatus.UNAVAILABLE
    assert [finding.code for finding in result.findings] == [
        CostQualityFindingCode.CONTRADICTORY_USAGE,
        CostQualityFindingCode.MISSING_USAGE,
        CostQualityFindingCode.SOURCE_CHECKPOINT_MISMATCH,
    ]
    assert result.baseline_cost is None
    assert result.candidate_cost is None


@pytest.mark.parametrize(
    ("aggregate_write", "write_5m", "write_1h", "write_unknown", "expected_status"),
    [
        (10, 0, 0, 0, CostQualityComparisonStatus.VERIFIED),
        (10, 4, 5, 1, CostQualityComparisonStatus.VERIFIED),
        (10, 11, 0, 0, CostQualityComparisonStatus.UNAVAILABLE),
        (0, 1, 0, 0, CostQualityComparisonStatus.UNAVAILABLE),
    ],
)
def test_cache_write_ttl_subcategories_cannot_contradict_aggregate_usage(
    aggregate_write: int,
    write_5m: int,
    write_1h: int,
    write_unknown: int,
    expected_status: CostQualityComparisonStatus,
) -> None:
    cost = _cost(session_id="candidate", input_tokens=20).model_copy(
        update={
            "input_tokens": 20,
            "cache_write_input_tokens": aggregate_write,
            "cache_write_5m_input_tokens": write_5m,
            "cache_write_1h_input_tokens": write_1h,
            "cache_write_unknown_ttl_input_tokens": write_unknown,
            "uncached_input_tokens": 20 - aggregate_write,
        }
    )
    pair = _pair(baseline_tokens=20, candidate_tokens=20)
    assert pair.candidate is not None
    candidate_attempt = pair.candidate.attempts[0].model_copy(update={"cost": cost})
    candidate = pair.candidate.model_copy(update={"attempts": (candidate_attempt,)})

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is expected_status


def test_hosted_search_usage_and_cost_survive_cost_quality_projection() -> None:
    pair = _pair(baseline_tokens=50, candidate_tokens=20)
    assert pair.candidate is not None
    cost = _cost(session_id="candidate", input_tokens=20).model_copy(
        update={
            "web_search_calls": 2,
            "web_search_cost": Decimal("20"),
            "total_cost": Decimal("40"),
        }
    )
    candidate_attempt = pair.candidate.attempts[0].model_copy(update={"cost": cost})
    candidate = pair.candidate.model_copy(update={"attempts": (candidate_attempt,)})

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.VERIFIED
    assert result.candidate is not None
    accepted = result.candidate.attempts[0].cost
    assert accepted is not None
    assert accepted.web_search_calls == 2
    assert accepted.web_search_cost == Decimal("20")
    assert result.candidate.whole_harness.web_search_calls == 2
    assert result.candidate_cost == Decimal("40")


def test_missing_attempt_identity_is_unavailable_instead_of_guessed() -> None:
    pair = _pair()
    assert pair.candidate is not None
    anonymous = pair.candidate.attempts[0].model_copy(update={"provider_name": None})
    candidate = pair.candidate.model_copy(update={"attempts": (anonymous,)})

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.UNAVAILABLE
    assert [finding.code for finding in result.findings] == [
        CostQualityFindingCode.MISSING_ATTEMPT_IDENTITY
    ]


def test_duplicate_attempts_and_incomplete_retry_sequences_are_unavailable() -> None:
    pair = _pair()
    assert pair.candidate is not None
    first = pair.candidate.attempts[0]
    invalid = first.model_copy(update={"attempt_id": "second", "attempt_ordinal": 3})
    candidate = pair.candidate.model_copy(update={"attempts": (first, first, invalid)})

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.UNAVAILABLE
    assert [finding.code for finding in result.findings] == [
        CostQualityFindingCode.DUPLICATE_ATTEMPT,
        CostQualityFindingCode.ATTEMPT_SEQUENCE_INVALID,
    ]


@pytest.mark.parametrize(
    "candidate_quality",
    [
        None,
        _quality(status=QualityEvidenceStatus.UNAVAILABLE, score=None),
    ],
)
def test_missing_or_unavailable_quality_keeps_cost_but_is_not_verified(
    candidate_quality: PairedQualityEvidence | None,
) -> None:
    pair = _pair()
    assert pair.candidate is not None
    candidate = pair.candidate.model_copy(update={"quality": candidate_quality})

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.MEASURED_UNMATCHED
    assert result.candidate_cost == Decimal("50")
    assert result.eligible_for_verified_aggregate is False


def test_mixed_currency_and_catalog_identity_cannot_produce_verified_savings() -> None:
    pair = _pair()
    assert pair.candidate is not None
    euro_cost = _cost(
        session_id="candidate-eur",
        input_tokens=50,
        price_book=_price_book(currency="EUR"),
    )
    candidate_attempt = pair.candidate.attempts[0].model_copy(update={"cost": euro_cost})
    candidate = pair.candidate.model_copy(
        update={
            "pricing_catalog": ComparisonPricingCatalog(
                price_book_version="other-v2",
                generated_at="2026-08-18T00:00:00Z",
            ),
            "attempts": (candidate_attempt,),
        }
    )

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.UNAVAILABLE
    assert [finding.code for finding in result.findings] == [
        CostQualityFindingCode.PRICING_CATALOG_MISMATCH,
        CostQualityFindingCode.MIXED_CURRENCY,
    ]
    assert result.currency is None
    assert result.savings_percentage is None


def test_equivalent_pricing_rows_with_different_provenance_are_unavailable() -> None:
    pair = _pair()
    assert pair.candidate is not None
    candidate_cost = pair.candidate.attempts[0].cost
    assert candidate_cost is not None
    changed_cost = candidate_cost.model_copy(
        update={
            "pricing_provenance": Provenance(
                source="different source",
                url="https://example.invalid/different-pricing",
                as_of="2026-08-18",
            )
        }
    )
    changed_attempt = pair.candidate.attempts[0].model_copy(update={"cost": changed_cost})
    candidate = pair.candidate.model_copy(update={"attempts": (changed_attempt,)})

    result = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    ).pairs[0]

    assert result.status is CostQualityComparisonStatus.UNAVAILABLE
    assert [finding.code for finding in result.findings] == [
        CostQualityFindingCode.PRICING_PROVENANCE_MISMATCH
    ]


def test_input_order_does_not_change_the_report() -> None:
    pair = _pair(pair_id="b")
    second = _pair(pair_id="a")
    assert pair.baseline is not None
    baseline = pair.baseline.model_copy(
        update={
            "attempts": (
                _attempt(
                    attempt_id="retry",
                    session_id="ordered",
                    input_tokens=10,
                    attempt_ordinal=2,
                ),
                _attempt(
                    attempt_id="first",
                    session_id="ordered",
                    input_tokens=20,
                ),
            )
        }
    )
    reordered_pair = pair.model_copy(update={"baseline": baseline})
    reversed_attempts = baseline.model_copy(update={"attempts": tuple(reversed(baseline.attempts))})
    reversed_pair = pair.model_copy(update={"baseline": reversed_attempts})

    forward = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(pairs=(reordered_pair, second))
    )
    reverse = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(pairs=(second, reversed_pair))
    )

    assert forward == reverse


def test_duplicate_and_empty_pair_sets_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        PairedCostQualityComparisonRequest(pairs=())
    with pytest.raises(ValidationError, match="distinct pair_id"):
        PairedCostQualityComparisonRequest(pairs=(_pair(), _pair()))


def test_default_serialization_is_bounded_and_excludes_raw_application_content() -> None:
    sensitive_cost = _cost(session_id="sensitive", input_tokens=50).model_copy(
        update={
            "billing_identity": BillingIdentity(
                provider_name="provider",
                resource_id="resource",
                request_evidence={"raw_prompt": "sensitive-prompt-marker"},
                completion_evidence={"model_output": "sensitive-output-marker"},
            )
        }
    )
    pair = _pair()
    assert pair.candidate is not None
    candidate_attempt = pair.candidate.attempts[0].model_copy(update={"cost": sensitive_cost})
    candidate = pair.candidate.model_copy(update={"attempts": (candidate_attempt,)})
    report = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    )

    serialized = report.model_dump_json()

    assert "raw_prompt" not in serialized
    assert "messages" not in serialized
    assert "model_output" not in serialized
    assert "credentials" not in serialized
    assert "arbitrary_metadata" not in serialized
    assert "billing_identity" not in serialized
    assert "sensitive-prompt-marker" not in serialized
    assert "sensitive-output-marker" not in serialized
    assert "sha256:" + "1" * 64 in serialized
    assert len(serialized.encode("utf-8")) < 20_000


def test_safe_cost_projection_rejects_unbounded_pricing_text() -> None:
    caller_cost = _cost(session_id="oversized", input_tokens=50)
    caller_cost = caller_cost.model_copy(
        update={
            "pricing_provenance": Provenance(
                source="x" * 2_049,
                url="https://example.invalid/pricing",
                as_of="2026-08-17",
            )
        }
    )

    with pytest.raises(ValidationError, match="at most 2048 characters"):
        ComparisonCostLineItem.from_cost_line_item(caller_cost)


def test_safe_cost_projection_rejects_unbounded_numeric_evidence() -> None:
    caller_cost = _cost(session_id="oversized", input_tokens=50)

    with pytest.raises(ValidationError, match="less than or equal to"):
        ComparisonCostLineItem.from_cost_line_item(
            caller_cost.model_copy(update={"input_tokens": MAX_DURABLE_JSON_INTEGER + 1})
        )
    with pytest.raises(ValidationError, match="at most 64 decimal digits"):
        ComparisonCostLineItem.from_cost_line_item(
            caller_cost.model_copy(update={"total_cost": Decimal("0." + "9" * 65)})
        )
    with pytest.raises(ValidationError, match="at most 64 decimal places"):
        ComparisonCostLineItem.from_cost_line_item(
            caller_cost.model_copy(update={"total_cost": Decimal("1e-65")})
        )


def test_comparison_arithmetic_is_independent_of_ambient_decimal_context() -> None:
    precise_cost = Decimal("0.1234567890123456789012345678901234567890")
    pair = _pair()
    assert pair.baseline is not None
    assert pair.candidate is not None

    def with_precise_cost(side: PairedCostQualitySide) -> PairedCostQualitySide:
        attempt = side.attempts[0]
        assert attempt.cost is not None
        cost = attempt.cost.model_copy(
            update={
                "input_cost": precise_cost,
                "total_cost": precise_cost,
            }
        )
        return side.model_copy(update={"attempts": (attempt.model_copy(update={"cost": cost}),)})

    request = PairedCostQualityComparisonRequest(
        pairs=(
            pair.model_copy(
                update={
                    "baseline": with_precise_cost(pair.baseline),
                    "candidate": with_precise_cost(pair.candidate),
                }
            ),
        )
    )

    with localcontext() as context:
        context.prec = 28
        low_precision_report = compare_paired_cost_quality(request)
    with localcontext() as context:
        context.prec = 70
        high_precision_report = compare_paired_cost_quality(request)

    assert low_precision_report == high_precision_report
    assert low_precision_report.status is CostQualityComparisonStatus.VERIFIED
    assert low_precision_report.pairs[0].baseline_cost == precise_cost


def test_comparison_request_rejects_excess_total_attempt_work() -> None:
    def missing_attempt(index: int) -> PairedCostAttempt:
        return PairedCostAttempt(
            attempt_id=f"attempt-{index}",
            operation_id=f"operation-{index}",
            session_id="bounded-session",
            role="final-answer",
            operation=CostQualityAttemptOperation.AGENT_STEP,
            attempt_ordinal=1,
            provider_name="provider",
            model="model",
            cost=None,
            usage_unavailable_reason="missing usage",
        )

    attempts = tuple(missing_attempt(index) for index in range(4_097))
    full_pair = PairedCostQualityPair(
        pair_id="full",
        baseline=_side(strategy_id="baseline", attempts=attempts[:2_048]),
        candidate=_side(strategy_id="candidate", attempts=attempts[2_048:4_096]),
    )
    overflow_pair = PairedCostQualityPair(
        pair_id="overflow",
        baseline=_side(strategy_id="baseline", attempts=(attempts[-1],)),
        candidate=None,
    )

    with pytest.raises(ValidationError, match="exceeds 4096 total attempts"):
        PairedCostQualityComparisonRequest(pairs=(full_pair, overflow_pair))


def test_comparison_request_rejects_excess_serialized_evidence_work() -> None:
    def missing_attempt(index: int) -> PairedCostAttempt:
        return PairedCostAttempt(
            attempt_id=f"oversized-{index}",
            operation_id=f"operation-{index}",
            session_id="oversized-session",
            role="final-answer",
            operation=CostQualityAttemptOperation.AGENT_STEP,
            attempt_ordinal=1,
            provider_name="provider",
            model="model",
            cost=None,
            usage_unavailable_reason="x" * 2_048,
        )

    attempts = tuple(missing_attempt(index) for index in range(4_096))
    pair = PairedCostQualityPair(
        pair_id="oversized",
        baseline=_side(strategy_id="baseline", attempts=attempts[:2_048]),
        candidate=_side(strategy_id="candidate", attempts=attempts[2_048:]),
    )

    with pytest.raises(ValidationError, match="exceeds 8388608 canonical JSON bytes"):
        PairedCostQualityComparisonRequest(pairs=(pair,))


def test_attempt_cost_evidence_is_detached_and_frozen() -> None:
    caller_cost = _cost(session_id="candidate", input_tokens=50)
    attempt = _attempt(
        attempt_id="candidate",
        session_id="candidate",
        input_tokens=50,
        cost=caller_cost,
    )
    caller_cost.total_cost = Decimal("999")

    pair = _pair()
    assert pair.candidate is not None
    candidate = pair.candidate.model_copy(update={"attempts": (attempt,)})
    report = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(pair.model_copy(update={"candidate": candidate}),)
        )
    )

    result = report.pairs[0]
    assert result.candidate is not None
    accepted_cost = result.candidate.attempts[0].cost
    assert accepted_cost is not None
    assert accepted_cost.total_cost == Decimal("50")
    with pytest.raises(ValidationError, match="frozen"):
        accepted_cost.total_cost = Decimal("1")


def test_schema_v3_minimal_unavailable_report_has_a_locked_json_shape() -> None:
    report = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(PairedCostQualityPair(pair_id="missing", baseline=None, candidate=None),)
        )
    )

    assert report.model_dump(mode="json") == {
        "schema_version": 3,
        "status": "unavailable",
        "pairs": [
            {
                "pair_id": "missing",
                "status": "unavailable",
                "findings": [
                    {
                        "code": "missing_side",
                        "field": "baseline",
                        "detail": "A required comparison side is missing.",
                    },
                    {
                        "code": "missing_side",
                        "field": "candidate",
                        "detail": "A required comparison side is missing.",
                    },
                ],
                "baseline": None,
                "candidate": None,
                "eligible_for_verified_aggregate": False,
                "currency": None,
                "baseline_cost": None,
                "candidate_cost": None,
                "savings": None,
                "savings_percentage": None,
                "savings_percentage_state": "unavailable",
                "cost_direction": "unavailable",
            }
        ],
        "aggregate": {
            "status": "unavailable",
            "pair_count": 1,
            "eligible_pair_ids": [],
            "exclusions": [
                {
                    "pair_id": "missing",
                    "status": "unavailable",
                    "findings": [
                        {
                            "code": "missing_side",
                            "field": "baseline",
                            "detail": "A required comparison side is missing.",
                        },
                        {
                            "code": "missing_side",
                            "field": "candidate",
                            "detail": "A required comparison side is missing.",
                        },
                    ],
                }
            ],
            "findings": [],
            "currency": None,
            "baseline_cost": None,
            "candidate_cost": None,
            "savings": None,
            "savings_percentage": None,
            "savings_percentage_state": "unavailable",
            "cost_direction": "unavailable",
        },
    }


def test_schema_v3_full_verified_report_matches_golden_json() -> None:
    report = compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(pairs=(_fully_attributed_pair(),))
    )
    golden_path = Path(__file__).parents[1] / "fixtures" / "cost_quality_verified_v3.json"

    assert report.model_dump(mode="json") == json.loads(golden_path.read_text())
