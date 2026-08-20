from __future__ import annotations

from decimal import Decimal
from typing import Any

from examples._advanced_support import (
    ComparisonSessionEvidence,
    paired_cost_quality_report,
)

from cayu import (
    ComparableGenerationSettings,
    ComparableOutputBudget,
    CostQualityAttemptOperation,
    Event,
    EventType,
    ModelPrice,
    PairedQualityEvidence,
    PriceBook,
    PriceSchedule,
    PriceTier,
    Provenance,
    QualityEvidenceStatus,
    TieredPricing,
)


def _price_book() -> PriceBook:
    return PriceBook(
        price_book_version="tiered-v1",
        generated_at="2026-07-13T00:00:00Z",
        prices=(
            ModelPrice(
                provider_name="provider",
                model="model",
                match="exact",
                schedules=(
                    PriceSchedule(
                        pricing=TieredPricing(
                            standard=(
                                PriceTier(
                                    max_input_tokens=100,
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("5"),
                                ),
                                PriceTier(
                                    input_per_million=Decimal("2"),
                                    output_per_million=Decimal("10"),
                                ),
                            )
                        ),
                        provenance=Provenance(
                            source="test fixture",
                            url="https://example.invalid/pricing",
                            as_of="2026-07-13",
                        ),
                    ),
                ),
            ),
        ),
    )


def _event(*, session_id: str, input_tokens: int, include_usage: bool = True) -> Event:
    payload: dict[str, Any] = {
        "provider_name": "provider",
        "model": "model",
    }
    if include_usage:
        payload["usage_metrics"] = {
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
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        payload=payload,
    )


def _session(
    session_id: str,
    *,
    input_tokens: int,
    include_usage: bool = True,
) -> ComparisonSessionEvidence:
    return ComparisonSessionEvidence(
        session_id=session_id,
        events=(
            _event(
                session_id=session_id,
                input_tokens=input_tokens,
                include_usage=include_usage,
            ),
        ),
        operation=CostQualityAttemptOperation.AGENT_STEP,
        role="answer",
        branch_id=session_id,
        source_checkpoint_id="checkpoint-1",
    )


def _quality(
    *,
    status: QualityEvidenceStatus = QualityEvidenceStatus.PASSED,
    score: Decimal | None = Decimal("1"),
) -> PairedQualityEvidence:
    return PairedQualityEvidence(
        contract_name="answer-quality",
        contract_version="v1",
        threshold=Decimal("1"),
        role="answer",
        status=status,
        score=score,
    )


def _report(
    *,
    baseline: tuple[ComparisonSessionEvidence, ...] | None,
    candidate: tuple[ComparisonSessionEvidence, ...] | None,
    price_book: PriceBook | None = None,
    baseline_quality: PairedQualityEvidence | None = None,
    candidate_quality: PairedQualityEvidence | None = None,
):
    return paired_cost_quality_report(
        pair_id="pair-1",
        workload_id="workload-1",
        task_id="task-1",
        source_id="checkpoint-1",
        role="answer",
        output_budget=ComparableOutputBudget(max_output_tokens=100),
        generation_settings=ComparableGenerationSettings(revision="sha256:" + "2" * 64),
        baseline_strategy_id="baseline",
        candidate_strategy_id="candidate",
        baseline=baseline,
        candidate=candidate,
        baseline_quality=baseline_quality or _quality(),
        candidate_quality=candidate_quality or _quality(),
        price_book=price_book,
    )


def test_adapter_emits_public_verified_report_and_retains_distinct_valid_tiers() -> None:
    report = _report(
        baseline=(_session("baseline", input_tokens=150),),
        candidate=(_session("candidate", input_tokens=50),),
        price_book=_price_book(),
    )

    assert report["schema_version"] == 2
    assert report["status"] == "verified"
    pair = report["pairs"][0]
    assert pair["baseline_cost"] == "0.0003"
    assert pair["candidate_cost"] == "0.00005"
    assert pair["savings_percentage"] == "83.33"
    assert pair["baseline"]["attempts"][0]["cost"]["pricing_tier_max_input_tokens"] is None
    assert pair["candidate"]["attempts"][0]["cost"]["pricing_tier_max_input_tokens"] == 100


def test_adapter_retains_usage_but_reports_unpriced_without_a_catalog() -> None:
    report = _report(
        baseline=(_session("baseline", input_tokens=100),),
        candidate=(_session("candidate", input_tokens=50),),
    )

    assert report["status"] == "unpriced"
    pair = report["pairs"][0]
    assert pair["baseline_cost"] is None
    assert pair["candidate_cost"] is None
    assert pair["baseline"]["whole_harness"]["input_tokens"] == 100
    assert pair["candidate"]["whole_harness"]["input_tokens"] == 50


def test_adapter_reports_missing_pair_side_as_unavailable() -> None:
    report = _report(
        baseline=(_session("baseline", input_tokens=100),),
        candidate=None,
        price_book=_price_book(),
    )

    assert report["status"] == "unavailable"
    assert report["pairs"][0]["findings"][0]["code"] == "missing_side"


def test_adapter_reports_missing_provider_usage_as_unavailable_not_zero() -> None:
    report = _report(
        baseline=(_session("baseline", input_tokens=100),),
        candidate=(_session("candidate", input_tokens=0, include_usage=False),),
        price_book=_price_book(),
    )

    assert report["status"] == "unavailable"
    candidate = report["pairs"][0]["candidate"]
    assert candidate["whole_harness"]["missing_usage_attempt_count"] == 1
    assert candidate["attempts"][0]["cost"] is None
    assert (
        candidate["attempts"][0]["usage_unavailable_reason"]
        == "model.completed event has no token usage metrics"
    )


def test_adapter_retains_dispatched_retry_without_usage_and_fails_closed() -> None:
    session_id = "candidate-retry"
    step_id = "mstep_retry"
    first_attempt_id = "matt_first"
    second_attempt_id = "matt_second"
    shared = {"provider": "provider", "model": "model", "step": 1, "max_attempts": 2}
    retry_session = ComparisonSessionEvidence(
        session_id=session_id,
        events=(
            Event(
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                payload={
                    **shared,
                    "attempt": 1,
                    "model_step_id": step_id,
                    "model_attempt_id": first_attempt_id,
                },
            ),
            Event(
                type=EventType.MODEL_RETRY,
                session_id=session_id,
                payload={
                    **shared,
                    "attempt": 1,
                    "next_attempt": 2,
                    "model_step_id": step_id,
                    "model_attempt_id": first_attempt_id,
                },
            ),
            Event(
                type=EventType.MODEL_ATTEMPT_DISCARDED,
                session_id=session_id,
                payload={
                    **shared,
                    "attempt": 1,
                    "next_attempt": 2,
                    "model_step_id": step_id,
                    "model_attempt_id": first_attempt_id,
                },
            ),
            Event(
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                payload={
                    **shared,
                    "attempt": 2,
                    "model_step_id": step_id,
                    "model_attempt_id": second_attempt_id,
                },
            ),
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                payload={
                    **_event(session_id=session_id, input_tokens=50).payload,
                    "step": 1,
                    "attempt": 2,
                    "max_attempts": 2,
                    "model_step_id": step_id,
                    "model_attempt_id": second_attempt_id,
                },
            ),
        ),
        operation=CostQualityAttemptOperation.AGENT_STEP,
        role="answer",
        branch_id=session_id,
        source_checkpoint_id="checkpoint-1",
    )

    report = _report(
        baseline=(_session("baseline", input_tokens=100),),
        candidate=(retry_session,),
        price_book=_price_book(),
    )

    assert report["status"] == "unavailable"
    candidate = report["pairs"][0]["candidate"]
    assert candidate["whole_harness"]["attempt_count"] == 2
    assert candidate["whole_harness"]["retry_attempt_count"] == 1
    assert candidate["whole_harness"]["missing_usage_attempt_count"] == 1
    assert candidate["attempts"][0]["attempt_id"] == first_attempt_id
    assert candidate["attempts"][0]["cost"] is None
    assert (
        candidate["attempts"][0]["usage_unavailable_reason"]
        == "provider attempt has no model.completed usage evidence"
    )


def test_adapter_keeps_measured_cost_when_the_quality_gate_fails() -> None:
    report = _report(
        baseline=(_session("baseline", input_tokens=100),),
        candidate=(_session("candidate", input_tokens=50),),
        price_book=_price_book(),
        candidate_quality=_quality(
            status=QualityEvidenceStatus.FAILED,
            score=Decimal("0"),
        ),
    )

    assert report["status"] == "measured_unmatched"
    pair = report["pairs"][0]
    assert pair["candidate_cost"] == "0.00005"
    assert pair["eligible_for_verified_aggregate"] is False
