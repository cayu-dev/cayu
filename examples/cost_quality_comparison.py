from __future__ import annotations

import hashlib
from decimal import Decimal

from cayu import (
    ComparableGenerationSettings,
    ComparableOutputBudget,
    ComparisonCostLineItem,
    ComparisonPricingCatalog,
    ComparisonPricingProvenance,
    CostQualityAttemptOperation,
    PairedCostAttempt,
    PairedCostQualityComparisonReport,
    PairedCostQualityComparisonRequest,
    PairedCostQualityPair,
    PairedCostQualitySide,
    PairedQualityEvidence,
    QualityEvidenceReference,
    QualityEvidenceStatus,
    compare_paired_cost_quality,
)

_CATALOG = ComparisonPricingCatalog(
    price_book_version="deterministic-demo-v1",
    generated_at="2026-08-17T00:00:00Z",
)
_PROVENANCE = ComparisonPricingProvenance(
    source="deterministic example fixture; not provider pricing",
    url="https://example.invalid/cayu/cost-quality-demo",
    as_of="2026-08-17",
)


def build_demo_report() -> PairedCostQualityComparisonReport:
    """Return one deterministic report containing every v1 proof status."""

    return compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(
            pairs=(
                _pair("verified", candidate_tokens=50),
                _pair(
                    "measured-unmatched",
                    candidate_tokens=50,
                    candidate_quality=QualityEvidenceStatus.FAILED,
                ),
                _pair("unpriced", candidate_tokens=50, candidate_priced=False),
                PairedCostQualityPair(
                    pair_id="unavailable",
                    baseline=_side(
                        pair_id="unavailable",
                        strategy_id="baseline",
                        tokens=100,
                    ),
                    candidate=None,
                ),
            )
        )
    )


def _pair(
    pair_id: str,
    *,
    candidate_tokens: int,
    candidate_quality: QualityEvidenceStatus = QualityEvidenceStatus.PASSED,
    candidate_priced: bool = True,
) -> PairedCostQualityPair:
    return PairedCostQualityPair(
        pair_id=pair_id,
        baseline=_side(pair_id=pair_id, strategy_id="baseline", tokens=100),
        candidate=_side(
            pair_id=pair_id,
            strategy_id="candidate",
            tokens=candidate_tokens,
            quality_status=candidate_quality,
            priced=candidate_priced,
        ),
    )


def _side(
    *,
    pair_id: str,
    strategy_id: str,
    tokens: int,
    quality_status: QualityEvidenceStatus = QualityEvidenceStatus.PASSED,
    priced: bool = True,
) -> PairedCostQualitySide:
    session_id = f"{pair_id}-{strategy_id}"
    return PairedCostQualitySide(
        strategy_id=strategy_id,
        workload_id="deterministic-comparison-demo",
        task_id="bounded-answer",
        source_id="fixture-source-v1",
        role="final-answer",
        output_budget=ComparableOutputBudget(max_output_tokens=100),
        generation_settings=ComparableGenerationSettings(revision="sha256:" + "5" * 64),
        pricing_catalog=_CATALOG,
        quality=_quality(
            pair_id=pair_id,
            strategy_id=strategy_id,
            status=quality_status,
        ),
        attempts=(
            PairedCostAttempt(
                attempt_id=f"{session_id}:1",
                operation_id=f"{session_id}:answer",
                session_id=session_id,
                branch_id=session_id,
                role="final-answer",
                operation=CostQualityAttemptOperation.AGENT_STEP,
                attempt_ordinal=1,
                provider_name="deterministic",
                model="fixture-model",
                source_checkpoint_id="fixture-source-v1",
                cost=_cost(tokens=tokens, priced=priced),
            ),
        ),
    )


def _quality(
    *,
    pair_id: str,
    strategy_id: str,
    status: QualityEvidenceStatus,
) -> PairedQualityEvidence:
    passed = status is QualityEvidenceStatus.PASSED
    return PairedQualityEvidence(
        contract_name="deterministic-equivalence",
        contract_version="v1",
        threshold=Decimal("0.8"),
        role="final-answer",
        status=status,
        score=Decimal("1") if passed else Decimal("0"),
        references=(
            QualityEvidenceReference(
                kind="fixture",
                reference=(
                    "sha256:"
                    + hashlib.sha256(
                        f"fixture://quality/{pair_id}/{strategy_id}".encode()
                    ).hexdigest()
                ),
            ),
        ),
    )


def _cost(*, tokens: int, priced: bool) -> ComparisonCostLineItem:
    total_cost = Decimal(tokens) / Decimal("1000000") if priced else Decimal("0")
    return ComparisonCostLineItem(
        model_step=1,
        provider_name="deterministic",
        requested_model="fixture-model",
        model="fixture-model",
        pricing_provider_name="deterministic" if priced else None,
        pricing_model="fixture-model" if priced else None,
        pricing_match="exact" if priced else None,
        pricing_provenance=_PROVENANCE if priced else None,
        priced=priced,
        currency="USD",
        input_tokens=tokens,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_write_input_tokens=0,
        uncached_input_tokens=tokens,
        input_cost=total_cost,
        output_cost=Decimal("0"),
        cache_read_input_cost=Decimal("0"),
        cache_write_input_cost=Decimal("0"),
        total_cost=total_cost,
        missing_pricing_reason=None if priced else "no matching model pricing",
    )


if __name__ == "__main__":
    print(build_demo_report().model_dump_json(indent=2))
