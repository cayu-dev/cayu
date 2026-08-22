from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Any

import pytest

from cayu._validation import canonical_durable_json_bytes
from cayu.memory import (
    AutomaticRecallContributor,
    AutomaticRecallMode,
    AutomaticRecallPolicy,
    RecallOffer,
    admit_recall,
)
from cayu.recall import (
    RECALL_ENGINE_VERSION,
    RecallCandidate,
    RecallEngine,
    RecallRecord,
    RecallResult,
    RecallSituation,
    RecallSourceDiagnostic,
    RecallSourceStatus,
)
from cayu.retrieval import (
    FusedChannelMatch,
    FusedRetrievalCandidate,
    RetrievalCandidateIdentity,
    RetrievalChannelDiagnostics,
    RetrievalFusionDiagnostics,
)

_FUSION_STRATEGY = "cayu.wrrf.v1"
_FUSION_CONFIGURATION = "memory-admission-tests-v1"


def _mutate_mapping(value: Any, key: str, replacement: str) -> None:
    value[key] = replacement


def _policy(**updates) -> AutomaticRecallPolicy:
    values: dict[str, object] = {
        "calibration_version": "memory-admission-calibration-v1",
        "fusion_strategy_version": _FUSION_STRATEGY,
        "fusion_configuration_version": _FUSION_CONFIGURATION,
        "minimum_inject_score": 0.03,
        "minimum_offer_score": 0.01,
    }
    values.update(updates)
    return AutomaticRecallPolicy.model_validate(values)


def _candidate(
    record_id: str,
    *,
    score: float,
    text: str | None = None,
    content_hash: str | None = None,
    record_type: str = "knowledge_entry",
) -> RecallCandidate:
    text = text or f"Evidence for {record_id}"
    content_hash = content_hash or sha256(text.encode("utf-8")).hexdigest()
    identity = RetrievalCandidateIdentity(
        record_type=record_type,
        record_id=record_id,
        revision="1",
    )
    return RecallCandidate(
        fused=FusedRetrievalCandidate(
            identity=identity,
            score=score,
            reciprocal_rank_score=score,
            feature_adjustment=0.0,
            features={"current_revision": 1.0},
            best_rank=1,
            channel_count=1,
            matches=(
                FusedChannelMatch(
                    channel="knowledge.lexical",
                    index_version="knowledge-test-v1",
                    rank=1,
                    representation="entry_text",
                    content_hash=content_hash,
                    explanations=("test match",),
                    raw_score=score,
                ),
            ),
        ),
        record=RecallRecord(
            identity=identity,
            representation="entry_text",
            text=text,
            text_complete=True,
            content_hash=content_hash,
            locator={"entry_id": record_id, "entry_revision": 1},
        ),
    )


def _result(*candidates: RecallCandidate, truncated: bool = False) -> RecallResult:
    return RecallResult(
        engine_version=RECALL_ENGINE_VERSION,
        situation_sha256="a" * 64,
        candidates=candidates,
        fusion=RetrievalFusionDiagnostics(
            strategy_version=_FUSION_STRATEGY,
            configuration_version=_FUSION_CONFIGURATION,
            configuration_sha256="b" * 64,
            channels=(
                RetrievalChannelDiagnostics(
                    channel="knowledge.lexical",
                    index_version="knowledge-test-v1",
                    candidate_limit=max(1, len(candidates)),
                    hit_count=len(candidates),
                    unique_candidate_count=len(candidates),
                    truncated=truncated,
                    continuation_available=False,
                ),
            ),
            unique_candidate_count=len(candidates),
            returned_candidate_count=len(candidates),
            omitted_candidate_count=0,
            truncated=truncated,
            truncation_reasons=(("source_truncated",) if truncated else ()),
            continuation_channels=(),
        ),
        sources=(
            RecallSourceDiagnostic(
                source="knowledge",
                required=True,
                status=(RecallSourceStatus.PARTIAL if truncated else RecallSourceStatus.COMPLETE),
                channels=("knowledge.lexical",),
                failure_code=("source_truncated" if truncated else None),
            ),
        ),
        truncated=truncated,
    )


def test_automatic_recall_admits_strong_offers_plausible_and_silences_weak() -> None:
    result = _result(
        _candidate("strong", score=0.04),
        _candidate("plausible", score=0.02),
        _candidate("weak", score=0.001),
    )

    contribution = admit_recall(result, _policy())

    assert contribution.focus is not None
    assert contribution.offer is not None
    assert [item.candidate.record.identity.record_id for item in contribution.focus.items] == [
        "strong"
    ]
    assert [item.identity.record_id for item in contribution.offer.items] == ["plausible"]
    assert [item.reason for item in contribution.offer.items] == ["calibrated_plausible_match"]
    assert contribution.diagnostics.injected_count == 1
    assert contribution.diagnostics.offered_count == 1
    assert contribution.diagnostics.silent_count == 1
    assert contribution.diagnostics.admission_truncated is False
    assert contribution.sources == result.sources
    assert contribution.continuations == {}


def test_automatic_recall_preserves_record_type_diversity_before_filling_capacity() -> None:
    result = _result(
        _candidate("knowledge-one", score=0.04),
        _candidate("knowledge-two", score=0.039),
        _candidate("knowledge-three", score=0.038),
        _candidate(
            "transcript-one",
            score=0.037,
            record_type="transcript_message",
        ),
    )

    contribution = admit_recall(
        result,
        _policy(max_injected_items=2),
    )

    assert contribution.focus is not None
    assert [item.candidate.record.identity.record_type for item in contribution.focus.items] == [
        "knowledge_entry",
        "transcript_message",
    ]
    assert [item.fused_rank for item in contribution.focus.items] == [1, 4]


def test_automatic_recall_clips_visible_items_to_the_complete_contribution_bound() -> None:
    result = _result(_candidate("large", score=0.04, text="evidence " * 80))
    initial = admit_recall(result, _policy(mode=AutomaticRecallMode.STRONG_MATCHES))
    initial_bytes = len(
        canonical_durable_json_bytes(
            initial.model_dump(mode="json"),
            "automatic recall contribution",
        )
    )
    total_bound = initial_bytes - 1

    bounded = admit_recall(
        result,
        _policy(
            mode=AutomaticRecallMode.STRONG_MATCHES,
            max_focus_bytes=total_bound,
            max_offer_bytes=total_bound,
            max_total_bytes=total_bound,
        ),
    )

    assert bounded.focus is None
    assert bounded.diagnostics.focus_bound_omitted == 1
    assert bounded.diagnostics.admission_truncated is True
    assert (
        len(
            canonical_durable_json_bytes(
                bounded.model_dump(mode="json"),
                "automatic recall contribution",
            )
        )
        <= total_bound
    )


@pytest.mark.parametrize(
    ("mode", "focused", "offered", "recall_performed"),
    [
        (AutomaticRecallMode.OFF, [], [], False),
        (AutomaticRecallMode.OFFER, [], ["strong", "plausible"], True),
        (AutomaticRecallMode.STRONG_MATCHES, ["strong"], [], True),
        (
            AutomaticRecallMode.OFFER_AND_STRONG_MATCHES,
            ["strong"],
            ["plausible"],
            True,
        ),
    ],
)
def test_automatic_recall_modes_are_explicit(
    mode: AutomaticRecallMode,
    focused: list[str],
    offered: list[str],
    recall_performed: bool,
) -> None:
    result = _result(
        _candidate("strong", score=0.04),
        _candidate("plausible", score=0.02),
    )

    contribution = admit_recall(result, _policy(mode=mode))

    assert (
        []
        if contribution.focus is None
        else [item.candidate.record.identity.record_id for item in contribution.focus.items]
    ) == focused
    assert (
        []
        if contribution.offer is None
        else [item.identity.record_id for item in contribution.offer.items]
    ) == offered
    assert contribution.diagnostics.recall_performed is recall_performed
    if mode is AutomaticRecallMode.STRONG_MATCHES:
        assert contribution.diagnostics.silent_count == 1
        assert contribution.diagnostics.offer_bound_omitted == 0
        assert contribution.diagnostics.admission_truncated is False


def test_automatic_recall_requires_the_calibrated_fusion_identity() -> None:
    result = _result(_candidate("strong", score=0.04))

    with pytest.raises(ValueError, match="strategy is not covered"):
        admit_recall(result, _policy(fusion_strategy_version="custom-fusion-v1"))
    with pytest.raises(ValueError, match="configuration is not covered"):
        admit_recall(result, _policy(fusion_configuration_version="other-config-v1"))


def test_automatic_recall_diversifies_duplicate_content_without_losing_the_reference() -> None:
    shared_hash = sha256(b"shared evidence").hexdigest()
    result = _result(
        _candidate("knowledge", score=0.05, text="shared evidence", content_hash=shared_hash),
        _candidate(
            "transcript",
            score=0.04,
            text="shared evidence",
            content_hash=shared_hash,
            record_type="transcript_message",
        ),
        _candidate("other", score=0.035),
    )

    contribution = admit_recall(result, _policy())

    assert contribution.focus is not None
    assert contribution.offer is not None
    assert [item.candidate.record.identity.record_id for item in contribution.focus.items] == [
        "knowledge",
        "other",
    ]
    assert [item.identity.record_id for item in contribution.offer.items] == ["transcript"]
    assert [item.reason for item in contribution.offer.items] == ["duplicate_strong_reference"]
    assert contribution.diagnostics.duplicate_content_omitted == 1
    assert contribution.diagnostics.silent_count == 0
    assert contribution.diagnostics.admission_truncated is True


def test_recall_offer_ticket_commits_to_calibration_and_frontier_state() -> None:
    contribution = admit_recall(
        _result(_candidate("plausible", score=0.02)),
        _policy(),
    )
    assert contribution.offer is not None
    serialized = contribution.offer.model_dump(mode="python")

    changed_calibration = {**serialized, "calibration_version": "other-calibration-v1"}
    with pytest.raises(ValueError, match="complete contents"):
        RecallOffer.model_validate(changed_calibration)

    changed_frontier = {
        **serialized,
        "recall_truncated": True,
        "truncated": True,
    }
    with pytest.raises(ValueError, match="complete contents"):
        RecallOffer.model_validate(changed_frontier)


def test_automatic_recall_focus_and_offer_honor_complete_serialized_byte_bounds() -> None:
    result = _result(
        _candidate("large-strong", score=0.05, text="strong " * 300),
        _candidate("plausible", score=0.02),
    )
    unbounded = admit_recall(result, _policy())
    assert unbounded.focus is not None
    full_focus_bytes = len(
        canonical_durable_json_bytes(unbounded.focus.model_dump(mode="json"), "focus")
    )

    bounded_policy = _policy(max_focus_bytes=full_focus_bytes - 1)
    bounded = admit_recall(result, bounded_policy)

    assert bounded.focus is None
    assert bounded.offer is not None
    assert [item.identity.record_id for item in bounded.offer.items] == [
        "large-strong",
        "plausible",
    ]
    assert [item.reason for item in bounded.offer.items] == [
        "strong_match_not_focused",
        "calibrated_plausible_match",
    ]
    assert (
        len(canonical_durable_json_bytes(bounded.offer.model_dump(mode="json"), "offer"))
        <= bounded_policy.max_offer_bytes
    )
    assert bounded.diagnostics.focus_bound_omitted == 1


def test_complete_bound_preserves_record_type_diversity_selection_order() -> None:
    result = _result(
        _candidate("knowledge-one", score=0.05),
        _candidate("knowledge-two", score=0.049, text="large evidence " * 80),
        _candidate(
            "transcript-one",
            score=0.048,
            record_type="transcript_message",
        ),
    )
    unbounded = admit_recall(
        result,
        _policy(
            mode=AutomaticRecallMode.STRONG_MATCHES,
            max_injected_items=3,
        ),
    )
    assert unbounded.focus is not None
    focus_bytes = len(
        canonical_durable_json_bytes(unbounded.focus.model_dump(mode="json"), "focus")
    )

    bounded = admit_recall(
        result,
        _policy(
            mode=AutomaticRecallMode.STRONG_MATCHES,
            max_injected_items=3,
            max_focus_bytes=focus_bytes,
            max_offer_bytes=focus_bytes,
            max_total_bytes=focus_bytes,
        ),
    )

    assert bounded.focus is not None
    assert [item.candidate.record.identity.record_id for item in bounded.focus.items] == [
        "knowledge-one",
        "transcript-one",
    ]
    assert bounded.diagnostics.focus_bound_omitted == 1


def test_automatic_recall_outputs_are_defensive_and_byte_stable() -> None:
    result = _result(_candidate("strong", score=0.04), truncated=True)
    policy = _policy()

    first = admit_recall(result, policy)
    second = admit_recall(result, policy)

    assert canonical_durable_json_bytes(first.model_dump(mode="json"), "first") == (
        canonical_durable_json_bytes(second.model_dump(mode="json"), "second")
    )
    assert first.focus is not None
    assert first.focus.recall_truncated is True
    assert first.focus.truncated is True
    with pytest.raises(TypeError):
        _mutate_mapping(first.continuations, "new", "cursor")
    with pytest.raises(TypeError):
        _mutate_mapping(
            first.focus.items[0].candidate.record.locator,
            "entry_id",
            "mutated",
        )


def test_disabled_contributor_does_not_run_recall() -> None:
    class NeverRunEngine(RecallEngine):
        def __init__(self) -> None:
            pass

        async def recall(self, situation):
            raise AssertionError("disabled automatic recall ran the engine")

    contribution = asyncio.run(
        AutomaticRecallContributor(
            NeverRunEngine(),
            _policy(mode=AutomaticRecallMode.OFF),
        ).contribute(RecallSituation(query="disabled"))
    )

    assert contribution.diagnostics.recall_performed is False
    assert contribution.focus is None
    assert contribution.offer is None


def test_policy_rejects_unversioned_or_incoherent_calibration() -> None:
    with pytest.raises(ValueError, match="minimum_offer_score"):
        _policy(minimum_offer_score=0.04)
    with pytest.raises(ValueError, match="max_injected_items"):
        _policy(max_evaluated_candidates=1, max_injected_items=2)
    with pytest.raises(ValueError, match="calibration_version"):
        _policy(calibration_version=" ")
