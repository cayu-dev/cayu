from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from cayu.retrieval import (
    RankedRetrievalChannel,
    RankedRetrievalHit,
    RetrievalCandidateIdentity,
    WeightedReciprocalRankFusion,
    WeightedReciprocalRankFusionConfig,
)


def _identity(record_id: str) -> RetrievalCandidateIdentity:
    return RetrievalCandidateIdentity(
        record_type="knowledge",
        record_id=record_id,
        revision="3",
    )


def _hit(
    record_id: str,
    rank: int,
    *,
    representation: str,
    raw_score: float,
    features: dict[str, float] | None = None,
) -> RankedRetrievalHit:
    return RankedRetrievalHit(
        identity=_identity(record_id),
        rank=rank,
        representation=representation,
        content_hash=f"sha256:{representation}:{record_id}",
        explanations=(f"matched {representation}",),
        raw_score=raw_score,
        features=features or {},
    )


def _config(**updates) -> WeightedReciprocalRankFusionConfig:
    values = {
        "configuration_version": "test-baseline-v1",
        "channel_weights": {"lexical": 2.0, "semantic": 1.0},
        "rrf_k": 10.0,
        "feature_weights": {"exact": 0.02},
        "max_candidates_per_channel": 4,
        "fused_head_limit": 4,
    }
    values.update(updates)
    return WeightedReciprocalRankFusionConfig(**values)


def _channels(*, reverse: bool = False, raw_scale: float = 1.0):
    lexical_hits = (
        _hit("a", 1, representation="content", raw_score=0.01 * raw_scale),
        _hit(
            "b",
            2,
            representation="content",
            raw_score=999.0 * raw_scale,
            features={"exact": 1.0},
        ),
    )
    semantic_hits = (
        _hit(
            "b",
            1,
            representation="applicability",
            raw_score=-500.0 * raw_scale,
            features={"exact": 1.0},
        ),
        _hit("a", 2, representation="applicability", raw_score=1.0 * raw_scale),
    )
    channels = (
        RankedRetrievalChannel(
            channel="lexical",
            index_version="fts-v4",
            candidate_limit=4,
            hits=tuple(reversed(lexical_hits)) if reverse else lexical_hits,
        ),
        RankedRetrievalChannel(
            channel="semantic",
            index_version="embedding-v9",
            candidate_limit=4,
            hits=tuple(reversed(semantic_hits)) if reverse else semantic_hits,
        ),
    )
    return tuple(reversed(channels)) if reverse else channels


def test_weighted_rrf_uses_rank_and_versioned_features_not_raw_score_scale() -> None:
    strategy = WeightedReciprocalRankFusion()
    result = strategy.fuse(_channels(), _config())

    assert [candidate.identity.record_id for candidate in result.candidates] == ["b", "a"]
    by_id = {candidate.identity.record_id: candidate for candidate in result.candidates}
    assert math.isclose(by_id["a"].reciprocal_rank_score, 2 / 11 + 1 / 12)
    assert math.isclose(by_id["b"].reciprocal_rank_score, 2 / 12 + 1 / 11)
    assert by_id["b"].feature_adjustment == 0.02
    rescaled = strategy.fuse(_channels(raw_scale=1_000_000), _config())
    assert [candidate.identity for candidate in rescaled.candidates] == [
        candidate.identity for candidate in result.candidates
    ]
    assert [candidate.score for candidate in rescaled.candidates] == [
        candidate.score for candidate in result.candidates
    ]


def test_weighted_rrf_is_independent_of_channel_and_hit_input_order() -> None:
    strategy = WeightedReciprocalRankFusion()

    forward = strategy.fuse(_channels(), _config())
    reversed_input = strategy.fuse(_channels(reverse=True), _config())

    assert reversed_input == forward
    assert forward.diagnostics.configuration_sha256 == _config().fingerprint()
    assert [channel.channel for channel in forward.diagnostics.channels] == [
        "lexical",
        "semantic",
    ]


def test_weighted_rrf_collapses_duplicate_channel_hits_to_the_best_rank() -> None:
    duplicate = RankedRetrievalChannel(
        channel="lexical",
        index_version="fts-v4",
        candidate_limit=3,
        hits=(
            _hit("a", 1, representation="title", raw_score=1.0),
            _hit("a", 2, representation="content", raw_score=100.0),
        ),
    )
    config = _config(
        channel_weights={"lexical": 2.0},
        feature_weights={},
        max_candidates_per_channel=3,
    )

    result = WeightedReciprocalRankFusion().fuse((duplicate,), config)

    candidate = result.candidates[0]
    assert candidate.reciprocal_rank_score == 2 / 11
    assert candidate.channel_count == 1
    assert [match.rank for match in candidate.matches] == [1, 2]
    assert result.diagnostics.channels[0].unique_candidate_count == 1


def test_weighted_rrf_reports_lane_and_fused_head_truncation() -> None:
    lexical = RankedRetrievalChannel(
        channel="lexical",
        index_version="fts-v4",
        candidate_limit=2,
        hits=(
            _hit("b", 1, representation="content", raw_score=1.0),
            _hit("a", 2, representation="content", raw_score=0.5),
        ),
        truncated=True,
        continuation="opaque-page-2",
    )
    config = _config(
        channel_weights={"lexical": 1.0},
        feature_weights={},
        max_candidates_per_channel=2,
        fused_head_limit=1,
    )

    result = WeightedReciprocalRankFusion().fuse((lexical,), config)

    assert [candidate.identity.record_id for candidate in result.candidates] == ["b"]
    assert result.diagnostics.omitted_candidate_count == 1
    assert result.diagnostics.truncated is True
    assert result.diagnostics.truncation_reasons == (
        "channel:lexical",
        "fused_head_limit",
    )
    assert result.diagnostics.continuation_channels == ("lexical",)


def test_weighted_rrf_does_not_claim_unavailable_channel_continuation() -> None:
    lexical = RankedRetrievalChannel(
        channel="lexical",
        index_version="fts-v4",
        candidate_limit=1,
        hits=(_hit("a", 1, representation="content", raw_score=1.0),),
        truncated=True,
    )
    config = _config(
        channel_weights={"lexical": 1.0},
        feature_weights={},
        max_candidates_per_channel=1,
        fused_head_limit=1,
    )

    result = WeightedReciprocalRankFusion().fuse((lexical,), config)

    assert result.diagnostics.truncated is True
    assert result.diagnostics.truncation_reasons == ("channel:lexical",)
    assert result.diagnostics.continuation_channels == ()


def test_weighted_rrf_uses_canonical_identity_as_the_final_tie_break() -> None:
    channel = RankedRetrievalChannel(
        channel="lexical",
        index_version="fts-v4",
        candidate_limit=2,
        hits=(
            _hit("z", 1, representation="content", raw_score=1.0),
            _hit("a", 2, representation="content", raw_score=1.0),
        ),
    )
    # Equalize reciprocal scores with duplicate best ranks across separate lanes.
    other = RankedRetrievalChannel(
        channel="semantic",
        index_version="embedding-v9",
        candidate_limit=2,
        hits=(
            _hit("a", 1, representation="applicability", raw_score=1.0),
            _hit("z", 2, representation="applicability", raw_score=1.0),
        ),
    )
    config = _config(channel_weights={"lexical": 1.0, "semantic": 1.0}, feature_weights={})

    result = WeightedReciprocalRankFusion().fuse((channel, other), config)

    assert [candidate.identity.record_id for candidate in result.candidates] == ["a", "z"]


def test_weighted_rrf_rejects_incomplete_unbounded_or_inconsistent_inputs() -> None:
    with pytest.raises(ValidationError, match="ranks must be unique"):
        RankedRetrievalChannel(
            channel="lexical",
            index_version="fts-v4",
            candidate_limit=2,
            hits=(
                _hit("a", 1, representation="title", raw_score=1.0),
                _hit("b", 1, representation="content", raw_score=1.0),
            ),
        )

    strategy = WeightedReciprocalRankFusion()
    with pytest.raises(ValueError, match="exactly match configured channels"):
        strategy.fuse((_channels()[0],), _config())
    with pytest.raises(ValueError, match="candidate limit exceeds"):
        strategy.fuse(
            _channels(),
            _config(max_candidates_per_channel=1),
        )

    inconsistent = list(_channels())
    inconsistent[1] = inconsistent[1].model_copy(
        update={
            "hits": (
                _hit(
                    "b",
                    1,
                    representation="applicability",
                    raw_score=1.0,
                    features={"exact": 0.5},
                ),
                inconsistent[1].hits[1],
            )
        }
    )
    with pytest.raises(ValueError, match="inconsistent deterministic features"):
        strategy.fuse(tuple(inconsistent), _config())


def test_weighted_rrf_configuration_is_defensively_copied_and_versioned() -> None:
    weights = {"lexical": 1.0}
    config = WeightedReciprocalRankFusionConfig(
        configuration_version="frozen-v1",
        channel_weights=weights,
    )
    fingerprint = config.fingerprint()

    weights["lexical"] = 999.0

    assert config.channel_weights == {"lexical": 1.0}
    assert config.fingerprint() == fingerprint
    with pytest.raises(ValidationError, match="strategy_version"):
        WeightedReciprocalRankFusionConfig(
            configuration_version="frozen-v1",
            channel_weights={"lexical": 1.0},
            strategy_version="unknown",
        )


def test_weighted_rrf_configuration_is_deeply_immutable_and_copies_revalidate() -> None:
    config = WeightedReciprocalRankFusionConfig(
        configuration_version="immutable-v1",
        channel_weights={"lexical": 1.0},
        feature_weights={"exact": 0.25},
    )
    fingerprint = config.fingerprint()

    with pytest.raises(TypeError, match="cannot be mutated"):
        config.channel_weights["lexical"] = 2.0
    with pytest.raises(TypeError, match="cannot be mutated"):
        config.feature_weights.clear()

    assert config.fingerprint() == fingerprint
    assert config.model_dump(mode="json")["channel_weights"] == {"lexical": 1.0}

    copied = config.model_copy(update={"channel_weights": {"semantic": 2.0}})
    assert copied.channel_weights == {"semantic": 2.0}
    with pytest.raises(TypeError, match="cannot be mutated"):
        copied.channel_weights["semantic"] = 3.0
    with pytest.raises(ValidationError, match="enable at least one channel"):
        config.model_copy(update={"channel_weights": {}})
    with pytest.raises(ValidationError):
        config.model_copy(update={"feature_weights": {"exact": 1e20}})


@pytest.mark.parametrize(
    "updates",
    [
        {"channel_weights": {"lexical": 1e20}},
        {"feature_weights": {"exact": -1e20}},
        {"rrf_k": 1e20},
        {"max_candidates_per_channel": 2**63},
        {"fused_head_limit": 2**63},
    ],
)
def test_weighted_rrf_rejects_configurations_that_cannot_be_fingerprinted(
    updates: dict[str, object],
) -> None:
    values = {
        "configuration_version": "portable-v1",
        "channel_weights": {"lexical": 1.0},
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        WeightedReciprocalRankFusionConfig(**values)


def test_weighted_rrf_every_accepted_configuration_is_immediately_fingerprintable() -> None:
    config = WeightedReciprocalRankFusionConfig(
        configuration_version="portable-boundary-v1",
        channel_weights={"lexical": float(2**62)},
        rrf_k=0.5,
        feature_weights={"exact": -1.25},
        max_candidates_per_channel=2**63 - 1,
        fused_head_limit=2**63 - 1,
    )

    assert len(config.fingerprint()) == 64
