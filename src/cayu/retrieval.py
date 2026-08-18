from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
from math import fsum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_json_value,
    require_finite,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)

WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION = "cayu.wrrf.v1"


class RetrievalCandidateIdentity(BaseModel):
    """Exact typed canonical revision identity shared by retrieval channels."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: str
    record_id: str
    revision: str

    @field_validator("record_type", "record_id", "revision")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    def sort_key(self) -> tuple[str, str, str]:
        return self.record_type, self.record_id, self.revision


class RankedRetrievalHit(BaseModel):
    """One bounded channel hit; raw scores are diagnostic and never fused."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: RetrievalCandidateIdentity
    rank: int
    representation: str
    content_hash: str
    explanations: tuple[str, ...] = ()
    raw_score: float | None = None
    features: dict[str, float] = Field(default_factory=dict)

    @field_validator("rank", mode="before")
    @classmethod
    def validate_rank(cls, value) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("`rank` must be a positive integer.")
        return value

    @field_validator("representation", "content_hash")
    @classmethod
    def validate_identity_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("explanations", mode="before")
    @classmethod
    def validate_explanations(cls, value) -> tuple[str, ...]:
        if type(value) is tuple:
            value = list(value)
        copied = copy_json_value(value, "explanations")
        if type(copied) is not list:
            raise ValueError("`explanations` must be a sequence.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`explanations[{index}]` must be a string.")
            explanation = require_clean_nonblank(item, f"explanations[{index}]")
            if explanation not in result:
                result.append(explanation)
        return tuple(result)

    @field_validator("raw_score", mode="before")
    @classmethod
    def validate_raw_score(cls, value) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`raw_score` must be a number.")
        return require_finite(float(value), "raw_score")

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, value) -> dict[str, float]:
        copied = copy_json_value(value, "features")
        if type(copied) is not dict:
            raise ValueError("`features` must be an object.")
        result: dict[str, float] = {}
        for raw_name, raw_value in copied.items():
            if type(raw_name) is not str:
                raise ValueError("Feature names must be strings.")
            name = require_clean_nonblank(raw_name, "feature name")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                raise ValueError(f"Feature {name!r} must be a number.")
            feature = require_finite(float(raw_value), f"features[{name!r}]")
            if feature < 0.0 or feature > 1.0:
                raise ValueError(f"Feature {name!r} must be between 0.0 and 1.0.")
            result[name] = feature
        return result


class RankedRetrievalChannel(BaseModel):
    """Bounded, independently ranked output from one retrieval channel."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    channel: str
    index_version: str
    candidate_limit: int
    hits: tuple[RankedRetrievalHit, ...] = ()
    truncated: bool = False
    continuation: str | None = None

    @field_validator("channel", "index_version")
    @classmethod
    def validate_identity_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("candidate_limit", mode="before")
    @classmethod
    def validate_candidate_limit(cls, value) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("`candidate_limit` must be a positive integer.")
        return value

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @field_validator("continuation")
    @classmethod
    def validate_continuation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "continuation")

    @model_validator(mode="after")
    def validate_bounded_ranking(self) -> RankedRetrievalChannel:
        if len(self.hits) > self.candidate_limit:
            raise ValueError("Channel hits exceed `candidate_limit`.")
        ranks = [hit.rank for hit in self.hits]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Channel hit ranks must be unique.")
        if any(rank > self.candidate_limit for rank in ranks):
            raise ValueError("Channel hit rank exceeds `candidate_limit`.")
        if self.continuation is not None and not self.truncated:
            raise ValueError("A channel continuation requires `truncated=True`.")
        return self


class WeightedReciprocalRankFusionConfig(BaseModel):
    """Versioned, evaluation-owned WRRF weights and hard work budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    configuration_version: str
    channel_weights: dict[str, float]
    rrf_k: float = 60.0
    feature_weights: dict[str, float] = Field(default_factory=dict)
    max_candidates_per_channel: int = 100
    fused_head_limit: int = 50
    strategy_version: str = WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION

    @field_validator("configuration_version", "strategy_version")
    @classmethod
    def validate_versions(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("strategy_version")
    @classmethod
    def validate_strategy_version(cls, value: str) -> str:
        if value != WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION:
            raise ValueError(
                f"`strategy_version` must be {WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION!r}."
            )
        return value

    @field_validator("channel_weights", mode="before")
    @classmethod
    def validate_channel_weights(cls, value) -> dict[str, float]:
        return _copy_weight_map(value, "channel_weights", strictly_positive=True)

    @field_validator("feature_weights", mode="before")
    @classmethod
    def validate_feature_weights(cls, value) -> dict[str, float]:
        return _copy_weight_map(value, "feature_weights", strictly_positive=False)

    @field_validator("rrf_k", mode="before")
    @classmethod
    def validate_rrf_k(cls, value) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`rrf_k` must be a number.")
        result = require_finite(float(value), "rrf_k")
        if result <= 0.0:
            raise ValueError("`rrf_k` must be greater than 0.")
        return result

    @field_validator("max_candidates_per_channel", "fused_head_limit", mode="before")
    @classmethod
    def validate_limits(cls, value, info) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"`{info.field_name}` must be a positive integer.")
        return value

    @model_validator(mode="after")
    def validate_enabled_channels(self) -> WeightedReciprocalRankFusionConfig:
        if not self.channel_weights:
            raise ValueError("`channel_weights` must enable at least one channel.")
        return self

    def fingerprint(self) -> str:
        payload = canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "weighted reciprocal-rank fusion configuration",
        )
        return sha256(payload).hexdigest()


class FusedChannelMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    channel: str
    index_version: str
    rank: int
    representation: str
    content_hash: str
    explanations: tuple[str, ...]
    raw_score: float | None


class FusedRetrievalCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: RetrievalCandidateIdentity
    score: float
    reciprocal_rank_score: float
    feature_adjustment: float
    features: dict[str, float]
    best_rank: int
    channel_count: int
    matches: tuple[FusedChannelMatch, ...]


class RetrievalChannelDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    channel: str
    index_version: str
    candidate_limit: int
    hit_count: int
    unique_candidate_count: int
    truncated: bool
    continuation_available: bool


class RetrievalFusionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    strategy_version: str
    configuration_version: str
    configuration_sha256: str
    channels: tuple[RetrievalChannelDiagnostics, ...]
    unique_candidate_count: int
    returned_candidate_count: int
    omitted_candidate_count: int
    truncated: bool
    truncation_reasons: tuple[str, ...]
    continuation_channels: tuple[str, ...]


class RetrievalFusionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    candidates: tuple[FusedRetrievalCandidate, ...]
    diagnostics: RetrievalFusionDiagnostics


class RetrievalFusionStrategy(ABC):
    """Typed replacement seam for bounded, diagnostic-preserving fusion."""

    @abstractmethod
    def fuse(
        self,
        channels: tuple[RankedRetrievalChannel, ...],
        config: WeightedReciprocalRankFusionConfig,
    ) -> RetrievalFusionResult:
        """Fuse bounded ranked channels without comparing their raw scores."""


class WeightedReciprocalRankFusion(RetrievalFusionStrategy):
    """Deterministic weighted reciprocal-rank fusion reference strategy."""

    def fuse(
        self,
        channels: tuple[RankedRetrievalChannel, ...],
        config: WeightedReciprocalRankFusionConfig,
    ) -> RetrievalFusionResult:
        config = _copy_fusion_config(config)
        copied_channels = tuple(_copy_ranked_channel(channel) for channel in channels)
        _validate_fusion_channels(copied_channels, config)

        candidates: dict[tuple[str, str, str], _CandidateAccumulator] = {}
        channel_diagnostics: list[RetrievalChannelDiagnostics] = []
        continuation_channels: list[str] = []
        for channel in sorted(copied_channels, key=lambda item: item.channel):
            unique_in_channel: set[tuple[str, str, str]] = set()
            if channel.continuation is not None:
                continuation_channels.append(channel.channel)
            for hit in sorted(channel.hits, key=lambda item: item.rank):
                identity_key = hit.identity.sort_key()
                unique_in_channel.add(identity_key)
                accumulator = candidates.get(identity_key)
                if accumulator is None:
                    accumulator = _CandidateAccumulator(hit)
                    candidates[identity_key] = accumulator
                else:
                    accumulator.require_consistent_features(hit)
                accumulator.add_match(channel, hit)
            channel_diagnostics.append(
                RetrievalChannelDiagnostics(
                    channel=channel.channel,
                    index_version=channel.index_version,
                    candidate_limit=channel.candidate_limit,
                    hit_count=len(channel.hits),
                    unique_candidate_count=len(unique_in_channel),
                    truncated=channel.truncated,
                    continuation_available=channel.continuation is not None,
                )
            )

        fused = [
            accumulator.fused_candidate(config) for _, accumulator in sorted(candidates.items())
        ]
        fused.sort(
            key=lambda item: (
                -item.score,
                item.best_rank,
                -item.channel_count,
                item.identity.sort_key(),
            )
        )
        returned = tuple(fused[: config.fused_head_limit])
        omitted = len(fused) - len(returned)
        truncation_reasons = [
            f"channel:{channel.channel}"
            for channel in sorted(copied_channels, key=lambda item: item.channel)
            if channel.truncated
        ]
        if omitted:
            truncation_reasons.append("fused_head_limit")
        diagnostics = RetrievalFusionDiagnostics(
            strategy_version=config.strategy_version,
            configuration_version=config.configuration_version,
            configuration_sha256=config.fingerprint(),
            channels=tuple(channel_diagnostics),
            unique_candidate_count=len(fused),
            returned_candidate_count=len(returned),
            omitted_candidate_count=omitted,
            truncated=bool(truncation_reasons),
            truncation_reasons=tuple(truncation_reasons),
            continuation_channels=tuple(continuation_channels),
        )
        return RetrievalFusionResult(candidates=returned, diagnostics=diagnostics)


class _CandidateAccumulator:
    def __init__(self, hit: RankedRetrievalHit) -> None:
        self.identity = _copy_candidate_identity(hit.identity)
        self.features = dict(hit.features)
        self.matches: list[tuple[RankedRetrievalChannel, RankedRetrievalHit]] = []

    def require_consistent_features(self, hit: RankedRetrievalHit) -> None:
        if hit.features != self.features:
            raise ValueError(
                "Retrieval channels reported inconsistent deterministic features "
                f"for candidate {self.identity.sort_key()!r}."
            )

    def add_match(self, channel: RankedRetrievalChannel, hit: RankedRetrievalHit) -> None:
        self.matches.append((channel, hit))

    def fused_candidate(
        self,
        config: WeightedReciprocalRankFusionConfig,
    ) -> FusedRetrievalCandidate:
        best_by_channel: dict[str, int] = {}
        for channel, hit in self.matches:
            current = best_by_channel.get(channel.channel)
            if current is None or hit.rank < current:
                best_by_channel[channel.channel] = hit.rank
        reciprocal_rank_score = fsum(
            config.channel_weights[channel] / (config.rrf_k + best_by_channel[channel])
            for channel in sorted(best_by_channel)
        )
        feature_adjustment = fsum(
            config.feature_weights[name] * self.features.get(name, 0.0)
            for name in sorted(config.feature_weights)
        )
        matches = tuple(
            FusedChannelMatch(
                channel=channel.channel,
                index_version=channel.index_version,
                rank=hit.rank,
                representation=hit.representation,
                content_hash=hit.content_hash,
                explanations=tuple(hit.explanations),
                raw_score=hit.raw_score,
            )
            for channel, hit in sorted(
                self.matches,
                key=lambda item: (
                    item[0].channel,
                    item[1].rank,
                    item[1].representation,
                    item[1].content_hash,
                ),
            )
        )
        return FusedRetrievalCandidate(
            identity=_copy_candidate_identity(self.identity),
            score=reciprocal_rank_score + feature_adjustment,
            reciprocal_rank_score=reciprocal_rank_score,
            feature_adjustment=feature_adjustment,
            features=dict(self.features),
            best_rank=min(best_by_channel.values()),
            channel_count=len(best_by_channel),
            matches=matches,
        )


def _copy_weight_map(
    value: Any,
    field_name: str,
    *,
    strictly_positive: bool,
) -> dict[str, float]:
    copied = copy_json_value(value, field_name)
    if type(copied) is not dict:
        raise ValueError(f"`{field_name}` must be an object.")
    result: dict[str, float] = {}
    for raw_name, raw_weight in copied.items():
        if type(raw_name) is not str:
            raise ValueError(f"`{field_name}` keys must be strings.")
        name = require_clean_nonblank(raw_name, f"{field_name} key")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
            raise ValueError(f"Weight {name!r} must be a number.")
        weight = require_finite(float(raw_weight), f"{field_name}[{name!r}]")
        if strictly_positive and weight <= 0.0:
            raise ValueError(f"Channel weight {name!r} must be greater than 0.")
        result[name] = weight
    return result


def _copy_candidate_identity(identity: RetrievalCandidateIdentity) -> RetrievalCandidateIdentity:
    if type(identity) is not RetrievalCandidateIdentity:
        raise TypeError("identity must be a RetrievalCandidateIdentity.")
    return RetrievalCandidateIdentity(
        record_type=identity.record_type,
        record_id=identity.record_id,
        revision=identity.revision,
    )


def _copy_ranked_hit(hit: RankedRetrievalHit) -> RankedRetrievalHit:
    if type(hit) is not RankedRetrievalHit:
        raise TypeError("Channel hits must be RankedRetrievalHit instances.")
    return RankedRetrievalHit(
        identity=_copy_candidate_identity(hit.identity),
        rank=hit.rank,
        representation=hit.representation,
        content_hash=hit.content_hash,
        explanations=tuple(hit.explanations),
        raw_score=hit.raw_score,
        features=dict(hit.features),
    )


def _copy_ranked_channel(channel: RankedRetrievalChannel) -> RankedRetrievalChannel:
    if type(channel) is not RankedRetrievalChannel:
        raise TypeError("channels must contain RankedRetrievalChannel instances.")
    return RankedRetrievalChannel(
        channel=channel.channel,
        index_version=channel.index_version,
        candidate_limit=channel.candidate_limit,
        hits=tuple(_copy_ranked_hit(hit) for hit in channel.hits),
        truncated=channel.truncated,
        continuation=channel.continuation,
    )


def _copy_fusion_config(
    config: WeightedReciprocalRankFusionConfig,
) -> WeightedReciprocalRankFusionConfig:
    if type(config) is not WeightedReciprocalRankFusionConfig:
        raise TypeError("config must be a WeightedReciprocalRankFusionConfig.")
    return WeightedReciprocalRankFusionConfig(
        configuration_version=config.configuration_version,
        channel_weights=dict(config.channel_weights),
        rrf_k=config.rrf_k,
        feature_weights=dict(config.feature_weights),
        max_candidates_per_channel=config.max_candidates_per_channel,
        fused_head_limit=config.fused_head_limit,
        strategy_version=config.strategy_version,
    )


def _validate_fusion_channels(
    channels: tuple[RankedRetrievalChannel, ...],
    config: WeightedReciprocalRankFusionConfig,
) -> None:
    channel_ids = [channel.channel for channel in channels]
    if len(channel_ids) != len(set(channel_ids)):
        raise ValueError("Retrieval channel identities must be unique.")
    configured = set(config.channel_weights)
    supplied = set(channel_ids)
    if supplied != configured:
        missing = sorted(configured - supplied)
        unexpected = sorted(supplied - configured)
        raise ValueError(
            "Supplied retrieval channels must exactly match configured channels "
            f"(missing={missing!r}, unexpected={unexpected!r})."
        )
    for channel in channels:
        if channel.candidate_limit > config.max_candidates_per_channel:
            raise ValueError(
                f"Channel {channel.channel!r} candidate limit exceeds `max_candidates_per_channel`."
            )


__all__ = [
    "WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION",
    "FusedChannelMatch",
    "FusedRetrievalCandidate",
    "RankedRetrievalChannel",
    "RankedRetrievalHit",
    "RetrievalCandidateIdentity",
    "RetrievalChannelDiagnostics",
    "RetrievalFusionDiagnostics",
    "RetrievalFusionResult",
    "RetrievalFusionStrategy",
    "WeightedReciprocalRankFusion",
    "WeightedReciprocalRankFusionConfig",
]
