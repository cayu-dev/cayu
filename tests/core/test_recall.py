from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from itertools import repeat

import pytest

from cayu._validation import canonical_durable_json_bytes
from cayu.recall import (
    RecallEngine,
    RecallEngineConfig,
    RecallRecord,
    RecallResult,
    RecallSituation,
    RecallSource,
    RecallSourceResult,
    RecallSourceStatus,
    RecallSourceUnavailable,
)
from cayu.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    RankedRetrievalChannel,
    RankedRetrievalHit,
    RetrievalCandidateIdentity,
    RetrievalFusionResult,
    RetrievalFusionStrategy,
    WeightedReciprocalRankFusion,
    WeightedReciprocalRankFusionConfig,
)
from cayu.storage.memory import KnowledgeAccessScope

_NOW = datetime(2026, 8, 21, 7, 0, tzinfo=UTC)


def _record(source: str, record_id: str, text: str) -> RecallRecord:
    encoded = text.encode("utf-8")
    return RecallRecord(
        identity=RetrievalCandidateIdentity(
            record_type=source,
            record_id=record_id,
            revision="1",
        ),
        representation=f"{source}_text",
        text=text,
        text_complete=True,
        content_hash=sha256(encoded).hexdigest(),
        locator={"source": source, "nested": [record_id]},
    )


class _StaticRecallSource(RecallSource):
    def __init__(
        self,
        *,
        name: str,
        channel: str,
        record_id: str,
        delay: float = 0.0,
        required: bool = True,
        fail: bool = False,
    ) -> None:
        self.name = name
        self.channel_names = (channel,)
        self._record = _record(name, record_id, f"{name} evidence")
        self._delay = delay
        self._fail = fail
        super().__init__(required=required, candidate_limit=2)

    async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
        del situation
        await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError("private source detail")
        hit = RankedRetrievalHit(
            identity=self._record.identity,
            rank=1,
            representation=self._record.representation,
            content_hash=self._record.content_hash,
            explanations=(f"matched {self.name}",),
            raw_score=999.0,
        )
        return RecallSourceResult(
            source=self.name,
            channels=(
                RankedRetrievalChannel(
                    channel=self.channel_names[0],
                    index_version=f"{self.name}-v1",
                    candidate_limit=2,
                    hits=(hit,),
                ),
            ),
            records=(self._record,),
            coverage_complete=True,
        )


def _fusion_config(
    *channels: str,
    strategy_version: str = WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
) -> WeightedReciprocalRankFusionConfig:
    return WeightedReciprocalRankFusionConfig(
        configuration_version="recall-tests-v1",
        channel_weights={channel: 1.0 for channel in channels},
        max_candidates_per_channel=20,
        fused_head_limit=20,
        strategy_version=strategy_version,
    )


def _situation(**updates) -> RecallSituation:
    values = {"query": "Atlas", "current_time": _NOW}
    values.update(updates)
    return RecallSituation(**values)


def test_recall_situation_is_bounded_defensive_and_resolves_short_followups() -> None:
    recent = ["We selected project Atlas."]
    continuations = {"transcript.lexical": "cursor"}
    access_scope = KnowledgeAccessScope.for_namespace(
        "project:cayu",
        required_labels={"project": "cayu"},
    )
    situation = RecallSituation(
        query="Why?",
        recent_conversation=recent,
        work_context="Release investigation",
        knowledge_access_scope=access_scope,
        transcript_session_ids=("session-b", "session-a"),
        continuations=continuations,
        current_time=_NOW,
    )
    recent[0] = "mutated"
    continuations["transcript.lexical"] = "mutated"
    access_scope.allowed_namespaces.append("project:other")
    access_scope.required_labels["project"] = "other"

    assert situation.recent_conversation == ("We selected project Atlas.",)
    assert situation.transcript_session_ids == ("session-a", "session-b")
    assert situation.continuations == {"transcript.lexical": "cursor"}
    assert situation.knowledge_access_scope is not None
    assert situation.knowledge_access_scope.allowed_namespaces == ("project:cayu",)
    assert situation.knowledge_access_scope.required_labels == {"project": "cayu"}
    assert situation.retrieval_text() == ("Release investigation\nWe selected project Atlas.\nWhy?")
    with pytest.raises(TypeError):
        situation.continuations["new"] = "cursor"  # type: ignore[index]
    with pytest.raises(AttributeError):
        situation.knowledge_access_scope.allowed_namespaces.append("project:other")
    with pytest.raises(TypeError, match="cannot be mutated"):
        situation.knowledge_access_scope.required_labels["project"] = "other"
    with pytest.raises(ValueError, match="interaction-item bound"):
        RecallSituation(query="bounded", recent_conversation=repeat("message"))
    with pytest.raises(ValueError, match="more than 100 ids"):
        RecallSituation(query="bounded", transcript_session_ids=repeat("session"))
    default_situation = RecallSituation(query="bounded")
    with pytest.raises(TypeError, match="cannot be mutated"):
        default_situation.continuations["new"] = "cursor"  # type: ignore[index]


def test_recall_engine_is_completion_order_independent_and_byte_stable() -> None:
    async def run(reverse_delays: bool):
        first = _StaticRecallSource(
            name="alpha",
            channel="alpha.lexical",
            record_id="a",
            delay=0.02 if reverse_delays else 0.0,
        )
        second = _StaticRecallSource(
            name="beta",
            channel="beta.lexical",
            record_id="b",
            delay=0.0 if reverse_delays else 0.02,
        )
        engine = RecallEngine(
            (second, first),
            fusion_config=_fusion_config("alpha.lexical", "beta.lexical"),
        )
        return await engine.recall(_situation())

    slow_alpha = asyncio.run(run(True))
    slow_beta = asyncio.run(run(False))

    assert slow_alpha == slow_beta
    assert [candidate.record.identity.record_id for candidate in slow_alpha.candidates] == [
        "a",
        "b",
    ]
    assert canonical_durable_json_bytes(
        slow_alpha.model_dump(mode="json"), "recall result"
    ) == canonical_durable_json_bytes(slow_beta.model_dump(mode="json"), "recall result")


def test_recall_engine_reports_optional_failure_and_fails_closed_when_required() -> None:
    optional = _StaticRecallSource(
        name="optional",
        channel="optional.lexical",
        record_id="optional",
        required=False,
        fail=True,
    )
    optional_result = asyncio.run(
        RecallEngine(
            (optional,),
            fusion_config=_fusion_config("optional.lexical"),
        ).recall(_situation())
    )
    assert optional_result.candidates == ()
    assert optional_result.truncated is True
    assert optional_result.sources[0].status is RecallSourceStatus.UNAVAILABLE
    assert optional_result.sources[0].failure_code == "failed"
    assert "private source detail" not in optional_result.model_dump_json()

    required = _StaticRecallSource(
        name="required",
        channel="required.lexical",
        record_id="required",
        fail=True,
    )
    with pytest.raises(RecallSourceUnavailable) as caught:
        asyncio.run(
            RecallEngine(
                (required,),
                fusion_config=_fusion_config("required.lexical"),
            ).recall(_situation())
        )
    assert caught.value.source == "required"
    assert caught.value.code == "failed"


def test_recall_engine_rejects_custom_fusion_that_alters_source_provenance() -> None:
    class TamperingFusion(RetrievalFusionStrategy):
        strategy_version = "test.tampering.v1"

        def fuse(
            self,
            channels: tuple[RankedRetrievalChannel, ...],
            config: WeightedReciprocalRankFusionConfig,
        ) -> RetrievalFusionResult:
            reference_config = config.model_copy(
                update={"strategy_version": WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION}
            )
            valid = WeightedReciprocalRankFusion().fuse(channels, reference_config)
            diagnostics = valid.diagnostics.model_copy(
                update={
                    "strategy_version": config.strategy_version,
                    "configuration_sha256": config.fingerprint(),
                }
            )
            candidate = valid.candidates[0]
            tampered_match = candidate.matches[0].model_copy(update={"content_hash": "0" * 64})
            tampered_candidate = candidate.model_copy(update={"matches": (tampered_match,)})
            return valid.model_copy(
                update={
                    "candidates": (tampered_candidate,),
                    "diagnostics": diagnostics,
                }
            )

    source = _StaticRecallSource(
        name="source",
        channel="source.lexical",
        record_id="record",
    )
    engine = RecallEngine(
        (source,),
        fusion_config=_fusion_config(
            "source.lexical",
            strategy_version=TamperingFusion.strategy_version,
        ),
        fusion_strategy=TamperingFusion(),
    )

    with pytest.raises(ValueError, match="match provenance"):
        asyncio.run(engine.recall(_situation()))


def test_recall_engine_records_and_accepts_honest_custom_fusion_identity() -> None:
    class ReverseFusion(RetrievalFusionStrategy):
        strategy_version = "test.reverse.v1"

        def fuse(
            self,
            channels: tuple[RankedRetrievalChannel, ...],
            config: WeightedReciprocalRankFusionConfig,
        ) -> RetrievalFusionResult:
            reference_config = config.model_copy(
                update={"strategy_version": WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION}
            )
            valid = WeightedReciprocalRankFusion().fuse(channels, reference_config)
            return valid.model_copy(
                update={
                    "candidates": tuple(reversed(valid.candidates)),
                    "diagnostics": valid.diagnostics.model_copy(
                        update={
                            "strategy_version": config.strategy_version,
                            "configuration_sha256": config.fingerprint(),
                        }
                    ),
                }
            )

    sources = (
        _StaticRecallSource(name="alpha", channel="alpha.lexical", record_id="a"),
        _StaticRecallSource(name="beta", channel="beta.lexical", record_id="b"),
    )
    config = _fusion_config(
        "alpha.lexical",
        "beta.lexical",
        strategy_version=ReverseFusion.strategy_version,
    )

    result = asyncio.run(
        RecallEngine(
            sources,
            fusion_config=config,
            fusion_strategy=ReverseFusion(),
        ).recall(_situation())
    )

    assert [candidate.record.identity.record_id for candidate in result.candidates] == [
        "b",
        "a",
    ]
    assert result.fusion.strategy_version == ReverseFusion.strategy_version
    assert result.fusion.configuration_sha256 == config.fingerprint()


def test_recall_engine_rejects_fusion_identity_mismatch_at_registration() -> None:
    class CustomFusion(RetrievalFusionStrategy):
        strategy_version = "test.custom.v1"

        def fuse(
            self,
            channels: tuple[RankedRetrievalChannel, ...],
            config: WeightedReciprocalRankFusionConfig,
        ) -> RetrievalFusionResult:  # pragma: no cover - registration rejects first
            raise AssertionError((channels, config))

    source = _StaticRecallSource(
        name="source",
        channel="source.lexical",
        record_id="record",
    )
    with pytest.raises(ValueError, match="identity must match"):
        RecallEngine(
            (source,),
            fusion_config=_fusion_config("source.lexical"),
            fusion_strategy=CustomFusion(),
        )


def test_recall_engine_cancels_outstanding_source_work_at_global_timeout() -> None:
    class BlockingSource(RecallSource):
        name = "blocking"
        channel_names = ("blocking.lexical",)

        def __init__(self) -> None:
            super().__init__(required=False, candidate_limit=1)
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
            del situation
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

    async def run() -> None:
        source = BlockingSource()
        engine = RecallEngine(
            (source,),
            fusion_config=_fusion_config("blocking.lexical"),
            config=RecallEngineConfig(
                source_timeout_seconds=1,
                overall_timeout_seconds=0.01,
            ),
        )
        with pytest.raises(RecallSourceUnavailable) as caught:
            await engine.recall(_situation())
        assert caught.value.code == "overall_timeout"
        assert source.started.is_set()
        assert source.cancelled.is_set()

    asyncio.run(run())


def test_recall_byte_selection_handles_continuation_metadata_that_shrinks() -> None:
    class ShrinkingContinuationSource(RecallSource):
        name = "shrinking"
        channel_names = ("shrinking.lexical",)
        continuation_channels = channel_names

        def __init__(self) -> None:
            self._records = (
                _record(self.name, "first", "first evidence"),
                _record(self.name, "second", "second evidence"),
            )
            super().__init__(required=True, candidate_limit=2)

        async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
            del situation
            hits = tuple(
                RankedRetrievalHit(
                    identity=record.identity,
                    rank=rank,
                    representation=record.representation,
                    content_hash=record.content_hash,
                    continuation=("c" * 4_000 if rank == 1 else "finished"),
                )
                for rank, record in enumerate(self._records, start=1)
            )
            return RecallSourceResult(
                source=self.name,
                channels=(
                    RankedRetrievalChannel(
                        channel=self.channel_names[0],
                        index_version="shrinking-v1",
                        candidate_limit=2,
                        hits=hits,
                    ),
                ),
                records=self._records,
                coverage_complete=True,
            )

    async def run() -> tuple[RecallResult, RecallResult, int]:
        source = ShrinkingContinuationSource()
        full_config = _fusion_config("shrinking.lexical").model_copy(update={"fused_head_limit": 2})
        one_config = full_config.model_copy(update={"fused_head_limit": 1})
        full = await RecallEngine((source,), fusion_config=full_config).recall(_situation())
        one = await RecallEngine((source,), fusion_config=one_config).recall(_situation())
        full_bytes = len(
            canonical_durable_json_bytes(full.model_dump(mode="json"), "full recall result")
        )
        bounded = await RecallEngine(
            (source,),
            fusion_config=full_config,
            config=RecallEngineConfig(max_result_bytes=full_bytes),
        ).recall(_situation())
        return one, bounded, full_bytes

    one, bounded, full_bytes = asyncio.run(run())

    assert (
        len(canonical_durable_json_bytes(one.model_dump(mode="json"), "one-candidate result"))
        > full_bytes
    )
    assert [candidate.record.identity.record_id for candidate in bounded.candidates] == [
        "first",
        "second",
    ]
    assert bounded.continuations == {}


def test_recall_engine_rejects_continuations_for_non_pageable_channels() -> None:
    source = _StaticRecallSource(
        name="static",
        channel="static.lexical",
        record_id="static",
    )
    engine = RecallEngine(
        (source,),
        fusion_config=_fusion_config("static.lexical"),
    )

    with pytest.raises(ValueError, match="non-pageable channel"):
        asyncio.run(engine.recall(_situation(continuations={"static.lexical": "cursor"})))


def test_recall_engine_rejects_cursor_advancement_without_a_returned_hit() -> None:
    class EmptyAdvancingSource(RecallSource):
        name = "pageable"
        channel_names = ("pageable.lexical",)
        continuation_channels = channel_names

        def __init__(self) -> None:
            super().__init__(required=True, candidate_limit=1)

        async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
            del situation
            return RecallSourceResult(
                source=self.name,
                channels=(
                    RankedRetrievalChannel(
                        channel=self.channel_names[0],
                        index_version="pageable-v1",
                        candidate_limit=1,
                        truncated=True,
                        continuation="unearned-cursor",
                    ),
                ),
                records=(),
                coverage_complete=True,
            )

    engine = RecallEngine(
        (EmptyAdvancingSource(),),
        fusion_config=_fusion_config("pageable.lexical"),
    )

    with pytest.raises(ValueError, match="cannot advance its cursor without a hit"):
        asyncio.run(engine.recall(_situation()))
