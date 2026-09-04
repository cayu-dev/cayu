from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import repeat

import pytest

from cayu._validation import canonical_durable_json_bytes
from cayu.core.messages import Message
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    TRANSCRIPT_LEXICAL_CHANNEL,
    KnowledgeRecallSource,
    RecallEngine,
    RecallEngineConfig,
    RecallRecord,
    RecallResult,
    RecallSituation,
    RecallSource,
    RecallSourceResult,
    RecallSourceStatus,
    RecallSourceUnavailable,
    TranscriptRecallSource,
)
from cayu.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    RankedRetrievalChannel,
    RankedRetrievalHit,
    RetrievalCandidateIdentity,
    WeightedReciprocalRankFusionConfig,
)
from cayu.runtime.sessions import (
    MAX_SESSION_ID_BYTES,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    TranscriptSearchQuery,
)
from cayu.storage.memory import (
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeLineageRole,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeSearchMode,
    KnowledgeStatus,
)
from cayu.storage.sqlite import SQLiteSessionStore

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
    before_indexes = {"session-a": 3}
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
        transcript_before_indexes=before_indexes,
        continuations=continuations,
        current_time=_NOW,
    )
    recent[0] = "mutated"
    continuations["transcript.lexical"] = "mutated"
    before_indexes["session-a"] = 99
    access_scope.allowed_namespaces.append("project:other")
    access_scope.required_labels["project"] = "other"

    assert situation.recent_conversation == ("We selected project Atlas.",)
    assert situation.transcript_session_ids == ("session-a", "session-b")
    assert situation.continuations == {"transcript.lexical": "cursor"}
    assert situation.transcript_before_indexes == {"session-a": 3}
    assert situation.knowledge_access_scope is not None
    assert situation.knowledge_access_scope.allowed_namespaces == ("project:cayu",)
    assert situation.knowledge_access_scope.required_labels == {"project": "cayu"}
    assert situation.retrieval_text() == ("Release investigation\nWe selected project Atlas.\nWhy?")
    with pytest.raises(TypeError):
        situation.continuations["new"] = "cursor"  # type: ignore[index]
    with pytest.raises(TypeError):
        situation.transcript_before_indexes["session-a"] = 4  # type: ignore[index]
    with pytest.raises(AttributeError):
        situation.knowledge_access_scope.allowed_namespaces.append("project:other")
    with pytest.raises(TypeError, match="cannot be mutated"):
        situation.knowledge_access_scope.required_labels["project"] = "other"
    with pytest.raises(ValueError, match="interaction-item bound"):
        RecallSituation(query="bounded", recent_conversation=repeat("message"))
    with pytest.raises(ValueError, match="more than 100 ids"):
        RecallSituation(query="bounded", transcript_session_ids=repeat("session"))
    with pytest.raises(ValueError, match="more than 100"):
        TranscriptSearchQuery(text="bounded", session_ids=repeat("session"))
    with pytest.raises(ValueError, match="selected session scope"):
        TranscriptSearchQuery(
            text="bounded",
            session_ids=("session-a",),
            before_transcript_indexes={"session-b": 1},
        )
    with pytest.raises(ValueError, match="belong to transcript_session_ids"):
        RecallSituation(
            query="bounded",
            transcript_session_ids=("session-a",),
            transcript_before_indexes={"session-b": 1},
        )
    with pytest.raises(ValueError, match="at most .* UTF-8 bytes"):
        TranscriptSearchQuery(
            text="bounded",
            session_ids=("s" * (MAX_SESSION_ID_BYTES + 1),),
        )
    default_situation = RecallSituation(query="bounded")
    with pytest.raises(TypeError, match="cannot be mutated"):
        default_situation.continuations["new"] = "cursor"  # type: ignore[index]
    with pytest.raises(ValueError, match="cannot contain more than 6 groups"):
        RecallSituation(
            query="bounded",
            knowledge_aspect_groups=tuple((f"aspect:{index}",) for index in range(7)),
        )
    with pytest.raises(ValueError, match="requires exact aspect groups"):
        RecallSituation(query="bounded", knowledge_filter_only=True)


def test_in_memory_transcript_search_is_cooperatively_cancellable() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(agent_name="agent", session_id="bounded-search", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            "bounded-search",
            [Message.text("assistant", f"cancellation marker {index}") for index in range(600)],
            interaction_id="bulk-interaction",
        )
        search = asyncio.create_task(
            store.search_transcript(
                TranscriptSearchQuery(
                    text="cancellation marker",
                    session_ids=("bounded-search",),
                    max_records_scanned=1_000,
                )
            )
        )
        await asyncio.sleep(0)
        assert not search.done()
        search.cancel()
        with pytest.raises(asyncio.CancelledError):
            await search

        await store.append_transcript_messages(
            "bounded-search",
            [Message.text("assistant", "postcancellationevidence")],
            interaction_id="post-cancellation-interaction",
        )
        recovered = await store.search_transcript(
            TranscriptSearchQuery(
                text="postcancellationevidence",
                session_ids=("bounded-search",),
            )
        )
        assert [hit.transcript_index for hit in recovered.hits] == [600]

    asyncio.run(run())


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


def test_recall_timeout_returns_promptly_while_sqlite_read_remains_fenced(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "recall-timeout.sqlite")
    worker_started = threading.Event()
    release_worker = threading.Event()

    class BlockingSQLiteSource(RecallSource):
        name = "sqlite"
        channel_names = ("sqlite.lexical",)

        def __init__(self) -> None:
            super().__init__(required=False, candidate_limit=1)

        async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
            del situation

            def blocked_read(_connection) -> None:
                worker_started.set()
                if not release_worker.wait(timeout=5):
                    raise TimeoutError("test did not release the SQLite reader")

            await store._run_read(blocked_read)
            raise AssertionError("cancelled read unexpectedly returned")

    async def run() -> tuple[RecallResult, float]:
        engine = RecallEngine(
            (BlockingSQLiteSource(),),
            fusion_config=_fusion_config("sqlite.lexical"),
            config=RecallEngineConfig(
                source_timeout_seconds=0.02,
                overall_timeout_seconds=0.1,
            ),
        )
        started = time.monotonic()
        result = await engine.recall(_situation())
        elapsed = time.monotonic() - started
        assert worker_started.is_set()
        assert store._read_lock.locked()
        release_worker.set()
        await store.close()
        return result, elapsed

    result, elapsed = asyncio.run(run())

    assert elapsed < 0.08
    assert result.sources[0].status is RecallSourceStatus.UNAVAILABLE
    assert result.sources[0].failure_code == "timeout"


def test_recall_engine_bounds_the_complete_serialized_result_and_keeps_a_fused_prefix() -> None:
    sources = (
        _StaticRecallSource(name="alpha", channel="alpha.lexical", record_id="a"),
        _StaticRecallSource(name="beta", channel="beta.lexical", record_id="b"),
    )

    async def run() -> tuple[RecallResult, int]:
        complete = await RecallEngine(
            sources,
            fusion_config=_fusion_config("alpha.lexical", "beta.lexical"),
        ).recall(_situation())
        one_candidate = RecallResult(
            engine_version=complete.engine_version,
            situation_sha256=complete.situation_sha256,
            candidates=complete.candidates[:1],
            fusion=complete.fusion,
            sources=complete.sources,
            continuations=complete.continuations,
            truncated=True,
            omitted_by_result_bytes=1,
        )
        exact_limit = len(
            canonical_durable_json_bytes(
                one_candidate.model_dump(mode="json"),
                "recall result",
            )
        )
        bounded = await RecallEngine(
            sources,
            fusion_config=_fusion_config("alpha.lexical", "beta.lexical"),
            config=RecallEngineConfig(max_result_bytes=exact_limit),
        ).recall(_situation())
        with pytest.raises(ValueError, match="metadata exceeded"):
            await RecallEngine(
                sources,
                fusion_config=_fusion_config("alpha.lexical", "beta.lexical"),
                config=RecallEngineConfig(max_result_bytes=1),
            ).recall(_situation())
        return bounded, exact_limit

    result, exact_limit = asyncio.run(run())
    assert len(result.candidates) == 1
    assert result.omitted_by_result_bytes == 1
    assert result.truncated is True
    assert (
        len(canonical_durable_json_bytes(result.model_dump(mode="json"), "recall result"))
        == exact_limit
    )
    with pytest.raises(TypeError, match="cannot be mutated"):
        result.candidates[0].fused.features["mutated"] = 1.0
    default_payload = result.model_dump(mode="python")
    default_payload.pop("continuations")
    default_result = RecallResult.model_validate(default_payload)
    with pytest.raises(TypeError, match="cannot be mutated"):
        default_result.continuations["new"] = "cursor"  # type: ignore[index]


def test_recall_result_returns_exact_channel_continuations() -> None:
    async def run():
        sessions = InMemorySessionStore()
        await sessions.create(
            RunRequest(agent_name="agent", session_id="atlas-session", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await sessions.append_transcript_messages(
            "atlas-session",
            [
                Message.text("assistant", "Older Atlas evidence"),
                Message.text("assistant", "Newer Atlas evidence"),
            ],
            interaction_id="atlas-interaction",
        )
        engine = RecallEngine(
            (TranscriptRecallSource(sessions, required=True, candidate_limit=1),),
            fusion_config=_fusion_config(TRANSCRIPT_LEXICAL_CHANNEL),
        )
        first = await engine.recall(_situation(transcript_session_ids=("atlas-session",)))
        second = await engine.recall(
            _situation(
                transcript_session_ids=("atlas-session",),
                continuations=first.continuations,
            )
        )
        with pytest.raises(ValueError, match="unknown channel"):
            await engine.recall(_situation(continuations={"unknown": "cursor"}))
        return first, second

    first, second = asyncio.run(run())

    assert set(first.continuations) == {TRANSCRIPT_LEXICAL_CHANNEL}
    assert first.fusion.continuation_channels == (TRANSCRIPT_LEXICAL_CHANNEL,)
    assert first.candidates[0].record.locator["transcript_index"] == 1
    assert second.continuations == {}
    assert second.candidates[0].record.locator["transcript_index"] == 0
    with pytest.raises(TypeError):
        first.continuations["other"] = "cursor"  # type: ignore[index]


def test_recall_continuation_advances_only_past_the_visible_fused_prefix() -> None:
    async def run() -> list[int]:
        sessions = InMemorySessionStore()
        await sessions.create(
            RunRequest(agent_name="agent", session_id="atlas-session", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await sessions.append_transcript_messages(
            "atlas-session",
            [Message.text("assistant", f"Atlas evidence {index}") for index in range(5)],
            interaction_id="atlas-interaction",
        )
        source = TranscriptRecallSource(sessions, required=True, candidate_limit=4)
        config = _fusion_config(TRANSCRIPT_LEXICAL_CHANNEL).model_copy(
            update={"fused_head_limit": 2}
        )
        engine = RecallEngine((source,), fusion_config=config)
        continuations: dict[str, str] = {}
        observed: list[int] = []
        while True:
            page = await engine.recall(
                _situation(
                    transcript_session_ids=("atlas-session",),
                    continuations=continuations,
                )
            )
            observed.extend(
                int(candidate.record.locator["transcript_index"]) for candidate in page.candidates
            )
            continuations = dict(page.continuations)
            if not continuations:
                break
        return observed

    assert asyncio.run(run()) == [4, 3, 2, 1, 0]


def test_recall_byte_clipping_does_not_advance_past_omitted_transcript_hits() -> None:
    async def run() -> tuple[list[int], list[int]]:
        sessions = InMemorySessionStore()
        await sessions.create(
            RunRequest(agent_name="agent", session_id="atlas-session", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await sessions.append_transcript_messages(
            "atlas-session",
            [
                Message.text("assistant", f"Atlas evidence {index} " + "x" * 1_000)
                for index in range(6)
            ],
            interaction_id="atlas-interaction",
        )
        source = TranscriptRecallSource(sessions, required=True, candidate_limit=5)
        fusion_config = _fusion_config(TRANSCRIPT_LEXICAL_CHANNEL)
        complete = await RecallEngine((source,), fusion_config=fusion_config).recall(
            _situation(transcript_session_ids=("atlas-session",))
        )
        complete_bytes = len(
            canonical_durable_json_bytes(
                complete.model_dump(mode="json"),
                "complete recall result",
            )
        )
        clipped = await RecallEngine(
            (source,),
            fusion_config=fusion_config,
            config=RecallEngineConfig(max_result_bytes=complete_bytes - 1),
        ).recall(_situation(transcript_session_ids=("atlas-session",)))
        remainder = await RecallEngine((source,), fusion_config=fusion_config).recall(
            _situation(
                transcript_session_ids=("atlas-session",),
                continuations=clipped.continuations,
            )
        )
        return (
            [int(candidate.record.locator["transcript_index"]) for candidate in clipped.candidates],
            [
                int(candidate.record.locator["transcript_index"])
                for candidate in remainder.candidates
            ],
        )

    clipped, remainder = asyncio.run(run())
    assert 0 < len(clipped) < 5
    assert clipped + remainder == [5, 4, 3, 2, 1, 0]


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


def test_built_in_sources_fuse_exact_current_knowledge_and_transcript_evidence() -> None:
    async def run():
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = InMemoryKnowledgeStore(access_scope=scope)
        revision_one = await knowledge.create_entry(
            KnowledgeEntry(
                id="release-date",
                text="Obsolete release date was Thursday",
                namespace="project:cayu",
            )
        )
        await knowledge.append_entry_revision(
            revision_one.model_copy(
                update={
                    "revision": 2,
                    "text": "Atlas canonical release date is Friday",
                    "updated_at": revision_one.updated_at + timedelta(microseconds=1),
                }
            ),
            expected_revision=1,
        )

        sessions = InMemorySessionStore()
        await sessions.create(
            RunRequest(agent_name="agent", session_id="atlas-session", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await sessions.append_transcript_messages(
            "atlas-session",
            [Message.text("assistant", "Atlas planning discussion preferred Monday")],
            interaction_id="atlas-interaction",
        )

        engine = RecallEngine(
            (
                KnowledgeRecallSource(knowledge),
                TranscriptRecallSource(sessions),
            ),
            fusion_config=_fusion_config(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
        )
        return await engine.recall(
            _situation(
                knowledge_access_scope=scope,
                knowledge_namespace="project:cayu",
                transcript_session_ids=("atlas-session",),
            )
        )

    result = asyncio.run(run())

    assert {candidate.record.identity.record_type for candidate in result.candidates} == {
        "knowledge_entry",
        "transcript_message",
    }
    knowledge_candidate = next(
        candidate
        for candidate in result.candidates
        if candidate.record.identity.record_type == "knowledge_entry"
    )
    transcript_candidate = next(
        candidate
        for candidate in result.candidates
        if candidate.record.identity.record_type == "transcript_message"
    )
    assert knowledge_candidate.record.identity.revision == "2"
    assert knowledge_candidate.record.locator == {
        "entry_id": "release-date",
        "entry_revision": 2,
    }
    assert "Obsolete" not in knowledge_candidate.record.text
    assert transcript_candidate.record.locator == {
        "session_id": "atlas-session",
        "interaction_id": "atlas-interaction",
        "transcript_index": 0,
        "text_part_indexes": [0],
    }
    assert [source.status for source in result.sources] == [
        RecallSourceStatus.PARTIAL,
        RecallSourceStatus.COMPLETE,
    ]
    assert result.sources[0].failure_code == "semantic_unsupported"
    assert result.truncated is True


def test_knowledge_recall_runs_raw_semantic_text_with_exact_facet_fallback() -> None:
    class RecordingEmbeddingProvider(TextEmbeddingProvider):
        name = "recording-test"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.calls.append(list(request.texts))
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=index, vector=[1.0])
                    for index, _text in enumerate(request.texts)
                ],
            )

    async def run() -> tuple[RecallSourceResult, RecallSourceResult, list[list[str]]]:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        facet = "scope:repository:cayu"
        plain = InMemoryKnowledgeStore(access_scope=scope)
        provider = RecordingEmbeddingProvider()
        embedded = InMemoryEmbeddingKnowledgeStore(
            access_scope=scope,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=1,
        )
        for store in (plain, embedded):
            await store.create_entry(
                KnowledgeEntry(
                    id="facet-target",
                    text="Exact facet target",
                    namespace="project:cayu",
                    aspects=[facet],
                )
            )
        await embedded.process_embedding_changes("raw-semantic-index", "worker:test")
        provider.calls.clear()
        situation = _situation(
            query="!!!",
            knowledge_access_scope=scope,
            knowledge_namespace="project:cayu",
            knowledge_aspect_groups=((facet,),),
        )
        return (
            await KnowledgeRecallSource(plain).retrieve(situation),
            await KnowledgeRecallSource(embedded).retrieve(situation),
            provider.calls,
        )

    plain_result, embedded_result, calls = asyncio.run(run())

    assert [record.locator["entry_id"] for record in plain_result.records] == ["facet-target"]
    assert plain_result.partial_reason == "semantic_unsupported"
    assert [record.locator["entry_id"] for record in embedded_result.records] == ["facet-target"]
    assert embedded_result.partial_reason is None
    assert calls == [["!!!"]]


def test_knowledge_recall_can_explain_supersession_and_unresolved_alternatives() -> None:
    async def run() -> RecallResult:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = InMemoryKnowledgeStore(access_scope=scope)
        replacement = await knowledge.create_entry(
            KnowledgeEntry(
                id="release-current",
                text="Atlas release date is Friday",
                namespace="project:cayu",
            )
        )
        predecessor = await knowledge.create_entry(
            KnowledgeEntry(
                id="release-obsolete",
                text="Atlas release date was Thursday",
                namespace="project:cayu",
            )
        )
        alternative = await knowledge.create_entry(
            KnowledgeEntry(
                id="release-alternative",
                text="Atlas release date may be Monday",
                namespace="project:cayu",
            )
        )
        await knowledge.publish_relations(
            [
                KnowledgeRelation(
                    id="release-supersession",
                    subject=KnowledgeRevisionRef(
                        entry_id=replacement.id,
                        revision=replacement.revision,
                    ),
                    object=KnowledgeRevisionRef(
                        entry_id=predecessor.id,
                        revision=predecessor.revision,
                    ),
                    kind=KnowledgeRelationKind.SUPERSEDES,
                    metadata={"private_review": "not recall material"},
                ),
                KnowledgeRelation(
                    id="release-contradiction",
                    subject=KnowledgeRevisionRef(
                        entry_id=replacement.id,
                        revision=replacement.revision,
                    ),
                    object=KnowledgeRevisionRef(
                        entry_id=alternative.id,
                        revision=alternative.revision,
                    ),
                    kind=KnowledgeRelationKind.CONTRADICTS,
                ),
            ],
            operation_id="release-lineage",
        )
        await knowledge.append_entry_revision(
            predecessor.model_copy(
                update={
                    "revision": 2,
                    "status": KnowledgeStatus.ARCHIVED,
                    "updated_at": predecessor.updated_at + timedelta(microseconds=1),
                }
            ),
            expected_revision=1,
        )
        return await RecallEngine(
            (KnowledgeRecallSource(knowledge, lineage_limit=10),),
            fusion_config=_fusion_config(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
            ),
        ).recall(
            _situation(
                knowledge_access_scope=scope,
                knowledge_namespace="project:cayu",
            )
        )

    result = asyncio.run(run())

    by_entry = {candidate.record.locator["entry_id"]: candidate for candidate in result.candidates}
    assert set(by_entry) == {"release-current", "release-alternative"}
    current_lineage = by_entry["release-current"].record.lineage
    alternative_lineage = by_entry["release-alternative"].record.lineage
    assert current_lineage is not None
    assert alternative_lineage is not None
    assert {
        (link.role, link.counterpart.entry_id, link.unresolved_contradiction)
        for link in current_lineage.links
    } == {
        (KnowledgeLineageRole.SUPERSEDES, "release-obsolete", False),
        (KnowledgeLineageRole.CONTRADICTS, "release-alternative", True),
    }
    assert alternative_lineage.links[0].role is KnowledgeLineageRole.CONTRADICTS
    assert alternative_lineage.links[0].counterpart.entry_id == "release-current"
    assert alternative_lineage.links[0].unresolved_contradiction is True
    assert "private_review" not in result.model_dump_json()


def test_knowledge_recall_lineage_has_zero_record_overhead() -> None:
    class CountingLineageStore(InMemoryKnowledgeStore):
        lineage_calls = 0

        async def inspect_lineage(self, query, *, access_scope=None):
            self.lineage_calls += 1
            return await super().inspect_lineage(query, access_scope=access_scope)

    async def run() -> int:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = CountingLineageStore(access_scope=scope)
        result = await KnowledgeRecallSource(
            knowledge,
            lineage_limit=5,
        ).retrieve(
            _situation(
                knowledge_access_scope=scope,
                knowledge_namespace="project:cayu",
            )
        )
        assert result.records == ()
        return knowledge.lineage_calls

    assert asyncio.run(run()) == 0
    assert "lineage" not in _record("knowledge_entry", "plain", "plain").model_dump(mode="json")


def test_knowledge_recall_lineage_candidate_limit_counts_chunk_records() -> None:
    class CountingLineageStore(InMemoryKnowledgeStore):
        lineage_calls = 0

        async def inspect_lineage(self, query, *, access_scope=None):
            self.lineage_calls += 1
            return await super().inspect_lineage(query, access_scope=access_scope)

    async def run() -> tuple[dict[tuple[str, str, str], RecallRecord], int]:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = CountingLineageStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="chunked-entry",
                text="Atlas canonical guidance",
                namespace="project:cayu",
            )
        )
        await knowledge.create_entry(
            KnowledgeEntry(
                id="source-entry",
                text="Atlas source guidance",
                namespace="project:cayu",
            )
        )
        await knowledge.publish_relations(
            [
                KnowledgeRelation(
                    id="chunked-lineage",
                    subject=KnowledgeRevisionRef(entry_id="chunked-entry", revision=1),
                    object=KnowledgeRevisionRef(entry_id="source-entry", revision=1),
                    kind=KnowledgeRelationKind.DERIVED_FROM,
                )
            ],
            operation_id="chunked-lineage",
        )
        records: dict[tuple[str, str, str], RecallRecord] = {}
        hits = []
        for rank, chunk_id in enumerate(("chunk-a", "chunk-b"), start=1):
            text = f"Atlas {chunk_id}"
            record = RecallRecord(
                identity=RetrievalCandidateIdentity(
                    record_type="knowledge_chunk",
                    record_id=chunk_id,
                    revision="1",
                ),
                representation="chunk_text",
                text=text,
                text_complete=True,
                content_hash=sha256(text.encode()).hexdigest(),
                locator={
                    "entry_id": "chunked-entry",
                    "entry_revision": 1,
                    "chunk_id": chunk_id,
                    "chunk_index": rank - 1,
                },
            )
            records[record.identity.sort_key()] = record
            hits.append(
                RankedRetrievalHit(
                    identity=record.identity,
                    rank=rank,
                    representation=record.representation,
                    content_hash=record.content_hash,
                )
            )
        source = KnowledgeRecallSource(
            knowledge,
            lineage_limit=5,
            lineage_candidate_limit=1,
        )
        await source._attach_lineage(
            records,
            channels=(
                RankedRetrievalChannel(
                    channel=KNOWLEDGE_LEXICAL_CHANNEL,
                    index_version="test-v1",
                    candidate_limit=2,
                    hits=tuple(hits),
                ),
            ),
            access_scope=scope,
        )
        return records, knowledge.lineage_calls

    records, lineage_calls = asyncio.run(run())

    assert lineage_calls == 1
    assert sum(record.lineage is not None for record in records.values()) == 1
    enriched = records[("knowledge_chunk", "chunk-a", "1")]
    assert enriched.lineage is not None
    assert records[("knowledge_chunk", "chunk-b", "1")].lineage is None
    invalid = enriched.model_dump(mode="python")
    invalid["locator"]["chunk_id"] = "different-chunk"
    with pytest.raises(ValueError, match="Knowledge-chunk lineage"):
        RecallRecord.model_validate(invalid)


@pytest.mark.parametrize(
    "successor_status",
    (KnowledgeStatus.ACTIVE, KnowledgeStatus.ARCHIVED),
)
def test_knowledge_recall_rejects_a_candidate_that_advances_before_lineage(
    successor_status: KnowledgeStatus,
) -> None:
    class AdvancingLineageStore(InMemoryKnowledgeStore):
        candidate: KnowledgeEntry | None = None

        async def inspect_lineage(self, query, *, access_scope=None):
            if self.candidate is not None:
                candidate = self.candidate
                self.candidate = None
                await self.append_entry_revision(
                    candidate.model_copy(
                        update={
                            "revision": candidate.revision + 1,
                            "status": successor_status,
                            "updated_at": candidate.updated_at + timedelta(microseconds=1),
                        }
                    ),
                    expected_revision=candidate.revision,
                    access_scope=KnowledgeAccessScope.privileged(),
                )
            return await super().inspect_lineage(query, access_scope=access_scope)

    async def run() -> None:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = AdvancingLineageStore()
        knowledge.candidate = await knowledge.create_entry(
            KnowledgeEntry(
                id="advancing-candidate",
                text="Atlas answer before advancement",
                namespace="project:cayu",
            ),
            access_scope=KnowledgeAccessScope.privileged(),
        )
        with pytest.raises(RuntimeError, match="authority changed"):
            await KnowledgeRecallSource(knowledge, lineage_limit=5).retrieve(
                _situation(
                    knowledge_access_scope=scope,
                    knowledge_namespace="project:cayu",
                )
            )

    asyncio.run(run())


def test_knowledge_recall_rejects_a_store_that_expands_the_lineage_query() -> None:
    class ExpandingLineageStore(InMemoryKnowledgeStore):
        expanded_link_count = 0

        async def inspect_lineage(self, query, *, access_scope=None):
            expanded = query.model_copy(update={"limit": 100, "max_bytes": 64_000})
            result = await super().inspect_lineage(expanded, access_scope=access_scope)
            assert result is not None
            self.expanded_link_count = len(result.links)
            return result

    async def run() -> int:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = ExpandingLineageStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="bounded-anchor",
                text="Atlas bounded anchor",
                namespace="project:cayu",
            )
        )
        for entry_id in ("bounded-source-a", "bounded-source-b"):
            await knowledge.create_entry(
                KnowledgeEntry(
                    id=entry_id,
                    text=f"Source material for {entry_id}",
                    namespace="project:cayu",
                )
            )
        await knowledge.publish_relations(
            [
                KnowledgeRelation(
                    id=f"bounded-relation-{index}",
                    subject=KnowledgeRevisionRef(entry_id="bounded-anchor", revision=1),
                    object=KnowledgeRevisionRef(entry_id=entry_id, revision=1),
                    kind=KnowledgeRelationKind.DERIVED_FROM,
                )
                for index, entry_id in enumerate(
                    ("bounded-source-a", "bounded-source-b"),
                    start=1,
                )
            ],
            operation_id="bounded-lineage",
        )
        with pytest.raises(RuntimeError, match="altered the bounded recall query"):
            await KnowledgeRecallSource(knowledge, lineage_limit=1).retrieve(
                _situation(
                    knowledge_access_scope=scope,
                    knowledge_namespace="project:cayu",
                )
            )
        return knowledge.expanded_link_count

    assert asyncio.run(run()) == 2


@pytest.mark.parametrize(
    "configuration",
    (
        {"lineage_limit": 101},
        {"lineage_candidate_limit": 101},
        {"lineage_max_bytes": 64_001},
        {"lineage_candidate_limit": 100, "lineage_max_bytes": 20_000},
    ),
)
def test_knowledge_recall_rejects_unbounded_lineage_configuration(configuration) -> None:
    scope = KnowledgeAccessScope.for_namespace("project:cayu")
    with pytest.raises(ValueError):
        KnowledgeRecallSource(
            InMemoryKnowledgeStore(access_scope=scope),
            **configuration,
        )


@pytest.mark.parametrize(
    ("semantic_behavior", "expected_reason"),
    (("fail", "semantic_failed"), ("timeout", "semantic_timeout")),
)
def test_knowledge_recall_preserves_lexical_results_when_semantic_lane_degrades(
    semantic_behavior: str,
    expected_reason: str,
) -> None:
    class DegradedSemanticStore(InMemoryKnowledgeStore):
        def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
            return (
                KnowledgeSearchMode.AUTO,
                KnowledgeSearchMode.KEYWORD,
                KnowledgeSearchMode.SEMANTIC,
            )

        async def search(self, query, *, access_scope=None):
            if query.mode is KnowledgeSearchMode.SEMANTIC:
                if semantic_behavior == "timeout":
                    await asyncio.Event().wait()
                raise RuntimeError("private semantic failure")
            return await super().search(query, access_scope=access_scope)

    async def run() -> RecallResult:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = DegradedSemanticStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="release-date",
                text="Atlas release date is Friday",
                namespace="project:cayu",
            )
        )
        return await RecallEngine(
            (
                KnowledgeRecallSource(
                    knowledge,
                    semantic_timeout_seconds=0.01,
                ),
            ),
            fusion_config=_fusion_config(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
            ),
        ).recall(
            _situation(
                knowledge_access_scope=scope,
                knowledge_namespace="project:cayu",
            )
        )

    result = asyncio.run(run())

    assert [candidate.record.identity.record_id for candidate in result.candidates] == [
        "release-date"
    ]
    assert result.sources[0].status is RecallSourceStatus.PARTIAL
    assert result.sources[0].failure_code == expected_reason
    semantic = next(
        channel
        for channel in result.fusion.channels
        if channel.channel == KNOWLEDGE_SEMANTIC_CHANNEL
    )
    assert semantic.index_version == "unavailable"
    assert semantic.hit_count == 0
    assert result.truncated is True
