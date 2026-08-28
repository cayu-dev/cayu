from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    RecallSituation,
)
from cayu.recall_processing import (
    AgentRecallProcessingError,
    AgentRecallProcessingMode,
    AgentRecallProcessingRequest,
    AgentRecallProcessingResult,
    AgentRecallProcessor,
    AgentRecallProcessorConfig,
)
from cayu.retrieval import WeightedReciprocalRankFusionConfig
from cayu.storage import (
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    KnowledgeQuery,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeSearchMode,
    KnowledgeStatus,
    SQLiteKnowledgeStore,
    knowledge_chunk_embedding_identity,
)
from cayu.work_context import AgentWorkContext

_NOW = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
_NAMESPACE = "project:cayu"
_QUERY = "checkpoint aware delta target phrase memory"


def _scope() -> KnowledgeAccessScope:
    return KnowledgeAccessScope.for_namespace(_NAMESPACE)


def _fusion_config(*, candidate_limit: int = 20) -> WeightedReciprocalRankFusionConfig:
    return WeightedReciprocalRankFusionConfig(
        configuration_version="checkpoint-recall-tests-v1",
        channel_weights={
            KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
            KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
        },
        max_candidates_per_channel=candidate_limit,
        fused_head_limit=candidate_limit,
    )


def _context(
    *, revision: int = 1, goal: str = "Implement checkpoint-aware memory"
) -> AgentWorkContext:
    return AgentWorkContext.create(
        task_id="task-memory",
        goal=goal,
        revision=revision,
        operation_id=f"work-context-{revision}",
        published_by="test-suite",
        published_at=_NOW,
        repository_paths=("src/cayu/recall_processing.py",),
    )


def _request(
    *,
    context: AgentWorkContext,
    checkpoint=None,
    frontier=None,
    operation: str,
) -> AgentRecallProcessingRequest:
    return AgentRecallProcessingRequest(
        agent_id="agent-reviewer",
        work_context=context,
        situation=RecallSituation(
            query=_QUERY,
            knowledge_access_scope=_scope(),
            knowledge_namespace=_NAMESPACE,
            current_time=_NOW,
        ),
        checkpoint=checkpoint,
        frontier=frontier,
        processing_id=f"processing-{operation}",
        operation_id=operation,
        updated_by="test-suite",
        updated_at=_NOW,
    )


async def _create_entry(store, entry_id: str, text: str) -> None:
    await store.create_entry(
        KnowledgeEntry(
            id=entry_id,
            namespace=_NAMESPACE,
            text=text,
        ),
        [
            KnowledgeChunk(
                id=f"{entry_id}-chunk",
                entry_id=entry_id,
                text=text,
                chunk_index=0,
            )
        ],
        access_scope=_scope(),
    )


class _RecallEmbeddingProvider(TextEmbeddingProvider):
    name = "recall-frontier-test"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.calls.append(list(request.texts))
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=[1.0, 0.0, 0.0])
                for index, _ in enumerate(request.texts)
            ],
        )


class _ToggleSemanticFailureStore(InMemoryKnowledgeStore):
    semantic_enabled = False

    def supported_search_modes(self):
        modes = (
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.KEYWORD,
        )
        return modes + ((KnowledgeSearchMode.SEMANTIC,) if self.semantic_enabled else ())

    async def search(self, query, *, access_scope=None):
        if query.mode is KnowledgeSearchMode.SEMANTIC:
            raise RuntimeError("simulated semantic outage")
        return await super().search(query, access_scope=access_scope)

    async def search_revisions(
        self,
        query,
        revision_refs,
        *,
        knowledge_sequence=None,
        index_readiness_sequence=None,
        access_scope=None,
    ):
        if query.mode is KnowledgeSearchMode.SEMANTIC:
            raise RuntimeError("simulated semantic outage")
        return await super().search_revisions(
            query,
            revision_refs,
            knowledge_sequence=knowledge_sequence,
            index_readiness_sequence=index_readiness_sequence,
            access_scope=access_scope,
        )


def test_processor_selects_full_no_work_delta_and_context_refresh() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        old_text = " ".join([_QUERY] * 8)
        await _create_entry(store, "old-global-winner", old_text)
        processor = AgentRecallProcessor(
            store,
            fusion_config=_fusion_config(candidate_limit=1),
            config=AgentRecallProcessorConfig(candidate_limit=1),
        )

        initial_request = _request(context=_context(), operation="initial-full")
        assert (
            AgentRecallProcessingRequest.model_validate_json(initial_request.model_dump_json())
            == initial_request
        )
        initial = await processor.process(initial_request)
        assert initial.mode is AgentRecallProcessingMode.FULL_INDEX
        assert initial.proposed_checkpoint is not None
        assert [
            candidate.record.locator["entry_id"] for candidate in initial.recall.candidates
        ] == ["old-global-winner"]

        unchanged = await processor.process(
            _request(
                context=_context(),
                checkpoint=initial.proposed_checkpoint,
                operation="unchanged",
            )
        )
        assert unchanged.mode is AgentRecallProcessingMode.NO_WORK
        assert unchanged.recall is None
        assert unchanged.proposed_checkpoint is None
        assert unchanged.processed_frontier == unchanged.frontier
        assert unchanged.work_context_sha256 == _context().content_sha256

        await _create_entry(store, "new-delta-target", _QUERY)
        global_result = await store.search(
            KnowledgeQuery(
                text=_QUERY,
                namespace=_NAMESPACE,
                mode=KnowledgeSearchMode.KEYWORD,
                limit=1,
            ),
            access_scope=_scope(),
        )
        assert [hit.entry.id for hit in global_result.hits] == ["old-global-winner"]

        delta_request = _request(
            context=_context(),
            checkpoint=initial.proposed_checkpoint,
            operation="delta",
        )
        assert (
            AgentRecallProcessingRequest.model_validate_json(delta_request.model_dump_json())
            == delta_request
        )
        delta = await processor.process(delta_request)
        assert delta.mode is AgentRecallProcessingMode.DELTA
        assert [(item.entry_id, item.revision) for item in delta.eligible_revisions] == [
            ("new-delta-target", 1)
        ]
        assert [event.entry_id for event in delta.knowledge_events] == ["new-delta-target"]
        assert delta.index_readiness_events == ()
        assert [candidate.record.locator["entry_id"] for candidate in delta.recall.candidates] == [
            "new-delta-target"
        ]
        assert delta.proposed_checkpoint is not None
        assert delta.proposed_checkpoint.knowledge_sequence == delta.frontier.knowledge_sequence

        refreshed = await processor.process(
            _request(
                context=_context(revision=2, goal="Review checkpoint-aware memory"),
                checkpoint=delta.proposed_checkpoint,
                operation="context-refresh",
            )
        )
        assert refreshed.mode is AgentRecallProcessingMode.FULL_INDEX
        assert refreshed.reason == "work_context_changed"
        assert refreshed.proposed_checkpoint is not None
        assert refreshed.proposed_checkpoint.work_context_revision == 2

    asyncio.run(run())


def test_full_index_does_not_cross_its_captured_frontier() -> None:
    class BlockingFrontierStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=_scope())
            self.block = False
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def search_at_frontier(
            self,
            query,
            *,
            knowledge_sequence,
            index_readiness_sequence,
            access_scope=None,
        ):
            if self.block and query.mode is KnowledgeSearchMode.KEYWORD:
                self.started.set()
                await self.release.wait()
            return await super().search_at_frontier(
                query,
                knowledge_sequence=knowledge_sequence,
                index_readiness_sequence=index_readiness_sequence,
                access_scope=access_scope,
            )

    async def run() -> None:
        store = BlockingFrontierStore()
        await _create_entry(store, "before-full-frontier", _QUERY)
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        store.block = True
        task = asyncio.create_task(
            processor.process(_request(context=_context(), operation="bounded-full"))
        )
        await asyncio.wait_for(store.started.wait(), timeout=2)
        await _create_entry(store, "after-full-frontier", _QUERY)
        store.release.set()

        initial = await task
        assert initial.mode is AgentRecallProcessingMode.FULL_INDEX
        assert initial.frontier.knowledge_sequence == 1
        assert [
            candidate.record.locator["entry_id"] for candidate in initial.recall.candidates
        ] == ["before-full-frontier"]
        assert initial.proposed_checkpoint is not None

        delta = await processor.process(
            _request(
                context=_context(),
                checkpoint=initial.proposed_checkpoint,
                operation="bounded-full-next-delta",
            )
        )
        assert delta.mode is AgentRecallProcessingMode.DELTA
        assert [reference.entry_id for reference in delta.eligible_revisions] == [
            "after-full-frontier"
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("failing_read", "expected_code"),
    [
        ("changes", "knowledge_changes_failed"),
        ("readiness", "index_readiness_failed"),
    ],
)
def test_processor_classifies_operational_freshness_read_failures(
    failing_read: str,
    expected_code: str,
) -> None:
    class FailingFreshnessStore(InMemoryKnowledgeStore):
        async def read_changes(self, **kwargs):
            if failing_read == "changes":
                raise RuntimeError("knowledge change storage is offline")
            return await super().read_changes(**kwargs)

        async def read_index_readiness(self, **kwargs):
            if failing_read == "readiness":
                raise RuntimeError("index readiness storage is offline")
            return await super().read_index_readiness(**kwargs)

    async def run() -> None:
        store = FailingFreshnessStore(access_scope=_scope())
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())

        with pytest.raises(AgentRecallProcessingError) as raised:
            await processor.process(
                _request(context=_context(), operation=f"failed-{failing_read}")
            )

        assert raised.value.code == expected_code
        assert isinstance(raised.value.__cause__, RuntimeError)

    asyncio.run(run())


def test_delta_proposes_only_the_bounded_processed_prefix() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _create_entry(store, "initial", _QUERY)
        processor = AgentRecallProcessor(
            store,
            fusion_config=_fusion_config(),
            config=AgentRecallProcessorConfig(
                knowledge_change_limit=1,
                index_readiness_limit=1,
            ),
        )
        initial = await processor.process(_request(context=_context(), operation="partial-initial"))
        assert initial.proposed_checkpoint is not None
        await _create_entry(store, "delta-a", _QUERY)
        await _create_entry(store, "delta-b", _QUERY)

        first_page = await processor.process(
            _request(
                context=_context(),
                checkpoint=initial.proposed_checkpoint,
                operation="partial-delta-a",
            )
        )
        assert not first_page.frontier_complete
        assert first_page.proposed_checkpoint is not None
        assert first_page.proposed_checkpoint.knowledge_sequence < (
            first_page.proposed_checkpoint.knowledge_high_water_sequence
        )
        assert [item.entry_id for item in first_page.eligible_revisions] == ["delta-a"]

        second_page = await processor.process(
            _request(
                context=_context(),
                checkpoint=first_page.proposed_checkpoint,
                operation="partial-delta-b",
            )
        )
        assert second_page.frontier_complete
        assert [item.entry_id for item in second_page.eligible_revisions] == ["delta-b"]

    asyncio.run(run())


def test_request_rejects_a_scope_that_would_couple_namespace_frontiers() -> None:
    with pytest.raises(ValueError, match="exact single-namespace"):
        AgentRecallProcessingRequest(
            agent_id="agent-reviewer",
            work_context=_context(),
            situation=RecallSituation(
                query=_QUERY,
                knowledge_access_scope=KnowledgeAccessScope(
                    allowed_namespaces=[_NAMESPACE, "project:other"],
                ),
                knowledge_namespace=_NAMESPACE,
                current_time=_NOW,
            ),
            processing_id="processing-broad-scope",
            operation_id="broad-scope",
            updated_by="test-suite",
            updated_at=_NOW,
        )


def test_captured_delta_frontier_replays_identically_across_later_changes() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _create_entry(store, "initial-replay", _QUERY)
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        initial = await processor.process(_request(context=_context(), operation="replay-initial"))
        assert initial.proposed_checkpoint is not None
        await _create_entry(store, "replay-delta-a", _QUERY)
        replay_request = _request(
            context=_context(),
            checkpoint=initial.proposed_checkpoint,
            operation="replay-delta",
        )
        first = await processor.process(replay_request)
        await _create_entry(store, "replay-delta-b", _QUERY)
        replay = await processor.process(
            replay_request.model_copy(update={"frontier": first.frontier})
        )

        assert replay == first
        assert replay.fingerprint() == first.fingerprint()
        assert AgentRecallProcessingResult.model_validate_json(first.model_dump_json()) == first
        assert [item.entry_id for item in replay.eligible_revisions] == ["replay-delta-a"]

    asyncio.run(run())


def test_captured_delta_frontier_excludes_later_relation_lineage() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _create_entry(store, "lineage-subject", "initial subject")
        await _create_entry(store, "lineage-object", "related object")
        processor = AgentRecallProcessor(
            store,
            fusion_config=_fusion_config(),
            config=AgentRecallProcessorConfig(lineage_limit=10),
        )
        initial = await processor.process(
            _request(context=_context(), operation="lineage-frontier-initial")
        )
        assert initial.proposed_checkpoint is not None
        subject = await store.get_entry("lineage-subject", access_scope=_scope())
        assert subject is not None
        revised = await store.append_entry_revision(
            subject.model_copy(
                update={
                    "revision": 2,
                    "text": _QUERY,
                    "updated_at": subject.updated_at + timedelta(microseconds=1),
                }
            ),
            [
                KnowledgeChunk(
                    id="lineage-subject-r2-chunk",
                    entry_id=subject.id,
                    entry_revision=2,
                    text=_QUERY,
                    chunk_index=0,
                )
            ],
            expected_revision=1,
            access_scope=_scope(),
        )
        request = _request(
            context=_context(),
            checkpoint=initial.proposed_checkpoint,
            operation="lineage-frontier-delta",
        )
        first = await processor.process(request)
        assert first.recall is not None
        assert first.recall.candidates[0].record.lineage is not None
        assert first.recall.candidates[0].record.lineage.links == []

        await store.publish_relations(
            [
                KnowledgeRelation(
                    id="lineage-after-frontier",
                    subject=KnowledgeRevisionRef(
                        entry_id=revised.id,
                        revision=revised.revision,
                    ),
                    object=KnowledgeRevisionRef(entry_id="lineage-object", revision=1),
                    kind=KnowledgeRelationKind.DERIVED_FROM,
                )
            ],
            operation_id="lineage-after-frontier",
            access_scope=_scope(),
        )
        replay = await processor.process(request.model_copy(update={"frontier": first.frontier}))

        assert replay == first
        assert replay.fingerprint() == first.fingerprint()

    asyncio.run(run())


def test_captured_delta_frontier_excludes_later_semantic_readiness() -> None:
    async def run() -> None:
        provider = _RecallEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=provider,
            embedding_model="recall-frontier-test",
            embedding_dimensions=3,
            access_scope=_scope(),
        )
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        initial = await processor.process(
            _request(context=_context(), operation="readiness-frontier-initial")
        )
        assert initial.proposed_checkpoint is not None
        await _create_entry(store, "readiness-frontier-delta", _QUERY)
        request = _request(
            context=_context(),
            checkpoint=initial.proposed_checkpoint,
            operation="readiness-frontier-delta",
        )
        first = await processor.process(request)
        assert first.frontier.index_readiness_sequence == 0
        assert provider.calls == []

        worker = await store.process_embedding_changes(
            "readiness-frontier-consumer",
            "readiness-frontier-worker",
            access_scope=_scope(),
        )
        assert worker.acknowledged_changes == 1
        calls_after_indexing = len(provider.calls)
        replay = await processor.process(request.model_copy(update={"frontier": first.frontier}))

        assert replay == first
        assert replay.fingerprint() == first.fingerprint()
        assert len(provider.calls) == calls_after_indexing

    asyncio.run(run())


def test_captured_delta_frontier_rechecks_exact_revision_currentness() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _create_entry(store, "currentness-initial", _QUERY)
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        initial = await processor.process(
            _request(context=_context(), operation="currentness-initial")
        )
        assert initial.proposed_checkpoint is not None
        await _create_entry(store, "currentness-delta", _QUERY)
        request = _request(
            context=_context(),
            checkpoint=initial.proposed_checkpoint,
            operation="currentness-delta",
        )
        first = await processor.process(request)
        current = await store.get_entry("currentness-delta", access_scope=_scope())
        assert current is not None
        await store.append_entry_revision(
            current.model_copy(update={"revision": 2, "text": f"revised {_QUERY}"}),
            [
                KnowledgeChunk(
                    id="currentness-delta-revision-2-chunk",
                    entry_id="currentness-delta",
                    entry_revision=2,
                    text=f"revised {_QUERY}",
                    chunk_index=0,
                )
            ],
            expected_revision=1,
            access_scope=_scope(),
        )
        replay = await processor.process(request.model_copy(update={"frontier": first.frontier}))

        assert [item.revision for item in replay.eligible_revisions] == [1]
        assert replay.recall is not None
        assert replay.recall.candidates == ()
        assert replay.fingerprint() != first.fingerprint()
        assert replay.frontier == first.frontier

    asyncio.run(run())


def test_semantic_failure_keeps_readiness_frontier_retryable() -> None:
    async def run() -> None:
        store = _ToggleSemanticFailureStore(access_scope=_scope())
        await _create_entry(store, "semantic-retry", _QUERY)
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        initial = await processor.process(
            _request(context=_context(), operation="semantic-retry-initial")
        )
        assert initial.proposed_checkpoint is not None
        store.semantic_enabled = True
        full_retry = await processor.process(
            _request(
                context=_context(revision=2, goal="Changed context must retry full recall"),
                checkpoint=initial.proposed_checkpoint,
                operation="semantic-retry-full",
            )
        )
        assert full_retry.mode is AgentRecallProcessingMode.FULL_INDEX
        assert full_retry.frontier_complete
        assert full_retry.retry_required
        assert full_retry.proposed_checkpoint is None
        chunk = (await store.read_chunks("semantic-retry", access_scope=_scope()))[0]
        identity = knowledge_chunk_embedding_identity(
            chunk,
            embedding_model="semantic-retry-model",
            dimensions=3,
        )
        pending = await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.PENDING,
                attempt_id="semantic-retry-attempt",
            ),
            expected_sequence=None,
            operation_id="semantic-retry-pending",
            access_scope=_scope(),
        )
        await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.READY,
                attempt_id="semantic-retry-attempt",
            ),
            expected_sequence=pending.sequence,
            operation_id="semantic-retry-ready",
            access_scope=_scope(),
        )

        result = await processor.process(
            _request(
                context=_context(),
                checkpoint=initial.proposed_checkpoint,
                operation="semantic-retry-delta",
            )
        )

        assert result.mode is AgentRecallProcessingMode.DELTA
        assert result.retry_required
        assert not result.frontier_complete
        assert result.proposed_checkpoint is None
        assert result.frontier.index_readiness_sequence == 2
        assert result.processed_frontier.index_readiness_sequence == 0
        assert [event.state for event in result.index_readiness_events] == [
            KnowledgeIndexState.PENDING,
            KnowledgeIndexState.READY,
        ]
        assert result.task_id == _context().task_id
        assert [item.entry_id for item in result.eligible_revisions] == ["semantic-retry"]
        assert AgentRecallProcessingResult.model_validate_json(result.model_dump_json()) == result
        with pytest.raises(ValueError, match="unique ascending"):
            result.model_copy(
                update={"index_readiness_events": tuple(reversed(result.index_readiness_events))}
            )

    asyncio.run(run())


def test_semantic_failure_without_readiness_progress_cannot_drop_the_retry() -> None:
    async def run() -> None:
        store = _ToggleSemanticFailureStore(access_scope=_scope())
        await _create_entry(store, "semantic-no-readiness-initial", _QUERY)
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        initial = await processor.process(
            _request(context=_context(), operation="semantic-no-readiness-initial")
        )
        assert initial.proposed_checkpoint is not None

        store.semantic_enabled = True
        await _create_entry(store, "semantic-no-readiness-delta", _QUERY)
        result = await processor.process(
            _request(
                context=_context(),
                checkpoint=initial.proposed_checkpoint,
                operation="semantic-no-readiness-delta",
            )
        )

        assert result.mode is AgentRecallProcessingMode.DELTA
        assert result.frontier_complete
        assert result.retry_required
        assert result.proposed_checkpoint is None
        assert [item.entry_id for item in result.eligible_revisions] == [
            "semantic-no-readiness-delta"
        ]

    asyncio.run(run())


def test_partial_semantic_failure_without_replayable_readiness_keeps_knowledge_retryable() -> None:
    async def run() -> None:
        store = _ToggleSemanticFailureStore(access_scope=_scope())
        await _create_entry(store, "semantic-partial-initial", _QUERY)
        processor = AgentRecallProcessor(
            store,
            fusion_config=_fusion_config(),
            config=AgentRecallProcessorConfig(
                knowledge_change_limit=1,
                index_readiness_limit=1,
            ),
        )
        initial = await processor.process(
            _request(context=_context(), operation="semantic-partial-initial")
        )
        assert initial.proposed_checkpoint is not None

        store.semantic_enabled = True
        await _create_entry(store, "semantic-partial-a", _QUERY)
        await _create_entry(store, "semantic-partial-b", _QUERY)
        request = _request(
            context=_context(),
            checkpoint=initial.proposed_checkpoint,
            operation="semantic-partial-first",
        )
        first = await processor.process(request)
        retry = await processor.process(
            request.model_copy(update={"operation_id": "semantic-partial-retry"})
        )

        assert first.retry_required
        assert not first.frontier_complete
        assert first.proposed_checkpoint is None
        assert [reference.entry_id for reference in first.eligible_revisions] == [
            "semantic-partial-a"
        ]
        assert retry.eligible_revisions == first.eligible_revisions
        assert retry.frontier == first.frontier

    asyncio.run(run())


def test_partial_semantic_failure_advances_only_when_ready_evidence_replays_every_revision() -> (
    None
):
    async def run() -> None:
        store = _ToggleSemanticFailureStore(access_scope=_scope())
        await _create_entry(store, "semantic-covered-initial", _QUERY)
        processor = AgentRecallProcessor(
            store,
            fusion_config=_fusion_config(),
            config=AgentRecallProcessorConfig(
                knowledge_change_limit=1,
                index_readiness_limit=2,
            ),
        )
        initial = await processor.process(
            _request(context=_context(), operation="semantic-covered-initial")
        )
        assert initial.proposed_checkpoint is not None

        await _create_entry(store, "semantic-covered-a", _QUERY)
        chunk = (await store.read_chunks("semantic-covered-a", access_scope=_scope()))[0]
        identity = knowledge_chunk_embedding_identity(
            chunk,
            embedding_model="semantic-covered-model",
            dimensions=3,
        )
        pending = await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.PENDING,
                attempt_id="semantic-covered-attempt",
            ),
            expected_sequence=None,
            operation_id="semantic-covered-pending",
            access_scope=_scope(),
        )
        await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.READY,
                attempt_id=pending.attempt_id,
            ),
            expected_sequence=pending.sequence,
            operation_id="semantic-covered-ready",
            access_scope=_scope(),
        )
        await _create_entry(store, "semantic-covered-b", _QUERY)
        store.semantic_enabled = True

        first = await processor.process(
            _request(
                context=_context(),
                checkpoint=initial.proposed_checkpoint,
                operation="semantic-covered-first",
            )
        )
        assert first.retry_required
        assert first.proposed_checkpoint is not None
        assert (
            first.proposed_checkpoint.knowledge_sequence
            == first.processed_frontier.knowledge_sequence
        )
        assert first.proposed_checkpoint.index_readiness_sequence == 0

        retry = await processor.process(
            _request(
                context=_context(),
                checkpoint=first.proposed_checkpoint,
                operation="semantic-covered-retry",
            )
        )
        assert {reference.entry_id for reference in retry.eligible_revisions} == {
            "semantic-covered-a",
            "semantic-covered-b",
        }

    asyncio.run(run())


def test_retired_entry_cannot_regress_a_proven_readiness_frontier() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _create_entry(store, "retired-readiness", _QUERY)
        chunk = (await store.read_chunks("retired-readiness", access_scope=_scope()))[0]
        identity = knowledge_chunk_embedding_identity(
            chunk,
            embedding_model="retired-readiness-model",
            dimensions=3,
        )
        pending = await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.PENDING,
                attempt_id="retired-readiness-attempt",
            ),
            expected_sequence=None,
            operation_id="retired-readiness-pending",
            access_scope=_scope(),
        )
        await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.READY,
                attempt_id=pending.attempt_id,
            ),
            expected_sequence=pending.sequence,
            operation_id="retired-readiness-ready",
            access_scope=_scope(),
        )
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        initial = await processor.process(
            _request(context=_context(), operation="retired-readiness-initial")
        )
        assert initial.proposed_checkpoint is not None
        assert initial.frontier.index_readiness_sequence == 2

        await store.transition_entry_status(
            "retired-readiness",
            expected_revision=1,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
            access_scope=_scope(),
        )
        result = await processor.process(
            _request(
                context=_context(),
                checkpoint=initial.proposed_checkpoint,
                operation="retired-readiness-delta",
            )
        )

        assert result.mode is AgentRecallProcessingMode.DELTA
        assert result.frontier.index_readiness_sequence == 2
        assert result.processed_frontier.index_readiness_sequence == 2
        assert result.proposed_checkpoint is not None
        assert result.recall is not None
        assert result.recall.candidates == ()

    asyncio.run(run())


def test_cancellation_leaves_the_prior_delta_frontier_retryable() -> None:
    class BlockingDeltaStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=_scope())
            self.block = False
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def search_revisions(
            self,
            query,
            revision_refs,
            *,
            knowledge_sequence=None,
            index_readiness_sequence=None,
            access_scope=None,
        ):
            if self.block and query.mode is KnowledgeSearchMode.KEYWORD:
                self.started.set()
                await self.release.wait()
            return await super().search_revisions(
                query,
                revision_refs,
                knowledge_sequence=knowledge_sequence,
                index_readiness_sequence=index_readiness_sequence,
                access_scope=access_scope,
            )

    async def run() -> None:
        store = BlockingDeltaStore()
        await _create_entry(store, "cancellation-initial", _QUERY)
        processor = AgentRecallProcessor(store, fusion_config=_fusion_config())
        initial = await processor.process(
            _request(context=_context(), operation="cancellation-initial")
        )
        assert initial.proposed_checkpoint is not None
        await _create_entry(store, "cancellation-delta", _QUERY)
        store.block = True
        request = _request(
            context=_context(),
            checkpoint=initial.proposed_checkpoint,
            operation="cancellation-delta",
        )
        task = asyncio.create_task(processor.process(request))
        await asyncio.wait_for(store.started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        store.block = False
        store.release.set()
        retry = await processor.process(request)
        assert retry.mode is AgentRecallProcessingMode.DELTA
        assert [item.entry_id for item in retry.eligible_revisions] == ["cancellation-delta"]

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_exact_revision_search_filters_before_ranking_and_requires_current_revision(
    backend: str,
    tmp_path,
) -> None:
    async def run() -> None:
        scope = _scope()
        store = (
            InMemoryKnowledgeStore(access_scope=scope)
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "exact-revisions.sqlite", access_scope=scope)
        )
        await _create_entry(store, "strong-unselected", " ".join([_QUERY] * 8))
        await _create_entry(store, "selected", _QUERY)
        query = KnowledgeQuery(
            text=_QUERY,
            namespace=_NAMESPACE,
            mode=KnowledgeSearchMode.KEYWORD,
            limit=1,
        )

        global_result = await store.search(query, access_scope=scope)
        exact_entry_reads: list[str] = []
        if isinstance(store, InMemoryKnowledgeStore):
            current_entry = store._current_entry

            def track_current_entry(entry_id: str):
                exact_entry_reads.append(entry_id)
                return current_entry(entry_id)

            store._current_entry = track_current_entry
        restricted = await store.search_revisions(
            query,
            (KnowledgeRevisionRef(entry_id="selected", revision=1),),
            access_scope=scope,
        )
        if isinstance(store, InMemoryKnowledgeStore):
            assert exact_entry_reads == ["selected"]
        current = await store.get_entry("selected", access_scope=scope)
        assert current is not None
        await store.append_entry_revision(
            current.model_copy(update={"revision": 2, "text": f"revised {_QUERY}"}),
            [
                KnowledgeChunk(
                    id="selected-revision-2-chunk",
                    entry_id="selected",
                    entry_revision=2,
                    text=f"revised {_QUERY}",
                    chunk_index=0,
                )
            ],
            expected_revision=1,
            access_scope=scope,
        )
        stale = await store.search_revisions(
            query,
            (KnowledgeRevisionRef(entry_id="selected", revision=1),),
            access_scope=scope,
        )

        assert [hit.entry.id for hit in global_result.hits] == ["strong-unselected"]
        assert [hit.entry.id for hit in restricted.hits] == ["selected"]
        assert stale.hits == []
        if isinstance(store, SQLiteKnowledgeStore):
            await store.close()

    asyncio.run(run())


def test_sqlite_exact_revision_search_is_authorized_and_avoids_global_fts_work(tmp_path) -> None:
    async def run() -> None:
        scope = _scope()
        store = SQLiteKnowledgeStore(tmp_path / "bounded-exact-revisions.sqlite")
        denied_entry = KnowledgeEntry(
            id="bounded-denied",
            namespace="project:other",
            text=_QUERY,
        )
        await store.create_entry(
            denied_entry,
            [
                KnowledgeChunk(
                    id="bounded-denied-chunk",
                    entry_id=denied_entry.id,
                    text=denied_entry.text,
                    chunk_index=0,
                )
            ],
            access_scope=KnowledgeAccessScope.privileged(),
        )
        for index in range(300):
            await _create_entry(store, f"bounded-{index:04d}", _QUERY)
        query = KnowledgeQuery(
            text=_QUERY,
            namespace=_NAMESPACE,
            mode=KnowledgeSearchMode.KEYWORD,
            limit=1,
        )

        steps = 0

        def count_steps() -> int:
            nonlocal steps
            steps += 1
            return 0

        loaded_chunk_refs: list[KnowledgeRevisionRef] = []
        load_chunks = store._load_chunks_for_revision_refs_unlocked

        def track_loaded_chunk_refs(revision_refs):
            loaded_chunk_refs.extend(revision_refs)
            return load_chunks(revision_refs)

        store._load_chunks_for_revision_refs_unlocked = track_loaded_chunk_refs
        try:
            store._connection.set_progress_handler(count_steps, 1)
            restricted = await store.search_revisions(
                query,
                (
                    KnowledgeRevisionRef(entry_id="bounded-0000", revision=1),
                    KnowledgeRevisionRef(entry_id="bounded-denied", revision=1),
                ),
                access_scope=scope,
            )
            restricted_steps = steps
            steps = 0
            global_result = await store.search(query, access_scope=scope)
            global_steps = steps
        finally:
            store._connection.set_progress_handler(None, 0)
            await store.close()

        assert [hit.entry.id for hit in restricted.hits] == ["bounded-0000"]
        assert loaded_chunk_refs == [KnowledgeRevisionRef(entry_id="bounded-0000", revision=1)]
        assert global_result.total_hits_known == 300
        assert restricted_steps * 3 < global_steps

    asyncio.run(run())


def test_request_rejects_caller_supplied_ephemeral_work_context() -> None:
    with pytest.raises(ValueError, match="must be omitted"):
        AgentRecallProcessingRequest(
            agent_id="agent-reviewer",
            work_context=_context(),
            situation=RecallSituation(
                query=_QUERY,
                work_context="ambiguous caller text",
                knowledge_access_scope=_scope(),
                knowledge_namespace=_NAMESPACE,
                current_time=_NOW,
            ),
            processing_id="processing-invalid",
            operation_id="invalid",
            updated_by="test-suite",
            updated_at=_NOW,
        )


def test_replay_frontier_requires_an_unchanged_checkpoint_context() -> None:
    from cayu.recall_processing import AgentRecallFrontier

    with pytest.raises(ValueError, match="unchanged checkpoint work context"):
        _request(
            context=_context(),
            frontier=AgentRecallFrontier(
                knowledge_sequence=0,
                index_readiness_sequence=0,
            ),
            operation="invalid-full-replay",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_limit", 101),
        ("source_max_bytes", 1_000_001),
        ("lineage_limit", 101),
        ("lineage_candidate_limit", 101),
        ("lineage_max_bytes", 64_001),
    ],
)
def test_processor_config_rejects_bounds_the_recall_source_cannot_enforce(
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValueError):
        AgentRecallProcessorConfig(**{field: value})


def test_processor_rejects_a_candidate_limit_above_the_fusion_ceiling() -> None:
    with pytest.raises(ValueError, match="fusion ceiling"):
        AgentRecallProcessor(
            InMemoryKnowledgeStore(access_scope=_scope()),
            fusion_config=_fusion_config(candidate_limit=1),
            config=AgentRecallProcessorConfig(candidate_limit=2),
        )
