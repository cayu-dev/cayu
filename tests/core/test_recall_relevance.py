from __future__ import annotations

import asyncio
import importlib

import pytest

from cayu.cli import main
from cayu.cli.project import project_context
from cayu.memory import admit_recall
from cayu.recall import KnowledgeRecallSource, RecallEngine, RecallSituation
from cayu.recall_relevance import query_concept_eligibility
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeSearchMode,
)


@pytest.mark.parametrize(
    "query,text,eligible",
    [
        (
            "How should we configure deployment rollback safeguards?",
            "The deployment picnic uses table number 0.",
            False,
        ),
        (
            "How should we configure deployment rollback safeguards?",
            "Safely revert a release using the previous healthy image.",
            True,
        ),
        ("Atlas-42", "Atlas-42 migration uses an expand-contract sequence.", True),
        ("src/cayu/recall.py", "Edit src/cayu/recall.py to change recall.", True),
        ("login credentials", "Store authentication passwords in the vault.", True),
        ("request deadlines", "Bound every request with a timeout.", True),
        ("request deadlines", "The request picnic has a table.", False),
        ("what about that?", "deployment rollback safeguards", False),
        (None, "deployment rollback safeguards", False),
    ],
)
def test_concept_support_independent_of_rank(query, text, eligible):
    assert (query_concept_eligibility(query, text)[0] == "eligible") is eligible


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_title_aware_coding_recall_keeps_related_contracts_and_rejects_control(tmp_path, backend):
    from test_memory_admission import _policy

    from cayu import WeightedReciprocalRankFusionConfig

    async def run():
        scope = KnowledgeAccessScope.for_namespace("default")
        store = (
            InMemoryKnowledgeStore(access_scope=scope)
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "coding.sqlite", access_scope=scope)
        )
        entries = [
            KnowledgeEntry(
                id="cache",
                title="Harbor cache contract",
                text="Caller-owned batch caches must use tenant-scoped keys and copy returned records.",
            ),
            KnowledgeEntry(
                id="retry",
                title="Harbor retry contract",
                text="retry_batch persists the receipt before acknowledgement; a failed send is never settled.",
            ),
            KnowledgeEntry(
                id="window",
                title="Harbor reporting bucket contract",
                text="Reporting buckets are half-open; counts compare aware timestamps in UTC.",
            ),
        ]
        try:
            for entry in entries:
                await store.create_entry(entry)
            policy = _policy(
                relevance_policy="cayu.query_concepts.v2",
                minimum_inject_score=0.01,
                minimum_offer_score=0.005,
            )
            engine = RecallEngine(
                (KnowledgeRecallSource(store),),
                fusion_config=WeightedReciprocalRankFusionConfig(
                    configuration_version=policy.fusion_configuration_version,
                    channel_weights={"knowledge.lexical": 1.0, "knowledge.semantic": 1.0},
                ),
            )
            for query, expected in [
                ("Add cached batch lookup to Harbor.", "cache"),
                ("Add retry batch delivery to Harbor.", "retry"),
                ("Add reporting bucket counts to Harbor.", "window"),
                ("Add item pagination to Harbor.", None),
                ("Harbor authentication credential rotation", None),
            ]:
                result = await engine.recall(
                    RecallSituation(query=query, knowledge_access_scope=scope)
                )
                contribution = admit_recall(result, policy)
                focused = (
                    []
                    if contribution.focus is None
                    else [
                        item.candidate.record.locator["entry_id"]
                        for item in contribution.focus.items
                    ]
                )
                assert focused == ([] if expected is None else [expected])
                for candidate in result.candidates:
                    assert candidate.record.title is not None
                    assert candidate.record.content_hash
                if expected == "cache":
                    legacy = admit_recall(
                        result, _policy(relevance_policy="cayu.query_concepts.v1")
                    )
                    assert legacy.focus is None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "query,text,expected",
    [
        ("cached caches caching", "cache", True),
        ("retry retries retrying", "retry", True),
        ("cache-42", "caches-42", False),
        ("src/cache.py", "src/caches.py", False),
        ("deployment rollback safeguards", "deployment picnic", False),
    ],
)
def test_v2_inflections_do_not_inflate_concepts_or_fuzz_identifiers(query, text, expected):
    assert (
        query_concept_eligibility(query, text, version="cayu.query_concepts.v2")[0] == "eligible"
    ) is expected


@pytest.mark.parametrize(
    "relevance_policy",
    ["rank_only.v1", "cayu.query_concepts.v1", "cayu.query_concepts.v2", "cayu.query_concepts.v3"],
)
@pytest.mark.parametrize("bound", ["below_body", "body", "below_combined", "combined"])
def test_title_byte_admission_and_candidate_diagnostics_agree(relevance_policy, bound):
    from test_memory_admission import _candidate, _policy, _result

    candidate = _candidate("cache", score=0.04, text="batch caches")
    candidate = candidate.model_copy(
        update={"record": candidate.record.model_copy(update={"title": "Caché contract"})}
    )
    body_bytes = len(candidate.record.text.encode("utf-8"))
    combined_bytes = body_bytes + len(candidate.record.title.encode("utf-8"))
    limit = {
        "below_body": body_bytes - 1,
        "body": body_bytes,
        "below_combined": combined_bytes - 1,
        "combined": combined_bytes,
    }[bound]
    result = _result(candidate).model_copy(update={"relevance_query": "batch caches"})
    contribution = admit_recall(
        result, _policy(relevance_policy=relevance_policy, max_candidate_text_bytes=limit)
    )
    oversized = bound == "below_body" or (
        relevance_policy in {"cayu.query_concepts.v2", "cayu.query_concepts.v3"}
        and bound != "combined"
    )
    assert (contribution.focus is None) is oversized
    assert contribution.diagnostics.oversized_candidate_omitted == int(oversized)
    assert contribution.diagnostics.candidate_decisions[0].outcome == (
        "oversized" if oversized else "focused"
    )


def test_v2_policy_identity_and_title_byte_bound():
    from test_memory_admission import _candidate, _policy, _result

    from cayu.recall_relevance import TITLE_RELEVANCE_TEXT_VERSION

    legacy = _policy(relevance_policy="cayu.query_concepts.v1")
    policy = _policy(relevance_policy="cayu.query_concepts.v2")
    assert policy.relevance_text_version == TITLE_RELEVANCE_TEXT_VERSION
    assert legacy.fingerprint() != policy.fingerprint()
    with pytest.raises(ValueError, match="Unicode version"):
        policy.model_copy(update={"relevance_text_version": legacy.relevance_text_version})
    candidate = _candidate("cache", score=0.04, text="batch caches")
    candidate = candidate.model_copy(
        update={"record": candidate.record.model_copy(update={"title": "Harbor cache"})}
    )
    result = _result(candidate).model_copy(update={"relevance_query": "Harbor cached batch"})
    assert admit_recall(result, policy).focus is not None
    bounded = policy.model_copy(
        update={"max_candidate_text_bytes": len(candidate.record.text.encode())}
    )
    assert admit_recall(result, bounded).focus is None
    with pytest.raises(ValueError, match="1024 bytes"):
        type(candidate.record).model_validate({**candidate.record.model_dump(), "title": "é" * 513})


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("population", [1, 20])
def test_generated_default_rejects_picnics_and_keeps_useful_procedure(
    tmp_path, backend, population
):
    assert main(["new", "relevance_app", "--dir", str(tmp_path)]) == 0
    with project_context(tmp_path / "relevance_app"):
        policy = importlib.import_module("memory.context").build_context_policy()
    assert policy.admission_policy.relevance_policy == "cayu.query_concepts.v3"

    async def run():
        scope = KnowledgeAccessScope.for_namespace("default")
        store = (
            InMemoryKnowledgeStore(access_scope=scope)
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "probe.sqlite", access_scope=scope)
        )
        try:
            for index in range(population):
                await store.create_entry(
                    KnowledgeEntry(
                        id=f"picnic-{index}",
                        text=f"The deployment picnic uses table number {index}.",
                    )
                )
            engine = RecallEngine(
                (KnowledgeRecallSource(store),), fusion_config=policy.fusion_config
            )
            situation = RecallSituation(
                query="How should we configure deployment rollback safeguards?",
                knowledge_access_scope=scope,
            )
            weak_result = await engine.recall(situation)
            weak = admit_recall(weak_result, policy.admission_policy)
            assert weak.focus is None and weak.offer is None
            assert len(weak.diagnostics.candidate_decisions) == population
            assert {item.outcome for item in weak.diagnostics.candidate_decisions} == {
                "low_relevance"
            }
            from cayu.runtime._memory_evidence import MemoryEvidenceKey, build_recall_receipt

            receipt = build_recall_receipt(
                session_id="relevance",
                interaction_id="interaction",
                model_step_id="mstep_00000000000000000000000000000000",
                situation=situation,
                result=weak_result,
                contribution=weak,
                admission_policy=policy.admission_policy,
                source_configuration={},
                key=MemoryEvidenceKey(key_id="test", key=b"k" * 32),
            )
            assert receipt.silent_count == population
            assert receipt.omitted_count == 0
            assert receipt.candidate_decisions == weak.diagnostics.candidate_decisions
            assert type(receipt).model_validate_json(receipt.model_dump_json()) == receipt
            await store.create_entry(
                KnowledgeEntry(
                    id="rollback",
                    text="Deployment rollback safeguards: verify health, restore the previous image, and monitor errors.",
                )
            )
            result = await engine.recall(situation)
            useful = admit_recall(result, policy.admission_policy)
            assert [item.candidate.record.locator["entry_id"] for item in useful.focus.items] == [
                "rollback"
            ]
            assert (
                admit_recall(
                    type(result).model_validate_json(result.model_dump_json()),
                    policy.admission_policy,
                )
                == useful
            )
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("semantic", ["hybrid", "semantic_only", "timeout", "partial"])
def test_semantic_agreement_and_coverage_cannot_manufacture_relevance(semantic):
    from test_memory_admission import _policy

    from cayu import InMemoryEmbeddingKnowledgeStore
    from cayu.embeddings import TextEmbedding, TextEmbeddingProvider, TextEmbeddingResult
    from cayu.retrieval import WeightedReciprocalRankFusionConfig

    class TiedEmbeddingProvider(TextEmbeddingProvider):
        name = "tied-test"

        async def embed_texts(self, request):
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=index, vector=[1.0])
                    for index, _ in enumerate(request.texts)
                ],
            )

    class SemanticStore(InMemoryEmbeddingKnowledgeStore):
        async def search(self, query, *, access_scope=None):
            if query.mode is KnowledgeSearchMode.SEMANTIC and semantic == "timeout":
                await asyncio.Event().wait()
            return await super().search(query, access_scope=access_scope)

    async def run():
        scope = KnowledgeAccessScope.for_namespace("default")
        store = SemanticStore(
            access_scope=scope,
            embedding_provider=TiedEmbeddingProvider(),
            embedding_model="test-model",
            embedding_dimensions=1,
        )
        good = "Safely revert a release using the previous healthy image."
        if semantic in {"timeout", "partial"}:
            good = "deployment rollback safeguards"
        weak = "picnic table" if semantic == "semantic_only" else "deployment picnic"
        for identity, text in [("weak", weak), ("good", good)]:
            await store.create_entry(KnowledgeEntry(id=identity, text=text))
        await store.process_embedding_changes("relevance-test-index", "worker:test")
        if semantic == "partial":
            await store.create_entry(KnowledgeEntry(id="unindexed", text="unindexed picnic"))
        fusion = WeightedReciprocalRankFusionConfig(
            configuration_version="memory-admission-tests-v1",
            channel_weights={"knowledge.lexical": 1.0, "knowledge.semantic": 1.0},
        )
        result = await RecallEngine(
            (KnowledgeRecallSource(store, semantic_timeout_seconds=0.01),), fusion_config=fusion
        ).recall(
            RecallSituation(query="deployment rollback safeguards", knowledge_access_scope=scope)
        )
        policy = _policy(
            relevance_policy="cayu.query_concepts.v1",
            minimum_inject_score=0.01,
            minimum_offer_score=0.005,
        )
        contribution = admit_recall(result, policy)
        assert [item.candidate.record.locator["entry_id"] for item in contribution.focus.items] == [
            "good"
        ]
        assert (
            next(
                item
                for item in contribution.diagnostics.candidate_decisions
                if item.identity.record_id == "weak" or item.identity.record_id.startswith("weak:")
            ).outcome
            == "low_relevance"
        )
        expected_failure = {"timeout": "semantic_timeout", "partial": "semantic_index_partial"}.get(
            semantic
        )
        assert result.sources[0].failure_code == expected_failure
        if semantic == "semantic_only":
            assert (
                next(
                    channel
                    for channel in result.fusion.channels
                    if channel.channel == "knowledge.lexical"
                ).hit_count
                == 0
            )

    asyncio.run(run())


@pytest.mark.parametrize("population,score", [(1, 0.04), (20, 0.04), (20, 100.0)])
def test_rank_population_and_ties_do_not_create_eligibility(population, score):
    from test_memory_admission import _candidate, _policy, _result

    result = _result(
        *(
            _candidate(f"weak-{index}", score=score, text=f"deployment picnic {index}")
            for index in range(population)
        )
    )
    result = result.model_copy(update={"relevance_query": "deployment rollback safeguards"})
    contribution = admit_recall(result, _policy(relevance_policy="cayu.query_concepts.v1"))
    assert contribution.focus is None and contribution.offer is None
    assert contribution.diagnostics.strong_candidate_count == 0


def test_relevance_capacity_and_missing_evidence_are_distinct():
    from test_memory_admission import _candidate, _policy, _result

    result = _result(
        *(
            _candidate(f"good-{index}", score=0.04, text=f"rollback safeguards {index}")
            for index in range(8)
        )
    )
    policy = _policy(
        relevance_policy="cayu.query_concepts.v1", max_injected_items=1, max_offered_items=1
    )
    missing = admit_recall(result, policy)
    assert {item.outcome for item in missing.diagnostics.candidate_decisions} == {
        "insufficient_evidence"
    }
    selected = admit_recall(
        result.model_copy(update={"relevance_query": "rollback safeguards"}), policy
    )
    assert [item.outcome for item in selected.diagnostics.candidate_decisions] == [
        "focused",
        "offered",
    ] + ["capacity"] * 6


def test_relevance_configuration_binds_unicode_and_preserves_rank_only_identity():
    from test_memory_admission import _policy

    from cayu.recall_relevance import RELEVANCE_TEXT_VERSION

    legacy = _policy()
    assert "relevance_policy" not in legacy.model_dump(mode="json")
    assert "relevance_text_version" not in legacy.model_dump(mode="json")
    fixed = _policy(relevance_policy="cayu.query_concepts.v1")
    assert fixed.relevance_text_version == RELEVANCE_TEXT_VERSION
    assert fixed.fingerprint() != legacy.fingerprint()
    with pytest.raises(ValueError, match="Unicode version"):
        fixed.model_copy(update={"relevance_text_version": "cayu.query_concepts.v1+unicode-other"})


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Weather tomorrow?", "weather"),
        ("And in production?", "rollback"),
        ("what about that?", "rollback"),
        ("Why?", "rollback"),
    ],
)
def test_generated_context_combines_resolution_and_strong_admission(tmp_path, query, expected):
    import json

    from test_automatic_recall_context import _fixture, _request

    from cayu.core.messages import Message
    from cayu.runtime._memory_evidence import MemoryEvidenceKey, memory_evidence_key_scope

    assert main(["new", "combined_app", "--dir", str(tmp_path)]) == 0
    with project_context(tmp_path / "combined_app"):
        policy = importlib.import_module("memory.context").build_context_policy()

    async def run():
        sessions, _, session, previous = await _fixture()
        store = InMemoryKnowledgeStore(
            access_scope=KnowledgeAccessScope.for_namespace(policy.sources.knowledge_namespace)
        )
        for identity, text in [
            ("rollback", "Deployment rollback safeguards use a previous healthy image."),
            ("picnic", "deployment picnic table"),
            ("weather", "Weather tomorrow sunny."),
        ]:
            await store.create_entry(
                KnowledgeEntry(id=identity, text=text, namespace=policy.sources.knowledge_namespace)
            )
        messages = [
            *previous,
            Message.text("user", "Deployment rollback safeguards"),
            Message.text("assistant", "picnic table " * 3000),
            Message.text("user", query),
        ]
        await sessions.append_transcript_messages(
            session.id, messages[len(previous) :], interaction_id="combined"
        )
        with memory_evidence_key_scope(MemoryEvidenceKey(key_id="combined-test", key=b"k" * 32)):
            result = await policy.build_with_checkpoint(
                _request(sessions=sessions, knowledge=store, session=session, messages=messages),
                checkpoint=None,
            )
        projection = result.checkpoint["automatic_recall"]["projection"]
        assert projection is not None
        assert {
            json.loads(item["locator_json"])["entry_id"] for item in projection["focus"]["items"]
        } == {expected}

    asyncio.run(run())
