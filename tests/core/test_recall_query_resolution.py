from __future__ import annotations

import asyncio

import pytest

from cayu.recall import KnowledgeRecallSource, RecallEngine, RecallSituation
from cayu.retrieval import WeightedReciprocalRankFusionConfig
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry


@pytest.mark.parametrize(
    "query",
    [
        "Weather tomorrow?",
        "New topic: taxes",
        "Atlas",
        "#1417",
        "src/cayu/recall.py",
        "How should we configure deployment rollback safeguards?",
        "Météo demain ?",
        "明天天气\uff1f",
        "¿Tiempo mañana?",
        "Погода завтра?",
    ],
)
def test_independent_queries_do_not_inherit_history(query):
    situation = RecallSituation(
        query=query,
        work_context="deployment picnic",
        recent_conversation=(
            "user: Explain deployment picnics",
            "assistant: deployment picnic " * 500,
        ),
    )
    assert situation.retrieval_text() == query
    assert situation.query_resolution()["decision"] == "independent_query"


@pytest.mark.parametrize(
    "query",
    [
        "And in production?",
        "what about that?",
        "Why?",
        "Y en producción?",
        "Et en production ?",
        "А в продакшене?",  # noqa: RUF001 - intentional Cyrillic discourse cue
        "那在生产环境呢\uff1f",
    ],
)
def test_followups_resolve_only_user_antecedent(query):
    situation = RecallSituation(
        query=query,
        recent_conversation=(
            "user: deployment rollback",
            "assistant: picnic " * 500,
        ),
    )
    assert situation.retrieval_text() == query + "\ndeployment rollback"
    assert situation.query_resolution()["decision"] == "resolved_followup"
    restored = RecallSituation.model_validate_json(situation.model_dump_json())
    assert restored.query_resolution() == situation.query_resolution()
    assert restored.fingerprint() == situation.fingerprint()


@pytest.mark.parametrize(
    "context", [(), ("assistant: deployment rollback",), ("user: " + "界" * 1000,)]
)
def test_missing_or_clipped_antecedent_is_explicit(context):
    situation = RecallSituation(query="And in production?", recent_conversation=context)
    assert situation.retrieval_text() == situation.query
    assert situation.query_resolution()["decision"] == "insufficient_context"
    assert situation.query_resolution()["context_clipped"] == bool(
        context and context[0].startswith("user:")
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_query_resolution_selects_current_topic_and_followup(tmp_path, backend):
    async def run():
        scope = KnowledgeAccessScope.for_namespace("default")
        store = (
            InMemoryKnowledgeStore(access_scope=scope)
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "query.sqlite", access_scope=scope)
        )
        try:
            for identity, text in [
                ("picnic", "deployment picnic table"),
                ("weather", "Weather tomorrow sunny"),
                ("rollback", "rollback production safeguards"),
            ]:
                await store.create_entry(KnowledgeEntry(id=identity, text=text))
            engine = RecallEngine(
                (KnowledgeRecallSource(store),),
                fusion_config=WeightedReciprocalRankFusionConfig(
                    configuration_version="query-test-v2",
                    channel_weights={"knowledge.lexical": 1.0, "knowledge.semantic": 1.0},
                ),
            )
            independent = await engine.recall(
                RecallSituation(
                    query="Weather tomorrow?",
                    recent_conversation=("user: deployment picnic", "assistant: picnic table"),
                    knowledge_access_scope=scope,
                )
            )
            assert {item.record.locator["entry_id"] for item in independent.candidates} == {
                "weather"
            }
            followup = await engine.recall(
                RecallSituation(
                    query="And in production?",
                    recent_conversation=("user: rollback safeguards", "assistant: picnic table"),
                    knowledge_access_scope=scope,
                )
            )
            assert {item.record.locator["entry_id"] for item in followup.candidates} == {"rollback"}
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


def test_generated_policy_binds_resolution_version(tmp_path):
    import importlib

    from cayu.cli import main
    from cayu.cli.project import project_context

    assert main(["new", "query_app", "--dir", str(tmp_path)]) == 0
    with project_context(tmp_path / "query_app"):
        policy = importlib.import_module("memory.context").build_context_policy()
    assert policy.configuration_material()["query_resolution_version"] == "cayu.query_resolution.v2"


def test_capture_reports_clipping_and_skips_verbose_assistant():
    from cayu.core.messages import Message
    from cayu.runtime.memory_context import _recent_conversation

    history = [
        Message.text("user", "rollback safeguards"),
        Message.text("assistant", "picnic " * 8000),
    ]
    recent, clipped = _recent_conversation(
        history, before_index=2, excluded_user_anchors=set(), max_items=8, max_bytes=100
    )
    assert recent == ("user: rollback safeguards",)
    assert not clipped
    recent, clipped = _recent_conversation(
        [Message.text("user", "prefix " * 100 + "rollback")],
        before_index=1,
        excluded_user_anchors=set(),
        max_items=8,
        max_bytes=100,
    )
    situation = RecallSituation(
        query="And in production?", recent_conversation=recent, recent_user_context_clipped=clipped
    )
    assert situation.retrieval_text() == situation.query
    assert situation.query_resolution()["context_clipped"]
    assert situation.query_resolution()["decision"] == "insufficient_context"
