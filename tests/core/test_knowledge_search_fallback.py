"""Runtime composition checks, not model-quality or semantic-ranking benchmarks."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from cayu import (
    AgentSpec,
    AutomaticRecallContextPolicy,
    AutomaticRecallPolicy,
    AutomaticRecallSourceConfig,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    Message,
    ModelProvider,
    ModelStreamEvent,
    ReadKnowledgeTool,
    RequestFootprintConfig,
    RunRequest,
    ScriptedModelProvider,
    SearchKnowledgeTool,
    WeightedReciprocalRankFusionConfig,
)
from cayu.core.messages import ToolResultPart
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.providers.base import ModelRequest
from cayu.recall_relevance import query_concept_eligibility


class _FixtureEmbeddings(TextEmbeddingProvider):
    """Deterministic wiring fixture, deliberately not a relevance model."""

    name = "fallback-fixture"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=[1.0, 0.0])
                for index, _ in enumerate(request.texts)
            ],
        )


def test_model_can_expand_the_search_revision_using_only_tool_text() -> None:
    class ReferenceReader(ModelProvider):
        name = "reference-reader"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    name="search_knowledge", arguments={"query": "release authority", "limit": 1}
                )
            elif len(self.requests) == 2:
                results = [
                    part
                    for message in request.messages
                    for part in message.content
                    if isinstance(part, ToolResultPart)
                ]
                assert len(results) == 1 and not results[0].is_error
                # Deliberately read the model-visible content, not ToolResult.structured.
                reference = re.search(r"entry_id='([^']+)' revision=(\d+)", results[0].content)
                assert reference is not None
                yield ModelStreamEvent.tool_call(
                    name="read_knowledge",
                    arguments={
                        "entry_id": reference.group(1),
                        "revision": int(reference.group(2)),
                    },
                )
            else:
                yield ModelStreamEvent.text_delta("Reference expanded.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    async def run() -> None:
        namespace = "project:references"
        scope = KnowledgeAccessScope.for_namespace(namespace)
        store = InMemoryKnowledgeStore(access_scope=scope)
        indexer = KnowledgeIndexer(store)
        for text in ("Release authority: retired team.", "Release authority: reviewed team."):
            await indexer.index_text(
                KnowledgeIndexRequest(entry_id="release", namespace=namespace, text=text)
            )
        provider = ReferenceReader()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="project"), knowledge_store=store, knowledge_access_scope=scope
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"),
            tools=[SearchKnowledgeTool(default_namespace=namespace), ReadKnowledgeTool()],
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Who approves releases?")],
                    max_steps=3,
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
        assert len(provider.requests) == 3
        results = [
            part
            for message in provider.requests[-1].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert len(results) == 2 and all(not result.is_error for result in results)
        assert "[chunk_index=0 revision=2]" in results[-1].content
        assert "reviewed team" in results[-1].content
        assert all("retired team" not in request.model_dump_json() for request in provider.requests)

    asyncio.run(run())


@pytest.mark.parametrize("lookup", ["current", "missing", "denied"])
def test_search_can_follow_silent_automatic_recall_without_widening_scope(lookup: str) -> None:
    async def run() -> None:
        namespace = "project:example"
        scope = KnowledgeAccessScope.for_namespace(namespace)
        knowledge = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=_FixtureEmbeddings(),
            embedding_model="fallback-fixture",
            embedding_dimensions=2,
        )
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        original = KnowledgeEntry(
            id="release-authority",
            namespace=namespace,
            text="Release approval authority: retired-board.",
            created_at=stamp,
            updated_at=stamp,
        )
        current = original.model_copy(
            update={"revision": 2, "text": "Release approval authority: delivery-board."}
        )
        await knowledge.create_entry(original, access_scope=scope)
        await knowledge.append_entry_revision(current, expected_revision=1, access_scope=scope)
        foreign = KnowledgeEntry(
            id="other-release-authority",
            namespace="project:other",
            text="Release approval authority: foreign-board.",
        )
        await knowledge.create_entry(
            foreign, access_scope=KnowledgeAccessScope.for_namespace("project:other")
        )
        for target in (namespace, "project:other"):
            worker = await knowledge.process_embedding_changes(
                target,
                "test-worker",
                limit=20,
                access_scope=KnowledgeAccessScope.for_namespace(target),
            )
            assert worker.failed_records == 0
        question = "Who gives approval for deployments?"
        assert query_concept_eligibility(
            question, current.text, version="cayu.query_concepts.v4"
        ) == (
            "low_relevance",
            "weak_query_support",
        )
        arguments: dict = {
            "query": "approval" if lookup != "missing" else "nonexistent-term",
            "limit": 5,
            "mode": "keyword",
        }
        if lookup == "denied":
            arguments["namespace"] = "project:other"
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="search_knowledge", arguments=arguments),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta(
                        "Scripted completion; not an answer-quality assertion."
                    ),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        fusion = WeightedReciprocalRankFusionConfig(
            configuration_version="fallback-test-v1",
            channel_weights={"knowledge.lexical": 1.0, "knowledge.semantic": 1.0},
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=AutomaticRecallPolicy(
                calibration_version="fallback-test-v1",
                relevance_policy="cayu.query_concepts.v4",
                fusion_strategy_version=fusion.strategy_version,
                fusion_configuration_version=fusion.configuration_version,
                mode="strong_matches",
                minimum_inject_score=0.01,
                minimum_offer_score=0.005,
            ),
            fusion_config=fusion,
            sources=AutomaticRecallSourceConfig(
                knowledge_namespace=namespace,
                knowledge_required=True,
                include_knowledge=True,
                include_transcript=False,
            ),
        )
        app = CayuApp(
            enable_logging=False,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="fallback-test",
                fingerprint_key="test-only-fallback-evidence-key-material",
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="project"),
                knowledge_store=knowledge,
                knowledge_access_scope=scope,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"),
            context_policy=policy,
            tools=[SearchKnowledgeTool(default_namespace=namespace)],
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", question)],
                    max_steps=2,
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert current.text not in provider.requests[0].model_dump_json()
        results = [
            part
            for message in provider.requests[1].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert len(results) == 1
        assert (current.text in results[0].content) is (lookup == "current")
        if lookup != "denied":
            assert not results[0].is_error
        if lookup != "current":
            assert current.text not in provider.requests[1].model_dump_json()
        for request in provider.requests:
            assert "retired-board" not in request.model_dump_json()
            assert "foreign-board" not in request.model_dump_json()
        recalls = [event for event in events if event.type is EventType.AUTOMATIC_RECALL_COMPLETED]
        assert (
            len(recalls) == 1
        )  # Explicit search does not refresh the frozen automatic contribution.
        assert recalls[0].payload["recall_candidate_count"] > 0
        assert all(s["status"] == "complete" for s in recalls[0].payload["source_statuses"])

    asyncio.run(run())
