"""Execute the documented recipe; scripted decisions are not quality evaluations."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cayu import (
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    ModelProvider,
    ModelStreamEvent,
)
from cayu.core.messages import TextPart, ToolResultPart
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.providers.base import ModelRequest
from cayu.runtime.sessions import InMemorySessionStore


class _FixtureEmbeddings(TextEmbeddingProvider):
    """Constant vectors exercise semantic wiring, not semantic quality."""

    name = "offer-recipe-fixture"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=[1.0, 0.0])
                for index, _ in enumerate(request.texts)
            ],
        )


def _offer_payload(request: ModelRequest) -> dict:
    matches = [
        match
        for message in request.messages
        for part in message.content
        if isinstance(part, TextPart)
        for match in re.findall(
            r'<cayu_automatic_memory version="2">\n(.*?)\n</cayu_automatic_memory>',
            part.text,
            flags=re.DOTALL,
        )
    ]
    assert len(matches) == 1
    return json.loads(matches[0])


@pytest.mark.parametrize("lookup", ["read", "search"])
@pytest.mark.parametrize("semantic", [False, True])
def test_documented_offers_bound_previews_and_support_scoped_tools(
    lookup: str, semantic: bool
) -> None:
    class RecipeProvider(ModelProvider):
        name = "recipe-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                payload = _offer_payload(request)
                assert "focus" not in payload
                offer = payload["offer"]
                assert len(offer["items"]) == 5
                assert offer["omitted"] > 0 and offer["partial"]
                assert all(len(item["preview"].encode("utf-8")) <= 240 for item in offer["items"])
                assert "z-contact" not in {item["read"]["entry_id"] for item in offer["items"]}
                guide = next(
                    item for item in offer["items"] if item["read"]["entry_id"] == "a-guide"
                )
                assert not guide["preview_complete"] and guide["read"]["revision"] == 2
                if lookup == "read":
                    yield ModelStreamEvent.tool_call(name="read_knowledge", arguments=guide["read"])
                else:
                    yield ModelStreamEvent.tool_call(
                        name="search_knowledge",
                        arguments={"query": "On-call contact", "mode": "keyword", "limit": 1},
                    )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            else:
                assert _offer_payload(request) == _offer_payload(self.requests[0])
                yield ModelStreamEvent.text_delta("Scripted recipe completed.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        store = (
            InMemoryEmbeddingKnowledgeStore(
                embedding_provider=_FixtureEmbeddings(),
                embedding_model="fixture",
                embedding_dimensions=2,
            )
            if semantic
            else InMemoryKnowledgeStore()
        )
        namespace = "project:example"
        scope = KnowledgeAccessScope.for_namespace(namespace)
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        guide = KnowledgeEntry(
            id="a-guide",
            namespace=namespace,
            text="Obsolete guide.",
            created_at=stamp,
            updated_at=stamp,
        )
        await store.create_entry(guide, access_scope=scope)
        text = (
            "Release procedure. "
            + "Record the deployment details. " * 15
            + "Approval gate: obtain sign-off."
        )
        await store.append_entry_revision(
            guide.model_copy(update={"revision": 2, "text": text}),
            expected_revision=1,
            access_scope=scope,
        )
        for index in range(5):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"b-reference-{index}",
                    namespace=namespace,
                    text=f"Release procedure reference {index}.",
                    created_at=stamp,
                    updated_at=stamp,
                ),
                access_scope=scope,
            )
        await store.create_entry(
            KnowledgeEntry(
                id="z-contact",
                namespace=namespace,
                text="On-call contact: operations.",
                created_at=stamp,
                updated_at=stamp,
            ),
            access_scope=scope,
        )
        await store.create_entry(
            KnowledgeEntry(
                id="foreign",
                namespace="project:other",
                text="Release procedure: forbidden material.",
            ),
            access_scope=KnowledgeAccessScope.for_namespace("project:other"),
        )
        if semantic:
            for target in (namespace, "project:other"):
                result = await store.process_embedding_changes(
                    target,
                    "test-worker",
                    limit=20,
                    access_scope=KnowledgeAccessScope.for_namespace(target),
                )
                assert result.failed_records == 0
        provider = RecipeProvider()
        bindings = {
            "provider": provider,
            "model": "fixture-model",
            "knowledge_store": store,
            "session_store": InMemorySessionStore(),
            "memory_evidence_secret": "recipe-test-only-fingerprint-key-material",
        }
        document = Path(__file__).resolve().parents[2] / "docs/knowledge-offers.md"
        blocks = re.findall(r"```python\n(.*?)\n```", document.read_text(), flags=re.DOTALL)
        assert len(blocks) == 1
        exec(compile(blocks[0], str(document), "exec"), bindings)
        assert bindings["recall_policy"].admission_policy.relevance_policy == "rank_only.v1"
        assert bindings["recall_policy"].delta_policy is None
        events = [event async for event in bindings["app"].run(bindings["request"])]
        assert str(events[-1].type) == "session.completed", events[-1].payload
        assert len(provider.requests) == 2
        assert "Approval gate" not in provider.requests[0].model_dump_json()
        assert "On-call contact: operations." not in provider.requests[0].model_dump_json()
        results = [
            part
            for message in provider.requests[1].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert len(results) == 1 and not results[0].is_error
        expected = (
            "Approval gate: obtain sign-off."
            if lookup == "read"
            else "On-call contact: operations."
        )
        assert expected in results[0].content
        for request in provider.requests:
            assert "Obsolete guide" not in request.model_dump_json()
            assert "forbidden material" not in request.model_dump_json()

    asyncio.run(run())
