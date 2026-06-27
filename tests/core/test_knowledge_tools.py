from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from cayu import (
    Environment,
    EnvironmentSpec,
    InMemoryKnowledgeStore,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    ReadKnowledgeTool,
    SearchKnowledgeTool,
    ToolContext,
)
from cayu.environments import copy_environment


def test_environment_accepts_and_copies_knowledge_store() -> None:
    store = InMemoryKnowledgeStore()
    environment = Environment(
        EnvironmentSpec(name="local"),
        knowledge_store=store,
    )

    copied = copy_environment(environment)

    assert copied.knowledge_store is store


def test_environment_rejects_invalid_knowledge_store() -> None:
    with pytest.raises(TypeError, match="knowledge_store must implement KnowledgeStore"):
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=cast("Any", object()),
        )


def test_search_knowledge_requires_configured_store() -> None:
    async def run():
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1"),
            {"query": "refund policy"},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "missing_knowledge_store"}


def test_search_knowledge_returns_ranked_hits_with_filters() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="payments",
                namespace="ops",
                title="Payment reminders",
                kind="procedure",
                labels={"project": "billing"},
                aspects=["payments"],
                impact_targets=["operator.workflow"],
                text=(
                    "# Payment reminders\n\n"
                    "Do not send payment reminders when the PO number is missing."
                ),
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="deploy",
                namespace="ops",
                kind="procedure",
                labels={"project": "infra"},
                text="Run migrations before deploy.",
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "query": "payment reminders PO",
                "namespace": "ops",
                "labels": {"project": "billing"},
                "kinds": ["procedure"],
                "aspects": ["payments"],
                "impact_targets": ["operator.workflow"],
                "limit": 5,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert "entry_id='payments'" in result.content
    assert "read_knowledge" in result.content
    assert result.structured is not None
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["payments"]
    assert result.structured["hits"][0]["title"] == "Payment reminders"


def test_search_knowledge_delegates_mode_support_to_store() -> None:
    async def run():
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=InMemoryKnowledgeStore()),
            {"query": "refund policy", "mode": "semantic"},
        )

    with pytest.raises(ValueError, match="InMemoryKnowledgeStore supports only"):
        asyncio.run(run())


def test_read_knowledge_returns_bounded_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="doc",
                text="# Guide\n\nFirst chunk has setup steps.\n\nSecond chunk has approval rules.",
                chunk_target_bytes=45,
                chunk_overlap_bytes=0,
            )
        )
        return await ReadKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "entry_id": "doc",
                "chunk_index": 1,
                "around": 1,
                "max_chunks": 3,
                "max_bytes": 10_000,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert "Knowledge chunks for entry_id 'doc'" in result.content
    assert result.structured is not None
    assert result.structured["entry_id"] == "doc"
    assert result.structured["chunks"]
    assert result.structured["chunks"][0]["entry_id"] == "doc"


def test_read_knowledge_without_chunk_index_reads_from_start() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(entry_id="policy", text="Always verify bank details.")
        )
        return await ReadKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"entry_id": "policy"},
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert "Always verify bank details" in result.content


def test_read_knowledge_requires_configured_store() -> None:
    async def run():
        return await ReadKnowledgeTool().run(
            ToolContext(session_id="session_1"),
            {"entry_id": "policy"},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "missing_knowledge_store"}
