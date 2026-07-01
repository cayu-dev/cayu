from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from cayu import (
    Environment,
    EnvironmentSpec,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    ListKnowledgeTool,
    ReadKnowledgeTool,
    SearchKnowledgeTool,
    ToolContext,
)
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.environments import copy_environment


class KeywordEmbeddingProvider(TextEmbeddingProvider):
    name = "keyword-test"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=_test_embedding_vector(text))
                for index, text in enumerate(request.texts)
            ],
        )


def _test_embedding_vector(text: str) -> list[float]:
    folded = text.casefold()
    return [
        1.0
        if any(
            term in folded for term in ("auth", "broker", "credential", "github", "proxy", "token")
        )
        else 0.0,
        1.0 if any(term in folded for term in ("invoice", "payment", "refund")) else 0.0,
        1.0 if any(term in folded for term in ("sendgrid", "email")) else 0.0,
    ]


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
    query = result.structured["query"]
    assert query["query"] == "payment reminders PO"
    assert query["all"] == []
    assert "text" not in query
    assert "all_terms" not in query
    assert "statuses" not in query
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["payments"]
    assert result.structured["hits"][0]["title"] == "Payment reminders"


def test_search_knowledge_accepts_structured_boolean_terms() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="github",
                text="GitHub push needs a credential broker.",
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="sendgrid",
                text="SendGrid email needs a secret proxy.",
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "any": ["credential", "secret"],
                "all": ["github push"],
                "none": ["fixture only"],
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    query = result.structured["query"]
    assert query["query"] is None
    assert query["any"] == ["credential", "secret"]
    assert query["all"] == ["github push"]
    assert query["none"] == ["fixture only"]
    assert "any_terms" not in query
    assert "all_terms" not in query
    assert "none_terms" not in query
    assert "statuses" not in query
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["github"]


def test_search_knowledge_schema_keeps_portable_validation_hints() -> None:
    schema = SearchKnowledgeTool.spec.input_schema

    assert "broad keyword query" in SearchKnowledgeTool.spec.description
    assert "truncated facet value" in SearchKnowledgeTool.spec.description
    assert "semantic or hybrid" in schema["properties"]["mode"]["description"]
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert schema["properties"]["query"]["minLength"] == 1
    assert schema["properties"]["query"]["pattern"] == "\\S"
    assert schema["properties"]["any"]["minItems"] == 1
    assert schema["properties"]["any"]["items"]["minLength"] == 1
    assert schema["properties"]["any"]["items"]["pattern"] == "\\S"
    assert schema["properties"]["all"]["minItems"] == 1
    assert schema["properties"]["none"]["minItems"] == 1
    assert schema["properties"]["phrases"]["minItems"] == 1
    assert "propertyNames" not in schema["properties"]["labels"]
    assert schema["properties"]["labels"]["additionalProperties"]["pattern"] == "\\S"
    assert "untruncated discovery result" in schema["properties"]["aspects"]["description"]


def test_search_knowledge_semantic_mode_uses_embedding_store() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="remote_git_credentials",
                namespace="ops",
                text=(
                    "Use a brokered Git HTTP proxy for GitHub pushes from a remote "
                    "sandbox. Keep credentials outside the sandbox."
                ),
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="invoice_approval",
                namespace="ops",
                text="Invoice refunds require approval before payment.",
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "query": "remote sandbox auth",
                "namespace": "ops",
                "mode": "semantic",
                "limit": 5,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["query"]["mode"] == "semantic"
    assert result.structured["search_modes"] == ["auto", "keyword", "semantic", "hybrid"]
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["remote_git_credentials"]
    assert result.structured["hits"][0]["score_kind"] == "inmemory_semantic"
    assert "chunk_index=0" in result.content


def test_search_knowledge_runtime_requires_a_positive_search_field() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert "requires `text`, `any_terms`, `all_terms`, or `phrases`" in result.content


def test_search_knowledge_caps_model_facing_preview_per_hit() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="long",
                text="credential " + ("important guidance " * 20),
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "query": "credential",
                "preview_bytes": 24,
                "max_bytes": 10_000,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["preview_bytes"] == 24
    hit = result.structured["hits"][0]
    assert hit["entry_id"] == "long"
    assert hit["text_preview_truncated"] is True
    assert len(hit["text_preview"].encode("utf-8")) <= 24
    assert "[preview truncated]" in result.content
    assert "read_knowledge" in result.content


def test_list_knowledge_discovers_entries_and_facets() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook",
                namespace="ops",
                kind="procedure",
                labels={"project": "billing"},
                text="Payment reminder runbook.",
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="warning",
                namespace="ops",
                kind="warning",
                labels={"project": "billing"},
                text="Approval warning.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "namespace": "ops",
                "labels": {"project": "billing"},
                "group_by": "kind",
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert "Knowledge discovery" in result.content
    assert result.structured is not None
    assert "statuses" not in result.structured["query"]
    assert result.structured["query"]["group_by"] == ["kind"]
    assert result.structured["search_modes"] == ["auto", "keyword"]
    assert result.structured["include_entries"] is False
    assert result.structured["truncated"] is False
    assert result.structured["entries"] == []
    assert "entry_id=" not in result.content
    assert "Search modes: auto, keyword" in result.content
    assert [(facet["value"], facet["count"]) for facet in result.structured["facets"]] == [
        ("procedure", 1),
        ("warning", 1),
    ]


def test_list_knowledge_can_include_entries_with_facets() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook",
                namespace="ops",
                kind="procedure",
                text="Payment reminder runbook.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "namespace": "ops",
                "group_by": "kind",
                "include_entries": True,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["include_entries"] is True
    assert [entry["entry_id"] for entry in result.structured["entries"]] == ["runbook"]
    assert "entry_id='runbook'" in result.content


def test_list_knowledge_advertises_embedding_search_modes() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook",
                namespace="ops",
                kind="procedure",
                text="Remote sandbox credential proxy runbook.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "namespace": "ops",
                "group_by": "kind",
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["search_modes"] == ["auto", "keyword", "semantic", "hybrid"]
    assert "Search modes: auto, keyword, semantic, hybrid" in result.content


def test_list_knowledge_can_return_multiple_facet_groups() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook",
                namespace="ops",
                kind="procedure",
                labels={"project": "billing"},
                aspects=["payments"],
                text="Payment reminder runbook.",
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="warning",
                namespace="ops",
                kind="warning",
                labels={"project": "billing"},
                aspects=["approvals"],
                text="Approval warning.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "namespace": "ops",
                "group_by": ["kind", "aspect", "label"],
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["query"]["group_by"] == ["kind", "aspect", "label"]
    assert result.structured["include_entries"] is False
    assert result.structured["truncated"] is False
    assert result.structured["entries"] == []
    facet_groups = result.structured["facet_groups"]
    assert {facet["value"] for facet in facet_groups["kind"]} == {"procedure", "warning"}
    assert {facet["value"] for facet in facet_groups["aspect"]} == {
        "approvals",
        "payments",
    }
    assert facet_groups["label"] == [
        {"field": "label", "key": "project", "value": "billing", "count": 2}
    ]
    assert "- kind: procedure (1)" in result.content
    assert "- aspect: payments (1)" in result.content
    assert "- label: project=billing (2)" in result.content


def test_list_knowledge_schema_advertises_group_by_as_portable_array() -> None:
    schema = ListKnowledgeTool.spec.input_schema

    assert "facets were truncated" in ListKnowledgeTool.spec.description
    assert "anyOf" not in schema
    assert "oneOf" not in schema["properties"]["group_by"]
    assert schema["properties"]["group_by"]["type"] == "array"
    assert schema["properties"]["group_by"]["minItems"] == 1
    assert "propertyNames" not in schema["properties"]["labels"]
    assert "facets are truncated" in schema["properties"]["group_by"]["description"]
    assert "higher value" in schema["properties"]["limit"]["description"]


def test_list_knowledge_runtime_still_accepts_single_group_by_string() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook",
                namespace="ops",
                kind="procedure",
                text="Payment reminder runbook.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "namespace": "ops",
                "group_by": "kind",
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["query"]["group_by"] == ["kind"]
    assert result.structured["facet_groups"]["kind"] == [
        {"field": "kind", "key": None, "value": "procedure", "count": 1}
    ]


def test_list_knowledge_returns_tool_error_for_invalid_arguments() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"group_by": []},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "group_by" in result.content


def test_list_knowledge_reports_facet_truncation() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        for index in range(5):
            await KnowledgeIndexer(store).index_text(
                KnowledgeIndexRequest(
                    entry_id=f"entry_{index}",
                    labels={"area": f"area_{index}"},
                    text=f"Knowledge entry {index}.",
                )
            )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "group_by": "label",
                "limit": 3,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["include_entries"] is False
    assert result.structured["facets_truncated"] is True
    assert result.structured["truncated"] is True
    assert len(result.structured["facets"]) == 3
    assert "Facet list truncated" in result.content
    assert "Increase limit or narrow filters" in result.content


def test_list_knowledge_reports_later_facet_truncation_with_entries() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="a",
                namespace="ops",
                kind="procedure",
                labels={"a": "1", "b": "2", "c": "3", "d": "4"},
                text="Entry A.",
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="b",
                namespace="ops",
                kind="procedure",
                labels={"e": "5"},
                text="Entry B.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "namespace": "ops",
                "group_by": ["kind", "label"],
                "include_entries": True,
                "limit": 3,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["include_entries"] is True
    assert len(result.structured["entries"]) == 2
    assert result.structured["facets_truncated"] is True
    assert result.structured["truncated"] is True
    assert "Facet list truncated" in result.content


def test_list_knowledge_does_not_claim_no_entries_when_hidden_entries_match() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook",
                namespace="ops",
                text="Payment reminder runbook.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "namespace": "ops",
                "group_by": "label",
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["entries"] == []
    assert result.structured["facets"] == []
    assert result.structured["total_entries_known"] == 1
    assert "found matching entries" in result.content
    assert "No knowledge entries found" not in result.content


def test_list_knowledge_includes_entries_by_default_without_group_by() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook",
                namespace="ops",
                kind="procedure",
                text="Payment reminder runbook.",
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"namespace": "ops"},
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["include_entries"] is True
    assert [entry["entry_id"] for entry in result.structured["entries"]] == ["runbook"]


def test_list_knowledge_caps_model_facing_preview_per_entry() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="long",
                text="Long operating note. " * 20,
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "preview_bytes": 20,
                "max_bytes": 10_000,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["preview_bytes"] == 20
    entry = result.structured["entries"][0]
    assert entry["entry_id"] == "long"
    assert entry["text_preview_truncated"] is True
    assert len(entry["text_preview"].encode("utf-8")) <= 20
    assert "[preview truncated]" in result.content


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


def test_read_knowledge_returns_tool_error_for_invalid_entry_id() -> None:
    async def run():
        return await ReadKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=InMemoryKnowledgeStore()),
            {"entry_id": 123},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "entry_id" in result.content


def test_read_knowledge_returns_tool_error_for_invalid_window() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(entry_id="policy", text="Always verify bank details.")
        )
        return await ReadKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"entry_id": "policy", "around": 1},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "around" in result.content
