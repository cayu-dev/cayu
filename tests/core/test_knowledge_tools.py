from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from cayu import (
    Environment,
    EnvironmentSpec,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    KnowledgeQuery,
    KnowledgeStatus,
    ListKnowledgeTool,
    ReadKnowledgeTool,
    RememberKnowledgePolicy,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
    ToolContext,
    ToolSpec,
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


class WeightedEmbeddingProvider(TextEmbeddingProvider):
    name = "weighted-test"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=_weighted_embedding_vector(text))
                for index, text in enumerate(request.texts)
            ],
        )


class FailingEmbeddingProvider(TextEmbeddingProvider):
    name = "failing-test"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        raise RuntimeError("embedding service unavailable")


class PartialWriteKnowledgeStore(InMemoryKnowledgeStore):
    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        if not chunks:
            raise AssertionError("test requires chunks")
        await super().put_entry_with_chunks(entry, chunks[:1])
        raise RuntimeError("partial write failure")


class CorruptingWriteKnowledgeStore(InMemoryKnowledgeStore):
    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        await super().put_entry_with_chunks(entry.model_copy(update={"labels": {}}), chunks)
        raise RuntimeError("corrupt write failure")


def _weighted_embedding_vector(text: str) -> list[float]:
    folded = text.casefold()
    if "sendgrid" in folded:
        return [0.2, 0.98]
    if "runbook" in folded:
        return [0.4, 0.9165]
    if "remote" in folded or "github" in folded or "auth" in folded:
        return [1.0, 0.0]
    return [0.0, 1.0]


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
    assert "min_score" not in schema["properties"]
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


def test_search_knowledge_score_override_is_opt_in() -> None:
    default_schema = SearchKnowledgeTool.spec.input_schema
    opt_in_schema = SearchKnowledgeTool(allow_score_override=True).spec.input_schema

    assert "min_score" not in default_schema["properties"]
    assert opt_in_schema["properties"]["min_score"]["minimum"] == 0.0
    assert opt_in_schema["properties"]["min_score"]["maximum"] == 1.0
    assert (
        "application-owned retrieval policy"
        in opt_in_schema["properties"]["min_score"]["description"]
    )
    assert "min_score" not in SearchKnowledgeTool.spec.input_schema["properties"]


def test_remember_knowledge_schema_describes_pending_policy() -> None:
    schema = RememberKnowledgeTool.spec.input_schema

    assert RememberKnowledgeTool.spec.name == "remember_knowledge"
    assert "pending review" in RememberKnowledgeTool.spec.description
    assert "edit, archive, or delete" in RememberKnowledgeTool.spec.description
    assert schema["required"] == ["text"]
    assert "entry_id" not in schema["properties"]
    assert "status" not in schema["properties"]
    assert "max_bytes" not in schema["properties"]
    assert "namespace" not in schema["properties"]
    assert "labels" not in schema["properties"]
    assert "impact_targets" not in schema["properties"]
    assert "importance" not in schema["properties"]
    assert "confidence" not in schema["properties"]
    assert "one stable" in schema["properties"]["text"]["description"]
    assert "large documents" in schema["properties"]["text"]["description"]


def test_remember_knowledge_schema_exposes_allowed_kinds() -> None:
    default_schema = RememberKnowledgeTool.spec.input_schema
    restricted_schema = RememberKnowledgeTool(
        policy=RememberKnowledgePolicy(
            allowed_kinds=("fact", "procedure"),
            default_kind="fact",
        )
    ).spec.input_schema

    assert "enum" not in default_schema["properties"]["kind"]
    assert restricted_schema["properties"]["kind"]["enum"] == ["fact", "procedure"]
    assert (
        "Choose one of: fact, procedure" in restricted_schema["properties"]["kind"]["description"]
    )
    assert "uses fact" in restricted_schema["properties"]["kind"]["description"]
    assert "enum" not in RememberKnowledgeTool.spec.input_schema["properties"]["kind"]


def test_remember_knowledge_schema_does_not_mutate_custom_spec() -> None:
    spec = RememberKnowledgeTool.spec.model_copy(deep=True)

    tool = RememberKnowledgeTool(
        spec=spec,
        policy=RememberKnowledgePolicy(
            allowed_kinds=("fact", "procedure"),
            default_kind="fact",
        ),
    )

    assert type(spec) is ToolSpec
    assert "enum" not in spec.input_schema["properties"]["kind"]
    assert tool.spec.input_schema["properties"]["kind"]["enum"] == ["fact", "procedure"]


def test_remember_knowledge_defaults_model_writes_to_pending() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        result = await RememberKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                agent_name="assistant",
                environment_name="local",
                workspace_id="workspace_1",
                knowledge_store=store,
            ),
            {
                "text": "Remote sandbox Git pushes should use a brokered credential proxy.",
                "title": "Remote sandbox Git credentials",
                "kind": "procedure",
                "aspects": ["git", "credentials"],
            },
        )
        assert result.structured is not None
        entry_id = result.structured["entry"]["entry_id"]
        entry = await store.get_entry(entry_id)
        default_search = await store.search(
            KnowledgeQuery(text="brokered credential proxy", namespace="default")
        )
        pending_search = await store.search(
            KnowledgeQuery(
                text="brokered credential proxy",
                namespace="default",
                statuses=[KnowledgeStatus.PENDING],
            )
        )
        chunks = await store.read_chunks(entry_id, max_chunks=5, max_bytes=20_000)
        return result, entry, default_search, pending_search, chunks

    result, entry, default_search, pending_search, chunks = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["status"] == "pending"
    assert entry is not None
    assert entry.status is KnowledgeStatus.PENDING
    assert entry.created_by_type.value == "model"
    assert entry.created_by == "assistant"
    assert entry.source_type == "tool"
    assert entry.source_uri == "cayu://sessions/session_1"
    assert entry.source_id == "session_1"
    assert entry.metadata == {
        "tool_name": "remember_knowledge",
        "session_id": "session_1",
        "agent_name": "assistant",
        "environment_name": "local",
        "workspace_id": "workspace_1",
    }
    assert entry.labels == {}
    assert entry.aspects == ["git", "credentials"]
    assert entry.importance is None
    assert entry.importance_source is None
    assert entry.confidence is None
    assert default_search.hits == []
    assert [hit.entry.id for hit in pending_search.hits] == [entry.id]
    assert len(chunks) == 1
    assert chunks[0].entry_id == entry.id


def test_remember_knowledge_status_is_policy_owned() -> None:
    async def run_default():
        store = InMemoryKnowledgeStore()
        result = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_1", agent_name="assistant", knowledge_store=store),
            {
                "text": "Use a trusted proxy for GitHub pushes from remote sandboxes.",
                "status": "active",
            },
        )
        assert result.structured is not None
        entry = await store.get_entry(result.structured["entry"]["entry_id"])
        return result, entry

    async def run_active():
        store = InMemoryKnowledgeStore()
        result = await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(
                default_status=KnowledgeStatus.ACTIVE,
                allow_active_writes=True,
                require_labels={"project": "cayu"},
            )
        ).run(
            ToolContext(session_id="session_2", agent_name="assistant", knowledge_store=store),
            {"text": "Invoice refunds require audit logging.", "kind": "warning"},
        )
        assert result.structured is not None
        entry = await store.get_entry(result.structured["entry"]["entry_id"])
        search = await store.search(KnowledgeQuery(text="invoice refunds audit"))
        return result, entry, search

    default_result, default_entry = asyncio.run(run_default())
    active_result, active_entry, active_search = asyncio.run(run_active())

    assert default_result.structured["status"] == "pending"
    assert default_entry.status is KnowledgeStatus.PENDING
    assert default_entry.metadata == {
        "tool_name": "remember_knowledge",
        "session_id": "session_1",
        "agent_name": "assistant",
    }
    assert "pending review" in default_result.content
    assert active_result.structured["status"] == "active"
    assert active_entry.status is KnowledgeStatus.ACTIVE
    assert active_entry.labels == {"project": "cayu"}
    assert [hit.entry.id for hit in active_search.hits] == [active_entry.id]


def test_remember_knowledge_accepts_policy_dict() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        result = await RememberKnowledgeTool(
            policy={
                "default_status": "active",
                "allow_active_writes": True,
                "default_namespace": "project:cayu",
            }
        ).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "Use the brokered Git HTTP proxy for sandbox pushes."},
        )
        assert result.structured is not None
        entry = await store.get_entry(result.structured["entry"]["entry_id"])
        return entry

    entry = asyncio.run(run())

    assert entry is not None
    assert entry.status is KnowledgeStatus.ACTIVE
    assert entry.namespace == "project:cayu"


def test_remember_knowledge_policy_owns_namespace_and_labels() -> None:
    async def run_policy_scope():
        store = InMemoryKnowledgeStore()
        result = await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(
                default_namespace="project:cayu",
                require_labels={"project": "cayu", "tenant": "trusted"},
            )
        ).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "text": "Sandbox pushes use a brokered credential proxy.",
                "namespace": "attacker",
                "labels": {"project": "wrong", "area": "git"},
                "impact_targets": ["sandbox.git.push"],
                "importance": 1.0,
                "confidence": 1.0,
            },
        )
        assert result.structured is not None
        return await store.get_entry(result.structured["entry"]["entry_id"])

    entry = asyncio.run(run_policy_scope())

    assert entry is not None
    assert entry.namespace == "project:cayu"
    assert entry.labels == {"project": "cayu", "tenant": "trusted"}
    assert entry.impact_targets == []
    assert entry.importance is None
    assert entry.confidence is None


def test_remember_knowledge_policy_restricts_kinds() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        return await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(
                allowed_kinds=("fact", "procedure"),
                default_kind="fact",
            )
        ).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "Never store this as a skill.", "kind": "skill"},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "`kind` must be one of: fact, procedure" in result.content


def test_remember_knowledge_rejects_oversized_text() -> None:
    async def run():
        return await RememberKnowledgeTool(max_text_bytes=5).run(
            ToolContext(session_id="session_1", knowledge_store=InMemoryKnowledgeStore()),
            {"text": "abcdef"},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "`text` must be at most 5 bytes" in result.content


def test_remember_knowledge_rejects_truncated_index_without_writing() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        result = await RememberKnowledgeTool(chunk_target_bytes=1_000, max_chunks=1).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "alpha " * 1_000},
        )
        search = await store.search(
            KnowledgeQuery(text="alpha", statuses=[KnowledgeStatus.PENDING])
        )
        return result, search

    result, search = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "chunk capacity" in result.content
    assert search.hits == []


def test_remember_knowledge_preserves_entry_on_embedding_write_failure() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=FailingEmbeddingProvider(),
            embedding_model="test-embedding",
        )
        result = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "Remembered text should survive failed embedding writes."},
        )
        assert result.structured is not None
        entry_id = result.structured["entry"]["entry_id"]
        entry = await store.get_entry(entry_id)
        chunks = await store.read_chunks(
            entry_id,
            max_chunks=5,
            max_bytes=20_000,
        )
        return result, entry, chunks

    result, entry, chunks = asyncio.run(run())

    assert result.is_error is False
    assert result.structured["post_write_error"] == "embedding service unavailable"
    assert "embedding service unavailable" not in result.content
    assert "Knowledge stored as pending" in result.content
    assert entry is not None
    assert entry.status is KnowledgeStatus.PENDING
    assert len(chunks) == 1


def test_remember_knowledge_rejects_partial_write_failure() -> None:
    async def run():
        store = PartialWriteKnowledgeStore()
        result = await RememberKnowledgeTool(chunk_target_bytes=1_000).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "alpha " * 400},
        )
        assert result.structured is not None
        entry = await store.get_entry(result.structured["entry_id"])
        chunks = await store.read_chunks(
            result.structured["entry_id"],
            max_chunks=5,
            max_bytes=20_000,
        )
        return result, entry, chunks

    result, entry, chunks = asyncio.run(run())

    assert result.is_error is True
    assert result.structured["error"] == "knowledge_write_failed"
    assert result.structured["cleanup"] == "completed"
    assert "partial write failure" in result.content
    assert entry is None
    assert chunks == []


def test_remember_knowledge_rejects_corrupt_post_write_failure() -> None:
    async def run():
        store = CorruptingWriteKnowledgeStore()
        result = await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(require_labels={"project": "cayu"})
        ).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "Sandbox Git pushes use brokered credentials."},
        )
        assert result.structured is not None
        entry = await store.get_entry(result.structured["entry_id"])
        chunks = await store.read_chunks(
            result.structured["entry_id"],
            max_chunks=5,
            max_bytes=20_000,
        )
        return result, entry, chunks

    result, entry, chunks = asyncio.run(run())

    assert result.is_error is True
    assert result.structured["error"] == "knowledge_write_failed"
    assert result.structured["cleanup"] == "completed"
    assert "corrupt write failure" in result.content
    assert entry is None
    assert chunks == []


def test_remember_knowledge_validates_chunk_configuration() -> None:
    with pytest.raises(ValueError, match="chunk_target_bytes"):
        RememberKnowledgeTool(chunk_target_bytes=799)

    with pytest.raises(ValueError, match="max_text_bytes"):
        RememberKnowledgeTool(max_text_bytes=0)


def test_remember_knowledge_runtime_requires_store() -> None:
    async def run():
        return await RememberKnowledgeTool().run(
            ToolContext(session_id="session_1"),
            {"text": "Remember this later."},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "missing_knowledge_store"}


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


def test_search_knowledge_auto_filters_weak_semantic_neighbors_by_default() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=WeightedEmbeddingProvider(),
            embedding_model="test-embedding",
            semantic_min_score=0.0,
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
                entry_id="sendgrid_proxy",
                namespace="ops",
                text="For SendGrid, prefer a trusted credential proxy outside the sandbox.",
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "query": "remote sandbox auth",
                "namespace": "ops",
                "limit": 5,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["query"]["mode"] == "auto"
    assert result.structured["min_score"] == 0.75
    assert result.structured["filtered_hits"] == 1
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["remote_git_credentials"]
    assert result.structured["hits"][0]["score_kind"] == "inmemory_hybrid"


def test_search_knowledge_auto_min_score_zero_keeps_weak_semantic_neighbors() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=WeightedEmbeddingProvider(),
            embedding_model="test-embedding",
            semantic_min_score=0.0,
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="remote_git_credentials",
                namespace="ops",
                text="GitHub remote sandbox auth should use a credential broker.",
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="sendgrid_proxy",
                namespace="ops",
                text="For SendGrid, prefer a trusted credential proxy outside the sandbox.",
            )
        )
        return await SearchKnowledgeTool(allow_score_override=True).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "query": "remote sandbox auth",
                "namespace": "ops",
                "min_score": 0,
                "limit": 5,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["min_score"] == 0.0
    assert result.structured["filtered_hits"] == 0
    assert [hit["entry_id"] for hit in result.structured["hits"]] == [
        "remote_git_credentials",
        "sendgrid_proxy",
    ]


def test_search_knowledge_auto_min_score_preserves_unscored_keyword_hits() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=WeightedEmbeddingProvider(),
            embedding_model="test-embedding",
            semantic_min_score=0.75,
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="remote_git_credentials",
                namespace="ops",
                text="GitHub remote sandbox auth should use a credential broker.",
            )
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="runbook_keyword_hit",
                namespace="ops",
                text="Remote sandbox auth runbook uses a documented fallback procedure.",
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "query": "remote sandbox auth",
                "namespace": "ops",
                "limit": 5,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["min_score"] == 0.75
    hit_by_id = {hit["entry_id"]: hit for hit in result.structured["hits"]}
    assert set(hit_by_id) == {"remote_git_credentials", "runbook_keyword_hit"}
    assert hit_by_id["remote_git_credentials"]["score_normalized"] == 1.0
    assert hit_by_id["runbook_keyword_hit"]["score_normalized"] is None
    assert "hybrid keyword match" in hit_by_id["runbook_keyword_hit"]["reason"]


def test_search_knowledge_default_rejects_score_override_argument() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=WeightedEmbeddingProvider(),
            embedding_model="test-embedding",
            semantic_min_score=0.0,
        )
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="remote_git_credentials",
                namespace="ops",
                text="GitHub remote sandbox auth should use a credential broker.",
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {
                "query": "remote sandbox auth",
                "namespace": "ops",
                "min_score": 0,
            },
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "not enabled" in result.content


def test_search_knowledge_keyword_store_auto_does_not_apply_min_score() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="github",
                text="GitHub push needs a credential broker.",
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"query": "github credential"},
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["search_modes"] == ["auto", "keyword"]
    assert result.structured["min_score"] is None
    assert result.structured["filtered_hits"] == 0
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["github"]


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
