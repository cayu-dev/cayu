from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pydantic import SecretStr, ValidationError

from cayu import (
    Environment,
    EnvironmentSpec,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    KnowledgeQuery,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    ListKnowledgeTool,
    ReadKnowledgeTool,
    RememberKnowledgePolicy,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
    SecretRedactor,
    ToolContext,
    ToolSpec,
    prepare_knowledge_publication,
)
from cayu.core.tools import Tool, ToolEffect, ToolResult
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.environments import copy_environment
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.runtime._tool_execution import run_tool
from cayu.storage.knowledge_indexer import (
    content_knowledge_entry_id,
    knowledge_source_hash,
)
from cayu.tools import knowledge as knowledge_module
from cayu.vaults import ResolvedSecret

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()


class _TestKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("access_scope", _ACCESS_SCOPE)
        super().__init__(*args, **kwargs)


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


class AcknowledgementLossKnowledgeStore(_TestKnowledgeStore):
    async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
        await super().publish_entry_with_chunks(
            entry,
            chunks,
            operation_id=operation_id,
        )
        raise RuntimeError("secret canary acknowledgement failure")


class CompetingLegacyKnowledgeStore(_TestKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self.legacy_put_calls = 0

    async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
        raise NotImplementedError

    async def load_entry_publication_receipt(self, operation_id):
        raise NotImplementedError

    async def put_entry_with_chunks(self, entry, chunks):
        self.legacy_put_calls += 1
        return await super().put_entry_with_chunks(entry, chunks)

    async def seed_competing_publication(self, entry, chunks) -> None:
        await super().put_entry_with_chunks(entry, chunks)


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
    store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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


def test_environment_rejects_scope_that_differs_from_bound_store() -> None:
    store = InMemoryKnowledgeStore(access_scope=KnowledgeAccessScope.for_namespace("project-a"))
    with pytest.raises(ValueError, match="must match the scope bound"):
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=store,
            knowledge_access_scope=KnowledgeAccessScope.for_namespace("project-b"),
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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


def test_search_knowledge_none_terms_exclude_sibling_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.put_entry_with_chunks(
            KnowledgeEntry(id="excluded", text="Integration summary."),
            [
                KnowledgeChunk(
                    id="excluded:0",
                    entry_id="excluded",
                    chunk_index=0,
                    text="GitHub push credential instructions.",
                ),
                KnowledgeChunk(
                    id="excluded:1",
                    entry_id="excluded",
                    chunk_index=1,
                    text="Deprecated proxy guidance.",
                ),
            ],
        )
        await store.put_entry(KnowledgeEntry(id="safe", text="GitHub credential instructions."))
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"query": "github", "none": ["deprecated"]},
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["safe"]


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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
            ToolContext(
                session_id="session_1",
                knowledge_store=InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            ),
            {"text": "abcdef"},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "`text` must be at most 5 bytes" in result.content


def test_remember_knowledge_rejects_truncated_index_without_writing() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
            access_scope=_ACCESS_SCOPE,
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
    assert result.structured["post_write_error"] == "publication_acknowledgement_lost"
    assert "embedding service unavailable" not in result.content
    assert "Knowledge stored as pending" in result.content
    assert entry is not None
    assert entry.status is KnowledgeStatus.PENDING
    assert len(chunks) == 1


def test_remember_knowledge_reconciles_acknowledgement_loss_without_exposing_error() -> None:
    async def run():
        store = AcknowledgementLossKnowledgeStore()
        ctx = ToolContext(
            session_id="session_1",
            idempotency_key="remember-operation-1",
            knowledge_store=store,
        )
        first = await RememberKnowledgeTool().run(ctx, {"text": "Rotate credentials weekly."})
        replay = await RememberKnowledgeTool().run(ctx, {"text": "Rotate credentials weekly."})
        receipt = await store.load_entry_publication_receipt("remember-operation-1")
        return first, replay, receipt

    first, replay, receipt = asyncio.run(run())

    assert first.is_error is False
    assert first.structured["post_write_error"] == "publication_acknowledgement_lost"
    assert "secret canary" not in first.content
    assert "secret canary" not in repr(first.structured)
    assert replay.is_error is False
    assert replay.structured["written"] is False
    assert replay.structured["already_known"] is None
    assert replay.structured["publication_replayed"] is True
    assert replay.structured["status"] is None
    assert receipt is not None


def test_remember_knowledge_never_dispatches_legacy_upsert_over_competing_winner() -> None:
    async def run():
        text = "Concurrent publication must survive unsupported legacy writes."
        source_hash = knowledge_source_hash(text)
        entry_id = content_knowledge_entry_id(
            namespace="default",
            kind="fact",
            source_hash=source_hash,
        )
        store = CompetingLegacyKnowledgeStore()
        winner = KnowledgeEntry(
            id=entry_id,
            text=text,
            source_hash=source_hash,
            created_by="concurrent-writer",
            metadata={"concurrent_owner": True},
        )
        winner_chunks = [
            KnowledgeChunk(
                id=f"{entry_id}:0",
                entry_id=entry_id,
                chunk_index=0,
                text=text,
                metadata={"concurrent_owner": True},
            )
        ]
        await store.seed_competing_publication(winner, winner_chunks)
        result = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": text, "title": "A legacy writer must not replace this title."},
        )
        entry = await store.get_entry(entry_id)
        chunks = await store.read_chunks(entry_id)
        return result, entry, chunks, store.legacy_put_calls

    result, entry, chunks, legacy_put_calls = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {
        "error": "knowledge_write_failed",
        "outcome": "owned_publication_unsupported",
        "cleanup": "not_attempted_unowned",
    }
    assert legacy_put_calls == 0
    assert entry is not None
    assert entry.created_by == "concurrent-writer"
    assert entry.title is None
    assert entry.metadata["concurrent_owner"] is True
    assert chunks[0].metadata["concurrent_owner"] is True


def test_remember_knowledge_rejects_store_with_only_receipt_lookup_support() -> None:
    class ReceiptOnlyKnowledgeStore(_TestKnowledgeStore):
        publish_entry_with_chunks = KnowledgeStore.publish_entry_with_chunks

    async def run():
        store = ReceiptOnlyKnowledgeStore()
        result = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "This must not reach a legacy publication path."},
        )
        return result, await store.search(
            KnowledgeQuery(text="legacy", statuses=[KnowledgeStatus.PENDING])
        )

    result, search = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {
        "error": "knowledge_write_failed",
        "outcome": "owned_publication_unsupported",
        "cleanup": "not_attempted_unowned",
    }
    assert search.hits == []


def test_remember_knowledge_rejects_invalid_operation_id_before_store_dispatch() -> None:
    class TrackingKnowledgeStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.receipt_reads = 0
            self.publications = 0

        async def load_entry_publication_receipt(self, operation_id):
            self.receipt_reads += 1
            return await super().load_entry_publication_receipt(operation_id)

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.publications += 1
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

    async def run():
        store = TrackingKnowledgeStore()
        result = await RememberKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                idempotency_key="x" * 257,
                knowledge_store=store,
            ),
            {"text": "Invalid operation identity must not reach the store."},
        )
        return result, store.receipt_reads, store.publications

    result, receipt_reads, publications = asyncio.run(run())

    assert result.is_error is True
    assert receipt_reads == 0
    assert publications == 0


def test_remember_knowledge_bounds_store_internal_entry_read_cancellation() -> None:
    class InternallyCancelledReadStore(_TestKnowledgeStore):
        async def get_entry(self, entry_id):
            del entry_id
            raise asyncio.CancelledError("private store cancellation")

    async def run():
        store = InternallyCancelledReadStore()
        task = asyncio.create_task(
            RememberKnowledgeTool().run(
                ToolContext(
                    session_id="session_1",
                    idempotency_key="internally-cancelled-read-operation",
                    knowledge_store=store,
                ),
                {"text": "An internal read cancellation does not cancel the caller."},
            )
        )
        result = await task
        return result, task.cancelling(), task.cancelled()

    result, cancelling, cancelled = asyncio.run(run())

    assert result.is_error is False
    assert result.structured["written"] is True
    assert cancelling == 0
    assert cancelled is False


def test_remember_knowledge_entry_read_preserves_real_caller_cancellation() -> None:
    class BlockingReadStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.read_started = asyncio.Event()
            self.release_read = asyncio.Event()
            self.read_finished = asyncio.Event()
            self.publish_calls = 0

        async def get_entry(self, entry_id):
            del entry_id
            self.read_started.set()
            try:
                await self.release_read.wait()
                return None
            finally:
                self.read_finished.set()

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.publish_calls += 1
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

    async def run() -> None:
        store = BlockingReadStore()
        tool = RememberKnowledgeTool()
        task = asyncio.create_task(
            tool.run(
                ToolContext(
                    session_id="session_1",
                    idempotency_key="caller-cancelled-read-operation",
                    knowledge_store=store,
                ),
                {"text": "Caller cancellation stays authoritative."},
            )
        )
        await asyncio.wait_for(store.read_started.wait(), timeout=2)
        task.cancel("authoritative caller cancellation")
        with pytest.raises(asyncio.CancelledError, match="authoritative caller cancellation"):
            await task
        assert task.cancelling() == 1
        assert task.cancelled() is True
        assert store.publish_calls == 0
        assert store.read_finished.is_set() is False
        assert len(tool._read_operations) == 1
        store.release_read.set()
        await asyncio.wait_for(store.read_finished.wait(), timeout=1)
        while tool._read_operations:
            await asyncio.sleep(0)
        assert len(tool._read_operations) == 0

    asyncio.run(run())


def test_remember_knowledge_cancellation_during_post_commit_confirmation_reconciles() -> None:
    class ConfirmationBarrierStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.confirmation_started = asyncio.Event()
            self.release_confirmation = asyncio.Event()
            self.reconciliation_started = asyncio.Event()
            self.release_reconciliation = asyncio.Event()
            self.reconciliation_completed = asyncio.Event()

        async def load_entry_publication_receipt(self, operation_id):
            receipt = await super().load_entry_publication_receipt(operation_id)
            if receipt is None:
                return None
            if not self.confirmation_started.is_set():
                self.confirmation_started.set()
                await self.release_confirmation.wait()
            self.reconciliation_started.set()
            await self.release_reconciliation.wait()
            self.reconciliation_completed.set()
            return receipt

    async def run() -> None:
        store = ConfirmationBarrierStore()
        operation_id = "post-commit-confirmation-cancellation"
        task = asyncio.create_task(
            RememberKnowledgeTool().run(
                ToolContext(
                    session_id="session_1",
                    idempotency_key=operation_id,
                    knowledge_store=store,
                ),
                {"text": "Post-commit cancellation still reconciles the receipt."},
            )
        )
        await asyncio.wait_for(store.confirmation_started.wait(), timeout=2)
        task.cancel("caller cancelled during receipt confirmation")
        with pytest.raises(
            asyncio.CancelledError,
            match="caller cancelled during receipt confirmation",
        ):
            await asyncio.wait_for(task, timeout=2)
        assert task.cancelling() == 1
        assert task.cancelled() is True
        # Cancellation stops waiting but the retained publication remains its
        # sole owner and finishes receipt reconciliation in the background.
        store.release_confirmation.set()
        await asyncio.wait_for(store.reconciliation_started.wait(), timeout=2)
        store.release_reconciliation.set()
        await asyncio.wait_for(store.reconciliation_completed.wait(), timeout=2)
        assert await store.load_entry_publication_receipt(operation_id) is not None

    asyncio.run(run())


@pytest.mark.parametrize("commit_before_release", [False, True])
def test_remember_knowledge_cancellation_retains_opaque_dispatch_and_preserves_retry(
    commit_before_release: bool,
) -> None:
    class OpaquePublicationStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.dispatched = threading.Event()
            self.release = threading.Event()
            self.publish_calls = 0
            self.retry_receipt_reads = 0
            self.block_retry_reads = False

        async def load_entry_publication_receipt(self, operation_id):
            if self.block_retry_reads and self.dispatched.is_set() and not self.release.is_set():
                self.retry_receipt_reads += 1
                await asyncio.to_thread(self.release.wait)
            return await super().load_entry_publication_receipt(operation_id)

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.publish_calls += 1
            if commit_before_release:
                receipt = await super().publish_entry_with_chunks(
                    entry,
                    chunks,
                    operation_id=operation_id,
                )
            await asyncio.to_thread(self._blocking_dispatch)
            if commit_before_release:
                return receipt
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

        def _blocking_dispatch(self) -> None:
            self.dispatched.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test publication barrier timed out")

    async def run():
        store = OpaquePublicationStore()
        ctx = ToolContext(
            session_id="session_1",
            idempotency_key="cancelled-remember-operation",
            knowledge_store=store,
        )
        tool = RememberKnowledgeTool()
        task = asyncio.create_task(tool.run(ctx, {"text": "Committed before caller cancellation."}))
        assert await asyncio.to_thread(store.dispatched.wait, 2)
        task.cancel("caller cancelled")
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await asyncio.wait_for(task, timeout=2)
        assert task.cancelling() == 1
        assert task.cancelled() is True
        early_receipt = await store.load_entry_publication_receipt("cancelled-remember-operation")
        assert (early_receipt is not None) is commit_before_release
        store.block_retry_reads = True
        conflict = await tool.run(
            ctx,
            {"text": "Different material cannot reuse the active operation identity."},
        )
        assert conflict.is_error is True
        assert conflict.structured["outcome"] == "operation_conflict"
        assert store.publish_calls == 1
        retry = asyncio.create_task(
            tool.run(ctx, {"text": "Committed before caller cancellation."})
        )
        await asyncio.sleep(0)
        assert retry.done() is False
        assert store.publish_calls == 1
        store.release.set()
        replay = await asyncio.wait_for(retry, timeout=2)
        receipt = await store.load_entry_publication_receipt("cancelled-remember-operation")
        return receipt, replay, store.publish_calls, store.retry_receipt_reads

    receipt, replay, publish_calls, retry_receipt_reads = asyncio.run(run())

    assert receipt is not None
    assert publish_calls == 1
    assert retry_receipt_reads == 0
    assert replay.is_error is False
    assert replay.structured["written"] is True
    assert replay.structured["already_known"] is False
    assert "publication_replayed" not in replay.structured


def test_remember_knowledge_workspace_identity_is_consistent_across_commit() -> None:
    class BarrierPublicationStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.publish_started = asyncio.Event()
            self.release_publish = asyncio.Event()
            self.publish_calls = 0

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.publish_calls += 1
            self.publish_started.set()
            await self.release_publish.wait()
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

    async def run():
        store = BarrierPublicationStore()
        tool = RememberKnowledgeTool()
        context_a = ToolContext(
            session_id="session_1",
            idempotency_key="workspace-authority-operation",
            workspace_id="workspace-a",
            knowledge_store=store,
        )
        context_b = context_a.model_copy(update={"workspace_id": "workspace-b"})
        arguments = {"text": "Workspace provenance is publication authority."}

        first = asyncio.create_task(tool.run(context_a, arguments))
        await asyncio.wait_for(store.publish_started.wait(), timeout=2)
        in_flight_conflict = await tool.run(context_b, arguments)
        store.release_publish.set()
        written = await asyncio.wait_for(first, timeout=2)
        await asyncio.sleep(0)
        settled_conflict = await tool.run(context_b, arguments)
        receipt = await store.load_entry_publication_receipt("workspace-authority-operation")
        assert receipt is not None
        entry = await store.get_entry(receipt.entry_id)
        return written, in_flight_conflict, settled_conflict, entry, store.publish_calls

    written, in_flight_conflict, settled_conflict, entry, publish_calls = asyncio.run(run())

    assert written.is_error is False
    assert in_flight_conflict.is_error is True
    assert in_flight_conflict.structured["outcome"] == "operation_conflict"
    assert settled_conflict.is_error is True
    assert settled_conflict.structured["outcome"] == "receipt_conflict"
    assert entry is not None
    assert entry.metadata["workspace_id"] == "workspace-a"
    assert publish_calls == 1


def test_remember_knowledge_capacity_allows_receipt_reconciliation_but_blocks_new_dispatch(
    monkeypatch,
) -> None:
    class SelectiveBarrierStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.active_started = asyncio.Event()
            self.release_active = asyncio.Event()
            self.published_operations: list[str] = []

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.published_operations.append(operation_id)
            if operation_id == "capacity-active-operation":
                self.active_started.set()
                await self.release_active.wait()
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

    async def run():
        store = SelectiveBarrierStore()
        replay_context = ToolContext(
            session_id="session_1",
            idempotency_key="capacity-replay-operation",
            workspace_id="workspace-1",
            knowledge_store=store,
        )
        replay_arguments = {"text": "This publication already committed."}
        seeded = await RememberKnowledgeTool().run(replay_context, replay_arguments)
        assert seeded.is_error is False

        monkeypatch.setattr(
            knowledge_module,
            "MAX_RETAINED_REMEMBER_KNOWLEDGE_PUBLICATIONS",
            1,
        )
        tool = RememberKnowledgeTool()
        active_context = replay_context.model_copy(
            update={"idempotency_key": "capacity-active-operation"}
        )
        active = asyncio.create_task(
            tool.run(active_context, {"text": "This publication is still active."})
        )
        await asyncio.wait_for(store.active_started.wait(), timeout=2)

        replayed = await tool.run(replay_context, replay_arguments)
        rejected = await tool.run(
            replay_context.model_copy(update={"idempotency_key": "capacity-new-operation"}),
            {"text": "This would require another publication."},
        )
        store.release_active.set()
        active_result = await asyncio.wait_for(active, timeout=2)
        return replayed, rejected, active_result, store.published_operations

    replayed, rejected, active_result, published_operations = asyncio.run(run())

    assert replayed.is_error is False
    assert replayed.structured["publication_replayed"] is True
    assert rejected.is_error is True
    assert rejected.structured["outcome"] == "publication_capacity_exhausted"
    assert active_result.is_error is False
    assert published_operations == [
        "capacity-replay-operation",
        "capacity-active-operation",
    ]


def test_remember_knowledge_cancelled_waiters_share_one_retained_publication() -> None:
    class ReconciliationBarrierStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.publish_started = asyncio.Event()
            self.release_publish = asyncio.Event()
            self.reconciliation_started = asyncio.Event()
            self.release_reconciliation = asyncio.Event()
            self.reconciliation_completed = asyncio.Event()

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.publish_started.set()
            await self.release_publish.wait()
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

        async def load_entry_publication_receipt(self, operation_id):
            receipt = await super().load_entry_publication_receipt(operation_id)
            if receipt is not None:
                self.reconciliation_started.set()
                await self.release_reconciliation.wait()
                self.reconciliation_completed.set()
            return receipt

    async def run() -> None:
        store = ReconciliationBarrierStore()
        context = ToolContext(
            session_id="session_1",
            idempotency_key="repeated-cancellation-operation",
            knowledge_store=store,
        )
        tool = RememberKnowledgeTool()
        first = asyncio.create_task(
            tool.run(
                context,
                {"text": "Receipt reconciliation survives repeated caller cancellation."},
            )
        )
        await asyncio.wait_for(store.publish_started.wait(), timeout=2)
        second = asyncio.create_task(
            tool.run(
                context,
                {"text": "Receipt reconciliation survives repeated caller cancellation."},
            )
        )
        await asyncio.sleep(0)
        first.cancel("first cancellation")
        second.cancel("second cancellation")
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await asyncio.wait_for(first, timeout=2)
        with pytest.raises(asyncio.CancelledError, match="second cancellation"):
            await asyncio.wait_for(second, timeout=2)
        assert first.cancelling() == 1
        assert second.cancelling() == 1
        assert first.cancelled() is True
        assert second.cancelled() is True
        store.release_publish.set()
        await asyncio.wait_for(store.reconciliation_started.wait(), timeout=2)
        store.release_reconciliation.set()
        await asyncio.wait_for(store.reconciliation_completed.wait(), timeout=2)
        receipt = await store.load_entry_publication_receipt("repeated-cancellation-operation")
        assert receipt is not None

    asyncio.run(run())


def test_remember_knowledge_reconciles_grouped_store_failure() -> None:
    class GroupedFailureStore(_TestKnowledgeStore):
        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )
            raise BaseExceptionGroup(
                "extension failures",
                [
                    asyncio.CancelledError("child cancellation"),
                    RuntimeError("private grouped failure canary"),
                ],
            )

    async def run():
        store = GroupedFailureStore()
        result = await RememberKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                idempotency_key="grouped-failure-operation",
                knowledge_store=store,
            ),
            {"text": "Grouped extension failures still reconcile committed knowledge."},
        )
        receipt = await store.load_entry_publication_receipt("grouped-failure-operation")
        return result, receipt

    result, receipt = asyncio.run(run())

    assert result.is_error is False
    assert result.structured["post_write_error"] == "publication_acknowledgement_lost"
    assert "canary" not in result.content
    assert "canary" not in repr(result.structured)
    assert receipt is not None


def test_remember_knowledge_keeps_caller_cancellation_authoritative_over_grouped_failure() -> None:
    class GroupedFailureBarrierStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.committed = asyncio.Event()
            self.release_failure = asyncio.Event()

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )
            self.committed.set()
            await self.release_failure.wait()
            raise BaseExceptionGroup(
                "extension failures",
                [
                    asyncio.CancelledError("child cancellation"),
                    RuntimeError("private grouped cancellation canary"),
                ],
            )

    async def run() -> None:
        store = GroupedFailureBarrierStore()
        operation_id = "cancelled-grouped-failure-operation"
        task = asyncio.create_task(
            RememberKnowledgeTool().run(
                ToolContext(
                    session_id="session_1",
                    idempotency_key=operation_id,
                    knowledge_store=store,
                ),
                {"text": "Caller cancellation remains authoritative after commit."},
            )
        )
        await asyncio.wait_for(store.committed.wait(), timeout=2)
        task.cancel("authoritative caller cancellation")
        store.release_failure.set()
        with pytest.raises(asyncio.CancelledError, match="authoritative caller cancellation"):
            await task
        assert task.cancelling() == 1
        assert task.cancelled() is True
        assert await store.load_entry_publication_receipt(operation_id) is not None

    asyncio.run(run())


def test_remember_knowledge_bounds_grouped_receipt_lookup_failure() -> None:
    class GroupedReceiptFailureStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.publish_calls = 0

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.publish_calls += 1
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

        async def load_entry_publication_receipt(self, operation_id):
            raise BaseExceptionGroup(
                "receipt failures",
                [
                    asyncio.CancelledError("child cancellation"),
                    RuntimeError("private receipt failure canary"),
                ],
            )

    async def run():
        store = GroupedReceiptFailureStore()
        result = await RememberKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                idempotency_key="grouped-receipt-operation",
                knowledge_store=store,
            ),
            {"text": "Receipt failures remain bounded."},
        )
        return result, store.publish_calls

    result, publish_calls = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {
        "error": "knowledge_write_failed",
        "outcome": "receipt_read_failed",
        "cleanup": "not_attempted_unowned",
    }
    assert "canary" not in result.content
    assert "canary" not in repr(result.structured)
    assert publish_calls == 0


@pytest.mark.parametrize(
    "conflicting_arguments",
    [
        {"text": "Different knowledge."},
        {"text": "Original knowledge.", "title": "Different title"},
    ],
)
def test_remember_knowledge_rejects_reused_operation_with_different_content(
    conflicting_arguments: dict[str, str],
) -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        ctx = ToolContext(
            session_id="session_1",
            idempotency_key="conflicting-remember-operation",
            knowledge_store=store,
        )
        first = await RememberKnowledgeTool().run(ctx, {"text": "Original knowledge."})
        conflict = await RememberKnowledgeTool().run(ctx, conflicting_arguments)
        original_id = first.structured["entry"]["entry_id"]
        return first, conflict, await store.get_entry(original_id)

    first, conflict, original = asyncio.run(run())

    assert first.is_error is False
    assert conflict.is_error is True
    assert conflict.structured["outcome"] == "receipt_conflict"
    assert conflict.structured["cleanup"] == "not_attempted_unowned"
    assert original is not None
    assert original.text == "Original knowledge."


def test_remember_knowledge_concurrent_exact_operation_reconciles_one_receipt() -> None:
    class BarrierPublicationStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.arrivals = 0
            self.ready = asyncio.Event()

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.arrivals += 1
            if self.arrivals == 2:
                self.ready.set()
            await self.ready.wait()
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

    async def run():
        store = BarrierPublicationStore()
        ctx = ToolContext(
            session_id="session_1",
            idempotency_key="concurrent-exact-operation",
            knowledge_store=store,
        )
        results = await asyncio.gather(
            RememberKnowledgeTool().run(ctx, {"text": "One exact concurrent operation."}),
            RememberKnowledgeTool().run(ctx, {"text": "One exact concurrent operation."}),
        )
        receipt = await store.load_entry_publication_receipt("concurrent-exact-operation")
        return results, receipt

    results, receipt = asyncio.run(run())

    assert receipt is not None
    assert all(result.is_error is False for result in results)
    assert sorted(result.structured["written"] for result in results) == [False, True]
    assert sum(result.structured.get("publication_replayed") is True for result in results) == 1
    assert sum(result.structured["already_known"] is False for result in results) == 1
    assert sum(result.structured["already_known"] is None for result in results) == 1
    assert {result.structured["entry"]["entry_id"] for result in results} == {receipt.entry_id}


def test_remember_knowledge_concurrent_exact_operation_converges_after_id_collision() -> None:
    class BarrierPublicationStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.arrivals = 0
            self.ready = asyncio.Event()
            self.masked_entry_id: str | None = None
            self.masked_reads = 0

        async def get_entry(self, entry_id):
            if entry_id == self.masked_entry_id and self.masked_reads == 0:
                # Give the two exact deliveries different pre-publication
                # snapshots: one sees the identity as free while the other sees
                # the incompatible occupant.
                self.masked_reads += 1
                return None
            return await super().get_entry(entry_id)

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.arrivals += 1
            if self.arrivals == 2:
                self.ready.set()
            await self.ready.wait()
            return await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )

    async def run():
        text = "Exact concurrent operations survive a deterministic-id collision."
        source_hash = knowledge_source_hash(text)
        deterministic_id = content_knowledge_entry_id(
            namespace="default",
            kind="fact",
            source_hash=source_hash,
        )
        occupant_text = "Unrelated material occupying the content-derived identity."
        store = BarrierPublicationStore()
        await store.put_entry(
            KnowledgeEntry(
                id=deterministic_id,
                text=occupant_text,
                source_hash=knowledge_source_hash(occupant_text),
                created_by="unrelated-writer",
            )
        )
        store.masked_entry_id = deterministic_id
        operation_id = "concurrent-collision-operation"
        context = ToolContext(
            session_id="session_1",
            idempotency_key=operation_id,
            knowledge_store=store,
        )
        results = await asyncio.gather(
            RememberKnowledgeTool().run(context, {"text": text}),
            RememberKnowledgeTool().run(context, {"text": text}),
        )
        receipt = await store.load_entry_publication_receipt(operation_id)
        occupant = await store.get_entry(deterministic_id)
        return results, receipt, occupant

    results, receipt, occupant = asyncio.run(run())

    assert receipt is not None
    assert all(result.is_error is False for result in results)
    assert {result.structured["entry"]["entry_id"] for result in results} == {receipt.entry_id}
    assert receipt.entry_id != occupant.id
    assert occupant.text == "Unrelated material occupying the content-derived identity."
    assert occupant.created_by == "unrelated-writer"


@pytest.mark.parametrize("later_state", ["archived", "deleted", "replaced"])
def test_remember_knowledge_exact_retry_reports_only_historical_commit(
    later_state: str,
) -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        ctx = ToolContext(
            session_id="session_1",
            idempotency_key="reviewed-remember-operation",
            knowledge_store=store,
        )
        first = await RememberKnowledgeTool().run(ctx, {"text": "Review this knowledge."})
        entry_id = first.structured["entry"]["entry_id"]
        if later_state == "archived":
            await store.update_entry_status(entry_id, KnowledgeStatus.ARCHIVED)
        elif later_state == "deleted":
            await store.delete_entry(entry_id, hard=True)
        else:
            await store.delete_entry(entry_id, hard=True)
            replacement_text = "A different publication now occupies this entry identity."
            await store.put_entry(
                KnowledgeEntry(
                    id=entry_id,
                    text=replacement_text,
                    source_hash=knowledge_source_hash(replacement_text),
                    status=KnowledgeStatus.ACTIVE,
                )
            )
        replay = await RememberKnowledgeTool().run(ctx, {"text": "Review this knowledge."})
        return first, replay, await store.get_entry(entry_id)

    first, replay, current = asyncio.run(run())

    assert first.is_error is False
    assert replay.is_error is False
    assert replay.content.endswith("Its current lifecycle state was not checked.")
    assert replay.structured["entry"] == {"entry_id": first.structured["entry"]["entry_id"]}
    assert replay.structured["written"] is False
    assert replay.structured["already_known"] is None
    assert replay.structured["publication_replayed"] is True
    assert replay.structured["status"] is None
    if later_state == "archived":
        assert current is not None
        assert current.status is KnowledgeStatus.ARCHIVED
    elif later_state == "deleted":
        assert current is None
    else:
        assert current is not None
        assert current.status is KnowledgeStatus.ACTIVE
        assert current.text == "A different publication now occupies this entry identity."


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
            access_scope=_ACCESS_SCOPE,
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
            access_scope=_ACCESS_SCOPE,
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
                text="For SendGrid, prefer a trusted email delivery configuration.",
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
    assert result.structured["filtered_hits"] == 0
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["remote_git_credentials"]
    assert result.structured["hits"][0]["score_kind"] == "inmemory_hybrid"


def test_search_knowledge_auto_min_score_zero_keeps_weak_semantic_neighbors() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
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


def test_search_knowledge_rejects_nan_auto_min_score() -> None:
    with pytest.raises(ValueError, match="auto_min_score"):
        SearchKnowledgeTool(auto_min_score=float("nan"))


def test_search_knowledge_auto_min_score_preserves_unscored_keyword_hits() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
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
    assert result.structured["min_score_applied"] is True
    hit_by_id = {hit["entry_id"]: hit for hit in result.structured["hits"]}
    assert set(hit_by_id) == {"remote_git_credentials", "runbook_keyword_hit"}
    assert hit_by_id["remote_git_credentials"]["score_normalized"] == 1.0
    assert hit_by_id["runbook_keyword_hit"]["score_normalized"] is None
    assert "hybrid keyword match" in hit_by_id["runbook_keyword_hit"]["reason"]


def test_search_knowledge_default_rejects_score_override_argument() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        return await SearchKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert "requires `text`, `any_terms`, `all_terms`, or `phrases`" in result.content


def test_search_knowledge_caps_model_facing_preview_per_hit() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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


@pytest.mark.parametrize("store_max_bytes", [16, 10_000])
def test_search_knowledge_redacts_preview_before_every_byte_bound(
    store_max_bytes: int,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="secret",
                text=secret,
            )
        )
        return await SearchKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=store,
                invocation_secret_redactor=lambda: SecretRedactor(secret),
            ),
            {
                "query": "workload",
                "preview_bytes": 16,
                "max_bytes": store_max_bytes,
            },
        )

    result = asyncio.run(run())

    rendered = repr(result.model_dump(mode="json"))
    assert secret not in rendered
    assert secret[:16] not in rendered
    assert result.structured["hits"][0]["text_preview_truncated"] is True


def test_search_knowledge_retries_when_secret_resolves_during_store_read() -> None:
    secret = "knowledge-search-race-secret-canary-ABCDEFGHIJKLMNOP"

    async def run():
        search_started = asyncio.Event()
        allow_search = asyncio.Event()

        class BlockingSearchStore(_TestKnowledgeStore):
            search_calls = 0

            async def search(self, query: KnowledgeQuery):
                self.search_calls += 1
                if self.search_calls == 1:
                    search_started.set()
                    await allow_search.wait()
                return await super().search(query)

        store = BlockingSearchStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="secret",
                text=secret,
            )
        )
        tracker = InvocationSecretTracker(SecretRedactor())
        task = asyncio.create_task(
            SearchKnowledgeTool().run(
                ToolContext(
                    session_id="session_search_revision",
                    knowledge_store=store,
                    invocation_secret_redactor=lambda: tracker.redactor,
                    invocation_secret_snapshot_provider=tracker.snapshot,
                    invocation_secret_capture_observer=(tracker.record_ambiguous_output_capture),
                ),
                {
                    "query": "knowledge",
                    "preview_bytes": 16,
                    "max_bytes": 16,
                },
            )
        )
        await search_started.wait()
        tracker.record(
            ResolvedSecret(
                name="token",
                value=SecretStr(secret),
            )
        )
        allow_search.set()
        result = await task
        return result, store.search_calls, tracker.seal_for_publication()

    result, search_calls, publication = asyncio.run(run())

    rendered = repr(result.model_dump(mode="json"))
    assert search_calls == 2
    assert publication.unsafe_output is False
    assert secret not in rendered
    assert secret[:16] not in rendered


def test_search_source_bounded_capture_fails_closed_when_secret_resolves_before_publication() -> (
    None
):
    secret = "knowledge-search-late-secret-canary-ABCDEFGHIJKLMNOP"

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(entry_id="secret", text=secret)
        )
        tracker = InvocationSecretTracker(SecretRedactor())
        ctx = ToolContext(
            session_id="session_search_late_publication",
            knowledge_store=store,
            invocation_secret_redactor=lambda: tracker.redactor,
            invocation_secret_snapshot_provider=tracker.snapshot,
            invocation_secret_capture_observer=tracker.record_ambiguous_output_capture,
        )

        class SearchThenResolveTool(Tool):
            spec = ToolSpec(
                name="search_then_resolve",
                description="Exercise the bounded search publication boundary.",
                input_schema={"type": "object", "properties": {}},
            )

            async def run(self, tool_ctx: ToolContext, args: dict) -> ToolResult:
                del args
                result = await SearchKnowledgeTool().run(
                    tool_ctx,
                    {
                        "query": "knowledge",
                        "preview_bytes": 100,
                        "max_bytes": 16,
                    },
                )
                tracker.record(ResolvedSecret(name="token", value=SecretStr(secret)))
                return result

        return await run_tool(
            tool=SearchThenResolveTool(),
            effect=ToolEffect.NONE,
            ctx=ctx,
            arguments={},
            redactor=lambda: tracker.redactor,
            finalize_publication=tracker.seal_for_publication,
        )

    outcome = asyncio.run(run())

    rendered = repr(outcome)
    assert outcome.result.is_error is True
    assert outcome.result.structured["terminal_outcome"] == "invalid_tool_output"
    assert secret not in rendered
    assert secret[:16] not in rendered


def test_search_preview_completeness_uses_the_selected_authoritative_field() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.put_entry_with_chunks(
            KnowledgeEntry(id="secret", text=secret[:16]),
            [
                KnowledgeChunk(
                    id="secret:0",
                    entry_id="secret",
                    text=secret,
                    chunk_index=0,
                )
            ],
        )
        return await SearchKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=store,
                invocation_secret_redactor=lambda: SecretRedactor(secret),
            ),
            {
                "query": "ABCDEFGHIJKLMNOP",
                "preview_bytes": 100,
                "max_bytes": 16,
            },
        )

    result = asyncio.run(run())

    rendered = repr(result.model_dump(mode="json"))
    assert secret not in rendered
    assert secret[:16] not in rendered
    assert result.structured["hits"][0]["text_preview_truncated"] is True


def test_search_result_limit_does_not_mark_complete_preview_as_truncated() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        indexer = KnowledgeIndexer(store)
        for entry_id in ("first", "second"):
            await indexer.index_text(
                KnowledgeIndexRequest(
                    entry_id=entry_id,
                    text="needle complete preview a",
                )
            )
        return await SearchKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=store,
                invocation_secret_redactor=lambda: SecretRedactor("abc-secret"),
            ),
            {
                "query": "needle",
                "limit": 1,
                "preview_bytes": 100,
                "max_bytes": 10_000,
            },
        )

    result = asyncio.run(run())

    assert result.structured["truncated"] is True
    assert result.structured["hits"][0]["text_preview"] == "needle complete preview a"
    assert result.structured["hits"][0]["text_preview_truncated"] is False
    assert "[preview truncated]" not in result.content


def test_list_knowledge_discovers_entries_and_facets() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
            access_scope=_ACCESS_SCOPE,
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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


@pytest.mark.parametrize("store_max_bytes", [16, 10_000])
def test_list_knowledge_redacts_preview_before_every_byte_bound(
    store_max_bytes: int,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="secret",
                text=secret,
            )
        )
        return await ListKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=store,
                invocation_secret_redactor=lambda: SecretRedactor(secret),
            ),
            {
                "preview_bytes": 16,
                "max_bytes": store_max_bytes,
            },
        )

    result = asyncio.run(run())

    rendered = repr(result.model_dump(mode="json"))
    assert secret not in rendered
    assert secret[:16] not in rendered
    assert result.structured["entries"][0]["text_preview_truncated"] is True


def test_list_knowledge_retries_when_secret_resolves_during_store_read() -> None:
    secret = "knowledge-list-race-secret-canary-ABCDEFGHIJKLMNOP"

    async def run():
        list_started = asyncio.Event()
        allow_list = asyncio.Event()

        class BlockingListStore(_TestKnowledgeStore):
            list_calls = 0

            async def list_entries(self, query):
                self.list_calls += 1
                if self.list_calls == 1:
                    list_started.set()
                    await allow_list.wait()
                return await super().list_entries(query)

        store = BlockingListStore()
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="secret",
                text=secret,
            )
        )
        tracker = InvocationSecretTracker(SecretRedactor())
        task = asyncio.create_task(
            ListKnowledgeTool().run(
                ToolContext(
                    session_id="session_list_revision",
                    knowledge_store=store,
                    invocation_secret_redactor=lambda: tracker.redactor,
                    invocation_secret_snapshot_provider=tracker.snapshot,
                    invocation_secret_capture_observer=(tracker.record_ambiguous_output_capture),
                ),
                {
                    "preview_bytes": 16,
                    "max_bytes": 16,
                },
            )
        )
        await list_started.wait()
        tracker.record(
            ResolvedSecret(
                name="token",
                value=SecretStr(secret),
            )
        )
        allow_list.set()
        result = await task
        return result, store.list_calls, tracker.seal_for_publication()

    result, list_calls, publication = asyncio.run(run())

    rendered = repr(result.model_dump(mode="json"))
    assert list_calls == 2
    assert publication.unsafe_output is False
    assert secret not in rendered
    assert secret[:16] not in rendered


def test_list_source_bounded_capture_fails_closed_when_secret_resolves_before_publication() -> None:
    secret = "knowledge-list-late-secret-canary-ABCDEFGHIJKLMNOP"

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(entry_id="secret", text=secret)
        )
        tracker = InvocationSecretTracker(SecretRedactor())
        ctx = ToolContext(
            session_id="session_list_late_publication",
            knowledge_store=store,
            invocation_secret_redactor=lambda: tracker.redactor,
            invocation_secret_snapshot_provider=tracker.snapshot,
            invocation_secret_capture_observer=tracker.record_ambiguous_output_capture,
        )

        class ListThenResolveTool(Tool):
            spec = ToolSpec(
                name="list_then_resolve",
                description="Exercise the bounded list publication boundary.",
                input_schema={"type": "object", "properties": {}},
            )

            async def run(self, tool_ctx: ToolContext, args: dict) -> ToolResult:
                del args
                result = await ListKnowledgeTool().run(
                    tool_ctx,
                    {
                        "preview_bytes": 100,
                        "max_bytes": 16,
                    },
                )
                tracker.record(ResolvedSecret(name="token", value=SecretStr(secret)))
                return result

        return await run_tool(
            tool=ListThenResolveTool(),
            effect=ToolEffect.NONE,
            ctx=ctx,
            arguments={},
            redactor=lambda: tracker.redactor,
            finalize_publication=tracker.seal_for_publication,
        )

    outcome = asyncio.run(run())

    rendered = repr(outcome)
    assert outcome.result.is_error is True
    assert outcome.result.structured["terminal_outcome"] == "invalid_tool_output"
    assert secret not in rendered
    assert secret[:16] not in rendered


def test_list_result_limit_does_not_mark_complete_preview_as_truncated() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        indexer = KnowledgeIndexer(store)
        for entry_id in ("first", "second"):
            await indexer.index_text(
                KnowledgeIndexRequest(
                    entry_id=entry_id,
                    text="complete listed preview a",
                )
            )
        return await ListKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=store,
                invocation_secret_redactor=lambda: SecretRedactor("abc-secret"),
            ),
            {
                "limit": 1,
                "preview_bytes": 100,
                "max_bytes": 10_000,
            },
        )

    result = asyncio.run(run())

    assert result.structured["truncated"] is True
    assert result.structured["entries"][0]["text_preview"] == "complete listed preview a"
    assert result.structured["entries"][0]["text_preview_truncated"] is False
    assert "[preview truncated]" not in result.content


def test_search_knowledge_delegates_mode_support_to_store() -> None:
    async def run():
        return await SearchKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            ),
            {"query": "refund policy", "mode": "semantic"},
        )

    with pytest.raises(ValueError, match="InMemoryKnowledgeStore supports only"):
        asyncio.run(run())


def test_read_knowledge_returns_bounded_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
            ToolContext(
                session_id="session_1",
                knowledge_store=InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            ),
            {"entry_id": 123},
        )

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "entry_id" in result.content

    result = asyncio.run(
        ReadKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            ),
            {"entry_id": " policy "},
        )
    )
    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "must not start or end with whitespace" in result.content


def test_read_knowledge_returns_tool_error_for_invalid_window() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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


def test_read_knowledge_preserves_operational_value_error() -> None:
    class FailingReadStore(_TestKnowledgeStore):
        async def read_chunks(self, entry_id: str, **kwargs):
            raise ValueError("knowledge backend state is invalid")

    async def run():
        return await ReadKnowledgeTool().run(
            ToolContext(session_id="session_1", knowledge_store=FailingReadStore()),
            {"entry_id": "policy"},
        )

    with pytest.raises(ValueError, match="knowledge backend state is invalid"):
        asyncio.run(run())


class ReadOnlyKnowledgeStore:
    """Minimal reader that only exposes the read-path store methods."""

    def __init__(self) -> None:
        self._store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)

    async def index(self, request: KnowledgeIndexRequest) -> None:
        await KnowledgeIndexer(self._store).index_text(request)

    async def search(self, query: KnowledgeQuery, *, access_scope=None):
        return await self._store.search(query, access_scope=access_scope)

    async def list_entries(self, query, *, access_scope=None):
        return await self._store.list_entries(query, access_scope=access_scope)

    async def read_chunks(self, entry_id: str, *, access_scope=None, **kwargs):
        return await self._store.read_chunks(
            entry_id,
            access_scope=access_scope,
            **kwargs,
        )


def test_read_tools_accept_read_only_knowledge_store() -> None:
    async def run():
        store = ReadOnlyKnowledgeStore()
        await store.index(
            KnowledgeIndexRequest(entry_id="policy", text="Always verify bank details.")
        )
        ctx = ToolContext(
            session_id="session_1",
            knowledge_store=store,
            knowledge_access_scope=_ACCESS_SCOPE,
        )
        search_result = await SearchKnowledgeTool().run(ctx, {"query": "bank details"})
        list_result = await ListKnowledgeTool().run(ctx, {})
        read_result = await ReadKnowledgeTool().run(ctx, {"entry_id": "policy"})
        return search_result, list_result, read_result

    search_result, list_result, read_result = asyncio.run(run())

    assert search_result.is_error is False
    assert search_result.structured is not None
    assert [hit["entry_id"] for hit in search_result.structured["hits"]] == ["policy"]
    assert list_result.is_error is False
    assert read_result.is_error is False


def test_knowledge_tools_name_missing_store_methods() -> None:
    class SearchOnlyStore:
        async def search(self, query):
            raise AssertionError("not reached")

    # A store missing the read surface is rejected when the context is built.
    with pytest.raises(ValidationError, match="KnowledgeStoreHandle"):
        ToolContext(session_id="session_1", knowledge_store=SearchOnlyStore())

    class ReadOnlyStore:
        async def search(self, query):
            raise AssertionError("not reached")

        async def list_entries(self, query):
            raise AssertionError("not reached")

        async def read_chunks(self, entry_id, **kwargs):
            raise AssertionError("not reached")

    ctx = ToolContext(session_id="session_1", knowledge_store=ReadOnlyStore())

    # Write-path tools name the owned-publication methods a read-only store lacks.
    with pytest.raises(
        TypeError,
        match=(
            r"remember_knowledge: get_entry, publish_entry_with_chunks, "
            r"load_entry_publication_receipt"
        ),
    ):
        asyncio.run(RememberKnowledgeTool().run(ctx, {"text": "fact"}))


def test_search_knowledge_forwards_min_score_in_store_query() -> None:
    received: list[float | None] = []

    class RecordingStore(InMemoryEmbeddingKnowledgeStore):
        async def search(self, query: KnowledgeQuery):
            received.append(query.min_score)
            return await super().search(query)

    async def run():
        store = RecordingStore(
            access_scope=_ACCESS_SCOPE,
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
            {"query": "remote sandbox auth", "namespace": "ops"},
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert received == [0.75]
    assert result.structured["query"]["min_score"] == 0.75
    assert result.structured["min_score"] == 0.75
    assert result.structured["min_score_applied"] is True


def test_search_knowledge_surfaces_threshold_not_applied() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await KnowledgeIndexer(store).index_text(
            KnowledgeIndexRequest(
                entry_id="github",
                text="GitHub push needs a credential broker.",
            )
        )
        return await SearchKnowledgeTool(allow_score_override=True).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"query": "github credential", "min_score": 0.5},
        )

    result = asyncio.run(run())

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["min_score"] == 0.5
    assert result.structured["query"]["min_score"] == 0.5
    assert result.structured["min_score_applied"] is False
    assert "was not applied" in result.content
    assert [hit["entry_id"] for hit in result.structured["hits"]] == ["github"]


def test_search_knowledge_keyword_auto_reports_no_threshold() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
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
    assert result.structured["min_score"] is None
    assert result.structured["query"]["min_score"] is None
    assert result.structured["min_score_applied"] is None
    assert "was not applied" not in result.content


def test_remember_knowledge_dedupes_identical_text() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        ctx = ToolContext(session_id="session_1", knowledge_store=store)
        args = {"text": "Always verify bank details before paying invoices."}
        first = await RememberKnowledgeTool().run(ctx, args)
        second = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_2", knowledge_store=store),
            args,
        )
        assert first.structured is not None
        chunks = await store.read_chunks(
            first.structured["entry"]["entry_id"], max_chunks=5, max_bytes=20_000
        )
        return first, second, chunks

    first, second, chunks = asyncio.run(run())

    assert first.is_error is False
    assert first.structured is not None
    assert first.structured["written"] is True
    assert first.structured["already_known"] is False
    assert second.is_error is False
    assert second.structured is not None
    assert second.structured["written"] is False
    assert second.structured["already_known"] is True
    assert second.structured["entry"]["entry_id"] == first.structured["entry"]["entry_id"]
    assert second.structured["source_hash"] == first.structured["source_hash"]
    assert "already known" in second.content
    assert len(chunks) == 1


@pytest.mark.parametrize(
    "stored_updates",
    [
        {"text": "Different material with a copied source hash."},
        {"labels": {"tenant": "other"}},
        {"visibility": KnowledgeVisibility.USER},
    ],
)
def test_remember_knowledge_fails_closed_on_incompatible_existing_material_or_scope(
    stored_updates: dict[str, Any],
) -> None:
    async def run():
        text = "Policy-scoped knowledge must match before deduplication."
        source_hash = knowledge_source_hash(text)
        entry_id = content_knowledge_entry_id(
            namespace="default",
            kind="fact",
            source_hash=source_hash,
        )
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        existing = KnowledgeEntry(
            id=entry_id,
            text=text,
            source_hash=source_hash,
            labels={"tenant": "expected"},
            visibility=KnowledgeVisibility.GLOBAL,
            status=KnowledgeStatus.PENDING,
        ).model_copy(update=stored_updates)
        existing = KnowledgeEntry.model_validate(existing.model_dump())
        await store.put_entry(existing)
        result = await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(require_labels={"tenant": "expected"})
        ).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": text},
        )
        return result, existing, await store.get_entry(entry_id)

    result, existing, stored = asyncio.run(run())

    assert result.is_error is True
    assert result.structured == {
        "error": "knowledge_write_failed",
        "outcome": "publication_conflict",
        "cleanup": "not_attempted_unowned",
        "entry_id": existing.id,
    }
    assert stored == existing


def test_remember_knowledge_rejects_incompatible_concurrent_winner() -> None:
    class ConcurrentScopedWinnerStore(_TestKnowledgeStore):
        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            winner = KnowledgeEntry.model_validate(
                entry.model_copy(update={"labels": {"tenant": "other"}}).model_dump()
            )
            await super().put_entry_with_chunks(winner, chunks)
            raise KnowledgePublicationConflict("entry_occupied")

    async def run():
        store = ConcurrentScopedWinnerStore()
        result = await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(require_labels={"tenant": "expected"})
        ).run(
            ToolContext(session_id="session_1", knowledge_store=store),
            {"text": "A concurrent winner must satisfy the active policy scope."},
        )
        assert result.structured is not None
        return result, await store.get_entry(result.structured["entry_id"])

    result, winner = asyncio.run(run())

    assert result.is_error is True
    assert result.structured["outcome"] == "publication_conflict"
    assert result.structured["cleanup"] == "not_attempted_unowned"
    assert winner is not None
    assert winner.labels == {"tenant": "other"}


def test_remember_knowledge_uses_one_policy_snapshot_across_publication() -> None:
    class ConcurrentScopedWinnerStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.publish_started = asyncio.Event()
            self.release_publish = asyncio.Event()

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            self.publish_started.set()
            await self.release_publish.wait()
            winner = KnowledgeEntry.model_validate(
                entry.model_copy(update={"labels": {"tenant": "other"}}).model_dump()
            )
            await super().put_entry_with_chunks(winner, chunks)
            raise KnowledgePublicationConflict("entry_occupied")

    async def run():
        caller_policy = RememberKnowledgePolicy(require_labels={"tenant": "expected"})
        tool = RememberKnowledgeTool(policy=caller_policy)
        store = ConcurrentScopedWinnerStore()
        task = asyncio.create_task(
            tool.run(
                ToolContext(session_id="session_1", knowledge_store=store),
                {"text": "Publication authority must remain stable across awaits."},
            )
        )
        await asyncio.wait_for(store.publish_started.wait(), timeout=2)
        caller_policy.require_labels.clear()
        caller_policy.require_labels["tenant"] = "other"
        tool._policy.require_labels.clear()
        tool._policy.require_labels["tenant"] = "other"
        store.release_publish.set()
        result = await task
        assert result.structured is not None
        winner = await store.get_entry(result.structured["entry_id"])
        return result, winner

    result, winner = asyncio.run(run())

    assert result.is_error is True
    assert result.structured["outcome"] == "publication_conflict"
    assert winner is not None
    assert winner.labels == {"tenant": "other"}


def test_remember_knowledge_revalidates_forged_policy_before_use() -> None:
    forged = RememberKnowledgePolicy().model_copy(update={"require_labels": {"tenant": 7}})

    with pytest.raises(ValidationError):
        RememberKnowledgeTool(policy=forged)


def test_custom_store_can_use_public_knowledge_publication_canonicalizer() -> None:
    class PublicCanonicalizerStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self._custom_publication_lock = asyncio.Lock()
            self._custom_receipts: dict[str, KnowledgePublicationReceipt] = {}

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            operation_id, entry, chunks, request_sha256 = prepare_knowledge_publication(
                entry,
                chunks,
                operation_id=operation_id,
            )
            async with self._custom_publication_lock:
                receipt = self._custom_receipts.get(operation_id)
                if receipt is not None:
                    if receipt.entry_id != entry.id or receipt.request_sha256 != request_sha256:
                        raise KnowledgePublicationConflict("operation_mismatch")
                    return receipt.model_copy(update={"replayed": True})
                if await self.get_entry(entry.id) is not None:
                    raise KnowledgePublicationConflict("entry_occupied")
                await super().put_entry_with_chunks(entry, chunks)
                receipt = KnowledgePublicationReceipt(
                    operation_id=operation_id,
                    entry_id=entry.id,
                    request_sha256=request_sha256,
                    entry_created_at=entry.created_at,
                    entry_updated_at=entry.updated_at,
                    committed_at=datetime.now(UTC),
                )
                self._custom_receipts[operation_id] = receipt
                return receipt

        async def load_entry_publication_receipt(self, operation_id):
            async with self._custom_publication_lock:
                return self._custom_receipts.get(operation_id)

    async def run():
        store = PublicCanonicalizerStore()
        context = ToolContext(
            session_id="session_1",
            idempotency_key="custom-canonical-publication",
            knowledge_store=store,
        )
        first = await RememberKnowledgeTool().run(
            context,
            {"text": "Custom stores use Cayu's public publication canonicalizer."},
        )
        replay = await RememberKnowledgeTool().run(
            context,
            {"text": "Custom stores use Cayu's public publication canonicalizer."},
        )
        return first, replay

    first, replay = asyncio.run(run())

    assert first.is_error is False
    assert first.structured is not None
    assert first.structured["written"] is True
    assert replay.is_error is False
    assert replay.structured is not None
    assert replay.structured["entry"]["entry_id"] == first.structured["entry"]["entry_id"]
    assert replay.structured["written"] is False
    assert replay.structured["already_known"] is None
    assert replay.structured["publication_replayed"] is True
    assert replay.structured["status"] is None


def test_remember_knowledge_does_not_authenticate_custom_store_input_mutation() -> None:
    class MutatingCanonicalizerStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self._custom_receipts: dict[str, KnowledgePublicationReceipt] = {}

        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            entry.labels.clear()
            entry.labels["tenant"] = "other"
            entry.metadata["mutated_by_store"] = True
            chunks[0].metadata["mutated_by_store"] = True
            operation_id, entry, chunks, request_sha256 = prepare_knowledge_publication(
                entry,
                chunks,
                operation_id=operation_id,
            )
            await super().put_entry_with_chunks(entry, chunks)
            receipt = KnowledgePublicationReceipt(
                operation_id=operation_id,
                entry_id=entry.id,
                request_sha256=request_sha256,
                entry_created_at=entry.created_at,
                entry_updated_at=entry.updated_at,
                committed_at=datetime.now(UTC),
            )
            self._custom_receipts[operation_id] = receipt
            return receipt

        async def load_entry_publication_receipt(self, operation_id):
            return self._custom_receipts.get(operation_id)

    async def run():
        store = MutatingCanonicalizerStore()
        result = await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(require_labels={"tenant": "expected"})
        ).run(
            ToolContext(
                session_id="session_1",
                idempotency_key="mutated-custom-publication",
                knowledge_store=store,
            ),
            {"text": "Custom store mutation cannot redefine publication authority."},
        )
        assert result.structured is not None
        stored_entry = await store.get_entry(result.structured["entry_id"])
        stored_chunks = await store.read_chunks(result.structured["entry_id"])
        return result, stored_entry, stored_chunks

    result, stored_entry, stored_chunks = asyncio.run(run())

    assert result.is_error is True
    assert result.structured["outcome"] == "invalid_publication_result"
    assert result.structured["cleanup"] == "not_attempted_unowned"
    assert stored_entry is not None
    assert stored_entry.labels == {"tenant": "other"}
    assert stored_entry.metadata["mutated_by_store"] is True
    assert stored_chunks[0].metadata["mutated_by_store"] is True


def test_remember_knowledge_does_not_dedupe_archived_entry() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        ctx = ToolContext(session_id="session_1", knowledge_store=store)
        args = {"text": "Always verify bank details before paying invoices."}
        first = await RememberKnowledgeTool().run(ctx, args)
        assert first.structured is not None
        await store.update_entry_status(
            first.structured["entry"]["entry_id"],
            KnowledgeStatus.ARCHIVED,
        )
        second = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_2", knowledge_store=store),
            args,
        )
        third = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_3", knowledge_store=store),
            args,
        )
        assert second.structured is not None
        await store.update_entry_status(
            second.structured["entry"]["entry_id"],
            KnowledgeStatus.ARCHIVED,
        )
        fourth = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_4", knowledge_store=store),
            args,
        )
        fifth = await RememberKnowledgeTool().run(
            ToolContext(session_id="session_5", knowledge_store=store),
            args,
        )
        return first, second, third, fourth, fifth

    first, second, third, fourth, fifth = asyncio.run(run())

    assert first.is_error is False
    assert second.is_error is False
    assert third.is_error is False
    assert fourth.is_error is False
    assert fifth.is_error is False
    assert first.structured is not None
    assert second.structured is not None
    assert third.structured is not None
    assert fourth.structured is not None
    assert fifth.structured is not None
    assert second.structured["written"] is True
    assert second.structured["already_known"] is False
    assert second.structured["source_hash"] == first.structured["source_hash"]
    assert second.structured["entry"]["entry_id"] != first.structured["entry"]["entry_id"]
    assert third.structured["written"] is False
    assert third.structured["already_known"] is True
    assert third.structured["entry"]["entry_id"] == second.structured["entry"]["entry_id"]
    assert fourth.structured["written"] is True
    assert fourth.structured["already_known"] is False
    assert fourth.structured["entry"]["entry_id"] != first.structured["entry"]["entry_id"]
    assert fourth.structured["entry"]["entry_id"] != second.structured["entry"]["entry_id"]
    assert fifth.structured["written"] is False
    assert fifth.structured["already_known"] is True
    assert fifth.structured["entry"]["entry_id"] == fourth.structured["entry"]["entry_id"]


def test_remember_knowledge_distinct_kinds_are_not_deduped() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        ctx = ToolContext(session_id="session_1", knowledge_store=store)
        text = "Always verify bank details before paying invoices."
        first = await RememberKnowledgeTool().run(ctx, {"text": text, "kind": "fact"})
        second = await RememberKnowledgeTool().run(ctx, {"text": text, "kind": "warning"})
        return first, second

    first, second = asyncio.run(run())

    assert first.is_error is False
    assert second.is_error is False
    assert first.structured is not None
    assert second.structured is not None
    assert second.structured["already_known"] is False
    assert second.structured["written"] is True
    assert second.structured["entry"]["entry_id"] != first.structured["entry"]["entry_id"]
