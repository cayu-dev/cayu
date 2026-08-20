from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.core.knowledge_access_scope_conformance import (
    assert_knowledge_access_scope_conformance,
)
from tests.core.knowledge_index_readiness_conformance import (
    assert_index_readiness_conformance,
)
from tests.core.knowledge_none_terms_conformance import (
    assert_entry_wide_none_terms_conformance,
)
from tests.core.knowledge_phrase_conformance import (
    assert_token_exact_phrase_search_conformance,
)
from tests.core.knowledge_publication_conformance import (
    assert_concurrent_publication_conformance,
    assert_owned_publication_conformance,
    assert_stale_operation_cannot_replace_newer_publication,
    publication_material,
)

from cayu._validation import DurableValueError, extract_durable_value_error
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.storage import (
    BUILTIN_KNOWLEDGE_KINDS,
    MAX_KNOWLEDGE_CHANGE_LIMIT,
    MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_KNOWLEDGE_CHUNK_INDEX,
    MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
    MAX_KNOWLEDGE_ENTRY_ID_BYTES,
    MAX_KNOWLEDGE_EVIDENCE_BYTES,
    MAX_KNOWLEDGE_REVISION,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEmbeddingProjection,
    KnowledgeEmbeddingProjectionConflict,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeRevisionConflict,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    knowledge_chunk_embedding_identity,
)
from cayu.storage.memory import (
    copy_knowledge_entry,
    copy_knowledge_hit,
    copy_knowledge_list_item,
)

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()


def test_in_memory_index_readiness_conformance() -> None:
    asyncio.run(
        assert_index_readiness_conformance(InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE))
    )


def test_embedding_identity_hashes_exact_projected_text() -> None:
    first = KnowledgeChunk(
        id="projection-hash-chunk",
        entry_id="projection-hash-entry",
        text="first projected text",
        chunk_index=0,
        content_hash="caller-supplied-source-hash",
    )
    second = first.model_copy(update={"text": "second projected text"})

    first_identity = knowledge_chunk_embedding_identity(
        first,
        embedding_model="test-embedding",
        dimensions=3,
    )
    second_identity = knowledge_chunk_embedding_identity(
        second,
        embedding_model="test-embedding",
        dimensions=3,
    )

    assert first_identity.projection_content_hash.startswith("sha256:")
    assert first_identity.projection_content_hash != second_identity.projection_content_hash
    with pytest.raises(ValidationError, match="dimensions.*less than or equal"):
        knowledge_chunk_embedding_identity(
            first,
            embedding_model="test-embedding",
            dimensions=2**31,
        )


def test_embedding_projection_rejects_malformed_vectors() -> None:
    identity = knowledge_chunk_embedding_identity(
        KnowledgeChunk(
            id="projection-validation-chunk",
            entry_id="projection-validation-entry",
            text="projected text",
            chunk_index=0,
        ),
        embedding_model="test-embedding",
        dimensions=3,
    )

    with pytest.raises(ValidationError, match="length must equal"):
        KnowledgeEmbeddingProjection(
            identity=identity,
            readiness_sequence=1,
            attempt_id="projection-validation",
            vector=[1.0, 0.0],
        )
    with pytest.raises(ValidationError, match=r"vector\[0\].*number"):
        KnowledgeEmbeddingProjection(
            identity=identity,
            readiness_sequence=1,
            attempt_id="projection-validation",
            vector=[True, 0.0, 0.0],
        )
    with pytest.raises(ValidationError, match=r"vector\[0\].*finite"):
        KnowledgeEmbeddingProjection(
            identity=identity,
            readiness_sequence=1,
            attempt_id="projection-validation",
            vector=[float("nan"), 0.0, 0.0],
        )


def test_knowledge_evidence_enforces_utf8_and_total_serialized_byte_limits() -> None:
    with pytest.raises(ValidationError, match="source_id.*256 UTF-8 bytes"):
        KnowledgeEvidence(
            id="evidence-byte-limit",
            entry_id="entry",
            source_type="document",
            source_id="é" * 129,
            source_revision="1",
        )

    with pytest.raises(
        ValidationError,
        match=rf"at most {MAX_KNOWLEDGE_EVIDENCE_BYTES} canonical UTF-8 bytes",
    ):
        KnowledgeEvidence(
            id="evidence-total-limit",
            entry_id="entry",
            source_type="document",
            source_id="source",
            source_revision="1",
            locator={"excerpt": "x" * 10_500},
            metadata={"context": "y" * 10_500},
        )


def test_knowledge_identities_enforce_portable_utf8_byte_limits() -> None:
    entry_id = "é" * (MAX_KNOWLEDGE_ENTRY_ID_BYTES // 2)
    chunk_id = "é" * (MAX_KNOWLEDGE_CHUNK_ID_BYTES // 2)
    entry = KnowledgeEntry(id=entry_id, text="bounded identity")
    chunk = KnowledgeChunk(
        id=chunk_id,
        entry_id=entry.id,
        chunk_index=0,
        text=entry.text,
    )
    evidence = KnowledgeEvidence(
        id="bounded-identity-evidence",
        entry_id=entry.id,
        chunk_id=chunk.id,
        source_type="document",
        source_id="bounded-source",
        source_revision="1",
    )

    assert len(entry.id.encode("utf-8")) == MAX_KNOWLEDGE_ENTRY_ID_BYTES
    assert len(chunk.id.encode("utf-8")) == MAX_KNOWLEDGE_CHUNK_ID_BYTES
    assert evidence.entry_id == entry.id
    assert evidence.chunk_id == chunk.id

    with pytest.raises(
        ValidationError,
        match=rf"id.*at most {MAX_KNOWLEDGE_ENTRY_ID_BYTES} UTF-8 bytes",
    ):
        KnowledgeEntry(id=entry_id + "x", text="too long")
    with pytest.raises(
        ValidationError,
        match=rf"id.*at most {MAX_KNOWLEDGE_CHUNK_ID_BYTES} UTF-8 bytes",
    ):
        KnowledgeChunk(
            id=chunk_id + "x",
            entry_id="entry",
            chunk_index=0,
            text="too long",
        )
    with pytest.raises(
        ValidationError,
        match=rf"entry_id.*at most {MAX_KNOWLEDGE_ENTRY_ID_BYTES} UTF-8 bytes",
    ):
        KnowledgeEvidence(
            id="oversized-entry-reference",
            entry_id=entry_id + "x",
            source_type="document",
            source_id="source",
            source_revision="1",
        )
    with pytest.raises(
        ValidationError,
        match=rf"chunk_id.*at most {MAX_KNOWLEDGE_CHUNK_ID_BYTES} UTF-8 bytes",
    ):
        KnowledgeEvidence(
            id="oversized-chunk-reference",
            entry_id="entry",
            chunk_id=chunk_id + "x",
            source_type="document",
            source_id="source",
            source_revision="1",
        )


def test_knowledge_change_pages_reject_unbounded_record_limits() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        with pytest.raises(ValueError, match=str(MAX_KNOWLEDGE_CHANGE_LIMIT)):
            await store.read_changes(limit=MAX_KNOWLEDGE_CHANGE_LIMIT + 1)

    asyncio.run(run())


def test_in_memory_knowledge_access_scope_conformance() -> None:
    asyncio.run(assert_knowledge_access_scope_conformance(InMemoryKnowledgeStore()))


def test_knowledge_store_requires_or_binds_an_explicit_access_scope() -> None:
    async def run() -> None:
        unbound = InMemoryKnowledgeStore()
        with pytest.raises(TypeError, match="requires `access_scope`"):
            await unbound.search(KnowledgeQuery(text="anything"))

        original = KnowledgeAccessScope.for_namespace("tenant-a")
        bound = InMemoryKnowledgeStore(access_scope=original)
        original.allowed_namespaces.append("tenant-b")
        await bound.create_entry(KnowledgeEntry(id="a", namespace="tenant-a", text="allowed"))
        with pytest.raises(KnowledgeAccessDenied):
            await bound.create_entry(KnowledgeEntry(id="b", namespace="tenant-b", text="denied"))
        with pytest.raises(KnowledgeAccessDenied, match="access_scope_override"):
            await bound.search(
                KnowledgeQuery(text="anything"),
                access_scope=KnowledgeAccessScope.privileged(),
            )

        multi_scope = KnowledgeAccessScope(
            allowed_namespaces=["tenant-b", "tenant-a"],
            allowed_visibilities=[
                KnowledgeVisibility.PROJECT,
                KnowledgeVisibility.GLOBAL,
            ],
        )
        equivalent = KnowledgeAccessScope(
            allowed_namespaces=["tenant-a", "tenant-b"],
            allowed_visibilities=[
                KnowledgeVisibility.GLOBAL,
                KnowledgeVisibility.PROJECT,
            ],
        )
        multi_bound = InMemoryKnowledgeStore(access_scope=multi_scope)
        await multi_bound.search(KnowledgeQuery(text="anything"), access_scope=equivalent)

    asyncio.run(run())


class KeywordEmbeddingProvider(TextEmbeddingProvider):
    name = "keyword-test"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.calls.append(list(request.texts))
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


def test_knowledge_entry_accepts_extensible_kind_and_core_fields() -> None:
    entry = KnowledgeEntry(
        id="entry_1",
        text="Refund requests require approval above $100.",
        namespace="support",
        labels={"project": "billing"},
        kind="support.playbook",
        visibility=KnowledgeVisibility.PROJECT,
        created_by_type=KnowledgeActorType.USER,
        created_by="user_1",
        aspects=["finance"],
        impact_targets=["finance.refunds"],
        importance=0.8,
        confidence=0.9,
        source_type="app_document",
        source_uri="kb://refunds",
        metadata={"nested": {"value": "original"}},
    )

    assert "skill" in BUILTIN_KNOWLEDGE_KINDS
    assert entry.kind == "support.playbook"
    assert entry.labels == {"project": "billing"}
    assert entry.visibility == KnowledgeVisibility.PROJECT
    assert entry.created_at.tzinfo is not None
    assert entry.updated_at.tzinfo is not None


def test_knowledge_entry_and_query_dedupe_list_filters() -> None:
    entry = KnowledgeEntry(
        id="entry_1",
        text="Refund process.",
        aspects=["finance", "finance"],
        impact_targets=["refunds", "refunds"],
    )
    query = KnowledgeQuery(
        text="refund",
        kinds=["warning", "warning"],
        aspects=["finance", "finance"],
        impact_targets=["refunds", "refunds"],
    )

    assert entry.aspects == ["finance"]
    assert entry.impact_targets == ["refunds"]
    assert query.kinds == ["warning"]
    assert query.aspects == ["finance"]
    assert query.impact_targets == ["refunds"]


def test_knowledge_entry_rejects_invalid_identity_labels_and_scores() -> None:
    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        KnowledgeEntry(id=" entry_1", text="memory")

    with pytest.raises(ValidationError, match="labels"):
        KnowledgeEntry(id="entry_1", text="memory", labels={"project": " "})

    with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
        KnowledgeEntry(id="entry_1", text="memory", confidence=1.5)

    with pytest.raises(ValidationError, match="timezone-aware"):
        KnowledgeEntry(
            id="entry_1",
            text="memory",
            created_at=datetime(2026, 1, 1),
        )

    with pytest.raises(ValidationError, match="updated_at"):
        KnowledgeEntry(
            id="entry_1",
            text="memory",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_knowledge_entry_and_chunk_enforce_portable_durable_values() -> None:
    numbers = {
        "ordinary": 1.0,
        "negative_zero": -0.0,
        "large": 1e18,
        "fractional": 1e-7,
    }
    entry = KnowledgeEntry(id="entry_numbers", text="memory", metadata={"numbers": numbers})
    chunk = KnowledgeChunk(
        id="chunk_numbers",
        entry_id=entry.id,
        chunk_index=0,
        text="chunk",
        metadata={"numbers": numbers},
    )
    boundary_chunk = KnowledgeChunk(
        id="chunk_boundary",
        entry_id=entry.id,
        chunk_index=MAX_KNOWLEDGE_CHUNK_INDEX,
        text="chunk",
    )
    assert boundary_chunk.chunk_index == MAX_KNOWLEDGE_CHUNK_INDEX

    boundary_entry = KnowledgeEntry(
        id="entry_revision_boundary",
        revision=MAX_KNOWLEDGE_REVISION,
        text="memory",
    )
    boundary_revision_chunk = KnowledgeChunk(
        id="chunk_revision_boundary",
        entry_id=boundary_entry.id,
        entry_revision=MAX_KNOWLEDGE_REVISION,
        chunk_index=0,
        text="chunk",
    )
    assert boundary_entry.revision == MAX_KNOWLEDGE_REVISION
    assert boundary_revision_chunk.entry_revision == MAX_KNOWLEDGE_REVISION

    with pytest.raises(ValidationError, match=str(MAX_KNOWLEDGE_REVISION)):
        KnowledgeEntry(
            id="entry_revision_out_of_range",
            revision=MAX_KNOWLEDGE_REVISION + 1,
            text="memory",
        )

    with pytest.raises(ValidationError, match=str(MAX_KNOWLEDGE_REVISION)):
        KnowledgeChunk(
            id="chunk_revision_out_of_range",
            entry_id=entry.id,
            entry_revision=MAX_KNOWLEDGE_REVISION + 1,
            chunk_index=0,
            text="chunk",
        )

    with pytest.raises(ValidationError, match=str(MAX_KNOWLEDGE_CHUNK_INDEX)):
        KnowledgeChunk(
            id="chunk_out_of_range",
            entry_id=entry.id,
            chunk_index=MAX_KNOWLEDGE_CHUNK_INDEX + 1,
            text="chunk",
        )

    for value in (entry.metadata["numbers"], chunk.metadata["numbers"]):
        assert value == {
            "ordinary": 1,
            "negative_zero": 0,
            "large": 1_000_000_000_000_000_000,
            "fractional": 1e-7,
        }
        assert type(value["ordinary"]) is int
        assert type(value["negative_zero"]) is int
        assert type(value["large"]) is int
        assert type(value["fractional"]) is float

    for factory, code in (
        (
            lambda: KnowledgeEntry(
                id="entry_invalid_metadata",
                text="memory",
                metadata={"bad": float("nan")},
            ),
            "non_finite_number",
        ),
        (
            lambda: KnowledgeChunk(
                id="chunk_invalid_metadata",
                entry_id="entry_1",
                chunk_index=0,
                text="chunk",
                metadata={"bad": "value\ud800"},
            ),
            "unicode_surrogate",
        ),
        (
            lambda: KnowledgeEntry(id="entry_invalid_text", text="memory\x00value"),
            "nul_character",
        ),
    ):
        with pytest.raises(ValidationError) as invalid_value:
            factory()
        durable_error = extract_durable_value_error(invalid_value.value)
        assert durable_error is not None
        assert durable_error.code == code


def test_in_memory_knowledge_store_revalidates_poisoned_chunk_batch_atomically() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        entry = KnowledgeEntry(id="entry_poisoned_batch", text="memory")
        valid_chunk = KnowledgeChunk(
            id="chunk_valid",
            entry_id=entry.id,
            chunk_index=0,
            text="valid",
        )
        poisoned_chunk = KnowledgeChunk(
            id="chunk_poisoned",
            entry_id=entry.id,
            chunk_index=1,
            text="poisoned",
            metadata={"safe": True},
        )
        poisoned_chunk.metadata["bad"] = float("inf")

        with pytest.raises((DurableValueError, ValidationError)) as invalid_batch:
            await store.create_entry(entry, [valid_chunk, poisoned_chunk])
        durable_error = extract_durable_value_error(invalid_batch.value)
        assert durable_error is not None
        assert durable_error.code == "non_finite_number"
        assert await store.get_entry(entry.id) is None
        assert await store.read_chunks(entry.id) == []

    asyncio.run(run())


def test_in_memory_knowledge_store_rejects_out_of_range_chunk_index_atomically() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        entry = KnowledgeEntry(id="entry_chunk_index", text="memory")
        chunk = KnowledgeChunk(
            id="chunk_index",
            entry_id=entry.id,
            chunk_index=0,
            text="chunk",
        )
        object.__setattr__(chunk, "chunk_index", MAX_KNOWLEDGE_CHUNK_INDEX + 1)

        with pytest.raises(ValidationError, match=str(MAX_KNOWLEDGE_CHUNK_INDEX)):
            await store.create_entry(entry, [chunk])
        assert await store.get_entry(entry.id) is None
        assert await store.read_chunks(entry.id) == []

    asyncio.run(run())


def test_in_memory_knowledge_store_owned_publication_conformance() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await assert_owned_publication_conformance(store)
        await assert_concurrent_publication_conformance(store)
        await assert_stale_operation_cannot_replace_newer_publication(store)

    asyncio.run(run())


def test_knowledge_store_owned_publication_hooks_are_not_abstract() -> None:
    assert "publish_entry_revision" not in KnowledgeStore.__abstractmethods__
    assert "load_entry_publication_receipt" not in KnowledgeStore.__abstractmethods__


def test_knowledge_hit_owns_copies() -> None:
    metadata = {"nested": {"value": "original"}}
    entry = KnowledgeEntry(id="entry_1", text="billing memory", metadata=metadata)
    hit = KnowledgeHit(entry=entry, score=2.0)

    metadata["nested"]["value"] = "mutated"
    entry.metadata["nested"]["value"] = "mutated again"

    assert hit.entry.id == "entry_1"
    assert hit.entry.metadata == {"nested": {"value": "original"}}


def test_knowledge_preview_completeness_is_typed_internal_provenance() -> None:
    entry = KnowledgeEntry(id="entry_1", text="complete preview")
    hit = KnowledgeHit(
        entry=entry,
        text_preview=entry.text,
        text_preview_complete=True,
    )
    item = KnowledgeListItem(
        entry=entry,
        text_preview=entry.text,
        text_preview_complete=True,
    )

    assert hit.text_preview_complete is True
    assert item.text_preview_complete is True
    assert copy_knowledge_hit(hit).text_preview_complete is True
    assert copy_knowledge_list_item(item).text_preview_complete is True
    assert "text_preview_complete" not in hit.model_dump()
    assert "text_preview_complete" not in item.model_dump()

    with pytest.raises(ValidationError, match="must be a boolean"):
        KnowledgeHit(
            entry=entry,
            text_preview=entry.text,
            text_preview_complete=1,
        )


def test_knowledge_hit_rejects_chunk_for_different_entry() -> None:
    entry = KnowledgeEntry(id="entry_1", text="billing memory")
    chunk = KnowledgeChunk(id="chunk_1", entry_id="entry_2", chunk_index=0, text="other")

    with pytest.raises(ValidationError, match="chunk.entry_id"):
        KnowledgeHit(entry=entry, chunk=chunk)


def test_knowledge_hit_rejects_chunk_for_different_revision() -> None:
    entry = KnowledgeEntry(id="entry_1", revision=2, text="current billing memory")
    chunk = KnowledgeChunk(
        id="chunk_1",
        entry_id=entry.id,
        entry_revision=1,
        chunk_index=0,
        text="superseded billing memory",
    )

    with pytest.raises(ValidationError, match="chunk.entry_revision"):
        KnowledgeHit(entry=entry, chunk=chunk)


def test_knowledge_search_result_rejects_impossible_known_total() -> None:
    hit = KnowledgeHit(entry=KnowledgeEntry(id="entry_1", text="billing memory"))

    with pytest.raises(ValidationError, match="total_hits_known"):
        KnowledgeSearchResult(
            query=KnowledgeQuery(text="billing"),
            hits=[hit],
            limit=10,
            max_bytes=20_000,
            total_hits_known=0,
        )


def test_knowledge_search_result_requires_limits_to_match_query() -> None:
    query = KnowledgeQuery(text="billing", limit=3, max_bytes=100)

    with pytest.raises(ValidationError, match="limit"):
        KnowledgeSearchResult(query=query, hits=[], limit=2, max_bytes=100)

    with pytest.raises(ValidationError, match="max_bytes"):
        KnowledgeSearchResult(query=query, hits=[], limit=3, max_bytes=99)


def test_knowledge_search_result_rejects_too_many_hits_and_duplicate_ranks() -> None:
    query = KnowledgeQuery(text="billing", limit=1)
    first = KnowledgeHit(entry=KnowledgeEntry(id="entry_1", text="billing memory"), rank=1)
    second = KnowledgeHit(entry=KnowledgeEntry(id="entry_2", text="billing policy"), rank=2)

    with pytest.raises(ValidationError, match="more entries than `limit`"):
        KnowledgeSearchResult(
            query=query,
            hits=[first, second],
            limit=1,
            max_bytes=20_000,
        )

    with pytest.raises(ValidationError, match="ranks"):
        KnowledgeSearchResult(
            query=KnowledgeQuery(text="billing", limit=2),
            hits=[
                first,
                KnowledgeHit(entry=KnowledgeEntry(id="entry_3", text="billing runbook"), rank=1),
            ],
            limit=2,
            max_bytes=20_000,
        )


def test_in_memory_knowledge_store_searches_filters_and_scopes() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(
                id="invoice_warning",
                text="Do not send invoice reminders when the PO number is missing.",
                namespace="ops",
                labels={"project": "invoice_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="other_project_warning",
                text="Do not send invoice reminders when the PO number is missing.",
                namespace="ops",
                labels={"project": "other_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="invoice_procedure",
                text="Payment reminders should include invoice number and vendor name.",
                namespace="ops",
                labels={"project": "invoice_agent", "user": "alice"},
                kind="procedure",
                visibility=KnowledgeVisibility.PROJECT,
            )
        )

        query = KnowledgeQuery(
            text="invoice reminders",
            namespace="ops",
            labels={"project": "invoice_agent"},
            kinds=["warning"],
        )
        return await store.search(query)

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["invoice_warning"]
    assert result.hits[0].entry.kind == "warning"
    assert result.hits[0].rank == 1
    assert "invoice reminders" in result.hits[0].text_preview
    assert result.limit == 10
    assert result.max_bytes == 20_000
    assert result.total_hits_known == 1


def test_in_memory_knowledge_store_rejects_duplicate_seed_entry_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate knowledge entry id"):
        InMemoryKnowledgeStore(
            [
                KnowledgeEntry(id="same", text="first"),
                KnowledgeEntry(id="same", text="second"),
            ],
            access_scope=_ACCESS_SCOPE,
        )


def test_in_memory_knowledge_store_excludes_non_active_and_expired_by_default() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(
                id="active",
                text="Active deployment warning.",
                namespace="deploy",
                kind="warning",
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="pending",
                text="Pending deployment warning.",
                namespace="deploy",
                kind="warning",
                status=KnowledgeStatus.PENDING,
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="expired",
                text="Expired deployment warning.",
                namespace="deploy",
                kind="warning",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        active_result = await store.search(KnowledgeQuery(text="deployment", namespace="deploy"))
        pending_result = await store.search(
            KnowledgeQuery(
                text="deployment",
                namespace="deploy",
                statuses=[KnowledgeStatus.PENDING],
            )
        )
        expired_result = await store.search(
            KnowledgeQuery(
                text="deployment",
                namespace="deploy",
                include_expired=True,
            )
        )
        return active_result, pending_result, expired_result

    active_result, pending_result, expired_result = asyncio.run(run())

    assert [hit.entry.id for hit in active_result.hits] == ["active"]
    assert [hit.entry.id for hit in pending_result.hits] == ["pending"]
    assert [hit.entry.id for hit in expired_result.hits] == ["expired", "active"]


def test_in_memory_knowledge_store_prune_expired_removes_expired_entries() -> None:
    # MEM-05: the read filter only hides expired entries (include_expired=True still surfaces them);
    # prune_expired reclaims them for good.
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(id="active", text="Active deployment warning.", kind="warning")
        )
        await store.create_entry(
            KnowledgeEntry(
                id="expired",
                text="Expired deployment warning.",
                kind="warning",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        pruned = await store.prune_expired()
        leftover = await store.search(KnowledgeQuery(text="deployment", include_expired=True))
        return pruned, leftover, await store.get_entry("expired"), await store.get_entry("active")

    pruned, leftover, expired_entry, active_entry = asyncio.run(run())

    assert pruned == 1
    assert expired_entry is None
    assert active_entry is not None
    assert [hit.entry.id for hit in leftover.hits] == ["active"]


def test_in_memory_embedding_store_prune_expired_drops_embeddings() -> None:
    # MEM-05: the derived-index consumer reclaims vectors after canonical pruning.
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(
            KnowledgeEntry(
                id="expired",
                text="Expired secret rotation runbook.",
                kind="procedure",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        await store.process_embedding_changes("embedding-prune", "worker")
        embeddings_before = len(store._chunk_embeddings)
        pruned = await store.prune_expired()
        await store.process_embedding_changes("embedding-prune", "worker")
        return (
            embeddings_before,
            pruned,
            len(store._chunk_embeddings),
            await store.get_entry("expired"),
        )

    embeddings_before, pruned, embeddings_after, entry = asyncio.run(run())

    assert embeddings_before == 1
    assert pruned == 1
    assert embeddings_after == 0
    assert entry is None


def test_in_memory_embedding_lifecycle_revisions_replace_stale_derived_rows() -> None:
    async def run() -> tuple[KnowledgeEntry, KnowledgeEntry, dict[str, object]]:
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        created = await store.create_entry(
            KnowledgeEntry(id="lifecycle-embedding", text="GitHub credential proxy.")
        )
        archived = await store.transition_entry_status(
            created.id,
            expected_revision=created.revision,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
        )
        deleted = await store.delete_entry(
            archived.id,
            expected_revision=archived.revision,
        )
        assert deleted is not None
        await store.process_embedding_changes("embedding-lifecycle", "worker")
        return archived, deleted, dict(store._chunk_embeddings)

    archived, deleted, embeddings = asyncio.run(run())

    assert archived.revision == 2
    assert deleted.revision == 3
    assert embeddings == {}


def test_in_memory_semantic_search_keeps_one_authorized_snapshot_across_provider_awaits() -> None:
    class QueryBlockingEmbeddingProvider(TextEmbeddingProvider):
        name = "query-blocking-keyword-test"

        def __init__(self) -> None:
            self.query_started = asyncio.Event()
            self.release_query = asyncio.Event()
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            if self.call_count == 2:
                self.query_started.set()
                await self.release_query.wait()
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=index, vector=_test_embedding_vector(text))
                    for index, text in enumerate(request.texts)
                ],
            )

    async def run() -> tuple[KnowledgeEntry, KnowledgeChunk]:
        provider = QueryBlockingEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        privileged = KnowledgeAccessScope.privileged()
        tenant_scope = KnowledgeAccessScope.for_namespace("tenant-a")
        original = KnowledgeEntry(
            id="shared-entry",
            namespace="tenant-a",
            text="Tenant A credential policy.",
        )
        original_chunk = KnowledgeChunk(
            id="shared-entry:0",
            entry_id=original.id,
            chunk_index=0,
            text=original.text,
        )
        await store.create_entry(
            original,
            [original_chunk],
            access_scope=privileged,
        )
        await store.process_embedding_changes(
            "snapshot-index",
            "worker",
            access_scope=privileged,
        )

        search_task = asyncio.create_task(
            store.search(
                KnowledgeQuery(
                    text="credential",
                    namespace="tenant-a",
                    mode=KnowledgeSearchMode.SEMANTIC,
                ),
                access_scope=tenant_scope,
            )
        )
        await asyncio.wait_for(provider.query_started.wait(), timeout=2)
        replacement = original.model_copy(
            update={
                "revision": 2,
                "status": KnowledgeStatus.ARCHIVED,
                "text": "Archived secret credential policy.",
            }
        )
        replacement_chunk = KnowledgeChunk(
            id="shared-entry:r2:0",
            entry_id=replacement.id,
            entry_revision=replacement.revision,
            chunk_index=0,
            text=replacement.text,
        )
        await store.append_entry_revision(
            replacement,
            [replacement_chunk],
            expected_revision=original.revision,
            access_scope=privileged,
        )
        provider.release_query.set()
        result = await search_task

        active_scope = KnowledgeAccessScope.for_namespace(
            "tenant-a",
            allowed_statuses=[KnowledgeStatus.ACTIVE],
        )
        assert await store.get_entry(original.id, access_scope=active_scope) is None
        assert result.hits
        returned_chunk = result.hits[0].chunk
        assert returned_chunk is not None
        return result.hits[0].entry, returned_chunk

    entry, chunk = asyncio.run(run())

    assert entry.namespace == "tenant-a"
    assert chunk.text == "Tenant A credential policy."


def test_in_memory_embedding_worker_does_not_apply_stale_derived_embeddings() -> None:
    class FirstCallBlockingEmbeddingProvider(TextEmbeddingProvider):
        name = "blocking-keyword-test"

        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            if self.call_count == 1:
                self.first_started.set()
                await self.release_first.wait()
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=index, vector=_test_embedding_vector(text))
                    for index, text in enumerate(request.texts)
                ],
            )

    async def run():
        provider = FirstCallBlockingEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        old_entry, old_chunks = publication_material(
            entry_id="reused_embedding_publication",
            text="GitHub credential proxy policy.",
        )
        old_chunks = [old_chunks[0].model_copy(update={"id": "old-publication-chunk"})]
        await store.publish_entry_revision(
            old_entry,
            old_chunks,
            operation_id="old-embedding-publication",
        )
        old_task = asyncio.create_task(
            store.process_embedding_changes(
                "stale-embedding-worker",
                "worker",
            )
        )
        await asyncio.wait_for(provider.first_started.wait(), timeout=2)
        await store.delete_entry(
            old_entry.id,
            expected_revision=old_entry.revision,
            hard=True,
        )
        new_entry, new_chunks = publication_material(
            entry_id=old_entry.id,
            text="Invoice payment refund policy.",
            timestamp_offset=1,
        )
        new_chunks = [new_chunks[0].model_copy(update={"id": "new-publication-chunk"})]
        await store.publish_entry_revision(
            new_entry,
            new_chunks,
            operation_id="new-embedding-publication",
        )
        provider.release_first.set()
        worker_result = await old_task
        return dict(store._chunk_embeddings), new_chunks[0], worker_result

    embeddings, new_chunk, worker_result = asyncio.run(run())

    identities = [stored["identity"] for stored in embeddings.values()]
    assert all(identity.chunk_id != "old-publication-chunk" for identity in identities)
    assert [identity.chunk_id for identity in identities] == [new_chunk.id]
    assert worker_result.acknowledged_changes == 3


def test_in_memory_embedding_worker_fences_superseded_attempt_vector_write() -> None:
    class FirstCallBlockingEmbeddingProvider(TextEmbeddingProvider):
        name = "blocking-attempt-fence-test"

        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            call = self.call_count
            if call == 1:
                self.first_started.set()
                await self.release_first.wait()
            vector = [1.0, 0.0, 0.0] if call == 1 else [0.0, 1.0, 0.0]
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=index, vector=vector)
                    for index, _ in enumerate(request.texts)
                ],
            )

    async def run() -> tuple[list[float], int]:
        provider = FirstCallBlockingEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        entry, chunks = publication_material(
            entry_id="embedding-attempt-fence",
            text="GitHub credential proxy policy.",
        )
        await store.publish_entry_revision(
            entry,
            chunks,
            operation_id="embedding-attempt-fence-publication",
        )
        slow = asyncio.create_task(
            store.process_embedding_changes("slow-attempt-index", "worker-a")
        )
        await asyncio.wait_for(provider.first_started.wait(), timeout=2)
        fast = await store.process_embedding_changes("fast-attempt-index", "worker-b")
        provider.release_first.set()
        await slow
        assert fast.indexed_records == 1
        stored = next(iter(store._chunk_embeddings.values()))
        return list(stored["vector"]), provider.call_count

    vector, call_count = asyncio.run(run())

    assert vector == [0.0, 1.0, 0.0]
    assert call_count == 2


def test_in_memory_publication_replay_does_not_repeat_failed_embedding_work() -> None:
    class CountingFailingEmbeddingProvider(TextEmbeddingProvider):
        name = "counting-failing-test"

        def __init__(self) -> None:
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            raise RuntimeError("embedding provider is unavailable")

    async def run():
        provider = CountingFailingEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        entry, chunks = publication_material(
            entry_id="failed-derived-replay",
            text="Durable source publication survives derived work failure.",
        )
        first_receipt = await store.publish_entry_revision(
            entry,
            chunks,
            operation_id="failed-derived-replay-operation",
        )
        receipt = await store.publish_entry_revision(
            entry,
            chunks,
            operation_id="failed-derived-replay-operation",
        )
        worker_result = await store.process_embedding_changes(
            "failed-embedding-worker",
            "worker",
        )
        return (
            provider.call_count,
            first_receipt,
            receipt,
            worker_result,
            await store.get_entry(entry.id),
        )

    call_count, first_receipt, receipt, worker_result, stored_entry = asyncio.run(run())

    assert call_count == 1
    assert first_receipt.replayed is False
    assert receipt.replayed is True
    assert worker_result.failed_records == 1
    assert stored_entry is not None


def test_knowledge_store_prune_expired_is_not_abstract() -> None:
    # prune_expired is a concrete NotImplementedError default (not @abstractmethod), so existing
    # out-of-tree KnowledgeStore subclasses still instantiate. Mirrors the SessionStore convention.
    assert "prune_expired" not in KnowledgeStore.__abstractmethods__


def test_in_memory_knowledge_store_chunks_are_bounded_and_scope_checked() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        entry = KnowledgeEntry(
            id="long_doc",
            text="Short summary.",
            namespace="docs",
            labels={"project": "agent_a"},
            kind="document",
        )
        await store.create_entry(
            entry,
            [
                KnowledgeChunk(id="chunk_0", entry_id="long_doc", chunk_index=0, text="alpha beta"),
                KnowledgeChunk(
                    id="chunk_1", entry_id="long_doc", chunk_index=1, text="gamma delta"
                ),
                KnowledgeChunk(
                    id="chunk_2", entry_id="long_doc", chunk_index=2, text="epsilon zeta"
                ),
            ],
        )
        chunks = await store.read_chunks(
            "long_doc",
            chunk_index=1,
            around=1,
            max_chunks=3,
            max_bytes=64,
        )
        bounded_chunks = await store.read_chunks("long_doc", chunk_index=1, around=1, max_chunks=2)
        centered_chunks = await store.read_chunks(
            "long_doc", chunk_index=2, around=10, max_chunks=1
        )
        search_result = await store.search(KnowledgeQuery(text="gamma", namespace="docs"))
        denied_result = await store.search(
            KnowledgeQuery(text="gamma", namespace="docs", labels={"project": "agent_b"})
        )
        return chunks, bounded_chunks, centered_chunks, search_result, denied_result

    chunks, bounded_chunks, centered_chunks, search_result, denied_result = asyncio.run(run())

    assert [chunk.id for chunk in chunks] == ["chunk_0", "chunk_1", "chunk_2"]
    assert [chunk.id for chunk in bounded_chunks] == ["chunk_0", "chunk_1"]
    assert [chunk.id for chunk in centered_chunks] == ["chunk_2"]
    assert search_result.hits[0].entry.id == "long_doc"
    assert search_result.hits[0].chunk.id == "chunk_1"
    assert denied_result.hits == []


def test_in_memory_knowledge_store_truncated_chunk_clears_content_hash() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(id="doc", text="Document summary."),
            [
                KnowledgeChunk(
                    id="chunk_0",
                    entry_id="doc",
                    chunk_index=0,
                    text="alpha beta",
                    content_hash="full-content-hash",
                )
            ],
        )
        return await store.read_chunks("doc", max_bytes=5)

    chunks = asyncio.run(run())

    assert len(chunks) == 1
    assert chunks[0].text == "alpha"
    assert chunks[0].content_hash is None


def test_in_memory_knowledge_store_rejects_ambiguous_chunk_window() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(KnowledgeEntry(id="entry_1", text="memory"))
        with pytest.raises(ValueError, match="around"):
            await store.read_chunks("entry_1", around=1)

    asyncio.run(run())


def test_in_memory_knowledge_store_rejects_unsupported_search_modes() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(KnowledgeEntry(id="entry_1", text="billing memory"))
        with pytest.raises(ValueError, match="supports only auto and keyword"):
            await store.search(KnowledgeQuery(text="billing", mode=KnowledgeSearchMode.SEMANTIC))

    asyncio.run(run())


def test_in_memory_embedding_knowledge_store_semantic_search() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(
            KnowledgeEntry(
                id="git_policy",
                text="GitHub pushes should use a trusted proxy for the secret token.",
                kind="procedure",
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="invoice_policy",
                text="Invoice refunds require payment approval.",
                kind="procedure",
            )
        )
        await store.process_embedding_changes("semantic-index", "worker")
        result = await store.search(
            KnowledgeQuery(text="auth broker", mode=KnowledgeSearchMode.SEMANTIC)
        )
        return result, provider.calls

    result, calls = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["git_policy"]
    assert result.hits[0].score_kind == "inmemory_semantic"
    assert result.hits[0].chunk is not None
    assert result.hits[0].reason == "semantic chunk match"
    assert len(calls) == 3


def test_in_memory_embedding_worker_continues_one_change_within_record_budget() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        entry = KnowledgeEntry(id="bounded-embedding-change", text="Bounded embedding work.")
        chunks = [
            KnowledgeChunk(
                id=f"bounded-embedding-change:{index}",
                entry_id=entry.id,
                chunk_index=index,
                text=text,
            )
            for index, text in enumerate(("github auth", "invoice payment", "refund approval"))
        ]
        await store.create_entry(entry, chunks)
        first = await store.process_embedding_changes(
            "bounded-embedding-index",
            "worker",
            limit=1,
            record_limit=2,
        )
        second = await store.process_embedding_changes(
            "bounded-embedding-index",
            "worker",
            limit=1,
            record_limit=2,
        )
        state = await store.load_change_consumer_state("bounded-embedding-index")
        return first, second, state, provider.calls

    first, second, state, calls = asyncio.run(run())

    assert first.processed_records == 2
    assert first.claimed_changes == 1
    assert first.acknowledged_changes == 0
    assert second.processed_records == 1
    assert second.claimed_changes == 1
    assert second.acknowledged_changes == 1
    assert state is not None
    assert state.cursor_sequence == 1
    assert calls == [["github auth", "invoice payment"], ["refund approval"]]


def test_in_memory_embedding_worker_pages_stale_cleanup_within_record_budget() -> None:
    async def run():
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        entry = KnowledgeEntry(id="bounded-cleanup", text="Old projection set.")
        await store.create_entry(
            entry,
            [
                KnowledgeChunk(
                    id=f"bounded-cleanup:{index}",
                    entry_id=entry.id,
                    chunk_index=index,
                    text=f"old projection {index}",
                )
                for index in range(5)
            ],
        )
        await store.process_embedding_changes(
            "bounded-cleanup-index",
            "worker",
            record_limit=10,
        )
        await store.append_entry_revision(
            entry.model_copy(update={"revision": 2, "text": "Current projection."}),
            [
                KnowledgeChunk(
                    id="bounded-cleanup:current",
                    entry_id=entry.id,
                    entry_revision=2,
                    chunk_index=0,
                    text="current projection",
                )
            ],
            expected_revision=1,
        )
        results = []
        for _ in range(4):
            result = await store.process_embedding_changes(
                "bounded-cleanup-index",
                "worker",
                limit=1,
                record_limit=2,
            )
            results.append(result)
            if result.acknowledged_changes:
                break
        return results, len(store._chunk_embeddings)

    results, remaining = asyncio.run(run())

    assert all(result.processed_records <= 2 for result in results)
    assert sum(result.removed_records for result in results) == 5
    assert results[-1].acknowledged_changes == 1
    assert remaining == 1


def test_in_memory_embedding_backfill_retries_failed_projections_with_a_bounded_result() -> None:
    class ToggleEmbeddingProvider(KeywordEmbeddingProvider):
        fail = True

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            if self.fail:
                raise RuntimeError("embedding provider unavailable")
            return await super().embed_texts(request)

    async def run():
        provider = ToggleEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="retry-failed", text="GitHub proxy."))
        failed = await store.process_embedding_changes("retry-failed-index", "worker")
        provider.fail = False
        recovered = await store.backfill_embeddings(limit=1)
        result = await store.search(KnowledgeQuery(text="auth", mode=KnowledgeSearchMode.SEMANTIC))
        with pytest.raises(ValueError, match="limit.*less than or equal"):
            await store.backfill_embeddings(limit=MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT + 1)
        return failed, recovered, result

    failed, recovered, result = asyncio.run(run())

    assert failed.failed_records == 1
    assert recovered.model_dump() == {
        "scanned_records": 1,
        "indexed_records": 1,
        "failed_records": 0,
        "skipped_records": 0,
        "limit": 1,
        "refresh_existing": False,
        "next_cursor": None,
    }
    assert [hit.entry.id for hit in result.hits] == ["retry-failed"]


def test_in_memory_accepts_precomputed_projection_only_for_current_pending_attempt() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="external-projection", text="GitHub proxy."))
        chunk = (await store.read_chunks("external-projection"))[0]
        identity = knowledge_chunk_embedding_identity(
            chunk,
            embedding_model="test-embedding",
            dimensions=3,
        )
        pending = await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.PENDING,
                attempt_id="external-attempt",
            ),
            expected_sequence=None,
            operation_id="external-projection:pending",
        )
        vector = [1.0, 0.0, 0.0]
        projection = KnowledgeEmbeddingProjection(
            identity=identity,
            readiness_sequence=pending.sequence,
            attempt_id=pending.attempt_id,
            vector=vector,
        )
        vector[0] = 0.0
        stored = await store.store_embedding_projections([projection])
        replayed = await store.store_embedding_projections([projection])
        with pytest.raises(KnowledgeEmbeddingProjectionConflict) as raised:
            await store.store_embedding_projections(
                [projection.model_copy(update={"vector": [0.0, 1.0, 0.0]})]
            )
        await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.READY,
                attempt_id=pending.attempt_id,
            ),
            expected_sequence=pending.sequence,
            operation_id="external-projection:ready",
        )
        stale = await store.store_embedding_projections([projection])
        result = await store.search(
            KnowledgeQuery(text="github", mode=KnowledgeSearchMode.SEMANTIC, min_score=0.0)
        )
        return projection, stored, replayed, raised.value, stale, result, provider.calls

    projection, stored, replayed, conflict, stale, result, provider_calls = asyncio.run(run())

    assert projection.vector == [1.0, 0.0, 0.0]
    assert stored.stored_identities == [projection.identity]
    assert replayed.stored_identities == [projection.identity]
    assert conflict.reason == "attempt_vector_conflict"
    assert stale.stored_identities == []
    assert [hit.entry.id for hit in result.hits] == ["external-projection"]
    assert provider_calls == [["github"]]


def test_in_memory_embedding_refresh_continues_across_bounded_pages() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        observed_at = datetime(2026, 1, 1, tzinfo=UTC)
        for index, text in enumerate(("GitHub proxy.", "Invoice policy.", "SendGrid email.")):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"refresh-page-{index}",
                    text=text,
                    importance=1.0 - index / 10,
                    created_at=observed_at,
                    updated_at=observed_at,
                )
            )
        await store.process_embedding_changes("refresh-pages", "worker")
        provider.calls.clear()

        first = await store.backfill_embeddings(limit=1, refresh_existing=True)
        assert first.next_cursor is not None
        with pytest.raises(ValueError, match="does not match"):
            await store.backfill_embeddings(
                KnowledgeListQuery(namespace="different"),
                limit=1,
                refresh_existing=True,
                cursor=first.next_cursor,
            )
        second = await store.backfill_embeddings(
            limit=1,
            refresh_existing=True,
            cursor=first.next_cursor,
        )
        assert second.next_cursor is not None
        third = await store.backfill_embeddings(
            limit=1,
            refresh_existing=True,
            cursor=second.next_cursor,
        )
        return first, second, third, provider.calls

    first, second, third, calls = asyncio.run(run())

    assert [first.scanned_records, second.scanned_records, third.scanned_records] == [1, 1, 1]
    assert [first.indexed_records, second.indexed_records, third.indexed_records] == [1, 1, 1]
    assert third.next_cursor is None
    assert sorted(call[0] for call in calls) == [
        "GitHub proxy.",
        "Invoice policy.",
        "SendGrid email.",
    ]


def test_in_memory_embedding_search_reports_partial_coverage_without_index_mutation() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="ready", text="GitHub credential proxy."))
        await store.create_entry(KnowledgeEntry(id="pending", text="Invoice payment policy."))
        calls_after_writes = list(provider.calls)
        worker_result = await store.process_embedding_changes(
            "partial-coverage-index",
            "worker",
            limit=1,
        )
        provider.calls.clear()
        search_result = await store.search(
            KnowledgeQuery(
                text="auth broker",
                mode=KnowledgeSearchMode.SEMANTIC,
                min_score=0.0,
            )
        )
        return calls_after_writes, worker_result, search_result, provider.calls

    calls_after_writes, worker_result, search_result, search_calls = asyncio.run(run())

    assert calls_after_writes == []
    assert worker_result.indexed_records == 1
    assert [hit.entry.id for hit in search_result.hits] == ["ready"]
    assert search_result.index_coverage[0].model_dump() == {
        "projection_type": "knowledge_chunk_text",
        "embedding_model": "test-embedding",
        "dimensions": 3,
        "preprocessing_version": "cayu:knowledge-chunk-text:v1",
        "generator": "cayu:canonical-knowledge-chunk",
        "generator_version": "1",
        "index_representation_version": "float32-cosine-v1",
        "eligible_records": 2,
        "ready_records": 1,
        "pending_records": 1,
        "failed_records": 0,
        "high_water_sequence": 2,
        "complete": False,
    }
    assert search_calls == [["auth broker"]]


def test_in_memory_embedding_worker_repairs_vector_committed_before_ready() -> None:
    class CrashAfterVectorStore(InMemoryEmbeddingKnowledgeStore):
        fail_ready_once = True

        async def publish_index_readiness(self, update, **kwargs):
            if self.fail_ready_once and update.state is KnowledgeIndexState.READY:
                self.fail_ready_once = False
                raise RuntimeError("simulated crash after vector commit")
            return await super().publish_index_readiness(update, **kwargs)

    async def run():
        provider = KeywordEmbeddingProvider()
        store = CrashAfterVectorStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="crash-window", text="GitHub proxy."))
        with pytest.raises(RuntimeError, match="simulated crash"):
            await store.process_embedding_changes("crash-window-index", "worker-a")
        calls_after_crash = list(provider.calls)
        retry = await store.process_embedding_changes("crash-window-index", "worker-b")
        result = await store.search(KnowledgeQuery(text="auth", mode=KnowledgeSearchMode.SEMANTIC))
        return calls_after_crash, provider.calls, retry, result

    calls_after_crash, all_calls, retry, result = asyncio.run(run())

    assert calls_after_crash == [["GitHub proxy."]]
    assert all_calls == [["GitHub proxy."], ["auth"]]
    assert retry.indexed_records == 1
    assert retry.acknowledged_changes == 1
    assert [hit.entry.id for hit in result.hits] == ["crash-window"]
    assert result.index_coverage[0].complete is True


def test_in_memory_embedding_worker_replays_ready_projection_after_ack_failure() -> None:
    class AckFailureStore(InMemoryEmbeddingKnowledgeStore):
        fail_ack_once = True

        async def acknowledge_change(self, claim, **kwargs):
            if self.fail_ack_once:
                self.fail_ack_once = False
                raise RuntimeError("simulated crash before outbox acknowledgement")
            return await super().acknowledge_change(claim, **kwargs)

    async def run():
        provider = KeywordEmbeddingProvider()
        store = AckFailureStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="ack-window", text="GitHub proxy."))
        with pytest.raises(RuntimeError, match="simulated crash"):
            await store.process_embedding_changes("ack-window-index", "worker-a")
        calls_after_crash = list(provider.calls)
        retry = await store.process_embedding_changes("ack-window-index", "worker-b")
        return calls_after_crash, provider.calls, retry

    calls_after_crash, all_calls, retry = asyncio.run(run())

    assert calls_after_crash == [["GitHub proxy."]]
    assert all_calls == calls_after_crash
    assert retry.indexed_records == 0
    assert retry.acknowledged_changes == 1


def test_in_memory_embedding_knowledge_store_auto_uses_hybrid_search() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(
            KnowledgeEntry(id="credential_policy", text="Use a credential proxy.")
        )
        await store.create_entry(KnowledgeEntry(id="email_policy", text="SendGrid email guide."))
        await store.process_embedding_changes("auto-index", "worker")
        result = await store.search(KnowledgeQuery(text="credential"))
        return result

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["credential_policy"]
    assert result.hits[0].score_kind == "inmemory_hybrid"
    assert "entry text match" in result.hits[0].reason


def test_in_memory_embedding_knowledge_store_hybrid_labels_keyword_only_hits() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=1.0,
        )
        await store.create_entry(KnowledgeEntry(id="deploy", text="Deployment checklist."))
        await store.process_embedding_changes("hybrid-index", "worker")
        return await store.search(KnowledgeQuery(text="deployment"))

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["deploy"]
    assert result.hits[0].reason == "hybrid keyword match; entry text match"
    assert result.hits[0].score_normalized is None


def test_in_memory_embedding_knowledge_store_min_score_uses_normalized_score() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=0.70,
        )
        await store.create_entry(KnowledgeEntry(id="matching", text="GitHub credential proxy."))
        await store.create_entry(KnowledgeEntry(id="unrelated", text="Deployment checklist."))
        await store.process_embedding_changes("min-score-index", "worker")
        return await store.search(
            KnowledgeQuery(text="auth broker", mode=KnowledgeSearchMode.SEMANTIC)
        )

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["matching"]
    assert result.hits[0].score_normalized == 1.0


def test_in_memory_embedding_knowledge_store_query_min_score_overrides_store_default() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=1.0,
        )
        await store.create_entry(KnowledgeEntry(id="matching", text="GitHub credential proxy."))
        await store.create_entry(KnowledgeEntry(id="orthogonal", text="Invoice payment policy."))
        await store.process_embedding_changes("override-index", "worker")
        return await store.search(
            KnowledgeQuery(
                text="auth broker",
                mode=KnowledgeSearchMode.SEMANTIC,
                min_score=0.0,
            )
        )

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["matching", "orthogonal"]
    assert result.hits[0].score_normalized == 1.0
    assert result.hits[1].score_normalized == 0.5


def test_in_memory_embedding_knowledge_store_refreshes_changed_chunks() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="policy", text="GitHub token policy."))
        await store.process_embedding_changes("refresh-index", "worker")
        first = await store.search(KnowledgeQuery(text="auth", mode=KnowledgeSearchMode.SEMANTIC))
        current = await store.get_entry("policy")
        assert current is not None
        await store.append_entry_revision(
            current.model_copy(
                update={"revision": current.revision + 1, "text": "Invoice payment policy."}
            ),
            expected_revision=current.revision,
        )
        await store.process_embedding_changes("refresh-index", "worker")
        second = await store.search(KnowledgeQuery(text="auth", mode=KnowledgeSearchMode.SEMANTIC))
        third = await store.search(KnowledgeQuery(text="refund", mode=KnowledgeSearchMode.SEMANTIC))
        return first, second, third

    first, second, third = asyncio.run(run())

    assert [hit.entry.id for hit in first.hits] == ["policy"]
    assert second.hits == []
    assert [hit.entry.id for hit in third.hits] == ["policy"]


def test_in_memory_embedding_knowledge_store_drops_replaced_custom_chunk_ids() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(
            KnowledgeEntry(id="policy", text="Policy summary."),
            [
                KnowledgeChunk(
                    id="custom-old", entry_id="policy", chunk_index=0, text="GitHub token."
                )
            ],
        )
        await store.process_embedding_changes("custom-chunk-index", "worker")
        first = await store.search(KnowledgeQuery(text="auth", mode=KnowledgeSearchMode.SEMANTIC))
        current = await store.get_entry("policy")
        assert current is not None
        await store.append_entry_revision(
            current.model_copy(update={"revision": current.revision + 1}),
            [
                KnowledgeChunk(
                    id="custom-new",
                    entry_id="policy",
                    entry_revision=current.revision + 1,
                    chunk_index=0,
                    text="Invoice refund.",
                )
            ],
            expected_revision=current.revision,
        )
        await store.process_embedding_changes("custom-chunk-index", "worker")
        second = await store.search(KnowledgeQuery(text="auth", mode=KnowledgeSearchMode.SEMANTIC))
        third = await store.search(KnowledgeQuery(text="refund", mode=KnowledgeSearchMode.SEMANTIC))
        return first, second, third

    first, second, third = asyncio.run(run())

    assert [hit.chunk.id for hit in first.hits if hit.chunk is not None] == ["custom-old"]
    assert second.hits == []
    assert [hit.chunk.id for hit in third.hits if hit.chunk is not None] == ["custom-new"]


def test_in_memory_embedding_store_rejects_cross_entry_chunk_id_collision_atomically() -> None:
    async def run() -> tuple[
        KnowledgeEntry | None,
        list[KnowledgeChunk],
        dict[str, object],
        dict[str, object],
    ]:
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(
            KnowledgeEntry(id="alpha", text="GitHub auth policy."),
            [
                KnowledgeChunk(
                    id="shared-chunk",
                    entry_id="alpha",
                    chunk_index=0,
                    text="GitHub auth policy.",
                )
            ],
        )
        await store.process_embedding_changes("collision-index", "worker")
        embeddings_before = dict(store._chunk_embeddings)

        with pytest.raises(KnowledgeChunkConflict):
            await store.create_entry(
                KnowledgeEntry(id="beta", text="Invoice policy."),
                [
                    KnowledgeChunk(
                        id="shared-chunk",
                        entry_id="beta",
                        chunk_index=0,
                        text="Invoice policy.",
                    )
                ],
            )

        return (
            await store.get_entry("beta"),
            await store.read_chunks("alpha"),
            embeddings_before,
            dict(store._chunk_embeddings),
        )

    beta, alpha_chunks, embeddings_before, embeddings_after = asyncio.run(run())

    assert beta is None
    assert [(chunk.entry_id, chunk.id) for chunk in alpha_chunks] == [("alpha", "shared-chunk")]
    assert embeddings_after == embeddings_before


def test_in_memory_store_rejects_default_chunk_id_collision_atomically() -> None:
    async def run() -> tuple[KnowledgeEntry | None, list[KnowledgeChunk]]:
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(id="alpha", text="GitHub auth policy."),
            [
                KnowledgeChunk(
                    id="beta:r1:0",
                    entry_id="alpha",
                    chunk_index=0,
                    text="GitHub auth policy.",
                )
            ],
        )

        with pytest.raises(KnowledgeChunkConflict):
            await store.create_entry(KnowledgeEntry(id="beta", text="Invoice policy."))

        return await store.get_entry("beta"), await store.read_chunks("alpha")

    beta, alpha_chunks = asyncio.run(run())

    assert beta is None
    assert [(chunk.entry_id, chunk.id) for chunk in alpha_chunks] == [("alpha", "beta:r1:0")]


def test_in_memory_embedding_knowledge_store_honors_none_terms() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="safe", text="GitHub credential proxy."))
        await store.create_entry(
            KnowledgeEntry(id="excluded", text="GitHub credential proxy deprecated.")
        )
        await store.process_embedding_changes("none-index", "worker")
        return await store.search(
            KnowledgeQuery(
                text="auth broker",
                none_terms=["deprecated"],
                mode=KnowledgeSearchMode.SEMANTIC,
            )
        )

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["safe"]


def test_in_memory_embedding_search_does_not_mutate_index_for_none_term_candidates() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(KnowledgeEntry(id="safe", text="GitHub credential proxy."))
        await store.create_entry(
            KnowledgeEntry(id="excluded", text="GitHub credential proxy deprecated.")
        )
        current = await store.get_entry("safe")
        assert current is not None
        await store.append_entry_revision(
            current.model_copy(update={"revision": current.revision + 1}),
            [
                KnowledgeChunk(
                    id="safe:r2:0",
                    entry_id="safe",
                    entry_revision=current.revision + 1,
                    chunk_index=0,
                    text="GitHub auth proxy.",
                )
            ],
            expected_revision=current.revision,
        )
        await store.process_embedding_changes("none-mutation-index", "worker")
        provider.calls.clear()
        result = await store.search(
            KnowledgeQuery(
                text="auth broker",
                none_terms=["deprecated"],
                mode=KnowledgeSearchMode.SEMANTIC,
            )
        )
        return result, provider.calls

    result, calls = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["safe"]
    assert ["GitHub auth proxy."] not in calls
    assert ["GitHub credential proxy deprecated."] not in calls
    assert ["auth broker"] in calls


def test_in_memory_embedding_knowledge_store_empty_candidates_do_not_embed_query() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        await store.create_entry(
            KnowledgeEntry(
                id="policy",
                text="GitHub credential proxy.",
                labels={"project": "cayu"},
            )
        )
        await store.process_embedding_changes("empty-candidate-index", "worker")
        provider.calls.clear()
        result = await store.search(
            KnowledgeQuery(
                text="auth broker",
                labels={"project": "other"},
                mode=KnowledgeSearchMode.SEMANTIC,
            )
        )
        return result, provider.calls

    result, calls = asyncio.run(run())

    assert result.hits == []
    assert result.total_hits_known == 0
    assert calls == []


def test_in_memory_embedding_knowledge_store_rejects_wrong_dimensions() -> None:
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=_ACCESS_SCOPE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=2,
        )
        await store.create_entry(KnowledgeEntry(id="policy", text="GitHub credential proxy."))
        worker_result = await store.process_embedding_changes("dimension-index", "worker")
        search_result = await store.search(
            KnowledgeQuery(text="GitHub", mode=KnowledgeSearchMode.SEMANTIC)
        )
        return worker_result, search_result

    worker_result, search_result = asyncio.run(run())

    assert worker_result.failed_records == 1
    assert search_result.hits == []
    assert search_result.index_coverage[0].failed_records == 1


def test_text_embedding_usage_rejects_bool_token_counts() -> None:
    with pytest.raises(ValidationError, match="input_tokens"):
        TextEmbeddingResult(
            model="test-embedding",
            embeddings=[TextEmbedding(index=0, vector=[1.0])],
            usage=cast("Any", {"input_tokens": True}),
        )


def test_in_memory_knowledge_store_keyword_search_does_not_match_substrings() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(KnowledgeEntry(id="substring", text="the deployment checklist"))
        await store.create_entry(KnowledgeEntry(id="token", text="he should approve deployment"))
        return await store.search(KnowledgeQuery(text="he"))

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["token"]


def test_in_memory_knowledge_store_title_match_uses_title_preview() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(
                id="entry",
                title="Invoice approval warning",
                text="Operators should inspect extracted fields before sending reminders.",
            )
        )
        return await store.search(KnowledgeQuery(text="invoice approval"))

    result = asyncio.run(run())

    assert len(result.hits) == 1
    assert result.hits[0].reason == "title match"
    assert result.hits[0].text_preview == "Invoice approval warning"


def test_in_memory_knowledge_store_uses_importance_as_ranking_tiebreaker() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        older = datetime(2026, 1, 1, tzinfo=UTC)
        newer = datetime(2026, 1, 2, tzinfo=UTC)
        await store.create_entry(
            KnowledgeEntry(
                id="high_importance",
                text="invoice reminder policy",
                importance=1.0,
                created_at=older,
                updated_at=older,
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="low_importance",
                text="invoice reminder policy",
                importance=0.0,
                created_at=newer,
                updated_at=newer,
            )
        )
        return await store.search(KnowledgeQuery(text="invoice reminder"))

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["high_importance", "low_importance"]


def test_in_memory_knowledge_store_structured_keyword_search() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(id="github_secret", text="GitHub push requires a credential broker.")
        )
        await store.create_entry(
            KnowledgeEntry(id="sendgrid_secret", text="SendGrid email uses a secret proxy.")
        )
        await store.create_entry(
            KnowledgeEntry(id="github_test", text="GitHub test credentials are fixture-only.")
        )
        return await store.search(
            KnowledgeQuery(
                any_terms=["credential", "secret"],
                all_terms=["github push"],
                none_terms=["fixture only"],
            )
        )

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["github_secret"]


def test_in_memory_knowledge_store_phrase_search_conformance() -> None:
    asyncio.run(
        assert_token_exact_phrase_search_conformance(
            InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        )
    )


@pytest.mark.parametrize(
    "mode",
    [
        KnowledgeSearchMode.KEYWORD,
        KnowledgeSearchMode.SEMANTIC,
        KnowledgeSearchMode.HYBRID,
    ],
)
def test_in_memory_knowledge_stores_apply_none_terms_to_the_complete_entry(
    mode: KnowledgeSearchMode,
) -> None:
    async def run() -> None:
        store: KnowledgeStore
        if mode is KnowledgeSearchMode.KEYWORD:
            store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        else:
            store = InMemoryEmbeddingKnowledgeStore(
                access_scope=_ACCESS_SCOPE,
                embedding_provider=KeywordEmbeddingProvider(),
                embedding_model="test-embedding",
                embedding_dimensions=3,
            )
        await assert_entry_wide_none_terms_conformance(store, mode=mode)

    asyncio.run(run())


def test_in_memory_knowledge_store_searches_entry_text_with_custom_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(
                id="broker_summary",
                text="Remote sandbox Git operations need a brokered credential boundary.",
            ),
            [
                KnowledgeChunk(
                    id="broker_summary:0",
                    entry_id="broker_summary",
                    chunk_index=0,
                    text="Implementation details live in the separate chunk body.",
                )
            ],
        )
        return await store.search(KnowledgeQuery(text="brokered credential"))

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["broker_summary"]
    assert result.hits[0].reason == "entry text match"
    assert "brokered credential" in result.hits[0].text_preview


def test_in_memory_knowledge_store_matches_singular_plural_token_variants() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(
                id="remote_git",
                title="Remote sandbox Git credential boundary",
                text=(
                    "GitHub clone or push from a remote sandbox should use a brokered "
                    "proxy. The trusted side injects the credential outside the sandbox."
                ),
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="fixture",
                text="Fixture credentials in local tests are not production guidance.",
            )
        )
        return await store.search(
            KnowledgeQuery(
                all_terms=["GitHub", "credentials"],
                any_terms=["sandbox", "push", "token"],
            )
        )

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["remote_git"]


def test_in_memory_knowledge_store_matches_y_plural_token_variants() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(KnowledgeEntry(id="keys", text="Store API keys securely."))
        await store.create_entry(KnowledgeEntry(id="policies", text="Security policies apply."))
        key_result = await store.search(KnowledgeQuery(text="key"))
        policy_result = await store.search(KnowledgeQuery(text="policy"))
        return key_result, policy_result

    key_result, policy_result = asyncio.run(run())

    assert [hit.entry.id for hit in key_result.hits] == ["keys"]
    assert [hit.entry.id for hit in policy_result.hits] == ["policies"]


def test_in_memory_knowledge_store_all_terms_match_across_entry_document() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(
                id="split_match",
                title="GitHub credential policy",
                text="Remote sandbox operations use a trusted boundary.",
            ),
            [
                KnowledgeChunk(
                    id="split_match:0",
                    entry_id="split_match",
                    chunk_index=0,
                    text="Use a brokered proxy for push operations.",
                )
            ],
        )
        return await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["split_match"]


def test_in_memory_knowledge_store_all_terms_do_not_match_across_unrelated_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(id="split_chunks", text="General operations note."),
            [
                KnowledgeChunk(
                    id="split_chunks:0",
                    entry_id="split_chunks",
                    chunk_index=0,
                    text="GitHub push requires special handling.",
                ),
                KnowledgeChunk(
                    id="split_chunks:1",
                    entry_id="split_chunks",
                    chunk_index=1,
                    text="Use a brokered proxy for remote credentials.",
                ),
            ],
        )
        return await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))

    result = asyncio.run(run())

    assert result.hits == []


def test_in_memory_knowledge_store_lists_entries_and_facets() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await store.create_entry(
            KnowledgeEntry(
                id="runbook",
                namespace="ops",
                kind="procedure",
                labels={"project": "billing"},
                aspects=["payments"],
                text="Payment reminder runbook.",
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="warning",
                namespace="ops",
                kind="warning",
                labels={"project": "billing"},
                aspects=["payments"],
                text="Do not send reminders without approval.",
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="archived",
                namespace="ops",
                kind="warning",
                status=KnowledgeStatus.ARCHIVED,
                text="Old warning.",
            )
        )
        return await store.list_entries(
            KnowledgeListQuery(
                namespace="ops",
                labels={"project": "billing"},
                group_by=KnowledgeListGroup.KIND,
            )
        )

    result = asyncio.run(run())

    assert result.total_entries_known == 2
    assert [item.entry.id for item in result.entries] == ["warning", "runbook"]
    assert [(facet.value, facet.count) for facet in result.facets] == [
        ("procedure", 1),
        ("warning", 1),
    ]


def test_in_memory_knowledge_store_caps_facets() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        for index in range(5):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"entry_{index}",
                    labels={"area": f"area_{index}"},
                    text=f"Knowledge entry {index}.",
                )
            )
        return await store.list_entries(
            KnowledgeListQuery(
                group_by=KnowledgeListGroup.LABEL,
                limit=3,
            )
        )

    result = asyncio.run(run())

    assert len(result.facets) == 3
    assert result.facets_truncated is True
    assert result.truncated is True


def test_knowledge_list_result_validates_result_envelope() -> None:
    query = KnowledgeListQuery(group_by=KnowledgeListGroup.KIND, limit=1)
    entry = KnowledgeEntry(id="entry_1", text="Knowledge entry.")
    facet = KnowledgeFacet(field=KnowledgeListGroup.KIND, value="fact", count=1)

    result = KnowledgeListResult(
        query=query,
        entries=[KnowledgeListItem(entry=entry)],
        facets=[facet],
        limit=query.limit,
        max_bytes=query.max_bytes,
        total_entries_known=1,
    )

    assert result.entries[0].entry.id == "entry_1"
    assert result.facets[0].field == KnowledgeListGroup.KIND

    with pytest.raises(ValidationError, match="entries"):
        KnowledgeListResult(
            query=query,
            entries=[
                KnowledgeListItem(entry=KnowledgeEntry(id="entry_1", text="One.")),
                KnowledgeListItem(entry=KnowledgeEntry(id="entry_2", text="Two.")),
            ],
            limit=query.limit,
            max_bytes=query.max_bytes,
        )

    with pytest.raises(ValidationError, match="query.group_by"):
        KnowledgeListResult(
            query=query,
            facets=[KnowledgeFacet(field=KnowledgeListGroup.LABEL, value="cayu", count=1)],
            limit=query.limit,
            max_bytes=query.max_bytes,
        )


def test_in_memory_knowledge_store_entry_and_chunk_lifecycle() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        entry = KnowledgeEntry(id="runbook", text="Deploy with the blue-green checklist.")
        written = await store.create_entry(entry)
        default_chunks = await store.read_chunks("runbook")
        updated = await store.append_entry_revision(
            entry.model_copy(update={"revision": 2, "text": "Updated deploy checklist."}),
            expected_revision=1,
        )
        updated_default_chunks = await store.read_chunks("runbook")
        revised = await store.append_entry_revision(
            updated.model_copy(update={"revision": 3}),
            [
                KnowledgeChunk(
                    id="runbook:r3:1",
                    entry_id="runbook",
                    entry_revision=3,
                    chunk_index=1,
                    text="Run smoke tests.",
                ),
                KnowledgeChunk(
                    id="runbook:r3:0",
                    entry_id="runbook",
                    entry_revision=3,
                    chunk_index=0,
                    text="Deploy to blue.",
                ),
                KnowledgeChunk(
                    id="runbook:r3:2",
                    entry_id="runbook",
                    entry_revision=3,
                    chunk_index=2,
                    text="Shift traffic.",
                ),
            ],
            expected_revision=2,
        )
        replaced_chunks = await store.read_chunks("runbook", revision=revised.revision)
        window = await store.read_chunks("runbook", chunk_index=1, around=1, max_chunks=3)
        archived = await store.transition_entry_status(
            "runbook",
            expected_revision=3,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
        )
        soft_deleted = await store.delete_entry("runbook", expected_revision=4)
        hard_deleted = await store.delete_entry("runbook", expected_revision=5, hard=True)
        missing = await store.get_entry("runbook")
        return (
            written,
            default_chunks,
            updated,
            updated_default_chunks,
            replaced_chunks,
            window,
            archived,
            soft_deleted,
            hard_deleted,
            missing,
        )

    (
        written,
        default_chunks,
        updated,
        updated_default_chunks,
        replaced_chunks,
        window,
        archived,
        soft_deleted,
        hard_deleted,
        missing,
    ) = asyncio.run(run())

    assert written.id == "runbook"
    assert [chunk.chunk_index for chunk in default_chunks] == [0]
    assert updated.text == "Updated deploy checklist."
    assert [chunk.text for chunk in updated_default_chunks] == ["Updated deploy checklist."]
    assert [chunk.id for chunk in replaced_chunks] == [
        "runbook:r3:0",
        "runbook:r3:1",
        "runbook:r3:2",
    ]
    assert [chunk.id for chunk in window] == [
        "runbook:r3:0",
        "runbook:r3:1",
        "runbook:r3:2",
    ]
    assert archived.status == KnowledgeStatus.ARCHIVED
    assert soft_deleted.status == KnowledgeStatus.DELETED
    assert hard_deleted is not None
    assert hard_deleted.status == KnowledgeStatus.DELETED
    assert missing is None


def test_in_memory_knowledge_store_preserves_custom_single_chunk_on_entry_update() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        current = await store.create_entry(
            KnowledgeEntry(id="doc", text="Document summary.", metadata={"version": 1}),
            [
                KnowledgeChunk(
                    id="doc:0",
                    entry_id="doc",
                    chunk_index=0,
                    text="Custom indexed body.",
                    metadata={"indexer": "custom"},
                )
            ],
        )
        await store.append_entry_revision(
            current.model_copy(update={"revision": 2, "metadata": {"version": 2}}),
            expected_revision=1,
        )
        return await store.read_chunks("doc")

    chunks = asyncio.run(run())

    assert len(chunks) == 1
    assert chunks[0].id == "doc:r2:0"
    assert chunks[0].entry_revision == 2
    assert chunks[0].text == "Custom indexed body."
    assert chunks[0].metadata == {"indexer": "custom"}


def test_in_memory_knowledge_store_status_update_is_monotonic_for_imported_timestamps() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        imported_at = datetime.now(UTC) + timedelta(days=1)
        await store.create_entry(
            KnowledgeEntry(
                id="future_import",
                text="Imported knowledge.",
                created_at=imported_at,
                updated_at=imported_at,
            )
        )
        return await store.transition_entry_status(
            "future_import",
            expected_revision=1,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
        )

    updated = asyncio.run(run())

    assert updated.status == KnowledgeStatus.ARCHIVED
    assert updated.updated_at >= updated.created_at


def test_in_memory_knowledge_store_rejects_invalid_revision_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        missing = KnowledgeEntry(id="missing", revision=2, text="text")
        with pytest.raises(KnowledgeRevisionConflict):
            await store.append_entry_revision(
                missing,
                [
                    KnowledgeChunk(
                        id="chunk",
                        entry_id="missing",
                        entry_revision=2,
                        chunk_index=0,
                        text="text",
                    )
                ],
                expected_revision=1,
            )
        current = await store.create_entry(KnowledgeEntry(id="entry", text="text"))
        successor = current.model_copy(update={"revision": 2})
        with pytest.raises(ValueError, match="cannot be empty"):
            await store.append_entry_revision(successor, [], expected_revision=1)
        with pytest.raises(ValueError, match="belong"):
            await store.append_entry_revision(
                successor,
                [
                    KnowledgeChunk(
                        id="chunk",
                        entry_id="other",
                        entry_revision=2,
                        chunk_index=0,
                        text="text",
                    )
                ],
                expected_revision=1,
            )
        with pytest.raises(ValueError, match="ids"):
            await store.append_entry_revision(
                successor,
                [
                    KnowledgeChunk(
                        id="chunk",
                        entry_id="entry",
                        entry_revision=2,
                        chunk_index=0,
                        text="first",
                    ),
                    KnowledgeChunk(
                        id="chunk",
                        entry_id="entry",
                        entry_revision=2,
                        chunk_index=1,
                        text="second",
                    ),
                ],
                expected_revision=1,
            )
        with pytest.raises(ValueError, match="indexes"):
            await store.append_entry_revision(
                successor,
                [
                    KnowledgeChunk(
                        id="chunk_1",
                        entry_id="entry",
                        entry_revision=2,
                        chunk_index=0,
                        text="first",
                    ),
                    KnowledgeChunk(
                        id="chunk_2",
                        entry_id="entry",
                        entry_revision=2,
                        chunk_index=0,
                        text="second",
                    ),
                ],
                expected_revision=1,
            )

    asyncio.run(run())


def test_in_memory_knowledge_store_search_result_reports_truncation() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        for index in range(3):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"entry_{index}",
                    text="billing reminder policy",
                    created_at=datetime(2026, 1, index + 1, tzinfo=UTC),
                    updated_at=datetime(2026, 1, index + 1, tzinfo=UTC),
                )
            )
        await store.create_entry(
            KnowledgeEntry(
                id="single_entry",
                text="policy content that exceeds the byte cap",
                labels={"single": "true"},
            )
        )
        limit_result = await store.search(KnowledgeQuery(text="billing", limit=2))
        byte_result = await store.search(KnowledgeQuery(text="billing", max_bytes=1))
        single_hit_byte_result = await store.search(
            KnowledgeQuery(text="policy", labels={"single": "true"}, max_bytes=4)
        )
        return limit_result, byte_result, single_hit_byte_result

    limit_result, byte_result, single_hit_byte_result = asyncio.run(run())

    assert [hit.entry.id for hit in limit_result.hits] == ["entry_2", "entry_1"]
    assert limit_result.total_hits_known == 3
    assert not hasattr(limit_result, "total_hits")
    assert limit_result.truncated is True
    assert len(byte_result.hits) == 1
    assert byte_result.truncated is True
    assert [hit.entry.id for hit in single_hit_byte_result.hits] == ["single_entry"]
    assert single_hit_byte_result.hits[0].text_preview == "poli"
    assert single_hit_byte_result.truncated is True


def test_in_memory_knowledge_store_get_enforces_query_scope_and_owns_copies() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        entry = KnowledgeEntry(
            id="entry_1",
            text="Project-specific memory.",
            namespace="projects",
            labels={"project": "alpha"},
        )
        await store.create_entry(entry)
        entry.labels["project"] = "mutated-outside"
        loaded = await store.get_entry("entry_1")
        allowed_query = KnowledgeQuery(
            text="memory", namespace="projects", labels={"project": "alpha"}
        )
        denied_query = KnowledgeQuery(
            text="memory", namespace="projects", labels={"project": "beta"}
        )
        allowed = (
            loaded
            if loaded is not None
            and loaded.labels.get("project") == allowed_query.labels["project"]
            else None
        )
        denied = (
            loaded
            if loaded is not None and loaded.labels.get("project") == denied_query.labels["project"]
            else None
        )
        assert allowed is not None
        allowed.labels["project"] = "mutated-copy"
        loaded_again = await store.get_entry("entry_1")
        return allowed, denied, loaded_again

    allowed, denied, loaded_again = asyncio.run(run())

    assert allowed.labels["project"] == "mutated-copy"
    assert denied is None
    assert loaded_again is not None
    assert loaded_again.text == "Project-specific memory."
    assert loaded_again.labels == {"project": "alpha"}


def test_copy_knowledge_entry_rejects_subclasses_before_attribute_access() -> None:
    class BadEntry(KnowledgeEntry):
        def __getattribute__(self, name):
            if name == "id":
                raise RuntimeError("entry id access should not run")
            return super().__getattribute__(name)

    entry = BadEntry.model_construct(
        id="entry_1",
        text="memory",
        namespace="default",
        labels={},
        kind="fact",
        visibility=KnowledgeVisibility.GLOBAL,
        status=KnowledgeStatus.ACTIVE,
        created_by_type=KnowledgeActorType.APP,
        created_by="app",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata={},
    )

    with pytest.raises(TypeError, match="KnowledgeEntry"):
        copy_knowledge_entry(entry)

    with pytest.raises(TypeError, match="KnowledgeEntry"):
        KnowledgeHit(entry=entry)
