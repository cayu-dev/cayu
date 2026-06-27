from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cayu.storage import (
    BUILTIN_KNOWLEDGE_KINDS,
    InMemoryKnowledgeStore,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeVisibility,
)
from cayu.storage.memory import copy_knowledge_entry


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


def test_knowledge_hit_owns_copies() -> None:
    metadata = {"nested": {"value": "original"}}
    entry = KnowledgeEntry(id="entry_1", text="billing memory", metadata=metadata)
    hit = KnowledgeHit(entry=entry, score=2.0)

    metadata["nested"]["value"] = "mutated"
    entry.metadata["nested"]["value"] = "mutated again"

    assert hit.entry.id == "entry_1"
    assert hit.entry.metadata == {"nested": {"value": "original"}}


def test_knowledge_hit_rejects_chunk_for_different_entry() -> None:
    entry = KnowledgeEntry(id="entry_1", text="billing memory")
    chunk = KnowledgeChunk(id="chunk_1", entry_id="entry_2", chunk_index=0, text="other")

    with pytest.raises(ValidationError, match="chunk.entry_id"):
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
        store = InMemoryKnowledgeStore()
        await store.put_entry(
            KnowledgeEntry(
                id="invoice_warning",
                text="Do not send invoice reminders when the PO number is missing.",
                namespace="ops",
                labels={"project": "invoice_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
            )
        )
        await store.put_entry(
            KnowledgeEntry(
                id="other_project_warning",
                text="Do not send invoice reminders when the PO number is missing.",
                namespace="ops",
                labels={"project": "other_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
            )
        )
        await store.put_entry(
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
            ]
        )


def test_in_memory_knowledge_store_excludes_non_active_and_expired_by_default() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await store.put_entry(
            KnowledgeEntry(
                id="active",
                text="Active deployment warning.",
                namespace="deploy",
                kind="warning",
            )
        )
        await store.put_entry(
            KnowledgeEntry(
                id="pending",
                text="Pending deployment warning.",
                namespace="deploy",
                kind="warning",
                status=KnowledgeStatus.PENDING,
            )
        )
        await store.put_entry(
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


def test_in_memory_knowledge_store_chunks_are_bounded_and_scope_checked() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        entry = KnowledgeEntry(
            id="long_doc",
            text="Short summary.",
            namespace="docs",
            labels={"project": "agent_a"},
            kind="document",
        )
        await store.put_entry_with_chunks(
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
        store = InMemoryKnowledgeStore()
        await store.put_entry_with_chunks(
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
        store = InMemoryKnowledgeStore()
        await store.put_entry(KnowledgeEntry(id="entry_1", text="memory"))
        with pytest.raises(ValueError, match="around"):
            await store.read_chunks("entry_1", around=1)

    asyncio.run(run())


def test_in_memory_knowledge_store_rejects_unsupported_search_modes() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await store.put_entry(KnowledgeEntry(id="entry_1", text="billing memory"))
        with pytest.raises(ValueError, match="supports only auto and keyword"):
            await store.search(KnowledgeQuery(text="billing", mode=KnowledgeSearchMode.SEMANTIC))

    asyncio.run(run())


def test_in_memory_knowledge_store_keyword_search_does_not_match_substrings() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await store.put_entry(KnowledgeEntry(id="substring", text="the deployment checklist"))
        await store.put_entry(KnowledgeEntry(id="token", text="he should approve deployment"))
        return await store.search(KnowledgeQuery(text="he"))

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["token"]


def test_in_memory_knowledge_store_title_match_uses_title_preview() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await store.put_entry(
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
        store = InMemoryKnowledgeStore()
        older = datetime(2026, 1, 1, tzinfo=UTC)
        newer = datetime(2026, 1, 2, tzinfo=UTC)
        await store.put_entry(
            KnowledgeEntry(
                id="high_importance",
                text="invoice reminder policy",
                importance=1.0,
                created_at=older,
                updated_at=older,
            )
        )
        await store.put_entry(
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
        store = InMemoryKnowledgeStore()
        await store.put_entry(
            KnowledgeEntry(id="github_secret", text="GitHub push requires a credential broker.")
        )
        await store.put_entry(
            KnowledgeEntry(id="sendgrid_secret", text="SendGrid email uses a secret proxy.")
        )
        await store.put_entry(
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


def test_in_memory_knowledge_store_searches_entry_text_with_custom_chunks() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await store.put_entry_with_chunks(
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
        store = InMemoryKnowledgeStore()
        await store.put_entry(
            KnowledgeEntry(
                id="remote_git",
                title="Remote sandbox Git credential boundary",
                text=(
                    "GitHub clone or push from a remote sandbox should use a brokered "
                    "proxy. The trusted side injects the credential outside the sandbox."
                ),
            )
        )
        await store.put_entry(
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
        store = InMemoryKnowledgeStore()
        await store.put_entry(KnowledgeEntry(id="keys", text="Store API keys securely."))
        await store.put_entry(KnowledgeEntry(id="policies", text="Security policies apply."))
        key_result = await store.search(KnowledgeQuery(text="key"))
        policy_result = await store.search(KnowledgeQuery(text="policy"))
        return key_result, policy_result

    key_result, policy_result = asyncio.run(run())

    assert [hit.entry.id for hit in key_result.hits] == ["keys"]
    assert [hit.entry.id for hit in policy_result.hits] == ["policies"]


def test_in_memory_knowledge_store_all_terms_match_across_entry_document() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await store.put_entry_with_chunks(
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
        store = InMemoryKnowledgeStore()
        await store.put_entry_with_chunks(
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
        store = InMemoryKnowledgeStore()
        await store.put_entry(
            KnowledgeEntry(
                id="runbook",
                namespace="ops",
                kind="procedure",
                labels={"project": "billing"},
                aspects=["payments"],
                text="Payment reminder runbook.",
            )
        )
        await store.put_entry(
            KnowledgeEntry(
                id="warning",
                namespace="ops",
                kind="warning",
                labels={"project": "billing"},
                aspects=["payments"],
                text="Do not send reminders without approval.",
            )
        )
        await store.put_entry(
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
        store = InMemoryKnowledgeStore()
        for index in range(5):
            await store.put_entry(
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
        store = InMemoryKnowledgeStore()
        entry = KnowledgeEntry(id="runbook", text="Deploy with the blue-green checklist.")
        written = await store.put_entry(entry)
        default_chunks = await store.read_chunks("runbook")
        updated = await store.put_entry(
            KnowledgeEntry(id="runbook", text="Updated deploy checklist.")
        )
        updated_default_chunks = await store.read_chunks("runbook")
        replaced_chunks = await store.replace_chunks(
            "runbook",
            [
                KnowledgeChunk(
                    id="runbook:1", entry_id="runbook", chunk_index=1, text="Run smoke tests."
                ),
                KnowledgeChunk(
                    id="runbook:0", entry_id="runbook", chunk_index=0, text="Deploy to blue."
                ),
                KnowledgeChunk(
                    id="runbook:2", entry_id="runbook", chunk_index=2, text="Shift traffic."
                ),
            ],
        )
        window = await store.read_chunks("runbook", chunk_index=1, around=1, max_chunks=3)
        archived = await store.update_entry_status("runbook", KnowledgeStatus.ARCHIVED)
        soft_deleted = await store.delete_entry("runbook")
        hard_deleted = await store.delete_entry("runbook", hard=True)
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
    assert [chunk.id for chunk in replaced_chunks] == ["runbook:0", "runbook:1", "runbook:2"]
    assert [chunk.id for chunk in window] == ["runbook:0", "runbook:1", "runbook:2"]
    assert archived.status == KnowledgeStatus.ARCHIVED
    assert soft_deleted.status == KnowledgeStatus.DELETED
    assert hard_deleted is not None
    assert hard_deleted.status == KnowledgeStatus.DELETED
    assert missing is None


def test_in_memory_knowledge_store_preserves_custom_single_chunk_on_entry_update() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await store.put_entry_with_chunks(
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
        await store.put_entry(
            KnowledgeEntry(id="doc", text="Document summary.", metadata={"version": 2})
        )
        return await store.read_chunks("doc")

    chunks = asyncio.run(run())

    assert len(chunks) == 1
    assert chunks[0].id == "doc:0"
    assert chunks[0].text == "Custom indexed body."
    assert chunks[0].metadata == {"indexer": "custom"}


def test_in_memory_knowledge_store_status_update_is_monotonic_for_imported_timestamps() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        imported_at = datetime.now(UTC) + timedelta(days=1)
        await store.put_entry(
            KnowledgeEntry(
                id="future_import",
                text="Imported knowledge.",
                created_at=imported_at,
                updated_at=imported_at,
            )
        )
        return await store.update_entry_status("future_import", KnowledgeStatus.ARCHIVED)

    updated = asyncio.run(run())

    assert updated.status == KnowledgeStatus.ARCHIVED
    assert updated.updated_at >= updated.created_at


def test_in_memory_knowledge_store_rejects_invalid_chunk_replacement() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        with pytest.raises(KeyError):
            await store.replace_chunks(
                "missing",
                [KnowledgeChunk(id="chunk", entry_id="missing", chunk_index=0, text="text")],
            )
        await store.put_entry(KnowledgeEntry(id="entry", text="text"))
        with pytest.raises(ValueError, match="cannot be empty"):
            await store.replace_chunks("entry", [])
        with pytest.raises(ValueError, match="belong"):
            await store.replace_chunks(
                "entry",
                [KnowledgeChunk(id="chunk", entry_id="other", chunk_index=0, text="text")],
            )
        with pytest.raises(ValueError, match="ids"):
            await store.replace_chunks(
                "entry",
                [
                    KnowledgeChunk(id="chunk", entry_id="entry", chunk_index=0, text="first"),
                    KnowledgeChunk(id="chunk", entry_id="entry", chunk_index=1, text="second"),
                ],
            )
        with pytest.raises(ValueError, match="indexes"):
            await store.replace_chunks(
                "entry",
                [
                    KnowledgeChunk(id="chunk_1", entry_id="entry", chunk_index=0, text="first"),
                    KnowledgeChunk(id="chunk_2", entry_id="entry", chunk_index=0, text="second"),
                ],
            )

    asyncio.run(run())


def test_in_memory_knowledge_store_search_result_reports_truncation() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        for index in range(3):
            await store.put_entry(
                KnowledgeEntry(
                    id=f"entry_{index}",
                    text="billing reminder policy",
                    created_at=datetime(2026, 1, index + 1, tzinfo=UTC),
                    updated_at=datetime(2026, 1, index + 1, tzinfo=UTC),
                )
            )
        await store.put_entry(
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
        store = InMemoryKnowledgeStore()
        entry = KnowledgeEntry(
            id="entry_1",
            text="Project-specific memory.",
            namespace="projects",
            labels={"project": "alpha"},
        )
        await store.put_entry(entry)
        entry.text = "mutated outside"
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
        allowed.text = "mutated copy"
        loaded_again = await store.get_entry("entry_1")
        return allowed, denied, loaded_again

    allowed, denied, loaded_again = asyncio.run(run())

    assert allowed.text == "mutated copy"
    assert denied is None
    assert loaded_again is not None
    assert loaded_again.text == "Project-specific memory."


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
