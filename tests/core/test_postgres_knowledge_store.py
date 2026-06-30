from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListGroup,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeVisibility,
)
from cayu.storage.migrations import LATEST_REVISION, MIN_SUPPORTED_REVISION, SchemaMode

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
    "cayu_knowledge_embeddings",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


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


async def _drop_all(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_store(dsn: str):
    from cayu import PostgresKnowledgeStore

    return PostgresKnowledgeStore(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)


def _new_embedding_store(dsn: str, provider: KeywordEmbeddingProvider):
    from cayu import PostgresEmbeddingKnowledgeStore

    return PostgresEmbeddingKnowledgeStore(
        dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
        embedding_provider=provider,
        embedding_model="test-embedding",
        embedding_dimensions=3,
        semantic_min_score=0.70,
    )


async def _skip_if_pgvector_unavailable(dsn: str) -> None:
    import psycopg

    try:
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.commit()
    except Exception as exc:
        pytest.skip(f"pgvector extension is not available: {exc}")


def _run(dsn: str, coro_factory):
    async def runner():
        await _drop_all(dsn)
        store = _new_store(dsn)
        try:
            return await coro_factory(store)
        finally:
            await store.close()

    return asyncio.run(runner())


def test_postgres_knowledge_store_persists_entries_chunks_and_filters(postgres_dsn: str) -> None:
    async def ops(store):
        await store.put_entry_with_chunks(
            KnowledgeEntry(
                id="invoice_warning",
                text="Do not send invoice reminders when the PO number is missing.",
                namespace="ops",
                labels={"project": "invoice_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
                aspects=["finance"],
                impact_targets=["finance.reminders"],
                source_type="manual",
                source_id="invoice_rules",
                importance=0.8,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            [
                KnowledgeChunk(
                    id="invoice_warning:0",
                    entry_id="invoice_warning",
                    chunk_index=0,
                    text="Invoice reminders require a PO number.",
                    source_uri="manual://invoice_rules",
                )
            ],
        )
        await store.put_entry(
            KnowledgeEntry(
                id="other_project_warning",
                text="Invoice reminders require a PO number.",
                namespace="ops",
                labels={"project": "other_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
            )
        )

        loaded = await store.get_entry("invoice_warning")
        result = await store.search(
            KnowledgeQuery(
                text="invoice reminders",
                namespace="ops",
                labels={"project": "invoice_agent"},
                kinds=["warning"],
                visibilities=[KnowledgeVisibility.PROJECT],
                aspects=["finance"],
                impact_targets=["finance.reminders"],
                source_type="manual",
                source_id="invoice_rules",
            )
        )
        denied = await store.search(
            KnowledgeQuery(
                text="invoice reminders",
                namespace="ops",
                labels={"project": "missing"},
            )
        )
        return loaded, result, denied

    loaded, result, denied = _run(postgres_dsn, ops)

    assert loaded is not None
    assert loaded.labels == {"project": "invoice_agent", "user": "alice"}
    assert loaded.aspects == ["finance"]
    assert loaded.impact_targets == ["finance.reminders"]
    assert [hit.entry.id for hit in result.hits] == ["invoice_warning"]
    assert result.hits[0].chunk is not None
    assert result.hits[0].chunk.id == "invoice_warning:0"
    assert result.hits[0].score_kind == "postgres_full_text"
    assert result.total_hits_known == 1
    assert denied.hits == []


def test_postgres_embedding_knowledge_store_persists_semantic_vectors(postgres_dsn: str) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            await store.put_entry(
                KnowledgeEntry(
                    id="git_policy",
                    text="Use a credential broker for GitHub auth from remote sandboxes.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                    aspects=["credentials", "git"],
                )
            )
            await store.put_entry(
                KnowledgeEntry(
                    id="invoice_policy",
                    text="Invoice refunds require payment approval.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                    aspects=["invoices"],
                )
            )
            result = await store.search(
                KnowledgeQuery(
                    text="auth broker",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
        finally:
            await store.close()

        reopened_provider = KeywordEmbeddingProvider()
        reopened = _new_embedding_store(postgres_dsn, reopened_provider)
        try:
            reopened_result = await reopened.search(
                KnowledgeQuery(
                    text="github credential proxy",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
        finally:
            await reopened.close()
        return result, reopened_result, provider.calls, reopened_provider.calls

    result, reopened_result, calls, reopened_calls = asyncio.run(ops())

    assert [hit.entry.id for hit in result.hits] == ["git_policy"]
    assert result.hits[0].score_kind == "postgres_semantic"
    assert result.hits[0].chunk is not None
    assert [hit.entry.id for hit in reopened_result.hits] == ["git_policy"]
    assert reopened_calls == [["github credential proxy"]]
    assert calls[:2] == [
        ["Use a credential broker for GitHub auth from remote sandboxes."],
        ["Invoice refunds require payment approval."],
    ]


def test_postgres_embedding_knowledge_store_skips_hnsw_for_large_dimensions(
    postgres_dsn: str,
) -> None:
    async def ops():
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="large-test-embedding",
            embedding_dimensions=3072,
        )
        try:
            await store._ensure_ready()
        finally:
            await store.close()

        import psycopg

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT to_regclass('idx_cayu_knowledge_embeddings_embedding_hnsw')")
            row = await cur.fetchone()
        return None if row is None else row[0]

    index_name = asyncio.run(ops())

    assert index_name is None


def test_postgres_embedding_knowledge_store_reports_dimension_mismatch_before_indexing(
    postgres_dsn: str,
) -> None:
    async def ops():
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        first = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="large-test-embedding",
            embedding_dimensions=3072,
        )
        try:
            await first._ensure_ready()
        finally:
            await first.close()

        second = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="small-test-embedding",
            embedding_dimensions=3,
        )
        try:
            with pytest.raises(RuntimeError, match="dimension mismatch"):
                await second._ensure_ready()
        finally:
            await second.close()

    asyncio.run(ops())


def test_postgres_embedding_knowledge_store_backfills_existing_chunks(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        base = _new_store(postgres_dsn)
        try:
            await base.put_entry(
                KnowledgeEntry(
                    id="git_policy",
                    text="Use a credential broker for GitHub auth from remote sandboxes.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                )
            )
            await base.put_entry(
                KnowledgeEntry(
                    id="invoice_policy",
                    text="GitHub token pushes should use the broker.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                )
            )
            await base.put_entry(
                KnowledgeEntry(
                    id="other_policy",
                    text="Invoice refunds require payment approval.",
                    namespace="ops",
                    labels={"project": "other"},
                    kind="procedure",
                )
            )
        finally:
            await base.close()

        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            before = await store.search(
                KnowledgeQuery(
                    text="auth broker",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
            first_backfill = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=1,
            )
            after = await store.search(
                KnowledgeQuery(
                    text="auth broker",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
            second_backfill = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=1,
            )
            third_backfill = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=10,
            )
            refresh = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=10,
                refresh_existing=True,
            )
        finally:
            await store.close()
        return (
            before,
            first_backfill,
            after,
            second_backfill,
            third_backfill,
            refresh,
            provider.calls,
        )

    before, first_backfill, after, second_backfill, third_backfill, refresh, calls = asyncio.run(
        ops()
    )

    assert before.hits == []
    assert first_backfill.scanned_chunks == 1
    assert first_backfill.embedded_chunks == 1
    assert first_backfill.skipped_current_chunks == 0
    assert len(after.hits) == 1
    assert after.hits[0].entry.id in {"git_policy", "invoice_policy"}
    assert second_backfill.scanned_chunks == 1
    assert second_backfill.embedded_chunks == 1
    assert second_backfill.skipped_current_chunks == 0
    assert third_backfill.scanned_chunks == 0
    assert third_backfill.embedded_chunks == 0
    assert third_backfill.skipped_current_chunks == 0
    assert refresh.scanned_chunks == 2
    assert refresh.embedded_chunks == 2
    embed_calls = [call for call in calls if call != ["auth broker"]]
    assert len(embed_calls) == 3
    assert sorted(embed_calls[:2]) == sorted(
        [
            ["GitHub token pushes should use the broker."],
            ["Use a credential broker for GitHub auth from remote sandboxes."],
        ]
    )
    assert sorted(embed_calls[2]) == sorted(
        [
            "GitHub token pushes should use the broker.",
            "Use a credential broker for GitHub auth from remote sandboxes.",
        ]
    )


def test_postgres_knowledge_store_defaults_hide_inactive_and_expired(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.put_entry(KnowledgeEntry(id="active", text="deployment warning"))
        await store.put_entry(
            KnowledgeEntry(
                id="pending",
                text="deployment warning",
                status=KnowledgeStatus.PENDING,
            )
        )
        await store.put_entry(
            KnowledgeEntry(
                id="expired",
                text="deployment warning",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        active = await store.search(KnowledgeQuery(text="deployment"))
        pending = await store.search(
            KnowledgeQuery(text="deployment", statuses=[KnowledgeStatus.PENDING])
        )
        expired = await store.search(KnowledgeQuery(text="deployment", include_expired=True))
        return active, pending, expired

    active, pending, expired = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in active.hits] == ["active"]
    assert [hit.entry.id for hit in pending.hits] == ["pending"]
    assert [hit.entry.id for hit in expired.hits] == ["expired", "active"]


def test_postgres_knowledge_store_preserves_custom_chunks_on_entry_update(
    postgres_dsn: str,
) -> None:
    async def ops(store):
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
        chunks = await store.read_chunks("doc")
        result = await store.search(KnowledgeQuery(text="custom indexed"))
        return chunks, result

    chunks, result = _run(postgres_dsn, ops)

    assert len(chunks) == 1
    assert chunks[0].text == "Custom indexed body."
    assert chunks[0].metadata == {"indexer": "custom"}
    assert [hit.entry.id for hit in result.hits] == ["doc"]


def test_postgres_knowledge_store_empty_kind_filter_returns_no_matches(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.put_entry(KnowledgeEntry(id="entry", text="billing memory"))
        return await store.search(KnowledgeQuery(text="billing", kinds=[]))

    result = _run(postgres_dsn, ops)

    assert result.hits == []
    assert result.total_hits_known == 0


def test_postgres_knowledge_store_search_reports_preview_truncation(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.put_entry(KnowledgeEntry(id="entry", text="billing memory has more text"))
        return await store.search(KnowledgeQuery(text="billing", max_bytes=7))

    result = _run(postgres_dsn, ops)

    assert len(result.hits) == 1
    assert result.hits[0].text_preview == "billing"
    assert result.truncated is True


def test_postgres_knowledge_store_search_dedupes_across_large_chunk_matches(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.put_entry_with_chunks(
            KnowledgeEntry(id="large", text="invoice corpus"),
            [
                KnowledgeChunk(
                    id=f"large:{index}",
                    entry_id="large",
                    chunk_index=index,
                    text=f"invoice repeated chunk {index}",
                )
                for index in range(1200)
            ],
        )
        await store.put_entry(KnowledgeEntry(id="small", text="invoice policy"))
        return await store.search(KnowledgeQuery(text="invoice", limit=2))

    result = _run(postgres_dsn, ops)

    assert {hit.entry.id for hit in result.hits} == {"large", "small"}
    assert result.total_hits_known == 2
    assert result.truncated is False


def test_postgres_knowledge_store_structured_keyword_search(postgres_dsn: str) -> None:
    async def ops(store):
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

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["github_secret"]


def test_postgres_knowledge_store_searches_entry_text_with_custom_chunks(
    postgres_dsn: str,
) -> None:
    async def ops(store):
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

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["broker_summary"]
    assert result.hits[0].reason == "entry text match"
    assert "brokered credential" in result.hits[0].text_preview


def test_postgres_knowledge_store_matches_singular_plural_token_variants(
    postgres_dsn: str,
) -> None:
    async def ops(store):
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

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["remote_git"]


def test_postgres_knowledge_store_matches_y_plural_token_variants(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.put_entry(KnowledgeEntry(id="keys", text="Store API keys securely."))
        await store.put_entry(KnowledgeEntry(id="policies", text="Security policies apply."))
        key_result = await store.search(KnowledgeQuery(text="key"))
        policy_result = await store.search(KnowledgeQuery(text="policy"))
        return key_result, policy_result

    key_result, policy_result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in key_result.hits] == ["keys"]
    assert [hit.entry.id for hit in policy_result.hits] == ["policies"]


def test_postgres_knowledge_store_all_terms_match_across_entry_document(
    postgres_dsn: str,
) -> None:
    async def ops(store):
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

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["split_match"]


def test_postgres_knowledge_store_all_terms_do_not_match_across_unrelated_chunks(
    postgres_dsn: str,
) -> None:
    async def ops(store):
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

    result = _run(postgres_dsn, ops)

    assert result.hits == []


def test_postgres_knowledge_store_lists_entries_and_facets(postgres_dsn: str) -> None:
    async def ops(store):
        await store.put_entry(
            KnowledgeEntry(
                id="runbook",
                namespace="ops",
                kind="procedure",
                labels={"project": "billing"},
                text="Payment reminder runbook.",
            )
        )
        await store.put_entry(
            KnowledgeEntry(
                id="warning",
                namespace="ops",
                kind="warning",
                labels={"project": "billing"},
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

    result = _run(postgres_dsn, ops)

    assert result.total_entries_known == 2
    assert {item.entry.id for item in result.entries} == {"runbook", "warning"}
    assert [(facet.value, facet.count) for facet in result.facets] == [
        ("procedure", 1),
        ("warning", 1),
    ]


def test_postgres_knowledge_store_caps_facets(postgres_dsn: str) -> None:
    async def ops(store):
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

    result = _run(postgres_dsn, ops)

    assert len(result.facets) == 3
    assert result.facets_truncated is True
    assert result.truncated is True


def test_postgres_knowledge_store_chunk_windows_and_truncation(postgres_dsn: str) -> None:
    async def ops(store):
        await store.put_entry_with_chunks(
            KnowledgeEntry(id="doc", text="summary"),
            [
                KnowledgeChunk(id="chunk_0", entry_id="doc", chunk_index=0, text="alpha beta"),
                KnowledgeChunk(
                    id="chunk_1",
                    entry_id="doc",
                    chunk_index=1,
                    text="gamma delta",
                    content_hash="full-hash",
                ),
                KnowledgeChunk(id="chunk_2", entry_id="doc", chunk_index=2, text="epsilon zeta"),
            ],
        )
        window = await store.read_chunks("doc", chunk_index=1, around=1, max_chunks=3)
        centered = await store.read_chunks("doc", chunk_index=2, around=10, max_chunks=1)
        truncated = await store.read_chunks("doc", chunk_index=1, around=0, max_bytes=5)
        return window, centered, truncated

    window, centered, truncated = _run(postgres_dsn, ops)

    assert [chunk.id for chunk in window] == ["chunk_0", "chunk_1", "chunk_2"]
    assert [chunk.id for chunk in centered] == ["chunk_2"]
    assert truncated[0].text == "gamma"
    assert truncated[0].content_hash is None


def test_postgres_knowledge_store_title_match_uses_title_preview(postgres_dsn: str) -> None:
    async def ops(store):
        await store.put_entry(
            KnowledgeEntry(
                id="title_match",
                title="Invoice approval warning",
                text="The body does not include the searched approval terms.",
            )
        )
        return await store.search(KnowledgeQuery(text="invoice approval"))

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["title_match"]
    assert result.hits[0].reason == "title match"
    assert result.hits[0].text_preview == "Invoice approval warning"


def test_postgres_knowledge_store_updates_status_and_deletes_entries(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.put_entry(KnowledgeEntry(id="runbook", text="deployment rollback procedure"))
        archived = await store.update_entry_status("runbook", KnowledgeStatus.ARCHIVED)
        archived_search = await store.search(
            KnowledgeQuery(text="deployment", statuses=[KnowledgeStatus.ARCHIVED])
        )
        soft_deleted = await store.delete_entry("runbook")
        deleted_search = await store.search(
            KnowledgeQuery(text="deployment", statuses=[KnowledgeStatus.DELETED])
        )
        hard_deleted = await store.delete_entry("runbook", hard=True)
        missing = await store.get_entry("runbook")
        missing_delete = await store.delete_entry("runbook", hard=True)
        return (
            archived,
            archived_search,
            soft_deleted,
            deleted_search,
            hard_deleted,
            missing,
            missing_delete,
        )

    (
        archived,
        archived_search,
        soft_deleted,
        deleted_search,
        hard_deleted,
        missing,
        missing_delete,
    ) = _run(postgres_dsn, ops)

    assert archived.status is KnowledgeStatus.ARCHIVED
    assert [hit.entry.id for hit in archived_search.hits] == ["runbook"]
    assert soft_deleted is not None
    assert soft_deleted.status is KnowledgeStatus.DELETED
    assert [hit.entry.id for hit in deleted_search.hits] == ["runbook"]
    assert hard_deleted is not None
    assert hard_deleted.status is KnowledgeStatus.DELETED
    assert missing is None
    assert missing_delete is None


def test_postgres_knowledge_store_rejects_invalid_chunk_replacement(
    postgres_dsn: str,
) -> None:
    async def ops(store):
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

    _run(postgres_dsn, ops)


def test_postgres_knowledge_store_rejects_unsupported_search_modes(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.put_entry(KnowledgeEntry(id="entry", text="billing memory"))
        with pytest.raises(ValueError, match="supports only auto and keyword"):
            await store.search(KnowledgeQuery(text="billing", mode=KnowledgeSearchMode.SEMANTIC))

    _run(postgres_dsn, ops)


def test_postgres_knowledge_schema_migrates_and_coexists_with_session_store(
    postgres_dsn: str,
) -> None:
    async def ops():
        import psycopg

        from cayu import PostgresKnowledgeStore, PostgresSessionStore
        from cayu.core import Message
        from cayu.runtime import RunRequest, SessionIdentity

        await _drop_all(postgres_dsn)
        session_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await session_store.create(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "hi")]),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        finally:
            await session_store.close()

        knowledge_store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await knowledge_store.put_entry(
                KnowledgeEntry(id="entry", text="shared database memory")
            )
            result = await knowledge_store.search(KnowledgeQuery(text="shared database"))
        finally:
            await knowledge_store.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT revision, compatible_from FROM cayu_schema_migrations ORDER BY revision"
            )
            revisions = [(int(row[0]), int(row[1])) for row in await cur.fetchall()]
            await cur.execute("SELECT to_regclass('cayu_knowledge_entries')")
            knowledge_row = await cur.fetchone()
            assert knowledge_row is not None
            knowledge_table = knowledge_row[0]
            await cur.execute("SELECT to_regclass('cayu_knowledge_chunks')")
            chunks_row = await cur.fetchone()
            assert chunks_row is not None
            chunks_table = chunks_row[0]
        return result, revisions, knowledge_table, chunks_table

    result, revisions, knowledge_table, chunks_table = asyncio.run(ops())

    assert [hit.entry.id for hit in result.hits] == ["entry"]
    assert revisions[-1] == (LATEST_REVISION, MIN_SUPPORTED_REVISION)
    assert knowledge_table == "cayu_knowledge_entries"
    assert chunks_table == "cayu_knowledge_chunks"
