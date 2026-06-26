from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeVisibility,
)
from cayu.storage.migrations import LATEST_REVISION, MIN_SUPPORTED_REVISION, SchemaMode

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
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
