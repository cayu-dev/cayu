from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListGroup,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
)
from cayu.storage import migrations as schema_migrations


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def test_sqlite_knowledge_store_persists_entries_chunks_and_filters(tmp_path) -> None:
    db_path = tmp_path / "knowledge.sqlite"
    store = SQLiteKnowledgeStore(db_path)

    async def write() -> None:
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
        await _close(store)

    asyncio.run(write())

    reopened = SQLiteKnowledgeStore(db_path)

    async def read():
        loaded = await reopened.get_entry("invoice_warning")
        result = await reopened.search(
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
        denied = await reopened.search(
            KnowledgeQuery(
                text="invoice reminders",
                namespace="ops",
                labels={"project": "missing"},
            )
        )
        await _close(reopened)
        return loaded, result, denied

    loaded, result, denied = asyncio.run(read())

    assert loaded is not None
    assert loaded.labels == {"project": "invoice_agent", "user": "alice"}
    assert loaded.aspects == ["finance"]
    assert loaded.impact_targets == ["finance.reminders"]
    assert [hit.entry.id for hit in result.hits] == ["invoice_warning"]
    assert result.hits[0].chunk is not None
    assert result.hits[0].chunk.id == "invoice_warning:0"
    assert result.hits[0].score_kind == "sqlite_fts5_bm25"
    assert result.total_hits_known == 1
    assert denied.hits == []


def test_sqlite_knowledge_store_defaults_hide_inactive_and_expired(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        await _close(store)
        return active, pending, expired

    active, pending, expired = asyncio.run(run())

    assert [hit.entry.id for hit in active.hits] == ["active"]
    assert [hit.entry.id for hit in pending.hits] == ["pending"]
    assert [hit.entry.id for hit in expired.hits] == ["expired", "active"]


def test_sqlite_knowledge_store_preserves_custom_chunks_on_entry_update(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        await _close(store)
        return chunks, result

    chunks, result = asyncio.run(run())

    assert len(chunks) == 1
    assert chunks[0].text == "Custom indexed body."
    assert chunks[0].metadata == {"indexer": "custom"}
    assert [hit.entry.id for hit in result.hits] == ["doc"]


def test_sqlite_knowledge_store_empty_kind_filter_returns_no_matches(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(KnowledgeEntry(id="entry", text="billing memory"))
        result = await store.search(KnowledgeQuery(text="billing", kinds=[]))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert result.hits == []
    assert result.total_hits_known == 0


def test_sqlite_knowledge_store_search_reports_preview_truncation(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(KnowledgeEntry(id="entry", text="billing memory has more text"))
        result = await store.search(KnowledgeQuery(text="billing", max_bytes=7))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert len(result.hits) == 1
    assert result.hits[0].text_preview == "billing"
    assert result.truncated is True


def test_sqlite_knowledge_store_search_dedupes_across_large_chunk_matches(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        result = await store.search(KnowledgeQuery(text="invoice", limit=2))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert {hit.entry.id for hit in result.hits} == {"large", "small"}
    assert result.total_hits_known == 2
    assert result.truncated is False


def test_sqlite_knowledge_store_structured_keyword_search(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(
            KnowledgeEntry(id="github_secret", text="GitHub push requires a credential broker.")
        )
        await store.put_entry(
            KnowledgeEntry(id="sendgrid_secret", text="SendGrid email uses a secret proxy.")
        )
        await store.put_entry(
            KnowledgeEntry(id="github_test", text="GitHub test credentials are fixture-only.")
        )
        result = await store.search(
            KnowledgeQuery(
                any_terms=["credential", "secret"],
                all_terms=["github push"],
                none_terms=["fixture only"],
            )
        )
        await _close(store)
        return result

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["github_secret"]


def test_sqlite_knowledge_store_searches_entry_text_with_custom_chunks(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        result = await store.search(KnowledgeQuery(text="brokered credential"))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["broker_summary"]
    assert result.hits[0].reason == "entry text match"
    assert "brokered credential" in result.hits[0].text_preview


def test_sqlite_knowledge_store_matches_singular_plural_token_variants(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        result = await store.search(
            KnowledgeQuery(
                all_terms=["GitHub", "credentials"],
                any_terms=["sandbox", "push", "token"],
            )
        )
        await _close(store)
        return result

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["remote_git"]


def test_sqlite_knowledge_store_matches_y_plural_token_variants(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(KnowledgeEntry(id="keys", text="Store API keys securely."))
        await store.put_entry(KnowledgeEntry(id="policies", text="Security policies apply."))
        key_result = await store.search(KnowledgeQuery(text="key"))
        policy_result = await store.search(KnowledgeQuery(text="policy"))
        await _close(store)
        return key_result, policy_result

    key_result, policy_result = asyncio.run(run())

    assert [hit.entry.id for hit in key_result.hits] == ["keys"]
    assert [hit.entry.id for hit in policy_result.hits] == ["policies"]


def test_sqlite_knowledge_store_all_terms_match_across_entry_document(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        result = await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["split_match"]


def test_sqlite_knowledge_store_all_terms_do_not_match_across_unrelated_chunks(
    tmp_path,
) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        result = await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert result.hits == []


def test_sqlite_knowledge_store_lists_entries_and_facets(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        result = await store.list_entries(
            KnowledgeListQuery(
                namespace="ops",
                labels={"project": "billing"},
                group_by=KnowledgeListGroup.KIND,
            )
        )
        await _close(store)
        return result

    result = asyncio.run(run())

    assert result.total_entries_known == 2
    assert {item.entry.id for item in result.entries} == {"runbook", "warning"}
    assert [(facet.value, facet.count) for facet in result.facets] == [
        ("procedure", 1),
        ("warning", 1),
    ]


def test_sqlite_knowledge_store_caps_facets(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        for index in range(5):
            await store.put_entry(
                KnowledgeEntry(
                    id=f"entry_{index}",
                    labels={"area": f"area_{index}"},
                    text=f"Knowledge entry {index}.",
                )
            )
        result = await store.list_entries(
            KnowledgeListQuery(
                group_by=KnowledgeListGroup.LABEL,
                limit=3,
            )
        )
        await _close(store)
        return result

    result = asyncio.run(run())

    assert len(result.facets) == 3
    assert result.facets_truncated is True
    assert result.truncated is True


def test_sqlite_knowledge_store_chunk_windows_and_truncation(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        await _close(store)
        return window, centered, truncated

    window, centered, truncated = asyncio.run(run())

    assert [chunk.id for chunk in window] == ["chunk_0", "chunk_1", "chunk_2"]
    assert [chunk.id for chunk in centered] == ["chunk_2"]
    assert truncated[0].text == "gamma"
    assert truncated[0].content_hash is None


def test_sqlite_knowledge_store_rejects_invalid_chunk_replacement(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
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
        await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_rejects_unsupported_search_modes(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(KnowledgeEntry(id="entry", text="billing memory"))
        with pytest.raises(ValueError, match="supports only auto and keyword"):
            await store.search(KnowledgeQuery(text="billing", mode=KnowledgeSearchMode.SEMANTIC))
        await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_schema_migrates_and_coexists_with_session_store(tmp_path) -> None:
    db_path = tmp_path / "cayu.sqlite"
    session_store = SQLiteSessionStore(db_path)

    async def close_session_store() -> None:
        await _close(session_store)

    asyncio.run(close_session_store())

    knowledge_store = SQLiteKnowledgeStore(db_path)

    async def write_knowledge() -> None:
        await knowledge_store.put_entry(KnowledgeEntry(id="entry", text="shared database memory"))
        result = await knowledge_store.search(KnowledgeQuery(text="shared database"))
        assert [hit.entry.id for hit in result.hits] == ["entry"]
        await _close(knowledge_store)

    asyncio.run(write_knowledge())

    connection = sqlite3.connect(db_path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        revisions = connection.execute(
            "SELECT revision, compatible_from FROM cayu_schema_migrations ORDER BY revision"
        ).fetchall()
        knowledge_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_knowledge_entries'"
        ).fetchone()
        knowledge_fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_knowledge_chunks_fts'"
        ).fetchone()
    finally:
        connection.close()

    assert version == schema_migrations.LATEST_REVISION
    assert revisions[-1] == (
        schema_migrations.LATEST_REVISION,
        schema_migrations.MIN_SUPPORTED_REVISION,
    )
    assert knowledge_table is not None
    assert knowledge_fts is not None
