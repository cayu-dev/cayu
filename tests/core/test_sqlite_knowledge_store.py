from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from tests.core.knowledge_none_terms_conformance import (
    assert_entry_wide_none_terms_conformance,
    assert_entry_wide_none_terms_precede_chunk_pagination,
)
from tests.core.knowledge_publication_conformance import (
    assert_concurrent_publication_conformance,
    assert_failed_publication_left_no_state,
    assert_owned_publication_conformance,
    assert_stale_operation_cannot_replace_newer_publication,
    publication_material,
)

from cayu._validation import DurableValueError, extract_durable_value_error
from cayu.core.tools import ToolContext
from cayu.storage import (
    MAX_KNOWLEDGE_CHUNK_INDEX,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListGroup,
    KnowledgeListQuery,
    KnowledgePublicationConflict,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
)
from cayu.storage import migrations as schema_migrations
from cayu.tools import RememberKnowledgeTool


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def test_sqlite_knowledge_store_rejects_out_of_range_chunk_index_atomically(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "chunk-index.sqlite")
        try:
            entry = KnowledgeEntry(id="entry_chunk_index", text="memory")
            chunk = KnowledgeChunk(
                id="chunk_index",
                entry_id=entry.id,
                chunk_index=0,
                text="chunk",
            )
            chunk.chunk_index = MAX_KNOWLEDGE_CHUNK_INDEX + 1

            with pytest.raises(ValidationError, match=str(MAX_KNOWLEDGE_CHUNK_INDEX)):
                await store.put_entry_with_chunks(entry, [chunk])
            assert await store.get_entry(entry.id) is None
            assert await store.read_chunks(entry.id) == []
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_owned_publication_conformance(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "owned-publication.sqlite")
        try:
            await assert_owned_publication_conformance(store)
            await assert_concurrent_publication_conformance(store)
            await assert_stale_operation_cannot_replace_newer_publication(store)
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_publication_serializes_independent_writers(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "concurrent-publication.sqlite"
        first_store = SQLiteKnowledgeStore(path)
        second_store = SQLiteKnowledgeStore(path)
        entry_a, chunks_a = publication_material(entry_id="cross-connection-publication")
        entry_b, chunks_b = publication_material(
            entry_id="cross-connection-publication",
            text="A different SQLite connection owns this candidate.",
            timestamp_offset=1,
        )

        def publish(store, operation_id, entry, chunks):
            try:
                return (
                    operation_id,
                    asyncio.run(
                        store.publish_entry_with_chunks(
                            entry,
                            chunks,
                            operation_id=operation_id,
                        )
                    ),
                )
            except Exception as exc:
                return operation_id, exc

        try:
            outcomes = await asyncio.gather(
                asyncio.to_thread(
                    publish,
                    first_store,
                    "cross-connection-a",
                    entry_a,
                    chunks_a,
                ),
                asyncio.to_thread(
                    publish,
                    second_store,
                    "cross-connection-b",
                    entry_b,
                    chunks_b,
                ),
            )
            successes = [outcome for outcome in outcomes if not isinstance(outcome[1], Exception)]
            conflicts = [outcome for outcome in outcomes if isinstance(outcome[1], Exception)]
            assert len(successes) == 1
            assert len(conflicts) == 1
            assert isinstance(conflicts[0][1], KnowledgePublicationConflict)
            expected_entry, expected_chunks = (
                (entry_a, chunks_a)
                if successes[0][0] == "cross-connection-a"
                else (entry_b, chunks_b)
            )
            assert await first_store.get_entry(expected_entry.id) == expected_entry
            assert await first_store.read_chunks(expected_entry.id) == expected_chunks
        finally:
            await _close(first_store)
            await _close(second_store)

    asyncio.run(run())


def test_sqlite_knowledge_publication_receipt_survives_restart(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "publication-restart.sqlite"
        entry, chunks = publication_material(entry_id="restart-publication")
        store = SQLiteKnowledgeStore(path)
        receipt = await store.publish_entry_with_chunks(
            entry,
            chunks,
            operation_id="restart-operation",
        )
        reviewed = await store.update_entry_status(entry.id, KnowledgeStatus.ARCHIVED)
        await _close(store)

        reopened = SQLiteKnowledgeStore(path)
        try:
            replay = await reopened.publish_entry_with_chunks(
                entry,
                chunks,
                operation_id="restart-operation",
            )
            assert replay.replayed is True
            assert replay.committed_at == receipt.committed_at
            assert await reopened.get_entry(entry.id) == reviewed
            assert await reopened.read_chunks(entry.id) == chunks
        finally:
            await _close(reopened)

    asyncio.run(run())


def test_sqlite_remember_knowledge_reconciles_ack_loss_and_restart(tmp_path) -> None:
    class AcknowledgementLossSQLiteStore(SQLiteKnowledgeStore):
        async def publish_entry_with_chunks(self, entry, chunks, *, operation_id):
            await super().publish_entry_with_chunks(
                entry,
                chunks,
                operation_id=operation_id,
            )
            raise RuntimeError("secret canary acknowledgement failure")

    async def run() -> None:
        path = tmp_path / "remember-ack-loss.sqlite"
        context_options = {
            "session_id": "session_1",
            "idempotency_key": "durable-remember-operation",
        }
        store = AcknowledgementLossSQLiteStore(path)
        first = await RememberKnowledgeTool().run(
            ToolContext(knowledge_store=store, **context_options),
            {"text": "Durable knowledge publication survives acknowledgement loss."},
        )
        await _close(store)

        reopened = SQLiteKnowledgeStore(path)
        try:
            replay = await RememberKnowledgeTool().run(
                ToolContext(knowledge_store=reopened, **context_options),
                {"text": "Durable knowledge publication survives acknowledgement loss."},
            )
            assert first.is_error is False
            assert first.structured["post_write_error"] == ("publication_acknowledgement_lost")
            assert "secret canary" not in first.content
            assert "secret canary" not in repr(first.structured)
            assert replay.is_error is False
            assert replay.structured["written"] is False
            assert replay.structured["already_known"] is None
            assert replay.structured["publication_replayed"] is True
            assert replay.structured["status"] is None
        finally:
            await _close(reopened)

    asyncio.run(run())


@pytest.mark.parametrize("failure_phase", ["entry", "chunks", "receipt"])
def test_sqlite_knowledge_publication_rolls_back_each_material_write(
    tmp_path,
    failure_phase: str,
) -> None:
    class FailingPublicationStore(SQLiteKnowledgeStore):
        def _insert_entry_unlocked(self, entry) -> None:
            super()._insert_entry_unlocked(entry)
            self._fail_after("entry")

        def _insert_chunks_unlocked(self, entry, chunks) -> None:
            super()._insert_chunks_unlocked(entry, chunks)
            self._fail_after("chunks")

        def _insert_publication_receipt_unlocked(self, receipt) -> None:
            super()._insert_publication_receipt_unlocked(receipt)
            self._fail_after("receipt")

        def _fail_after(self, phase: str) -> None:
            if phase == failure_phase:
                raise RuntimeError(f"injected {phase}-boundary failure")

    async def run() -> None:
        path = tmp_path / "publication-rollback.sqlite"
        entry, chunks = publication_material(entry_id="rollback-publication")
        failing = FailingPublicationStore(path)
        try:
            with pytest.raises(RuntimeError, match=rf"{failure_phase}-boundary"):
                await failing.publish_entry_with_chunks(
                    entry,
                    chunks,
                    operation_id="rollback-operation",
                )
            await assert_failed_publication_left_no_state(
                failing,
                entry_id=entry.id,
                operation_id="rollback-operation",
            )
        finally:
            await _close(failing)

        reopened = SQLiteKnowledgeStore(path)
        try:
            receipt = await reopened.publish_entry_with_chunks(
                entry,
                chunks,
                operation_id="rollback-operation",
            )
            assert receipt.replayed is False
        finally:
            await _close(reopened)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("invalid_text", "code"),
    [
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ],
)
def test_sqlite_knowledge_store_rejects_nonportable_lookup_text(
    tmp_path,
    invalid_text: str,
    code: str,
) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "portable-knowledge.sqlite")
        try:
            with pytest.raises(DurableValueError) as invalid_id:
                await store.get_entry(invalid_text)
            assert invalid_id.value.code == code
            assert "workload-secret-value" not in str(invalid_id.value)

            query = KnowledgeQuery(namespace="safe", text="probe")
            query.namespace = invalid_text
            with pytest.raises(ValidationError) as invalid_query:
                await store.search(query)
            query_error = extract_durable_value_error(invalid_query.value)
            assert query_error is not None
            assert query_error.code == code
            assert "workload-secret-value" not in str(invalid_query.value)
        finally:
            await _close(store)

    asyncio.run(run())


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
                metadata={"numbers": {"ordinary": 1.0, "zero": -0.0, "fractional": 1e-7}},
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
                    metadata={"numbers": {"ordinary": 1.0, "zero": -0.0, "fractional": 1e-7}},
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
    assert loaded.metadata["numbers"] == {"ordinary": 1, "zero": 0, "fractional": 1e-7}
    assert type(loaded.metadata["numbers"]["ordinary"]) is int
    assert type(loaded.metadata["numbers"]["zero"]) is int
    assert type(loaded.metadata["numbers"]["fractional"]) is float
    assert [hit.entry.id for hit in result.hits] == ["invoice_warning"]
    assert result.hits[0].chunk is not None
    assert result.hits[0].chunk.id == "invoice_warning:0"
    assert result.hits[0].chunk.metadata["numbers"] == {
        "ordinary": 1,
        "zero": 0,
        "fractional": 1e-7,
    }
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


def test_sqlite_knowledge_store_prune_expired_hard_deletes(tmp_path) -> None:
    # MEM-05: prune_expired reclaims expired entries (and their chunks/FTS) rather than just hiding them.
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(KnowledgeEntry(id="active", text="deployment warning"))
        await store.put_entry(
            KnowledgeEntry(
                id="expired",
                text="deployment warning",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        pruned = await store.prune_expired()
        leftover = await store.search(KnowledgeQuery(text="deployment", include_expired=True))
        expired_entry = await store.get_entry("expired")
        active_entry = await store.get_entry("active")
        await _close(store)
        return pruned, leftover, expired_entry, active_entry

    pruned, leftover, expired_entry, active_entry = asyncio.run(run())

    assert pruned == 1
    assert expired_entry is None
    assert active_entry is not None
    assert [hit.entry.id for hit in leftover.hits] == ["active"]


def test_sqlite_knowledge_store_conditionally_transitions_status(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(
            KnowledgeEntry(
                id="pending",
                text="Remote sandbox Git pushes should use a brokered credential proxy.",
                namespace="project:cayu",
                labels={"project": "cayu"},
                status=KnowledgeStatus.PENDING,
            )
        )
        active = await store.transition_entry_status(
            "pending",
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ACTIVE,
            expected_namespace="project:cayu",
            expected_labels={"project": "cayu"},
        )
        with pytest.raises(ValueError, match="not 'pending'"):
            await store.transition_entry_status(
                "pending",
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ARCHIVED,
                expected_namespace="project:cayu",
                expected_labels={"project": "cayu"},
            )
        await store.put_entry(
            KnowledgeEntry(
                id="pending_other",
                text="Other project knowledge.",
                namespace="project:other",
                labels={"project": "other"},
                status=KnowledgeStatus.PENDING,
            )
        )
        with pytest.raises(ValueError, match="expected namespace"):
            await store.transition_entry_status(
                "pending_other",
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
                expected_namespace="project:cayu",
            )
        with pytest.raises(ValueError, match="expected labels"):
            await store.transition_entry_status(
                "pending_other",
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
                expected_labels={"project": "cayu"},
            )
        await _close(store)
        return active

    active = asyncio.run(run())

    assert active.status is KnowledgeStatus.ACTIVE


def test_sqlite_knowledge_store_update_entry_status_guards_concurrent_delete(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        # Simulate the check-then-write race: the entry is seen by the load but is
        # gone by the time the UPDATE runs (e.g. a concurrent hard delete). The
        # in-statement rowcount guard must surface this as a missing entry rather
        # than silently succeeding.
        stale = KnowledgeEntry(id="ghost", text="already deleted", status=KnowledgeStatus.PENDING)
        store._load_entry_unlocked = lambda entry_id: stale  # type: ignore[assignment]
        with pytest.raises(KeyError, match="ghost"):
            await store.update_entry_status("ghost", KnowledgeStatus.ARCHIVED)
        await _close(store)

    asyncio.run(run())


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


def test_sqlite_knowledge_store_applies_none_terms_to_the_complete_entry(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "entry-wide-none-terms.sqlite")
        try:
            await assert_entry_wide_none_terms_conformance(
                store,
                mode=KnowledgeSearchMode.KEYWORD,
            )
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_filters_none_terms_before_chunk_pagination(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "none-terms-pagination.sqlite")
        try:
            await assert_entry_wide_none_terms_precede_chunk_pagination(store)
        finally:
            await _close(store)

    asyncio.run(run())


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


def test_sqlite_knowledge_store_batches_multi_entry_hit_hydration(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        for index in range(3):
            await store.put_entry(
                KnowledgeEntry(
                    id=f"entry_{index}",
                    text=f"Shared deployment warning number {index}.",
                    labels={"project": f"proj_{index}", "shared": "yes"},
                    aspects=[f"aspect_{index}"],
                    impact_targets=[f"target_{index}"],
                )
            )
        result = await store.search(KnowledgeQuery(text="deployment warning", limit=10))
        await _close(store)
        return result

    result = asyncio.run(run())

    # Every hit hydrates through the batched loaders, so per-entry label/aspect/
    # impact-target lists must stay correctly grouped by entry (not cross-contaminated).
    by_entry = {hit.entry.id: hit for hit in result.hits}
    assert set(by_entry) == {"entry_0", "entry_1", "entry_2"}
    for index in range(3):
        hit = by_entry[f"entry_{index}"]
        assert hit.entry.labels == {"project": f"proj_{index}", "shared": "yes"}
        assert hit.entry.aspects == [f"aspect_{index}"]
        assert hit.entry.impact_targets == [f"target_{index}"]
        assert hit.chunk is not None
        assert hit.chunk.entry_id == f"entry_{index}"


def test_sqlite_knowledge_store_list_reports_multi_chunk_counts(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        await store.put_entry(KnowledgeEntry(id="single", text="Single chunk entry."))
        await store.put_entry_with_chunks(
            KnowledgeEntry(id="multi", text="Multi chunk entry."),
            [
                KnowledgeChunk(
                    id=f"multi:{index}",
                    entry_id="multi",
                    chunk_index=index,
                    text=f"Body part {index}.",
                )
                for index in range(3)
            ],
        )
        result = await store.list_entries(KnowledgeListQuery(limit=10))
        await _close(store)
        return result

    result = asyncio.run(run())

    counts = {item.entry.id: item.chunk_count for item in result.entries}
    assert counts == {"single": 1, "multi": 3}


def test_sqlite_knowledge_store_search_survives_entry_text_only_update(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")

    async def run():
        # Custom chunk whose body differs from the entry text drives put_entry down
        # the untouched-chunks branch, which now refreshes the FTS rows exactly once.
        await store.put_entry_with_chunks(
            KnowledgeEntry(id="doc", title="Original title", text="Original entry summary."),
            [
                KnowledgeChunk(
                    id="doc:0",
                    entry_id="doc",
                    chunk_index=0,
                    text="Unrelated chunk body.",
                )
            ],
        )
        await store.put_entry(
            KnowledgeEntry(id="doc", title="Revised heading", text="Revised entry summary.")
        )
        by_title = await store.search(KnowledgeQuery(text="revised heading"))
        by_text = await store.search(KnowledgeQuery(text="revised summary"))
        stale = await store.search(KnowledgeQuery(text="original title"))
        await _close(store)
        return by_title, by_text, stale

    by_title, by_text, stale = asyncio.run(run())

    assert [hit.entry.id for hit in by_title.hits] == ["doc"]
    assert [hit.entry.id for hit in by_text.hits] == ["doc"]
    assert stale.hits == []


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
        publication_receipts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_knowledge_publication_receipts'"
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
    assert publication_receipts is not None
