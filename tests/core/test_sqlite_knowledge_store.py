from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from tests.core.knowledge_access_scope_conformance import (
    assert_knowledge_access_scope_conformance,
)
from tests.core.knowledge_index_readiness_conformance import (
    assert_index_readiness_conformance,
)
from tests.core.knowledge_maintenance_conformance import (
    _create_proposal_entries,
    maintenance_decision,
    maintenance_proposal,
)
from tests.core.knowledge_none_terms_conformance import (
    assert_entry_wide_none_terms_conformance,
    assert_entry_wide_none_terms_precede_chunk_pagination,
)
from tests.core.knowledge_phrase_conformance import (
    assert_token_exact_phrase_search_conformance,
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
    MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_KNOWLEDGE_CHUNK_INDEX,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListGroup,
    KnowledgeListQuery,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeQuery,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeRevisionConflict,
    KnowledgeRevisionRef,
    KnowledgeRevisionResetRequired,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations
from cayu.tools import RememberKnowledgeTool

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def test_sqlite_index_readiness_conformance(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(
            tmp_path / "index-readiness.sqlite",
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await assert_index_readiness_conformance(store)
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_relation_change_access_fails_closed_for_malformed_audiences(
    tmp_path,
) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(
            tmp_path / "relation-change-audiences.sqlite",
            access_scope=_ACCESS_SCOPE,
        )
        try:
            for entry_id in ("audience-subject", "audience-object"):
                await store.create_entry(KnowledgeEntry(id=entry_id, text=entry_id))
            relation = KnowledgeRelation(
                id="audience-relation",
                subject=KnowledgeRevisionRef(entry_id="audience-subject", revision=1),
                object=KnowledgeRevisionRef(entry_id="audience-object", revision=1),
                kind=KnowledgeRelationKind.DERIVED_FROM,
            )
            await store.publish_relations([relation], operation_id="audience-operation")
            relation_change = next(
                change
                for change in (await store.read_changes()).changes
                if change.relation_id == relation.id
            )
            object_audience = store._connection.execute(
                """
                SELECT namespace, visibility, status, source_type, source_id,
                       requires_include_expired
                FROM cayu_knowledge_change_audiences
                WHERE change_sequence = ? AND audience_kind = 'object_current'
                """,
                (relation_change.sequence,),
            ).fetchone()
            assert object_audience is not None
            store._connection.execute(
                """
                DELETE FROM cayu_knowledge_change_audiences
                WHERE change_sequence = ? AND audience_kind = 'object_current'
                """,
                (relation_change.sequence,),
            )
            store._connection.execute(
                """
                INSERT INTO cayu_knowledge_change_audiences (
                    change_sequence, audience_kind, namespace, visibility, status,
                    source_type, source_id, requires_include_expired
                ) VALUES (?, 'after', ?, ?, ?, ?, ?, ?)
                """,
                (relation_change.sequence, *object_audience),
            )
            store._connection.commit()

            visible = await store.read_changes(
                after_sequence=relation_change.sequence - 1,
            )
            assert all(change.relation_id != relation.id for change in visible.changes)
        finally:
            await store.close()

    asyncio.run(run())


def _reconcile_sqlite_through_revision_41(connection: sqlite3.Connection) -> None:
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 41
        )
        sqlite_support.reconcile_schema(
            connection,
            schema_migrations.SchemaMode.MIGRATE,
            app_min_supported=41,
        )
    finally:
        schema_migrations.REVISIONS = revisions


def _reconcile_sqlite_through_revision_42(connection: sqlite3.Connection) -> None:
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 42
        )
        sqlite_support.reconcile_schema(
            connection,
            schema_migrations.SchemaMode.MIGRATE,
            app_min_supported=42,
        )
    finally:
        schema_migrations.REVISIONS = revisions


def _reconcile_sqlite_through_revision_43(connection: sqlite3.Connection) -> None:
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 43
        )
        sqlite_support.reconcile_schema(
            connection,
            schema_migrations.SchemaMode.MIGRATE,
            app_min_supported=43,
        )
    finally:
        schema_migrations.REVISIONS = revisions


def _reconcile_sqlite_through_revision_59(connection: sqlite3.Connection) -> None:
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 59
        )
        sqlite_support.reconcile_schema(
            connection,
            schema_migrations.SchemaMode.MIGRATE,
            app_min_supported=59,
        )
    finally:
        schema_migrations.REVISIONS = revisions


def _reconcile_sqlite_through_revision_60(connection: sqlite3.Connection) -> None:
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 60
        )
        sqlite_support.reconcile_schema(
            connection,
            schema_migrations.SchemaMode.MIGRATE,
            app_min_supported=60,
        )
    finally:
        schema_migrations.REVISIONS = revisions


def _reconcile_sqlite_through_revision_62(connection: sqlite3.Connection) -> None:
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 62
        )
        sqlite_support.reconcile_schema(
            connection,
            schema_migrations.SchemaMode.MIGRATE,
            app_min_supported=62,
        )
    finally:
        schema_migrations.REVISIONS = revisions


def test_sqlite_knowledge_access_scope_conformance(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "access-scope.sqlite")
        try:
            await assert_knowledge_access_scope_conformance(store)
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_scoped_entry_hydration_uses_one_read_snapshot(tmp_path) -> None:
    database = tmp_path / "scoped-read-snapshot.sqlite"
    privileged = SQLiteKnowledgeStore(database, access_scope=_ACCESS_SCOPE)

    async def seed() -> None:
        await privileged.create_entry(
            KnowledgeEntry(
                id="entry",
                text="snapshot protected",
                labels={"project": "alpha"},
            )
        )
        await privileged.close()

    asyncio.run(seed())

    class RacingStore(SQLiteKnowledgeStore):
        changed = False

        def _load_labels_unlocked(self, entry_id: str, revision: int) -> dict[str, str]:
            if not self.changed:
                self.changed = True
                peer = sqlite3.connect(database)
                try:
                    peer.execute(
                        "UPDATE cayu_knowledge_labels SET value = ? WHERE entry_id = ? AND key = ?",
                        ("beta", entry_id, "project"),
                    )
                    peer.commit()
                finally:
                    peer.close()
            return super()._load_labels_unlocked(entry_id, revision)

    scope = KnowledgeAccessScope.for_namespace(
        "default",
        required_labels={"project": "alpha"},
    )
    racing = RacingStore(database, access_scope=scope)

    async def read() -> KnowledgeEntry | None:
        try:
            return await racing.get_entry("entry")
        finally:
            await racing.close()

    loaded = asyncio.run(read())
    assert loaded is not None
    assert loaded.labels == {"project": "alpha"}

    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT value FROM cayu_knowledge_labels WHERE entry_id = ? AND key = ?",
            ("entry", "project"),
        ).fetchone() == ("beta",)
    finally:
        connection.close()


def test_sqlite_knowledge_store_rejects_out_of_range_chunk_index_atomically(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "chunk-index.sqlite", access_scope=_ACCESS_SCOPE)
        try:
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
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_owned_publication_conformance(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(
            tmp_path / "owned-publication.sqlite", access_scope=_ACCESS_SCOPE
        )
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
        first_store = SQLiteKnowledgeStore(path, access_scope=_ACCESS_SCOPE)
        second_store = SQLiteKnowledgeStore(path, access_scope=_ACCESS_SCOPE)
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
                        store.publish_entry_revision(
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
            assert isinstance(conflicts[0][1], KnowledgeRevisionConflict)
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
        store = SQLiteKnowledgeStore(path, access_scope=_ACCESS_SCOPE)
        receipt = await store.publish_entry_revision(
            entry,
            chunks,
            operation_id="restart-operation",
        )
        reviewed = await store.transition_entry_status(
            entry.id,
            expected_revision=entry.revision,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
        )
        await _close(store)

        reopened = SQLiteKnowledgeStore(path, access_scope=_ACCESS_SCOPE)
        try:
            replay = await reopened.publish_entry_revision(
                entry,
                chunks,
                operation_id="restart-operation",
            )
            assert replay.replayed is True
            assert replay.committed_at == receipt.committed_at
            assert await reopened.get_entry(entry.id) == reviewed
            assert await reopened.read_chunks(entry.id, revision=entry.revision) == chunks
            current_chunks = await reopened.read_chunks(entry.id)
            assert [chunk.text for chunk in current_chunks] == [chunk.text for chunk in chunks]
            assert all(chunk.entry_revision == reviewed.revision for chunk in current_chunks)
        finally:
            await _close(reopened)

    asyncio.run(run())


def test_sqlite_remember_knowledge_reconciles_ack_loss_and_restart(tmp_path) -> None:
    class AcknowledgementLossSQLiteStore(SQLiteKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            access_scope=None,
            operation_id,
            expected_revision=None,
        ):
            await super().publish_entry_revision(
                entry,
                chunks,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )
            raise RuntimeError("secret canary acknowledgement failure")

    async def run() -> None:
        path = tmp_path / "remember-ack-loss.sqlite"
        context_options = {
            "session_id": "session_1",
            "idempotency_key": "durable-remember-operation",
        }
        store = AcknowledgementLossSQLiteStore(path, access_scope=_ACCESS_SCOPE)
        first = await RememberKnowledgeTool().run(
            ToolContext(knowledge_store=store, **context_options),
            {"text": "Durable knowledge publication survives acknowledgement loss."},
        )
        await _close(store)

        reopened = SQLiteKnowledgeStore(path, access_scope=_ACCESS_SCOPE)
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

        def _insert_publication_receipt_unlocked(self, receipt, entry) -> None:
            super()._insert_publication_receipt_unlocked(receipt, entry)
            self._fail_after("receipt")

        def _fail_after(self, phase: str) -> None:
            if phase == failure_phase:
                raise RuntimeError(f"injected {phase}-boundary failure")

    async def run() -> None:
        path = tmp_path / "publication-rollback.sqlite"
        entry, chunks = publication_material(entry_id="rollback-publication")
        failing = FailingPublicationStore(path, access_scope=_ACCESS_SCOPE)
        try:
            with pytest.raises(RuntimeError, match=rf"{failure_phase}-boundary"):
                await failing.publish_entry_revision(
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

        reopened = SQLiteKnowledgeStore(path, access_scope=_ACCESS_SCOPE)
        try:
            receipt = await reopened.publish_entry_revision(
                entry,
                chunks,
                operation_id="rollback-operation",
            )
            assert receipt.replayed is False
        finally:
            await _close(reopened)

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure_phase",
    ["replacement", "predecessor", "relations", "relation_change", "decision"],
)
def test_sqlite_maintenance_rolls_back_every_material_boundary(
    tmp_path,
    failure_phase: str,
) -> None:
    class FailingMaintenanceStore(SQLiteKnowledgeStore):
        lifecycle_writes = 0

        def _append_revision_unlocked(self, *args, **kwargs) -> None:
            super()._append_revision_unlocked(*args, **kwargs)
            self.lifecycle_writes += 1
            self._fail_after("replacement" if self.lifecycle_writes == 1 else "predecessor")

        def _insert_relations_unlocked(self, relations) -> None:
            super()._insert_relations_unlocked(relations)
            self._fail_after("relations")

        def _insert_relation_change_unlocked(self, *args, **kwargs):
            change = super()._insert_relation_change_unlocked(*args, **kwargs)
            self._fail_after("relation_change")
            return change

        def _insert_maintenance_record_unlocked(self, *args, **kwargs) -> None:
            super()._insert_maintenance_record_unlocked(*args, **kwargs)
            self._fail_after("decision")

        def _fail_after(self, phase: str) -> None:
            if phase == failure_phase:
                raise RuntimeError(f"injected {phase}-boundary failure")

    async def assert_unchanged(store, proposal, operation_id: str, baseline: int) -> None:
        replacement = await store.get_entry(proposal.replacement.entry_id)
        source = await store.get_entry(proposal.sources[0].entry_id)
        assert replacement is not None
        assert replacement.revision == 1
        assert replacement.status is KnowledgeStatus.PENDING
        assert source is not None
        assert source.revision == 1
        assert source.status is KnowledgeStatus.ACTIVE
        relations = await store.read_relations(
            KnowledgeRelationQuery(reference=proposal.sources[0])
        )
        assert relations is not None
        assert relations.relations == []
        assert await store.load_maintenance_decision_receipt(operation_id) is None
        assert (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence == (
            baseline
        )

    async def run() -> None:
        path = tmp_path / f"maintenance-rollback-{failure_phase}.sqlite"
        proposal = maintenance_proposal(f"sqlite-rollback-{failure_phase}")
        decision = maintenance_decision(
            proposal,
            operation_id=f"sqlite-rollback-{failure_phase}-operation",
            kind=KnowledgeMaintenanceDecisionKind.APPROVE,
        )
        failing = FailingMaintenanceStore(path, access_scope=_ACCESS_SCOPE)
        try:
            await _create_proposal_entries(failing, proposal)
            baseline = (await failing.read_changes(after_sequence=0, limit=100)).high_water_sequence
            with pytest.raises(RuntimeError, match=rf"{failure_phase}-boundary"):
                await failing.apply_maintenance_decision(proposal, decision)
            await assert_unchanged(failing, proposal, decision.operation_id, baseline)
        finally:
            await failing.close()

        reopened = SQLiteKnowledgeStore(path, access_scope=_ACCESS_SCOPE)
        try:
            await assert_unchanged(reopened, proposal, decision.operation_id, baseline)
            receipt = await reopened.apply_maintenance_decision(proposal, decision)
            assert receipt.replayed is False
        finally:
            await reopened.close()

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
        store = SQLiteKnowledgeStore(
            tmp_path / "portable-knowledge.sqlite", access_scope=_ACCESS_SCOPE
        )
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
    store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)

    async def write() -> None:
        await store.create_entry(
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
        await store.create_entry(
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

    reopened = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)

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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(KnowledgeEntry(id="active", text="deployment warning"))
        await store.create_entry(
            KnowledgeEntry(
                id="pending",
                text="deployment warning",
                status=KnowledgeStatus.PENDING,
            )
        )
        await store.create_entry(
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(KnowledgeEntry(id="active", text="deployment warning"))
        await store.create_entry(
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(
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
            expected_revision=1,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ACTIVE,
            expected_namespace="project:cayu",
            expected_labels={"project": "cayu"},
        )
        with pytest.raises(ValueError, match="not 'pending'"):
            await store.transition_entry_status(
                "pending",
                expected_revision=active.revision,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ARCHIVED,
                expected_namespace="project:cayu",
                expected_labels={"project": "cayu"},
            )
        await store.create_entry(
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
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
                expected_namespace="project:cayu",
            )
        with pytest.raises(ValueError, match="expected labels"):
            await store.transition_entry_status(
                "pending_other",
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
                expected_labels={"project": "cayu"},
            )
        await _close(store)
        return active

    active = asyncio.run(run())

    assert active.status is KnowledgeStatus.ACTIVE


def test_sqlite_knowledge_store_transition_rejects_missing_entry(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        with pytest.raises(KeyError, match="ghost"):
            await store.transition_entry_status(
                "ghost",
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ARCHIVED,
            )
        await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_preserves_custom_chunks_on_entry_update(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(KnowledgeEntry(id="entry", text="billing memory"))
        result = await store.search(KnowledgeQuery(text="billing", kinds=[]))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert result.hits == []
    assert result.total_hits_known == 0


def test_sqlite_knowledge_store_search_reports_preview_truncation(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(KnowledgeEntry(id="entry", text="billing memory has more text"))
        result = await store.search(KnowledgeQuery(text="billing", max_bytes=7))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert len(result.hits) == 1
    assert result.hits[0].text_preview == "billing"
    assert result.truncated is True


def test_sqlite_knowledge_store_search_dedupes_across_large_chunk_matches(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(
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
        await store.create_entry(KnowledgeEntry(id="small", text="invoice policy"))
        result = await store.search(KnowledgeQuery(text="invoice", limit=2))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert {hit.entry.id for hit in result.hits} == {"large", "small"}
    assert result.total_hits_known == 2
    assert result.truncated is False


def test_sqlite_knowledge_store_structured_keyword_search(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(
            KnowledgeEntry(id="github_secret", text="GitHub push requires a credential broker.")
        )
        await store.create_entry(
            KnowledgeEntry(id="sendgrid_secret", text="SendGrid email uses a secret proxy.")
        )
        await store.create_entry(
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


def test_sqlite_knowledge_store_phrase_search_conformance(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(
            tmp_path / "phrase-conformance.sqlite",
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await assert_token_exact_phrase_search_conformance(store)
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_applies_none_terms_to_the_complete_entry(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(
            tmp_path / "entry-wide-none-terms.sqlite", access_scope=_ACCESS_SCOPE
        )
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
        store = SQLiteKnowledgeStore(
            tmp_path / "none-terms-pagination.sqlite", access_scope=_ACCESS_SCOPE
        )
        try:
            await assert_entry_wide_none_terms_precede_chunk_pagination(store)
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_searches_entry_text_with_custom_chunks(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
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
        result = await store.search(KnowledgeQuery(text="brokered credential"))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["broker_summary"]
    assert result.hits[0].reason == "entry text match"
    assert "brokered credential" in result.hits[0].text_preview


def test_sqlite_knowledge_store_matches_singular_plural_token_variants(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(KnowledgeEntry(id="keys", text="Store API keys securely."))
        await store.create_entry(KnowledgeEntry(id="policies", text="Security policies apply."))
        key_result = await store.search(KnowledgeQuery(text="key"))
        policy_result = await store.search(KnowledgeQuery(text="policy"))
        await _close(store)
        return key_result, policy_result

    key_result, policy_result = asyncio.run(run())

    assert [hit.entry.id for hit in key_result.hits] == ["keys"]
    assert [hit.entry.id for hit in policy_result.hits] == ["policies"]


def test_sqlite_knowledge_store_all_terms_match_across_entry_document(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
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
        result = await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert [hit.entry.id for hit in result.hits] == ["split_match"]


def test_sqlite_knowledge_store_all_terms_do_not_match_across_unrelated_chunks(
    tmp_path,
) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
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
        result = await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))
        await _close(store)
        return result

    result = asyncio.run(run())

    assert result.hits == []


def test_sqlite_knowledge_store_lists_entries_and_facets(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(
            KnowledgeEntry(
                id="runbook",
                namespace="ops",
                kind="procedure",
                labels={"project": "billing"},
                text="Payment reminder runbook.",
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="warning",
                namespace="ops",
                kind="warning",
                labels={"project": "billing"},
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        for index in range(5):
            await store.create_entry(
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(
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


def test_sqlite_knowledge_store_rejects_invalid_revision_chunks(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
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
        await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_schema_rejects_a_dangling_current_revision(tmp_path) -> None:
    database = tmp_path / "current-revision-fk.sqlite"

    async def create() -> None:
        store = SQLiteKnowledgeStore(database, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="entry", text="current"))
        finally:
            await store.close()

    asyncio.run(create())

    connection = sqlite_support.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError), sqlite_support._transaction(connection):
            connection.execute(
                "UPDATE cayu_knowledge_entries SET current_revision = ? WHERE id = ?",
                (2, "entry"),
            )
        assert (
            connection.execute(
                "SELECT current_revision FROM cayu_knowledge_entries WHERE id = ?",
                ("entry",),
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("object_name", "drop_sql"),
    (
        (
            "cayu_knowledge_current_entries",
            "DROP VIEW cayu_knowledge_current_entries",
        ),
        (
            "idx_cayu_knowledge_revisions_status",
            "DROP INDEX idx_cayu_knowledge_revisions_status",
        ),
        (
            "cayu_knowledge_chunks_fts",
            "DROP TABLE cayu_knowledge_chunks_fts",
        ),
    ),
)
def test_sqlite_revision_schema_validation_rejects_missing_structural_objects(
    tmp_path,
    object_name: str,
    drop_sql: str,
) -> None:
    database = tmp_path / f"missing-{object_name}.sqlite"
    store = SQLiteKnowledgeStore(database, access_scope=_ACCESS_SCOPE)
    asyncio.run(store.close())

    connection = sqlite_support.connect(database)
    try:
        connection.execute(drop_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match=object_name):
        SQLiteKnowledgeStore(
            database,
            access_scope=_ACCESS_SCOPE,
            schema_mode=schema_migrations.SchemaMode.VALIDATE,
        )


def test_sqlite_knowledge_store_rejects_unsupported_search_modes(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(KnowledgeEntry(id="entry", text="billing memory"))
        with pytest.raises(ValueError, match="supports only auto and keyword"):
            await store.search(KnowledgeQuery(text="billing", mode=KnowledgeSearchMode.SEMANTIC))
        await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_store_batches_multi_entry_hit_hydration(tmp_path) -> None:
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        for index in range(3):
            await store.create_entry(
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        await store.create_entry(KnowledgeEntry(id="single", text="Single chunk entry."))
        await store.create_entry(
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
    store = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)

    async def run():
        # Custom chunk whose body differs from the entry text drives create_entry down
        # the untouched-chunks branch, which now refreshes the FTS rows exactly once.
        await store.create_entry(
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
        current = await store.get_entry("doc")
        assert current is not None
        await store.append_entry_revision(
            current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "title": "Revised heading",
                    "text": "Revised entry summary.",
                }
            ),
            expected_revision=current.revision,
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

    knowledge_store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)

    async def write_knowledge() -> None:
        await knowledge_store.create_entry(
            KnowledgeEntry(id="entry", text="shared database memory")
        )
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


def test_sqlite_revision_60_refuses_populated_knowledge_without_backfill(
    tmp_path,
) -> None:
    database = tmp_path / "revision-59-to-60-populated.sqlite"
    connection = sqlite_support.connect(database)
    entry = KnowledgeEntry(id="preserved-entry", text="Must remain untouched.")
    try:
        _reconcile_sqlite_through_revision_59(connection)
        seed_store = SQLiteKnowledgeStore.__new__(SQLiteKnowledgeStore)
        seed_store._connection = connection
        with sqlite_support._transaction(connection):
            seed_store._insert_entry_unlocked(entry)
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match="clean prerelease knowledge-lineage break",
    ):
        SQLiteKnowledgeStore(
            database,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )

    connection = sqlite_support.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 59
        assert (
            connection.execute(
                "SELECT text FROM cayu_knowledge_revisions WHERE entry_id = ? AND revision = 1",
                (entry.id,),
            ).fetchone()[0]
            == entry.text
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cayu_knowledge_relations'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_sqlite_revision_60_initializes_empty_pre_relation_schema_directly(
    tmp_path,
) -> None:
    database = tmp_path / "revision-59-to-60-empty.sqlite"
    connection = sqlite_support.connect(database)
    try:
        _reconcile_sqlite_through_revision_59(connection)
    finally:
        connection.close()

    store = SQLiteKnowledgeStore(
        database,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
        access_scope=_ACCESS_SCOPE,
    )
    store._connection.close()

    connection = sqlite_support.connect(database)
    try:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == schema_migrations.LATEST_REVISION
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cayu_knowledge_relations'"
            ).fetchone()
            is not None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cayu_knowledge_relation_publication_receipts'"
            ).fetchone()
            is not None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cayu_knowledge_maintenance_decisions'"
            ).fetchone()
            is not None
        )
    finally:
        connection.close()


def test_sqlite_revision_63_refuses_populated_knowledge_without_interpretation(
    tmp_path,
) -> None:
    database = tmp_path / "revision-62-to-63-populated.sqlite"
    connection = sqlite_support.connect(database)
    entry = KnowledgeEntry(id="revision-62-entry", text="Must remain untouched.")
    try:
        _reconcile_sqlite_through_revision_62(connection)
        seed_store = SQLiteKnowledgeStore.__new__(SQLiteKnowledgeStore)
        seed_store._connection = connection
        with sqlite_support._transaction(connection):
            seed_store._insert_entry_unlocked(entry)
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match="clean prerelease reviewed-maintenance break",
    ):
        SQLiteKnowledgeStore(
            database,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )

    connection = sqlite_support.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 62
        assert (
            connection.execute(
                "SELECT text FROM cayu_knowledge_revisions WHERE entry_id = ? AND revision = 1",
                (entry.id,),
            ).fetchone()[0]
            == entry.text
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cayu_knowledge_maintenance_decisions'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_sqlite_revision_63_initializes_empty_knowledge_schema_directly(tmp_path) -> None:
    database = tmp_path / "revision-62-to-63-empty.sqlite"
    connection = sqlite_support.connect(database)
    try:
        _reconcile_sqlite_through_revision_62(connection)
    finally:
        connection.close()

    store = SQLiteKnowledgeStore(
        database,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
        access_scope=_ACCESS_SCOPE,
    )
    store._connection.close()

    connection = sqlite_support.connect(database)
    try:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == schema_migrations.LATEST_REVISION
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cayu_knowledge_maintenance_decisions'"
            ).fetchone()
            is not None
        )
    finally:
        connection.close()


def test_sqlite_revision_63_rejects_a_malformed_maintenance_table(tmp_path) -> None:
    database = tmp_path / "revision-63-malformed-maintenance.sqlite"
    store = SQLiteKnowledgeStore(database, access_scope=_ACCESS_SCOPE)
    store._connection.close()
    connection = sqlite_support.connect(database)
    try:
        connection.execute("DROP TABLE cayu_knowledge_maintenance_decisions")
        connection.execute(
            "CREATE TABLE cayu_knowledge_maintenance_decisions (operation_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="reviewed knowledge maintenance contract"):
        SQLiteKnowledgeStore(database, access_scope=_ACCESS_SCOPE)


def test_sqlite_revision_63_requires_lowercase_sha_constraints(tmp_path) -> None:
    database = tmp_path / "revision-63-weak-maintenance-hashes.sqlite"
    store = SQLiteKnowledgeStore(database, access_scope=_ACCESS_SCOPE)
    store._connection.close()
    connection = sqlite_support.connect(database)
    try:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_knowledge_maintenance_decisions'"
        ).fetchone()
        assert row is not None
        definition = str(row[0])
        weakened = definition.replace(
            "AND proposal_fingerprint NOT GLOB '*[^0-9a-f]*'",
            "",
        ).replace(
            "AND request_sha256 NOT GLOB '*[^0-9a-f]*'",
            "",
        )
        assert weakened != definition
        connection.execute("DROP TABLE cayu_knowledge_maintenance_decisions")
        connection.execute(weakened)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="reviewed knowledge maintenance contract"):
        SQLiteKnowledgeStore(database, access_scope=_ACCESS_SCOPE)


def test_sqlite_revision_43_rejects_out_of_contract_revision_42_identities(
    tmp_path,
) -> None:
    database = tmp_path / "revision-42-oversized-identity.sqlite"
    connection = sqlite_support.connect(database)
    entry = KnowledgeEntry(id="bounded-entry", text="Valid revision-42 entry.")
    oversized_chunk_id = "c" * (MAX_KNOWLEDGE_CHUNK_ID_BYTES + 1)
    try:
        _reconcile_sqlite_through_revision_42(connection)
        seed_store = SQLiteKnowledgeStore.__new__(SQLiteKnowledgeStore)
        seed_store._connection = connection
        with sqlite_support._transaction(connection):
            seed_store._insert_entry_unlocked(entry)
            connection.execute(
                """
                INSERT INTO cayu_knowledge_chunks (
                    id, entry_id, entry_revision, chunk_index,
                    text, content_hash, source_uri, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    oversized_chunk_id,
                    entry.id,
                    entry.revision,
                    0,
                    entry.text,
                    None,
                    None,
                    "{}",
                ),
            )
    finally:
        connection.close()

    connection = sqlite_support.connect(database)
    try:
        with pytest.raises(schema_migrations.SchemaTooOld, match="bounds knowledge"):
            _reconcile_sqlite_through_revision_43(connection)
    finally:
        connection.close()

    connection = sqlite_support.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 42
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' "
                "AND name = 'idx_cayu_knowledge_chunks_identity_owner'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_sqlite_revision_migration_refuses_populated_legacy_knowledge_unchanged(
    tmp_path,
) -> None:
    database = tmp_path / "populated-revision-41.sqlite"
    connection = sqlite_support.connect(database)
    try:
        _reconcile_sqlite_through_revision_41(connection)
        connection.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id, namespace, text, kind, visibility, status,
                created_by_type, created_by, created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-entry",
                "default",
                "legacy text must survive",
                "fact",
                "global",
                "active",
                "system",
                "legacy-test",
                "2026-08-18T09:00:00+00:00",
                "2026-08-18T09:00:00+00:00",
                '{"proof":"unchanged"}',
            ),
        )
        connection.execute(
            "INSERT INTO cayu_knowledge_labels (entry_id, key, value) VALUES (?, ?, ?)",
            ("legacy-entry", "project", "cayu"),
        )
        connection.execute(
            """
            INSERT INTO cayu_knowledge_chunks (
                id, entry_id, chunk_index, text, metadata_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("legacy-entry:0", "legacy-entry", 0, "legacy chunk must survive", "{}"),
        )
        connection.commit()

        schema_before = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name LIKE 'cayu_knowledge_%' ORDER BY type, name"
        ).fetchall()
        data_before = (
            connection.execute("SELECT * FROM cayu_knowledge_entries").fetchall(),
            connection.execute("SELECT * FROM cayu_knowledge_labels").fetchall(),
            connection.execute("SELECT * FROM cayu_knowledge_chunks").fetchall(),
        )
        ledger_before = connection.execute(
            "SELECT revision, kind, compatible_from FROM cayu_schema_migrations ORDER BY revision"
        ).fetchall()
        connection.close()

        with pytest.raises(KnowledgeRevisionResetRequired) as raised:
            SQLiteKnowledgeStore(
                database,
                schema_mode=schema_migrations.SchemaMode.MIGRATE,
                access_scope=_ACCESS_SCOPE,
            )

        connection = sqlite_support.connect(database)
        connection.execute("BEGIN IMMEDIATE")
        connection.rollback()
        assert raised.value.assessment.populated_tables == (
            "cayu_knowledge_chunks",
            "cayu_knowledge_entries",
            "cayu_knowledge_labels",
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 41
        assert (
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name LIKE 'cayu_knowledge_%' ORDER BY type, name"
            ).fetchall()
            == schema_before
        )
        assert (
            connection.execute("SELECT * FROM cayu_knowledge_entries").fetchall(),
            connection.execute("SELECT * FROM cayu_knowledge_labels").fetchall(),
            connection.execute("SELECT * FROM cayu_knowledge_chunks").fetchall(),
        ) == data_before
        assert (
            connection.execute(
                "SELECT revision, kind, compatible_from FROM cayu_schema_migrations "
                "ORDER BY revision"
            ).fetchall()
            == ledger_before
        )
    finally:
        connection.close()


def test_sqlite_revision_migration_refuses_populated_unversioned_knowledge_before_ddl(
    tmp_path,
) -> None:
    database = tmp_path / "populated-unversioned.sqlite"
    connection = sqlite_support.connect(database)
    try:
        connection.execute(
            "CREATE TABLE cayu_knowledge_entries (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO cayu_knowledge_entries (id, text) VALUES (?, ?)",
            ("unversioned-entry", "must survive"),
        )
        connection.commit()
        schema_before = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        data_before = connection.execute("SELECT * FROM cayu_knowledge_entries").fetchall()
        connection.close()

        with pytest.raises(KnowledgeRevisionResetRequired):
            SQLiteKnowledgeStore(
                database,
                schema_mode=schema_migrations.SchemaMode.MIGRATE,
                access_scope=_ACCESS_SCOPE,
            )

        connection = sqlite_support.connect(database)
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'cayu_schema_migrations'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
            == schema_before
        )
        assert connection.execute("SELECT * FROM cayu_knowledge_entries").fetchall() == data_before
    finally:
        connection.close()
