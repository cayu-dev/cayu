from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from tests.core.knowledge_access_scope_conformance import (
    assert_knowledge_access_scope_conformance,
)

from cayu.storage import (
    MAX_KNOWLEDGE_CHANGE_LIMIT,
    MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_KNOWLEDGE_ENTRY_ID_BYTES,
    InMemoryKnowledgeStore,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeChangeConsumerConflict,
    KnowledgeChangeKind,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceConflict,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceRole,
    KnowledgeListQuery,
    KnowledgePublicationConflict,
    KnowledgeQuery,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
    PostgresKnowledgeStore,
    SQLiteKnowledgeStore,
)
from cayu.storage.migrations import SchemaMode

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()


@dataclass(frozen=True)
class KnowledgeStoreCase:
    name: str
    location: Path | str | None
    durable: bool


@dataclass
class _MutableClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


@pytest.fixture(params=("memory", "sqlite", "postgres"))
def knowledge_store_case(request, tmp_path: Path) -> KnowledgeStoreCase:
    if request.param == "memory":
        return KnowledgeStoreCase("memory", None, False)
    if request.param == "sqlite":
        return KnowledgeStoreCase("sqlite", tmp_path / "knowledge.sqlite", True)
    return KnowledgeStoreCase(
        "postgres",
        request.getfixturevalue("postgres_dsn"),
        True,
    )


async def _open_store(
    case: KnowledgeStoreCase,
    *,
    access_scope: KnowledgeAccessScope | None = _ACCESS_SCOPE,
    clock: Callable[[], datetime] | None = None,
):
    if case.name == "memory":
        return InMemoryKnowledgeStore(access_scope=access_scope, clock=clock)
    if case.name == "sqlite":
        assert isinstance(case.location, Path)
        return SQLiteKnowledgeStore(case.location, access_scope=access_scope, clock=clock)
    assert isinstance(case.location, str)
    return PostgresKnowledgeStore(
        case.location,
        access_scope=access_scope,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
        clock=clock,
    )


async def _close_store(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        await close()


async def _reset_case(case: KnowledgeStoreCase) -> None:
    if case.name != "postgres":
        return
    import psycopg
    from psycopg import sql

    assert isinstance(case.location, str)
    async with await psycopg.AsyncConnection.connect(case.location) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
            )
            tables = [str(row[0]) for row in await cursor.fetchall()]
            for table in tables:
                await cursor.execute(sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(table)))
        await connection.commit()


def test_knowledge_store_shared_revision_contract(knowledge_store_case) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            timestamp = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)
            revision_1 = KnowledgeEntry(
                id="shared-revision",
                text="obsoletequasar marker",
                namespace="project:cayu",
                labels={"project": "cayu"},
                status=KnowledgeStatus.PENDING,
                created_at=timestamp,
                updated_at=timestamp,
                metadata={"nested": {"value": 1}},
            )
            chunk_1 = KnowledgeChunk(
                id="shared-revision:r1:0",
                entry_id=revision_1.id,
                entry_revision=1,
                chunk_index=0,
                text=revision_1.text,
                metadata={"nested": {"value": 1}},
            )
            created = await store.create_entry(revision_1, [chunk_1])
            revision_1.metadata["nested"]["value"] = 99
            chunk_1.metadata["nested"]["value"] = 99
            assert created.revision == 1
            assert (await store.get_entry(created.id, revision=1)).metadata == {
                "nested": {"value": 1}
            }
            assert (await store.read_chunks(created.id, revision=1))[0].metadata == {
                "nested": {"value": 1}
            }

            revision_2 = created.model_copy(
                update={
                    "revision": 2,
                    "text": "currentnebula marker",
                    "updated_at": timestamp,
                }
            )
            chunk_2 = KnowledgeChunk(
                id="shared-revision:r2:0",
                entry_id=created.id,
                entry_revision=2,
                chunk_index=0,
                text=revision_2.text,
            )
            appended = await store.append_entry_revision(
                revision_2,
                [chunk_2],
                expected_revision=1,
            )
            assert appended.revision == 2
            assert await store.get_entry(created.id) == revision_2
            assert await store.get_entry(created.id, revision=1) == created
            assert await store.read_chunks(created.id, revision=1) == [
                chunk_1.model_copy(update={"metadata": {"nested": {"value": 1}}})
            ]
            assert await store.read_chunks(created.id) == [chunk_2]
            truncated_chunks = await store.read_chunks(created.id, max_bytes=7)
            assert len(truncated_chunks) == 1
            assert truncated_chunks[0].entry_revision == 2
            assert truncated_chunks[0].content_hash is None
            pending_query = {
                "namespace": "project:cayu",
                "statuses": [KnowledgeStatus.PENDING],
            }
            assert (
                await store.search(KnowledgeQuery(text="obsoletequasar", **pending_query))
            ).hits == []
            assert [
                hit.entry.revision
                for hit in (
                    await store.search(KnowledgeQuery(text="currentnebula", **pending_query))
                ).hits
            ] == [2]
            listing = await store.list_entries(
                KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING])
            )
            assert [(item.entry.id, item.entry.revision) for item in listing.entries] == [
                (created.id, 2)
            ]

            active = await store.transition_entry_status(
                created.id,
                expected_revision=2,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
            )
            assert active.revision == 3
            assert active.status is KnowledgeStatus.ACTIVE
            assert (await store.get_entry(created.id, revision=2)).status is KnowledgeStatus.PENDING
            assert all(chunk.entry_revision == 3 for chunk in await store.read_chunks(created.id))

            tombstone = await store.delete_entry(created.id, expected_revision=3)
            assert tombstone is not None
            assert tombstone.revision == 4
            assert tombstone.status is KnowledgeStatus.DELETED
            assert (await store.get_entry(created.id, revision=3)).status is KnowledgeStatus.ACTIVE
            erased = await store.delete_entry(created.id, expected_revision=4, hard=True)
            assert erased == tombstone
            assert await store.get_entry(created.id) is None
            assert await store.get_entry(created.id, revision=1) is None
            assert await store.read_chunks(created.id, revision=1) == []
            changes = await store.read_changes()
            assert [change.kind for change in changes.changes] == [
                KnowledgeChangeKind.CREATED,
                KnowledgeChangeKind.REVISION_APPENDED,
                KnowledgeChangeKind.STATUS_TRANSITIONED,
                KnowledgeChangeKind.TOMBSTONED,
                KnowledgeChangeKind.HARD_DELETED,
            ]
            assert [change.entry_revision for change in changes.changes] == [1, 2, 3, 4, 4]
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_expiration_change_contract(knowledge_store_case) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            cutoff = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
            await store.create_entry(
                KnowledgeEntry(
                    id="shared-expiration-change",
                    text="Expired knowledge.",
                    expires_at=cutoff - timedelta(seconds=1),
                ),
                evidence=[
                    KnowledgeEvidence(
                        id="shared-expiration-evidence",
                        entry_id="shared-expiration-change",
                        source_type="document",
                        source_id="expired-source",
                        source_revision="1",
                    )
                ],
            )
            assert await store.prune_expired(now=cutoff) == 1
            assert await store.get_entry("shared-expiration-change") is None
            assert await store.read_evidence("shared-expiration-change", revision=1) is None
            changes = await store.read_changes()
            assert [change.kind for change in changes.changes] == [
                KnowledgeChangeKind.CREATED,
                KnowledgeChangeKind.EXPIRED,
            ]
            expired = changes.changes[-1]
            assert expired.entry_id == "shared-expiration-change"
            assert expired.entry_revision == 1
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_expiration_change_order_is_portable(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            cutoff = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
            for entry_id in ("shared-expiration-z", "shared-expiration-a"):
                await store.create_entry(
                    KnowledgeEntry(
                        id=entry_id,
                        text=f"Expired knowledge for {entry_id}.",
                        expires_at=cutoff - timedelta(seconds=1),
                    )
                )

            assert await store.prune_expired(now=cutoff) == 2
            changes = await store.read_changes()
            assert [
                change.entry_id
                for change in changes.changes
                if change.kind is KnowledgeChangeKind.EXPIRED
            ] == ["shared-expiration-a", "shared-expiration-z"]
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_revision_bound_evidence_contract(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            timestamp = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
            revision_1 = KnowledgeEntry(
                id="shared-evidence",
                text="Evidence-bound knowledge.",
                status=KnowledgeStatus.PENDING,
                created_at=timestamp,
                updated_at=timestamp,
            )
            chunk_1 = KnowledgeChunk(
                id="shared-evidence:r1:0",
                entry_id=revision_1.id,
                entry_revision=1,
                chunk_index=0,
                text=revision_1.text,
            )
            evidence = [
                KnowledgeEvidence(
                    id="shared-evidence-supporting",
                    entry_id=revision_1.id,
                    entry_revision=1,
                    role=KnowledgeEvidenceRole.SUPPORTING,
                    source_type="issue",
                    source_uri="https://example.invalid/issues/989",
                    source_hash="a" * 64,
                    disposition=KnowledgeEvidenceDisposition.RETAINED,
                    created_at=timestamp,
                ),
                KnowledgeEvidence(
                    id="shared-evidence-origin",
                    entry_id=revision_1.id,
                    entry_revision=1,
                    chunk_id=chunk_1.id,
                    role=KnowledgeEvidenceRole.ORIGIN,
                    source_type="document",
                    source_id="architecture-v5.1",
                    source_revision="commit-a",
                    locator={"section": "evidence", "nested": {"line": 1}},
                    created_at=timestamp,
                    metadata={"reviewed": True},
                ),
            ]
            origin_evidence = evidence[1]

            invalid_bindings = (
                (
                    "shared-evidence-wrong-entry",
                    evidence[0].model_copy(update={"entry_id": "another-entry"}),
                ),
                (
                    "shared-evidence-wrong-revision",
                    evidence[0].model_copy(update={"entry_revision": 2}),
                ),
                (
                    "shared-evidence-missing-chunk",
                    evidence[0].model_copy(update={"chunk_id": "missing-chunk"}),
                ),
            )
            for invalid_entry_id, invalid_evidence in invalid_bindings:
                with pytest.raises(ValueError):
                    await store.create_entry(
                        KnowledgeEntry(id=invalid_entry_id, text="Must remain absent."),
                        evidence=[
                            invalid_evidence.model_copy(update={"entry_id": invalid_entry_id})
                            if invalid_entry_id != "shared-evidence-wrong-entry"
                            else invalid_evidence
                        ],
                    )
                assert await store.get_entry(invalid_entry_id) is None

            await store.create_entry(revision_1, [chunk_1], evidence=evidence)
            origin_evidence.locator["nested"]["line"] = 99

            if knowledge_store_case.durable:
                await _close_store(store)
                store = await _open_store(knowledge_store_case)

            before_conflict = await store.read_changes()
            conflicting_entry = KnowledgeEntry(
                id="shared-evidence-conflict",
                text="Must remain absent.",
            )
            with pytest.raises(KnowledgeEvidenceConflict):
                await store.create_entry(
                    conflicting_entry,
                    evidence=[
                        evidence[0].model_copy(
                            update={
                                "entry_id": conflicting_entry.id,
                                "entry_revision": 1,
                                "chunk_id": None,
                            }
                        )
                    ],
                )
            assert await store.get_entry(conflicting_entry.id) is None
            assert await store.read_changes() == before_conflict

            stored_1 = await store.read_evidence(revision_1.id, revision=1)
            assert stored_1 is not None
            assert [item.id for item in stored_1.evidence] == [
                "shared-evidence-origin",
                "shared-evidence-supporting",
            ]
            assert stored_1.evidence[0].locator == {
                "section": "evidence",
                "nested": {"line": 1},
            }
            bounded = await store.read_evidence(
                revision_1.id,
                revision=1,
                max_records=1,
            )
            assert bounded is not None
            assert len(bounded.evidence) == 1
            assert bounded.truncated is True
            assert bounded.total_evidence_known == 2
            byte_bounded = await store.read_evidence(
                revision_1.id,
                revision=1,
                max_bytes=1,
            )
            assert byte_bounded is not None
            assert byte_bounded.evidence == []
            assert byte_bounded.truncated is True
            assert byte_bounded.total_evidence_known == 2

            revision_2 = await store.transition_entry_status(
                revision_1.id,
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
            )
            stored_2 = await store.read_evidence(revision_1.id, revision=2)
            assert stored_2 is not None
            assert len(stored_2.evidence) == 2
            assert {item.id for item in stored_2.evidence}.isdisjoint(
                item.id for item in stored_1.evidence
            )
            inherited_origin = next(
                item for item in stored_2.evidence if item.role is KnowledgeEvidenceRole.ORIGIN
            )
            assert inherited_origin.entry_revision == 2
            assert inherited_origin.chunk_id == "shared-evidence:r2:0"
            assert inherited_origin.locator == stored_1.evidence[0].locator

            revision_3 = revision_2.model_copy(
                update={
                    "revision": 3,
                    "text": "Materially revised knowledge.",
                    "updated_at": revision_2.updated_at,
                }
            )
            await store.append_entry_revision(revision_3, expected_revision=2)
            stored_3 = await store.read_evidence(revision_1.id, revision=3)
            assert stored_3 is not None
            assert stored_3.evidence == []
            assert stored_3.total_evidence_known == 0
            assert (await store.read_evidence(revision_1.id, revision=1)) == stored_1
            await store.delete_entry(revision_1.id, expected_revision=3, hard=True)
            assert await store.read_evidence(revision_1.id, revision=1) is None
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_evidence_order_uses_portable_scalar_identity(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            entry = KnowledgeEntry(id="shared-evidence-order", text="Ordered evidence.")
            await store.create_entry(
                entry,
                evidence=[
                    KnowledgeEvidence(
                        id=evidence_id,
                        entry_id=entry.id,
                        source_type="document",
                        source_id=evidence_id,
                        source_revision="1",
                    )
                    for evidence_id in (
                        "shared-evidence-order-ä",
                        "shared-evidence-order-z",
                    )
                ],
            )

            result = await store.read_evidence(entry.id, max_records=1)
            assert result is not None
            assert [item.id for item in result.evidence] == ["shared-evidence-order-z"]
            assert result.truncated is True
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_atomic_change_and_snapshot_contract(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        privileged = KnowledgeAccessScope.privileged()
        alpha = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"project": "alpha"},
        )
        beta = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"project": "beta"},
        )
        try:
            created = await store.create_entry(
                KnowledgeEntry(
                    id="shared-change-alpha",
                    text="alpha revision",
                    labels={"project": "alpha"},
                ),
                evidence=[
                    KnowledgeEvidence(
                        id="shared-change-alpha-evidence",
                        entry_id="shared-change-alpha",
                        source_type="document",
                        source_id="alpha-source",
                        source_revision="1",
                    )
                ],
                access_scope=privileged,
            )
            assert (
                await store.read_evidence(created.id, revision=1, access_scope=alpha)
            ) is not None
            with pytest.raises(ValueError, match="max_records"):
                await store.read_evidence(
                    created.id,
                    revision=1,
                    access_scope=beta,
                    max_records=0,
                )
            with pytest.raises(KnowledgeAccessDenied):
                await store.create_entry(
                    KnowledgeEntry(
                        id="shared-change-hidden-collision",
                        text="must remain absent",
                        labels={"project": "beta"},
                    ),
                    evidence=[
                        KnowledgeEvidence(
                            id="shared-change-alpha-evidence",
                            entry_id="shared-change-hidden-collision",
                            source_type="document",
                            source_id="beta-source",
                            source_revision="1",
                        )
                    ],
                    access_scope=beta,
                )
            assert (
                await store.get_entry(
                    "shared-change-hidden-collision",
                    access_scope=privileged,
                )
                is None
            )
            with pytest.raises(KnowledgeRevisionConflict):
                await store.append_entry_revision(
                    created.model_copy(
                        update={
                            "revision": 3,
                            "text": "failed append",
                        }
                    ),
                    expected_revision=2,
                    access_scope=privileged,
                )
            await store.create_entry(
                KnowledgeEntry(
                    id="shared-change-beta",
                    text="beta revision",
                    labels={"project": "beta"},
                ),
                access_scope=privileged,
            )
            await store.append_entry_revision(
                created.model_copy(
                    update={
                        "revision": 2,
                        "text": "alpha moved to beta",
                        "labels": {"project": "beta"},
                    }
                ),
                expected_revision=1,
                access_scope=privileged,
            )
            assert (await store.read_evidence(created.id, revision=1, access_scope=alpha)) is None
            assert (await store.read_evidence(created.id, revision=1, access_scope=beta)) is None
            assert (
                await store.read_evidence(created.id, revision=1, access_scope=privileged)
            ) is not None

            all_changes = await store.read_changes(access_scope=privileged)
            assert [change.kind for change in all_changes.changes] == [
                KnowledgeChangeKind.CREATED,
                KnowledgeChangeKind.CREATED,
                KnowledgeChangeKind.REVISION_APPENDED,
            ]
            assert [change.sequence for change in all_changes.changes] == sorted(
                change.sequence for change in all_changes.changes
            )
            assert all_changes.high_water_sequence == all_changes.changes[-1].sequence
            alpha_changes = await store.read_changes(access_scope=alpha)
            assert [
                (change.entry_id, change.entry_revision) for change in alpha_changes.changes
            ] == [
                ("shared-change-alpha", 1),
                ("shared-change-alpha", 2),
            ]
            beta_changes = await store.read_changes(access_scope=beta)
            assert [
                (change.entry_id, change.entry_revision) for change in beta_changes.changes
            ] == [
                ("shared-change-beta", 1),
                ("shared-change-alpha", 2),
            ]
            page = await store.read_changes(limit=1, access_scope=privileged)
            assert len(page.changes) == 1
            assert page.truncated is True
            remainder = await store.read_changes(
                after_sequence=page.next_after_sequence,
                access_scope=privileged,
            )
            assert [*page.changes, *remainder.changes] == all_changes.changes

            publication_entry = KnowledgeEntry(
                id="shared-change-publication",
                text="atomic publication",
            )
            publication_chunk = KnowledgeChunk(
                id="shared-change-publication:r1:0",
                entry_id=publication_entry.id,
                entry_revision=1,
                chunk_index=0,
                text=publication_entry.text,
            )
            publication_evidence = KnowledgeEvidence(
                id="shared-change-publication-evidence",
                entry_id=publication_entry.id,
                entry_revision=1,
                chunk_id=publication_chunk.id,
                source_type="document",
                source_id="source-a",
                source_revision="1",
            )
            receipt = await store.publish_entry_revision(
                publication_entry,
                [publication_chunk],
                evidence=[publication_evidence],
                operation_id="shared-change-publication-operation",
                access_scope=privileged,
            )
            replay = await store.publish_entry_revision(
                publication_entry,
                [publication_chunk],
                evidence=[publication_evidence],
                operation_id="shared-change-publication-operation",
                access_scope=privileged,
            )
            assert receipt.replayed is False
            assert replay.replayed is True
            with pytest.raises(KnowledgePublicationConflict):
                await store.publish_entry_revision(
                    publication_entry,
                    [publication_chunk],
                    evidence=[publication_evidence.model_copy(update={"source_revision": "2"})],
                    operation_id="shared-change-publication-operation",
                    access_scope=privileged,
                )
            after_publication = await store.read_changes(access_scope=privileged)
            assert len(after_publication.changes) == len(all_changes.changes) + 1
            assert after_publication.changes[-1].operation_id == receipt.operation_id
            assert after_publication.changes[-1].kind is KnowledgeChangeKind.CREATED
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_change_scope_exit_delivery_contract(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        privileged = KnowledgeAccessScope.privileged()
        alpha = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"project": "alpha-exit"},
        )
        active_cleanup = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"project": "status-exit"},
        )
        expiry_cleanup = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"project": "expiry-exit"},
        )
        timestamp = datetime.now(UTC)
        try:
            alpha_entry = await store.create_entry(
                KnowledgeEntry(
                    id="shared-change-alpha-exit",
                    text="alpha before relabel",
                    labels={"project": "alpha-exit"},
                ),
                access_scope=privileged,
            )
            alpha_created = await store.claim_change(
                "shared-alpha-exit-consumer",
                "worker",
                access_scope=alpha,
            )
            assert alpha_created is not None
            await store.acknowledge_change(alpha_created, access_scope=alpha)
            await store.append_entry_revision(
                alpha_entry.model_copy(
                    update={
                        "revision": 2,
                        "text": "beta after relabel",
                        "labels": {"project": "beta-entry"},
                    }
                ),
                expected_revision=1,
                access_scope=privileged,
            )
            relabel = await store.claim_change(
                "shared-alpha-exit-consumer",
                "worker",
                access_scope=alpha,
            )
            assert relabel is not None
            assert relabel.change.entry_revision == 2
            assert relabel.change.kind is KnowledgeChangeKind.REVISION_APPENDED

            status_entry = await store.create_entry(
                KnowledgeEntry(
                    id="shared-change-status-exit",
                    text="active before tombstone",
                    labels={"project": "status-exit"},
                ),
                access_scope=privileged,
            )
            status_created = await store.claim_change(
                "shared-status-exit-consumer",
                "worker",
                access_scope=active_cleanup,
            )
            assert status_created is not None
            await store.acknowledge_change(status_created, access_scope=active_cleanup)
            await store.delete_entry(
                status_entry.id,
                expected_revision=1,
                access_scope=privileged,
            )
            tombstone = await store.claim_change(
                "shared-status-exit-consumer",
                "worker",
                access_scope=active_cleanup,
            )
            assert tombstone is not None
            assert tombstone.change.kind is KnowledgeChangeKind.TOMBSTONED

            expiring_entry = await store.create_entry(
                KnowledgeEntry(
                    id="shared-change-expiry-exit",
                    text="active before expiry",
                    labels={"project": "expiry-exit"},
                    expires_at=timestamp + timedelta(minutes=5),
                ),
                access_scope=privileged,
            )
            expiry_created = await store.claim_change(
                "shared-expiry-exit-consumer",
                "worker",
                access_scope=expiry_cleanup,
            )
            assert expiry_created is not None
            await store.acknowledge_change(expiry_created, access_scope=expiry_cleanup)
            assert (
                await store.prune_expired(
                    access_scope=privileged,
                    now=timestamp + timedelta(minutes=10),
                )
                == 1
            )
            expired = await store.claim_change(
                "shared-expiry-exit-consumer",
                "worker",
                access_scope=expiry_cleanup,
            )
            assert expired is not None
            assert expired.change.entry_id == expiring_entry.id
            assert expired.change.kind is KnowledgeChangeKind.EXPIRED

            never_visible = KnowledgeAccessScope.for_namespace(
                "default",
                required_labels={"project": "already-expired"},
            )
            already_expired = await store.create_entry(
                KnowledgeEntry(
                    id="shared-change-already-expired",
                    text="expired before publication",
                    labels={"project": "already-expired"},
                    expires_at=timestamp - timedelta(minutes=5),
                ),
                access_scope=privileged,
            )
            assert (await store.read_changes(access_scope=never_visible)).changes == []
            assert (
                await store.prune_expired(
                    access_scope=privileged,
                    now=timestamp,
                )
                == 1
            )
            assert (
                await store.get_entry(
                    already_expired.id,
                    access_scope=privileged,
                )
                is None
            )
            assert (await store.read_changes(access_scope=never_visible)).changes == []
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_change_high_water_is_store_owned(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        privileged = KnowledgeAccessScope.privileged()
        alpha = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"project": "alpha"},
        )
        try:
            with pytest.raises(ValueError, match="after_sequence"):
                await store.read_changes(
                    after_sequence=1,
                    access_scope=privileged,
                )
            await store.create_entry(
                KnowledgeEntry(
                    id="shared-high-water-beta",
                    text="visible only to beta",
                    labels={"project": "beta"},
                ),
                access_scope=privileged,
            )
            global_page = await store.read_changes(access_scope=privileged)
            assert global_page.high_water_sequence == 1
            scoped_page = await store.read_changes(
                after_sequence=global_page.high_water_sequence,
                access_scope=alpha,
            )
            assert scoped_page.changes == []
            assert scoped_page.high_water_sequence == 0
            assert scoped_page.next_after_sequence == global_page.high_water_sequence
            with pytest.raises(ValueError, match="after_sequence"):
                await store.read_changes(
                    after_sequence=global_page.high_water_sequence + 1,
                    access_scope=alpha,
                )
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_portable_identity_boundaries(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        privileged = KnowledgeAccessScope.privileged()
        entry_id = "e" * MAX_KNOWLEDGE_ENTRY_ID_BYTES
        entry = KnowledgeEntry(id=entry_id, text="long canonical identity")
        chunk = KnowledgeChunk(
            id="c" * MAX_KNOWLEDGE_CHUNK_ID_BYTES,
            entry_id=entry_id,
            entry_revision=1,
            chunk_index=0,
            text=entry.text,
        )
        evidence = KnowledgeEvidence(
            id="shared-long-identity-evidence",
            entry_id=entry_id,
            entry_revision=1,
            chunk_id=chunk.id,
            source_type="document",
            source_id="long-identity-source",
            source_revision="1",
        )
        try:
            receipt = await store.publish_entry_revision(
                entry,
                [chunk],
                evidence=[evidence],
                operation_id="shared-long-identity-publication",
                access_scope=privileged,
            )
            assert receipt.entry_id == entry_id
            stored = await store.read_evidence(entry_id, access_scope=privileged)
            assert stored is not None
            assert stored.evidence == [evidence]
            deleted = await store.delete_entry(
                entry_id,
                expected_revision=1,
                hard=True,
                access_scope=privileged,
            )
            assert deleted == entry
            assert (await store.read_changes(access_scope=privileged)).changes[
                -1
            ].entry_id == entry_id
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_change_page_limit_is_hard_bounded(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        try:
            with pytest.raises(ValueError, match=str(MAX_KNOWLEDGE_CHANGE_LIMIT)):
                await store.read_changes(
                    limit=MAX_KNOWLEDGE_CHANGE_LIMIT + 1,
                    access_scope=KnowledgeAccessScope.privileged(),
                )
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_change_consumer_lease_contract(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        clock = _MutableClock(datetime.now(UTC))
        store = await _open_store(
            knowledge_store_case,
            access_scope=None,
            clock=clock,
        )
        privileged = KnowledgeAccessScope.privileged()
        alternate_scope = KnowledgeAccessScope.for_namespace("default")
        timestamp = clock.current
        try:
            await store.create_entry(
                KnowledgeEntry(id="shared-lease-1", text="first"),
                access_scope=privileged,
            )
            await store.create_entry(
                KnowledgeEntry(id="shared-lease-2", text="second"),
                access_scope=privileged,
            )
            contenders = await asyncio.gather(
                store.claim_change(
                    "shared-consumer",
                    "worker-a",
                    lease_seconds=10,
                    access_scope=privileged,
                ),
                store.claim_change(
                    "shared-consumer",
                    "worker-b",
                    lease_seconds=10,
                    access_scope=privileged,
                ),
            )
            claimed = [claim for claim in contenders if claim is not None]
            assert len(claimed) == 1
            first = claimed[0]
            losing_worker = "worker-b" if first.worker_id == "worker-a" else "worker-a"
            clock.current = timestamp + timedelta(seconds=1)
            replay = await store.claim_change(
                "shared-consumer",
                first.worker_id,
                lease_seconds=10,
                access_scope=privileged,
            )
            assert replay == first
            assert (
                await store.claim_change(
                    "shared-consumer",
                    losing_worker,
                    access_scope=privileged,
                )
                is None
            )
            with pytest.raises(
                KnowledgeChangeConsumerConflict,
                match="conflicts with durable state",
            ):
                await store.claim_change(
                    "shared-consumer",
                    "worker-a",
                    access_scope=alternate_scope,
                )

            clock.current = timestamp + timedelta(seconds=11)
            retried = await store.claim_change(
                "shared-consumer",
                losing_worker,
                lease_seconds=10,
                access_scope=privileged,
            )
            assert retried is not None
            assert retried.change == first.change
            assert retried.claim_id != first.claim_id
            assert retried.attempt == first.attempt + 1
            clock.current = timestamp + timedelta(seconds=12)
            with pytest.raises(KnowledgeChangeConsumerConflict):
                await store.acknowledge_change(
                    first,
                    access_scope=privileged,
                )
            acknowledged = await store.acknowledge_change(
                retried,
                access_scope=privileged,
            )
            assert acknowledged.cursor_sequence == retried.change.sequence
            assert (
                await store.acknowledge_change(
                    retried,
                    access_scope=privileged,
                )
            ) == acknowledged

            clock.current = timestamp + timedelta(seconds=13)
            second = await store.claim_change(
                "shared-consumer",
                "worker-a",
                access_scope=privileged,
            )
            assert second is not None
            assert second.change.sequence > retried.change.sequence
            clock.current = timestamp + timedelta(seconds=13, milliseconds=500)
            released = await store.release_change(
                second,
                access_scope=privileged,
            )
            assert released.cursor_sequence == acknowledged.cursor_sequence
            assert released.pending_change_sequence is None
            clock.current = timestamp + timedelta(seconds=14)
            reclaimed = await store.claim_change(
                "shared-consumer",
                "worker-b",
                access_scope=privileged,
            )
            assert reclaimed is not None
            assert reclaimed.change == second.change
            assert reclaimed.attempt == second.attempt + 1
            clock.current = timestamp + timedelta(seconds=15)
            final = await store.acknowledge_change(
                reclaimed,
                access_scope=privileged,
            )
            assert final.cursor_sequence == reclaimed.change.sequence
            assert (await store.acknowledge_change(retried, access_scope=privileged)) == final
            with pytest.raises(KnowledgeChangeConsumerConflict):
                await store.acknowledge_change(
                    retried.model_copy(update={"worker_id": "tampered-worker"}),
                    access_scope=privileged,
                )

            await store.create_entry(
                KnowledgeEntry(id="shared-lease-3", text="third"),
                access_scope=privileged,
            )
            clock.current = timestamp + timedelta(seconds=16)
            pending = await store.claim_change(
                "shared-consumer",
                "worker-a",
                lease_seconds=30,
                access_scope=privileged,
            )
            assert pending is not None

            if knowledge_store_case.durable:
                await _close_store(store)
                store = await _open_store(
                    knowledge_store_case,
                    access_scope=None,
                    clock=clock,
                )
                restored = await store.load_change_consumer_state(
                    "shared-consumer",
                    access_scope=privileged,
                )
                assert restored is not None
                assert restored.pending_claim_id == pending.claim_id
                assert restored.cursor_sequence == final.cursor_sequence
                assert (
                    await store.acknowledge_change(retried, access_scope=privileged)
                ) == restored
                clock.current = timestamp + timedelta(seconds=17)
                replayed = await store.claim_change(
                    "shared-consumer",
                    pending.worker_id,
                    lease_seconds=30,
                    access_scope=privileged,
                )
                assert replayed == pending
            clock.current = timestamp + timedelta(seconds=18)
            completed = await store.acknowledge_change(
                pending,
                access_scope=privileged,
            )
            assert completed.cursor_sequence == pending.change.sequence
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_change_consumer_baseline_contract(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        clock = _MutableClock(datetime.now(UTC))
        store = await _open_store(
            knowledge_store_case,
            access_scope=None,
            clock=clock,
        )
        privileged = KnowledgeAccessScope.privileged()
        alternate_scope = KnowledgeAccessScope.for_namespace("default")
        timestamp = clock.current
        try:
            await store.create_entry(
                KnowledgeEntry(id="shared-baseline-1", text="first baseline entry"),
                access_scope=privileged,
            )
            await store.create_entry(
                KnowledgeEntry(id="shared-baseline-2", text="second baseline entry"),
                access_scope=privileged,
            )
            captured = await store.read_changes(limit=1, access_scope=privileged)
            assert captured.truncated is True
            initialized = await store.initialize_change_consumer(
                "shared-baseline-consumer",
                baseline_sequence=captured.high_water_sequence,
                access_scope=privileged,
            )
            assert initialized.cursor_sequence == captured.high_water_sequence
            assert initialized.pending_change_sequence is None
            clock.current = timestamp + timedelta(seconds=1)
            assert (
                await store.initialize_change_consumer(
                    "shared-baseline-consumer",
                    baseline_sequence=captured.high_water_sequence,
                    access_scope=privileged,
                )
            ) == initialized
            assert (
                await store.claim_change(
                    "shared-baseline-consumer",
                    "baseline-worker",
                    access_scope=privileged,
                )
                is None
            )
            with pytest.raises(KnowledgeChangeConsumerConflict):
                await store.initialize_change_consumer(
                    "shared-baseline-consumer",
                    baseline_sequence=captured.high_water_sequence,
                    access_scope=alternate_scope,
                )
            with pytest.raises(ValueError, match="cannot exceed"):
                await store.initialize_change_consumer(
                    "shared-baseline-ahead",
                    baseline_sequence=captured.high_water_sequence + 100,
                    access_scope=privileged,
                )

            await store.create_entry(
                KnowledgeEntry(id="shared-baseline-3", text="post-baseline entry"),
                access_scope=privileged,
            )
            clock.current = timestamp + timedelta(seconds=2)
            claim = await store.claim_change(
                "shared-baseline-consumer",
                "baseline-worker",
                access_scope=privileged,
            )
            assert claim is not None
            assert claim.change.entry_id == "shared-baseline-3"
            clock.current = timestamp + timedelta(seconds=3)
            with pytest.raises(KnowledgeChangeConsumerConflict):
                await store.initialize_change_consumer(
                    "shared-baseline-consumer",
                    baseline_sequence=claim.change.sequence,
                    access_scope=privileged,
                )

            if knowledge_store_case.durable:
                await _close_store(store)
                store = await _open_store(
                    knowledge_store_case,
                    access_scope=None,
                    clock=clock,
                )
                restored = await store.load_change_consumer_state(
                    "shared-baseline-consumer",
                    access_scope=privileged,
                )
                assert restored is not None
                assert restored.pending_claim_id == claim.claim_id
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_durable_knowledge_store_rolls_back_when_change_publication_fails(
    knowledge_store_case,
    monkeypatch,
) -> None:
    if not knowledge_store_case.durable:
        pytest.skip("Transaction fault injection applies to durable stores.")

    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            if knowledge_store_case.name == "sqlite":

                def fail_change(*_args, **_kwargs):
                    raise RuntimeError("injected change publication failure")

            else:

                async def fail_change(*_args, **_kwargs):
                    raise RuntimeError("injected change publication failure")

            monkeypatch.setattr(store, "_insert_change_unlocked", fail_change, raising=False)
            monkeypatch.setattr(store, "_insert_change", fail_change, raising=False)
            entry = KnowledgeEntry(id="shared-failed-change", text="must roll back")
            evidence = KnowledgeEvidence(
                id="shared-failed-change-evidence",
                entry_id=entry.id,
                source_type="document",
                source_id="fault-source",
                source_revision="1",
            )
            with pytest.raises(RuntimeError, match="injected change publication failure"):
                await store.create_entry(entry, evidence=[evidence])
            assert await store.get_entry(entry.id) is None
            assert await store.read_evidence(entry.id) is None
            assert (await store.read_changes()).changes == []
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_cas_has_one_winner(knowledge_store_case) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            original = await store.create_entry(KnowledgeEntry(id="shared-cas", text="original"))
            candidates = [
                original.model_copy(update={"revision": 2, "text": text})
                for text in ("winner alpha", "winner beta")
            ]

            async def append(candidate: KnowledgeEntry):
                try:
                    return await store.append_entry_revision(
                        candidate,
                        expected_revision=1,
                    )
                except KnowledgeRevisionConflict as conflict:
                    return conflict

            outcomes = await asyncio.gather(*(append(candidate) for candidate in candidates))
            winners = [outcome for outcome in outcomes if isinstance(outcome, KnowledgeEntry)]
            conflicts = [
                outcome for outcome in outcomes if isinstance(outcome, KnowledgeRevisionConflict)
            ]
            assert len(winners) == 1
            assert len(conflicts) == 1
            assert conflicts[0].entry_id == original.id
            assert conflicts[0].expected_revision == 1
            assert conflicts[0].actual_revision == 2
            assert await store.get_entry(original.id) == winners[0]
            assert await store.get_entry(original.id, revision=1) == original
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_mutations_reject_invalid_cas_inputs(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        try:
            created = await store.create_entry(
                KnowledgeEntry(id="shared-invalid-cas", text="preserve me")
            )

            with pytest.raises(ValueError, match="expected_revision.*integer"):
                await store.transition_entry_status(
                    created.id,
                    expected_revision=True,
                    from_status=KnowledgeStatus.ACTIVE,
                    to_status=KnowledgeStatus.ARCHIVED,
                )
            with pytest.raises(ValueError, match="expected_revision.*greater than 0"):
                await store.delete_entry("missing", expected_revision=0, hard=True)
            with pytest.raises(ValueError, match="hard.*boolean"):
                await store.delete_entry(created.id, expected_revision=1, hard=1)

            assert await store.get_entry(created.id) == created
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_shared_access_contract(knowledge_store_case) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        try:
            await assert_knowledge_access_scope_conformance(store)
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_historical_reads_require_current_and_snapshot_access(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        privileged = KnowledgeAccessScope.privileged()
        active_scope = KnowledgeAccessScope.for_namespace(
            "tenant-a",
            allowed_statuses=[KnowledgeStatus.ACTIVE],
        )
        try:
            original = KnowledgeEntry(
                id="shared-historical-authorization",
                namespace="tenant-a",
                text="historical material",
            )
            chunk = KnowledgeChunk(
                id="shared-historical-authorization:r1:0",
                entry_id=original.id,
                entry_revision=1,
                chunk_index=0,
                text=original.text,
            )
            created = await store.create_entry(
                original,
                [chunk],
                access_scope=privileged,
            )
            tombstone = await store.delete_entry(
                created.id,
                expected_revision=created.revision,
                access_scope=privileged,
            )

            assert tombstone is not None
            assert tombstone.status is KnowledgeStatus.DELETED
            assert await store.get_entry(created.id, access_scope=active_scope) is None
            assert (
                await store.get_entry(
                    created.id,
                    revision=created.revision,
                    access_scope=active_scope,
                )
                is None
            )
            assert (
                await store.read_chunks(
                    created.id,
                    revision=created.revision,
                    access_scope=active_scope,
                )
                == []
            )

            assert (
                await store.get_entry(
                    created.id,
                    revision=created.revision,
                    access_scope=privileged,
                )
                == created
            )
            assert await store.read_chunks(
                created.id,
                revision=created.revision,
                access_scope=privileged,
            ) == [chunk]
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_scope_can_retire_but_not_promote_outside_read_statuses(
    knowledge_store_case,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case, access_scope=None)
        active_scope = KnowledgeAccessScope.for_namespace(
            "tenant-a",
            allowed_statuses=[KnowledgeStatus.ACTIVE],
        )
        pending_scope = KnowledgeAccessScope.for_namespace(
            "tenant-a",
            allowed_statuses=[KnowledgeStatus.PENDING],
        )
        privileged = KnowledgeAccessScope.privileged()
        try:
            archived_source = await store.create_entry(
                KnowledgeEntry(
                    id="shared-scope-archive",
                    namespace="tenant-a",
                    text="archive me",
                ),
                access_scope=active_scope,
            )
            archived = await store.transition_entry_status(
                archived_source.id,
                expected_revision=archived_source.revision,
                from_status=KnowledgeStatus.ACTIVE,
                to_status=KnowledgeStatus.ARCHIVED,
                access_scope=active_scope,
            )
            assert archived.status is KnowledgeStatus.ARCHIVED
            assert await store.get_entry(archived.id, access_scope=active_scope) is None
            assert await store.get_entry(archived.id, access_scope=privileged) == archived

            deleted_source = await store.create_entry(
                KnowledgeEntry(
                    id="shared-scope-delete",
                    namespace="tenant-a",
                    text="delete me",
                ),
                access_scope=active_scope,
            )
            deleted = await store.delete_entry(
                deleted_source.id,
                expected_revision=deleted_source.revision,
                access_scope=active_scope,
            )
            assert deleted is not None
            assert deleted.status is KnowledgeStatus.DELETED
            assert await store.get_entry(deleted.id, access_scope=active_scope) is None
            assert await store.get_entry(deleted.id, access_scope=privileged) == deleted

            pending = await store.create_entry(
                KnowledgeEntry(
                    id="shared-scope-promote",
                    namespace="tenant-a",
                    text="do not promote",
                    status=KnowledgeStatus.PENDING,
                ),
                access_scope=pending_scope,
            )
            with pytest.raises(KnowledgeAccessDenied):
                await store.transition_entry_status(
                    pending.id,
                    expected_revision=pending.revision,
                    from_status=KnowledgeStatus.PENDING,
                    to_status=KnowledgeStatus.ACTIVE,
                    access_scope=pending_scope,
                )
            assert await store.get_entry(pending.id, access_scope=pending_scope) == pending
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_revision_exhaustion_fails_before_lifecycle_mutation(
    knowledge_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        await _reset_case(knowledge_store_case)
        monkeypatch.setattr("cayu.storage.memory.MAX_KNOWLEDGE_REVISION", 2)
        store = await _open_store(knowledge_store_case)
        try:
            original = await store.create_entry(
                KnowledgeEntry(id="shared-revision-exhaustion", text="revision one")
            )
            successor = original.model_copy(update={"revision": 2, "text": "revision two"})
            current = await store.append_entry_revision(
                successor,
                expected_revision=original.revision,
            )

            with pytest.raises(ValueError, match="cannot advance beyond 2"):
                await store.transition_entry_status(
                    current.id,
                    expected_revision=current.revision,
                    from_status=KnowledgeStatus.ACTIVE,
                    to_status=KnowledgeStatus.ARCHIVED,
                )
            with pytest.raises(ValueError, match="cannot advance beyond 2"):
                await store.delete_entry(
                    current.id,
                    expected_revision=current.revision,
                )

            assert await store.get_entry(current.id) == current
            assert await store.get_entry(current.id, revision=1) == original
        finally:
            await _close_store(store)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())


def test_knowledge_store_durable_reopen_preserves_revisions(knowledge_store_case) -> None:
    if not knowledge_store_case.durable:
        pytest.skip("The in-memory registration has no restart boundary.")

    async def run() -> None:
        await _reset_case(knowledge_store_case)
        store = await _open_store(knowledge_store_case)
        original = await store.create_entry(KnowledgeEntry(id="shared-reopen", text="before"))
        successor = original.model_copy(update={"revision": 2, "text": "after"})
        await store.append_entry_revision(successor, expected_revision=1)
        await _close_store(store)

        reopened = await _open_store(knowledge_store_case)
        try:
            assert await reopened.get_entry(original.id) == successor
            assert await reopened.get_entry(original.id, revision=1) == original
            assert [chunk.entry_revision for chunk in await reopened.read_chunks(original.id)] == [
                2
            ]
            assert (
                await reopened.search(KnowledgeQuery(text="before", include_expired=True))
            ).hits == []
            assert [
                hit.entry.revision
                for hit in (
                    await reopened.search(KnowledgeQuery(text="after", include_expired=True))
                ).hits
            ] == [2]
        finally:
            await _close_store(reopened)
            await _reset_case(knowledge_store_case)

    asyncio.run(run())
