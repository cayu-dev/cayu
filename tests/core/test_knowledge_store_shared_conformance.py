from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.core.knowledge_access_scope_conformance import (
    assert_knowledge_access_scope_conformance,
)

from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListQuery,
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
):
    if case.name == "memory":
        return InMemoryKnowledgeStore(access_scope=access_scope)
    if case.name == "sqlite":
        assert isinstance(case.location, Path)
        return SQLiteKnowledgeStore(case.location, access_scope=access_scope)
    assert isinstance(case.location, str)
    return PostgresKnowledgeStore(
        case.location,
        access_scope=access_scope,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
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
