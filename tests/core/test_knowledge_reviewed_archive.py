"""Reviewed archive of kept knowledge, and reviewed drop, on every knowledge store."""

from __future__ import annotations

import asyncio

import pytest

from cayu.knowledge.publication_contracts import KnowledgePublicationConflict
from cayu.knowledge.search import KnowledgeQuery
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStatus,
    PostgresKnowledgeStore,
    SQLiteKnowledgeStore,
)
from cayu.storage.knowledge_review import KnowledgeReviewWorkflow
from cayu.storage.migrations import SchemaMode

_SCOPE = KnowledgeAccessScope.privileged()


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def open_store(request, tmp_path):
    """An async factory for an empty store of each kind."""

    if request.param == "memory":

        async def open_memory():
            return InMemoryKnowledgeStore(access_scope=_SCOPE)

        return open_memory
    if request.param == "sqlite":

        async def open_sqlite():
            return SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_SCOPE)

        return open_sqlite
    location = request.getfixturevalue("postgres_dsn")

    async def open_postgres():
        import psycopg
        from psycopg import sql

        # Each test starts from an empty schema in the module's database.
        async with await psycopg.AsyncConnection.connect(location) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
                )
                for (table,) in await cursor.fetchall():
                    await cursor.execute(
                        sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(str(table)))
                    )
            await connection.commit()
        return PostgresKnowledgeStore(
            location, access_scope=_SCOPE, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
        )

    return open_postgres


async def _kept_entry(store, workflow: KnowledgeReviewWorkflow, entry_id: str) -> KnowledgeEntry:
    """An entry kept through reviewed approval, the way a reviewer keeps a proposal."""

    await store.create_entry(
        KnowledgeEntry(
            id=entry_id,
            text=f"The scout runs one process ({entry_id}).",
            status=KnowledgeStatus.PENDING,
        ),
        access_scope=_SCOPE,
    )
    approval = await workflow.approve(
        entry_id,
        operation_id=f"keep-{entry_id}",
        reviewer_identity="reviewer",
        reviewer_version="1",
        annotations={"reason": "Accurate when kept."},
    )
    assert approval.entry.status is KnowledgeStatus.ACTIVE
    return approval.entry


async def _found(store, text: str) -> list[str]:
    result = await store.search(KnowledgeQuery(any_terms=text.split()), access_scope=_SCOPE)
    return list(dict.fromkeys(hit.entry.id for hit in result.hits))


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def test_reviewed_archive_takes_a_kept_entry_out_of_search_and_records_why(open_store):
    async def scenario():
        store = await open_store()
        try:
            workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
            kept = await _kept_entry(store, workflow, "stale")
            assert await _found(store, "scout process") == ["stale"]

            archived = await workflow.archive_reviewed(
                "stale",
                expected_revision=kept.revision,
                operation_id="archive-stale",
                reviewer_identity="reviewer",
                reviewer_version="1",
                reason="The worker now runs three processes.",
                annotations={"channel": "test"},
            )
            assert archived.entry.status is KnowledgeStatus.ARCHIVED
            assert archived.entry.revision == kept.revision + 1
            assert archived.entry.text == kept.text
            assert archived.decision.reason == "The worker now runs three processes."
            assert archived.receipt.replayed is False
            current = await store.get_entry("stale", access_scope=_SCOPE)
            assert current.status is KnowledgeStatus.ARCHIVED
            assert current.metadata["cayu:review_archive"]["reason"] == (
                "The worker now runs three processes."
            )
            assert await _found(store, "scout process") == []

            # The same decision under the same operation replays; it doesn't conflict.
            replayed = await workflow.archive_reviewed(
                "stale",
                expected_revision=kept.revision,
                operation_id="archive-stale",
                reviewer_identity="reviewer",
                reviewer_version="1",
                reason="The worker now runs three processes.",
                annotations={"channel": "test"},
            )
            assert replayed.receipt.replayed is True
            assert replayed.entry == archived.entry
            loaded = await workflow.load_review_archive(
                "stale", expected_revision=kept.revision, operation_id="archive-stale"
            )
            assert loaded is not None and loaded.decision == archived.decision

            # A different reason under that operation, or archiving again, conflicts.
            with pytest.raises(KnowledgePublicationConflict):
                await workflow.archive_reviewed(
                    "stale",
                    expected_revision=kept.revision,
                    operation_id="archive-stale",
                    reviewer_identity="reviewer",
                    reviewer_version="1",
                    reason="Another reason.",
                )
            with pytest.raises(KnowledgePublicationConflict):
                await workflow.archive_reviewed(
                    "stale",
                    expected_revision=archived.entry.revision,
                    operation_id="archive-again",
                    reviewer_identity="reviewer",
                    reviewer_version="1",
                    reason="Twice.",
                )
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_reviewed_archive_refuses_pending_and_drop_refuses_kept(open_store):
    async def scenario():
        store = await open_store()
        try:
            workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
            kept = await _kept_entry(store, workflow, "kept")
            await store.create_entry(
                KnowledgeEntry(id="proposal", text="A proposal", status=KnowledgeStatus.PENDING),
                access_scope=_SCOPE,
            )
            with pytest.raises(KnowledgePublicationConflict):
                await workflow.archive_reviewed(
                    "proposal",
                    expected_revision=1,
                    operation_id="archive-proposal",
                    reviewer_identity="reviewer",
                    reviewer_version="1",
                    reason="Not kept yet.",
                )
            with pytest.raises(KnowledgePublicationConflict):
                await workflow.reject_reviewed(
                    "kept",
                    expected_revision=kept.revision,
                    operation_id="drop-kept",
                    reviewer_identity="reviewer",
                    reviewer_version="1",
                    reason="Already kept.",
                )
            dropped = await workflow.reject_reviewed(
                "proposal",
                expected_revision=1,
                operation_id="drop-proposal",
                reviewer_identity="reviewer",
                reviewer_version="1",
                reason="A duplicate.",
            )
            assert dropped.entry.status is KnowledgeStatus.ARCHIVED
            assert dropped.decision.reason == "A duplicate."
            # A drop's receipt isn't an archive's, and the reverse.
            assert (
                await workflow.load_review_rejection(
                    "proposal", expected_revision=1, operation_id="drop-proposal"
                )
                is not None
            )
            with pytest.raises(KnowledgePublicationConflict):
                await workflow.load_review_archive(
                    "proposal", expected_revision=1, operation_id="drop-proposal"
                )
            assert (await store.get_entry("kept", access_scope=_SCOPE)).status is (
                KnowledgeStatus.ACTIVE
            )
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_reviewed_archive_stays_inside_the_review_scope(open_store):
    async def scenario():
        store = await open_store()
        try:
            workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
            await store.create_entry(
                KnowledgeEntry(
                    id="other-team",
                    text="Another team's entry",
                    status=KnowledgeStatus.PENDING,
                    labels={"team": "other"},
                ),
                access_scope=_SCOPE,
            )
            await workflow.approve(
                "other-team",
                operation_id="keep-other",
                reviewer_identity="reviewer",
                reviewer_version="1",
            )
            scoped = KnowledgeReviewWorkflow(store, access_scope=_SCOPE, labels={"team": "mine"})
            with pytest.raises(PermissionError):
                await scoped.archive_reviewed(
                    "other-team",
                    expected_revision=2,
                    operation_id="archive-other",
                    reviewer_identity="reviewer",
                    reviewer_version="1",
                    reason="Outside the scope.",
                )
            assert (await store.get_entry("other-team", access_scope=_SCOPE)).status is (
                KnowledgeStatus.ACTIVE
            )
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_reviewed_archive_copies_a_kept_entry_larger_than_a_proposal_may_be(open_store):
    from cayu.storage import KnowledgeChunk

    async def scenario():
        store = await open_store()
        try:
            workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
            # An indexed document: 300 chunks of 4,000 bytes, past the 1 MiB proposal bound.
            entry = KnowledgeEntry(id="doc", text="A large indexed document.")
            chunks = [
                KnowledgeChunk(
                    id=f"doc-{index}",
                    entry_id="doc",
                    chunk_index=index,
                    text=f"{index:04d}" + "x" * 3_996,
                )
                for index in range(300)
            ]
            await store.create_entry(entry, chunks, access_scope=_SCOPE)
            archived = await workflow.archive_reviewed(
                "doc",
                expected_revision=1,
                operation_id="archive-doc",
                reviewer_identity="reviewer",
                reviewer_version="1",
                reason="Superseded.",
            )
            assert archived.entry.status is KnowledgeStatus.ARCHIVED
            copied = await store.read_chunks(
                "doc", revision=2, access_scope=_SCOPE, max_chunks=1_000, max_bytes=2_000_000
            )
            assert [chunk.text for chunk in copied] == [chunk.text for chunk in chunks]
        finally:
            await _close(store)

    asyncio.run(scenario())
