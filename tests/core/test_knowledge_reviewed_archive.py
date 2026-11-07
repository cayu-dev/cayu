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


def test_an_archive_by_revision_alone_records_no_earlier_reviewed_decision(open_store):
    """A later revision copies the metadata of a reviewed archive unchanged; the archived
    list attributes the decision only to the revision the decision wrote."""

    async def scenario():
        store = await open_store()
        try:
            workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
            kept = await _kept_entry(store, workflow, "restored")
            archived = await workflow.archive_reviewed(
                "restored",
                expected_revision=kept.revision,
                operation_id="archive-restored",
                reviewer_identity="reviewer",
                reviewer_version="1",
                reason="Stale.",
            )
            assert await workflow.archived_decision(archived.entry) == {
                "decision": "archive",
                "decided_by": "reviewer",
                "reason": "Stale.",
            }
            # Restored through the store, then retired by revision alone.
            for revision, from_status, to_status in (
                (archived.entry.revision, KnowledgeStatus.ARCHIVED, KnowledgeStatus.ACTIVE),
                (archived.entry.revision + 1, KnowledgeStatus.ACTIVE, KnowledgeStatus.ARCHIVED),
            ):
                retired = await store.transition_entry_status(
                    "restored",
                    expected_revision=revision,
                    access_scope=_SCOPE,
                    from_status=from_status,
                    to_status=to_status,
                )
            assert retired.metadata["cayu:review_archive"]["reason"] == "Stale."
            (listed,) = (await workflow.list_archived()).entries
            assert listed.entry.revision == retired.revision
            assert await workflow.archived_decision(listed.entry) == {
                "decision": None,
                "decided_by": None,
                "reason": None,
            }
            # Restored again and archived under a new operation with the same decision:
            # that publication wrote this revision, so the decision is attributed to it.
            restored = await store.transition_entry_status(
                "restored",
                expected_revision=retired.revision,
                access_scope=_SCOPE,
                from_status=KnowledgeStatus.ARCHIVED,
                to_status=KnowledgeStatus.ACTIVE,
            )
            again = await workflow.archive_reviewed(
                "restored",
                expected_revision=restored.revision,
                operation_id="archive-restored-again",
                reviewer_identity="reviewer",
                reviewer_version="1",
                reason="Stale.",
            )
            assert again.entry.metadata == retired.metadata
            assert await workflow.archived_decision(again.entry) == {
                "decision": "archive",
                "decided_by": "reviewer",
                "reason": "Stale.",
            }
            receipt = await store.load_entry_revision_publication_receipt(
                again.entry, access_scope=_SCOPE
            )
            assert receipt is not None and receipt.operation_id == "archive-restored-again"
            assert (
                await store.load_entry_revision_publication_receipt(retired, access_scope=_SCOPE)
                is None
            )
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_a_reused_entry_id_attributes_each_archive_to_its_own_entry(open_store):
    """Publication receipts outlive a hard delete, so a recreated entry id has receipts
    for the same revision from its earlier life; each archive keeps its own decision."""

    async def scenario():
        store = await open_store()
        try:
            workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
            for life, reason in (("first", "Stale."), ("second", "Wrong.")):
                kept = await store.create_entry(
                    KnowledgeEntry(
                        id="reused", text=f"The {life} note.", status=KnowledgeStatus.ACTIVE
                    ),
                    access_scope=_SCOPE,
                )
                archived = await workflow.archive_reviewed(
                    "reused",
                    expected_revision=kept.revision,
                    operation_id=f"archive-{life}",
                    reviewer_identity="reviewer",
                    reviewer_version="1",
                    reason=reason,
                )
                (listed,) = (await workflow.list_archived()).entries
                assert listed.entry == archived.entry
                assert await workflow.archived_decision(listed.entry) == {
                    "decision": "archive",
                    "decided_by": "reviewer",
                    "reason": reason,
                }
                receipt = await store.load_entry_revision_publication_receipt(
                    listed.entry, access_scope=_SCOPE
                )
                assert receipt is not None and receipt.operation_id == f"archive-{life}"
                if life == "first":
                    await store.delete_entry(
                        "reused",
                        expected_revision=archived.entry.revision,
                        access_scope=_SCOPE,
                        hard=True,
                    )
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_another_publication_of_copied_review_metadata_records_no_decision(open_store):
    """A revision some other publication wrote, carrying an earlier decision's metadata,
    isn't attributed to that decision: only the publication the decision itself makes is."""

    from cayu.storage import KnowledgeChunk

    async def scenario():
        store = await open_store()
        try:
            workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
            kept = await _kept_entry(store, workflow, "edited")
            archived = await workflow.archive_reviewed(
                "edited",
                expected_revision=kept.revision,
                operation_id="archive-edited",
                reviewer_identity="Alice",
                reviewer_version="1",
                reason="Old text is stale",
            )
            restored = await store.transition_entry_status(
                "edited",
                expected_revision=archived.entry.revision,
                access_scope=_SCOPE,
                from_status=KnowledgeStatus.ARCHIVED,
                to_status=KnowledgeStatus.ACTIVE,
            )
            successor = restored.model_copy(
                update={
                    "revision": restored.revision + 1,
                    "status": KnowledgeStatus.ARCHIVED,
                    "text": "Edited text.",
                }
            )
            await store.publish_entry_revision(
                successor,
                [
                    KnowledgeChunk(
                        id="edited:edit:0",
                        entry_id="edited",
                        entry_revision=successor.revision,
                        chunk_index=0,
                        text="Edited text.",
                    )
                ],
                operation_id="publish-edited",
                expected_revision=restored.revision,
                access_scope=_SCOPE,
            )
            (listed,) = (await workflow.list_archived()).entries
            assert listed.entry.metadata["cayu:review_archive"]["reviewer_identity"] == "Alice"
            assert await workflow.archived_decision(listed.entry) == {
                "decision": None,
                "decided_by": None,
                "reason": None,
            }
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_a_bound_store_without_review_reads_still_records_fresh_decisions():
    """A bound store written before review reads refuses the review scope; reviewed drops
    and archives still record there, as before, and archived entries just aren't shown."""

    from cayu.storage import KnowledgeAccessDenied

    recall = KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )

    class OlderStore(InMemoryKnowledgeStore):
        def _review_read_access_scope(self, access_scope):
            return self._operation_access_scope(access_scope)

    async def scenario():
        store = OlderStore(access_scope=recall)
        for entry_id, status in (
            ("kept", KnowledgeStatus.ACTIVE),
            ("proposal", KnowledgeStatus.PENDING),
        ):
            await store.create_entry(
                KnowledgeEntry(id=entry_id, text=f"Made-up {entry_id}", status=status)
            )
        workflow = KnowledgeReviewWorkflow(store)
        decision = {"reviewer_identity": "reviewer", "reviewer_version": "1"}
        dropped = await workflow.reject_reviewed(
            "proposal", expected_revision=1, operation_id="drop", reason="Duplicate.", **decision
        )
        archived = await workflow.archive_reviewed(
            "kept", expected_revision=1, operation_id="archive", reason="Stale.", **decision
        )
        assert dropped.entry.status is archived.entry.status is KnowledgeStatus.ARCHIVED
        assert await workflow.get_archived("kept") is None
        assert await workflow.archived_decision(archived.entry) == {
            "decision": None,
            "decided_by": None,
            "reason": None,
        }
        with pytest.raises(KnowledgeAccessDenied):
            await workflow.list_archived()

    asyncio.run(scenario())
