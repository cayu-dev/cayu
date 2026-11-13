"""Knowledge lists drive from ``cayu_knowledge_entry_order``, kept by triggers.

SQLite and Postgres keep one order row per entry with its current status and sort
key. These tests hold that the rows follow creation, new revisions, status
changes and deletion, that a list read without a total pages exactly as one with
it, and that it reports the total only on the last page. With the unrestricted
scope these lists use, the total is counted from the order table alone.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeListQuery, encode_knowledge_list_cursor
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.migrations import SchemaMode

SCOPE = KnowledgeAccessScope.privileged()
NOW = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.fixture(params=["sqlite", "postgres"])
def open_store(request, tmp_path):
    if request.param == "sqlite":
        return lambda: SQLiteKnowledgeStore(tmp_path / "order.sqlite", access_scope=SCOPE)
    dsn = request.getfixturevalue("postgres_dsn")

    def postgres():
        from cayu.storage import PostgresKnowledgeStore

        return PostgresKnowledgeStore(dsn, schema_mode=SchemaMode.CREATE, access_scope=SCOPE)

    return postgres


async def _page_ids(store, namespace: str, status: KnowledgeStatus, *, count_total: bool):
    pages, cursor = [], None
    while True:
        listed = await store.list_entries(
            KnowledgeListQuery(
                namespace=namespace,
                statuses=[status],
                cursor=cursor,
                limit=2,
                count_total=count_total,
            ),
            access_scope=SCOPE,
        )
        pages.append(([item.entry.id for item in listed.entries], listed.total_entries_known))
        more = listed.truncated and (
            listed.total_entries_known is None or listed.total_entries_known > len(listed.entries)
        )
        if not more:
            return pages
        cursor = encode_knowledge_list_cursor(listed.entries[-1].entry)


def test_order_rows_follow_entries_and_paging_matches_without_a_total(open_store) -> None:
    namespace = "order-" + uuid4().hex[:8]

    async def scenario():
        store = open_store()
        try:
            for index, importance in enumerate((None, 0.9, None, 0.5, 0.9)):
                at = NOW + timedelta(minutes=index)
                await store.create_entry(
                    KnowledgeEntry(
                        id=f"{namespace}-{index}",
                        namespace=namespace,
                        text=f"Entry {index}.",
                        status=KnowledgeStatus.PENDING,
                        importance=importance,
                        created_at=at,
                        updated_at=at,
                    ),
                    access_scope=SCOPE,
                )
            # An expired proposal is neither listed nor counted.
            await store.create_entry(
                KnowledgeEntry(
                    id=f"{namespace}-expired",
                    namespace=namespace,
                    text="Expired entry.",
                    status=KnowledgeStatus.PENDING,
                    importance=1.0,
                    created_at=NOW,
                    updated_at=NOW,
                    expires_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                access_scope=SCOPE,
            )
            counted = await _page_ids(store, namespace, KnowledgeStatus.PENDING, count_total=True)
            uncounted = await _page_ids(
                store, namespace, KnowledgeStatus.PENDING, count_total=False
            )
            # Keep one entry, and give another a new revision that raises it to the top.
            await store.transition_entry_status(
                f"{namespace}-0",
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
                access_scope=SCOPE,
            )
            current = await store.get_entry(f"{namespace}-2", access_scope=SCOPE)
            assert current is not None
            await store.append_entry_revision(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "importance": 1.0,
                        "updated_at": NOW + timedelta(hours=1),
                    }
                ),
                expected_revision=current.revision,
                access_scope=SCOPE,
            )
            await store.delete_entry(f"{namespace}-3", expected_revision=1, access_scope=SCOPE)
            pending = await _page_ids(store, namespace, KnowledgeStatus.PENDING, count_total=False)
            active = await _page_ids(store, namespace, KnowledgeStatus.ACTIVE, count_total=False)
            return counted, uncounted, pending, active
        finally:
            await store.close()

    counted, uncounted, pending, active = asyncio.run(scenario())
    order = [f"{namespace}-{index}" for index in (4, 1, 3, 2, 0)]
    assert [ids for ids, _ in counted] == [order[:2], order[2:4], order[4:]]
    # A page after a cursor counts the entries from there on.
    assert [total for _, total in counted] == [5, 3, 1]
    assert [ids for ids, _ in uncounted] == [ids for ids, _ in counted]
    # Without a total, only the last page knows how many matched.
    assert [total for _, total in uncounted] == [None, None, 1]
    after = [f"{namespace}-{index}" for index in (2, 4, 1)]
    assert [ids for ids, _ in pending] == [after[:2], after[2:]]
    assert [ids for ids, _ in active] == [[f"{namespace}-0"]]
