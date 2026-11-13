"""The ``cayu_artifact_index`` contract on the SQLite and Postgres session stores.

Rows follow commits and deletions by the time each writer observed them, reads
filter, search, order and page with keysets, counts cover live session-scoped
rows, titles follow the session, and the migration adds the tables to an older
store.
"""

from __future__ import annotations

import asyncio
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any, LiteralString, cast
from uuid import uuid4

import pytest

from cayu import RunRequest
from cayu.artifacts import _index as artifact_index
from cayu.artifacts import _scan as artifact_scan
from cayu.artifacts._index import ArtifactIndexQuery, bind_artifact_index, reindex_artifact_store
from cayu.artifacts.base import ArtifactMetadata, ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.sessions.records import SessionIdentity
from cayu.storage.migrations import SchemaMode, SchemaTooOld
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.anyio
NOW = datetime(2026, 5, 1, 12, tzinfo=UTC)


class _Backend:
    def __init__(self, name: str, tmp_path, request) -> None:
        self.name = name
        if name == "sqlite":
            self.path = tmp_path / "artifact-index.sqlite"
        else:
            self.dsn = request.getfixturevalue("postgres_dsn")

    async def open(self, schema_mode: SchemaMode = SchemaMode.CREATE):
        if self.name == "sqlite":
            return SQLiteSessionStore(self.path, schema_mode=schema_mode)
        from cayu.storage.postgres import PostgresSessionStore

        store = PostgresSessionStore(self.dsn, schema_mode=schema_mode)
        try:
            await store.ensure_schema()
        except BaseException:
            await store.close()
            raise
        return store

    async def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self.name == "sqlite":
            connection = sqlite3.connect(self.path)
            try:
                rows = [tuple(row) for row in connection.execute(sql, params).fetchall()]
                connection.commit()
                return rows
            finally:
                connection.close()
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            cursor = await conn.execute(cast("LiteralString", sql.replace("?", "%s")), params)
            return [tuple(row) for row in await cursor.fetchall()] if cursor.description else []


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path) -> _Backend:
    return _Backend(request.param, tmp_path, request)


def _artifact(index: int, **update) -> ArtifactMetadata:
    values: dict[str, Any] = {
        "id": f"artifact-{index:04}",
        "filename": f"Report-{index:04}.txt",
        "content_type": "text/plain",
        "size_bytes": (index * 37) % 101,
        "session_id": f"session-{index % 3}",
        "agent_name": f"agent-{index % 2}",
        "environment_name": f"env-{index % 4}",
        "created_at": NOW + timedelta(seconds=index // 2),
    }
    values.update(update)
    return ArtifactMetadata(**values)


async def _session(store, session_id: str, title: str | None) -> None:
    await store.create(
        RunRequest(session_id=session_id, agent_name="writer", title=title, messages=[]),
        identity=SessionIdentity(provider_name="configured", model="configured"),
    )


async def _all_pages(store, **query) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    after = None
    while True:
        page = await store.query_artifact_index(ArtifactIndexQuery(after=after, **query))
        rows.extend((row.store_id, row.artifact.id) for row in page.rows)
        if page.next_cursor is None:
            return rows
        after = page.next_cursor


def _reference(artifacts: dict[tuple[str, str], ArtifactMetadata], order_by: str, keep=None):
    def key(item):
        (store_id, artifact_id), artifact = item
        if order_by.startswith("filename_"):
            primary = artifact.filename.lower()
        elif order_by.startswith("size_bytes_"):
            primary = artifact.size_bytes
        else:
            primary = artifact.created_at
        return primary, store_id, artifact_id

    selected = [item for item in artifacts.items() if keep is None or keep(item[1])]
    return [
        identity for identity, _ in sorted(selected, key=key, reverse=order_by.endswith("_desc"))
    ]


async def test_rows_follow_writes_by_the_time_their_writer_observed_them(backend) -> None:
    store = await backend.open()
    try:
        artifact = _artifact(1)
        await store.record_indexed_artifacts("s", [artifact], observed_at=NOW)
        # A deletion that finished later wins over a slow commit hook.
        await store.record_deleted_artifacts("s", ["artifact-0001"], deleted_at=NOW + timedelta(1))
        await store.record_indexed_artifacts("s", [artifact], observed_at=NOW)
        query = ArtifactIndexQuery(store_ids=("s",))
        assert (await store.query_artifact_index(query)).total_count == 0
        # A newer commit of the same identity revives it.
        await store.record_indexed_artifacts("s", [artifact], observed_at=NOW + timedelta(2))
        page = await store.query_artifact_index(query)
        assert [row.artifact for row in page.rows] == [artifact]
        # A deletion of an artifact the index never saw leaves a tombstone a
        # slower backfill cannot overwrite.
        await store.record_deleted_artifacts("s", ["artifact-0002"], deleted_at=NOW + timedelta(3))
        await store.record_indexed_artifacts("s", [_artifact(2)], observed_at=NOW)
        assert [row.artifact.id for row in (await store.query_artifact_index(query)).rows] == [
            "artifact-0001"
        ]
        assert await store.scan_artifact_index("s", after=None, limit=10) == (
            ("artifact-0001", _json(artifact)),
        )
        assert await store.read_artifact_index_stores(["s", "t"]) == {}
        await store.mark_artifact_index_store("s", backfilled_at=NOW)
        assert await store.read_artifact_index_stores(["s", "t"]) == {"s": NOW}
        await store.mark_artifact_index_store("s", backfilled_at=None)
        assert await store.read_artifact_index_stores(["s"]) == {"s": None}
    finally:
        await store.close()


def _json(artifact: ArtifactMetadata) -> str:
    from cayu.artifacts._index import metadata_json

    return metadata_json(artifact)


async def test_pages_filter_order_and_continue_like_the_reference(backend) -> None:
    store = await backend.open()
    try:
        generator = random.Random(3947)
        live: dict[tuple[str, str], ArtifactMetadata] = {}
        clock = NOW
        for step in range(160):
            clock += timedelta(milliseconds=1)
            store_id = generator.choice(("a", "b"))
            index = generator.randrange(60)
            if generator.random() < 0.25:
                await store.record_deleted_artifacts(
                    store_id, [f"artifact-{index:04}"], deleted_at=clock
                )
                live.pop((store_id, f"artifact-{index:04}"), None)
            else:
                artifact = _artifact(
                    index,
                    size_bytes=generator.randrange(5),
                    filename=generator.choice(("b.txt", "A.txt", "a.txt", f"x-{step}.md")),
                )
                await store.record_indexed_artifacts(store_id, [artifact], observed_at=clock)
                live[(store_id, artifact.id)] = artifact
        for order_by in (
            "created_at_desc",
            "created_at_asc",
            "filename_asc",
            "filename_desc",
            "size_bytes_desc",
            "size_bytes_asc",
        ):
            cases: tuple[tuple[dict[str, Any], Any], ...] = (
                ({}, None),
                ({"session_id": "session-1"}, lambda a: a.session_id == "session-1"),
                ({"environment_name": "env-2"}, lambda a: a.environment_name == "env-2"),
                ({"agent_name": "agent-0"}, lambda a: a.agent_name == "agent-0"),
                ({"q": "X-1"}, lambda a: "x-1" in a.filename.lower()),
            )
            for filters, keep in cases:
                expected = _reference(live, order_by, keep)
                got = await _all_pages(
                    store, store_ids=("a", "b"), order_by=order_by, limit=7, **filters
                )
                assert got == expected, (order_by, filters)
                total = await store.query_artifact_index(
                    ArtifactIndexQuery(store_ids=("a", "b"), order_by=order_by, **filters)
                )
                assert total.total_count == len(expected)
            offset = await store.query_artifact_index(
                ArtifactIndexQuery(store_ids=("a",), order_by=order_by, offset=3, limit=4)
            )
            only_a = _reference({k: v for k, v in live.items() if k[0] == "a"}, order_by)
            assert [("a", row.artifact.id) for row in offset.rows] == only_a[3:7]
        counts = await store.count_indexed_session_artifacts(
            ["a", "b"], ["session-0", "session-1", "session-9"]
        )
        assert counts == {
            session_id: sum(1 for a in live.values() if a.session_id == session_id)
            for session_id in ("session-0", "session-1", "session-9")
        }
    finally:
        await store.close()


async def test_search_reads_titles_that_follow_the_session(backend) -> None:
    store = await backend.open()
    try:
        session_id = "titled-" + uuid4().hex
        await _session(store, session_id, "Quarterly report")
        saved = _artifact(7, session_id=session_id, filename="chart.png", content_type="image/png")
        other = _artifact(8, session_id=None, scope=ArtifactScope.ENVIRONMENT)
        await store.record_indexed_artifacts("s", [saved, other], observed_at=NOW)

        async def found(q: str) -> list[tuple[str, str | None]]:
            page = await store.query_artifact_index(ArtifactIndexQuery(store_ids=("s",), q=q))
            return [(row.artifact.id, row.title) for row in page.rows]

        assert await found("QUARTER") == [("artifact-0007", "Quarterly report")]
        assert await found("png image") == [("artifact-0007", "Quarterly report")]
        assert await found("rl") == [("artifact-0007", "Quarterly report")]
        assert await found("report-0008") == [("artifact-0008", None)]
        # A literal: wildcards and quotes match only themselves.
        assert await found("%") == []
        assert await found('"') == []
        # Any writer that changes the title, older binaries included, fires the trigger.
        if backend.name == "sqlite":
            await backend.execute(
                "UPDATE cayu_sessions SET metadata_json = "
                "json_set(metadata_json, '$.\"cayu:title\"', ?) WHERE id = ?",
                ("Annual review", session_id),
            )
        else:
            await backend.execute(
                "UPDATE cayu_sessions SET metadata = "
                "jsonb_set(metadata, '{cayu:title}', to_jsonb(?::text)) WHERE id = ?",
                ("Annual review", session_id),
            )
        assert await found("quarter") == []
        assert await found("annual") == [("artifact-0007", "Annual review")]
        await store.delete_session(session_id)
        assert await found("annual") == []
        assert await found("chart") == [("artifact-0007", None)]
        counts = await store.count_indexed_session_artifacts(["s"], [session_id])
        assert counts == {session_id: 1}
    finally:
        await store.close()


async def test_migration_adds_the_index_to_an_older_store(backend) -> None:
    store = await backend.open()
    await store.close()
    on_sessions = "" if backend.name == "sqlite" else " ON cayu_sessions"
    for name in ("insert", "update", "delete"):
        await backend.execute(f"DROP TRIGGER cayu_artifact_index_title_{name}{on_sessions}")
    if backend.name == "sqlite":
        await backend.execute("DROP TABLE cayu_artifact_index_fts")
    await backend.execute("DROP TABLE cayu_artifact_index")
    await backend.execute("DROP TABLE cayu_artifact_index_stores")
    from cayu.storage._artifact_index_schema import ARTIFACT_INDEX_REVISION

    await backend.execute(
        "DELETE FROM cayu_schema_migrations WHERE revision >= ?", (ARTIFACT_INDEX_REVISION,)
    )
    with pytest.raises(SchemaTooOld, match="requires >= "):
        await (await backend.open()).close()
    store = await backend.open(SchemaMode.MIGRATE)
    try:
        await store.record_indexed_artifacts("s", [_artifact(1)], observed_at=NOW)
        page = await store.query_artifact_index(ArtifactIndexQuery(store_ids=("s",), q="report"))
        assert page.total_count == 1
    finally:
        await store.close()


async def test_cursors_continue_after_their_row_is_deleted(backend) -> None:
    store = await backend.open()
    try:
        await store.record_indexed_artifacts(
            "s", [_artifact(index) for index in range(6)], observed_at=NOW
        )
        query = ArtifactIndexQuery(store_ids=("s",), order_by="filename_desc", limit=2)
        first = await store.query_artifact_index(query)
        assert first.next_cursor is not None
        # The cursor names its row, never the row's filename.
        assert "report" not in first.next_cursor.encode().lower()
        await store.record_deleted_artifacts(
            "s", [first.rows[-1].artifact.id], deleted_at=NOW + timedelta(1)
        )
        rest = await _all_pages(store, store_ids=("s",), order_by="filename_desc", limit=2)
        page = await store.query_artifact_index(
            ArtifactIndexQuery(
                store_ids=("s",), order_by="filename_desc", limit=10, after=first.next_cursor
            )
        )
        assert [row.artifact.id for row in page.rows] == [
            f"artifact-{index:04}" for index in (3, 2, 1, 0)
        ]
        assert len(rest) == 5
    finally:
        await store.close()


async def test_a_deletion_that_finishes_during_a_reindex_wins(tmp_path) -> None:
    sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowDeletes(LocalArtifactStore):
        async def delete(self, artifact_id: str) -> None:
            entered.set()
            await release.wait()
            await super().delete(artifact_id)

    artifacts = SlowDeletes(tmp_path / "artifacts", store_id="slow")
    bind_artifact_index(artifacts, sessions)
    try:
        saved = await artifacts.put_bytes(b"x", filename="x.txt", session_id="owner")
        deleting = asyncio.create_task(artifacts.delete(saved.id))
        await entered.wait()
        # The walk begins after the deletion began, and still sees the artifact.
        drift = await reindex_artifact_store(sessions, "slow", artifacts)
        assert drift.store_count == 1
        release.set()
        await deleting
        page = await sessions.query_artifact_index(ArtifactIndexQuery(store_ids=("slow",)))
        assert page.total_count == 0
    finally:
        await sessions.close()


async def test_a_store_over_the_scan_bound_is_filled_from_its_whole_list(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(artifact_scan, "ARTIFACT_SCAN_RECORD_MAX", 5)
    monkeypatch.setattr(artifact_index, "ARTIFACT_INDEX_BATCH_MAX", 5)
    sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="big")
    bind_artifact_index(artifacts, sessions)
    try:
        for index in range(7):
            await artifacts.put_bytes(b"x", filename=f"file-{index}.txt", session_id="owner")
        drift = await reindex_artifact_store(sessions, "big", artifacts)
        assert (drift.store_count, drift.drift) == (7, False)
        assert drift.backfilled_at is not None
    finally:
        await sessions.close()
