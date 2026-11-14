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
from cayu.artifacts._settlement import _settle_artifact_write
from cayu.artifacts.aws_s3 import S3ArtifactStore
from cayu.artifacts.base import ArtifactMetadata, ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.artifacts.settlement import ArtifactWriteSettlementStatus, artifact_write_settlements
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
        assert await store.read_artifact_index_stores(["s"]) == {}
        # A fill completes only if its claim is the newest and nothing unfilled the
        # store since it began.
        a, b, c = NOW + timedelta(4), NOW + timedelta(5), NOW + timedelta(6)
        await store.claim_artifact_index_fill("s", claimed_at=a)
        assert await store.read_artifact_index_stores(["s"]) == {}
        assert await store.complete_artifact_index_fill("s", claimed_at=a, backfilled_at=b)
        assert await store.read_artifact_index_stores(["s"]) == {"s": b}
        assert not await store.complete_artifact_index_fill("s", claimed_at=a, backfilled_at=c)
        await store.claim_artifact_index_fill("s", claimed_at=a)
        assert await store.read_artifact_index_stores(["s"]) == {"s": b}
        await store.mark_artifact_index_store("s", backfilled_at=None)
        assert not await store.complete_artifact_index_fill("s", claimed_at=a, backfilled_at=c)
        assert await store.read_artifact_index_stores(["s"]) == {}
        # Fill a began, a late write unfilled the store, fill b began: a cannot
        # complete on b's claim, and b can.
        await store.claim_artifact_index_fill("s", claimed_at=a)
        await store.mark_artifact_index_store("s", backfilled_at=None)
        await store.claim_artifact_index_fill("s", claimed_at=b)
        assert not await store.complete_artifact_index_fill("s", claimed_at=a, backfilled_at=c)
        assert await store.read_artifact_index_stores(["s"]) == {}
        assert await store.complete_artifact_index_fill("s", claimed_at=b, backfilled_at=c)
        assert await store.read_artifact_index_stores(["s"]) == {"s": c}
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


async def test_an_unfill_racing_a_completing_postgres_fill_leaves_the_store_unfilled(
    request,
) -> None:
    import psycopg

    from cayu.storage import _artifact_index_sql as index_sql
    from cayu.storage.postgres import PostgresSessionStore

    dsn = request.getfixturevalue("postgres_dsn")
    store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
    await store.ensure_schema()
    try:
        await store.claim_artifact_index_fill("race", claimed_at=NOW)
        async with await psycopg.AsyncConnection.connect(dsn) as fill:
            # The fill takes back its claim but has not committed yet.
            await fill.execute(
                cast("LiteralString", index_sql.release_claim_sql("postgres")),
                index_sql.claim_fill_params("postgres", "race", NOW),
            )
            unfill = asyncio.create_task(
                store.mark_artifact_index_store("race", backfilled_at=None)
            )
            async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as watch:
                for _ in range(200):
                    cursor = await watch.execute(
                        "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock'"
                        " AND query LIKE 'DELETE FROM cayu_artifact_index_stores%%'"
                    )
                    row = await cursor.fetchone()
                    if row is not None and row[0] > 0:
                        break
                    await asyncio.sleep(0.025)
                else:
                    pytest.fail("The unfill never waited on the fill's claim.")
            # The fill writes its marker and commits while the unfill waits.
            await fill.execute(
                cast("LiteralString", index_sql.mark_store_sql("postgres")),
                index_sql.mark_store_params("postgres", "race", NOW + timedelta(1)),
            )
            await fill.commit()
        await unfill
        assert await store.read_artifact_index_stores(["race"]) == {}
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


async def _filled_s3_store(tmp_path):
    from tests.artifacts.test_aws_s3 import _BlockingMetadataCommitS3Client

    sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    client = _BlockingMetadataCommitS3Client()
    artifacts = S3ArtifactStore("bucket", client=client, store_id="s3")
    bind_artifact_index(artifacts, sessions)
    await reindex_artifact_store(sessions, "s3", artifacts)
    assert (await sessions.read_artifact_index_stores(["s3"]))["s3"] is not None
    return sessions, client, artifacts


async def _cancel_during_commit(client, artifacts) -> asyncio.Task[Any]:
    put = asyncio.create_task(artifacts.put_bytes(b"late", filename="late.txt", session_id="owner"))
    assert await asyncio.to_thread(client.metadata_commit_started.wait, 2)
    put.cancel("caller stopped")
    return put


async def test_a_write_that_commits_while_its_caller_is_cancelled_unfills_the_store(
    tmp_path,
) -> None:
    sessions, client, artifacts = await _filled_s3_store(tmp_path)
    try:
        put = await _cancel_during_commit(client, artifacts)
        client.release_metadata_commit.set()
        with pytest.raises(asyncio.CancelledError, match="caller stopped") as raised:
            await put
        (evidence,) = artifact_write_settlements(raised.value)
        assert evidence.status is ArtifactWriteSettlementStatus.COMMITTED
        # Reads walk the store until the next fill, which finds the artifact.
        assert await sessions.read_artifact_index_stores(["s3"]) == {}
        drift = await reindex_artifact_store(sessions, "s3", artifacts)
        assert drift.missing == (evidence.artifact_id,)
    finally:
        await sessions.close()


async def test_a_write_still_settling_unfills_the_store_again_once_it_lands(
    tmp_path, monkeypatch
) -> None:
    defaults = dict(_settle_artifact_write.__kwdefaults__ or {})
    monkeypatch.setattr(
        _settle_artifact_write, "__kwdefaults__", {**defaults, "settlement_timeout_s": 0.05}
    )
    sessions, client, artifacts = await _filled_s3_store(tmp_path)
    try:
        put = await _cancel_during_commit(client, artifacts)
        with pytest.raises(asyncio.CancelledError) as raised:
            await put
        (evidence,) = artifact_write_settlements(raised.value)
        assert evidence.status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
        # A fill that finishes before the write lands misses it and marks the store filled.
        await sessions.mark_artifact_index_store("s3", backfilled_at=datetime.now(UTC))
        assert (await sessions.read_artifact_index_stores(["s3"]))["s3"] is not None
        client.release_metadata_commit.set()
        await asyncio.gather(*artifact_index._PENDING_REPAIRS)
        assert await sessions.read_artifact_index_stores(["s3"]) == {}
        drift = await reindex_artifact_store(sessions, "s3", artifacts)
        assert drift.missing == (evidence.artifact_id,)
    finally:
        await sessions.close()


async def test_a_fill_walking_the_store_when_a_late_write_lands_leaves_it_unfilled(
    tmp_path, monkeypatch
) -> None:
    sessions, client, artifacts = await _filled_s3_store(tmp_path)
    try:
        client.release_metadata_commit.set()
        await artifacts.put_bytes(b"late", filename="late.txt", session_id="owner")
        await sessions.mark_artifact_index_store("s3", backfilled_at=None)
        complete = sessions.complete_artifact_index_fill

        async def after_a_late_write_lands(
            store_id: str, *, claimed_at: datetime, backfilled_at: datetime
        ) -> bool:
            # A write that settles while the refill walks the store marks it unfilled
            # (what the commit hook does once a late write lands).
            await artifact_index._invalidate(artifacts)
            return await complete(store_id, claimed_at=claimed_at, backfilled_at=backfilled_at)

        monkeypatch.setattr(sessions, "complete_artifact_index_fill", after_a_late_write_lands)
        refill = await reindex_artifact_store(sessions, "s3", artifacts)
        assert refill.backfilled_at is None
        assert await sessions.read_artifact_index_stores(["s3"]) == {}
        monkeypatch.undo()
        drift = await reindex_artifact_store(sessions, "s3", artifacts)
        assert drift.backfilled_at is not None
        assert (await sessions.read_artifact_index_stores(["s3"]))["s3"] is not None
    finally:
        await sessions.close()


async def test_a_scoped_access_denial_keeps_the_store_filled(tmp_path) -> None:
    from cayu.artifacts._index import _may_have_changed
    from cayu.sessions.access import SessionAccessDenied

    sessions, _, artifacts = await _filled_s3_store(tmp_path)
    try:
        assert not _may_have_changed(artifacts, "delete", SessionAccessDenied())
        assert not _may_have_changed(artifacts, "put", SessionAccessDenied())
    finally:
        await sessions.close()


async def test_a_rejected_write_keeps_the_store_filled(tmp_path) -> None:
    sessions, _, artifacts = await _filled_s3_store(tmp_path)
    try:
        with pytest.raises(ValueError):
            await artifacts.put_bytes(b"x", filename="", session_id="owner")
        assert (await sessions.read_artifact_index_stores(["s3"]))["s3"] is not None
    finally:
        await sessions.close()
