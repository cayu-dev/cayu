from __future__ import annotations

import asyncio
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu.runtime import RunRequest, SessionIdentity
from cayu.storage import (
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeQuery,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()


async def _close(store: object) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def _reconcile_through_revision_37(
    connection: sqlite3.Connection,
    schema_mode: schema_migrations.SchemaMode,
) -> None:
    """Exercise the historical migration without crossing the revision-42 reset."""

    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 37
        )
        sqlite_support.reconcile_schema(
            connection,
            schema_mode,
            app_min_supported=37,
        )
    finally:
        schema_migrations.REVISIONS = revisions


def _migrate_legacy_through_revision_37(db_path: Path) -> None:
    connection = sqlite_support.connect(db_path)
    try:
        _reconcile_through_revision_37(
            connection,
            schema_migrations.SchemaMode.MIGRATE,
        )
    finally:
        connection.close()


def _assert_peer_writer_can_begin(db_path: Path) -> None:
    """Prove a failed migration released SQLite's writer lock at any schema revision."""

    peer = sqlite_support.connect(db_path)
    try:
        peer.execute("PRAGMA busy_timeout = 100")
        peer.execute("BEGIN IMMEDIATE")
        peer.commit()
    finally:
        peer.close()


def _chunks(entry_id: str, *texts: str) -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            id=f"{entry_id}:{index}",
            entry_id=entry_id,
            chunk_index=index,
            text=text,
        )
        for index, text in enumerate(texts)
    ]


def _downgrade_knowledge_layout_to_revision_36(db_path: Path) -> None:
    connection = sqlite_support.connect(db_path)
    try:
        # This helper deliberately dismantles revision 42's mutually dependent
        # entry/revision tables to fabricate a historical revision-36 database.
        # Disable enforcement only for that test-only reconstruction; the
        # resulting legacy schema is reopened with enforcement enabled.
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        connection.executescript(
            """
            CREATE TEMP TABLE legacy_knowledge_entries AS
            SELECT * FROM cayu_knowledge_current_entries;
            CREATE TEMP TABLE legacy_knowledge_labels AS
            SELECT labels.entry_id, labels.key, labels.value
            FROM cayu_knowledge_labels AS labels
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = labels.entry_id
             AND logical.current_revision = labels.entry_revision;
            CREATE TEMP TABLE legacy_knowledge_aspects AS
            SELECT aspects.entry_id, aspects.aspect
            FROM cayu_knowledge_aspects AS aspects
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = aspects.entry_id
             AND logical.current_revision = aspects.entry_revision;
            CREATE TEMP TABLE legacy_knowledge_impact_targets AS
            SELECT targets.entry_id, targets.impact_target
            FROM cayu_knowledge_impact_targets AS targets
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = targets.entry_id
             AND logical.current_revision = targets.entry_revision;
            CREATE TEMP TABLE legacy_knowledge_chunks AS
            SELECT chunks.*
            FROM cayu_knowledge_chunks AS chunks
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = chunks.entry_id
             AND logical.current_revision = chunks.entry_revision;

            DROP VIEW cayu_knowledge_current_entries;
            DROP TABLE cayu_knowledge_chunks_fts;
            DROP TABLE cayu_knowledge_publication_receipts;
            DROP TABLE cayu_knowledge_chunks;
            DROP TABLE cayu_knowledge_impact_targets;
            DROP TABLE cayu_knowledge_aspects;
            DROP TABLE cayu_knowledge_labels;
            DROP TABLE cayu_knowledge_revisions;
            DROP TABLE cayu_knowledge_entries;
            """
        )
        connection.executescript(sqlite_support._MIGRATION_STEPS[6])
        connection.execute("DROP TABLE cayu_knowledge_chunks_fts")
        connection.execute("DROP TABLE cayu_knowledge_chunks")
        connection.execute(
            """
            CREATE TABLE cayu_knowledge_chunks (
                id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL
                    REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT,
                source_uri TEXT,
                metadata_json TEXT NOT NULL,
                UNIQUE (entry_id, chunk_index)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id, namespace, text, kind, visibility, status,
                created_by_type, created_by, created_at, updated_at,
                source_type, source_uri, source_id, source_hash,
                importance, importance_source, confidence, last_used_at,
                expires_at, title, metadata_json
            )
            SELECT
                id, namespace, text, kind, visibility, status,
                created_by_type, COALESCE(created_by, ''), created_at, updated_at,
                source_type, source_uri, source_id, source_hash,
                importance, importance_source, confidence, last_used_at,
                expires_at, title, metadata_json
            FROM legacy_knowledge_entries
            """
        )
        connection.execute(
            "INSERT INTO cayu_knowledge_labels (entry_id, key, value) "
            "SELECT entry_id, key, value FROM legacy_knowledge_labels"
        )
        connection.execute(
            "INSERT INTO cayu_knowledge_aspects (entry_id, aspect) "
            "SELECT entry_id, aspect FROM legacy_knowledge_aspects"
        )
        connection.execute(
            "INSERT INTO cayu_knowledge_impact_targets (entry_id, impact_target) "
            "SELECT entry_id, impact_target FROM legacy_knowledge_impact_targets"
        )
        connection.execute(
            """
            INSERT INTO cayu_knowledge_chunks (
                rowid, id, entry_id, chunk_index, text,
                content_hash, source_uri, metadata_json
            )
            SELECT
                fts_rowid, id, entry_id, chunk_index, text,
                content_hash, source_uri, metadata_json
            FROM legacy_knowledge_chunks
            ORDER BY fts_rowid
            """
        )
        connection.execute(
            "CREATE INDEX idx_cayu_knowledge_chunks_entry_index "
            "ON cayu_knowledge_chunks(entry_id, chunk_index)"
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE cayu_knowledge_chunks_fts
            USING fts5(entry_id UNINDEXED, chunk_id UNINDEXED, title, text)
            """
        )
        # Legacy FTS rowids were unrelated to source rowids. Deliberately offset
        # them so the migration test cannot pass through accidental insertion order.
        connection.execute(
            """
            INSERT INTO cayu_knowledge_chunks_fts (
                rowid, entry_id, chunk_id, title, text
            )
            SELECT
                chunk.rowid + 100000,
                chunk.entry_id,
                chunk.id,
                COALESCE(entry.title, ''),
                CASE
                    WHEN chunk.text = entry.text THEN chunk.text
                    ELSE entry.text || char(10) || chunk.text
                END
            FROM cayu_knowledge_chunks AS chunk
            JOIN cayu_knowledge_entries AS entry ON entry.id = chunk.entry_id
            ORDER BY chunk.rowid
            """
        )
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 37")
        connection.execute("PRAGMA user_version = 36")
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _raw_ranked_matches(db_path: Path, query: str) -> list[tuple[str, float]]:
    connection = sqlite3.connect(db_path)
    try:
        return [
            (str(row[0]), float(row[1]))
            for row in connection.execute(
                """
                SELECT chunk_id, bm25(cayu_knowledge_chunks_fts)
                FROM cayu_knowledge_chunks_fts
                WHERE cayu_knowledge_chunks_fts MATCH ?
                ORDER BY bm25(cayu_knowledge_chunks_fts), chunk_id
                """,
                (query,),
            )
        ]
    finally:
        connection.close()


class _MigrationBoundaryConnection:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        failure: str,
        rollback_failure: bool = False,
    ) -> None:
        self._connection = connection
        self.failure: str | None = failure
        self.rollback_failure = rollback_failure
        self.rollback_calls = 0
        self.close_calls = 0

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def execute(self, sql: str, *args):
        cursor = self._connection.execute(sql, *args)
        if self.failure == "after_begin" and " ".join(sql.upper().split()) == "BEGIN IMMEDIATE":
            self.failure = None
            signal.raise_signal(signal.SIGINT)
        return cursor

    def commit(self) -> None:
        # `reconcile_schema` commits its idempotent bookkeeping-table creation
        # before entering the revision transaction. Inject only at the commit
        # owned by `_transaction`, where SQLite still reports active ownership.
        failure = self.failure if self._connection.in_transaction else None
        if failure == "before_commit":
            self.failure = None
            raise sqlite3.OperationalError("injected commit failure with live transaction")
        self._connection.commit()
        if failure == "after_commit":
            self.failure = None
            raise sqlite3.OperationalError("injected commit acknowledgement loss")

    def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_failure:
            raise sqlite3.OperationalError("injected rollback failure")
        self._connection.rollback()

    def close(self) -> None:
        self.close_calls += 1
        self._connection.close()

    def __getattr__(self, name: str):
        return getattr(self._connection, name)


def _assert_ordered_transaction_failures(
    failure: BaseExceptionGroup,
    *,
    primary_type: type[BaseException],
    primary_message: str,
) -> None:
    assert len(failure.exceptions) == 2
    primary, rollback = failure.exceptions
    assert isinstance(primary, primary_type)
    assert str(primary) == primary_message
    assert isinstance(rollback, sqlite3.OperationalError)
    assert str(rollback) == "injected rollback failure"
    assert rollback.__context__ is None


def test_sqlite_transaction_successful_rollback_preserves_primary_context(
    tmp_path: Path,
) -> None:
    connection = sqlite_support.connect(tmp_path / "primary-context.sqlite")
    connection.execute("CREATE TABLE evidence (value TEXT NOT NULL)")
    connection.commit()
    try:
        with (
            pytest.raises(RuntimeError, match="primary transaction failure") as caught,
            sqlite_support._transaction(connection),
        ):
            connection.execute("INSERT INTO evidence (value) VALUES ('partial')")
            try:
                raise ValueError("original causal evidence")
            except ValueError as cause:
                raise RuntimeError("primary transaction failure") from cause
        assert isinstance(caught.value.__cause__, ValueError)
        assert str(caught.value.__cause__) == "original causal evidence"
        assert not connection.in_transaction
        assert connection.execute("SELECT value FROM evidence").fetchall() == []
    finally:
        connection.close()


def _assert_exact_fts_mapping(
    connection: sqlite3.Connection,
    *,
    entry_id: str,
    chunk_ids: list[str],
) -> None:
    entry_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(cayu_knowledge_entries)")
    }
    current_join = (
        "JOIN cayu_knowledge_entries AS logical "
        "ON logical.id = chunk.entry_id "
        "AND logical.current_revision = chunk.entry_revision"
        if "current_revision" in entry_columns
        else ""
    )
    rows = connection.execute(
        f"""
        SELECT
            chunk.fts_rowid,
            fts.rowid,
            chunk.id,
            fts.chunk_id,
            chunk.entry_id,
            fts.entry_id
        FROM cayu_knowledge_chunks AS chunk
        JOIN cayu_knowledge_chunks_fts AS fts ON fts.rowid = chunk.fts_rowid
        {current_join}
        WHERE chunk.entry_id = ?
        ORDER BY chunk.chunk_index
        """,
        (entry_id,),
    ).fetchall()
    assert [str(row[2]) for row in rows] == chunk_ids
    assert all(int(row[0]) == int(row[1]) for row in rows)
    assert all(str(row[2]) == str(row[3]) for row in rows)
    assert all(str(row[4]) == str(row[5]) == entry_id for row in rows)


def _seed_unrelated_corpus(
    connection: sqlite3.Connection,
    *,
    chunk_count: int,
) -> None:
    if chunk_count == 0:
        return
    entry = KnowledgeEntry(id="unrelated", title="Other", text="unrelated corpus")
    with connection:
        connection.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id, namespace, current_revision, created_at, updated_at
            )
            VALUES (?, ?, 1, ?, ?)
            """,
            (
                entry.id,
                entry.namespace,
                sqlite_support.format_datetime(entry.created_at),
                sqlite_support.format_datetime(entry.updated_at),
            ),
        )
        connection.execute(
            """
            INSERT INTO cayu_knowledge_revisions (
                entry_id, revision, text, kind, visibility, status,
                created_by_type, created_by, created_at, updated_at,
                source_type, source_uri, source_id, source_hash,
                importance, importance_source, confidence, last_used_at,
                expires_at, title, metadata_json
            )
            VALUES (
                ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                entry.id,
                entry.text,
                str(entry.kind),
                str(entry.visibility),
                str(entry.status),
                str(entry.created_by_type),
                entry.created_by,
                sqlite_support.format_datetime(entry.created_at),
                sqlite_support.format_datetime(entry.updated_at),
                entry.source_type,
                entry.source_uri,
                entry.source_id,
                entry.source_hash,
                entry.importance,
                entry.importance_source,
                entry.confidence,
                None,
                None,
                entry.title,
                "{}",
            ),
        )
        connection.executemany(
            """
            INSERT INTO cayu_knowledge_chunks (
                id, entry_id, entry_revision, chunk_index, text,
                content_hash, source_uri, metadata_json
            )
            VALUES (?, 'unrelated', 1, ?, ?, NULL, NULL, '{}')
            """,
            [
                (f"unrelated:r1:{index}", index, f"unrelated body {index}")
                for index in range(chunk_count)
            ],
        )
        connection.execute(
            """
            INSERT INTO cayu_knowledge_chunks_fts (
                rowid, entry_id, entry_revision, chunk_id, title, text
            )
            SELECT
                fts_rowid, entry_id, entry_revision, id, 'Other',
                'unrelated corpus' || char(10) || text
            FROM cayu_knowledge_chunks
            WHERE entry_id = 'unrelated'
            ORDER BY chunk_index
            """
        )


def _unrelated_fts_rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in connection.execute(
            """
            SELECT rowid, entry_id, chunk_id, title, text
            FROM cayu_knowledge_chunks_fts
            WHERE entry_id = 'unrelated'
            ORDER BY rowid
            """
        )
    ]


def _wait_for_file(path: Path, process: subprocess.Popen[bytes]) -> None:
    for _ in range(500):
        if path.exists():
            return
        if process.poll() is not None:
            raise AssertionError(f"child exited before publishing {path.name}")
        time.sleep(0.01)
    raise AssertionError(f"child did not publish {path.name}")


def test_revision_37_migrates_legacy_fts_and_preserves_ranking(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-knowledge.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(
                KnowledgeEntry(id="alpha", title="Shared", text="shared summary"),
                _chunks("alpha", "shared first", "shared second"),
            )
            await store.create_entry(
                KnowledgeEntry(id="beta", title="Other", text="shared summary"),
                _chunks("beta", "shared third"),
            )
        finally:
            await _close(store)

    asyncio.run(seed())
    _downgrade_knowledge_layout_to_revision_36(db_path)
    before = _raw_ranked_matches(db_path, "shared")

    legacy = sqlite3.connect(db_path)
    try:
        assert legacy.execute("PRAGMA user_version").fetchone()[0] == 36
        assert (
            legacy.execute(
                """
            SELECT COUNT(*)
            FROM cayu_knowledge_chunks AS chunk
            JOIN cayu_knowledge_chunks_fts AS fts
              ON fts.rowid = chunk.rowid
            """
            ).fetchone()[0]
            == 0
        )
    finally:
        legacy.close()

    _migrate_legacy_through_revision_37(db_path)
    after = _raw_ranked_matches(db_path, "shared")

    assert [row[0] for row in after] == [row[0] for row in before]
    assert [row[1] for row in after] == pytest.approx([row[1] for row in before])
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 37
        assert connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 37"
        ).fetchone() == ("breaking", 37)
        columns = connection.execute("PRAGMA table_info(cayu_knowledge_chunks)").fetchall()
        assert columns[0][1:3] == ("fts_rowid", "INTEGER")
        assert columns[0][5] == 1
        _assert_exact_fts_mapping(
            connection,
            entry_id="alpha",
            chunk_ids=["alpha:0", "alpha:1"],
        )
        _assert_exact_fts_mapping(
            connection,
            entry_id="beta",
            chunk_ids=["beta:0"],
        )
    finally:
        connection.close()

    _migrate_legacy_through_revision_37(db_path)


def test_revision_37_failure_rolls_back_legacy_schema_and_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "failed-migration.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="legacy", text="searchable legacy"))
        finally:
            await _close(store)

    asyncio.run(seed())
    _downgrade_knowledge_layout_to_revision_36(db_path)
    original = sqlite_support._validate_revision_37_knowledge_fts_data

    def fail_after_rebuild(connection: sqlite3.Connection) -> None:
        original(connection)
        raise RuntimeError("injected revision-37 validation failure")

    monkeypatch.setattr(
        sqlite_support,
        "_validate_revision_37_knowledge_fts_data",
        fail_after_rebuild,
    )
    connection = sqlite_support.connect(db_path)
    try:
        with pytest.raises(RuntimeError, match="injected revision-37"):
            _reconcile_through_revision_37(
                connection,
                schema_migrations.SchemaMode.MIGRATE,
            )
    finally:
        connection.close()

    check = sqlite3.connect(db_path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 36
        assert "fts_rowid" not in {
            str(row[1]) for row in check.execute("PRAGMA table_info(cayu_knowledge_chunks)")
        }
        assert (
            check.execute(
                """
            SELECT chunk_id
            FROM cayu_knowledge_chunks_fts
            WHERE cayu_knowledge_chunks_fts MATCH 'searchable'
            """
            ).fetchone()[0]
            == "legacy:r1:0"
        )
    finally:
        check.close()

    monkeypatch.setattr(
        sqlite_support,
        "_validate_revision_37_knowledge_fts_data",
        original,
    )
    _migrate_legacy_through_revision_37(db_path)


@pytest.mark.parametrize(
    ("failure", "expected_exception"),
    [
        ("after_begin", KeyboardInterrupt),
        ("before_commit", sqlite3.OperationalError),
    ],
)
def test_revision_37_transaction_boundary_failure_rolls_back_and_retries(
    tmp_path: Path,
    failure: str,
    expected_exception: type[BaseException],
) -> None:
    db_path = tmp_path / f"migration-{failure}.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="legacy", text="legacy searchable"))
        finally:
            await _close(store)

    asyncio.run(seed())
    _downgrade_knowledge_layout_to_revision_36(db_path)
    connection = sqlite_support.connect(db_path)
    boundary = _MigrationBoundaryConnection(connection, failure=failure)
    previous_sigint_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(
            expected_exception, match="injected" if failure != "after_begin" else None
        ):
            _reconcile_through_revision_37(
                boundary,  # type: ignore[arg-type]
                schema_migrations.SchemaMode.MIGRATE,
            )
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    try:
        assert boundary.rollback_calls == 1
        assert not connection.in_transaction
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 36

        _assert_peer_writer_can_begin(db_path)

        _reconcile_through_revision_37(
            boundary,  # type: ignore[arg-type]
            schema_migrations.SchemaMode.MIGRATE,
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 37
    finally:
        connection.close()

    assert [chunk_id for chunk_id, _ in _raw_ranked_matches(db_path, "legacy searchable")] == [
        "legacy:r1:0"
    ]


def test_revision_37_commit_acknowledgement_loss_preserves_committed_migration(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "migration-commit-acknowledgement.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="legacy", text="legacy searchable"))
        finally:
            await _close(store)

    asyncio.run(seed())
    _downgrade_knowledge_layout_to_revision_36(db_path)
    connection = sqlite_support.connect(db_path)
    boundary = _MigrationBoundaryConnection(connection, failure="after_commit")
    try:
        with pytest.raises(sqlite3.OperationalError, match="acknowledgement loss"):
            _reconcile_through_revision_37(
                boundary,  # type: ignore[arg-type]
                schema_migrations.SchemaMode.MIGRATE,
            )
        assert boundary.rollback_calls == 0
        assert not connection.in_transaction
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 37

        # Retrying on the same connection observes the durable marker and only
        # validates the already-complete source/FTS relationship.
        _reconcile_through_revision_37(
            boundary,  # type: ignore[arg-type]
            schema_migrations.SchemaMode.MIGRATE,
        )
    finally:
        connection.close()

    _assert_peer_writer_can_begin(db_path)
    assert [chunk_id for chunk_id, _ in _raw_ranked_matches(db_path, "legacy searchable")] == [
        "legacy:r1:0"
    ]


def test_revision_37_commit_and_rollback_failure_fences_connection_and_retries(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "migration-commit-rollback-failure.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="legacy", text="legacy searchable"))
        finally:
            await _close(store)

    asyncio.run(seed())
    _downgrade_knowledge_layout_to_revision_36(db_path)
    connection = sqlite_support.connect(db_path)
    boundary = _MigrationBoundaryConnection(
        connection,
        failure="before_commit",
        rollback_failure=True,
    )
    with pytest.raises(ExceptionGroup) as caught:
        _reconcile_through_revision_37(
            boundary,  # type: ignore[arg-type]
            schema_migrations.SchemaMode.MIGRATE,
        )
    _assert_ordered_transaction_failures(
        caught.value,
        primary_type=sqlite3.OperationalError,
        primary_message="injected commit failure with live transaction",
    )
    assert caught.value.__suppress_context__
    assert boundary.rollback_calls == 1
    assert boundary.close_calls == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")

    # Closing the uncertain owner releases the writer without publishing any of
    # the interrupted schema or FTS rebuild.
    check = sqlite_support.connect(db_path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 36
        assert (
            sqlite_support._sqlite_table_columns(check, "cayu_knowledge_chunks")
            == sqlite_support._KNOWLEDGE_CHUNK_LEGACY_COLUMNS
        )
        assert (
            check.execute(
                "SELECT chunk_id FROM cayu_knowledge_chunks_fts "
                "WHERE cayu_knowledge_chunks_fts MATCH 'searchable'"
            ).fetchone()[0]
            == "legacy:r1:0"
        )
    finally:
        check.close()

    _assert_peer_writer_can_begin(db_path)

    retry = sqlite_support.connect(db_path)
    try:
        _reconcile_through_revision_37(retry, schema_migrations.SchemaMode.MIGRATE)
        assert retry.execute("PRAGMA user_version").fetchone()[0] == 37
    finally:
        retry.close()

    assert [chunk_id for chunk_id, _ in _raw_ranked_matches(db_path, "legacy searchable")] == [
        "legacy:r1:0"
    ]


@pytest.mark.parametrize("primary_kind", ["ordinary", "sigint"])
def test_sqlite_knowledge_rollback_failure_preserves_primary_and_fences_store(
    tmp_path: Path,
    primary_kind: str,
) -> None:
    class FailingStore(SQLiteKnowledgeStore):
        failure_kind: str | None = None

        def _insert_entry_fts_unlocked(self, entry, chunks) -> None:
            super()._insert_entry_fts_unlocked(entry, chunks)
            if self.failure_kind == "ordinary":
                raise RuntimeError("injected knowledge mutation failure")
            if self.failure_kind == "sigint":
                signal.raise_signal(signal.SIGINT)

    db_path = tmp_path / f"knowledge-rollback-failure-{primary_kind}.sqlite"

    async def run() -> None:
        store = FailingStore(db_path, access_scope=_ACCESS_SCOPE)
        peer: SQLiteKnowledgeStore | None = None
        try:
            await store.create_entry(KnowledgeEntry(id="target", text="originaltoken"))
            connection = store._connection
            boundary = _MigrationBoundaryConnection(
                connection,
                failure="none",
                rollback_failure=True,
            )
            store._connection = boundary  # type: ignore[assignment]
            store.failure_kind = primary_kind
            previous_sigint_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
            try:
                with pytest.raises(BaseExceptionGroup) as caught:
                    await store.append_entry_revision(
                        KnowledgeEntry(
                            id="target",
                            revision=2,
                            text="replacementtoken",
                            created_at=(await store.get_entry("target")).created_at,
                        ),
                        expected_revision=1,
                    )
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)

            primary_type: type[BaseException]
            primary_message: str
            if primary_kind == "ordinary":
                assert isinstance(caught.value, ExceptionGroup)
                primary_type = RuntimeError
                primary_message = "injected knowledge mutation failure"
            else:
                assert not isinstance(caught.value, ExceptionGroup)
                primary_type = KeyboardInterrupt
                primary_message = ""
            _assert_ordered_transaction_failures(
                caught.value,
                primary_type=primary_type,
                primary_message=primary_message,
            )
            assert caught.value.__suppress_context__
            assert boundary.rollback_calls == 1
            assert boundary.close_calls == 1
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                await store.get_entry("target")

            peer = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
            peer._connection.execute("PRAGMA busy_timeout = 100")
            loaded = await peer.get_entry("target")
            assert loaded is not None and loaded.text == "originaltoken"
            assert [
                hit.entry.id
                for hit in (await peer.search(KnowledgeQuery(text="originaltoken"))).hits
            ] == ["target"]
            assert (await peer.search(KnowledgeQuery(text="replacementtoken"))).hits == []
            await peer.create_entry(KnowledgeEntry(id="peer", text="peer write"))
        finally:
            if peer is not None:
                await _close(peer)
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_mutations_keep_exact_fts_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "mutations.sqlite"

    async def run() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            entry = KnowledgeEntry(id="mutable", title="Original", text="old summary")
            await store.create_entry(
                entry,
                _chunks("mutable", "old first", "old second"),
            )
            _assert_exact_fts_mapping(
                store._connection,
                entry_id="mutable",
                chunk_ids=["mutable:0", "mutable:1"],
            )

            await store.append_entry_revision(
                entry.model_copy(update={"revision": 2, "title": "Revised", "text": "new summary"}),
                expected_revision=1,
            )
            assert [
                hit.entry.id for hit in (await store.search(KnowledgeQuery(text="Revised"))).hits
            ] == ["mutable"]
            assert (await store.search(KnowledgeQuery(text="Original"))).hits == []

            await store.append_entry_revision(
                entry.model_copy(update={"revision": 3, "title": "Revised", "text": "new summary"}),
                [
                    KnowledgeChunk(
                        id="mutable:replacement",
                        entry_id="mutable",
                        entry_revision=3,
                        chunk_index=0,
                        text="replacement body",
                    )
                ],
                expected_revision=2,
            )
            _assert_exact_fts_mapping(
                store._connection,
                entry_id="mutable",
                chunk_ids=["mutable:replacement"],
            )
            assert (await store.search(KnowledgeQuery(text="old second"))).hits == []

            published = KnowledgeEntry(id="published", text="published body")
            await store.publish_entry_revision(
                published,
                _chunks("published", "published body"),
                operation_id="publish-operation",
            )
            _assert_exact_fts_mapping(
                store._connection,
                entry_id="published",
                chunk_ids=["published:0"],
            )

            expired = KnowledgeEntry(
                id="expired",
                text="expired body",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
            await store.create_entry(expired)
            assert await store.prune_expired() == 1
            assert (
                store._connection.execute(
                    "SELECT 1 FROM cayu_knowledge_chunks_fts WHERE entry_id = 'expired'"
                ).fetchone()
                is None
            )

            assert await store.delete_entry("mutable", expected_revision=3, hard=True) is not None
            assert (
                store._connection.execute(
                    "SELECT 1 FROM cayu_knowledge_chunks_fts WHERE entry_id = 'mutable'"
                ).fetchone()
                is None
            )
            source_count = store._connection.execute(
                "SELECT COUNT(*) FROM cayu_knowledge_chunks"
            ).fetchone()[0]
            fts_count = store._connection.execute(
                "SELECT COUNT(*) FROM cayu_knowledge_chunks_fts"
            ).fetchone()[0]
            assert source_count == fts_count == 1
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("failure_phase", ["before_fts_insert", "after_fts_insert"])
def test_sqlite_knowledge_refresh_failure_rolls_back_source_and_fts(
    tmp_path: Path,
    failure_phase: str,
) -> None:
    class FailingStore(SQLiteKnowledgeStore):
        fail_at: str | None = None

        def _insert_entry_fts_unlocked(self, entry, chunks) -> None:
            if self.fail_at == "before_fts_insert":
                raise RuntimeError("injected before FTS insert")
            super()._insert_entry_fts_unlocked(entry, chunks)
            if self.fail_at == "after_fts_insert":
                raise RuntimeError("injected after FTS insert")

    async def run() -> None:
        store = FailingStore(
            tmp_path / f"rollback-{failure_phase}.sqlite",
            access_scope=_ACCESS_SCOPE,
        )
        try:
            original = await store.create_entry(KnowledgeEntry(id="atomic", text="original body"))
            store.fail_at = failure_phase
            with pytest.raises(RuntimeError, match="injected"):
                await store.append_entry_revision(
                    original.model_copy(update={"revision": 2, "text": "revised body"}),
                    expected_revision=1,
                )
            store.fail_at = None

            loaded = await store.get_entry("atomic")
            assert loaded is not None and loaded.text == "original body"
            assert [
                hit.entry.id for hit in (await store.search(KnowledgeQuery(text="original"))).hits
            ] == ["atomic"]
            assert (await store.search(KnowledgeQuery(text="revised"))).hits == []
            _assert_exact_fts_mapping(
                store._connection,
                entry_id="atomic",
                chunk_ids=["atomic:r1:0"],
            )
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_knowledge_cancellation_before_writer_admission_is_atomic(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "cancelled.sqlite", access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="cancelled", text="original"))
            await store._lock.acquire()
            task = asyncio.create_task(
                store.create_entry(KnowledgeEntry(id="cancelled", text="replacement"))
            )
            try:
                await asyncio.sleep(0)
                assert task.cancel()
                assert task.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()
            finally:
                store._lock.release()
            loaded = await store.get_entry("cancelled")
            assert loaded is not None and loaded.text == "original"
            assert [
                hit.entry.id for hit in (await store.search(KnowledgeQuery(text="original"))).hits
            ] == ["cancelled"]
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("entrance", "interrupt_hook"),
    [
        ("append_default", "fts"),
        ("append_chunks", "fts"),
        ("publish_entry_revision", "receipt"),
    ],
)
def test_sqlite_knowledge_interruption_rolls_back_and_releases_writer(
    tmp_path: Path,
    entrance: str,
    interrupt_hook: str,
) -> None:
    class InterruptingStore(SQLiteKnowledgeStore):
        interrupt_at: str | None = None

        def _insert_entry_fts_unlocked(self, entry, chunks) -> None:
            super()._insert_entry_fts_unlocked(entry, chunks)
            if self.interrupt_at == "fts":
                signal.raise_signal(signal.SIGINT)

        def _insert_publication_receipt_unlocked(self, receipt, entry) -> None:
            super()._insert_publication_receipt_unlocked(receipt, entry)
            if self.interrupt_at == "receipt":
                signal.raise_signal(signal.SIGINT)

    db_path = tmp_path / f"interrupted-{entrance}.sqlite"

    async def run() -> None:
        store = InterruptingStore(db_path, access_scope=_ACCESS_SCOPE)
        peer = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        peer._connection.execute("PRAGMA busy_timeout = 100")
        try:
            original: KnowledgeEntry | None = None
            if entrance != "publish_entry_revision":
                original = await store.create_entry(
                    KnowledgeEntry(id="target", title="Original", text="originaltoken"),
                    _chunks("target", "original chunk"),
                )

            store.interrupt_at = interrupt_hook
            previous_sigint_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
            try:
                with pytest.raises(KeyboardInterrupt):
                    if entrance == "append_default":
                        assert original is not None
                        await store.append_entry_revision(
                            original.model_copy(
                                update={
                                    "revision": 2,
                                    "title": "Replacement",
                                    "text": "replacementtoken",
                                }
                            ),
                            expected_revision=1,
                        )
                    elif entrance == "append_chunks":
                        assert original is not None
                        await store.append_entry_revision(
                            original.model_copy(
                                update={
                                    "revision": 2,
                                    "title": "Replacement",
                                    "text": "replacementtoken",
                                }
                            ),
                            [
                                KnowledgeChunk(
                                    id="target:r2:0",
                                    entry_id="target",
                                    entry_revision=2,
                                    chunk_index=0,
                                    text="replacement chunk",
                                )
                            ],
                            expected_revision=1,
                        )
                    else:
                        await store.publish_entry_revision(
                            KnowledgeEntry(id="target", text="replacementtoken"),
                            _chunks("target", "replacement chunk"),
                            operation_id="interrupted-publication",
                        )
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)
            store.interrupt_at = None

            assert not store._connection.in_transaction
            if entrance == "publish_entry_revision":
                assert await store.get_entry("target") is None
                assert await store.load_entry_publication_receipt("interrupted-publication") is None
                assert (await store.search(KnowledgeQuery(text="replacementtoken"))).hits == []
            else:
                loaded = await store.get_entry("target")
                assert loaded is not None
                assert loaded.title == "Original"
                assert loaded.text == "originaltoken"
                assert [chunk.id for chunk in await store.read_chunks("target")] == ["target:0"]
                assert [
                    hit.entry.id
                    for hit in (await store.search(KnowledgeQuery(text="originaltoken"))).hits
                ] == ["target"]
                assert (await store.search(KnowledgeQuery(text="replacementtoken"))).hits == []

            await peer.create_entry(KnowledgeEntry(id="peer", text="peer write"))
            assert await peer.get_entry("peer") is not None
        finally:
            await _close(peer)
            await _close(store)

    asyncio.run(run())


def _measure_single_entry_refresh(
    db_path: Path,
    *,
    unrelated_chunk_count: int,
) -> tuple[int, list[str], list[tuple[object, ...]], list[tuple[object, ...]], str]:
    async def run() -> tuple[
        int,
        list[str],
        list[tuple[object, ...]],
        list[tuple[object, ...]],
        str,
    ]:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            target = await store.create_entry(
                KnowledgeEntry(id="target", title="Before", text="target summary"),
                _chunks("target", "target one", "target two"),
            )
            _seed_unrelated_corpus(
                store._connection,
                chunk_count=unrelated_chunk_count,
            )
            unrelated_before = _unrelated_fts_rows(store._connection)
            statements: list[str] = []
            progress_calls = 0

            def count_progress() -> int:
                nonlocal progress_calls
                progress_calls += 1
                return 0

            store._connection.set_trace_callback(statements.append)
            store._connection.set_progress_handler(count_progress, 100)
            try:
                await store.append_entry_revision(
                    target.model_copy(
                        update={"revision": 2, "title": "After", "text": "revised summary"}
                    ),
                    expected_revision=1,
                )
            finally:
                store._connection.set_progress_handler(None, 0)
                store._connection.set_trace_callback(None)
            unrelated_after = _unrelated_fts_rows(store._connection)
            plan = " ".join(
                str(row[3])
                for row in store._connection.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT id, fts_rowid
                    FROM cayu_knowledge_chunks
                        INDEXED BY idx_cayu_knowledge_chunks_entry_revision_index
                    WHERE entry_id = 'target' AND entry_revision = 2
                    ORDER BY chunk_index
                    """
                )
            )
            return (
                progress_calls,
                statements,
                unrelated_before,
                unrelated_after,
                plan,
            )
        finally:
            await _close(store)

    return asyncio.run(run())


@pytest.mark.stress
def test_single_entry_fts_refresh_is_independent_of_unrelated_corpus(tmp_path: Path) -> None:
    small = _measure_single_entry_refresh(
        tmp_path / "small.sqlite",
        unrelated_chunk_count=0,
    )
    large = _measure_single_entry_refresh(
        tmp_path / "large.sqlite",
        unrelated_chunk_count=5000,
    )

    small_progress, _, _, _, _ = small
    large_progress, statements, before, after, plan = large
    assert large_progress <= small_progress + 20
    assert before == after
    normalized = [" ".join(statement.lower().split()) for statement in statements]
    fts_inserts = [
        statement
        for statement in normalized
        if statement.startswith("insert into cayu_knowledge_chunks_fts")
    ]
    assert fts_inserts
    assert all("entry_revision" in statement for statement in fts_inserts)
    assert "idx_cayu_knowledge_chunks_entry_revision_index" in plan


@pytest.mark.stress
def test_bounded_knowledge_refresh_does_not_starve_shared_checkpoint_write(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "shared.sqlite"

    async def seed() -> None:
        knowledge = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        session_store = SQLiteSessionStore(db_path)
        try:
            await knowledge.create_entry(
                KnowledgeEntry(id="target", text="target summary"),
                _chunks("target", "target one", "target two"),
            )
            _seed_unrelated_corpus(knowledge._connection, chunk_count=5000)
            await session_store.create(
                RunRequest(agent_name="agent", session_id="shared-session", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        finally:
            await _close(knowledge)
            await _close(session_store)

    asyncio.run(seed())
    knowledge = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
    checkpoint_store = SQLiteSessionStore(db_path)
    knowledge_paused = threading.Event()
    release_knowledge = threading.Event()
    checkpoint_begin = threading.Event()
    errors: list[BaseException] = []
    progress_calls = 0

    original_insert_fts = knowledge._insert_entry_fts_unlocked

    def pause_after_target_fts_insert(
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> None:
        original_insert_fts(entry, chunks)
        if entry.id == "target" and entry.revision == 2:
            knowledge_paused.set()
            if not release_knowledge.wait(5):
                raise RuntimeError("checkpoint writer did not contend for SQLite ownership")

    def bound_knowledge_work() -> int:
        nonlocal progress_calls
        progress_calls += 1
        return int(progress_calls > 100)

    knowledge._insert_entry_fts_unlocked = pause_after_target_fts_insert  # type: ignore[method-assign]
    knowledge._connection.set_progress_handler(bound_knowledge_work, 100)
    checkpoint_store._connection.set_trace_callback(
        lambda statement: checkpoint_begin.set() if statement.startswith("BEGIN") else None
    )

    def refresh_knowledge() -> None:
        try:
            current = asyncio.run(knowledge.get_entry("target"))
            assert current is not None
            asyncio.run(
                knowledge.append_entry_revision(
                    current.model_copy(update={"revision": 2, "text": "revised target summary"}),
                    expected_revision=1,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    def write_checkpoint() -> None:
        try:
            asyncio.run(checkpoint_store.checkpoint("shared-session", {"step": 1}))
        except BaseException as exc:
            errors.append(exc)
        finally:
            asyncio.run(_close(checkpoint_store))

    knowledge_thread = threading.Thread(target=refresh_knowledge)
    checkpoint_thread = threading.Thread(target=write_checkpoint)
    try:
        knowledge_thread.start()
        assert knowledge_paused.wait(5)
        checkpoint_thread.start()
        assert checkpoint_begin.wait(5)
        release_knowledge.set()
        knowledge_thread.join(5)
        checkpoint_thread.join(5)
        assert not knowledge_thread.is_alive()
        assert not checkpoint_thread.is_alive()
        assert errors == []
        assert progress_calls <= 100
        knowledge._connection.set_progress_handler(None, 0)

        verifier = SQLiteSessionStore(db_path)
        try:
            assert asyncio.run(verifier.load_checkpoint("shared-session")) == {"step": 1}
        finally:
            asyncio.run(_close(verifier))
        assert [
            hit.entry.id
            for hit in asyncio.run(
                knowledge.search(KnowledgeQuery(text="revised target summary"))
            ).hits
        ] == ["target"]
    finally:
        release_knowledge.set()
        knowledge_thread.join(5)
        if checkpoint_thread.ident is not None:
            checkpoint_thread.join(5)
        knowledge._connection.set_progress_handler(None, 0)
        asyncio.run(_close(knowledge))


@pytest.mark.process
def test_sigkill_during_revision_37_migration_rolls_back_and_retries(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "migration-kill.sqlite"
    marker = tmp_path / "migration-started"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="legacy", text="legacy searchable"))
            _seed_unrelated_corpus(store._connection, chunk_count=5000)
        finally:
            await _close(store)

    asyncio.run(seed())
    _downgrade_knowledge_layout_to_revision_36(db_path)
    program = """
import pathlib
import sys
import time
from cayu.storage import _sqlite_support as support
from cayu.storage import migrations

db_path = pathlib.Path(sys.argv[1])
marker = pathlib.Path(sys.argv[2])
connection = support.connect(db_path)
published = False

def pause_in_transaction():
    global published
    if connection.in_transaction and not published:
        published = True
        marker.write_text('migration transaction active')
        while True:
            time.sleep(1)
    return 0

connection.set_progress_handler(pause_in_transaction, 100)
migrations.REVISIONS = tuple(
    revision for revision in migrations.REVISIONS if revision.revision <= 37
)
support.reconcile_schema(
    connection,
    migrations.SchemaMode.MIGRATE,
    app_min_supported=37,
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(db_path), str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_file(marker, process)
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 36
        assert "fts_rowid" not in {
            str(row[1]) for row in connection.execute("PRAGMA table_info(cayu_knowledge_chunks)")
        }
        assert (
            connection.execute(
                """
            SELECT chunk_id
            FROM cayu_knowledge_chunks_fts
            WHERE cayu_knowledge_chunks_fts MATCH 'searchable'
            """
            ).fetchone()[0]
            == "legacy:r1:0"
        )
    finally:
        connection.close()

    _migrate_legacy_through_revision_37(db_path)
    assert (
        next(chunk_id for chunk_id, _ in _raw_ranked_matches(db_path, "legacy searchable"))
        == "legacy:r1:0"
    )


@pytest.mark.process
def test_sigkill_during_knowledge_refresh_cannot_publish_half_state(tmp_path: Path) -> None:
    db_path = tmp_path / "refresh-kill.sqlite"
    marker = tmp_path / "refresh-paused"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
        try:
            await store.create_entry(KnowledgeEntry(id="target", text="originaltoken"))
        finally:
            await _close(store)

    asyncio.run(seed())
    program = """
import asyncio
import pathlib
import sys
import time
from cayu.storage import KnowledgeAccessScope, SQLiteKnowledgeStore

db_path = pathlib.Path(sys.argv[1])
marker = pathlib.Path(sys.argv[2])
_ACCESS_SCOPE = KnowledgeAccessScope.privileged()
store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
original = store._insert_entry_fts_unlocked

def pause_before_fts(entry, chunks):
    marker.write_text('source changed and old FTS removed')
    while True:
        time.sleep(1)

store._insert_entry_fts_unlocked = pause_before_fts
current = asyncio.run(store.get_entry('target'))
asyncio.run(
    store.append_entry_revision(
        current.model_copy(update={'revision': 2, 'text': 'replacementtoken'}),
        expected_revision=1,
    )
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(db_path), str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_file(marker, process)
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    store = SQLiteKnowledgeStore(db_path, access_scope=_ACCESS_SCOPE)
    try:
        loaded = asyncio.run(store.get_entry("target"))
        assert loaded is not None and loaded.text == "originaltoken"
        original = asyncio.run(store.search(KnowledgeQuery(text="originaltoken")))
        replacement = asyncio.run(store.search(KnowledgeQuery(text="replacementtoken")))
        assert [hit.entry.id for hit in original.hits] == ["target"]
        assert replacement.hits == []
        _assert_exact_fts_mapping(
            store._connection,
            entry_id="target",
            chunk_ids=["target:r1:0"],
        )
    finally:
        asyncio.run(_close(store))
