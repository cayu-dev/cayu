"""Postgres schema-migrator behavior (ADR 0001, Phases 1-2).

Proves the per-backend realization of the shared migration model against a real
Postgres: validate-at-startup fail-fast, create-records-baseline, migrate, and
advisory-lock-coordinated reconciliation across stores that share a database.
These skip automatically when Postgres is unavailable (see ``conftest.py``).
"""

from __future__ import annotations

import asyncio

import pytest

from cayu import PostgresSessionStore, PostgresTaskStore
from cayu.core import Message
from cayu.runtime import RunRequest, SessionIdentity
from cayu.storage import migrations as schema
from cayu.storage.migrations import SchemaMode

pytestmark = pytest.mark.usefixtures("postgres_dsn")


def _request(agent_name: str) -> RunRequest:
    return RunRequest(agent_name=agent_name, messages=[Message.text("user", "hi")])


_TABLES = (
    "cayu_budget_reservations",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_dead_letters",
    "cayu_event_watcher_state",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _expected_revisions() -> list[tuple[int, str, int]]:
    return [(rev.revision, str(rev.kind), rev.compatible_from) for rev in schema.REVISIONS]


async def _drop_all(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


async def _recorded_revisions(dsn: str) -> list[tuple[int, str, int]]:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT revision, kind, compatible_from FROM cayu_schema_migrations "
            "ORDER BY revision ASC"
        )
        return [tuple(row) for row in await cur.fetchall()]


def test_validate_mode_fails_fast_on_uninitialized(postgres_dsn: str) -> None:
    async def runner() -> None:
        await _drop_all(postgres_dsn)
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaUninitialized):
                await store.create(_request("a"), identity=_identity())
        finally:
            await store.close()

    asyncio.run(runner())


def test_create_mode_initializes_and_records_baseline(postgres_dsn: str) -> None:
    async def runner() -> None:
        await _drop_all(postgres_dsn)
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            session = await store.create(_request("a"), identity=_identity())
            assert session.id
        finally:
            await store.close()
        # A new database is initialized through every known revision.
        assert await _recorded_revisions(postgres_dsn) == _expected_revisions()

    asyncio.run(runner())


def test_validate_mode_succeeds_after_create(postgres_dsn: str) -> None:
    async def runner() -> None:
        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.create(_request("a"), identity=_identity())
        finally:
            await creator.close()
        # A second process that only validates now starts cleanly.
        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            session = await validator.create(_request("b"), identity=_identity())
            assert session.id
        finally:
            await validator.close()

    asyncio.run(runner())


def test_validate_mode_rejects_pre_insert_xid_postgres_schema(postgres_dsn: str) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 14")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 16"):
                await validator.ensure_schema()
        finally:
            await validator.close()

    asyncio.run(runner())


def test_revision_fourteen_requires_cascade_index_migration(postgres_dsn: str) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 15")
                await cur.execute("DROP INDEX idx_cayu_checkpoints_pending_interruption_cascade")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 16"):
                await validator.ensure_schema()
        finally:
            await validator.close()

        migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT 1 FROM pg_indexes WHERE schemaname = current_schema() "
                "AND indexname = "
                "'idx_cayu_checkpoints_pending_interruption_cascade'"
            )
            assert await cur.fetchone() is not None

    asyncio.run(runner())


def test_revision_fifteen_requires_session_sequence_index_migration(postgres_dsn: str) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision = 16")
                await cur.execute("DROP INDEX idx_cayu_events_session_sequence")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 16"):
                await validator.ensure_schema()
        finally:
            await validator.close()

        first_migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        second_migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await asyncio.gather(
                first_migrator.ensure_schema(),
                second_migrator.ensure_schema(),
            )
        finally:
            await first_migrator.close()
            await second_migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT index_definition.indisvalid
                FROM pg_catalog.pg_class AS index_class
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = index_class.relnamespace
                JOIN pg_catalog.pg_index AS index_definition
                  ON index_definition.indexrelid = index_class.oid
                WHERE namespace.nspname = current_schema()
                  AND index_class.relname = 'idx_cayu_events_session_sequence'
                """
            )
            assert await cur.fetchone() == (True,)

            await cur.execute("SELECT COUNT(*) FROM cayu_schema_migrations WHERE revision = 16")
            assert await cur.fetchone() == (1,)

    asyncio.run(runner())


@pytest.mark.parametrize(
    ("conflict_ddl", "cleanup_ddl"),
    [
        (
            "CREATE INDEX idx_cayu_events_session_sequence ON cayu_events(session_id)",
            "DROP INDEX idx_cayu_events_session_sequence",
        ),
        (
            "CREATE TABLE idx_cayu_events_session_sequence (id INTEGER)",
            "DROP TABLE idx_cayu_events_session_sequence",
        ),
    ],
    ids=["wrong-index-definition", "non-index-relation"],
)
def test_revision_sixteen_rejects_conflicting_schema_objects(
    postgres_dsn: str,
    conflict_ddl: str,
    cleanup_ddl: str,
) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision = 16")
                await cur.execute("DROP INDEX idx_cayu_events_session_sequence")
                await cur.execute(conflict_ddl)
            await conn.commit()

        try:
            migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
            try:
                with pytest.raises(RuntimeError, match="conflicts with the required B-tree index"):
                    await migrator.ensure_schema()
            finally:
                await migrator.close()

            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute("SELECT COUNT(*) FROM cayu_schema_migrations WHERE revision = 16")
                assert await cur.fetchone() == (0,)
        finally:
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(cleanup_ddl)
                await conn.commit()

    asyncio.run(runner())


def test_migrate_mode_initializes_baseline_idempotently(postgres_dsn: str) -> None:
    async def runner() -> None:
        await _drop_all(postgres_dsn)
        first = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await first.create(_request("a"), identity=_identity())
        finally:
            await first.close()
        # Re-running migrate is a no-op: still exactly the known revisions.
        second = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await second.create(_request("b"), identity=_identity())
        finally:
            await second.close()
        assert await _recorded_revisions(postgres_dsn) == _expected_revisions()

    asyncio.run(runner())


def test_session_and_task_stores_share_one_baseline(postgres_dsn: str) -> None:
    async def runner() -> None:
        await _drop_all(postgres_dsn)
        # The production pattern: two stores, each reconciling the shared schema.
        sessions = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        tasks = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await sessions.create(_request("a"), identity=_identity())
            listed = await tasks.list_tasks()
            assert listed == []
        finally:
            await sessions.close()
            await tasks.close()
        # The advisory lock serialized init: revisions are recorded once, not twice.
        assert await _recorded_revisions(postgres_dsn) == _expected_revisions()

    asyncio.run(runner())
