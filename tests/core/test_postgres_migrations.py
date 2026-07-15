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
from cayu.core import Event, EventType, Message
from cayu.runtime import RunRequest, SessionIdentity
from cayu.runtime.pending_actions import pending_action_lookup_key
from cayu.storage import migrations as schema
from cayu.storage import postgres as postgres_storage
from cayu.storage.migrations import SchemaMode

pytestmark = pytest.mark.usefixtures("postgres_dsn")


def test_revision_seventeen_builds_hot_indexes_concurrently() -> None:
    assert all(
        "CREATE INDEX" not in statement
        for statement in postgres_storage._MIGRATION_STEPS.get(17, ())
    )
    indexes = {
        index.index_name: index for index in postgres_storage._CONCURRENT_INDEX_MIGRATIONS[17]
    }
    assert {
        "idx_cayu_checkpoints_pending_control_action",
        "idx_cayu_events_pending_action_barrier",
        "idx_cayu_events_pending_action_lookup",
    } == indexes.keys()
    assert all("CREATE INDEX CONCURRENTLY" in index.create_statement for index in indexes.values())
    barrier_index = indexes["idx_cayu_events_pending_action_barrier"]
    assert barrier_index.key_definitions == ("session_id", "sequence")
    assert all(
        event_type in (barrier_index.predicate_definition or "")
        for event_type in ("session.resumed", "session.completed", "session.failed")
    )
    assert "tool.call" not in (barrier_index.predicate_definition or "")
    assert "session_id > %s" in postgres_storage._REVISION_17_CHECKPOINT_BACKFILL_SQL
    assert "sequence > %s" in postgres_storage._REVISION_17_EVENT_BACKFILL_SMALL_SQL
    assert "LIMIT 25" in postgres_storage._REVISION_17_EVENT_BACKFILL_SMALL_SQL
    assert "sequence > %s" in postgres_storage._REVISION_17_EVENT_BACKFILL_LARGE_SQL
    assert "LIMIT 1" in postgres_storage._REVISION_17_EVENT_BACKFILL_LARGE_SQL


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
    "cayu_session_message_queue",
    "cayu_checkpoints",
    "cayu_session_operations",
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


def test_revision_nineteen_migrates_durable_session_message_queue(
    postgres_dsn: str,
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
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision = 19")
                await cur.execute("DROP TABLE cayu_session_message_queue")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 19"):
                await validator.ensure_schema()
        finally:
            await validator.close()

        task_validator = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            await task_validator.ensure_schema()
        finally:
            await task_validator.close()

        migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT to_regclass('cayu_session_message_queue')")
            assert (await cur.fetchone())[0] == "cayu_session_message_queue"
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 19"
            )
            assert await cur.fetchone() == ("breaking", 19)

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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 19"):
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 19"):
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
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 16")
                await cur.execute("DROP INDEX idx_cayu_events_session_sequence")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 19"):
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


def test_revision_seventeen_requires_pending_action_index_migration(
    postgres_dsn: str,
) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
            long_id_session = await creator.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="revision_17_long_identifier",
                    messages=[Message.text("user", "hello")],
                ),
                identity=_identity(),
            )
            await creator.append_event(
                long_id_session.id,
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=long_id_session.id,
                    payload={"tool_call_id": "x" * 10_000},
                ),
            )
            await creator.append_event(
                long_id_session.id,
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=long_id_session.id,
                    payload={
                        "tool_round_id": "revision_17_terminal_round",
                        "tool_call_id": "revision_17_valid_terminal",
                        "result": {"content": "done"},
                    },
                ),
            )
            await creator.append_event(
                long_id_session.id,
                Event(
                    type=EventType.TOOL_CALL_FAILED,
                    session_id=long_id_session.id,
                    payload={
                        "tool_round_id": "revision_17_terminal_round",
                        "tool_call_id": "revision_17_invalid_terminal",
                    },
                ),
            )
            await creator.append_event(
                long_id_session.id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=long_id_session.id,
                    payload={
                        "approval_id": "revision_17_large_event",
                        "error": "x"
                        * (postgres_storage._REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES + 1),
                    },
                ),
            )
            await creator.append_event(
                long_id_session.id,
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=long_id_session.id,
                    payload={
                        "approval_id": "\t",
                        "approval": {
                            "approval_id": "revision_17_nested_approval",
                            "tool_name": "deploy",
                        },
                    },
                ),
            )
            await creator.checkpoint(
                long_id_session.id,
                {
                    "pending_tool_round": {
                        "round_id": "revision_17_round",
                        "agent_name": "assistant",
                        "tool_calls": [{"tool_call_id": "revision_17_call"}],
                    }
                },
            )
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 17")
                await cur.execute("DROP INDEX idx_cayu_checkpoints_pending_control_action")
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_barrier")
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_lookup")
                await cur.execute(
                    "ALTER TABLE cayu_events DROP COLUMN pending_action_lookup_key, "
                    "DROP COLUMN pending_action_projection, "
                    "DROP COLUMN pending_action_projection_bytes"
                )
                await cur.execute(
                    "ALTER TABLE cayu_checkpoints DROP COLUMN pending_action_source_bytes, "
                    "DROP COLUMN pending_action_tool_call_count, "
                    "DROP COLUMN pending_action_flags, "
                    "DROP COLUMN pending_action_metrics_ready"
                )
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 19"):
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
                "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
                "AND indexname = 'idx_cayu_checkpoints_pending_control_action'"
            )
            row = await cur.fetchone()
            assert row is not None
            assert "pending_action_flags" in row[0]
            await cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'cayu_checkpoints'
                  AND column_name IN (
                      'pending_action_source_bytes',
                      'pending_action_tool_call_count',
                      'pending_action_flags',
                      'pending_action_metrics_ready'
                  )
                """
            )
            assert {row[0] for row in await cur.fetchall()} == {
                "pending_action_source_bytes",
                "pending_action_tool_call_count",
                "pending_action_flags",
                "pending_action_metrics_ready",
            }
            await cur.execute(
                "SELECT pending_action_source_bytes, pending_action_tool_call_count, "
                "pending_action_flags, pending_action_metrics_ready FROM cayu_checkpoints "
                "WHERE session_id = 'revision_17_long_identifier'"
            )
            metric_row = await cur.fetchone()
            assert metric_row is not None
            assert metric_row[0] > 0
            assert metric_row[1:] == (1, 4, True)
            await cur.execute(
                """
                SELECT index_definition.indisvalid
                FROM pg_catalog.pg_class AS index_class
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = index_class.relnamespace
                JOIN pg_catalog.pg_index AS index_definition
                  ON index_definition.indexrelid = index_class.oid
                WHERE namespace.nspname = current_schema()
                  AND index_class.relname = 'idx_cayu_events_pending_action_barrier'
                """
            )
            assert await cur.fetchone() == (True,)
            await cur.execute(
                """
                SELECT index_definition.indisvalid, pg_get_indexdef(index_class.oid)
                FROM pg_catalog.pg_class AS index_class
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = index_class.relnamespace
                JOIN pg_catalog.pg_index AS index_definition
                  ON index_definition.indexrelid = index_class.oid
                WHERE namespace.nspname = current_schema()
                  AND index_class.relname = 'idx_cayu_events_pending_action_lookup'
                """
            )
            lookup_row = await cur.fetchone()
            assert lookup_row is not None
            assert lookup_row[0] is True
            assert "pending_action_lookup_key" in lookup_row[1]
            assert "event_type" in lookup_row[1]
            assert "IS NOT NULL" in lookup_row[1]
            await cur.execute(
                "SELECT pending_action_lookup_key, pending_action_projection, "
                "pending_action_projection_bytes FROM cayu_events "
                "WHERE session_id = 'revision_17_long_identifier' "
                "AND event_type = 'tool.call.started'"
            )
            event_metric_row = await cur.fetchone()
            assert event_metric_row is not None
            assert event_metric_row[0] == pending_action_lookup_key("x" * 10_000)
            assert event_metric_row[1]["payload"] == {"tool_call_id": "x" * 10_000}
            assert event_metric_row[2] > 10_000
            await cur.execute(
                "SELECT pending_action_lookup_key FROM cayu_events "
                "WHERE session_id = 'revision_17_long_identifier' "
                "AND event_type = 'tool.call.approval_requested'"
            )
            assert await cur.fetchone() == (
                pending_action_lookup_key("revision_17_nested_approval"),
            )
            await cur.execute(
                "SELECT pending_action_lookup_key, pending_action_projection_bytes "
                "FROM cayu_events WHERE session_id = 'revision_17_long_identifier' "
                "AND event_type = 'session.interrupted'"
            )
            large_event_row = await cur.fetchone()
            assert large_event_row is not None
            assert large_event_row[0] == pending_action_lookup_key("revision_17_large_event")
            assert (
                large_event_row[1] > postgres_storage._REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES
            )
            await cur.execute(
                "SELECT event_type, pending_action_projection -> 'payload' "
                "->> '__cayu_terminal_result_valid__' "
                "FROM cayu_events WHERE session_id = 'revision_17_long_identifier' "
                "AND event_type IN ('tool.call.completed', 'tool.call.failed') "
                "ORDER BY event_type"
            )
            assert await cur.fetchall() == [
                ("tool.call.completed", "true"),
                ("tool.call.failed", "false"),
            ]

    asyncio.run(runner())


def test_revision_seventeen_requires_session_operation_migration(postgres_dsn: str) -> None:
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
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 18")
                await cur.execute("DROP TABLE cayu_session_operations")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 19"):
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
            await cur.execute("SELECT to_regclass('cayu_session_operations')")
            assert await cur.fetchone() == ("cayu_session_operations",)

    asyncio.run(runner())


def test_revision_seventeen_rejects_incomplete_lookup_index(postgres_dsn: str) -> None:
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
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 17")
                await cur.execute("DROP INDEX idx_cayu_checkpoints_pending_control_action")
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_barrier")
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_lookup")
                await cur.execute(
                    """
                    CREATE INDEX idx_cayu_events_pending_action_lookup
                    ON cayu_events(
                        session_id,
                        md5(COALESCE(
                            payload ->> 'approval_id',
                            payload #>> '{approval,approval_id}',
                            payload ->> 'input_id',
                            payload #>> '{user_input,input_id}',
                            payload ->> 'tool_call_id',
                            payload ->> 'tool_round_id'
                        )),
                        sequence
                    )
                    WHERE event_type IN (
                        'tool.call.approval_requested',
                        'tool.call.approval_denied'
                    )
                    """
                )
            await conn.commit()

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
            await cur.execute("SELECT COUNT(*) FROM cayu_schema_migrations WHERE revision = 17")
            assert await cur.fetchone() == (0,)

    asyncio.run(runner())


def test_recorded_revision_seventeen_validates_and_repairs_missing_index(
    postgres_dsn: str,
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
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_lookup")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(
                RuntimeError,
                match="Required Cayu Postgres index is missing",
            ):
                await validator.ensure_schema()
        finally:
            await validator.close()

        migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        validated = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            await validated.ensure_schema()
        finally:
            await validated.close()

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
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 16")
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
