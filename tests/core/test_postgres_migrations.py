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
    "cayu_budget_settlements",
    "cayu_budget_reservations",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_dead_letters",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
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


def test_revision_twenty_six_rejects_populated_session_database(
    postgres_dsn: str,
) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            session = await creator.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_transcript_order_migration",
                    messages=[],
                ),
                identity=_identity(),
            )
            await creator.append_transcript_messages(
                session.id,
                [Message.text("user", "first")],
                interaction_id="interaction-one",
            )
            await creator.append_transcript_messages(
                session.id,
                [Message.text("user", "other")],
                interaction_id="interaction-two",
            )
            await creator.append_transcript_messages(
                session.id,
                [Message.text("assistant", "second")],
                interaction_id="interaction-one",
            )
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DROP TRIGGER cayu_assign_transcript_order ON cayu_transcript_messages"
                )
                await cur.execute("DROP FUNCTION cayu_assign_transcript_order()")
                await cur.execute("ALTER TABLE cayu_transcript_messages DROP COLUMN session_order")
                await cur.execute("ALTER TABLE cayu_sessions DROP COLUMN transcript_seq")
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 26")
            await conn.commit()

        migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            with pytest.raises(
                schema.SchemaTooOld,
                match="clean prerelease break",
            ):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT COUNT(*) FROM cayu_transcript_messages WHERE session_id = %s",
                ("sess_transcript_order_migration",),
            )
            assert (await cur.fetchone())[0] == 3
            await cur.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert (await cur.fetchone())[0] == 25
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'cayu_transcript_messages' "
                "AND column_name = 'session_order'"
            )
            assert await cur.fetchone() is None

    asyncio.run(runner())


def test_latest_migrates_queue_and_event_side_effect_handoff(
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
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 19")
                await cur.execute("DROP TABLE cayu_mcp_manifest_baselines")
                await cur.execute("DROP TABLE cayu_persisted_event_side_effects")
                await cur.execute("DROP TABLE cayu_session_message_queue")
                await cur.execute("DROP INDEX idx_cayu_sessions_parent_created_id")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="requires >= 26"):
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
            await cur.execute("SELECT to_regclass('cayu_persisted_event_side_effects')")
            assert (await cur.fetchone())[0] == "cayu_persisted_event_side_effects"
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 20"
            )
            assert await cur.fetchone() == ("additive", 19)
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 21"
            )
            assert await cur.fetchone() == ("breaking", 21)
            await cur.execute("SELECT to_regclass('cayu_mcp_manifest_baselines')")
            assert (await cur.fetchone())[0] == "cayu_mcp_manifest_baselines"
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 22"
            )
            assert await cur.fetchone() == ("breaking", 22)
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 23"
            )
            assert await cur.fetchone() == ("breaking", 23)
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 24"
            )
            assert await cur.fetchone() == ("additive", 23)
            await cur.execute("SELECT to_regclass('idx_cayu_sessions_parent_created_id')")
            assert (await cur.fetchone())[0] == "idx_cayu_sessions_parent_created_id"
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 25"
            )
            assert await cur.fetchone() == ("breaking", 25)
            await cur.execute("SELECT to_regclass('cayu_budget_settlements')")
            assert (await cur.fetchone())[0] == "cayu_budget_settlements"
            await cur.execute(
                "SELECT to_regclass('idx_cayu_budget_reservation_identities_session')"
            )
            assert (await cur.fetchone())[0] == ("idx_cayu_budget_reservation_identities_session")
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'cayu_budget_reservations' "
                "AND column_name = 'budget_limit_id'"
            )
            assert await cur.fetchone() == ("budget_limit_id",)
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'cayu_budget_reservations' "
                "AND column_name IN ('model_step_id', 'model_attempt_id') "
                "ORDER BY column_name"
            )
            assert await cur.fetchall() == [
                ("model_attempt_id",),
                ("model_step_id",),
            ]
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'cayu_budget_reservations' "
                "AND column_name IN ("
                "'environment_name', 'settlement_event_payload', "
                "'settlement_fallback', 'dispatch_id', 'dispatched_at'"
                ") ORDER BY column_name"
            )
            assert await cur.fetchall() == [
                ("dispatch_id",),
                ("dispatched_at",),
                ("environment_name",),
                ("settlement_event_payload",),
                ("settlement_fallback",),
            ]
            await cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_trigger "
                "WHERE tgname = 'cayu_events_enqueue_persisted_side_effect' "
                "AND NOT tgisinternal)"
            )
            assert await cur.fetchone() == (False,)

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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 26"):
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 26"):
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 26"):
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 26"):
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


def test_recorded_revision_twenty_three_requires_the_unique_reservation_index(
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
                await cur.execute("DROP INDEX idx_cayu_events_budget_reservation_identity")
                await cur.execute(
                    "CREATE INDEX idx_cayu_events_budget_reservation_identity "
                    "ON cayu_events ((payload ->> 'reservation_id')) "
                    "WHERE event_type = 'budget.reserved'"
                )
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(RuntimeError, match="required unique B-tree index"):
                await validator.ensure_schema()
        finally:
            await validator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP INDEX idx_cayu_events_budget_reservation_identity")
            await conn.commit()

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
                "SELECT index_definition.indisunique "
                "FROM pg_catalog.pg_class AS index_class "
                "JOIN pg_catalog.pg_namespace AS namespace "
                "ON namespace.oid = index_class.relnamespace "
                "JOIN pg_catalog.pg_index AS index_definition "
                "ON index_definition.indexrelid = index_class.oid "
                "WHERE namespace.nspname = current_schema() "
                "AND index_class.relname = "
                "'idx_cayu_events_budget_reservation_identity'"
            )
            assert await cur.fetchone() == (True,)

    asyncio.run(runner())


def test_recorded_revision_twenty_three_fails_closed_without_reservation_registry(
    postgres_dsn: str,
) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            session = await creator.create(
                RunRequest(
                    session_id="sess_registry_repair",
                    agent_name="assistant",
                    messages=[Message.text("user", "seed")],
                ),
                identity=_identity(),
            )
            reservation_id = "bres_registry_repair"
            reserved = Event(
                type=EventType.BUDGET_RESERVED,
                session_id=session.id,
                payload={"reservation_id": reservation_id},
            )
            await creator.append_event(session.id, reserved)
            reserved_claim = await creator.claim_persisted_event_side_effect(
                session_id=session.id,
                event_id=reserved.id,
            )
            assert reserved_claim is not None
            await creator.mark_persisted_event_side_effect_delivered(reserved_claim)
            released = Event(
                type=EventType.BUDGET_RESERVATION_RELEASED,
                session_id=session.id,
                payload={"reservation_id": reservation_id},
            )
            await creator.append_event(session.id, released)
            released_claim = await creator.claim_persisted_event_side_effect(
                session_id=session.id,
                event_id=released.id,
            )
            assert released_claim is not None
            await creator.mark_persisted_event_side_effect_delivered(released_claim)
            await creator.delete_session(session.id)
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE cayu_budget_reservation_identities")
                await cur.execute(
                    "CREATE TABLE cayu_budget_reservation_identities ("
                    "reservation_id TEXT PRIMARY KEY, publication_id TEXT NOT NULL)"
                )
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(RuntimeError, match="reservation identity contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE cayu_budget_reservation_identities")
            await conn.commit()

        migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            with pytest.raises(RuntimeError, match="permanent reservation ownership registry"):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT to_regclass('cayu_budget_reservation_identities')")
            assert (await cur.fetchone())[0] is None

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
