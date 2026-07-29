"""Postgres schema-migrator behavior (ADR 0001, Phases 1-2).

Proves the per-backend realization of the shared migration model against a real
Postgres: validate-at-startup fail-fast, create-records-baseline, migrate, and
advisory-lock-coordinated reconciliation across stores that share a database.
These skip automatically when Postgres is unavailable (see ``conftest.py``).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cayu import PostgresSessionStore, PostgresTaskStore
from cayu.core import Event, EventType, Message
from cayu.runtime import (
    PendingActionQuery,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime.pending_actions import (
    pending_action_event_storage_values,
    pending_action_lookup_key,
)
from cayu.runtime.sessions import (
    MAX_PENDING_ACTION_RESULT_BYTES,
    BudgetReservationIdentityConflict,
)
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 24"):
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 24"):
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 24"):
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 24"):
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
                    payload={
                        "tool_call_id": "x" * 10_000,
                        "approval_id": "revision_17_pause",
                    },
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
            identity_payload = {
                "model_step_id": f"mstep_{'1' * 32}",
                "model_attempt_id": f"matt_{'2' * 32}",
                "tool_round_id": f"tround_{'3' * 32}",
            }
            projected_action_events = [
                Event(
                    id="revision_17_projected_approval",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=long_id_session.id,
                    tool_name="deploy",
                    payload={
                        **identity_payload,
                        "approval_id": "revision_17_projected_approval_id",
                        "tool_call_id": "revision_17_projected_approval_call",
                        "approval": {
                            "approval_id": "revision_17_projected_approval_id",
                            **identity_payload,
                            "tool_call_id": "revision_17_projected_approval_call",
                            "reason": "review",
                            "tool_name": "deploy",
                        },
                    },
                ),
                Event(
                    id="revision_17_projected_input",
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id=long_id_session.id,
                    tool_name="ask_user",
                    payload={
                        **identity_payload,
                        "input_id": "revision_17_projected_input_id",
                        "tool_call_id": "revision_17_projected_input_call",
                        "question": "Deploy?",
                        "options": ["yes", "no"],
                    },
                ),
                Event(
                    id="revision_17_projected_interruption",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=long_id_session.id,
                    tool_name="deploy",
                    payload={
                        **identity_payload,
                        "interruption_type": "runtime_interrupted",
                        "manual_recovery_required": True,
                        "approval_id": "revision_17_projected_approval_id",
                        "tool_call_id": "revision_17_projected_approval_call",
                        "message": "reconcile",
                        "tool_name": "deploy",
                        "tool_evidence_conflict": True,
                        "approval": {
                            "approval_id": "revision_17_projected_approval_id",
                            **identity_payload,
                            "tool_call_id": "revision_17_projected_approval_call",
                            "reason": "review",
                            "tool_name": "deploy",
                        },
                        "user_input": {
                            "input_id": "revision_17_projected_input_id",
                            "tool_call_id": "revision_17_projected_input_call",
                            "question": "Deploy?",
                            "options": ["yes", "no"],
                        },
                    },
                ),
                Event(
                    id="revision_17_projected_failure",
                    type=EventType.SESSION_FAILED,
                    session_id=long_id_session.id,
                    payload={"tool_evidence_conflict": True},
                ),
                Event(
                    id="revision_17_projected_resume",
                    type=EventType.SESSION_RESUMED,
                    session_id=long_id_session.id,
                    payload=identity_payload,
                ),
                Event(
                    id="revision_17_projected_oversized_input",
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id=long_id_session.id,
                    payload={
                        **identity_payload,
                        "input_id": "revision_17_projected_oversized_input_id",
                        "tool_call_id": "revision_17_projected_oversized_input_call",
                        "question": "x" * MAX_PENDING_ACTION_RESULT_BYTES,
                        "options": [],
                    },
                ),
            ]
            await creator.append_events(long_id_session.id, projected_action_events)
            pending_approval_session = await creator.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="revision_17_pending_approval",
                    messages=[Message.text("user", "hello")],
                ),
                identity=_identity(),
            )
            await creator.append_event(
                pending_approval_session.id,
                Event(
                    id="revision_17_pending_approval_event",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=pending_approval_session.id,
                    agent_name="assistant",
                    tool_name="deploy",
                    payload={
                        **identity_payload,
                        "approval_id": "revision_17_pending_approval_id",
                        "tool_call_id": "revision_17_pending_approval_call",
                        "approval": {
                            "approval_id": "revision_17_pending_approval_id",
                            **identity_payload,
                            "tool_call_id": "revision_17_pending_approval_call",
                            "tool_name": "deploy",
                            "arguments": {},
                            "agent_name": "assistant",
                            "tool_calls": [
                                {
                                    "tool_call_id": "revision_17_pending_approval_call",
                                    "tool_name": "deploy",
                                    "arguments": {},
                                    "policy_decision": None,
                                    "reason": None,
                                    "metadata": {},
                                    "active_taint_labels": [],
                                }
                            ],
                        },
                    },
                ),
            )
            await creator.checkpoint(
                pending_approval_session.id,
                {
                    "pending_tool_approval": {
                        **identity_payload,
                        "approval_id": "revision_17_pending_approval_id",
                        "tool_call_id": "revision_17_pending_approval_call",
                        "tool_name": "deploy",
                        "arguments": {},
                        "agent_name": "assistant",
                        "tool_calls": [
                            {
                                "tool_call_id": "revision_17_pending_approval_call",
                                "tool_name": "deploy",
                                "arguments": {},
                                "policy_decision": None,
                                "reason": None,
                                "metadata": {},
                                "active_taint_labels": [],
                            }
                        ],
                    }
                },
            )
            await creator.update_status(
                pending_approval_session.id,
                SessionStatus.INTERRUPTED,
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
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_round_scope")
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_attempt_scope")
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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 24"):
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
                "SELECT pending_action_source_bytes, pending_action_tool_call_count, "
                "pending_action_flags, pending_action_metrics_ready FROM cayu_checkpoints "
                "WHERE session_id = 'revision_17_pending_approval'"
            )
            approval_metric_row = await cur.fetchone()
            assert approval_metric_row is not None
            assert approval_metric_row[0] > 0
            assert approval_metric_row[1:] == (1, 1, True)
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
            await cur.execute(
                "SELECT event_id, pending_action_projection FROM cayu_events "
                "WHERE event_id = ANY(%s)",
                ([event.id for event in projected_action_events],),
            )
            migrated_projections = {
                str(event_id): projection for event_id, projection in await cur.fetchall()
            }
            expected_projections = {}
            for event in projected_action_events:
                _lookup_key, projection_json, _projection_bytes = (
                    pending_action_event_storage_values(event)
                )
                assert projection_json is not None
                expected_projections[event.id] = json.loads(projection_json)
            assert migrated_projections == expected_projections

        reader = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            pending_actions = await reader.query_pending_actions(
                PendingActionQuery(session_id="revision_17_pending_approval")
            )
        finally:
            await reader.close()
        assert len(pending_actions.actions) == 1
        assert pending_actions.actions[0].approval_id == "revision_17_pending_approval_id"
        assert pending_actions.issues == []

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
            with pytest.raises(schema.SchemaTooOld, match="requires >= 24"):
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


def test_revision_twenty_three_preserves_existing_reservation_ownership(
    postgres_dsn: str,
) -> None:
    async def runner() -> None:
        import psycopg

        await _drop_all(postgres_dsn)
        reservation_id = "bres_revision_23_existing"
        publication_id = "evt_revision_23_existing"
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            session = await creator.create(
                RunRequest(
                    session_id="sess_revision_23_existing",
                    agent_name="assistant",
                    messages=[Message.text("user", "seed")],
                ),
                identity=_identity(),
            )
            reserved = Event(
                id=publication_id,
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
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 23")
                await cur.execute("DROP INDEX idx_cayu_events_budget_reservation_identity")
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_round_scope")
                await cur.execute("DROP INDEX idx_cayu_events_pending_action_attempt_scope")
                await cur.execute("DROP TABLE cayu_budget_reservation_identities")
            await conn.commit()

        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await store.ensure_schema()
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute(
                    "SELECT publication_session_id, publication_id, published "
                    "FROM cayu_budget_reservation_identities "
                    "WHERE reservation_id = %s",
                    (reservation_id,),
                )
                assert await cur.fetchone() == (
                    "sess_revision_23_existing",
                    publication_id,
                    True,
                )

            await store.delete_session("sess_revision_23_existing")
            replacement = await store.create(
                RunRequest(
                    session_id="sess_revision_23_replacement",
                    agent_name="assistant",
                    messages=[Message.text("user", "reuse")],
                ),
                identity=_identity(),
            )
            with pytest.raises(BudgetReservationIdentityConflict):
                await store.append_event(
                    replacement.id,
                    Event(
                        id="evt_revision_23_reuse",
                        type=EventType.BUDGET_RESERVED,
                        session_id=replacement.id,
                        payload={"reservation_id": reservation_id},
                    ),
                )
        finally:
            await store.close()

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
