"""Regression for the published revision-96 schema (df5a41a9cb53).

That revision contained only task-group quiescence DDL, not participant bindings.
Build that schema forward from an empty database; never rewind a current schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing

import pytest
from tests.core.test_cli_storage import (
    _alias_codec,
    _breaking_acknowledgements_after,
    _configure_alias_environment,
)

from cayu import AgentSpec, CayuApp, SQLiteSessionStore, SQLiteTaskStore
from cayu.cli import main
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import ResumeRequest, RunRequest
from cayu.storage import _sqlite_support as sql
from cayu.storage import migrations as schema
from cayu.storage._participant_bindings_schema import validate_sqlite_participant_bindings
from cayu.tasks.base import TaskCreate


def _app(store):
    app = CayuApp(session_store=store)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


def _rows(connection):
    return {
        name: connection.execute(f'SELECT * FROM "{name}"').fetchall()
        for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not name.startswith("sqlite_") and name != schema.MIGRATIONS_TABLE and "_fts" not in name
    }


@pytest.mark.parametrize("revision", [96, 99, 101])
def test_historical_sqlite_upgrade_continues_session(
    revision, tmp_path, monkeypatch, capsys, sqlite_resources
):
    codec = _alias_codec(7)
    _configure_alias_environment(monkeypatch, 7)
    seed = tmp_path / "seed.sqlite"
    alias = codec.encode("existing", field_name="session_id")

    async def seed_data():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(seed, public_authority_alias_codec=codec))
            tasks = resources.own(SQLiteTaskStore(seed))
            events = [
                event
                async for event in _app(store).run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="existing",
                        messages=[Message.text("user", "first")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
            await tasks.create_task(TaskCreate(task_id="existing-task", type="ordinary"))
            await store.register_public_authority_alias(
                alias, field_name="session_id", private_value="existing"
            )

        return store

    seed_store = asyncio.run(seed_data())
    with closing(sqlite3.connect(seed)) as connection:
        seeded = _rows(connection)

    db = tmp_path / "historical.sqlite"
    with monkeypatch.context() as historical:
        historical.setattr(
            schema, "REVISIONS", tuple(r for r in schema.REVISIONS if r.revision <= revision)
        )
        # Exact historical revision-96 step, verified against df5a41a9cb53.
        historical.setitem(sql._MIGRATION_STEPS, 96, sql.SQLITE_TASK_GROUP_QUIESCENCE_DDL)
        with closing(sqlite3.connect(db)) as connection:
            sql._register_sqlite_functions(connection)
            seed_store._register_public_authority_alias_sql_function(connection)
            connection.row_factory = sqlite3.Row
            connection.execute(sql._MIGRATIONS_TABLE_DDL)
            sql._apply_pending(connection, sql.read_schema_state(connection))
            assert (
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name='cayu_participant_session_bindings'"
                ).fetchone()
                is None
            )
            # Import persisted rows without replaying writer-side projection triggers.
            triggers = connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
            for name, _ in triggers:
                connection.execute(f'DROP TRIGGER "{name}"')
            existing_tables = _rows(connection)
            authority_tables = {
                "cayu_public_authority_alias_keys",
                "cayu_public_authority_alias_config",
            }
            for table, rows in sorted(
                seeded.items(), key=lambda item: item[0] not in authority_tables
            ):
                if table not in existing_tables or not rows:
                    continue
                connection.execute(f'DELETE FROM "{table}"')
                connection.executemany(
                    f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in rows[0])})', rows
                )
            connection.execute("CREATE TABLE product_records (id TEXT PRIMARY KEY, payload TEXT)")
            connection.execute(
                "INSERT INTO product_records VALUES ('retained', 'application data')"
            )
            for _, ddl in triggers:
                connection.execute(ddl)
            connection.commit()
    with closing(sqlite3.connect(db)) as connection:
        before = _rows(connection)
    backup = tmp_path / "retained.backup.sqlite"
    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                "--backup",
                str(backup),
                *_breaking_acknowledgements_after(revision),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)["migration_receipt"]
    assert receipt["input_revision"] == revision
    assert receipt["output_revision"] == schema.LATEST_REVISION
    assert receipt["migration_steps"] == list(range(revision + 1, schema.LATEST_REVISION + 1))
    assert receipt["backup"]["sha256"] == hashlib.sha256(backup.read_bytes()).hexdigest()
    assert receipt["checks"] == {"foreign_key_violations": 0, "integrity_check": "ok"}
    with closing(sqlite3.connect(backup)) as connection:
        assert sql.read_schema_state(connection).revision == revision
        assert _rows(connection) == before
    with closing(sqlite3.connect(db)) as connection:
        after = _rows(connection)
        assert all(after[table] == rows for table, rows in before.items())
        validate_sqlite_participant_bindings(connection)
    # Repeated public migration is a no-op, with its own retained backup/receipt.
    assert main(["storage", "migrate", "--sqlite", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["migration_receipt"]["migration_steps"] == []
    with closing(sqlite3.connect(db)) as connection:
        assert _rows(connection) == after

    async def continuation():
        store = SQLiteSessionStore(
            db, schema_mode=schema.SchemaMode.VALIDATE, public_authority_alias_codec=codec
        )
        try:
            assert await store.load_participant_session_binding("existing") is None
            assert (
                await store.resolve_public_authority_alias(alias, field_name="session_id")
                == "existing"
            )
            events = [
                event
                async for event in _app(store).resume(
                    ResumeRequest(
                        session_id="existing",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
        finally:
            await store.close()

    asyncio.run(continuation())


@pytest.mark.parametrize("damage", ["table", "index", "column", "wrong_index"])
@pytest.mark.parametrize("mode", list(schema.SchemaMode))
def test_current_sqlite_revision_rejects_missing_binding_structure(
    tmp_path, damage, mode, sqlite_resources
):
    async def run():
        async with sqlite_resources as resources:
            db = tmp_path / "damaged.sqlite"
            store = resources.own(SQLiteSessionStore(db))
            await store.close()
            with closing(sqlite3.connect(db)) as connection:
                if damage == "table":
                    connection.execute("DROP TABLE cayu_participant_session_bindings")
                elif damage == "column":
                    connection.execute(
                        "ALTER TABLE cayu_participant_session_bindings DROP COLUMN receipt_json"
                    )
                else:
                    connection.execute(
                        "DROP INDEX idx_cayu_participant_session_bindings_participant"
                    )
                    if damage == "wrong_index":
                        connection.execute(
                            "CREATE INDEX idx_cayu_participant_session_bindings_participant ON cayu_participant_session_bindings(session_id)"
                        )
            with pytest.raises(schema.SchemaError, match="Participant session bindings schema"):
                SQLiteSessionStore(db, schema_mode=mode)

    asyncio.run(run())


@pytest.mark.parametrize("revision", [96, 99, 101])
def test_historical_postgres_upgrade_continues_session(postgres_dsn, revision, monkeypatch):
    import psycopg

    from cayu.storage import postgres as pg

    async def run():
        async with await psycopg.AsyncConnection.connect(
            postgres_dsn, autocommit=True
        ) as connection:
            await connection.execute("DROP SCHEMA public CASCADE")
            await connection.execute("CREATE SCHEMA public")
        with monkeypatch.context() as historical:
            historical.setattr(
                schema, "REVISIONS", tuple(r for r in schema.REVISIONS if r.revision <= revision)
            )
            historical.setitem(pg._MIGRATION_STEPS, 96, pg.POSTGRES_TASK_GROUP_QUIESCENCE_DDL)
            store = pg.PostgresSessionStore(postgres_dsn, schema_mode=schema.SchemaMode.CREATE)
            try:
                events = [
                    event
                    async for event in _app(store).run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="existing",
                            messages=[Message.text("user", "historical input")],
                        )
                    )
                ]
                assert events[-1].type == EventType.SESSION_COMPLETED
            finally:
                await store.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            assert await (
                await connection.execute("SELECT to_regclass('cayu_participant_session_bindings')")
            ).fetchone() == (None,)
        for mode in (
            schema.SchemaMode.MIGRATE,
            schema.SchemaMode.MIGRATE,
            schema.SchemaMode.VALIDATE,
        ):
            store = pg.PostgresSessionStore(postgres_dsn, schema_mode=mode)
            try:
                assert await store.load_participant_session_binding("existing") is None
                assert await store.load("existing") is not None
                if mode == schema.SchemaMode.VALIDATE:
                    events = [
                        event
                        async for event in _app(store).resume(
                            ResumeRequest(
                                session_id="existing",
                                messages=[Message.text("user", "continue")],
                            )
                        )
                    ]
                    assert events[-1].type == EventType.SESSION_COMPLETED
            finally:
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("damage", ["table", "index", "column", "wrong_index"])
@pytest.mark.parametrize("mode", list(schema.SchemaMode))
def test_current_postgres_revision_rejects_missing_binding_structure(postgres_dsn, damage, mode):
    import psycopg

    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        async with await psycopg.AsyncConnection.connect(
            postgres_dsn, autocommit=True
        ) as connection:
            await connection.execute("DROP SCHEMA public CASCADE")
            await connection.execute("CREATE SCHEMA public")
        store = PostgresSessionStore(postgres_dsn, schema_mode=schema.SchemaMode.CREATE)
        try:
            assert await store.load_participant_session_binding("absent") is None
        finally:
            await store.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            if damage == "table":
                await connection.execute("DROP TABLE cayu_participant_session_bindings")
            elif damage == "column":
                await connection.execute(
                    "ALTER TABLE cayu_participant_session_bindings DROP COLUMN receipt_json"
                )
            else:
                await connection.execute(
                    "DROP INDEX idx_cayu_participant_session_bindings_participant"
                )
                if damage == "wrong_index":
                    await connection.execute(
                        "CREATE INDEX idx_cayu_participant_session_bindings_participant ON cayu_participant_session_bindings(session_id)"
                    )
        store = PostgresSessionStore(postgres_dsn, schema_mode=mode)
        try:
            with pytest.raises(schema.SchemaError, match="Participant session bindings schema"):
                await store.load_participant_session_binding("absent")
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("extra_column", [False, True])
def test_repair_preserves_existing_participant_binding(
    backend, extra_column, tmp_path, request, monkeypatch
):
    from tests.core.test_context_views import (
        _test_memory_participant_creation_is_atomic_and_replayable,
    )

    from cayu.storage.postgres import PostgresSessionStore

    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def exercise(own):
        if dsn is not None:
            import psycopg

            async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as connection:
                await connection.execute("DROP SCHEMA public CASCADE")
                await connection.execute("CREATE SCHEMA public")

        def factory(mode):
            if dsn is not None:
                return PostgresSessionStore(dsn, schema_mode=mode)
            return own(SQLiteSessionStore(tmp_path / "existing-bindings.sqlite", schema_mode=mode))

        with monkeypatch.context() as historical:
            historical.setattr(
                schema, "REVISIONS", tuple(r for r in schema.REVISIONS if r.revision <= 101)
            )
            store = factory(schema.SchemaMode.CREATE)
            try:
                (
                    creation,
                    receipt,
                    session_id,
                ) = await _test_memory_participant_creation_is_atomic_and_replayable(
                    store, cleanup=False
                )
                binding = receipt.binding
            finally:
                await store.close()
        if extra_column:
            ddl = "ALTER TABLE cayu_participant_session_bindings ADD COLUMN optional_metadata TEXT"
            if dsn is not None:
                async with await psycopg.AsyncConnection.connect(dsn) as connection:
                    await connection.execute(ddl)
            else:
                with closing(sqlite3.connect(tmp_path / "existing-bindings.sqlite")) as connection:
                    connection.execute(ddl)
                    connection.commit()
        for mode in (
            schema.SchemaMode.MIGRATE,
            schema.SchemaMode.MIGRATE,
            schema.SchemaMode.VALIDATE,
            schema.SchemaMode.CREATE,
        ):
            store = factory(mode)
            try:
                restored = await store.lookup_participant_session_creation(creation)
                assert restored is not None
                assert restored[0].id == session_id
                assert restored[1] == receipt
                assert await store.load_participant_session_binding(session_id) == binding
            finally:
                await store.close()

    async def run():
        if backend == "sqlite":
            async with request.getfixturevalue("sqlite_resources") as resources:
                await exercise(resources.own)
        else:
            await exercise(lambda store: store)

    asyncio.run(run())
