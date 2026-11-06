"""Revision 140: the Postgres function that checks the alias keyring on the server."""

from __future__ import annotations

import asyncio

import pytest

from cayu.sessions import RunRequest, SessionIdentity
from cayu.storage import migrations as schema
from cayu.storage.migrations import SchemaMode


def test_revision_140_is_additive_and_required_by_postgres_session_stores() -> None:
    from cayu.storage import postgres

    revision = schema.revision(140)
    assert revision.kind is schema.RevisionKind.ADDITIVE
    assert revision.compatible_from == schema.revision(139).compatible_from
    assert postgres._POSTGRES_SESSION_MIN_REQUIRED_REVISION >= 140


def test_postgres_revision_140_installs_and_validates_the_alias_check_function(
    postgres_dsn,
) -> None:
    import psycopg

    from cayu.storage import PostgresSessionStore

    async def run() -> None:
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.create(
                RunRequest(agent_name="scout", session_id="root", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
        finally:
            await store.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            await conn.execute(
                "DROP FUNCTION cayu_require_public_authority_alias_config(text, text)"
            )
            await conn.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 140")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="cayu storage migrate"):
                await validator.ensure_schema()
        finally:
            await validator.close()
        migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await migrator.ensure_schema()
            assert (await migrator.load("root")) is not None
        finally:
            await migrator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            cur = await conn.execute(
                "SELECT cayu_require_public_authority_alias_config(%s, %s)", (None, None)
            )
            assert await cur.fetchone() == (True,)
            # A function that checks nothing must not pass for the real one.
            await conn.execute(
                "CREATE OR REPLACE FUNCTION cayu_require_public_authority_alias_config("
                "expected_active_key_id TEXT, expected_keyring_fingerprint TEXT) "
                "RETURNS BOOLEAN AS $$ BEGIN RETURN TRUE; END; $$ LANGUAGE plpgsql"
            )
            await conn.commit()
        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(RuntimeError, match="alias check function"):
                await validator.ensure_schema()
        finally:
            await validator.close()

    asyncio.run(run())
