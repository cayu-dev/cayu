"""Schema owners compose with caller-owned cursors, transactions and imports."""

import asyncio
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cayu

_OWNERS = [
    "_postgres_catalog",
    "_postgres_eval_schema",
    "_postgres_knowledge_schema",
    "_postgres_task_schema",
    "_postgres_verified_work_schema",
    "_postgres_work_context_schema",
]


def _validators(owner):
    module = importlib.import_module("cayu.storage." + owner)
    return [value for name, value in inspect.getmembers(module) if name.startswith("_validate_")]


def _arguments(validator):
    parameters = inspect.signature(validator).parameters
    arguments = {
        name: True
        for name in (
            "allow_revision_43",
            "require_payload_bytes",
            "relation_aware",
            "require_processing_schema_version",
            "require_verifier_profiles",
            "require",
            "verify_event_ownership",
        )
        if name in parameters
    }
    if "expected_guard_sql" in parameters:
        from cayu.storage import _postgres_schema_history

        arguments["expected_guard_sql"] = _postgres_schema_history._MIGRATION_STEPS[88][1]
    if "tokenizer_version" in parameters:
        from cayu.sessions.transcript_queries import TRANSCRIPT_SEARCH_TOKENIZER_VERSION

        arguments["tokenizer_version"] = TRANSCRIPT_SEARCH_TOKENIZER_VERSION
    return arguments


@pytest.mark.parametrize("owner", _OWNERS)
def test_schema_owner_imports_without_drivers_or_store_owners(owner):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys

blocked = (
    "psycopg", "psycopg_pool", "cayu.storage._postgres_base",
    "cayu.storage._postgres_support", "cayu.storage._postgres_schema_history",
    "cayu.storage.postgres", "cayu.storage.tasks_postgres",
    "cayu.storage.work_context_postgres", "cayu.storage.knowledge_postgres",
    "cayu.storage.knowledge_embedding_postgres", "cayu.storage.budget_postgres",
    "cayu.storage.event_watchers_postgres", "cayu.storage.migrations",
    "cayu.sessions", "cayu.tasks", "cayu.runtime",
)
class RejectOwners(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("schema owner imported " + fullname)

sys.meta_path.insert(0, RejectOwners())
importlib.import_module("cayu.storage." + sys.argv[1])
assert not any(name in sys.modules for name in blocked)
""",
            owner,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("owner", [owner for owner in _OWNERS if owner != "_postgres_catalog"])
@pytest.mark.parametrize("failure_type", [asyncio.CancelledError, RuntimeError])
def test_schema_owner_propagates_cursor_failure(owner, failure_type):
    async def run():
        for validator in _validators(owner):
            failure = failure_type("caller-owned cursor failure")

            class Cursor:
                def __init__(self, failure):
                    self.failure = failure
                    self.calls = 0

                async def execute(self, *args, **kwargs):
                    self.calls += 1
                    raise self.failure

            cursor = Cursor(failure)
            with pytest.raises(failure_type) as caught:
                await validator(cursor, **_arguments(validator))
            assert caught.value is failure
            assert cursor.calls == 1

    asyncio.run(run())


@pytest.fixture(scope="module")
def current_schema(postgres_dsn):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def prepare():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.ensure_schema()
        finally:
            await store.close()

    asyncio.run(prepare())
    return postgres_dsn


@pytest.mark.parametrize("owner", [owner for owner in _OWNERS if owner != "_postgres_catalog"])
def test_schema_owner_preserves_caller_transaction(current_schema, owner):
    import psycopg
    from psycopg.pq import TransactionStatus

    async def run():
        async with await psycopg.AsyncConnection.connect(current_schema) as connection:
            await connection.execute("CREATE TEMP TABLE owner_sentinel (value TEXT)")
            await connection.execute("INSERT INTO owner_sentinel VALUES ('uncommitted')")
            before = await (await connection.execute("SELECT txid_current()")).fetchone()
            async with connection.cursor() as cursor:
                for validator in _validators(owner):
                    await validator(cursor, **_arguments(validator))
            assert connection.info.transaction_status == TransactionStatus.INTRANS
            assert await (await connection.execute("SELECT txid_current()")).fetchone() == before
            assert await (
                await connection.execute("SELECT value FROM owner_sentinel")
            ).fetchall() == [("uncommitted",)]
            await connection.rollback()
            assert await (
                await connection.execute("SELECT to_regclass('pg_temp.owner_sentinel')")
            ).fetchone() == (None,)
            await connection.rollback()
            await connection.execute("SET TRANSACTION READ ONLY")
            async with connection.cursor() as cursor:
                for validator in _validators(owner):
                    await validator(cursor, **_arguments(validator))
            assert connection.info.transaction_status == TransactionStatus.INTRANS
            await connection.rollback()

    asyncio.run(run())
