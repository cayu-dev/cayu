"""Shared PostgreSQL resets must remove the worker schema's dependent tables."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from cayu import PostgresTaskStore
from cayu.storage.migrations import SchemaMode


@pytest.mark.parametrize(
    ("module", "tables_name"),
    [
        ("test_session_store_shared_conformance", "_POSTGRES_TABLES"),
        ("test_event_watchers", "_POSTGRES_TABLES"),
        ("test_postgres_budget_ledger", "_TABLES"),
        ("postgres_contention_support", "POSTGRES_CONTENTION_TABLES"),
        ("test_postgres_session_store", "_TABLES"),
        ("test_postgres_task_store", "_TABLES"),
        ("test_postgres_migrations", "_TABLES"),
        ("test_postgres_knowledge_store", "_TABLES"),
    ],
)
def test_worker_schema_survives_shared_test_reset(postgres_dsn, module, tables_name):
    import psycopg
    from psycopg import sql

    tables = getattr(importlib.import_module(f"tests.core.{module}"), tables_name)

    async def scenario():
        for index, mode in enumerate((SchemaMode.CREATE, SchemaMode.CREATE, SchemaMode.VALIDATE)):
            store = PostgresTaskStore(postgres_dsn, schema_mode=mode)
            try:
                assert await store.list_tasks() == []
            finally:
                await store.close()
            if index == 0:
                # Only reset once: the second create must reconstruct complete
                # constraints, and validate must independently accept them.
                async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                    for table in tables:
                        await connection.execute(
                            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table))
                        )
                    for table in (
                        "cayu_work_attempt_preparation_holds",
                        "cayu_work_attempt_lifecycle_receipts",
                    ):
                        cursor = await connection.execute("SELECT to_regclass(%s)", (table,))
                        assert await cursor.fetchone() == (None,)

    asyncio.run(scenario())
