"""The shared PostgreSQL connection owner does not depend on concrete stores."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from psycopg_pool import AsyncConnectionPool

import cayu
from cayu.storage._postgres_base import _PostgresStoreBase
from cayu.storage.migrations import SchemaMode


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("_postgres_base", "_PostgresStoreBase"),
        ("budget_postgres", "PostgresBudgetLedger"),
        ("tasks_postgres", "PostgresTaskStore"),
        ("event_watchers_postgres", "PostgresEventWatcherStore"),
        ("evals_postgres", "PostgresEvalStore"),
        ("collaboration_postgres", "PostgresCollaborationStore"),
        ("product_operations_postgres", "PostgresProductOperationStore"),
        ("model_policy_postgres", "PostgresModelPolicyStore"),
    ],
)
def test_postgres_subclass_import_does_not_load_monolithic_stores(module_name, class_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import importlib
import importlib.abc
import sys

class NoMonolithicStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cayu.storage.postgres":
            raise AssertionError("standalone owner imported monolithic stores")

sys.meta_path.insert(0, NoMonolithicStores())
owner = importlib.import_module("cayu.storage." + sys.argv[1])
store_type = getattr(owner, sys.argv[2])
if sys.argv[2] in {"PostgresBudgetLedger", "PostgresEventWatcherStore"}:
    import cayu
    import cayu.storage
    for public in (cayu, cayu.storage):
        assert getattr(public, sys.argv[2]) is store_type
        assert sys.argv[2] not in public.__all__
store = store_type("postgresql://example/cayu")
assert not store._opened
assert store._owns_pool
asyncio.run(store.close())
assert "cayu.storage.postgres" not in sys.modules
""",
            module_name,
            class_name,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_postgres_base_preserves_borrowed_pool_and_lazy_schema_lock(postgres_dsn):
    async def exercise():
        async with AsyncConnectionPool(postgres_dsn, open=False) as pool:
            first = _PostgresStoreBase(pool=pool, schema_mode=SchemaMode.CREATE)
            second = _PostgresStoreBase(pool=pool, schema_mode=SchemaMode.CREATE)
            try:
                await asyncio.gather(first.ensure_schema(), second.ensure_schema())
                assert first._schema_ready and second._schema_ready
                assert first._pool is pool and second._pool is pool
            finally:
                await first.close()
                await second.close()
            assert not pool.closed
            async with pool.connection() as connection:
                assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
        assert pool.closed

    asyncio.run(exercise())
