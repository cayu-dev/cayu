"""Supported PostgreSQL class imports retain one identity after relocation."""

import asyncio
import importlib
import os
import pickle
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import get_type_hints

import pytest
from psycopg_pool import AsyncConnectionPool

import cayu
from cayu.storage._diagnostic_inspection import diagnostic_store_inspection
from cayu.support_bundles import _store_schema_readiness


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("event_watchers_postgres", "PostgresEventWatcherStore"),
        ("budget_postgres", "PostgresBudgetLedger"),
        ("tasks_postgres", "PostgresTaskStore"),
        ("work_context_postgres", "PostgresAgentWorkContextStore"),
        ("knowledge_postgres", "PostgresKnowledgeStore"),
    ],
)
def test_postgres_owner_preserves_public_imports_and_pickled_class(module_name, class_name):
    owner = importlib.import_module("cayu.storage." + module_name)
    store_type = getattr(owner, class_name)
    public_stores = importlib.import_module("cayu.storage.postgres")
    assert getattr(public_stores, class_name) is store_type
    assert getattr(cayu, class_name) is store_type
    assert class_name not in cayu.__all__
    storage = importlib.import_module("cayu.storage")
    assert getattr(storage, class_name) is store_type
    assert class_name not in storage.__all__
    assert get_type_hints(store_type.__init__)["pool"] == AsyncConnectionPool | None
    assert pickle.loads(pickle.dumps(store_type)) is store_type
    historical_reference = f"ccayu.storage.postgres\n{class_name}\n.".encode()
    assert pickle.loads(historical_reference) is store_type


@pytest.mark.parametrize(
    "module_name",
    [
        "_postgres_base",
        "budget_postgres",
        "event_watchers_postgres",
        "tasks_postgres",
        "work_context_postgres",
        "knowledge_postgres",
        "postgres",
    ],
)
def test_postgres_owner_preserves_optional_dependency_error(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys

class MissingPostgresDriver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"psycopg", "psycopg_pool"}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, MissingPostgresDriver())
import cayu
assert "cayu.storage.postgres" not in sys.modules
try:
    importlib.import_module("cayu.storage." + sys.argv[1])
except RuntimeError as error:
    assert 'pip install "cayu[postgres]"' in str(error)
else:
    raise AssertionError("missing driver did not produce the installation guidance")
""",
            module_name,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "class_name",
    [
        "PostgresBudgetLedger",
        "PostgresEmbeddingKnowledgeStore",
        "PostgresEvalStore",
        "PostgresEventWatcherStore",
        "PostgresKnowledgeStore",
        "PostgresSessionStore",
        "PostgresTaskStore",
    ],
)
@pytest.mark.parametrize("scenario", ["healthy", "invalid", "writable", "subclass", "cancelled"])
def test_postgres_schema_readiness_preserves_validation_boundary(class_name, scenario, monkeypatch):
    builtin = getattr(cayu, class_name)
    store_type = type("CustomStore", (builtin,), {}) if scenario == "subclass" else builtin
    options = {}
    if class_name == "PostgresEmbeddingKnowledgeStore":
        from cayu.embeddings import TextEmbeddingProvider

        class UnusedProvider(TextEmbeddingProvider):
            name = "schema-readiness"

            async def embed_texts(self, request):
                raise AssertionError("Schema diagnostics must not call the embedding provider.")

        options = {
            "embedding_provider": UnusedProvider(),
            "embedding_model": "schema-readiness",
            "embedding_dimensions": 3,
        }

    async def exercise():
        calls = []

        async def ensure_schema(self):
            calls.append(self)
            if scenario == "invalid":
                raise ValueError("incompatible schema")
            if scenario == "cancelled":
                raise asyncio.CancelledError

        monkeypatch.setattr(builtin, "ensure_schema", ensure_schema)
        inspection = nullcontext() if scenario == "writable" else diagnostic_store_inspection()
        with inspection:
            store = store_type("postgresql://example/cayu", **options)
        try:
            assert store._read_only is (scenario != "writable")
            if scenario == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await _store_schema_readiness(store)
            else:
                expected = {
                    "healthy": "validated_compatible",
                    "invalid": "validation_failed",
                    "writable": "unavailable",
                    "subclass": "unavailable",
                }[scenario]
                assert await _store_schema_readiness(store) == expected
            assert calls == ([] if scenario in {"writable", "subclass"} else [store])
            assert not store._opened
        finally:
            await store.close()

    asyncio.run(exercise())
