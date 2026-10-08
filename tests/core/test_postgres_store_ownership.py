"""Supported PostgreSQL class imports retain one identity after relocation."""

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest
from psycopg_pool import AsyncConnectionPool

import cayu


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("event_watchers_postgres", "PostgresEventWatcherStore"),
        ("budget_postgres", "PostgresBudgetLedger"),
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
    "module_name", ["_postgres_base", "budget_postgres", "event_watchers_postgres", "postgres"]
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
