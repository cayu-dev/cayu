"""PostgreSQL schema declarations are usable without connection or store owners."""

import os
import subprocess
import sys
from pathlib import Path

import cayu


def test_postgres_history_creates_baseline_without_store_imports(postgres_dsn):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
import psycopg

blocked = {
    "cayu.storage.postgres",
    "cayu.storage._postgres_base",
    "cayu.storage._postgres_support",
    "cayu.storage.event_watchers_postgres",
    "cayu.storage.budget_postgres",
}

class NoStoreImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("history imported " + fullname)

sys.meta_path.insert(0, NoStoreImports())
from cayu.storage import _postgres_schema_history as history

with psycopg.connect(sys.argv[1]) as connection:
    connection.execute("CREATE TABLE application_data (value TEXT NOT NULL)")
    connection.execute("INSERT INTO application_data VALUES ('retained')")
    for _ in range(2):
        for statement in history.SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(history.MIGRATIONS_TABLE_DDL)
        connection.execute(history.MIGRATION_RECEIPTS_TABLE_DDL)
    assert connection.execute("SELECT value FROM application_data").fetchall() == [("retained",)]
    assert connection.execute("SELECT generation FROM cayu_accounting_state").fetchall() == [(0,)]
    assert connection.execute("SELECT revision FROM cayu_schema_migrations").fetchall() == []
    assert history._required_concurrent_indexes(17)

assert not blocked.intersection(sys.modules)
""",
            postgres_dsn,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
