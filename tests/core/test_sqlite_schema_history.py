"""Schema declarations work independently and remain authoritative during upgrades."""

import asyncio
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_schema_history as sqlite_schema_history
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations


def test_sqlite_schema_history_creates_baseline_without_migration_execution(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sqlite3
import sys
from contextlib import closing

blocked = {
    "cayu.storage._sqlite_support",
    "cayu.storage._sqlite_connection",
    "cayu.storage.sqlite",
    "cayu.storage.tasks_sqlite",
    "cayu.storage.postgres",
}

class NoExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("schema declarations imported " + fullname)

sys.meta_path.insert(0, NoExecutionImports())
from cayu.storage import _sqlite_schema_history as history

with closing(sqlite3.connect(sys.argv[1])) as connection:
    connection.execute("CREATE TABLE application_data (value TEXT NOT NULL)")
    connection.execute("INSERT INTO application_data VALUES ('retained')")
    connection.commit()
    for _ in range(2):
        connection.executescript(history._BASELINE_DDL)
        connection.execute(history._MIGRATIONS_TABLE_DDL)
        connection.commit()
    assert connection.execute("SELECT value FROM application_data").fetchall() == [("retained",)]
    assert connection.execute("SELECT generation FROM cayu_accounting_state").fetchall() == [(0,)]
    assert connection.execute("SELECT revision FROM cayu_schema_migrations").fetchall() == []
    assert connection.execute("PRAGMA user_version").fetchone() == (0,)
    assert not connection.in_transaction

assert not blocked.intersection(sys.modules)
""",
            str(tmp_path / "baseline.sqlite"),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_revision_uses_history_owner_and_rolls_back_failed_ddl(
    tmp_path, sqlite_resources, monkeypatch
):
    def snapshot(connection):
        return (
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall(),
            connection.execute("SELECT * FROM cayu_schema_migrations ORDER BY revision").fetchall(),
            connection.execute("PRAGMA user_version").fetchone(),
        )

    async def exercise():
        async with sqlite_resources as resources:
            connection = resources.own(
                sqlite_connection.connect(tmp_path / "upgrade.sqlite"), kind="connection"
            )
            connection.execute(sqlite_schema_history._MIGRATIONS_TABLE_DDL)
            sqlite_support._apply_baseline(connection)
            pending = list(migrations.pending(migrations.BASELINE_REVISION))
            for revision in pending[:-1]:
                sqlite_support._apply_revision(connection, revision)
            final = pending[-1]
            before = snapshot(connection)
            with monkeypatch.context() as fault:
                fault.setattr(
                    sqlite_schema_history,
                    "_MIGRATION_STEPS",
                    {
                        **sqlite_schema_history._MIGRATION_STEPS,
                        final.revision: sqlite_schema_history._MIGRATION_STEPS.get(
                            final.revision, ""
                        )
                        + "\nCREATE TABLE cayu_failed_upgrade (value TEXT);"
                        + "\nINSERT INTO cayu_absent_upgrade_target VALUES (1);",
                    },
                )
                with pytest.raises(sqlite3.OperationalError, match="cayu_absent_upgrade_target"):
                    sqlite_support._apply_revision(connection, final)
            assert not connection.in_transaction
            assert snapshot(connection) == before
            sqlite_support._apply_revision(connection, final)
            sqlite_support.reconcile_schema(connection, migrations.SchemaMode.VALIDATE)
            assert (
                sqlite_support.read_schema_state(connection).revision == migrations.LATEST_REVISION
            )
            assert not connection.in_transaction

    asyncio.run(exercise())
