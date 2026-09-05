"""Real database cleanup across missing pytest shutdown/report hooks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from scripts import run_runtime_qualification as runner

import cayu
from tests.qualification.postgres_cleanup import postgres_operation

pytestmark = pytest.mark.postgres


def _database_exists(dsn, database):
    import psycopg

    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database,),
            ).fetchone()
            is not None
        )


def _env(dsn):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(root), str(root / "src"))),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        CAYU_QUALIFICATION_PACKAGE=str(Path(cayu.__file__).resolve()),
        CAYU_TEST_POSTGRES_DSN=dsn,
        CAYU_QUALIFICATION_POSTGRES="1",
    )
    env.pop("PYTEST_ADDOPTS", None)
    return env


@pytest.mark.parametrize("shutdown", ["normal", "abrupt-exit", "timeout"])
def test_parent_drops_database_without_child_cleanup(tmp_path, postgres_dsn, monkeypatch, shutdown):
    env = _env(postgres_dsn)
    result = tmp_path / "result.json"
    marker = tmp_path / "database"
    env["CAYU_QUALIFICATION_RESULT"] = str(result)
    fixture = tmp_path / "test_database.py"
    fixture.write_text(
        "import os, signal, time, psycopg\nfrom pathlib import Path\n"
        "def test_database_is_provisioned():\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    with psycopg.connect(os.environ['CAYU_TEST_POSTGRES_DSN']) as connection:\n"
        "        name = connection.execute('SELECT current_database()').fetchone()[0]\n"
        f"    Path({str(marker)!r}).write_text(name)\n"
        + {"normal": "", "abrupt-exit": "    os._exit(0)\n", "timeout": "    time.sleep(60)\n"}[
            shutdown
        ]
    )
    operations = []

    def observe(python, database, child_env, operation):
        operations.append((operation, database))
        return postgres_operation(python, database, child_env, operation)

    monkeypatch.setattr(runner, "postgres_operation", observe)
    cleanup = {}
    try:
        code = runner.run_bounded(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.qualification.report_plugin",
                str(fixture),
            ],
            cwd=tmp_path,
            env=env,
            timeout=10,
            cleanup=cleanup,
        )
        assert code == (124 if shutdown == "timeout" else 0)
        assert marker.exists(), "The fixture must connect to the owned database before exit"
        database = marker.read_text()
        assert operations == [("create", database), ("drop", database)]
        assert not _database_exists(postgres_dsn, database)
        assert cleanup == {"postgres_databases_retained": 0}
        assert result.exists() is (shutdown == "normal")
        if result.exists():
            assert json.loads(result.read_text())["exit_code"] == 0
    finally:
        for _, database in operations:
            assert postgres_operation(sys.executable, database, env, "drop")


@pytest.mark.parametrize("failure", ["create-ack-loss", "drop-failure", "spawn-failure"])
def test_parent_keeps_database_ownership_on_lifecycle_failure(
    tmp_path,
    postgres_dsn,
    monkeypatch,
    failure,
):
    env = _env(postgres_dsn)
    operations = []
    dispatched = []

    def operation(python, database, child_env, action):
        operations.append((action, database))
        if failure == "drop-failure" and action == "drop":
            return False
        succeeded = postgres_operation(python, database, child_env, action)
        assert succeeded
        return not (failure == "create-ack-loss" and action == "create")

    def dispatch(*args, **kwargs):
        dispatched.append(True)
        if failure == "spawn-failure":
            raise OSError("process creation failed")
        return 0

    monkeypatch.setattr(runner, "postgres_operation", operation)
    monkeypatch.setattr(runner, "_run_bounded_process", dispatch)
    cleanup = {}
    try:
        if failure == "spawn-failure":
            with pytest.raises(OSError, match="process creation failed"):
                runner.run_bounded(
                    [sys.executable], cwd=tmp_path, env=env, timeout=10, cleanup=cleanup
                )
        else:
            code = runner.run_bounded(
                [sys.executable], cwd=tmp_path, env=env, timeout=10, cleanup=cleanup
            )
            assert code == (2 if failure == "create-ack-loss" else 1)
        assert [action for action, _ in operations] == ["create", "drop"]
        database = operations[0][1]
        assert operations[1][1] == database
        assert bool(dispatched) is (failure != "create-ack-loss")
        assert cleanup["postgres_databases_retained"] == int(failure == "drop-failure")
        assert _database_exists(postgres_dsn, database) is (failure == "drop-failure")
    finally:
        for _, database in operations:
            assert postgres_operation(sys.executable, database, env, "drop")
