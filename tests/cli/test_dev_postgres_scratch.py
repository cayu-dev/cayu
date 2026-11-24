"""`cayu.testing.postgres.scratch_database` and `cayu dev postgres scratch -- COMMAND`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.testing.postgres import (
    ScratchDatabaseError,
    database_environment,
    database_url,
    scratch_database,
)


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        (
            "postgresql://postgres:secret@127.0.0.1:5432/postgres",
            "postgresql://postgres:secret@127.0.0.1:5432/scratch",
        ),
        (
            "host=db.example port=6543 user=app password='p@ss word' sslmode=require",
            "postgresql://app:p%40ss%20word@db.example:6543/scratch?sslmode=require",
        ),
        (
            "postgresql://u@[::1]:5432/postgres",
            "postgresql://u@[::1]:5432/scratch",
        ),
        (
            "host=h user=u options='-c statement_timeout=0'",
            "postgresql://u@h/scratch?options=-c%20statement_timeout%3D0",
        ),
        (
            "host=/var/run/postgresql user=me",
            "postgresql://me@/scratch?host=%2Fvar%2Frun%2Fpostgresql",
        ),
    ],
)
def test_the_url_names_the_scratch_database_on_the_same_server(server: str, expected: str) -> None:
    assert database_url(server, "scratch") == expected


def test_the_environment_points_only_at_the_scratch_database() -> None:
    base = {"CAYU_DATABASE_DIRECT_URL": "postgresql://elsewhere/db", "KEEP": "1"}

    assert database_environment("postgresql://h/scratch", base) == {
        "CAYU_DATABASE_URL": "postgresql://h/scratch",
        "KEEP": "1",
    }


def test_without_a_server_it_says_how_to_get_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAYU_TEST_POSTGRES_DSN", raising=False)

    with pytest.raises(ScratchDatabaseError, match="cayu dev postgres up"), scratch_database():
        pass


def _databases(server: str) -> set[str]:
    import psycopg

    with psycopg.connect(server, autocommit=True) as connection:
        return {row[0] for row in connection.execute("SELECT datname FROM pg_database")}


def test_a_scratch_database_is_migrated_then_dropped(
    tmp_path: Path, _postgres_server_dsn: str
) -> None:
    import psycopg

    with scratch_database(_postgres_server_dsn, project=tmp_path, prefix="scratch_test") as url:
        name = url.rsplit("/", 1)[1].partition("?")[0]
        assert name.startswith("scratch_test_")
        assert name in _databases(_postgres_server_dsn)
        with psycopg.connect(url) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                )
            }
        assert "cayu_schema_migrations" in tables
    assert name not in _databases(_postgres_server_dsn)


def test_a_failed_migration_drops_the_database_and_says_why(
    tmp_path: Path, _postgres_server_dsn: str
) -> None:
    # An alembic.ini with no migrations: Alembic fails, and the database goes anyway.
    (tmp_path / "alembic.ini").write_text("[alembic]\nscript_location = missing\n")
    before = _databases(_postgres_server_dsn)

    with (
        pytest.raises(ScratchDatabaseError, match="alembic upgrade head"),
        scratch_database(
            _postgres_server_dsn, project=tmp_path, cayu_migrations=False, prefix="scratch_test"
        ),
    ):
        pytest.fail("the scratch database should not be yielded")
    assert _databases(_postgres_server_dsn) == before


def test_the_command_runs_against_a_scratch_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _postgres_server_dsn: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "probe"\nversion = "0"\n')
    report = tmp_path / "seen.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_DATABASE_DIRECT_URL", "postgresql://elsewhere/db")
    probe = (
        "import json, os, sys; json.dump({k: os.environ.get(k) for k in "
        "('CAYU_DATABASE_URL', 'CAYU_DATABASE_DIRECT_URL')}, open(sys.argv[1], 'w')); "
        "sys.exit(7)"
    )

    code = main(
        [
            "dev",
            "postgres",
            "scratch",
            "--server",
            _postgres_server_dsn,
            "--no-cayu-migrations",
            "--",
            sys.executable,
            "-c",
            probe,
            str(report),
        ]
    )

    assert code == 7
    seen = json.loads(report.read_text())
    assert seen["CAYU_DATABASE_DIRECT_URL"] is None
    name = seen["CAYU_DATABASE_URL"].rsplit("/", 1)[1].partition("?")[0]
    assert name.startswith("cayu_scratch_")
    assert name not in _databases(_postgres_server_dsn)


def test_the_command_needs_a_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "probe"\nversion = "0"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAYU_TEST_POSTGRES_DSN", raising=False)
    monkeypatch.setattr("cayu.cli.dev_postgres.find", lambda project: None)

    assert main(["dev", "postgres", "scratch", "--", "true"]) == 3
    assert "cayu dev postgres up" in capsys.readouterr().err
