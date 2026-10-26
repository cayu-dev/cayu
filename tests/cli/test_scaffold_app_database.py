"""`cayu new --with app-database`: application tables in their own PostgreSQL schema."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from sqlalchemy.engine import URL

from cayu.cli import main
from cayu.cli.scaffold import project_files

PRESETS = ("agent", "service", "long-running", "coding")
APP_DATABASE_FILES = (
    "alembic.ini",
    "database/__init__.py",
    "database/models.py",
    "database/session.py",
    "database/CAPABILITY.md",
    "migrations/env.py",
    "migrations/script.py.mako",
    "migrations/versions/0001_app_tables.py",
    "tests/test_app_database.py",
)


def _dependency_names(pyproject: str) -> list[str]:
    return [
        requirement.split("[")[0].split(">")[0].split("=")[0]
        for requirement in tomllib.loads(pyproject)["project"]["dependencies"]
    ]


def test_dry_run_lists_the_capability_its_files_and_dependencies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = ["new", "demo", "--with", "app-database", "--dir", str(tmp_path), "--dry-run"]
    assert main([*command, "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert "app-database" in plan["capabilities"]
    assert set(APP_DATABASE_FILES) <= set(plan["files"])
    assert {"alembic>=1.13,<2", "sqlalchemy>=2,<3"} <= set(plan["dependencies"])
    assert not (tmp_path / "demo").exists()


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_generates_the_app_database_convention(preset: str) -> None:
    files = project_files("demo", preset=preset, with_capabilities=("app-database",))
    assert set(APP_DATABASE_FILES) <= set(files)
    assert {"sqlalchemy", "alembic"} <= set(_dependency_names(files["pyproject.toml"]))
    scaffold = tomllib.loads(files["pyproject.toml"])["tool"]["cayu"]["scaffold"]
    assert "app-database" in scaffold["capabilities"]
    guidance = files["AGENTS.md"]
    assert "`app` schema" in guidance
    assert "alembic upgrade head" in guidance
    assert "with their own table prefix" not in guidance
    # Cayu's own tables are never in reach of the application's migrations.
    environment = files["migrations/env.py"]
    assert "version_table_schema=SCHEMA" in environment
    assert "include_name=include_name" in environment
    assert "set search_path" not in environment.lower()
    # `configuration/settings.py` stays the project's only environment reader.
    assert "def configured_app_database_url(" in files["configuration/settings.py"]
    for relative in ("database/session.py", "migrations/env.py"):
        assert "os.environ" not in files[relative], relative


@pytest.mark.parametrize("preset", PRESETS)
def test_without_the_capability_nothing_changes(preset: str) -> None:
    files = project_files("demo", preset=preset)
    assert not set(APP_DATABASE_FILES) & set(files)
    assert not {"sqlalchemy", "alembic"} & set(_dependency_names(files["pyproject.toml"]))
    assert "with their own table prefix" in files["AGENTS.md"]


def test_generated_suite_runs_without_a_database_and_against_postgres(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    _postgres_server_dsn: str,
) -> None:
    assert main(["new", "demo", "--with", "app-database", "--dir", str(tmp_path), "--json"]) == 0
    capsys.readouterr()
    project = tmp_path / "demo"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CAYU_DATABASE_", "CAYU_TEST_POSTGRES"))
    }
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )

    def generated_suite(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-p",
                "no:cacheprovider",
                "tests/test_app_database.py",
            ],
            cwd=project,
            env={**environment, **extra},
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )

    # Without a database the hermetic checks pass and the PostgreSQL ones skip.
    offline = generated_suite({})
    assert offline.returncode == 0, offline.stdout + offline.stderr
    assert "SKIPPED" in offline.stdout

    online = generated_suite({"CAYU_TEST_POSTGRES_DSN": _postgres_server_dsn})
    assert online.returncode == 0, online.stdout + online.stderr
    assert "SKIPPED" not in online.stdout


def _generated_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    assert main(["new", "demo", "--with", "app-database", "--dir", str(tmp_path), "--json"]) == 0
    capsys.readouterr()
    return tmp_path / "demo"


def _alembic(project: Path, *arguments: str, database_url: str | None = None):
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("CAYU_DATABASE_")
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None, (str(Path(__file__).resolve().parents[2] / "src"), environment.get("PYTHONPATH"))
        )
    )
    if database_url is not None:
        environment["CAYU_DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_offline_sql_keeps_percent_signs_as_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _generated_project(tmp_path, capsys)
    (project / "migrations/versions/0002_percent.py").write_text(
        'from alembic import op\n\nrevision = "0002"\ndown_revision = "0001"\n'
        "branch_labels = None\ndepends_on = None\n\n\n"
        "def upgrade() -> None:\n    op.execute(\"UPDATE app.notes SET body = '100%'\")\n\n\n"
        "def downgrade() -> None:\n    pass\n",
        encoding="utf-8",
    )
    result = _alembic(project, "upgrade", "head", "--sql")
    assert result.returncode == 0, result.stderr
    assert 'CREATE SCHEMA IF NOT EXISTS "app"' in result.stdout
    assert "SET body = '100%';" in result.stdout


def test_a_role_named_app_is_refused_before_its_schema_is_kept(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    _postgres_server_dsn: str,
) -> None:
    from uuid import uuid4

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    project = _generated_project(tmp_path, capsys)
    database = f"app_role_{uuid4().hex}"
    # CI's server requires a password over TCP, so the role gets one.
    password = uuid4().hex
    with psycopg.connect(_postgres_server_dsn, autocommit=True) as connection:
        connection.execute("DROP ROLE IF EXISTS app")
        connection.execute(
            sql.SQL("CREATE ROLE app LOGIN PASSWORD {}").format(sql.Literal(password))
        )
        connection.execute(sql.SQL("CREATE DATABASE {} OWNER app").format(sql.Identifier(database)))
    server = conninfo_to_dict(_postgres_server_dsn)
    host = server.get("host", "127.0.0.1")
    socket = host.startswith("/")
    url = URL.create(
        "postgresql",
        username="app",
        password=password,
        host=None if socket else host,
        port=int(server["port"]) if "port" in server else None,
        database=database,
        query={"host": host} if socket else {},
    ).render_as_string(hide_password=False)
    try:
        result = _alembic(project, "upgrade", "head", database_url=url)
        assert result.returncode != 0
        assert "default schema is 'app'" in result.stderr
        with psycopg.connect(
            make_conninfo(_postgres_server_dsn, dbname=database), autocommit=True
        ) as connection:
            kept = connection.execute(
                "SELECT count(*) FROM pg_namespace WHERE nspname = 'app'"
            ).fetchone()
        assert kept == (0,)
    finally:
        with psycopg.connect(_postgres_server_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database))
            )
            connection.execute("DROP ROLE IF EXISTS app")
