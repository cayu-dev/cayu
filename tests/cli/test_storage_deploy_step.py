"""`cayu storage` as a deployment step: target resolution, backup authority, status."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import pytest
from pydantic import SecretStr

from cayu.cli import main
from cayu.cli import storage as storage_cli
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
)
from cayu.storage import migrations as schema

_UNREACHABLE = "postgresql://admin:pooled-s3cr3t@127.0.0.1:1/nope?sslmode=disable"
_UNREACHABLE_DIRECT = "postgresql://admin:direct-s3cr3t@127.0.0.1:1/direct?sslmode=disable"
_LATEST_BREAKING = max(
    item.revision for item in schema.REVISIONS if item.kind is schema.RevisionKind.BREAKING
)


@pytest.fixture(autouse=True)
def _isolated_target_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "CAYU_DATABASE_URL",
        storage_cli.DIRECT_DATABASE_URL_ENV,
        PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
        PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _alias_codec(byte: int) -> PublicAuthorityAliasCodec:
    encoded = base64.urlsafe_b64encode(bytes([byte]) * 32).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="primary", keys={"primary": SecretStr(encoded)})
    )


def _sqlite_url(path: Path) -> str:
    return f"sqlite://{path.resolve()}"


def _write_project(root: Path, config: str) -> None:
    (root / "pyproject.toml").write_text(config, encoding="utf-8")


def _json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    return json.loads(captured.out)


def _append_sqlite_revision(path: Path, revision: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO cayu_schema_migrations "
            "(revision, kind, compatible_from, checksum, applied_at) "
            "VALUES (?, 'breaking', ?, NULL, '2026-09-29T00:00:00Z')",
            (revision, revision),
        )
        connection.commit()
    finally:
        connection.close()


def _rewind_sqlite_before_latest_breaking(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "DELETE FROM cayu_schema_migrations WHERE revision >= ?",
            (_LATEST_BREAKING,),
        )
        connection.execute(f"PRAGMA user_version = {_LATEST_BREAKING - 1}")
        connection.commit()
    finally:
        connection.close()


# --- target resolution -------------------------------------------------------


def test_status_migrate_and_export_resolve_database_url(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "env.sqlite"
    monkeypatch.setenv("CAYU_DATABASE_URL", _sqlite_url(db))

    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_MIGRATION_AVAILABLE
    status = _json(capsys)
    assert status["target"] == str(db.resolve())
    assert status["target_source"] == "environment:CAYU_DATABASE_URL"

    assert main(["storage", "migrate"]) == 0
    migrated = _json(capsys)
    assert migrated["target_source"] == "environment:CAYU_DATABASE_URL"
    assert migrated["migration_receipt"]["target"] == str(db.resolve())

    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_UP_TO_DATE
    assert _json(capsys)["migration"]["need"] == "up_to_date"

    assert main(["storage", "export", "--jsonl"]) == 0
    assert "exported 0 session(s)" in capsys.readouterr().err


def test_storage_resolves_project_session_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(
        tmp_path,
        '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/project.db"\n',
    )
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    db = (tmp_path / "data" / "project.db").resolve()

    assert main(["storage", "migrate"]) == 0
    migrated = _json(capsys)
    assert migrated["target"] == str(db)
    assert migrated["target_source"] == "project"
    assert db.is_file()

    assert main(["storage", "status", "--table"]) == 0
    assert "migration need: up_to_date (exit 0)" in capsys.readouterr().out


def test_explicit_flag_wins_over_environment_and_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(
        tmp_path,
        '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/project.db"\n',
    )
    from_env = tmp_path / "env.sqlite"
    explicit = tmp_path / "explicit.sqlite"
    monkeypatch.setenv("CAYU_DATABASE_URL", _sqlite_url(from_env))
    monkeypatch.setenv(storage_cli.DIRECT_DATABASE_URL_ENV, _UNREACHABLE_DIRECT)

    assert main(["storage", "migrate", "--sqlite", str(explicit)]) == 0
    migrated = _json(capsys)
    assert migrated["target"] == str(explicit)
    assert migrated["target_source"] == "explicit"
    assert explicit.is_file()
    assert not from_env.exists()
    assert not (tmp_path / "data" / "project.db").exists()


def test_environment_wins_over_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(
        tmp_path,
        '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/project.db"\n',
    )
    from_env = tmp_path / "env.sqlite"
    monkeypatch.setenv("CAYU_DATABASE_URL", _sqlite_url(from_env))

    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_MIGRATION_AVAILABLE
    assert _json(capsys)["target"] == str(from_env.resolve())


def test_missing_target_is_an_actionable_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["storage", "migrate"]) == 1
    message = _json(capsys)["error"]["message"]
    assert "CAYU_DATABASE_URL" in message
    assert "--postgres" in message


def test_direct_url_is_used_only_by_resolved_migrate_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_DATABASE_URL", _UNREACHABLE)
    monkeypatch.setenv(storage_cli.DIRECT_DATABASE_URL_ENV, _UNREACHABLE_DIRECT)

    def resolved(command: str, **flags: str | None) -> argparse.Namespace:
        args = argparse.Namespace(
            storage_command=command,
            sqlite=flags.get("sqlite"),
            postgres=flags.get("postgres"),
        )
        storage_cli._resolve_storage_target(args)
        return args

    for command in ("migrate", "status"):
        args = resolved(command)
        assert args.postgres == _UNREACHABLE_DIRECT
        assert args.target_source == "environment:CAYU_DATABASE_DIRECT_URL"
    export = resolved("export")
    assert export.postgres == _UNREACHABLE
    assert export.target_source == "environment:CAYU_DATABASE_URL"
    explicit = resolved("migrate", postgres="postgresql://explicit/db")
    assert explicit.postgres == "postgresql://explicit/db"
    assert explicit.target_source == "explicit"


@pytest.mark.parametrize(
    ("database_url", "direct_url", "expected"),
    [
        (None, "sqlite:///tmp/x.db", "CAYU_DATABASE_DIRECT_URL must contain a Postgres URL"),
        (None, "  ", "CAYU_DATABASE_DIRECT_URL must contain a Postgres URL"),
        ("sqlite:///tmp/app.db", _UNREACHABLE_DIRECT, "resolved session store is SQLite"),
    ],
)
def test_direct_url_misconfiguration_is_rejected_without_echo(
    database_url: str | None,
    direct_url: str,
    expected: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_DATABASE_URL", database_url or _UNREACHABLE)
    monkeypatch.setenv(storage_cli.DIRECT_DATABASE_URL_ENV, direct_url)

    assert main(["storage", "status"]) == 1
    captured = capsys.readouterr()
    assert expected in json.loads(captured.out)["error"]["message"]
    assert "s3cr3t" not in captured.out + captured.err


@pytest.mark.parametrize(
    "arguments",
    [
        ["storage", "status"],
        ["storage", "status", "--table"],
        ["storage", "migrate", "--backup-managed", "rds-pitr:2026-09-29T12:00:00Z"],
        ["storage", "migrate", "--waive-backup", "--table"],
    ],
)
@pytest.mark.parametrize("direct", [False, True])
def test_resolved_postgres_connection_failure_never_prints_connection_string(
    arguments: list[str],
    direct: bool,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_DATABASE_URL", _UNREACHABLE)
    if direct:
        monkeypatch.setenv(storage_cli.DIRECT_DATABASE_URL_ENV, _UNREACHABLE_DIRECT)

    assert main(arguments) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "error" in output
    for secret in (_UNREACHABLE, _UNREACHABLE_DIRECT, "pooled-s3cr3t", "direct-s3cr3t"):
        assert secret not in output


def test_resolved_postgres_export_failure_never_prints_connection_string(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_DATABASE_URL", _UNREACHABLE)

    assert main(["storage", "export", "--jsonl"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "STORAGE_COMMAND_FAILED"
    for secret in (_UNREACHABLE, "pooled-s3cr3t"):
        assert secret not in captured.out + captured.err


# --- provider-managed backup references --------------------------------------


@pytest.mark.parametrize(
    "reference",
    [
        "rds-snapshot:cayu-release-42",
        "rds-snapshot:a",
        "rds-snapshot:rds:agent-db-2026-09-29-05-10",
        "rds-snapshot:" + "a" * 255,
        "rds-pitr:2026-09-29T12:00:00Z",
        "rds-pitr:2026-09-29T12:00:00.123456Z",
        "rds-pitr:2026-09-29t12:00:00z",
        "rds-pitr:2024-02-29T23:59:59+00:00",
    ],
)
def test_managed_backup_reference_accepts_documented_forms(reference: str) -> None:
    assert storage_cli._managed_backup_reference(reference) == reference


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("rds-snapshot:", "RDS snapshot identifier"),
        ("rds-snapshot:1starts-with-digit", "RDS snapshot identifier"),
        ("rds-snapshot:double--hyphen", "RDS snapshot identifier"),
        ("rds-snapshot:trailing-", "RDS snapshot identifier"),
        ("rds-snapshot:has_underscore", "RDS snapshot identifier"),
        ("rds-snapshot:" + "a" * 256, "RDS snapshot identifier"),
        ("rds-pitr:", "RFC 3339 UTC timestamp"),
        ("rds-pitr:2026-09-29", "RFC 3339 UTC timestamp"),
        ("rds-pitr:2026-09-29T12:00:00", "RFC 3339 UTC timestamp"),
        ("rds-pitr:2026-09-29T12:00:00+02:00", "RFC 3339 UTC timestamp"),
        ("rds-pitr:2026-09-29T12:00:00-00:00", "RFC 3339 UTC timestamp"),
        ("rds-pitr:2026-02-30T12:00:00Z", "RFC 3339 UTC timestamp"),
        ("rds-pitr:2026-09-29 12:00:00Z", "RFC 3339 UTC timestamp"),
        ("snapshot:cayu", "must be rds-snapshot:"),
        ("rds-snapshot", "must be rds-snapshot:"),
        ("RDS-PITR:2026-09-29T12:00:00Z", "must be rds-snapshot:"),
    ],
)
def test_malformed_managed_backup_reference_fails_before_connecting(
    reference: str,
    expected: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_DATABASE_URL", _UNREACHABLE)

    assert main(["storage", "migrate", "--backup-managed", reference]) == 1
    message = _json(capsys)["error"]["message"]
    assert expected in message
    assert "connection" not in message


@pytest.mark.parametrize("other", [["--waive-backup"], ["--backup-sha256", "a" * 64]])
def test_managed_backup_is_mutually_exclusive_with_other_backup_authority(
    other: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "storage",
                "migrate",
                "--postgres",
                _UNREACHABLE,
                "--backup-managed",
                "rds-snapshot:pre-deploy",
                *other,
            ]
        )
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_managed_backup_is_rejected_for_sqlite_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "managed.sqlite"
    monkeypatch.setenv("CAYU_DATABASE_URL", _sqlite_url(db))

    assert main(["storage", "migrate", "--backup-managed", "rds-pitr:2026-09-29T12:00:00Z"]) == 1
    assert "only valid for a Postgres target" in _json(capsys)["error"]["message"]
    assert not db.exists()


# --- SQLite status and empty-database receipts --------------------------------


def test_sqlite_first_migration_records_empty_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "fresh.sqlite"

    assert main(["storage", "migrate", "--sqlite", str(db)]) == 0
    receipt = _json(capsys)["migration_receipt"]
    assert receipt["input_revision"] == schema.UNINITIALIZED
    assert receipt["input_empty"] is True
    assert receipt["backup"]["mode"] == "retained"


def test_sqlite_status_json_and_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "status.sqlite"
    status = ["storage", "status", "--sqlite", str(db)]

    assert main(status) == storage_cli.STATUS_EXIT_MIGRATION_AVAILABLE
    fresh = _json(capsys)["migration"]
    assert fresh == {
        "need": "forward_migration",
        "exit_code": 3,
        "input_revision": schema.UNINITIALIZED,
        "input_empty": True,
        "target_revision": schema.LATEST_REVISION,
        "breaking_boundary": False,
        "breaking_revisions": [],
        "missing_acknowledgements": [],
        "resuming": False,
        "reason": None,
        "detail": None,
    }
    assert not db.exists()

    assert main(["storage", "migrate", "--sqlite", str(db)]) == 0
    capsys.readouterr()
    assert main(status) == storage_cli.STATUS_EXIT_UP_TO_DATE
    current = _json(capsys)
    assert current["up_to_date"] is True
    assert current["migration"]["need"] == "up_to_date"
    assert current["migration"]["input_revision"] == schema.LATEST_REVISION
    assert current["migration"]["input_empty"] is False

    _rewind_sqlite_before_latest_breaking(db)
    before = db.read_bytes()
    assert main(status) == storage_cli.STATUS_EXIT_INCOMPATIBLE
    blocked = _json(capsys)["migration"]
    assert blocked["need"] == "incompatible"
    assert blocked["reason"] == "breaking_acknowledgement_required"
    assert blocked["breaking_boundary"] is True
    assert blocked["breaking_revisions"] == [_LATEST_BREAKING]
    assert blocked["missing_acknowledgements"] == [_LATEST_BREAKING]

    acknowledged = [*status, "--acknowledge-breaking", str(_LATEST_BREAKING)]
    assert main(acknowledged) == storage_cli.STATUS_EXIT_MIGRATION_AVAILABLE
    ready = _json(capsys)["migration"]
    assert ready["need"] == "forward_migration"
    assert ready["missing_acknowledgements"] == []
    assert db.read_bytes() == before

    # A keyring the database never recorded is a preflight rejection, not a
    # connection failure: the deploy step learns migrate cannot proceed.
    monkeypatch.setenv(PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV, "primary")
    monkeypatch.setenv(PUBLIC_AUTHORITY_ALIAS_KEYS_ENV, "not json")
    assert main(acknowledged) == storage_cli.STATUS_EXIT_INCOMPATIBLE
    rejected = _json(capsys)["migration"]
    assert rejected["reason"] == "migration_preflight_rejected"
    assert db.read_bytes() == before


def test_sqlite_status_reports_database_newer_than_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "newer.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(db)]) == 0
    capsys.readouterr()
    _append_sqlite_revision(db, schema.LATEST_REVISION + 1)

    assert main(["storage", "status", "--sqlite", str(db)]) == storage_cli.STATUS_EXIT_INCOMPATIBLE
    payload = _json(capsys)
    assert payload["up_to_date"] is False
    migration = payload["migration"]
    assert migration["need"] == "incompatible"
    assert migration["reason"] == "database_newer_than_build"
    assert migration["input_revision"] == schema.LATEST_REVISION + 1
    assert migration["breaking_boundary"] is None

    assert main(["storage", "status", "--sqlite", str(db), "--table"]) == 4
    assert "reason: database_newer_than_build" in capsys.readouterr().out


# --- Postgres -----------------------------------------------------------------


def _postgres_url(dsn: str) -> str:
    from psycopg.conninfo import conninfo_to_dict

    params = conninfo_to_dict(dsn)
    user = quote(str(params.get("user", "")), safe="")
    password = params.get("password")
    credentials = user if password is None else f"{user}:{quote(str(password), safe='')}"
    host = params.get("host", "localhost")
    port = params.get("port", "5432")
    return f"postgresql://{credentials}@{host}:{port}/{quote(str(params['dbname']), safe='')}"


async def _reset_postgres(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("DROP SCHEMA public CASCADE")
        await conn.execute("CREATE SCHEMA public")


async def _postgres_query(dsn: str, query: str, params: tuple[object, ...] = ()) -> object:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        row = await (await conn.execute(query, params)).fetchone()
        await conn.commit()
        return None if row is None else row[0]


def _postgres_revision(dsn: str) -> int:
    async def read() -> int:
        import psycopg

        from cayu.storage import postgres

        async with await psycopg.AsyncConnection.connect(dsn) as conn, conn.cursor() as cur:
            return (await postgres.read_schema_state(cur)).revision

    return asyncio.run(read())


def _pending_receipt_count(dsn: str) -> int:
    registered = asyncio.run(
        _postgres_query(dsn, "SELECT to_regclass('cayu_schema_migration_receipts')")
    )
    if registered is None:
        return 0
    return int(
        asyncio.run(_postgres_query(dsn, "SELECT COUNT(*) FROM cayu_schema_migration_receipts"))
    )


def _prepare_postgres_before_latest_breaking(dsn: str, *, codec_byte: int | None = None) -> None:
    from cayu.storage.postgres import PostgresSessionStore

    async def prepare() -> None:
        await _reset_postgres(dsn)
        codec = None if codec_byte is None else _alias_codec(codec_byte)
        store = PostgresSessionStore(
            dsn,
            schema_mode=schema.SchemaMode.CREATE,
            public_authority_alias_codec=codec,
        )
        try:
            await store.ensure_schema()
        finally:
            await store.close()
        await _postgres_query(
            dsn,
            "DELETE FROM cayu_schema_migrations WHERE revision >= %s RETURNING revision",
            (_LATEST_BREAKING,),
        )

    asyncio.run(prepare())


def test_postgres_empty_database_migrates_without_backup_authority(
    postgres_dsn: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_reset_postgres(postgres_dsn))
    url = _postgres_url(postgres_dsn)
    monkeypatch.setenv("CAYU_DATABASE_URL", url)

    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_MIGRATION_AVAILABLE
    status = capsys.readouterr().out
    assert url not in status
    migration = json.loads(status)["migration"]
    assert migration["input_empty"] is True
    assert migration["breaking_boundary"] is False

    assert main(["storage", "migrate"]) == 0
    output = capsys.readouterr().out
    assert url not in output
    payload = json.loads(output)
    assert payload["target_source"] == "environment:CAYU_DATABASE_URL"
    receipt = payload["migration_receipt"]
    assert receipt["input_revision"] == schema.UNINITIALIZED
    assert receipt["input_empty"] is True
    assert receipt["backup"] == {"mode": "not_required", "path": None, "sha256": None}
    assert receipt["output_revision"] == schema.LATEST_REVISION

    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_UP_TO_DATE
    assert json.loads(capsys.readouterr().out)["migration"]["need"] == "up_to_date"


def test_postgres_direct_url_migration_records_managed_backup_for_empty_database(
    postgres_dsn: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_reset_postgres(postgres_dsn))
    direct = _postgres_url(postgres_dsn)
    monkeypatch.setenv("CAYU_DATABASE_URL", _UNREACHABLE)
    monkeypatch.setenv(storage_cli.DIRECT_DATABASE_URL_ENV, direct)

    # A platform may always pass a reference; an empty database records it.
    reference = "rds-pitr:2026-09-29T12:00:00Z"
    assert main(["storage", "migrate", "--backup-managed", reference]) == 0
    migrated = capsys.readouterr().out
    assert direct not in migrated
    payload = json.loads(migrated)
    assert payload["target_source"] == "environment:CAYU_DATABASE_DIRECT_URL"
    receipt = payload["migration_receipt"]
    assert receipt["input_revision"] == schema.UNINITIALIZED
    assert receipt["input_empty"] is True
    assert receipt["backup"] == {
        "mode": "provider_managed",
        "path": None,
        "sha256": None,
        "reference": reference,
    }

    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_UP_TO_DATE
    status = json.loads(capsys.readouterr().out)
    assert status["target_source"] == "environment:CAYU_DATABASE_DIRECT_URL"

    storage_cli._render_migration(
        "postgres",
        receipt["target"],
        schema.SchemaState(
            revision=receipt["output_revision"],
            compatible_from=receipt["output_compatible_from"],
        ),
        receipt,
        target_source="explicit",
        output_format="table",
        output=None,
    )
    assert f"managed backup: {reference}" in capsys.readouterr().out


def test_postgres_project_env_target_and_managed_snapshot_receipt(
    postgres_dsn: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_postgres_before_latest_breaking(postgres_dsn)
    _write_project(
        tmp_path,
        '[tool.cayu.session_store]\nbackend = "postgres"\nenv = "APP_DATABASE_URL"\n',
    )
    url = _postgres_url(postgres_dsn)
    monkeypatch.setenv("APP_DATABASE_URL", url)

    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_INCOMPATIBLE
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["target_source"] == "project:APP_DATABASE_URL"
    assert blocked["migration"]["reason"] == "breaking_acknowledgement_required"
    assert blocked["migration"]["breaking_revisions"] == [_LATEST_BREAKING]
    acknowledgement = ["--acknowledge-breaking", str(_LATEST_BREAKING)]
    assert main(["storage", "status", *acknowledgement]) == 3
    capsys.readouterr()

    # A database with Cayu data still requires explicit backup authority.
    assert main(["storage", "migrate", *acknowledgement]) == 1
    error = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "--backup-managed" in error
    assert _postgres_revision(postgres_dsn) == _LATEST_BREAKING - 1
    assert _pending_receipt_count(postgres_dsn) == 0

    reference = "rds-snapshot:cayu-pre-release-42"
    assert main(["storage", "migrate", "--backup-managed", reference, *acknowledgement]) == 0
    output = capsys.readouterr().out
    assert url not in output
    receipt = json.loads(output)["migration_receipt"]
    assert receipt["input_revision"] == _LATEST_BREAKING - 1
    assert receipt["input_empty"] is False
    assert receipt["acknowledged_breaking_revisions"] == [_LATEST_BREAKING]
    assert receipt["backup"] == {
        "mode": "provider_managed",
        "path": None,
        "sha256": None,
        "reference": reference,
    }
    assert _postgres_revision(postgres_dsn) == schema.LATEST_REVISION

    newer = schema.LATEST_REVISION + 1
    asyncio.run(
        _postgres_query(
            postgres_dsn,
            "INSERT INTO cayu_schema_migrations "
            "(revision, kind, compatible_from, checksum, applied_at) "
            "VALUES (%s, 'breaking', %s, NULL, now()) RETURNING revision",
            (newer, newer),
        )
    )
    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_INCOMPATIBLE
    migration = json.loads(capsys.readouterr().out)["migration"]
    assert migration["reason"] == "database_newer_than_build"
    assert migration["input_revision"] == newer


def test_postgres_preflight_rejects_missing_inputs_before_any_write(
    postgres_dsn: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_postgres_before_latest_breaking(postgres_dsn, codec_byte=3)
    monkeypatch.setenv("CAYU_DATABASE_URL", _postgres_url(postgres_dsn))
    backup = ["--backup-managed", "rds-pitr:2026-09-29T12:00:00Z"]
    acknowledgement = ["--acknowledge-breaking", str(_LATEST_BREAKING)]

    def unchanged() -> None:
        assert _postgres_revision(postgres_dsn) == _LATEST_BREAKING - 1
        assert _pending_receipt_count(postgres_dsn) == 0

    assert main(["storage", "migrate", *backup]) == 1
    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "Breaking migration acknowledgement" in message
    unchanged()

    assert main(["storage", "migrate", *backup, *acknowledgement]) == 1
    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "configure the deployment's alias keyring" in message
    unchanged()

    assert main(["storage", "status", *acknowledgement]) == storage_cli.STATUS_EXIT_INCOMPATIBLE
    migration = json.loads(capsys.readouterr().out)["migration"]
    assert migration["reason"] == "migration_preflight_rejected"
    assert "alias keyring" in migration["detail"]
    unchanged()


def test_postgres_status_follows_an_interrupted_first_migration(
    postgres_dsn: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_reset_postgres(postgres_dsn))
    monkeypatch.setenv("CAYU_DATABASE_URL", _postgres_url(postgres_dsn))

    def fail_delivery(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected receipt delivery failure")

    with monkeypatch.context() as patch:
        patch.setattr(storage_cli, "_render_migration", fail_delivery)
        assert main(["storage", "migrate"]) == 1
    capsys.readouterr()
    assert _pending_receipt_count(postgres_dsn) == 1
    # Simulate an interruption before the latest breaking revision committed.
    asyncio.run(
        _postgres_query(
            postgres_dsn,
            "DELETE FROM cayu_schema_migrations WHERE revision >= %s RETURNING revision",
            (_LATEST_BREAKING,),
        )
    )

    # The recorded input was empty, so resuming needs no acknowledgement even
    # though the live revision now sits before a breaking boundary.
    assert main(["storage", "status"]) == storage_cli.STATUS_EXIT_MIGRATION_AVAILABLE
    migration = json.loads(capsys.readouterr().out)["migration"]
    assert migration["resuming"] is True
    assert migration["input_revision"] == _LATEST_BREAKING - 1
    assert migration["breaking_revisions"] == []
