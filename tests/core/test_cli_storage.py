"""`cayu storage` CLI: status / migrate / export (ADR 0001 Phases 2-3)."""

from __future__ import annotations

import json

from cayu.cli import main
from cayu.storage import migrations as schema


def test_storage_status_reports_uninitialized(tmp_path, capsys):
    db = tmp_path / "s.sqlite"
    assert main(["storage", "status", "--sqlite", str(db)]) == 0
    out = capsys.readouterr().out
    assert "uninitialized" in out
    # A fresh DB shows every known revision as pending.
    pending = ", ".join(str(rev.revision) for rev in schema.REVISIONS)
    assert f"pending migrations: {pending}" in out


def test_storage_migrate_then_status_is_up_to_date(tmp_path, capsys):
    db = tmp_path / "s.sqlite"

    assert main(["storage", "migrate", "--sqlite", str(db)]) == 0
    migrate_out = capsys.readouterr().out
    assert f"revision {schema.LATEST_REVISION}" in migrate_out

    assert main(["storage", "status", "--sqlite", str(db)]) == 0
    status_out = capsys.readouterr().out
    assert "pending migrations: none (up to date)" in status_out


def test_storage_export_emits_jsonl(tmp_path, capsys):
    db = tmp_path / "s.sqlite"
    # Seed one session via a create-mode store, then export it.
    import asyncio

    from cayu import SQLiteSessionStore
    from cayu.core import Message
    from cayu.runtime import RunRequest, SessionIdentity

    async def seed() -> None:
        store = SQLiteSessionStore(db, schema_mode=schema.SchemaMode.CREATE)
        try:
            await store.create(
                RunRequest(agent_name="a", messages=[Message.text("user", "hi")]),
                identity=SessionIdentity(provider_name="fake", model="m"),
            )
        finally:
            await store.close()

    asyncio.run(seed())

    out_file = tmp_path / "dump.jsonl"
    assert main(["storage", "export", "--sqlite", str(db), "--output", str(out_file)]) == 0

    lines = [line for line in out_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "session"
    assert record["session"]["agent_name"] == "a"


def test_storage_export_uninitialized_fails_cleanly(tmp_path, capsys):
    db = tmp_path / "empty.sqlite"
    # Export uses validate mode, so an empty DB fails fast with a clean message.
    assert main(["storage", "export", "--sqlite", str(db)]) == 1
    assert "error:" in capsys.readouterr().err


def test_storage_export_failure_does_not_truncate_existing_output(tmp_path, capsys):
    db = tmp_path / "empty.sqlite"
    out_file = tmp_path / "existing.jsonl"
    out_file.write_text("keep me\n", encoding="utf-8")

    assert main(["storage", "export", "--sqlite", str(db), "--output", str(out_file)]) == 1

    assert "error:" in capsys.readouterr().err
    assert out_file.read_text(encoding="utf-8") == "keep me\n"


def test_redact_dsn_strips_credentials():
    from cayu.cli.storage import _redact_dsn, _sanitize

    secret = "postgresql://admin:s3cr3t@db.internal:5432/cayu?sslmode=require"
    redacted = _redact_dsn(secret)
    assert "s3cr3t" not in redacted
    assert "admin" not in redacted
    assert "db.internal:5432" in redacted
    # key/value (libpq) form is handled too.
    assert "topsecret" not in _redact_dsn("host=db user=admin password=topsecret dbname=cayu")
    assert "top secret" not in _redact_dsn("host=db password='top secret' user=admin")
    assert "top secret" not in _redact_dsn('host=db password="top secret" user=admin')
    assert "top secret" not in _redact_dsn("host=db password = 'top secret' user=admin")
    assert "top secret" not in _redact_dsn("host=db password= 'top secret' user=admin")
    assert "top\\ secret" not in _redact_dsn(r"host=db password=top\ secret user=admin")
    # Error-message sanitizer scrubs the password even if a driver echoes it.
    assert "s3cr3t" not in _sanitize(f"connection failed for {secret}", secret)
    encoded_secret = "postgresql://admin:top%20secret@db.internal:5432/cayu"
    assert "top secret" not in _sanitize(
        "authentication failed for password top secret",
        encoded_secret,
    )
    assert "top%20secret" not in _sanitize(
        "authentication failed for password top%20secret",
        encoded_secret,
    )
    libpq_secret = "host=db user=admin password='top secret' dbname=cayu"
    assert "top secret" not in _sanitize(f"connection failed for {libpq_secret}", libpq_secret)
    assert "top secret" not in _sanitize(
        "authentication failed for password top secret",
        libpq_secret,
    )
    assert "top secret" not in _sanitize(
        "authentication failed for password top secret",
        r"host=db user=admin password=top\ secret dbname=cayu",
    )


def test_storage_status_connection_error_does_not_leak_dsn(capsys):
    # An unreachable Postgres must not echo the password in the error output.
    dsn = "postgresql://admin:s3cr3t@127.0.0.1:1/nope"
    assert main(["storage", "status", "--postgres", dsn]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "s3cr3t" not in err
