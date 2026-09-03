"""`cayu storage` CLI: status / migrate / export (ADR 0001 Phases 2-3)."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

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


def _encoded_alias_key(byte: int) -> str:
    return base64.urlsafe_b64encode(bytes([byte]) * 32).decode("ascii").rstrip("=")


def _alias_codec(byte: int) -> PublicAuthorityAliasCodec:
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="primary",
            keys={"primary": SecretStr(_encoded_alias_key(byte))},
        )
    )


def _configure_alias_environment(monkeypatch, byte: int) -> str:
    encoded = _encoded_alias_key(byte)
    monkeypatch.setenv(PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV, "primary")
    monkeypatch.setenv(PUBLIC_AUTHORITY_ALIAS_KEYS_ENV, json.dumps({"primary": encoded}))
    return encoded


def _rewind_schema_revision(path, revision: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "DELETE FROM cayu_schema_migrations WHERE revision > ?",
            (revision,),
        )
        connection.execute(f"PRAGMA user_version = {revision}")
        connection.commit()
    finally:
        connection.close()


def _install_empty_legacy_recall_checkpoint(path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "cayu_agent_recall_delivery_states",
            "cayu_agent_recall_delivery_releases",
            "cayu_agent_recall_delivery_claims",
            "cayu_agent_recall_deliveries",
            "cayu_agent_recall_checkpoint_heads",
            "cayu_agent_recall_checkpoints",
        ):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute(
            """
            CREATE TABLE cayu_agent_recall_checkpoints (
                agent_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                knowledge_namespace TEXT NOT NULL,
                access_policy_sha256 TEXT NOT NULL,
                revision INTEGER NOT NULL,
                PRIMARY KEY (
                    agent_id, task_id, knowledge_namespace,
                    access_policy_sha256, revision
                )
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def _breaking_acknowledgements_after(revision: int) -> list[str]:
    arguments: list[str] = []
    for item in schema.pending(revision):
        if item.kind is schema.RevisionKind.BREAKING:
            arguments.extend(("--acknowledge-breaking", str(item.revision)))
    return arguments


def test_storage_status_reports_uninitialized(tmp_path, capsys):
    db = tmp_path / "s.sqlite"
    assert main(["storage", "status", "--sqlite", str(db)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["database"]["initialized"] is False
    # A fresh DB shows every known revision as pending.
    assert payload["pending_migrations"] == [rev.revision for rev in schema.REVISIONS]


def test_storage_migrate_then_status_is_up_to_date(tmp_path, capsys):
    db = tmp_path / "s.sqlite"

    assert main(["storage", "migrate", "--sqlite", str(db)]) == 0
    migrate = json.loads(capsys.readouterr().out)
    assert migrate["database"]["revision"] == schema.LATEST_REVISION

    assert main(["storage", "status", "--sqlite", str(db), "--table"]) == 0
    status_out = capsys.readouterr().out
    assert "pending migrations: none (up to date)" in status_out


def test_storage_migrate_receipt_binds_backup_path_authority_and_checks(
    tmp_path,
    capsys,
    monkeypatch,
):
    db = tmp_path / "receipt.sqlite"
    backup = tmp_path / "receipt.backup.sqlite"
    secret = _configure_alias_environment(monkeypatch, 7)

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                "--backup",
                str(backup),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    receipt = payload["migration_receipt"]
    assert receipt["input_revision"] == schema.UNINITIALIZED
    assert receipt["migration_steps"] == [revision.revision for revision in schema.REVISIONS]
    assert receipt["output_revision"] == schema.LATEST_REVISION
    assert receipt["backup"] == {
        "mode": "retained",
        "path": str(backup),
        "sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
    }
    assert receipt["authority_configuration"]["configured"] is True
    assert receipt["authority_configuration"]["active_key_id"] == "primary"
    assert (
        receipt["authority_configuration"]["fingerprint"] == _alias_codec(7).keyring_fingerprint()
    )
    assert receipt["checks"] == {
        "foreign_key_violations": 0,
        "integrity_check": "ok",
    }
    assert len(receipt["runtime"]["migration_identity_sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64
    assert secret not in json.dumps(payload)


def test_storage_migrate_missing_breaking_acknowledgement_is_byte_identical(
    tmp_path,
    capsys,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "missing-ack.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 72)
    before = db.read_bytes()

    assert main(["storage", "migrate", "--sqlite", str(db)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert "Breaking migration acknowledgement" in payload["error"]["message"]
    assert db.read_bytes() == before
    assert not tuple(tmp_path.glob("missing-ack.sqlite.cayu-backup-*"))


def test_storage_migrate_missing_alias_authority_is_byte_identical(
    tmp_path,
    capsys,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "missing-authority.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db, public_authority_alias_codec=_alias_codec(3))
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 72)
    before = db.read_bytes()

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                *_breaking_acknowledgements_after(72),
            ]
        )
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    assert "configure the deployment's alias keyring" in payload["error"]["message"]
    assert db.read_bytes() == before
    assert not tuple(tmp_path.glob("missing-authority.sqlite.cayu-backup-*"))


def test_storage_migrate_requires_empty_recall_reset_input_before_ddl(
    tmp_path,
    capsys,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "legacy-recall.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _install_empty_legacy_recall_checkpoint(db)
    _rewind_schema_revision(db, 72)
    before = db.read_bytes()

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                *_breaking_acknowledgements_after(72),
            ]
        )
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    assert "does not migrate the prerelease checkpoint schema" in payload["error"]["message"]
    assert db.read_bytes() == before

    backup = tmp_path / "legacy-recall.backup.sqlite"
    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                "--backup",
                str(backup),
                "--reset-empty-recall-state",
                *_breaking_acknowledgements_after(72),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)["migration_receipt"]
    assert receipt["migration_inputs"] == {"reset_empty_recall_state": True}
    assert receipt["output_revision"] == schema.LATEST_REVISION


def test_storage_migrate_recall_reset_rejects_populated_state_byte_identically(
    tmp_path,
    capsys,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "populated-legacy-recall.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _install_empty_legacy_recall_checkpoint(db)
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            "INSERT INTO cayu_agent_recall_checkpoints "
            "(agent_id, task_id, knowledge_namespace, access_policy_sha256, revision) "
            "VALUES ('agent', 'task', 'namespace', ?, 1)",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    _rewind_schema_revision(db, 72)
    before = db.read_bytes()

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                "--reset-empty-recall-state",
                *_breaking_acknowledgements_after(72),
            ]
        )
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    assert "only when all six checkpoint/delivery tables are empty" in payload["error"]["message"]
    assert db.read_bytes() == before
    assert not tuple(tmp_path.glob("populated-legacy-recall.sqlite.cayu-backup-*"))


def test_storage_migrate_incompatible_alias_authority_is_byte_identical(
    tmp_path,
    capsys,
    monkeypatch,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "incompatible-authority.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db, public_authority_alias_codec=_alias_codec(3))
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 72)
    before = db.read_bytes()
    secret = _configure_alias_environment(monkeypatch, 4)

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                *_breaking_acknowledgements_after(72),
            ]
        )
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    assert "different key material" in payload["error"]["message"]
    assert secret not in json.dumps(payload)
    assert db.read_bytes() == before
    assert not tuple(tmp_path.glob("incompatible-authority.sqlite.cayu-backup-*"))


def test_storage_migrate_existing_database_publishes_only_completed_staging_path(
    tmp_path,
    capsys,
    monkeypatch,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "existing.sqlite"
    backup = tmp_path / "existing.backup.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db, public_authority_alias_codec=_alias_codec(8))
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 72)
    _configure_alias_environment(monkeypatch, 8)

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                "--backup",
                str(backup),
                *_breaking_acknowledgements_after(72),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    receipt = payload["migration_receipt"]
    assert receipt["input_revision"] == 72
    assert receipt["migration_steps"] == list(range(73, schema.LATEST_REVISION + 1))
    assert receipt["acknowledged_breaking_revisions"] == [
        revision.revision
        for revision in schema.pending(72)
        if revision.kind is schema.RevisionKind.BREAKING
    ]
    backup_connection = sqlite3.connect(backup)
    try:
        assert schema.validate_migration_input(
            storage_cli.sqlite_support.read_schema_state(backup_connection)
        ) == schema.pending(72)
    finally:
        backup_connection.close()

    migrated_connection = sqlite3.connect(db)
    try:
        assert storage_cli.sqlite_support.read_schema_state(migrated_connection).revision == (
            schema.LATEST_REVISION
        )
        assert migrated_connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert migrated_connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        migrated_connection.close()


@pytest.mark.parametrize(
    ("owner", "attribute"),
    (
        (storage_cli, "_write_sqlite_snapshot"),
        (storage_cli.shutil, "copyfile"),
        (storage_cli, "_validate_sqlite_migration_result"),
        (storage_cli, "_checkpoint_and_sync_sqlite"),
        (storage_cli.os, "replace"),
    ),
)
def test_storage_migrate_injected_failure_never_publishes_partial_database(
    tmp_path,
    capsys,
    monkeypatch,
    owner,
    attribute,
):
    db = tmp_path / f"staging-failure-{attribute}.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup"]) == 0
    capsys.readouterr()
    before = db.read_bytes()

    def fail_step(*_args, **_kwargs):
        raise RuntimeError(f"injected {attribute} failure")

    monkeypatch.setattr(owner, attribute, fail_step)
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["message"] == f"injected {attribute} failure"
    assert db.read_bytes() == before
    assert not tuple(tmp_path.glob(".*.cayu-migration-*"))


def test_storage_migrate_invalid_output_fails_before_sqlite_publication(
    tmp_path,
    capsys,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "invalid-output.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 78)
    before = db.read_bytes()

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                "--waive-backup",
                "--acknowledge-breaking",
                "79",
                "--output",
                str(tmp_path / "missing" / "receipt.json"),
            ]
        )
        == 1
    )

    assert json.loads(capsys.readouterr().out)["error"]["code"] == "STORAGE_COMMAND_FAILED"
    assert db.read_bytes() == before
    assert not Path(f"{db}.cayu-migration-receipt.pending.json").exists()
    assert not Path(f"{db}.cayu-migration-receipt.json").exists()


def test_storage_migrate_rejects_output_that_aliases_retained_backup(
    tmp_path,
    capsys,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "backup-output-collision.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 78)
    before = db.read_bytes()
    backup_and_output = tmp_path / "backup-and-output.sqlite"

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(db),
                "--backup",
                str(backup_and_output),
                "--acknowledge-breaking",
                "79",
                "--output",
                str(backup_and_output),
            ]
        )
        == 1
    )

    error = json.loads(capsys.readouterr().out)["error"]
    assert "output path must differ" in error["message"]
    assert db.read_bytes() == before
    assert not backup_and_output.exists()


def test_storage_migrate_reuses_authenticated_backup_after_publication_failure(
    tmp_path,
    capsys,
    monkeypatch,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "retained-backup-retry.sqlite"
    backup = tmp_path / "retained-backup.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 78)
    before = db.read_bytes()
    real_replace = storage_cli.os.replace

    def fail_publication(source, destination):
        if Path(destination) == db and ".cayu-migration-" in Path(source).name:
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    arguments = [
        "storage",
        "migrate",
        "--sqlite",
        str(db),
        "--backup",
        str(backup),
        "--acknowledge-breaking",
        "79",
    ]
    monkeypatch.setattr(storage_cli.os, "replace", fail_publication)
    assert main(arguments) == 1
    assert "injected publication failure" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert db.read_bytes() == before
    pending_receipt = Path(f"{db}.cayu-migration-receipt.pending.json")
    persisted = json.loads(pending_receipt.read_text(encoding="utf-8"))
    assert backup.exists()
    assert persisted["backup"]["sha256"] == hashlib.sha256(backup.read_bytes()).hexdigest()

    monkeypatch.setattr(storage_cli.os, "replace", real_replace)
    assert main(arguments) == 0
    recovered = json.loads(capsys.readouterr().out)["migration_receipt"]
    assert recovered["receipt_sha256"] == persisted["receipt_sha256"]
    assert recovered["input_revision"] == 78
    assert recovered["output_revision"] == 79
    assert backup.exists()
    assert not pending_receipt.exists()


def test_storage_migrate_refuses_to_overwrite_same_revision_writes_on_retry(
    tmp_path,
    capsys,
    monkeypatch,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "changed-input-retry.sqlite"
    backup = tmp_path / "changed-input-backup.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 78)
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE app_data(value TEXT NOT NULL)")
    connection.execute("INSERT INTO app_data VALUES ('old')")
    connection.commit()
    connection.close()
    real_replace = storage_cli.os.replace

    def fail_publication(source, destination):
        if Path(destination) == db and ".cayu-migration-" in Path(source).name:
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    arguments = [
        "storage",
        "migrate",
        "--sqlite",
        str(db),
        "--backup",
        str(backup),
        "--acknowledge-breaking",
        "79",
    ]
    monkeypatch.setattr(storage_cli.os, "replace", fail_publication)
    assert main(arguments) == 1
    capsys.readouterr()

    connection = sqlite3.connect(db)
    connection.execute("UPDATE app_data SET value = 'new'")
    connection.commit()
    connection.close()

    monkeypatch.setattr(storage_cli.os, "replace", real_replace)
    assert main(arguments) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert "refusing to overwrite same-revision writes" in error["message"]
    connection = sqlite3.connect(db)
    try:
        assert connection.execute("SELECT value FROM app_data").fetchone() == ("new",)
        assert storage_cli.sqlite_support.read_schema_state(connection).revision == 78
    finally:
        connection.close()
    assert backup.exists()
    assert Path(f"{db}.cayu-migration-receipt.pending.json").exists()


def test_storage_migrate_rejects_symbolic_link_target(
    tmp_path,
    capsys,
):
    from cayu import SQLiteSessionStore

    referent = tmp_path / "referent.sqlite"
    link = tmp_path / "link.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(referent)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(referent, 78)
    before = referent.read_bytes()
    link.symlink_to(referent)

    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(link),
                "--waive-backup",
                "--acknowledge-breaking",
                "79",
            ]
        )
        == 1
    )

    assert "must not be a symbolic link" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert link.is_symlink()
    assert referent.read_bytes() == before


def test_storage_migrate_serializes_receipt_recovery_with_publication(
    tmp_path,
    monkeypatch,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "concurrent-receipt-recovery.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 78)
    output = tmp_path / "concurrent-receipt.json"
    args = argparse.Namespace(
        sqlite=str(db),
        backup=None,
        backup_sha256=None,
        waive_backup=True,
        acknowledge_breaking=[79],
        reset_empty_recall_state=False,
        output=str(output),
        output_format="json",
    )
    pending_receipt = Path(f"{db}.cayu-migration-receipt.pending.json")
    real_write_receipt = storage_cli._write_durable_sqlite_receipt
    real_path_lock = storage_cli.cooperative_path_lock
    receipt_written = threading.Event()
    release_publication = threading.Event()
    both_attempted_lock = threading.Event()
    lock_attempts = 0
    attempts_guard = threading.Lock()

    def pause_after_receipt(path, receipt):
        real_write_receipt(path, receipt)
        receipt_written.set()
        assert release_publication.wait(timeout=5)

    @contextlib.contextmanager
    def observed_path_lock(*lock_args, **lock_kwargs):
        nonlocal lock_attempts
        with attempts_guard:
            lock_attempts += 1
            if lock_attempts == 2:
                both_attempted_lock.set()
        with real_path_lock(*lock_args, **lock_kwargs):
            yield

    monkeypatch.setattr(storage_cli, "_write_durable_sqlite_receipt", pause_after_receipt)
    monkeypatch.setattr(storage_cli, "cooperative_path_lock", observed_path_lock)
    outcomes: list[int | BaseException] = []

    def migrate() -> None:
        try:
            outcomes.append(storage_cli._migrate_sqlite(args))
        except BaseException as exc:
            outcomes.append(exc)

    first = threading.Thread(target=migrate)
    first.start()
    assert receipt_written.wait(timeout=5)
    assert pending_receipt.exists()
    second = threading.Thread(target=migrate)
    second.start()
    assert both_attempted_lock.wait(timeout=5)
    assert pending_receipt.exists()
    release_publication.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert 0 in outcomes
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(failures) == 1
    assert "exact pending path (unexpected 79)" in str(failures[0])
    assert not pending_receipt.exists()
    assert not Path(f"{db}.cayu-migration-receipt.json").exists()


def test_storage_migrate_recovers_receipt_after_sqlite_publication(
    tmp_path,
    capsys,
    monkeypatch,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "receipt-recovery.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 78)
    original_render = storage_cli._render_migration

    def fail_delivery(*_args, **_kwargs):
        raise OSError("injected receipt delivery failure")

    monkeypatch.setattr(storage_cli, "_render_migration", fail_delivery)
    arguments = [
        "storage",
        "migrate",
        "--sqlite",
        str(db),
        "--waive-backup",
        "--acknowledge-breaking",
        "79",
    ]
    assert main(arguments) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert "migration committed but receipt delivery did not complete" in error["message"]

    durable_receipt = Path(f"{db}.cayu-migration-receipt.json")
    receipt = json.loads(durable_receipt.read_text(encoding="utf-8"))
    connection = sqlite3.connect(db)
    try:
        assert storage_cli.sqlite_support.read_schema_state(connection).revision == 79
    finally:
        connection.close()

    monkeypatch.setattr(storage_cli, "_render_migration", original_render)
    assert main(arguments) == 0
    recovered = json.loads(capsys.readouterr().out)["migration_receipt"]
    assert recovered["receipt_sha256"] == receipt["receipt_sha256"]
    assert not durable_receipt.exists()
    assert not Path(f"{db}.cayu-migration-receipt.pending.json").exists()


def test_storage_migrate_recovers_pending_receipt_after_sqlite_publication(
    tmp_path,
    capsys,
    monkeypatch,
):
    from cayu import SQLiteSessionStore

    db = tmp_path / "pending-receipt-recovery.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(db)
        await store.close()

    asyncio.run(initialize())
    _rewind_schema_revision(db, 78)
    pending_receipt = Path(f"{db}.cayu-migration-receipt.pending.json")
    durable_receipt = Path(f"{db}.cayu-migration-receipt.json")
    real_replace = storage_cli.os.replace

    def fail_receipt_promotion(source, destination):
        if Path(source) == pending_receipt and Path(destination) == durable_receipt:
            raise OSError("injected receipt promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(storage_cli.os, "replace", fail_receipt_promotion)
    arguments = [
        "storage",
        "migrate",
        "--sqlite",
        str(db),
        "--waive-backup",
        "--acknowledge-breaking",
        "79",
    ]
    assert main(arguments) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert "migration committed but receipt delivery did not complete" in error["message"]
    assert pending_receipt.exists()
    assert not durable_receipt.exists()

    monkeypatch.setattr(storage_cli.os, "replace", real_replace)
    assert main(arguments) == 0
    recovered = json.loads(capsys.readouterr().out)["migration_receipt"]
    assert recovered["input_revision"] == 78
    assert recovered["output_revision"] == 79
    assert not pending_receipt.exists()
    assert not durable_receipt.exists()


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
    assert (
        main(
            [
                "storage",
                "export",
                "--sqlite",
                str(db),
                "--jsonl",
                "--output",
                str(out_file),
            ]
        )
        == 0
    )

    lines = [line for line in out_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "session"
    assert record["session"]["agent_name"] == "a"

    json_file = tmp_path / "dump.json"
    assert main(["storage", "export", "--sqlite", str(db), "-o", str(json_file)]) == 0
    records = json.loads(json_file.read_text(encoding="utf-8"))
    assert [record["session"]["agent_name"] for record in records] == ["a"]


def test_storage_export_uninitialized_fails_cleanly(tmp_path, capsys):
    db = tmp_path / "empty.sqlite"
    # Export uses validate mode, so an empty DB fails fast with a clean message.
    assert main(["storage", "export", "--sqlite", str(db), "--jsonl"]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["error"]["code"] == "STORAGE_COMMAND_FAILED"


def test_storage_migrate_alias_keyring_runtime_failure_is_rendered(
    tmp_path,
    capsys,
    monkeypatch,
):
    def fail_migration(_args):
        raise RuntimeError("Postgres public authority alias key configuration is stale")

    monkeypatch.setattr(storage_cli, "_migrate", fail_migration)
    db = tmp_path / "configured.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(db)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "STORAGE_COMMAND_FAILED",
        "message": "Postgres public authority alias key configuration is stale",
    }


def test_storage_export_failure_does_not_truncate_existing_output(tmp_path, capsys):
    db = tmp_path / "empty.sqlite"
    out_file = tmp_path / "existing.jsonl"
    out_file.write_text("keep me\n", encoding="utf-8")

    assert (
        main(
            [
                "storage",
                "export",
                "--sqlite",
                str(db),
                "--jsonl",
                "--output",
                str(out_file),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["error"]["code"] == "STORAGE_COMMAND_FAILED"
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
    assert main(["storage", "status", "--postgres", dsn, "--table"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "s3cr3t" not in err


def test_storage_export_connection_error_does_not_leak_dsn(capsys, monkeypatch):
    # Export must redact the DSN password on a Postgres connection failure too.
    dsn = "postgresql://admin:s3cr3t@127.0.0.1:1/nope"

    class FailingStore:
        async def ensure_schema(self) -> None:
            raise RuntimeError(f"connection failed for {dsn}")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(storage_cli, "_session_store", lambda _args: FailingStore())

    assert main(["storage", "export", "--postgres", dsn, "--jsonl"]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "STORAGE_COMMAND_FAILED"
    assert "s3cr3t" not in payload["error"]["message"]
