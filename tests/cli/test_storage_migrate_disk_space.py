"""`cayu storage migrate` refuses to start a SQLite migration that may not fit on disk."""

from __future__ import annotations

import base64
import collections
import json
import math
import sqlite3
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli import storage as storage_cli
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
)
from cayu.storage import _disk_space as disk_space
from cayu.storage import migrations as schema

_Usage = collections.namedtuple("_Usage", "total used free")


@pytest.fixture(autouse=True)
def _isolated_target_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CAYU_DATABASE_URL",
        storage_cli.DIRECT_DATABASE_URL_ENV,
        PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
        PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _fake_free_space(monkeypatch: pytest.MonkeyPatch, free: int) -> list[Path]:
    """Report ``free`` bytes on every file system; return the directories asked about."""

    asked: list[Path] = []

    def disk_usage(path: Path) -> _Usage:
        asked.append(Path(path))
        return _Usage(0, 0, free)

    monkeypatch.setattr(disk_space.shutil, "disk_usage", disk_usage)
    return asked


def _migrated_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    db = tmp_path / "data" / "sessions.db"
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup"]) == 0
    capsys.readouterr()
    _write_and_checkpoint(
        db,
        "CREATE TABLE filler (value BLOB)",
        "INSERT INTO filler VALUES (zeroblob(200000))",
    )
    return db


def _write_and_checkpoint(db: Path, *statements: str) -> None:
    """Run ``statements``, then fold the log into the file so its size stays put."""

    connection = sqlite3.connect(db)
    try:
        with connection:
            for statement in statements:
                connection.execute(statement)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


def _files_besides_sqlite_sidecars(directory: Path) -> set[Path]:
    # Read-only preflight connections may recreate an empty -wal and -shm.
    return {path for path in directory.iterdir() if not path.name.endswith(("-wal", "-shm"))}


def _error_message(capsys: pytest.CaptureFixture[str]) -> str:
    return json.loads(capsys.readouterr().out)["error"]["message"]


def _working_space(size: int) -> str:
    working = math.ceil(size * disk_space.SQLITE_MIGRATION_WORKING_SPACE_RATIO)
    return f"{disk_space.format_bytes(working)} for the copy's write-ahead log"


def test_needs_on_one_file_system_add_up(tmp_path, monkeypatch):
    asked = _fake_free_space(monkeypatch, 250)
    needs = (
        disk_space.SpaceNeed(tmp_path, 100, "for a"),
        disk_space.SpaceNeed(tmp_path / "not" / "created" / "yet", 200, "for b"),
        disk_space.SpaceNeed(tmp_path, 0, "for nothing"),
    )

    [shortfall] = disk_space.find_space_shortfalls(needs)

    assert shortfall.directory == tmp_path
    assert (shortfall.needed, shortfall.available) == (300, 250)
    assert [need.purpose for need in shortfall.needs] == ["for a", "for b"]
    assert asked == [tmp_path]
    with pytest.raises(disk_space.InsufficientDiskSpaceError) as raised:
        disk_space.require_free_space(needs, operation="Copying", hint="Free some space.")
    assert str(raised.value) == (
        f"Copying needs about 300 bytes free in {tmp_path} (100 bytes for a, 200 bytes for b); "
        "250 bytes is free. Free some space."
    )
    assert raised.value.shortfalls == (shortfall,)

    _fake_free_space(monkeypatch, 300)
    assert disk_space.find_space_shortfalls(needs) == ()
    disk_space.require_free_space(needs, operation="Copying")


def test_format_bytes_uses_decimal_units():
    assert disk_space.format_bytes(999) == "999 bytes"
    assert disk_space.format_bytes(7_300_000_000) == "7.3 GB"
    assert disk_space.format_bytes(15_000_000_000_000_000) == "15000.0 TB"


def test_migration_estimate_covers_backup_copy_and_working_space(tmp_path):
    db = tmp_path / "sessions.db"
    db.write_bytes(b"x" * 1000)
    Path(f"{db}-wal").write_bytes(b"x" * 200)
    backups = tmp_path / "backups"

    needs = disk_space.sqlite_migration_space_needs(db, backup_directory=backups)

    assert [(need.directory, need.bytes) for need in needs] == [
        (backups, 1200),
        (tmp_path, 1200),
        (tmp_path, 900),
    ]
    waived = disk_space.sqlite_migration_space_needs(
        db, backup_directory=None, snapshot_directory=tmp_path, migrates=False
    )
    assert [(need.bytes, need.purpose) for need in waived] == [
        (1200, "for a temporary snapshot"),
        (1200, "for the migrated copy"),
    ]
    assert disk_space.sqlite_migration_space_needs(tmp_path / "new.db", backup_directory=None) == ()


def test_migrate_refuses_before_writing_when_the_backup_and_copy_do_not_fit(
    tmp_path, monkeypatch, capsys
):
    db = _migrated_database(tmp_path, capsys)
    size = disk_space.sqlite_file_set_bytes(db)
    old_backup = db.parent / "sessions.db.cayu-backup-old.sqlite3"
    old_backup.write_bytes(b"x" * 1234)
    before = _files_besides_sqlite_sidecars(db.parent)
    asked = _fake_free_space(monkeypatch, size)

    assert main(["storage", "migrate", "--sqlite", str(db)]) == 1

    message = _error_message(capsys)
    size_text = disk_space.format_bytes(size)
    assert message.startswith(
        f"SQLite migration needs about {disk_space.format_bytes(2 * size)} free in {db.parent} "
        f"({size_text} for the backup, {size_text} for the migrated copy); {size_text} is free."
    )
    # Nothing is pending, so no working space is reserved for revisions.
    assert "write-ahead log" not in message
    assert f"1 older backup(s) of this database in {db.parent} take 1.2 KB" in message
    assert "point --backup-dir at another disk" in message
    assert "--skip-disk-space-check" in message
    assert asked == [db.parent]
    assert _files_besides_sqlite_sidecars(db.parent) == before

    assert main(["storage", "migrate", "--sqlite", str(db), "--skip-disk-space-check"]) == 0
    assert json.loads(capsys.readouterr().out)["backup"]["path"] is not None


def test_migrate_counts_a_waived_backup_as_a_temporary_snapshot(tmp_path, monkeypatch, capsys):
    db = _migrated_database(tmp_path, capsys)
    _fake_free_space(monkeypatch, disk_space.sqlite_file_set_bytes(db))

    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup"]) == 1

    message = _error_message(capsys)
    assert "for a temporary snapshot" in message
    assert "for the backup" not in message
    assert "--backup-dir" not in message
    assert message.endswith("Free some space, or pass --skip-disk-space-check to start anyway.")


def test_migrate_checks_the_backup_directory_and_starts_when_space_is_enough(
    tmp_path, monkeypatch, capsys
):
    db = _migrated_database(tmp_path, capsys)
    backups = tmp_path / "backups"
    asked = _fake_free_space(monkeypatch, 2 * disk_space.sqlite_file_set_bytes(db))

    assert main(["storage", "migrate", "--sqlite", str(db), "--backup-dir", str(backups)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["backup"]["path"]).parent == backups
    # The backup directory did not exist yet, so its nearest existing ancestor
    # stands for its file system; both needs share tmp_path's file system.
    assert asked == [tmp_path]


def test_skip_disk_space_check_is_sqlite_only(capsys):
    postgres = "postgresql://cayu@127.0.0.1:1/none"
    command = ["storage", "migrate", "--postgres", postgres, "--waive-backup"]

    assert main([*command, "--skip-disk-space-check"]) == 1

    assert "--skip-disk-space-check is only valid with --sqlite" in _error_message(capsys)


def test_migrate_reserves_working_space_when_revisions_are_pending(tmp_path, monkeypatch, capsys):
    db = _migrated_database(tmp_path, capsys)
    _write_and_checkpoint(
        db, f"DELETE FROM cayu_schema_migrations WHERE revision = {schema.LATEST_REVISION}"
    )
    size = disk_space.sqlite_file_set_bytes(db)
    # Room for the backup and the copy, but not for the working space.
    _fake_free_space(monkeypatch, 2 * size)
    command = ["storage", "migrate", "--sqlite", str(db), "--waive-backup"]
    command += ["--acknowledge-breaking", str(schema.LATEST_REVISION)]

    assert main(command) == 1

    assert _working_space(size) in _error_message(capsys)


def test_migrate_reserves_working_space_for_a_new_alias_key_backfill(tmp_path, monkeypatch, capsys):
    db = _migrated_database(tmp_path, capsys)
    size = disk_space.sqlite_file_set_bytes(db)
    _fake_free_space(monkeypatch, 2 * size)
    key = base64.urlsafe_b64encode(b"\x07" * 32).decode("ascii").rstrip("=")
    monkeypatch.setenv(PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV, "primary")
    monkeypatch.setenv(PUBLIC_AUTHORITY_ALIAS_KEYS_ENV, json.dumps({"primary": key}))

    # No revision is pending, but opening the store backfills aliases for the key.
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup"]) == 1

    assert _working_space(size) in _error_message(capsys)


def test_migrate_gives_backup_advice_only_when_the_backup_disk_is_short(
    tmp_path, monkeypatch, capsys
):
    db = _migrated_database(tmp_path, capsys)
    size = disk_space.sqlite_file_set_bytes(db)
    backups = tmp_path / "backups"
    backups.mkdir()
    (backups / f"{storage_cli._sqlite_backup_prefix(db, backups)}old.sqlite3").write_bytes(b"x")
    monkeypatch.setattr(
        disk_space, "file_system_id", lambda directory: 1 if directory == backups else 0
    )
    monkeypatch.setattr(
        disk_space.shutil,
        "disk_usage",
        lambda directory: _Usage(0, 0, 10 * size if Path(directory) == backups else size // 2),
    )

    assert main(["storage", "migrate", "--sqlite", str(db), "--backup-dir", str(backups)]) == 1

    message = _error_message(capsys)
    assert f"in {db.parent} ({disk_space.format_bytes(size)} for the migrated copy)" in message
    assert "for the backup" not in message
    assert "older backup" not in message
    assert message.endswith("Free some space, or pass --skip-disk-space-check to start anyway.")


def test_migrate_names_the_backup_option_the_operator_used(tmp_path, monkeypatch, capsys):
    db = _migrated_database(tmp_path, capsys)
    _fake_free_space(monkeypatch, disk_space.sqlite_file_set_bytes(db))
    backup = tmp_path / "chosen.sqlite3"

    assert main(["storage", "migrate", "--sqlite", str(db), "--backup", str(backup)]) == 1

    message = _error_message(capsys)
    assert "point --backup at another disk" in message
    assert "--backup-dir" not in message
    assert not backup.exists()
