"""`cayu keys status`, `cayu storage reset-aliases` and the check finding for a keyed store."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli._local_alias_keys import keyring_required_diagnostic
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    _AliasStoreKeyringRequired,
    public_authority_alias_codec_from_environment,
)
from cayu.storage import SQLiteSessionStore

_KEY = "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE"
_OTHER_KEY = "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"


def _environment(key: str = _KEY, key_id: str = "primary") -> dict[str, str]:
    return {
        PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV: key_id,
        PUBLIC_AUTHORITY_ALIAS_KEYS_ENV: json.dumps({key_id: key}),
    }


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "probe"\n\n[tool.cayu]\nfactory = "app:build_app"\n\n'
        '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    for name in (PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV, PUBLIC_AUTHORITY_ALIAS_KEYS_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    return root


def _open_with_keyring(path: Path) -> None:
    codec = public_authority_alias_codec_from_environment(_environment())
    store = SQLiteSessionStore(path, public_authority_alias_codec=codec)
    asyncio.run(store.close())


def _status(capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    code = main(["keys", "status", "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_status_of_a_store_that_never_used_a_keyring(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _status(capsys)
    assert code == 0
    assert payload["store"]["exists"] is False
    assert payload["writable_open"] == "succeeds"

    asyncio.run(SQLiteSessionStore(project / "data" / "cayu.db").close())
    code, payload = _status(capsys)
    assert code == 0
    assert payload["store"]["requires_keyring"] is False


def test_status_names_the_recorded_keys_and_compares_the_environment(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    database = project / "data" / "cayu.db"
    _open_with_keyring(database)

    code, payload = _status(capsys)
    assert code == 3
    assert payload["environment"]["state"] == "unset"
    assert payload["store"]["requires_keyring"] is True
    assert payload["store"]["active_key_id"] == "primary"
    (key,) = payload["store"]["keys"]
    assert key["key_id"] == "primary"
    assert key["first_used_at"].endswith("Z")
    assert key["environment"] == "not_in_environment"
    assert payload["writable_open"] == "refused"

    for name, value in _environment().items():
        monkeypatch.setenv(name, value)
    code, payload = _status(capsys)
    assert code == 0
    assert payload["store"]["keys"][0]["environment"] == "matches"

    for name, value in _environment(_OTHER_KEY).items():
        monkeypatch.setenv(name, value)
    code, payload = _status(capsys)
    assert code == 3
    assert payload["store"]["keys"][0]["environment"] == "different"

    # The human form says the same, and reading never wrote to the database.
    before = database.stat().st_mtime_ns
    assert main(["keys", "status"]) == 3
    text = capsys.readouterr().out
    assert "requires a keyring" in text
    assert "DIFFERENT material" in text
    assert database.stat().st_mtime_ns == before


def test_reset_aliases_lets_the_store_open_without_a_keyring(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = project / "data" / "cayu.db"
    _open_with_keyring(database)
    with pytest.raises(_AliasStoreKeyringRequired):
        SQLiteSessionStore(database)

    assert main(["storage", "reset-aliases", "--yes"]) == 0
    assert "dropped" in capsys.readouterr().out

    asyncio.run(SQLiteSessionStore(database).close())
    code, payload = _status(capsys)
    assert code == 0
    assert payload["store"]["requires_keyring"] is False
    # Nothing left to reset.
    assert main(["storage", "reset-aliases", "--yes"]) == 0
    assert "nothing to reset" in capsys.readouterr().out


def test_reset_aliases_needs_confirmation(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    database = project / "data" / "cayu.db"
    _open_with_keyring(database)

    # No terminal and no --yes: refuse.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["storage", "reset-aliases"]) == 1
    assert "--yes" in capsys.readouterr().err

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    assert main(["storage", "reset-aliases"]) == 1
    assert "Not reset" in capsys.readouterr().out
    with pytest.raises(_AliasStoreKeyringRequired):
        SQLiteSessionStore(database)

    monkeypatch.setattr("builtins.input", lambda _prompt: "cayu.db")
    assert main(["storage", "reset-aliases"]) == 0
    asyncio.run(SQLiteSessionStore(database).close())


def test_reset_aliases_refuses_outside_data_and_postgres(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = project / "elsewhere.db"
    _open_with_keyring(outside)

    assert main(["storage", "reset-aliases", "--sqlite", str(outside), "--yes"]) == 1
    assert "outside" in capsys.readouterr().err
    with pytest.raises(_AliasStoreKeyringRequired):
        SQLiteSessionStore(outside)

    monkeypatch.setenv("CAYU_DATABASE_URL", "postgresql://user@localhost/db")
    assert main(["storage", "reset-aliases", "--yes"]) == 1
    assert "PostgreSQL" in capsys.readouterr().err


def test_reset_aliases_refuses_a_symlinked_data_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "probe"\n\n[tool.cayu]\nfactory = "app:build_app"\n',
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)
    _open_with_keyring(outside / "cayu.db")
    monkeypatch.chdir(root)
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)

    assert main(["storage", "reset-aliases", "--yes"]) == 1
    assert "symbolic link" in capsys.readouterr().err
    with pytest.raises(_AliasStoreKeyringRequired):
        SQLiteSessionStore(outside / "cayu.db")


def test_status_does_not_claim_an_unchecked_store_opens(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAYU_DATABASE_URL", "postgresql://user@localhost/db")
    assert main(["keys", "status", "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["writable_open"] == "not_checked"

    monkeypatch.setenv("CAYU_DATABASE_URL", "not-a-url")
    assert main(["keys", "status", "--json"]) == 1
    assert "error" in capsys.readouterr().err


def test_reset_aliases_refuses_outside_a_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    database = tmp_path / "data" / "cayu.db"
    database.parent.mkdir()
    _open_with_keyring(database)

    assert main(["storage", "reset-aliases", "--sqlite", str(database), "--yes"]) == 1
    assert "inside a Cayu project" in capsys.readouterr().err


def test_check_reports_the_database_not_the_project(project: Path) -> None:
    database = project / "data" / "cayu.db"
    _open_with_keyring(database)
    with pytest.raises(_AliasStoreKeyringRequired) as raised:
        SQLiteSessionStore(database)

    finding = keyring_required_diagnostic(raised.value, root=project)

    assert finding.code == "SESSION_STORE_ALIAS_KEYRING_REQUIRED"
    assert finding.path == "data/cayu.db"
    assert "The database is the problem, not the project's code" in finding.message
    assert "'primary'" in finding.message
    assert "cayu storage reset-aliases" in finding.hint
    assert list(finding.parameters["key_ids"]) == ["primary"]
    assert finding.parameters["first_used_at"] is not None
    # Nothing in the store changed.
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM cayu_public_authority_alias_keys"
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_cli_check_names_the_database_when_its_keyring_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV, PUBLIC_AUTHORITY_ALIAS_KEYS_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    _open_with_keyring(project / "data" / "cayu.db")
    monkeypatch.chdir(project)

    assert main(["check", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert [item["code"] for item in report["diagnostics"]] == [
        "SESSION_STORE_ALIAS_KEYRING_REQUIRED"
    ]
    assert report["diagnostics"][0]["path"] == "data/cayu.db"
