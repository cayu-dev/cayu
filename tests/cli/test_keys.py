from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    public_authority_alias_codec_from_environment,
)

dotenv = pytest.importorskip("dotenv")


def _codec_from_env_file(path: Path):
    values = dotenv.dotenv_values(path)
    return public_authority_alias_codec_from_environment(
        {name: value for name, value in values.items() if value is not None}
    )


def test_generate_prints_env_lines_the_runtime_loader_accepts(tmp_path: Path, capsys) -> None:
    assert main(["keys", "generate"]) == 0
    first = capsys.readouterr().out
    lines = first.splitlines()
    assert lines[0] == f"{PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV}=primary"
    assert lines[1].startswith(f'{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}=\'{{"primary":')
    assert len(lines) == 2

    # The printed lines parse as a .env file into a keyring with one 32-byte key.
    env_file = tmp_path / ".env"
    env_file.write_text(first)
    codec = _codec_from_env_file(env_file)
    assert codec is not None
    assert codec.keyring.active_key_id == "primary"
    assert codec.keyring.key_ids == ("primary",)
    assert len(codec.keyring.keys["primary"].get_secret_value()) == 43

    assert main(["keys", "generate"]) == 0
    assert capsys.readouterr().out != first


def test_keys_only_prints_the_json_value_for_the_chosen_key_id(capsys) -> None:
    assert main(["keys", "generate", "--key-id", "k1", "--keys-only"]) == 0
    output = capsys.readouterr().out
    keys = json.loads(output)
    assert list(keys) == ["k1"]
    codec = public_authority_alias_codec_from_environment(
        {
            PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV: "k1",
            PUBLIC_AUTHORITY_ALIAS_KEYS_ENV: output.strip(),
        }
    )
    assert codec is not None


@pytest.mark.parametrize("key_id", ["Primary", "1key", "a" * 33, ""])
def test_generate_rejects_a_key_id_the_runtime_would_refuse(key_id: str, capsys) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["keys", "generate", "--key-id", key_id])
    assert exited.value.code == 2
    assert "--key-id" in capsys.readouterr().err


def test_env_file_is_created_private_and_appended_to_existing_content(
    tmp_path: Path, capsys
) -> None:
    created = tmp_path / "new.env"
    assert main(["keys", "generate", "--env-file", str(created)]) == 0
    output = capsys.readouterr().out
    assert "Added" in output
    # Only the confirmation is printed, never the key.
    assert "{" not in output
    assert stat.S_IMODE(created.stat().st_mode) == 0o600
    assert _codec_from_env_file(created) is not None

    existing = tmp_path / ".env"
    existing.write_text("# provider\nOPENAI_API_KEY=sk-test")
    existing.chmod(0o640)
    assert main(["keys", "generate", "--env-file", str(existing), "--key-id", "k1"]) == 0
    content = existing.read_text()
    assert content.startswith("# provider\nOPENAI_API_KEY=sk-test\n")
    assert content.endswith("\n")
    assert stat.S_IMODE(existing.stat().st_mode) == 0o640
    values = dotenv.dotenv_values(existing)
    assert values["OPENAI_API_KEY"] == "sk-test"
    codec = _codec_from_env_file(existing)
    assert codec is not None and codec.keyring.active_key_id == "k1"
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env", "new.env"]


@pytest.mark.parametrize(
    "line",
    [
        f"{PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV}=old",
        f'export {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}=\'{{"old":"value"}}\'',
        f"  {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} = x",
        f"'{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}'=x",
    ],
)
def test_env_file_refuses_to_overwrite_either_variable(tmp_path: Path, line: str, capsys) -> None:
    env_file = tmp_path / ".env"
    original = f"OTHER=1\n{line}\n"
    env_file.write_text(original)

    assert main(["keys", "generate", "--env-file", str(env_file)]) == 1

    error = capsys.readouterr().err
    assert "already sets" in error
    assert "--replace" in error
    assert env_file.read_text() == original


def test_replace_overwrites_in_place_and_drops_duplicate_definitions(
    tmp_path: Path, capsys
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "A=1\r\n"
        f'{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}=\'{{"old":"value"}}\'\r\n'
        "B=2\n"
        f"export {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}=stale\n"
    )

    assert main(["keys", "generate", "--env-file", str(env_file), "--replace"]) == 0

    assert "Replaced" in capsys.readouterr().out
    rows = env_file.read_bytes().decode().splitlines(keepends=True)
    assert rows[0] == "A=1\r\n"
    assert rows[1].startswith(f'export {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}=\'{{"primary":')
    assert rows[1].endswith("'\r\n")
    assert rows[2] == "B=2\n"
    # The variable the file didn't set is appended.
    assert rows[3] == f"{PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV}=primary\n"
    assert len(rows) == 4
    assert _codec_from_env_file(env_file) is not None


def test_replace_needs_env_file(capsys) -> None:
    assert main(["keys", "generate", "--replace"]) == 2
    assert "--replace needs --env-file" in capsys.readouterr().err


def test_env_file_and_keys_only_are_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["keys", "generate", "--keys-only", "--env-file", str(tmp_path / ".env")])
    assert exited.value.code == 2


def test_env_file_refuses_a_multiline_value_it_would_split(tmp_path: Path, capsys) -> None:
    env_file = tmp_path / ".env"
    original = f'{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}=\'{{\n  "old": "value"\n}}\'\n'
    env_file.write_text(original)

    assert main(["keys", "generate", "--env-file", str(env_file), "--replace"]) == 1

    assert "spans several lines" in capsys.readouterr().err
    assert env_file.read_text() == original


def test_env_file_symlink_updates_its_target(tmp_path: Path) -> None:
    target = tmp_path / "shared.env"
    target.write_text("A=1\n")
    link = tmp_path / ".env"
    os.symlink(target, link)

    assert main(["keys", "generate", "--env-file", str(link)]) == 0

    assert link.is_symlink()
    assert _codec_from_env_file(target) is not None


@pytest.mark.parametrize("quote", ["'", '"'])
def test_replace_refuses_escaped_quotes_in_multiline_values(tmp_path: Path, capsys, quote) -> None:
    env_file = tmp_path / ".env"
    original = (
        f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}={quote}{{\\{quote}primary\\{quote}:\nvalue}}{quote}\n"
    )
    env_file.write_text(original)
    assert main(["keys", "generate", "--env-file", str(env_file), "--replace"]) == 1
    assert "spans several lines" in capsys.readouterr().err
    assert env_file.read_text() == original


def test_replace_preserves_export_for_child_process(tmp_path: Path) -> None:
    import subprocess

    env_file = tmp_path / ".env"
    env_file.write_text(
        f"export {PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV}=old\n"
        f"'{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}'=old\n"
        f"export {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}=duplicate\n"
    )
    assert main(["keys", "generate", "--env-file", str(env_file), "--replace"]) == 0
    result = subprocess.run(
        ["sh", "-c", '. "$1"; env', "sh", str(env_file)],
        env={},
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert public_authority_alias_codec_from_environment(values) is not None


@pytest.mark.skipif(os.name != "posix", reason="POSIX file ownership")
@pytest.mark.parametrize("permitted", [True, False])
def test_replacement_preserves_ownership_or_leaves_original(
    tmp_path: Path,
    monkeypatch,
    permitted: bool,
    capsys,
) -> None:
    from types import SimpleNamespace

    from cayu.cli import keys

    target = tmp_path / ".env"
    target.write_text("SECRET=old\n")
    target.chmod(0o640)
    original = target.stat()
    operations = []
    real_chmod = os.chmod

    def fchown(fd, uid, gid):
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
        operations.append((uid, gid))
        if not permitted:
            raise PermissionError("ownership cannot be preserved")

    real_fstat = os.fstat
    monkeypatch.setattr(
        keys.os,
        "fstat",
        lambda fd: SimpleNamespace(
            st_uid=original.st_uid,
            st_gid=original.st_gid + 1,
            st_mode=real_fstat(fd).st_mode,
        ),
    )
    monkeypatch.setattr(keys.os, "fchown", fchown)

    def chmod(path, mode):
        assert operations == [(original.st_uid, original.st_gid)]
        real_chmod(path, mode)

    monkeypatch.setattr(keys.os, "chmod", chmod)
    result = main(["keys", "generate", "--env-file", str(target)])
    assert result == (0 if permitted else 1)
    assert operations == [(original.st_uid, original.st_gid)]
    if permitted:
        assert _codec_from_env_file(target) is not None
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
    else:
        assert target.read_text() == "SECRET=old\n"
        assert "ownership cannot be preserved" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [target]
