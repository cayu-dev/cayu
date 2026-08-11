"""Unit tests for the shared runner secret/env helpers."""

from __future__ import annotations

import os
import stat

import pytest

from cayu.runners._secrets import DOCKER_ENV_FILE_MAX_LINE_BYTES, runner_env_file


def test_runner_env_file_writes_key_value_lines_and_cleans_up() -> None:
    captured_path = None
    with runner_env_file({"API_TOKEN": "sk-secret", "PLAIN": "x=y"}) as path:
        assert path is not None
        captured_path = path
        content = open(path, encoding="utf-8").read()  # noqa: SIM115
        # KEY=VALUE per line; a '=' inside a value is preserved (docker splits on first '=').
        assert "API_TOKEN=sk-secret\n" in content
        assert "PLAIN=x=y\n" in content
        # The file is private (owner read/write only) — no group/other access to secrets.
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode & 0o077 == 0
    # Unlinked on exit.
    assert captured_path is not None and not os.path.exists(captured_path)


def test_runner_env_file_yields_none_for_empty_env() -> None:
    with runner_env_file({}) as path:
        assert path is None


@pytest.mark.parametrize(
    "environment",
    (
        {"BAD\nNAME": "v"},
        {"BAD=NAME": "v"},
        {"INNER SPACE": "v"},
        {"INNER\tTAB": "v"},
        {"#COMMENT": "v"},
        {"\ufeffBOM": "v"},
        {"OK": "value\nwith-newline"},
        {"OK": "value\rwith-carriage-return"},
    ),
)
def test_runner_env_file_rejects_unrepresentable_environment(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="env-file"), runner_env_file(environment):
        pass


def test_runner_env_file_enforces_docker_line_size_boundary() -> None:
    key = "VALUE"
    value_at_limit = "x" * (DOCKER_ENV_FILE_MAX_LINE_BYTES - len(f"{key}="))

    with runner_env_file({key: value_at_limit}) as path:
        assert path is not None
        assert os.path.getsize(path) == DOCKER_ENV_FILE_MAX_LINE_BYTES + 1

    with (
        pytest.raises(ValueError, match="supported size"),
        runner_env_file({key: f"{value_at_limit}x"}),
    ):
        pass
