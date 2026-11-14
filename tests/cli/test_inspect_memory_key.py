"""`cayu inspect` in a fresh checkout of a memory project, which has no evidence key."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cayu.cli import main


@pytest.fixture
def project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.delenv("CAYU_MEMORY_EVIDENCE_KEY", raising=False)
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    root = tmp_path / "project"
    monkeypatch.chdir(root)
    return root


def _inspect(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, dict, str]:
    code = main(["inspect", *arguments])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_inspect_lends_a_throwaway_key_in_a_fresh_checkout(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A fresh clone: the git-ignored key file isn't there and nothing sets the variable.
    (project / "data" / "memory-evidence.key").unlink()

    code, manifest, err = _inspect(capsys, "--json")

    assert code == 0, err
    assert manifest["agents"]
    (notice,) = manifest["diagnostics"]
    assert notice["code"] == "MEMORY_EVIDENCE_KEY_UNSET"
    assert notice["severity"] == "info"
    assert "`cayu inspect` built the application with a throwaway" in notice["message"]
    assert notice["verification_command"] == "cayu inspect --json"
    assert "Cayu Cloud" not in notice["hint"]
    assert err == ""
    # The throwaway key was only lent while the factory ran.
    assert "CAYU_MEMORY_EVIDENCE_KEY" not in os.environ
    assert not (project / "data" / "memory-evidence.key").exists()

    assert main(["inspect", "--table"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("Cayu application ")
    assert captured.err.startswith("note: CAYU_MEMORY_EVIDENCE_KEY is unset")
    assert "cayu guide diagnostics#memory-evidence-key-unset" in captured.err


def test_inspect_reads_the_task_store_with_a_lent_key(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The task store keeps no memory evidence, so its reads work with a throwaway key.
    (project / "data" / "memory-evidence.key").unlink()

    code, fenced, err = _inspect(capsys, "--fenced-tasks", "--json")
    assert code == 0, err
    assert fenced["fenced_tasks"] == []
    assert [item["code"] for item in fenced["diagnostics"]] == ["MEMORY_EVIDENCE_KEY_UNSET"]

    code, missing, err = _inspect(capsys, "--retry-key", "nightly", "--json")
    assert code == 1, err
    assert missing["error"]["code"] == "SUBJECT_NOT_FOUND"
    assert [item["code"] for item in missing["diagnostics"]] == ["MEMORY_EVIDENCE_KEY_UNSET"]

    assert main(["inspect", "--fenced-tasks", "--table"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "No fenced tasks."
    assert captured.err.startswith("note: CAYU_MEMORY_EVIDENCE_KEY is unset")


def test_inspect_uses_the_project_key_and_lends_none(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, manifest, err = _inspect(capsys, "--json")
    assert code == 0, err
    assert "diagnostics" not in manifest
    assert err == ""

    (project / "data" / "memory-evidence.key").unlink()
    monkeypatch.setenv("CAYU_MEMORY_EVIDENCE_KEY", "k" * 48)
    code, manifest, err = _inspect(capsys, "--json")
    assert code == 0, err
    assert "diagnostics" not in manifest
    assert os.environ["CAYU_MEMORY_EVIDENCE_KEY"] == "k" * 48

    assert main(["inspect", "--table"]) == 0
    assert capsys.readouterr().err == ""
