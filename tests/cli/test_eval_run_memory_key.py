"""`cayu eval run` in a fresh checkout of a memory project, which has no evidence key."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cayu.cli import main

_KEY = "CAYU_MEMORY_EVIDENCE_KEY"


@pytest.fixture
def project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.delenv(_KEY, raising=False)
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    root = tmp_path / "project"
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def fresh_checkout(project: Path) -> Path:
    # A fresh clone: the git-ignored key file and database aren't there.
    (project / "data" / "memory-evidence.key").unlink()
    assert not (project / "data" / "cayu.db").exists()
    return project


def _eval_run(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, dict, str]:
    code = main(["eval", "run", "--json", *arguments])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_eval_run_lends_a_throwaway_key_when_the_store_is_new(
    fresh_checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, run, err = _eval_run(capsys)

    assert code == 0, (run, err)
    assert run["status"] == "passed"
    assert run["metadata"]["cayu_memory_evidence_key"] == {"source": "throwaway"}
    assert err.startswith("note: CAYU_MEMORY_EVIDENCE_KEY is unset")
    assert "throwaway memory evidence key that covers only this run" in err
    assert "cayu guide diagnostics#memory-evidence-key-required" in err
    # The key was only lent for the run and nothing kept it.
    assert _KEY not in os.environ
    assert not (fresh_checkout / "data" / "memory-evidence.key").exists()


def test_eval_run_lends_the_same_key_to_every_worker_process(
    fresh_checkout: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Both workers build the app during admission, so each needs the key.
    code, run, err = _eval_run(
        capsys,
        "--processes",
        "2",
        "--max-concurrency",
        "2",
        "--process-directory",
        str(tmp_path / "process-run"),
    )

    assert code == 0, (run, err)
    assert run["status"] == "passed"
    assert len(run["metadata"]["cayu_process_execution"]["workers"]) >= 1
    assert run["metadata"]["cayu_memory_evidence_key"] == {"source": "throwaway"}
    assert "throwaway memory evidence key that covers only this run" in err
    assert sorted(path.name for path in (tmp_path / "process-run").glob("ready-*.json")) == [
        "ready-0.json",
        "ready-1.json",
    ]
    assert _KEY not in os.environ


def _store_a_session(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS cayu_sessions (id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO cayu_sessions VALUES ('earlier-run')")


def test_eval_run_lends_a_key_after_cayu_check_created_an_empty_store(
    fresh_checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `cayu check` borrows a key too and builds the app, which creates the database.
    assert main(["check", "--json"]) == 0
    capsys.readouterr()
    assert (fresh_checkout / "data" / "cayu.db").is_file()

    code, run, err = _eval_run(capsys)

    assert code == 0, (run, err)
    assert run["metadata"]["cayu_memory_evidence_key"] == {"source": "throwaway"}
    assert "holds no sessions yet" in err


@pytest.mark.parametrize("store", ["sessions", "unreadable", "external"])
def test_eval_run_refuses_a_throwaway_key_for_a_store_that_keeps_evidence(
    fresh_checkout: Path,
    store: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = fresh_checkout / "data" / "cayu.db"
    named = "data/cayu.db"
    if store == "sessions":
        _store_a_session(database)
    elif store == "unreadable":
        database.parent.mkdir(exist_ok=True)
        database.write_bytes(b"not a database" * 100)
    else:
        monkeypatch.setenv("CAYU_DATABASE_URL", "postgresql://cayu@db.invalid/cayu")
        named = "the database CAYU_DATABASE_URL names"

    code, document, err = _eval_run(capsys)

    assert code == 2, (document, err)
    assert document["schema_version"] == "1"
    assert document["error"]["code"] == "MEMORY_EVIDENCE_KEY_REQUIRED"
    message = document["error"]["message"]
    assert message.startswith("`cayu eval run` needs a memory evidence key")
    assert f"into {named}, which keeps it" in message
    assert "cayu guide diagnostics#memory-evidence-key-required" in message
    assert "db.invalid" not in message
    assert err == ""
    assert _KEY not in os.environ


def test_eval_run_uses_a_real_key_and_lends_none(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, run, err = _eval_run(capsys)
    assert code == 0, (run, err)
    assert run["metadata"]["cayu_memory_evidence_key"] == {"source": "project_file"}
    assert err == ""

    # A real key may use a store that holds sessions.
    (project / "data" / "memory-evidence.key").unlink()
    _store_a_session(project / "data" / "cayu.db")
    monkeypatch.setenv(_KEY, "k" * 48)
    code, run, err = _eval_run(capsys)
    assert code == 0, (run, err)
    assert run["metadata"]["cayu_memory_evidence_key"] == {"source": "environment"}
    assert err == ""
    assert os.environ[_KEY] == "k" * 48


def test_eval_run_of_a_project_without_memory_reports_no_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_KEY, raising=False)
    assert main(["new", "plain", "--dir", str(tmp_path), "--without", "memory"]) == 0
    capsys.readouterr()
    monkeypatch.chdir(tmp_path / "plain")

    code, run, err = _eval_run(capsys)

    assert code == 0, (run, err)
    assert "cayu_memory_evidence_key" not in run["metadata"]
    assert err == ""


def test_a_run_that_stores_sessions_with_a_throwaway_key_says_so(
    fresh_checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cayu.cli._eval_memory_key import eval_memory_evidence_key

    with eval_memory_evidence_key(fresh_checkout) as source:
        assert source == "throwaway"
        assert len(os.environ[_KEY]) >= 32
        _store_a_session(fresh_checkout / "data" / "cayu.db")

    err = capsys.readouterr().err
    assert "note: this run stored sessions in data/cayu.db with a throwaway memory" in err
    assert _KEY not in os.environ
