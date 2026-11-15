"""`cayu memberships` against an application's membership tables on PostgreSQL."""

from __future__ import annotations

import json

import pytest

psycopg = pytest.importorskip("psycopg")

from tests.memberships_support import _create, _execute  # noqa: E402

from cayu.cli import main  # noqa: E402

pytestmark = pytest.mark.postgres


@pytest.fixture
def database(postgres_dsn: str, postgres_url: str, monkeypatch, tmp_path) -> str:
    _create(postgres_dsn)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_DATABASE_URL", postgres_url)
    return postgres_dsn


def _json(capsys, *argv: str) -> tuple[int, dict]:
    status = main(["memberships", *argv, "--json"])
    return status, json.loads(capsys.readouterr().out)


def test_members_are_added_listed_moved_and_removed(database, capsys) -> None:
    assert (
        main(
            [
                "memberships",
                "add",
                "alice",
                "acme",
                "--department",
                "acme/finance",
                "--role",
                "admin",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "alice is an admin of acme/finance.\n"
    assert main(["memberships", "add", "carol", "acme"]) == 0
    capsys.readouterr()

    status, listed = _json(capsys, "list", "--organization", "acme")
    assert status == 0
    assert listed == {
        "members": [
            {
                "subject": "alice",
                "organization": "acme",
                "department": "acme/finance",
                "role": "admin",
            },
            {"subject": "carol", "organization": "acme", "department": None, "role": "member"},
        ]
    }
    assert main(["memberships", "add", "carol", "acme", "--department", "acme/legal"]) == 0
    capsys.readouterr()
    assert main(["memberships", "list", "--subject", "carol"]) == 0
    table = capsys.readouterr().out
    assert "acme/legal" in table and "member" in table

    status, removed = _json(capsys, "remove", "carol", "acme")
    assert (status, removed["removed"]) == (0, True)
    assert main(["memberships", "remove", "carol", "acme"]) == 0
    assert "nothing changed" in capsys.readouterr().out
    assert main(["memberships", "list", "--organization", "acme", "--table"]) == 0
    assert "carol" not in capsys.readouterr().out
    with psycopg.connect(database) as connection:
        revision = connection.execute("SELECT revision FROM app.access_revision").fetchone()
    assert revision == (5,)


def test_the_last_admin_and_a_foreign_department_are_refused(database, capsys) -> None:
    assert main(["memberships", "add", "alice", "acme", "--role", "admin"]) == 0
    capsys.readouterr()

    assert main(["memberships", "remove", "alice", "acme"]) == 1
    assert "at least one admin" in capsys.readouterr().err
    status, refused = _json(capsys, "add", "alice", "acme", "--role", "member")
    assert status == 1
    assert refused["error"]["code"] == "MEMBERSHIPS_COMMAND_FAILED"
    assert "at least one admin" in refused["error"]["message"]
    assert main(["memberships", "add", "bob", "acme", "--department", "globex/support"]) == 1
    assert "isn't a department of 'acme'" in capsys.readouterr().err
    status, listed = _json(capsys, "list", "--organization", "acme")
    assert [item["subject"] for item in listed["members"]] == ["alice"]


def test_it_says_when_the_tables_or_the_database_are_missing(database, capsys, monkeypatch) -> None:
    _execute(database, "DROP SCHEMA app CASCADE")
    assert main(["memberships", "list", "--organization", "acme"]) == 1
    assert "There is no table app.memberships." in capsys.readouterr().err

    monkeypatch.delenv("CAYU_DATABASE_URL")
    assert main(["memberships", "list", "--organization", "acme"]) == 1
    assert "set CAYU_DATABASE_URL" in capsys.readouterr().err
