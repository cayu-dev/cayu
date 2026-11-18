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


def test_members_are_added_listed_changed_and_removed(database, capsys) -> None:
    assert main(["memberships", "add", "alice", "acme", "--role", "admin"]) == 0
    assert capsys.readouterr().out == "alice is an admin of acme.\n"
    assert main(["memberships", "add", "carol", "acme"]) == 0
    assert capsys.readouterr().out == "carol is a member of acme.\n"

    status, listed = _json(capsys, "list", "--organization", "acme")
    assert status == 0
    assert listed == {
        "members": [
            {"subject": "alice", "organization": "acme", "role": "admin"},
            {"subject": "carol", "organization": "acme", "role": "member"},
        ]
    }
    status, added = _json(capsys, "add", "carol", "acme", "--role", "admin")
    assert (status, added) == (
        0,
        {"added": {"subject": "carol", "organization": "acme", "role": "admin"}},
    )
    assert main(["memberships", "list", "--subject", "carol"]) == 0
    table = capsys.readouterr().out
    assert "acme" in table and "admin" in table

    status, removed = _json(capsys, "remove", "carol", "acme")
    assert (status, removed["removed"]) == (0, True)
    assert main(["memberships", "remove", "carol", "acme"]) == 0
    assert "nothing changed" in capsys.readouterr().out
    assert main(["memberships", "list", "--organization", "acme", "--table"]) == 0
    assert "carol" not in capsys.readouterr().out
    with psycopg.connect(database) as connection:
        revision = connection.execute("SELECT revision FROM app.access_revision").fetchone()
    assert revision == (5,)


def test_the_last_admin_and_an_unknown_organization_are_refused(database, capsys) -> None:
    assert main(["memberships", "add", "alice", "acme", "--role", "admin"]) == 0
    capsys.readouterr()

    assert main(["memberships", "remove", "alice", "acme"]) == 1
    assert "at least one admin" in capsys.readouterr().err
    status, refused = _json(capsys, "add", "alice", "acme", "--role", "member")
    assert status == 1
    assert refused["error"]["code"] == "MEMBERSHIPS_COMMAND_FAILED"
    assert "at least one admin" in refused["error"]["message"]
    assert main(["memberships", "add", "bob", "initech"]) == 1
    assert "'initech' doesn't exist" in capsys.readouterr().err
    status, listed = _json(capsys, "list", "--organization", "acme")
    assert [item["subject"] for item in listed["members"]] == ["alice"]


def test_it_works_on_the_older_tables_with_departments(database, capsys) -> None:
    _create(database, older=True)
    assert main(["memberships", "add", "alice", "acme", "--role", "admin"]) == 0
    capsys.readouterr()
    status, listed = _json(capsys, "list", "--subject", "alice")
    assert (status, listed["members"]) == (
        0,
        [{"subject": "alice", "organization": "acme", "role": "admin"}],
    )


def test_it_says_when_the_tables_or_the_database_are_missing(database, capsys, monkeypatch) -> None:
    _execute(database, "DROP SCHEMA app CASCADE")
    assert main(["memberships", "list", "--organization", "acme"]) == 1
    assert "There is no table app.memberships." in capsys.readouterr().err

    monkeypatch.delenv("CAYU_DATABASE_URL")
    assert main(["memberships", "list", "--organization", "acme"]) == 1
    assert "set CAYU_DATABASE_URL" in capsys.readouterr().err


def test_knowledge_is_granted_to_work_admitted_before_the_policy_had_it(database, capsys) -> None:
    """`grant-knowledge` lists the conversations admitted without knowledge, records a grant
    for each admission only when not a dry run, and the policy then admits that work."""

    import asyncio
    from functools import partial

    from cayu import CayuApp
    from cayu.memberships import MembershipAccessPolicy, PostgresMemberships, conversation_labels
    from cayu.memberships.directory import Membership
    from cayu.sessions.records import SessionIdentity
    from cayu.sessions.requests import RunRequest
    from cayu.storage import PostgresSessionStore, PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    assert main(["memberships", "add", "alice", "acme", "--role", "admin"]) == 0
    capsys.readouterr()
    memberships = PostgresMemberships(partial(psycopg.connect, database))

    async def start(knowledge: bool) -> tuple[str, object]:
        sessions = PostgresSessionStore(database, schema_mode=SchemaMode.CREATE)
        tasks = PostgresTaskStore(database, schema_mode=SchemaMode.CREATE)
        try:
            app = CayuApp(
                session_store=sessions,
                task_store=tasks,
                resource_access_policy=MembershipAccessPolicy(memberships, knowledge=knowledge),
                enable_logging=False,
            )
            access = await app.access("alice")
            created = await access.sessions.create(
                RunRequest(
                    agent_name="assistant",
                    messages=[],
                    labels=conversation_labels(
                        subject="alice", membership=Membership("acme"), sharing="organization"
                    ),
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await app.aclose()
            return created.id, created.invocation.resource_access
        finally:
            await tasks.close()
            await sessions.close()

    old_id, old_binding = asyncio.run(start(knowledge=False))
    asyncio.run(start(knowledge=True))
    policy = MembershipAccessPolicy(memberships, knowledge=True)

    status, listed = _json(capsys, "grant-knowledge", "--dry-run")
    assert status == 0
    assert listed["dry_run"] is True
    (admission,) = listed["admissions"]
    assert admission["subject"] == "alice"
    assert admission["organizations"] == ["acme"]
    assert admission["session_ids"] == [old_id]
    assert admission["status"] == "would_grant"
    assert listed["counts"]["would_grant"] == 1
    assert asyncio.run(policy.knowledge_admission(old_binding)) is None

    status, granted = _json(capsys, "grant-knowledge")
    assert status == 0
    assert [row["status"] for row in granted["admissions"]] == ["granted"]
    assert asyncio.run(policy.knowledge_admission(old_binding)) is not None
    # The record is the audit of what was granted.
    (grant,) = asyncio.run(memberships.knowledge_grants.list())
    assert (grant.subject, grant.organizations, grant.sessions) == ("alice", ("acme",), 1)
    assert grant.binding_sha256 == admission["binding_sha256"]

    status, again = _json(capsys, "grant-knowledge")
    assert [row["status"] for row in again["admissions"]] == ["already_granted"]
    assert again["counts"]["granted"] == 0
