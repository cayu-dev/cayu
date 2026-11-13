"""`cayu inbox --as SUBJECT`: a person's own inbox, through the resource access policy."""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

pytest.importorskip("fastapi")

from tests.server.test_scoped_inbox import _app

import cayu.cli.inbox as inbox_cli
from cayu import CayuApp, Message, RunRequest
from cayu.cli import main
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.storage import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor

_KEYRING = PublicAuthorityAliasKeyring(
    active_key_id="test",
    keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
)


def _paused(
    monkeypatch,
    tmp_path,
    *,
    agent: str = "approver",
    session_id: str = "alice-1",
    secret: str | None = None,
) -> None:
    """alice's paused session in a SQLite store, and the CLI opening a fresh app over it.

    With ``secret``, the CLI's app redacts it, so a session ID containing it is
    listed by its public alias.
    """

    path = tmp_path / "sessions.sqlite"
    aliased: dict = {}
    if secret is not None:
        aliased = {"public_authority_alias_keyring": _KEYRING}

    def store() -> SQLiteSessionStore:
        if secret is None:
            return SQLiteSessionStore(path)
        return SQLiteSessionStore(
            path, public_authority_alias_codec=PublicAuthorityAliasCodec(_KEYRING)
        )

    async def pause() -> None:
        sessions = store()
        app, *_rest = _app(session_store=sessions, **aliased)
        alice = await app.access("alice")
        async for _event in alice.run(
            RunRequest(
                agent_name=agent,
                session_id=session_id,
                messages=[Message.text("user", "go")],
                labels={"org": "acme"},
            )
        ):
            pass
        await app.aclose()
        await sessions.close()

    asyncio.run(pause())

    def build(_target, **_kwargs) -> CayuApp:
        redacting = {} if secret is None else {"secret_redactor": SecretRedactor(secret)}
        return _app(session_store=store(), **aliased, **redacting)[0]

    monkeypatch.setattr(
        inbox_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(
            root=tmp_path, target=target or "app:build", service_target=None
        ),
    )
    monkeypatch.setattr(inbox_cli, "project_context", lambda _root: nullcontext())
    monkeypatch.setattr(inbox_cli, "build_project_app", build)


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_a_developer_sees_and_decides_what_alice_does(monkeypatch, tmp_path, capsys) -> None:
    _paused(monkeypatch, tmp_path)

    assert main(["inbox", "list", "--as", "bob", "--json"]) == 0
    assert _json(capsys)["items"] == []
    assert main(["inbox", "list", "--as", "alice", "--json"]) == 0
    [item] = _json(capsys)["items"]
    assert item["kind"] == "approval"

    assert main(["inbox", "approve", item["id"], "--as", "bob", "--json"]) == 1
    capsys.readouterr()
    assert main(["inbox", "approve", item["id"], "--as", "alice", "--json"]) == 0
    decided = _json(capsys)
    assert decided["item"]["settlement"]["decided_by"]["subject"] == "alice"
    assert decided["session_status"] == "completed"


def test_as_cant_be_combined_with_a_reviewer(capsys) -> None:
    assert main(["inbox", "list", "--as", "alice", "--reviewer", "ops", "--json"]) == 1
    assert "--as" in _json(capsys)["error"]["message"]


def test_as_follows_a_session_named_by_its_public_alias(monkeypatch, tmp_path, capsys) -> None:
    # Started before its ID's secret was configured, the session is listed by an alias.
    secret = "inbox-as-secret-canary"
    _paused(monkeypatch, tmp_path, session_id=f"alice-{secret}", secret=secret)
    assert main(["inbox", "list", "--as", "alice", "--json"]) == 0
    [item] = _json(capsys)["items"]
    assert secret not in item["session_id"]

    assert main(["inbox", "approve", item["id"], "--as", "alice", "--json"]) == 0
    decided = _json(capsys)
    assert decided["item"]["settlement"]["decided_by"]["subject"] == "alice"
    assert decided["session_status"] == "completed"
