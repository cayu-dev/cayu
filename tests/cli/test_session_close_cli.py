"""`cayu session close` ends a waiting session locally or on a Cloud Agent."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.server.test_server_inbox import _app

from cayu import EventType, Message, RunRequest, SessionStatus
from cayu.cli import main
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth
from cayu.storage import SQLiteSessionStore

_DATABASE_ENV = "CAYU_TEST_SESSION_CLOSE_DB"


def build_app():
    """The project factory `--target` names; it opens the test's store."""

    return _app(session_store=SQLiteSessionStore(os.environ[_DATABASE_ENV]))


def _seed(database: Path, session_id: str, agent: str) -> None:
    async def run() -> None:
        app = _app(session_store=SQLiteSessionStore(database))
        try:
            async for _event in app.run(
                RunRequest(
                    agent_name=agent, session_id=session_id, messages=[Message.text("user", "go")]
                )
            ):
                pass
        finally:
            await app.aclose()

    asyncio.run(run())


def _status_and_failure(database: Path, session_id: str):
    async def read():
        store = SQLiteSessionStore(database)
        try:
            session = await store.load(session_id)
            events = await store.load_events(session_id)
        finally:
            await store.close()
        failed = [event for event in events if event.type is EventType.SESSION_FAILED]
        return session.status, failed

    return asyncio.run(read())


@pytest.fixture
def cloud_agent(tmp_path: Path, monkeypatch):
    database = tmp_path / "cayu.db"
    _seed(database, "waiting", "asker")
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
    server = create_server(
        _app(session_store=SQLiteSessionStore(database)),
        config=ServerConfig.protected(BasicAuth(username="operator", password="s3cret-password")),
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield database


def test_close_on_a_cloud_agent_fails_the_waiting_session(cloud_agent, capsys) -> None:
    argv = ["session", "close", "waiting", "--application", "bug-hunter-agent"]
    assert main([*argv, "--reason", "Stranded by an old release.", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["closed"], answer["where"], answer["status"]) == (True, "cloud", "failed")
    assert "closed" in answer["actions"]
    status, [failed] = _status_and_failure(cloud_agent, "waiting")
    assert status is SessionStatus.FAILED
    assert failed.payload["closed"] is True
    assert failed.payload["metadata"]["reason"] == "Stranded by an old release."
    assert failed.payload["metadata"]["resolved_by"]["subject"] == "operator"

    # A finished session is refused with the Agent's reason.
    assert (
        main(
            [
                "session",
                "close",
                "missing",
                "--application",
                "bug-hunter-agent",
                "--reason",
                "x",
                "--json",
            ]
        )
        == 1
    )
    refused = json.loads(capsys.readouterr().out)
    assert refused["error"]["code"] == "SESSION_CLOSE_FAILED"
    assert "not found" in refused["error"]["message"].lower()


def test_close_in_the_project_application_records_the_local_user(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    database = tmp_path / "local.db"
    _seed(database, "local-waiting", "approver")
    monkeypatch.setenv(_DATABASE_ENV, str(database))
    target = "tests.cli.test_session_close_cli:build_app"
    argv = ["session", "close", "local-waiting", "--target", target, "--as", "local:ops"]
    assert main([*argv, "--reason", "Nobody will approve this.", "--table"]) == 0
    assert "Closed session local-waiting; it is failed." in capsys.readouterr().out
    status, [failed] = _status_and_failure(database, "local-waiting")
    assert status is SessionStatus.FAILED
    assert failed.payload["metadata"]["resolved_by"]["subject"] == "local:ops"
    assert "approval_id" in failed.payload

    # Closing it again replays; a blank reason and a Cloud-only option are refused.
    assert main([*argv, "--reason", "Nobody will approve this.", "--json"]) == 0
    capsys.readouterr()
    assert main([*argv, "--reason", " ", "--json"]) == 1
    assert "blank" in json.loads(capsys.readouterr().out)["error"]["message"]
    local_with_context = ["session", "close", "local-waiting", "--context", "c.json"]
    assert main([*local_with_context, "--reason", "x", "--json"]) == 1
    assert "--application" in json.loads(capsys.readouterr().out)["error"]["message"]
