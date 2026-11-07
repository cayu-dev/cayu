"""`cayu session show` reports execution presence and `session list --owner-lost` finds
sessions whose owner is gone, from a local store, a Cayu server and a Cloud Agent."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import cayu.sessions._process_liveness as processes
from cayu import CayuApp, Message, RunRequest
from cayu.cli import main
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.storage import SQLiteSessionStore

_LOST = "sess_lost"
_LIVE = "sess_live"
_IDLE = "sess_idle"
_LOST_OWNER = "a" * 32
_LIVE_OWNER = "b" * 32


class _FakeCloud:
    """The Cayu Cloud reads the CLI makes before it reads the Agent."""

    def request(self, method: str, path: str, **_: object) -> dict[str, object]:
        if path == "/v1/applications":
            return {"items": [{"id": "scout-agent", "name": "Scout"}]}
        if path == "/v1/applications/scout-agent/service":
            return {"status": "running", "cayu_url": "http://127.0.0.1/cayu/"}
        if path == "/v1/applications/scout-agent/operator-credentials":
            return {
                "username": "operator",
                "password": "s3cret-password",
                "env": {"username": "CAYU_OPERATOR_USERNAME", "password": "CAYU_OPERATOR_PASSWORD"},
            }
        raise AssertionError(f"unexpected Cloud request {method} {path}")


def _seed(path: Path) -> None:
    """A running session whose owner's lease expired, one still held, and one not started."""

    async def run() -> None:
        store = SQLiteSessionStore(path)
        try:
            for session_id in (_LOST, _LIVE, _IDLE):
                await store.create(
                    RunRequest(
                        agent_name="scout",
                        session_id=session_id,
                        messages=[Message.text("user", "go")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="fake-model"),
                )
            for session_id, owner_id, lease_seconds in (
                (_LOST, _LOST_OWNER, 0.2),
                (_LIVE, _LIVE_OWNER, 600),
            ):
                await store.transition_status(
                    session_id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                )
                claimed = await store._claim_session_execution(
                    session_id,
                    token=uuid4().hex,
                    owner_id=owner_id,
                    owner_kind="task_worker",
                    owner_label="scout-worker",
                    # Another host's process: only the lease says whether it is alive.
                    process_identity=processes.ProcessIdentity(
                        host_boot_id="another-host", pid=os.getpid()
                    ),
                    lease_seconds=lease_seconds,
                )
                assert claimed is not None
            await asyncio.sleep(0.25)
            assert (await store.inspect_session_execution(_LOST)).state == "owner_lost"
            assert (await store.inspect_session_execution(_LIVE)).state == "executing"
        finally:
            await store.close()

    asyncio.run(run())


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"
    _seed(path)
    return path


@pytest.fixture
def served(database: Path, monkeypatch):
    """The same store served by a protected Cayu server, reachable as a Cloud Agent."""

    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
    server = create_server(
        CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False),
        config=ServerConfig.protected(BasicAuth(username="operator", password="s3cret-password")),
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield client


def _json(capsys, *argv: str) -> dict:
    code = main(["session", *argv, "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured
    return json.loads(captured.out)


def _targets(database: Path) -> dict[str, tuple[str, ...]]:
    return {
        "store": ("--sqlite", str(database)),
        "server": ("--server-url", "http://127.0.0.1/cayu"),
        "agent": ("--application", "scout-agent"),
    }


@pytest.fixture
def server_authorization(monkeypatch) -> None:
    # Basic operator:s3cret-password, as `cayu session ... --server-url` reads it.
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", "Basic b3BlcmF0b3I6czNjcmV0LXBhc3N3b3Jk")


@pytest.mark.parametrize("target", ["store", "server", "agent"])
def test_show_reports_who_holds_the_session_and_whether_it_is_gone(
    database, served, server_authorization, capsys, target
) -> None:
    argv = _targets(database)[target]

    lost = _json(capsys, "show", _LOST, *argv)["execution"]
    live = _json(capsys, "show", _LIVE, *argv)["execution"]
    idle = _json(capsys, "show", _IDLE, *argv)["execution"]

    assert lost["state"] == "owner_lost"
    assert lost["owner_id"] == _LOST_OWNER
    assert lost["owner_kind"] == "task_worker"
    assert lost["owner_label"] == "scout-worker"
    assert lost["local_owner"] is False
    assert lost["lease_expires_at"] is not None and lost["heartbeat_at"] is not None
    assert live["state"] == "executing" and live["owner_id"] == _LIVE_OWNER
    assert idle["state"] == "idle" and idle["owner_id"] is None


@pytest.mark.parametrize("target", ["store", "server", "agent"])
def test_list_owner_lost_finds_only_sessions_whose_owner_is_gone(
    database, served, server_authorization, capsys, target
) -> None:
    argv = _targets(database)[target]

    listed = _json(capsys, "list", "--owner-lost", *argv)

    assert [row["id"] for row in listed["sessions"]] == [_LOST]
    row = listed["sessions"][0]
    assert row["status"] == "running"
    assert row["execution"]["state"] == "owner_lost"
    assert row["execution"]["owner_id"] == _LOST_OWNER
    # Both running sessions were checked; the pending one can't have lost an owner.
    assert listed["scanned"] == 2
    assert listed["has_more"] is False


def test_list_owner_lost_stops_at_the_limit(database, capsys, monkeypatch) -> None:
    async def lost(self, session_id):
        return {"session_id": session_id, "state": "owner_lost", "owner_id": "c" * 32}

    monkeypatch.setattr("cayu.cli._session_reader.StoreSessionReader.execution", lost)

    listed = _json(capsys, "list", "--owner-lost", "--limit", "1", "--sqlite", str(database))

    assert len(listed["sessions"]) == 1
    assert listed["scanned"] == 1
    assert listed["has_more"] is True


def test_list_owner_lost_checks_the_newest_sessions_first_whatever_their_status(
    database, capsys, monkeypatch
) -> None:
    async def interrupt_a_newer_session() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="scout",
                    session_id="sess_newer",
                    messages=[Message.text("user", "go")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status("sess_newer", SessionStatus.RUNNING)
            await store.update_status("sess_newer", SessionStatus.INTERRUPTING)
        finally:
            await store.close()

    asyncio.run(interrupt_a_newer_session())

    async def lost(self, session_id):
        return {"session_id": session_id, "state": "owner_lost", "owner_id": "c" * 32}

    monkeypatch.setattr("cayu.cli._session_reader.StoreSessionReader.execution", lost)

    listed = _json(capsys, "list", "--owner-lost", "--limit", "1", "--sqlite", str(database))
    everything = _json(capsys, "list", "--owner-lost", "--sqlite", str(database))

    assert [row["id"] for row in listed["sessions"]] == ["sess_newer"]
    assert listed["has_more"] is True
    assert [row["id"] for row in everything["sessions"]] == ["sess_newer", _LIVE, _LOST]
    assert everything["scanned"] == 3


def test_list_owner_lost_table_shows_the_owner(database, capsys) -> None:
    assert main(["session", "list", "--owner-lost", "--sqlite", str(database), "--table"]) == 0
    out = capsys.readouterr().out

    assert _LOST in out and _LOST_OWNER in out and "task_worker" in out
    assert _LIVE not in out


def test_show_table_lists_the_execution_fields(database, capsys) -> None:
    assert main(["session", "show", _LOST, "--sqlite", str(database), "--table"]) == 0
    out = capsys.readouterr().out

    assert "execution.state" in out and "owner_lost" in out
    assert "execution.lease_expires_at" in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--status", "completed"), "can't have lost its owner"),
        (("--cursor", "abc"), "--owner-lost can't be combined"),
        (("--overview",), "--owner-lost can't be combined"),
    ],
)
def test_list_owner_lost_refuses_what_it_cant_honor(database, capsys, extra, message) -> None:
    assert main(["session", "list", "--owner-lost", *extra, "--sqlite", str(database)]) == 1
    assert message in json.loads(capsys.readouterr().out)["error"]["message"]


def test_an_agent_that_predates_execution_presence(database, served, capsys, monkeypatch) -> None:
    from cayu.cli._cloud_agent_api import CayuApi

    original = CayuApi.get

    def without_execution(self, path, **kwargs):
        body = original(self, path, **kwargs)
        if path.endswith("/state"):
            body = {key: value for key, value in body.items() if key != "execution"}
        return body

    monkeypatch.setattr(CayuApi, "get", without_execution)

    shown = _json(capsys, "show", _LOST, "--application", "scout-agent")
    assert shown["execution"] is None
    assert main(["session", "list", "--owner-lost", "--application", "scout-agent"]) == 1
    assert "newer cayu release" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_server_url_keeps_its_authorization_out_of_errors(
    database, served, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", "Basic d3Jvbmc6czNjcmV0LW90aGVy")

    code = main(["session", "show", _LOST, "--server-url", "http://127.0.0.1/cayu", "--json"])
    out = capsys.readouterr().out

    assert code == 1
    assert "s3cret" not in out and "d3Jvbmc6czNjcmV0LW90aGVy" not in out
    assert json.loads(out)["error"]["message"]


def test_server_url_refuses_credentials_in_the_url(capsys) -> None:
    code = main(
        ["session", "show", _LOST, "--server-url", "https://user:pw@cayu.example.com", "--json"]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "pw@" not in out
    assert "canonical" in json.loads(out)["error"]["message"]


def test_lease_expiry_is_reported_in_time(database, capsys) -> None:
    shown = _json(capsys, "show", _LIVE, "--sqlite", str(database))["execution"]
    expires = shown["lease_expires_at"]

    # The live lease runs for minutes from the claim, in UTC.
    assert datetime.fromisoformat(expires).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(expires) > datetime.now(UTC)
