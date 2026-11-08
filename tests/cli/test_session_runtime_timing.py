"""`cayu session runtime-timing` and `GET /api/sessions/{id}/runtime-timing`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.cli import main
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.runtime._session_execution_presence import process_owner_id
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth
from cayu.storage import SQLiteSessionStore

_SESSION = "sess_timed"
_PASSWORD = "s3cret-password"


class _Echo(Tool):
    spec = ToolSpec(
        name="echo",
        description="Return ok.",
        max_terminal_payload_bytes=1024,
        input_schema={"type": "object", "properties": {}},
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="test:echo", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx, args):
        return ToolResult(content="ok")


_LONG_SESSION = "sess_long_call"
# Longer than a timing record keeps: the record holds a reference in its place.
_LONG_CALL_ID = "call-" + "x" * 600
_CALL_IDS: list[str] = []


def _response(request):
    if request.messages[-1].role.value == "tool":
        return [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    # One call per session, in the order the sessions run.
    call_id = _CALL_IDS.pop(0)
    return [
        ModelStreamEvent.tool_call(id=call_id, name="echo", arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


class _FakeCloud:
    """The Cayu Cloud reads the CLI makes before it reads the Agent."""

    def request(self, method: str, path: str, **_: object) -> dict[str, object]:
        if path == "/v1/applications":
            return {"items": [{"id": "scout-agent", "name": "Scout"}]}
        if path == "/v1/applications/scout-agent/service":
            return {"status": "running", "cayu_url": "http://127.0.0.1/"}
        if path == "/v1/applications/scout-agent/operator-credentials":
            return {
                "username": "operator",
                "password": _PASSWORD,
                "env": {"username": "CAYU_OPERATOR_USERNAME", "password": "CAYU_OPERATOR_PASSWORD"},
            }
        raise AssertionError(f"unexpected Cloud request {method} {path}")


@pytest.fixture
def served(tmp_path: Path, monkeypatch):
    """An app that ran one session with a tool round, served as a Cloud Agent."""

    app = CayuApp(session_store=SQLiteSessionStore(tmp_path / "cayu.db"), enable_logging=False)
    app.register_provider(ScriptedModelProvider(response_factory=_response), default=True)
    app.register_agent(AgentSpec(name="a", model="scripted-model"), tools=[_Echo()])

    _CALL_IDS[:] = ["call-1", _LONG_CALL_ID]

    async def run() -> None:
        for session_id in (_SESSION, _LONG_SESSION):
            async for _ in app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="a",
                    max_steps=3,
                    messages=[Message.text("user", "go")],
                )
            ):
                pass

    asyncio.run(run())
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
    server = create_server(
        app, config=ServerConfig.protected(BasicAuth(username="operator", password=_PASSWORD))
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield tmp_path / "cayu.db", client


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(["session", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_an_agent_returns_the_records_it_holds_without_private_ids(served, capsys) -> None:
    code, out, _ = _run(
        capsys, "runtime-timing", _SESSION, "--application", "scout-agent", "--json"
    )

    assert code == 0, out
    assert _PASSWORD not in out
    payload = json.loads(out)
    assert payload["source"] == {"kind": "cayu_cloud", "application": "scout-agent"}
    assert payload["scope"] == "process"
    assert payload["process"] == {"owner_id": process_owner_id()}
    assert payload["session_execution"] == {
        "state": "terminal",
        "owner_id": None,
        "in_this_process": False,
    }
    assert payload["recorder"]["enabled"] is True
    assert payload["recorder"]["recent_capacity"] == 64
    kinds = [record["kind"] for record in payload["records"]]
    assert "tool_round" in kinds and "model_step_preparation" in kinds
    starts = [record["started_at"] for record in payload["records"]]
    assert starts == sorted(starts)
    for private in ("model_step_id", "model_attempt_id", "tool_round_id", "after_tool_round_id"):
        assert private not in out
    (round_,) = [record for record in payload["records"] if record["kind"] == "tool_round"]
    (call,) = round_["calls"]
    assert call["tool_name"] == "echo"
    assert {phase["name"] for phase in round_["phases"]} >= {"execution", "round_commit"}
    # A call is named as `cayu session tools` and its tool events name it.
    tools = json.loads(_run(capsys, "tools", _SESSION, "--application", "scout-agent", "--json")[1])
    assert call["tool_call_id"] == tools["calls"][0]["tool_call_id"]
    assert any(
        record["after_tool_round"] is True
        for record in payload["records"]
        if record["kind"] == "model_step_preparation"
    )
    totals = {entry["phase"]: entry for entry in payload["phase_totals"]}
    assert totals["execution"]["records"] == 1
    assert totals["round_commit"]["store_transaction_count"] >= 1


def test_a_call_is_named_by_its_original_identity(served, capsys) -> None:
    payload = json.loads(
        _run(capsys, "runtime-timing", _LONG_SESSION, "--application", "scout-agent")[1]
    )
    tools = json.loads(
        _run(capsys, "tools", _LONG_SESSION, "--application", "scout-agent", "--json")[1]
    )

    (round_,) = [record for record in payload["records"] if record["kind"] == "tool_round"]
    assert round_["calls"][0]["tool_call_id"] == tools["calls"][0]["tool_call_id"]
    assert _LONG_CALL_ID not in json.dumps(payload)


def test_the_table_shows_records_and_phase_totals(served, capsys) -> None:
    code, out, _ = _run(
        capsys, "runtime-timing", _SESSION, "--application", "scout-agent", "--table"
    )

    assert code == 0, out
    assert f"as process {process_owner_id()} holds it in memory" in out
    assert "tool round" in out and "model step preparation" in out
    assert "Phases over these records" in out
    assert "tx: sql " in out and "lock/pool wait" in out
    assert _PASSWORD not in out


def test_limit_keeps_the_newest_records(served, capsys) -> None:
    everything = json.loads(
        _run(capsys, "runtime-timing", _SESSION, "--application", "scout-agent")[1]
    )["records"]
    newest = json.loads(
        _run(capsys, "runtime-timing", _SESSION, "--application", "scout-agent", "--limit", "1")[1]
    )

    assert newest["limit"] == 1
    assert newest["records"] == everything[-1:]


def test_a_server_url_reads_the_same_route(served, capsys, monkeypatch) -> None:
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", "Basic b3BlcmF0b3I6czNjcmV0LXBhc3N3b3Jk")

    code, out, _ = _run(
        capsys, "runtime-timing", _SESSION, "--server-url", "http://127.0.0.1/", "--json"
    )

    assert code == 0, out
    payload = json.loads(out)
    assert payload["source"] == {"kind": "server", "url": "http://127.0.0.1/"}
    assert payload["records"]
    assert "b3BlcmF0b3I6" not in out


@pytest.mark.parametrize("target", [(), ("--sqlite", "cayu.db")])
def test_a_session_store_has_no_records(served, capsys, target) -> None:
    code, out, _ = _run(capsys, "runtime-timing", _SESSION, *target, "--json")

    assert code == 1
    message = json.loads(out)["error"]["message"]
    assert "memory of the process that ran the session" in message
    assert "--server-url" in message and "--application" in message


def test_an_unknown_session_is_not_found(served, capsys) -> None:
    code, out, _ = _run(
        capsys, "runtime-timing", "missing", "--application", "scout-agent", "--json"
    )

    assert code == 1
    assert json.loads(out)["error"]["message"] == "Session not found: missing"


def test_the_route_needs_credentials_and_bounds_the_limit(served) -> None:
    _, client = served
    path = f"/api/sessions/{_SESSION}/runtime-timing"

    assert client.get(path).status_code == 401
    auth = ("operator", _PASSWORD)
    assert client.get(path, params={"limit": 0}, auth=auth).status_code == 422
    assert client.get(path, params={"limit": 1025}, auth=auth).status_code == 422
    answered = client.get(path, auth=auth)
    assert answered.status_code == 200
    assert answered.headers["cache-control"] == "private, no-store"
