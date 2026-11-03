"""`cayu session ... --application NAME` reads a Cloud Agent through its own Cayu API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.core.test_tool_completion import FinalTool

import cayu.sessions._call_inspection as call_inspection
from cayu import AgentSpec, CayuApp, Message, ResumeRequest, RunRequest
from cayu.cli import main
from cayu.context import StructuredOutputSpec
from cayu.evals.testing import ScriptedModelProvider, scripted_structured_output
from cayu.events import Event, EventType
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth
from cayu.sessions.records import SessionIdentity
from cayu.storage import SQLiteSessionStore
from cayu.tools import AlwaysRequireApprovalToolPolicy

_USAGE = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
_SESSION = "sess_cloud"


class _FakeCloud:
    """The three Cayu Cloud reads the CLI makes before it reads the Agent."""

    def __init__(self, *, cayu_url: str | None = "http://127.0.0.1/cayu/") -> None:
        self.cayu_url = cayu_url
        self.requests: list[str] = []

    def request(self, method: str, path: str, **_: object) -> dict[str, object]:
        self.requests.append(f"{method} {path}")
        if path == "/v1/applications":
            return {"items": [{"id": "bug-hunter-agent", "name": "Bug Hunter"}]}
        if path == "/v1/applications/bug-hunter-agent/service":
            return {"status": "running", "cayu_url": self.cayu_url}
        if path == "/v1/applications/bug-hunter-agent/operator-credentials":
            return {
                "username": "operator",
                "password": "s3cret-password",
                "env": {"username": "CAYU_OPERATOR_USERNAME", "password": "CAYU_OPERATOR_PASSWORD"},
            }
        raise AssertionError(f"unexpected Cloud request {method} {path}")


def _seed(path: Path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(path)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1"),
                        ModelStreamEvent.completed(
                            {"finish_reason": "tool_calls", "usage": _USAGE}
                        ),
                    ],
                    [
                        ModelStreamEvent.text_delta("Done"),
                        ModelStreamEvent.completed({"finish_reason": "stop", "usage": _USAGE}),
                    ],
                ],
                name="fixture",
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[FinalTool()])
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="support",
                    session_id=_SESSION,
                    messages=[Message.text("user", "Help")],
                    labels={"team": "docs"},
                    max_steps=4,
                )
            ):
                pass
        finally:
            await store.close()

    asyncio.run(run())


def _seed_ledger(path: Path) -> None:
    """A session whose budget reservations no model call claims."""

    async def run() -> None:
        store = SQLiteSessionStore(path)
        try:
            await store.create(
                RunRequest(
                    agent_name="analyst",
                    session_id="sess_ledger",
                    messages=[Message.text("user", "analyze")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            for event_type, payload in (
                (
                    EventType.BUDGET_RESERVED,
                    {
                        "accepted": True,
                        "reservation_id": "res-1",
                        "currency": "USD",
                        "requested": "0.5",
                    },
                ),
                (
                    EventType.MODEL_COMPLETED,
                    {
                        "transcript_cursor": 1,
                        "usage_metrics": {"input_tokens": 4, "output_tokens": 1},
                    },
                ),
                (EventType.BUDGET_RESERVATION_RELEASED, {"reservation_id": "res-1"}),
                (
                    EventType.BUDGET_RESERVED,
                    {
                        "accepted": True,
                        "reservation_id": "res-2",
                        "currency": "USD",
                        "requested": "0.7",
                    },
                ),
            ):
                await store.append_event(
                    "sess_ledger", Event(type=event_type, session_id="sess_ledger", payload=payload)
                )
            # A call that never ran, recorded with its raw arguments before write-time
            # quarantine existed.
            await store.create(
                RunRequest(
                    agent_name="analyst",
                    session_id="sess_legacy",
                    messages=[Message.text("user", "look up")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            await store.append_event(
                "sess_legacy",
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_legacy",
                    tool_name="lookup",
                    payload={
                        "tool_call_id": "c9",
                        "model_step_id": "mstep_" + "9" * 32,
                        "model_attempt_id": "matt_" + "9" * 32,
                        "tool_round_id": "tround_" + "9" * 32,
                        "arguments": {"patient": "Jane Roe", "token": "s3cret-token"},
                    },
                ),
            )
        finally:
            await store.close()

    asyncio.run(run())


def _seed_outputs(path: Path) -> None:
    """A session answered twice with structured output, and one paused for an approval."""

    async def run() -> None:
        store = SQLiteSessionStore(path)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    list(scripted_structured_output({"verdict": "bug", "score": 0.9})),
                    list(scripted_structured_output({"verdict": "not a bug", "score": 0.2})),
                    [
                        ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1"),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                ],
                name="fixture",
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="judge", model="scripted-model"))
        app.register_agent(
            AgentSpec(name="gated", model="scripted-model"),
            tools=[FinalTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["ask_customer"]),
        )
        spec = StructuredOutputSpec(
            name="Verdict",
            json_schema={
                "type": "object",
                "properties": {"verdict": {"type": "string"}, "score": {"type": "number"}},
                "required": ["verdict", "score"],
                "additionalProperties": False,
            },
        )
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="judge",
                    session_id="sess_answer",
                    messages=[Message.text("user", "Is this a bug?")],
                    structured_output=spec,
                )
            ):
                pass
            async for _ in app.resume(
                ResumeRequest(
                    session_id="sess_answer", messages=[Message.text("user", "Look again.")]
                )
            ):
                pass
            async for _ in app.run(
                RunRequest(
                    agent_name="gated",
                    session_id="sess_paused",
                    messages=[Message.text("user", "Ask the customer.")],
                )
            ):
                pass
        finally:
            await app.aclose()
            await store.close()

    asyncio.run(run())


@pytest.fixture
def agent(tmp_path: Path, monkeypatch):
    """A seeded store, the same store served as a Cloud Agent, and a signed-in CLI."""

    database = tmp_path / "cayu.db"
    _seed(database)
    _seed_ledger(database)
    _seed_outputs(database)
    cloud = _FakeCloud()
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: cloud)
    server = create_server(
        CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False),
        config=ServerConfig.protected(BasicAuth(username="operator", password="s3cret-password")),
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        seen: list = []
        client.event_hooks["request"].append(seen.append)
        # Each command closes the client it opened; keep this shared one for the next.
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield database, cloud, seen


def _json(capsys, *argv: str) -> dict:
    assert main(["session", *argv, "--json"]) == 0, capsys.readouterr()
    return json.loads(capsys.readouterr().out)


def _both(capsys, database: Path, *argv: str) -> tuple[dict, dict]:
    local = _json(capsys, *argv, "--sqlite", str(database))
    remote = _json(capsys, *argv, "--application", "bug-hunter-agent")
    return local, remote


@pytest.mark.parametrize(
    "argv",
    [
        ("usage", _SESSION),
        ("usage", _SESSION, "--after-sequence", "6", "--limit", "1"),
        ("transcript", _SESSION, "--sizes", "--include-content"),
        ("transcript", _SESSION, "--offset", "1", "--limit", "1"),
    ],
)
def test_cloud_agent_reads_match_the_local_store(agent, capsys, argv) -> None:
    database, _, _ = agent

    local, remote = _both(capsys, database, *argv)

    assert remote == local
    assert remote.get("calls") or remote.get("messages")
    if argv == ("usage", _SESSION):
        assert [call["finish_reason"] for call in remote["calls"]] == ["tool_calls", "stop"]


def test_cloud_usage_pairs_reservations_on_the_agent_without_publishing_them(
    agent, capsys, monkeypatch
) -> None:
    database, _, _ = agent

    local, remote = _both(capsys, database, "usage", "sess_ledger")

    assert [row["reservation_id"] for row in local["unmatched_ledger"]] == ["res-1", "res-2"]
    assert remote["unmatched_ledger"] == [
        {**row, "reservation_id": None} for row in local["unmatched_ledger"]
    ]
    assert [row["outcome"] for row in remote["unmatched_ledger"]] == ["released", "open"]
    assert {k: v for k, v in remote.items() if k != "unmatched_ledger"} == {
        k: v for k, v in local.items() if k != "unmatched_ledger"
    }

    # The Agent applies the same retained-event bound and says how to narrow the read.
    monkeypatch.setattr(call_inspection, "_MAX_COLLECTED_EVENT_RECORDS", 0)
    assert main(["session", "usage", "sess_ledger", "--application", "bug-hunter-agent"]) == 1
    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "rejected" in message and "--after-sequence" in message


def test_cloud_events_are_the_agents_public_view_of_the_same_records(agent, capsys) -> None:
    database, _, _ = agent

    def visible(rows: list[dict]) -> list[dict]:
        # The Agent publishes record-scoped aliases for private ids, so payload sizes differ.
        return [{k: v for k, v in row.items() if not k.startswith("payload_")} for row in rows]

    for argv in (
        ("events", _SESSION),
        ("events", _SESSION, "--type", "tool.call.started,model.completed"),
        ("events", _SESSION, "--limit", "3", "--after-sequence", "2"),
    ):
        local, remote = _both(capsys, database, *argv)
        assert visible(remote["events"]) == visible(local["events"]), argv
        assert {k: v for k, v in remote.items() if k != "events"} == {
            k: v for k, v in local.items() if k != "events"
        }
    local, remote = _both(capsys, database, "events", _SESSION, "--type", "model.completed")
    assert [row["type"] for row in remote["events"]] == ["model.completed"] * 2

    local, remote = _both(capsys, database, "interactions", _SESSION)
    ids = {"start_event_id", "terminal_event_id"}
    assert [{k: v for k, v in row.items() if k not in ids} for row in remote["interactions"]] == [
        {k: v for k, v in row.items() if k not in ids} for row in local["interactions"]
    ]


def test_cloud_tool_calls_pair_on_the_agent_and_name_calls_publicly(agent, capsys) -> None:
    database, _, _ = agent

    local, remote = _both(capsys, database, "tools", _SESSION)

    private = ("tool_call_id", "model_step_id", "model_attempt_id", "tool_round_id")
    assert [{k: v for k, v in row.items() if k not in private} for row in remote["calls"]] == [
        {k: v for k, v in row.items() if k not in private} for row in local["calls"]
    ]
    assert remote["calls"][0]["status"] == "success"
    assert {k: remote["calls"][0][k] for k in private[1:]} == dict.fromkeys(private[1:])
    # The call's public name is the correlation its tool events carry.
    events = _json(
        capsys,
        "events",
        _SESSION,
        "--type",
        "tool.call.completed",
        "--include-payload",
        "65536",
        "--application",
        "bug-hunter-agent",
    )
    payload = json.loads(events["events"][0]["payload_preview"])
    assert remote["calls"][0]["tool_call_id"] == payload["tool_call_correlation_id"]


def test_cloud_session_list_and_show_report_what_the_agent_api_exposes(agent, capsys) -> None:
    database, cloud, seen = agent

    local, remote = _both(capsys, database, "list", "--label", "team=docs")
    hidden = {"last_activity_at": None, "run_epoch": None}
    assert remote["sessions"] == [{**row, **hidden} for row in local["sessions"]]
    assert remote["total_count"] == local["total_count"] == 1

    local, remote = _both(capsys, database, "show", _SESSION)
    for section in ("usage", "provider_operation", "terminal_failure", "pending_action"):
        assert remote[section] == local[section], section
    assert remote["activity"] == {**local["activity"], "model_calls_with_usage": None}
    assert remote["transcript"]["message_count"] == local["transcript"]["message_count"]
    assert remote["events"]["event_count"] == local["events"]["event_count"]
    assert remote["session"] == {**local["session"], "run_epoch": None}
    assert remote["budget"] is None
    assert remote["timing"] == local["timing"]
    assert remote["timing"]["model_calls"] == 2

    # The CLI signed in once per command and read the Agent where `cayu serve` mounts the
    # API, after the embedded `/cayu/api` layout answered 404, with the Agent's login.
    assert cloud.requests.count("GET /v1/applications/bug-hunter-agent/operator-credentials") == 2
    paths = [request.url.path for request in seen]
    assert paths[0] == "/cayu/api/sessions" and paths[1] == "/api/sessions"
    assert all(request.headers["authorization"].startswith("Basic ") for request in seen)


def test_cloud_show_without_timing_from_an_older_agent(agent, capsys, monkeypatch) -> None:
    from cayu.cli import _agent_session_reader

    original = _agent_session_reader.AgentSessionReader._session_get

    def older_agent(self, session_id, *resource, params=None):
        if resource == ("timing",):
            raise _agent_session_reader._unsupported("/sessions/x/timing")
        return original(self, session_id, *resource, params=params)

    monkeypatch.setattr(_agent_session_reader.AgentSessionReader, "_session_get", older_agent)

    assert _json(capsys, "show", _SESSION, "--application", "bug-hunter-agent")["timing"] is None
    assert main(["session", "show", _SESSION, "--application", "bug-hunter-agent", "--table"]) == 0
    assert "doesn't report session timing" in capsys.readouterr().out


def test_cloud_reads_name_missing_sessions_and_never_print_the_login(agent, capsys) -> None:
    assert main(["session", "show", "missing", "--application", "bug-hunter-agent"]) == 1
    captured = capsys.readouterr()
    assert "Session not found: missing" in captured.out + captured.err
    assert "s3cret-password" not in captured.out + captured.err


def test_cloud_target_rejects_local_targets_and_snapshots(agent, capsys) -> None:
    database, _, _ = agent
    with pytest.raises(SystemExit):
        main(["session", "list", "--application", "a", "--sqlite", str(database)])
    with pytest.raises(SystemExit):
        main(["session", "snapshots", _SESSION, "--application", "bug-hunter-agent"])
    assert main(["session", "list", "--context", "ctx.json", "--sqlite", str(database)]) == 1
    captured = capsys.readouterr()
    assert "--context selects the Cayu Cloud" in captured.out + captured.err


def test_cloud_agent_without_cayu_address_is_named(monkeypatch, capsys) -> None:
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud(cayu_url=None))

    assert main(["session", "list", "--application", "bug-hunter-agent"]) == 1
    captured = capsys.readouterr()
    assert "serves no Cayu address" in captured.out + captured.err


def test_cloud_rollup_and_creation_window_match_the_local_store(agent, capsys) -> None:
    database, _, seen = agent

    for argv in (("rollup",), ("rollup", "--by", "label:team"), ("rollup", "--max-sessions", "1")):
        local, remote = _both(capsys, database, *argv)
        assert remote == local, argv
    assert any(request.url.path == "/api/sessions/rollup" for request in seen)

    local, remote = _both(capsys, database, "list", "--since", "2000-01-01")
    assert [row["id"] for row in remote["sessions"]] == [row["id"] for row in local["sessions"]]
    local, remote = _both(capsys, database, "list", "--until", "2000-01-01")
    assert remote["sessions"] == local["sessions"] == []


def test_cloud_tool_calls_quarantine_arguments_recorded_before_the_call_ran(agent, capsys) -> None:
    database, _, _ = agent

    local, remote = _both(capsys, database, "tools", "sess_legacy")

    assert "Jane Roe" in local["calls"][0]["argument_summary"]
    assert (remote["calls"][0]["arguments_state"], remote["calls"][0]["argument_summary"]) == (
        "quarantined",
        "[quarantined]",
    )
    assert remote["calls"][0]["status"] == local["calls"][0]["status"] == "running"


def test_server_disclosure_redacts_arguments_before_bounding_them() -> None:
    from cayu.sessions._call_inspection import ArgumentDisclosure, tool_inspection_record
    from cayu.sessions.records import EventRecord

    secret = "SECRETVALUE123"
    record = EventRecord(
        sequence=1,
        event=Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="s",
            payload={"arguments_state": "finalized", "arguments": {"q": "x" * 240 + secret}},
        ),
    )

    def redact(value):
        return json.loads(json.dumps(value).replace(secret, "[REDACTED]"))

    local = tool_inspection_record(record).event.payload["_argument_summary"]
    server = tool_inspection_record(record, ArgumentDisclosure(redact=redact)).event.payload[
        "_argument_summary"
    ]
    # Bounding first would leave a prefix of the secret that an exact-match redactor misses.
    assert "SECRET" in local
    assert "SECRET" not in server and len(server.encode()) <= 256


def test_session_output_prints_the_latest_interactions_structured_output(agent, capsys) -> None:
    database, _, _ = agent

    local, remote = _both(capsys, database, "output", "sess_answer")

    assert remote == local
    assert (local["name"], local["output"]) == ("Verdict", {"verdict": "not a bug", "score": 0.2})
    assert local["session_status"] == "completed"
    interactions = _json(capsys, "interactions", "sess_answer", "--sqlite", str(database))
    first, latest = interactions["interactions"][1], interactions["interactions"][0]
    assert local["interaction_id"] == latest["interaction_id"]

    # An earlier interaction's answer is read by naming it.
    local = _json(
        capsys,
        "output",
        "sess_answer",
        "--interaction-id",
        first["interaction_id"],
        "--sqlite",
        str(database),
    )
    assert local["output"] == {"verdict": "bug", "score": 0.9}

    assert main(["session", "output", "sess_answer", "--sqlite", str(database), "--table"]) == 0
    table = capsys.readouterr().out
    assert "Verdict" in table and '"verdict": "not a bug"' in table


@pytest.mark.parametrize(
    ("session_id", "code", "exit_code", "message"),
    [
        (
            _SESSION,
            "NO_STRUCTURED_OUTPUT",
            3,
            "Session sess_cloud has no structured output from its latest interaction; "
            "it is completed.",
        ),
        (
            "sess_paused",
            "SESSION_NOT_FINISHED",
            4,
            "Session sess_paused hasn't finished: it is interrupted and its latest interaction "
            "is paused, waiting for an approval or an answer.",
        ),
        ("sess_ledger", "SESSION_NOT_FINISHED", 4, "Session sess_ledger hasn't finished: it is"),
    ],
)
def test_session_output_says_when_there_is_none_or_the_session_hasnt_finished(
    agent, capsys, session_id, code, exit_code, message
) -> None:
    database, _, _ = agent

    for target in (("--sqlite", str(database)), ("--application", "bug-hunter-agent")):
        assert main(["session", "output", session_id, *target, "--json"]) == exit_code
        error = json.loads(capsys.readouterr().out)["error"]
        assert error["code"] == code, target
        assert error["message"].startswith(message), target

    assert (
        main(["session", "output", session_id, "--sqlite", str(database), "--table"]) == exit_code
    )
    assert capsys.readouterr().err.startswith(f"error: {message}")


def test_session_output_escapes_a_hostile_agents_status_for_people(
    agent, capsys, monkeypatch
) -> None:
    from cayu.cli._agent_session_reader import AgentSessionReader

    async def hostile(self, session_id: str) -> str:
        return "completed\x1b]52;c;ZWNobyBwd25lZA==\x07\x1b[2J"

    monkeypatch.setattr(AgentSessionReader, "session_status", hostile)

    code = main(["session", "output", _SESSION, "--application", "bug-hunter-agent", "--table"])

    err = capsys.readouterr().err
    assert code == 3
    assert "\x1b" not in err and "\x07" not in err
    assert "it is completed\\x1b]52;c;" in err
