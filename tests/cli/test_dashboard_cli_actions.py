"""Dashboard task/knowledge mutations remain reachable through the CLI."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.server import ServerConfig, create_server
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskTypeSpec


def test_graph_http_admits_once_and_rejects_changed_content():
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        body = {
            "operation_id": "graph-one",
            "nodes": [
                {"key": "first", "type": "report", "input": {}},
                {"key": "second", "type": "report", "input": {}, "prerequisite_keys": ["first"]},
            ],
        }
        response = client.post("/api/task-graphs", json=body)
        assert response.status_code == 200, response.text
        assert client.post("/api/task-graphs", json=body).json() == response.json()
        assert len(client.get("/api/tasks").json()) == 2
        body["nodes"][1]["input"] = {"changed": True}
        assert client.post("/api/task-graphs", json=body).status_code == 409


@pytest.mark.parametrize(
    "command",
    [
        ["task", "create", "report", "--operation-id", "one"],
        ["task", "block", "one"],
        ["task", "hold", "one"],
        ["task", "resume", "one"],
        ["task", "cancel", "one", "--reason", "No longer needed"],
        ["knowledge", "keep", "one", "--operation-id", "review"],
        ["knowledge", "drop", "one"],
        ["knowledge", "drop", "one", "--operation-id", "drop-one", "--reason", "Duplicate"],
        ["knowledge", "archive", "one", "--reason", "Stale"],
    ],
)
def test_actions_use_server_api(command, monkeypatch, capsys):
    from cayu.cli._cloud_agent_api import CayuApi

    calls = []
    if command[1] == "create":
        receipt = {"task_id": "one"}
    elif command[0] == "task":
        status = {"hold": "paused", "block": "blocked", "resume": "pending"}
        receipt = {"id": "one", "status": status.get(command[1], "cancelled")}
    else:
        receipt = {
            "entry_id": "one",
            "revision": 2,
            "status": "active" if command[1] == "keep" else "archived",
        }
        if "--reason" in command:
            receipt["reason"] = command[command.index("--reason") + 1]
    monkeypatch.setattr(
        CayuApi,
        "mutate",
        lambda self, path, body, **kw: calls.append((path, body, kw)) or receipt,
        raising=False,
    )
    assert main([*command, "--server-url", "http://127.0.0.1:8123", "--json"]) == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["result"] == receipt


def test_serve_accepts_price_book_flag():
    from cayu.cli.serve import parse_serve_arguments

    assert (
        str(parse_serve_arguments(["--dev", "--price-book", "prices.json"]).price_book)
        == "prices.json"
    )


@pytest.mark.parametrize("failure", ["timeout", "starting", "invalid_json"])
def test_mutation_discovers_by_read_and_never_retries_uncertain_response(failure):
    import httpx

    from cayu.cli._cloud_agent_api import CayuApi, CayuApiError

    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            if request.url.path.startswith("/mounted"):
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"name": "Agent"})
        assert request.url.path == "/api/tasks"
        assert request.headers["Idempotency-Key"] == "kept-key"
        if failure == "timeout":
            raise httpx.ReadTimeout("secret-canary")
        if failure == "starting":
            return httpx.Response(503, headers={"retry-after": "0"})
        return httpx.Response(200, text="not json")

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        api = CayuApi("http://localhost/mounted", http=http)
        with pytest.raises(CayuApiError) as raised:
            api.mutate(
                "/tasks", {"operation_id": "kept-key"}, resource="task", operation_id="kept-key"
            )
    assert raised.value.category == "outcome_unknown"
    assert [r.method for r in requests] == ["GET", "GET", "POST"]


def test_cli_real_routes_create_hold_block_graph_and_list(tmp_path, monkeypatch, capsys):
    from cayu.cli import _control_plane_actions, task
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._task_readers import ApiTaskReader

    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )
        monkeypatch.setattr(
            task,
            "open_task_reader",
            lambda target: ApiTaskReader(
                CayuApi("http://testserver", http=client), source={"kind": "server"}
            ),
        )

        def run(*args):
            assert main(["task", *args, "--json"]) == 0
            return json.loads(capsys.readouterr().out)

        created = run("create", "report", "--operation-id", "cli-create")["result"]["task_id"]
        assert (
            run("create", "report", "--operation-id", "cli-create")["result"]["task_id"] == created
        )
        assert run("hold", created, "--reason", "Review")["result"]["status"] == "paused"
        assert run("block", created, "--reason", "Missing input")["result"]["status"] == "blocked"
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(
            json.dumps({"nodes": [{"key": "first", "type": "report", "input": {}}]})
        )
        assert run("graph-create", str(graph_file), "--operation-id", "cli-graph")["result"][
            "graph_id"
        ]
        assert len(run("list", "--server-url", "http://localhost")["tasks"]) == 2


def test_cli_cancels_resumes_and_replays_tasks_through_the_real_routes(
    monkeypatch, capsys, tmp_path
):
    """`cayu task cancel|resume|replay` reach the API's task controls."""

    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )

        def run(*args, code=0):
            assert main(["task", *args, "--json"]) == code
            return json.loads(capsys.readouterr().out)

        held = run("create", "report", "--operation-id", "held")["result"]["task_id"]
        run("hold", held, "--reason", "Review")
        # `hold` pauses "until resumed", and `resume` does.
        assert run("resume", held)["result"]["status"] == "pending"

        cancelled = run("cancel", held, "--reason", "Wrong input")["result"]
        assert cancelled["status"] == "cancelled"
        assert cancelled["error"] == {"code": "task_cancelled", "message": "Wrong input"}
        # Cancelling again is refused by the task's lifecycle, without echoing the server.
        assert "error" in run("cancel", held, code=1)

        replayed = run("replay", held, "--operation-id", "again")["result"]
        assert replayed["replayed_task_id"] == held and replayed["task_id"] != held
        # A retry sends the request it kept, not a fresh offer whose proof may have changed.
        offered = _control_plane_actions._offered_replay
        monkeypatch.setattr(
            _control_plane_actions,
            "_offered_replay",
            lambda *_args: pytest.fail("a retry must reuse the kept replay request"),
        )
        assert run("replay", held, "--operation-id", "again")["result"] == replayed
        monkeypatch.setattr(_control_plane_actions, "_offered_replay", offered)

        # A retry that can't reach the Agent keeps the request for the next retry.
        unreachable = _control_plane_actions._replay_record("http://127.0.0.1:9", held, "again")
        unreachable.parent.mkdir(parents=True, exist_ok=True)
        unreachable.write_text(
            json.dumps({"operation_id": "again", "type": "report", "source": {"task_id": held}})
        )
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi(
                "http://127.0.0.1:9", timeout_seconds=0.5, sleep=lambda _seconds: None
            ),
        )
        run("replay", held, "--operation-id", "again", code=1)
        assert unreachable.exists()
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )

        # A replay the server refuses outright isn't kept: the next try asks for a fresh offer.
        stale = run("create", "report", "--operation-id", "stale")["result"]["task_id"]
        record = _control_plane_actions._replay_record("http://testserver", stale, "stale-1")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps({"operation_id": "stale-1", "type": "report", "source": {"task_id": stale}})
        )
        run("replay", stale, "--operation-id", "stale-1", code=1)
        assert not record.exists()
        assert (
            run("replay", stale, "--operation-id", "stale-1")["result"]["replayed_task_id"] == stale
        )
        task = client.get(f"/api/tasks/{replayed['task_id']}").json()
        assert (task["type"], task["status"]) == ("report", "pending")

        # A task whose type the agent no longer declares can't be replayed, and says why.
        asyncio.run(store.create_task(TaskCreate(task_id="old", type="retired")))
        refused = run("replay", "old", "--operation-id", "old-again", code=1)
        assert "task_type_not_declared" in refused["error"]["message"]

        # Resuming a task whose prerequisite hasn't finished puts it back to wait for it.
        graph = {
            "operation_id": "chain",
            "nodes": [
                {"key": "first", "type": "report", "input": {}},
                {"key": "second", "type": "report", "input": {}, "prerequisite_keys": ["first"]},
            ],
        }
        second = client.post("/api/task-graphs", json=graph).json()["task_ids"]["second"]
        run("hold", second)
        assert run("resume", second)["result"]["status"] == "waiting_dependencies"


def test_cli_real_knowledge_review_and_existing_pending_listing(monkeypatch, capsys):
    from cayu.cli import _control_plane_actions, knowledge
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._knowledge_readers import ApiKnowledgeReader
    from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
    from cayu.knowledge.scopes import KnowledgeAccessScope
    from cayu.storage.memory import InMemoryKnowledgeStore

    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(id=name, text="Made-up proposal", status=KnowledgeStatus.PENDING)
            for name in ("keep-me", "drop-me")
        ],
        access_scope=scope,
    )
    app = CayuApp(knowledge_store=store, enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )
        monkeypatch.setattr(
            knowledge,
            "open_knowledge_reader",
            lambda target: ApiKnowledgeReader(
                CayuApi("http://testserver", http=client), source={"kind": "server"}
            ),
        )

        def run(*args):
            assert main(["knowledge", *args, "--json"]) == 0
            return json.loads(capsys.readouterr().out)

        pending = run("list", "--status", "pending", "--server-url", "http://localhost")
        assert {entry["entry_id"] for entry in pending["entries"]} == {"keep-me", "drop-me"}
        first = run("keep", "keep-me", "--operation-id", "keep-one")["result"]
        assert first["status"] == "active"
        assert run("keep", "keep-me", "--operation-id", "keep-one")["result"] == first
        assert run("drop", "drop-me")["result"]["status"] == "archived"
        assert (
            run("list", "--status", "pending", "--server-url", "http://localhost")["entries"] == []
        )


@pytest.mark.parametrize(
    ("command", "path", "body", "key"),
    [
        (["keep", "one", "--operation-id", "k"], "/knowledge/one/approve", {}, "k"),
        (
            ["keep", "one", "--operation-id", "k", "--reason", "Checked"],
            "/knowledge/one/approve",
            {"reason": "Checked"},
            "k",
        ),
        (["drop", "one"], "/knowledge/one/reject", {}, None),
        (
            ["drop", "one", "--operation-id", "d", "--reason", "Duplicate"],
            "/knowledge/one/reject",
            {"reason": "Duplicate"},
            "d",
        ),
        (
            ["archive", "one", "--reason", "Stale", "--revision", "3"],
            "/knowledge/one/retire",
            {"reason": "Stale", "expected_revision": 3},
            None,
        ),
        (
            ["archive", "one", "--reason", "Stale", "--operation-id", "a"],
            "/knowledge/one/retire",
            {"reason": "Stale"},
            "a",
        ),
    ],
)
def test_knowledge_decisions_send_reason_and_operation_id(command, path, body, key, monkeypatch):
    from cayu.cli._cloud_agent_api import CayuApi

    calls = []
    receipt = {
        "entry_id": "one",
        "revision": 4,
        "status": "active" if command[0] == "keep" else "archived",
        "reason": body.get("reason"),
    }
    monkeypatch.setattr(
        CayuApi,
        "mutate",
        lambda self, path, body, **kw: calls.append((path, body, kw)) or receipt,
        raising=False,
    )
    assert main(["knowledge", *command, "--server-url", "http://127.0.0.1:8123", "--json"]) == 0
    assert calls == [(path, body, {"resource": "knowledge", "operation_id": key})]


@pytest.mark.parametrize(
    "command",
    [
        ["archive", "one"],
        ["archive", "one", "--reason", "  "],
        ["archive", "one", "--reason", "Stale", "--revision", "0"],
        ["keep", "one"],
    ],
)
def test_knowledge_decisions_refuse_missing_or_blank_inputs(command, monkeypatch):
    from cayu.cli import _control_plane_actions

    seen = []
    monkeypatch.setattr(_control_plane_actions, "open_action_api", lambda args: seen.append(args))
    try:
        assert main(["knowledge", *command, "--json"]) != 0
    except SystemExit as exc:
        assert exc.code != 0
    assert seen == []


def test_reason_an_older_agent_ignores_is_reported(monkeypatch, capsys):
    from cayu.cli._cloud_agent_api import CayuApi

    monkeypatch.setattr(
        CayuApi,
        "mutate",
        lambda self, path, body, **kw: {"entry_id": "one", "revision": 2, "status": "archived"},
        raising=False,
    )
    assert (
        main(
            [
                "knowledge",
                "drop",
                "one",
                "--reason",
                "Duplicate",
                "--server-url",
                "http://127.0.0.1:8123",
            ]
        )
        == 0
    )
    assert "doesn't record review reasons" in capsys.readouterr().err


def test_cli_archives_a_kept_entry_with_its_reason_and_retries(tmp_path, monkeypatch, capsys):
    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
    from cayu.knowledge.scopes import KnowledgeAccessScope
    from cayu.storage import SQLiteKnowledgeStore

    scope = KnowledgeAccessScope.privileged()
    path = tmp_path / "knowledge.sqlite"
    store = SQLiteKnowledgeStore(path, access_scope=scope)
    for name in ("stale", "dup"):
        asyncio.run(
            store.create_entry(
                KnowledgeEntry(id=name, text="Made-up entry", status=KnowledgeStatus.PENDING),
                access_scope=scope,
            )
        )
    app = CayuApp(knowledge_store=store, enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )

        def run(*args):
            assert main(["knowledge", *args, "--json"]) == 0
            return json.loads(capsys.readouterr().out)

        kept = run("keep", "stale", "--operation-id", "keep-stale", "--reason", "Was right")
        assert kept["result"]["reason"] == "Was right"
        dropped = run("drop", "dup", "--operation-id", "drop-dup", "--reason", "Duplicate")
        assert dropped["result"]["status"] == "archived"
        assert run("drop", "dup", "--operation-id", "drop-dup", "--reason", "Duplicate") == (
            dropped
        )
        archive = ("archive", "stale", "--operation-id", "archive-stale", "--reason", "Stale now")
        archived = run(*archive)
        assert archived["result"] == {
            "entry_id": "stale",
            "revision": 3,
            "status": "archived",
            "reason": "Stale now",
        }
        assert archived["operation_id"] == "archive-stale"
        assert run(*archive) == archived
        assert main(["knowledge", *archive[:4], "--reason", "Other", "--json"]) == 1
        assert "HTTP 409" in json.loads(capsys.readouterr().out)["error"]["message"]
        assert client.get("/api/knowledge").json()["entries"] == []
    asyncio.run(store.close())
    # The local store shows who archived it and why.
    assert main(["knowledge", "show", "stale", "--sqlite", str(path)]) == 0
    shown = capsys.readouterr().out
    assert "archived_reason" in shown and "Stale now" in shown
    assert main(["knowledge", "show", "dup", "--sqlite", str(path)]) == 0
    shown = capsys.readouterr().out
    assert "dropped_reason" in shown and "Duplicate" in shown


def test_cli_shows_and_lists_what_it_archived_on_a_recall_scope(monkeypatch, capsys):
    """On an Agent whose knowledge scope admits only active and pending entries (the
    scaffold's), `show` and `list --status archived` read back an archived entry with its
    reason, and repeating the archive replays it."""

    from cayu.cli import _control_plane_actions, knowledge
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._knowledge_readers import ApiKnowledgeReader
    from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
    from cayu.knowledge.scopes import KnowledgeAccessScope
    from cayu.storage.memory import InMemoryKnowledgeStore

    recall = KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )
    store = InMemoryKnowledgeStore(
        [KnowledgeEntry(id="stale", text="Made-up kept entry", status=KnowledgeStatus.ACTIVE)],
        access_scope=recall,
    )
    app = CayuApp(knowledge_store=store, enable_logging=False)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )
        monkeypatch.setattr(
            knowledge,
            "open_knowledge_reader",
            lambda target: ApiKnowledgeReader(
                CayuApi("http://testserver", http=client), source={"kind": "server"}
            ),
        )

        def run(*args):
            assert main(["knowledge", *args]) == 0
            return capsys.readouterr().out

        server = ("--server-url", "http://localhost")
        archive = ("archive", "stale", "--operation-id", "archive-stale", "--reason", "Stale now")
        archived = json.loads(run(*archive, "--json"))
        assert archived["result"]["status"] == "archived"
        assert json.loads(run(*archive, "--json")) == archived
        listed = json.loads(run("list", "--status", "archived", *server, "--json"))
        (row,) = listed["entries"]
        assert (row["entry_id"], row["decision"], row["reason"]) == (
            "stale",
            "archive",
            "Stale now",
        )
        assert json.loads(run("list", *server, "--json"))["entries"] == []
        shown = run("show", "stale", *server)
        assert "archived_reason" in shown and "Stale now" in shown


@pytest.mark.parametrize("status", [401, 403, 409, 422])
def test_cli_refusal_does_not_echo_remote_body_or_credentials(status, monkeypatch, capsys):
    import httpx

    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi

    seen = []
    answers = iter(
        [
            # A body carrying the login itself, and one that isn't JSON.
            httpx.Response(status, json={"detail": "login operator:secret-canary"}),
            httpx.Response(status, text="<html>secret-canary</html>"),
        ]
    )

    def respond(request):
        seen.append(request.method)
        return (
            httpx.Response(200, json={"name": "Agent"})
            if request.method == "GET"
            else next(answers)
        )

    for _ in range(2):
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            monkeypatch.setattr(
                _control_plane_actions,
                "open_action_api",
                lambda args, http=http: CayuApi(
                    "http://localhost",
                    authorization="Basic b3BlcmF0b3I6c2VjcmV0LWNhbmFyeQ==",
                    http=http,
                ),
            )
            assert main(["task", "block", "one", "--json"]) == 1
        captured = capsys.readouterr()
        assert "secret-canary" not in captured.out + captured.err
        assert json.loads(captured.out)["error"]
    assert seen == ["GET", "POST", "GET", "POST"]


@pytest.mark.parametrize(
    ("status", "body", "category", "shown"),
    [
        (
            403,
            {"detail": "Knowledge access denied."},
            "forbidden",
            "Server refused the request (forbidden, HTTP 403): Knowledge access denied.",
        ),
        (
            403,
            {"detail": {"code": "role_forbidden", "message": "Your role can't use this."}},
            "forbidden",
            "Server refused the request (forbidden, HTTP 403): Your role can't use this. "
            "(role_forbidden)",
        ),
        (
            409,
            {"detail": "Knowledge retirement conflicts with its current revision or scope."},
            "request_rejected",
            "Server refused the request (request_rejected, HTTP 409): Knowledge retirement "
            "conflicts with its current revision or scope.",
        ),
        (
            401,
            {"detail": "Not authenticated"},
            "credentials_rejected",
            "Server refused the request (credentials_rejected, HTTP 401).",
        ),
    ],
    ids=["forbidden", "role-code", "conflict", "login"],
)
def test_cli_refusal_keeps_the_routes_own_reason_and_code(
    status, body, category, shown, monkeypatch, capsys
):
    """A 403 means the login was accepted and something else refused, so say what did."""

    import httpx

    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"name": "Agent"})
        return httpx.Response(status, json=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://localhost", http=http),
        )
        assert main(["knowledge", "archive", "one", "--reason", "Stale", "--json"]) == 1
        error = json.loads(capsys.readouterr().out)["error"]
        assert error["message"] == shown
        assert error["category"] == category
        assert error["status_code"] == status
        assert ("code" in error) is (shown.endswith("(role_forbidden)"))
        assert main(["knowledge", "archive", "one", "--reason", "Stale"]) == 1
        assert capsys.readouterr().err.strip() == f"error: {shown}"


def test_invalid_json_and_remote_http_fail_before_any_mutation(tmp_path, monkeypatch, capsys):
    from cayu.cli import _control_plane_actions

    seen = []
    monkeypatch.setattr(_control_plane_actions, "open_action_api", lambda args: seen.append(args))
    path = tmp_path / "input.json"
    path.write_text('{"secret-canary": NaN}')
    assert (
        main(
            [
                "task",
                "create",
                "report",
                "--operation-id",
                "one",
                "--input-file",
                str(path),
                "--json",
            ]
        )
        == 1
    )
    assert seen == []
    assert "secret-canary" not in capsys.readouterr().out


def test_unwritable_output_is_reported_without_dispatch(tmp_path, monkeypatch, capsys):
    from cayu.cli import _control_plane_actions

    seen = []
    monkeypatch.setattr(_control_plane_actions, "open_action_api", lambda args: seen.append(args))
    assert main(["task", "block", "one", "--output", str(tmp_path)]) == 1
    assert seen == []
    assert "could not write command output" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "purpose", "next_step"),
    [
        (["task", "create"], "Queue a declared task", "cayu task show TASK_ID"),
        (["task", "graph-create"], "Atomically queue", "cayu task show TASK_ID"),
        (["task", "block"], "Block a task", "cayu task show TASK_ID"),
        (["task", "hold"], "Pause a task", "cayu task show TASK_ID"),
        (
            ["knowledge", "keep"],
            "Keep a pending knowledge proposal",
            "cayu knowledge show ENTRY_ID",
        ),
        (
            ["knowledge", "drop"],
            "Drop a pending knowledge proposal",
            "cayu knowledge show ENTRY_ID",
        ),
        (
            ["knowledge", "archive"],
            "Archive a kept (active) knowledge entry",
            "cayu knowledge show ENTRY_ID",
        ),
    ],
)
def test_mutation_help_explains_purpose_and_inspection(command, purpose, next_step, capsys):
    with pytest.raises(SystemExit) as raised:
        main([*command, "--help"])
    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert purpose in help_text
    assert next_step in help_text


@pytest.mark.parametrize("command", ["create", "graph-create", "block", "hold"])
@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink", "input_symlink"])
def test_output_cannot_overwrite_mutation_input(tmp_path, monkeypatch, capsys, command, alias):
    from cayu.cli import _control_plane_actions

    original = b'{"nodes": [{"key": "first", "type": "report"}]}\n'
    source = tmp_path / "request.json"
    source.write_bytes(original)
    destination = source if alias == "same" else tmp_path / "output.json"
    if alias == "symlink":
        destination.symlink_to(source)
    elif alias == "hardlink":
        destination.hardlink_to(source)
    elif alias == "input_symlink":
        destination = source
        source = tmp_path / "linked-input.json"
        source.symlink_to(destination)
    calls = []
    monkeypatch.setattr(_control_plane_actions, "open_action_api", lambda args: calls.append(args))
    arguments = ["task", command]
    if command == "graph-create":
        arguments += [str(source), "--operation-id", "preserve-graph"]
    elif command == "create":
        arguments += ["report", "--input-file", str(source), "--operation-id", "preserve-task"]
    else:
        arguments += ["one", "--payload-file", str(source)]
    assert main([*arguments, "--json", "--output", str(destination)]) == 1
    assert source.read_bytes() == original
    assert destination.read_bytes() == original
    assert calls == []
    assert "input file" in capsys.readouterr().err


def test_output_preserves_explicit_cloud_context(tmp_path, monkeypatch, capsys):
    from cayu.cli import _control_plane_actions

    context = tmp_path / "context.json"
    original = b'{"made-up-context": "retain this input"}\n'
    context.write_bytes(original)
    destination = tmp_path / "output.json"
    destination.hardlink_to(context)
    calls = []
    monkeypatch.setattr(_control_plane_actions, "open_action_api", lambda args: calls.append(args))
    assert (
        main(
            [
                "knowledge",
                "keep",
                "one",
                "--operation-id",
                "keep-one",
                "--application",
                "demo",
                "--context",
                str(context),
                "--output",
                str(destination),
                "--json",
            ]
        )
        == 1
    )
    assert context.read_bytes() == original
    assert calls == []
    assert "input file" in capsys.readouterr().err


def test_archive_reaches_a_cloud_agent_through_its_api(monkeypatch, capsys):
    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi

    opened, calls = [], []
    monkeypatch.setattr(
        _control_plane_actions,
        "open_cloud_agent_api",
        lambda name, **kw: opened.append(name) or CayuApi("https://agent.example"),
    )
    monkeypatch.setattr(
        CayuApi,
        "mutate",
        lambda self, path, body, **kw: (
            calls.append((path, body, kw))
            or {"entry_id": "one", "revision": 3, "status": "archived", "reason": "Stale"}
        ),
        raising=False,
    )
    command = ["knowledge", "archive", "one", "--reason", "Stale", "--operation-id", "a"]
    assert main([*command, "--application", "cayu-scout", "--json"]) == 0
    assert opened == ["cayu-scout"]
    assert calls == [
        (
            "/knowledge/one/retire",
            {"reason": "Stale"},
            {"resource": "knowledge", "operation_id": "a"},
        )
    ]
    assert json.loads(capsys.readouterr().out)["result"]["reason"] == "Stale"
