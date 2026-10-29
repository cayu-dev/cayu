"""Dashboard task/knowledge mutations remain reachable through the CLI."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.server import ServerConfig, create_server
from cayu.tasks import InMemoryTaskStore, TaskTypeSpec


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
        ["knowledge", "keep", "one", "--operation-id", "review"],
        ["knowledge", "drop", "one"],
    ],
)
def test_actions_use_server_api(command, monkeypatch, capsys):
    from cayu.cli._cloud_agent_api import CayuApi

    calls = []
    if command[1] == "create":
        receipt = {"task_id": "one"}
    elif command[0] == "task":
        receipt = {"id": "one", "status": "paused" if command[1] == "hold" else "blocked"}
    else:
        receipt = {
            "entry_id": "one",
            "revision": 2,
            "status": "active" if command[1] == "keep" else "archived",
        }
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


@pytest.mark.parametrize("status", [401, 403, 409, 422])
def test_cli_refusal_does_not_echo_remote_body_or_credentials(status, monkeypatch, capsys):
    import httpx

    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi

    seen = []

    def respond(request):
        seen.append(request.method)
        return (
            httpx.Response(200, json={"name": "Agent"})
            if request.method == "GET"
            else httpx.Response(status, json={"detail": "secret-canary"})
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://localhost", http=http),
        )
        assert main(["task", "block", "one", "--json"]) == 1
    captured = capsys.readouterr()
    assert "secret-canary" not in captured.out + captured.err
    assert json.loads(captured.out)["error"]
    assert seen == ["GET", "POST"]


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
