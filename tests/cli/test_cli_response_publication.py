"""Authenticated API output never publishes active credentials or corrupts receipts."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cayu.cli import main


@pytest.fixture
def authenticated_host():
    state = {"credential": "", "answer": {}, "seen": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, mutation):
            observed = self.headers.get("Authorization")
            state["seen"].append((self.command, observed))
            if observed != state["credential"]:
                self.send_response(401)
                self.end_headers()
                return
            if mutation:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
            body = state["answer"] if mutation or self.path != "/api/app" else {"name": "Demo"}
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.answer(False)

        def do_POST(self):
            self.answer(True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def credential(form):
    password = "arbitrary-credential-canary-8753"
    encoded = base64.b64encode(f"operator:{password}".encode()).decode()
    if form == "basic_header":
        return "Basic " + encoded, "Basic " + encoded
    if form == "basic_spaced":
        return "Basic  " + encoded, encoded
    if form == "basic_payload":
        return "Basic " + encoded, encoded
    if form == "basic_password":
        return "Basic " + encoded, password
    return "Bearer opaque-unstructured-canary-9864", "opaque-unstructured-canary-9864"


@pytest.mark.parametrize(
    "form", ["basic_header", "basic_payload", "basic_password", "basic_spaced", "bearer_raw"]
)
@pytest.mark.parametrize("where", ["value", "nested", "key"])
@pytest.mark.parametrize("output", ["json", "table", "file"])
def test_mutation_echo_is_unknown_without_credential_publication(
    authenticated_host, monkeypatch, tmp_path, capsys, form, where, output
):
    state, url = authenticated_host
    header, secret = credential(form)
    state["credential"] = header
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", header)
    state["answer"] = {"task_id": "task-safe"}
    if where == "value":
        state["answer"]["message"] = secret
    elif where == "nested":
        state["answer"]["details"] = {"items": [{"value": "prefix " + secret + " suffix"}]}
    else:
        state["answer"][secret] = "ordinary value"
    output_file = tmp_path / "result.json"
    options = ["--table"] if output == "table" else ["--json"]
    if output == "file":
        options += ["--output", str(output_file)]
    code = main(
        ["task", "create", "report", "--operation-id", "retained", "--server-url", url, *options]
    )
    captured = capsys.readouterr()
    emitted = (
        captured.out + captured.err + (output_file.read_text() if output_file.exists() else "")
    )
    assert code == 1
    assert "unknown" in emitted
    assert secret not in emitted and header not in emitted
    assert [method for method, _ in state["seen"]] == ["GET", "POST"]
    assert all(observed == header for _, observed in state["seen"])


@pytest.mark.parametrize(
    "command", [["task", "show", "one"], ["knowledge", "list", "--status", "pending"]]
)
@pytest.mark.parametrize("where", ["value", "key"])
def test_server_url_readers_reject_echoed_credentials(
    authenticated_host, monkeypatch, capsys, command, where
):
    state, url = authenticated_host
    header, secret = credential("bearer_raw")
    state["credential"] = header
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", header)
    state["answer"] = {
        "id": "one",
        "type": "report",
        "status": "pending",
        "entries": [],
        "truncated": False,
    }
    state["answer"].update({"title": secret} if where == "value" else {secret: "safe"})
    assert main([*command, "--server-url", url, "--json"]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert state["seen"] and all(observed == header for _, observed in state["seen"])


@pytest.mark.parametrize("answer", [None, [], {}, {"task_id": True}, {"task_id": ""}])
def test_invalid_success_receipt_does_not_claim_success_or_retry(
    authenticated_host, monkeypatch, capsys, answer
):
    state, url = authenticated_host
    header, _ = credential("bearer_raw")
    state.update(credential=header, answer=answer)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", header)
    assert (
        main(
            [
                "task",
                "create",
                "report",
                "--operation-id",
                "retained",
                "--server-url",
                url,
                "--json",
            ]
        )
        == 1
    )
    assert "unknown" in capsys.readouterr().out
    assert [method for method, _ in state["seen"]] == ["GET", "POST"]


def test_safe_authenticated_receipt_is_published(authenticated_host, monkeypatch, capsys):
    state, url = authenticated_host
    header, _ = credential("basic_payload")
    state.update(credential=header, answer={"task_id": "task-safe"})
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", header)
    assert (
        main(
            [
                "task",
                "create",
                "report",
                "--operation-id",
                "retained",
                "--server-url",
                url,
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["result"] == {"task_id": "task-safe"}
    assert [method for method, _ in state["seen"]] == ["GET", "POST"]


@pytest.mark.parametrize("output", ["json", "table", "file"])
def test_graph_structural_keys_survive_real_cli_admission_replay_and_use(
    tmp_path, monkeypatch, capsys, output
):
    from fastapi.testclient import TestClient

    from cayu import CayuApp
    from cayu.cli import _control_plane_actions
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.server import ServerConfig, create_server
    from cayu.tasks import InMemoryTaskStore, TaskTypeSpec

    keys = ["token", "password", "auth", "api_secret", "ordinary"]
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(
        json.dumps({"nodes": [{"key": key, "type": "report", "input": {}} for key in keys]})
    )
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _control_plane_actions,
            "open_action_api",
            lambda args: CayuApi("http://testserver", http=client),
        )
        output_file = tmp_path / "result.json"
        options = ["--table"] if output == "table" else ["--json"]
        if output == "file":
            options += ["--output", str(output_file)]
        arguments = [
            "task",
            "graph-create",
            str(graph_file),
            "--operation-id",
            "structural-keys",
            *options,
        ]
        assert main(arguments) == 0
        captured = capsys.readouterr()
        receipt = json.loads(output_file.read_text() if output == "file" else captured.out)[
            "result"
        ]
        assert set(receipt["task_ids"]) == set(keys)
        for task_id in receipt["task_ids"].values():
            assert client.get(f"/api/tasks/{task_id}").status_code == 200
        assert main(arguments) == 0
        captured = capsys.readouterr()
        assert (
            json.loads(output_file.read_text() if output == "file" else captured.out)["result"]
            == receipt
        )
        assert main(["task", "hold", receipt["task_ids"]["token"], "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["result"]["status"] == "paused"


@pytest.mark.parametrize(
    ("command", "answer"),
    [
        (["task", "hold", "one"], {"id": "other", "status": "paused"}),
        (["task", "block", "one"], {"id": "one", "status": "pending"}),
        (
            ["knowledge", "keep", "one", "--operation-id", "keep-one"],
            {"entry_id": "one", "revision": True, "status": "active"},
        ),
        (["knowledge", "drop", "one"], {"entry_id": "one", "revision": 2, "status": "active"}),
    ],
)
def test_unconfirmed_transition_receipts_are_unknown(
    authenticated_host, monkeypatch, capsys, command, answer
):
    state, url = authenticated_host
    header, _ = credential("bearer_raw")
    state.update(credential=header, answer=answer)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", header)
    assert main([*command, "--server-url", url, "--json"]) == 1
    assert "unknown" in capsys.readouterr().out
    assert [method for method, _ in state["seen"]] == ["GET", "POST"]


@pytest.mark.parametrize(
    "mapping",
    [{}, {"other": "task-one"}, {"token": True}, {"token": ""}, {"token": "Bearer secret"}],
)
def test_unusable_graph_mapping_does_not_report_success(
    authenticated_host, monkeypatch, tmp_path, capsys, mapping
):
    state, url = authenticated_host
    header, _ = credential("bearer_raw")
    state.update(credential=header, answer={"graph_id": "graph-one", "task_ids": mapping})
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", header)
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps({"nodes": [{"key": "token", "type": "report"}]}))
    assert (
        main(
            [
                "task",
                "graph-create",
                str(graph_file),
                "--operation-id",
                "graph-one",
                "--server-url",
                url,
                "--json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "unknown" in captured.out
    assert "Bearer secret" not in captured.out + captured.err
    assert [method for method, _ in state["seen"]] == ["GET", "POST"]


@pytest.mark.parametrize("kind", ["success", "detail", "invalid_json"])
def test_api_decodes_each_response_once_before_publication(monkeypatch, kind):
    import httpx

    from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound

    decoded = []
    original_json = httpx.Response.json

    def decode(response, **kwargs):
        decoded.append(response)
        return original_json(response, **kwargs)

    monkeypatch.setattr(httpx.Response, "json", decode)
    header, _ = credential("basic_payload")

    def respond(request):
        assert request.headers["Authorization"] == header
        if kind == "detail":
            return httpx.Response(404, json={"detail": "Unknown task"})
        if kind == "invalid_json":
            return httpx.Response(200, text="not json")
        return httpx.Response(200, json={"task_id": "safe"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        api = CayuApi("http://localhost", authorization=header, http=http)
        if kind == "success":
            assert api.get("/tasks/one", resource="task") == {"task_id": "safe"}
        elif kind == "detail":
            with pytest.raises(CayuApiNotFound, match="Unknown task"):
                api.get("/tasks/one", resource="task")
        else:
            with pytest.raises(CayuApiError) as raised:
                api.get("/tasks/one", resource="task")
            assert raised.value.category == "not_served"
    assert len(decoded) == 1
