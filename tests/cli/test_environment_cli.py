"""Live environment CLI catalog reads share the HTTP contract."""

import json

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient
from tests.core.test_cloud_reporting import _app

from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.server import ServerConfig, create_server


def _http_bridge(client: TestClient) -> httpx.Client:
    # Starlette may use httpx2; keep CayuApi on its actual httpx transport contract.
    def request(request: httpx.Request) -> httpx.Response:
        response = client.request(
            request.method, str(request.url), content=request.content, headers=request.headers
        )
        return httpx.Response(
            response.status_code, content=response.content, headers=dict(response.headers)
        )

    return httpx.Client(transport=httpx.MockTransport(request))


@pytest.mark.parametrize("name", ["dev", "dev/team é%", ".", ".."])
def test_environment_cli_reads_same_http_data(monkeypatch, capsys, name):
    app = _app()
    app.register_environment(Environment(EnvironmentSpec(name=name)))
    with (
        TestClient(create_server(app, config=ServerConfig.local_development())) as client,
        _http_bridge(client) as http,
    ):
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=http, **kwargs),
        )
        assert main(["environment", "list", "--server-url", "http://127.0.0.1", "--json"]) == 0
        body = json.loads(capsys.readouterr().out)
        assert body["environments"][0]["name"] == name
        assert body["environments"][0]["session_count"] == "0"
        assert (
            main(["environment", "show", name, "--server-url", "http://127.0.0.1", "--table"]) == 0
        )
        assert name in capsys.readouterr().out


def test_environment_cli_help_requires_server_or_cloud(capsys):
    for command in ("list", "show"):
        with pytest.raises(SystemExit) as error:
            main(["environment", command, "--help"])
        assert error.value.code == 0
        output = capsys.readouterr().out
        assert "--server-url" in output and "--application" in output
        assert "--sqlite" not in output and "--postgres" not in output
    with pytest.raises(SystemExit) as error:
        main(["environment", "list"])
    assert error.value.code == 2


def test_environment_cli_cloud_and_custom_unknown_activity(monkeypatch, capsys):
    from cayu.sessions.memory import InMemorySessionStore

    class Unsupported(InMemorySessionStore):
        supports_environment_activity = False

    app = _app(Unsupported())
    app.register_environment(Environment(EnvironmentSpec(name="dev")))
    with (
        TestClient(create_server(app, config=ServerConfig.local_development())) as client,
        _http_bridge(client) as http,
    ):

        def cloud(application, *, context, timeout_seconds):
            assert application == "demo"
            return CayuApi("http://127.0.0.1", http=http, timeout_seconds=timeout_seconds)

        monkeypatch.setattr("cayu.cli._cloud_agent_api.open_cloud_agent_api", cloud)
        assert main(["environment", "show", "dev", "--application", "demo", "--table"]) == 0
        assert "Unavailable" in capsys.readouterr().out


def test_environment_cli_protects_implicit_context_before_output(monkeypatch, tmp_path, capsys):
    context = tmp_path / "context.json"
    context.write_text('{"private":"preserved"}')
    output = tmp_path / "output.json"
    output.hardlink_to(context)
    monkeypatch.setenv("CAYU_CLOUD_CONTEXT", str(context))
    assert main(["environment", "list", "--application", "demo", "--output", str(output)]) == 1
    assert context.read_text() == '{"private":"preserved"}'
    assert "output must not refer" in capsys.readouterr().err


def test_environment_cli_rejects_incomplete_catalog(monkeypatch, capsys):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "environments": [{"name": "dev", "artifact_store_id": "files"}],
                    "total_count": 1,
                },
            )
        )
    ) as http:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=http, **kwargs),
        )
        assert main(["environment", "list", "--server-url", "http://127.0.0.1", "--json"]) == 1
    body = json.loads(capsys.readouterr().out)
    assert "error" in body and "environments" not in body


def test_environment_cli_live_loopback(capsys):
    import socket
    import threading
    import time

    import uvicorn

    app = _app()
    app.register_environment(Environment(EnvironmentSpec(name="dev")))
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(app, config=ServerConfig.local_development()), log_level="error"
        )
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started
            assert (
                main(["environment", "list", "--server-url", f"http://127.0.0.1:{port}", "--json"])
                == 0
            )
            body = json.loads(capsys.readouterr().out)
            assert body["environments"][0]["session_count"] == "0"
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            assert not worker.is_alive()


@pytest.mark.parametrize("command", ["task", "knowledge"])
def test_environment_parser_preserves_sibling_store_targets(command):
    from cayu.cli import _build_parser

    parsed = _build_parser().parse_args([command, "list", "--sqlite", "sessions.sqlite"])
    assert parsed.sqlite == "sessions.sqlite"
