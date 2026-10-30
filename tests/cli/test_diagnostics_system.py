"""System CLI publishes the same protected deployment metadata as HTTP."""

import json

import pytest
from tests.cli.test_environment_cli import _http_bridge

from cayu.applications import CayuApp
from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi
from cayu.server import OpenAccess, mount_cayu

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize("target", ["server", "cloud"])
def test_diagnostics_system_uses_real_nested_http(monkeypatch, capsys, target):
    server = FastAPI()
    mount_cayu(server, CayuApp(), path="/nested", access=OpenAccess())
    with TestClient(server) as client, _http_bridge(client) as http:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=http, **kwargs),
        )
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.open_cloud_agent_api",
            lambda application, **kwargs: CayuApi("http://127.0.0.1/nested", http=http),
        )
        args = (
            ["--server-url", "http://127.0.0.1/nested"]
            if target == "server"
            else ["--application", "demo"]
        )
        assert main(["diagnostics", "system", *args, "--json"]) == 0
        actual = json.loads(capsys.readouterr().out)
        expected = client.get("/nested/api/system/diagnostics").json()
        assert actual["deployment"] == expected["deployment"]
        assert main(["diagnostics", "system", *args, "--table"]) == 0
        assert "embedded" in capsys.readouterr().out


def test_diagnostics_system_help_and_context_output_guard(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as result:
        main(["diagnostics", "system", "--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--application" in help_text and "--server-url" in help_text
    source = tmp_path / "context.json"
    source.write_text('{"preserved":true}')
    output = tmp_path / "output.json"
    output.symlink_to(source)
    monkeypatch.setenv("CAYU_CLOUD_CONTEXT", str(source))
    assert main(["diagnostics", "system", "--application", "demo", "--output", str(output)]) == 1
    assert source.read_text() == '{"preserved":true}'


def test_diagnostics_system_live_authenticated_loopback(monkeypatch, capsys):
    from base64 import b64encode

    from tests.cli.test_diagnostics_requests import _running_server

    from cayu.server import BasicAuth, ServerConfig, create_server

    server = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig.protected(
            BasicAuth(username="operator", password="made-up-kind-password"),
            deployment_kind="cloud",
        ),
    )
    with _running_server(server) as url:
        monkeypatch.delenv("CAYU_API_AUTHORIZATION", raising=False)
        assert main(["diagnostics", "system", "--server-url", url, "--json"]) == 1
        denied = json.loads(capsys.readouterr().out)
        assert "error" in denied and "deployment" not in denied
        authorization = "Basic " + b64encode(b"operator:made-up-kind-password").decode()
        monkeypatch.setenv("CAYU_API_AUTHORIZATION", authorization)
        assert main(["diagnostics", "system", "--server-url", url, "--json"]) == 0
        output = capsys.readouterr()
        assert json.loads(output.out)["deployment"]["kind"] == "cloud"
        assert "made-up-kind-password" not in output.out + output.err
        assert authorization not in output.out + output.err
