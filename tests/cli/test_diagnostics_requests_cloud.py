"""`cayu diagnostics requests --application` reads a Cayu Cloud Agent's request timing."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from tests.cli.test_diagnostics_requests import _summary

from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError

_PASSWORD = "made-up-operator-password"


class _FakeCloud:
    """Cloud's answers for the Agent: its service address and its Developer dashboard login."""

    def __init__(self, *, credentials_error: CloudApiError | None = None) -> None:
        self.credentials_error = credentials_error
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **_options):
        self.requests.append((method, path))
        if path == "/v1/applications/app_scout/service":
            return {"status": "running", "cayu_url": "https://scout.agents.example/cayu"}
        if path == "/v1/applications/app_scout/operator-credentials":
            if self.credentials_error is not None:
                raise self.credentials_error
            return {
                "username": "operator",
                "password": _PASSWORD,
                "env": {"username": "CAYU_OPERATOR_USERNAME", "password": "CAYU_OPERATOR_PASSWORD"},
            }
        raise AssertionError(f"unexpected Cloud request {method} {path}")


@pytest.fixture
def cloud_agent(monkeypatch):
    state: dict = {"cloud": _FakeCloud(), "agent_requests": [], "status": 200}
    real_client = httpx.Client

    def agent(request: httpx.Request) -> httpx.Response:
        state["agent_requests"].append(request)
        expected = "Basic " + base64.b64encode(f"operator:{_PASSWORD}".encode()).decode()
        if request.headers.get("authorization") != expected:
            return httpx.Response(401, json={"detail": "Not authenticated"})
        if request.url.path != "/cayu/api/diagnostics/requests":
            return httpx.Response(404, json={"detail": "Not Found"})
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"detail": "Not Found"})
        return httpx.Response(200, content=_summary(cpu_seconds=0.6).model_dump_json())

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(agent), **kwargs),
    )
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_kwargs: state["cloud"])
    monkeypatch.setattr(
        "cayu.cli.cloud.resolve_application_id",
        lambda _client, reference: "app_scout" if reference == "cayu-scout" else "missing",
    )
    return state


def test_application_reads_the_agent_with_its_dashboard_login(cloud_agent, capsys) -> None:
    command = ["diagnostics", "requests", "--application", "cayu-scout", "--since", "5m"]
    assert main([*command, "--vcpu", "0.5"]) == 0

    captured = capsys.readouterr()
    assert "Top routes by CPU:" in captured.out
    assert "GET     /api/state      29.0   280.0   350.0" in captured.out
    (request,) = cloud_agent["agent_requests"]
    assert request.url.host == "scout.agents.example"
    assert dict(request.url.params) == {"since_seconds": "300.0", "vcpu": "0.5"}
    assert ("GET", "/v1/applications/app_scout/operator-credentials") in cloud_agent[
        "cloud"
    ].requests
    assert _PASSWORD not in captured.out + captured.err

    assert main([*command, "--json", "--budget-idle-cpu", "0.01"]) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["summary"]["routes"][0]["route"] == "/api/state"
    assert document["budget"]["exceeded"] is True


def test_application_reports_a_refused_login_and_a_missing_route(cloud_agent, capsys) -> None:
    cloud_agent["cloud"] = _FakeCloud(
        credentials_error=CloudApiError(
            "forbidden", "Only the Agent's developers and admins can do this.", status_code=403
        )
    )
    assert main(["diagnostics", "requests", "--application", "cayu-scout"]) == 2
    assert "developers and admins" in capsys.readouterr().err
    assert cloud_agent["agent_requests"] == []

    cloud_agent["cloud"] = _FakeCloud()
    cloud_agent["status"] = 404
    assert main(["diagnostics", "requests", "--application", "cayu-scout"]) == 2
    error = capsys.readouterr().err
    assert "The Agent app_scout does not serve Cayu's API" in error
    assert _PASSWORD not in error


def test_application_options_are_checked(cloud_agent, capsys, tmp_path) -> None:
    with pytest.raises(SystemExit) as result:
        main(
            [
                "diagnostics",
                "requests",
                "--application",
                "cayu-scout",
                "--server-url",
                "http://127.0.0.1:8000",
            ]
        )
    assert result.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err

    # Refused before the output is opened, so the context is never truncated.
    context = tmp_path / "context.json"
    context.write_text('{"preserved": true}')
    command = ["diagnostics", "requests", "--context", str(context)]
    assert main([*command, "--output", str(context)]) == 2
    assert "--context applies only with --application" in capsys.readouterr().err
    assert context.read_text() == '{"preserved": true}'

    # The output never overwrites the Cloud login it reads.
    output = tmp_path / "summary.json"
    output.symlink_to(context)
    command = ["diagnostics", "requests", "--application", "cayu-scout"]
    assert main([*command, "--context", str(context), "--output", str(output)]) == 2
    assert context.read_text() == '{"preserved": true}'
    assert cloud_agent["agent_requests"] == []
