"""`cayu cloud service credentials`: the Agent's per-Agent `/cayu/` operator login."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cayu.cli import main

PASSWORD = "operator-password-canary"
CREDENTIALS = {
    "env": {"password": "CAYU_OPERATOR_PASSWORD", "username": "CAYU_OPERATOR_USERNAME"},
    "password": PASSWORD,
    "username": "operator",
}


class FakeCloud:
    def __init__(self) -> None:
        self.credentials: tuple[int, object] = (200, CREDENTIALS)
        self.paths: list[str] = []


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[FakeCloud]:
    state = FakeCloud()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.paths.append(self.path)
            if self.path.startswith("/v1/applications?"):
                status, body = 200, {"items": [{"id": "voice-agent-one", "name": "Voice Agent"}]}
            elif self.path == "/v1/applications/voice-agent-one/operator-credentials":
                status, body = state.credentials
            else:
                status, body = 404, {"detail": "Not Found"}
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


def _run(capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, object]]:
    code = main(["cloud", "service", "credentials", "--application", "Voice Agent"])
    return code, json.loads(capsys.readouterr().out)


def test_prints_the_agents_operator_login_and_records_no_evidence(
    cloud: FakeCloud,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output = _run(capsys)

    assert code == 0
    assert output == {"ok": True, "operation": "service.credentials", "result": CREDENTIALS}
    assert cloud.paths[-1] == "/v1/applications/voice-agent-one/operator-credentials"
    # The password is a credential: it reaches stdout only, never local evidence.
    evidence = tmp_path / "evidence"
    assert not evidence.exists() or all(
        PASSWORD not in path.read_text() for path in evidence.rglob("*") if path.is_file()
    )


def test_an_older_cloud_without_the_endpoint_says_so(
    cloud: FakeCloud,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cloud.credentials = (404, {"detail": "Not Found"})

    code, output = _run(capsys)

    assert code == 2
    assert output == {
        "error": {
            "category": "operator_credentials_unsupported",
            "message": "This Cayu Cloud does not support per-Agent operator credentials yet.",
        },
        "ok": False,
    }


def test_an_agent_without_credentials_yet_is_told_to_deploy_again(
    cloud: FakeCloud,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cloud.credentials = (
        409,
        {
            "detail": {
                "code": "operator_credentials_not_provisioned",
                "message": "The Agent has no operator credentials yet.",
            }
        },
    )

    code, output = _run(capsys)

    assert code == 2
    error = output["error"]
    assert isinstance(error, dict)
    assert error["code"] == "operator_credentials_not_provisioned"
    assert error["category"] == "api_request_rejected"
    assert "deploy it again with `cayu cloud deploy`" in str(error["message"])


@pytest.mark.parametrize(
    "body",
    [
        {"username": "operator"},
        {"username": "operator", "password": "", "env": CREDENTIALS["env"]},
        {"username": "operator", "password": PASSWORD, "env": None},
    ],
)
def test_a_malformed_answer_is_rejected_without_echoing_it(
    cloud: FakeCloud,
    capsys: pytest.CaptureFixture[str],
    body: dict[str, object],
) -> None:
    cloud.credentials = (200, body)

    code, output = _run(capsys)

    assert code == 2
    assert output["error"] == {
        "category": "api_response_invalid",
        "message": "Operator credentials response is invalid.",
    }
    assert PASSWORD not in json.dumps(output)


def test_an_unknown_agent_is_not_found_before_any_credential_read(
    cloud: FakeCloud,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["cloud", "service", "credentials", "--application", "someone-elses-agent"])
    output = json.loads(capsys.readouterr().out)

    assert code == 2
    assert output["error"]["category"] == "application_not_found"
    assert not any(path.endswith("/operator-credentials") for path in cloud.paths)
