"""`cayu cloud service check`: Cayu Cloud runs the Agent's declared operator check."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cayu.cli import main

_PATH = "/v1/applications/voice-agent-one/service/check"
PASSED = {
    "application_id": "voice-agent-one",
    "deployment_id": "dep_123",
    "method": "POST",
    "path": "/operator-check",
    "outcome": "passed",
    "http_status": 200,
    "duration_ms": 412,
    "checked_at": "2026-10-31T17:00:00+00:00",
    "body": {
        "kind": "json",
        "ok": True,
        "summary": "2 of 2 checks passed.",
        "checks": [
            {"name": "ai_gateway", "ok": True, "detail": "Answered in 210 ms"},
            {"name": "workflow", "ok": True, "detail": None},
        ],
        "checks_truncated": False,
    },
    "hint": None,
}
STARTING = (
    503,
    {
        "detail": {
            "code": "agent_starting",
            "message": "The Agent is starting, so its check wasn't sent. Try again shortly.",
            "retry_after": 1,
        }
    },
)


class FakeCloud:
    def __init__(self) -> None:
        self.answers: list[tuple[int, object]] = [(200, PASSED)]
        self.posts: list[str] = []


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeCloud]:
    state = FakeCloud()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, status: int, body: object) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path.startswith("/v1/applications?"):
                self._answer(200, {"items": [{"id": "voice-agent-one", "name": "Voice Agent"}]})
            else:
                self._answer(404, {"detail": "Not Found"})

        def do_POST(self) -> None:
            state.posts.append(self.path)
            if self.path != _PATH:
                self._answer(404, {"detail": "Not Found"})
                return
            status, body = state.answers.pop(0) if len(state.answers) > 1 else state.answers[0]
            self._answer(status, body)

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
    # A waking Agent is asked again after Cloud's Retry-After; tests don't wait for it.
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


def _check(*options: str) -> int:
    return main(["cloud", "service", "check", "--application", "Voice Agent", *options])


def test_prints_what_the_agent_answered(cloud: FakeCloud, capsys: pytest.CaptureFixture) -> None:
    assert _check() == 0

    out = capsys.readouterr().out
    assert "Check passed: POST /operator-check answered HTTP 200 in 412 ms" in out
    assert "Release: dep_123" in out
    assert "Summary: 2 of 2 checks passed." in out
    assert "ok   ai_gateway  Answered in 210 ms" in out
    assert "ok   workflow" in out
    assert cloud.posts == [_PATH]


def test_json_prints_the_envelope(cloud: FakeCloud, capsys: pytest.CaptureFixture) -> None:
    assert _check("--json") == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["operation"] == "service.check"
    result = output["result"]
    assert (result["schema_version"], result["outcome"], result["http_status"]) == (
        1,
        "passed",
        200,
    )
    assert result["body"]["checks"][0]["name"] == "ai_gateway"


def test_a_failed_check_exits_2_with_the_agents_answer(
    cloud: FakeCloud, capsys: pytest.CaptureFixture
) -> None:
    cloud.answers = [
        (
            200,
            {
                **PASSED,
                "outcome": "failed",
                "http_status": 401,
                "body": {"kind": "empty"},
                "hint": "The Agent refused Cloud's operator login at its check route.",
            },
        )
    ]

    assert _check("--json") == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "operator_check_failed"
    assert error["result"]["http_status"] == 401

    assert _check() == 2
    err = capsys.readouterr().err
    assert "Check FAILED: POST /operator-check answered HTTP 401" in err
    assert "Hint: The Agent refused Cloud's operator login" in err


def test_a_waking_agent_is_asked_again_until_it_answers(cloud: FakeCloud) -> None:
    cloud.answers = [STARTING, STARTING, (200, PASSED)]

    assert _check("--json") == 0
    assert len(cloud.posts) == 3


def test_an_agent_still_starting_after_the_wait_says_so(
    cloud: FakeCloud, capsys: pytest.CaptureFixture
) -> None:
    cloud.answers = [STARTING]

    assert _check("--json", "--wait-seconds", "0") == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "agent_starting"
    assert len(cloud.posts) == 1


def test_clouds_refusal_is_reported_with_its_code_and_never_retried(
    cloud: FakeCloud, capsys: pytest.CaptureFixture
) -> None:
    message = "The serving release declares no [operator_check] in its cayu-cloud.toml."
    cloud.answers = [(409, {"detail": {"code": "operator_check_not_declared", "message": message}})]

    assert _check("--json") == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "operator_check_not_declared"
    assert message in error["message"]
    assert len(cloud.posts) == 1

    cloud.answers = [(504, {"detail": {"code": "operator_check_timeout", "message": "slow"}})]
    assert _check("--json") == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "operator_check_timeout"
    assert len(cloud.posts) == 2


def test_an_older_cloud_without_the_route_says_so(
    cloud: FakeCloud, capsys: pytest.CaptureFixture
) -> None:
    cloud.answers = [(404, {"detail": "Not Found"})]

    assert _check("--json") == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "service_check_unsupported"


def test_the_agents_text_cannot_drive_the_terminal(
    cloud: FakeCloud, capsys: pytest.CaptureFixture
) -> None:
    body = {
        "kind": "text",
        "excerpt": "ok\x1b]52;c;YXR0YWNrZXI=\x07\nline two\x1b[2K",
        "truncated": False,
    }
    cloud.answers = [(200, {**PASSED, "outcome": "failed", "http_status": 503, "body": body})]

    assert _check() == 2

    err = capsys.readouterr().err
    assert "\x1b" not in err
    assert "\x07" not in err
    assert "  ok?]52;c;YXR0YWNrZXI=?" in err
    assert "  line two?[2K" in err
