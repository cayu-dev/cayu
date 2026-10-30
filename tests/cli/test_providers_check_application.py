"""`cayu providers check --application` runs the check on a Cloud Agent's own server."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelProviderError, ModelStreamEvent
from cayu.runtime._model_check import _MODEL_CHECK_PROMPT
from cayu.server import AuthenticatedAccess, BasicAuth, ServerConfig, create_server

_ANSWER = "AGENT-PROBE-ANSWER"
_PROVIDER_TEXT = "Incorrect API key provided: sk-live-AGENT-PROBE-CREDENTIAL"
_OPERATOR_PASSWORD = "operator-password"
# What the Agent's server answers: (method, path under its mount, body, headers) to
# (status, JSON body, extra headers).
AgentAnswer = Callable[[str, str, bytes, dict[str, str]], tuple[int, Any, dict[str, str]]]


class _RaisingProvider(ScriptedModelProvider):
    async def stream(self, request):
        raise ModelProviderError(
            _PROVIDER_TEXT, provider=self.name, status_code=401, error_code="invalid_api_key"
        )
        yield


def _agent_app() -> CayuApp:
    def answer(request):
        return [
            ModelStreamEvent.text_delta(_ANSWER),
            ModelStreamEvent.completed(
                {"finish_reason": "stop", "usage": {"input_tokens": 9, "output_tokens": 2}}
            ),
        ]

    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(response_factory=answer), model_patterns=["probe-*"], default=True
    )
    app.register_provider(_RaisingProvider([], name="failing"))
    return app


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A fake Cayu Cloud whose Agent `bug-hunter` serves Cayu at `/cayu/`.

    ``state["agent"]`` answers the Agent's requests; ``state["login"]`` is the Developer
    dashboard password Cloud hands out; ``state["agent_requests"]`` records the requests.
    """

    state: dict[str, Any] = {"agent": None, "login": _OPERATOR_PASSWORD, "agent_requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle(b"")

        def do_POST(self) -> None:
            self._handle(self.rfile.read(int(self.headers.get("content-length") or 0)))

        def _handle(self, body: bytes) -> None:
            path = urlsplit(self.path).path
            origin = f"http://{self.headers['Host']}"
            cloud_answers = {
                "/v1/applications": {"items": [{"id": "bug-hunter", "name": "Bug Hunter"}]},
                "/v1/applications/bug-hunter/service": {
                    "status": "running",
                    "cayu_url": f"{origin}/cayu/",
                },
                "/v1/applications/bug-hunter/operator-credentials": {
                    "username": "operator",
                    "password": state["login"],
                    "env": {"username": "U", "password": "P"},
                },
            }
            headers: dict[str, str] = {}
            if path in cloud_answers:
                status, answer = 200, cloud_answers[path]
            elif path.startswith("/cayu/"):
                state["agent_requests"].append((self.command, path, body))
                status, answer, headers = state["agent"](
                    self.command, path.removeprefix("/cayu"), body, dict(self.headers)
                )
            else:
                status, answer = 404, {"detail": "Not Found"}
            payload = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def served(cloud: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The Agent is a real Cayu server behind the operator login Cloud hands out."""

    access = AuthenticatedAccess(
        dependency=BasicAuth(username="operator", password=_OPERATOR_PASSWORD)
    )
    with TestClient(create_server(_agent_app(), config=ServerConfig(access=access))) as client:

        def forward(
            method: str, path: str, body: bytes, headers: dict[str, str]
        ) -> tuple[int, Any, dict[str, str]]:
            sent = {
                name: value
                for name, value in headers.items()
                if name.lower() in {"authorization", "content-type", "accept"}
            }
            response = client.request(method, path, content=body or None, headers=sent)
            kept = {
                name: value
                for name, value in response.headers.items()
                if name.lower() in {"retry-after", "cache-control"}
            }
            return response.status_code, response.json(), kept

        cloud["agent"] = forward
        yield cloud


def _fake_agent(model_check: tuple[int, Any]) -> AgentAnswer:
    """An Agent that serves its contract and answers the check with ``model_check``."""

    def answer(
        method: str, path: str, body: bytes, headers: dict[str, str]
    ) -> tuple[int, Any, dict[str, str]]:
        if (method, path) == ("GET", "/api/contract"):
            return 200, {"server_contract_version": "79"}, {}
        if path == "/api/diagnostics/model-check":
            return (*model_check, {})
        return 404, {"detail": "Not Found"}, {}

    return answer


def _run(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, Any, str]:
    code = main(list(arguments))
    captured = capsys.readouterr()
    out = captured.out
    return code, json.loads(out) if out.startswith("{") else out, captured.err


def _assert_private_text_absent(text: str) -> None:
    for private in (
        _MODEL_CHECK_PROMPT,
        _ANSWER,
        _PROVIDER_TEXT,
        _OPERATOR_PASSWORD,
        "customer-secret-material",
    ):
        assert private not in text


def test_the_check_runs_on_the_agent_with_its_providers_and_login(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, report, _ = _run(
        capsys, "providers", "check", "probe-model", "--application", "bug-hunter"
    )

    assert code == 0
    _assert_private_text_absent(json.dumps(report))
    assert report["ok"] is True
    assert report["route"] == {
        "provider": "scripted",
        "model": "probe-model",
        "resolution": "model_pattern",
    }
    assert report["answer"] == {"finish_reason": "stop", "text_received": True}
    assert report["usage"]["output_tokens"] == 2
    assert report["pricing"]["price_book"]["source"] == "bundled"
    method, path, body = served["agent_requests"][-1]
    # The CLI names only the model; the Agent supplies the prompt, cap and deadline.
    assert (method, path, json.loads(body)) == (
        "POST",
        "/cayu/api/diagnostics/model-check",
        {"model": "probe-model"},
    )


def test_a_provider_failure_on_the_agent_exits_1_without_its_text(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, human, _ = _run(
        capsys,
        "providers",
        "check",
        "probe-model",
        "--provider",
        "failing",
        "--application",
        "bug-hunter",
        "--table",
    )

    assert code == 1
    _assert_private_text_absent(human)
    assert human.startswith("FAILED failing/probe-model in ")
    assert "authentication_failed HTTP 401" in human


def test_doctor_model_runs_on_the_agent_too(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, report, _ = _run(
        capsys, "doctor", "--model", "probe-model", "--application", "bug-hunter"
    )
    assert code == 0
    assert report["ok"] is True

    assert main(["doctor", "--bundle", "bundle.zip", "--application", "bug-hunter"]) == 2
    assert main(["doctor", "app:build", "--model", "probe-model", "--application", "x"]) == 2
    assert "drop the target" in capsys.readouterr().err


def test_the_agents_refusals_are_reported_as_not_run(
    served: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _run(
        capsys,
        "providers",
        "check",
        "probe-model",
        "--provider",
        "missing",
        "--application",
        "bug-hunter",
    )
    assert code == 2
    assert out["error"]["code"] == "MODEL_CHECK_NOT_RUN"
    assert out["error"]["category"] == "request_rejected"
    assert (
        "Provider not registered: missing. Registered providers: failing, scripted."
        in (out["error"]["message"])
    )

    check = ("providers", "check", "probe-model", "--application", "bug-hunter")
    assert _run(capsys, *check)[0] == 0
    code, out, _ = _run(capsys, *check)
    assert code == 2
    assert out["error"]["category"] == "busy"
    assert "try again in a few seconds" in out["error"]["message"]

    served["login"] = "rotated-password"
    code, out, _ = _run(capsys, *check)
    assert code == 2
    assert out["error"]["category"] == "credentials_rejected"
    _assert_private_text_absent(json.dumps(out))


@pytest.mark.parametrize(
    "answer",
    [(404, {"detail": "Not Found"}), (405, {"detail": "Method Not Allowed"})],
    ids=["no-route", "other-method"],
)
def test_an_agent_whose_cayu_predates_the_check_is_told_apart(
    cloud: dict[str, Any], capsys: pytest.CaptureFixture[str], answer: tuple[int, Any]
) -> None:
    cloud["agent"] = _fake_agent(answer)

    code, out, _ = _run(capsys, "providers", "check", "probe-model", "--application", "bug-hunter")

    assert code == 2
    assert out["error"]["category"] == "unsupported"
    assert out["error"]["message"] == (
        "The Agent bug-hunter runs a Cayu that can't check models. Deploy a release built "
        "with this CLI's Cayu version or newer, then run the check again."
    )

    code, human, err = _run(
        capsys, "providers", "check", "probe-model", "--application", "bug-hunter", "--table"
    )
    assert code == 2 and human == ""
    assert err.startswith("error: The Agent bug-hunter runs a Cayu that can't check models.")


def test_an_answer_that_is_not_a_report_is_not_printed(
    cloud: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud["agent"] = _fake_agent((200, {"ok": True, "answer": _ANSWER}))

    code, out, _ = _run(capsys, "providers", "check", "probe-model", "--application", "bug-hunter")

    assert code == 2
    assert out["error"]["category"] == "invalid_response"
    assert _ANSWER not in json.dumps(out)


_ANSWERED = {"finish_reason": "stop", "text_received": True}
_FAILED = {"category": "timeout", "message": "The provider did not answer before the deadline."}


@pytest.mark.parametrize(
    ("ok", "answer", "error"),
    [
        (False, None, None),
        (True, None, _FAILED),
        (True, _ANSWERED, _FAILED),
        (False, _ANSWERED, None),
    ],
)
def test_a_report_whose_outcome_contradicts_itself_is_not_printed(
    cloud: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    ok: bool,
    answer: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> None:
    report = {
        "schema_version": "1",
        "ok": ok,
        "route": {"provider": "scripted", "model": "probe-model", "resolution": "model_pattern"},
        "limits": {"max_output_tokens": 16, "timeout_seconds": 30.0},
        "latency_ms": 5,
        "answer": answer,
        "usage": None,
        "pricing": {
            "priced": False,
            "price_book": {"source": "bundled", "version": "v1", "generated_at": "unspecified"},
            "unpriced_reason": "no matching model pricing",
        },
        "error": error,
    }
    cloud["agent"] = _fake_agent((200, report))
    check = ("providers", "check", "probe-model", "--application", "bug-hunter")

    code, human, err = _run(capsys, *check, "--table")
    assert (code, human) == (2, "")
    assert "answered the model check with a report this CLI can't read" in err

    code, out, _ = _run(capsys, *check)
    assert code == 2
    assert out["error"]["category"] == "invalid_response"


def test_cloud_options_are_checked_before_anything_is_sent(
    cloud: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = tmp_path / "context.json"
    context.write_text('{"private":"preserved"}')

    code, out, _ = _run(capsys, "providers", "check", "probe-model", "--context", str(context))
    assert code == 2
    assert out["error"]["message"] == "--context applies only with --application."

    output = tmp_path / "report.json"
    output.hardlink_to(context)
    code, out, _ = _run(
        capsys,
        "providers",
        "check",
        "probe-model",
        "--application",
        "bug-hunter",
        "--context",
        str(context),
        "--output",
        str(output),
    )
    assert code == 2
    assert out["error"]["message"] == "output must not refer to an input file."
    assert context.read_text() == '{"private":"preserved"}'

    with pytest.raises(SystemExit) as raised:
        main(["providers", "check", "probe-model", "--target", "app:build", "--application", "x"])
    assert raised.value.code == 2
    with pytest.raises(SystemExit):
        main(["providers", "check", "probe-model", "--application", " padded"])
    assert cloud["agent_requests"] == []
