"""`--wait` on `cayu cloud schedule run` and `service` changes while Cloud switches the service.

While Cayu Cloud publishes a release or changes an Agent's service, it refuses a run on demand
or another service change with HTTP 409 and a plain-text reason that says nothing changed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_service_switch import ServiceStillSwitchingError, send_when_not_switching

_APP = "/v1/applications/cayu-scout"
_RUN = f"{_APP}/schedules/scout/run"
_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENTS = {"items": [{"id": "cayu-scout", "name": "Cayu Scout"}]}
_SWITCHING = (
    "The Agent's service is being deployed or changed right now; nothing changed. Try again "
    "once that finishes."
)
_STARTED = {
    "application_id": "cayu-scout",
    "deployment_id": "dep_new",
    "input_overridden": True,
    "run": "task-1",
    "schedule": "scout",
    "started_at": "2026-10-28T18:00:00Z",
}


def _switching() -> httpx.Response:
    return httpx.Response(409, json={"detail": _SWITCHING})


class _Cloud:
    """Cloud's answers to each request, in order; the last one stands once the others are used."""

    def __init__(self, routes: dict[tuple[str, str], list[httpx.Response]]) -> None:
        self.routes = {_LIST: [httpx.Response(200, json=_AGENTS)], **routes}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answers = self.routes[(request.method, request.url.raw_path.decode())]
        return answers.pop(0) if len(answers) > 1 else answers[0]

    def sent(self, method: str, path: str) -> list[httpx.Request]:
        return [
            request
            for request in self.requests
            if (request.method, request.url.raw_path.decode()) == (method, path)
        ]


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], list[httpx.Response]]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], list[httpx.Response]]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: _client())
        return fake

    return install


def _client() -> CloudApiClient:
    return CloudApiClient(api_url="https://cloud.example", api_key="login-token")


def _clock() -> Callable[[], float]:
    ticks: Iterator[int] = iter(range(0, 1_000_000, 5))
    return lambda: float(next(ticks))


def _run_schedule(*, wait: bool, sleeps: list[float] | None = None) -> dict[str, Any]:
    return cloud_cli._schedule(
        SimpleNamespace(
            application="cayu-scout",
            input={"prompt": "Look again"},
            name="scout",
            poll_seconds=5.0,
            wait=wait,
            wait_seconds=30.0,
        ),
        client=_client(),
        sleep=(sleeps if sleeps is not None else []).append,
        monotonic=_clock(),
    )


def test_schedule_run_with_wait_sends_the_same_request_again_once_the_switch_finishes(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("POST", _RUN): [_switching(), _switching(), httpx.Response(202, json=_STARTED)]})
    sleeps: list[float] = []

    assert _run_schedule(wait=True, sleeps=sleeps) == {
        "operation": "schedule.run",
        "result": _STARTED,
    }

    runs = fake.sent("POST", _RUN)
    assert len(runs) == 3
    assert {request.content for request in runs} == {b'{"input":{"prompt":"Look again"}}'}
    assert sleeps == [5.0, 5.0]
    # Said once, not on every refusal.
    assert capsys.readouterr().err.count("being deployed or changed right now") == 1


def test_schedule_run_with_wait_knows_the_switch_by_its_code(
    cloud: Callable[..., _Cloud],
) -> None:
    coded = httpx.Response(409, json={"detail": {"code": "service_busy", "message": _SWITCHING}})
    fake = cloud({("POST", _RUN): [coded, httpx.Response(202, json=_STARTED)]})

    assert _run_schedule(wait=True)["result"] == _STARTED
    assert len(fake.sent("POST", _RUN)) == 2


def test_schedule_run_with_wait_gives_up_when_the_switch_outlasts_the_wait(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("POST", _RUN): [_switching()]})

    with pytest.raises(CloudApiError) as raised:
        _run_schedule(wait=True)

    assert cloud_cli._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error == {
        "application": "cayu-scout",
        "category": "service_busy",
        "commands": {"status": "cayu cloud service status --application cayu-scout"},
        "message": (
            "The Agent's service was still being deployed or changed after 30s, so Cayu Cloud "
            "kept refusing the request; nothing changed. Run the command again once that "
            "finishes."
        ),
        "waited_seconds": 30,
    }
    # The test clock moves 5 seconds on every read, and each round reads it after the
    # refusal and after the sleep: the first refusal starts the 30-second wait, and three
    # more requests fit in it.
    assert len(fake.sent("POST", _RUN)) == 4


def test_the_request_is_never_sent_again_once_the_wait_is_over() -> None:
    now = [0.0]
    sent: list[float] = []

    def refused() -> dict[str, Any]:
        sent.append(now[0])
        # Cloud waits for the service lock for a while before it refuses.
        now[0] += 10.0
        raise CloudApiError("api_request_rejected", "HTTP 409.", status_code=409, detail=_SWITCHING)

    def sleep(seconds: float) -> None:
        now[0] += seconds

    with pytest.raises(ServiceStillSwitchingError) as raised:
        send_when_not_switching(
            refused,
            wait=True,
            application_id="cayu-scout",
            poll_seconds=40.0,
            wait_seconds=60.0,
            sleep=sleep,
            monotonic=lambda: now[0],
        )

    # The last sleep is cut to the time left, and nothing is sent after it.
    assert sent == [0.0, 50.0]
    assert raised.value.details["waited_seconds"] == 60


def test_schedule_run_never_sends_again_after_another_conflict(
    cloud: Callable[..., _Cloud],
) -> None:
    reason = "Application service is archived."
    fake = cloud({("POST", _RUN): [httpx.Response(409, json={"detail": reason})]})

    with pytest.raises(CloudApiError) as raised:
        _run_schedule(wait=True)

    assert str(raised.value) == f"Cayu Cloud API returned HTTP 409: {reason}"
    assert len(fake.sent("POST", _RUN)) == 1


def test_schedule_run_without_wait_says_that_wait_would_send_it_again(
    cloud: Callable[..., _Cloud],
) -> None:
    fake = cloud({("POST", _RUN): [_switching(), httpx.Response(202, json=_STARTED)]})

    with pytest.raises(CloudApiError) as raised:
        _run_schedule(wait=False)

    assert raised.value.status_code == 409
    assert str(raised.value) == (
        f"Cayu Cloud API returned HTTP 409: {_SWITCHING} Pass --wait to wait for it and send "
        "the request again."
    )
    assert len(fake.sent("POST", _RUN)) == 1


def test_schedule_run_wait_from_the_command_line(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("POST", _RUN): [_switching(), httpx.Response(202, json=_STARTED)]})

    code = main(
        [
            "cloud",
            "schedule",
            "run",
            "scout",
            "--application",
            "cayu-scout",
            "--wait",
            "--poll-seconds",
            "0.001",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "operation": "schedule.run",
        "result": _STARTED,
    }


@pytest.mark.parametrize(
    ("action", "method", "path"),
    [
        ("restart", "POST", f"{_APP}/service/restart"),
        ("sleep", "POST", f"{_APP}/service/sleep"),
        ("wake", "POST", f"{_APP}/service/wake"),
        ("pause", "POST", f"{_APP}/service/pause"),
        ("resume", "POST", f"{_APP}/service/resume"),
        ("destroy", "DELETE", f"{_APP}/service"),
    ],
)
def test_service_changes_with_wait_send_the_request_again_once_the_switch_finishes(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    action: str,
    method: str,
    path: str,
) -> None:
    service = {"application_id": "cayu-scout", "status": "running"}
    fake = cloud({(method, path): [_switching(), httpx.Response(200, json=service)]})

    code = main(
        [
            "cloud",
            "service",
            action,
            "--application",
            "cayu-scout",
            "--wait",
            "--poll-seconds",
            "0.001",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["result"] == service
    assert len(fake.sent(method, path)) == 2


@pytest.mark.parametrize("action", ["status", "logs", "credentials"])
def test_service_reads_take_no_wait(action: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["cloud", "service", action, "--application", "cayu-scout", "--wait"])

    assert code == 2
    assert "--wait" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_wait_options_are_checked_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid wait input reached Cloud authentication")

    monkeypatch.setattr(cloud_cli, "_cloud_client", unexpected_client)

    code = main(
        [
            "cloud",
            "schedule",
            "run",
            "scout",
            "--application",
            "cayu-scout",
            "--wait",
            "--poll-seconds",
            "10",
            "--wait-seconds",
            "1",
        ]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "invalid_wait"
