"""`cayu cloud schedule runs` and `cayu cloud schedule run --wait-for`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_APP = "/v1/applications/acme-reports"
_RUNS = f"{_APP}/schedule-runs"
_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENTS = httpx.Response(200, json={"items": [{"id": "acme-reports", "name": "Acme Reports"}]})
_STARTED = {
    "active_runs": [],
    "already_running": False,
    "application_id": "acme-reports",
    "deployment_id": "release-1",
    "input_overridden": False,
    "run": "job-run-1",
    "schedule": "nightly",
    "started_at": "2026-10-26T18:00:00Z",
}
_SCHEDULER = "The scheduler reports: 0/6 nodes are available: 6 Insufficient cpu."
_UNSUPPORTED = {
    "detail": {
        "code": "schedule_runs_unsupported",
        "message": "This Cayu Cloud install can't list schedule runs.",
    }
}


def _run_state(status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schedule": "nightly",
        "run": "job-run-1",
        "status": status,
        "created_at": "2026-10-26T18:00:00Z",
        "on_demand": True,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "reason": None,
        "hint": None,
        **fields,
    }


_PENDING = _run_state(
    "pending", reason="No Agent node has room for its pod (30s waiting).", hint=_SCHEDULER
)


class _Cloud:
    """Answers each route in turn; the last answer repeats."""

    def __init__(self, routes: dict[tuple[str, str], list[httpx.Response]]) -> None:
        self.routes = {_LIST: [_AGENTS], **routes}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answers = self.routes[(request.method, request.url.raw_path.decode())]
        return answers.pop(0) if len(answers) > 1 else answers[0]


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], list[httpx.Response]]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], list[httpx.Response]]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _listing(*items: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"application_id": "acme-reports", "items": list(items)})


def _json(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_runs_prints_a_table_with_why_a_run_waits(cloud, capsys) -> None:
    finished = _run_state(
        "succeeded",
        run="job-1",
        on_demand=False,
        started_at="2026-10-26T03:00:05Z",
        finished_at="2026-10-26T03:01:00Z",
        exit_code=0,
    )
    fake = cloud({("GET", f"{_RUNS}?schedule=nightly"): [_listing(_PENDING, finished)]})

    code = main(["cloud", "schedule", "runs", "nightly", "--application", "acme-reports"])

    out = capsys.readouterr().out
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == [
        "RUN",
        "SCHEDULE",
        "STATUS",
        "BY",
        "CREATED",
        "STARTED",
        "FINISHED",
        "EXIT",
    ]
    assert lines[1].split()[:4] == ["job-run-1", "nightly", "pending", "run"]
    assert lines[2].split()[:4] == ["job-1", "nightly", "succeeded", "schedule"]
    assert f"job-run-1: No Agent node has room for its pod (30s waiting). {_SCHEDULER}" in lines
    assert fake.requests[-1].url.params["schedule"] == "nightly"


def test_runs_json_prints_the_envelope(cloud, capsys) -> None:
    fake = cloud({("GET", _RUNS): [_listing(_PENDING)]})

    code, output = _json(["schedule", "runs", "--application", "acme-reports", "--json"], capsys)

    assert code == 0
    assert output["operation"] == "schedule.runs"
    assert output["result"] == {
        "application": "acme-reports",
        "schedule": None,
        "items": [_PENDING],
    }
    assert "schedule" not in fake.requests[-1].url.params


def test_runs_with_none_says_so(cloud, capsys) -> None:
    cloud({("GET", _RUNS): [_listing()]})

    assert main(["cloud", "schedule", "runs", "--application", "acme-reports"]) == 0
    assert capsys.readouterr().out.strip() == "No schedule runs for acme-reports."


def test_runs_on_a_cloud_without_the_listing_says_so(cloud, capsys) -> None:
    cloud({("GET", _RUNS): [httpx.Response(404, json={"detail": "Not Found"})]})

    code, output = _json(["schedule", "runs", "--application", "acme-reports", "--json"], capsys)

    assert code == 2
    assert output["error"] == {
        "category": "schedule_runs_unsupported",
        "message": "This Cayu Cloud doesn't list schedule runs yet.",
    }


def test_runs_on_an_install_that_cant_list_them_carries_its_reason(cloud, capsys) -> None:
    cloud({("GET", _RUNS): [httpx.Response(501, json=_UNSUPPORTED)]})

    code, output = _json(["schedule", "runs", "--application", "acme-reports", "--json"], capsys)

    assert code == 2
    assert output["error"]["code"] == "schedule_runs_unsupported"
    assert "can't list schedule runs" in output["error"]["message"]


def _wait_for(until: str, *extra: str) -> list[str]:
    return [
        "schedule",
        "run",
        "nightly",
        "--application",
        "acme-reports",
        "--wait-for",
        until,
        "--poll-seconds",
        "0.01",
        *extra,
    ]


def test_wait_for_finish_follows_the_run_until_it_succeeds(cloud, capsys) -> None:
    running = _run_state("running", started_at="2026-10-26T18:00:40Z")
    succeeded = _run_state("succeeded", finished_at="2026-10-26T18:02:00Z", exit_code=0)
    fake = cloud(
        {
            ("POST", f"{_APP}/schedules/nightly/run"): [httpx.Response(202, json=_STARTED)],
            ("GET", f"{_RUNS}?schedule=nightly"): [
                _listing(_PENDING),
                _listing(running),
                _listing(succeeded),
            ],
        }
    )

    code = main(["cloud", *_wait_for("finish")])

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert code == 0
    assert output["result"]["run"] == "job-run-1"
    assert output["result"]["run_state"] == succeeded
    assert captured.err.splitlines()[-3:] == [
        f"Run job-run-1: pending. No Agent node has room for its pod (30s waiting). {_SCHEDULER}",
        "Run job-run-1: running",
        "Run job-run-1: succeeded",
    ]
    assert [request.url.path for request in fake.requests].count(_RUNS) == 3


def test_wait_for_start_stops_once_the_run_runs(cloud, capsys) -> None:
    running = _run_state("running", started_at="2026-10-26T18:00:40Z")
    cloud(
        {
            ("POST", f"{_APP}/schedules/nightly/run"): [httpx.Response(202, json=_STARTED)],
            ("GET", f"{_RUNS}?schedule=nightly"): [_listing(running)],
        }
    )

    code, output = _json(_wait_for("start"), capsys)

    assert code == 0
    assert output["result"]["run_state"]["status"] == "running"


def test_wait_for_a_run_that_stays_pending_exits_with_the_schedulers_reason(cloud, capsys) -> None:
    cloud(
        {
            ("POST", f"{_APP}/schedules/nightly/run"): [httpx.Response(202, json=_STARTED)],
            ("GET", f"{_RUNS}?schedule=nightly"): [_listing(_PENDING)],
        }
    )

    code, output = _json(_wait_for("start", "--wait-seconds", "0.05"), capsys)

    assert code == 2
    error = output["error"]
    assert error["category"] == "schedule_run_wait_timeout"
    assert error["message"].startswith(
        "Run job-run-1 of schedule nightly is still pending after 0.05 seconds. No Agent node"
    )
    assert _SCHEDULER in error["message"]
    assert error["run"] == _PENDING


def test_wait_for_a_failed_run_exits_nonzero_with_why(cloud, capsys) -> None:
    failed = _run_state(
        "failed",
        exit_code=137,
        reason="Its process exited with code 137 (OOMKilled).",
        hint="Its output is in the service logs of its schedule process.",
    )
    cloud(
        {
            ("POST", f"{_APP}/schedules/nightly/run"): [httpx.Response(202, json=_STARTED)],
            ("GET", f"{_RUNS}?schedule=nightly"): [_listing(failed)],
        }
    )

    code, output = _json(_wait_for("finish"), capsys)

    assert code == 2
    assert output["error"]["category"] == "schedule_run_failed"
    assert output["error"]["message"] == (
        "Run job-run-1 of schedule nightly failed. Its process exited with code 137 (OOMKilled). "
        "Its output is in the service logs of its schedule process."
    )
    assert output["error"]["run"] == failed


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(404, json={"detail": "Not Found"}),
        httpx.Response(501, json=_UNSUPPORTED),
    ],
)
def test_wait_for_on_a_cloud_without_the_listing_says_the_run_started(
    cloud, capsys, answer: httpx.Response
) -> None:
    cloud(
        {
            ("POST", f"{_APP}/schedules/nightly/run"): [httpx.Response(202, json=_STARTED)],
            ("GET", f"{_RUNS}?schedule=nightly"): [answer],
        }
    )

    code, output = _json(_wait_for("finish"), capsys)

    assert code == 2
    assert output["error"]["category"] == "schedule_runs_unsupported"
    assert output["error"]["message"].startswith("Run job-run-1 started, but this Cayu Cloud")
    assert output["error"]["run"] == "job-run-1"


def test_wait_for_checks_its_timings_before_starting_the_run(cloud, capsys) -> None:
    fake = cloud({})

    code, output = _json(_wait_for("finish", "--wait-seconds", "0.001"), capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_wait"
    assert all(request.method == "GET" for request in fake.requests)
