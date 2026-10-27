"""`cayu cloud service pause|resume`, `cayu cloud schedule run` and schedule `input`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_project import CloudProjectManifest

_APP = "/v1/applications/cayu-scout"
_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENTS = httpx.Response(200, json={"items": [{"id": "cayu-scout", "name": "Cayu Scout"}]})


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = {_LIST: _AGENTS, **routes}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.routes[(request.method, request.url.raw_path.decode())]


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], httpx.Response]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], httpx.Response]) -> _Cloud:
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


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", *arguments])
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("action", ["pause", "resume"])
def test_service_pause_and_resume_answer_the_service(
    cloud, capsys: pytest.CaptureFixture[str], action: str
) -> None:
    paused = action == "pause"
    service = {
        "status": "running",
        "paused": paused,
        "paused_at": "2026-10-26T18:00:00Z" if paused else None,
    }
    fake = cloud({("POST", f"{_APP}/service/{action}"): httpx.Response(200, json=service)})

    code, output = _run(["service", action, "--application", "cayu-scout"], capsys)

    assert code == 0
    assert output["operation"] == f"service.{action}"
    assert output["result"] == service
    assert fake.requests[-1].content == b""


def test_service_status_shows_paused(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    service = {"status": "running", "paused": True, "paused_at": "2026-10-26T18:00:00Z"}
    cloud({("GET", f"{_APP}/service"): httpx.Response(200, json=service)})

    code, output = _run(["service", "status", "--application", "cayu-scout"], capsys)

    assert code == 0
    assert output["result"]["paused"] is True


def test_service_pause_on_a_cloud_without_it_says_so(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("POST", f"{_APP}/service/pause"): httpx.Response(404, json={"detail": "Not Found"})})

    code, output = _run(["service", "pause", "--application", "cayu-scout"], capsys)

    assert code == 2
    assert output["error"] == {
        "category": "service_pause_unsupported",
        "message": "This Cayu Cloud can't pause Agents yet.",
    }


def test_service_pause_conflict_carries_clouds_reason(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    reason = "Agent service is being deleted."
    cloud({("POST", f"{_APP}/service/pause"): httpx.Response(409, json={"detail": reason})})

    code, output = _run(["service", "pause", "--application", "cayu-scout"], capsys)

    assert code == 2
    assert output["error"]["message"] == f"Cayu Cloud API returned HTTP 409: {reason}"


def test_schedule_run_sends_the_input_and_answers_the_run(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    run = {
        "application_id": "cayu-scout",
        "deployment_id": "dep_one",
        "input_overridden": True,
        "run": "task-1",
        "schedule": "scout",
        "started_at": "2026-10-26T18:00:00Z",
    }
    fake = cloud({("POST", f"{_APP}/schedules/scout/run"): httpx.Response(202, json=run)})

    code, output = _run(
        [
            "schedule",
            "run",
            "scout",
            "--application",
            "cayu-scout",
            "--input",
            '{"prompt": "Look again", "limit": 3}',
        ],
        capsys,
    )

    assert code == 0
    assert output == {"ok": True, "operation": "schedule.run", "result": run}
    assert json.loads(fake.requests[-1].content) == {"input": {"limit": 3, "prompt": "Look again"}}


def test_schedule_run_without_input_sends_no_body(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {("POST", f"{_APP}/schedules/scout/run"): httpx.Response(202, json={"run": "task-2"})}
    )

    code, _ = _run(["schedule", "run", "scout", "--application", "cayu-scout"], capsys)

    assert code == 0
    assert fake.requests[-1].content == b""


def test_schedule_run_reports_an_unknown_schedule(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    detail = {
        "code": "schedule_not_found",
        "message": "The deployed Release declares no schedule named nightly.",
    }
    cloud({("POST", f"{_APP}/schedules/nightly/run"): httpx.Response(404, json={"detail": detail})})

    code, output = _run(["schedule", "run", "nightly", "--application", "cayu-scout"], capsys)

    assert code == 2
    assert output["error"]["code"] == "schedule_not_found"
    assert output["error"]["message"] == f"Cayu Cloud API returned HTTP 404: {detail['message']}"


@pytest.mark.parametrize(
    ("value", "reason"),
    [("{not json", "must be JSON"), (json.dumps("x" * 4095), "at most 4096 bytes")],
)
def test_schedule_run_refuses_bad_input_before_calling_cloud(
    capsys: pytest.CaptureFixture[str], value: str, reason: str
) -> None:
    code, output = _run(
        ["schedule", "run", "scout", "--application", "cayu-scout", "--input", value], capsys
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert reason in output["error"]["message"]


_MANIFEST = """
schema_version = 2
application = "cayu-scout"
name = "Cayu Scout"
version = "0.1.0"
entrypoint = "python -m scout"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 2048
timeout_seconds = 900
environment = "python"
compatibility = "cayu>=0.1,<1"
policy_version = "cayu-egress-v1"

[[schedules]]
name = "scout"
command = "python -m scout.run"
expression = "rate(6 hours)"
"""


def _schedules(manifest: CloudProjectManifest) -> list[dict[str, object]]:
    runtime = manifest.runtime_payload()
    assert runtime is not None
    schedules = runtime["schedules"]
    assert isinstance(schedules, list)
    return schedules


def test_a_schedule_without_input_keeps_the_manifest_payload_unchanged() -> None:
    assert _schedules(CloudProjectManifest.loads(_MANIFEST)) == [
        {"command": "python -m scout.run", "expression": "rate(6 hours)", "name": "scout"}
    ]


def test_a_schedule_input_is_sent_with_the_deploy() -> None:
    manifest = CloudProjectManifest.loads(
        _MANIFEST + 'input = { prompt = "Look for new pricing pages", limit = 20 }\n'
    )

    assert _schedules(manifest)[0]["input"] == {"prompt": "Look for new pricing pages", "limit": 20}


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("input = 1979-05-27T07:32:00Z", "must be JSON"),
        ("input = nan", "must be JSON"),
        ('input = "' + "\\u00e9" * 683 + '"', "at most 4096 bytes"),
    ],
)
def test_a_schedule_input_cloud_would_refuse_is_refused_locally(line: str, reason: str) -> None:
    with pytest.raises(CloudApiError, match=reason) as raised:
        CloudProjectManifest.loads(_MANIFEST + line + "\n")

    assert raised.value.category == "manifest_invalid"


@pytest.mark.parametrize(
    ("health", "expected"),
    [
        ({"fenced_after_worker_loss": 1, "complete": True}, "1 task stayed claimed"),
        ({"fenced_after_worker_loss": 100, "complete": False}, "100 tasks or more stayed claimed"),
    ],
)
def test_service_status_warns_about_tasks_held_by_a_stopped_worker(
    cloud, capsys: pytest.CaptureFixture[str], health: dict[str, Any], expected: str
) -> None:
    task_health = {**health, "read_at": "2026-10-27T09:30:00Z"}
    service = {"status": "running", "task_health": task_health}
    cloud({("GET", f"{_APP}/service"): httpx.Response(200, json=service)})

    code, output = _run(["service", "status", "--application", "cayu-scout"], capsys)

    assert code == 0
    assert output["result"]["task_health"] == task_health
    (warning,) = output["warnings"]
    assert warning.startswith(expected)
    assert "`cayu inspect --fenced-tasks`" in warning
    assert "--reconcile-abandoned-after-seconds" in warning


@pytest.mark.parametrize(
    "task_health",
    [
        None,
        {"fenced_after_worker_loss": 0, "complete": True, "read_at": None},
        {"fenced_after_worker_loss": "2", "complete": True, "read_at": None},
        "unexpected",
    ],
)
def test_service_status_says_nothing_without_fenced_tasks(
    cloud, capsys: pytest.CaptureFixture[str], task_health: object
) -> None:
    cloud(
        {
            ("GET", f"{_APP}/service"): httpx.Response(
                200, json={"status": "running", "task_health": task_health}
            )
        }
    )

    code, output = _run(["service", "status", "--application", "cayu-scout"], capsys)

    assert code == 0
    assert "warnings" not in output
