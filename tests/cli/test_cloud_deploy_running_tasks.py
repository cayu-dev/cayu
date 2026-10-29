"""`cayu cloud deploy` lists the tasks the Agent's current workers run and warns."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_reused_release import ReusingCloud
from tests.cli.test_cloud_reused_release import _deploy as reused_deploy

from cayu.cli import _cloud_deploy_tasks as deploy_tasks
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiError

CAYU_URL = "https://agent.example.test/cayu/"


class FakeCloud:
    def __init__(self, *, service: dict[str, Any] | None = None, credentials: Any = None):
        self.service = service or {"status": "running", "cayu_url": CAYU_URL}
        self.credentials = credentials or {
            "username": "operator",
            "password": "pw",
            "env": {"username": "U", "password": "P"},
        }

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/v1/applications/scanner-agent/service":
            if isinstance(self.service, Exception):
                raise self.service
            return self.service
        if path == "/v1/applications/scanner-agent/operator-credentials":
            if isinstance(self.credentials, Exception):
                raise self.credentials
            return self.credentials
        raise AssertionError(f"unexpected Cloud request: {method} {path}")


def _task(task_id: str, status: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "type": "scan",
        "status": status,
        "status_reason": None,
        "worker_id": "worker-1",
        "assigned_agent_name": "scanner",
        "updated_at": "2026-10-28T16:19:00+00:00",
        "retry_series": None,
    }


def _agent(tasks: dict[str, list[dict[str, Any]]], *, status: int = 200) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Basic ")
        if status != 200:
            return httpx.Response(status, json={"detail": "Not authenticated"})
        assert request.url.path == "/cayu/api/tasks"
        return httpx.Response(200, json=tasks.get(request.url.params["status"], []))

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_reads_running_and_claimed_tasks_and_warns() -> None:
    report = deploy_tasks.read_running_tasks(
        FakeCloud(),  # type: ignore[arg-type]
        "scanner-agent",
        http=_agent(
            {"running": [_task("scan-1", "running")], "claimed": [_task("scan-2", "claimed")]}
        ),
    )

    assert report["status"] == "read" and report["complete"] is True
    assert [task["id"] for task in report["tasks"]] == ["scan-2", "scan-1"]
    warning = deploy_tasks.running_tasks_warning(report, application_id="scanner-agent")
    assert warning is not None
    assert warning.startswith("2 tasks are claimed or running on the Agent's current workers")
    assert "scan-2 (scan, claimed), scan-1 (scan, running)" in warning
    assert "cancelled or left fenced" in warning
    assert (
        "cayu task list --application scanner-agent --status claimed` and `--status running`"
        in warning
    )


def test_no_tasks_means_no_warning_and_a_full_page_says_or_more() -> None:
    quiet = deploy_tasks.read_running_tasks(
        FakeCloud(),  # type: ignore[arg-type]
        "scanner-agent",
        http=_agent({}),
    )
    assert quiet == {"status": "read", "tasks": [], "complete": True}
    assert deploy_tasks.running_tasks_warning(quiet, application_id="scanner-agent") is None

    many = [_task(f"scan-{index}", "running") for index in range(deploy_tasks.TASK_LIMIT + 1)]
    busy = deploy_tasks.read_running_tasks(
        FakeCloud(),  # type: ignore[arg-type]
        "scanner-agent",
        http=_agent({"running": many}),
    )
    assert busy["complete"] is False and len(busy["tasks"]) == deploy_tasks.TASK_LIMIT
    warning = deploy_tasks.running_tasks_warning(busy, application_id="scanner-agent")
    assert warning is not None and warning.startswith(f"{deploy_tasks.TASK_LIMIT} or more tasks")


@pytest.mark.parametrize(
    "cloud",
    [
        FakeCloud(service={"status": "stopped", "cayu_url": None}),
        FakeCloud(
            service=CloudApiError(
                "api_request_rejected", "Cayu Cloud API returned HTTP 404.", status_code=404
            )  # type: ignore[arg-type]
        ),
    ],
)
def test_an_agent_without_a_running_service_has_nothing_to_interrupt(cloud: FakeCloud) -> None:
    report = deploy_tasks.read_running_tasks(cloud, "scanner-agent")  # type: ignore[arg-type]

    assert report == {"status": "not_running"}
    assert deploy_tasks.unavailable_notice(report) is None


def test_a_read_that_fails_is_reported_without_stopping_the_deploy() -> None:
    refused = deploy_tasks.read_running_tasks(
        FakeCloud(
            credentials=CloudApiError(
                "api_request_rejected",
                "A signed-in organization administrator is required.",
                status_code=403,
            )
        ),  # type: ignore[arg-type]
        "scanner-agent",
    )
    assert refused["status"] == "unavailable"
    notice = deploy_tasks.unavailable_notice(refused)
    assert notice is not None and "administrator is required" in notice

    rejected = deploy_tasks.read_running_tasks(
        FakeCloud(),  # type: ignore[arg-type]
        "scanner-agent",
        http=_agent({}, status=401),
    )
    assert rejected["status"] == "unavailable"
    assert rejected["reason"] == "credentials_rejected"


def test_deploy_reports_running_tasks_before_uploading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[str] = []
    report = {"status": "read", "tasks": [_task("scan-1", "running")], "complete": True}

    def read(client: Any, application_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.append(application_id)
        # Nothing has been created yet when the tasks are read.
        assert not any(method == "POST" for method, _ in client.requests)
        return report

    monkeypatch.setattr(deploy_tasks, "read_running_tasks", read)
    client = ReusingCloud(
        service={"deployment_id": "dep_existing", "issues": [], "status": "running"}
    )

    response = reused_deploy(client, tmp_path, no_wait=True)

    assert seen == ["smoke-agent"]
    assert response["result"]["running_tasks"]["tasks"][0]["id"] == "scan-1"
    [warning] = response["warnings"]
    assert "1 task is claimed or running" in warning
    assert "cayu cloud: 1 task is claimed or running" in capsys.readouterr().err


def test_candidate_and_first_deploys_skip_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def read(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("no running-task read expected")

    monkeypatch.setattr(deploy_tasks, "read_running_tasks", read)
    monkeypatch.setattr(cloud_cli, "_requested_candidate", lambda arguments: None)
    project = SimpleNamespace(manifest=SimpleNamespace(runtime_payload=lambda: {"web": {}}))
    arguments = SimpleNamespace()

    first = cloud_cli._running_tasks_before_deploy(
        arguments,
        client=None,  # type: ignore[arg-type]
        application={"id": "scanner-agent", "current_deployment_id": None},
        project=project,  # type: ignore[arg-type]
    )
    assert first is None

    monkeypatch.setattr(cloud_cli, "_requested_candidate", lambda arguments: {"id": "candidate"})
    candidate = cloud_cli._running_tasks_before_deploy(
        arguments,
        client=None,  # type: ignore[arg-type]
        application={"id": "scanner-agent", "current_deployment_id": "dep_1"},
        project=project,  # type: ignore[arg-type]
    )
    assert candidate is None


def test_a_running_agent_without_an_address_is_unavailable_not_safe() -> None:
    report = deploy_tasks.read_running_tasks(
        FakeCloud(service={"status": "running", "cayu_url": None}),  # type: ignore[arg-type]
        "scanner-agent",
    )

    assert report["status"] == "unavailable" and report["reason"] == "no_address"
    notice = deploy_tasks.unavailable_notice(report)
    assert notice is not None and "serves no Cayu address" in notice


def test_a_task_moving_from_claimed_to_running_between_reads_is_listed_once() -> None:
    task = _task("scan-1", "claimed")
    reads: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        status = request.url.params["status"]
        reads.append(status)
        # It is claimed at the first read and running by the second.
        return httpx.Response(200, json=[{**task, "status": status}])

    report = deploy_tasks.read_running_tasks(
        FakeCloud(),  # type: ignore[arg-type]
        "scanner-agent",
        http=httpx.Client(transport=httpx.MockTransport(handle)),
    )

    assert reads == ["claimed", "running"]
    assert [(row["id"], row["status"]) for row in report["tasks"]] == [("scan-1", "running")]
