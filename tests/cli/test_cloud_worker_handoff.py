"""What `cayu cloud` says about the tasks a deploy hands from the old worker to the new one."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.cli.test_cloud_session_choice import (
    _DEPLOYMENT,
    _SESSION_REFUSAL,
    _failure_envelope,
    _PublicationClient,
    _wait_for_service,
)

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_auth import CloudAuthError
from cayu.cli._cloud_worker_handoff import (
    drain_warnings,
    interrupted_tasks_warning,
    task_check_unsupported_notice,
)

NOW = datetime(2026, 10, 29, 3, 15, tzinfo=UTC)
DRAIN = {
    "release_id": "dep_old",
    "replaced_by": "dep_new",
    "requested_at": "2026-10-29T02:57:00+00:00",
    "deadline_at": "2026-10-29T05:57:00+00:00",
    "instances": ["arn:aws:ecs:us-east-1:123456789012:task/agents/0f3c"],
}
SCAN = {
    "id": "scan-7f2",
    "type": "scout_run",
    "title": "Nightly scan",
    "status": "running",
    "worker_id": "worker-a",
    "outcome": "interrupted",
    "blocking": True,
    "acknowledged": True,
    "updated_at": "2026-10-29T01:00:00+00:00",
    "lease_expires_at": "2026-10-29T03:20:00+00:00",
}


def test_a_drain_is_named_with_its_deadline_and_the_time_left() -> None:
    assert drain_warnings({"worker_drains": [DRAIN]}, now=NOW) == [
        "Draining: release dep_old's worker (1 task on the platform) keeps running to finish "
        "the tasks it holds until 2026-10-29T05:57:00+00:00 (2 h 42 min left), while release "
        "dep_new's worker takes new ones."
    ]
    assert drain_warnings({"worker_drains": [DRAIN]}, replaced_by="dep_other", now=NOW) == []
    # An older Cloud sends no drains; a malformed entry is skipped.
    assert drain_warnings({}, now=NOW) == []
    assert drain_warnings({"worker_drains": [{"replaced_by": "x"}, "x"]}, now=NOW) == []


def test_interrupted_tasks_and_a_cloud_without_the_task_check_are_reported() -> None:
    interrupted = {
        "session_preflight": {"tasks_read": "complete", "interrupted_tasks": ["scan-7f2"]}
    }
    assert interrupted_tasks_warning(interrupted) == (
        "1 task the previous worker held was interrupted by this release, as acknowledged: "
        "scan-7f2. Cayu fences each one until the Agent settles it; `cayu inspect "
        "--fenced-tasks` lists them."
    )
    assert interrupted_tasks_warning({"session_preflight": {"interrupted_tasks": []}}) is None
    assert interrupted_tasks_warning({}) is None

    running = {"status": "read", "tasks": [{"id": "scan-7f2"}], "complete": True}
    older = {"session_preflight": {"state": "clear", "read": "complete"}}
    notice = task_check_unsupported_notice(older, running)
    assert notice is not None and "doesn't check the tasks" in notice
    assert task_check_unsupported_notice(interrupted, running) is None
    assert task_check_unsupported_notice(older, {"status": "read", "tasks": []}) is None


def test_a_held_tasks_refusal_prints_the_retry_that_interrupts_them(capsys) -> None:
    refusal = {
        **_SESSION_REFUSAL,
        "acknowledge_sessions": ["scan-7f2"],
        "code": "held_tasks",
        "message": "1 task the serving release's worker holds would be interrupted by this Release.",
        "detail": "The serving release's worker holds 1 task: scan-7f2 (scout_run, running).",
    }
    with pytest.raises(CloudApiError) as raised:
        _wait_for_service(_PublicationClient(refusal))
    error = _failure_envelope(raised, capsys)

    base = f"cayu cloud deployment %s {_DEPLOYMENT} --application ready-agent"
    retry = base % "retry" + " --acknowledge-session scan-7f2"
    assert error["code"] == "held_tasks"
    assert error["commands"]["retry"] == retry
    assert error["commands"]["retry_wait"] == base % "retry" + " --session-policy wait"
    assert (
        f"Let the tasks finish and retry, or accept that they are interrupted with `{retry}`"
        in (error["message"])
    )
    assert "wait for them again with" in error["message"]


EXITS = {
    "application_id": "scout-agent",
    "looked_at": "2026-10-29T03:14:00Z",
    "exits": [
        {
            "instance": "arn:aws:ecs:us-east-1:123456789012:task/agents/0f3c",
            "state": "stopped",
            "stopped_at": "2026-10-29T03:10:00Z",
            "reason": "Essential container in task exited",
            "exit_code": 0,
        }
    ],
}


class _StatusCloud:
    def __init__(
        self,
        *,
        release: dict[str, Any],
        service: dict[str, Any] | Exception,
        exits: dict[str, Any] | Exception = EXITS,
    ) -> None:
        self.release = release
        self.service = service
        self.exits = exits

    def request(self, method: str, path: str, **kwargs: object) -> dict[str, Any]:
        del method, kwargs
        if path == "/v1/applications":
            return {"items": [{"id": "scout-agent", "name": "Scout Agent"}]}
        if path.endswith("/worker-exits"):
            if isinstance(self.exits, Exception):
                raise self.exits
            return self.exits
        if path.endswith("/service"):
            if isinstance(self.service, Exception):
                raise self.service
            return self.service
        if "/deployments/" in path:
            return self.release
        raise AssertionError(path)


def _run(monkeypatch, capsys, client, *argv: str) -> dict[str, Any]:
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: client)
    assert main(["cloud", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_service_status_lists_each_drain(monkeypatch, capsys) -> None:
    service = {
        "application_id": "scout-agent",
        "deployment_id": "dep_new",
        "issues": [],
        "status": "running",
        "worker_drain_seconds": 10800,
        "worker_drains": [DRAIN],
    }

    output = _run(
        monkeypatch,
        capsys,
        _StatusCloud(release={}, service=service),
        "service",
        "status",
        "--application",
        "scout-agent",
    )

    (warning,) = output["warnings"]
    assert warning.startswith("Draining: release dep_old's worker (1 task on the platform)")
    assert output["result"]["worker_drains"] == [DRAIN]
    assert output["result"]["worker_exits"] == {"status": "read", **EXITS}


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (CloudApiError("api_request_rejected", "HTTP 404.", status_code=404), "unsupported"),
        (CloudApiError("api_unavailable", "Cayu Cloud API is unavailable."), "unavailable"),
    ],
)
def test_service_status_says_when_worker_exits_cant_be_read(
    monkeypatch, capsys, failure: CloudApiError, status: str
) -> None:
    service = {"application_id": "scout-agent", "deployment_id": "dep_new", "status": "running"}

    output = _run(
        monkeypatch,
        capsys,
        _StatusCloud(release={}, service=service, exits=failure),
        "service",
        "status",
        "--application",
        "scout-agent",
    )

    assert output["result"]["worker_exits"]["status"] == status
    assert output["result"]["status"] == "running"
    if status == "unsupported":
        assert "doesn't report worker exits yet" in output["result"]["worker_exits"]["message"]


def test_deployment_status_reports_interrupted_tasks_and_the_drain_it_started(
    monkeypatch, capsys
) -> None:
    release = {
        "id": "dep_new",
        "status": "promoted",
        "session_preflight": {
            "state": "acknowledged",
            "tasks_read": "complete",
            "tasks": [SCAN],
            "interrupted_tasks": ["scan-7f2"],
        },
    }
    service = {"deployment_id": "dep_new", "status": "running", "worker_drains": [DRAIN]}

    output = _run(
        monkeypatch,
        capsys,
        _StatusCloud(release=release, service=service),
        "deployment",
        "status",
        "dep_new",
        "--application",
        "scout-agent",
    )

    interrupted, draining = output["warnings"]
    assert interrupted.startswith("1 task the previous worker held was interrupted")
    assert draining.startswith("Draining: release dep_old's worker")

    # Without the service (or with a login that can't be refreshed) the Release's own status
    # stands.
    for unavailable in (
        CloudApiError("api_unavailable", "Cayu Cloud API is unavailable."),
        CloudAuthError("login_refresh_unavailable", "The login could not be refreshed."),
    ):
        output = _run(
            monkeypatch,
            capsys,
            _StatusCloud(release=release, service=unavailable),
            "deployment",
            "status",
            "dep_new",
            "--application",
            "scout-agent",
        )
        assert [line[:20] for line in output["warnings"]] == ["1 task the previous "]


def test_a_published_deploy_reports_what_its_check_did_with_the_tasks_read_before() -> None:
    running = {
        "status": "read",
        "tasks": [{"id": "scan-7f2", "type": "scout_run", "status": "running"}],
        "complete": True,
    }
    release = {
        "id": "dep_new",
        "status": "promoted",
        "session_preflight": {"tasks_read": "complete", "interrupted_tasks": ["scan-7f2"]},
    }
    published = {"deployment_id": "dep_new", "status": "running", "worker_drains": [DRAIN]}

    def warnings(client: Any, **changes: Any) -> list[str]:
        return cloud_cli._deploy_handoff_warnings(
            client,
            **{
                "application_id": "scout-agent",
                "deployment": {"id": "dep_new", "status": "promoted"},
                "service": published,
                "running_tasks": running,
                "reread": True,
                "waited": True,
                "replaced": True,
                **changes,
            },
        )

    listed, interrupted, draining = warnings(_StatusCloud(release=release, service={}))
    assert listed.startswith("1 task is claimed or running")
    assert interrupted.startswith("1 task the previous worker held was interrupted")
    assert draining.startswith("Draining: release dep_old's worker")
    # Not waited for (--no-wait), or a rerun that found the release already serving: only the
    # tasks read before the upload.
    assert warnings(_StatusCloud(release=release, service={}), waited=False) == [listed]
    assert warnings(_StatusCloud(release=release, service={}), replaced=False) == [listed]

    for failure in (
        CloudApiError("api_unavailable", "Cayu Cloud API is unavailable."),
        CloudAuthError("login_refresh_unavailable", "The login could not be refreshed."),
    ):

        class Unreadable(_StatusCloud):
            def request(
                self, method: str, path: str, failure=failure, **kwargs: object
            ) -> dict[str, Any]:
                raise failure

        # The deploy already succeeded: a failed re-read of the release only loses its warnings.
        assert warnings(Unreadable(release=release, service={}))[0] == listed


def test_a_deploy_the_cli_promotes_names_the_tasks_it_interrupted(tmp_path, monkeypatch) -> None:
    from tests.cli.test_cloud_session_choice import _promoted_rerun

    from cayu.cli import _cloud_deploy_tasks as deploy_tasks

    running = {
        "status": "read",
        "tasks": [{"id": "scan-7f2", "type": "scout_run", "status": "running"}],
        "complete": True,
    }
    monkeypatch.setattr(deploy_tasks, "read_running_tasks", lambda *args, **kwargs: running)
    preflight = {
        "state": "acknowledged",
        "policy": "proceed",
        "read": "complete",
        "sessions": [],
        "truncated": False,
        "tasks_read": "complete",
        "tasks": [SCAN],
        "interrupted_tasks": ["scan-7f2"],
    }
    _client, deploy = _promoted_rerun(
        tmp_path, "--acknowledge-session", "scan-7f2", on_promote={"session_preflight": preflight}
    )

    warnings = deploy()["warnings"]

    assert any(
        line.startswith("1 task the previous worker held was interrupted") for line in warnings
    )
