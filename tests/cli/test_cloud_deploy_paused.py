"""`cayu cloud deploy --paused`: the release starts with its worker and schedules paused."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_reused_release import ReusingCloud, _project
from tests.cli.test_cloud_schedules_pause import _LIST, cloud  # noqa: F401

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_evidence import EvidenceRecorder

_PAUSE = "/v1/applications/smoke-agent/service/pause"
_APP = "/v1/applications/acme-desk"
_AGENTS = {_LIST: httpx.Response(200, json={"items": [{"id": "acme-desk", "name": "Acme Desk"}]})}
_PAUSED = {
    "application_id": "smoke-agent",
    "message": "The Agent's service is not deployed. Its worker is paused.",
    "paused": True,
    "paused_at": "2026-11-25T18:00:00Z",
    "status": "not_deployed",
}


class PausingCloud(ReusingCloud):
    """Cloud that answers the pause `deploy --paused` sends first with ``pause``."""

    def __init__(self, pause: dict[str, Any] | Exception, **options: Any) -> None:
        super().__init__(**options)
        self.pause = pause

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and path == _PAUSE:
            self.requests.append((method, path))
            if isinstance(self.pause, Exception):
                raise self.pause
            return dict(self.pause)
        return super().request(method, path, **kwargs)


def _deploy(
    client: ReusingCloud, tmp_path: Path, *, paused: bool = True, **options: Any
) -> dict[str, Any]:
    clock = iter(range(0, 10_000, 5))
    return cloud_cli._deploy(
        SimpleNamespace(
            **options,
            application=None,
            no_promote=False,
            no_wait=False,
            paused=paused,
            poll_seconds=5.0,
            wait_seconds=600.0,
        ),
        client=client,  # type: ignore[arg-type]
        recorder=EvidenceRecorder(tmp_path / "evidence"),
        project=_project(tmp_path),
        sleep=lambda _: None,
        monotonic=lambda: float(next(clock)),
    )


def test_the_agent_is_paused_before_anything_is_deployed(tmp_path: Path) -> None:
    client = PausingCloud(_PAUSED, service=None)

    result = _deploy(client, tmp_path)["result"]

    posts = [path for method, path in client.requests if method in {"POST", "PUT"}]
    assert posts[0] == _PAUSE
    assert client.count("POST", "/deployments") == 1
    assert result["paused"] == {
        "paused": True,
        "paused_at": "2026-11-25T18:00:00Z",
        "status": "not_deployed",
    }
    assert "cayu cloud service resume --application smoke-agent" in result["message"]


def test_the_resume_it_prints_acts_where_the_deploy_did(tmp_path: Path) -> None:
    client = PausingCloud(_PAUSED, service=None)

    result = _deploy(client, tmp_path, context="staging", intended_organization="example-org")

    assert (
        "cayu cloud --context staging --organization example-org service resume "
        "--application smoke-agent"
    ) in result["result"]["message"]


def test_a_deploy_without_paused_sends_no_pause(tmp_path: Path) -> None:
    client = PausingCloud(AssertionError("no pause"), service=None)

    result = _deploy(client, tmp_path, paused=False)["result"]

    assert client.count("POST", "/service/pause") == 0
    assert "paused" not in result


@pytest.mark.parametrize(
    "refusal",
    [
        # A Cloud that can pause only a deployed service.
        CloudApiError(
            "api_request_rejected",
            "Cayu Cloud API returned HTTP 409: Application service is not deployed.",
            status_code=409,
            detail="Application service is not deployed.",
        ),
        # One without pausing at all.
        CloudApiError("api_request_rejected", "Cayu Cloud API returned HTTP 404.", status_code=404),
        CloudApiError("api_request_rejected", "Cayu Cloud API returned HTTP 405.", status_code=405),
    ],
)
def test_a_cloud_that_cant_hold_the_worker_is_refused_before_any_upload(
    tmp_path: Path, refusal: CloudApiError
) -> None:
    client = PausingCloud(refusal, service=None)

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path)

    assert raised.value.category == "deploy_paused_unsupported"
    assert "Nothing was deployed" in str(raised.value)
    assert client.requests[-1] == ("POST", _PAUSE)
    assert client.count("POST", "/deployments") == 0


def test_an_answer_that_isnt_paused_is_refused(tmp_path: Path) -> None:
    client = PausingCloud({"status": "running", "deployment_id": "dep_existing"}, service=None)

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(client, tmp_path)

    assert raised.value.category == "deploy_paused_unsupported"
    assert client.count("POST", "/deployments") == 0


def test_another_pause_refusal_is_cloud_s_own(tmp_path: Path) -> None:
    busy = CloudApiError(
        "api_request_rejected",
        "Cayu Cloud API returned HTTP 409: Application service is being deleted or is archived.",
        status_code=409,
        detail="Application service is being deleted or is archived.",
    )
    client = PausingCloud(busy, service=None)

    with pytest.raises(CloudApiError) as raised:
        _deploy(client, tmp_path)

    assert raised.value is busy
    assert client.count("POST", "/deployments") == 0


def test_paused_with_a_candidate_is_refused_before_any_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "cloud",
            "deploy",
            "--paused",
            "--no-promote",
            "--evaluated-against",
            "dep_one",
            "--skip-deploy-check",
        ]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert "--paused" in error["message"]


def test_service_status_says_a_pause_waits_for_the_next_deploy(
    cloud,  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    detail = {
        "code": "service_not_deployed",
        "message": "The Agent's service is not deployed.",
        "paused": True,
        "paused_at": "2026-11-25T18:00:00+00:00",
    }
    cloud({**_AGENTS, ("GET", f"{_APP}/service"): httpx.Response(404, json={"detail": detail})})

    code = main(["cloud", "service", "status", "--application", "acme-desk"])

    assert code == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "service_not_deployed"
    assert "paused (since 2026-11-25T18:00:00+00:00)" in error["message"]
    assert "cayu cloud service resume --application acme-desk" in error["message"]


def test_service_status_of_an_agent_without_a_service_or_pause_is_unchanged(
    cloud,  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    cloud(
        {**_AGENTS, ("GET", f"{_APP}/service"): httpx.Response(404, json={"detail": "Not Found"})}
    )

    code = main(["cloud", "service", "status", "--application", "acme-desk"])

    assert code == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] != "service_not_deployed"


def test_the_not_deployed_refusal_parses_from_cloud(
    cloud,  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The 409 an older Cloud answers, as the client reads it.
    cloud(
        {
            **_AGENTS,
            ("POST", f"{_APP}/service/pause"): httpx.Response(
                409, json={"detail": "Application service is not deployed."}
            ),
        }
    )

    code = main(["cloud", "service", "pause", "--application", "acme-desk"])

    assert code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["message"].endswith("Application service is not deployed.")
    with pytest.raises(cloud_cli.CloudCommandError):
        cloud_cli._pause_before_deploy(cloud_cli._cloud_client(), application_id="acme-desk")
