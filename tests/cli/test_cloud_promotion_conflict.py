"""`cayu cloud deploy` when Cayu Cloud answers the CLI's promote with HTTP 409.

Cloud's deployment worker promotes every smoke-tested release itself. The CLI's promote
carries the Agent revision it read before uploading, so when another release (typically
the previous one, still finishing) is promoted in between, Cloud rejects the CLI's
promote with "Application changed after the expected revision." and then promotes the
new release anyway. These tests reproduce that sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api
from cayu.cli import _cloud_project as cloud_project
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_evidence import EvidenceRecorder

_STALE_REVISION = "Application changed after the expected revision."

_MANIFEST = """
schema_version = 2
application = "staging-smoke"
name = "Staging Smoke"
version = "0.1.0"
entrypoint = "python -m smoke"
capabilities = ["model.generate"]
cpu_millis = 512
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
[web]
command = "python -m smoke.web"
port = 8000
"""


def _project(tmp_path: Path) -> cloud_project.ResolvedCloudProject:
    return cloud_project.ResolvedCloudProject(
        root=None,
        manifest_path=tmp_path / "cayu-cloud.toml",
        manifest=cloud_project.CloudProjectManifest.loads(_MANIFEST),
        repository="https://github.com/example/staging-smoke",
        revision="b" * 40,
    )


class RacingCloud:
    """Cloud where the previous release is promoted while the new one is building."""

    def __init__(self, statuses_after_conflict: list[str]) -> None:
        # The previous release's late promotion moved the Agent from revision 4 to 5.
        self.revision = 5
        self.current = "dep_previous"
        self.statuses_after_conflict = statuses_after_conflict
        self.conflicted = False
        self.requests: list[tuple[str, str]] = []

    def application(self) -> dict[str, object]:
        return {
            "current_deployment_id": self.current,
            "id": "staging-smoke",
            "name": "Staging Smoke",
            "revision": self.revision,
        }

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path))
        if path == "/v1/applications":
            # Read before the upload, while the previous release was still finishing.
            return {"items": [{**self.application(), "current_deployment_id": None, "revision": 4}]}
        if path == "/v1/applications/staging-smoke":
            return self.application()
        if method == "POST" and path.endswith("/deployments"):
            return {"id": "dep_second", "status": "accepted"}
        if method == "POST" and path.endswith("/dep_second/promote"):
            assert kwargs["payload"] == {"expected_application_revision": 4}
            self.conflicted = True
            raise CloudApiError(
                "api_request_rejected",
                f"Cayu Cloud API returned HTTP 409: {_STALE_REVISION}",
                status_code=409,
                detail=_STALE_REVISION,
            )
        if path.endswith("/deployments/dep_second"):
            if not self.conflicted:
                return {"id": "dep_second", "status": "smoke_tested"}
            status = self.statuses_after_conflict.pop(0)
            if status == "promoted":
                # The deployment worker promotes with the revision it reads itself.
                self.revision, self.current = 6, "dep_second"
            return {"id": "dep_second", "status": status}
        if path.endswith("/deployments/dep_second/timeline"):
            return {"failure": None, "items": []}
        if method == "GET" and path.endswith("/service"):
            return {"deployment_id": self.current, "issues": [], "status": "running"}
        raise AssertionError(f"unexpected Cloud request: {method} {path}")


def _deploy(client: RacingCloud, tmp_path: Path, *, no_wait: bool = False) -> dict[str, Any]:
    clock = iter(range(0, 10_000, 5))
    return cloud_cli._deploy(
        SimpleNamespace(
            application=None,
            no_promote=False,
            no_wait=no_wait,
            poll_seconds=5.0,
            wait_seconds=60.0,
        ),
        client=client,  # type: ignore[arg-type]
        recorder=EvidenceRecorder(tmp_path / "evidence"),
        project=_project(tmp_path),
        sleep=lambda _: None,
        monotonic=lambda: float(next(clock)),
    )


def test_deploy_keeps_waiting_when_cloud_promotes_the_release_after_a_promote_conflict(
    tmp_path: Path,
) -> None:
    client = RacingCloud(["smoke_tested", "smoke_tested", "promoted"])

    result = _deploy(client, tmp_path)["result"]

    assert result["deployment"] == {"id": "dep_second", "status": "promoted"}
    assert result["application"]["current_deployment_id"] == "dep_second"
    assert result["application"]["revision"] == 6
    assert result["service"]["deployment_id"] == "dep_second"
    assert result["service_publication_pending"] is False
    assert ("POST", "/v1/applications/staging-smoke/deployments/dep_second/promote") in (
        client.requests
    )


def test_no_wait_deploy_accepts_a_release_cloud_already_promoted(tmp_path: Path) -> None:
    client = RacingCloud(["promoted"])
    original = client.request

    def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        # A repeated deploy replays the release, which is already smoke tested.
        if method == "POST" and path.endswith("/deployments"):
            client.requests.append((method, path))
            return {"id": "dep_second", "status": "smoke_tested"}
        return original(method, path, **kwargs)

    client.request = request  # type: ignore[method-assign]

    result = _deploy(client, tmp_path, no_wait=True)["result"]

    assert result["deployment"]["status"] == "promoted"
    assert result["application"]["current_deployment_id"] == "dep_second"


def test_promote_conflict_that_cloud_never_resolves_reports_cloud_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = RacingCloud(["smoke_tested"] * 100)

    with pytest.raises(CloudApiError) as raised:
        _deploy(client, tmp_path)

    assert cloud_cli._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "deployment_promotion_conflict"
    assert error["message"] == (
        "Cayu Cloud refused to promote this release and has not promoted it. "
        f"Cayu Cloud said: {_STALE_REVISION}"
    )
    assert error["deployment_id"] == "dep_second"
    assert error["status"] == "smoke_tested"
    assert error["commands"]["promote"] == (
        "cayu cloud deployment promote dep_second --application staging-smoke"
    )


def test_promote_conflict_followed_by_cancellation_reports_the_terminal_release(
    tmp_path: Path,
) -> None:
    client = RacingCloud(["cancelled"])

    with pytest.raises(CloudApiError) as raised:
        _deploy(client, tmp_path)

    assert raised.value.category == "deployment_failed"
    assert raised.value.details["status"] == "cancelled"  # type: ignore[attr-defined]


def test_promote_rejections_other_than_conflicts_are_not_waited_out(tmp_path: Path) -> None:
    class Forbidden(RacingCloud):
        def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if method == "POST" and path.endswith("/promote"):
                raise CloudApiError("api_request_rejected", "denied", status_code=403)
            return super().request(method, path, **kwargs)

    with pytest.raises(CloudApiError) as raised:
        _deploy(Forbidden([]), tmp_path)

    assert raised.value.status_code == 403


def _client_response(
    monkeypatch: pytest.MonkeyPatch, status_code: int, payload: object
) -> CloudApiError:
    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        return real_client(
            transport=httpx.MockTransport(lambda _: httpx.Response(status_code, json=payload)),
            **kwargs,
        )

    monkeypatch.setattr(_cloud_api.httpx, "Client", client)
    with pytest.raises(CloudApiError) as raised:
        CloudApiClient(api_url="https://cloud.example", api_key="key").request(
            "POST", "/v1/applications/staging-smoke/deployments/dep_second/promote"
        )
    return raised.value


@pytest.mark.parametrize(
    "reason",
    [
        _STALE_REVISION,
        "Agent deployment has not passed the required release gate.",
        "Agent deployment has a pending lifecycle action.",
        "Application already has a deployment with this version.",
    ],
)
def test_conflict_message_carries_cloud_plain_text_reason(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    error = _client_response(monkeypatch, 409, {"detail": reason})

    assert str(error) == f"Cayu Cloud API returned HTTP 409: {reason}"
    assert error.detail == reason


def test_conflict_message_carries_a_structured_reason_with_a_cloud_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _client_response(
        monkeypatch,
        409,
        {"detail": {"code": "archive_exit_not_blocking", "message": "The exit is not blocking."}},
    )

    assert str(error) == "Cayu Cloud API returned HTTP 409: The exit is not blocking."
    assert error.code is None


@pytest.mark.parametrize(
    "status_code,payload",
    [
        (409, {"detail": "token: leaked-value"}),
        (409, {"detail": "x" * 513}),
        (409, {"detail": {"code": "Not-A-Code", "message": "Free text."}}),
        (409, {"detail": {"code": "some_conflict", "message": "password=hunter2"}}),
        (400, {"detail": "Some other rejection."}),
    ],
)
def test_unsafe_or_non_conflict_details_stay_hidden(
    monkeypatch: pytest.MonkeyPatch, status_code: int, payload: object
) -> None:
    error = _client_response(monkeypatch, status_code, payload)

    assert str(error) == f"Cayu Cloud API returned HTTP {status_code}."


def test_documented_conflict_codes_keep_their_documented_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _client_response(
        monkeypatch,
        409,
        {"detail": {"code": "application_archived", "message": "server wording"}},
    )

    assert str(error) == (
        "Cayu Cloud API returned HTTP 409: The Agent is archived; it can't be deployed or changed."
    )
    assert error.code == "application_archived"
