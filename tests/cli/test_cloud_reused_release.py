"""`cayu cloud deploy` of unchanged source that Cayu Cloud resolves to an earlier Release.

Cloud answers an unchanged-source deploy with the Release it already built. When that
Release was promoted long ago, no promotion finalizer is going to publish it again, so
the deploy must either start the Agent service itself or say why it can't, instead of
polling a missing service until the wait expires (cayu#2256).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cayu.cli import _cloud_project as cloud_project
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_evidence import EvidenceRecorder

_MANIFEST = """
schema_version = 2
application = "smoke-agent"
name = "Smoke Agent"
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
        repository="https://github.com/example/smoke-agent",
        revision="c" * 40,
    )


def _not_found() -> CloudApiError:
    return CloudApiError(
        "api_request_rejected", "Cayu Cloud API returned HTTP 404.", status_code=404
    )


class ReusingCloud:
    """Cloud whose deployment create replays an earlier Release."""

    def __init__(
        self,
        *,
        replay_status: str = "promoted",
        current: str | None = "dep_existing",
        service: dict[str, Any] | None = None,
        starting_reads: int = 1,
    ) -> None:
        self.replay_status = replay_status
        self.current = current
        self.service = service
        self.starting_reads = starting_reads
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path))
        if path == "/v1/applications":
            return {
                "items": [
                    {
                        "current_deployment_id": self.current,
                        "id": "smoke-agent",
                        "name": "Smoke Agent",
                        "revision": 7,
                    }
                ]
            }
        if method == "POST" and path.endswith("/deployments"):
            return {"id": "dep_existing", "status": self.replay_status}
        if method == "POST" and path.endswith("/dep_existing/promote"):
            self.replay_status = "promoted"
            self.current = "dep_existing"
            return {"current_deployment_id": "dep_existing", "id": "smoke-agent", "revision": 8}
        if method == "GET" and path.endswith("/deployments/dep_existing"):
            return {"id": "dep_existing", "publication_error": None, "status": self.replay_status}
        if method == "PUT" and path.endswith("/service"):
            # Cloud queues publication of the selected Release.
            self.service = {"deployment_id": self.current, "issues": [], "status": "deploying"}
            return dict(self.service)
        if method == "GET" and path.endswith("/service"):
            if self.service is None:
                raise _not_found()
            if self.service["status"] == "deploying":
                if self.starting_reads:
                    self.starting_reads -= 1
                else:
                    self.service = {**self.service, "status": "running"}
            return dict(self.service)
        raise AssertionError(f"unexpected Cloud request: {method} {path}")

    def count(self, method: str, suffix: str) -> int:
        return sum(1 for m, p in self.requests if m == method and p.endswith(suffix))


def _deploy(
    client: ReusingCloud,
    tmp_path: Path,
    *,
    no_wait: bool = False,
    sleeps: list[float] | None = None,
) -> dict[str, Any]:
    clock = iter(range(0, 10_000, 5))
    return cloud_cli._deploy(
        SimpleNamespace(
            application=None,
            no_promote=False,
            no_wait=no_wait,
            poll_seconds=5.0,
            wait_seconds=600.0,
        ),
        client=client,  # type: ignore[arg-type]
        recorder=EvidenceRecorder(tmp_path / "evidence"),
        project=_project(tmp_path),
        sleep=(lambda _: None) if sleeps is None else sleeps.append,
        monotonic=lambda: float(next(clock)),
    )


def test_reused_selected_release_without_a_service_starts_it(tmp_path: Path) -> None:
    client = ReusingCloud(service=None)

    result = _deploy(client, tmp_path)["result"]

    assert client.count("PUT", "/service") == 1
    assert result["service_publication_requested"] is True
    assert result["service"] == {"deployment_id": "dep_existing", "issues": [], "status": "running"}
    assert result["service_publication_pending"] is False
    assert client.count("POST", "/promote") == 0


def test_reused_selected_release_without_a_service_starts_it_without_waiting(
    tmp_path: Path,
) -> None:
    client = ReusingCloud(service=None)
    sleeps: list[float] = []

    result = _deploy(client, tmp_path, no_wait=True, sleeps=sleeps)["result"]

    assert client.count("PUT", "/service") == 1
    assert result["service_publication_requested"] is True
    assert result["service"]["status"] == "deploying"
    assert result["service_publication_pending"] is False
    assert sleeps == []


def test_reused_release_already_running_is_left_alone(tmp_path: Path) -> None:
    client = ReusingCloud(
        service={"deployment_id": "dep_existing", "issues": [], "status": "running"}
    )

    result = _deploy(client, tmp_path)["result"]

    assert client.count("PUT", "/service") == 0
    assert result["service_publication_requested"] is False
    assert result["service"]["status"] == "running"


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ("dep_newer", "the Agent's selected Release is dep_newer"),
        (None, "the Agent has no selected Release"),
    ],
)
def test_reused_release_that_is_not_selected_fails_promptly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    current: str | None,
    expected: str,
) -> None:
    client = ReusingCloud(
        current=current,
        service=None
        if current is None
        else {"deployment_id": current, "issues": [], "status": "running"},
    )
    sleeps: list[float] = []

    with pytest.raises(CloudApiError) as raised:
        _deploy(client, tmp_path, sleeps=sleeps)

    assert sleeps == []
    assert client.count("PUT", "/service") == 0
    assert cloud_cli._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "release_not_selected"
    assert error["deployment_id"] == "dep_existing"
    assert error["current_deployment_id"] == current
    assert expected in error["message"]
    assert "dep_existing" in error["message"]
    assert error["commands"] == {
        "rollback": "cayu cloud rollback dep_existing --application smoke-agent --wait",
        "status": "cayu cloud service status --application smoke-agent",
    }


def test_freshly_promoted_release_leaves_publication_to_cloud(tmp_path: Path) -> None:
    client = ReusingCloud(replay_status="smoke_tested", current=None, service=None)
    original = client.request

    def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "GET" and path.endswith("/service") and client.service is None:
            client.requests.append((method, path))
            # Cloud's promotion finalizer publishes the service on its own.
            client.service = {"deployment_id": "dep_existing", "issues": [], "status": "deploying"}
            raise _not_found()
        return original(method, path, **kwargs)

    client.request = request  # type: ignore[method-assign]

    result = _deploy(client, tmp_path)["result"]

    assert client.count("POST", "/dep_existing/promote") == 1
    assert client.count("PUT", "/service") == 0
    assert result["service_publication_requested"] is False
    assert result["service"]["status"] == "running"


def test_transient_service_read_keeps_polling_instead_of_publishing(tmp_path: Path) -> None:
    client = ReusingCloud(
        service={"deployment_id": "dep_existing", "issues": [], "status": "running"}
    )
    original = client.request
    unavailable = {"reads": 1}

    def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path.endswith("/service") and (method == "PUT" or unavailable["reads"]):
            client.requests.append((method, path))
            if method == "GET":
                unavailable["reads"] -= 1
            raise CloudApiError(
                "api_request_rejected", "Cayu Cloud API returned HTTP 503.", status_code=503
            )
        return original(method, path, **kwargs)

    client.request = request  # type: ignore[method-assign]

    result = _deploy(client, tmp_path)["result"]

    assert client.count("PUT", "/service") == 0
    assert result["service_publication_requested"] is False
    assert result["service"]["status"] == "running"


def test_promoted_retry_found_from_a_failed_original_is_treated_as_reused(
    tmp_path: Path,
) -> None:
    client = ReusingCloud(current="dep_retry", service=None)
    original = client.request
    family = {
        "manifest_digest": "d" * 64,
        "policy_version": "v1",
    }

    def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and path.endswith("/deployments"):
            client.requests.append((method, path))
            # Unchanged source replays the failed original Release...
            return {"id": "dep_existing", "status": "failed", "version": "0.1.0", **family}
        if method == "GET" and path.endswith("/deployments"):
            client.requests.append((method, path))
            # ...whose retry was promoted and selected by an earlier deploy.
            return {
                "items": [
                    {
                        "created_at": "2026-10-04T00:00:00Z",
                        "id": "dep_retry",
                        "status": "promoted",
                        "version": "0.1.0-retry-1",
                        **family,
                    }
                ],
                "next_cursor": None,
            }
        if method == "GET" and path.endswith("/deployments/dep_retry"):
            client.requests.append((method, path))
            return {"id": "dep_retry", "publication_error": None, "status": "promoted"}
        return original(method, path, **kwargs)

    client.request = request  # type: ignore[method-assign]

    result = _deploy(client, tmp_path)["result"]

    assert result["deployment"]["id"] == "dep_retry"
    assert client.count("PUT", "/service") == 1
    assert result["service_publication_requested"] is True
    assert result["service"]["status"] == "running"
