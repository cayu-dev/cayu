"""`cayu cloud deploy` and `cayu cloud deployment wait --live` wait until the Agent's service runs
the release, not only until Cayu Cloud promotes it.

Cloud selects a release when it promotes it, then migrates the Agent's database and updates its
service; until that finishes, the service still reports the previous release. These tests replay
that sequence with a Cloud whose service keeps reporting the previous release for a while.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cayu.cli import _cloud_project as cloud_project
from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_evidence import EvidenceRecorder
from cayu.cli.cloud import CloudCommandError

_APP = "/v1/applications/live-agent"
_MANIFEST = """
schema_version = 2
application = "live-agent"
name = "Live Agent"
version = "0.2.0"
entrypoint = "python -m agent"
capabilities = ["model.generate"]
cpu_millis = 512
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
[web]
command = "python -m agent.web"
port = 8000
"""
_PREVIOUS = {"deployment_id": "dep_old", "issues": [], "status": "running"}
_STARTING = {"deployment_id": "dep_new", "issues": [], "status": "starting"}
_RUNNING = {"deployment_id": "dep_new", "issues": [], "status": "running"}
# Cloud's refusal of the selected release for the serving release's unfinished sessions.
_REFUSAL = {
    "acknowledge_breaking": [],
    "acknowledge_sessions": ["s-input"],
    "code": "unfinished_sessions",
    "detail": (
        "s-input is paused on a pending user_input bound to the serving release. Nothing was "
        "changed; the previous release keeps serving."
    ),
    "hint": (
        "Answer or finish them and retry, or accept that they may not resume: in the Cayu "
        "Cloud portal open this Release and choose Acknowledge and retry (or Retry and wait)."
    ),
    "message": "1 unfinished session of the serving release may not resume on this Release.",
    "retryable": True,
}
# Cloud keeps the refused release promoted and selected; only `publication_error` says so.
_REFUSED = {
    "status": "promoted",
    "publication_error": _REFUSAL,
    "session_preflight": {"state": "blocked", "truncated": False},
}


class SwitchingCloud:
    """Cloud that has promoted `dep_new` while the Agent's service still runs `dep_old`.

    ``services`` and ``releases`` are Cloud's answers in order; the last one stands once the
    others are used up.
    """

    def __init__(
        self,
        services: list[dict[str, Any] | CloudApiError | None],
        releases: list[dict[str, Any]] | None = None,
        *,
        selected: str = "dep_new",
        selection: str = "complete",
    ) -> None:
        self.services = list(services)
        self.releases = list(releases or [{"status": "promoted"}])
        self.selected = selected
        # The state of the release's timeline step "release_selected".
        self.selection = selection
        self.requests: list[tuple[str, str]] = []

    @staticmethod
    def _next(answers: list[Any]) -> Any:
        return answers.pop(0) if len(answers) > 1 else answers[0]

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.requests.append((method, path))
        if path == "/v1/applications":
            # Nothing was selected when the deploy read the Agent, so it reads no tasks.
            agent = {"id": "live-agent", "name": "Live Agent", "revision": 4}
            return {"items": [{**agent, "current_deployment_id": None}]}
        if path == _APP:
            return {"current_deployment_id": self.selected, "id": "live-agent", "revision": 5}
        if method == "POST" and path == f"{_APP}/deployments":
            return {"id": "dep_new", "status": "accepted"}
        if method == "GET" and path == f"{_APP}/deployments/dep_new":
            return {"id": "dep_new", **self._next(self.releases)}
        if method == "GET" and path == f"{_APP}/deployments/dep_new/timeline":
            return {"items": [{"key": "release_selected", "state": self.selection}]}
        if method == "GET" and path == f"{_APP}/service":
            service = self._next(self.services)
            if service is None:
                raise CloudApiError("api_request_rejected", "Not found.", status_code=404)
            if isinstance(service, CloudApiError):
                raise service
            return service
        raise AssertionError(f"unexpected Cloud request: {method} {path}")

    def service_reads(self) -> int:
        return self.requests.count(("GET", f"{_APP}/service"))


def _clock() -> Any:
    ticks = iter(range(0, 1_000_000, 5))
    return lambda: float(next(ticks))


def _deploy(
    client: SwitchingCloud,
    tmp_path: Path,
    *,
    no_wait_live: bool = False,
    wait_seconds: float = 300.0,
) -> dict[str, Any]:
    return cloud_cli._deploy(
        SimpleNamespace(
            application=None,
            no_promote=False,
            no_wait=False,
            no_wait_live=no_wait_live,
            poll_seconds=5.0,
            wait_seconds=wait_seconds,
        ),
        client=client,  # type: ignore[arg-type]
        recorder=EvidenceRecorder(tmp_path / "evidence"),
        project=cloud_project.ResolvedCloudProject(
            root=None,
            manifest_path=tmp_path / "cayu-cloud.toml",
            manifest=cloud_project.CloudProjectManifest.loads(_MANIFEST),
            repository="https://github.com/example/live-agent",
            revision="c" * 40,
        ),
        sleep=lambda _: None,
        monotonic=_clock(),
    )


def _wait(client: SwitchingCloud, *, live: bool = True) -> dict[str, Any]:
    return cloud_cli._deployment(
        SimpleNamespace(
            application="live-agent",
            deployment_command="wait",
            deployment_id="dep_new",
            live=live,
            poll_seconds=5.0,
            wait_seconds=300.0,
        ),
        client=client,  # type: ignore[arg-type]
        sleep=lambda _: None,
        monotonic=_clock(),
    )


def _error(raised: pytest.ExceptionInfo[CloudApiError], capsys: pytest.CaptureFixture[str]) -> Any:
    assert cloud_cli._cloud_failure(raised.value) == 2
    return json.loads(capsys.readouterr().out)["error"]


def test_deploy_returns_once_the_service_runs_the_new_release(tmp_path: Path) -> None:
    client = SwitchingCloud([_PREVIOUS, _PREVIOUS, _PREVIOUS, _STARTING, _RUNNING])

    result = _deploy(client, tmp_path)["result"]

    assert result["deployment"]["status"] == "promoted"
    assert result["service"] == _RUNNING
    assert result["service_publication_pending"] is False
    assert client.service_reads() == 5


def test_no_wait_live_returns_once_the_release_is_promoted(tmp_path: Path) -> None:
    client = SwitchingCloud([_PREVIOUS, _RUNNING])

    result = _deploy(client, tmp_path, no_wait_live=True)["result"]

    assert result["deployment"]["status"] == "promoted"
    assert result["service"] == _PREVIOUS
    assert result["service_publication_pending"] is True
    assert client.service_reads() == 1


def test_no_wait_live_returns_while_cloud_holds_the_promoted_release_for_sessions(
    tmp_path: Path,
) -> None:
    held = {
        "status": "promoted",
        "session_preflight": {
            "state": "waiting",
            "message": "Waiting for 1 unfinished session of the serving release.",
            "waiting_for": "sessions",
            "checked_at": "2026-10-28T18:00:00Z",
            "deadline_at": "2026-10-28T18:15:00Z",
        },
    }
    client = SwitchingCloud([_PREVIOUS], [held])

    result = _deploy(client, tmp_path, no_wait_live=True)["result"]

    assert result["deployment"]["status"] == "promoted"
    assert result["service_publication_pending"] is True
    assert client.requests.count(("GET", f"{_APP}/deployments/dep_new")) == 1


def test_no_wait_live_is_a_deploy_option() -> None:
    parsed = cloud_cli._build_parser().parse_args(["deploy", "--no-wait-live"])

    assert parsed.no_wait_live is True
    assert parsed.no_wait is False


def test_deploy_reports_a_selected_release_cloud_refused_to_publish_and_the_way_past_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = SwitchingCloud([_PREVIOUS], [_REFUSED])

    with pytest.raises(CloudApiError) as raised:
        _deploy(client, tmp_path)

    error = _error(raised, capsys)
    assert error["category"] == "service_publication_failed"
    assert error["code"] == "unfinished_sessions"
    assert error["publication_error"]["message"] == _REFUSAL["message"]
    retry = "cayu cloud deployment retry dep_new --application live-agent"
    assert error["commands"]["retry"] == f"{retry} --acknowledge-session s-input"
    assert error["commands"]["retry_wait"] == f"{retry} --session-policy wait"
    assert error["message"].startswith(f"{_REFUSAL['message']} {_REFUSAL['detail']} ")
    assert f"`{retry} --acknowledge-session s-input`" in error["message"]


def test_deploy_wait_ends_while_the_service_still_runs_the_previous_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = SwitchingCloud([_PREVIOUS])

    with pytest.raises(CloudApiError) as raised:
        _deploy(client, tmp_path, wait_seconds=30.0)

    error = _error(raised, capsys)
    assert error["category"] == "service_still_starting"
    assert error["status"] == "starting"
    assert error["waited_seconds"] >= 30
    assert error["commands"] == {"status": "cayu cloud service status --application live-agent"}


def test_deployment_wait_live_waits_until_the_service_runs_the_release() -> None:
    client = SwitchingCloud(
        [_PREVIOUS, _PREVIOUS, _STARTING, _RUNNING],
        # Ready once smoke tested; Cloud selects it while the CLI waits for the service.
        [{"status": "smoke_tested"}, {"status": "smoke_tested"}, {"status": "promoted"}],
    )

    answer = _wait(client)

    assert answer["operation"] == "deployment.wait"
    assert answer["service"] == _RUNNING
    assert answer["result"] == {"id": "dep_new", "status": "promoted"}
    assert client.service_reads() == 4


def test_deployment_wait_without_live_returns_once_the_release_is_ready() -> None:
    client = SwitchingCloud([_PREVIOUS])

    answer = _wait(client, live=False)

    assert answer == {
        "operation": "deployment.wait",
        "result": {"id": "dep_new", "status": "promoted"},
    }
    assert client.service_reads() == 0


def test_deployment_wait_live_accepts_a_running_release_whose_web_sleeps() -> None:
    asleep = {**_RUNNING, "status": "sleeping"}
    client = SwitchingCloud([asleep])

    assert _wait(client)["service"] == asleep


def test_deployment_wait_live_accepts_a_sleeping_release_after_a_failed_service_read() -> None:
    asleep = {**_RUNNING, "status": "sleeping"}
    unavailable = CloudApiError("api_request_rejected", "Unavailable.", status_code=503)
    client = SwitchingCloud([unavailable, asleep])

    assert _wait(client)["service"] == asleep
    assert client.service_reads() == 2


def test_deployment_wait_live_retries_a_failed_read_before_judging_the_selection() -> None:
    # dep_new still serves while the Agent already selects a later release.
    unavailable = CloudApiError("api_request_rejected", "Unavailable.", status_code=503)
    client = SwitchingCloud([unavailable, _RUNNING], selected="dep_later")

    assert _wait(client)["service"] == _RUNNING
    assert client.service_reads() == 2


def test_deployment_wait_live_returns_for_a_release_without_a_service() -> None:
    client = SwitchingCloud([None], [{"status": "promoted", "manifest": {"runtime": None}}])

    answer = _wait(client)

    assert answer["service"] is None
    assert client.service_reads() == 0


def test_deployment_wait_live_follows_a_promoted_candidate_while_cloud_selects_it() -> None:
    candidate = {"status": "smoke_tested", "stage_only": True, "candidate": {}}
    client = SwitchingCloud(
        [_PREVIOUS, _PREVIOUS, _RUNNING],
        [candidate, candidate, {"status": "promoted"}],
        selected="dep_old",
        selection="active",
    )

    assert _wait(client)["service"] == _RUNNING


def test_deployment_wait_live_treats_an_omitted_runtime_as_no_service() -> None:
    client = SwitchingCloud(
        [None], [{"status": "promoted", "manifest": {"entrypoint": "python -m agent"}}]
    )

    assert _wait(client)["service"] is None
    assert client.service_reads() == 0


def test_deployment_wait_live_reports_a_refused_publication(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = SwitchingCloud(
        [_PREVIOUS],
        [{"status": "promoted"}, _REFUSED],
    )

    with pytest.raises(CloudApiError) as raised:
        _wait(client)

    error = _error(raised, capsys)
    assert error["category"] == "service_publication_failed"
    assert error["commands"]["retry"] == (
        "cayu cloud deployment retry dep_new --application live-agent --acknowledge-session s-input"
    )


def test_deployment_wait_live_does_not_wait_for_a_release_the_agent_no_longer_selects(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = SwitchingCloud([{**_PREVIOUS, "deployment_id": "dep_later"}], selected="dep_later")

    with pytest.raises(CloudApiError) as raised:
        _wait(client)

    error = _error(raised, capsys)
    assert error["category"] == "release_not_selected"
    assert error["message"] == (
        "Release dep_new was promoted, but the Agent's selected Release is dep_later, so the "
        "Agent's service won't run it. Select it again with `cayu cloud rollback`."
    )
    assert error["commands"]["rollback"] == (
        "cayu cloud rollback dep_new --application live-agent --wait"
    )
    assert client.service_reads() == 1


def test_deployment_wait_live_does_not_wait_for_a_staged_release(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = SwitchingCloud([_PREVIOUS], [{"status": "smoke_tested", "stage_only": True}])

    with pytest.raises(CloudCommandError) as raised:
        _wait(client)

    assert cloud_cli._cloud_failure(raised.value) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "release_not_promoted"
    assert "`cayu cloud deployment promote`" in error["message"]


def test_deployment_wait_live_from_the_command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    client = SwitchingCloud([_RUNNING])
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: client)

    assert (
        main(["cloud", "deployment", "wait", "dep_new", "--application", "live-agent", "--live"])
        == 0
    )

    answer = json.loads(capsys.readouterr().out)
    assert answer["ok"] is True
    assert answer["service"] == _RUNNING
