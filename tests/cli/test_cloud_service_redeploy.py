"""`cayu cloud service redeploy`: Cloud's forced publication of the selected Release."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError

_AGENT = "outbound-agent"
_SERVICE = f"/v1/applications/{_AGENT}/service"
_RELEASE = f"/v1/applications/{_AGENT}/deployments/dep_live"
_QUEUED = {"application_id": _AGENT, "deployment_id": "dep_live", "status": "deploying"}


class _Cloud:
    """Cloud's answers to the request and to the Release and Agent reads around it."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | Exception | None = None,
        publish: Callable[[], dict[str, Any]] | None = None,
        agent: dict[str, Any] | None = None,
    ) -> None:
        self.release = release or {"id": "dep_live", "status": "promoted"}
        self.publish = publish or (lambda: dict(_QUEUED))
        self.agent = agent
        self.requests: list[tuple[str, str, dict[str, str] | None]] = []

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((method, path, kwargs.get("query")))
        if path == "/v1/applications":
            return {"items": [{"id": _AGENT, "name": "Outbound Agent"}]}
        if method == "PUT" and path == _SERVICE:
            return self.publish()
        if method == "GET" and path == _RELEASE:
            if isinstance(self.release, Exception):
                raise self.release
            return self.release
        if method == "GET" and path == f"/v1/applications/{_AGENT}":
            if self.agent is None:
                raise CloudApiError(
                    "api_request_rejected", "Cayu Cloud API returned HTTP 404.", status_code=404
                )
            return self.agent
        raise AssertionError(f"unexpected Cloud request: {method} {path}")


def _run(monkeypatch: pytest.MonkeyPatch, cloud: _Cloud, *global_flags: str) -> int:
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    return main(["cloud", *global_flags, "service", "redeploy", "--application", _AGENT])


def test_redeploy_queues_a_forced_publication_and_names_the_next_step(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud = _Cloud()

    assert _run(monkeypatch, cloud) == 0

    streams = capsys.readouterr()
    status = f"cayu cloud service status --application {_AGENT}"
    message = (
        "Cayu Cloud queued a new publication of Release dep_live to the Agent's service; its "
        f"deployment worker publishes it shortly. Run `{status}` to see the service running."
    )
    assert json.loads(streams.out) == {
        "commands": {"status": status},
        "message": message,
        "ok": True,
        "operation": "service.redeploy",
        "result": _QUEUED,
        "status": "queued",
    }
    assert streams.err == f"cayu cloud: {message}\n"
    # One request and no polling: the Release is read only for its declared processes.
    assert cloud.requests[1:] == [
        ("PUT", _SERVICE, {"force": "true"}),
        ("GET", _RELEASE, None),
    ]


def test_redeploy_keeps_the_selected_context_in_its_next_step(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    key = tmp_path / "key"
    key.write_text("private-key\n")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "api_key_file": str(key),
                "api_url": "https://staging.example.test",
                "schema_version": 1,
                "status": "ready",
            }
        )
    )

    assert _run(monkeypatch, _Cloud(), "--context", str(context)) == 0

    assert json.loads(capsys.readouterr().out)["commands"] == {
        "status": f"cayu cloud --context {context} service status --application {_AGENT}",
    }


def test_redeploy_of_a_release_without_processes_warns_that_nothing_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    headless = {"id": "dep_live", "manifest": {"runtime": None}, "status": "promoted"}

    assert _run(monkeypatch, _Cloud(release=headless)) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "queued"
    assert "runs no web process, worker or schedule, so nothing will run" in output["warnings"][0]


@pytest.mark.parametrize("status_code", [403, 503])
def test_redeploy_is_reported_queued_whatever_happens_to_the_release_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status_code: int
) -> None:
    refused = CloudApiError(
        "api_request_rejected",
        f"Cayu Cloud API returned HTTP {status_code}.",
        status_code=status_code,
    )

    assert _run(monkeypatch, _Cloud(release=refused)) == 0

    output = json.loads(capsys.readouterr().out)
    assert (output["status"], "warnings" in output) == ("queued", False)


def _refused(status_code: int, code: str | None = None) -> Callable[[], dict[str, Any]]:
    def refuse() -> dict[str, Any]:
        raise CloudApiError(
            "api_request_rejected",
            f"Cayu Cloud API returned HTTP {status_code}.",
            status_code=status_code,
            code=code,
        )

    return refuse


def test_redeploy_names_an_agent_without_a_selected_release(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud = _Cloud(publish=_refused(404), agent={"id": _AGENT})

    assert _run(monkeypatch, cloud) == 2

    assert json.loads(capsys.readouterr().out)["error"] == {
        "category": "no_selected_release",
        "message": (
            "The Agent has no selected Release to publish; deploy one with `cayu cloud deploy`."
        ),
    }


def test_redeploy_names_an_agent_deleted_after_it_was_found(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(monkeypatch, _Cloud(publish=_refused(404))) == 2

    assert json.loads(capsys.readouterr().out)["error"] == {
        "category": "application_not_found",
        "message": "Cayu Cloud no longer has this Agent.",
    }


def test_redeploy_passes_on_cloud_refusing_an_archived_agent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud = _Cloud(publish=_refused(409, "application_archived"))

    assert _run(monkeypatch, cloud) == 2

    error = json.loads(capsys.readouterr().out)["error"]
    assert (error["category"], error["code"]) == ("api_request_rejected", "application_archived")
    assert ("GET", f"/v1/applications/{_AGENT}", None) not in cloud.requests


def test_redeploy_and_restart_say_which_bindings_they_keep(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for action in ("redeploy", "restart"):
        with pytest.raises(SystemExit):
            main(["cloud", "service", action, "--help"])
        text = " ".join(capsys.readouterr().out.split())
        assert "bindings" in text
