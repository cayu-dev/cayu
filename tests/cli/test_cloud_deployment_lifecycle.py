"""`cayu cloud deployment cancel|destroy|resume`, and the session choice `promote` sends."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import _cloud_deployment_lifecycle, main

cloud = _cloud

_APPS = "/v1/applications?lifecycle=all"
_AGENTS = {"items": [{"id": "support-desk", "name": "Support desk"}]}
_RELEASE = "/v1/applications/support-desk/deployments/dep_1"


def _workflow(**changes: object) -> dict[str, Any]:
    return {
        "application_id": "support-desk",
        "deployment_id": "dep_1",
        "attempt_count": 1,
        "completed_phases": ["image_built"],
        "current_phase": "image_scanned",
        "deadline_at": "2026-11-25T12:00:00Z",
        "external_operation_ids": {},
        "last_error": None,
        "paused_at": None,
        "phase_attempt_count": 1,
        "requested_action": None,
        **changes,
    }


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "deployment", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _statuses(*statuses: str, **fields: object) -> Any:
    answers: Iterator[str] = iter(statuses)

    def answer(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "dep_1", "status": next(answers), **fields})

    return answer


class _Clock:
    """A fake clock that moves only when the command sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(_cloud_deployment_lifecycle, "_monotonic", fake.monotonic)
    monkeypatch.setattr(_cloud_deployment_lifecycle, "_sleep", fake.sleep)
    return fake


@pytest.mark.parametrize("action", ["cancel", "destroy", "resume"])
def test_each_action_posts_to_its_route_and_prints_the_workflow(
    cloud, capsys: pytest.CaptureFixture[str], action: str
) -> None:
    workflow = _workflow(requested_action=None if action == "resume" else action)
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/{action}"): httpx.Response(202, json=workflow),
        }
    )

    code, output = _run([action, "dep_1", "--application", "Support desk"], capsys)

    assert code == 0, output
    assert output["operation"] == f"deployment.{action}"
    assert output["result"] == workflow
    # A change to an Agent says which Organization it acts in.
    assert output["organization"]["id"] == "org_test"
    (post,) = [request for request in fake.requests if request.method == "POST"]
    assert post.content == b""
    assert "release" not in output


def test_cancel_waits_until_the_release_is_cancelled(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/cancel"): httpx.Response(
                202, json=_workflow(requested_action="cancel")
            ),
            ("GET", _RELEASE): _statuses("image_scanned", "image_scanned", "cancelled"),
        }
    )

    code = main(
        [
            "cloud",
            "deployment",
            "cancel",
            "dep_1",
            "--application",
            "support-desk",
            "--wait",
            "--poll-seconds",
            "1",
        ]
    )
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert code == 0, output
    assert output["release"] == {"id": "dep_1", "status": "cancelled"}
    reads = [request for request in fake.requests if request.url.raw_path == _RELEASE.encode()]
    assert len(reads) == 3
    assert captured.err.count("waiting for Release dep_1 to be cancelled") == 1


@pytest.mark.parametrize("final", ["failed", "promoted", "destroyed"])
def test_cancel_wait_stops_when_the_release_ends_otherwise(
    cloud, capsys: pytest.CaptureFixture[str], final: str
) -> None:
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/cancel"): httpx.Response(202, json=_workflow()),
            ("GET", _RELEASE): _statuses(final),
        }
    )

    code, output = _run(["cancel", "dep_1", "--application", "support-desk", "--wait"], capsys)

    assert code == 2
    assert output["error"]["category"] == "deployment_not_cancelled"
    assert f"`{final}`" in output["error"]["message"]


def test_destroy_wait_times_out_with_the_last_status(
    cloud, capsys: pytest.CaptureFixture[str], clock
) -> None:
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/destroy"): httpx.Response(
                202, json=_workflow(requested_action="destroy")
            ),
            ("GET", _RELEASE): _statuses(*["failed"] * 10),
        }
    )

    code, output = _run(
        [
            "destroy",
            "dep_1",
            "--application",
            "support-desk",
            "--wait",
            "--poll-seconds",
            "1",
            "--wait-seconds",
            "2",
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "wait_timeout"
    assert "last status: failed" in output["error"]["message"]
    assert clock.sleeps == [1.0, 1.0]


def test_a_wait_as_long_as_one_poll_reads_again_at_its_deadline(
    cloud, capsys: pytest.CaptureFixture[str], clock
) -> None:
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/destroy"): httpx.Response(202, json=_workflow()),
            ("GET", _RELEASE): _statuses("promoted", "destroyed"),
        }
    )

    code, output = _run(
        [
            "destroy",
            "dep_1",
            "--application",
            "support-desk",
            "--wait",
            "--poll-seconds",
            "5",
            "--wait-seconds",
            "5",
        ],
        capsys,
    )

    assert code == 0, output
    assert output["release"]["status"] == "destroyed"
    assert clock.sleeps == [5.0]


def test_cancel_wait_keeps_waiting_while_a_candidate_still_reads_smoke_tested(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = {"evaluated_against": "dep_0"}
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/cancel"): httpx.Response(
                202, json=_workflow(requested_action="cancel")
            ),
            ("GET", _RELEASE): _statuses(
                "smoke_tested", "smoke_tested", "cancelled", candidate=candidate
            ),
        }
    )

    code, output = _run(["cancel", "dep_1", "--application", "support-desk", "--wait"], capsys)

    assert code == 0, output
    assert output["release"]["status"] == "cancelled"


def test_destroy_wait_rides_out_a_brief_outage(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    reads = iter(
        (
            httpx.Response(503, json={"detail": "unavailable"}),
            httpx.Response(200, json={"id": "dep_1", "status": "destroyed"}),
        )
    )
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/destroy"): httpx.Response(202, json=_workflow()),
            ("GET", _RELEASE): lambda _request: next(reads),
        }
    )

    code, output = _run(["destroy", "dep_1", "--application", "support-desk", "--wait"], capsys)

    assert code == 0, output
    assert output["release"]["status"] == "destroyed"


def test_an_invalid_wait_is_refused_before_anything_is_sent(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    code, output = _run(
        [
            "destroy",
            "dep_1",
            "--application",
            "support-desk",
            "--wait",
            "--poll-seconds",
            "10",
            "--wait-seconds",
            "5",
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_wait"
    assert fake.requests == []


def test_clouds_refusal_carries_its_reason(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_RELEASE}/destroy"): httpx.Response(
                409, json={"detail": "The application's current deployment cannot be destroyed."}
            ),
        }
    )

    code, output = _run(["destroy", "dep_1", "--application", "support-desk"], capsys)

    assert code == 2
    assert output["error"]["category"] == "api_request_rejected"
    assert "current deployment cannot be destroyed" in output["error"]["message"]


def test_promote_sends_the_session_choice(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", "/v1/applications/support-desk"): httpx.Response(
                200, json={"id": "support-desk", "revision": 7}
            ),
            ("POST", f"{_RELEASE}/promote"): httpx.Response(
                200, json={"id": "support-desk", "revision": 8}
            ),
        }
    )

    code, output = _run(
        [
            "promote",
            "dep_1",
            "--application",
            "support-desk",
            "--acknowledge-session",
            "ses_1",
        ],
        capsys,
    )

    assert code == 0, output
    (post,) = [request for request in fake.requests if request.method == "POST"]
    assert json.loads(post.content) == {
        "acknowledge_sessions": ["ses_1"],
        "expected_application_revision": 7,
        "session_policy": "proceed",
    }


def test_promote_refuses_an_invalid_session_choice_before_signing_in(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    code, output = _run(
        ["promote", "dep_1", "--application", "support-desk", "--session-policy", "proceed"],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert fake.requests == []
