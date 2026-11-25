"""`cayu cloud compound` calls Cloud's Compound mission routes with the customer login."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import _cloud_compound, main

cloud = _cloud

_ME = "/v1/me"
_ORGANIZATIONS = "/v1/me/organizations"
_APPS = "/v1/applications?lifecycle=all"
_AGENTS = {"items": [{"id": "month-end-close", "name": "Month-end close"}]}
_MISSIONS = "/v1/compound/missions"
_MISSION = f"{_MISSIONS}/mis_1"
_FINGERPRINT = "a" * 64
_IDENTITY = httpx.Response(200, json={"organization_id": "org_1"})
_NAMED = httpx.Response(200, json={"items": [{"id": "org_1", "name": "Acme"}]})


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "compound", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _bodies(fake: Any) -> list[tuple[str, str, object, str | None]]:
    return [
        (
            request.method,
            request.url.raw_path.decode(),
            json.loads(request.content) if request.content else None,
            request.headers.get("Idempotency-Key"),
        )
        for request in fake.requests
    ]


def _detail(status: str, **waiting: int) -> dict[str, Any]:
    return {
        "mission_id": "mis_1",
        "compound": {"mission": {"status": status}},
        "hosted": {"status": "running"},
        "waiting_on_person": {"open_requests": 0, "unreviewed_changes": 0, **waiting},
    }


def test_help_lists_every_mission_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "compound", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "start",
        "readiness",
        "evaluation",
        "approve-plan",
        "list",
        "status",
        "watch",
        "steer",
        "pause",
        "resume",
        "stop",
        "requests",
        "candidates",
        "decisions",
        "changes",
    ):
        assert command in help_text


def test_start_resolves_the_agent_and_sends_the_brief_and_budget_under_a_key(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"goal": "Fewer manual accruals."}))
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"max_cost_microunits": 50_000_000}))
    fake = cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", _MISSIONS): httpx.Response(
                201, json={"mission_id": "mis_1", "status": "submitted"}
            ),
        }
    )

    code, output = _run(
        [
            "start",
            "--application",
            "Month-end close",
            "--brief",
            str(brief),
            "--budget",
            str(budget),
            "--acceptance-dimension",
            "cost",
            "--idempotency-key",
            "start-1",
        ],
        capsys,
    )

    assert code == 0, output
    assert output["operation"] == "compound.start"
    assert output["idempotency_key"] == "start-1"
    assert output["organization"]["id"] == "org_test"
    assert _bodies(fake)[-1] == (
        "POST",
        _MISSIONS,
        {
            "application_id": "month-end-close",
            "brief": {"goal": "Fewer manual accruals."},
            "budget": {"max_cost_microunits": 50_000_000},
            "acceptance_dimension": "cost",
        },
        "start-1",
    )


def test_a_brief_that_is_not_a_json_object_is_refused_before_anything_is_sent(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    brief = tmp_path / "brief.json"
    brief.write_text("[]")
    fake = cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
        }
    )

    code, output = _run(
        ["start", "--application", "month-end-close", "--brief", str(brief)], capsys
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert all(request.method == "GET" for request in fake.requests)


def test_a_mission_is_approved_steered_answered_and_its_change_reviewed(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    answered = {"receipt": {"replayed": False}}
    fake = cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("POST", f"{_MISSION}/approve-plan"): httpx.Response(200, json={"status": "running"}),
            ("POST", f"{_MISSION}/steer"): httpx.Response(200, json=answered),
            ("POST", f"{_MISSION}/requests/req%2F1/answer"): httpx.Response(200, json=answered),
            ("POST", f"{_MISSION}/pause"): httpx.Response(200, json={"status": "paused"}),
            ("POST", f"{_MISSION}/changes/chg_1/approve"): httpx.Response(
                200, json={"state": "approved"}
            ),
            ("POST", f"{_MISSION}/changes/chg_2/reject"): httpx.Response(
                200, json={"state": "rejected"}
            ),
        }
    )

    for arguments in (
        ["approve-plan", "mis_1", "--plan", _FINGERPRINT, "--idempotency-key", "plan-1"],
        ["steer", "mis_1", "--message", "Focus on accruals.", "--idempotency-key", "steer-1"],
        ["requests", "answer", "mis_1", "req/1", "--message", "EU", "--idempotency-key", "ans-1"],
        ["pause", "mis_1", "--reason", "Budget review.", "--idempotency-key", "pause-1"],
        ["changes", "approve", "mis_1", "chg_1", "--expires-in-seconds", "3600"],
        ["changes", "reject", "mis_1", "chg_2", "--reason", "Too risky."],
    ):
        code, output = _run(arguments, capsys)
        assert code == 0, output
        assert output["organization"]["id"] == "org_test"

    writes = [body for body in _bodies(fake) if body[0] == "POST"]
    assert writes[:4] == [
        ("POST", f"{_MISSION}/approve-plan", {"plan_fingerprint": _FINGERPRINT}, "plan-1"),
        ("POST", f"{_MISSION}/steer", {"message": "Focus on accruals."}, "steer-1"),
        ("POST", f"{_MISSION}/requests/req%2F1/answer", {"message": "EU"}, "ans-1"),
        ("POST", f"{_MISSION}/pause", {"reason": "Budget review."}, "pause-1"),
    ]
    approve, reject = writes[4:]
    assert approve[:3] == (
        "POST",
        f"{_MISSION}/changes/chg_1/approve",
        {"expires_in_seconds": 3600},
    )
    assert approve[3] is not None and approve[3].startswith("compound-changes-approve-")
    assert reject == ("POST", f"{_MISSION}/changes/chg_2/reject", {"reason": "Too risky."}, None)


def test_a_failed_write_prints_its_key_and_cloud_code_for_the_retry(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    message = tmp_path / "message.txt"
    message.write_text("Focus on accruals.")
    cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("POST", f"{_MISSION}/steer"): httpx.Response(
                503,
                json={
                    "detail": {
                        "code": "compound_unavailable",
                        "message": "Compound is unavailable.",
                    }
                },
            ),
        }
    )

    code, output = _run(["steer", "mis_1", "--message-file", str(message)], capsys)

    assert code == 2
    assert output["error"]["code"] == "compound_unavailable"
    assert output["error"]["idempotency_key"].startswith("compound-steer-")


def test_reads_send_their_paging_and_need_no_organization_check(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", f"{_MISSIONS}?application_id=month-end-close"): httpx.Response(
                200, json={"items": [{"mission_id": "mis_1"}]}
            ),
            ("GET", "/v1/compound/agents/month-end-close/readiness"): httpx.Response(
                200, json={"ready": True}
            ),
            ("GET", f"{_MISSION}/candidates?offset=10&limit=5"): httpx.Response(
                200, json={"items": []}
            ),
            ("GET", f"{_MISSION}/decisions?offset=0&limit=50"): httpx.Response(
                200, json={"items": []}
            ),
            ("GET", f"{_MISSION}/changes/chg_1"): httpx.Response(200, json={"stageable": True}),
        }
    )

    for arguments, operation in (
        (["list", "--application", "month-end-close"], "compound.list"),
        (["readiness", "--application", "month-end-close"], "compound.readiness"),
        (
            ["candidates", "list", "mis_1", "--offset", "10", "--limit", "5"],
            "compound.candidates.list",
        ),
        (["decisions", "mis_1"], "compound.decisions"),
        (["changes", "show", "mis_1", "chg_1"], "compound.changes.show"),
    ):
        code, output = _run(arguments, capsys)
        assert (code, output["operation"]) == (0, operation), output
        assert "organization" not in output
    assert all(request.url.path != _ME for request in fake.requests)


def test_evaluation_set_replaces_the_settings_and_refuses_a_repeated_name(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    path = "/v1/compound/agents/month-end-close/evaluation"
    fake = cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("PUT", path): httpx.Response(200, json={"settings": {"REGION": "eu"}}),
        }
    )

    code, output = _run(
        ["evaluation", "set", "--application", "month-end-close", "--setting", "REGION=eu"],
        capsys,
    )
    assert code == 0, output
    assert _bodies(fake)[-1] == ("PUT", path, {"settings": {"REGION": "eu"}}, None)

    code, output = _run(
        [
            "evaluation",
            "set",
            "--application",
            "month-end-close",
            "--setting",
            "REGION=eu",
            "--setting",
            "REGION=us",
        ],
        capsys,
    )
    assert code == 2
    assert output["error"]["category"] == "invalid_input"


def test_stop_sends_no_key(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("POST", f"{_MISSION}/stop"): httpx.Response(200, json={"status": "stopped"}),
        }
    )

    code, output = _run(["stop", "mis_1"], capsys)

    assert (code, output["result"]["status"]) == (0, "stopped")
    assert "idempotency_key" not in output
    assert _bodies(fake)[-1] == ("POST", f"{_MISSION}/stop", None, None)


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


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(_cloud_compound, "_monotonic", fake.monotonic)
    monkeypatch.setattr(_cloud_compound, "_sleep", fake.sleep)
    return fake


def test_watch_waits_until_the_mission_needs_a_person(
    cloud, capsys: pytest.CaptureFixture[str], clock: _Clock
) -> None:
    reads = iter(
        [
            httpx.Response(200, json=_detail("running")),
            httpx.Response(503, json={"error": {"code": "unavailable", "message": "Busy."}}),
            httpx.Response(200, json=_detail("running", unreviewed_changes=1)),
        ]
    )
    cloud({("GET", _MISSION): lambda _request: next(reads)})

    code = main(["cloud", "compound", "watch", "mis_1", "--poll-seconds", "1"])
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert code == 0, output
    assert output["result"]["polls"] == 2
    assert output["result"]["mission"]["waiting_on_person"]["unreviewed_changes"] == 1
    assert clock.sleeps == [1.0, 1.0]
    assert "waiting for mission mis_1 (to need a person); it is running." in captured.err


def test_watch_gives_up_at_its_deadline_and_says_how_to_go_on(
    cloud, capsys: pytest.CaptureFixture[str], clock: _Clock
) -> None:
    cloud({("GET", _MISSION): httpx.Response(200, json=_detail("running"))})

    code, output = _run(
        ["watch", "mis_1", "--until", "completed", "--poll-seconds", "2", "--wait-seconds", "5"],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "wait_timeout"
    assert "run watch again" in output["error"]["message"]
    # The last sleep is cut to what is left of the wait, and the mission is read once more.
    assert clock.sleeps == [2.0, 2.0, 1.0]


def test_watch_reads_again_at_the_deadline_when_the_wait_is_one_poll_long(
    cloud, capsys: pytest.CaptureFixture[str], clock: _Clock
) -> None:
    reads = iter(
        [
            httpx.Response(200, json=_detail("running")),
            httpx.Response(200, json=_detail("running", open_requests=1)),
        ]
    )
    cloud({("GET", _MISSION): lambda _request: next(reads)})

    code, output = _run(["watch", "mis_1", "--wait-seconds", "15"], capsys)

    assert code == 0, output
    assert output["result"]["polls"] == 2
    assert clock.sleeps == [15.0]


@pytest.mark.parametrize(
    ("detail", "until", "satisfied"),
    [
        (_detail("running"), "waiting", False),
        (_detail("paused"), "waiting", True),
        (_detail("running", open_requests=1), "waiting", True),
        (_detail("running", open_requests=1), "completed", False),
        (_detail("completed"), "completed", True),
        ({**_detail("running"), "hosted": {"status": "failed"}}, "completed", True),
        ({"hosted": {"status": "stopped"}}, "waiting", True),
        ({"waiting_on_person": {"open_requests": True}}, "waiting", False),
    ],
)
def test_watch_satisfied(detail: dict[str, Any], until: str, satisfied: bool) -> None:
    assert _cloud_compound.watch_satisfied(detail, until) is satisfied


def test_an_empty_application_is_refused_as_json(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("GET", _APPS): httpx.Response(200, json=_AGENTS)})

    code, output = _run(["readiness", "--application", ""], capsys)

    assert code == 2
    assert output["error"]["category"] == "application_not_found"
