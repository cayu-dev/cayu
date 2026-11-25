"""`cayu cloud admin`'s platform administrator commands and its repairs on one Organization's
Agent call Cloud's routes with the request shapes Cloud takes."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import _cloud_admin_operations, main

cloud = _cloud

_APPS = "/v1/applications?lifecycle=all"
_AGENTS = {"items": [{"id": "month-end-close", "name": "Month-end close"}]}
_OWNED = "/v1/admin/organizations/org_2/applications/month-end-close"
_REFUSED = httpx.Response(
    403, json={"detail": "Cayu Cloud platform administrator access is required."}
)


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "admin", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _sent(fake: Any) -> list[tuple[str, str, object]]:
    return [
        (
            request.method,
            request.url.raw_path.decode(),
            json.loads(request.content) if request.content else None,
        )
        for request in fake.requests
    ]


def test_help_lists_every_admin_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "admin", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "platform-admin",
        "access-requests",
        "organizations",
        "applications",
        "agent-lease",
        "archive",
        "connections",
        "runtimes",
        "cleanup",
        "activity-reader",
    ):
        assert command in help_text


@pytest.mark.parametrize(
    ("arguments", "method", "path", "body"),
    [
        (["platform-admin", "list"], "GET", "/v1/admin/platform-admins", None),
        (["access-requests", "list"], "GET", "/v1/admin/access-requests", None),
        (
            ["access-requests", "list", "--status", "pending", "--limit", "20", "--cursor", "c1"],
            "GET",
            "/v1/admin/access-requests?status=pending&limit=20&cursor=c1",
            None,
        ),
        (["access-requests", "get", "acr_1"], "GET", "/v1/admin/access-requests/acr_1", None),
        (["organizations", "list"], "GET", "/v1/admin/organizations", None),
        (
            ["organizations", "list", "--query", "Example Co"],
            "GET",
            "/v1/admin/organizations?query=Example+Co",
            None,
        ),
        (
            ["organizations", "agents", "org_1"],
            "GET",
            "/v1/admin/organizations/org_1/applications",
            None,
        ),
        (["platform-admin", "grant", "user_1"], "PUT", "/v1/admin/platform-admins/user_1", None),
        (
            ["platform-admin", "revoke", "user_1"],
            "DELETE",
            "/v1/admin/platform-admins/user_1",
            None,
        ),
        (
            ["access-requests", "approve", "acr_1", "--organization", "org_1"],
            "PUT",
            "/v1/admin/access-requests/acr_1/decision",
            {
                "outcome": "approved",
                "admission": {
                    "kind": "existing",
                    "organization_id": "org_1",
                    "role_slug": "member",
                },
            },
        ),
        (
            ["access-requests", "approve", "acr_1", "--new-organization", "New Co"],
            "PUT",
            "/v1/admin/access-requests/acr_1/decision",
            {"outcome": "approved", "admission": {"kind": "new", "organization_name": "New Co"}},
        ),
        (
            ["access-requests", "reject", "acr_1"],
            "PUT",
            "/v1/admin/access-requests/acr_1/decision",
            {"outcome": "rejected"},
        ),
        (
            ["connections", "move-model-credentials"],
            "POST",
            "/v1/admin/connections/model-credentials/move",
            None,
        ),
        (["archive", "blocked", "--limit", "5"], "GET", "/v1/admin/archives?limit=5", None),
        (
            ["archive", "show", "month-end-close", "--organization", "org_2"],
            "GET",
            f"{_OWNED}/archive",
            None,
        ),
        (
            [
                "archive",
                "confirm-exit",
                "month-end-close",
                "--organization",
                "org_2",
                "--kind",
                "agent_task",
                "--subject",
                "task_1",
                "--reason",
                "The task's container is gone.",
            ],
            "POST",
            f"{_OWNED}/archive/exit-confirmations",
            {
                "subject_kind": "agent_task",
                "subject": "task_1",
                "reason": "The task's container is gone.",
            },
        ),
        (
            ["agent-lease", "show", "month-end-close", "--organization", "org_2"],
            "GET",
            f"{_OWNED}/lease-work",
            None,
        ),
        (
            [
                "agent-lease",
                "confirm-exit",
                "month-end-close",
                "--organization",
                "org_2",
                "--ref",
                "job/publish-1",
                "--reason",
                "The Job finished.",
                "--outside-runtime",
            ],
            "POST",
            f"{_OWNED}/lease-work/exit-confirmations",
            {"ref": "job/publish-1", "reason": "The Job finished.", "outside_runtime": True},
        ),
    ],
)
def test_platform_admin_commands_send_cloud_s_request(
    cloud,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    method: str,
    path: str,
    body: object,
) -> None:
    answer = (
        httpx.Response(204)
        if method in {"PUT", "DELETE"} and "platform-admins" in path
        else (httpx.Response(200, json={"items": []}))
    )
    fake = cloud({(method, path): answer})

    code, output = _run(arguments, capsys)

    assert code == 0, output
    assert _sent(fake) == [(method, path, body)]
    assert output["operation"] == "admin." + ".".join(arguments[:2])


def test_grant_and_revoke_say_whose_access_changed(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("DELETE", "/v1/admin/platform-admins/user_1"): httpx.Response(204)})

    code, output = _run(["platform-admin", "revoke", "user_1"], capsys)

    assert code == 0
    assert output["result"] == {"user_id": "user_1", "platform_admin": False}


def test_anyone_but_a_platform_administrator_is_told_who_may(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", f"{_OWNED}/lease-work"): _REFUSED})

    code, output = _run(
        ["agent-lease", "show", "month-end-close", "--organization", "org_2"], capsys
    )

    assert code == 2
    assert output["error"]["category"] == "platform_admin_required"
    assert "platform administrator" in output["error"]["message"]
    assert "cayu cloud login" in output["error"]["message"]


def test_a_cloud_without_the_route_says_so(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("GET", "/v1/admin/platform-admins"): httpx.Response(404, json={"detail": "Not Found"})})

    code, output = _run(["platform-admin", "list"], capsys)

    assert code == 2
    assert output["error"]["category"] == "admin_command_unsupported"


def test_reading_access_requests_is_refused_to_anyone_else(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", "/v1/admin/access-requests"): _REFUSED})

    code, output = _run(["access-requests", "list"], capsys)

    assert code == 2
    assert output["error"]["category"] == "platform_admin_required"


def test_a_page_limit_outside_cloud_s_bounds_is_refused_before_anything_is_sent(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    code, output = _run(["access-requests", "list", "--limit", "101"], capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert fake.requests == []


def test_an_empty_organization_is_refused_before_anything_is_sent(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    code, output = _run(["archive", "show", "month-end-close", "--organization", " "], capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert fake.requests == []


def test_a_refused_move_reports_cloud_s_code(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("POST", "/v1/admin/applications/month-end-close/move"): httpx.Response(
                409,
                json={
                    "detail": {
                        "code": "sandboxes_live",
                        "message": "The Agent has live sandboxes; stop them, then move it.",
                    }
                },
            )
        }
    )

    code, output = _run(
        ["applications", "move", "month-end-close", "--to-organization", "org_2"], capsys
    )

    assert code == 2
    assert output["error"]["code"] == "sandboxes_live"
    assert "live sandboxes" in output["error"]["message"]


def test_a_move_whose_runtime_step_is_pending_prints_the_command_that_finishes_it(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/admin/applications/month-end-close/move"): httpx.Response(
                200, json={"application_id": "month-end-close", "status": "runtime_pending"}
            )
        }
    )

    code, output = _run(
        ["applications", "move", "month-end-close", "--to-organization", "org_2"], capsys
    )

    assert code == 0, output
    assert _sent(fake) == [
        ("POST", "/v1/admin/applications/month-end-close/move", {"organization_id": "org_2"})
    ]
    assert output["result"]["commands"]["finish_move"] == (
        "cayu cloud admin applications move month-end-close --to-organization org_2"
    )


@pytest.mark.parametrize(
    ("arguments", "path", "body"),
    [
        (
            ["hold", "rta_1", "--reason", "Keep for the incident review."],
            "hold",
            {"reason": "Keep for the incident review."},
        ),
        (["release-hold", "rta_1"], "release-hold", None),
        (["retire", "rta_1"], "retire", None),
    ],
)
def test_runtimes_change_an_artifact_of_the_resolved_agent(
    cloud,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    path: str,
    body: object,
) -> None:
    route = f"/v1/applications/month-end-close/runtime-artifacts/rta_1/{path}"
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", route): httpx.Response(200, json={"id": "rta_1", "state": "held"}),
        }
    )

    code, output = _run(["runtimes", *arguments, "--application", "Month-end close"], capsys)

    assert code == 0, output
    assert _sent(fake)[-1] == ("POST", route, body)
    assert output["operation"] == f"admin.runtimes.{path}"
    assert output["result"]["application_id"] == "month-end-close"
    assert output["organization"]["id"] == "org_test"


class _Clock:
    """A fake clock that moves only when the command sleeps, or a read takes time."""

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
    monkeypatch.setattr(_cloud_admin_operations.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(_cloud_admin_operations.time, "sleep", fake.sleep)
    return fake


def test_cleanup_reads_the_release_again_when_the_poll_fills_the_whole_wait(
    cloud, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    release = "/v1/applications/month-end-close/deployments/dep_1"
    statuses = iter(["destroying", "destroyed"])

    def read(_request: httpx.Request) -> httpx.Response:
        clock.now += 0.5  # Each read takes time, so a full poll would overrun the wait.
        return httpx.Response(200, json={"id": "dep_1", "status": next(statuses)})

    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{release}/destroy"): httpx.Response(202, json={}),
            ("GET", release): read,
        }
    )

    code, output = _run(
        [
            "cleanup",
            "--application",
            "month-end-close",
            "--deployment",
            "dep_1",
            "--poll-seconds",
            "30",
            "--wait-seconds",
            "30",
        ],
        capsys,
    )

    assert code == 0, output
    assert output["result"]["status"] == "destroyed"
    # The sleep is cut to what is left of the wait, and the Release is read once more.
    assert clock.sleeps == [29.5]


def test_cleanup_times_out_once_the_wait_has_passed(
    cloud, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    release = "/v1/applications/month-end-close/deployments/dep_1"
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{release}/destroy"): httpx.Response(202, json={}),
            ("GET", release): httpx.Response(200, json={"status": "destroying"}),
        }
    )

    code, output = _run(
        [
            "cleanup",
            "--application",
            "month-end-close",
            "--deployment",
            "dep_1",
            "--poll-seconds",
            "2",
            "--wait-seconds",
            "5",
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "wait_timeout"
    assert clock.sleeps == [2.0, 2.0, 1.0]


def test_cleanup_destroys_the_release_and_waits_until_it_is_destroyed(
    cloud, clock: _Clock, capsys: pytest.CaptureFixture[str]
) -> None:
    release = "/v1/applications/month-end-close/deployments/dep_1"
    statuses = iter(["destroying", "destroyed"])
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{release}/destroy"): httpx.Response(202, json={"status": "destroying"}),
            ("GET", release): lambda _request: httpx.Response(
                200, json={"id": "dep_1", "status": next(statuses)}
            ),
        }
    )

    code, output = _run(
        [
            "cleanup",
            "--application",
            "month-end-close",
            "--deployment",
            "dep_1",
            "--poll-seconds",
            "0.01",
        ],
        capsys,
    )

    assert code == 0, output
    assert output["operation"] == "admin.cleanup"
    assert output["result"]["status"] == "destroyed"
    assert [request[:2] for request in _sent(fake)].count(("GET", release)) == 2


def test_cleanup_refuses_an_unordered_wait_before_destroying_anything(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("GET", _APPS): httpx.Response(200, json=_AGENTS)})

    code, output = _run(
        [
            "cleanup",
            "--application",
            "month-end-close",
            "--deployment",
            "dep_1",
            "--poll-seconds",
            "10",
            "--wait-seconds",
            "1",
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_wait"
    assert all(request.method == "GET" for request in fake.requests)


def test_cleanup_fails_when_the_release_ends_another_way(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    release = "/v1/applications/month-end-close/deployments/dep_1"
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{release}/destroy"): httpx.Response(202, json={}),
            ("GET", release): httpx.Response(200, json={"status": "failed"}),
        }
    )

    code, output = _run(
        ["cleanup", "--application", "month-end-close", "--deployment", "dep_1"], capsys
    )

    assert code == 2
    assert output["error"]["category"] == "release_cleanup_failed"


def test_activity_reader_show_and_setup(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    path = "/v1/applications/month-end-close/activity-reader"
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", path): httpx.Response(200, json={"reader": None}),
            ("PUT", path): httpx.Response(200, json={"reader": {"revision": 1}}),
        }
    )

    code, output = _run(["activity-reader", "show", "--application", "month-end-close"], capsys)
    assert code == 0, output
    assert output["result"] == {"reader": None}

    code, output = _run(["activity-reader", "setup", "--application", "month-end-close"], capsys)
    assert code == 0, output
    assert _sent(fake)[-1] == ("PUT", path, {"action": "setup"})


def test_activity_reader_restart_sends_the_revision_it_read(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    path = "/v1/applications/month-end-close/activity-reader"
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", path): httpx.Response(200, json={"reader": {"revision": 7}}),
            ("PUT", path): httpx.Response(200, json={"reader": {"revision": 8}}),
        }
    )

    code, output = _run(["activity-reader", "restart", "--application", "month-end-close"], capsys)

    assert code == 0, output
    assert _sent(fake)[-1] == ("PUT", path, {"action": "restart", "expected_revision": 7})


def test_activity_reader_restart_without_a_reader_changes_nothing(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    path = "/v1/applications/month-end-close/activity-reader"
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", path): httpx.Response(200, json={"reader": None}),
        }
    )

    code, output = _run(["activity-reader", "restart", "--application", "month-end-close"], capsys)

    assert code == 2
    assert output["error"]["category"] == "activity_reader_not_set_up"
    assert all(request.method == "GET" for request in fake.requests)
