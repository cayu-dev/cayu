"""`cayu cloud sandboxes list|show` over Cloud's sandbox broker reads."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_AGENT = "/v1/applications/cayu-scout"
_SANDBOXES = f"{_AGENT}/sandboxes"
_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENTS = httpx.Response(200, json={"items": [{"id": "cayu-scout", "name": "Cayu Scout"}]})
_SELECTED = httpx.Response(200, json={"id": "cayu-scout", "current_deployment_id": "dep_new"})
_CAPACITY = httpx.Response(
    200,
    json={
        "available": 0,
        "limited_by": "agent",
        "agent": {"limit": 8, "in_use": 8, "queued": 0, "available": 0},
    },
)


def _sandbox(sandbox_id: str, **changes: object) -> dict[str, Any]:
    return {
        "id": sandbox_id,
        "application_id": "cayu-scout",
        "image": "coding-python",
        "egress": {"mode": "none"},
        "status": "ready",
        "failure": None,
        "labels": {},
        "on_behalf_of": None,
        "provenance": {"release_id": "dep_new", "session_id": "ses_1", "task_id": "task_1"},
        "queued_until": None,
        "provider": "aws-lambda-microvm",
        "created_at": "2026-10-29T09:27:00+00:00",
        "ready_at": "2026-10-29T09:27:20+00:00",
        "lease_expires_at": "2026-10-29T09:42:20+00:00",
        "max_expires_at": "2026-10-29T17:27:20+00:00",
        "destroyed_at": None,
        "destroy_reason": None,
        "owner_epoch": 1,
        **changes,
    }


_LEFTOVER = _sandbox(
    "sbx_old",
    provenance={"release_id": "dep_old", "session_id": "ses_0", "task_id": None},
)
_ENDED = _sandbox(
    "sbx_gone",
    status="destroyed",
    provenance={"release_id": "dep_old", "session_id": "ses_9", "task_id": None},
    destroy_reason="lease_expired",
    destroyed_at="2026-10-29T09:30:00+00:00",
)
_UNTRACED = _sandbox("sbx_bare", provenance=None)
# Waiting in the queue holds no place yet, whichever Release asked.
_QUEUED = _sandbox(
    "sbx_waiting",
    status="provisioning",
    ready_at=None,
    lease_expires_at=None,
    queued_until="2026-10-29T09:50:00+00:00",
    provenance={"release_id": "dep_old", "session_id": "ses_2", "task_id": None},
)


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = {_LIST: _AGENTS, ("GET", _AGENT): _SELECTED, **routes}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.routes[(request.method, request.url.raw_path.decode())]


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], httpx.Response]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], httpx.Response]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(["cloud", "sandboxes", *arguments, "--application", "Cayu Scout"])
    return code, capsys.readouterr().out


def _listing(*items: dict[str, Any], next_cursor: str | None = None) -> httpx.Response:
    return httpx.Response(200, json={"items": list(items), "next_cursor": next_cursor})


def test_cloud_help_lists_sandboxes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "sandboxes", "--help"])
    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for command in ("list", "show"):
        assert command in help_text


def test_list_marks_a_sandbox_still_holding_a_place_for_an_older_release(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=50"): _listing(
                _sandbox("sbx_new"), _LEFTOVER, _ENDED, _UNTRACED, _QUEUED
            ),
            ("GET", f"{_SANDBOXES}/capacity"): _CAPACITY,
        }
    )

    code, out = _run(["list", "--json"], capsys)

    assert code == 0
    result = json.loads(out)["result"]
    assert result["selected_release_id"] == "dep_new"
    assert result["capacity"] == {
        "limited_by": "agent",
        "available": 0,
        "agent": {"limit": 8, "in_use": 8, "queued": 0},
    }
    marks = {item["id"]: item["older_release"] for item in result["items"]}
    # Only a sandbox that still holds a place counts; one without provenance can't be told.
    assert marks == {
        "sbx_new": False,
        "sbx_old": True,
        "sbx_gone": False,
        "sbx_bare": None,
        "sbx_waiting": False,
    }
    assert result["items"][1]["provenance"] == {
        "release_id": "dep_old",
        "session_id": "ses_0",
        "task_id": None,
    }
    assert all(request.method == "GET" for request in fake.requests)


def test_list_prints_a_table_with_the_capacity_and_the_leftover(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=50"): _listing(
                _sandbox("sbx_new"), _LEFTOVER, _ENDED, _QUEUED
            ),
            ("GET", f"{_SANDBOXES}/capacity"): _CAPACITY,
        }
    )

    code, out = _run(["list"], capsys)

    assert code == 0
    lines = out.splitlines()
    assert lines[0] == (
        "Capacity: 8 of 8 places held under the Agent's limit, 0 queued; a create now waits "
        "or is refused (agent limit)."
    )
    assert lines[1].split() == [
        "SANDBOX",
        "STATUS",
        "CREATED",
        "LEASE",
        "ENDS",
        "ENDED",
        "RELEASE",
        "SESSION",
        "TASK",
    ]
    (old,) = [line for line in lines if line.startswith("sbx_old")]
    assert "2026-10-29 09:42Z" in old and "dep_old *" in old and "ses_0" in old
    (gone,) = [line for line in lines if line.startswith("sbx_gone")]
    assert "lease_expired 2026-10-29 09:30Z" in gone and "*" not in gone
    (waiting,) = [line for line in lines if line.startswith("sbx_waiting")]
    assert "queued to 2026-10-29 09:50Z" in waiting and "*" not in waiting
    assert "1 still holds a place for a Release the Agent no longer selects (dep_new)" in out


def test_list_pages_up_to_the_limit_and_filters_by_status(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=100&status=ready"): _listing(
                *[_sandbox(f"sbx_{index}") for index in range(100)], next_cursor="c1"
            ),
            ("GET", f"{_SANDBOXES}?limit=20&status=ready&cursor=c1"): _listing(
                *[_sandbox(f"sbx_{index}") for index in range(100, 120)], next_cursor="c2"
            ),
            ("GET", f"{_SANDBOXES}/capacity"): httpx.Response(404, json={}),
        }
    )

    code, out = _run(["list", "--status", "ready", "--limit", "120", "--json"], capsys)

    assert code == 0
    result = json.loads(out)["result"]
    assert len(result["items"]) == 120 and result["next_cursor"] == "c2"
    # A Cloud that doesn't report capacity still lists.
    assert result["capacity"] is None
    assert len([request for request in fake.requests if "cursor" in str(request.url)]) == 1


def test_show_prints_the_sandbox_its_provenance_and_events(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    events = [
        {
            "at": "2026-10-29T09:27:00+00:00",
            "kind": "requested",
            "actor": "agent",
            "detail": {"image": "coding-python", "provenance": _LEFTOVER["provenance"]},
        },
        {"at": "2026-10-29T09:27:20+00:00", "kind": "ready", "actor": "cloud", "detail": {}},
    ]
    cloud(
        {
            ("GET", f"{_SANDBOXES}/sbx_old"): httpx.Response(200, json=_LEFTOVER),
            ("GET", f"{_SANDBOXES}/sbx_old/events"): httpx.Response(200, json={"items": events}),
        }
    )

    code, out = _run(["show", "sbx_old"], capsys)

    assert code == 0
    assert "Sandbox: sbx_old (ready)" in out
    assert "Lease ends: 2026-10-29 09:42Z" in out
    assert "Release: dep_old (the Agent now selects dep_new)" in out
    assert "Session: ses_0" in out and "Task: -" in out
    (requested,) = [line for line in out.splitlines() if "requested" in line]
    assert "provenance={release_id=dep_old,session_id=ses_0,task_id=-}" in requested

    code, out = _run(["show", "sbx_old", "--json"], capsys)
    result = json.loads(out)["result"]
    assert [event["kind"] for event in result["events"]] == ["requested", "ready"]
    assert result["sandbox"]["older_release"] is True


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "lets only the Agent itself read them"),
        (403, "an API key with the read scope"),
        (404, "no such sandbox"),
        (503, "runs no sandbox broker"),
    ],
)
def test_a_refused_read_says_what_it_means(
    cloud, capsys: pytest.CaptureFixture[str], status: int, message: str
) -> None:
    cloud(
        {
            ("GET", f"{_SANDBOXES}/sbx_x"): httpx.Response(
                status, json={"error": {"code": "x", "message": "x", "detail": {}}}
            ),
        }
    )

    code = main(["cloud", "sandboxes", "show", "sbx_x", "--application", "cayu-scout", "--json"])

    assert code != 0
    assert message in capsys.readouterr().out
