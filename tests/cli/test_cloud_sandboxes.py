"""`cayu cloud sandboxes list|show|archives` over Cloud's sandbox broker reads."""

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


def _listing(
    *items: dict[str, Any], next_cursor: str | None = None, filters: dict[str, Any] | None = None
) -> httpx.Response:
    body: dict[str, Any] = {"items": list(items), "next_cursor": next_cursor}
    if filters is not None:
        body["filters"] = filters
    return httpx.Response(200, json=body)


def _activity(
    commands: int = 0,
    duration_ms: int = 0,
    *,
    failed: int = 0,
    put: tuple[int, int, int] = (0, 0, 0),
    got: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    return {
        "commands": {
            "count": commands,
            "running": 0,
            "failed": failed,
            "cancelled": 0,
            "duration_ms": duration_ms,
        },
        "transfers": {
            "in": {"count": put[0], "bytes": put[1], "files": put[2]},
            "out": {"count": got[0], "bytes": got[1]},
        },
    }


_NO_FILTERS = {"status": None, "session": None, "task": None, "labels": {}, "since": None}


def test_cloud_help_lists_sandboxes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "sandboxes", "--help"])
    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for command in ("list", "show", "archives"):
        assert command in help_text


def test_list_marks_a_sandbox_still_holding_a_place_for_an_older_release(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=50&activity=true"): _listing(
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
            ("GET", f"{_SANDBOXES}?limit=50&activity=true"): _listing(
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
        "COMMANDS",
        "TRANSFERS",
        "IN",
        "/",
        "OUT",
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
            ("GET", f"{_SANDBOXES}?limit=100&activity=true&status=ready"): _listing(
                *[_sandbox(f"sbx_{index}") for index in range(100)], next_cursor="c1"
            ),
            ("GET", f"{_SANDBOXES}?limit=20&activity=true&status=ready&cursor=c1"): _listing(
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
            ("GET", f"{_SANDBOXES}/sbx_old?activity=true"): httpx.Response(200, json=_LEFTOVER),
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
            ("GET", f"{_SANDBOXES}/sbx_x?activity=true"): httpx.Response(
                status, json={"error": {"code": "x", "message": "x", "detail": {}}}
            ),
        }
    )

    code = main(["cloud", "sandboxes", "show", "sbx_x", "--application", "cayu-scout", "--json"])

    assert code != 0
    assert message in capsys.readouterr().out


def test_list_asks_cloud_for_one_sessions_sandboxes_and_what_they_did(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    busy = _activity(185, 207_000, failed=3, put=(19, 129_000_000, 4200), got=(45, 138_000_000))
    applied = {
        "status": None,
        "session": "ses_1",
        "task": None,
        "labels": {"role": "builder", "mission": "m-17"},
        "since": "2026-10-29T09:00:00+00:00",
    }
    asked = (
        "session=ses_1&label=role%3Dbuilder&label=mission%3Dm-17"
        "&since=2026-10-29T09%3A00%3A00%2B00%3A00"
    )
    fake = cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=50&activity=true&{asked}"): _listing(
                _sandbox("sbx_new", activity=busy), filters=applied
            ),
            ("GET", f"{_SANDBOXES}/activity?{asked}"): httpx.Response(
                200, json={"filters": applied, "sandboxes": 1, "activity": busy}
            ),
            ("GET", f"{_SANDBOXES}/capacity"): _CAPACITY,
        }
    )
    arguments = [
        "list",
        "--session",
        "ses_1",
        "--label",
        "role=builder",
        "--label",
        "mission=m-17",
        "--since",
        "2026-10-29T02:00:00-07:00",
    ]

    code, out = _run([*arguments, "--json"], capsys)

    assert code == 0
    result = json.loads(out)["result"]
    assert result["filters"] == applied
    assert [item["id"] for item in result["items"]] == ["sbx_new"]
    assert result["items"][0]["activity"] == busy
    assert result["totals"] == {"sandboxes": 1, "activity": busy}
    assert all(request.method == "GET" for request in fake.requests)

    code, out = _run(arguments, capsys)

    assert code == 0
    (row,) = [line for line in out.splitlines() if line.startswith("sbx_new")]
    assert "185 3m27s 3 failed" in row and "19 129.0 MB / 45 138.0 MB" in row
    assert out.splitlines()[-1] == (
        "All 1 matching sandbox: 185 commands (3 failed), 3m27s in total; "
        "in 19 transfers, 129.0 MB, 4200 files; out 45 transfers, 138.0 MB."
    )


def test_list_refuses_filters_a_cloud_would_ignore(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    # A Cloud before the filters lists every sandbox and says nothing of them.
    cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=50&activity=true&task=task_1"): _listing(
                _sandbox("sbx_new"), _LEFTOVER
            ),
        }
    )

    code, out = _run(["list", "--task", "task_1", "--json"], capsys)

    assert code != 0
    error = json.loads(out)["error"]
    assert error["category"] == "sandbox_filters_unsupported"
    assert "needs cayu-tech/cayu-cloud#" in error["message"]


def test_list_without_filters_still_works_on_an_older_cloud(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=50&activity=true"): _listing(_sandbox("sbx_new")),
            ("GET", f"{_SANDBOXES}/capacity"): _CAPACITY,
        }
    )

    code, out = _run(["list"], capsys)

    assert code == 0
    (row,) = [line for line in out.splitlines() if line.startswith("sbx_new")]
    assert row.split()[-2:] == ["-", "-"]
    assert out.splitlines()[-1].startswith("Activity: not reported; this Cayu Cloud needs ")
    # It never asks the older Cloud for totals.
    assert not [request for request in fake.requests if "/activity" in str(request.url)]


def test_list_reports_no_totals_from_a_cloud_without_them(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", f"{_SANDBOXES}?limit=50&activity=true"): _listing(
                _sandbox("sbx_new", activity=_activity()), filters=_NO_FILTERS
            ),
            ("GET", f"{_SANDBOXES}/activity"): httpx.Response(
                404, json={"error": {"code": "not_found", "message": "x", "detail": {}}}
            ),
            ("GET", f"{_SANDBOXES}/capacity"): _CAPACITY,
        }
    )

    code, out = _run(["list", "--json"], capsys)

    assert code == 0
    result = json.loads(out)["result"]
    assert result["totals"] is None
    assert result["items"][0]["activity"] == _activity()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--label", "role"], "KEY=VALUE"),
        (["--label", "Role=x"], "KEY=VALUE"),
        (["--session", ""], "printable"),
        (["--since", "yesterday"], "ISO 8601"),
    ],
)
def test_list_checks_its_filters(
    capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    code = main(["cloud", "sandboxes", "list", *arguments, "--application", "cayu-scout"])

    assert code != 0
    assert message in capsys.readouterr().out


def test_list_takes_at_most_ten_labels(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({})
    labels = [item for index in range(11) for item in ("--label", f"k{index}=v")]

    code, out = _run(["list", *labels, "--json"], capsys)

    assert code != 0
    assert "at most 10 times" in json.loads(out)["error"]["message"]
    assert not [request for request in fake.requests if "/sandboxes" in str(request.url)]


def test_list_refuses_two_values_for_one_label(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({})

    code, out = _run(["list", "--label", "role=a", "--label", "role=b", "--json"], capsys)

    assert code != 0
    assert "names role twice" in json.loads(out)["error"]["message"]
    assert not [request for request in fake.requests if "/sandboxes" in str(request.url)]


def test_show_sums_what_the_sandbox_did(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    activity = _activity(2, 850, put=(1, 2048, 3), got=(1, 512))
    cloud(
        {
            ("GET", f"{_SANDBOXES}/sbx_new?activity=true"): httpx.Response(
                200, json=_sandbox("sbx_new", activity=activity)
            ),
            ("GET", f"{_SANDBOXES}/sbx_new/events"): httpx.Response(200, json={"items": []}),
        }
    )

    code, out = _run(["show", "sbx_new"], capsys)

    assert code == 0
    assert "  Commands: 2 commands, 850 ms in total" in out
    assert "  Transfers in: 1 transfer, 2.0 kB, 3 files" in out
    assert "  Transfers out: 1 transfer, 512 B" in out

    code, out = _run(["show", "sbx_new", "--json"], capsys)
    assert json.loads(out)["result"]["sandbox"]["activity"] == activity


def test_show_says_an_older_cloud_reports_no_activity(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", f"{_SANDBOXES}/sbx_old?activity=true"): httpx.Response(200, json=_LEFTOVER),
            ("GET", f"{_SANDBOXES}/sbx_old/events"): httpx.Response(200, json={"items": []}),
        }
    )

    code, out = _run(["show", "sbx_old"], capsys)

    assert code == 0
    assert "  Activity: not reported; this Cayu Cloud needs cayu-tech/cayu-cloud#" in out


_ARCHIVES = f"{_SANDBOXES}/copy-archives"
_REUSE = {"seconds": 21600, "max_archives": 8, "max_bytes": 1073741824}


def _archive(archive_id: str, **changes: object) -> dict[str, Any]:
    return {
        "id": archive_id,
        "sha256": "5f3a" + "0" * 60,
        "bytes": 9437696,
        "status": "ready",
        "created_at": "2026-11-08T17:02:11+00:00",
        "stored_at": "2026-11-08T17:02:12+00:00",
        "last_used_at": "2026-11-08T18:40:03+00:00",
        "retained_until": "2026-11-09T00:40:03+00:00",
        # A field the command doesn't show, as a later Cloud might add.
        "extra": "x",
        **changes,
    }


def _archives(*items: dict[str, Any], next_cursor: str | None = None, reuse=_REUSE):
    return httpx.Response(
        200, json={"items": list(items), "next_cursor": next_cursor, "reuse": reuse}
    )


def test_archives_lists_what_the_agent_keeps(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    storing = _archive("a" * 32, status="storing", stored_at=None, bytes=2048)
    fake = cloud({("GET", f"{_ARCHIVES}?limit=50"): _archives(storing, _archive("b" * 32))})

    code, out = _run(["archives", "--json"], capsys)

    assert code == 0
    envelope = json.loads(out)
    assert envelope["operation"] == "sandboxes.archives"
    result = envelope["result"]
    assert result["reuse"] == _REUSE
    assert result["next_cursor"] is None
    assert [(item["id"], item["status"]) for item in result["items"]] == [
        ("a" * 32, "storing"),
        ("b" * 32, "ready"),
    ]
    assert set(result["items"][0]) == {
        "id",
        "sha256",
        "bytes",
        "status",
        "created_at",
        "stored_at",
        "last_used_at",
        "retained_until",
    }
    assert all(request.method == "GET" for request in fake.requests)

    code, out = _run(["archives"], capsys)

    assert code == 0
    lines = out.splitlines()
    assert lines[0] == (
        "Reuse: each archive is kept 6h after its last use, at most 8 per Agent, each up to 1.1 GB."
    )
    assert lines[1].split() == [
        "ARCHIVE",
        "STATUS",
        "SIZE",
        "SHA256",
        "KEPT",
        "LAST",
        "USED",
        "LAPSES",
    ]
    (ready,) = [line for line in lines if line.startswith("b" * 32)]
    assert ready.split() == [
        "b" * 32,
        "ready",
        "9.4",
        "MB",
        "5f3a00000000",
        "2026-11-08",
        "17:02Z",
        "2026-11-08",
        "18:40Z",
        "2026-11-09",
        "00:40Z",
    ]


def test_archives_pages_from_a_cursor_up_to_the_limit(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_ARCHIVES}?limit=100&cursor=c0"): _archives(
                *[_archive(f"{index:032x}") for index in range(100)], next_cursor="c1"
            ),
            ("GET", f"{_ARCHIVES}?limit=20&cursor=c1"): _archives(
                *[_archive(f"{index:032x}") for index in range(100, 120)], next_cursor="c2"
            ),
        }
    )

    code, out = _run(["archives", "--limit", "120", "--cursor", "c0", "--json"], capsys)

    assert code == 0
    result = json.loads(out)["result"]
    assert len(result["items"]) == 120 and result["next_cursor"] == "c2"
    assert len(fake.requests) == 3

    code, out = _run(["archives", "--limit", "120", "--cursor", "c0"], capsys)
    assert out.splitlines()[-1] == "More: pass --cursor c2 for the next page."


def test_archives_says_when_none_are_kept(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("GET", f"{_ARCHIVES}?limit=50"): _archives(reuse=None)})

    code, out = _run(["archives"], capsys)

    assert code == 0
    assert out.splitlines() == [
        "Reuse: this Cayu Cloud keeps no copy-in archives.",
        "No copy-in archives kept for cayu-scout.",
    ]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (404, "cayu-tech/cayu-cloud#970"),
        (403, "an API key with the read scope"),
        (503, "runs no sandbox broker"),
    ],
)
def test_archives_says_what_a_refusal_means(
    cloud, capsys: pytest.CaptureFixture[str], status: int, message: str
) -> None:
    cloud(
        {
            ("GET", f"{_ARCHIVES}?limit=50"): httpx.Response(
                status, json={"error": {"code": "x", "message": "x", "detail": {}}}
            ),
        }
    )

    code, out = _run(["archives", "--json"], capsys)

    assert code != 0
    assert message in out


def test_archives_checks_its_cursor(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["cloud", "sandboxes", "archives", "--application", "a", "--cursor", ""])

    assert code != 0
    assert "--cursor: is not a cursor from an earlier page" in capsys.readouterr().out


def test_archives_refuses_pages_that_repeat_a_cursor(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_ARCHIVES}?limit=50"): _archives(next_cursor="c1"),
            ("GET", f"{_ARCHIVES}?limit=50&cursor=c1"): _archives(next_cursor="c1"),
        }
    )

    code, out = _run(["archives", "--json"], capsys)

    assert code != 0
    assert "repeat a cursor" in out
    assert len(fake.requests) == 3


def test_archives_prints_a_long_cursor_whole(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cursor = "c" * 300
    cloud({("GET", f"{_ARCHIVES}?limit=1"): _archives(_archive("a" * 32), next_cursor=cursor)})

    code, out = _run(["archives", "--limit", "1"], capsys)

    assert code == 0
    assert out.splitlines()[-1] == f"More: pass --cursor {cursor} for the next page."
