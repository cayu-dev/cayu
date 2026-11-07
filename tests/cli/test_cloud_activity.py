"""`cayu cloud activity status` over Cloud's activity reader status."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_activity import _ago, read_activity_reader_status
from cayu.cli._cloud_api import CloudApiClient

_AGENT = "/v1/applications/cayu-scout"
_STATUS = f"{_AGENT}/activity/reader"
_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENTS = httpx.Response(200, json={"items": [{"id": "cayu-scout", "name": "Cayu Scout"}]})


def _reader(**changes: Any) -> dict[str, Any]:
    return {
        "enabled": True,
        "availability": "runtime_unavailable",
        "error": "runtime_unavailable",
        "generation": "0b6f4d1e-8f0c-4b7a-9d2e-3c4b5a6f7e8d",
        "store_id": "a" * 32,
        "api_path": "/api",
        "rotation_pending": False,
        "cursor": {"position": 1234, "advanced_at": "2026-11-06T10:00:00+00:00"},
        "last_synced_at": "2026-11-06T10:00:00+00:00",
        "catching_up": True,
        "last_read": {
            "started_at": "2026-11-06T10:05:00+00:00",
            "duration_ms": 10003,
            "outcome": "timed_out",
            "error": "runtime_unavailable",
            "page_limit": 200,
            "next_page_limit": 50,
            "runs": None,
            "touches": None,
            "has_more": None,
        },
        "recent_timeouts": [
            "2026-11-06T10:04:20+00:00",
            "2026-11-06T10:05:00+00:00",
        ],
        "page_limit": {"next": 50, "max": 200, "min": 10},
        "read_deadline_seconds": 10.0,
        "restarted_at": None,
        "restart_reason": None,
        **changes,
    }


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = {_LIST: _AGENTS, **routes}
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


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(["cloud", "activity", "status", "--application", "Cayu Scout", *arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_status_reads_the_reader_by_the_agents_name(cloud, capsys) -> None:
    fake = cloud({("GET", _STATUS): httpx.Response(200, json={"reader": _reader()})})

    code, out, _ = _run(["--json"], capsys)

    assert code == 0
    answer = json.loads(out)
    assert answer["ok"] is True and answer["operation"] == "activity.status"
    result = answer["result"]
    assert result["schema_version"] == 1
    assert result["application_id"] == "cayu-scout"
    assert result["reader"] == _reader()
    assert [request.url.raw_path.decode() for request in fake.requests][-1] == _STATUS


def test_status_table_says_where_the_reader_stands(cloud, capsys) -> None:
    cloud({("GET", _STATUS): httpx.Response(200, json={"reader": _reader()})})

    code, out, _ = _run([], capsys)

    assert code == 0
    assert out.startswith("Activity reader for cayu-scout\n")
    lines = {line.split("  ", 2)[1].strip(): line for line in out.splitlines()[1:]}
    assert "after event 1234" in lines["cursor position"]
    assert "2026-11-06 10:00:00Z" in lines["cursor advanced"]
    assert "timed out, in 10003 ms" in lines["last read"]
    assert "asked for 200 runs" in lines["last read"]
    assert lines["recent timeouts"].split("  ")[-1].startswith("2, latest 2026-11-06 10:05:00Z")
    assert "50 runs (10 to 200)" in lines["next page size"]
    assert "10 s" in lines["read deadline"]


def test_a_successful_read_and_a_restart_are_shown(cloud, capsys) -> None:
    read = {
        "started_at": "2026-11-06T10:05:00+00:00",
        "duration_ms": 812,
        "outcome": "read",
        "error": None,
        "page_limit": 50,
        "next_page_limit": 100,
        "runs": 12,
        "touches": 1,
        "has_more": True,
    }
    reader = _reader(
        error=None,
        availability="catching_up",
        last_read=read,
        recent_timeouts=[],
        rotation_pending=True,
        restarted_at="2026-11-05T09:00:00+00:00",
        restart_reason="store_changed",
    )
    cloud({("GET", _STATUS): httpx.Response(200, json={"reader": reader})})

    code, out, _ = _run([], capsys)

    assert code == 0
    assert "a page arrived, in 812 ms" in out
    assert "got 12 runs and 1 touches, more to read" in out
    assert "recent timeouts  none" in out
    assert "(store_changed)" in out
    assert "a new credential is prepared" in out


def test_an_agent_without_a_reader_says_so(cloud, capsys) -> None:
    cloud({("GET", _STATUS): httpx.Response(200, json={"reader": None})})

    code, out, _ = _run([], capsys)

    assert code == 0
    assert out.strip() == "Cayu Cloud has no activity reader for cayu-scout yet."


def test_values_of_another_shape_never_reach_the_output(cloud, capsys) -> None:
    secret = "sk-live-0123456789abcdef"
    reader = _reader(
        error=f"token={secret}",
        availability="https://user:pass@agent.example/api",
        generation=secret + " \x1b[31m",
        api_path="https://agent.example/api",
        cursor={"position": f"{secret}", "advanced_at": "yesterday"},
        last_read={"outcome": secret, "error": "Bearer abc", "duration_ms": -4},
        recent_timeouts=["not a time", secret, "2026-11-06T10:05:00"],
        restart_reason="Authorization: x",
        cursor_text=secret,
    )
    cloud({("GET", _STATUS): httpx.Response(200, json={"reader": reader})})

    for arguments in ([], ["--json"]):
        code, out, _ = _run(arguments, capsys)
        assert code == 0
        assert secret not in out and "user:pass" not in out and "Bearer" not in out
        assert "\x1b" not in out
    result = json.loads(out)["result"]["reader"]
    assert result["error"] is None and result["availability"] is None
    assert result["restart_reason"] is None
    assert result["generation"] is None and result["api_path"] is None
    assert result["cursor"] == {"position": None, "advanced_at": None}
    assert result["last_read"]["outcome"] is None
    assert result["last_read"]["duration_ms"] is None
    assert result["recent_timeouts"] == []
    assert "cursor_text" not in result


def test_a_cloud_without_the_route_is_named(cloud, capsys) -> None:
    cloud({("GET", _STATUS): httpx.Response(404, json={"detail": "Not Found"})})

    code, out, err = _run(["--json"], capsys)

    assert code == 2
    error = json.loads(out)["error"]
    assert error["category"] == "activity_status_unsupported"
    code, out, err = _run([], capsys)
    assert code == 2
    assert err.strip() == "Error: This Cayu Cloud doesn't report its activity reader yet."


def test_a_member_who_cant_see_the_agent_gets_clouds_refusal(cloud, capsys) -> None:
    refusal = {
        "detail": {
            "code": "agent_not_yours",
            "message": "Only the Agent's people, developers and admins can see it.",
        }
    }
    cloud({("GET", _STATUS): httpx.Response(403, json=refusal)})

    code, out, _ = _run(["--json"], capsys)

    assert code == 2
    assert json.loads(out)["error"]["code"] == "agent_not_yours"


def test_the_reader_status_is_a_part_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.raw_path.decode() == "/v1/applications/finance%20agent/activity/reader"
        return httpx.Response(200, json={"reader": None})

    real_client = httpx.Client
    monkeypatch.setattr(
        _cloud_api.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(answer), **kwargs),
    )
    client = CloudApiClient(api_url="https://cloud.example", api_key="login-token")

    assert read_activity_reader_status(client, "finance agent") == {
        "application_id": "finance agent",
        "reader": None,
    }


def test_times_say_how_long_ago_they_were() -> None:
    now = datetime(2026, 11, 6, 10, 10, tzinfo=UTC)
    assert _ago("2026-11-06T10:09:30+00:00", now=now) == "2026-11-06 10:09:30Z (30 s ago)"
    assert _ago("2026-11-06T10:00:00+00:00", now=now) == "2026-11-06 10:00:00Z (10 min ago)"
    assert _ago("2026-11-04T10:00:00+00:00", now=now) == "2026-11-04 10:00:00Z (2 days ago)"
    assert _ago(None, now=now) == "-"


@pytest.mark.parametrize(
    "changes",
    [
        {"api_path": []},
        {"api_path": {}},
        {"last_read": {"outcome": {}, "runs": 3, "touches": "many"}},
        {"cursor": {"advanced_at": "0001-01-01T00:00:00+14:00"}},
        {"recent_timeouts": ["0001-01-01T00:00:00+14:00", {}]},
        {"restarted_at": "9999-12-31T23:59:59-14:00", "restart_reason": ["x"]},
    ],
)
def test_a_field_of_the_wrong_type_is_left_out_rather_than_failing(
    cloud, capsys, changes: dict[str, Any]
) -> None:
    cloud({("GET", _STATUS): httpx.Response(200, json={"reader": _reader(**changes)})})

    code, out, _ = _run([], capsys)
    assert code == 0, out
    assert "None" not in out
    code, out, _ = _run(["--json"], capsys)
    assert code == 0, out


def test_an_answer_without_a_reader_is_invalid(cloud, capsys) -> None:
    cloud({("GET", _STATUS): httpx.Response(200, json={"readers": None})})

    code, out, _ = _run(["--json"], capsys)

    assert code == 2
    assert json.loads(out)["error"]["category"] == "api_response_invalid"
