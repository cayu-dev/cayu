"""`cayu cloud service database` over Cloud's Agent database health."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient
from cayu.cli._cloud_database_health import window_minutes

_AGENT = "/v1/applications/cayu-scout"
_HEALTH = f"{_AGENT}/service/database"
_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENTS = httpx.Response(200, json={"items": [{"id": "cayu-scout", "name": "Cayu Scout"}]})

_ANSWER: dict[str, Any] = {
    "schema_version": 1,
    "application_id": "cayu-scout",
    "observed_at": "2026-11-06T18:30:00+00:00",
    "window": {"start": "2026-11-06T17:30:00+00:00", "end": "2026-11-06T18:30:00+00:00"},
    "round_trips": {
        "path": "direct",
        "connect_ms": 21.4,
        "samples": 10,
        "median_ms": 0.62,
        "min_ms": 0.51,
        "max_ms": 1.9,
    },
    "connections": {
        "total": 7,
        "active": 1,
        "idle": 6,
        "idle_in_transaction": 0,
        "other": 0,
        "waiting_on_locks": 0,
    },
    "pooler": {
        "samples": 120,
        "tasks": 2,
        "queries": 51120,
        "queries_per_second": 14.2,
        "avg_query_ms": 1.1,
        "avg_transaction_ms": 3.4,
        "avg_wait_ms": 0.02,
        "max_wait_ms": 0.0,
        "max_client_waiting": 0,
        "max_server_connections": 4,
    },
    "instance": {
        "shared": True,
        "instance_class": "db.t4g.small",
        "cpu_percent": {"average": 18.5, "max": 61.0},
        "cpu_credit_balance": {"latest": 140.2, "min": 132.0},
        "connections": {"max": 88},
    },
    "unavailable": {},
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


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(["cloud", "service", "database", "--application", "Cayu Scout", *arguments])
    return code, capsys.readouterr().out


def test_help_names_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "service", "database", "--help"])
    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--window" in help_text
    assert "PgBouncer" in help_text


def test_the_table_shows_every_part(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({("GET", f"{_HEALTH}?window_minutes=60"): httpx.Response(200, json=_ANSWER)})

    code, out = _run(["--window", "1h"], capsys)

    assert code == 0, out
    assert fake.requests[-1].headers["Authorization"] == "Bearer login-token"
    assert "Window: 2026-11-06 17:30Z to 2026-11-06 18:30Z" in out
    assert "Connect: 21.4 ms" in out
    assert "SELECT 1 x 10: median 0.62 ms, min 0.51 ms, max 1.9 ms" in out
    assert "TOTAL  ACTIVE  IDLE  IDLE IN TX  OTHER  WAITING ON LOCKS" in out
    assert "PgBouncer for this Agent (120 samples from 2 tasks):" in out
    assert "Queries: 51120 (14.2/s), average 1.1 ms" in out
    assert "Wait for a server connection: average 0.02 ms, longest 0.0 ms" in out
    assert "Most in a minute: 0 clients waiting, 4 server connections" in out
    assert "Shared db.t4g.small (every Agent of the install):" in out
    assert "CPU credits: now 140.2, lowest 132.0" in out


def test_json_keeps_only_the_known_fields(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    leaky = {
        **_ANSWER,
        "host": "agents.cluster.example.internal",
        "round_trips": {
            **_ANSWER["round_trips"],
            "dsn": "postgresql://admin:secret@host/db",
            "connect_ms": "postgresql://admin:secret@host/db",
            "path": "agents.cluster.example.internal",
        },
        "instance": {
            **_ANSWER["instance"],
            "endpoint": "agents.example:5432",
            "instance_class": "agents.cluster.example.internal:5432",
        },
        "pooler": {**_ANSWER["pooler"], "avg_query_ms": "secret"},
        "observed_at": "postgresql://admin:secret@host/db",
        "unavailable": {"pooler": "postgresql://admin:secret@host/db"},
    }
    cloud({("GET", f"{_HEALTH}?window_minutes=15"): httpx.Response(200, json=leaky)})

    code, out = _run(["--json"], capsys)

    assert code == 0, out
    envelope = json.loads(out)
    assert envelope["ok"] is True
    assert envelope["operation"] == "service.database"
    result = envelope["result"]
    assert result["schema_version"] == 1
    # Text where a number belongs is dropped, never printed.
    assert result["round_trips"] == {**_ANSWER["round_trips"], "connect_ms": None, "path": None}
    assert result["pooler"] == {**_ANSWER["pooler"], "avg_query_ms": None}
    assert result["instance"] == {**_ANSWER["instance"], "instance_class": None}
    assert result["observed_at"] is None
    assert result["unavailable"] == {}
    for text in ("secret", "agents.cluster", "endpoint", "dsn"):
        assert text not in out


@pytest.mark.parametrize("path", [[], {}, 3, None])
def test_a_path_that_isnt_text_is_dropped(
    cloud, capsys: pytest.CaptureFixture[str], path: object
) -> None:
    answer = {**_ANSWER, "round_trips": {**_ANSWER["round_trips"], "path": path}}
    cloud({("GET", f"{_HEALTH}?window_minutes=15"): httpx.Response(200, json=answer)})

    code, out = _run(["--json"], capsys)

    assert code == 0, out
    result = json.loads(out)["result"]
    assert result["round_trips"] == {**_ANSWER["round_trips"], "path": None}


def test_unavailable_parts_say_why(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    answer = {
        **_ANSWER,
        "pooler": None,
        "instance": None,
        "unavailable": {"pooler": "no_samples", "instance": "metrics_not_permitted"},
    }
    cloud({("GET", f"{_HEALTH}?window_minutes=15"): httpx.Response(200, json=answer)})

    code, out = _run([], capsys)

    assert code == 0, out
    assert "PgBouncer: unavailable (no PgBouncer lines for this Agent in the window" in out
    assert "Instance: unavailable (Cloud may not read these metrics yet)." in out
    assert "Connect: 21.4 ms" in out


def test_a_burstable_class_is_told_apart(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    answer = {
        **_ANSWER,
        "instance": {
            **_ANSWER["instance"],
            "instance_class": "db.m7g.large",
            "cpu_credit_balance": None,
        },
    }
    cloud({("GET", f"{_HEALTH}?window_minutes=15"): httpx.Response(200, json=answer)})

    code, out = _run([], capsys)

    assert code == 0, out
    assert "CPU credits: - (not a burstable class)" in out


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            httpx.Response(
                404,
                json={"detail": {"code": "agent_database_not_found", "message": "none yet"}},
            ),
            "The Agent has no database yet; deploy it first.",
        ),
        (
            httpx.Response(
                503,
                json={
                    "detail": {
                        "code": "agent_database_health_unavailable",
                        "message": "no databases",
                    }
                },
            ),
            "gives its Agents no database",
        ),
        (httpx.Response(404, json={"detail": "Not Found"}), "doesn't report an Agent's database"),
        # A 503 that isn't Cloud's own refusal (an outage) stays what it was.
        (httpx.Response(503, json={"detail": "Service Unavailable"}), "HTTP 503"),
    ],
)
def test_refusals_say_what_they_mean(
    cloud, capsys: pytest.CaptureFixture[str], response: httpx.Response, message: str
) -> None:
    cloud({("GET", f"{_HEALTH}?window_minutes=15"): response})

    code, out = _run(["--json"], capsys)

    assert code != 0
    assert message in out


@pytest.mark.parametrize(("value", "minutes"), [("15m", 15), ("1m", 1), ("2h", 120), ("6h", 360)])
def test_the_window_is_minutes_or_hours(value: str, minutes: int) -> None:
    assert window_minutes(value) == minutes


@pytest.mark.parametrize("value", ["0m", "7h", "361m", "90s", "1d", "soon", "15"])
def test_a_window_outside_the_bounds_is_refused(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        window_minutes(value)
