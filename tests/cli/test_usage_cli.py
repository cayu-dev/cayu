"""Event-window Usage CLI reads the same bounded HTTP data as the dashboard."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.core.test_cloud_reporting import _app, _start

from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi
from cayu.server import ServerConfig, create_server


@pytest.mark.parametrize(
    "target",
    [("--server-url", "http://127.0.0.1"), ("--application", "demo")],
    ids=["server", "cloud"],
)
def test_usage_cli_rollup_and_counts_use_real_http_contract(monkeypatch, capsys, target):
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "done", "writer")
        now = datetime.now(UTC) + timedelta(seconds=1)
        since = (now - timedelta(days=2)).isoformat()
        until = now.isoformat()

        def api(url, **kwargs):
            return CayuApi(url, http=client, **kwargs)

        monkeypatch.setattr("cayu.cli._cloud_agent_api.CayuApi", api)

        def cloud(
            application, *, context, timeout_seconds, wake_timeout_seconds, offers_wake_option
        ):
            assert application == "demo" and context is None
            assert wake_timeout_seconds == 180
            return api("http://127.0.0.1", timeout_seconds=timeout_seconds)

        monkeypatch.setattr("cayu.cli._cloud_agent_api.open_cloud_agent_api", cloud)
        arguments = [
            *target,
            "--since",
            since,
            "--until",
            until,
            "--bucket",
            "day",
            "--agent",
            "writer",
            "--json",
        ]
        assert main(["usage", "rollup", *arguments]) == 0
        output = json.loads(capsys.readouterr().out)
        expected = client.post(
            "/api/usage/rollup",
            json={
                "start_at": since,
                "end_at": until,
                "session_filter": {"agent_name": "writer"},
                "series": {"unit": "day", "timezone": "UTC"},
            },
        ).json()
        assert output["totals"] == expected["totals"]
        assert output["series"] == expected["series"]
        assert main(["usage", "counts", *arguments]) == 0
        output = json.loads(capsys.readouterr().out)
        assert output["total_count"] == "1"
        assert sum(int(b["total_count"]) for b in output["series"]["buckets"]) == 1


def test_usage_cli_rollup_prices_a_busy_day_exactly(monkeypatch, capsys, tmp_path):
    """A day with more model steps than pricing inputs still has an exact cost."""

    import asyncio
    from decimal import Decimal

    from cayu.applications import CayuApp
    from cayu.budgets.pricing import default_price_book, estimate_model_step_cost
    from cayu.budgets.usage import UsageMetrics
    from cayu.events import Event, EventType
    from cayu.messages import Message
    from cayu.sessions.records import SessionIdentity
    from cayu.sessions.requests import RunRequest
    from cayu.storage.sqlite import SQLiteSessionStore

    start = datetime(2026, 10, 1, tzinfo=UTC)

    def step(index: int) -> UsageMetrics:
        # Half the steps are above gpt-6-sol's 272k-token tier in the bundled book.
        input_tokens = (300_000 if index % 2 else 1_000) + index
        return UsageMetrics(
            provider_name="openai",
            model="gpt-6-sol",
            input_tokens=input_tokens,
            output_tokens=50,
            total_tokens=input_tokens + 50,
        )

    steps = 6000

    async def seed(store: SQLiteSessionStore) -> None:
        await store.create(
            RunRequest(
                session_id="busy", agent_name="writer", messages=[Message.text("user", "hi")]
            ),
            identity=SessionIdentity(provider_name="openai", model="gpt-6-sol"),
        )
        await store.append_events(
            "busy",
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="busy",
                    timestamp=start + timedelta(seconds=10 * index),
                    payload={"usage_metrics": step(index).model_dump(mode="json")},
                )
                for index in range(steps)
            ],
        )

    store = SQLiteSessionStore(tmp_path / "busy.sqlite")
    asyncio.run(seed(store))
    try:
        app = CayuApp(session_store=store, enable_logging=False)
        with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
            monkeypatch.setattr(
                "cayu.cli._cloud_agent_api.CayuApi",
                lambda url, **kwargs: CayuApi(url, http=client, **kwargs),
            )
            arguments = [
                "--server-url",
                "http://127.0.0.1",
                "--since",
                start.isoformat(),
                "--until",
                (start + timedelta(days=1)).isoformat(),
                "--json",
            ]
            assert main(["usage", "rollup", *arguments]) == 0
            cost = json.loads(capsys.readouterr().out)["cost"]
    finally:
        asyncio.run(store.close())

    price_book = default_price_book()
    expected = sum(
        (
            estimate_model_step_cost(
                metrics=step(index), pricing=price_book, effective_on=start.date()
            ).total_cost
            for index in range(steps)
        ),
        Decimal(0),
    )
    assert cost["accuracy"] == {"kind": "exact", "limit": None, "reason": None}
    assert int(cost["priced_model_steps"]) == steps
    assert int(cost["unevaluated_model_steps"]) == 0
    assert [(row["currency"], Decimal(row["total_cost"])) for row in cost["currencies"]] == [
        ("USD", expected)
    ]


def test_usage_help_and_output_input_alias_preserve_source(tmp_path, capsys):

    for name in ("rollup", "counts"):
        with pytest.raises(SystemExit) as exit_info:
            main(["usage", name, "--help"])
        assert exit_info.value.code == 0
        assert "--server-url" in capsys.readouterr().out
    source = tmp_path / "prices.json"
    source.write_text('{"preserve":"input"}')
    alias = tmp_path / "alias.json"
    alias.hardlink_to(source)
    assert (
        main(
            [
                "usage",
                "rollup",
                "--server-url",
                "http://127.0.0.1:1",
                "--since",
                "2026-01-01T00:00:00Z",
                "--until",
                "2026-01-02T00:00:00Z",
                "--price-book",
                str(source),
                "--output",
                str(alias),
            ]
        )
        == 1
    )
    assert source.read_text() == '{"preserve":"input"}'
    assert "output must not refer" in capsys.readouterr().err


def test_usage_cli_live_loopback_and_table_output(capsys):
    import asyncio
    import socket
    import threading
    import time

    import uvicorn

    app = _app()
    asyncio.run(_start(app, "live", "writer"))
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(app, config=ServerConfig.local_development()), log_level="error"
        )
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started
            until = datetime.now(UTC) + timedelta(seconds=1)
            arguments = [
                "--server-url",
                f"http://127.0.0.1:{port}",
                "--since",
                (until - timedelta(days=2)).isoformat(),
                "--until",
                until.isoformat(),
                "--bucket",
                "day",
                "--table",
            ]
            assert main(["usage", "rollup", *arguments]) == 0
            rollup = capsys.readouterr().out
            assert "Event-activity buckets" in rollup
            # Without --price-book, costs still use the bundled book.
            assert "Estimated cost (priced work)" in rollup
            assert main(["usage", "counts", *arguments]) == 0
            assert "Session-start buckets" in capsys.readouterr().out
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            assert not worker.is_alive()


@pytest.mark.parametrize("command", ["usage", "task", "knowledge"])
@pytest.mark.parametrize("selection", ["explicit", "environment", "persisted"])
@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_effective_cloud_context_is_preserved_before_output(
    monkeypatch, tmp_path, capsys, selection, alias_kind, command
):
    source = tmp_path / "context.json"
    original = '{"private":"preserve-context"}'
    source.write_text(original)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "active_context": str(source)}))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(config))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    extra = []
    if selection == "explicit":
        extra = ["--context", str(source)]
    elif selection == "environment":
        monkeypatch.setenv("CAYU_CLOUD_CONTEXT", str(source))
    output = source
    if alias_kind != "same":
        output = tmp_path / "output.json"
        if alias_kind == "symlink":
            output.symlink_to(source)
        else:
            output.hardlink_to(source)
    calls = []

    def cloud(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("authentication must not run")

    monkeypatch.setattr("cayu.cli._cloud_agent_api.open_cloud_agent_api", cloud)
    assert (
        main(
            [
                command,
                "counts" if command == "usage" else "list",
                "--application",
                "demo",
                *(
                    ["--since", "2026-01-01T00:00:00Z", "--until", "2026-01-02T00:00:00Z"]
                    if command == "usage"
                    else []
                ),
                *extra,
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert source.read_text() == original
    assert calls == []
    assert "output must not refer" in capsys.readouterr().err


@pytest.mark.parametrize(
    "surface",
    [
        None,
        {},
        {"read": None},
        {"read": {"enabled": False}},
        {"read": {"enabled": 1}},
        {"read": {"enabled": "true"}},
    ],
)
@pytest.mark.parametrize(
    "command,options",
    [
        ("counts", ["--agent", "writer"]),
        ("counts", ["--environment", "dev"]),
        ("counts", ["--provider", "actual"]),
        ("counts", ["--model", "m"]),
        ("counts", ["--bucket", "hour"]),
        ("rollup", ["--bucket", "hour"]),
    ],
)
def test_new_usage_options_require_positive_capability(
    monkeypatch, capsys, surface, command, options
):
    import httpx

    calls = []
    name = "session_overview_series" if command == "counts" else "usage_series"

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/api/contract":
            return httpx.Response(200, json={"capabilities": {"surfaces": {name: surface}}})
        return httpx.Response(200, json={"total_count": "99", "series": None})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=http, **kwargs),
        )
        assert (
            main(
                [
                    "usage",
                    command,
                    "--server-url",
                    "http://127.0.0.1",
                    "--since",
                    "2026-01-01T00:00:00Z",
                    "--until",
                    "2026-01-02T00:00:00Z",
                    *options,
                    "--json",
                ]
            )
            == 1
        )
    assert calls == ["/api/contract"]
    output = json.loads(capsys.readouterr().out)
    assert "error" in output and "total_count" not in output
    assert name in output["error"]["message"]


def test_plain_counts_does_not_require_series_capability(monkeypatch, capsys):
    import httpx

    calls = []

    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"total_count": "2"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=http, **kwargs),
        )
        assert (
            main(
                [
                    "usage",
                    "counts",
                    "--server-url",
                    "http://127.0.0.1",
                    "--since",
                    "2026-01-01T00:00:00Z",
                    "--until",
                    "2026-01-02T00:00:00Z",
                    "--json",
                ]
            )
            == 0
        )
    assert calls == ["/api/sessions/overview/counts"]
    assert json.loads(capsys.readouterr().out)["total_count"] == "2"


def test_usage_cli_relation_filters_match_the_http_headline_and_chart(monkeypatch, capsys):
    from tests.core.test_cloud_reporting import _drain

    from cayu.messages import Message
    from cayu.sessions.requests import RunRequest

    app = _app()

    async def seed():
        for session_id, labels, parent in (
            ("root", {"team": "red"}, None),
            ("child", {"team": "blue"}, "root"),
            ("other", {"team": "red", "tier": "gold"}, None),
        ):
            await _drain(
                app.run(
                    RunRequest(
                        agent_name="writer",
                        session_id=session_id,
                        messages=[Message.text("user", "go")],
                        labels=labels,
                        parent_session_id=parent,
                        causal_budget_id=parent,
                    )
                )
            )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(seed)
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=client, **kwargs),
        )
        now = datetime.now(UTC) + timedelta(seconds=1)
        window = ["--since", (now - timedelta(days=1)).isoformat(), "--until", now.isoformat()]
        for options, session_filter, expected in (
            (["--label", "team=red"], {"labels": {"team": "red"}}, "2"),
            (
                ["--label-selector", "team in (red,blue)", "--label-selector", "!tier"],
                {
                    "label_selectors": [
                        {"key": "team", "operator": "in", "values": ["red", "blue"]},
                        {"key": "tier", "operator": "not_exists", "values": []},
                    ]
                },
                "2",
            ),
            (["--parent-session", "root"], {"parent_session_id": "root"}, "1"),
            (
                ["--causal-budget", "root", "--label", "team=blue"],
                {"causal_budget_id": "root", "labels": {"team": "blue"}},
                "1",
            ),
        ):
            arguments = [
                "--server-url",
                "http://127.0.0.1",
                *window,
                "--bucket",
                "hour",
                "--timezone",
                "America/Los_Angeles",
                *options,
                "--json",
            ]
            assert main(["usage", "counts", *arguments]) == 0
            counts = json.loads(capsys.readouterr().out)
            assert counts["total_count"] == expected, options
            assert sum(int(b["total_count"]) for b in counts["series"]["buckets"]) == int(expected)
            assert main(["usage", "rollup", *arguments]) == 0
            rollup = json.loads(capsys.readouterr().out)
            direct = client.post(
                "/api/usage/rollup",
                json={"start_at": window[1], "end_at": window[3], "session_filter": session_filter},
            ).json()
            assert rollup["matching_session_count"] == direct["matching_session_count"] == expected
            assert rollup["totals"] == direct["totals"]


def test_usage_counts_relation_filters_require_capability_and_valid_syntax(monkeypatch, capsys):
    import httpx

    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/api/contract":
            enabled = {"read": {"enabled": True}}
            return httpx.Response(
                200, json={"capabilities": {"surfaces": {"session_overview_series": enabled}}}
            )
        return httpx.Response(200, json={"total_count": "99", "series": None})

    window = ["--since", "2026-01-01T00:00:00Z", "--until", "2026-01-02T00:00:00Z"]
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=http, **kwargs),
        )
        for options in (
            ["--label", "team=red"],
            ["--label-selector", "team"],
            ["--parent-session", "root"],
            ["--causal-budget", "root"],
        ):
            calls.clear()
            command = ["usage", "counts", "--server-url", "http://127.0.0.1", *window]
            assert main([*command, *options, "--json"]) == 1
            assert calls == ["/api/contract"]
            output = json.loads(capsys.readouterr().out)
            assert "session_overview_count_filters" in output["error"]["message"]
        for options in (["--label", "no-separator"], ["--label-selector", "team in red"]):
            for name in ("counts", "rollup"):
                calls.clear()
                command = ["usage", name, "--server-url", "http://127.0.0.1", *window]
                assert main([*command, *options, "--json"]) == 1
                assert calls == []
                assert "error" in json.loads(capsys.readouterr().out)


def test_usage_cli_recompute_checks_stored_totals_against_events(monkeypatch, capsys, tmp_path):
    import asyncio
    import sqlite3

    from cayu.applications import CayuApp
    from cayu.events import Event, EventType
    from cayu.messages import Message
    from cayu.sessions.records import SessionIdentity
    from cayu.sessions.requests import RunRequest
    from cayu.storage.sqlite import SQLiteSessionStore

    path = tmp_path / "recompute.sqlite"
    store = SQLiteSessionStore(path)
    app = CayuApp(session_store=store, enable_logging=False)
    start = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)

    async def populate() -> None:
        await store.create(
            RunRequest(session_id="s", agent_name="writer", messages=[Message.text("user", "x")]),
            identity=SessionIdentity(provider_name="anthropic", model="claude-x"),
        )
        for index in range(6):
            await store.append_event(
                "s",
                Event(
                    id=f"s-{index}",
                    session_id="s",
                    type=EventType.MODEL_COMPLETED,
                    timestamp=start + timedelta(minutes=40 * index),
                    payload={
                        "usage_metrics": {
                            "provider_name": "anthropic",
                            "model": "claude-x",
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                        }
                    },
                ),
            )

    asyncio.run(populate())
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=client, **kwargs),
        )
        arguments = [
            "usage",
            "rollup",
            "--server-url",
            "http://127.0.0.1",
            "--since",
            (start - timedelta(minutes=10)).isoformat(),
            "--until",
            (start + timedelta(hours=5)).isoformat(),
            "--recompute",
            "--json",
        ]
        assert main(arguments) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["matches"] is True and report["differences"] == []

        connection = sqlite3.connect(path)
        # An unfiltered rollup sums the hourly totals.
        connection.execute("UPDATE cayu_usage_hourly SET input_tokens = input_tokens + 1")
        connection.commit()
        connection.close()
        assert main(arguments) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["matches"] is False and report["rollup_matches"] is False
        assert report["tool_usage_matches"] is True
        # Either the sum differs, or the stored rollup fails its own pricing
        # conservation check against the changed total.
        assert any(
            difference["path"] in {"rollup.totals.usage.input_tokens", "rollup.error"}
            for difference in report["differences"]
        )


def test_recompute_differences_carry_leaf_values_only():
    from cayu.sessions.usage_recompute import _differences

    found = []
    count = _differences(
        {"records": [{"title": "a"}, {"title": "b"}], "group": {"x": 1}},
        {"records": [{"title": "a"}], "group": "missing"},
        "rollup",
        found,
    )
    # Elements past the shorter list are covered by the length difference.
    assert count == 2
    assert {(item.path, item.stored, item.recomputed) for item in found} == {
        ("rollup.records.length", 2, 1),
        ("rollup.group", "<object>", "missing"),
    }


def test_recompute_route_redacts_its_report_and_bounds_its_body(monkeypatch, tmp_path):
    from cayu.applications import CayuApp
    from cayu.sessions import usage_recompute
    from cayu.storage.sqlite import SQLiteSessionStore

    secret = "sk-live-recompute-canary"

    async def leaking(store, query):
        return usage_recompute.UsageRecomputeReport(
            matches=False,
            rollup_matches=False,
            tool_usage_matches=True,
            difference_count=1,
            differences=[
                usage_recompute.UsageRecomputeDifference(
                    path="rollup.session_breakdown.groups[0].title",
                    stored=f"title {secret}",
                    recomputed=None,
                )
            ],
        )

    monkeypatch.setattr("cayu.server._usage_recompute_routes.recompute_usage", leaking)
    app = CayuApp(
        session_store=SQLiteSessionStore(tmp_path / "redacted.sqlite"), enable_logging=False
    )
    # The server's redaction, as a configured secret redactor applies it.
    monkeypatch.setattr(
        app,
        "redact_json",
        lambda value: json.loads(json.dumps(value).replace(secret, "[redacted]")),
    )
    body = {"start_at": "2026-03-01T00:00:00Z", "end_at": "2026-03-02T00:00:00Z"}
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post("/api/usage/rollup/recompute", json=body)
        assert response.status_code == 200, response.text
        assert secret not in response.text
        oversized = client.post(
            "/api/usage/rollup/recompute", json={**body, "pad": "x" * (8 * 1024 * 1024)}
        )
        assert oversized.status_code == 413, oversized.text[:200]
