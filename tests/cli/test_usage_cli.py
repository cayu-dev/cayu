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

        def cloud(application, *, context, timeout_seconds):
            assert application == "demo" and context is None
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
            assert "Event-activity buckets" in capsys.readouterr().out
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
