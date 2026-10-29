"""Cost reports and their token fallback stay available through real HTTP-backed CLI."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient
from tests.server.test_usage_cost_details import WINDOW, prices, seed

from cayu.applications import CayuApp
from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi
from cayu.server import ServerConfig, create_server
from cayu.sessions.memory import InMemorySessionStore


@pytest.mark.parametrize(
    "target", [("--server-url", "http://localhost"), ("--application", "demo")]
)
@pytest.mark.parametrize("priced", [False, True])
def test_cost_cli_http_json_and_table(monkeypatch, tmp_path, capsys, target, priced):
    store = InMemorySessionStore()
    asyncio.run(seed(store))
    with TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:

        def api(url, **kwargs):
            return CayuApi(url, http=client, **kwargs)

        monkeypatch.setattr("cayu.cli._cloud_agent_api.CayuApi", api)
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.open_cloud_agent_api",
            lambda *a, **kw: api("http://localhost"),
        )
        args = [
            "usage",
            "rollup",
            *target,
            "--since",
            WINDOW["start_at"],
            "--until",
            WINDOW["end_at"],
            "--top-sessions",
            "1",
            "--session-order",
            "cost",
        ]
        if priced:
            path = tmp_path / "prices.json"
            path.write_text(json.dumps(prices()))
            args += ["--price-book", str(path)]
        assert main([*args, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["session_ranking"]["effective_order"] == ("cost" if priced else "tokens")
        assert main(args) == 0
        text = capsys.readouterr().out
        assert ("Top sessions by cost" if priced else "Top sessions by tokens") in text
        assert "Title cost-winner" in text if priced else "Title token-winner" in text
        if priced:
            assert "Cost by model" in text and "Cost by provider" in text
        else:
            assert "pricing not configured" in text


@pytest.mark.parametrize("enabled", [None, False, 1, "true"])
def test_cost_cli_rejects_unsupported_or_malformed_capability(monkeypatch, capsys, enabled):
    posts = []

    class Api:
        def get(self, *args, **kwargs):
            return {
                "capabilities": {"surfaces": {"usage_cost_details": {"read": {"enabled": enabled}}}}
            }

        def post(self, *args, **kwargs):
            posts.append(args)
            raise AssertionError("must not request unsupported cost ordering")

        def close(self):
            pass

    monkeypatch.setattr("cayu.cli._cloud_agent_api.CayuApi", lambda *a, **kw: Api())
    assert (
        main(
            [
                "usage",
                "rollup",
                "--server-url",
                "http://localhost",
                "--since",
                WINDOW["start_at"],
                "--until",
                WINDOW["end_at"],
                "--top-sessions",
                "1",
                "--session-order",
                "cost",
                "--json",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "does not advertise usage_cost_details" in output.out
    assert not posts
    assert "session_breakdown" not in output.out


def test_cost_cli_live_loopback(tmp_path, capsys):
    import socket
    import threading
    import time

    import uvicorn

    store = InMemorySessionStore()
    asyncio.run(seed(store))
    app = CayuApp(session_store=store, enable_logging=False)
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(app, config=ServerConfig.local_development()), log_level="error"
        )
    )
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(prices()))
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
            arguments = [
                "usage",
                "rollup",
                "--server-url",
                f"http://127.0.0.1:{port}",
                "--since",
                WINDOW["start_at"],
                "--until",
                WINDOW["end_at"],
                "--top-sessions",
                "1",
                "--session-order",
                "cost",
            ]
            assert main([*arguments, "--price-book", str(path), "--json"]) == 0
            data = json.loads(capsys.readouterr().out)
            assert data["session_ranking"]["effective_order"] == "cost"
            assert data["session_breakdown"]["groups"][0]["title"] == "Title cost-winner"
            assert main(arguments) == 0
            assert "Top sessions by tokens" in capsys.readouterr().out
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            assert not worker.is_alive()


def test_cost_cli_rejected_pricing_is_unavailable(monkeypatch, tmp_path, capsys):
    from tests.server.test_usage_cost_details import seed_rejected_pricing

    store = InMemorySessionStore()
    asyncio.run(seed(store))
    asyncio.run(seed_rejected_pricing(store))
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(prices()))
    with TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=client, **kwargs),
        )
        args = [
            "usage",
            "rollup",
            "--server-url",
            "http://localhost",
            "--since",
            WINDOW["start_at"],
            "--until",
            WINDOW["end_at"],
            "--price-book",
            str(path),
            "--top-sessions",
            "1",
            "--session-order",
            "cost",
        ]
        assert main(args) == 0
        output = capsys.readouterr().out
        for dimension in ("provider", "model"):
            section = (
                output.split(f"Cost by {dimension}\n", 1)[1]
                .split("Top sessions", 1)[0]
                .split("Cost by", 1)[0]
            )
            assert "Unavailable" in section
            assert " USD" not in section
            assert "0 (no priced work)" not in section
        assert "Cost ranking unavailable: prices incomplete" in output
