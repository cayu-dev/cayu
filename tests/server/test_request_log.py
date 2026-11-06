"""Cayu's request log: a line with its duration per finished request, and one for a
request still running past `long_request_seconds`, with route templates only."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cayu.applications import CayuApp
from cayu.cli.serve import ServeError, _serve_settings, serve_uvicorn_options
from cayu.server import (
    OpenAccess,
    RequestTimingConfig,
    ServerConfig,
    create_server,
    mount_cayu,
)
from cayu.server._request_timing import (
    REQUEST_LOGGER_NAME,
    RequestTimingMiddleware,
    RequestTimingRecorder,
)


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Capture the request log even after a uvicorn config in this process (such as
    `cayu serve`'s) gave its logger its own handler and stopped it propagating."""

    logger = logging.getLogger(REQUEST_LOGGER_NAME)
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "disabled", False)
    monkeypatch.setattr(logger, "handlers", [])
    caplog.set_level(logging.INFO, logger=REQUEST_LOGGER_NAME)
    return caplog


def _lines(caplog: pytest.LogCaptureFixture, level: int | None = None) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == REQUEST_LOGGER_NAME and (level is None or record.levelno == level)
    ]


def _scope(path: str, *, method: str = "GET", query: bytes = b"") -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query,
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"authorization", b"Bearer header-secret")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }


def test_each_finished_request_logs_its_route_status_and_duration(caplog) -> None:
    app = FastAPI()

    @app.get("/api/items/{item_id}")
    async def item(item_id: str) -> dict:
        return {"id": item_id}

    @app.get("/api/broken")
    async def broken() -> dict:
        raise RuntimeError("handler failed")

    app.add_middleware(
        RequestTimingMiddleware, recorder=RequestTimingRecorder(RequestTimingConfig())
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        assert (
            client.get("/api/items/item-path-secret", params={"token": "query-secret"}).status_code
            == 200
        )
        assert client.get("/missing/path-secret").status_code == 404
        assert client.get("/api/broken").status_code == 500

    found, missing, failed = _lines(caplog, logging.INFO)
    assert re.fullmatch(r"GET /api/items/\{item_id\} 200 \d+(\.\d)? ms", found)
    assert re.fullmatch(r"GET \(unmatched\) 404 \d+(\.\d)? ms", missing)
    assert re.fullmatch(r"GET /api/broken 500 error \d+(\.\d)? ms", failed)
    text = caplog.text
    for secret in ("item-path-secret", "query-secret", "path-secret", "header-secret"):
        assert secret not in text
    (record,) = [r for r in caplog.records if r.getMessage() == found]
    assert record.cayu_route == "/api/items/{item_id}"
    assert record.cayu_status == 200
    assert record.cayu_duration_ms >= 0


def test_a_request_still_running_is_logged_before_it_answers(caplog) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        app = FastAPI()

        @app.post("/api/inbox/{item_id}/settle")
        async def settle(item_id: str) -> dict:
            await asyncio.wait_for(release.wait(), timeout=10)
            return {"settled": item_id}

        config = RequestTimingConfig(long_request_seconds=0.01)
        app.add_middleware(RequestTimingMiddleware, recorder=RequestTimingRecorder(config))

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict] = []

        async def send(message: dict) -> None:
            sent.append(message)

        request = asyncio.create_task(
            app(_scope("/api/inbox/item-secret/settle", method="POST"), receive, send)
        )
        # Wait for the line rather than a fixed time; the request is held open until then.
        for _ in range(1000):
            if _lines(caplog, logging.WARNING):
                break
            await asyncio.sleep(0.01)
        (running,) = _lines(caplog, logging.WARNING)
        assert sent == []
        release.set()
        await asyncio.wait_for(request, timeout=10)

        match = re.fullmatch(
            r"POST /api/inbox/\{item_id\}/settle still running after \S+ m?s, "
            r"no response yet \(request ([0-9a-f]{8})\)",
            running,
        )
        assert match is not None, running
        (finished,) = _lines(caplog, logging.INFO)
        assert re.fullmatch(
            r"POST /api/inbox/\{item_id\}/settle 200 \S+ m?s \(request " + match.group(1) + r"\)",
            finished,
        ), finished
        assert "item-secret" not in caplog.text

    asyncio.run(scenario())


def test_event_streams_and_turned_off_lines_stay_quiet(caplog) -> None:
    async def scenario(config: RequestTimingConfig, path: str) -> None:
        app = FastAPI()
        delivered = asyncio.Event()

        async def events():
            yield "data: first\n\n"
            await asyncio.wait_for(delivered.wait(), timeout=10)
            # Long past the threshold: an open stream is not a stalled request.
            await asyncio.sleep(0.1)
            yield "data: second\n\n"

        @app.get("/api/stream")
        async def stream() -> StreamingResponse:
            return StreamingResponse(events(), media_type="text/event-stream")

        @app.get("/api/slow")
        async def slow() -> dict:
            await asyncio.sleep(0.1)
            return {}

        app.add_middleware(RequestTimingMiddleware, recorder=RequestTimingRecorder(config))

        async def receive() -> dict:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            if b"first" in message.get("body", b""):
                delivered.set()

        await asyncio.wait_for(app(_scope(path), receive, send), timeout=10)

    asyncio.run(scenario(RequestTimingConfig(long_request_seconds=0.01), "/api/stream"))
    (line,) = _lines(caplog)
    assert re.fullmatch(r"GET /api/stream 200 stream \S+ m?s", line)

    caplog.clear()
    quiet = RequestTimingConfig(request_log=False, long_request_seconds=None)
    asyncio.run(scenario(quiet, "/api/slow"))
    assert _lines(caplog) == []


def test_long_request_seconds_is_validated() -> None:
    assert RequestTimingConfig().long_request_seconds == 5.0
    assert RequestTimingConfig(long_request_seconds=None).long_request_seconds is None
    for value in (0, -1, float("inf"), True, 86_401):
        with pytest.raises(ValidationError):
            RequestTimingConfig(long_request_seconds=value)


def test_server_says_whether_its_request_log_covers_every_request() -> None:
    server = create_server(CayuApp(enable_logging=False), config=ServerConfig.local_development())
    assert server.state.cayu_request_log_covers_server is True

    host = FastAPI()
    mount_cayu(host, CayuApp(enable_logging=False), path="/cayu", access=OpenAccess())
    assert host.state.cayu_request_log_covers_server is False

    observed = FastAPI()
    mount_cayu(
        observed,
        CayuApp(enable_logging=False),
        path="/cayu",
        access=OpenAccess(),
        observe_host_requests=True,
    )
    assert observed.state.cayu_request_log_covers_server is True


def test_serve_logs_cayu_request_lines_in_place_of_the_access_log() -> None:
    import uvicorn

    options = serve_uvicorn_options(uvicorn, covered=True)
    assert options["access_log"] is False
    logger = options["log_config"]["loggers"][REQUEST_LOGGER_NAME]
    assert logger == {"handlers": ["cayu_request"], "level": "INFO", "propagate": False}
    assert options["log_config"]["handlers"]["cayu_request"]["stream"] == "ext://sys.stdout"
    # uvicorn's own configuration is kept as it is.
    for key, value in uvicorn.config.LOGGING_CONFIG.items():
        if key not in {"handlers", "loggers"}:
            assert options["log_config"][key] == value
    for key in ("handlers", "loggers"):
        assert uvicorn.config.LOGGING_CONFIG[key].items() <= options["log_config"][key].items()
    assert serve_uvicorn_options(uvicorn, covered=False)["access_log"] is True


def _project(tmp_path: Path, serve: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n\n'
        '[tool.cayu]\nfactory = "app:build_app"\n\n'
        f"[tool.cayu.serve]\n{serve}\n"
    )
    return tmp_path


def test_serve_settings_read_long_request_seconds(tmp_path: Path) -> None:
    settings = _serve_settings(_project(tmp_path, "long_request_seconds = 30"))
    assert RequestTimingConfig(**dict(settings.request_timing)).long_request_seconds == 30.0
    settings = _serve_settings(_project(tmp_path, "long_request_seconds = false"))
    assert RequestTimingConfig(**dict(settings.request_timing)).long_request_seconds is None
    assert _serve_settings(_project(tmp_path, 'auth = "a:b"')).request_timing == ()
    for value in ("0", "true", '"5"', "100000"):
        with pytest.raises(ServeError, match="long_request_seconds"):
            _serve_settings(_project(tmp_path, f"long_request_seconds = {value}"))


def test_a_background_task_after_the_response_is_not_a_request_still_running(caplog) -> None:
    from starlette.background import BackgroundTask
    from starlette.responses import JSONResponse

    async def scenario() -> None:
        app = FastAPI()

        @app.get("/api/report")
        async def report() -> JSONResponse:
            # Runs after the whole body was sent, long past the threshold.
            return JSONResponse({}, background=BackgroundTask(asyncio.sleep, 0.1))

        config = RequestTimingConfig(long_request_seconds=0.01)
        app.add_middleware(RequestTimingMiddleware, recorder=RequestTimingRecorder(config))

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_message: dict) -> None:
            return None

        await asyncio.wait_for(app(_scope("/api/report"), receive, send), timeout=10)

    asyncio.run(scenario())
    (line,) = _lines(caplog)
    assert re.fullmatch(r"GET /api/report 200 \S+ m?s", line), line


def test_reload_leaves_out_the_access_log_only_for_an_app_the_log_covers(monkeypatch) -> None:
    from types import SimpleNamespace

    import uvicorn

    from cayu.cli import _serve_reload

    # uvicorn applies its logging configuration to this process; restore it afterwards.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", REQUEST_LOGGER_NAME):
        logger = logging.getLogger(name)
        for attribute in ("handlers", "propagate", "level", "disabled"):
            monkeypatch.setattr(logger, attribute, getattr(logger, attribute))
    served: list[bool] = []
    monkeypatch.setattr(
        uvicorn.Server, "run", lambda self, sockets=None: served.append(self.config.access_log)
    )
    for covered in (False, True):
        app = SimpleNamespace(request_log_covers_server=covered, close_sync=lambda: None)
        monkeypatch.setattr(_serve_reload, "create_reload_app", lambda app=app: app)
        config = uvicorn.Config(app="unused:app", **serve_uvicorn_options(uvicorn, covered=False))
        assert config.access_log is True
        _serve_reload._ReloadServer(config).run()
        assert bool(logging.getLogger("uvicorn.access").handlers) is not covered
    assert served == [True, False]
