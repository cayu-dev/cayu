"""Reload through the real CLI supervisor, spawned workers and filesystem edits."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from cayu.cli.serve import parse_serve_arguments

ROOT = Path(__file__).resolve().parents[2]


def _source(version: str, persistent: bool, bootstrap_async: bool = False) -> str:
    return f'''import os
from pathlib import Path
from cayu import AgentSpec, CayuApp
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.storage import SQLiteSessionStore
import asyncio

async def load_settings():
    return "settings"

def build_app():
    if {bootstrap_async!r}:
        assert asyncio.run(load_settings()) == "settings"
    with Path("builds.log").open("a") as log:
        log.write(str(os.getpid()) + ":{version}\\n")
    app = CayuApp(enable_logging=False{', session_store=SQLiteSessionStore("sessions.db")' if persistent else ""})
    app.register_provider(ScriptedModelProvider([[ModelStreamEvent.text_delta("ok"), ModelStreamEvent.completed()]]), default=True)
    app.register_agent(AgentSpec(name="{version}", model="scripted"))
    return app
'''


def _wait(process, log_path, predicate, *, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None, log_path.read_text()
        try:
            result = predicate()
            if result:
                return result
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.1)
    pytest.fail(f"Server did not reach expected state:\n{log_path.read_text()}")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX subprocess signal/reaping acceptance")
@pytest.mark.parametrize(
    ("persistent", "bootstrap_async"), [(False, False), (True, False), (False, True)]
)
def test_reload_restarts_on_file_edits_and_warns_about_state(tmp_path, persistent, bootstrap_async):
    (tmp_path / "pyproject.toml").write_text('[tool.cayu]\nfactory = "reload_app:build_app"\n')
    source = tmp_path / "reload_app.py"
    source.write_text(_source("before", persistent, bootstrap_async))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cayu.cli", "serve", "--dev", "--reload", "--port", str(port)],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:

                def has_agent(name):
                    response = client.get("/api/agents")
                    return response.status_code == 200 and any(
                        agent["name"] == name for agent in response.json()["agents"]
                    )

                _wait(process, log_path, lambda: has_agent("before"))
                response = client.post(
                    "/api/run",
                    json={"session_id": "retained", "agent": "before", "prompt": "go"},
                    timeout=15,
                )
                assert response.status_code == 200, response.text
                assert client.get("/api/sessions/retained").status_code == 200
                # Ensure source mtime differs even with coarse filesystem timestamps.
                time.sleep(1.1)
                source.write_text(_source("after_edit", persistent, bootstrap_async))
                _wait(process, log_path, lambda: has_agent("after_edit"))
                assert client.get("/api/sessions/retained").status_code == (
                    200 if persistent else 404
                )
                first_builds = (tmp_path / "builds.log").read_text().splitlines()
                assert len(first_builds) == 2
                assert all(int(line.split(":")[0]) != process.pid for line in first_builds)
                time.sleep(1.1)
                source.write_text(_source("third_version", persistent, bootstrap_async))
                _wait(process, log_path, lambda: has_agent("third_version"))
                warnings = log_path.read_text()
                assert warnings.count("WARNING: cayu serve --reload") == 3
                assert "in-memory state is lost" in warnings
                assert "active runs can be interrupted" in warnings
                assert "Persistent stores retain data" in warnings
                time.sleep(1.1)
                source.write_text('raise RuntimeError("temporary invalid edit")\n')
                _wait(
                    process, log_path, lambda: "Application startup failed" in log_path.read_text()
                )
                time.sleep(1.1)
                source.write_text(_source("after_repair", persistent, bootstrap_async))
                _wait(process, log_path, lambda: has_agent("after_repair"))
                assert log_path.read_text().count("WARNING: cayu serve --reload") == 5
                assert client.get("/api/sessions/retained").status_code == (
                    200 if persistent else 404
                )
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                pytest.fail(f"Reload supervisor did not terminate:\n{log_path.read_text()}")
        assert process.returncode == 0, log_path.read_text()
        for line in (tmp_path / "builds.log").read_text().splitlines():
            with pytest.raises(ProcessLookupError):
                os.kill(int(line.split(":")[0]), 0)


def test_reload_option_is_explicit_and_shared_with_command_readers():
    assert parse_serve_arguments(["--dev"]).reload is False
    assert parse_serve_arguments(["--reload", "--auth", "project:auth"]).reload is True


@pytest.mark.parametrize("reload", [False, True])
def test_serve_modes_preserve_auth_and_project_lifetime(tmp_path, monkeypatch, capsys, reload):
    import uvicorn
    from fastapi.testclient import TestClient

    import cayu.cli._serve_reload as reload_cli
    from cayu.cli import main

    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "reload_app:build_app"\n'
        '[tool.cayu.serve]\nauth = "reload_app:AUTH"\n'
        'startup_recovery_statuses = ["running"]\nrecovery_inactive_after_seconds = 900\n'
    )
    (tmp_path / "reload_app.py").write_text(
        _source("protected", False)
        + '\nfrom cayu.server import BasicAuth\nAUTH = BasicAuth(username="test", password="test")\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("_CAYU_SERVE_RELOAD", "previous value")
    called = []

    def supervise(supervisor):
        assert "reload_app" not in sys.modules
        assert not (tmp_path / "builds.log").exists()
        assert supervisor.config.reload is True
        assert supervisor.config.lifespan == "on"
        assert supervisor.config.reload_dirs == [tmp_path]
        supervisor.target(sockets=[])

    def run(server, **options):
        assert options == {"host": "127.0.0.1", "port": 8000}
        with TestClient(server) as client:
            assert Path.cwd() == tmp_path
            assert client.get("/api/agents").status_code == 401
            assert client.get("/api/agents", auth=("test", "wrong")).status_code == 401
            assert client.get("/api/agents", auth=("test", "test")).status_code == 200
            actual = server._server if reload else server
            config = actual.state.cayu_server_config
            assert config.lifecycle.recovery_inactive_after_seconds == 900
            assert config.request_timing is not None
        called.append(True)

    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(uvicorn.Config, "bind_socket", lambda self: socket.socket())
    monkeypatch.setattr(reload_cli.WatchFilesReload, "run", supervise)
    monkeypatch.setattr(
        uvicorn.Server,
        "run",
        lambda self, sockets=None: run(
            self.config.app, host=self.config.host, port=self.config.port
        ),
    )
    assert main(["serve", *(["--reload"] if reload else [])]) == 0
    assert called == [True]
    assert "reload_app" not in sys.modules
    assert os.environ["_CAYU_SERVE_RELOAD"] == "previous value"
    assert ("WARNING: cayu serve --reload" in capsys.readouterr().err) is reload


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "Refusing to start an unauthenticated server"),
        (["--dev", "--host", "0.0.0.0"], "non-loopback host"),
    ],
)
def test_reload_refuses_unsafe_access_before_import(
    tmp_path, monkeypatch, capsys, arguments, message
):
    from cayu.cli import main

    (tmp_path / "pyproject.toml").write_text('[tool.cayu]\nfactory = "never_import:build_app"\n')
    (tmp_path / "never_import.py").write_text('raise AssertionError("factory imported")\n')
    monkeypatch.chdir(tmp_path)
    assert main(["serve", "--reload", *arguments]) == 1
    assert message in capsys.readouterr().err
    assert "never_import" not in sys.modules


def test_reload_keeps_maintained_service_and_deployment_policy(tmp_path, monkeypatch, capsys):
    import uvicorn
    from fastapi.testclient import TestClient

    import cayu.cli._serve_reload as reload_cli
    from cayu.cli import main
    from cayu.cli.scaffold import project_files

    for relative, content in project_files("reload-service", preset="service").items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRODUCT_AUTH_TOKENS_JSON", raising=False)
    monkeypatch.delenv("CAYU_OPERATOR_BEARER_TOKEN", raising=False)
    mode = "development"
    checked = []

    def supervise(supervisor):
        supervisor.target(sockets=[])
        checked.append(mode)

    def run(server, sockets=None):
        assert mode == "development", "unsafe service must fail before serving"
        with TestClient(server.config.app) as client:
            assert client.get("/cayu/api/contract").status_code == 200
            assert (
                server.config.app._server.state.cayu_public_service_manifest.mode == "development"
            )

    monkeypatch.setattr(uvicorn.Config, "bind_socket", lambda self: socket.socket())
    monkeypatch.setattr(uvicorn.Server, "run", run)
    monkeypatch.setattr(reload_cli.WatchFilesReload, "run", supervise)
    assert main(["serve", "--dev", "--reload"]) == 0
    mode = "production"
    assert main(["serve", "--reload"]) == 0
    assert checked == ["development", "production"]
    assert capsys.readouterr().err.count("WARNING: cayu serve --reload") == 2


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_reload_worker_closes_context_after_startup_failure(tmp_path, monkeypatch, cleanup_fails):
    import json
    from types import SimpleNamespace

    import cayu.cli._serve_reload as reload_cli

    (tmp_path / "pyproject.toml").write_text('[tool.cayu]\nfactory = "unused:build_app"\n')
    monkeypatch.setenv(
        "_CAYU_SERVE_RELOAD", json.dumps({"root": str(tmp_path), "arguments": ["--dev"]})
    )
    closed = []
    primary = ValueError("startup failure")
    cleanup = RuntimeError("cleanup failure")

    async def close():
        closed.append(True)
        if cleanup_fails:
            raise cleanup

    def build(*_args):
        raise primary

    monkeypatch.setattr(
        reload_cli,
        "serve_project_control_plane",
        lambda *a, **kw: SimpleNamespace(close=close),
    )
    monkeypatch.setattr(reload_cli, "_build_serve_app", build)
    cwd = Path.cwd()
    with pytest.raises(ExceptionGroup if cleanup_fails else ValueError) as caught:
        reload_cli.create_reload_app()
    assert closed == [True]
    assert Path.cwd() == cwd
    if cleanup_fails:
        assert caught.value.exceptions == (primary, cleanup)
    else:
        assert caught.value is primary


def test_reload_worker_preserves_delivered_cancellation_and_closes_context(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import cayu.cli._serve_reload as reload_cli

    (tmp_path / "pyproject.toml").write_text('[tool.cayu]\nfactory = "unused:build_app"\n')
    monkeypatch.setenv(
        "_CAYU_SERVE_RELOAD", json.dumps({"root": str(tmp_path), "arguments": ["--dev"]})
    )
    closed = []

    async def close():
        closed.append(True)

    async def exercise():
        entered = asyncio.Event()

        async def inner(scope, receive, send):
            assert await receive() == {"type": "lifespan.startup"}
            entered.set()
            await asyncio.Event().wait()

        async def receive():
            return {"type": "lifespan.startup"}

        monkeypatch.setattr(
            reload_cli,
            "serve_project_control_plane",
            lambda *a, **kw: SimpleNamespace(close=close),
        )
        monkeypatch.setattr(reload_cli, "_build_serve_app", lambda *a: inner)
        server = reload_cli.create_reload_app()
        task = asyncio.create_task(server({"type": "lifespan"}, receive, None))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 1
        assert task.cancelled()
        assert closed == [True]
        assert server._server is None

    asyncio.run(exercise())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX subprocess signal/reaping acceptance")
@pytest.mark.parametrize("action", ["edit", "terminate"])
def test_reload_bounds_live_run_request_drain(tmp_path, action):
    (tmp_path / "pyproject.toml").write_text('[tool.cayu]\nfactory = "reload_app:build_app"\n')
    source = tmp_path / "reload_app.py"
    code = (
        _source("blocked", True)
        .replace(
            "def build_app():",
            """import asyncio
class BlockingProvider(ScriptedModelProvider):
    async def stream(self, request):
        Path("provider-entered").touch()
        try:
            await asyncio.Event().wait()
            yield ModelStreamEvent.completed()
        finally:
            Path("provider-stopped").touch()

def build_app():""",
        )
        .replace(
            "app.register_provider(ScriptedModelProvider(",
            "app.register_provider(BlockingProvider(",
        )
    )
    source.write_text(code)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cayu.cli", "serve", "--dev", "--reload", "--port", str(port)],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:
                _wait(process, log_path, lambda: client.get("/api/agents").status_code == 200)
                with (
                    socket.create_connection(("127.0.0.1", port), timeout=3) as pending_body,
                    client.stream(
                        "POST",
                        "/api/run",
                        json={"session_id": "in-flight", "agent": "blocked", "prompt": "hold"},
                    ) as response,
                ):
                    assert response.status_code == 200
                    _wait(process, log_path, lambda: (tmp_path / "provider-entered").exists())
                    # 100 Continue proves the server entered a second request's
                    # body read. Keep that request and the run stream open.
                    pending_body.sendall(
                        b"POST /api/run HTTP/1.1\r\nHost: localhost\r\n"
                        b"Content-Type: application/json\r\nContent-Length: 100000\r\n"
                        b"Expect: 100-continue\r\n\r\n"
                    )
                    assert b"100 Continue" in pending_body.recv(1024)
                    old_pid = int((tmp_path / "builds.log").read_text().split(":")[0])
                    if action == "edit":
                        source.write_text(_source("replacement", True))

                        def replaced():
                            body = client.get("/api/agents")
                            return body.status_code == 200 and any(
                                row["name"] == "replacement" for row in body.json()["agents"]
                            )

                        _wait(process, log_path, replaced, timeout=45)
                        session = client.get("/api/sessions/in-flight").json()
                        # Reload does not invent an automatic recovery policy. A blocked
                        # durable run remains visible for explicit recovery.
                        assert session["status"] == "running", session
                    else:
                        process.send_signal(signal.SIGTERM)
                        process.wait(timeout=45)
                    with pytest.raises(ProcessLookupError):
                        os.kill(old_pid, 0)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        assert process.returncode == 0, log_path.read_text()


@pytest.mark.parametrize("phase", ["shutdown", "startup_failure"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_reload_waits_for_outer_cleanup_before_uvicorn_ack(
    tmp_path, monkeypatch, caplog, phase, cleanup_fails
):
    import json
    import re

    from uvicorn import Config
    from uvicorn.lifespan.on import LifespanOn

    from cayu.cli.scaffold import project_files
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    for relative, content in project_files("reload-service", preset="service").items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    project = tmp_path / "pyproject.toml"
    config = project.read_text()
    original_target = re.search(r'service_factory = "([^"]+)"', config).group(1)
    module, attribute = original_target.split(":")
    project.write_text(config.replace(original_target, "no_context:build_service"))
    (tmp_path / "no_context.py").write_text(f"""
from {module} import {attribute} as original

def build_service(*, mode):
    return original(mode=mode)
""")
    if phase == "startup_failure":
        from cayu import CayuApp

        async def fail_startup(self):
            raise ValueError("primary startup failed")

        monkeypatch.setattr(CayuApp, "start_model_policy", fail_startup)
    import logging

    monkeypatch.setattr(logging.getLogger("uvicorn"), "handlers", [])
    monkeypatch.setattr(logging.getLogger("uvicorn"), "propagate", True)
    monkeypatch.setattr(logging.getLogger("uvicorn.error"), "handlers", [])
    monkeypatch.setattr(logging.getLogger("uvicorn.error"), "propagate", True)
    monkeypatch.setenv(
        "_CAYU_SERVE_RELOAD", json.dumps({"root": str(tmp_path), "arguments": ["--dev"]})
    )
    original_close = SQLiteEvalStore.close
    cleanup_error = RuntimeError("outer cleanup failed")

    config = Config(
        "cayu.cli._serve_reload:create_reload_app",
        factory=True,
        lifespan="on",
        log_config=None,
        ws="none",
    )
    config.load()

    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()

        async def close(store):
            entered.set()
            await release.wait()
            await original_close(store)
            completed.set()
            if cleanup_fails:
                raise cleanup_error

        monkeypatch.setattr(SQLiteEvalStore, "close", close)
        lifecycle = LifespanOn(config)
        if phase == "shutdown":
            await lifecycle.startup()
            assert not lifecycle.should_exit
            waiter = asyncio.create_task(lifecycle.shutdown())
        else:
            waiter = asyncio.create_task(lifecycle.startup())
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            await asyncio.sleep(0)
            acknowledged_early = waiter.done() or (
                lifecycle.shutdown_event.is_set()
                if phase == "shutdown"
                else lifecycle.startup_event.is_set()
            )
            assert not completed.is_set()
        finally:
            release.set()
            await asyncio.wait_for(waiter, timeout=10)
            await asyncio.wait_for(completed.wait(), timeout=10)
        assert not acknowledged_early
        assert lifecycle.error_occurred is (cleanup_fails or phase == "startup_failure")
        if cleanup_fails:
            logged = [record.exc_info[1] for record in caplog.records if record.exc_info]
            failure = logged[-1]
            if phase == "startup_failure":
                assert isinstance(failure, ExceptionGroup)
                assert len(failure.exceptions) == 2
                assert str(failure.exceptions[0]) == "primary startup failed"
                assert failure.exceptions[1] is cleanup_error
            else:
                assert failure is cleanup_error
        assert lifecycle.should_exit is (cleanup_fails or phase == "startup_failure")

    asyncio.run(exercise())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX subprocess signal/reaping acceptance")
def test_reload_detects_python_file_creation_and_deletion(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.cayu]\nfactory = "reload_app:build_app"\n')
    (tmp_path / "reload_app.py").write_text(
        "from added_module import VALUE\n" + _source("files", False)
    )
    dependency = tmp_path / "added_module.py"

    def write_dependency(text: str) -> None:
        # One atomic rename, so the watcher sees one added file. A plain write
        # can reach it as separate create and modify batches on a loaded
        # runner, and each batch restarts the worker.
        staged = tmp_path / "added_module.staged"
        staged.write_text(text)
        os.replace(staged, dependency)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cayu.cli", "serve", "--dev", "--reload", "--port", str(port)],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait(process, log_path, lambda: "ModuleNotFoundError" in log_path.read_text())
            write_dependency('VALUE = "created"\n')
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2) as client:
                _wait(process, log_path, lambda: client.get("/api/agents").status_code == 200)
                first_pid = int((tmp_path / "builds.log").read_text().split(":")[0])
                missing_count = log_path.read_text().count("ModuleNotFoundError")
                dependency.unlink()
                # The startup warning precedes the import. Wait until the
                # replacement worker has observed the deletion before recreating
                # the module, or that worker can build an extra application.
                _wait(
                    process,
                    log_path,
                    lambda: log_path.read_text().count("ModuleNotFoundError") > missing_count,
                )
                with pytest.raises(ProcessLookupError):
                    os.kill(first_pid, 0)
                assert len((tmp_path / "builds.log").read_text().splitlines()) == 1
                write_dependency('VALUE = "recreated"\n')
                _wait(process, log_path, lambda: client.get("/api/agents").status_code == 200)
                assert len((tmp_path / "builds.log").read_text().splitlines()) == 2
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        assert process.returncode == 0, log_path.read_text()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_reload_worker_closes_bootstrap_resources_when_server_setup_fails(
    tmp_path, monkeypatch, cleanup_fails
):
    import json
    from types import SimpleNamespace

    import uvicorn

    import cayu.cli._serve_reload as reload_cli

    (tmp_path / "pyproject.toml").write_text('[tool.cayu]\nfactory = "unused:build_app"\n')
    monkeypatch.setenv(
        "_CAYU_SERVE_RELOAD", json.dumps({"root": str(tmp_path), "arguments": ["--dev"]})
    )
    primary = ValueError("server configuration failed")
    cleanup = RuntimeError("cleanup failed")
    closed = []

    async def close():
        closed.append(True)
        if cleanup_fails:
            raise cleanup

    def start(server, sockets=None):
        assert Path.cwd() == tmp_path
        raise primary

    monkeypatch.setattr(
        reload_cli,
        "serve_project_control_plane",
        lambda *a, **kw: SimpleNamespace(close=close),
    )
    monkeypatch.setattr(reload_cli, "_build_serve_app", lambda *a: object())
    monkeypatch.setattr(uvicorn.Server, "run", start)
    server = reload_cli._ReloadServer(uvicorn.Config("unused:app", log_config=None))
    cwd = Path.cwd()
    with pytest.raises(ExceptionGroup if cleanup_fails else ValueError) as caught:
        server.run()
    assert closed == [True]
    assert Path.cwd() == cwd
    if cleanup_fails:
        assert caught.value.exceptions == (primary, cleanup)
    else:
        assert caught.value is primary
