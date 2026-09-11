"""Real Runtime task loop with controlled deployment readiness/close boundaries."""

import asyncio
import importlib
import inspect
import signal
import sys
import time
import tomllib
from types import SimpleNamespace

import pytest

from cayu import CayuApp, InMemoryTaskStore, TaskCreate, TaskStatus, complete_managed_task
from cayu.cli.project import project_context
from tests.cli.test_worker import _running_worker, _wait_for_worker_start
from tests.qualification.test_repository_maintenance_application import project as project


@pytest.fixture(params=["coding", "git_preparation", "git_delivery", "github_delivery"])
def role(project, monkeypatch, request):
    with project_context(project):
        module = importlib.import_module("operations.maintenance_roles")
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        calls = []

        async def ready():
            calls.append("ready")

        async def close(**kwargs):
            calls.append("close")
            return True

        owned = SimpleNamespace(
            application=SimpleNamespace(app=app, artifact_store=object()),
            reservations=object(),
            validate_startup_schema=ready,
            aclose=close,
        )

        def bind(actual, *, agent_name):
            assert actual is app and agent_name == module.AGENT.name
            calls.append("bind")
            return owned

        monkeypatch.setattr(module, "bind_maintenance_deployment", bind)
        monkeypatch.setattr(module, "configured_git_broker", lambda store: object())
        monkeypatch.setattr(
            module, "configured_github_connector_factory", lambda store: lambda: object()
        )
        yield module, app, owned, calls, request.param


def test_named_role_has_exact_cli_contract(project):
    config = tomllib.loads((project / "pyproject.toml").read_text())
    with project_context(project):
        for name in ("coding", "git_preparation", "git_delivery", "github_delivery"):
            assert (
                config["tool"]["cayu"]["workers"][name]
                == f"operations.maintenance_roles:run_{name}"
            )
            target = getattr(importlib.import_module("operations.maintenance_roles"), f"run_{name}")
            assert inspect.iscoroutinefunction(target)
            parameters = tuple(inspect.signature(target).parameters.values())
            assert [value.name for value in parameters] == ["app", "stop"]
            assert all(value.default is inspect.Parameter.empty for value in parameters)


@pytest.mark.parametrize("cancel", [False, True])
def test_stop_retains_real_claim_until_handler_and_cleanup_settle(role, monkeypatch, cancel):
    module, app, _owned, calls, name = role

    async def scenario():
        stop, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        queued = await app.task_store.create_task(
            TaskCreate(type=f"maintenance.{name}", title="fix")
        )

        async def handle(application, _reservations, claimed, worker_id, *broker):
            assert application.app is app
            calls.append("handler")
            entered.set()
            await release.wait()
            await complete_managed_task(app.task_store, claimed, worker_id, {"fixture": True})
            calls.append("settled")

        monkeypatch.setattr(module, f"handle_{name}_task", handle)
        owner = asyncio.create_task(getattr(module, f"run_{name}")(app, stop))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                owner.cancel("first-stop")
                await asyncio.sleep(0)
                owner.cancel("second-stop")
            else:
                stop.set()
            await asyncio.sleep(0)
            assert not owner.done() and "close" not in calls
            current = await app.task_store.load_task(queued.id)
            assert current is not None and current.status is TaskStatus.CLAIMED
            release.set()
            if cancel:
                done, _ = await asyncio.wait((owner,), timeout=5)
                assert done
                with pytest.raises(asyncio.CancelledError, match="first-stop"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 2
            else:
                await asyncio.wait_for(owner, 5)
            assert calls == ["bind", "ready", "handler", "settled", "close"]
            terminal = await app.task_store.load_task(queued.id)
            assert terminal is not None and terminal.status is TaskStatus.COMPLETED
            assert terminal.worker_id is None
        finally:
            stop.set()
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX CLI signal contract")
@pytest.mark.parametrize("blocked_close", [False, True])
@pytest.mark.parametrize("name", ["coding", "git_preparation", "git_delivery", "github_delivery"])
def test_named_cli_sigterm_settles_or_reports_failed_shutdown(project, blocked_close, name):
    # Only native dependency construction/readiness/closure is replaced. The
    # emitted role, public Runtime worker and real CLI signal owner all execute.
    root = project / "app.py"
    root.write_text(
        root.read_text()
        + """

def build_maintenance_app():
    import asyncio
    from pathlib import Path
    from types import SimpleNamespace
    from cayu import CayuApp, InMemoryTaskStore
    from operations import maintenance_roles

    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)

    async def ready():
        Path("role-started").write_text("ready")

    async def close(**kwargs):
        Path("role-close-started").write_text("closing")
        if BLOCKED_CLOSE:
            await asyncio.Event().wait()
        Path("role-closed").write_text("closed")
        return True

    owned = SimpleNamespace(application=SimpleNamespace(app=app, artifact_store=None), reservations=None,
                            validate_startup_schema=ready, aclose=close)
    maintenance_roles.bind_maintenance_deployment = lambda actual, **kwargs: owned
    maintenance_roles.configured_git_broker = lambda store: object()
    maintenance_roles.configured_github_connector_factory = lambda store: lambda: object()
    return app
""".replace("BLOCKED_CLOSE", repr(blocked_close))
    )
    with _running_worker(cwd=project, name=name, shutdown_grace_seconds="0.5") as process:
        _wait_for_worker_start(process, project / "role-started")
        began = time.monotonic()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == (124 if blocked_close else 143), stdout + stderr
        assert time.monotonic() - began < 3
        assert (project / "role-close-started").is_file()
        assert (project / "role-closed").is_file() is not blocked_close
        if blocked_close:
            assert "did not stop within 0.5 seconds after SIGTERM" in stderr


def test_primary_and_cleanup_errors_keep_ordered_originals(role, monkeypatch):
    module, _app, owned, calls, name = role
    primary, secondary = RuntimeError("primary"), OSError("cleanup")

    async def fail_ready():
        raise primary

    async def fail_close(**kwargs):
        calls.append("close")
        raise secondary

    monkeypatch.setattr(owned, "validate_startup_schema", fail_ready)
    monkeypatch.setattr(owned, "aclose", fail_close)

    async def scenario():
        stop = asyncio.Event()
        with pytest.raises(ExceptionGroup) as caught:
            await getattr(module, f"run_{name}")(owned.application.app, stop)
        assert caught.value.exceptions == (primary, secondary)
        assert stop.is_set() and calls == ["bind", "close"]

    asyncio.run(scenario())


def test_handler_configuration_failure_closes_without_claiming(role, monkeypatch):
    module, app, _owned, calls, name = role
    failure = ValueError("invalid host configuration")

    def fail(*args):
        raise failure

    monkeypatch.setattr(
        module,
        "_coding_handler"
        if name == "coding"
        else "configured_github_connector_factory"
        if name == "github_delivery"
        else "configured_git_broker",
        fail,
    )

    async def scenario():
        queued = await app.task_store.create_task(
            TaskCreate(type=f"maintenance.{name}", title="fix")
        )
        stop = asyncio.Event()
        with pytest.raises(ValueError) as caught:
            await getattr(module, f"run_{name}")(app, stop)
        assert caught.value is failure
        assert calls == ["bind", "ready", "close"] and stop.is_set()
        assert await app.task_store.load_task(queued.id) == queued

    asyncio.run(scenario())


def test_incomplete_close_retains_role_and_observer_cancellation(role, monkeypatch):
    module, app, owned, calls, name = role

    async def scenario():
        stop = asyncio.Event()
        stop.set()  # Real Runtime loop returns without claiming new work.
        observed, release = asyncio.Event(), asyncio.Event()
        count = 0
        failure = OSError("close failed")

        async def close(**kwargs):
            nonlocal count
            count += 1
            if count == 1:
                observed.set()
                return False
            await release.wait()
            raise failure

        monkeypatch.setattr(owned, "aclose", close)
        owner = asyncio.create_task(getattr(module, f"run_{name}")(app, stop))
        try:
            await asyncio.wait_for(observed.wait(), 5)
            assert not owner.done()
            owner.cancel("retain-cleanup")
            await asyncio.sleep(0)
            assert not owner.done()
            release.set()
            done, _ = await asyncio.wait((owner,), timeout=5)
            assert done
            with pytest.raises(asyncio.CancelledError, match="retain-cleanup") as caught:
                await owner
            assert caught.value.__cause__ is failure
            assert owner.cancelled() and owner.cancelling() == 1
            assert count == 2 and calls == ["bind", "ready"]
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())
