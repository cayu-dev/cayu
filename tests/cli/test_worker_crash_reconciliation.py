"""The actual CLI repairs a child run left by a hard shutdown on its next start."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import pytest
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus
from cayu.storage import open_application_stores

_PROJECT = """import asyncio
import os
from pathlib import Path

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.providers import ModelStreamEvent
from cayu.storage import open_application_stores
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import _Crashing, _request, _slot

class HeldTool(_Crashing):
    async def run(self, ctx, args):
        Path("dispatched").write_text("ready")
        await asyncio.Event().wait()

def build_app():
    stores = open_application_stores(os.environ["CAYU_SIGNAL_TEST_DATABASE_URL"], sqlite_path=os.environ["CAYU_SIGNAL_TEST_STORE"], tasks=False, event_watchers=False)
    app = CayuApp(session_store=stores.session_store, enable_logging=False)
    app.register_provider(VersionedFakeProvider([
        [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed({"finish_reason": "stop"})],
        [ModelStreamEvent.tool_call(id="call-1", name="record", arguments={}), ModelStreamEvent.completed({"finish_reason": "tool_calls"})],
    ]), default=True)
    app.register_agent(AgentSpec(name="scanner", model="fake-model"), tools=[HeldTool()])
    return app

async def scan(app, stop):
    async for _ in app.run(RunRequest(session_id="parent", agent_name="scanner", messages=[Message.text("user", "go")])):
        pass
    slot = _slot()
    app.register_environment_factory(slot.environment.spec, slot)
    async for _ in app.run(_request("scan").model_copy(update={"parent_session_id": "parent"})):
        pass

async def repair(app, stop):
    for _ in range(200):
        events = await app.session_store.load_events("scan")
        if any(event.type is EventType.INTERACTION_FAILED for event in events):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("The replacement worker left the fresh orphan running.")
"""


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGKILL])
def test_next_worker_start_repairs_a_child_after_hard_exit(
    tmp_path, database_url, termination
) -> None:
    (tmp_path / "project_worker.py").write_text(_PROJECT)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "project_worker:build_app"\n'
        '[tool.cayu.workers]\nscan = "project_worker:scan"\nrepair = "project_worker:repair"\n'
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    environment["CAYU_SIGNAL_TEST_DATABASE_URL"] = database_url
    environment["CAYU_SIGNAL_TEST_STORE"] = str(tmp_path / "sessions.db")
    command = [sys.executable, "-m", "cayu", "worker"]
    worker = subprocess.Popen(
        [*command, "scan", "--processes", "1", "--shutdown-grace-seconds", "0.1"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "dispatched").exists():
            if worker.poll() is not None or time.monotonic() >= deadline:
                worker.kill()
                raise AssertionError(worker.communicate(timeout=5))
            time.sleep(0.02)
        worker.send_signal(termination)
        stdout, stderr = worker.communicate(timeout=10)
        assert worker.returncode == (124 if termination == signal.SIGTERM else -signal.SIGKILL), (
            stdout,
            stderr,
        )

        async def inspect_crash():
            stores = open_application_stores(
                database_url,
                sqlite_path=tmp_path / "sessions.db",
                tasks=False,
                event_watchers=False,
            )
            try:
                assert (await stores.session_store.load("scan")).status is SessionStatus.RUNNING
                assert (
                    await stores.session_store.inspect_session_execution("scan")
                ).state == "owner_lost"
            finally:
                await stores.close()

        asyncio.run(inspect_crash())
        replacement = subprocess.run(
            [*command, "repair", "--processes", "1"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert replacement.returncode == 0, (replacement.stdout, replacement.stderr)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=5)

    async def inspect() -> None:
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "sessions.db", tasks=False, event_watchers=False
        )
        store = stores.session_store
        try:
            session = await store.load("scan")
            assert session.parent_session_id == "parent"
            assert session.status is SessionStatus.FAILED
            assert (await store.load("parent")).status is SessionStatus.COMPLETED
            events = await store.load_events("scan")
            failures = [e for e in events if e.type is EventType.SESSION_FAILED]
            assert len(failures) == 1
            assert failures[0].payload["reason"] == "orphaned_session_reconciled"
            assert failures[0].payload["tool_effect_outcome_unknown"][0]["tool_call_id"] == "call-1"
            assert [e.type for e in events].count(EventType.INTERACTION_FAILED) == 1
        finally:
            await stores.close()

    asyncio.run(inspect())


_RETAINED_PROJECT = """import asyncio
import json
import os
from pathlib import Path

import httpx

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate, run_task_worker
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import _Crashing, _slot

class HeldProvider(VersionedFakeProvider):
    async def stream(self, request):
        if os.environ.get("CAYU_SIGNAL_TEST_REPLACEMENT"):
            Path("redispatched").write_text("provider")
            raise AssertionError("Recovery must not dispatch the provider")
        if os.environ["CAYU_SIGNAL_TEST_DISPATCH"] == "model":
            Path("dispatched").write_text("model")
            await asyncio.Event().wait()
        async for event in super().stream(request):
            yield event

class HeldTool(_Crashing):
    async def run(self, ctx, args):
        if os.environ.get("CAYU_SIGNAL_TEST_REPLACEMENT"):
            Path("redispatched").write_text("tool")
            raise AssertionError("Recovery must not dispatch the tool")
        Path("dispatched").write_text("tool")
        await asyncio.Event().wait()

def build_app():
    stores = open_application_stores(
        os.environ["CAYU_SIGNAL_TEST_DATABASE_URL"],
        sqlite_path=os.environ["CAYU_SIGNAL_TEST_STORE"], event_watchers=False,
    )
    app = CayuApp(
        session_store=stores.session_store, task_store=stores.task_store,
        owned_resources=(stores,), enable_logging=False,
    )
    app.register_provider(HeldProvider([
        ModelStreamEvent.tool_call(id="call-1", name="record", arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]), default=True)
    app.register_agent(AgentSpec(name="scanner", model="fake-model"), tools=[HeldTool()])
    if os.environ["CAYU_SIGNAL_TEST_ENVIRONMENT"] == "registered":
        slot = _slot()
        app.register_environment_factory(slot.environment.spec, slot)
    return app

async def scan(app, stop):
    request = RunRequest(
        session_id="scan", agent_name="scanner", messages=[Message.text("user", "go")],
        environment_name=(
            "run-1-slot-1" if os.environ["CAYU_SIGNAL_TEST_ENVIRONMENT"] == "registered" else None
        ),
    )
    async def run():
        async for _ in app.run(request):
            pass
    if os.environ["CAYU_SIGNAL_TEST_TOPOLOGY"] == "direct":
        await run()
        return
    await app.create_task(TaskCreate(task_id="job", type="scan", restartable=True))
    async def handler(_app, task, worker_id):
        await run()
    await run_task_worker(
        app, app.task_store, handler, worker_id="lost-worker", max_tasks=1,
        lease_seconds=1, reclaim=False, stop=stop,
    )

async def repair(app, stop):
    for _ in range(100):
        session = await app.session_store.load("scan")
        events = await app.session_store.load_events("scan")
        if session.status is SessionStatus.INTERRUPTED and any(
            event.type is EventType.SESSION_INTERRUPTED for event in events
        ):
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("The replacement worker left the fresh orphan running.")
    # Wait for the startup pass to release its claim and finish the interaction.
    for _ in range(100):
        execution = await app.inspect_session_execution("scan")
        if execution.state != "executing":
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("The startup recovery never released its claim.")
    server = create_server(app, config=ServerConfig.local_development())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server), base_url="http://runtime.test"
    ) as client:
        sessions = await client.get("/api/sessions", params={"include": "execution_profile"})
        actions = await client.get("/api/pending-actions")
    assert sessions.status_code == actions.status_code == 200
    Path("recovered.json").write_text(json.dumps({
        "sessions": sessions.json(), "pending": actions.json(),
    }))
    stable = await app.session_store.load_events("scan")
    assert await app.reconcile_orphaned_sessions() == ()
    assert await app.session_store.load_events("scan") == stable
"""


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("environment_name", ["none", "registered"])
@pytest.mark.parametrize("dispatch", ["model", "tool"])
@pytest.mark.parametrize("topology", ["direct", "requeued_handler"])
def test_worker_restart_recovers_retained_environment_and_handler_sessions(
    tmp_path, database_url, termination, environment_name, dispatch, topology
) -> None:
    _assert_worker_restart_recovers(
        tmp_path, database_url, termination, environment_name, dispatch, topology
    )


def test_multiple_replacement_processes_recover_an_inherited_model_run(tmp_path, database_url):
    # The replacement uses ordinary orphan reconciliation, not the explicitly
    # single-process abandoned-task restart mode. Process zero owns the sweep;
    # both entrypoints observe its durable result through the shared database.
    _assert_worker_restart_recovers(
        tmp_path,
        database_url,
        signal.SIGTERM,
        "none",
        "model",
        "requeued_handler",
        replacement_processes=2,
    )


def _assert_worker_restart_recovers(
    tmp_path,
    database_url,
    termination,
    environment_name,
    dispatch,
    topology,
    *,
    replacement_processes=1,
) -> None:
    """Startup settles the lost invocation, not its independently restarted queue job.

    The model and tool stand-ins expose a real dispatch barrier. The real CLI is
    signalled while awaiting them, then restarted over the same native database.
    HTTP projection is in-process here; this does not claim Cloud deployment or
    remote provider abort coverage.
    """

    from cayu.tasks import TaskStatus
    from cayu.tasks.restart import restart_abandoned_task

    (tmp_path / "project_worker.py").write_text(_RETAINED_PROJECT)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "project_worker:build_app"\n'
        '[tool.cayu.workers]\nscan = "project_worker:scan"\nrepair = "project_worker:repair"\n'
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        "CAYU_SIGNAL_TEST_DATABASE_URL": database_url,
        "CAYU_SIGNAL_TEST_STORE": str(tmp_path / "sessions.db"),
        "CAYU_SIGNAL_TEST_ENVIRONMENT": environment_name,
        "CAYU_SIGNAL_TEST_DISPATCH": dispatch,
        "CAYU_SIGNAL_TEST_TOPOLOGY": topology,
    }
    command = [sys.executable, "-m", "cayu", "worker"]
    worker = subprocess.Popen(
        [*command, "scan", "--processes", "1", "--shutdown-grace-seconds", "0.1"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "dispatched").exists():
            if worker.poll() is not None or time.monotonic() >= deadline:
                worker.kill()
                raise AssertionError(worker.communicate(timeout=5))
            time.sleep(0.02)
        worker.send_signal(termination)
        stdout, stderr = worker.communicate(timeout=10)
        assert worker.returncode == (124 if termination == signal.SIGTERM else -signal.SIGKILL), (
            stdout,
            stderr,
        )

        async def inspect_crash_and_requeue():
            stores = open_application_stores(
                database_url, sqlite_path=tmp_path / "sessions.db", event_watchers=False
            )
            try:
                session = await stores.session_store.load("scan")
                assert session.status is SessionStatus.RUNNING
                assert (
                    await stores.session_store.inspect_session_execution("scan")
                ).state == "owner_lost"
                if topology == "direct":
                    return session, None
                old = await stores.task_store.load_task("job")
                assert old.session_id is None
                assert session.metadata["cayu:task_id"] == "job"
                now = await stores.task_store.lease_clock()
                await asyncio.sleep(max(0, (old.lease_expires_at - now).total_seconds()) + 0.05)
                await stores.task_store.reclaim_expired()
                restarted = await restart_abandoned_task(
                    stores.task_store,
                    "job",
                    validator_id="tests.real-cli-process-exit",
                    validator_version="1",
                    evidence_id=f"worker-exited:{worker.pid}:{worker.returncode}",
                )
                assert restarted.status is TaskStatus.PENDING
                successor = await stores.task_store.claim_task("replacement-job-owner")
                assert successor.claim_generation > old.claim_generation
                return session, successor
            finally:
                await stores.close()

        before, successor = asyncio.run(inspect_crash_and_requeue())
        replacement = subprocess.run(
            [*command, "repair", "--processes", str(replacement_processes)],
            cwd=tmp_path,
            env={**environment, "CAYU_SIGNAL_TEST_REPLACEMENT": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert replacement.returncode == 0, (replacement.stdout, replacement.stderr)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=5)

    assert not (tmp_path / "redispatched").exists()
    projection = json.loads((tmp_path / "recovered.json").read_text())
    [projected] = projection["sessions"]["sessions"]
    [profile] = projection["sessions"]["execution_profiles"]
    assert projected["status"] == "interrupted"
    assert profile["boundary"] == ("resume" if dispatch == "model" else "continuation")
    pending = projection["pending"]
    assert pending["issues"] == []
    assert pending["has_more"] is False and pending["next_cursor"] is None
    if dispatch == "model":
        assert pending["actions"] == []
    else:
        [action] = pending["actions"]
        assert action["kind"] == "manual_recovery"
        assert action["session"]["id"] == "scan"

    async def inspect_recovered():
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "sessions.db", event_watchers=False
        )
        try:
            session = await stores.session_store.load("scan")
            assert session.instance_id == before.instance_id
            assert session.run_epoch > before.run_epoch
            assert session.status is SessionStatus.INTERRUPTED
            events = await stores.session_store.load_events("scan")
            interrupted = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
            assert len(interrupted) == 1
            assert interrupted[0].payload["reason"] == "orphaned_session_reconciled"
            if dispatch == "tool":
                assert any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in events)
            if successor is not None:
                assert await stores.task_store.load_task("job") == successor
        finally:
            await stores.close()

    asyncio.run(inspect_recovered())
