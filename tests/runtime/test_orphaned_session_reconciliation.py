"""Sessions a lost worker left ``running`` are reconciled, and can be closed.

A Cayu Cloud Agent registers a per-run environment only in the worker process
that runs the session. When a deploy replaces that worker mid-run, the session
stays ``running``: no later process registers its environment, so ordinary
recovery can't take it over, and nothing else ends it. Once its owner is
provably gone (its execution lease lapsed or its process died, or its task
finished), the runtime reconciles it and ``cayu session close`` closes it,
without the environment. A live owner, an unfinished task and recent activity
still keep it.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.runtime.runtime_environment_support import RecordingEnvironmentFactory
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.server.test_server_inbox import _as, _authenticate, _items

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolEffect,
    ToolSpec,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.server import ServerConfig, ServerLifecycleConfig, create_server
from cayu.sessions import IncompleteSessionRecoveryAction
from cayu.sessions.records import SessionIdentity, SessionStatusConflict
from cayu.storage import SQLiteSessionStore
from cayu.tasks import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore

_ENVIRONMENT = "run-1-slot-1"


# Identities that hold across processes, so a process with the environment may
# continue the session under the profile the worker started it with.
_TOOL = ToolSpec(
    name="record",
    effect=ToolEffect.EXTERNAL,
    execution_profile_identity=ExecutionProfileBehaviorIdentity(
        name="tests:orphan-record", behavior_version="1", implementation_version="1"
    ),
)


class _Crashing(Tool):
    """An external call the worker died inside: its outcome is unknown."""

    spec = _TOOL

    async def run(self, ctx, args):
        os._exit(137)


class _Blocking(Tool):
    spec = _TOOL

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, ctx, args):
        self.started.set()
        await self.release.wait()
        return "recorded"


class _Slot(RecordingEnvironmentFactory):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:orphan-slot", behavior_version="1", implementation_version="1"
        )


class _CrashingSlot(_Slot):
    """The worker died while it set up the session's sandbox."""

    async def create(self, request):
        os._exit(137)


class _CrashingProvider(VersionedFakeProvider):
    """The worker died while the model answered."""

    async def stream(self, request):
        os._exit(137)
        yield  # pragma: no cover


def _application(
    store, *, environment=None, tool: Tool | None = None, task_store=None, provider=None
):
    app = CayuApp(session_store=store, task_store=task_store, enable_logging=False)
    app.register_provider(
        (provider or VersionedFakeProvider)(
            [
                ModelStreamEvent.tool_call(id="call-1", name="record", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="scanner", model="fake-model"), tools=[tool or _Crashing()])
    if environment is not None:
        app.register_environment_factory(environment.environment.spec, environment)
    return app


def _slot(factory=_Slot):
    identity = ExecutionProfileBehaviorIdentity(
        name="tests:orphan-environment", behavior_version="1", implementation_version="1"
    )
    return factory(
        Environment(EnvironmentSpec(name=_ENVIRONMENT, execution_profile_identity=identity))
    )


def _request(session_id: str) -> RunRequest:
    return RunRequest(
        session_id=session_id,
        agent_name="scanner",
        environment_name=_ENVIRONMENT,
        messages=[Message.text("user", "go")],
    )


def _run_worker() -> None:
    """The scan's worker: it registers the run's environment and dies mid-session."""

    async def scenario() -> None:
        store = SQLiteSessionStore(os.environ["CAYU_ORPHAN_TEST_STORE"])
        where = os.environ["CAYU_ORPHAN_TEST_CRASH"]
        app = _application(
            store,
            environment=_slot(_CrashingSlot if where == "setup" else _Slot),
            provider=_CrashingProvider if where == "model" else None,
        )
        async for _ in app.run(_request(os.environ["CAYU_ORPHAN_TEST_SESSION"])):
            pass
        raise AssertionError("The worker did not die inside the session.")

    asyncio.run(scenario())


def _orphan(path, session_id: str, *, crash: str = "tool") -> None:
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.runtime.test_orphaned_session_reconciliation import _run_worker; "
            "_run_worker()",
        ],
        env={
            **os.environ,
            # The worker imports what this process does.
            "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
            "CAYU_ORPHAN_TEST_STORE": str(path),
            "CAYU_ORPHAN_TEST_SESSION": session_id,
            "CAYU_ORPHAN_TEST_CRASH": crash,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 137, (child.stdout, child.stderr)


async def _latest_interaction_event(store, session_id: str) -> EventType:
    [latest] = [
        event
        for event in await store.load_events(session_id)
        if event.type.value.startswith("interaction.")
    ][-1:]
    return latest.type


async def _failure(store, session_id: str) -> dict:
    session = await store.load(session_id)
    assert session.status is SessionStatus.FAILED
    [failed] = [
        event
        for event in await store.load_events(session_id)
        if event.type is EventType.SESSION_FAILED
    ]
    assert failed.payload["error_type"] == "SessionAbandoned"
    return failed.payload


@pytest.mark.parametrize("crash", ["tool", "setup"])
def test_reconciliation_ends_a_run_its_lost_worker_left_where_the_environment_isnt(
    tmp_path, crash
) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash=crash)

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        # A later Release's web process: the run's environment isn't registered here.
        reader = _application(store, task_store=InMemoryTaskStore())
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        assert (await store.inspect_session_execution("scan")).state == "owner_lost"
        # Its activity is too recent for an hour's window.
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=3600) == ()
        assert (await store.load("scan")).status is SessionStatus.RUNNING

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.session_id == "scan"
        assert result.status is SessionStatus.FAILED
        payload = await _failure(store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"
        unknown = payload.get("tool_effect_outcome_unknown", [])
        if crash == "tool":
            # The worker died inside the external call: its outcome is unknown, and
            # it is never dispatched again.
            assert result.actions == (IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT,)
            assert [call["tool_call_id"] for call in unknown] == ["call-1"]
        else:
            assert result.actions == (IncompleteSessionRecoveryAction.ABANDONED_ORPHANED_RUN,)
            assert unknown == []
        # The run's interaction ended with it.
        assert await _latest_interaction_event(store, "scan") is EventType.INTERACTION_FAILED
        # The run is fenced and nothing holds the session any longer.
        execution = await store.inspect_session_execution("scan")
        assert execution.state == "terminal"
        checkpoint = await store.load_checkpoint("scan")
        assert "incomplete_session_recovery_claim" not in (checkpoint or {})
        # A second pass finds nothing left to do.
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_reconciliation_leaves_a_recovery_item_where_the_environment_is_registered(
    tmp_path,
) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan")

    store = SQLiteSessionStore(path)
    reader = _application(store, environment=_slot())
    with TestClient(create_server(reader, config=_config())) as client:
        [result] = client.portal.call(
            lambda: reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        )
        assert IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT in result.actions
        assert client.portal.call(store.load, "scan").status is SessionStatus.INTERRUPTED
        # A person decides about the call the worker died inside.
        [item] = _items(client, "priya")
        assert (item["kind"], item["session_id"]) == ("recovery", "scan")


def _config() -> ServerConfig:
    # The test drives reconciliation itself.
    return ServerConfig.protected(
        _authenticate,
        lifecycle=ServerLifecycleConfig(orphaned_session_reconcile_interval_seconds=None),
    )


def _close(client, session_id: str, reason: str = "Worker lost."):
    return client.post(
        f"/api/sessions/{session_id}/close", json={"reason": reason}, headers=_as("priya")
    )


@pytest.mark.parametrize("crash", ["tool", "setup"])
def test_close_ends_a_session_its_lost_worker_left_running(tmp_path, crash) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash=crash)
    store = SQLiteSessionStore(path)
    reader = _application(store)
    # `cayu session close --application` calls this route on the Agent's web process.
    with TestClient(create_server(reader, config=_config())) as client:
        closed = _close(client, "scan")
        assert closed.status_code == 200, closed.text
        body = closed.json()
        assert body["session"]["status"] == "failed"
        assert body["actions"] == ["closed"]
        payload = client.portal.call(_failure, store, "scan")
        assert payload["reason"] == "session_closed"
        assert payload["closed"] is True
        assert payload["metadata"]["reason"] == "Worker lost."
        if crash == "tool":
            [unknown] = payload["tool_effect_outcome_unknown"]
            assert unknown["tool_call_id"] == "call-1"
        latest = client.portal.call(_latest_interaction_event, store, "scan")
        assert latest is EventType.INTERACTION_FAILED
        again = _close(client, "scan")
        assert again.status_code == 200, again.text
        assert again.json()["session"]["status"] == "failed"


def test_close_and_reconciliation_leave_a_live_owner_alone(tmp_path) -> None:
    async def scenario() -> None:
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        blocking = _Blocking()
        writer = _application(store, environment=_slot(), tool=blocking)

        async def run() -> None:
            async for _ in writer.run(_request("live")):
                pass

        running = asyncio.create_task(run())
        try:
            await asyncio.wait_for(blocking.started.wait(), 10)
            reader = _application(store)
            assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
            with pytest.raises(SessionStatusConflict, match="nothing shows its owner is gone"):
                await reader.close_session("live", reason="Worker lost.")
            assert (await store.load("live")).status is SessionStatus.RUNNING
            await reader.aclose()
        finally:
            blocking.release.set()
            await asyncio.wait_for(running, 10)
        # Its own run ended it; nothing here did.
        checkpoint = await store.load_checkpoint("live")
        assert "abandoned_session" not in (checkpoint or {})
        assert (await store.load("live")).status is not SessionStatus.FAILED
        await writer.aclose()
        await store.close()

    asyncio.run(scenario())


def test_close_and_reconciliation_leave_a_session_whose_task_runs(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan")

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        tasks = InMemoryTaskStore()
        await tasks.create_task(TaskCreate(task_id="scan-job", type="scan", session_id="scan"))
        reader = _application(store, task_store=tasks)
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        with pytest.raises(SessionStatusConflict, match="task that is not finished"):
            await reader.close_session("scan", reason="Worker lost.")
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        # Once its task ends, the session can be closed.
        await tasks.cancel_task("scan-job")
        closed = await reader.close_session("scan", reason="Worker lost.")
        assert closed.status is SessionStatus.FAILED
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


async def _running_without_an_owner(store, session_id: str) -> None:
    """A run that recorded no execution owner, as one an older release started."""

    await store.create(
        RunRequest(
            agent_name="scanner", session_id=session_id, messages=[Message.text("user", "go")]
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    await store.transition_status(
        session_id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
    )


def test_a_running_session_without_an_attested_owner_needs_its_finished_task(tmp_path) -> None:
    async def scenario() -> None:
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        await _running_without_an_owner(store, "legacy")
        tasks = InMemoryTaskStore()
        reader = _application(store, task_store=tasks)
        assert (await store.inspect_session_execution("legacy")).state == "unknown"
        # Nothing proves its owner gone, however long it has been quiet.
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        with pytest.raises(SessionStatusConflict, match="nothing shows its owner is gone"):
            await reader.close_session("legacy", reason="Worker lost.")
        assert (await store.load("legacy")).status is SessionStatus.RUNNING
        # Its run's task finished, so its run did too.
        await tasks.create_task(TaskCreate(task_id="legacy-job", type="scan", session_id="legacy"))
        await tasks.cancel_task("legacy-job")
        closed = await reader.close_session("legacy", reason="Worker lost.")
        assert closed.status is SessionStatus.FAILED
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_the_server_reconciles_on_its_schedule(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash="setup")
    store = SQLiteSessionStore(path)
    reader = _application(store)
    config = ServerConfig.protected(
        _authenticate,
        lifecycle=ServerLifecycleConfig(
            orphaned_session_reconcile_interval_seconds=0.05,
            orphaned_session_inactive_after_seconds=0,
        ),
    )

    async def failed() -> None:
        for _attempt in range(400):
            if (await store.load("scan")).status is SessionStatus.FAILED:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("The server did not reconcile the orphaned session.")

    with TestClient(create_server(reader, config=config)) as client:
        client.portal.call(failed)
        payload = client.portal.call(_failure, store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"


def test_reconciliation_settles_a_lost_model_call_where_the_environment_is_registered(
    tmp_path,
) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash="model")

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        reader = _application(store, environment=_slot())
        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert IncompleteSessionRecoveryAction.FAILED not in result.actions, result.message
        assert (await store.load("scan")).status is SessionStatus.INTERRUPTED
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_without_the_environment_a_lost_model_call_is_left_to_its_own_recovery(
    tmp_path,
) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash="model")

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        reader = _application(store)
        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert result.actions == (IncompleteSessionRecoveryAction.FAILED,)
        assert "inside a model call" in result.message
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_close_without_the_environment_refuses_a_stale_pause_id(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash="setup")

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        reader = _application(store)
        with pytest.raises(SessionStatusConflict, match="no longer waits"):
            await reader.close_session("scan", reason="Old question.", waiting_on="input-1")
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_reconciliation_finishes_an_abandonment_that_stopped_before_its_evidence(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan")

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        reader = _application(store)
        finalization = reader._incomplete_recovery.terminal_finalization
        repair = finalization.repair
        calls = 0

        async def lost_once(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                # The process stops after the session failed, before session.failed.
                raise RuntimeError("terminal publication lost")
            return await repair(**kwargs)

        monkeypatch.setattr(finalization, "repair", lost_once)
        stopped, finished = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert stopped.actions == (IncompleteSessionRecoveryAction.FAILED,)
        # The same sweep finds the failed session still missing its evidence and finishes it.
        assert finished.session_id == "scan"
        assert IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE in finished.actions
        payload = await _failure(store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"
        assert await _latest_interaction_event(store, "scan") is EventType.INTERACTION_FAILED
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_model_work_the_lost_run_started_before_the_fence_is_left_alone(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash="setup")

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        reader = _application(store)
        runtime_store = reader._incomplete_recovery._session_store
        load_stage = runtime_store.load_active_model_completion_stage
        checks = 0

        async def started_after_the_first_check(session_id):
            nonlocal checks
            checks += 1
            # The lost run started a model call between the first check and the fence.
            return await load_stage(session_id) if checks == 1 else object()

        monkeypatch.setattr(
            runtime_store, "load_active_model_completion_stage", started_after_the_first_check
        )
        with pytest.raises(SessionStatusConflict, match="inside a model call"):
            await reader.close_session("scan", reason="Worker lost.")
        assert checks >= 2
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        assert "abandoned_session" not in (await store.load_checkpoint("scan") or {})
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())
