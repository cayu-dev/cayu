"""Sessions a lost worker left ``running`` are reconciled, and can be closed.

A Cayu Cloud Agent registers a per-run environment only in the worker process
that runs the session. When a deploy replaces that worker mid-run, the session
stays ``running``: no later process registers its environment, so ordinary
recovery can't take it over, and nothing else ends it. Once its owner is
provably gone (its execution lease lapsed or its process died, or its task
finished), the runtime reconciles it and ``cayu session close`` closes it,
without the environment. A live owner, an unfinished task and recent activity
still keep it.

A run that stopped inside an assistant model call is ended the same way: the
model call is abandoned with the run, with an unknown outcome when it had
reached its provider, and its budget reservations are charged at their reserved
amount. Those tests run on SQLite and PostgreSQL; an in-memory store can't
outlive the worker that crashes.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.runtime.runtime_budget_support import fake_budget_limit
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
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.server import ServerConfig, ServerLifecycleConfig, create_server
from cayu.sessions import IncompleteSessionRecoveryAction
from cayu.sessions._model_completion_publication import ModelCompletionStageDisposition
from cayu.sessions.records import SessionIdentity, SessionStatusConflict
from cayu.storage import SQLiteSessionStore, open_application_stores
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


class _BlockingProvider(VersionedFakeProvider):
    """A live worker whose model call is still waiting on the provider."""

    started: asyncio.Event
    release: asyncio.Event

    async def stream(self, request):
        self.started.set()
        await self.release.wait()
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _budget() -> BudgetPolicy:
    # Each model call reserves its budget before it is dispatched.
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("10"),
                pricing=fake_budget_limit("10").pricing,
                reservation=BudgetReservation(max_input_tokens=1_000_000, max_output_tokens=0),
            ),
        )
    )


def _application(
    store,
    *,
    environment=None,
    tool: Tool | None = None,
    task_store=None,
    provider=None,
    budget_policy: BudgetPolicy | None = None,
):
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        budget_policy=budget_policy,
        enable_logging=False,
    )
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
        url = os.environ.get("CAYU_ORPHAN_TEST_DATABASE_URL")
        if url is None:
            store = SQLiteSessionStore(os.environ["CAYU_ORPHAN_TEST_STORE"])
        else:
            store = _stores(url, os.environ["CAYU_ORPHAN_TEST_STORE"]).session_store
        where = os.environ["CAYU_ORPHAN_TEST_CRASH"]
        if where == "prepared":
            # The worker died after it prepared the model call, before dispatching it.
            async def die(*args, **kwargs):
                os._exit(137)

            type(store).mark_model_completion_stage_dispatched = die
        app = _application(
            store,
            environment=_slot(_CrashingSlot if where == "setup" else _Slot),
            provider=_CrashingProvider if where == "model" else None,
            budget_policy=_budget() if url is not None else None,
        )
        async for _ in app.run(_request(os.environ["CAYU_ORPHAN_TEST_SESSION"])):
            pass
        raise AssertionError("The worker did not die inside the session.")

    asyncio.run(scenario())


def _run_abandoner() -> None:
    """A sweep without the environment that dies before it releases its recovery claim."""

    import datetime

    import cayu.runtime._recovery_ownership as recovery_ownership

    async def die(*args, **kwargs):
        os._exit(137)

    # The claim lapses soon after the process dies, as a real one does after minutes.
    recovery_ownership._INCOMPLETE_RECOVERY_CLAIM_LEASE = datetime.timedelta(seconds=1)
    recovery_ownership.RecoveryOwnership.cleanup_claim = die

    async def scenario() -> None:
        stores = _stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"], os.environ["CAYU_ORPHAN_TEST_STORE"]
        )
        app = _application(stores.session_store, budget_policy=_budget())
        await app.reconcile_orphaned_sessions(inactive_for_seconds=0)
        raise AssertionError("The sweep did not die before releasing its claim.")

    asyncio.run(scenario())


def _orphan(
    path,
    session_id: str,
    *,
    crash: str = "tool",
    database_url: str | None = None,
    entry: str = "_run_worker",
    module: str = __name__,
) -> None:
    database = {} if database_url is None else {"CAYU_ORPHAN_TEST_DATABASE_URL": database_url}
    child = subprocess.run(
        [sys.executable, "-c", f"from {module} import {entry}; {entry}()"],
        env={
            **os.environ,
            # The worker imports what this process does.
            "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
            "CAYU_ORPHAN_TEST_STORE": str(path),
            "CAYU_ORPHAN_TEST_SESSION": session_id,
            "CAYU_ORPHAN_TEST_CRASH": crash,
            **database,
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
            # The sweep fails the session, then publishes session.failed.
            if (await store.load("scan")).status is SessionStatus.FAILED and any(
                event.type is EventType.SESSION_FAILED for event in await store.load_events("scan")
            ):
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


def _stores(database_url: str, sqlite_path):
    return open_application_stores(
        database_url, sqlite_path=sqlite_path, tasks=False, event_watchers=False
    )


@pytest.fixture(params=["sqlite", "postgres"])
def database_url(request, tmp_path):
    """A database of its own for the test: a SQLite file, or a PostgreSQL database."""

    if request.param == "sqlite":
        yield f"sqlite:///{tmp_path / 'cayu.db'}"
        return
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    server = request.getfixturevalue("_postgres_server_dsn")
    database = f"cayu_test_{uuid4().hex}"
    with psycopg.connect(server, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    dsn = make_conninfo(server, dbname=database)

    async def create_schema() -> None:
        creator = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

    try:
        asyncio.run(create_schema())
        params = conninfo_to_dict(dsn)
        credentials = str(params.get("user", ""))
        if params.get("password"):
            credentials += f":{params['password']}"
        host, port = params.get("host", "localhost"), params.get("port", 5432)
        yield f"postgresql://{credentials}@{host}:{port}/{database}"
    finally:
        with psycopg.connect(server, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


def _model_orphan(tmp_path, database_url: str) -> None:
    """A worker with a budget died while its model call waited on the provider."""

    _orphan(tmp_path / "cayu.db", "scan", crash="model", database_url=database_url)


async def _lost_model_call(store):
    """The worker's model call, still in flight at its provider."""

    active = await store.load_active_model_completion_stage("scan")
    assert active is not None and active.stage.state == "in_flight"
    assert active.stage.purpose == "assistant-turn"
    assert await store.load_model_completion_stage_dispatch("scan", active.stage.stage_id)
    # It reserved budget before it was dispatched.
    assert active.stage.reservation_ids
    return active.stage


async def _require_abandoned_model_call(stores, stage, payload: dict) -> None:
    """The model call ended with the run: unknown outcome, settled, its budget charged."""

    store = stores.session_store
    [unknown] = payload["model_call_outcome_unknown"]
    assert unknown["stage_id"] == stage.stage_id
    assert unknown["model_step_id"] == stage.logical_step_id
    assert unknown["provider_name"] == "fake"
    assert payload.get("tool_effect_outcome_unknown", []) == []
    assert await store.load_active_model_completion_stage("scan") is None
    settlement = await store.load_model_completion_stage_settlement("scan", stage.stage_id)
    assert settlement.disposition is ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
    # Nothing knows what the provider charged, so each reservation is charged in full.
    for reservation_id in stage.reservation_ids:
        reservation = await stores.budget_ledger.load_reservation(reservation_id)
        assert reservation.status == "reconciled"
    reconciled = [
        event
        for event in await store.load_events("scan")
        if event.type is EventType.BUDGET_RECONCILED
    ]
    assert len(reconciled) == len(stage.reservation_ids)
    assert await _latest_interaction_event(store, "scan") is EventType.INTERACTION_FAILED
    interaction_failures = [
        event
        for event in await store.load_events("scan")
        if event.type is EventType.INTERACTION_FAILED
    ]
    assert len(interaction_failures) == 1
    checkpoint = await store.load_checkpoint("scan")
    assert "incomplete_session_recovery_claim" not in (checkpoint or {})


def test_reconciliation_ends_a_run_its_lost_worker_left_inside_a_model_call(
    tmp_path, database_url
) -> None:
    _model_orphan(tmp_path, database_url)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        reader = _application(store, budget_policy=_budget())
        stage = await _lost_model_call(store)
        assert (await store.inspect_session_execution("scan")).state == "owner_lost"
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=3600) == ()

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.status is SessionStatus.FAILED, result.message
        assert result.actions == (IncompleteSessionRecoveryAction.ABANDONED_ORPHANED_RUN,)
        assert "model call it was making has an unknown outcome" in result.message
        payload = await _failure(store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"
        await _require_abandoned_model_call(stores, stage, payload)
        assert (await store.inspect_session_execution("scan")).state == "terminal"
        # A second pass finds nothing left to do.
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


def test_a_model_call_the_lost_run_never_dispatched_is_settled_without_an_effect(
    tmp_path, database_url
) -> None:
    _orphan(tmp_path / "cayu.db", "scan", crash="prepared", database_url=database_url)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        reader = _application(store, budget_policy=_budget())
        active = await store.load_active_model_completion_stage("scan")
        stage = active.stage
        assert await store.load_model_completion_stage_dispatch("scan", stage.stage_id) is None
        assert stage.reservation_ids

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.status is SessionStatus.FAILED, result.message
        assert result.actions == (IncompleteSessionRecoveryAction.ABANDONED_ORPHANED_RUN,)
        assert "unknown outcome" not in result.message
        payload = await _failure(store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"
        # It never reached the provider, so nothing about it is unknown.
        assert "model_call_outcome_unknown" not in payload
        assert await store.load_active_model_completion_stage("scan") is None
        settlement = await store.load_model_completion_stage_settlement("scan", stage.stage_id)
        assert (
            settlement.disposition is ModelCompletionStageDisposition.FAILED_BEFORE_PROVIDER_EFFECT
        )
        for reservation_id in stage.reservation_ids:
            reservation = await stores.budget_ledger.load_reservation(reservation_id)
            assert reservation.status == "released"
        assert await _latest_interaction_event(store, "scan") is EventType.INTERACTION_FAILED
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


def test_close_ends_a_session_its_lost_worker_left_inside_a_model_call(
    tmp_path, database_url
) -> None:
    _model_orphan(tmp_path, database_url)
    stores = _stores(database_url, tmp_path / "cayu.db")
    store = stores.session_store
    reader = _application(store, budget_policy=_budget())
    with TestClient(create_server(reader, config=_config())) as client:
        stage = client.portal.call(_lost_model_call, store)
        closed = _close(client, "scan")
        assert closed.status_code == 200, closed.text
        assert closed.json()["session"]["status"] == "failed"
        assert closed.json()["actions"] == ["closed"]
        payload = client.portal.call(_failure, store, "scan")
        assert payload["reason"] == "session_closed"
        assert payload["metadata"]["reason"] == "Worker lost."
        client.portal.call(_require_abandoned_model_call, stores, stage, payload)
        # A repeated close replays and settles nothing twice.
        again = _close(client, "scan")
        assert again.status_code == 200, again.text
        assert again.json()["session"]["status"] == "failed"
        client.portal.call(_require_abandoned_model_call, stores, stage, payload)
        client.portal.call(stores.close)


def test_a_live_owner_inside_a_model_call_is_left_alone(tmp_path, database_url) -> None:
    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store

        class Provider(_BlockingProvider):
            started = asyncio.Event()
            release = asyncio.Event()

        writer = _application(store, environment=_slot(), provider=Provider)

        async def run() -> None:
            async for _ in writer.run(_request("scan")):
                pass

        running = asyncio.create_task(run())
        try:
            await asyncio.wait_for(Provider.started.wait(), 10)
            reader = _application(store)
            assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
            with pytest.raises(SessionStatusConflict, match="nothing shows its owner is gone"):
                await reader.close_session("scan", reason="Worker lost.")
            # Even past that check, the store refuses to fence a live owner's run.
            with pytest.raises(SessionStatusConflict, match="owner is live"):
                await reader._incomplete_recovery.abandon_orphaned_without_environment(
                    await store.load("scan"),
                    reason="orphaned_session_reconciled",
                    metadata={},
                    inactive_for_seconds=0,
                )
            assert (await store.load("scan")).status is SessionStatus.RUNNING
            assert await store.load_active_model_completion_stage("scan") is not None
            await reader.aclose()
        finally:
            Provider.release.set()
            await asyncio.wait_for(running, 10)
        # Its own run ended it; nothing here did.
        assert "abandoned_session" not in (await store.load_checkpoint("scan") or {})
        assert (await store.load("scan")).status is SessionStatus.COMPLETED
        await writer.aclose()
        await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("moved_on", ["started", "dispatched"])
def test_model_work_the_lost_run_moved_on_to_before_the_fence_is_left_alone(
    tmp_path, database_url, monkeypatch, moved_on
) -> None:
    _model_orphan(tmp_path, database_url)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        reader = _application(store, budget_policy=_budget())
        stage = await _lost_model_call(store)
        recovery = reader._incomplete_recovery
        check = recovery._orphaned_model_call
        checks = 0

        async def moved_on_after_the_first_check(session, *, unavailable):
            nonlocal checks
            checks += 1
            current = await check(session, unavailable=unavailable)
            if checks > 1:
                return current
            # What the first check saw: the run hadn't started its model call yet,
            # or hadn't dispatched it; it did both before the fence.
            return None if moved_on == "started" else replace(current, dispatched=False)

        monkeypatch.setattr(recovery, "_orphaned_model_call", moved_on_after_the_first_check)
        with pytest.raises(SessionStatusConflict, match="moved on to other model work"):
            await reader.close_session("scan", reason="Worker lost.")
        assert checks == 2
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        checkpoint = await store.load_checkpoint("scan") or {}
        assert "abandoned_session" not in checkpoint
        assert "incomplete_session_recovery_claim" not in checkpoint
        assert await store.load_model_completion_stage_settlement("scan", stage.stage_id) is None
        for reservation_id in stage.reservation_ids:
            reservation = await stores.budget_ledger.load_reservation(reservation_id)
            assert reservation.status == "active"
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("lost", ["budget", "before_settlement", "after_settlement"])
def test_an_abandoned_model_call_settles_once_across_a_lost_acknowledgement(
    tmp_path, database_url, monkeypatch, lost
) -> None:
    _model_orphan(tmp_path, database_url)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        reader = _application(store, budget_policy=_budget())
        stage = await _lost_model_call(store)
        if lost == "budget":
            owner, name = (
                reader._incomplete_recovery._model_completion_recovery,
                "settle_abandoned_dispatch_budget",
            )
        else:
            owner, name = reader._session_recovery, "settle_released_open_interaction"
        original = getattr(owner, name)
        calls = 0

        async def lost_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1 and lost == "before_settlement":
                # The process stopped after the session failed, before the call settled.
                raise RuntimeError("stopped before the model call settled")
            result = await original(*args, **kwargs)
            if calls == 1:
                # The store committed; the acknowledgement was lost.
                raise RuntimeError("acknowledgement lost")
            return result

        monkeypatch.setattr(owner, name, lost_once)
        first = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert first and first[0].actions == (IncompleteSessionRecoveryAction.FAILED,)
        if lost == "budget":
            # The budget settled, the claim was released and the session still runs.
            assert (await store.load("scan")).status is SessionStatus.RUNNING
            assert "incomplete_session_recovery_claim" not in (
                await store.load_checkpoint("scan") or {}
            )
            [retried] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
            assert retried.status is SessionStatus.FAILED, retried.message
        elif lost == "before_settlement":
            # The same sweep finished the abandonment.
            [_, finished] = first
            assert finished.session_id == "scan"
        payload = await _failure(store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"
        await _require_abandoned_model_call(stores, stage, payload)
        # Neither a later sweep nor a close settles anything again.
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        closed = await reader.close_session("scan", reason="Worker lost.")
        assert closed.status is SessionStatus.FAILED
        await _require_abandoned_model_call(stores, stage, payload)
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


def test_an_application_with_the_environment_finishes_an_abandoned_model_call(
    tmp_path, monkeypatch
) -> None:
    database_url = f"sqlite:///{tmp_path / 'cayu.db'}"
    _model_orphan(tmp_path, database_url)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        stage = await _lost_model_call(store)
        without = _application(store, budget_policy=_budget())

        async def stopped(session_id):
            raise RuntimeError("stopped before the model call settled")

        # The application without the environment failed the run and stopped there.
        monkeypatch.setattr(without._session_recovery, "settle_released_open_interaction", stopped)
        await without.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert (await store.load("scan")).status is SessionStatus.FAILED
        assert await store.load_active_model_completion_stage("scan") is not None
        await without.aclose()

        # One that registers the environment finishes it from the abandonment record.
        with_environment = _application(store, environment=_slot(), budget_policy=_budget())
        [finished] = await with_environment.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert IncompleteSessionRecoveryAction.FAILED not in finished.actions, finished.message
        await _require_abandoned_model_call(stores, stage, await _failure(store, "scan"))
        assert await with_environment.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await with_environment.aclose()
        await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("environment_first", [True, False])
def test_an_abandonment_that_died_before_releasing_its_claim_finishes(
    tmp_path, database_url, environment_first
) -> None:
    _model_orphan(tmp_path, database_url)
    stage_holder = {}

    async def before() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        stage_holder["stage"] = await _lost_model_call(stores.session_store)
        await stores.close()

    asyncio.run(before())
    # It failed the session and published session.failed, then died holding its claim.
    _orphan(tmp_path / "cayu.db", "scan", database_url=database_url, entry="_run_abandoner")

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        assert (await store.load("scan")).status is SessionStatus.FAILED
        assert await store.load_active_model_completion_stage("scan") is not None
        await asyncio.sleep(1.5)
        orders = [_application(store, environment=_slot(), budget_policy=_budget())]
        orders.append(_application(store, budget_policy=_budget()))
        if not environment_first:
            orders.reverse()
        first, second = orders
        await first.reconcile_orphaned_sessions(inactive_for_seconds=0)
        await _require_abandoned_model_call(
            stores, stage_holder["stage"], await _failure(store, "scan")
        )
        assert await first.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        assert await second.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        closed = await second.close_session("scan", reason="Worker lost.")
        assert closed.status is SessionStatus.FAILED
        await _require_abandoned_model_call(
            stores, stage_holder["stage"], await _failure(store, "scan")
        )
        await first.aclose()
        await second.aclose()
        await stores.close()

    asyncio.run(scenario())


def test_the_server_reconciles_as_soon_as_it_starts(tmp_path) -> None:
    # A Cloud Agent's web process sleeps when nobody uses it and starts again on the
    # next request, such as the next deploy's session check: it reconciles then, not
    # an interval later.
    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash="setup")
    store = SQLiteSessionStore(path)
    reader = _application(store)
    config = ServerConfig.protected(
        _authenticate,
        lifecycle=ServerLifecycleConfig(
            orphaned_session_reconcile_interval_seconds=3600,
            orphaned_session_inactive_after_seconds=0,
        ),
    )

    with TestClient(create_server(reader, config=config)) as client:
        client.portal.call(_wait_until_failed, store, "scan")
        payload = client.portal.call(_failure, store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"


async def _wait_until_failed(store, session_id: str) -> None:
    for _attempt in range(200):
        # The sweep fails the session, then publishes session.failed.
        if (await store.load(session_id)).status is SessionStatus.FAILED and any(
            event.type is EventType.SESSION_FAILED for event in await store.load_events(session_id)
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("The orphaned session was not reconciled.")


def test_the_reconciliation_loop_runs_at_once_and_until_stopped(tmp_path) -> None:
    from cayu.runtime.orphaned_sessions import run_orphaned_session_reconciliation

    path = tmp_path / "sessions.sqlite"
    _orphan(path, "scan", crash="setup")

    async def scenario() -> None:
        store = SQLiteSessionStore(path)
        reader = _application(store)
        stop = asyncio.Event()
        loop = asyncio.create_task(
            run_orphaned_session_reconciliation(
                reader, stop, interval_seconds=3600, inactive_for_seconds=0
            )
        )
        await _wait_until_failed(store, "scan")
        assert (await _failure(store, "scan"))["reason"] == "orphaned_session_reconciled"
        stop.set()
        await asyncio.wait_for(loop, 5)
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("interval", "inactive", "error"),
    [(0, 0, ValueError), (float("inf"), 0, ValueError), (True, 0, TypeError), (1, -1, ValueError)],
)
def test_the_reconciliation_loop_refuses_bad_timing(interval, inactive, error) -> None:
    from cayu.runtime.orphaned_sessions import run_orphaned_session_reconciliation

    app = CayuApp(enable_logging=False)
    with pytest.raises(error):
        asyncio.run(
            run_orphaned_session_reconciliation(
                app, asyncio.Event(), interval_seconds=interval, inactive_for_seconds=inactive
            )
        )
