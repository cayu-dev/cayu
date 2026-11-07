"""Edges of external-event waits in task-owned sessions, on every built-in store.

Peers that don't host waits, interruptions racing a service, the ``cancel`` grant,
lost registrations, crashes, and until-idle reporting.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_external_wait_hosting import WORKER, RegistrarPolicy, _delivery, _reserve
from tests.core.test_external_waits_in_tasks import (
    _JOB,
    _answers,
    _drain,
    _job,
    _parked,
    _session_id,
    _WorkerDied,
)
from tests.core.test_sleep_until import _Clock

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.sessions.external_waits import ExternalWaitScope
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.storage import PostgresSessionStore, SQLiteSessionStore, SQLiteTaskStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    run_task_worker_until_idle,
    session_wakes,
)
from cayu.tasks.queries import TaskQuery
from cayu.tools import SleepUntilTool

BACKENDS = ["memory", "sqlite", "postgres"]


def _stores(backend, clock, tmp_path, request):
    if backend == "memory":
        return (
            InMemorySessionStore(ownership_clock=clock),
            InMemoryTaskStore(clock=clock, ownership_clock=clock),
        )
    if backend == "sqlite":
        return (
            SQLiteSessionStore(tmp_path / "sessions.sqlite", ownership_clock=clock),
            SQLiteTaskStore(tmp_path / "tasks.sqlite", clock=clock, ownership_clock=clock),
        )
    dsn = request.getfixturevalue("postgres_dsn")
    return (
        PostgresSessionStore(
            dsn, schema_mode=SchemaMode.CREATE, min_size=1, max_size=2, clock=clock
        ),
        PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=clock),
    )


def _app(sessions, tasks, clock, provider, *, sleeps=False):
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="root", model="model"), tools=[SleepUntilTool()] if sleeps else []
    )
    return app


@asynccontextmanager
async def clocked(backend, tmp_path, request, *, policy=None, dispatches=2):
    """A hosted app on one store whose task and lease times follow an injected clock.

    It also yields a peer: an app on the same stores that sleeps but hosts no waits.
    """

    clock = _Clock(datetime.now(UTC))
    sessions, tasks = _stores(backend, clock, tmp_path, request)
    provider = _answers(dispatches)
    app = _app(sessions, tasks, clock, provider)
    peer = _app(sessions, tasks, clock, provider, sleeps=True)
    scope = ExternalWaitScope(application_scope="video-" + uuid4().hex, generation=1)
    hosting = app.host_external_waits(
        access_policy=policy or RegistrarPolicy(), context=WORKER, scopes={scope: ("renderer",)}
    )
    try:
        yield clock, app, peer, hosting, scope, provider
    finally:
        await peer.aclose()
        await app.aclose()
        await hosting.aclose()
        if backend != "memory":
            await tasks.close()
            await sessions.close()


async def _park(app, hosting, scope, *, deadline=None):
    job = f"job-{uuid4().hex[:8]}"
    correlation, registration = await _reserve(hosting, scope, deadline=deadline, key=job)
    await hosting.register(registration, context=WORKER)
    await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
    handler = _job(hosting, registration)
    await _drain(app, handler)
    assert (await _parked(app, job)).kind == "external_wait"
    return job, correlation, registration, handler


async def _generic(app, *, worker_id="after-stop"):
    continued: list[str] = []

    async def generic(app, task, worker_id):
        continued.append(task.id)
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def refuse(app, task, worker_id):
        raise AssertionError(f"The application handler got {task.type}.")

    await run_task_worker_until_idle(
        app,
        app.task_store,
        refuse,
        worker_id=worker_id,
        recovered_interrupted_task_handler=generic,
    )
    return continued


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_peer_that_hosts_no_waits_never_releases_an_external_wait_park(
    backend, tmp_path, request
):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (_, app, peer, hosting, scope, _):
            job, _, _, _ = await _park(app, hosting, scope)
            assert await session_wakes.reconcile_parked_handoffs(peer) == 0
            assert (await _parked(app, job)).kind == "external_wait"
            # The peer's own worker passes leave it parked too.
            assert await _generic(peer) == []
            assert (await _parked(app, job)).kind == "external_wait"

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_peer_that_hosts_no_waits_refuses_to_interrupt_a_waiting_task(backend, tmp_path, request):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (_, app, peer, hosting, scope, _):
            job, correlation, _, _ = await _park(app, hosting, scope)
            with pytest.raises(ValueError, match="host_external_waits"):
                async for _ in peer.interrupt_session(
                    InterruptSessionRequest(session_id=_session_id(job), reason="stop")
                ):
                    pass
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome is None and wait.pending_handoff
            assert (await _parked(app, job)).kind == "external_wait"

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_peer_that_hosts_no_waits_never_reruns_a_turn_that_died_before_parking(
    backend, tmp_path, request, monkeypatch, caplog
):
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        async with clocked(backend, tmp_path, request) as (
            clock,
            app,
            peer,
            hosting,
            scope,
            provider,
        ):
            job = f"job-{uuid4().hex[:8]}"
            _, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            with monkeypatch.context() as patch:
                patch.setattr(_ExternalExecutionToWait, "park", _dies)
                await _die(app, _job(hosting, registration))
            await _expire(backend, clock)
            # A peer running the same agent, which could recover the session plainly,
            # can't bring the turn to its wait, and leaves it alone.
            same_agent = _app(app.session_store, app.task_store, clock, provider)
            try:
                assert await _generic(same_agent) == []
            finally:
                await same_agent.aclose()
            session = await app.session_store.load(_session_id(job))
            assert session.status is SessionStatus.RUNNING
            # A peer worker's passes report it only once.
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                await _expire(backend, clock)
                assert await _generic(peer) == []
                assert await _generic(peer) == []
            reported = [
                record
                for record in caplog.records
                if "hosts the waits" in record.getMessage()
                and _session_id(job) in record.getMessage()
            ]
            assert len(reported) == 1
            # A worker that hosts the waits recovers it to its wait.
            assert await _generic(app, worker_id="hosting-worker") == []
            assert (await _parked(app, job)).kind == "external_wait"
            assert len(provider.requests) == 1

    asyncio.run(scenario())


async def _dies(*args, **kwargs):
    raise _WorkerDied


# A worker that dies holds its task this long; PostgreSQL leases follow the database's
# own clock, so on it the test waits this out instead of advancing the injected clock.
# Every store's worker also times its lease renewals on its own monotonic clock, so the
# lease is long enough that a renewal acknowledged late on a loaded runner keeps it:
# a lost lease would end the worker with the loss, not with its death.
_DYING_LEASE_SECONDS = 3


async def _die(app, handler):
    with pytest.raises(_WorkerDied):
        await run_task_worker_until_idle(
            app,
            app.task_store,
            handler,
            worker_id="worker-1",
            lease_seconds=_DYING_LEASE_SECONDS,
            recovered_interrupted_task_handler=_never,
        )


async def _never(app, task, worker_id):
    raise AssertionError(f"A generic continuation claimed {task.id}.")


async def _expire(backend, clock):
    """Let the dead worker's leases run out."""
    clock.advance(3600)
    if backend == "postgres":
        await asyncio.sleep(_DYING_LEASE_SECONDS + 0.5)


async def _deliver_and_die_mid_admission(app, hosting, correlation, handler, monkeypatch):
    """Deliver the event, and let its service die after preparing the admission."""
    from cayu.runtime import _session_continuation_owner

    await hosting.deliver(_delivery(correlation), context=WORKER)

    async def dies_after_preparing(native_store, consumption, command):
        await native_store.consume_continuation(consumption)
        raise RuntimeError("the servicing worker died after preparing its admission")

    with monkeypatch.context() as patch:
        patch.setattr(_session_continuation_owner, "admit_continuation", dies_after_preparing)
        await _drain(app, handler)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_service_that_died_mid_admission_is_finished_under_a_new_claim(
    backend, tmp_path, request, monkeypatch
):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _, handler = await _park(app, hosting, scope)
            await _deliver_and_die_mid_admission(app, hosting, correlation, handler, monkeypatch)
            assert len(provider.requests) == 1
            # Still the wait's: no generic continuation may take the task meanwhile.
            assert (await _parked(app, job)).kind == "external_wait"
            clock.advance(60)
            await _drain(app, handler, worker_id="worker-2")
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


# On PostgreSQL the admission these tests interrupt rolls back with its failure, so
# its window can't be produced this way.
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_interrupting_while_the_event_is_being_delivered_is_refused_clearly(
    backend, tmp_path, request, monkeypatch
):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _, handler = await _park(app, hosting, scope)
            await _deliver_and_die_mid_admission(app, hosting, correlation, handler, monkeypatch)
            with pytest.raises(ValueError, match="already being delivered"):
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id=_session_id(job), reason="stop")
                ):
                    pass
            # The delivery still finishes, and the session can be interrupted once it runs.
            clock.advance(60)
            await _drain(app, handler, worker_id="worker-2")
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


# On PostgreSQL the admission these tests interrupt rolls back with its failure, so
# its window can't be produced this way.
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancelling_while_the_event_is_being_delivered_never_continues_it(
    backend, tmp_path, request, monkeypatch, caplog
):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _, handler = await _park(app, hosting, scope)
            await _deliver_and_die_mid_admission(app, hosting, correlation, handler, monkeypatch)
            await app.task_store.cancel_task(job)
            clock.advance(60)
            with caplog.at_level(logging.WARNING):
                await _drain(app, handler, worker_id="worker-2")
                # Later passes neither reschedule it nor repeat the warning.
                attempts = await _service_attempts(app, job)
                for _ in range(3):
                    clock.advance(3600)
                    await _drain(app, handler, worker_id="worker-3")
                    assert await hosting.sweep() == 0
            assert await _service_attempts(app, job) == attempts
            assert len(provider.requests) == 1
            session = await app.session_store.load(_session_id(job))
            assert session.status is SessionStatus.INTERRUPTED
            warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
            stuck = [m for m in warnings if "ended while its event was being delivered" in m]
            assert len(stuck) == 1 and "capacity" in stuck[0]
            assert not any("service_once" in m for m in warnings)

    asyncio.run(scenario())


async def _service_attempts(app, job):
    return sorted(
        task.id
        for task in await app.task_store.list_tasks(
            TaskQuery(type="cayu.external-wait-service.v1", limit=200)
        )
        if task.input.get("correlation_key") == job
    )


class _NoCancelPolicy(RegistrarPolicy):
    def authorize(self, context, *, scope, source, action):
        return action != "cancel" and super().authorize(
            context, scope=scope, source=source, action=action
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_task_owned_wait_needs_the_hosting_context_to_cancel(backend, tmp_path, request):
    async def scenario():
        async with clocked(backend, tmp_path, request, policy=_NoCancelPolicy()) as (
            _,
            app,
            _,
            hosting,
            scope,
            provider,
        ):
            job = f"job-{uuid4().hex[:8]}"
            _, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            failed = await _drain(app, _job(hosting, registration))
            assert failed.handled == 1
            task = await app.task_store.load_task(job)
            assert task.status is TaskStatus.FAILED
            assert "'cancel'" in task.error["message"]
            # Refused before its turn ran.
            assert provider.requests == []

    asyncio.run(scenario())


def test_a_failing_ended_task_check_still_lets_the_sweep_repair_the_wait(
    tmp_path, request, monkeypatch
):
    from cayu import external_wait_tasks

    async def scenario():
        async with clocked("memory", tmp_path, request) as (clock, app, _, hosting, scope, _):
            job = f"job-{uuid4().hex[:8]}"
            _, registration = await _reserve(
                hosting, scope, key=job, deadline=clock() + timedelta(hours=1)
            )
            create = app.task_store.create_task

            async def lost(request):
                raise ConnectionError("task store unavailable")

            # Registered, but its deadline task was never created.
            monkeypatch.setattr(app.task_store, "create_task", lost)
            with pytest.raises(ConnectionError):
                await hosting.register(registration, context=WORKER)
            monkeypatch.setattr(app.task_store, "create_task", create)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            # A worker that claims only the job parks it, and sweeps nothing.
            await run_task_worker_until_idle(
                app,
                app.task_store,
                _job(hosting, registration),
                worker_id="jobs",
                query=TaskQuery(type=_JOB),
            )
            wait = await hosting.waits.inspect(registration.correlation, context=WORKER)
            assert wait.pending_timer and wait.handoff == "pending"

            async def fails(*args, **kwargs):
                raise ConnectionError("task store unavailable")

            monkeypatch.setattr(external_wait_tasks, "stop_wait_of_ended_task", fails)
            assert await hosting.sweep() == 1
            wait = await hosting.waits.inspect(registration.correlation, context=WORKER)
            assert not wait.pending_timer

    asyncio.run(scenario())


def _forget_wait(app, monkeypatch, key):
    """The session's wait is no longer registered, as after its scope was pruned."""
    read = app.session_store._read_external_wait

    async def gone(scope, correlation_key):
        return None if correlation_key == key else await read(scope, correlation_key)

    monkeypatch.setattr(app.session_store, "_read_external_wait", gone)


def test_interrupting_a_session_whose_wait_is_gone_just_interrupts_it(
    tmp_path, request, monkeypatch
):
    async def scenario():
        async with clocked("memory", tmp_path, request) as (_, app, _, hosting, scope, _):
            job, _, _, _ = await _park(app, hosting, scope)
            _forget_wait(app, monkeypatch, job)
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id=_session_id(job), reason="stop")
            ):
                pass
            # Nothing waits on it any more: the task goes to its ordinary continuation.
            assert await _generic(app) == [job]

    asyncio.run(scenario())


def test_a_turn_whose_wait_is_gone_is_recovered_plainly(tmp_path, request, monkeypatch):
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        async with clocked("memory", tmp_path, request) as (clock, app, _, hosting, scope, _):
            job = f"job-{uuid4().hex[:8]}"
            _, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            with monkeypatch.context() as patch:
                patch.setattr(_ExternalExecutionToWait, "park", _dies)
                await _die(app, _job(hosting, registration))
            _forget_wait(app, monkeypatch, job)
            await _expire("memory", clock)
            # There's no wait left to protect: the task goes to its ordinary continuation.
            assert await _generic(app) == [job]

    asyncio.run(scenario())


def test_until_idle_reports_no_continuation_once_a_service_settled_its_session(tmp_path, request):
    async def scenario():
        async with clocked("memory", tmp_path, request, dispatches=4) as (
            _,
            app,
            _,
            hosting,
            scope,
            _,
        ):
            job, correlation, _, handler = await _park(app, hosting, scope)
            await hosting.deliver(_delivery(correlation), context=WORKER)
            done = await _drain(app, handler)
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED
            assert done.idle and not done.continuations_pending

            # A root session's wait, continued inside its service task, leaves none either.
            _, root = await _reserve(hosting, scope, key="root-" + job)
            await hosting.register(root, context=WORKER)
            await hosting.run_to_wait(
                RunRequest(
                    agent_name="root",
                    session_id="root-" + job,
                    messages=[Message.text("user", "Submit the render.")],
                ),
                root,
                context=WORKER,
            )
            await hosting.deliver(_delivery(root.correlation), context=WORKER)
            done = await _drain(app, handler)
            assert done.handled == 1 and not done.continuations_pending

    asyncio.run(scenario())


def test_until_idle_reports_a_continuation_when_a_service_left_its_task_unparked(tmp_path, request):
    from cayu.external_wait_hosting import _attempt
    from cayu.tasks.worker import _may_await_continuation

    async def scenario():
        async with clocked("memory", tmp_path, request) as (clock, app, _, hosting, scope, _):
            job, correlation, _, _ = await _park(app, hosting, scope)
            await app.task_store.create_task(_attempt(correlation, "probe", 1, clock()))
            [service] = await app.task_store.list_tasks(
                TaskQuery(type="cayu.external-wait-service.v1")
            )
            # Parked on its wait: nothing awaits a generic continuation.
            assert not await _may_await_continuation(app, app.task_store, service)
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id=_session_id(job), reason="stop")
            ):
                pass
            assert await session_wakes.reconcile_parked_handoffs(app) == 1
            # Handed back unparked: its recovered-task handler is due.
            assert await _may_await_continuation(app, app.task_store, service)

    asyncio.run(scenario())


def test_the_sweep_reads_a_root_waits_turn_only_if_a_task_could_own_it(
    tmp_path, request, monkeypatch
):
    async def scenario():
        async with clocked("memory", tmp_path, request) as (_, _, _, hosting, scope, _):
            _, root = await _reserve(hosting, scope, key="root-wait")
            await hosting.register(root, context=WORKER)
            await hosting.run_to_wait(
                RunRequest(
                    agent_name="root",
                    session_id="root-wait",
                    messages=[Message.text("user", "Submit the render.")],
                ),
                root,
                context=WORKER,
            )
            reads: list[str] = []
            turn_task = hosting.adapter._waiting_turn_task_id

            async def counted(retained, *, context):
                reads.append(retained.continuation.intent.session_id)
                return await turn_task(retained, context=context)

            monkeypatch.setattr(hosting.adapter, "_waiting_turn_task_id", counted)
            assert await hosting.sweep() == 0
            assert reads == []

    asyncio.run(scenario())


async def _start_and_die(app, hosting, scope, monkeypatch, owner, name, *, after):
    job = f"job-{uuid4().hex[:8]}"
    correlation, registration = await _reserve(hosting, scope, key=job)
    await hosting.register(registration, context=WORKER)
    await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
    original = getattr(owner, name)

    async def dies(*args, **kwargs):
        if after:
            await original(*args, **kwargs)
        raise _WorkerDied

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, dies)
        await _die(app, _job(hosting, registration))
    task = await app.task_store.load_task(job)
    assert task.status is TaskStatus.RUNNING and task.worker_id is not None
    return job, correlation, _job(hosting, registration)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_turn_that_died_after_its_answer_is_recovered_to_its_wait(
    backend, tmp_path, request, monkeypatch
):
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, handler = await _start_and_die(
                app, hosting, scope, monkeypatch, _ExternalExecutionToWait, "park", after=False
            )
            await _expire(backend, clock)
            # The answer was complete: the wait's recovery parks it, and the task with it.
            assert await _generic(app) == []
            assert (await _parked(app, job)).kind == "external_wait"
            await hosting.deliver(_delivery(correlation), context=WORKER)
            await _drain(app, handler)
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_turn_that_died_before_its_answer_is_left_to_ordinary_recovery(
    backend, tmp_path, request, monkeypatch
):
    from cayu.runtime._session_task_lifecycle import SessionTaskLifecycle

    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _ = await _start_and_die(
                app, hosting, scope, monkeypatch, SessionTaskLifecycle, "start_task", after=True
            )
            await _expire(backend, clock)
            # Nothing can bring the turn to its wait, and it never reruns blind: the wait
            # is stopped and the task goes to the ordinary continuation.
            assert await _generic(app) == [job]
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "cancelled" and not wait.pending_handoff
            assert provider.requests == []
            session = await app.session_store.load(_session_id(job))
            assert session.status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_a_turn_that_died_during_its_model_call_waits_for_manual_recovery(
    tmp_path, request, caplog
):
    async def scenario():
        async with clocked("memory", tmp_path, request) as (
            clock,
            app,
            _,
            hosting,
            scope,
            provider,
        ):

            async def dying_stream(request):
                raise _WorkerDied
                yield  # pragma: no cover - makes this an async generator

            provider.stream = dying_stream
            job = f"job-{uuid4().hex[:8]}"
            _, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            await _die(app, _job(hosting, registration))
            del provider.stream
            await _expire("memory", clock)
            with caplog.at_level(logging.WARNING):
                assert await _generic(app) == []
            # The model call is the person's to settle; nothing reruns it.
            assert "recover_model_completion_stage" in caplog.text
            assert provider.requests == []
            session = await app.session_store.load(_session_id(job))
            assert session.status is SessionStatus.RUNNING

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_cancelled_task_retires_its_wait_instead_of_continuing_it(backend, tmp_path, request):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _, handler = await _park(app, hosting, scope)
            assert (await app.task_store.cancel_task(job)).status is TaskStatus.CANCELLED
            await hosting.deliver(_delivery(correlation), context=WORKER)
            # PostgreSQL elects the event at the database's time; its service is due then.
            clock.advance(60)
            await _drain(app, handler)
            assert len(provider.requests) == 1
            session = await app.session_store.load(_session_id(job))
            assert session.status is SessionStatus.INTERRUPTED
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.retirement_requested and not wait.pending_handoff

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_deleting_a_waiting_task_owned_session_waits_for_its_task_to_be_cancelled(
    backend, tmp_path, request
):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (_, app, _, hosting, scope, provider):
            job, correlation, _, _ = await _park(app, hosting, scope)
            with pytest.raises(ValueError, match="Cancel the task first"):
                await app.delete_session(_session_id(job), finished_only=False)
            await app.task_store.cancel_task(job)
            # The worker's sweep stops the wait of a task that ended; the session can go.
            await hosting.sweep()
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "cancelled" and not wait.pending_handoff
            assert await app.delete_session(_session_id(job), finished_only=False)
            assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_handoff_parked_after_an_interrupt_retired_its_wait_is_released(
    backend, tmp_path, request, monkeypatch
):
    from cayu.tasks import worker

    async def scenario():
        async with clocked(backend, tmp_path, request) as (_, app, _, hosting, scope, _):
            job = f"job-{uuid4().hex[:8]}"
            correlation, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            parked_wait = worker._session_parked_wait

            async def interrupted_meanwhile(app, session_id):
                # The worker read the wait; an operator stops it before the handoff commits.
                parked = await parked_wait(app, session_id)
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id=session_id, reason="stop")
                ):
                    pass
                return parked

            with monkeypatch.context() as patch:
                patch.setattr(worker, "_session_parked_wait", interrupted_meanwhile)
                await _drain(app, _job(hosting, registration))
            assert (await _parked(app, job)).kind == "external_wait"
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "cancelled" and not wait.pending_handoff
            # Parked on a stopped wait: reconciliation hands it back.
            assert await session_wakes.reconcile_parked_handoffs(app) == 1
            assert await _generic(app) == [job]

    asyncio.run(scenario())


def test_a_process_that_registers_no_sleep_never_releases_a_sleeping_task():
    from tests.core.test_sleep_until_tasks import _app as _sleeper_app
    from tests.core.test_sleep_until_tasks import _parked_job

    async def scenario():
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        sleeper = _sleeper_app(sessions, tasks, [])
        job, _ = await _parked_job(sleeper, "job-sleeps")
        # Hosts external waits but registers no sleep_until: it can't follow a wake.
        hosting_only = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        hosting_only.host_external_waits(
            access_policy=RegistrarPolicy(),
            context=WORKER,
            scopes={ExternalWaitScope(application_scope="other", generation=1): ("renderer",)},
        )
        try:
            assert await session_wakes.reconcile_parked_handoffs(hosting_only) == 0
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, (await tasks.load_task(job.id)).interrupted_handoff_id
            )
            assert receipt.request.parked.kind == "session_wake"
        finally:
            await hosting_only.aclose()
            await sleeper.aclose()

    asyncio.run(scenario())


async def _recover_publicly(app, job, entrance):
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

    if entrance == "abandoned":
        return await app.recover_abandoned_sessions(inactive_for_seconds=0)
    return (
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=_session_id(job), inactive_for_seconds=0)
        ),
    )


@pytest.mark.parametrize("entrance", ["abandoned", "incomplete"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_public_recovery_leaves_a_turn_that_died_after_its_answer_to_its_wait(
    backend, entrance, tmp_path, request, monkeypatch
):
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, handler = await _start_and_die(
                app, hosting, scope, monkeypatch, _ExternalExecutionToWait, "park", after=False
            )
            await _expire(backend, clock)
            # An application that recovers abandoned sessions at startup, before its worker.
            [result] = [
                r
                for r in await _recover_publicly(app, job, entrance)
                if r.session_id == _session_id(job)
            ]
            assert result.status is SessionStatus.RUNNING
            assert await _generic(app) == []
            assert (await _parked(app, job)).kind == "external_wait"
            await hosting.deliver(_delivery(correlation), context=WORKER)
            await _drain(app, handler)
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["abandoned", "incomplete"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_public_recovery_leaves_a_turn_that_died_before_its_answer_to_its_wait(
    backend, entrance, tmp_path, request, monkeypatch
):
    from cayu.runtime._session_task_lifecycle import SessionTaskLifecycle

    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _ = await _start_and_die(
                app, hosting, scope, monkeypatch, SessionTaskLifecycle, "start_task", after=True
            )
            await _expire(backend, clock)
            [result] = [
                r
                for r in await _recover_publicly(app, job, entrance)
                if r.session_id == _session_id(job)
            ]
            assert result.status is SessionStatus.RUNNING
            assert await _generic(app) == [job]
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "cancelled" and not wait.pending_handoff
            assert provider.requests == []

    asyncio.run(scenario())


class _RevocablePolicy(RegistrarPolicy):
    revoked = False

    def authorize(self, context, *, scope, source, action):
        if action == "cancel" and self.revoked:
            return False
        return super().authorize(context, scope=scope, source=source, action=action)


def test_the_sweep_says_what_a_revoked_cancel_grant_needs(tmp_path, request, caplog):
    async def scenario():
        policy = _RevocablePolicy()
        async with clocked("memory", tmp_path, request, policy=policy) as (
            _,
            app,
            _,
            hosting,
            scope,
            _,
        ):
            job, _, _, _ = await _park(app, hosting, scope)
            policy.revoked = True
            await app.task_store.cancel_task(job)
            with caplog.at_level(logging.WARNING):
                await hosting.sweep()
            assert "grant it 'cancel'" in caplog.text

    asyncio.run(scenario())


def test_deleting_a_session_whose_wait_is_gone_is_not_refused_for_that_wait(
    tmp_path, request, monkeypatch
):
    async def scenario():
        async with clocked("memory", tmp_path, request) as (_, app, _, hosting, scope, _):
            job, _, _, _ = await _park(app, hosting, scope)
            _forget_wait(app, monkeypatch, job)
            try:
                await app.delete_session(_session_id(job), finished_only=False)
            except ValueError as error:
                assert "waits on an external event" not in str(error)

    asyncio.run(scenario())


def test_a_worker_that_registers_no_sleep_reconciles_no_wakes(tmp_path, request, monkeypatch):
    from cayu.tasks.worker import _reconcile_session_wakes

    async def scenario():
        async with clocked("memory", tmp_path, request) as (_, app, _, _, _, _):
            reconciled: list[str] = []

            async def wakes(app):
                reconciled.append("wakes")
                return 0

            monkeypatch.setattr(session_wakes, "reconcile_session_wakes", wakes)
            assert await _reconcile_session_wakes(app, app.task_store) == 0
            assert reconciled == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_continuation_longer_than_the_observation_bound_completes_its_task(
    backend, tmp_path, request, monkeypatch
):
    from cayu.collaboration import _ownership

    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _, handler = await _park(app, hosting, scope)
            # The public observation bound, shortened from 10 s; the resumed model call
            # outlasts it.
            init = _ownership._MutationOwners.__init__

            def short_bound(self):
                init(self)
                self.observation_timeout = 0.3

            monkeypatch.setattr(_ownership._MutationOwners, "__init__", short_bound)
            answer = provider.stream

            async def slow(model_request):
                await asyncio.sleep(1.0)
                async for event in answer(model_request):
                    yield event

            provider.stream = slow
            await hosting.deliver(_delivery(correlation), context=WORKER)
            clock.advance(60)
            await _drain(app, handler)
            task = await app.task_store.load_task(job)
            assert task.status is TaskStatus.COMPLETED
            session = await app.session_store.load(_session_id(job))
            assert session.status is SessionStatus.COMPLETED
            assert len(provider.requests) == 2

    asyncio.run(scenario())


def _storeless(app, provider):
    """A process on the same session store that has no task store."""
    from cayu.external_wait_host import ExternalWaitHost
    from cayu.external_waits import ExternalEventWaits
    from cayu.session_external_waits import SessionExternalWaitAdapter

    storeless = CayuApp(session_store=app.session_store, enable_logging=False)
    storeless.register_provider(provider, default=True)
    storeless.register_agent(AgentSpec(name="root", model="model"))
    adapter = SessionExternalWaitAdapter(
        storeless, ExternalEventWaits(store=app.session_store, access_policy=RegistrarPolicy())
    )
    return storeless, ExternalWaitHost(adapter, context=WORKER)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_servicer_without_the_task_store_leaves_a_task_owned_wait_pending(
    backend, tmp_path, request
):
    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, correlation, _, handler = await _park(app, hosting, scope)
            await hosting.deliver(_delivery(correlation), context=WORKER)
            storeless, host = _storeless(app, provider)
            try:
                wait = await hosting.waits.inspect(correlation, context=WORKER)
                assert await host.service_row(wait) == ("pending", None)
                with pytest.raises(ValueError, match="task store"):
                    async for _ in storeless.interrupt_session(
                        InterruptSessionRequest(session_id=_session_id(job), reason="stop")
                    ):
                        pass
                with pytest.raises(ValueError, match="task store"):
                    await storeless.delete_session(_session_id(job), finished_only=False)
            finally:
                await storeless.aclose()
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.pending_handoff and not wait.retirement_requested
            assert (await _parked(app, job)).kind == "external_wait"
            # Its own worker still delivers the event under the task.
            clock.advance(60)
            await _drain(app, handler)
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_recovery_without_the_task_store_leaves_a_task_owned_turn_to_its_wait(
    backend, tmp_path, request, monkeypatch
):
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job, _, _ = await _start_and_die(
                app, hosting, scope, monkeypatch, _ExternalExecutionToWait, "park", after=False
            )
            await _expire(backend, clock)
            storeless, _ = _storeless(app, provider)
            try:
                results = await storeless.recover_abandoned_sessions(inactive_for_seconds=0)
            finally:
                await storeless.aclose()
            [result] = [r for r in results if r.session_id == _session_id(job)]
            assert result.status is SessionStatus.RUNNING
            assert await _generic(app) == []
            assert (await _parked(app, job)).kind == "external_wait"

    asyncio.run(scenario())
