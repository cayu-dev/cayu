"""``app.host_external_waits``: the task worker continues parked sessions by itself."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_wait_hosting import (
    EXTERNAL_WAIT_SERVICE_TASK_TYPE,
    ExternalEventSource,
    _attempt,
)
from cayu.external_wait_scheduler import EXTERNAL_WAIT_TASK_TYPE
from cayu.external_waits import ExternalWaitAccessPolicy, ExternalWaitContext
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.external_waits import (
    ExternalCorrelationRequest,
    ExternalEventDelivery,
    ExternalWaitRegistration,
    ExternalWaitScope,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import run_task_worker, run_task_worker_until_idle
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery

WORKER = ExternalWaitContext(principal="worker")
RENDERER = ExternalWaitContext(principal="renderer-webhook")


class Policy(ExternalWaitAccessPolicy):
    def authorize(self, context, *, scope, source, action):
        if source != "renderer":
            return False
        if context.principal == "renderer-webhook":
            return action == "deliver"
        return context.principal == "worker"


class RegistrarPolicy(Policy):
    def authorize(self, context, *, scope, source, action):
        if context.principal == "registrar":
            return source == "renderer" and action in {"reserve", "register"}
        return super().authorize(context, scope=scope, source=source, action=action)


class ReadOnlyPolicy(ExternalWaitAccessPolicy):
    def authorize(self, context, *, scope, source, action):
        return action == "read"


class Renderer(ExternalEventSource):
    source = "renderer"

    async def receive(self, request):
        return None


@asynccontextmanager
async def hosted_app(backend, tmp_path, request, *, dispatches=2, clocks=None):
    if backend == "memory" and clocks is not None:
        # The task store's clock runs ahead of the wait store's by clocks["ahead"], and
        # the session store's by clocks.get("sessions_ahead", 0).
        sessions = InMemorySessionStore(
            ownership_clock=lambda: datetime.now(UTC) + clocks.get("sessions_ahead", timedelta(0))
        )
        tasks = InMemoryTaskStore(
            clock=lambda: datetime.now(UTC) + clocks["ahead"],
            ownership_clock=lambda: datetime.now(UTC) + clocks["ahead"],
        )
    elif backend == "memory":
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
    elif backend == "sqlite":
        sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        tasks = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    else:
        dsn = request.getfixturevalue("postgres_dsn")
        sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE, min_size=1, max_size=2)
        tasks = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(dispatches)
        ]
    )
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="root", model="model"))
    scope = ExternalWaitScope(application_scope="video-" + uuid4().hex, generation=1)
    hosting = app.host_external_waits(
        access_policy=RegistrarPolicy(), context=WORKER, scopes={scope: ("renderer",)}
    )
    try:
        yield app, hosting, scope, provider
    finally:
        await app.aclose()
        await hosting.aclose()
        if backend != "memory":
            await tasks.close()
            await sessions.close()


async def _reserve(hosting, scope, *, deadline=None, key="job-42"):
    correlation = await hosting.waits.reserve_correlation(
        ExternalCorrelationRequest(
            scope=scope, source="renderer", correlation_key=key, deadline=deadline
        ),
        context=WORKER,
    )
    return correlation, ExternalWaitRegistration(
        correlation=correlation,
        operation_key="wait-" + key,
        projector_id="json",
        projector_version=1,
    )


async def _park(hosting, registration, session_id="video-session"):
    return await hosting.run_to_wait(
        RunRequest(
            agent_name="root",
            session_id=session_id,
            messages=[Message.text("user", "The render was submitted.")],
        ),
        registration,
        context=WORKER,
    )


async def _refuse(app, task, worker_id):
    raise AssertionError(f"The application handler got {task.type}.")


async def _tasks(app, task_type):
    return await app.task_store.list_tasks(TaskQuery(type=task_type))


async def _service_tasks(app, correlation):
    return [
        task
        for task in await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
        if task.input["incarnation"] == correlation.incarnation
    ]


async def _drain(app):
    return await run_task_worker_until_idle(app, app.task_store, _refuse, worker_id="worker-1")


@asynccontextmanager
async def running_worker(app):
    """A worker that is already running, past its startup sweep."""
    stop = asyncio.Event()
    worker = asyncio.create_task(
        run_task_worker(
            app, app.task_store, _refuse, worker_id="worker-1", stop=stop, poll_interval_s=0.05
        )
    )
    await asyncio.sleep(0.5)
    try:
        yield
    finally:
        stop.set()
        await worker


async def _completed(app, task_id):
    async with asyncio.timeout(20):
        while (task := await app.task_store.load_task(task_id)).status.value != "completed":
            await asyncio.sleep(0.05)
    return task


def _delivery(correlation, delivery_id="render-done-1", payload='{"url":"v.mp4"}'):
    return ExternalEventDelivery(
        correlation=correlation, delivery_id=delivery_id, payload_json=payload
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_delivered_event_continues_the_session_once(backend, tmp_path, request):
    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)
            assert len(provider.requests) == 1

            # A worker that is already running continues the session from the
            # delivery's task, long before its next sweep.
            async with running_worker(app):
                receipt = await hosting.deliver(_delivery(correlation), context=RENDERER)
                assert receipt.disposition == "accepted"
                # A retried webhook converges on the same service task.
                assert await hosting.deliver(_delivery(correlation), context=RENDERER) == receipt
                [service] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
                done = await _completed(app, service.id)
                # A later, different event doesn't win and schedules nothing.
                later = await hosting.deliver(
                    _delivery(correlation, "render-done-2"), context=RENDERER
                )
                assert later.disposition == "settled"
                assert len(await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)) == 1
            assert len(provider.requests) == 2
            assert "v.mp4" in str(provider.requests[1].messages[-1])
            assert done.result == {"service": "settled"}
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "event" and wait.handoff == "settled"
            assert (await _drain(app)).handled == 0
            assert len(provider.requests) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_the_deadline_task_continues_the_session_on_timeout(backend, tmp_path, request):
    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            deadline = datetime.now(UTC) + timedelta(seconds=2)
            correlation, registration = await _reserve(hosting, scope, deadline=deadline)
            registered = await hosting.register(registration, context=WORKER)
            assert registered.timer_published
            [timer] = await _tasks(app, EXTERNAL_WAIT_TASK_TYPE)
            await _park(hosting, registration)
            assert await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE) == []

            early = await _drain(app)
            assert early.handled == 0 and early.next_available_at == timer.available_at
            async with running_worker(app):
                done = await _completed(app, timer.id)
            assert len(provider.requests) == 2
            assert done.result == {
                "outcome_kind": "timeout",
                "pending_handoff": True,
                "service": "settled",
            }
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "timeout" and wait.handoff == "settled"

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_an_event_before_the_session_parks_is_serviced_after_it_parks(backend, tmp_path, request):
    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.deliver(_delivery(correlation), context=RENDERER)
            # Nothing is parked yet, so there's nothing to service: parking schedules it.
            assert await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE) == []
            assert (await _drain(app)).handled == 0
            assert provider.requests == []

            await hosting.register(registration, context=WORKER)
            parked = await _park(hosting, registration)
            assert parked.wait.outcome.kind == "event" and parked.wait.pending_handoff
            assert len(await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)) == 1
            assert (await _drain(app)).handled == 1
            assert len(provider.requests) == 2
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.handoff == "settled"

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_the_worker_sweep_schedules_a_wait_whose_task_was_lost(
    backend, tmp_path, request, monkeypatch
):
    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)

            async def lost(*args, **kwargs):
                raise ConnectionError("task store unavailable")

            patch = monkeypatch
            with monkeypatch.context() as lose:
                lose.setattr(hosting, "_ensure_service", lost)
                receipt = await hosting.deliver(_delivery(correlation), context=RENDERER)
            assert receipt.disposition == "accepted"
            assert await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE) == []

            # A worker that only claims other tasks neither sweeps nor continues it.
            narrow = await run_task_worker_until_idle(
                app, app.task_store, _refuse, worker_id="narrow", query=TaskQuery(type="other")
            )
            assert narrow.handled == 0 and len(provider.requests) == 1
            assert await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE) == []

            # The sweep schedules the missing task; the continuation runs in it.
            result = await _drain(app)
            assert result.idle and result.handled == 1
            assert len(provider.requests) == 2
            [swept] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
            assert swept.result == {"service": "settled"}
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.handoff == "settled"

            # Waits that need nothing are only read.
            writes = []
            mutate = app.session_store._mutate_external_wait

            async def counted(command):
                writes.append(command.kind)
                return await mutate(command)

            patch.setattr(app.session_store, "_mutate_external_wait", counted)
            assert await hosting.sweep() == 0
            assert writes == []

    asyncio.run(scenario())


def test_a_worker_on_another_task_store_leaves_hosted_tasks_to_the_application(tmp_path):
    async def scenario():
        async with hosted_app("memory", tmp_path, None) as (app, hosting, scope, _provider):
            other = InMemoryTaskStore()
            correlation, _ = await _reserve(hosting, scope)
            await app.task_store.create_task(_attempt(correlation, "manual", 1, datetime.now(UTC)))
            [task] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
            await other.create_task(_copy(task))
            seen = []

            async def handler(app, task, worker_id):
                seen.append(task.type)
                from cayu.tasks.worker import complete_managed_task

                await complete_managed_task(other, task, worker_id, {})

            await run_task_worker(app, other, handler, worker_id="other", max_tasks=1)
            assert seen == [EXTERNAL_WAIT_SERVICE_TASK_TYPE]

    asyncio.run(scenario())


def _copy(task):
    from cayu.tasks.creation import TaskCreate

    return TaskCreate(task_id=task.id, type=task.type, input=task.input)  # unmanaged copy


def test_hosting_refuses_what_it_could_not_run(tmp_path):
    async def scenario():
        scope = ExternalWaitScope(application_scope="video", generation=1)
        sessions = InMemorySessionStore()
        no_tasks = CayuApp(session_store=sessions, enable_logging=False)
        with pytest.raises(ValueError, match="task store that schedules tasks"):
            no_tasks.host_external_waits(
                access_policy=Policy(), context=WORKER, scopes={scope: ("renderer",)}
            )
        app = CayuApp(session_store=sessions, task_store=InMemoryTaskStore(), enable_logging=False)
        with pytest.raises(ValueError, match="must grant 'service'"):
            app.host_external_waits(
                access_policy=ReadOnlyPolicy(), context=WORKER, scopes={scope: ("renderer",)}
            )
        with pytest.raises(ValueError, match="must grant 'read'"):
            app.host_external_waits(
                access_policy=Policy(), context=RENDERER, scopes={scope: ("renderer",)}
            )
        for options, message in (
            ({"scopes": {}}, "at least one scope"),
            ({"scopes": {scope: ()}}, "distinct, nonblank"),
            ({"sources": (Renderer(),)}, "need a delivery_path"),
            ({"delivery_path": "/webhooks/waits"}, "at least one external event source"),
            ({"sources": (Renderer(),), "delivery_path": "/webhooks/"}, "fixed path"),
            ({"sources": (Renderer(),), "delivery_path": "/{x}/waits"}, "fixed path"),
            ({"sources": (Renderer(),), "delivery_path": "/web hooks"}, "fixed path"),
            ({"sources": (Renderer(),), "delivery_path": "/hooks/.."}, "fixed path"),
            (
                {"sources": (Renderer(),), "delivery_path": "/w", "max_request_bytes": 0},
                "max_request_bytes",
            ),
            ({"sources": (Renderer(), Renderer()), "delivery_path": "/w"}, "configured twice"),
        ):
            with pytest.raises(ValueError, match=message):
                app.host_external_waits(
                    **{
                        "access_policy": Policy(),
                        "context": WORKER,
                        "scopes": {scope: ("renderer",)},
                        **options,
                    }
                )

        class Stranger(Renderer):
            source = "stranger"

        with pytest.raises(ValueError, match="must name a source of a configured scope"):
            app.host_external_waits(
                access_policy=Policy(),
                context=WORKER,
                scopes={scope: ("renderer",)},
                sources=(Stranger(),),
                delivery_path="/w",
            )
        app.host_external_waits(
            access_policy=Policy(), context=WORKER, scopes={scope: ("renderer",)}
        )
        with pytest.raises(ValueError, match="already hosts"):
            app.host_external_waits(
                access_policy=Policy(), context=WORKER, scopes={scope: ("renderer",)}
            )
        await app.aclose()
        await no_tasks.aclose()

    asyncio.run(scenario())


def test_a_service_task_for_another_incarnation_is_obsolete(tmp_path):
    async def scenario():
        async with hosted_app("memory", tmp_path, None) as (app, hosting, scope, _provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)
            await hosting.waits.deliver(_delivery(correlation), context=WORKER)
            stale = correlation.model_copy(update={"incarnation": "replaced"})
            await app.task_store.create_task(_attempt(stale, "stale", 1, datetime.now(UTC)))
            [task] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
            async with running_worker(app):
                done = await _completed(app, task.id)
            assert done.result == {"service": "obsolete"}

    asyncio.run(scenario())


def test_a_deadline_task_that_runs_early_retries_shortly_after_the_deadline(tmp_path):
    async def scenario():
        deadline = datetime.now(UTC) + timedelta(seconds=2)
        clocks = {"ahead": timedelta(seconds=5)}
        async with hosted_app("memory", tmp_path, None, clocks=clocks) as (
            app,
            hosting,
            scope,
            provider,
        ):
            correlation, registration = await _reserve(hosting, scope, deadline=deadline)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)
            [timer] = await _tasks(app, EXTERNAL_WAIT_TASK_TYPE)

            # The task store says the deadline passed; the wait store doesn't yet.
            early = await _drain(app)
            assert early.handled == 1 and len(provider.requests) == 1
            retry_at = deadline + timedelta(seconds=30)
            assert early.next_available_at == retry_at
            done = await app.task_store.load_task(timer.id)
            assert done.result == {
                "outcome_kind": None,
                "pending_handoff": True,
                "retry_at": retry_at.isoformat(),
            }

            # Once both clocks are past it, the retry continues the session on timeout.
            await asyncio.sleep((deadline - datetime.now(UTC)).total_seconds() + 0.1)
            clocks["ahead"] = timedelta(seconds=35)
            assert (await _drain(app)).handled == 1
            assert len(provider.requests) == 2
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "timeout" and wait.handoff == "settled"

    asyncio.run(scenario())


def test_a_hand_wired_schedulers_deadline_task_goes_to_the_application(tmp_path):
    async def scenario():
        async with hosted_app("memory", tmp_path, None) as (app, hosting, scope, _provider):
            from cayu.external_wait_scheduler import TaskStoreWaitScheduler
            from cayu.tasks.worker import complete_managed_task

            _, registration = await _reserve(
                hosting, scope, deadline=datetime.now(UTC) - timedelta(seconds=1)
            )
            await hosting.waits.register(registration, context=WORKER)
            own = TaskStoreWaitScheduler(
                waits=hosting.waits, task_store=app.task_store, scheduler_id="hand-wired"
            )
            await own.schedule(registration, context=WORKER)
            seen = []

            async def handler(app, task, worker_id):
                seen.append(task.type)
                await complete_managed_task(app.task_store, task, worker_id, {})

            await run_task_worker_until_idle(app, app.task_store, handler, worker_id="w")
            assert seen == [EXTERNAL_WAIT_TASK_TYPE]

    asyncio.run(scenario())


def test_a_registrar_without_service_rights_still_gets_its_deadline_scheduled(tmp_path):
    async def scenario():
        async with hosted_app("memory", tmp_path, None) as (app, hosting, scope, _provider):
            registrar = ExternalWaitContext(principal="registrar")
            correlation = await hosting.waits.reserve_correlation(
                ExternalCorrelationRequest(
                    scope=scope,
                    source="renderer",
                    correlation_key="job-9",
                    deadline=datetime.now(UTC) + timedelta(hours=1),
                ),
                context=registrar,
            )
            registered = await hosting.register(
                ExternalWaitRegistration(
                    correlation=correlation,
                    operation_key="wait-job-9",
                    projector_id="json",
                    projector_version=1,
                ),
                context=registrar,
            )
            assert registered.timer_published
            assert len(await _tasks(app, EXTERNAL_WAIT_TASK_TYPE)) == 1

    asyncio.run(scenario())


def test_one_failing_wait_does_not_stop_the_sweep(tmp_path, monkeypatch):
    async def scenario():
        async with hosted_app("memory", tmp_path, None, dispatches=4) as (
            app,
            hosting,
            scope,
            _provider,
        ):
            for key in ("a-job", "b-job"):
                correlation, registration = await _reserve(hosting, scope, key=key)
                await hosting.register(registration, context=WORKER)
                await _park(hosting, registration, session_id="session-" + key)
                await hosting.waits.deliver(_delivery(correlation), context=RENDERER)
            sweep_row = hosting._sweep_row

            async def failing(row):
                if row.correlation.request.correlation_key == "a-job":
                    raise ConnectionError("store hiccup")
                return await sweep_row(row)

            monkeypatch.setattr(hosting, "_sweep_row", failing)
            assert await hosting.sweep() == 1
            [task] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
            assert task.input["correlation_key"] == "b-job"

    asyncio.run(scenario())


def test_an_additional_early_event_schedules_no_second_task(tmp_path):
    async def scenario():
        async with hosted_app("memory", tmp_path, None) as (app, hosting, scope, _provider):
            correlation, _ = await _reserve(hosting, scope)
            first = await hosting.deliver(_delivery(correlation), context=RENDERER)
            second = await hosting.deliver(_delivery(correlation, "other"), context=RENDERER)
            assert (first.disposition, second.disposition) == ("accepted", "additional")
            # Neither schedules work before the session parks.
            assert await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE) == []

    asyncio.run(scenario())


def test_the_sweep_replaces_a_service_task_that_failed(tmp_path):
    async def scenario():
        clocks = {"ahead": timedelta(0)}
        async with hosted_app("memory", tmp_path, None, clocks=clocks) as (
            app,
            hosting,
            scope,
            provider,
        ):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)
            await hosting.waits.deliver(_delivery(correlation), context=RENDERER)
            assert await hosting.sweep() == 1
            [first] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
            # The attempt ends without servicing, as when its worker raised.
            claimed = await app.task_store.claim_task(
                "crashed", TaskQuery(type=EXTERNAL_WAIT_SERVICE_TASK_TYPE)
            )
            assert claimed is not None and claimed.id == first.id
            await app.task_store.fail_task(
                first.id,
                {"message": "transient"},
                worker_id="crashed",
                lease_expires_at=claimed.lease_expires_at,
            )

            assert await hosting.sweep() == 1
            assert await hosting.sweep() == 0  # The successor is live.
            successor = next(
                task
                for task in await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
                if task.id != first.id
            )
            assert successor.input["attempt"] == first.input["attempt"] + 1
            clocks["ahead"] = timedelta(seconds=60)
            assert (await _drain(app)).handled == 1
            assert len(provider.requests) == 2
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.handoff == "settled"

    asyncio.run(scenario())


def test_a_deadline_task_that_cannot_finish_servicing_schedules_a_successor(tmp_path, monkeypatch):
    async def scenario():
        clocks = {"ahead": timedelta(0)}
        async with hosted_app("memory", tmp_path, None, clocks=clocks) as (
            app,
            hosting,
            scope,
            provider,
        ):
            deadline = datetime.now(UTC) + timedelta(seconds=1)
            correlation, registration = await _reserve(hosting, scope, deadline=deadline)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)
            assert await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE) == []
            await asyncio.sleep((deadline - datetime.now(UTC)).total_seconds() + 0.1)
            service_row = hosting.host.service_row
            calls = []

            async def busy_twice(row):
                calls.append(row)
                if len(calls) <= 2:
                    return "pending", "fenced"
                return await service_row(row)

            monkeypatch.setattr(hosting.host, "service_row", busy_twice)
            # The deadline task alone leaves a service attempt behind it.
            await run_task_worker(app, app.task_store, _refuse, worker_id="one", max_tasks=1)
            [attempt] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
            assert attempt.status.value == "pending"
            first = await _drain(app)
            # The deadline task and the service attempt it scheduled both met a busy
            # writer, so a later attempt waits, and the host learns when to come back.
            assert first.handled == 1 and len(provider.requests) == 1
            assert first.next_available_at is not None
            retry = next(
                task
                for task in await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
                if task.status.value == "pending"
            )
            assert retry.available_at == first.next_available_at

            clocks["ahead"] = timedelta(seconds=60)
            assert (await _drain(app)).handled == 1
            assert len(provider.requests) == 2
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "timeout" and wait.handoff == "settled"

    asyncio.run(scenario())


def test_retries_are_timed_from_when_the_attempt_ran(tmp_path, monkeypatch):
    async def scenario():
        # The worker comes back two days after the event was accepted.
        clocks = {"ahead": timedelta(days=2)}
        async with hosted_app("memory", tmp_path, None, clocks=clocks) as (
            app,
            hosting,
            scope,
            _provider,
        ):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)
            await hosting.deliver(_delivery(correlation), context=RENDERER)

            async def busy(row):
                return "pending", "fenced"

            monkeypatch.setattr(hosting.host, "service_row", busy)
            result = await _drain(app)
            ran_at = datetime.now(UTC) + clocks["ahead"]
            # One attempt ran and scheduled one successor, in the future of when it ran.
            assert result.handled == 1
            assert result.next_available_at is not None
            assert result.next_available_at > ran_at
            assert result.next_available_at - ran_at <= timedelta(seconds=31)

    asyncio.run(scenario())


def test_a_failing_service_attempt_leaves_a_successor(tmp_path, monkeypatch):
    async def scenario():
        async with hosted_app("memory", tmp_path, None) as (app, hosting, scope, _provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await _park(hosting, registration)
            await hosting.deliver(_delivery(correlation), context=RENDERER)
            [attempt] = await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)

            async def broken(row):
                raise ConnectionError("store hiccup")

            monkeypatch.setattr(hosting.host, "service_row", broken)
            await run_task_worker(app, app.task_store, _refuse, worker_id="one", max_tasks=1)
            failed = await app.task_store.load_task(attempt.id)
            assert failed.status.value == "failed"
            successor = next(
                task
                for task in await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)
                if task.id != attempt.id
            )
            assert successor.status.value == "pending" and successor.input["attempt"] == 2

    asyncio.run(scenario())


def test_a_registered_wait_that_has_not_parked_schedules_nothing(tmp_path):
    async def scenario():
        async with hosted_app("memory", tmp_path, None) as (app, hosting, scope, provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            receipt = await hosting.deliver(_delivery(correlation), context=RENDERER)
            assert receipt.disposition == "accepted"
            assert await hosting.sweep() == 0
            assert await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE) == []
            await _park(hosting, registration)
            assert len(await _tasks(app, EXTERNAL_WAIT_SERVICE_TASK_TYPE)) == 1
            assert (await _drain(app)).handled == 1
            assert len(provider.requests) == 2

    asyncio.run(scenario())


def test_an_event_during_the_parking_turn_continues_the_session_once(tmp_path):
    async def scenario():
        clocks = {"ahead": timedelta(0)}
        async with hosted_app("memory", tmp_path, None, clocks=clocks) as (
            app,
            hosting,
            scope,
            provider,
        ):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            stream = provider.stream

            async def deliver_mid_turn(request):
                await hosting.deliver(_delivery(correlation), context=RENDERER)
                # The attempt can't continue a turn that is still running.
                await run_task_worker(app, app.task_store, _refuse, worker_id="w", max_tasks=1)
                async for event in stream(request):
                    yield event

            provider.stream = deliver_mid_turn
            await _park(hosting, registration)
            provider.stream = stream
            assert len(provider.requests) == 1
            # Its successor continues the parked session once.
            clocks["ahead"] = timedelta(seconds=60)
            await _drain(app)
            assert len(provider.requests) == 2
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.handoff == "settled"
            assert (await _drain(app)).handled == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("scheduled_by", ["delivery", "sweep"])
def test_an_event_for_a_wait_whose_binding_publication_failed_is_serviced(
    backend, scheduled_by, tmp_path, request, monkeypatch
):
    from cayu.sessions.external_waits import ExternalWaitUnavailable

    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            mutate = hosting.waits._mutate

            async def fail_binding(command):
                if command.kind == "bind":
                    raise RuntimeError("binding publication unavailable")
                return await mutate(command)

            # The native preparation commits, then publishing its binding fails.
            with monkeypatch.context() as patch:
                patch.setattr(hosting.waits, "_mutate", fail_binding)
                with pytest.raises(ExternalWaitUnavailable):
                    await _park(hosting, registration, session_id="binding-" + uuid4().hex)
            key = correlation.request.correlation_key
            record = await app.session_store._read_external_wait(scope, key)
            assert record.execution is not None and record.continuation is None
            assert record.handoff == "unbound" and record.pending_handoff

            if scheduled_by == "delivery":
                await hosting.deliver(_delivery(correlation), context=RENDERER)
            else:
                await hosting.waits.deliver(_delivery(correlation), context=RENDERER)
                assert await _service_tasks(app, correlation) == []
                assert await hosting.sweep() == 1
            [attempt] = await _service_tasks(app, correlation)

            # The attempt repairs the binding. The binding is published before the
            # turn runs, so there is no completed turn to recover and nothing is
            # dispatched blind; the attempt retries until the wait is retired.
            await _drain(app)
            repaired = await app.session_store._read_external_wait(scope, key)
            assert repaired.continuation is not None and repaired.handoff == "pending"
            done = await app.task_store.load_task(attempt.id)
            assert done.result["service"] == "pending" and "retry_at" in done.result
            assert provider.requests == []

            retired = await hosting.adapter.retire_execution(
                registration, operation_key="retire-failed-start", context=WORKER
            )
            assert retired.handoff == "excluded" and retired.outcome.kind == "event"
            assert await hosting.sweep() == 0
            assert provider.requests == []

    asyncio.run(scenario())
