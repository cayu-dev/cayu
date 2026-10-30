"""External-event waits in sessions a task owns: the task parks until its wait is serviced."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_external_wait_hosting import (
    WORKER,
    RegistrarPolicy,
    _delivery,
    _reserve,
    _service_tasks,
    hosted_app,
)
from tests.core.test_sleep_until import _Clock

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_waits import ExternalEventWaits
from cayu.providers.base import ModelStreamEvent
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.external_waits import ExternalWaitScope
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, ResumeRequest
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    run_task_worker,
    run_task_worker_until_idle,
)
from cayu.tasks.handoff import TaskParkedWait

_JOB = "job"


def _session_id(task_id: str) -> str:
    return f"s-{task_id}"


def _job(hosting, registration, *, calls: list[str] | None = None):
    """The documented handler: run the task's session to its wait, then hand it off."""

    async def handle(app, task, worker_id):
        if task.type != _JOB:
            raise AssertionError(f"The application handler got {task.type}.")
        if calls is not None:
            calls.append(task.id)
        await hosting.run_to_wait(
            RunRequest(
                agent_name="root",
                session_id=_session_id(task.id),
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "Submit the render and wait for it.")],
            ),
            registration,
            context=WORKER,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    return handle


async def _never_continued(app, task, worker_id):
    raise AssertionError(f"A generic continuation claimed parked task {task.id}.")


async def _drain(app, handler, *, worker_id="worker-1"):
    return await run_task_worker_until_idle(
        app,
        app.task_store,
        handler,
        worker_id=worker_id,
        recovered_interrupted_task_handler=_never_continued,
    )


async def _parked(app, task_id):
    task = await app.task_store.load_task(task_id)
    assert task.status is TaskStatus.RUNNING and task.worker_id is None
    receipt = await app.task_store.load_interrupted_task_handoff_receipt(
        task.id, task.interrupted_handoff_id
    )
    return receipt.request.parked


async def _park_job(app, hosting, scope, *, task_id="job-1", deadline=None, calls=None):
    correlation, registration = await _reserve(hosting, scope, deadline=deadline)
    await hosting.register(registration, context=WORKER)
    await app.task_store.create_task(TaskCreate(task_id=task_id, type=_JOB))
    handler = _job(hosting, registration, calls=calls)
    await _drain(app, handler)
    return correlation, registration, handler


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_an_event_continues_the_task_owned_session_under_its_task(backend, tmp_path, request):
    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            correlation, _, handler = await _park_job(app, hosting, scope)
            parked = await _parked(app, "job-1")
            assert parked == TaskParkedWait(kind="external_wait", wait_id=parked.wait_id)
            assert len(provider.requests) == 1

            # Parked: no worker continues it before its event.
            await _drain(app, handler)
            assert len(provider.requests) == 1

            await hosting.deliver(_delivery(correlation), context=WORKER)
            result = await _drain(app, handler)
            assert result.handled >= 1
            assert len(provider.requests) == 2
            task = await app.task_store.load_task("job-1")
            assert task.status is TaskStatus.COMPLETED
            session = await app.session_store.load(_session_id("job-1"))
            assert session.status is SessionStatus.COMPLETED
            [service] = await _service_tasks(app, correlation)
            assert service.status is TaskStatus.COMPLETED

    asyncio.run(scenario())


def _answers(count: int) -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(count)
        ]
    )


@asynccontextmanager
async def clocked_app(clock: _Clock, *, dispatches: int = 2):
    """A memory app whose wait deadlines and task times follow an injected clock."""

    sessions = InMemorySessionStore(ownership_clock=clock)
    tasks = InMemoryTaskStore(clock=clock, ownership_clock=clock)
    provider = _answers(dispatches)
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
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


def test_the_deadline_continues_the_task_owned_session_on_timeout():
    async def scenario():
        clock = _Clock(datetime.now(UTC))
        async with clocked_app(clock) as (app, hosting, scope, provider):
            deadline = clock() + timedelta(minutes=10)
            correlation, _, handler = await _park_job(app, hosting, scope, deadline=deadline)
            assert (await _parked(app, "job-1")).kind == "external_wait"
            early = await _drain(app, handler)
            assert early.handled == 0 and early.next_available_at == deadline
            assert not early.continuations_pending

            clock.advance(11 * 60)
            await _drain(app, handler)
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task("job-1")).status is TaskStatus.COMPLETED
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "timeout" and wait.handoff == "settled"

    asyncio.run(scenario())


def test_a_service_that_runs_before_the_worker_hands_the_task_off_retries(monkeypatch):
    from cayu.tasks import parked_continuation

    monkeypatch.setattr(parked_continuation, "OWNER_HANDOFF_WAIT", timedelta(0))

    async def scenario():
        async with hosted_app("memory", None, None) as (app, hosting, scope, provider):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await hosting.deliver(_delivery(correlation), context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id="job-1", type=_JOB))
            dispositions = []
            parking = _job(hosting, registration)

            async def handle(app, task, worker_id):
                outcome = await parking(app, task, worker_id)
                # The event already won, so its service is due now, while this worker
                # still holds the task: it must not continue the session.
                wait = await hosting.waits.inspect(correlation, context=WORKER)
                dispositions.append(await hosting.host.service_row(wait))
                assert len(provider.requests) == 1
                return outcome

            await run_task_worker(
                app,
                app.task_store,
                handle,
                worker_id="worker-1",
                max_tasks=1,
                poll_interval_s=0.01,
            )
            assert dispositions == [("pending", None)]
            assert (await _parked(app, "job-1")).kind == "external_wait"
            await _drain(app, handle)
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task("job-1")).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_a_handler_rerun_under_a_new_claim_replays_its_wait():
    """A worker that died after parking leaves the same wait to a handler run again."""

    async def scenario():
        async with hosted_app("memory", None, None) as (app, hosting, scope, provider):
            _, registration, _ = await _park_job(app, hosting, scope)
            task = await app.task_store.load_task("job-1")
            receipt = await hosting.run_to_wait(
                RunRequest(
                    agent_name="root",
                    session_id=_session_id(task.id),
                    task_id=task.id,
                    task_worker_id="another-worker",
                    task_lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
                    messages=[Message.text("user", "Submit the render and wait for it.")],
                ),
                registration,
                context=WORKER,
            )
            assert receipt.session_id == _session_id(task.id)
            assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_interrupting_the_session_retires_its_wait_and_hands_the_task_back(
    backend, tmp_path, request
):
    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            job = f"job-{uuid4().hex[:8]}"
            correlation, _, handler = await _park_job(app, hosting, scope, task_id=job)
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id=_session_id(job), reason="stop waiting")
            ):
                pass
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "cancelled" and not wait.pending_handoff

            continued: list[str] = []

            async def generic(app, task, worker_id):
                continued.append(task.id)
                return TaskHandlerOutcome.SESSION_INTERRUPTED

            await run_task_worker(
                app,
                app.task_store,
                handler,
                worker_id="after-stop",
                poll_interval_s=0.01,
                max_tasks=1,
                recovered_interrupted_task_handler=generic,
            )
            assert continued == [job]
            # A late event finds the wait retired and continues nothing.
            await hosting.deliver(_delivery(correlation), context=WORKER)
            await _drain(app, handler)
            assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_interrupting_a_root_session_keeps_its_wait():
    async def scenario():
        async with hosted_app("memory", None, None) as (app, hosting, scope, _):
            correlation, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await hosting.run_to_wait(
                RunRequest(
                    agent_name="root",
                    session_id="root-session",
                    messages=[Message.text("user", "Submit the render.")],
                ),
                registration,
                context=WORKER,
            )
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id="root-session", reason="stop")
            ):
                pass
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert not wait.retirement_requested and wait.pending_handoff

    asyncio.run(scenario())


def test_what_cannot_park_a_task_is_refused():
    async def scenario():
        async with hosted_app("memory", None, None) as (app, hosting, scope, provider):
            _, registration = await _reserve(hosting, scope)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id="job-1", type=_JOB))
            with pytest.raises(ValueError, match="task worker's claim"):
                await hosting.run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id="s",
                        task_id="job-1",
                        messages=[Message.text("user", "go")],
                    ),
                    registration,
                    context=WORKER,
                )
            with pytest.raises(ValueError, match="subagent sessions"):
                await hosting.run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id="child",
                        parent_session_id="parent",
                        messages=[Message.text("user", "go")],
                    ),
                    registration,
                    context=WORKER,
                )
            hand_wired = SessionExternalWaitAdapter(
                app,
                ExternalEventWaits(store=app.session_store, access_policy=RegistrarPolicy()),
            )
            with pytest.raises(ValueError, match="host_external_waits"):
                await hand_wired.run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id="s",
                        task_id="job-1",
                        task_worker_id="w",
                        task_lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
                        messages=[Message.text("user", "go")],
                    ),
                    registration,
                    context=WORKER,
                )
            assert provider.requests == []

            # A subagent session can't be resumed into a wait either.
            async for _ in app.run(RunRequest(agent_name="root", session_id="parent", messages=[])):
                pass
            async for _ in app.run(
                RunRequest(
                    agent_name="root",
                    session_id="child",
                    parent_session_id="parent",
                    messages=[Message.text("user", "hi")],
                )
            ):
                pass
            with pytest.raises(ValueError, match="subagent sessions"):
                await hosting.resume_to_wait(
                    ResumeRequest(session_id="child", messages=[Message.text("user", "go")]),
                    registration,
                    context=WORKER,
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["postgres"])
def test_two_workers_racing_on_one_event_continue_the_session_once(backend, tmp_path, request):
    async def scenario():
        async with hosted_app(backend, tmp_path, request) as (app, hosting, scope, provider):
            job = f"job-{uuid4().hex[:8]}"
            correlation, _, handler = await _park_job(app, hosting, scope, task_id=job)
            await hosting.deliver(_delivery(correlation), context=WORKER)
            # The service task and a sweep-made attempt race the same claim.
            await asyncio.gather(
                _drain(app, handler, worker_id="worker-a"),
                _drain(app, handler, worker_id="worker-b"),
                hosting.host.service_row(await hosting.waits.inspect(correlation, context=WORKER)),
            )
            await _drain(app, handler)
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task(job)).status is TaskStatus.COMPLETED

    asyncio.run(scenario())


class _WorkerDied(BaseException):
    """Stands in for the process dying: nothing below it gets to clean up."""


def test_a_service_that_died_before_settling_its_handoff_settles_it_next_time(monkeypatch):
    from cayu.runtime._external_wait_receiver import ExternalWaitLatchReceiver

    async def scenario():
        clock = _Clock(datetime.now(UTC))
        async with clocked_app(clock) as (app, hosting, scope, provider):
            correlation, _, handler = await _park_job(app, hosting, scope)
            await hosting.deliver(_delivery(correlation), context=WORKER)

            async def dies(receiver):
                raise RuntimeError("the servicing worker died after continuing the session")

            with monkeypatch.context() as patch:
                patch.setattr(ExternalWaitLatchReceiver, "reconcile_handoff", dies)
                await _drain(app, handler)
            # The session continued under the task, which completed with it.
            assert len(provider.requests) == 2
            assert (await app.task_store.load_task("job-1")).status is TaskStatus.COMPLETED
            assert (await hosting.waits.inspect(correlation, context=WORKER)).pending_handoff

            clock.advance(60)
            await _drain(app, handler, worker_id="worker-2")
            assert len(provider.requests) == 2
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.handoff == "settled" and not wait.retirement_requested

    asyncio.run(scenario())
