"""Cancelling a session-attached task whose worker died settles it as cancelled.

The worker that held the task is gone, so nobody drains the cancellation. Once the
dead worker's lease has expired, the next worker fences the task's session as
interrupted, the same way it recovers any expired attached task, and settles the
task as cancelled instead of handing it off. It never runs the model or a tool
again, and it leaves a task whose worker still holds its lease alone.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_external_wait_hosting import WORKER, _delivery, _reserve
from tests.core.test_external_waits_in_tasks import _JOB, _drain, _job, _session_id, _WorkerDied
from tests.core.test_external_waits_in_tasks_edges import (
    _DYING_LEASE_SECONDS,
    BACKENDS,
    _die,
    _expire,
    clocked,
)
from tests.core.test_external_waits_in_tasks_edges import _stores as _shared_stores
from tests.core.test_foreground_subagent_recovery import _identity
from tests.core.test_sleep_until import (
    _Clock,
    _own_postgres_database,
    _StableScriptedProvider,
    _tool_results,
)
from tests.core.test_sleep_until_children import (
    _due,
    _owned_family,
    _parked_parent,
    _until_idle,
)

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.storage import PostgresSessionStore, SQLiteSessionStore, SQLiteTaskStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    run_task_worker,
    run_task_worker_until_idle,
    session_wakes,
)
from cayu.tasks import worker as task_worker
from cayu.tasks.abandoned import _declaring_worker_actor
from cayu.tasks.continuation_claims import TaskContinuationExpectation
from cayu.tasks.handoff import new_interrupted_task_continuation_handoff_id
from cayu.tasks.queries import TaskQuery
from cayu.tools import SleepUntilTool
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.subagents import SubagentSpec, SubagentTool


def _stores(backend: str, clock, tmp_path, request):
    """The edge tests' stores, on a database of this test's own for PostgreSQL."""

    if backend != "postgres":
        return _shared_stores(backend, clock, tmp_path, request)
    dsn = _own_postgres_database(request)
    return (
        PostgresSessionStore(
            dsn, schema_mode=SchemaMode.CREATE, min_size=1, max_size=2, clock=clock
        ),
        PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=clock),
    )


def _wake_id_of(held) -> str:
    """The wake whose continuation claim holds ``held``."""

    return held.worker_id.rpartition(":continue:")[2]


class _Charge(Tool):
    """The worker dies while this tool runs; counting calls proves it never runs again."""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="charge",
            description="Charge the card once.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.IDEMPOTENT,
            max_terminal_payload_bytes=4096,
        )
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        raise _WorkerDied


def _charge_then_answer(request: ModelRequest):
    if not _tool_results(request):
        return [
            ModelStreamEvent.tool_call(id="call_charge", name="charge", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    return [ModelStreamEvent.text_delta("charged"), ModelStreamEvent.completed()]


def _sleep_then_charge(request: ModelRequest):
    results = _tool_results(request)
    if not results:
        return [
            ModelStreamEvent.tool_call(
                id="call_sleep", name="sleep_until", arguments={"duration_seconds": 30}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    if len(results) == 1:
        return [
            ModelStreamEvent.tool_call(id="call_charge", name="charge", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    return [ModelStreamEvent.text_delta("charged"), ModelStreamEvent.completed()]


def _app(sessions, tasks, clock, respond) -> tuple[CayuApp, _StableScriptedProvider, _Charge]:
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    provider = _StableScriptedProvider(response_factory=respond)
    app.register_provider(provider, default=True)
    charge = _Charge()
    app.register_agent(
        AgentSpec(name="charger", model="scripted-model"), tools=[SleepUntilTool(), charge]
    )
    return app, provider, charge


async def _run_job(app, task, worker_id):
    async for _ in app.run(
        RunRequest(
            agent_name="charger",
            session_id=_session_id(task.id),
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
            messages=[Message.text("user", "charge the card")],
        )
    ):
        pass
    session = await app.session_store.load(_session_id(task.id))
    return (
        TaskHandlerOutcome.SESSION_INTERRUPTED
        if session.status is SessionStatus.INTERRUPTED
        else None
    )


async def _sleep_and_die_before_the_handoff(app, task, worker_id):
    """The session sleeps, and the worker dies before it hands the task off parked."""

    assert await _run_job(app, task, worker_id) is TaskHandlerOutcome.SESSION_INTERRUPTED
    raise _WorkerDied


async def _refuse(app, task, worker_id):
    raise AssertionError(f"The application handler got {task.id}.")


async def _never_continued(app, task, worker_id):
    raise AssertionError(f"A continuation claimed cancelled task {task.id}.")


async def _next_worker(app, **options):
    return await run_task_worker_until_idle(
        app,
        app.task_store,
        _refuse,
        worker_id="after-crash",
        recovered_interrupted_task_handler=_never_continued,
        **options,
    )


async def _cancel_fenced(app, task_id: str):
    """Cancel the task while its dead worker still holds it: only a fence is recorded."""

    requested = await app.task_store.cancel_task(task_id)
    assert requested.status is TaskStatus.RUNNING
    assert requested.status_reason == "cancellation_requested"
    assert requested.worker_id is not None
    return requested


async def _assert_cancelled_and_interrupted(app, task_id: str) -> None:
    task = await app.task_store.load_task(task_id)
    assert task.status is TaskStatus.CANCELLED
    assert task.worker_id is None and task.lease_expires_at is None
    session = await app.session_store.load(_session_id(task_id))
    assert session.status is SessionStatus.INTERRUPTED


async def _died_mid_tool(app) -> str:
    """A plain task-worker run of ``app.run`` whose worker dies inside a tool."""

    task_id = f"job-plain-{uuid4().hex[:8]}"
    await app.task_store.create_task(TaskCreate(task_id=task_id, type="job"))
    await _die(app, _run_job)
    session = await app.session_store.load(_session_id(task_id))
    assert session.status is SessionStatus.RUNNING
    return task_id


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancelling_a_task_whose_worker_died_mid_run_settles_it_as_cancelled(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, provider, charge = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            task_id = await _died_mid_tool(app)
            await _cancel_fenced(app, task_id)
            await _expire(backend, clock)
            result = await _next_worker(app)
            await _assert_cancelled_and_interrupted(app, task_id)
            assert not result.attention_required
            # Neither the model nor the tool ran again.
            assert len(provider.requests) == 1 and charge.calls == 1
            # A later run finds nothing left to do.
            assert not (await _next_worker(app)).attention_required
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_long_running_worker_settles_a_cancelled_task_whose_worker_died(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, provider, charge = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            task_id = await _died_mid_tool(app)
            await _cancel_fenced(app, task_id)
            await _expire(backend, clock)
            stop = asyncio.Event()
            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    tasks,
                    _refuse,
                    worker_id="long-running",
                    poll_interval_s=0.01,
                    stop=stop,
                    recovered_interrupted_task_handler=_never_continued,
                )
            )
            try:
                for _ in range(500):
                    if (await tasks.load_task(task_id)).status is TaskStatus.CANCELLED:
                        break
                    await asyncio.sleep(0.01)
            finally:
                stop.set()
                await worker
            await _assert_cancelled_and_interrupted(app, task_id)
            assert len(provider.requests) == 1 and charge.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_cancelled_task_whose_worker_still_holds_its_lease_is_left_to_it(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, provider, charge = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            task_id = await _died_mid_tool(app)
            requested = await _cancel_fenced(app, task_id)
            await _next_worker(app)
            # The lease is live: its holder may still be draining, so nothing settles it.
            task = await tasks.load_task(task_id)
            assert task.status is TaskStatus.RUNNING
            assert task.status_reason == "cancellation_requested"
            assert task.worker_id == requested.worker_id
            session = await sessions.load(_session_id(task_id))
            assert session.status is SessionStatus.RUNNING
            await _expire(backend, clock)
            await _next_worker(app)
            await _assert_cancelled_and_interrupted(app, task_id)
            assert len(provider.requests) == 1 and charge.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancelling_a_sleeping_task_whose_worker_died_before_parking_it_stops_the_sleep(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, provider, charge = _app(sessions, tasks, clock, _sleep_then_charge)
        task_id = f"job-sleep-{uuid4().hex[:8]}"
        try:
            await tasks.create_task(TaskCreate(task_id=task_id, type="job"))
            await _die(app, _sleep_and_die_before_the_handoff)
            held = await tasks.load_task(task_id)
            assert held.worker_id is not None and held.interrupted_handoff_id is None
            await _cancel_fenced(app, task_id)
            await _expire(backend, clock)
            await _next_worker(app)
            await _assert_cancelled_and_interrupted(app, task_id)
            # The wake comes due and finds its task ended: the sleep stops, nothing runs.
            await _next_worker(app)
            checkpoint = await sessions.load_checkpoint(_session_id(task_id))
            assert "pending_wake" not in checkpoint
            assert len(provider.requests) == 1 and charge.calls == 0
            assert (await sessions.load(_session_id(task_id))).status is SessionStatus.INTERRUPTED
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancelling_a_task_whose_wake_worker_died_mid_tool_settles_it_as_cancelled(
    backend, tmp_path, request, monkeypatch
) -> None:
    # The wake's claim of the task expires soon after the dying worker's own lease. On
    # PostgreSQL it follows the database clock, so it is long enough to renew reliably.
    postgres = backend == "postgres"
    claim_lease = _DYING_LEASE_SECONDS + 2 if postgres else _DYING_LEASE_SECONDS
    monkeypatch.setattr(session_wakes, "_CONTINUATION_LEASE_SECONDS", claim_lease)

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, provider, charge = _app(sessions, tasks, clock, _sleep_then_charge)
        task_id = f"job-woken-{uuid4().hex[:8]}"
        try:
            await tasks.create_task(TaskCreate(task_id=task_id, type="job"))
            await run_task_worker_until_idle(
                app, tasks, _run_job, worker_id="first", query=TaskQuery(type="job")
            )
            parked = await tasks.load_task(task_id)
            assert parked.worker_id is None and parked.interrupted_handoff_id is not None
            # The wake comes due; its claim of the task dies inside the woken round's tool.
            clock.advance(31)
            await _die(app, _refuse)
            held = await tasks.load_task(task_id)
            assert held.worker_id is not None and charge.calls == 1
            [attempt] = await session_wakes.session_wake_attempts(app, _wake_id_of(held))
            assert attempt.worker_id == "worker-1"
            await _cancel_fenced(app, task_id)
            await _expire(backend, clock)
            if postgres:
                await asyncio.sleep(claim_lease - _DYING_LEASE_SECONDS)
            assert not (await _next_worker(app)).attention_required
            await _assert_cancelled_and_interrupted(app, task_id)
            # The dead worker's wake attempt is settled with its owner, so nothing waits
            # on a person, and nothing runs again.
            assert (await tasks.load_task(attempt.id)).status is TaskStatus.CANCELLED
            assert not (await _next_worker(app)).attention_required
            assert len(provider.requests) == 2 and charge.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancelling_a_task_whose_worker_died_before_handing_off_its_external_wait_settles_it(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        async with clocked(backend, tmp_path, request) as (clock, app, _, hosting, scope, provider):
            job = f"job-wait-{uuid4().hex[:8]}"
            correlation, registration = await _reserve(hosting, scope, key=job)
            await hosting.register(registration, context=WORKER)
            await app.task_store.create_task(TaskCreate(task_id=job, type=_JOB))
            handler = _job(hosting, registration)

            async def parks_and_dies_before_the_handoff(app, task, worker_id):
                assert await handler(app, task, worker_id) is TaskHandlerOutcome.SESSION_INTERRUPTED
                raise _WorkerDied

            await _die(app, parks_and_dies_before_the_handoff)
            assert (await app.task_store.load_task(job)).interrupted_handoff_id is None
            await _cancel_fenced(app, job)
            await _expire(backend, clock)
            await _next_worker(app)
            await _assert_cancelled_and_interrupted(app, job)
            # The task ended, so its wait is stopped and never continues the session.
            wait = await hosting.waits.inspect(correlation, context=WORKER)
            assert wait.outcome.kind == "cancelled" and not wait.pending_handoff
            await hosting.deliver(_delivery(correlation), context=WORKER)
            clock.advance(60)
            await _drain(app, handler)
            await _assert_cancelled_and_interrupted(app, job)
            assert len(provider.requests) == 1

    asyncio.run(scenario())


def _chain_stores(backend: str, tmp_path, request):
    """Wake and lease times on one injected clock; PostgreSQL leases follow its own."""

    clock = _Clock(datetime.now(UTC))
    if backend == "memory":
        return clock, InMemorySessionStore(), InMemoryTaskStore(clock=clock, ownership_clock=clock)
    if backend == "sqlite":
        return (
            clock,
            SQLiteSessionStore(tmp_path / "sessions.db"),
            SQLiteTaskStore(tmp_path / "tasks.db", clock=clock, ownership_clock=clock),
        )
    dsn = _own_postgres_database(request)
    return (
        clock,
        PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
        PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=clock),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_cancelled_task_in_a_subagent_chain_is_left_alone(backend, tmp_path, request) -> None:
    """The woken child ended, and the claim that would continue its parent died.

    Its parent's session waits on a subagent, so the task isn't settled
    automatically: it stays fenced and reported, and nothing runs.
    """

    async def scenario() -> None:
        clock, sessions, tasks = _chain_stores(backend, tmp_path, request)
        calls: list[ModelRequest] = []
        app = _owned_family(sessions, tasks, clock, calls)
        try:
            job, child, wake = await _parked_parent(app, f"job-chain-{uuid4().hex[:8]}")
            _due(clock)
            # The dead worker claimed the wake's attempt, then the task under it.
            attempt = await tasks.claim_task(
                "dead-worker",
                TaskQuery(type=session_wakes.SESSION_WAKE_TASK_TYPE),
                lease_seconds=_DYING_LEASE_SECONDS,
            )
            assert attempt is not None and attempt.input["wake_id"] == wake["input_id"]
            await tasks.mark_claimed_task_execution_started(
                attempt.id, "dead-worker", attempt.lease_expires_at
            )
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, job.interrupted_handoff_id
            )
            claimed = await tasks.claim_interrupted_task_continuation(
                f"dead-worker:continue:{wake['input_id']}",
                handoff_id=new_interrupted_task_continuation_handoff_id(),
                task_id=job.id,
                expected=TaskContinuationExpectation(
                    task_id=job.id,
                    session_id=job.session_id,
                    session_instance_id=job.session_instance_id,
                    prior_handoff_id=job.interrupted_handoff_id,
                    operation_sha256="0" * 64,
                    parked_wait=receipt.request.parked,
                ),
                lease_seconds=_DYING_LEASE_SECONDS,
            )
            assert claimed.task is not None
            await session_wakes._deliver(app, session_id=child.id, wake_id=wake["input_id"])
            assert (await sessions.load(child.id)).status is SessionStatus.COMPLETED
            assert (await sessions.load(job.session_id)).status is SessionStatus.INTERRUPTED
            model_calls = len(calls)

            parent_epoch = (await sessions.load(job.session_id)).run_epoch
            fenced = await _cancel_fenced(app, job.id)
            await _expire(backend, clock)
            for _ in range(2):
                assert (await _until_idle(app)).attention_required
                task = await tasks.load_task(job.id)
                assert task.status is TaskStatus.RUNNING and task.worker_id == fenced.worker_id
                assert (await tasks.load_task(attempt.id)).status is not TaskStatus.CANCELLED
                parent = await sessions.load(job.session_id)
                assert parent.status is SessionStatus.INTERRUPTED
                assert parent.run_epoch == parent_epoch
                assert len(calls) == model_calls
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _recover_page(app, worker_id: str = "recoverer"):
    """One pass of the expired attached-task recovery a task worker runs."""

    return await task_worker._recover_expired_interrupted_task_handoffs(
        app, app.task_store, worker_id=worker_id, after=None, limit=10, stop=None
    )


async def _mark_retry_cancellation(store, task_id: str, backend: str) -> None:
    """Give a held task the retry-series cancellation marker no supported API attaches."""

    reason = "retry_cancellation_requested"
    if backend == "memory":
        store._tasks[task_id] = store._tasks[task_id].model_copy(update={"status_reason": reason})
    elif backend == "sqlite":
        store._connection.execute(
            "UPDATE cayu_tasks SET status_reason = ? WHERE id = ?", (reason, task_id)
        )
        store._connection.commit()
    else:
        import psycopg

        with psycopg.connect(store._conninfo, autocommit=True) as connection:
            connection.execute(
                "UPDATE cayu_tasks SET status_reason = %s WHERE id = %s", (reason, task_id)
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_expired_owner_listing_includes_pending_cancellations_but_not_retry_ones(
    backend, tmp_path, request
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, _, _ = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            plain = await _died_mid_tool(app)
            cancelled = await _died_mid_tool(app)
            retrying = await _died_mid_tool(app)
            await _cancel_fenced(app, cancelled)
            await _mark_retry_cancellation(tasks, retrying, backend)
            await _expire(backend, clock)
            listed = await tasks.list_expired_interrupted_task_handoff_candidates()
            assert sorted(task.id for task in listed) == sorted([plain, cancelled])
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_cancellation_landing_after_the_listing_is_settled_by_the_next_pass(
    backend, tmp_path, request, monkeypatch
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, provider, charge = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            task_id = await _died_mid_tool(app)
            await _expire(backend, clock)
            listing = tasks.list_expired_interrupted_task_handoff_candidates

            async def listed_then_cancelled(**kwargs):
                candidates = await listing(**kwargs)
                await _cancel_fenced(app, task_id)
                return candidates

            with monkeypatch.context() as patch:
                patch.setattr(
                    tasks, "list_expired_interrupted_task_handoff_candidates", listed_then_cancelled
                )
                page = await _recover_page(app)
            # The handoff, built from the uncancelled listing, conflicts: nothing changes.
            assert page.recovered == 0 and not page.failed
            held = await tasks.load_task(task_id)
            assert held.status_reason == "cancellation_requested"
            assert held.interrupted_handoff_id is None and held.worker_id is not None
            assert (await _recover_page(app)).recovered == 1
            await _assert_cancelled_and_interrupted(app, task_id)
            assert len(provider.requests) == 1 and charge.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_recovering_workers_settle_a_cancelled_task_once(backend, tmp_path, request) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, _, _ = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            task_id = await _died_mid_tool(app)
            snapshot = await _cancel_fenced(app, task_id)
            await _expire(backend, clock)
            # Both listed the same expired claim and both found its session interrupted.
            settled = await asyncio.gather(
                *(
                    task_worker._settle_expired_owner_cancellation(
                        tasks,
                        snapshot,
                        actor=_declaring_worker_actor(worker),
                        evidence_id="session-interrupted:test:2",
                    )
                    for worker in ("recoverer-a", "recoverer-b")
                )
            )
            assert sorted(settled) == [False, True]
            assert (await tasks.load_task(task_id)).status is TaskStatus.CANCELLED

            # A worker whose pass listed a claim before another worker's pass settled
            # it counts nothing.
            other = await _died_mid_tool(app)
            stale = await _cancel_fenced(app, other)
            await _expire(backend, clock)
            assert (await _recover_page(app, "recoverer-a")).recovered == 1

            async def stale_listing(**kwargs):
                return [stale]

            tasks.list_expired_interrupted_task_handoff_candidates = stale_listing
            assert (await _recover_page(app, "recoverer-b")).recovered == 0
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_store_error_while_settling_is_retried_by_the_next_pass(
    backend, tmp_path, request, monkeypatch
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, _, _ = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            task_id = await _died_mid_tool(app)
            await _cancel_fenced(app, task_id)
            await _expire(backend, clock)

            async def unavailable(request):
                raise ConnectionError("the task store is briefly unavailable")

            with monkeypatch.context() as patch:
                patch.setattr(tasks, "reconcile_task_cancellation", unavailable)
                page = await _recover_page(app)
            assert page.recovered == 0 and page.failed == frozenset({task_id})
            assert (await tasks.load_task(task_id)).status is TaskStatus.RUNNING
            assert (await _recover_page(app)).recovered == 1
            await _assert_cancelled_and_interrupted(app, task_id)
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_cancelled_task_whose_session_needs_a_person_is_left_for_them(tmp_path, request) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores("memory", clock, tmp_path, request)
        app, provider, _ = _app(sessions, tasks, clock, _charge_then_answer)

        async def dies_during_the_model_call(model_request):
            raise _WorkerDied
            yield  # pragma: no cover - makes this an async generator

        try:
            task_id = f"job-model-{uuid4().hex[:8]}"
            await tasks.create_task(TaskCreate(task_id=task_id, type="job"))
            provider.stream = dies_during_the_model_call
            await _die(app, _run_job)
            del provider.stream
            await _cancel_fenced(app, task_id)
            await _expire("memory", clock)
            page = await _recover_page(app)
            assert page.recovered == 0 and page.manual_recovery_required
            assert (await _next_worker(app)).attention_required
            # Only a person can settle the model call; the task waits for them.
            held = await tasks.load_task(task_id)
            assert held.status is TaskStatus.RUNNING
            assert held.status_reason == "cancellation_requested"
            assert (await sessions.load(_session_id(task_id))).status is SessionStatus.RUNNING
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_continuation_cancelled_while_draining_settles_after_its_lease_expired(
    backend, tmp_path, request
) -> None:
    """A recovered continuation's owner-lost settlement names its claim generation."""

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, _, _ = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            task_id = await _died_mid_tool(app)
            await _expire(backend, clock)
            assert (await _recover_page(app)).recovered == 1
            claimed = await tasks.claim_interrupted_task_continuation(
                "continuing-worker",
                handoff_id=new_interrupted_task_continuation_handoff_id(),
                task_id=task_id,
                lease_seconds=_DYING_LEASE_SECONDS,
            )
            assert claimed.task is not None and claimed.task.interrupted_handoff_id is not None
            await _cancel_fenced(app, task_id)
            # The continuation drained, but its own lease expired before it settled.
            await _expire(backend, clock)
            await task_worker._settle_ordinary_task_cancellation_after_quiescence(
                tasks, task_id, "continuing-worker"
            )
            assert (await tasks.load_task(task_id)).status is TaskStatus.CANCELLED
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_lost_wake_attempt_is_left_alone_while_its_task_runs(tmp_path, request) -> None:
    async def scenario() -> None:
        clock, sessions, tasks = _chain_stores("memory", tmp_path, request)
        app = _owned_family(sessions, tasks, clock)
        try:
            job, _, wake = await _parked_parent(app, f"job-live-{uuid4().hex[:8]}")
            _due(clock)
            attempt = await tasks.claim_task(
                "slow-worker",
                TaskQuery(type=session_wakes.SESSION_WAKE_TASK_TYPE),
                lease_seconds=_DYING_LEASE_SECONDS,
            )
            await tasks.mark_claimed_task_execution_started(
                attempt.id, "slow-worker", attempt.lease_expires_at
            )
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                job.id, job.interrupted_handoff_id
            )
            claimed = await tasks.claim_interrupted_task_continuation(
                f"slow-worker:continue:{wake['input_id']}",
                handoff_id=new_interrupted_task_continuation_handoff_id(),
                task_id=job.id,
                expected=TaskContinuationExpectation(
                    task_id=job.id,
                    session_id=job.session_id,
                    session_instance_id=job.session_instance_id,
                    prior_handoff_id=job.interrupted_handoff_id,
                    operation_sha256="0" * 64,
                    parked_wait=receipt.request.parked,
                ),
                lease_seconds=600,
            )
            assert claimed.task is not None
            owner = await _cancel_fenced(app, job.id)
            # The attempt's own lease lapsed, but the claim on the task is still live.
            clock.advance(_DYING_LEASE_SECONDS + 1)
            assert not await task_worker._settle_expired_owner_cancellation(
                tasks,
                owner,
                actor=_declaring_worker_actor("recoverer"),
                evidence_id="session-interrupted:test:2",
            )
            # Reclamation fences the lapsed attempt, but its task hasn't ended.
            await _until_idle(app)
            assert (await tasks.load_task(job.id)).status is TaskStatus.RUNNING
            fenced = await tasks.load_task(attempt.id)
            assert fenced.status is TaskStatus.CLAIMED
            assert fenced.status_reason == "cancellation_requested"
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _sleep_then_die_in_the_woken_round(app, tasks, clock, *, attempt_lease: int) -> str:
    """Park a sleeping task, then let the wake's worker die inside the woken round's tool."""

    task_id = f"job-woken-{uuid4().hex[:8]}"
    await tasks.create_task(TaskCreate(task_id=task_id, type="job"))
    await run_task_worker_until_idle(
        app, tasks, _run_job, worker_id="first", query=TaskQuery(type="job")
    )
    clock.advance(31)
    with pytest.raises(_WorkerDied):
        await run_task_worker_until_idle(
            app,
            tasks,
            _refuse,
            worker_id="worker-1",
            lease_seconds=attempt_lease,
            recovered_interrupted_task_handler=_never_continued,
        )
    return task_id


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_wake_attempt_outliving_its_cancelled_task_is_settled_once_its_lease_expires(
    backend, tmp_path, request, monkeypatch
) -> None:
    """The dying worker's own lease is longer than its continuation claim on the task."""

    postgres = backend == "postgres"
    claim_lease, attempt_lease = (3, 6) if postgres else (1, 600)
    monkeypatch.setattr(session_wakes, "_CONTINUATION_LEASE_SECONDS", claim_lease)

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, provider, charge = _app(sessions, tasks, clock, _sleep_then_charge)
        try:
            task_id = await _sleep_then_die_in_the_woken_round(
                app, tasks, clock, attempt_lease=attempt_lease
            )
            held = await tasks.load_task(task_id)
            [attempt] = await session_wakes.session_wake_attempts(app, _wake_id_of(held))
            await _cancel_fenced(app, task_id)
            # The claim on the task expires; the attempt's own lease is still live.
            clock.advance(claim_lease + 1)
            if postgres:
                await asyncio.sleep(claim_lease + 0.5)
            await _next_worker(app)
            await _assert_cancelled_and_interrupted(app, task_id)
            assert (await tasks.load_task(attempt.id)).status is TaskStatus.CLAIMED
            # Once it expires too, a later pass settles it.
            clock.advance(attempt_lease + 1)
            if postgres:
                await asyncio.sleep(attempt_lease - claim_lease)
            assert not (await _next_worker(app)).attention_required
            assert (await tasks.load_task(attempt.id)).status is TaskStatus.CANCELLED
            assert len(provider.requests) == 2 and charge.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_store_error_settling_a_wake_attempt_is_retried_by_a_later_pass(
    tmp_path, request, monkeypatch
) -> None:
    monkeypatch.setattr(session_wakes, "_CONTINUATION_LEASE_SECONDS", _DYING_LEASE_SECONDS)

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores("memory", clock, tmp_path, request)
        app, _, _ = _app(sessions, tasks, clock, _sleep_then_charge)
        try:
            task_id = await _sleep_then_die_in_the_woken_round(
                app, tasks, clock, attempt_lease=_DYING_LEASE_SECONDS
            )
            held = await tasks.load_task(task_id)
            [attempt] = await session_wakes.session_wake_attempts(app, _wake_id_of(held))
            await _cancel_fenced(app, task_id)
            clock.advance(3600)
            reconcile = task_worker._reconcile_observed_abandoned_task_cancellation
            failures: list[str] = []

            async def attempt_fails_once(task_store, current, **kwargs):
                if current.id == attempt.id and not failures:
                    failures.append(current.id)
                    raise ConnectionError("the task store is briefly unavailable")
                return await reconcile(task_store, current, **kwargs)

            monkeypatch.setattr(
                task_worker, "_reconcile_observed_abandoned_task_cancellation", attempt_fails_once
            )
            await _next_worker(app)
            assert failures == [attempt.id]
            await _assert_cancelled_and_interrupted(app, task_id)
            assert (await tasks.load_task(attempt.id)).status is not TaskStatus.CANCELLED
            assert not (await _next_worker(app)).attention_required
            assert (await tasks.load_task(attempt.id)).status is TaskStatus.CANCELLED
        finally:
            await app.aclose()

    asyncio.run(scenario())


class _Sessions:
    def __init__(self, *sessions, checkpoints=None) -> None:
        self._sessions = {session.id: session for session in sessions}
        self._checkpoints = checkpoints or {}

    async def load(self, session_id):
        return self._sessions.get(session_id)

    async def load_checkpoint(self, session_id):
        return self._checkpoints.get(session_id, {})

    async def list_sessions(self, query):
        from types import SimpleNamespace

        children = [
            session
            for session in self._sessions.values()
            if session.parent_session_id == query.parent_session_id
        ]
        return SimpleNamespace(sessions=children[query.offset : query.offset + query.limit])


class _Tasks:
    def __init__(self, *tasks) -> None:
        self._tasks = tasks

    async def list_tasks(self, query):
        matching = [task for task in self._tasks if task.session_id == query.session_id]
        return matching[query.offset : query.offset + query.limit]


@pytest.mark.parametrize(
    ("session_status", "owner_statuses", "ended"),
    [
        (SessionStatus.INTERRUPTED, (TaskStatus.CANCELLED,), True),
        (SessionStatus.INTERRUPTED, (TaskStatus.COMPLETED, TaskStatus.CANCELLED), True),
        # The session still runs: whatever ended, something may still act on it.
        (SessionStatus.RUNNING, (TaskStatus.CANCELLED,), False),
        # Its task still runs and may be continued by this very attempt.
        (SessionStatus.INTERRUPTED, (TaskStatus.CANCELLED, TaskStatus.RUNNING), False),
        # No task owns it: the attempt continues the session itself.
        (SessionStatus.INTERRUPTED, (), False),
    ],
)
def test_a_wake_attempt_counts_as_orphaned_only_when_its_sessions_tasks_ended(
    session_status, owner_statuses, ended
) -> None:
    from types import SimpleNamespace

    session = SimpleNamespace(
        id="s-top",
        status=session_status,
        parent_session_id=None,
        metadata={},
        instance_id="instance",
        run_epoch=4,
    )
    owners = [SimpleNamespace(session_id="s-top", status=status) for status in owner_statuses]
    attempt = SimpleNamespace(input={"session_id": "s-top"})
    app = SimpleNamespace(session_store=_Sessions(session))
    evidence = asyncio.run(session_wakes._ended_owner_evidence(app, _Tasks(*owners), attempt))
    assert evidence == ("task-ended:instance:4" if ended else None)


def test_one_until_idle_run_settles_owners_and_wake_attempts_across_recovery_pages(
    tmp_path, request, monkeypatch
) -> None:
    """More expired owners than one recovery page, each held by a lost wake's claim."""

    monkeypatch.setattr(session_wakes, "_CONTINUATION_LEASE_SECONDS", _DYING_LEASE_SECONDS)
    monkeypatch.setattr(task_worker, "_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE", 1)

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores("memory", clock, tmp_path, request)
        app, _, charge = _app(sessions, tasks, clock, _sleep_then_charge)
        try:
            task_ids = [f"job-paged-{index}-{uuid4().hex[:6]}" for index in range(3)]
            for task_id in task_ids:
                await tasks.create_task(TaskCreate(task_id=task_id, type="job"))
            await run_task_worker_until_idle(
                app, tasks, _run_job, worker_id="first", query=TaskQuery(type="job")
            )
            clock.advance(31)
            # Each wake's worker dies inside its woken round, holding the task's claim.
            for _ in task_ids:
                await _die(app, _refuse)
            assert charge.calls == len(task_ids)
            attempts = []
            for task_id in task_ids:
                held = await tasks.load_task(task_id)
                [attempt] = await session_wakes.session_wake_attempts(app, _wake_id_of(held))
                attempts.append(attempt.id)
                await _cancel_fenced(app, task_id)
            clock.advance(3600)
            assert not (await _next_worker(app)).attention_required
            for task_id in task_ids:
                await _assert_cancelled_and_interrupted(app, task_id)
            for attempt_id in attempts:
                assert (await tasks.load_task(attempt_id)).status is TaskStatus.CANCELLED
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("statuses", "ended"),
    [
        # A history longer than one page, every task of it ended.
        ((TaskStatus.COMPLETED,) * 100 + (TaskStatus.CANCELLED,), True),
        ((TaskStatus.COMPLETED,) * 100, True),
        # A running task on a later page still owns the session.
        ((TaskStatus.COMPLETED,) * 150 + (TaskStatus.RUNNING,), False),
    ],
)
def test_every_page_of_a_sessions_task_history_is_read(statuses, ended) -> None:
    from types import SimpleNamespace

    session = SimpleNamespace(
        id="s-top",
        status=SessionStatus.INTERRUPTED,
        parent_session_id=None,
        metadata={},
        instance_id="instance",
        run_epoch=4,
    )
    owners = [SimpleNamespace(session_id="s-top", status=status) for status in statuses]
    attempt = SimpleNamespace(input={"session_id": "s-top"})
    app = SimpleNamespace(session_store=_Sessions(session))
    evidence = asyncio.run(session_wakes._ended_owner_evidence(app, _Tasks(*owners), attempt))
    assert evidence == ("task-ended:instance:4" if ended else None)


def _session(session_id: str, status: SessionStatus, *, parent: str | None = None):
    from types import SimpleNamespace

    return SimpleNamespace(
        id=session_id,
        status=status,
        parent_session_id=parent,
        metadata={} if parent is None else {"subagent": {"mode": "foreground"}},
        instance_id="instance",
        run_epoch=4,
    )


@pytest.mark.parametrize(
    ("sessions", "finished", "ended"),
    [
        # Its foreground subagent finished, interrupted or not: the chain is quiescent.
        (
            (
                _session("s-top", SessionStatus.INTERRUPTED),
                _session("s-child", SessionStatus.INTERRUPTED, parent="s-top"),
            ),
            {"s-child"},
            True,
        ),
        # Its foreground subagent hasn't finished (it waits on something, or runs).
        (
            (
                _session("s-top", SessionStatus.INTERRUPTED),
                _session("s-child", SessionStatus.INTERRUPTED, parent="s-top"),
            ),
            set(),
            False,
        ),
        # The attempt's session is itself a foreground subagent that hasn't finished.
        (
            (
                _session("s-top", SessionStatus.INTERRUPTED, parent="s-parent"),
                _session("s-parent", SessionStatus.INTERRUPTED),
            ),
            set(),
            False,
        ),
    ],
)
def test_a_wake_attempt_in_a_live_subagent_chain_is_not_counted_as_orphaned(
    sessions, finished, ended, monkeypatch
) -> None:
    from types import SimpleNamespace

    from cayu.runtime import _foreground_child_wait

    async def judged(store, child):
        return child.id in finished

    monkeypatch.setattr(_foreground_child_wait, "foreground_child_finished", judged)
    owner = SimpleNamespace(session_id="s-top", status=TaskStatus.CANCELLED)
    attempt = SimpleNamespace(input={"session_id": "s-top"})
    app = SimpleNamespace(session_store=_Sessions(*sessions))
    evidence = asyncio.run(session_wakes._ended_owner_evidence(app, _Tasks(owner), attempt))
    assert evidence == ("task-ended:instance:4" if ended else None)


class _Note(Tool):
    """A harmless tool a subagent calls before its step limit stops it."""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="note",
            description="Write a note.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.NONE,
            max_terminal_payload_bytes=4096,
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="noted")


def _first_user_text(request: ModelRequest) -> str:
    return next(
        part.text
        for message in request.messages
        if message.role == "user"
        for part in message.content
        if hasattr(part, "text")
    )


def _call_tool(call_id: str, name: str, arguments: dict) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _delegating(subagent: str, *, then_sleep: bool):
    """The task's agent first delegates to a subagent that finishes, then charges.

    ``then_sleep`` puts a sleep between the subagent and the charge.
    """

    def respond(request: ModelRequest):
        results = len(_tool_results(request))
        user = _first_user_text(request)
        if user == "note it":
            return _call_tool("call_note", "note", {})
        if user == "finish":
            return [ModelStreamEvent.text_delta("finished"), ModelStreamEvent.completed()]
        steps = ["spawn", *(["sleep"] if then_sleep else []), "charge"]
        if results >= len(steps):
            return [ModelStreamEvent.text_delta("charged"), ModelStreamEvent.completed()]
        step = steps[results]
        if step == "spawn":
            task = "note it" if subagent == "limited" else "finish"
            return _call_tool("spawn", "subagent", {"agent": subagent, "task": task})
        if step == "sleep":
            return _call_tool("call_sleep", "sleep_until", {"duration_seconds": 30})
        return _call_tool("call_charge", "charge", {})

    return respond


def _delegating_app(sessions, tasks, clock, respond):
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    provider = _StableScriptedProvider(response_factory=respond)
    app.register_provider(provider, default=True)
    charge = _Charge()
    app.register_agent(
        AgentSpec(name="charger", model="scripted-model"),
        tools=[
            SleepUntilTool(),
            charge,
            SubagentTool(
                app,
                agents={
                    # Stopped by its step limit after one tool call: it ends interrupted.
                    "limited": SubagentSpec(agent_name="helper", max_steps=1),
                    "done": SubagentSpec(agent_name="helper"),
                },
                execution_profile_identity=_identity("cancel-orphaned-delegating"),
            ),
        ],
    )
    app.register_agent(AgentSpec(name="helper", model="scripted-model"), tools=[_Note()])
    return app, provider, charge


async def _child_status(sessions, parent_id: str) -> SessionStatus:
    from cayu.sessions.queries import SessionQuery

    [child] = (await sessions.list_sessions(SessionQuery(parent_session_id=parent_id))).sessions
    return child.status


@pytest.mark.parametrize(
    ("subagent", "child_status"),
    [("limited", SessionStatus.INTERRUPTED), ("done", SessionStatus.COMPLETED)],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_task_whose_session_had_a_finished_subagent_is_still_settled(
    backend, subagent, child_status, tmp_path, request
) -> None:
    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, _, charge = _delegating_app(
            sessions, tasks, clock, _delegating(subagent, then_sleep=False)
        )
        try:
            task_id = await _died_mid_tool(app)
            # The subagent finished and the session went on to the charge it died in.
            assert await _child_status(sessions, _session_id(task_id)) is child_status
            assert charge.calls == 1
            await _cancel_fenced(app, task_id)
            await _expire(backend, clock)
            assert not (await _next_worker(app)).attention_required
            await _assert_cancelled_and_interrupted(app, task_id)
            assert charge.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_sleeping_task_whose_session_had_a_finished_subagent_has_its_wake_attempt_settled(
    backend, tmp_path, request, monkeypatch
) -> None:
    monkeypatch.setattr(session_wakes, "_CONTINUATION_LEASE_SECONDS", _DYING_LEASE_SECONDS)

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, _, charge = _delegating_app(
            sessions, tasks, clock, _delegating("limited", then_sleep=True)
        )
        try:
            task_id = await _sleep_then_die_in_the_woken_round(
                app, tasks, clock, attempt_lease=_DYING_LEASE_SECONDS
            )
            session_id = _session_id(task_id)
            assert await _child_status(sessions, session_id) is SessionStatus.INTERRUPTED
            held = await tasks.load_task(task_id)
            [attempt] = await session_wakes.session_wake_attempts(app, _wake_id_of(held))
            await _cancel_fenced(app, task_id)
            await _expire(backend, clock)
            assert not (await _next_worker(app)).attention_required
            await _assert_cancelled_and_interrupted(app, task_id)
            assert (await tasks.load_task(attempt.id)).status is TaskStatus.CANCELLED
            assert charge.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())
