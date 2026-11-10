"""A continuation callback owns its own task-group execution, keyed by its claim generation.

A worker name and a start time do not identify an execution: a worker may
continue an interrupted session under its original name, and the continuation
keeps the task's original start time. The claim generation does change, so the
group barrier records the continuation as a new, unsettled execution and keeps
the finalizer waiting until that callback returns.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from tests.core.task_invocation_fixtures import stored_session_invocation
from tests.core.test_task_group_quiescence import store as store
from tests.core.test_task_groups import anyio_backend as anyio_backend

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.recovery import RecoveryPlanItem
from cayu.sessions.requests import RunRequest, run_request_with_task_invocation
from cayu.tasks import (
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupFinalizerStatus,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskGroupQuiescenceStatus,
    TaskInvocationSnapshot,
    TaskQuery,
    TaskStatus,
)
from cayu.tasks._execution_settlement import (
    TaskExecutionSettlementPending,
    has_task_execution_settlement_pending,
)
from cayu.tasks.worker import TaskHandlerOutcome, fail_managed_task, run_task_worker

pytestmark = pytest.mark.anyio

_WAIT_S = 5
_FAST_POLL_S = 0.01
_REPEATED_SCANS = 4


async def _interrupted_member(store) -> CayuApp:
    """A first-success group whose member's first callback paused its session."""
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    await app.create_task_group(
        TaskGroupCreate(
            group_id="continuation-race",
            graph=TaskGraphCreate(
                graph_id="continuation-graph",
                nodes=(
                    TaskGraphNode(task=TaskCreate(task_id="continued", type="job")),
                    TaskGraphNode(task=TaskCreate(task_id="winner", type="winner")),
                    TaskGraphNode(task=TaskCreate(task_id="finalizer", type="finalizer")),
                ),
            ),
            member_task_ids=("continued", "winner"),
            policy=TaskGroupPolicy(kind="first_success"),
            quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
            finalizer_task_id="finalizer",
        )
    )
    await _interrupt(app, store, "continued")
    return app


async def _interrupt(app: CayuApp, store, task_id: str) -> None:
    """Run the task's first callback, which pauses its session and hands it off."""
    created = await store.load_task(task_id)
    session_id = f"{task_id}-session"
    await app.session_store.create(
        run_request_with_task_invocation(
            RunRequest(
                agent_name="worker-agent",
                session_id=session_id,
                task_id=created.id,
                messages=[Message.text("user", "pause")],
            ),
            TaskInvocationSnapshot(
                id=created.id, session_id=created.session_id, invocation=created.invocation
            ),
        ),
        identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
    )
    await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    async def first_handler(_app, task, worker_id):
        await store.attach_task(
            task.id,
            session_id=session_id,
            session_invocation=await stored_session_invocation(app.session_store, session_id),
            worker_id=worker_id,
            claim_generation=task.claim_generation,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    handled = await run_task_worker(
        app,
        store,
        first_handler,
        worker_id="original",
        query=TaskQuery(type="job"),
        max_tasks=1,
        reclaim=False,
        recover_interrupted_handoffs=False,
    )
    assert handled == 1


async def test_same_worker_continuation_retains_its_group_barrier_until_return(store):
    app = await _interrupted_member(store)
    first = (await app.load_task_group("continuation-race")).quiescence.executions[0]
    assert first.settled_at is not None
    release = asyncio.Event()
    callback_running = asyncio.Event()
    observed = []

    async def continuation(_app, task, worker_id):
        observed.append(task.claim_generation)
        # The callback settles its task before its remaining cleanup stops; the
        # group must keep that callback's positive obligation.
        await fail_managed_task(store, task, worker_id, {"code": "no_result"})
        callback_running.set()
        await release.wait()

    async def no_fresh_work(*_args):
        pytest.fail("Only the interrupted continuation should be dispatched.")

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            no_fresh_work,
            worker_id="original",
            query=TaskQuery(type="job"),
            recovered_interrupted_task_handler=continuation,
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
        )
    )
    try:
        await asyncio.wait_for(callback_running.wait(), _WAIT_S)
        assert (await store.load_task("continued")).status is TaskStatus.FAILED
        await store.complete_task("winner", {"ok": True})
        draining = await app.reconcile_task_group("continuation-race")
        assert (
            draining.quiescence.status,
            draining.quiescence.finalizer_status,
            (await store.load_task("finalizer")).status,
        ) == (
            TaskGroupQuiescenceStatus.DRAINING,
            TaskGroupFinalizerStatus.WAITING,
            TaskStatus.WAITING_GROUP,
        )
        execution = draining.quiescence.executions[0]
        assert (execution.worker_id, execution.started_at) == (first.worker_id, first.started_at)
        assert execution.claim_generation == observed[0] > first.claim_generation
        assert execution.settled_at is None
        release.set()
        assert await asyncio.wait_for(worker, _WAIT_S) == 1
        settled = await app.reconcile_task_group("continuation-race")
        assert settled.quiescence.executions[0].settled_at is not None
        assert settled.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert settled.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), _WAIT_S)
        await app.aclose()


async def test_group_refused_continuation_skips_only_its_own_candidate(store):
    app = await _interrupted_member(store)
    # The winner decides the group, so the handed-off member becomes a fenced
    # loser that no claim may resume.
    await store.complete_task("winner", {"ok": True})
    await store.create_task(TaskCreate(task_id="unrelated", type="job"))
    await _interrupt(app, store, "unrelated")
    continued = []

    async def continuation(_app, task, worker_id):
        continued.append(task.id)
        await fail_managed_task(store, task, worker_id, {"code": "no_result"})

    async def no_fresh_work(*_args):
        pytest.fail("Only interrupted continuations should be dispatched.")

    try:
        handled = await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                no_fresh_work,
                worker_id="original",
                query=TaskQuery(type="job"),
                recovered_interrupted_task_handler=continuation,
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
            ),
            _WAIT_S,
        )
        assert (handled, continued) == (1, ["unrelated"])
        refused = await store.load_task("continued")
        assert (refused.status, refused.worker_id) == (TaskStatus.RUNNING, None)
        assert (await store.load_task("unrelated")).status is TaskStatus.FAILED
    finally:
        await app.aclose()


async def test_group_waiting_continuation_is_reported_as_waiting_once(store, caplog, monkeypatch):
    app = await _interrupted_member(store)
    await store.complete_task("winner", {"ok": True})
    page = await store.claim_interrupted_task_continuation(
        "original", TaskQuery(type="job"), handoff_id="probe-handoff"
    )
    assert (page.task, page.rejected_candidates, page.group_waiting_task_ids) == (
        None,
        0,
        ("continued",),
    )
    scans = 0
    scanned_enough = asyncio.Event()
    from cayu.tasks import worker as task_worker

    scan = task_worker._run_recovered_interrupted_task_handler

    async def counted_scan(*args, **kwargs):
        nonlocal scans
        result = await scan(*args, **kwargs)
        scans += 1
        if scans >= _REPEATED_SCANS:
            scanned_enough.set()
        return result

    monkeypatch.setattr(task_worker, "_run_recovered_interrupted_task_handler", counted_scan)
    stop = asyncio.Event()

    async def no_work(*_args):
        pytest.fail("A group-waiting continuation must not be dispatched.")

    caplog.set_level("INFO", logger="cayu.tasks.worker")
    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            no_work,
            worker_id="original",
            query=TaskQuery(type="job"),
            recovered_interrupted_task_handler=no_work,
            poll_interval_s=_FAST_POLL_S,
            continuation_poll_interval_s=_FAST_POLL_S,
            reclaim=False,
            recover_interrupted_handoffs=False,
            stop=stop,
        )
    )
    try:
        await asyncio.wait_for(scanned_enough.wait(), _WAIT_S)
    finally:
        stop.set()
        assert await asyncio.wait_for(worker, _WAIT_S) == 0
        await app.aclose()
    messages = [record.getMessage() for record in caplog.records]
    assert not [message for message in messages if "invalid authority" in message]
    assert [message for message in messages if "earlier execution to settle" in message] == [
        "Interrupted-task continuation is waiting for its group's earlier execution to "
        "settle: task_id=continued"
    ]


async def test_continuation_cancelled_before_dispatch_settles_its_execution(store, monkeypatch):
    app = await _interrupted_member(store)
    first = (await app.load_task_group("continuation-race")).quiescence.executions[0]
    from cayu.tasks import worker as task_worker

    renew = task_worker._renew_task_claim_before_dispatch

    async def decide_before_dispatch(*args, **kwargs):
        # The winner completes after the continuation is claimed and before its
        # callback is dispatched, so the group cancels the continuation.
        await store.complete_task("winner", {"ok": True})
        return await renew(*args, **kwargs)

    monkeypatch.setattr(task_worker, "_renew_task_claim_before_dispatch", decide_before_dispatch)

    async def not_dispatched(*_args):
        pytest.fail("A cancelled continuation must not be dispatched.")

    try:
        handled = await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                not_dispatched,
                worker_id="original",
                query=TaskQuery(type="job"),
                recovered_interrupted_task_handler=not_dispatched,
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
            ),
            _WAIT_S,
        )
        assert handled == 1
        assert (await store.load_task("continued")).status is TaskStatus.CANCELLED
        settled = await app.reconcile_task_group("continuation-race")
        execution = settled.quiescence.executions[0]
        assert execution.claim_generation > first.claim_generation
        assert execution.settled_at is not None
        assert settled.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert settled.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
    finally:
        await app.aclose()


async def test_continuation_failing_before_dispatch_settles_its_execution(store, monkeypatch):
    app = await _interrupted_member(store)
    from cayu.tasks import worker as task_worker

    async def read_fails(*_args, **_kwargs):
        # Any store read before dispatch, such as the event budget, may fail.
        raise ConnectionError("store unavailable before dispatch")

    monkeypatch.setattr(task_worker, "_renew_task_claim_before_dispatch", read_fails)

    async def not_dispatched(*_args):
        pytest.fail("A continuation that failed before dispatch must not be dispatched.")

    try:
        with contextlib.suppress(ConnectionError):
            await asyncio.wait_for(
                run_task_worker(
                    app,
                    store,
                    not_dispatched,
                    worker_id="original",
                    query=TaskQuery(type="job"),
                    recovered_interrupted_task_handler=not_dispatched,
                    max_tasks=1,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                ),
                _WAIT_S,
            )
        snapshot = await app.load_task_group("continuation-race")
        assert snapshot.quiescence.executions[0].settled_at is not None
        continued = await store.load_task("continued")
        assert snapshot.quiescence.executions[0].claim_generation == continued.claim_generation
    finally:
        await app.aclose()


async def test_recovery_plan_continuation_settles_its_group_execution(store):
    app = await _interrupted_member(store)
    first = (await app.load_task_group("continuation-race")).quiescence.executions[0]
    session_id = "continued-session"
    await app.session_store.update_status(session_id, SessionStatus.FAILED)
    task = await store.load_task("continued")
    session = await app.session_store.load(session_id)
    assert task is not None and session is not None
    try:
        # An operator plan marks the handed-off member's model operation failed:
        # it claims the continuation itself and terminalizes the task.
        await app._recovery_plan_coordinator._fail_recoverable_task_after_model_disposition(
            task=task,
            session=session,
            item=RecoveryPlanItem.model_construct(item_id="model-failure-item"),
            execution_id="model-failure-execution",
        )
        assert (await store.load_task("continued")).status is TaskStatus.FAILED
        await store.complete_task("winner", {"ok": True})
        settled = await app.reconcile_task_group("continuation-race")
        execution = settled.quiescence.executions[0]
        assert execution.claim_generation > first.claim_generation
        assert execution.settled_at is not None
        assert settled.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert settled.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
    finally:
        await app.aclose()


async def test_recovery_plan_keeps_a_pending_acknowledgement_retryable(store, monkeypatch):
    app = await _interrupted_member(store)
    session_id = "continued-session"
    await app.session_store.update_status(session_id, SessionStatus.FAILED)
    task = await store.load_task("continued")
    session = await app.session_store.load(session_id)
    assert task is not None and session is not None

    async def unavailable(_store, _claim):
        raise ConnectionError("group acknowledgement storage unavailable")

    _replace_group_settlement(monkeypatch, unavailable)
    try:
        with pytest.raises(TaskExecutionSettlementPending) as pending:
            await app._recovery_plan_coordinator._fail_recoverable_task_after_model_disposition(
                task=task,
                session=session,
                item=RecoveryPlanItem.model_construct(item_id="model-failure-item"),
                execution_id="model-failure-execution",
            )
        assert (await store.load_task("continued")).status is TaskStatus.FAILED
        # Storage recovers; the retained handle finishes the acknowledgement.
        monkeypatch.undo()
        await pending.value.settlement.retry()
        await store.complete_task("winner", {"ok": True})
        settled = await app.reconcile_task_group("continuation-race")
        assert settled.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT
        assert settled.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
    finally:
        await app.aclose()


async def test_cancelled_acknowledgement_stays_a_cancellation_with_its_handle(monkeypatch):
    from cayu.tasks import InMemoryTaskStore
    from cayu.tasks._task_run_settlement import continuation_execution

    store = InMemoryTaskStore()
    await store.create_task(TaskCreate(task_id="claimed", type="job"))
    claim = await store.claim_task("worker", TaskQuery(type="job"))
    assert claim is not None
    claim = await store.mark_claimed_task_execution_started(
        claim.id, "worker", claim_generation=claim.claim_generation
    )
    acknowledging = asyncio.Event()

    async def slow_acknowledgement(_store, _claim):
        acknowledging.set()
        await asyncio.Event().wait()

    _replace_group_settlement(monkeypatch, slow_acknowledgement)

    async def work_then_acknowledge():
        async with continuation_execution(store, claim):
            pass

    owner = asyncio.create_task(work_then_acknowledge())
    await asyncio.wait_for(acknowledging.wait(), _WAIT_S)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await owner
    assert has_task_execution_settlement_pending(cancelled.value)


def _replace_group_settlement(monkeypatch, replacement) -> None:
    """Replace each store's group settlement step, below its proven mutation boundary."""
    from cayu.storage import _postgres_task_groups, _sqlite_task_groups
    from cayu.tasks import _memory_groups

    for module in (_memory_groups, _sqlite_task_groups, _postgres_task_groups):
        monkeypatch.setattr(module, "settle_execution", replacement)
