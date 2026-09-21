"""Exact callback/nondispatch proof survives failed group acknowledgements."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal

import pytest
from tests.core.test_dispatch import _SecretFreeDispatchRuntime, _test_dispatch_envelope
from tests.core.test_task_group_quiescence import store as store

from cayu import (
    CayuApp,
    TaskCreate,
    TaskExecutionSettlementPending,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryPolicy,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.invocation import TaskExecutionSource
from cayu.tasks.base import (
    TaskQuery,
    TaskStatus,
    task_create_with_runtime_invocation,
)
from cayu.tasks.dispatch import (
    DispatchRequest,
    DispatchStatus,
    TaskStoreDispatcher,
    _queued_dispatch_persisted_envelope,
    _queued_dispatch_task_id,
)
from cayu.tasks.worker import run_task_worker
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def create_group(app, member):
    await app.create_task_group(
        TaskGroupCreate(
            group_id="owned",
            graph=TaskGraphCreate(
                graph_id="owned-graph",
                nodes=tuple(
                    TaskGraphNode(task=task)
                    for task in (
                        TaskCreate(task_id="winner", type="winner"),
                        member,
                        TaskCreate(task_id="finalizer", type="finalizer"),
                    )
                ),
            ),
            member_task_ids=("winner", member.task_id),
            policy=TaskGroupPolicy(kind="first_success"),
            quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
            finalizer_task_id="finalizer",
        )
    )


def fail_acknowledgements(monkeypatch, store, *, failures, commit):
    original = store._settle_task_group_execution
    calls = []

    async def settle(self, task):
        calls.append((task.id, task.worker_id, task.started_at))
        if commit or len(calls) > failures:
            await original(task)
        if len(calls) <= failures:
            raise RuntimeError("secret-settlement-canary")

    monkeypatch.setattr(type(store), "_settle_task_group_execution", settle)
    return calls


async def assert_settled(app, identity):
    snapshot = await app.load_task_group("owned")
    execution = next(x for x in snapshot.quiescence.executions if x.task_id == identity)
    assert execution.settled_at is not None


async def retry_until_settled(settlement):
    # Observation timeout is not a database completion deadline. Join the same
    # owner until storage finishes, with a finite outer test/cleanup deadline.
    async with asyncio.timeout(10):
        while True:
            try:
                return await settlement.retry()
            except TaskExecutionSettlementPending as pending:
                assert pending.settlement is settlement
                await asyncio.sleep(0.01)


@pytest.mark.parametrize("commit", [False, True])
async def test_exact_settlement_retry_survives_permitted_task_deletion(store, monkeypatch, commit):
    from tests.core.task_invocation_fixtures import task_backed_session_invocation
    from tests.core.test_task_group_quiescence import _postgres_address

    from cayu.storage.postgres import PostgresTaskStore
    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.tasks.base import TaskSessionClosureClaim
    from cayu.tasks.graphs import TaskGraphConflict
    from cayu.tasks.groups import TaskGroupConflict, TaskGroupEventType

    app = CayuApp(task_store=store, enable_logging=False)
    await create_group(app, TaskCreate(task_id="member", type="member"))
    original = type(store)._settle_task_group_execution
    calls = fail_acknowledgements(monkeypatch, store, failures=3, commit=commit)
    authorities = []

    async def handler(_app, task, _worker):
        authorities.append(task.model_copy(deep=True))
        await store.attach_task(
            task.id,
            session_id="member-session",
            session_invocation=await task_backed_session_invocation(
                store, task.id, "member-session"
            ),
            worker_id=_worker,
            lease_expires_at=task.lease_expires_at,
        )
        await store.complete_task(
            task.id, {}, worker_id=_worker, lease_expires_at=task.lease_expires_at
        )

    with pytest.raises(TaskExecutionSettlementPending) as pending:
        await asyncio.wait_for(
            run_task_worker(
                app, store, handler, worker_id="owner", query=TaskQuery(type="member"), max_tasks=1
            ),
            10,
        )
    assert len(authorities) == 1 and len(calls) == 3
    authority = authorities[0]
    assert authority.started_at is not None
    assert (await store.load_task("member")).status is TaskStatus.COMPLETED
    await store.complete_task("finalizer", {})
    before = await app.load_task_group("owned")
    events = await app.list_task_group_events("owned")
    graph_events = await store.list_task_graph_events("owned-graph")
    assert sum(event.type is TaskGroupEventType.FINALIZER_RELEASED for event in events) == 1
    closure = TaskSessionClosureClaim(
        session_id="member-session", plan_id="a" * 64, task_ids=("member",)
    )
    if not commit:
        with pytest.raises(TaskGraphConflict, match="retains its execution evidence"):
            await store.claim_session_closure(closure)
        with pytest.raises(TaskGraphConflict, match="retains its execution evidence"):
            await store.delete_session_tasks("member-session", task_ids=("member",), policy=None)
        assert await store.load_task("member") is not None
        assert await app.load_task_group("owned") == before
        await pending.value.settlement.retry()
        before = await app.load_task_group("owned")
        assert await app.list_task_group_events("owned") == events
        assert await store.list_task_graph_events("owned-graph") == graph_events
    await store.claim_session_closure(closure)
    await store.delete_session_tasks("member-session", task_ids=("member",), policy=None)
    assert await store.load_task("member") is None
    assert await pending.value.settlement.retry() is None
    assert await pending.value.settlement.retry() is None
    assert len(calls) == 4 and len(authorities) == 1
    monkeypatch.setattr(type(store), "_settle_task_group_execution", original)

    async def assert_replay(owner):
        for update in (
            {"worker_id": "other-owner"},
            {"started_at": authority.started_at + timedelta(microseconds=1)},
        ):
            with pytest.raises(TaskGroupConflict, match="retained group owner"):
                await owner._settle_task_group_execution(authority.model_copy(update=update))
        await owner._settle_task_group_execution(authority)
        assert await owner.load_task_group("owned") == before
        assert await owner.list_task_group_events("owned") == events
        assert await owner.list_task_graph_events("owned-graph") == graph_events
        assert await owner.load_task("member") is None
        assert await owner.claim_task("late", TaskQuery(type="finalizer")) is None

    await assert_replay(store)
    if isinstance(store, SQLiteTaskStore):
        address = store.path
        await store.close()
        reopened = SQLiteTaskStore(address)
    elif isinstance(store, PostgresTaskStore):
        address = _postgres_address(store)
        await store.close()
        reopened = PostgresTaskStore(address)
    else:
        return
    try:
        await assert_replay(reopened)
    finally:
        await reopened.close()


@pytest.mark.parametrize("failures", [1, 3])
@pytest.mark.parametrize("commit", [False, True])
async def test_worker_keeps_retry_report_separate_from_acknowledgement(
    store, monkeypatch, failures, commit, capsys, caplog
):
    app = CayuApp(
        task_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor("secret-settlement-canary"),
    )
    await create_group(
        app,
        TaskCreate(
            task_id="member",
            type="member",
            retry_policy=TaskRetryPolicy(max_attempts=3, initial_backoff_seconds=0),
            metadata={"execution_profile_fingerprint": "b" * 64, "effect_fingerprint": "c" * 64},
        ),
    )
    calls = fail_acknowledgements(monkeypatch, store, failures=failures, commit=commit)
    report = TaskRetryAttemptReport(
        idempotency_key="returned-report",
        disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        error={"code": "try_again"},
        token_count=17,
        estimated_cost=Decimal("0.03"),
    )
    dispatched = 0

    async def handler(_app, task, _worker):
        nonlocal dispatched
        dispatched += 1
        # A callback's mutable task view must not alter the captured authority.
        task.worker_id = "not-the-owner"
        return report

    async def run():
        return await run_task_worker(
            app, store, handler, worker_id="owner", query=TaskQuery(type="member"), max_tasks=1
        )

    if failures == 3:
        with pytest.raises(TaskExecutionSettlementPending) as raised:
            await run()
        pending = raised.value
        assert "secret-settlement-canary" not in str(pending.__cause__)
        # The caller can retain and retry this exact owner after the worker
        # exits. No callback or retry disposition is replayed by the handle.
        assert await pending.settlement.retry() == report
        assert await pending.settlement.retry() == report
    else:
        assert await run() == 1
    receipt = await store.load_task_retry_settlement("member", report.idempotency_key)
    assert receipt is not None and receipt.successor is not None
    assert receipt.task.retry_series.cumulative_tokens == 17
    assert receipt.task.retry_series.cumulative_estimated_cost == Decimal("0.03")
    assert dispatched == 1
    assert len(set(calls)) == 1 and calls[0][1] == "owner" and calls[0][2] is not None
    await assert_settled(app, "member")
    captured = capsys.readouterr()
    assert "secret-settlement-canary" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("failures", [0, 3])
async def test_worker_start_ack_deadline_has_proven_nondispatch_owner(store, monkeypatch, failures):
    app = CayuApp(task_store=store, enable_logging=False)
    await create_group(app, TaskCreate(task_id="member", type="member"))
    calls = fail_acknowledgements(monkeypatch, store, failures=failures, commit=False)
    original = store.mark_claimed_task_execution_started

    async def delayed_start(self, *args):
        task = await original(*args)
        await asyncio.sleep(0.72)
        return task

    monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", delayed_start)

    async def handler(*args):
        pytest.fail("Expired local dispatch authority must never invoke the callback")

    async def run():
        return await run_task_worker(
            app,
            store,
            handler,
            worker_id="owner",
            query=TaskQuery(type="member"),
            lease_seconds=1,
            max_tasks=1,
        )

    if failures:
        with pytest.raises(TaskExecutionSettlementPending) as raised:
            await run()
        assert (await store.load_task("member")).status is TaskStatus.CANCELLED
        assert await raised.value.settlement.retry() is None
    else:
        assert await run() == 1
    assert (await store.load_task("member")).status is TaskStatus.CANCELLED
    assert len(set(calls)) == 1
    await assert_settled(app, "member")
    await store.complete_task("winner", {})
    assert (await store.load_task("finalizer")).status is TaskStatus.PENDING


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "precommit",
        "acknowledgement",
        "read_failure",
        "cancel_read",
        "generic_read_failure",
        "generic_cancel_read",
    ],
)
async def test_grouped_busy_session_requeues_with_settled_execution(store, monkeypatch, fault):
    from tests.core.test_dispatch import FakeProvider, _batch, _configured_app

    from cayu.sessions.base import (
        InMemorySessionStore,
        ResumeRequest,
        RunRequest,
        SessionStatusConflict,
    )
    from cayu.tasks._execution_settlement import _failure_chain
    from cayu.tasks.groups import TaskGroupEventType

    entered, release = asyncio.Event(), asyncio.Event()
    read_started, read_release = asyncio.Event(), asyncio.Event()
    process = None

    class BusyProvider(FakeProvider):
        async def stream(self, request):
            if len(self.requests) == 1:
                entered.set()
                await release.wait()
            async for event in super().stream(request):
                yield event

    provider = BusyProvider([_batch("initial"), _batch("busy"), _batch("queued")])
    dispatcher = TaskStoreDispatcher(store)
    app = _configured_app(
        session_store=InMemorySessionStore(),
        task_store=store,
        dispatcher=dispatcher,
        provider=provider,
    )
    async for _ in app.run(
        RunRequest(
            agent_name="assistant", session_id="busy", messages=[Message.text("user", "initial")]
        )
    ):
        pass
    request = DispatchRequest(
        session_id="busy", dispatch_id="queued", messages=[Message.text("user", "queued")]
    )
    identity = _queued_dispatch_task_id(request, task_type="cayu.dispatch")
    envelope = await app._prepare_queued_dispatch(request, queue_task_id=identity)
    member = task_create_with_runtime_invocation(
        TaskCreate(
            task_id=identity,
            type="cayu.dispatch",
            input={"dispatch": _queued_dispatch_persisted_envelope(envelope)},
        ),
        source=TaskExecutionSource.TASK_DISPATCH,
        session_invocation=await app.session_invocation_for_dispatch("busy"),
    )
    await create_group(app, member)

    async def occupy():
        async for _ in app.resume(
            ResumeRequest(session_id="busy", messages=[Message.text("user", "occupy")])
        ):
            pass

    busy = asyncio.create_task(occupy())
    try:
        await asyncio.wait_for(entered.wait(), 15)
        generic = fault.startswith("generic_")
        if generic:
            dispatch = app._dispatch_queued
            refused = False

            async def refuse_before_admission(envelope):
                nonlocal refused
                if not refused:
                    refused = True
                    raise RuntimeError("Pre-admission failure")
                async for event in dispatch(envelope):
                    yield event

            monkeypatch.setattr(app, "_dispatch_queued", refuse_before_admission)
        lookup_fault = fault.removeprefix("generic_")
        calls = fail_acknowledgements(
            monkeypatch,
            store,
            failures=3 if fault in {"precommit", "acknowledgement"} else 0,
            commit=fault == "acknowledgement",
        )
        if lookup_fault in {"read_failure", "cancel_read"}:
            lookup = app._queued_dispatch_settlement_state
            read_count = 0

            async def delayed_lookup(envelope):
                nonlocal read_count
                read_count += 1
                if 2 <= read_count <= 4 and lookup_fault == "read_failure":
                    raise ConnectionError("Non-admission read is unavailable")
                if read_count == 2 and lookup_fault == "cancel_read":
                    read_started.set()
                    await read_release.wait()
                return await lookup(envelope)

            monkeypatch.setattr(app, "_queued_dispatch_settlement_state", delayed_lookup)
            process = asyncio.create_task(dispatcher.process_next(app, worker_id="owner"))
            if lookup_fault == "cancel_read":
                await asyncio.wait_for(read_started.wait(), 10)
                process.cancel()
                with pytest.raises(asyncio.CancelledError) as interrupted:
                    await process
                assert process.cancelled() and process.cancelling() == 1
                failure = interrupted.value
            else:
                with pytest.raises(RuntimeError if generic else SessionStatusConflict) as rejected:
                    await process
                failure = rejected.value
            pending = next(
                e for e in _failure_chain(failure) if isinstance(e, TaskExecutionSettlementPending)
            )
            assert not calls
            execution = (await app.load_task_group("owned")).quiescence.executions[0]
            assert execution.settled_at is None
            assert (await store.load_task(identity)).status is TaskStatus.CLAIMED
            read_release.set()
            await pending.settlement.retry()
            assert read_count == (2 if lookup_fault == "cancel_read" else 5)
            # The failed process did not requeue without proof. Finish ordinary
            # claim disposition only after its acknowledgement-only retry.
            claimed = await store.load_task(identity)
            await store.release_task(identity, "owner", lease_expires_at=claimed.lease_expires_at)
        elif fault == "none":
            handle = await dispatcher.process_next(app, worker_id="owner")
            assert handle.metadata["requeued"] is True
        else:
            with pytest.raises(TaskExecutionSettlementPending) as raised:
                await dispatcher.process_next(app, worker_id="owner")
            await raised.value.settlement.retry()
        assert (await store.load_task(identity)).status is TaskStatus.PENDING
        assert len(provider.requests) == 1
        await assert_settled(app, identity)
        first = (await app.load_task_group("owned")).quiescence.executions[0]
        release.set()
        await asyncio.wait_for(busy, 15)
        handle = await dispatcher.process_next(app, worker_id="owner")
        assert handle.status is DispatchStatus.COMPLETED
        assert len(provider.requests) == 3
        assert (await store.load_task(identity)).status is TaskStatus.COMPLETED
        await assert_settled(app, identity)
        second = (await app.load_task_group("owned")).quiescence.executions[0]
        assert second.started_at > first.started_at
        assert len(set(calls)) == 2
        events = await app.list_task_group_events("owned")
        assert sum(e.type is TaskGroupEventType.FINALIZER_RELEASED for e in events) == 1
        assert (await store.load_task("finalizer")).status is TaskStatus.PENDING
    finally:
        read_release.set()
        release.set()
        if process is not None:
            if not process.done():
                process.cancel()
            await asyncio.gather(process, return_exceptions=True)
        await asyncio.gather(busy, return_exceptions=True)


class Runtime(_SecretFreeDispatchRuntime):
    calls = 0

    async def dispatch_inline(self, request):
        self.calls += 1
        yield Event(type=EventType.SESSION_COMPLETED, session_id=request.session_id)


async def dispatch_group(app, runtime):
    request = DispatchRequest(
        session_id="session", dispatch_id="dispatch", messages=[Message.text("user", "continue")]
    )
    identity = _queued_dispatch_task_id(request, task_type="cayu.dispatch")
    envelope = _test_dispatch_envelope(request, queue_task_id=identity)
    member = task_create_with_runtime_invocation(
        TaskCreate(
            task_id=identity,
            type="cayu.dispatch",
            input={"dispatch": _queued_dispatch_persisted_envelope(envelope)},
        ),
        source=TaskExecutionSource.TASK_DISPATCH,
        session_invocation=await runtime.session_invocation_for_dispatch(request.session_id),
    )
    await create_group(app, member)
    return identity


async def test_session_conflict_after_admission_retains_unsettled_execution(store):
    from cayu.runtime.authority import SessionRunFenced
    from cayu.tasks.dispatch import _QueuedDispatchSettlement, _QueuedDispatchSettlementState

    class AdmittedRuntime(Runtime):
        released = False

        async def dispatch_inline(self, request):
            self.calls += 1
            raise SessionRunFenced("Already admitted execution lost its session fence")
            yield  # pragma: no cover - preserve the streaming interface

        async def _queued_dispatch_settlement_state(self, envelope):
            if self.calls:
                if self.released:
                    return _QueuedDispatchSettlement(
                        _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE,
                        terminal_status=DispatchStatus.COMPLETED,
                    )
                return _QueuedDispatchSettlement(
                    _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_PENDING
                )
            return await super()._queued_dispatch_settlement_state(envelope)

    app = CayuApp(task_store=store, enable_logging=False)
    runtime = AdmittedRuntime()
    identity = await dispatch_group(app, runtime)
    with pytest.raises(TaskExecutionSettlementPending) as pending:
        await TaskStoreDispatcher(store).process_next(runtime, worker_id="owner")
    task = await store.load_task(identity)
    assert task.status is TaskStatus.PENDING
    execution = (await app.load_task_group("owned")).quiescence.executions[0]
    assert execution.settled_at is None
    await store.complete_task("winner", {})
    assert (await store.load_task("finalizer")).status is TaskStatus.WAITING_GROUP
    with pytest.raises(TaskExecutionSettlementPending):
        await pending.value.settlement.retry()
    assert (await store.load_task("finalizer")).status is TaskStatus.WAITING_GROUP
    runtime.released = True
    await pending.value.settlement.retry()
    await app.reconcile_task_group("owned")
    assert (await store.load_task("finalizer")).status is TaskStatus.PENDING
    assert runtime.calls == 1


async def test_grouped_dispatch_recovers_missing_terminal_event_without_redispatch(store):
    from tests.core.test_dispatch import FakeProvider, _batch, _configured_app

    from cayu.sessions.base import InMemorySessionStore, RunRequest
    from cayu.tasks.groups import TaskGroupEventType

    class RejectTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        blocked_event_id = None

        async def append_event(self, session_id, event):
            if event.id == self.blocked_event_id:
                raise ConnectionError("Terminal publication unavailable")
            await super().append_event(session_id, event)

    sessions = RejectTerminalStore()
    provider = FakeProvider([_batch("initial"), _batch("queued")])
    dispatcher = TaskStoreDispatcher(store, recover_stalled_sessions_after_seconds=0)
    app = _configured_app(
        session_store=sessions, task_store=store, dispatcher=dispatcher, provider=provider
    )
    async for _ in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="publication",
            messages=[Message.text("user", "initial")],
        )
    ):
        pass
    request = DispatchRequest(
        session_id="publication", dispatch_id="queued", messages=[Message.text("user", "queued")]
    )
    identity = _queued_dispatch_task_id(request, task_type="cayu.dispatch")
    envelope = await app._prepare_queued_dispatch(request, queue_task_id=identity)
    await create_group(
        app,
        task_create_with_runtime_invocation(
            TaskCreate(
                task_id=identity,
                type="cayu.dispatch",
                input={"dispatch": _queued_dispatch_persisted_envelope(envelope)},
            ),
            source=TaskExecutionSource.TASK_DISPATCH,
            session_invocation=await app.session_invocation_for_dispatch("publication"),
        ),
    )
    sessions.blocked_event_id = envelope.terminal_event_id
    with pytest.raises(TaskExecutionSettlementPending) as pending:
        await dispatcher.process_next(app, worker_id="original")
    group = await app.load_task_group("owned")
    original = group.quiescence.executions[0]
    assert original.settled_at is None
    assert (await store.load_task("finalizer")).status is TaskStatus.WAITING_GROUP
    assert (await store.load_task(identity)).status is TaskStatus.PENDING

    sessions.blocked_event_id = None
    for worker in ("repair", "finish", "replay"):
        handle = await dispatcher.process_next(app, worker_id=worker)
        if handle is not None and handle.status is not DispatchStatus.SUBMITTED:
            break
    assert handle is not None and handle.status in {
        DispatchStatus.COMPLETED,
        DispatchStatus.FAILED,
        DispatchStatus.INTERRUPTED,
    }
    await pending.value.settlement.retry()
    await pending.value.settlement.retry()
    group = await app.load_task_group("owned")
    execution = group.quiescence.executions[0]
    assert execution.worker_id == original.worker_id
    assert execution.started_at == original.started_at
    assert execution.settled_at is not None
    if (await store.load_task("winner")).status is TaskStatus.PENDING:
        await store.complete_task("winner", {})
    assert (await store.load_task("finalizer")).status is TaskStatus.PENDING
    events = await app.list_task_group_events("owned")
    assert sum(e.type is TaskGroupEventType.FINALIZER_RELEASED for e in events) == 1
    assert len(provider.requests) == 2


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("commit", [False, True])
async def test_dispatch_worker_transfers_pending_ack_owner(store, monkeypatch, wrapped, commit):
    from cayu.tasks._execution_settlement import _failure_chain

    app = CayuApp(task_store=store, enable_logging=False)

    class LosingRuntime(Runtime):
        async def dispatch_inline(self, request):
            await store.complete_task("winner", {})
            async for event in super().dispatch_inline(request):
                yield event

    class WrappedDispatcher(TaskStoreDispatcher):
        async def process_next(self, runtime, *, worker_id):
            try:
                return await super().process_next(runtime, worker_id=worker_id)
            except TaskExecutionSettlementPending as pending:
                raise RuntimeError("Outer dispatch failure") from pending

    runtime = LosingRuntime()
    identity = await dispatch_group(app, runtime)
    calls = fail_acknowledgements(monkeypatch, store, failures=3, commit=commit)
    dispatcher = (WrappedDispatcher if wrapped else TaskStoreDispatcher)(store)
    with pytest.raises(RuntimeError) as raised:
        await asyncio.wait_for(
            dispatcher.run_worker(
                runtime,
                worker_id="owner",
                stop=asyncio.Event(),
                reconcile_terminal_receipts=False,
                reclaim_expired_leases=False,
            ),
            5,
        )
    pending = next(
        item
        for item in _failure_chain(raised.value)
        if isinstance(item, TaskExecutionSettlementPending)
    )
    assert type(raised.value) is (RuntimeError if wrapped else TaskExecutionSettlementPending)
    assert runtime.calls == 1 and len(calls) == 3
    if not commit:
        assert await store.claim_task("finalizer", TaskQuery(type="finalizer")) is None
    await pending.settlement.retry()
    await assert_settled(app, identity)
    assert runtime.calls == 1 and len(set(calls)) == 1
    assert await store.claim_task("finalizer", TaskQuery(type="finalizer")) is not None


@pytest.mark.parametrize("dispatch", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "precommit",
        "ack_loss",
        "cancel",
        "cancel_return",
        "repeated_cancel",
        "readback",
        "wrong_claim",
        "settlement",
    ],
)
async def test_failed_execution_entry_keeps_exact_nondispatch_owner(
    store, monkeypatch, dispatch, failure
):
    import cayu.tasks._execution_settlement as ownership

    monkeypatch.setattr(ownership, "_OBSERVATION_TIMEOUT_SECONDS", 0.05)
    app = CayuApp(task_store=store, enable_logging=False)
    runtime = Runtime()
    identity = await dispatch_group(app, runtime) if dispatch else "member"
    if not dispatch:
        await create_group(app, TaskCreate(task_id=identity, type="member"))
    mark, load = type(store).mark_claimed_task_execution_started, type(store).load_task
    settle = type(store)._settle_task_group_execution
    entered, release = asyncio.Event(), asyncio.Event()
    entry_calls, settlements, entry_cancellations = [], [], []
    repaired = False

    async def lost_entry(instance, *args):
        task = (
            await load(instance, args[0]) if failure == "precommit" else await mark(instance, *args)
        )
        entry_calls.append(task.model_copy(deep=True))
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            entry_cancellations.append(True)
            raise
        if failure == "cancel_return":
            return task
        raise OSError("Start acknowledgement lost.")

    async def readback(instance, task_id):
        task = await load(instance, task_id)
        if task_id == identity and entered.is_set() and not repaired:
            if failure == "readback":
                raise OSError("Entry readback unavailable.")
            if failure == "wrong_claim":
                return task.model_copy(update={"worker_id": "replacement"})
        return task

    async def acknowledge(instance, task):
        settlements.append(task.model_copy(deep=True))
        if failure == "settlement" and not repaired:
            raise OSError("Entry settlement unavailable.")
        await settle(instance, task)

    monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", lost_entry)
    monkeypatch.setattr(type(store), "load_task", readback)
    monkeypatch.setattr(type(store), "_settle_task_group_execution", acknowledge)

    async def handler(*args):
        pytest.fail("Failed entry must not dispatch the handler")

    owner = asyncio.create_task(
        TaskStoreDispatcher(store).process_next(runtime, worker_id="owner")
        if dispatch
        else run_task_worker(
            app, store, handler, worker_id="owner", query=TaskQuery(type="member"), max_tasks=1
        )
    )
    pending = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if failure != "precommit":
            await store.complete_task("winner", {})
        if failure in {"cancel", "cancel_return", "repeated_cancel"}:
            owner.cancel()
            if failure == "repeated_cancel":
                owner.cancel()
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(owner, 5)
            assert owner.cancelled() and owner.cancelling() == (
                2 if failure == "repeated_cancel" else 1
            )
        else:
            release.set()
            with pytest.raises(RuntimeError) as raised:
                await asyncio.wait_for(owner, 5)
        pending = next(
            item
            for item in ownership._failure_chain(raised.value)
            if isinstance(item, TaskExecutionSettlementPending)
        )
        assert len(entry_calls) == 1 and not runtime.calls and not settlements
        assert await store.claim_task("finalizer", TaskQuery(type="finalizer")) is None
        if not release.is_set():
            # A retry joins the unresolved original store call, never a new
            # start. Neither cancellation nor its observation timeout aborts it.
            with pytest.raises(TaskExecutionSettlementPending):
                await pending.settlement.retry()
            assert len(entry_calls) == 1 and not entry_cancellations
            release.set()
        if failure in {"readback", "wrong_claim", "settlement"}:
            with pytest.raises(TaskExecutionSettlementPending):
                await pending.settlement.retry()
            assert await store.claim_task("finalizer", TaskQuery(type="finalizer")) is None
            if failure != "settlement":
                assert not settlements
        repaired = True
        await retry_until_settled(pending.settlement)
        if failure == "precommit":
            assert not settlements and not runtime.calls
            await store.complete_task("winner", {})
        else:
            await assert_settled(app, identity)
        assert len(entry_calls) == 1 and not runtime.calls and not entry_cancellations
        assert bool(settlements) is (failure != "precommit")
        assert all(
            (item.id, item.worker_id, item.started_at)
            == (entry_calls[0].id, entry_calls[0].worker_id, entry_calls[0].started_at)
            for item in settlements
        )
        # Still-live leases remain a separate barrier after proven nondispatch;
        # settle ordinary cancellation through its existing positive evidence.
        from cayu.tasks.worker import _task_cancellation_terminalization_request

        current = await load(store, identity)
        request = _task_cancellation_terminalization_request(current, worker_id="owner")
        assert request is not None
        await store.terminalize_task(request)
        assert await store.claim_task("finalizer", TaskQuery(type="finalizer")) is not None
    finally:
        repaired = True
        release.set()
        if not owner.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        if pending is not None:
            await retry_until_settled(pending.settlement)


@pytest.mark.parametrize("nondispatch", [False, True])
@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("failures", [1, 3])
async def test_dispatcher_ack_owner_does_not_turn_success_into_callback_failure(
    store, monkeypatch, nondispatch, commit, failures
):
    app = CayuApp(task_store=store, enable_logging=False)
    runtime = Runtime()
    identity = await dispatch_group(app, runtime)
    calls = fail_acknowledgements(monkeypatch, store, failures=failures, commit=commit)
    dispatcher = TaskStoreDispatcher(store, lease_seconds=1 if nondispatch else 300)
    if nondispatch:
        original = store.mark_claimed_task_execution_started

        async def delayed_start(self, *args):
            task = await original(*args)
            await asyncio.sleep(0.72)
            return task

        monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", delayed_start)
    expected = DispatchStatus.CANCELLED if nondispatch else DispatchStatus.COMPLETED
    if failures == 3:
        with pytest.raises(TaskExecutionSettlementPending) as raised:
            await dispatcher.process_next(runtime, worker_id="owner")
        assert "secret-settlement-canary" not in str(raised.value.__cause__)
        assert await raised.value.settlement.retry() is expected
    else:
        result = await dispatcher.process_next(runtime, worker_id="owner")
        assert result.status is expected
    assert runtime.calls == (0 if nondispatch else 1)
    assert (await store.load_task(identity)).status is (
        TaskStatus.CANCELLED if nondispatch else TaskStatus.COMPLETED
    )
    assert len(set(calls)) == 1
    await assert_settled(app, identity)


@pytest.mark.parametrize("dispatch", [False, True])
async def test_real_owner_cancellation_keeps_pending_acknowledgement_retryable(
    store, monkeypatch, dispatch
):
    app = CayuApp(task_store=store, enable_logging=False)
    runtime = Runtime()
    if dispatch:
        identity = await dispatch_group(app, runtime)
    else:
        identity = "member"
        await create_group(app, TaskCreate(task_id=identity, type="member"))
    entered = asyncio.Event()
    release = asyncio.Event()
    original = store._settle_task_group_execution
    calls = []

    async def blocked_ack(self, task):
        calls.append((task.id, task.worker_id, task.started_at))
        if len(calls) <= 3:
            entered.set()
            await release.wait()
            raise RuntimeError("acknowledgement unavailable")
        await original(task)

    monkeypatch.setattr(type(store), "_settle_task_group_execution", blocked_ack)
    callback_calls = 0

    async def handler(*args):
        nonlocal callback_calls
        callback_calls += 1

    if dispatch:
        owner = asyncio.create_task(TaskStoreDispatcher(store).process_next(runtime, worker_id="w"))
    else:
        owner = asyncio.create_task(
            run_task_worker(
                app, store, handler, worker_id="w", query=TaskQuery(type="member"), max_tasks=1
            )
        )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        owner.cancel()
        assert owner.cancelling() == 1
        await asyncio.sleep(0.01)
        assert not owner.done()
        # Existing worker draining may temporarily consume the request; it
        # must restore it before propagation below.
        assert owner.cancelling() in {0, 1}
        release.set()
        # Use the ordinary handler shape, not an except* workaround.
        try:
            await asyncio.wait_for(owner, 3)
        except asyncio.CancelledError as cancellation:
            pending = cancellation.__cause__
            assert isinstance(pending, TaskExecutionSettlementPending)
        else:
            pytest.fail("Owner cancellation was lost")
        assert owner.cancelled() and owner.cancelling() == 1
        assert pending.settlement.result is (DispatchStatus.COMPLETED if dispatch else None)
        await pending.settlement.retry()
        assert len(calls) == 4 and len(set(calls)) == 1
        assert (runtime.calls if dispatch else callback_calls) == 1
        await assert_settled(app, identity)
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.parametrize("nondispatch", [False, True])
async def test_ordinary_store_without_group_capability_never_needs_ack_owner(
    store, monkeypatch, nondispatch
):
    monkeypatch.setattr(store, "supports_task_group_quiescence", False)
    app = CayuApp(task_store=store, enable_logging=False)
    await store.create_task(TaskCreate(task_id="ordinary", type="ordinary"))

    async def unexpected_ack(task):
        pytest.fail("Ordinary stores do not implement group settlement")

    monkeypatch.setattr(store, "_settle_task_group_execution", unexpected_ack)
    if nondispatch:
        original = store.mark_claimed_task_execution_started

        async def delayed_start(self, *args):
            task = await original(*args)
            await asyncio.sleep(0.72)
            return task

        monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", delayed_start)
    calls = 0

    async def handler(_app, task, worker):
        from cayu.tasks.worker import complete_managed_task

        nonlocal calls
        calls += 1
        await complete_managed_task(store, task, worker, {})

    assert (
        await run_task_worker(
            app,
            store,
            handler,
            worker_id="w",
            query=TaskQuery(type="ordinary"),
            lease_seconds=1 if nondispatch else 300,
            max_tasks=1,
        )
        == 1
    )
    assert calls == (0 if nondispatch else 1)
    assert (await store.load_task("ordinary")).status is (
        TaskStatus.CANCELLED if nondispatch else TaskStatus.COMPLETED
    )


@pytest.mark.parametrize("dispatch", [False, True])
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_timed_out_write_and_cancelled_retries_join_the_same_late_commit(
    store, monkeypatch, dispatch, cancel_owner
):
    import cayu.tasks._execution_settlement as ownership

    monkeypatch.setattr(ownership, "_OBSERVATION_TIMEOUT_SECONDS", 0.1)
    app = CayuApp(task_store=store, enable_logging=False)
    runtime = Runtime()
    if dispatch:
        identity = await dispatch_group(app, runtime)
    else:
        identity = "member"
        await create_group(app, TaskCreate(task_id=identity, type="member"))
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    original = store._settle_task_group_execution
    calls = []
    write_cancellations = []

    async def late_commit(self, task):
        calls.append((task.id, task.worker_id, task.started_at))
        entered.set()
        try:
            await release.wait()
            await original(task)
            completed.set()
        except asyncio.CancelledError:
            write_cancellations.append(True)
            raise

    monkeypatch.setattr(type(store), "_settle_task_group_execution", late_commit)
    handler_calls = 0

    async def handler(*args):
        nonlocal handler_calls
        handler_calls += 1

    owner = asyncio.create_task(
        TaskStoreDispatcher(store).process_next(runtime, worker_id="owner")
        if dispatch
        else run_task_worker(
            app, store, handler, worker_id="owner", query=TaskQuery(type="member"), max_tasks=1
        )
    )
    retry = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if cancel_owner:
            owner.cancel()
            assert owner.cancelling() == 1
            await asyncio.sleep(0)
            owner.cancel()
            try:
                await asyncio.wait_for(owner, 3)
            except asyncio.CancelledError as cancellation:
                pending = cancellation.__cause__
                assert isinstance(pending, TaskExecutionSettlementPending)
                # Repeated publication must neither duplicate the handle nor
                # introduce cycles into the cancellation cause graph.
                pending.settlement.finish(cancellation)
                assert cancellation.__cause__ is pending
            else:
                pytest.fail("A pending write swallowed owner cancellation")
            assert owner.cancelled() and owner.cancelling() == 2
        else:
            with pytest.raises(TaskExecutionSettlementPending) as raised:
                await asyncio.wait_for(owner, 3)
            pending = raised.value
        assert len(calls) == 1 and not completed.is_set()
        assert (runtime.calls if dispatch else handler_calls) == 1
        assert (await store.load_task(identity)).status in {
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
            TaskStatus.COMPLETED,
        }

        retry = asyncio.create_task(pending.settlement.retry())
        await asyncio.sleep(0)
        retry.cancel()
        assert retry.cancelling() == 1
        try:
            await retry
        except asyncio.CancelledError as cancellation:
            assert isinstance(cancellation.__cause__, TaskExecutionSettlementPending)
            assert cancellation.__cause__.settlement is pending.settlement
        else:
            pytest.fail("Retry observation swallowed cancellation")
        assert retry.cancelled() and retry.cancelling() == 1
        # A second bounded observation still joins the unresolved original.
        with pytest.raises(TaskExecutionSettlementPending) as raised:
            await pending.settlement.retry()
        assert raised.value.settlement is pending.settlement
        assert len(calls) == 1 and not write_cancellations
        release.set()
        result = await retry_until_settled(pending.settlement)
        assert result is (DispatchStatus.COMPLETED if dispatch else None)
        assert completed.is_set() and len(calls) == 1 and not write_cancellations
        assert pending.settlement._operation is None
        await assert_settled(app, identity)
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        if retry is not None:
            await asyncio.gather(retry, return_exceptions=True)
        await asyncio.wait_for(completed.wait(), 3)
