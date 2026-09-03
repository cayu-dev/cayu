"""Tests for the generic ``run_task_worker`` durable-worker helper."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import (
    admit_test_invocation,
    interrupt_and_release_test_invocation,
    profiled_session_identity,
    runtime_interaction_started_event,
)
from tests.core.task_invocation_fixtures import (
    stored_session_invocation,
    task_backed_session_invocation,
)
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.runtime.task_worker as task_worker_module
from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    Event,
    EventQuery,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfileMismatchError,
    InMemorySessionStore,
    InMemoryTaskStore,
    InterruptedTaskContinuationClaimPage,
    Message,
    ModelStreamEvent,
    PendingActionQuery,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
    ResumeRequest,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    ScriptedModelProvider,
    SessionRunFenced,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskHandlerOutcome,
    TaskInterruptedHandoffConflict,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
    TaskInvocationSnapshot,
    TaskQuery,
    TaskRetryPolicy,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolCapabilityCeiling,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolRoundRecoveryRequest,
    ToolSpec,
    UserInputRecoveryRequest,
    UserInputResponse,
    complete_managed_task,
    fail_managed_task,
    interrupted_task_handoff_request,
    run_task_worker,
)
from cayu.runtime import SessionStatus
from cayu.runtime.provider_operations import provider_operation_resolution_request_digest
from cayu.runtime.sessions import (
    ModelCompletionStageDisposition,
    SessionIdentity,
    run_request_with_task_invocation,
)
from cayu.runtime.tasks import TaskExecutionSource, task_create_with_runtime_invocation
from cayu.tools.user_input import UserInputTool
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def _build(
    tmp_path: Path,
    *,
    secret_redactor: SecretRedactor | None = None,
) -> tuple[CayuApp, SQLiteTaskStore]:
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    app = CayuApp(task_store=store, secret_redactor=secret_redactor)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    return app, store


async def _seed_interrupted_worker_handoff(
    app: CayuApp,
    task_store: TaskStore,
    *,
    task_id: str,
    session_id: str,
) -> None:
    created = await task_store.create_task(TaskCreate(task_id=task_id, type="job"))
    await app.session_store.create(
        run_request_with_task_invocation(
            RunRequest(
                agent_name="worker-agent",
                session_id=session_id,
                task_id=task_id,
                messages=[Message.text("user", "pause")],
            ),
            TaskInvocationSnapshot(
                id=created.id,
                session_id=created.session_id,
                invocation=created.invocation,
            ),
        ),
        identity=SessionIdentity(
            provider_name="scripted",
            model="scripted-model",
        ),
    )
    await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)


async def _seed_receipt_backed_continuation(
    app: CayuApp,
    task_store: TaskStore,
    *,
    task_id: str,
    session_id: str,
) -> None:
    await _seed_interrupted_worker_handoff(
        app,
        task_store,
        task_id=task_id,
        session_id=session_id,
    )
    worker_id = f"prior-{task_id}"
    claimed = await task_store.claim_task(worker_id, TaskQuery(type="job"))
    assert claimed is not None and claimed.id == task_id
    attached = await task_store.attach_task(
        task_id,
        session_id=session_id,
        session_invocation=await stored_session_invocation(
            app.session_store,
            session_id,
        ),
        worker_id=worker_id,
        lease_expires_at=claimed.lease_expires_at,
    )
    session = await app.session_store.load(session_id)
    assert session is not None
    await task_store.release_interrupted_task_worker(
        interrupted_task_handoff_request(
            attached,
            session_run_epoch=session.run_epoch,
        )
    )


def test_run_task_worker_rejects_nan_poll_interval(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        return None

    async def scenario() -> None:
        with pytest.raises(ValueError, match="poll_interval_s"):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker-a",
                poll_interval_s=float("nan"),
                max_tasks=0,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"minimum_idle_delay_s": 0.0}, "minimum_idle_delay_s"),
        ({"maximum_idle_delay_s": 2.0}, "maximum_idle_delay_s"),
        ({"idle_backoff_multiplier": 0.5}, "backoff_multiplier"),
        ({"idle_jitter_ratio": 1.1}, "jitter_ratio"),
        ({"reclaim_every_s": 0.0}, "reclaim_every_s"),
        (
            {"interrupted_handoff_recovery_every_s": 0.0},
            "interrupted_handoff_recovery_every_s",
        ),
    ),
)
def test_run_task_worker_validates_demand_economics(
    overrides: dict[str, float],
    match: str,
) -> None:
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        return None

    async def scenario() -> None:
        with pytest.raises(ValueError, match=match):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="economics-worker",
                max_tasks=0,
                **overrides,
            )

    asyncio.run(scenario())


def test_interrupted_handoff_recovery_has_an_independent_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    stop = asyncio.Event()
    recovery_calls = 0

    async def recover(*_args, **_kwargs):
        nonlocal recovery_calls
        recovery_calls += 1
        if recovery_calls == 2:
            stop.set()
        return task_worker_module._InterruptedHandoffRecoveryPage(
            recovered=0,
            next_after=None,
            exhausted=True,
        )

    monkeypatch.setattr(
        task_worker_module,
        "_recover_expired_interrupted_task_handoffs",
        recover,
    )

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise AssertionError("The empty recovery-cadence test cannot claim a task.")

    async def scenario() -> int:
        return await run_task_worker(
            app,
            store,
            handler,
            worker_id="recovery-cadence-worker",
            poll_interval_s=0.2,
            minimum_idle_delay_s=0.2,
            idle_jitter_ratio=0.0,
            reclaim=False,
            interrupted_handoff_recovery_every_s=0.02,
            stop=stop,
        )

    assert asyncio.run(scenario()) == 0
    assert recovery_calls == 2


def test_reclaim_cadence_can_wake_before_claim_poll() -> None:
    stop = asyncio.Event()

    class ObservedStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.reclaim_calls = 0

        async def reclaim_expired(self, *, query=None, max_reclaims=100):
            self.reclaim_calls += 1
            reclaimed = await super().reclaim_expired(
                query=query,
                max_reclaims=max_reclaims,
            )
            if self.reclaim_calls == 2:
                stop.set()
            return reclaimed

    store = ObservedStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise AssertionError("The empty reclaim-cadence test cannot claim a task.")

    async def scenario() -> int:
        return await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="reclaim-cadence-worker",
                poll_interval_s=10.0,
                minimum_idle_delay_s=10.0,
                maximum_idle_delay_s=10.0,
                idle_jitter_ratio=0.0,
                reclaim=True,
                reclaim_every_s=0.02,
                recover_interrupted_handoffs=False,
                stop=stop,
            ),
            timeout=0.5,
        )

    assert asyncio.run(scenario()) == 0
    assert store.reclaim_calls == 2


def test_run_task_worker_rejects_incomplete_interrupted_handoff_capability() -> None:
    class IncompleteHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True
        load_interrupted_task_handoff_receipt = TaskStore.load_interrupted_task_handoff_receipt

    store = IncompleteHandoffStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise AssertionError("Capability validation must precede task handling.")

    async def scenario() -> None:
        with pytest.raises(NotImplementedError, match="complete idempotent|capability requires"):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker-incomplete-handoff",
                poll_interval_s=0.001,
                max_tasks=1,
            )

    asyncio.run(scenario())


def test_task_worker_revalidates_before_dispatch_after_delayed_claim_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store_time = [datetime(2026, 9, 1, tzinfo=UTC)]
        monotonic_time = [0.0]

        class DelayedClaimAcknowledgementStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self) -> None:
                super().__init__(ownership_clock=lambda: store_time[0])
                self.claim_committed = asyncio.Event()
                self.release_claim_acknowledgement = asyncio.Event()

            async def claim_task(self, worker_id, query=None, *, lease_seconds=300):
                claimed = await super().claim_task(
                    worker_id,
                    query,
                    lease_seconds=lease_seconds,
                )
                if worker_id == "stale-worker" and claimed is not None:
                    self.claim_committed.set()
                    await self.release_claim_acknowledgement.wait()
                return claimed

        monkeypatch.setattr(task_worker_module, "monotonic", lambda: monotonic_time[0])
        store = DelayedClaimAcknowledgementStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="delayed-claim-ack", type="job"))
        handler_workers: list[str] = []

        async def handler(_app: CayuApp, _task: Task, worker_id: str) -> None:
            handler_workers.append(worker_id)

        stale_worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="stale-worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        await store.claim_committed.wait()
        store_time[0] += timedelta(seconds=2)
        monotonic_time[0] += 2
        try:
            assert (
                await run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="replacement-worker",
                    lease_seconds=1,
                    poll_interval_s=0.001,
                    max_tasks=1,
                )
                == 1
            )
        finally:
            store.release_claim_acknowledgement.set()
        with pytest.raises(TaskClaimLost):
            await stale_worker
        assert handler_workers == ["replacement-worker"]

    asyncio.run(scenario())


def test_task_worker_rejects_heartbeat_acknowledged_after_its_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store_time = [datetime(2026, 9, 1, tzinfo=UTC)]
        monotonic_time = [0.0]

        class DelayedHeartbeatAcknowledgementStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self) -> None:
                super().__init__(ownership_clock=lambda: store_time[0])
                self.heartbeat_committed = asyncio.Event()
                self.release_heartbeat_acknowledgement = asyncio.Event()

            async def heartbeat(
                self,
                task_id,
                worker_id,
                *,
                lease_expires_at,
                handoff_id=None,
                extend_seconds=300,
            ):
                renewed = await super().heartbeat(
                    task_id,
                    worker_id,
                    lease_expires_at=lease_expires_at,
                    handoff_id=handoff_id,
                    extend_seconds=extend_seconds,
                )
                if not self.heartbeat_committed.is_set():
                    self.heartbeat_committed.set()
                    await self.release_heartbeat_acknowledgement.wait()
                return renewed

        monkeypatch.setattr(task_worker_module, "monotonic", lambda: monotonic_time[0])
        store = DelayedHeartbeatAcknowledgementStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="delayed-heartbeat-ack", type="job"))
        handler_called = False

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            nonlocal handler_called
            handler_called = True

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        await store.heartbeat_committed.wait()
        store_time[0] += timedelta(seconds=2)
        monotonic_time[0] += 2
        store.release_heartbeat_acknowledgement.set()
        with pytest.raises(TaskClaimLost, match="acknowledgement consumed"):
            await worker
        assert handler_called is False

    asyncio.run(scenario())


def test_task_worker_stops_handler_when_heartbeat_stalls_past_lease() -> None:
    async def scenario() -> None:
        class BlockingPeriodicHeartbeatStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self) -> None:
                super().__init__()
                self.heartbeat_calls = 0
                self.periodic_heartbeat_started = asyncio.Event()
                self.release_periodic_heartbeat = asyncio.Event()

            async def heartbeat(
                self,
                task_id,
                worker_id,
                *,
                lease_expires_at,
                handoff_id=None,
                extend_seconds=300,
            ):
                self.heartbeat_calls += 1
                if worker_id == "stale-worker" and self.heartbeat_calls == 2:
                    self.periodic_heartbeat_started.set()
                    await self.release_periodic_heartbeat.wait()
                return await super().heartbeat(
                    task_id,
                    worker_id,
                    lease_expires_at=lease_expires_at,
                    handoff_id=handoff_id,
                    extend_seconds=extend_seconds,
                )

        store = BlockingPeriodicHeartbeatStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="stalled-periodic-heartbeat", type="job"))
        stale_handler_started = asyncio.Event()
        stale_handler_stopped = asyncio.Event()
        release_stale_handler = threading.Event()

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            if worker_id == "stale-worker":
                stale_handler_started.set()
                try:
                    await asyncio.to_thread(release_stale_handler.wait)
                finally:
                    stale_handler_stopped.set()
                return
            raise AssertionError("A replacement must not run while stale work drains.")

        stale_worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="stale-worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        await asyncio.wait_for(stale_handler_started.wait(), timeout=1)
        await asyncio.wait_for(store.periodic_heartbeat_started.wait(), timeout=1)
        await asyncio.sleep(1.05)
        assert stale_worker.done() is False
        assert stale_handler_stopped.is_set() is False
        for _attempt in range(100):
            draining = await store.load_task("stalled-periodic-heartbeat")
            if draining is not None and draining.status_reason == "cancellation_requested":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Task worker did not durably fence its draining handler.")
        assert await store.reclaim_expired(query=TaskQuery(type="job")) == []
        assert (
            await store.claim_task(
                "replacement-worker",
                TaskQuery(type="job"),
                lease_seconds=1,
            )
            is None
        )
        release_stale_handler.set()
        await asyncio.wait_for(stale_handler_stopped.wait(), timeout=2)
        with pytest.raises(TaskClaimLost, match="positively known lease deadline"):
            await stale_worker
        terminal = await store.load_task("stalled-periodic-heartbeat")
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert terminal.worker_id is None
        assert terminal.lease_expires_at is None

        store.release_periodic_heartbeat.set()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_task_worker_cancellation_fences_opaque_handler_until_natural_settlement() -> None:
    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        class CancellationAcknowledgementLossStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self) -> None:
                super().__init__()
                self.cancel_calls = 0

            async def request_claimed_task_cancellation(
                self,
                task_id,
                worker_id,
                lease_expires_at,
                error=None,
            ):
                self.cancel_calls += 1
                cancelled = await super().request_claimed_task_cancellation(
                    task_id,
                    worker_id,
                    lease_expires_at,
                    error,
                )
                if self.cancel_calls == 1:
                    raise ConnectionError("cancellation acknowledgement lost")
                return cancelled

        store = CancellationAcknowledgementLossStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="opaque-cancelled-handler", type="job"))
        handler_started = asyncio.Event()
        handler_stopped = asyncio.Event()
        release_handler = threading.Event()

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            handler_started.set()
            try:
                await asyncio.to_thread(release_handler.wait)
            finally:
                handler_stopped.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="cancelled-worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        worker.cancel("stop opaque worker")
        cancelling = worker.cancelling()
        for _attempt in range(100):
            draining = await store.load_task("opaque-cancelled-handler")
            if draining is not None and draining.status_reason == "cancellation_requested":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Cancelled worker did not publish its durable drain fence.")
        await asyncio.sleep(1.05)
        assert worker.done() is False
        assert handler_stopped.is_set() is False
        assert await store.reclaim_expired(query=TaskQuery(type="job")) == []
        assert (
            await store.claim_task(
                "replacement-worker",
                TaskQuery(type="job"),
                lease_seconds=1,
            )
            is None
        )

        release_handler.set()
        await asyncio.wait_for(handler_stopped.wait(), timeout=2)
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker
        terminal = await store.load_task("opaque-cancelled-handler")
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert terminal.worker_id is None
        assert terminal.lease_expires_at is None
        assert store.cancel_calls == 1
        return raised.value, cancelling, worker.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("stop opaque worker",)
    assert cancelling == 1
    assert cancelled is True


def test_expired_dispatched_task_cannot_be_reclaimed_while_fence_publication_waits() -> None:
    async def scenario() -> None:
        now = {"value": datetime.now(UTC)}

        class DelayedCancellationStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self) -> None:
                super().__init__(ownership_clock=lambda: now["value"])
                self.fence_started = asyncio.Event()
                self.release_fence = asyncio.Event()

            async def request_claimed_task_cancellation(
                self,
                task_id,
                worker_id,
                lease_expires_at,
                error=None,
            ):
                self.fence_started.set()
                await self.release_fence.wait()
                return await super().request_claimed_task_cancellation(
                    task_id,
                    worker_id,
                    lease_expires_at,
                    error,
                )

        store = DelayedCancellationStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="stale-cancellation", type="job"))
        handler_started = asyncio.Event()
        handler_stopped = asyncio.Event()
        release_handler = threading.Event()

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            handler_started.set()
            try:
                await asyncio.to_thread(release_handler.wait)
            finally:
                handler_stopped.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="stale-worker",
                lease_seconds=60,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        try:
            await asyncio.wait_for(handler_started.wait(), timeout=1)
            claimed = await store.load_task("stale-cancellation")
            assert claimed is not None
            assert claimed.lease_expires_at is not None

            worker.cancel("stop stale worker")
            await asyncio.wait_for(store.fence_started.wait(), timeout=1)
            now["value"] = claimed.lease_expires_at + timedelta(seconds=1)
            reclaimed = await store.reclaim_expired(query=TaskQuery(type="job"))
            assert reclaimed == []
            replacement = await store.claim_task(
                "replacement-worker",
                TaskQuery(type="job"),
                lease_seconds=300,
            )
            assert replacement is None

            draining = await store.load_task(claimed.id)
            assert draining is not None
            assert draining.worker_id == "stale-worker"
            assert draining.lease_expires_at == claimed.lease_expires_at
            assert draining.status_reason == "cancellation_requested"
            assert handler_stopped.is_set() is False

            store.release_fence.set()
            await asyncio.sleep(0)
            assert worker.done() is False
            assert handler_stopped.is_set() is False
            assert (
                await store.claim_task(
                    "replacement-worker",
                    TaskQuery(type="job"),
                    lease_seconds=300,
                )
                is None
            )

            release_handler.set()
            await asyncio.wait_for(handler_stopped.wait(), timeout=2)
            with pytest.raises(asyncio.CancelledError):
                await worker
            terminal = await store.load_task(claimed.id)
            assert terminal is not None
            assert terminal.status is TaskStatus.CANCELLED
            assert terminal.worker_id is None
            assert terminal.lease_expires_at is None
        finally:
            store.release_fence.set()
            release_handler.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_task_retry_worker_cancellation_fences_opaque_handler_until_natural_settlement() -> None:
    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="opaque-cancelled-retry-handler",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        handler_started = threading.Event()
        handler_stopped = threading.Event()
        release_handler = threading.Event()

        def opaque_work() -> None:
            handler_started.set()
            release_handler.wait()
            handler_stopped.set()

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            await asyncio.to_thread(opaque_work)

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="cancelled-retry-worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        assert await asyncio.to_thread(handler_started.wait, 1)
        worker.cancel("stop opaque retry worker")
        cancelling = worker.cancelling()
        await asyncio.sleep(1.05)
        assert worker.done() is False
        assert handler_stopped.is_set() is False
        draining = await store.load_task("opaque-cancelled-retry-handler")
        assert draining is not None
        assert draining.status is TaskStatus.CLAIMED
        assert draining.worker_id == "cancelled-retry-worker"
        assert draining.lease_expires_at is not None
        assert await store.reclaim_expired(query=TaskQuery(type="job")) == []
        assert await store.claim_task("replacement-retry-worker") is None

        release_handler.set()
        with pytest.raises(asyncio.CancelledError, match="stop opaque retry worker") as raised:
            await asyncio.wait_for(worker, timeout=2)
        released = await store.load_task("opaque-cancelled-retry-handler")
        assert released is not None
        assert released.status is TaskStatus.PENDING
        assert released.worker_id is None
        assert released.lease_expires_at is None
        assert released.retry_series is not None
        assert released.retry_series.successor_task_id is None
        return raised.value, cancelling, worker.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("stop opaque retry worker",)
    assert cancelling == 1
    assert cancelled is True


def test_task_retry_worker_revalidates_cancellation_after_deadline_query() -> None:
    class CancellationDuringDeadlineQueryStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def task_retry_deadline_elapsed(
            self,
            task_id,
            worker_id,
            *,
            lease_expires_at,
        ):
            await self.cancel_task(task_id, {"code": "cancel_during_deadline_query"})
            return await super().task_retry_deadline_elapsed(
                task_id,
                worker_id,
                lease_expires_at=lease_expires_at,
            )

    async def scenario() -> None:
        store = CancellationDuringDeadlineQueryStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(
            TaskCreate(
                task_id="retry-cancelled-during-deadline-query",
                type="job",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_elapsed_seconds=300,
                ),
            )
        )
        handler_called = False

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            nonlocal handler_called
            handler_called = True

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="deadline-query-worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
            == 1
        )
        assert handler_called is False
        terminal = await store.load_task("retry-cancelled-during-deadline-query")
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert terminal.worker_id is None

    asyncio.run(scenario())


def test_completed_handler_still_fences_a_stalled_heartbeat() -> None:
    class BlockingHeartbeatStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_calls = 0
            self.periodic_heartbeat_started = asyncio.Event()
            self.release_periodic_heartbeat = asyncio.Event()

        async def heartbeat(
            self,
            task_id,
            worker_id,
            *,
            lease_expires_at,
            handoff_id=None,
            extend_seconds=300,
        ):
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 2:
                self.periodic_heartbeat_started.set()
                await self.release_periodic_heartbeat.wait()
            return await super().heartbeat(
                task_id,
                worker_id,
                lease_expires_at=lease_expires_at,
                handoff_id=handoff_id,
                extend_seconds=extend_seconds,
            )

    async def scenario() -> None:
        store = BlockingHeartbeatStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="completed-before-heartbeat", type="job"))
        handler_started = asyncio.Event()
        allow_handler_completion = asyncio.Event()
        effects: list[str] = []

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            handler_started.set()
            await allow_handler_completion.wait()
            effects.append("completed")

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="completed-handler-worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        await asyncio.wait_for(store.periodic_heartbeat_started.wait(), timeout=1)
        allow_handler_completion.set()

        with pytest.raises(TaskClaimLost, match="positively known lease deadline"):
            await asyncio.wait_for(worker, timeout=2)
        assert effects == ["completed"]
        terminal = await store.load_task("completed-before-heartbeat")
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert terminal.worker_id is None
        assert await store.reclaim_expired(query=TaskQuery(type="job")) == []
        assert await store.claim_task("replacement-worker") is None

        store.release_periodic_heartbeat.set()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_task_worker_rechecks_cancellation_marker_before_handler_dispatch() -> None:
    async def scenario() -> None:
        class CancellationDuringRenewalStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            async def heartbeat(
                self,
                task_id,
                worker_id,
                *,
                lease_expires_at,
                handoff_id=None,
                extend_seconds=300,
            ):
                renewed = await super().heartbeat(
                    task_id,
                    worker_id,
                    lease_expires_at=lease_expires_at,
                    handoff_id=handoff_id,
                    extend_seconds=extend_seconds,
                )
                if renewed.status_reason is None:
                    return await self.cancel_task(
                        task_id,
                        {"code": "cancellation_won_before_dispatch"},
                    )
                return renewed

        store = CancellationDuringRenewalStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="cancel-before-dispatch", type="job"))
        handler_called = False

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            nonlocal handler_called
            handler_called = True

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="cancelled-before-dispatch-worker",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
            == 1
        )
        terminal = await store.load_task("cancel-before-dispatch")
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert handler_called is False

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_method", ["reclaim_expired", "claim_task"])
def test_ordinary_task_worker_preserves_store_failure_identity_and_traceback(
    failure_method: str,
) -> None:
    class OrdinaryFailureStore(InMemoryTaskStore):
        supports_verified_work_contracts = False
        hold_claimed_work_contract_task = TaskStore.hold_claimed_work_contract_task

        def __init__(self) -> None:
            super().__init__()
            self.failure = KeyError(f"ordinary {failure_method} failure")

        async def reclaim_expired(self, *, query=None):
            if failure_method == "reclaim_expired":
                raise self.failure
            return await super().reclaim_expired(query=query)

        async def claim_task(self, worker_id, query, *, lease_seconds):
            if failure_method == "claim_task":
                raise self.failure
            return await super().claim_task(
                worker_id,
                query,
                lease_seconds=lease_seconds,
            )

    async def scenario() -> tuple[BaseException, OrdinaryFailureStore]:
        store = OrdinaryFailureStore()
        app = CayuApp(enable_logging=False)

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            raise AssertionError("An ordinary store failure must precede task handling.")

        with pytest.raises(KeyError) as raised:
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="ordinary-worker-failure",
                poll_interval_s=0.001,
                max_tasks=1,
            )
        return raised.value, store

    failure, store = asyncio.run(scenario())

    assert failure is store.failure
    traceback_names: list[str] = []
    traceback = failure.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert failure_method in traceback_names


@pytest.mark.anyio
async def test_one_second_task_lease_heartbeats_after_one_third(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elapsed = 0.0
    stop = asyncio.Event()
    observed_heartbeats: list[tuple[str, str, int, float]] = []

    class ExpiringLeaseStore:
        async def heartbeat(
            self,
            task_id: str,
            worker_id: str,
            *,
            lease_expires_at,
            handoff_id: str | None = None,
            extend_seconds: int,
        ) -> None:
            assert handoff_id is None
            assert elapsed < 1.0
            del lease_expires_at
            observed_heartbeats.append((task_id, worker_id, extend_seconds, elapsed))
            stop.set()

    async def advance_clock(seconds: float, wait_stop: asyncio.Event) -> bool:
        nonlocal elapsed
        elapsed += seconds
        return wait_stop.is_set()

    monkeypatch.setattr(task_worker_module, "_wait_or_stop", advance_clock)

    await task_worker_module._heartbeat_until(
        ExpiringLeaseStore(),  # type: ignore[arg-type]
        "task-1",
        "worker-a",
        1,
        stop,
        lease_authority=task_worker_module._TaskLeaseAuthority(datetime.now(UTC)),
    )

    assert observed_heartbeats == [("task-1", "worker-a", 1, pytest.approx(1 / 3))]


@pytest.mark.anyio
async def test_retry_deadline_inspection_failure_reconciles_terminal_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection_failure = RuntimeError("deadline read failed")
    authoritative_store = InMemoryTaskStore()
    await authoritative_store.create_task(TaskCreate(task_id="task-1", type="job"))
    renewed = await authoritative_store.claim_task("worker-a", lease_seconds=3)
    assert renewed is not None
    terminal = await authoritative_store.complete_task(
        "task-1",
        {"ok": True},
        worker_id="worker-a",
        lease_expires_at=renewed.lease_expires_at,
    )
    load_task_calls = 0

    class ConcurrentTerminalStore:
        async def heartbeat(
            self,
            task_id: str,
            worker_id: str,
            *,
            lease_expires_at: datetime,
            handoff_id: str | None = None,
            extend_seconds: int,
        ) -> Task:
            del task_id, worker_id, lease_expires_at, handoff_id, extend_seconds
            return renewed

        async def task_retry_deadline_elapsed(
            self,
            task_id: str,
            worker_id: str,
            *,
            lease_expires_at: datetime,
        ) -> bool:
            del task_id, worker_id, lease_expires_at
            raise inspection_failure

        async def load_task(self, task_id: str) -> Task:
            nonlocal load_task_calls
            del task_id
            load_task_calls += 1
            return terminal

    async def advance_clock(_seconds: float, _stop: asyncio.Event) -> bool:
        return False

    monkeypatch.setattr(task_worker_module, "_wait_or_stop", advance_clock)

    outcome = await task_worker_module._heartbeat_until(
        ConcurrentTerminalStore(),  # type: ignore[arg-type]
        "task-1",
        "worker-a",
        3,
        asyncio.Event(),
        lease_authority=task_worker_module._TaskLeaseAuthority(renewed.lease_expires_at),
        enforce_retry_deadline=True,
    )

    assert outcome is task_worker_module._TaskHeartbeatOutcome.TERMINAL
    assert load_task_calls == 1


@pytest.mark.anyio
async def test_task_heartbeat_rejects_acknowledgement_that_consumed_renewed_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_time = 0.0
    stop = asyncio.Event()

    class DelayedHeartbeatStore:
        async def heartbeat(
            self,
            task_id: str,
            worker_id: str,
            *,
            lease_expires_at,
            handoff_id: str | None = None,
            extend_seconds: int,
        ) -> None:
            nonlocal monotonic_time
            del task_id, worker_id, lease_expires_at, handoff_id, extend_seconds
            monotonic_time += 2

    async def advance_clock(seconds: float, wait_stop: asyncio.Event) -> bool:
        nonlocal monotonic_time
        monotonic_time += seconds
        return wait_stop.is_set()

    monkeypatch.setattr(task_worker_module, "monotonic", lambda: monotonic_time)
    monkeypatch.setattr(task_worker_module, "_wait_or_stop", advance_clock)

    with pytest.raises(TaskClaimLost, match="acknowledgement consumed"):
        await task_worker_module._heartbeat_until(
            DelayedHeartbeatStore(),  # type: ignore[arg-type]
            "task-1",
            "worker-a",
            1,
            stop,
            lease_authority=task_worker_module._TaskLeaseAuthority(datetime.now(UTC)),
        )


@pytest.mark.anyio
async def test_task_cancellation_fence_rejects_reused_worker_lease_generation() -> None:
    ownership_now = datetime(2026, 1, 1, tzinfo=UTC)
    store = InMemoryTaskStore(ownership_clock=lambda: ownership_now)
    await store.create_task(TaskCreate(task_id="same-worker-cancellation", type="job"))
    stale = await store.claim_task("shared-worker", lease_seconds=1)
    assert stale is not None
    assert stale.lease_expires_at is not None
    stale_authority = task_worker_module._TaskLeaseAuthority(stale.lease_expires_at)

    ownership_now += timedelta(seconds=1)
    assert [task.id for task in await store.reclaim_expired()] == [stale.id]
    successor = await store.claim_task("shared-worker", lease_seconds=30)
    assert successor is not None
    assert successor.lease_expires_at != stale.lease_expires_at

    with pytest.raises(TaskClaimLost, match="expected lease generation"):
        await task_worker_module._request_task_cancellation_fence(
            store,
            stale,
            "shared-worker",
            {"code": "stale-owner-cancelled"},
            lease_authority=stale_authority,
        )

    assert await store.load_task(successor.id) == successor


def test_handler_may_finish_cleanup_after_terminalizing_its_task() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="terminal-cleanup", type="job"))
        cleanup_finished = asyncio.Event()

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            await store.complete_task(
                task.id,
                {"ok": True},
                worker_id=worker_id,
                lease_expires_at=task.lease_expires_at,
            )
            await asyncio.sleep(0.5)
            cleanup_finished.set()

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="worker-a",
                lease_seconds=1,
                poll_interval_s=0.01,
                max_tasks=1,
            )
            == 1
        )
        assert cleanup_finished.is_set()
        terminal = await store.load_task("terminal-cleanup")
        assert terminal is not None
        assert terminal.status is TaskStatus.COMPLETED

    asyncio.run(scenario())


async def _run_handler(app: CayuApp, task: Task, worker_id: str) -> None:
    async for _event in app.run(
        RunRequest(
            agent_name="worker-agent",
            session_id=f"sess-{task.id}",
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
            messages=[Message.text("user", "go")],
        )
    ):
        pass


class _PublishChangeTool(Tool):
    spec = ToolSpec(
        name="publish_change",
        description="Publish one reviewed change.",
        input_schema={
            "type": "object",
            "properties": {"change": {"type": "string"}},
            "required": ["change"],
        },
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:task-worker:publish-change-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"Published {args['change']}",
            structured={"session_id": ctx.session_id},
        )


class _VersionedSessionFailureHook(RuntimeHook):
    def __init__(self, implementation_version: str) -> None:
        self.implementation_version = implementation_version
        self.failed_sessions: list[str] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:task-worker:session-failure-hook",
            behavior_version="1",
            implementation_version=self.implementation_version,
        )

    async def after_session_failed(self, context: RuntimeHookContext) -> None:
        self.failed_sessions.append(context.session.id)


def _register_approval_agent(app: CayuApp) -> None:
    app.register_agent(
        AgentSpec(name="worker-agent", model="scripted-model"),
        tools=[_PublishChangeTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )


def test_run_task_worker_claims_runs_and_completes_a_task(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> tuple[int, Task | None]:
        created = await store.create_task(
            TaskCreate(type="job", assigned_agent_name="worker-agent")
        )
        handled = await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return handled, await store.load_task(created.id)

    handled, task = asyncio.run(scenario())
    assert handled == 1
    assert task is not None
    assert task.status == "completed"


def test_worker_uses_current_hidden_lease_for_session_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periodic_heartbeat = asyncio.Event()
    acknowledged_leases: list[datetime] = []

    class HeartbeatRecordingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def heartbeat(
            self,
            task_id: str,
            worker_id: str,
            *,
            lease_expires_at: datetime,
            handoff_id: str | None = None,
            extend_seconds: int = 300,
        ) -> Task:
            updated = await super().heartbeat(
                task_id,
                worker_id,
                lease_expires_at=lease_expires_at,
                handoff_id=handoff_id,
                extend_seconds=extend_seconds,
            )
            assert updated.lease_expires_at is not None
            acknowledged_leases.append(updated.lease_expires_at)
            if len(acknowledged_leases) >= 2:
                periodic_heartbeat.set()
            return updated

    async def scenario() -> Task | None:
        store = HeartbeatRecordingStore()
        provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
        created = await store.create_task(TaskCreate(task_id="renew-before-attach", type="job"))
        original_start_task = app._session_engine._start_task

        async def start_after_periodic_renewal(**kwargs):
            await asyncio.wait_for(periodic_heartbeat.wait(), timeout=3)
            assert kwargs["lease_expires_at"] == acknowledged_leases[0]
            assert acknowledged_leases[-1] != kwargs["lease_expires_at"]
            return await original_start_task(**kwargs)

        monkeypatch.setattr(app._session_engine, "_start_task", start_after_periodic_renewal)
        handled = await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="stable-worker",
            query=TaskQuery(type="job"),
            lease_seconds=3,
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        assert handled == 1
        assert len(provider.requests) == 1
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.COMPLETED


@pytest.mark.parametrize("terminal_kind", ["complete", "fail"])
@pytest.mark.parametrize("entrypoint", ["managed_helper", "store_method"])
def test_managed_handler_terminalization_uses_latest_acknowledged_lease(
    terminal_kind: str,
    entrypoint: str,
) -> None:
    periodic_heartbeat = asyncio.Event()
    acknowledged_leases: list[datetime] = []
    terminal_leases: list[datetime | None] = []

    class LeaseRecordingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def heartbeat(self, task_id, worker_id, **kwargs):
            updated = await super().heartbeat(task_id, worker_id, **kwargs)
            assert updated.lease_expires_at is not None
            acknowledged_leases.append(updated.lease_expires_at)
            if len(acknowledged_leases) >= 2:
                periodic_heartbeat.set()
            return updated

        async def complete_task(self, task_id, result, **kwargs):
            terminal_leases.append(kwargs.get("lease_expires_at"))
            return await super().complete_task(task_id, result, **kwargs)

        async def fail_task(self, task_id, error, **kwargs):
            terminal_leases.append(kwargs.get("lease_expires_at"))
            return await super().fail_task(task_id, error, **kwargs)

    async def scenario() -> Task | None:
        store = LeaseRecordingStore()
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(
            TaskCreate(task_id=f"managed-{terminal_kind}-latest-lease", type="job")
        )

        async def handler(_app: CayuApp, task: Task, worker_id: str) -> None:
            original_lease = task.lease_expires_at
            await asyncio.wait_for(periodic_heartbeat.wait(), timeout=3)
            assert acknowledged_leases[-1] != original_lease
            if terminal_kind == "complete":
                if entrypoint == "managed_helper":
                    await complete_managed_task(store, task, worker_id, {"ok": True})
                else:
                    await store.complete_task(
                        task.id,
                        {"ok": True},
                        worker_id=worker_id,
                        lease_expires_at=original_lease,
                    )
            else:
                if entrypoint == "managed_helper":
                    await fail_managed_task(store, task, worker_id, {"code": "expected"})
                else:
                    await store.fail_task(
                        task.id,
                        {"code": "expected"},
                        worker_id=worker_id,
                        lease_expires_at=original_lease,
                    )

        assert (
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="managed-terminal-worker",
                query=TaskQuery(type="job"),
                lease_seconds=3,
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status is (
        TaskStatus.COMPLETED if terminal_kind == "complete" else TaskStatus.FAILED
    )
    assert len(terminal_leases) == 1
    if entrypoint == "managed_helper":
        assert terminal_leases[0] == acknowledged_leases[-1]
    else:
        assert terminal_leases[0] != acknowledged_leases[-1]


@pytest.mark.parametrize("provider_fails", [False, True])
def test_session_engine_terminalizes_worker_task_with_nonreceipt_custom_store(
    provider_fails: bool,
) -> None:
    class NonReceiptStore(InMemoryTaskStore):
        supports_idempotent_terminalization = False
        verified_work_mutations_are_cancellation_quiescent = True

    async def scenario() -> tuple[list[Event], Task | None]:
        store = NonReceiptStore()
        provider = ScriptedModelProvider(
            [
                [
                    *([ModelStreamEvent.error("provider unavailable")] if provider_fails else []),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
        created = await store.create_task(
            TaskCreate(task_id="nonreceipt-terminalization", type="job")
        )
        claimed = await store.claim_task("custom-worker", TaskQuery(type="job"))
        assert claimed is not None
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="nonreceipt-terminalization-session",
                    task_id=claimed.id,
                    task_worker_id="custom-worker",
                    task_lease_expires_at=claimed.lease_expires_at,
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert len(provider.requests) == 1
        return events, await store.load_task(created.id)

    events, task = asyncio.run(scenario())
    assert events[-1].type is (
        EventType.SESSION_FAILED if provider_fails else EventType.SESSION_COMPLETED
    )
    assert task is not None
    assert task.status is (TaskStatus.FAILED if provider_fails else TaskStatus.COMPLETED)


def test_running_ordinary_task_cancellation_is_worker_terminalized(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> None:
        started = asyncio.Event()
        stopped = asyncio.Event()
        release = asyncio.Event()
        created = await store.create_task(TaskCreate(task_id="cancel-live", type="job"))

        async def blocking_handler(
            _app: CayuApp,
            _task: Task,
            _worker_id: str,
        ) -> None:
            started.set()
            try:
                await release.wait()
            finally:
                stopped.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                blocking_handler,
                worker_id="worker-cancel-live",
                query=TaskQuery(type="job"),
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        requested = await store.cancel_task(
            created.id,
            {"reason": "operator cancelled live builder"},
        )
        assert requested.status is TaskStatus.CLAIMED
        assert requested.worker_id == "worker-cancel-live"
        assert requested.lease_expires_at is not None
        assert requested.status_reason == "cancellation_requested"
        assert requested.status_payload is not None

        await asyncio.sleep(0)
        assert worker.done() is False
        assert stopped.is_set() is False
        release.set()
        assert await asyncio.wait_for(worker, timeout=3) == 1
        assert stopped.is_set()

        terminal = await store.load_task(created.id)
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert terminal.worker_id is None
        assert terminal.lease_expires_at is None
        assert terminal.error == {"reason": "operator cancelled live builder"}
        key = requested.status_payload["terminalization_idempotency_key"]
        receipt = await store.load_task_terminalization_receipt(created.id, key)
        assert receipt is not None
        assert receipt.kind is TaskTerminalKind.CANCELLED
        assert receipt.task == terminal

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "missing_capability",
    ["cancellation_reconciliation", "idempotent_terminalization"],
)
def test_run_task_worker_rejects_incomplete_cancellation_reconciliation_before_claim(
    missing_capability: str,
) -> None:
    class IncompleteCancellationStore(InMemoryTaskStore):
        pass

    async def scenario() -> None:
        store = IncompleteCancellationStore()
        if missing_capability == "cancellation_reconciliation":
            store.supports_task_cancellation_reconciliation = False
        else:
            store.supports_idempotent_terminalization = False
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(TaskCreate(task_id="unsupported-worker", type="job"))
        handler_called = False

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            nonlocal handler_called
            handler_called = True

        with pytest.raises(
            NotImplementedError,
            match="cancellation reconciliation before it can claim work",
        ):
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="unsupported-worker",
                query=TaskQuery(type="job"),
                max_tasks=1,
            )
        assert handler_called is False
        assert await store.load_task(created.id) == created

    asyncio.run(scenario())


def test_owner_lost_cancellation_reconciliation_uses_store_owned_time() -> None:
    async def scenario() -> None:
        now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}
        store = InMemoryTaskStore(ownership_clock=lambda: now["value"])
        created = await store.create_task(TaskCreate(task_id="store-time-cancel", type="job"))
        claimed = await store.claim_task(
            "store-time-worker",
            TaskQuery(type="job"),
            lease_seconds=5,
        )
        assert claimed is not None
        requested = await store.cancel_task(created.id, {"code": "worker_cancelled"})
        assert requested.lease_expires_at is not None
        now["value"] = requested.lease_expires_at + timedelta(seconds=1)

        await task_worker_module._settle_ordinary_task_cancellation_after_quiescence(
            store,
            created.id,
            "store-time-worker",
        )
        terminal = await store.load_task(created.id)
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert terminal.status_payload is not None
        reconciliation = terminal.status_payload["cancellation_reconciliation"]
        assert datetime.fromisoformat(reconciliation["reconciliation_requested_at"]) == (
            requested.lease_expires_at
        )

    asyncio.run(scenario())


def test_owner_lost_retry_reconciliation_uses_store_owned_time() -> None:
    class CapturingTaskStore(InMemoryTaskStore):
        reconciliation_requested_at: datetime | None = None

        async def reconcile_task_retry_cancellation(self, request):
            self.reconciliation_requested_at = request.reconciliation_requested_at
            return await super().reconcile_task_retry_cancellation(request)

    async def scenario() -> None:
        now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}
        store = CapturingTaskStore(
            clock=lambda: now["value"],
            ownership_clock=lambda: now["value"],
        )
        created = await store.create_task(
            TaskCreate(
                task_id="store-time-retry-cancel",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        claimed = await store.claim_task(
            "store-time-retry-worker",
            TaskQuery(type="job"),
            lease_seconds=5,
        )
        assert claimed is not None
        requested = await store.cancel_task(created.id, {"code": "worker_cancelled"})
        assert requested.lease_expires_at is not None
        now["value"] = requested.lease_expires_at + timedelta(seconds=1)

        await task_worker_module._settle_retry_task_cancellation_after_quiescence(
            store,
            created.id,
            "store-time-retry-worker",
            cancellation_baseline=0,
            report=None,
        )
        terminal = await store.load_task(created.id)
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED
        assert store.reconciliation_requested_at == requested.lease_expires_at

    asyncio.run(scenario())


def test_run_task_worker_reconciles_failure_terminalization_acknowledgement_loss() -> None:
    class CommitThenRaiseTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

    store = CommitThenRaiseTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def scenario() -> tuple[int, Task | None]:
        await store.create_task(TaskCreate(task_id="worker-failure", type="job"))

        async def fail_handler(_app: CayuApp, _task: Task, _worker_id: str):
            raise RuntimeError("handler failed")

        handled = await run_task_worker(
            app,
            store,
            fail_handler,
            worker_id="worker-a",
            max_tasks=1,
        )
        return handled, await store.load_task("worker-failure")

    handled, task = asyncio.run(scenario())

    assert handled == 1
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert store.terminalize_calls == 1


def test_run_task_worker_continues_after_handler_terminalizes_then_raises() -> None:
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def scenario() -> tuple[int, Task | None, Task | None]:
        await store.create_task(TaskCreate(task_id="worker-first", type="job"))
        await store.create_task(TaskCreate(task_id="worker-second", type="job"))

        async def terminalize_then_raise(_app: CayuApp, task: Task, worker_id: str):
            await store.complete_task(
                task.id,
                {"winner": "handler"},
                worker_id=worker_id,
                lease_expires_at=task.lease_expires_at,
            )
            if task.id == "worker-first":
                raise RuntimeError("handler raised after terminalizing")

        handled = await run_task_worker(
            app,
            store,
            terminalize_then_raise,
            worker_id="worker-a",
            max_tasks=2,
        )
        return (
            handled,
            await store.load_task("worker-first"),
            await store.load_task("worker-second"),
        )

    handled, first, second = asyncio.run(scenario())
    assert handled == 2
    assert first is not None
    assert first.status is TaskStatus.COMPLETED
    assert first.result == {"winner": "handler"}
    assert second is not None
    assert second.status is TaskStatus.COMPLETED
    assert second.result == {"winner": "handler"}


def test_run_task_worker_keeps_same_key_changed_intent_conflict_explicit() -> None:
    class ChangedIntentStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            await super().terminalize_task(
                TaskTerminalizationRequest(
                    task_id=request.task_id,
                    worker_id=request.worker_id,
                    lease_expires_at=request.lease_expires_at,
                    kind=TaskTerminalKind.COMPLETED,
                    result={"winner": "other-intent"},
                    idempotency_key=request.idempotency_key,
                )
            )
            return await super().terminalize_task(request)

    store = ChangedIntentStore()
    app = CayuApp(task_store=store, enable_logging=False)

    async def scenario() -> Task | None:
        await store.create_task(TaskCreate(task_id="worker-conflict", type="job"))

        async def fail_handler(_app: CayuApp, _task: Task, _worker_id: str):
            raise RuntimeError("handler failed")

        with pytest.raises(TaskTerminalizationConflict):
            await run_task_worker(
                app,
                store,
                fail_handler,
                worker_id="worker-a",
                max_tasks=1,
            )
        return await store.load_task("worker-conflict")

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.result == {"winner": "other-intent"}


def test_run_task_worker_returns_immediately_when_stopped(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> int:
        stop = asyncio.Event()
        stop.set()
        return await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            reclaim=False,
            stop=stop,
        )

    assert asyncio.run(scenario()) == 0


def test_run_task_worker_rejects_negative_max_tasks(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def scenario() -> None:
        await run_task_worker(
            app,
            store,
            _run_handler,
            worker_id="w1",
            max_tasks=-1,
        )

    with pytest.raises(ValueError, match="max_tasks must be non-negative"):
        asyncio.run(scenario())


def test_run_task_worker_fails_task_when_handler_leaves_it_active(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def no_terminal_state(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        return None

    async def scenario() -> Task | None:
        created = await store.create_task(
            TaskCreate(type="job", assigned_agent_name="worker-agent")
        )
        handled = await run_task_worker(
            app,
            store,
            no_terminal_state,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        assert handled == 1
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Task handler returned without completing or failing the task.",
    }


@pytest.mark.parametrize(
    ("rejected_text", "error_code"),
    [
        ("handler failure\u0000with invalid text", "nul_character"),
        ("handler failure\ud800with invalid text", "unicode_surrogate"),
    ],
)
def test_run_task_worker_terminalizes_nonportable_handler_failures(
    tmp_path: Path,
    rejected_text: str,
    error_code: str,
) -> None:
    app, store = _build(tmp_path)

    async def fail_handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise RuntimeError(rejected_text)

    async def scenario():
        created = await store.create_task(TaskCreate(type="job"))
        handled = await run_task_worker(
            app,
            store,
            fail_handler,
            worker_id="w1",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        task = await store.load_task(created.id)
        reclaimed = await store.reclaim_expired(query=TaskQuery(type="job"))
        second_claim = await store.claim_task(
            "w2",
            TaskQuery(type="job"),
            lease_seconds=300,
        )
        return handled, task, reclaimed, second_claim

    handled, task, reclaimed, second_claim = asyncio.run(scenario())

    assert handled == 1
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Task handler failed with a non-portable diagnostic.",
        "durable_value_error_code": error_code,
        "durable_value_error_path": "$",
    }
    assert reclaimed == []
    assert second_claim is None


def test_run_task_worker_redacts_handler_failure_before_durable_write(tmp_path: Path) -> None:
    secret = "task-worker-diagnostic-secret-canary"
    app, store = _build(tmp_path, secret_redactor=SecretRedactor(secret))

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise RuntimeError(f"upstream rejected credential {secret}")

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            fail,
            worker_id="worker-1",
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())

    assert task is not None
    assert task.status == "failed"
    assert task.error is not None
    assert secret not in str(task.error)
    assert REDACTED_SECRET in task.error["message"]


def test_run_task_worker_redacts_secret_bearing_exception_type_before_durable_write(
    tmp_path: Path,
) -> None:
    secret = "TaskWorkerSecretTypeCanary"
    app, store = _build(tmp_path, secret_redactor=SecretRedactor(secret))
    secret_error_type = type(f"Failure{secret}", (Exception,), {})

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise secret_error_type

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            fail,
            worker_id="worker-1",
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())

    assert task is not None
    assert task.error == {
        "error": f"Failure{REDACTED_SECRET}",
        "message": f"Failure{REDACTED_SECRET}: task handler failed",
    }


def test_run_task_worker_continues_after_handler_error_with_broken_stringification(
    tmp_path: Path,
) -> None:
    app, store = _build(tmp_path)

    class BrokenStringError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("stringification failed")

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise BrokenStringError

    async def scenario() -> tuple[int, list[Task | None]]:
        created = [
            await store.create_task(TaskCreate(type="job")),
            await store.create_task(TaskCreate(type="job")),
        ]
        handled = await run_task_worker(
            app,
            store,
            fail,
            worker_id="worker-1",
            query=TaskQuery(type="job"),
            max_tasks=2,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return handled, [await store.load_task(task.id) for task in created]

    handled, tasks = asyncio.run(scenario())

    assert handled == 2
    assert all(task is not None and task.status == "failed" for task in tasks)
    assert [task.error for task in tasks if task is not None] == [
        {
            "error": "BrokenStringError",
            "message": "BrokenStringError: task handler failed",
        },
        {
            "error": "BrokenStringError",
            "message": "BrokenStringError: task handler failed",
        },
    ]


def test_task_worker_cancellation_during_failure_write_does_not_retain_raw_error() -> None:
    secret = "task-worker-cancelled-publication-secret-canary"

    class BlockingFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.failure_started = asyncio.Event()

        async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
            self.failure_started.set()
            await asyncio.Event().wait()
            return await super().terminalize_task(request)

    store = BlockingFailureStore()
    app = CayuApp(
        task_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def fail(_app: CayuApp, _task: Task, _worker_id: str) -> None:
        raise RuntimeError(f"upstream rejected credential {secret}")

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        await store.create_task(TaskCreate(type="job"))
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                store,
                fail,
                worker_id="worker-1",
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await store.failure_started.wait()
        worker_task.cancel("stop worker")
        cancelling = worker_task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await worker_task
        return exc_info.value, cancelling, worker_task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("stop worker",)
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


def test_run_task_worker_rejects_secret_bearing_worker_authority(tmp_path: Path) -> None:
    secret = "task-worker-authority-secret-canary"
    app, store = _build(tmp_path, secret_redactor=SecretRedactor(secret))

    with pytest.raises(ValueError, match="durable task authority"):
        asyncio.run(
            run_task_worker(
                app,
                store,
                _run_handler,
                worker_id=f"worker-{secret}",
                max_tasks=0,
            )
        )


def test_run_task_worker_hands_interrupted_session_to_reconstructed_control_plane(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "sessions.sqlite"
    task_path = tmp_path / "tasks.sqlite"
    first_app = CayuApp(
        session_store=SQLiteSessionStore(session_path),
        task_store=SQLiteTaskStore(task_path),
        enable_logging=False,
    )
    first_app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-change",
                        name="publish_change",
                        arguments={"change": "reviewed-release"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    _register_approval_agent(first_app)
    assert first_app.task_store is not None

    async def await_approval(
        app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in app.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-handoff",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "Publish the reviewed change.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[
        Task | None,
        SessionStatus,
        Task | None,
        SessionStatus,
        list[tuple[str, str]],
    ]:
        await first_app.create_task(
            TaskCreate(
                task_id="task-handoff",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        handled = await run_task_worker(
            first_app,
            first_app.task_store,
            await_approval,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        assert handled == 1

        handed_off_task = await first_app.task_store.load_task("task-handoff")
        handed_off_session = await first_app.session_store.load("session-handoff")
        pending = await first_app.session_store.query_pending_actions(
            PendingActionQuery(session_id="session-handoff")
        )
        assert handed_off_task is not None
        assert handed_off_task.status == "running"
        assert handed_off_task.session_id == "session-handoff"
        assert handed_off_task.worker_id is None
        assert handed_off_task.lease_expires_at is None
        assert handed_off_session is not None
        assert handed_off_session.status == SessionStatus.INTERRUPTED
        assert len(pending.actions) == 1
        assert pending.actions[0].approval_id is not None
        handoff_events = await first_app.session_store.query_events(
            EventQuery(
                session_id="session-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )

        reconstructed = CayuApp(
            session_store=SQLiteSessionStore(session_path),
            task_store=SQLiteTaskStore(task_path),
            enable_logging=False,
        )
        reconstructed.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("The durable work item is complete."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        _register_approval_agent(reconstructed)
        approval_id = pending.actions[0].approval_id
        assert approval_id is not None
        approval_round_id = pending.actions[0].round_id
        approval_tool_call_id = pending.actions[0].tool_call_id
        assert approval_round_id is not None
        assert approval_tool_call_id is not None
        continuation = (
            await reconstructed.task_store.claim_interrupted_task_continuation(
                "control-plane-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert continuation is not None
        async for _event in reconstructed.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="session-handoff",
                task_worker_id="control-plane-worker",
                task_handoff_id=continuation.interrupted_handoff_id,
                approval_id=approval_id,
                tool_round_id=approval_round_id,
                tool_call_id=approval_tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

        assert reconstructed.task_store is not None
        completed_task = await reconstructed.task_store.load_task("task-handoff")
        completed_session = await reconstructed.session_store.load("session-handoff")
        assert completed_session is not None
        return (
            handed_off_task,
            handed_off_session.status,
            completed_task,
            completed_session.status,
            [
                (
                    str(record.event.payload["task_id"]),
                    str(record.event.payload["handoff_status"]),
                )
                for record in handoff_events
            ],
        )

    (
        handed_off_task,
        handed_off_status,
        completed_task,
        completed_status,
        handoff_evidence,
    ) = asyncio.run(scenario())
    assert handed_off_task is not None
    assert handed_off_status == SessionStatus.INTERRUPTED
    assert completed_task is not None
    assert completed_task.status == "completed"
    assert completed_status == SessionStatus.COMPLETED
    assert handoff_evidence == [
        ("task-handoff", "released"),
    ]


def test_terminal_peer_winner_during_handoff_does_not_stop_worker() -> None:
    class TerminalPeerStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            if request.task_id == "task-handoff-peer-winner":
                await self.terminalize_task(
                    TaskTerminalizationRequest(
                        task_id=request.task_id,
                        worker_id=request.worker_id,
                        lease_expires_at=request.lease_expires_at,
                        kind=TaskTerminalKind.COMPLETED,
                        result={"winner": "peer"},
                        idempotency_key="terminal-peer-winner",
                    )
                )
            return await super().release_interrupted_task_worker(request)

    task_store = TerminalPeerStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome | None:
        if task.id == "task-handoff-peer-winner":
            await task_store.attach_task(
                task.id,
                session_id="session-handoff-peer-winner",
                session_invocation=await stored_session_invocation(
                    app.session_store,
                    "session-handoff-peer-winner",
                ),
                worker_id=worker_id,
                lease_expires_at=task.lease_expires_at,
            )
            return TaskHandlerOutcome.SESSION_INTERRUPTED
        await task_store.complete_task(
            task.id,
            {"handled": True},
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return None

    async def scenario() -> tuple[int, Task | None, Task | None]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-handoff-peer-winner",
            session_id="session-handoff-peer-winner",
        )
        await task_store.create_task(TaskCreate(task_id="task-after-peer-winner", type="job"))
        handled = await run_task_worker(
            app,
            task_store,
            handler,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=2,
            poll_interval_s=0.01,
            reclaim=False,
        )
        return (
            handled,
            await task_store.load_task("task-handoff-peer-winner"),
            await task_store.load_task("task-after-peer-winner"),
        )

    handled, peer_winner, following = asyncio.run(scenario())
    assert handled == 2
    assert peer_winner is not None
    assert peer_winner.status is TaskStatus.COMPLETED
    assert peer_winner.result == {"winner": "peer"}
    assert following is not None
    assert following.status is TaskStatus.COMPLETED
    assert following.result == {"handled": True}


@pytest.mark.parametrize(
    ("failure_point", "expected_calls", "expected_statuses"),
    [
        ("before_commit", 2, ["pending", "recovered"]),
        ("after_commit", 1, ["pending", "recovered"]),
    ],
)
def test_interrupted_handoff_retries_or_reads_back_without_failing_task(
    tmp_path: Path,
    failure_point: str,
    expected_calls: int,
    expected_statuses: list[str],
) -> None:
    class FaultingHandoffStore(SQLiteTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.release_calls = 0

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_calls += 1
            if failure_point == "before_commit" and self.release_calls == 1:
                raise RuntimeError("transient handoff write failure")
            receipt = await super().release_interrupted_task_worker(request)
            if failure_point == "after_commit" and self.release_calls == 1:
                raise RuntimeError("handoff acknowledgement lost")
            return receipt

    session_store = SQLiteSessionStore(tmp_path / f"sessions-{failure_point}.sqlite")
    task_store = FaultingHandoffStore(tmp_path / f"tasks-{failure_point}.sqlite")
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-faulted-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-faulted-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, list[str]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-faulted-handoff",
            session_id="session-faulted-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        task = await task_store.load_task("task-faulted-handoff")
        events = await session_store.query_events(
            EventQuery(
                session_id="session-faulted-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return task, [str(record.event.payload["handoff_status"]) for record in events]

    task, statuses = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-faulted-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error is None
    assert task_store.release_calls == expected_calls
    assert statuses == expected_statuses


@pytest.mark.parametrize(
    ("failure_point", "expected_statuses"),
    [
        ("before_commit", []),
        ("after_commit", ["released"]),
    ],
)
def test_interrupted_handoff_event_failure_never_owns_task_release(
    failure_point: str,
    expected_statuses: list[str],
) -> None:
    class FaultingEventStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type != EventType.TASK_INTERRUPTED_HANDOFF:
                await super().append_event(session_id, event)
                return
            if failure_point == "before_commit":
                raise RuntimeError("handoff event unavailable before commit")
            await super().append_event(session_id, event)
            raise RuntimeError("handoff event acknowledgement lost")

    session_store = FaultingEventStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-event-failure-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-event-failure-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, list[str]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-event-failure-handoff",
            session_id="session-event-failure-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        events = await session_store.query_events(
            EventQuery(
                session_id="session-event-failure-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return (
            await task_store.load_task("task-event-failure-handoff"),
            [str(record.event.payload["handoff_status"]) for record in events],
        )

    task, statuses = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-event-failure-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error is None
    assert statuses == expected_statuses


def test_interrupted_handoff_releases_before_slow_event_publication() -> None:
    class SlowEventStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if (
                event.type is EventType.TASK_INTERRUPTED_HANDOFF
                and event.payload.get("handoff_status") == "released"
            ):
                released = await task_store.load_task("task-slow-event-handoff")
                assert released is not None
                assert released.worker_id is None
                assert released.lease_expires_at is None
                await asyncio.sleep(1.1)
            await super().append_event(session_id, event)

    session_store = SlowEventStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-slow-event-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-slow-event-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-slow-event-handoff",
            session_id="session-slow-event-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        return await task_store.load_task("task-slow-event-handoff")

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id is None
    assert task.lease_expires_at is None


def test_interrupted_handoff_does_not_attest_secret_bearing_task_id() -> None:
    secret = "handoff-task-id-secret"
    task_id = f"task-{secret}-value"
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-secret-task-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-secret-task-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, list[object]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id=task_id,
            session_id="session-secret-task-handoff",
        )
        assert (
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        events = await app.session_store.query_events(
            EventQuery(
                session_id="session-secret-task-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return await task_store.load_task(task_id), events

    task, events = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert events == []


def test_interrupted_handoff_rejects_malformed_store_receipt_without_retry() -> None:
    class MalformedReceiptStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_calls = 0

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            self.release_calls += 1
            return object()  # type: ignore[return-value]

    task_store = MalformedReceiptStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-malformed-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-malformed-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-malformed-handoff",
            session_id="session-malformed-handoff",
        )
        with pytest.raises(TaskInterruptedHandoffConflict, match="malformed receipt"):
            await run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        return await task_store.load_task("task-malformed-handoff")

    task = asyncio.run(scenario())
    assert task_store.release_calls == 1
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id == "worker-a"
    assert task.error is None


def test_interrupted_handoff_exhaustion_recovers_and_resumes_original_task(
    tmp_path: Path,
) -> None:
    class UnavailableHandoffStore(SQLiteTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            raise RuntimeError("handoff store unavailable")

    session_path = tmp_path / "sessions-recovery.sqlite"
    task_path = tmp_path / "tasks-recovery.sqlite"
    session_store = SQLiteSessionStore(session_path)
    failing_store = UnavailableHandoffStore(task_path)
    first_app = CayuApp(
        session_store=session_store,
        task_store=failing_store,
        enable_logging=False,
    )
    first_app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-recovered-change",
                        name="publish_change",
                        arguments={"change": "recovered-release"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    _register_approval_agent(first_app)

    async def handoff_handler(
        app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in app.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-recovery-handoff",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "Publish after durable recovery.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def first_process() -> tuple[Task, str, str, str]:
        await first_app.create_task(
            TaskCreate(
                task_id="task-recovery-handoff",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        with pytest.raises(RuntimeError, match="handoff store unavailable"):
            await run_task_worker(
                first_app,
                failing_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        task = await failing_store.load_task("task-recovery-handoff")
        assert task is not None
        pending = await session_store.query_pending_actions(
            PendingActionQuery(session_id="session-recovery-handoff")
        )
        assert len(pending.actions) == 1
        approval = pending.actions[0]
        assert approval.approval_id is not None
        assert approval.round_id is not None
        assert approval.tool_call_id is not None
        await failing_store.close()
        return (
            task,
            approval.approval_id,
            approval.round_id,
            approval.tool_call_id,
        )

    stranded, approval_id, approval_round_id, approval_tool_call_id = asyncio.run(first_process())
    assert stranded.status is TaskStatus.RUNNING
    assert stranded.session_id == "session-recovery-handoff"
    assert stranded.worker_id == "worker-a"
    assert stranded.lease_expires_at is not None
    assert stranded.error is None

    async def second_process() -> tuple[
        Task | None,
        SessionStatus,
        list[str],
    ]:
        remaining = max(
            (stranded.lease_expires_at - datetime.now(UTC)).total_seconds(),
            0,
        )
        await asyncio.sleep(remaining + 0.05)
        recovered_store = SQLiteTaskStore(task_path)
        recovered_sessions = SQLiteSessionStore(session_path)
        recovered_app = CayuApp(
            session_store=recovered_sessions,
            task_store=recovered_store,
            enable_logging=False,
        )
        recovered_app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("The recovered task is complete."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        _register_approval_agent(recovered_app)
        stop = asyncio.Event()

        async def stop_after_recovery() -> None:
            await asyncio.sleep(0.05)
            stop.set()

        async def unexpected_handler(
            _app: CayuApp,
            _task: Task,
            _worker_id: str,
        ) -> None:
            raise AssertionError("Recovered attached work must not re-enter the fresh queue.")

        await asyncio.gather(
            run_task_worker(
                recovered_app,
                recovered_store,
                unexpected_handler,
                worker_id="worker-b",
                query=TaskQuery(type="job"),
                poll_interval_s=0.01,
                reclaim=False,
                stop=stop,
                max_tasks=1,
            ),
            stop_after_recovery(),
        )
        recovered = await recovered_store.load_task("task-recovery-handoff")
        assert recovered is not None
        assert recovered.status is TaskStatus.RUNNING
        assert recovered.session_id == "session-recovery-handoff"
        assert recovered.worker_id is None
        assert recovered.lease_expires_at is None
        continuation = (
            await recovered_store.claim_interrupted_task_continuation(
                "worker-b",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert continuation is not None

        async for _event in recovered_app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="session-recovery-handoff",
                task_worker_id="worker-b",
                task_handoff_id=continuation.interrupted_handoff_id,
                approval_id=approval_id,
                tool_round_id=approval_round_id,
                tool_call_id=approval_tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

        task = await recovered_store.load_task("task-recovery-handoff")
        session = await recovered_sessions.load("session-recovery-handoff")
        assert session is not None
        events = await recovered_sessions.query_events(
            EventQuery(
                session_id="session-recovery-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        await recovered_store.close()
        return (
            task,
            session.status,
            [str(record.event.payload["handoff_status"]) for record in events],
        )

    completed, session_status, statuses = asyncio.run(second_process())
    assert completed is not None
    assert completed.status is TaskStatus.COMPLETED
    assert completed.session_id == "session-recovery-handoff"
    assert completed.worker_id is None
    assert completed.lease_expires_at is None
    assert session_status is SessionStatus.COMPLETED
    assert statuses == [
        "pending",
        "retrying",
        "retrying",
        "recovery_required",
        "recovered",
    ]


def test_interrupted_handoff_recovery_yields_to_fresh_work_between_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_worker_module, "_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE", 2)
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def attach_expiring_task(
        *,
        task_id: str,
        session_id: str,
        interrupted: bool,
    ) -> None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id=task_id,
            session_id=session_id,
        )
        if not interrupted:
            await app.session_store.update_status(session_id, SessionStatus.RUNNING)
        claimed = await task_store.claim_task("expired-worker", lease_seconds=1)
        assert claimed is not None
        assert claimed.id == task_id
        await task_store.attach_task(
            task_id,
            session_id=session_id,
            session_invocation=await stored_session_invocation(
                app.session_store,
                session_id,
            ),
            worker_id="expired-worker",
            lease_expires_at=claimed.lease_expires_at,
        )

    async def handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> None:
        await task_store.complete_task(
            task.id,
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
            result={"ok": True},
        )

    async def scenario() -> tuple[Task | None, Task | None, Task | None]:
        await attach_expiring_task(
            task_id="stale-ineligible-a",
            session_id="session-ineligible-a",
            interrupted=False,
        )
        await attach_expiring_task(
            task_id="stale-ineligible-b",
            session_id="session-ineligible-b",
            interrupted=False,
        )
        await attach_expiring_task(
            task_id="stale-interrupted",
            session_id="session-interrupted",
            interrupted=True,
        )
        await task_store.create_task(TaskCreate(task_id="trigger-a", type="trigger"))
        await asyncio.sleep(1.05)

        handled = await run_task_worker(
            app,
            task_store,
            handler,
            worker_id="recovery-worker",
            query=TaskQuery(type="trigger"),
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        assert handled == 1
        before_next_page = await task_store.load_task("stale-interrupted")
        fresh = await task_store.load_task("trigger-a")

        stop = asyncio.Event()

        async def stop_worker() -> None:
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(
            run_task_worker(
                app,
                task_store,
                handler,
                worker_id="recovery-worker",
                query=TaskQuery(type="trigger"),
                poll_interval_s=0.01,
                reclaim=False,
                stop=stop,
            ),
            stop_worker(),
        )
        return (
            await task_store.load_task("stale-interrupted"),
            before_next_page,
            fresh,
        )

    recovered, before_next_page, fresh = asyncio.run(scenario())
    assert before_next_page is not None
    assert before_next_page.worker_id == "expired-worker"
    assert fresh is not None and fresh.status is TaskStatus.COMPLETED
    assert recovered is not None
    assert recovered.worker_id is None
    assert recovered.lease_expires_at is None


def test_interrupted_handoff_recovery_stops_between_candidates() -> None:
    stop = asyncio.Event()

    class StoppingSessionStore(InMemorySessionStore):
        armed = False
        recovery_loads = 0

        async def load(self, session_id: str):
            session = await super().load(session_id)
            if self.armed:
                self.recovery_loads += 1
                stop.set()
            return session

    session_store = StoppingSessionStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    fresh_calls: list[str] = []

    async def fresh_handler(
        _app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> None:
        fresh_calls.append(task.id)

    async def seed(task_id: str) -> None:
        session_id = f"session-{task_id}"
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id=task_id,
            session_id=session_id,
        )
        claimed = await task_store.claim_task("expired-worker", lease_seconds=1)
        assert claimed is not None and claimed.id == task_id
        await task_store.attach_task(
            task_id,
            session_id=session_id,
            session_invocation=await stored_session_invocation(
                session_store,
                session_id,
            ),
            worker_id="expired-worker",
            lease_expires_at=claimed.lease_expires_at,
        )

    async def scenario() -> int:
        await seed("stop-recovery-a")
        await seed("stop-recovery-b")
        await task_store.create_task(TaskCreate(task_id="fresh-after-stop", type="fresh"))
        await asyncio.sleep(1.05)
        session_store.armed = True
        return await run_task_worker(
            app,
            task_store,
            fresh_handler,
            worker_id="recovery-worker",
            query=TaskQuery(type="fresh"),
            poll_interval_s=0.01,
            reclaim=False,
            stop=stop,
        )

    assert asyncio.run(scenario()) == 0
    assert session_store.recovery_loads == 1
    assert fresh_calls == []


def test_continuation_execution_is_independent_of_recovery_scanner_election() -> None:
    class CountingRecoveryStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.recovery_scans = 0

        async def list_expired_interrupted_task_handoff_candidates(
            self,
            *,
            after: tuple[datetime, str] | None = None,
            limit: int = 100,
        ) -> list[Task]:
            self.recovery_scans += 1
            return await super().list_expired_interrupted_task_handoff_candidates(
                after=after,
                limit=limit,
            )

    task_store = CountingRecoveryStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    continued: list[str] = []

    async def fresh_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> None:
        raise AssertionError("A continuation must not enter the fresh queue.")

    async def continuation_handler(
        _app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> TaskHandlerOutcome:
        continued.append(task.id)
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> int:
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="executor-without-scanner",
            session_id="executor-without-scanner-session",
        )
        return await run_task_worker(
            app,
            task_store,
            fresh_handler,
            worker_id="continuation-executor",
            query=TaskQuery(type="job"),
            poll_interval_s=0.01,
            reclaim=False,
            recover_interrupted_handoffs=False,
            recovered_interrupted_task_handler=continuation_handler,
            max_tasks=1,
        )

    assert asyncio.run(scenario()) == 1
    assert continued == ["executor-without-scanner"]
    assert task_store.recovery_scans == 0


def test_all_configured_continuation_executors_can_run_concurrently() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    all_started = asyncio.Event()
    started: list[tuple[str, str]] = []

    async def fresh_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> None:
        raise AssertionError("A continuation must not enter the fresh queue.")

    async def continuation_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        started.append((task.id, worker_id))
        if len(started) == 2:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> list[int]:
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="concurrent-continuation-a",
            session_id="concurrent-continuation-session-a",
        )
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="concurrent-continuation-b",
            session_id="concurrent-continuation-session-b",
        )
        return await asyncio.gather(
            run_task_worker(
                app,
                task_store,
                fresh_handler,
                worker_id="continuation-executor-a",
                query=TaskQuery(type="job"),
                poll_interval_s=0.01,
                reclaim=False,
                recover_interrupted_handoffs=False,
                recovered_interrupted_task_handler=continuation_handler,
                max_tasks=1,
            ),
            run_task_worker(
                app,
                task_store,
                fresh_handler,
                worker_id="continuation-executor-b",
                query=TaskQuery(type="job"),
                poll_interval_s=0.01,
                reclaim=False,
                recover_interrupted_handoffs=False,
                recovered_interrupted_task_handler=continuation_handler,
                max_tasks=1,
            ),
        )

    assert asyncio.run(scenario()) == [1, 1]
    assert {task_id for task_id, _worker_id in started} == {
        "concurrent-continuation-a",
        "concurrent-continuation-b",
    }
    assert {worker_id for _task_id, worker_id in started} == {
        "continuation-executor-a",
        "continuation-executor-b",
    }


@pytest.mark.parametrize("authority_error", [SessionRunFenced, TaskClaimLost])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recovered_continuation_authority_loss_releases_for_retry(
    tmp_path: Path,
    authority_error: type[Exception],
    backend: str,
) -> None:
    task_store: TaskStore = (
        InMemoryTaskStore()
        if backend == "memory"
        else SQLiteTaskStore(tmp_path / f"continuation-{authority_error.__name__}.sqlite")
    )
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def fresh_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> None:
        raise AssertionError("A continuation must not enter the fresh queue.")

    async def fenced_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> None:
        raise authority_error("Recovered continuation lost durable authority.")

    async def scenario() -> tuple[int, Task | None, InterruptedTaskContinuationClaimPage]:
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="fenced-recovered-continuation",
            session_id="fenced-recovered-continuation-session",
        )
        handled = await run_task_worker(
            app,
            task_store,
            fresh_handler,
            worker_id="recovery-worker",
            query=TaskQuery(type="job"),
            poll_interval_s=0.01,
            reclaim=False,
            recover_interrupted_handoffs=False,
            recovered_interrupted_task_handler=fenced_handler,
            max_tasks=1,
        )
        released = await task_store.load_task("fenced-recovered-continuation")
        reclaimed = await task_store.claim_interrupted_task_continuation(
            "retry-worker",
            TaskQuery(type="job"),
            handoff_id=str(uuid4()),
        )
        return handled, released, reclaimed

    handled, released, reclaimed = asyncio.run(scenario())
    assert handled == 1
    assert released is not None
    assert released.status is TaskStatus.RUNNING
    assert released.worker_id is None
    assert released.lease_expires_at is None
    assert released.interrupted_handoff_id is not None
    assert released.error is None
    assert reclaimed.task is not None
    assert reclaimed.task.id == released.id
    assert reclaimed.task.worker_id == "retry-worker"
    assert reclaimed.task.interrupted_handoff_id is not None
    assert reclaimed.task.interrupted_handoff_id != released.interrupted_handoff_id


def test_recovered_continuation_session_fence_defers_to_expired_owner_recovery() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def fresh_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> None:
        raise AssertionError("A continuation must not enter the fresh queue.")

    async def peer_won_session_handler(
        recovery_app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> None:
        assert task.session_id is not None
        await recovery_app.session_store.update_status(
            task.session_id,
            SessionStatus.RUNNING,
        )
        raise SessionRunFenced("A peer session epoch won continuation admission.")

    async def scenario() -> tuple[Task | None, Task | None]:
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="peer-fenced-recovered-continuation",
            session_id="peer-fenced-recovered-continuation-session",
        )
        handled = await run_task_worker(
            app,
            task_store,
            fresh_handler,
            worker_id="recovery-worker",
            query=TaskQuery(type="job"),
            lease_seconds=1,
            poll_interval_s=0.01,
            reclaim=False,
            recover_interrupted_handoffs=False,
            recovered_interrupted_task_handler=peer_won_session_handler,
            max_tasks=1,
        )
        assert handled == 1
        deferred = await task_store.load_task("peer-fenced-recovered-continuation")
        assert deferred is not None
        assert deferred.worker_id == "recovery-worker"
        assert deferred.lease_expires_at is not None
        await asyncio.sleep(1.05)
        recovery = await task_worker_module._recover_expired_interrupted_task_handoffs(
            app,
            task_store,
            after=None,
            limit=10,
            stop=None,
        )
        assert recovery.recovered == 1
        reclaimed = await task_store.claim_interrupted_task_continuation(
            "retry-worker",
            TaskQuery(type="job"),
            handoff_id=str(uuid4()),
        )
        return deferred, reclaimed.task

    deferred, reclaimed = asyncio.run(scenario())
    assert deferred is not None
    assert reclaimed is not None
    assert reclaimed.id == deferred.id
    assert reclaimed.worker_id == "retry-worker"
    assert reclaimed.interrupted_handoff_id is not None
    assert reclaimed.interrupted_handoff_id != deferred.interrupted_handoff_id


def test_recovered_continuation_respects_worker_limit_before_fresh_claim() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    fresh_calls: list[str] = []

    async def fresh_handler(
        _app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> None:
        fresh_calls.append(task.id)

    async def continuation_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> TaskHandlerOutcome:
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[int, Task | None]:
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="limit-continuation",
            session_id="limit-continuation-session",
        )
        await task_store.create_task(TaskCreate(task_id="fresh-after-limit", type="job"))
        handled = await run_task_worker(
            app,
            task_store,
            fresh_handler,
            worker_id="limit-worker",
            query=TaskQuery(type="job"),
            poll_interval_s=0.01,
            reclaim=False,
            recover_interrupted_handoffs=False,
            recovered_interrupted_task_handler=continuation_handler,
            max_tasks=1,
        )
        return handled, await task_store.load_task("fresh-after-limit")

    handled, fresh = asyncio.run(scenario())
    assert handled == 1
    assert fresh_calls == []
    assert fresh is not None and fresh.status is TaskStatus.PENDING


def test_continuation_rediscovery_uses_configured_poll_objective() -> None:
    first_scan = asyncio.Event()

    class DelayedContinuationStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.continuation_scans = 0

        async def claim_interrupted_task_continuation(
            self,
            worker_id: str,
            query: TaskQuery | None = None,
            *,
            handoff_id: str,
            lease_seconds: int = 300,
            after: tuple[datetime, str] | None = None,
            scan_limit: int = 100,
        ) -> InterruptedTaskContinuationClaimPage:
            self.continuation_scans += 1
            if self.continuation_scans == 1:
                first_scan.set()
                return InterruptedTaskContinuationClaimPage(
                    scanned_candidates=0,
                    rejected_candidates=0,
                    exhausted=True,
                )
            return await super().claim_interrupted_task_continuation(
                worker_id,
                query,
                handoff_id=handoff_id,
                lease_seconds=lease_seconds,
                after=after,
                scan_limit=scan_limit,
            )

    task_store = DelayedContinuationStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def fresh_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> None:
        raise AssertionError("A continuation must not enter the fresh queue.")

    async def continuation_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> TaskHandlerOutcome:
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> int:
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="poll-objective-continuation",
            session_id="poll-objective-session",
        )
        return await asyncio.wait_for(
            run_task_worker(
                app,
                task_store,
                fresh_handler,
                worker_id="poll-objective-worker",
                query=TaskQuery(type="job"),
                poll_interval_s=1.0,
                reclaim=False,
                recover_interrupted_handoffs=False,
                recovered_interrupted_task_handler=continuation_handler,
                continuation_poll_interval_s=0.02,
                max_tasks=1,
            ),
            timeout=0.5,
        )

    assert asyncio.run(scenario()) == 1
    assert first_scan.is_set()
    assert task_store.continuation_scans == 2


def test_expired_attached_session_is_recovered_handed_off_and_resumed() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    resumed: list[str] = []

    async def fresh_handler(
        _app: CayuApp,
        _task: Task,
        _worker_id: str,
    ) -> None:
        raise AssertionError("An attached task must not re-enter the fresh queue.")

    async def resume_handler(
        recovery_app: CayuApp,
        task: Task,
        recovery_worker_id: str,
    ) -> None:
        assert task.session_id is not None
        resumed.append(task.id)
        async for _event in recovery_app.resume(
            ResumeRequest(
                session_id=task.session_id,
                task_worker_id=recovery_worker_id,
                task_handoff_id=task.interrupted_handoff_id,
                messages=[Message.text("user", "Continue the attached task.")],
            )
        ):
            pass

    async def scenario() -> tuple[int, Task | None, SessionStatus]:
        created = await task_store.create_task(
            TaskCreate(task_id="expired-attached-task", type="job")
        )
        await app.session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="expired-attached-session",
                    task_id=created.id,
                    messages=[Message.text("user", "Original task")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                TaskInvocationSnapshot(
                    id=created.id,
                    session_id=created.session_id,
                    invocation=created.invocation,
                ),
            ),
            identity=profiled_session_identity(
                provider_name=provider.name,
                model="scripted-model",
                agent_name="worker-agent",
                app=app,
            ),
        )
        await admit_test_invocation(
            app.session_store,
            "expired-attached-session",
            interaction_started_event=runtime_interaction_started_event(
                app,
                session_id="expired-attached-session",
                interaction_id="expired-attached-interaction",
                agent_name="worker-agent",
            ),
            interaction_source_messages=(Message.text("user", "Original task"),),
        )
        claimed = await task_store.claim_task("dead-worker", lease_seconds=1)
        assert claimed is not None
        await task_store.attach_task(
            claimed.id,
            session_id="expired-attached-session",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "expired-attached-session",
            ),
            worker_id="dead-worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        await asyncio.sleep(1.05)

        handled = await asyncio.wait_for(
            run_task_worker(
                app,
                task_store,
                fresh_handler,
                worker_id="recovery-worker",
                query=TaskQuery(type="job"),
                poll_interval_s=0.01,
                reclaim=False,
                recovered_interrupted_task_handler=resume_handler,
                max_tasks=1,
            ),
            timeout=15,
        )
        task = await task_store.load_task("expired-attached-task")
        session = await app.session_store.load("expired-attached-session")
        assert session is not None
        return handled, task, session.status

    handled, task, session_status = asyncio.run(scenario())
    assert handled == 1
    assert resumed == ["expired-attached-task"]
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert session_status is SessionStatus.COMPLETED


def test_continuation_session_race_isolated_with_claimed_lease(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SessionRacingStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True
        app: CayuApp

        async def claim_interrupted_task_continuation(
            self,
            worker_id: str,
            query: TaskQuery | None = None,
            *,
            handoff_id: str,
            lease_seconds: int = 300,
            after: tuple[datetime, str] | None = None,
            scan_limit: int = 100,
        ) -> InterruptedTaskContinuationClaimPage:
            page = await super().claim_interrupted_task_continuation(
                worker_id,
                query,
                handoff_id=handoff_id,
                lease_seconds=lease_seconds,
                after=after,
                scan_limit=scan_limit,
            )
            claimed = page.task
            if claimed is not None:
                assert claimed.session_id is not None
                await self.app.session_store.update_status(
                    claimed.session_id,
                    SessionStatus.RUNNING,
                )
            return page

    task_store = SessionRacingStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    task_store.app = app
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
    handler_calls: list[str] = []

    async def unexpected_handler(
        _app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> None:
        handler_calls.append(task.id)

    async def scenario() -> tuple[int, Task | None]:
        await _seed_receipt_backed_continuation(
            app,
            task_store,
            task_id="session-race-continuation",
            session_id="session-race-continuation-session",
        )
        handled = await run_task_worker(
            app,
            task_store,
            unexpected_handler,
            worker_id="recovery-owner",
            query=TaskQuery(type="job"),
            poll_interval_s=0.01,
            reclaim=False,
            recover_interrupted_handoffs=False,
            recovered_interrupted_task_handler=unexpected_handler,
            max_tasks=1,
        )
        return handled, await task_store.load_task("session-race-continuation")

    handled, task = asyncio.run(scenario())
    assert handled == 1
    assert handler_calls == []
    assert task is not None
    assert task.worker_id == "recovery-owner"
    assert task.lease_expires_at is not None
    assert "continuation session changed after claim" in caplog.text


def test_interrupted_handoff_recovery_skips_new_cancellation_request() -> None:
    class CancellationWinningStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def recover_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            await self.cancel_task(request.task_id, {"code": "operator_cancelled"})
            return await super().recover_interrupted_task_worker(request)

    task_store = CancellationWinningStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def scenario() -> tuple[Task | None, Task | None]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancellation-winner",
            session_id="session-cancellation-winner",
        )
        claimed = await task_store.claim_task("expired-worker", lease_seconds=1)
        assert claimed is not None
        await task_store.attach_task(
            claimed.id,
            session_id="session-cancellation-winner",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancellation-winner",
            ),
            worker_id="expired-worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        await task_store.create_task(TaskCreate(task_id="fresh-after-cancel", type="fresh"))
        await asyncio.sleep(1.05)

        async def handler(
            _app: CayuApp,
            task: Task,
            worker_id: str,
        ) -> None:
            await task_store.complete_task(
                task.id,
                {"ok": True},
                worker_id=worker_id,
                lease_expires_at=task.lease_expires_at,
            )

        assert (
            await run_task_worker(
                app,
                task_store,
                handler,
                worker_id="recovery-worker",
                query=TaskQuery(type="fresh"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
            == 1
        )
        return (
            await task_store.load_task("task-cancellation-winner"),
            await task_store.load_task("fresh-after-cancel"),
        )

    cancelled, fresh = asyncio.run(scenario())
    assert cancelled is not None
    assert cancelled.status is TaskStatus.RUNNING
    assert cancelled.status_reason == "cancellation_requested"
    assert fresh is not None
    assert fresh.status is TaskStatus.COMPLETED


def test_operator_cancellation_winning_live_handoff_is_terminalized() -> None:
    class CancellationWinningStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.cancellation_key: str | None = None

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            cancellation = await self.cancel_task(
                request.task_id,
                {"code": "operator_cancelled_during_handoff"},
            )
            assert cancellation.status_payload is not None
            key = cancellation.status_payload["terminalization_idempotency_key"]
            assert isinstance(key, str)
            self.cancellation_key = key
            return await super().release_interrupted_task_worker(request)

    task_store = CancellationWinningStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-live-cancellation-winner",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-live-cancellation-winner",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[int, Task | None, TaskTerminalizationReceipt | None]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-live-cancellation-winner",
            session_id="session-live-cancellation-winner",
        )
        handled = await run_task_worker(
            app,
            task_store,
            handoff_handler,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.01,
            reclaim=False,
        )
        terminal = await task_store.load_task("task-live-cancellation-winner")
        assert terminal is not None
        assert terminal.status_payload is None
        assert task_store.cancellation_key is not None
        return (
            handled,
            terminal,
            await task_store.load_task_terminalization_receipt(
                terminal.id,
                task_store.cancellation_key,
            ),
        )

    handled, task, receipt = asyncio.run(scenario())
    assert handled == 1
    assert task is not None
    assert task.status is TaskStatus.CANCELLED
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error == {"code": "operator_cancelled_during_handoff"}
    assert receipt is not None
    assert receipt.kind is TaskTerminalKind.CANCELLED
    assert receipt.task == task


def test_interrupted_handoff_recovery_retains_bounded_cursor_between_idle_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_worker_module, "_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE", 2)

    class CountingRecoveryStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.recovery_list_calls = 0

        async def list_expired_interrupted_task_handoff_candidates(
            self,
            *,
            after: tuple[datetime, str] | None = None,
            limit: int = 100,
        ) -> list[Task]:
            self.recovery_list_calls += 1
            return await super().list_expired_interrupted_task_handoff_candidates(
                after=after,
                limit=limit,
            )

    task_store = CountingRecoveryStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def attach_ineligible(task_id: str) -> None:
        session_id = f"session-{task_id}"
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id=task_id,
            session_id=session_id,
        )
        await app.session_store.update_status(session_id, SessionStatus.RUNNING)
        claimed = await task_store.claim_task("expired-worker", lease_seconds=1)
        assert claimed is not None
        await task_store.attach_task(
            task_id,
            session_id=session_id,
            session_invocation=await stored_session_invocation(
                app.session_store,
                session_id,
            ),
            worker_id="expired-worker",
            lease_expires_at=claimed.lease_expires_at,
        )

    async def scenario() -> None:
        await attach_ineligible("stale-bounded-a")
        await attach_ineligible("stale-bounded-b")
        await asyncio.sleep(1.05)
        stop = asyncio.Event()

        async def stop_worker() -> None:
            await asyncio.sleep(0.1)
            stop.set()

        async def unexpected_handler(
            _app: CayuApp,
            _task: Task,
            _worker_id: str,
        ) -> None:
            raise AssertionError("No fresh task should be claimed.")

        await asyncio.gather(
            run_task_worker(
                app,
                task_store,
                unexpected_handler,
                worker_id="recovery-worker",
                query=TaskQuery(type="fresh"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
                stop=stop,
            ),
            stop_worker(),
        )

    asyncio.run(scenario())
    assert task_store.recovery_list_calls == 2


def test_interrupted_handoff_cancellation_waits_for_dispatched_release() -> None:
    class BlockingHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_started.set()
            await self.allow_release.wait()
            return await super().release_interrupted_task_worker(request)

    task_store = BlockingHandoffStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-cancelled-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancelled-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, int, bool, tuple[object, ...], list[str]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancelled-handoff",
            session_id="session-cancelled-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await task_store.release_started.wait()
        worker_task.cancel("stop interrupted handoff")
        await asyncio.sleep(0)
        assert not worker_task.done()
        task_store.allow_release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        task = await task_store.load_task("task-cancelled-handoff")
        events = await app.session_store.query_events(
            EventQuery(
                session_id="session-cancelled-handoff",
                event_types=(EventType.TASK_INTERRUPTED_HANDOFF,),
            )
        )
        return (
            task,
            worker_task.cancelling(),
            worker_task.cancelled(),
            raised.value.args,
            [str(record.event.payload["handoff_status"]) for record in events],
        )

    task, cancelling, cancelled, cancellation_args, statuses = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-cancelled-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error is None
    assert cancelling == 1
    assert cancelled is True
    assert cancellation_args == ("stop interrupted handoff",)
    assert statuses == []


def test_interrupted_handoff_pending_cancellation_owns_dispatched_release() -> None:
    class BlockingHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_started.set()
            await self.allow_release.wait()
            return await super().release_interrupted_task_worker(request)

    task_store = BlockingHandoffStore()
    app = CayuApp(task_store=task_store, enable_logging=False)

    async def scenario() -> tuple[Task | None, int, bool, tuple[object, ...]]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-pending-cancelled-handoff",
            session_id="session-pending-cancelled-handoff",
        )
        claimed = await task_store.claim_task("worker-a", TaskQuery(type="job"))
        assert claimed is not None
        attached = await task_store.attach_task(
            claimed.id,
            session_id="session-pending-cancelled-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-pending-cancelled-handoff",
            ),
            worker_id="worker-a",
            lease_expires_at=claimed.lease_expires_at,
        )
        request = task_worker_module.interrupted_task_handoff_request(
            attached,
            session_run_epoch=1,
        )

        async def settle_with_pending_cancellation() -> TaskInterruptedHandoffReceipt:
            owner = asyncio.current_task()
            assert owner is not None
            owner.cancel("cancel before interrupted handoff settlement")
            return await task_worker_module._settle_interrupted_task_handoff_once(
                app,
                task_store,
                request,
                recover_expired=False,
            )

        settlement = asyncio.create_task(settle_with_pending_cancellation())
        await task_store.release_started.wait()
        await asyncio.sleep(0)
        assert not settlement.done()
        task_store.allow_release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await settlement
        return (
            await task_store.load_task(attached.id),
            settlement.cancelling(),
            settlement.cancelled(),
            raised.value.args,
        )

    task, cancelling, cancelled, cancellation_args = asyncio.run(scenario())
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.session_id == "session-pending-cancelled-handoff"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert cancelling == 1
    assert cancelled is True
    assert cancellation_args == ("cancel before interrupted handoff settlement",)


def test_interrupted_handoff_cancellation_sanitizes_concurrent_store_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "handoff-store-secret-canary"

    class BlockingFailureStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_failure = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            self.release_started.set()
            await self.allow_failure.wait()
            raise RuntimeError(f"store cleanup exposed {secret}")

    task_store = BlockingFailureStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-cancelled-failing-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancelled-failing-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[asyncio.CancelledError, Task | None, int, bool]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancelled-failing-handoff",
            session_id="session-cancelled-failing-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await task_store.release_started.wait()
        worker_task.cancel("cancel handoff with failing cleanup")
        await asyncio.sleep(0)
        assert not worker_task.done()
        task_store.allow_failure.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        return (
            raised.value,
            await task_store.load_task("task-cancelled-failing-handoff"),
            worker_task.cancelling(),
            worker_task.cancelled(),
        )

    with caplog.at_level("DEBUG"):
        cancellation, task, cancelling, cancelled = asyncio.run(scenario())
    rendered: list[str] = []
    pending: list[BaseException] = [cancellation]
    seen: set[int] = set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        rendered.extend((str(failure), repr(failure)))
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)

    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id == "worker-a"
    assert task.error is None
    assert cancelling == 1
    assert cancelled is True
    assert REDACTED_SECRET in " ".join(rendered)
    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            *rendered,
            captured.out,
            captured.err,
            *[record.getMessage() for record in caplog.records],
            *[str(warning.message) for warning in recwarn],
        ]
    )
    assert secret not in diagnostics


def test_interrupted_handoff_event_cancellation_drops_store_failure_context(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "handoff-event-context-secret-canary"

    class FailingHandoffStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            del request
            raise RuntimeError(f"store handoff exposed {secret}")

    class BlockingEventStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.publication_started = asyncio.Event()

        async def append_event(self, session_id: str, event: Event) -> None:
            if (
                event.type is EventType.TASK_INTERRUPTED_HANDOFF
                and event.payload.get("handoff_status") == "pending"
            ):
                self.publication_started.set()
                await asyncio.Event().wait()
            await super().append_event(session_id, event)

    task_store = FailingHandoffStore()
    session_store = BlockingEventStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-event-cancelled-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-event-cancelled-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-event-cancelled-handoff",
            session_id="session-event-cancelled-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await session_store.publication_started.wait()
        worker_task.cancel("cancel pending handoff event")
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        return raised.value, worker_task.cancelling(), worker_task.cancelled()

    with caplog.at_level("DEBUG"):
        cancellation, cancelling, cancelled = asyncio.run(scenario())
    rendered: list[str] = []
    pending: list[BaseException] = [cancellation]
    seen: set[int] = set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        rendered.extend((str(failure), repr(failure)))
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("cancel pending handoff event",)
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            *rendered,
            captured.out,
            captured.err,
            *[record.getMessage() for record in caplog.records],
            *[str(warning.message) for warning in recwarn],
        ]
    )
    assert secret not in diagnostics


def test_interrupted_handoff_cancellation_validates_commit_receipt_before_redelivery(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "handoff-malformed-receipt-secret-canary"

    class SecretBearingValue:
        def __eq__(self, _other: object) -> bool:
            raise RuntimeError(secret)

        def __repr__(self) -> str:
            return secret

    class BlockingMalformedReceiptStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def release_interrupted_task_worker(
            self,
            request: TaskInterruptedHandoffRequest,
        ) -> TaskInterruptedHandoffReceipt:
            self.release_started.set()
            await self.allow_release.wait()
            receipt = await super().release_interrupted_task_worker(request)
            object.__setattr__(receipt, "request", SecretBearingValue())
            return receipt

    task_store = BlockingMalformedReceiptStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def handoff_handler(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-cancelled-malformed-handoff",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "session-cancelled-malformed-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[asyncio.CancelledError, Task | None, int, bool]:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="task-cancelled-malformed-handoff",
            session_id="session-cancelled-malformed-handoff",
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handoff_handler,
                worker_id="worker-a",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.01,
                reclaim=False,
            )
        )
        await task_store.release_started.wait()
        worker_task.cancel("cancel handoff with malformed receipt")
        await asyncio.sleep(0)
        assert not worker_task.done()
        task_store.allow_release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker_task
        return (
            raised.value,
            await task_store.load_task("task-cancelled-malformed-handoff"),
            worker_task.cancelling(),
            worker_task.cancelled(),
        )

    with caplog.at_level("DEBUG"):
        cancellation, task, cancelling, cancelled = asyncio.run(scenario())
    rendered: list[str] = []
    pending: list[BaseException] = [cancellation]
    seen: set[int] = set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        rendered.extend((str(failure), repr(failure)))
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)

    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert cancelling == 1
    assert cancelled is True
    assert "malformed receipt evidence" in " ".join(rendered)
    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            *rendered,
            captured.out,
            captured.err,
            *[record.getMessage() for record in caplog.records],
            *[str(warning.message) for warning in recwarn],
        ]
    )
    assert secret not in diagnostics


@pytest.mark.parametrize(
    "session_status",
    [
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
    ],
)
def test_run_task_worker_fails_handoff_when_session_is_not_interrupted(
    tmp_path: Path,
    session_status: SessionStatus,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def invalid_handoff(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await task_store.attach_task(
            task.id,
            session_id="session-invalid-handoff",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-invalid-handoff",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await task_store.create_task(TaskCreate(type="job"))
        await session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="session-invalid-handoff",
                    task_id=created.id,
                    messages=[Message.text("user", "original")],
                ),
                TaskInvocationSnapshot(
                    id=created.id,
                    session_id=created.session_id,
                    invocation=created.invocation,
                ),
            ),
            identity=SessionIdentity(
                provider_name="scripted",
                model="scripted-model",
            ),
        )
        if session_status is not SessionStatus.PENDING:
            await session_store.update_status("session-invalid-handoff", session_status)
        await run_task_worker(
            app,
            task_store,
            invalid_handoff,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await task_store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": (
            "Task handler requested an interrupted-session handoff while session "
            f"session-invalid-handoff was {session_status}."
        ),
    }


def test_run_task_worker_fails_handoff_for_missing_attached_session(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def missing_session_handoff(
        _app: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        await store.attach_task(
            task.id,
            session_id="session-missing",
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                "session-missing",
            ),
            worker_id=worker_id,
            lease_expires_at=task.lease_expires_at,
        )
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            missing_session_handoff,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "failed"
    assert task.error == {
        "error": "RuntimeError",
        "message": "Attached session not found: session-missing.",
    }


def test_run_task_worker_preserves_terminal_task_before_handoff_cleanup(tmp_path: Path) -> None:
    app, store = _build(tmp_path)

    async def terminal_race(
        _app: CayuApp,
        task: Task,
        _worker_id: str,
    ) -> TaskHandlerOutcome:
        await store.complete_task(task.id, {"winner": "terminal-state"})
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> Task | None:
        created = await store.create_task(TaskCreate(type="job"))
        await run_task_worker(
            app,
            store,
            terminal_race,
            worker_id="worker-a",
            query=TaskQuery(type="job"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        return await store.load_task(created.id)

    task = asyncio.run(scenario())
    assert task is not None
    assert task.status == "completed"
    assert task.result == {"winner": "terminal-state"}


def test_resume_rejects_stale_or_missing_worker_authority_before_provider(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "fenced-sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "fenced-tasks.sqlite")
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def collect(worker_id: str | None) -> list[Event]:
        return [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="fenced-resume-session",
                    task_worker_id=worker_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

    async def scenario() -> Task | None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="fenced-resume-task",
            session_id="fenced-resume-session",
        )
        prior = await task_store.claim_task("stale-worker", lease_seconds=30)
        assert prior is not None
        attached = await task_store.attach_task(
            prior.id,
            session_id="fenced-resume-session",
            session_invocation=await stored_session_invocation(
                session_store,
                "fenced-resume-session",
            ),
            worker_id="stale-worker",
            lease_expires_at=prior.lease_expires_at,
        )
        session = await session_store.load("fenced-resume-session")
        assert session is not None
        await task_store.release_interrupted_task_worker(
            interrupted_task_handoff_request(
                attached,
                session_run_epoch=session.run_epoch,
            )
        )
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None

        with pytest.raises(TaskClaimLost):
            await collect("stale-worker")
        with pytest.raises(TaskClaimLost):
            await collect(None)
        assert provider.requests == []
        return await task_store.load_task("fenced-resume-task")

    task = asyncio.run(scenario())
    assert task is not None and task.status is TaskStatus.RUNNING
    assert task.worker_id == "elected-worker"
    assert provider.requests == []


@pytest.mark.parametrize(
    "continuation_kind",
    [
        "user_input",
        "user_input_recovery",
        "tool_approval",
        "tool_approval_recovery",
        "tool_round_recovery",
        "provider_operation",
    ],
)
@pytest.mark.parametrize("worker_id", [None, "stale-worker"])
def test_typed_continuations_require_elected_worker_authority_before_execution(
    continuation_kind: str,
    worker_id: str | None,
) -> None:
    task_store = InMemoryTaskStore()
    provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def invoke_continuation() -> None:
        if continuation_kind == "user_input":
            stream = app.resolve_user_input(
                UserInputResponse(
                    session_id="typed-continuation-session",
                    task_worker_id=worker_id,
                    input_id="input-id",
                    answer="answer",
                )
            )
        elif continuation_kind == "user_input_recovery":
            stream = app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id="typed-continuation-session",
                    task_worker_id=worker_id,
                    input_id="input-id",
                    answer="answer",
                    tool_call_id="tool-call-id",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="completed externally",
                )
            )
        elif continuation_kind == "tool_approval":
            stream = app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="typed-continuation-session",
                    task_worker_id=worker_id,
                    approval_id="approval-id",
                    tool_round_id="tool-round-id",
                    tool_call_id="tool-call-id",
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        elif continuation_kind == "tool_approval_recovery":
            stream = app.recover_tool_approval(
                ToolApprovalRecoveryRequest(
                    session_id="typed-continuation-session",
                    task_worker_id=worker_id,
                    approval_id="approval-id",
                    tool_round_id="tool-round-id",
                    tool_call_id="tool-call-id",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="completed externally",
                )
            )
        elif continuation_kind == "tool_round_recovery":
            stream = app.recover_tool_round(
                ToolRoundRecoveryRequest(
                    session_id="typed-continuation-session",
                    task_worker_id=worker_id,
                    round_id="tool-round-id",
                    tool_call_id="tool-call-id",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="completed externally",
                )
            )
        else:
            stream = app.resolve_provider_operation(
                ProviderOperationResolutionRequest(
                    session_id="typed-continuation-session",
                    task_worker_id=worker_id,
                    stage_id="provider-stage-id",
                    expected_run_epoch=0,
                    action=ProviderOperationResolutionAction.FAIL,
                )
            )
        async for _event in stream:
            pass

    async def scenario() -> Task | None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="typed-continuation-task",
            session_id="typed-continuation-session",
        )
        prior = await task_store.claim_task("prior-worker", lease_seconds=30)
        assert prior is not None
        attached = await task_store.attach_task(
            prior.id,
            session_id="typed-continuation-session",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "typed-continuation-session",
            ),
            worker_id="prior-worker",
            lease_expires_at=prior.lease_expires_at,
        )
        session = await app.session_store.load("typed-continuation-session")
        assert session is not None
        await task_store.release_interrupted_task_worker(
            interrupted_task_handoff_request(
                attached,
                session_run_epoch=session.run_epoch,
            )
        )
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None

        with pytest.raises(TaskClaimLost):
            await invoke_continuation()
        return await task_store.load_task("typed-continuation-task")

    task = asyncio.run(scenario())
    assert task is not None and task.status is TaskStatus.RUNNING
    assert task.worker_id == "elected-worker"
    assert provider.requests == []


def test_typed_continuation_rejects_secret_bearing_worker_authority_before_provider() -> None:
    secret = "typed-continuation-worker-secret-canary"
    provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="task continuation authority"):
            async for _event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="secret-worker-continuation",
                    task_worker_id=f"worker-{secret}",
                    approval_id="approval-id",
                    tool_round_id="tool-round-id",
                    tool_call_id="tool-call-id",
                    decision=ToolApprovalDecision.APPROVE,
                )
            ):
                pass

    asyncio.run(scenario())
    assert provider.requests == []


def test_provider_resolution_digest_excludes_transient_task_worker_authority() -> None:
    def request(worker_id: str) -> ProviderOperationResolutionRequest:
        return ProviderOperationResolutionRequest(
            session_id="provider-resolution-worker-handoff",
            task_worker_id=worker_id,
            stage_id="provider-stage-id",
            expected_run_epoch=7,
            action=ProviderOperationResolutionAction.FAIL,
            reason="operator selected a durable failure disposition",
        )

    assert provider_operation_resolution_request_digest(
        request("first-elected-worker")
    ) == provider_operation_resolution_request_digest(request("replacement-elected-worker"))


def test_elected_worker_completes_task_through_tool_approval_continuation() -> None:
    task_store = InMemoryTaskStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="publish-change",
                    name="publish_change",
                    arguments={"change": "reviewed-release"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    _register_approval_agent(app)

    async def pause_for_approval(
        runtime: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in runtime.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="elected-approval-session",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "Publish the reviewed change.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, SessionStatus]:
        await task_store.create_task(
            TaskCreate(
                task_id="elected-approval-task",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        handled = await run_task_worker(
            app,
            task_store,
            pause_for_approval,
            worker_id="prior-worker",
            query=TaskQuery(type="job"),
            max_tasks=1,
            reclaim=False,
        )
        assert handled == 1
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None and elected.worker_id == "elected-worker"
        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="elected-approval-session")
        )
        assert len(pending.actions) == 1
        approval = pending.actions[0]
        assert approval.approval_id is not None
        assert approval.round_id is not None
        assert approval.tool_call_id is not None

        async for _event in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="elected-approval-session",
                task_worker_id="elected-worker",
                task_handoff_id=elected.interrupted_handoff_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

        completed_session = await app.session_store.load("elected-approval-session")
        assert completed_session is not None
        return await task_store.load_task("elected-approval-task"), completed_session.status

    task, session_status = asyncio.run(scenario())
    assert task is not None and task.status is TaskStatus.COMPLETED
    assert task.result == {
        "session_id": "elected-approval-session",
        "agent_name": "worker-agent",
        "environment_name": None,
    }
    assert session_status is SessionStatus.COMPLETED
    assert len(provider.requests) == 2


@pytest.mark.parametrize("task_store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("loss_point", ["after_task_failure", "after_session_failure"])
def test_elected_worker_replays_approval_failure_after_terminalization_interruption(
    task_store_kind: str,
    loss_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ApprovalProcessLoss(BaseException):
        pass

    async def scenario() -> None:
        task_store: TaskStore = (
            InMemoryTaskStore()
            if task_store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "approval-failure-tasks.sqlite")
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-change",
                        name="publish_change",
                        arguments={"change": "reviewed-release"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        original_failure_hook = _VersionedSessionFailureHook("1")
        app = CayuApp(
            task_store=task_store,
            enable_logging=False,
            runtime_hooks=[original_failure_hook],
        )
        app.register_provider(provider, default=True)
        _register_approval_agent(app)

        async def pause_for_approval(
            runtime: CayuApp,
            task: Task,
            worker_id: str,
        ) -> TaskHandlerOutcome:
            async for _event in runtime.run(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="approval-failure-session",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "Publish the reviewed change.")],
                )
            ):
                pass
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        await task_store.create_task(TaskCreate(task_id="approval-failure-task", type="job"))
        assert (
            await run_task_worker(
                app,
                task_store,
                pause_for_approval,
                worker_id="prior-worker",
                query=TaskQuery(type="job"),
                max_tasks=1,
                reclaim=False,
            )
            == 1
        )
        elected = await task_store.claim_interrupted_task_continuation(
            "elected-worker",
            TaskQuery(type="job"),
            handoff_id=str(uuid4()),
        )
        assert elected.task is not None
        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="approval-failure-session")
        )
        assert len(pending.actions) == 1
        approval = pending.actions[0]
        assert approval.approval_id is not None
        assert approval.round_id is not None
        assert approval.tool_call_id is not None
        request = ToolApprovalRequest(
            session_id="approval-failure-session",
            task_worker_id="elected-worker",
            task_handoff_id=elected.task.interrupted_handoff_id,
            approval_id=approval.approval_id,
            tool_round_id=approval.round_id,
            tool_call_id=approval.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
        )

        original_materialize = app._recovery_coordinator.materialize_expected_deferred_input
        original_fan_out = app._event_writer.fan_out_persisted
        original_terminal = app._recovery_coordinator._emit_terminal_event_with_hooks

        async def fail_after_approval_close(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("injected failure after approval close")

        async def lose_after_task_failure(events: list[Event]) -> list[Event]:
            if any(event.type is EventType.TASK_FAILED for event in events):
                raise _ApprovalProcessLoss(
                    "process lost after task failure and before session failure"
                )
            return await original_fan_out(events)

        async def lose_after_session_failure(request):
            async for event in original_terminal(request):
                yield event
                if event.type is EventType.SESSION_FAILED:
                    raise _ApprovalProcessLoss(
                        "process lost after approval session failure and before terminal hooks"
                    )

        monkeypatch.setattr(
            app._recovery_coordinator,
            "materialize_expected_deferred_input",
            fail_after_approval_close,
        )
        if loss_point == "after_task_failure":
            monkeypatch.setattr(app._event_writer, "fan_out_persisted", lose_after_task_failure)
        else:
            monkeypatch.setattr(
                app._recovery_coordinator,
                "_emit_terminal_event_with_hooks",
                lose_after_session_failure,
            )
        with pytest.raises(_ApprovalProcessLoss):
            async for _event in app.resolve_tool_approval(request):
                pass

        failed_before_replay = await task_store.load_task("approval-failure-task")
        assert failed_before_replay is not None
        assert failed_before_replay.status is TaskStatus.FAILED
        session_before_replay = await app.session_store.load("approval-failure-session")
        assert session_before_replay is not None
        assert session_before_replay.status is (
            SessionStatus.RUNNING if loss_point == "after_task_failure" else SessionStatus.FAILED
        )

        monkeypatch.setattr(
            app._recovery_coordinator,
            "materialize_expected_deferred_input",
            original_materialize,
        )
        monkeypatch.setattr(app._event_writer, "fan_out_persisted", original_fan_out)
        monkeypatch.setattr(
            app._recovery_coordinator,
            "_emit_terminal_event_with_hooks",
            original_terminal,
        )
        drifted_hook = _VersionedSessionFailureHook("2")
        drifted_app = CayuApp(
            session_store=app.session_store,
            task_store=task_store,
            enable_logging=False,
            runtime_hooks=[drifted_hook],
        )
        drifted_app.register_provider(provider, default=True)
        _register_approval_agent(drifted_app)
        with pytest.raises(ExecutionProfileMismatchError):
            async for _event in drifted_app.resolve_tool_approval(request):
                pass
        assert drifted_hook.failed_sessions == []
        drift_events = await drifted_app.session_store.load_events("approval-failure-session")
        assert not any(
            event.type is EventType.SESSION_EXECUTION_PROFILE_REJECTED for event in drift_events
        )

        recovered_failure_hook = _VersionedSessionFailureHook("1")
        recovered_app = CayuApp(
            session_store=app.session_store,
            task_store=task_store,
            enable_logging=False,
            runtime_hooks=[recovered_failure_hook],
        )
        recovered_app.register_provider(provider, default=True)
        _register_approval_agent(recovered_app)
        events = [event async for event in recovered_app.resolve_tool_approval(request)]

        failed_session = await recovered_app.session_store.load("approval-failure-session")
        failed_task = await task_store.load_task("approval-failure-task")
        assert failed_session is not None
        assert failed_session.status is SessionStatus.FAILED
        assert failed_task is not None
        assert failed_task.status is TaskStatus.FAILED
        assert failed_task.worker_id is None
        if loss_point == "after_task_failure":
            assert EventType.TASK_FAILED in {event.type for event in events}
            assert sum(event.type is EventType.SESSION_FAILED for event in events) == 1
        else:
            assert EventType.TASK_FAILED not in {event.type for event in events}
            assert EventType.SESSION_FAILED not in {event.type for event in events}
        assert recovered_failure_hook.failed_sessions == ["approval-failure-session"]

        replay = [event async for event in recovered_app.resolve_tool_approval(request)]
        assert [event.type for event in replay] == [EventType.SESSION_CHECKPOINTED]
        with pytest.raises(TaskClaimLost):
            async for _event in recovered_app.resolve_tool_approval(
                request.model_copy(update={"task_worker_id": "stale-worker"})
            ):
                pass
        with pytest.raises(TaskClaimLost):
            async for _event in recovered_app.resolve_tool_approval(
                request.model_copy(update={"task_worker_id": None, "task_handoff_id": None})
            ):
                pass
        with pytest.raises(TaskClaimLost):
            async for _event in recovered_app.resolve_tool_approval(
                request.model_copy(update={"reason": "conflicting replay identity"})
            ):
                pass
        stored = await recovered_app.session_store.load_events("approval-failure-session")
        assert sum(event.type is EventType.TASK_FAILED for event in stored) == 1
        assert sum(event.type is EventType.SESSION_FAILED for event in stored) == 1

        if isinstance(task_store, SQLiteTaskStore):
            await task_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("task_store_kind", ["memory", "sqlite"])
def test_workerless_approval_failure_replays_after_task_terminalization_loss(
    task_store_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ApprovalProcessLoss(BaseException):
        pass

    async def scenario() -> None:
        task_store: TaskStore = (
            InMemoryTaskStore()
            if task_store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "workerless-approval-failure-tasks.sqlite")
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="publish-workerless-change",
                        name="publish_change",
                        arguments={"change": "reviewed-workerless-release"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = CayuApp(task_store=task_store, enable_logging=False)
        app.register_provider(provider, default=True)
        _register_approval_agent(app)

        await task_store.create_task(
            TaskCreate(task_id="workerless-approval-failure-task", type="job")
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="workerless-approval-failure-session",
                    task_id="workerless-approval-failure-task",
                    messages=[Message.text("user", "Publish the reviewed change.")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        attached = await task_store.load_task("workerless-approval-failure-task")
        assert attached is not None
        assert attached.status is TaskStatus.RUNNING
        assert attached.worker_id is None

        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="workerless-approval-failure-session")
        )
        assert len(pending.actions) == 1
        approval = pending.actions[0]
        assert approval.approval_id is not None
        assert approval.round_id is not None
        assert approval.tool_call_id is not None
        request = ToolApprovalRequest(
            session_id="workerless-approval-failure-session",
            approval_id=approval.approval_id,
            tool_round_id=approval.round_id,
            tool_call_id=approval.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
        )

        original_materialize = app._recovery_coordinator.materialize_expected_deferred_input
        original_fan_out = app._event_writer.fan_out_persisted

        async def fail_after_approval_close(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("injected failure after approval close")

        async def lose_after_task_failure(failure_events: list[Event]) -> list[Event]:
            if any(event.type is EventType.TASK_FAILED for event in failure_events):
                raise _ApprovalProcessLoss(
                    "process lost after workerless task failure and before session failure"
                )
            return await original_fan_out(failure_events)

        monkeypatch.setattr(
            app._recovery_coordinator,
            "materialize_expected_deferred_input",
            fail_after_approval_close,
        )
        monkeypatch.setattr(app._event_writer, "fan_out_persisted", lose_after_task_failure)
        with pytest.raises(_ApprovalProcessLoss):
            async for _event in app.resolve_tool_approval(request):
                pass

        failed_before_replay = await task_store.load_task("workerless-approval-failure-task")
        interrupted_before_replay = await app.session_store.load(
            "workerless-approval-failure-session"
        )
        assert failed_before_replay is not None
        assert failed_before_replay.status is TaskStatus.FAILED
        assert interrupted_before_replay is not None
        assert interrupted_before_replay.status is SessionStatus.RUNNING

        monkeypatch.setattr(
            app._recovery_coordinator,
            "materialize_expected_deferred_input",
            original_materialize,
        )
        monkeypatch.setattr(app._event_writer, "fan_out_persisted", original_fan_out)
        replay_events = [event async for event in app.resolve_tool_approval(request)]

        failed_session = await app.session_store.load("workerless-approval-failure-session")
        failed_task = await task_store.load_task("workerless-approval-failure-task")
        assert failed_session is not None
        assert failed_session.status is SessionStatus.FAILED
        assert failed_task == failed_before_replay
        assert EventType.TASK_FAILED in {event.type for event in replay_events}
        assert replay_events[-1].type is EventType.SESSION_FAILED

        completed_replay = [event async for event in app.resolve_tool_approval(request)]
        assert [event.type for event in completed_replay] == [EventType.SESSION_CHECKPOINTED]
        with pytest.raises(TaskClaimLost):
            async for _event in app.resolve_tool_approval(
                request.model_copy(update={"task_worker_id": "unrelated-worker"})
            ):
                pass
        with pytest.raises(TaskClaimLost):
            async for _event in app.resolve_tool_approval(
                request.model_copy(update={"reason": "conflicting direct replay"})
            ):
                pass
        stored = await app.session_store.load_events("workerless-approval-failure-session")
        assert sum(event.type is EventType.TASK_FAILED for event in stored) == 1
        assert sum(event.type is EventType.SESSION_FAILED for event in stored) == 1

        if isinstance(task_store, SQLiteTaskStore):
            await task_store.close()

    asyncio.run(scenario())


def test_elected_worker_completes_task_through_user_input_continuation() -> None:
    task_store = InMemoryTaskStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="ask-release-channel",
                    name="ask_user",
                    arguments={"question": "Which release channel?"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="worker-agent", model="scripted-model"),
        tools=[UserInputTool()],
    )

    async def pause_for_input(
        runtime: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in runtime.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="elected-input-session",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "Prepare the release.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, SessionStatus]:
        await task_store.create_task(
            TaskCreate(
                task_id="elected-input-task",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        handled = await run_task_worker(
            app,
            task_store,
            pause_for_input,
            worker_id="prior-worker",
            query=TaskQuery(type="job"),
            max_tasks=1,
            reclaim=False,
        )
        assert handled == 1
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None and elected.worker_id == "elected-worker"
        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="elected-input-session")
        )
        assert len(pending.actions) == 1
        input_action = pending.actions[0]
        assert input_action.input_id is not None

        async for _event in app.resolve_user_input(
            UserInputResponse(
                session_id="elected-input-session",
                task_worker_id="elected-worker",
                task_handoff_id=elected.interrupted_handoff_id,
                input_id=input_action.input_id,
                answer="stable",
            )
        ):
            pass

        completed_session = await app.session_store.load("elected-input-session")
        assert completed_session is not None
        return await task_store.load_task("elected-input-task"), completed_session.status

    task, session_status = asyncio.run(scenario())
    assert task is not None and task.status is TaskStatus.COMPLETED
    assert task.result == {
        "session_id": "elected-input-session",
        "agent_name": "worker-agent",
        "environment_name": None,
    }
    assert session_status is SessionStatus.COMPLETED
    assert len(provider.requests) == 2


@pytest.mark.parametrize("task_store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("session_store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("task_authority", ["elected", "workerless"])
@pytest.mark.parametrize(
    "loss_point",
    ["before_interaction", "before_session_event", "after_session_event"],
)
def test_generic_continuation_failure_replays_after_task_terminalization_loss(
    task_store_kind: str,
    session_store_kind: str,
    task_authority: str,
    loss_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ContinuationProcessLoss(BaseException):
        pass

    async def scenario() -> None:
        task_store: TaskStore = (
            InMemoryTaskStore()
            if task_store_kind == "memory"
            else SQLiteTaskStore(tmp_path / "generic-continuation-failure-tasks.sqlite")
        )
        session_store = (
            InMemorySessionStore()
            if session_store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "generic-continuation-failure-sessions.sqlite")
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="ask-release-channel-before-failure",
                        name="ask_user",
                        arguments={"question": "Which release channel?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.error("provider failed after continuation admission"),
                    ModelStreamEvent.completed({"finish_reason": "error"}),
                ],
            ]
        )
        original_failure_hook = _VersionedSessionFailureHook("1")
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
            runtime_hooks=[original_failure_hook],
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="worker-agent", model="scripted-model"),
            tools=[UserInputTool()],
        )

        async def pause_for_input(
            runtime: CayuApp,
            task: Task,
            worker_id: str,
        ) -> TaskHandlerOutcome:
            async for _event in runtime.run(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="generic-continuation-failure-session",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "Prepare the release.")],
                )
            ):
                pass
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        await task_store.create_task(
            TaskCreate(task_id="generic-continuation-failure-task", type="job")
        )
        if task_authority == "elected":
            assert (
                await run_task_worker(
                    app,
                    task_store,
                    pause_for_input,
                    worker_id="prior-worker",
                    query=TaskQuery(type="job"),
                    max_tasks=1,
                    reclaim=False,
                )
                == 1
            )
            elected = await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
            assert elected.task is not None
            task_worker_id = "elected-worker"
            task_handoff_id = elected.task.interrupted_handoff_id
        else:
            async for _event in app.run(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="generic-continuation-failure-session",
                    task_id="generic-continuation-failure-task",
                    messages=[Message.text("user", "Prepare the release.")],
                )
            ):
                pass
            task_worker_id = None
            task_handoff_id = None
        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="generic-continuation-failure-session")
        )
        assert len(pending.actions) == 1
        assert pending.actions[0].input_id is not None
        response = UserInputResponse(
            session_id="generic-continuation-failure-session",
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            input_id=pending.actions[0].input_id,
            answer="stable",
        )

        original_publish = app._session_engine._publish_sibling_interaction_transition
        original_terminal = app._session_engine._emit_terminal_event_with_hooks

        async def lose_after_task_failure(*args, **kwargs):
            if kwargs.get("to_status") is SessionStatus.FAILED:
                raise _ContinuationProcessLoss(
                    "process lost after task failure and before session failure"
                )
            return await original_publish(*args, **kwargs)

        async def lose_before_session_event(*args, **kwargs):
            event = kwargs.get("event")
            if (
                isinstance(event, Event)
                and event.type is EventType.SESSION_FAILED
                and event.payload.get("runtime_task_failure_id") is not None
            ):
                raise _ContinuationProcessLoss(
                    "process lost after interaction failure and before session event"
                )
            async for emitted in original_terminal(*args, **kwargs):
                yield emitted

        async def lose_after_session_event(*args, **kwargs):
            async for emitted in original_terminal(*args, **kwargs):
                yield emitted
                if (
                    emitted.type is EventType.SESSION_FAILED
                    and emitted.payload.get("runtime_task_failure_id") is not None
                ):
                    raise _ContinuationProcessLoss(
                        "process lost after session failure and before terminal hooks"
                    )

        if loss_point == "before_interaction":
            monkeypatch.setattr(
                app._session_engine,
                "_publish_sibling_interaction_transition",
                lose_after_task_failure,
            )
        elif loss_point == "before_session_event":
            monkeypatch.setattr(
                app._session_engine,
                "_emit_terminal_event_with_hooks",
                lose_before_session_event,
            )
        else:
            monkeypatch.setattr(
                app._session_engine,
                "_emit_terminal_event_with_hooks",
                lose_after_session_event,
            )
        with pytest.raises(_ContinuationProcessLoss):
            async for _event in app.resolve_user_input(response):
                pass

        failed_task = await task_store.load_task("generic-continuation-failure-task")
        live_session = await app.session_store.load("generic-continuation-failure-session")
        assert failed_task is not None and failed_task.status is TaskStatus.FAILED
        assert failed_task.error is not None
        assert failed_task.error["runtime_task_failure"]["schema"] == (
            "cayu.runtime-task-failure.v2"
        )
        assert live_session is not None
        assert failed_task.error["runtime_task_failure"]["run_epoch"] == (
            live_session.run_epoch
            if loss_point == "before_interaction"
            else live_session.run_epoch - 1
        )
        assert live_session.status is (
            SessionStatus.RUNNING if loss_point == "before_interaction" else SessionStatus.FAILED
        )
        active_stage_before_replay = await app.session_store.load_active_model_completion_stage(
            live_session.id
        )
        if loss_point == "before_interaction":
            assert active_stage_before_replay is not None
            unsettled_stage_id = active_stage_before_replay.stage.stage_id
        else:
            assert active_stage_before_replay is None
            unsettled_stage_id = None

        monkeypatch.setattr(
            app._session_engine,
            "_publish_sibling_interaction_transition",
            original_publish,
        )
        monkeypatch.setattr(
            app._session_engine,
            "_emit_terminal_event_with_hooks",
            original_terminal,
        )
        drifted_hook = _VersionedSessionFailureHook("2")
        drifted_app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
            runtime_hooks=[drifted_hook],
        )
        drifted_app.register_provider(provider, default=True)
        drifted_app.register_agent(
            AgentSpec(name="worker-agent", model="scripted-model"),
            tools=[UserInputTool()],
        )
        with pytest.raises(ExecutionProfileMismatchError):
            async for _event in drifted_app.resolve_user_input(response):
                pass
        assert drifted_hook.failed_sessions == []
        assert len(provider.requests) == 2
        drift_events = await drifted_app.session_store.load_events(response.session_id)
        assert not any(
            event.type is EventType.SESSION_EXECUTION_PROFILE_REJECTED for event in drift_events
        )

        recovered_failure_hook = _VersionedSessionFailureHook("1")
        recovered_app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
            runtime_hooks=[recovered_failure_hook],
        )
        recovered_app.register_provider(provider, default=True)
        recovered_app.register_agent(
            AgentSpec(name="worker-agent", model="scripted-model"),
            tools=[UserInputTool()],
        )

        async def replay_user_input() -> list[Event]:
            return [event async for event in recovered_app.resolve_user_input(response)]

        async def replay_ordinary_resume() -> list[Event]:
            return [
                event
                async for event in recovered_app.resume(
                    ResumeRequest(
                        session_id=response.session_id,
                        task_worker_id=task_worker_id,
                        task_handoff_id=task_handoff_id,
                        messages=[Message.text("user", "retry terminal convergence")],
                    )
                )
            ]

        first_replay, concurrent_replay = await asyncio.gather(
            replay_user_input(),
            replay_ordinary_resume(),
        )
        replay_events = [*first_replay, *concurrent_replay]

        terminal_session = await recovered_app.session_store.load(
            "generic-continuation-failure-session"
        )
        assert terminal_session is not None
        assert terminal_session.status is SessionStatus.FAILED
        assert (
            await recovered_app.session_store.load_active_model_completion_stage(
                terminal_session.id
            )
            is None
        )
        if unsettled_stage_id is not None:
            settlement = await recovered_app.session_store.load_model_completion_stage_settlement(
                terminal_session.id,
                unsettled_stage_id,
            )
            assert settlement is not None
            assert settlement.disposition is (
                ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
            )
            assert settlement.reason_code == "model_attempt_failed"
        assert EventType.SESSION_FAILED in {event.type for event in replay_events}
        assert len(provider.requests) == 2
        assert recovered_failure_hook.failed_sessions == ["generic-continuation-failure-session"]

        async def unavailable_terminal_cleanup(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("injected trailing terminal cleanup failure")

        monkeypatch.setattr(
            recovered_app._session_engine,
            "_clear_session_run_operation",
            unavailable_terminal_cleanup,
        )
        completed_replay = [event async for event in recovered_app.resolve_user_input(response)]
        assert [event.type for event in completed_replay] == [EventType.SESSION_FAILED]
        assert len(provider.requests) == 2
        with pytest.raises(TaskClaimLost):
            async for _event in recovered_app.resolve_user_input(
                response.model_copy(
                    update={
                        "task_worker_id": (
                            "stale-worker" if task_worker_id is not None else "unexpected-worker"
                        )
                    }
                )
            ):
                pass
        stored = await recovered_app.session_store.load_events(terminal_session.id)
        terminal_hook_events = [
            event
            for event in stored
            if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED}
            and event.payload.get("phase") == "after_session_failed"
            and event.payload.get("hook_name") == "_VersionedSessionFailureHook"
        ]
        assert [event.type for event in terminal_hook_events] == [
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        ]
        assert all(event.interaction_id is None for event in terminal_hook_events)
        assert (
            terminal_hook_events[0].payload["hook_invocation_id"]
            == terminal_hook_events[1].payload["hook_invocation_id"]
        )
        assert {event.payload["hook_index"] for event in terminal_hook_events} == {0}
        assert sum(event.type == EventType.TASK_FAILED for event in stored) == 1
        assert sum(event.type == EventType.INTERACTION_FAILED for event in stored) == 1
        failed_turns = [
            event
            for event in stored
            if event.type is EventType.TURN_COMPLETED
            and event.payload.get("status") == SessionStatus.FAILED.value
        ]
        assert len(failed_turns) == 1
        assert failed_turns[0].payload["interaction_ids"] == [
            failed_task.error["runtime_task_failure"]["interaction_id"]
        ]
        assert sum(event.type is EventType.SESSION_FAILED for event in stored) == 1
        terminal_checkpoint = await recovered_app.session_store.load_checkpoint(terminal_session.id)
        assert terminal_checkpoint is not None
        assert "session_run_operation" not in terminal_checkpoint

        if isinstance(task_store, SQLiteTaskStore):
            await task_store.close()
        if isinstance(session_store, SQLiteSessionStore):
            await session_store.close()

    asyncio.run(scenario())


def test_elected_worker_completes_task_through_manual_tool_recovery() -> None:
    class RecoverableTool(Tool):
        spec = ToolSpec(
            name="recoverable_tool",
            description="Exercise manual recovery after a simulated terminal-write crash.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.NONE,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:task-worker:recoverable-tool",
                behavior_version="1",
                implementation_version="1",
            ),
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult(content="executed before the simulated crash")

    class FailFirstToolTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_terminal_once = False

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            if not self.failed_terminal_once and any(
                event.type is EventType.TOOL_CALL_COMPLETED for event in events
            ):
                self.failed_terminal_once = True
                raise RuntimeError("simulated crash before the tool terminal became durable")
            await super().append_events(session_id, events)

    class PreserveTaskThroughSimulatedCrashStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        supports_verified_work_contracts = True
        verified_work_mutations_are_cancellation_quiescent = True
        suppress_terminalization = True

        async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
            if self.suppress_terminalization:
                raise RuntimeError("simulated worker loss before task terminalization")
            return await super().terminalize_task(request)

    session_store = FailFirstToolTerminalStore()
    task_store = PreserveTaskThroughSimulatedCrashStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="recoverable-call",
                    name="recoverable_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="worker-agent", model="scripted-model"),
        tools=[RecoverableTool()],
    )

    async def scenario() -> tuple[Task | None, SessionStatus]:
        await task_store.create_task(
            TaskCreate(
                task_id="elected-tool-recovery-task",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        claimed = await task_store.claim_task("prior-worker", TaskQuery(type="job"))
        assert claimed is not None
        initial_events: list[Event] = []
        async for event in app.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="elected-tool-recovery-session",
                task_id=claimed.id,
                task_worker_id="prior-worker",
                task_lease_expires_at=claimed.lease_expires_at,
                messages=[Message.text("user", "Publish the recovered release.")],
            )
        ):
            initial_events.append(event)

        crashed_session = await session_store.load("elected-tool-recovery-session")
        attached = await task_store.load_task("elected-tool-recovery-task")
        checkpoint = await session_store.load_checkpoint("elected-tool-recovery-session")
        assert initial_events[-1].type is EventType.SESSION_FAILED
        assert crashed_session is not None and crashed_session.status is SessionStatus.FAILED
        assert attached is not None and attached.worker_id == "prior-worker"
        assert checkpoint is not None
        pending_round = checkpoint["pending_tool_round"]
        assert pending_round["task_id"] == attached.id

        task_store.suppress_terminalization = False
        await task_store.release_interrupted_task_worker(
            interrupted_task_handoff_request(
                attached,
                session_run_epoch=crashed_session.run_epoch,
            )
        )
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None and elected.worker_id == "elected-worker"

        async for _event in app.recover_tool_round(
            ToolRoundRecoveryRequest(
                session_id="elected-tool-recovery-session",
                task_worker_id="elected-worker",
                task_handoff_id=elected.interrupted_handoff_id,
                round_id=pending_round["tool_round_id"],
                tool_call_id="recoverable-call",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="publication verified externally",
            )
        ):
            pass

        completed_session = await session_store.load("elected-tool-recovery-session")
        assert completed_session is not None
        return (
            await task_store.load_task("elected-tool-recovery-task"),
            completed_session.status,
        )

    task, session_status = asyncio.run(scenario())
    assert task is not None and task.status is TaskStatus.COMPLETED
    assert task.worker_id is None
    assert session_status is SessionStatus.COMPLETED
    assert len(provider.requests) == 2


def test_tool_approval_rechecks_worker_authority_after_session_admission() -> None:
    preflight_complete = asyncio.Event()
    release_preflight = asyncio.Event()

    class PausingAuthorityTaskStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        supports_verified_work_contracts = True
        verified_work_mutations_are_cancellation_quiescent = True
        reads = 0
        reject_future_reads = False

        async def load_active_attached_task_worker(
            self,
            task_id: str,
            worker_id: str,
            *,
            session_id: str,
            session_instance_id: str,
        ) -> Task:
            task = await super().load_active_attached_task_worker(
                task_id,
                worker_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
            )
            self.reads += 1
            if self.reads == 1:
                preflight_complete.set()
                await release_preflight.wait()
                return task
            if self.reject_future_reads:
                raise TaskClaimLost("Continuation worker authority was lost after preflight.")
            return task

    class AuthorityLossHook(RuntimeHook):
        def __init__(self) -> None:
            self.interrupted_sessions: list[str] = []

        async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
            self.interrupted_sessions.append(context.session.id)

    task_store = PausingAuthorityTaskStore()
    authority_loss_hook = AuthorityLossHook()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="publish-change",
                    name="publish_change",
                    arguments={"change": "reviewed-release"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        task_store=task_store,
        runtime_hooks=[authority_loss_hook],
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    _register_approval_agent(app)

    async def pause_for_approval(
        runtime: CayuApp,
        task: Task,
        worker_id: str,
    ) -> TaskHandlerOutcome:
        async for _event in runtime.run(
            RunRequest(
                agent_name="worker-agent",
                session_id="approval-race-session",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "Publish the reviewed change.")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario() -> tuple[Task | None, SessionStatus, list[Event]]:
        await task_store.create_task(
            TaskCreate(
                task_id="approval-race-task",
                type="job",
                assigned_agent_name="worker-agent",
            )
        )
        await run_task_worker(
            app,
            task_store,
            pause_for_approval,
            worker_id="prior-worker",
            query=TaskQuery(type="job"),
            max_tasks=1,
            reclaim=False,
        )
        authority_loss_hook.interrupted_sessions.clear()
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None
        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="approval-race-session")
        )
        assert len(pending.actions) == 1
        approval = pending.actions[0]
        assert approval.approval_id is not None
        assert approval.round_id is not None
        assert approval.tool_call_id is not None

        async def resolve() -> None:
            async for _event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="approval-race-session",
                    task_worker_id="elected-worker",
                    task_handoff_id=elected.interrupted_handoff_id,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            ):
                pass

        continuation = asyncio.create_task(resolve())
        await preflight_complete.wait()
        task_store.reject_future_reads = True
        release_preflight.set()
        with pytest.raises(TaskClaimLost):
            await continuation
        session = await app.session_store.load("approval-race-session")
        assert session is not None
        records = await app.session_store.query_events(
            EventQuery(
                session_id="approval-race-session",
                event_types=(EventType.TOOL_CALL_COMPLETED,),
            )
        )
        return (
            await task_store.load_task("approval-race-task"),
            session.status,
            [record.event for record in records],
        )

    task, session_status, completed_tool_events = asyncio.run(scenario())
    assert task is not None and task.status is TaskStatus.RUNNING
    assert task.worker_id == "elected-worker"
    assert session_status is SessionStatus.INTERRUPTED
    assert completed_tool_events == []
    assert len(provider.requests) == 1
    assert authority_loss_hook.interrupted_sessions == []


def test_ownerless_resume_cannot_cross_a_concurrent_continuation_claim() -> None:
    listed = asyncio.Event()
    continue_resume = asyncio.Event()

    class PausingListTaskStore(InMemoryTaskStore):
        async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
            tasks = await super().list_tasks(query)
            if (
                query is not None
                and query.session_id == "ownerless-race-session"
                and not listed.is_set()
            ):
                listed.set()
                await continue_resume.wait()
            return tasks

    task_store = PausingListTaskStore()
    provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def collect() -> list[Event]:
        return [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="ownerless-race-session",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

    async def scenario() -> Task | None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="ownerless-race-task",
            session_id="ownerless-race-session",
        )
        claimed = await task_store.claim_task("prior-worker", lease_seconds=30)
        assert claimed is not None
        attached = await task_store.attach_task(
            claimed.id,
            session_id="ownerless-race-session",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "ownerless-race-session",
            ),
            worker_id="prior-worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        session = await app.session_store.load("ownerless-race-session")
        assert session is not None
        await task_store.release_interrupted_task_worker(
            interrupted_task_handoff_request(
                attached,
                session_run_epoch=session.run_epoch,
            )
        )

        resume_task = asyncio.create_task(collect())
        await listed.wait()
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None
        continue_resume.set()
        with pytest.raises(TaskClaimLost):
            await resume_task
        return await task_store.load_task("ownerless-race-task")

    task = asyncio.run(scenario())
    assert task is not None and task.worker_id == "elected-worker"
    assert provider.requests == []


def test_resume_rechecks_worker_authority_after_session_admission() -> None:
    preflight_complete = asyncio.Event()
    release_preflight = asyncio.Event()

    class PausingAuthorityTaskStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        supports_verified_work_contracts = True
        verified_work_mutations_are_cancellation_quiescent = True
        reads = 0

        async def load_active_attached_task_worker(
            self,
            task_id: str,
            worker_id: str,
            *,
            session_id: str,
            session_instance_id: str,
        ) -> Task:
            task = await super().load_active_attached_task_worker(
                task_id,
                worker_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
            )
            self.reads += 1
            if self.reads == 1:
                preflight_complete.set()
                await release_preflight.wait()
            return task

    class AuthorityLossHook(RuntimeHook):
        def __init__(self) -> None:
            self.interrupted_sessions: list[str] = []

        async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
            self.interrupted_sessions.append(context.session.id)

    task_store = PausingAuthorityTaskStore()
    authority_loss_hook = AuthorityLossHook()
    provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
    app = CayuApp(
        task_store=task_store,
        runtime_hooks=[authority_loss_hook],
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def collect() -> list[Event]:
        return [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="post-admission-fence-session",
                    task_worker_id="stale-worker",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

    async def scenario() -> tuple[Task | None, SessionStatus]:
        created = await task_store.create_task(
            TaskCreate(task_id="post-admission-fence-task", type="job")
        )
        await app.session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="post-admission-fence-session",
                    task_id=created.id,
                    messages=[Message.text("user", "pause")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                TaskInvocationSnapshot(
                    id=created.id,
                    session_id=created.session_id,
                    invocation=created.invocation,
                ),
            ),
            identity=profiled_session_identity(
                provider_name=provider.name,
                model="scripted-model",
                agent_name="worker-agent",
                app=app,
            ),
        )
        await admit_test_invocation(
            app.session_store,
            "post-admission-fence-session",
            interaction_started_event=runtime_interaction_started_event(
                app,
                session_id="post-admission-fence-session",
                interaction_id="post-admission-fence-interaction",
                agent_name="worker-agent",
            ),
            interaction_source_messages=(Message.text("user", "pause"),),
        )
        await interrupt_and_release_test_invocation(
            app.session_store,
            "post-admission-fence-session",
        )
        claimed = await task_store.claim_task("stale-worker", lease_seconds=30)
        assert claimed is not None
        attached = await task_store.attach_task(
            claimed.id,
            session_id="post-admission-fence-session",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "post-admission-fence-session",
            ),
            worker_id="stale-worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        session = await app.session_store.load("post-admission-fence-session")
        assert session is not None
        handoff = interrupted_task_handoff_request(
            attached,
            session_run_epoch=session.run_epoch,
        )

        resume_task = asyncio.create_task(collect())
        await preflight_complete.wait()
        await task_store.release_interrupted_task_worker(handoff)
        release_preflight.set()
        with pytest.raises(TaskClaimLost):
            await resume_task
        final_session = await app.session_store.load("post-admission-fence-session")
        assert final_session is not None
        return await task_store.load_task("post-admission-fence-task"), final_session.status

    task, session_status = asyncio.run(scenario())
    assert task is not None and task.interrupted_handoff_id is not None
    assert session_status is SessionStatus.INTERRUPTED
    assert provider.requests == []
    assert authority_loss_hook.interrupted_sessions == []


def test_resume_rejects_a_replacement_session_incarnation_before_provider() -> None:
    authority_read = asyncio.Event()
    continue_resume = asyncio.Event()

    class PausingAuthorityTaskStore(InMemoryTaskStore):
        reads = 0

        async def load_active_attached_task_worker(
            self,
            task_id: str,
            worker_id: str,
            *,
            session_id: str,
            session_instance_id: str,
        ) -> Task:
            task = await super().load_active_attached_task_worker(
                task_id,
                worker_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
            )
            self.reads += 1
            if self.reads == 1:
                authority_read.set()
                await continue_resume.wait()
            return task

    task_store = PausingAuthorityTaskStore()
    session_store = InMemorySessionStore()
    provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
    app = CayuApp(
        task_store=task_store,
        session_store=session_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario() -> None:
        await _seed_interrupted_worker_handoff(
            app,
            task_store,
            task_id="replacement-session-task",
            session_id="replacement-session",
        )
        claimed = await task_store.claim_task("prior-worker", lease_seconds=30)
        assert claimed is not None
        attached = await task_store.attach_task(
            claimed.id,
            session_id="replacement-session",
            session_invocation=await stored_session_invocation(
                session_store,
                "replacement-session",
            ),
            worker_id="prior-worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        original = await session_store.load("replacement-session")
        assert original is not None
        await task_store.release_interrupted_task_worker(
            interrupted_task_handoff_request(
                attached,
                session_run_epoch=original.run_epoch,
            )
        )
        elected = (
            await task_store.claim_interrupted_task_continuation(
                "elected-worker",
                TaskQuery(type="job"),
                handoff_id=str(uuid4()),
            )
        ).task
        assert elected is not None

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="replacement-session",
                        task_worker_id="elected-worker",
                        task_handoff_id=elected.interrupted_handoff_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]

        resume_task = asyncio.create_task(collect())
        await authority_read.wait()
        await session_store.append_events(
            "replacement-session",
            [
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="replacement-session",
                    agent_name="worker-agent",
                )
            ],
        )
        await session_store.delete_session("replacement-session")
        replacement = await session_store.create(
            RunRequest(
                agent_name="worker-agent",
                session_id="replacement-session",
                messages=[Message.text("user", "replacement")],
            ),
            identity=SessionIdentity(
                provider_name="scripted",
                model="scripted-model",
            ),
        )
        assert replacement.instance_id != original.instance_id
        await session_store.update_status("replacement-session", SessionStatus.INTERRUPTED)
        continue_resume.set()
        with pytest.raises(TaskClaimLost):
            await resume_task

    asyncio.run(scenario())
    assert provider.requests == []


def test_session_engine_keeps_legacy_custom_store_direct_resume_compatible() -> None:
    class LegacyCustomTaskStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = False
        verified_work_mutations_are_cancellation_quiescent = True
        load_direct_attached_task_resume = TaskStore.load_direct_attached_task_resume

    task_store = LegacyCustomTaskStore()
    provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario() -> list[Event]:
        task = await task_store.create_task(TaskCreate(task_id="legacy-resume-task", type="job"))
        await app.session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="legacy-resume-session",
                    task_id=task.id,
                    messages=[Message.text("user", "original")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                TaskInvocationSnapshot(
                    id=task.id,
                    session_id=task.session_id,
                    invocation=task.invocation,
                ),
            ),
            identity=profiled_session_identity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await app.session_store.update_status(
            "legacy-resume-session",
            SessionStatus.INTERRUPTED,
        )
        await task_store.start_task(
            task.id,
            session_id="legacy-resume-session",
            session_invocation=await stored_session_invocation(
                app.session_store,
                "legacy-resume-session",
            ),
        )
        return [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="legacy-resume-session",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

    events = asyncio.run(scenario())
    assert provider.requests
    assert events[-1].type is EventType.SESSION_COMPLETED


def test_resume_completes_the_running_task_already_attached_to_the_session(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario():
        task = await task_store.create_task(TaskCreate(task_id="task-resume", type="job"))
        await session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker-agent",
                    session_id="session-resume",
                    task_id=task.id,
                    messages=[Message.text("user", "original")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                ),
                TaskInvocationSnapshot(
                    id=task.id,
                    session_id=task.session_id,
                    invocation=task.invocation,
                ),
            ),
            identity=profiled_session_identity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await session_store.update_status("session-resume", SessionStatus.INTERRUPTED)
        await task_store.start_task(
            task.id,
            session_id="session-resume",
            session_invocation=await stored_session_invocation(
                session_store,
                "session-resume",
            ),
        )

        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="session-resume",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        return await task_store.load_task(task.id), events

    task, events = asyncio.run(scenario())

    assert task is not None
    assert task.status == "completed"
    assert task.session_id == "session-resume"
    assert [event.type for event in events][-3:] == [
        EventType.TASK_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]


def test_resume_rejects_multiple_running_tasks_attached_to_the_same_session(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    provider = ScriptedModelProvider([])
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))

    async def scenario() -> None:
        await session_store.create(
            RunRequest(
                agent_name="worker-agent",
                session_id="session-ambiguous-tasks",
                messages=[Message.text("user", "original")],
            ),
            identity=SessionIdentity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await session_store.update_status(
            "session-ambiguous-tasks",
            SessionStatus.INTERRUPTED,
        )
        for task_id in ("task-ambiguous-a", "task-ambiguous-b"):
            session_invocation = await stored_session_invocation(
                session_store,
                "session-ambiguous-tasks",
            )
            await task_store.create_task(
                task_create_with_runtime_invocation(
                    TaskCreate(
                        task_id=task_id,
                        type="job",
                        session_id="session-ambiguous-tasks",
                    ),
                    source=TaskExecutionSource.SDK_TASK,
                    session_invocation=session_invocation,
                )
            )
            await task_store.start_task(
                task_id,
                session_id="session-ambiguous-tasks",
                session_invocation=session_invocation,
            )

        async for _ in app.resume(
            ResumeRequest(
                session_id="session-ambiguous-tasks",
                messages=[Message.text("user", "continue")],
            )
        ):
            pass

    with pytest.raises(
        RuntimeError,
        match="Session has multiple running tasks attached: session-ambiguous-tasks",
    ):
        asyncio.run(scenario())

    session = asyncio.run(session_store.load("session-ambiguous-tasks"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert provider.requests == []
