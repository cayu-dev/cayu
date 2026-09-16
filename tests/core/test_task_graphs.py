from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import (
    InMemoryTaskStore,
    TaskCreate,
    TaskQuery,
    TaskSessionClosureClaim,
    TaskStatus,
    TaskStore,
)
from cayu.tasks.graphs import (
    TaskGraphConflict,
    TaskGraphCreate,
    TaskGraphEventType,
    TaskGraphNode,
    TaskGraphUnavailable,
)
from cayu.tasks.scheduling import TaskSchedulePolicy

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryTaskStore()
    elif request.param == "sqlite":
        value = SQLiteTaskStore(tmp_path / "graphs.sqlite")
        try:
            yield value
        finally:
            await value.close()
    else:
        value = PostgresTaskStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
        try:
            await value._ensure_ready()
            # The fixture's module-specific database is disposable and owned by
            # this test module. Reset only this graph suite's persisted records.
            async with value._connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "TRUNCATE cayu_tasks, cayu_task_graphs, cayu_task_graph_members, cayu_task_graph_events CASCADE"
                )
                await conn.commit()
            yield value
        finally:
            await value.close()


def graph_request(*, graph_id: str = "graph") -> TaskGraphCreate:
    return TaskGraphCreate(
        graph_id=graph_id,
        nodes=(
            TaskGraphNode(
                task=TaskCreate(task_id="join", type="test"), prerequisite_task_ids=("b", "a")
            ),
            TaskGraphNode(
                task=TaskCreate(task_id="b", type="test"), prerequisite_task_ids=("root",)
            ),
            TaskGraphNode(
                task=TaskCreate(task_id="a", type="test"), prerequisite_task_ids=("root",)
            ),
            TaskGraphNode(task=TaskCreate(task_id="root", type="test")),
        ),
    )


@pytest.mark.parametrize("dimension", ["items", "bytes"])
@pytest.mark.parametrize("offset", [-1, 0, 1, 131])
async def test_contract_join_reserves_dependency_skip_capacity(
    store: TaskStore, monkeypatch: pytest.MonkeyPatch, dimension: str, offset: int
) -> None:
    from tests.core.test_verified_work_contracts import _contract

    from cayu import CayuApp
    from cayu._validation import canonical_durable_json_bytes
    from cayu.tasks import base
    from cayu.tasks.contracts import (
        WORK_CONTRACT_TASK_CREATION_MAX_BYTES,
        WORK_CONTRACT_TASK_CREATION_MAX_ITEMS,
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 16, 0, 0, 0, 123456, tzinfo=UTC)

    # Exact byte-boundary sizing must not depend on timestamp microsecond width.
    monkeypatch.setattr(base, "datetime", FixedDateTime)
    contract = await store.publish_work_contract(_contract())
    prerequisites = tuple(f"b{index:02d}" for index in range(64))
    join = TaskCreate(
        task_id="z",
        type="test",
        work_contract=contract.reference(),
        input={"padding": [] if dimension == "items" else ""},
    )
    prototype = base._task_from_create(
        join, task_id="z", parent_task=None, supports_verified_work_contracts=True
    ).model_copy(
        update={
            "graph_id": "capacity",
            "prerequisite_task_ids": prerequisites,
            "status": TaskStatus.WAITING_DEPENDENCIES,
        }
    )
    # Independently specify the expected public diagnostics, not the production
    # helper: changing their shape must force this capacity regression to change.
    evidence = prototype.model_dump(mode="json")
    evidence.update(
        status_reason="dependency_failed",
        status_payload={"failed_prerequisite_task_ids": list(prerequisites)},
        error={"code": "dependency_failed", "failed_prerequisite_task_ids": list(prerequisites)},
    )
    if dimension == "items":
        count = 0
        pending = [evidence]
        while pending:
            value = pending.pop()
            count += 1
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        padding = [0] * (WORK_CONTRACT_TASK_CREATION_MAX_ITEMS - count + offset)
    else:
        size = len(canonical_durable_json_bytes(evidence, "test snapshot"))
        padding = "x" * (WORK_CONTRACT_TASK_CREATION_MAX_BYTES - size + offset)
    join = join.model_copy(update={"input": {"padding": padding}})
    # Even rejected cases satisfy the old initial-snapshot check, including
    # offset=131, which fills its entire 32,704-value allowance.
    base.require_contract_bound_task_creation_snapshot(
        prototype.model_copy(update={"input": join.input})
    )
    request = TaskGraphCreate(
        graph_id="capacity",
        nodes=(
            TaskGraphNode(task=TaskCreate(task_id="a", type="test")),
            *(
                TaskGraphNode(
                    task=TaskCreate(task_id=identity, type="test"), prerequisite_task_ids=("a",)
                )
                for identity in prerequisites
            ),
            TaskGraphNode(task=join, prerequisite_task_ids=prerequisites),
        ),
    )
    app = CayuApp(task_store=store, enable_logging=False)
    if offset > 0:
        with pytest.raises(ValueError):
            await app.create_task_graph(request)
        assert await store.list_tasks() == []
        assert await app.load_task_graph(request.graph_id) is None
        with pytest.raises(KeyError):
            await app.list_task_graph_events(request.graph_id)
        # The native entrance also refuses before any durable admission writes.
        with pytest.raises(ValueError):
            await store.create_task_graph(request)
        assert await store.list_tasks() == []
        assert await store.load_task_graph(request.graph_id) is None
        return
    receipt = await app.create_task_graph(request)
    await store.fail_task("a", {"code": "source_failed"})
    snapshot = await app.load_task_graph(request.graph_id)
    assert snapshot is not None
    assert all(
        member.status
        is (TaskStatus.FAILED if member.task_id == "a" else TaskStatus.DEPENDENCY_SKIPPED)
        for member in snapshot.members
    )
    skipped = await store.load_task("z")
    assert skipped is not None
    assert skipped.error == evidence["error"]
    assert skipped.status_payload == evidence["status_payload"]
    events = await app.list_task_graph_events(request.graph_id, limit=1000)
    assert sum(event.type is TaskGraphEventType.SKIPPED for event in events) == 65
    assert sum(event.type is TaskGraphEventType.TERMINAL for event in events) == 1
    assert await app.create_task_graph(request) == receipt
    assert await app.list_task_graph_events(request.graph_id, limit=1000) == events
    assert await store.claim_task("worker") is None


async def test_graph_owner_lost_cancellation_recovery_skips_dependents(store: TaskStore) -> None:
    from tests.core.task_terminalization_conformance import (
        ordinary_cancellation_reconciliation_request,
    )

    await store.create_task_graph(graph_request())
    claimed = await store.claim_task("lost-worker", lease_seconds=1)
    assert claimed is not None and claimed.id == "root"
    requested = await store.cancel_task(claimed.id, {"code": "operator"})
    request = ordinary_cancellation_reconciliation_request(requested)
    await asyncio.sleep(1.05)
    result = await store.reconcile_task_cancellation(request)
    assert result.task.status is TaskStatus.CANCELLED
    snapshot = await store.load_task_graph("graph")
    assert snapshot is not None
    assert all(
        member.status is TaskStatus.DEPENDENCY_SKIPPED
        for member in snapshot.members
        if member.task_id != "root"
    )
    events = await store.list_task_graph_events("graph")
    assert sum(event.type is TaskGraphEventType.SKIPPED for event in events) == 3
    assert await store.reconcile_task_cancellation(request) == result
    assert await store.list_task_graph_events("graph") == events
    assert await store.claim_task("replacement") is None


async def test_graph_direct_cancellation_skips_dependents(store: TaskStore) -> None:
    await store.create_task_graph(graph_request())
    cancelled = await store.cancel_task("root", {"code": "operator"})
    assert cancelled.status is TaskStatus.CANCELLED
    snapshot = await store.load_task_graph("graph")
    assert snapshot is not None
    assert all(
        member.status is TaskStatus.DEPENDENCY_SKIPPED
        for member in snapshot.members
        if member.task_id != "root"
    )
    assert await store.claim_task("worker") is None


async def test_graph_retry_cancellation_recovery_skips_dependents(store: TaskStore) -> None:
    from tests.core.test_postgres_task_store import (
        _postgres_retry_cancellation_reconciliation_request,
    )

    from cayu.tasks.base import TaskRetryPolicy

    request = graph_request()
    root = next(node for node in request.nodes if node.task.task_id == "root")
    root.task.retry_policy = TaskRetryPolicy(max_attempts=2)
    await store.create_task_graph(request)
    claimed = await store.claim_task("lost-worker", lease_seconds=1)
    assert claimed is not None and claimed.id == "root"
    requested = await store.cancel_task("root", {"code": "operator"})
    reconciliation = _postgres_retry_cancellation_reconciliation_request(requested)
    await asyncio.sleep(1.05)
    result = await store.reconcile_task_retry_cancellation(reconciliation)
    assert result.task.status is TaskStatus.CANCELLED
    snapshot = await store.load_task_graph("graph")
    assert snapshot is not None
    assert all(
        member.status is TaskStatus.DEPENDENCY_SKIPPED
        for member in snapshot.members
        if member.task_id != "root"
    )
    events = await store.list_task_graph_events("graph")
    assert await store.reconcile_task_retry_cancellation(reconciliation) == result
    assert await store.list_task_graph_events("graph") == events


async def test_graph_active_retry_deadline_skips_dependents(store: TaskStore) -> None:
    from cayu.tasks.base import TaskRetryPolicy

    request = graph_request()
    root = next(node for node in request.nodes if node.task.task_id == "root")
    root.task.retry_policy = TaskRetryPolicy(max_attempts=2, max_elapsed_seconds=2)
    await store.create_task_graph(request)
    claimed = await store.claim_task("worker", lease_seconds=60)
    assert claimed is not None and claimed.id == "root"
    assert claimed.lease_expires_at is not None
    await asyncio.sleep(2.05)
    result = await store.enforce_task_retry_deadline(
        claimed.id, "worker", lease_expires_at=claimed.lease_expires_at
    )
    assert result is not None and result.task.status is TaskStatus.FAILED
    snapshot = await store.load_task_graph("graph")
    assert snapshot is not None
    assert all(
        member.status is TaskStatus.DEPENDENCY_SKIPPED
        for member in snapshot.members
        if member.task_id != "root"
    )
    assert await store.claim_task("replacement") is None


async def test_graph_validates_each_reference_to_same_contract(store: TaskStore) -> None:
    from tests.core.test_verified_work_contracts import _contract

    from cayu.tasks.contracts import WorkContractConflict

    contract = await store.publish_work_contract(_contract(contract_id="graph-contract"))
    reference = contract.reference()
    conflicting = reference.model_copy(update={"fingerprint": "0" * 64})
    request = TaskGraphCreate(
        graph_id="conflicting-contract",
        nodes=(
            TaskGraphNode(
                task=TaskCreate(task_id="a-invalid", type="test", work_contract=conflicting)
            ),
            TaskGraphNode(task=TaskCreate(task_id="z-valid", type="test", work_contract=reference)),
        ),
    )
    with pytest.raises(WorkContractConflict):
        await store.create_task_graph(request)
    assert await store.load_task_graph(request.graph_id) is None
    assert await store.load_task("a-invalid") is None
    assert await store.load_task("z-valid") is None
    with pytest.raises(KeyError, match="not found"):
        await store.list_task_graph_events(request.graph_id)
    repaired = TaskGraphCreate(
        graph_id=request.graph_id,
        nodes=tuple(
            TaskGraphNode(task=node.task.model_copy(update={"work_contract": reference}))
            for node in request.nodes
        ),
    )
    assert (await store.create_task_graph(repaired)).task_ids == ("a-invalid", "z-valid")


async def test_graph_diamond_releases_each_join_atomically(store: TaskStore) -> None:
    receipt = await store.create_task_graph(graph_request())
    assert receipt.task_ids == ("a", "b", "join", "root")
    root = await store.claim_task("worker")
    assert root is not None and root.id == "root"
    assert await store.claim_task("peer") is None
    await store.complete_task(
        root.id, {}, worker_id=root.worker_id, lease_expires_at=root.lease_expires_at
    )
    a, b = await asyncio.gather(store.claim_task("worker-a"), store.claim_task("worker-b"))
    assert a is not None and b is not None and {a.id, b.id} == {"a", "b"}
    await store.complete_task(a.id, {}, worker_id=a.worker_id, lease_expires_at=a.lease_expires_at)
    assert await store.claim_task("early") is None
    await store.complete_task(b.id, {}, worker_id=b.worker_id, lease_expires_at=b.lease_expires_at)
    claims = await asyncio.gather(store.claim_task("join-a"), store.claim_task("join-b"))
    assert [task.id for task in claims if task is not None] == ["join"]


async def test_maximum_graph_transitive_skip_is_bounded(store: TaskStore) -> None:
    nodes = tuple(
        TaskGraphNode(
            task=TaskCreate(task_id=f"bounded-{index:03}", type="test"),
            prerequisite_task_ids=() if index == 0 else (f"bounded-{index - 1:03}",),
        )
        for index in range(128)
    )
    await store.create_task_graph(TaskGraphCreate(graph_id="maximum", nodes=tuple(reversed(nodes))))
    await store.fail_task("bounded-000", {"code": "test"})
    snapshot = await store.load_task_graph("maximum")
    assert snapshot is not None and len(snapshot.members) == 128
    assert snapshot.members[0].status is TaskStatus.FAILED
    assert all(member.status is TaskStatus.DEPENDENCY_SKIPPED for member in snapshot.members[1:])
    events = await store.list_task_graph_events("maximum", limit=1000)
    assert len(events) == 257
    assert await store.claim_task("never") is None


async def test_graph_aggregate_bounds_do_not_accept_individually_valid_members() -> None:
    nodes = tuple(
        TaskGraphNode(
            task=TaskCreate(task_id=f"large-{i}", type="test", input={"blob": "x" * 32768})
        )
        for i in range(32)
    )
    with pytest.raises(ValueError, match="canonical byte limit"):
        TaskGraphCreate(graph_id="oversized", nodes=nodes)
    with pytest.raises(ValueError, match="node count"):
        TaskGraphCreate(graph_id="many", nodes=nodes * 5)
    dense = tuple(
        TaskGraphNode(
            task=TaskCreate(task_id=f"dense-{i}", type="test"),
            prerequisite_task_ids=tuple(f"dense-{j}" for j in range(i)),
        )
        for i in range(46)
    )
    with pytest.raises(ValueError, match="edge count"):
        TaskGraphCreate(graph_id="dense", nodes=dense)


async def test_graph_operational_counts_preserve_dependency_states(store: TaskStore) -> None:
    await store.create_task_graph(graph_request())
    initial = await store.aggregate_operational_snapshot()
    assert initial.total_count == 4
    assert initial.counts_by_status.pending == initial.claimable_pending_count == 1
    assert initial.counts_by_status.waiting_dependencies == 3
    assert initial.counts_by_status.dependency_skipped == 0
    await store.fail_task("root", {"code": "test"})
    terminal = await store.aggregate_operational_snapshot()
    assert terminal.total_count == 4
    assert terminal.counts_by_status.failed == 1
    assert terminal.counts_by_status.dependency_skipped == 3
    assert terminal.counts_by_status.waiting_dependencies == terminal.claimable_pending_count == 0


async def test_skipped_retry_member_has_matching_terminal_receipt(store: TaskStore) -> None:
    from cayu.tasks.base import TaskRetryEventType, TaskRetryPolicy, TaskRetrySeriesDisposition

    await store.create_task_graph(
        TaskGraphCreate(
            graph_id="skip-retry",
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id="skip-root", type="test")),
                TaskGraphNode(
                    task=TaskCreate(
                        task_id="skip-child",
                        type="test",
                        retry_policy=TaskRetryPolicy(max_attempts=2),
                    ),
                    prerequisite_task_ids=("skip-root",),
                ),
            ),
        )
    )
    await store.fail_task("skip-root", {"code": "test"})
    child = await store.load_task("skip-child")
    assert child is not None and child.retry_series is not None and child.status_payload is not None
    assert child.status is TaskStatus.DEPENDENCY_SKIPPED
    assert child.started_at is None
    assert child.retry_series.disposition is TaskRetrySeriesDisposition.NON_RETRYABLE_FAILURE
    receipt = await store.load_task_retry_settlement(
        child.id, child.status_payload["settlement_idempotency_key"]
    )
    assert receipt is not None and receipt.task == child and receipt.successor is None
    assert [event.type for event in receipt.events] == [
        TaskRetryEventType.ATTEMPT_SETTLED,
        TaskRetryEventType.SERIES_TERMINAL,
    ]
    assert await store.claim_task("late-worker") is None


async def test_graph_outcome_and_retry_skip_roll_back_together(
    store: TaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.storage import _postgres_task_graphs, _sqlite_task_graphs
    from cayu.tasks import _memory_graphs
    from cayu.tasks.base import TaskRetryPolicy

    module = (
        _memory_graphs
        if isinstance(store, InMemoryTaskStore)
        else (_sqlite_task_graphs if isinstance(store, SQLiteTaskStore) else _postgres_task_graphs)
    )
    await store.create_task_graph(
        TaskGraphCreate(
            graph_id="atomic-outcome",
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id="a-source", type="test")),
                TaskGraphNode(
                    task=TaskCreate(
                        task_id="z-dependent",
                        type="test",
                        retry_policy=TaskRetryPolicy(max_attempts=2),
                    ),
                    prerequisite_task_ids=("a-source",),
                ),
            ),
        )
    )
    before = await store.load_task_graph("atomic-outcome")
    before_events = await store.list_task_graph_events("atomic-outcome")
    original = module.member_from_task

    def late_failure(task):
        if task.id == "z-dependent":
            raise RuntimeError("later terminal evidence failed")
        return original(task)

    with monkeypatch.context() as patch:
        patch.setattr(module, "member_from_task", late_failure)
        with pytest.raises(RuntimeError, match="later terminal evidence"):
            await store.fail_task("a-source", {"code": "test"})
    assert await store.load_task_graph("atomic-outcome") == before
    assert await store.list_task_graph_events("atomic-outcome") == before_events
    await store.fail_task("a-source", {"code": "test"})
    child = await store.load_task("z-dependent")
    assert child is not None and child.status is TaskStatus.DEPENDENCY_SKIPPED


@pytest.mark.parametrize("cursor", [True, -1, 9007199254740992])
async def test_graph_event_cursor_is_portable_and_strict(store: TaskStore, cursor) -> None:
    await store.create_task_graph(graph_request())
    with pytest.raises(ValueError, match="cursor"):
        await store.list_task_graph_events("graph", after_sequence=cursor)
    assert await store.list_task_graph_events("graph", after_sequence=9007199254740991) == []


async def test_public_worker_executes_graph_in_dependency_order(store: TaskStore) -> None:
    from cayu import CayuApp
    from cayu.tasks.worker import complete_managed_task, run_task_worker

    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_graph(graph_request())
    observed = []

    async def handler(_app, task, worker_id):
        for identity in task.prerequisite_task_ids:
            prerequisite = await store.load_task(identity)
            assert prerequisite is not None and prerequisite.status is TaskStatus.COMPLETED
        observed.append(task.id)
        await complete_managed_task(store, task, worker_id, {"done": True})

    count = await asyncio.wait_for(
        run_task_worker(
            app,
            store,
            handler,
            worker_id="graph-worker",
            max_tasks=4,
            poll_interval_s=0.01,
            reclaim=False,
            recover_interrupted_handoffs=False,
        ),
        15,
    )
    assert count == 4
    assert observed[0] == "root" and observed[-1] == "join"
    assert set(observed[1:3]) == {"a", "b"}
    snapshot = await app.load_task_graph("graph")
    assert snapshot is not None and all(
        member.status is TaskStatus.COMPLETED for member in snapshot.members
    )


async def test_verified_completion_atomically_releases_graph_join(store: TaskStore) -> None:
    from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding
    from tests.core.test_completion_decision_application import (
        _contract,
        _persist_decision,
        _result,
    )

    from cayu import CayuApp
    from cayu.sessions.invocation import TaskExecutionSource
    from cayu.tasks.base import task_create_with_runtime_invocation
    from cayu.tasks.contracts import CompletionDecisionApplicationRequest, CompletionVerdict

    contract = await store.publish_work_contract(_contract())
    binding = unattributed_session_invocation_binding("graph-verification")
    root = task_create_with_runtime_invocation(
        TaskCreate(
            task_id="verified-root",
            type="test",
            session_id="graph-verification",
            work_contract=contract.reference(),
        ),
        source=TaskExecutionSource.SDK_TASK,
        session_invocation=binding,
    )
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_graph(
        TaskGraphCreate(
            graph_id="verified-graph",
            nodes=(
                TaskGraphNode(task=root),
                TaskGraphNode(
                    task=TaskCreate(task_id="verified-join", type="test"),
                    prerequisite_task_ids=("verified-root",),
                ),
            ),
        )
    )
    running = await store.start_task("verified-root", session_invocation=binding)
    decision_id, reference = await _persist_decision(
        store, task=running, ordinal=1, verdict=CompletionVerdict.ACCEPTED
    )
    request = CompletionDecisionApplicationRequest(
        task_id=running.id,
        decision_id=decision_id,
        idempotency_key="graph-accepted",
        result=_result("1"),
        result_reference=reference,
    )
    completed = await app.apply_completion_decision(request)
    assert completed.status is TaskStatus.COMPLETED
    assert await app.apply_completion_decision(request) == completed
    claimed = await store.claim_task("verified-join-worker")
    assert claimed is not None and claimed.id == "verified-join"
    events = await app.list_task_graph_events("verified-graph")
    assert (
        len(
            [
                event
                for event in events
                if event.type is TaskGraphEventType.READY and event.task_id == claimed.id
            ]
        )
        == 1
    )


async def test_retry_successor_cannot_replace_exact_graph_prerequisite(store: TaskStore) -> None:
    from cayu.tasks.base import (
        TaskRetryAttemptDisposition,
        TaskRetryPolicy,
        TaskRetrySettlementRequest,
    )

    await store.create_task_graph(
        TaskGraphCreate(
            graph_id="retry-graph",
            nodes=(
                TaskGraphNode(
                    task=TaskCreate(
                        task_id="retry-root",
                        type="test",
                        retry_policy=TaskRetryPolicy(
                            max_attempts=2,
                            initial_backoff_seconds=0,
                        ),
                    )
                ),
                TaskGraphNode(
                    task=TaskCreate(task_id="retry-child", type="test"),
                    prerequisite_task_ids=("retry-root",),
                ),
            ),
        )
    )
    claimed = await store.claim_task("retry-worker")
    assert claimed is not None and claimed.retry_series is not None
    receipt = await store.settle_task_retry_attempt(
        TaskRetrySettlementRequest(
            task_id=claimed.id,
            worker_id="retry-worker",
            lease_expires_at=claimed.lease_expires_at,
            causal_budget_id=claimed.retry_series.causal_budget_id,
            idempotency_key="retry-failure",
            disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
            error={"code": "temporary"},
        )
    )
    assert receipt.successor is not None
    assert receipt.successor.graph_id is None
    child = await store.load_task("retry-child")
    assert child is not None and child.status is TaskStatus.DEPENDENCY_SKIPPED
    successor = await store.claim_task("successor-worker")
    assert successor is not None and successor.id == receipt.successor.id
    assert successor.retry_series is not None
    await store.settle_task_retry_attempt(
        TaskRetrySettlementRequest(
            task_id=successor.id,
            worker_id="successor-worker",
            lease_expires_at=successor.lease_expires_at,
            causal_budget_id=successor.retry_series.causal_budget_id,
            idempotency_key="retry-success",
            disposition=TaskRetryAttemptDisposition.SUCCEEDED,
            result={"done": True},
        )
    )
    assert await store.load_task("retry-child") == child
    assert await store.claim_task("late-worker") is None


async def test_retained_graph_id_collision_cannot_partially_settle_retry(store: TaskStore) -> None:
    from cayu.tasks.base import (
        TaskRetryAttemptDisposition,
        TaskRetryPolicy,
        TaskRetrySettlementRequest,
        _task_retry_successor_id,
    )

    await store.create_task_graph(
        TaskGraphCreate(
            graph_id="source-graph",
            nodes=(
                TaskGraphNode(
                    task=TaskCreate(
                        task_id="collision-source",
                        type="test",
                        retry_policy=TaskRetryPolicy(max_attempts=2),
                    )
                ),
                TaskGraphNode(
                    task=TaskCreate(task_id="collision-child", type="test"),
                    prerequisite_task_ids=("collision-source",),
                ),
            ),
        )
    )
    claimed = await store.claim_task("worker")
    assert claimed is not None and claimed.retry_series is not None
    reserved = _task_retry_successor_id(claimed.retry_series.series_id, 2)
    await store.create_task_graph(
        TaskGraphCreate(
            graph_id="retained",
            nodes=(
                TaskGraphNode(
                    task=TaskCreate(task_id=reserved, type="test", session_id="retained-session")
                ),
            ),
        )
    )
    await store.complete_task(reserved, {})
    await store.delete_session_tasks("retained-session", task_ids=(reserved,), policy=None)
    assert await store.load_task(reserved) is None
    before = await store.load_task_graph("source-graph")
    events = await store.list_task_graph_events("source-graph")
    with pytest.raises(ValueError):
        await store.settle_task_retry_attempt(
            TaskRetrySettlementRequest(
                task_id=claimed.id,
                worker_id="worker",
                lease_expires_at=claimed.lease_expires_at,
                causal_budget_id=claimed.retry_series.causal_budget_id,
                idempotency_key="collision",
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
            )
        )
    assert await store.load_task(claimed.id) == claimed
    assert await store.load_task_graph("source-graph") == before
    assert await store.list_task_graph_events("source-graph") == events
    assert await store.load_task_retry_settlement(claimed.id, "collision") is None


async def test_graph_waiting_schedule_can_move_but_skipped_schedule_is_terminal(
    store: TaskStore,
) -> None:
    from cayu.tasks.scheduling import TaskRescheduleRequest, TaskScheduleConflict

    due = datetime.now(UTC) + timedelta(days=1)
    await store.create_task_graph(
        TaskGraphCreate(
            graph_id="scheduled-graph",
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id="scheduled-root", type="test")),
                TaskGraphNode(
                    task=TaskCreate(
                        task_id="scheduled-child",
                        type="test",
                        available_at=due,
                        schedule_policy=TaskSchedulePolicy(),
                    ),
                    prerequisite_task_ids=("scheduled-root",),
                ),
            ),
        )
    )
    request = TaskRescheduleRequest(
        task_id="scheduled-child",
        operation_id="move",
        expected_revision=1,
        available_at=due + timedelta(days=1),
    )
    await store.reschedule_task(request)
    child = await store.load_task("scheduled-child")
    assert child is not None and child.status is TaskStatus.WAITING_DEPENDENCIES
    assert child.available_at == request.available_at
    await store.fail_task("scheduled-root", {"code": "test"})
    skipped = await store.load_task("scheduled-child")
    assert skipped is not None and skipped.status is TaskStatus.DEPENDENCY_SKIPPED
    assert skipped.schedule is not None
    with pytest.raises(TaskScheduleConflict, match="Terminal"):
        await store.reschedule_task(
            TaskRescheduleRequest(
                task_id=skipped.id,
                operation_id="after-skip",
                expected_revision=skipped.schedule.revision,
                available_at=due,
            )
        )
    assert await store.load_task("scheduled-child") == skipped


async def test_expiry_batch_does_not_resettle_a_skipped_descendant(store: TaskStore) -> None:
    due = datetime.now(UTC) - timedelta(days=2)
    policy = TaskSchedulePolicy(expires_at=due + timedelta(days=1))
    await store.create_task_graph(
        TaskGraphCreate(
            graph_id="expiry-batch",
            nodes=(
                TaskGraphNode(task=TaskCreate(task_id="expiry-root", type="test")),
                TaskGraphNode(
                    task=TaskCreate(
                        task_id="a-expired", type="test", available_at=due, schedule_policy=policy
                    ),
                    prerequisite_task_ids=("expiry-root",),
                ),
                TaskGraphNode(
                    task=TaskCreate(
                        task_id="b-skipped", type="test", available_at=due, schedule_policy=policy
                    ),
                    prerequisite_task_ids=("a-expired",),
                ),
            ),
        )
    )
    claimed = await store.claim_task("expiry-worker", TaskQuery(status=TaskStatus.PENDING))
    assert claimed is not None and claimed.id == "expiry-root"
    expired = await store.load_task("a-expired")
    skipped = await store.load_task("b-skipped")
    assert expired is not None and expired.status is TaskStatus.CANCELLED
    assert skipped is not None and skipped.status is TaskStatus.DEPENDENCY_SKIPPED
    assert await store.claim_task("second-worker") is None
    assert await store.load_task("b-skipped") == skipped


async def test_graph_failure_is_transitive_and_replay_does_not_recreate(store: TaskStore) -> None:
    request = graph_request()
    receipt = await store.create_task_graph(request)
    await store.fail_task("root", {"code": "failed"})
    snapshot = await store.load_task_graph("graph")
    assert snapshot is not None
    assert {member.status for member in snapshot.members if member.task_id != "root"} == {
        TaskStatus.DEPENDENCY_SKIPPED
    }
    assert await store.create_task_graph(request) == receipt
    assert await store.claim_task("worker") is None
    events = await store.list_task_graph_events("graph")
    assert len([event for event in events if event.type is TaskGraphEventType.SKIPPED]) == 3
    assert await store.list_task_graph_events("graph", after_sequence=events[-1].sequence) == []


async def test_graph_waiting_cannot_be_completed_or_started(store: TaskStore) -> None:
    await store.create_task_graph(graph_request())
    with pytest.raises(TaskGraphConflict):
        await store.complete_task("join", {})
    with pytest.raises(ValueError):
        await store.start_task("join")
    waiting = await store.load_task("join")
    assert waiting is not None and waiting.status is TaskStatus.WAITING_DEPENDENCIES


async def test_graph_hold_and_resume_preserve_dependency_gate(store: TaskStore) -> None:
    await store.create_task_graph(graph_request())
    await store.pause_task("join", reason="operator")
    resumed = await store.resume_task("join")
    assert resumed.status is TaskStatus.WAITING_DEPENDENCIES
    await store.pause_task("a", reason="operator")
    await store.complete_task("root", {})
    held = await store.load_task("a")
    assert held is not None and held.status is TaskStatus.PAUSED
    assert (await store.resume_task("a")).status is TaskStatus.PENDING


@pytest.mark.parametrize(
    "changed",
    [
        "type",
        "title",
        "description",
        "session_id",
        "parent_task_id",
        "input",
        "metadata",
        "assigned_agent_name",
        "available_at",
        "retry_policy",
        "invocation_origin",
        "work_contract",
    ],
)
async def test_graph_exact_replay_rejects_changed_node(changed: str, store: TaskStore) -> None:
    from tests.core.test_verified_work_contracts import _contract

    from cayu.sessions.invocation import InvocationOriginClaim
    from cayu.tasks.base import TaskRetryPolicy

    request = graph_request()
    receipt = await store.create_task_graph(request)
    before = await store.load_task_graph(request.graph_id)
    values = {
        "input": {"changed": True},
        "metadata": {"changed": True},
        "available_at": datetime(2100, 1, 1, tzinfo=UTC),
        "retry_policy": TaskRetryPolicy(max_attempts=2),
        "invocation_origin": InvocationOriginClaim(subject="changed"),
        "work_contract": _contract().reference(),
    }
    setattr(request.nodes[0].task, changed, values.get(changed, "changed"))
    with pytest.raises(TaskGraphConflict):
        await store.create_task_graph(request)
    assert await store.load_task_graph(request.graph_id) == before
    assert await store.create_task_graph(graph_request()) == receipt


async def test_graph_collision_leaves_no_partial_members(store: TaskStore) -> None:
    await store.create_task(TaskCreate(task_id="join", type="existing"))
    with pytest.raises(TaskGraphConflict):
        await store.create_task_graph(graph_request())
    assert [task.id for task in await store.list_tasks(TaskQuery())] == ["join"]
    assert await store.load_task_graph("graph") is None


async def test_graph_late_parent_failure_leaves_no_partial_members(store: TaskStore) -> None:
    request = graph_request()
    request.nodes[-1].task.parent_task_id = "missing"
    with pytest.raises(ValueError):
        await store.create_task_graph(request)
    assert await store.list_tasks(TaskQuery()) == []
    assert await store.load_task_graph("graph") is None


@pytest.mark.parametrize("dependencies", [("missing",), ("a",), ("root", "root")])
def test_invalid_graph_rejected(dependencies: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        TaskGraphCreate(
            graph_id="g",
            nodes=(
                TaskGraphNode(
                    task=TaskCreate(task_id="a", type="test"), prerequisite_task_ids=dependencies
                ),
                TaskGraphNode(task=TaskCreate(task_id="root", type="test")),
            ),
        )


def test_cycle_and_empty_graph_rejected() -> None:
    with pytest.raises(ValueError):
        TaskGraphCreate(graph_id="g", nodes=())
    with pytest.raises(ValueError):
        TaskGraphCreate(
            graph_id="g",
            nodes=(
                TaskGraphNode(
                    task=TaskCreate(task_id="a", type="test"), prerequisite_task_ids=("b",)
                ),
                TaskGraphNode(
                    task=TaskCreate(task_id="b", type="test"), prerequisite_task_ids=("a",)
                ),
            ),
        )


async def test_public_sdk_graph_creation_and_inspection(store: TaskStore) -> None:
    from cayu import CayuApp
    from cayu import TaskGraphCreate as PublicGraphCreate

    assert PublicGraphCreate is TaskGraphCreate
    app = CayuApp(task_store=store, enable_logging=False)
    receipt = await app.create_task_graph(graph_request())
    snapshot = await app.load_task_graph(receipt.graph_id)
    assert snapshot is not None and snapshot.receipt == receipt
    assert len(await app.list_task_graph_events(receipt.graph_id)) == 5


async def test_runtime_evidence_preserves_graph_waiting_and_skipped_status(
    store: TaskStore,
) -> None:
    from tests.core.test_runtime_evidence import _create_session

    from cayu import CayuApp
    from cayu.runtime.evidence import RuntimeEvidenceRequest, runtime_evidence

    app = CayuApp(task_store=store, enable_logging=False)
    await _create_session(app.session_store, "graph-evidence")
    request = TaskGraphCreate(
        graph_id="evidence",
        nodes=(
            TaskGraphNode(
                task=TaskCreate(task_id="source", type="test", session_id="graph-evidence")
            ),
            TaskGraphNode(
                task=TaskCreate(task_id="dependent", type="test", session_id="graph-evidence"),
                prerequisite_task_ids=("source",),
            ),
        ),
    )
    await app.create_task_graph(request)
    evidence_request = RuntimeEvidenceRequest(
        root_session_id="graph-evidence", max_sessions=10, max_events=20
    )
    initial = await runtime_evidence(app, evidence_request)
    assert {task.task_id: task.status for task in initial.tasks}[
        "dependent"
    ] == "waiting_dependencies"
    await store.fail_task("source", {"code": "test"})
    final = await runtime_evidence(app, evidence_request)
    assert {task.task_id: task.status for task in final.tasks}["dependent"] == "dependency_skipped"


async def test_public_graph_exact_replay_needs_no_new_parent_lookup(
    store: TaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu import CayuApp

    await store.create_task(TaskCreate(task_id="replay-parent", type="test"))
    app = CayuApp(task_store=store, enable_logging=False)
    request = TaskGraphCreate(
        graph_id="parent-replay",
        nodes=(
            TaskGraphNode(
                task=TaskCreate(task_id="replay-child", type="test", parent_task_id="replay-parent")
            ),
        ),
    )
    receipt = await app.create_task_graph(request)
    before = await app.list_task_graph_events(request.graph_id)

    async def unavailable(*args, **kwargs):
        raise AssertionError("Read-only graph replay must not prepare a new admission.")

    monkeypatch.setattr(store, "load_invocation_snapshot", unavailable)
    monkeypatch.setattr(store, "create_task_graph", unavailable)
    assert await app.create_task_graph(request) == receipt
    assert await app.list_task_graph_events(request.graph_id) == before


async def test_public_graph_replay_binds_exact_resolved_session(
    store: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding

    from cayu import CayuApp
    from cayu.sessions.invocation import TaskExecutionSource
    from cayu.tasks.base import task_create_with_runtime_invocation

    app = CayuApp(task_store=store, enable_logging=False)
    binding = unattributed_session_invocation_binding("graph-session")

    async def resolve_session(session_id):
        assert session_id == "graph-session"
        return binding

    monkeypatch.setattr(app.session_store, "load_invocation_snapshot", resolve_session)
    request = TaskGraphCreate(
        graph_id="session-bound",
        nodes=(
            TaskGraphNode(
                task=TaskCreate(task_id="member", type="test", session_id="graph-session")
            ),
        ),
    )
    receipt = await app.create_task_graph(request)
    task = await store.load_task("member")
    assert task is not None and task.session_instance_id == binding.session_instance_id
    assert task.invocation.root_invocation_id == binding.invocation.root_invocation_id
    assert await app.create_task_graph(request) == receipt
    before = await app.list_task_graph_events(request.graph_id)
    binding = unattributed_session_invocation_binding("graph-session")
    # Replay returns the original committed operation, never rebinds its tasks.
    assert await app.create_task_graph(request) == receipt
    changed = TaskGraphCreate(
        graph_id=request.graph_id,
        nodes=(
            TaskGraphNode(
                task=task_create_with_runtime_invocation(
                    request.nodes[0].task,
                    source=TaskExecutionSource.SDK_TASK,
                    session_invocation=binding,
                )
            ),
        ),
    )
    with pytest.raises(TaskGraphConflict):
        await app.create_task_graph(changed)

    async def unavailable(session_id):
        raise AssertionError("Read-only replay must not resolve a replacement session.")

    monkeypatch.setattr(app.session_store, "load_invocation_snapshot", unavailable)
    assert await app.create_task_graph(request) == receipt
    assert await store.load_task("member") == task
    assert await app.list_task_graph_events(request.graph_id) == before


async def test_public_graph_mutated_input_has_no_diagnostic_side_channel(
    store: TaskStore, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    import warnings

    from cayu import CayuApp
    from cayu.vaults.redaction import SecretRedactor

    secret = "graph-mutated-input-secret-canary"

    class HostileValue:
        def __repr__(self):
            return secret

        __str__ = __repr__

    request = graph_request()
    request.nodes[0].task.input = {"invalid": HostileValue()}
    request.nodes[-1].task.title = secret
    app = CayuApp(task_store=store, secret_redactor=SecretRedactor(secret), enable_logging=False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, TypeError)) as raised:
            await app.create_task_graph(request)
    pending = [raised.value]
    seen = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        assert secret not in str(error) + repr(error)
        pending.extend(child for child in (error.__cause__, error.__context__) if child is not None)
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    output = capsys.readouterr()
    assert secret not in caplog.text + output.out + output.err
    assert all(secret not in str(item.message) for item in recorded)
    assert await store.load_task_graph("graph") is None
    assert await store.load_task("root") is None


async def test_public_graph_inherited_identity_is_validated_before_dispatch(
    store: TaskStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import warnings

    from cayu import CayuApp
    from cayu.sessions.invocation import InvocationOriginClaim
    from cayu.vaults.redaction import SecretRedactor

    secret = "graph-inherited-origin-secret-canary"
    await store.create_task(
        TaskCreate(
            task_id="sensitive-parent",
            type="test",
            invocation_origin=InvocationOriginClaim(subject=secret),
        )
    )
    calls = []
    original = store.create_task_graph

    async def recording(request):
        calls.append(request.graph_id)
        return await original(request)

    monkeypatch.setattr(store, "create_task_graph", recording)
    app = CayuApp(task_store=store, secret_redactor=SecretRedactor(secret), enable_logging=False)
    request = TaskGraphCreate(
        graph_id="inherited",
        nodes=(
            TaskGraphNode(
                task=TaskCreate(task_id="child", type="test", parent_task_id="sensitive-parent")
            ),
        ),
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="inherited invocation authority") as raised:
            await app.create_task_graph(request)
    assert calls == []
    assert await store.load_task("child") is None
    assert await store.load_task_graph("inherited") is None
    output = capsys.readouterr()
    assert (
        secret not in str(raised.value) + repr(raised.value) + caplog.text + output.out + output.err
    )
    assert all(secret not in str(item.message) for item in recorded)
    # An ordinary inherited identity still reaches the backend and survives
    # persistent reconstruction, independent of the negative case above.
    parent = await store.create_task(TaskCreate(task_id="safe-parent", type="test"))
    safe = TaskGraphCreate(
        graph_id="safe-inherited",
        nodes=(
            TaskGraphNode(
                task=TaskCreate(task_id="safe-child", type="test", parent_task_id="safe-parent")
            ),
        ),
    )
    await app.create_task_graph(safe)
    child = await store.load_task("safe-child")
    assert child is not None and child.invocation.origin == parent.invocation.origin
    assert calls == ["safe-inherited"]


@pytest.mark.parametrize("fault", ["replacement", "lost_preparation"])
async def test_public_graph_rejects_parent_replacement_after_preflight(
    store: TaskStore, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from cayu import CayuApp
    from cayu.sessions.invocation import InvocationOriginClaim
    from cayu.vaults.redaction import SecretRedactor

    secret = "graph-parent-replacement-secret-canary"
    await store.create_task(TaskCreate(task_id="parent", type="test", session_id="old-session"))
    await store.complete_task("parent", {})
    original = store.create_task_graph

    async def replace_parent_then_admit(request):
        if fault == "lost_preparation":
            request._parent_invocations = ()
        else:
            await store.delete_session_tasks("old-session", task_ids=("parent",), policy=None)
            await store.create_task(
                TaskCreate(
                    task_id="parent",
                    type="test",
                    invocation_origin=InvocationOriginClaim(subject=secret),
                )
            )
        return await original(request)

    monkeypatch.setattr(store, "create_task_graph", replace_parent_then_admit)
    app = CayuApp(task_store=store, secret_redactor=SecretRedactor(secret), enable_logging=False)
    request = TaskGraphCreate(
        graph_id="parent-replacement",
        nodes=(
            TaskGraphNode(task=TaskCreate(task_id="child", type="test", parent_task_id="parent")),
        ),
    )
    with pytest.raises(TaskGraphConflict, match="parent authority"):
        await app.create_task_graph(request)
    assert await store.load_task("child") is None
    assert await store.load_task_graph(request.graph_id) is None


async def test_late_member_event_failure_leaves_no_partial_admission(
    store: TaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.storage import postgres, sqlite
    from cayu.tasks import base

    module = (
        base
        if isinstance(store, InMemoryTaskStore)
        else postgres
        if isinstance(store, PostgresTaskStore)
        else sqlite
    )
    original = module.schedule_transition_events

    def fail_later(prior, current, **kwargs):
        if current.id == "b":
            raise RuntimeError("later member schedule publication failed")
        return original(prior, current, **kwargs)

    request = TaskGraphCreate(
        graph_id="atomic",
        nodes=tuple(
            TaskGraphNode(
                task=TaskCreate(
                    task_id=identity,
                    type="test",
                    available_at=datetime.now(UTC),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            for identity in ("a", "b")
        ),
    )
    with monkeypatch.context() as patch:
        patch.setattr(module, "schedule_transition_events", fail_later)
        with pytest.raises(RuntimeError, match="later member"):
            await store.create_task_graph(request)
    assert await store.list_tasks() == []
    assert await store.load_task_graph("atomic") is None
    assert (await store.create_task_graph(request)).task_ids == ("a", "b")


async def test_graph_retains_cross_session_members_until_terminal(store: TaskStore) -> None:
    request = TaskGraphCreate(
        graph_id="closure",
        nodes=(
            TaskGraphNode(task=TaskCreate(task_id="a", type="test", session_id="first")),
            TaskGraphNode(
                task=TaskCreate(task_id="b", type="test", session_id="second"),
                prerequisite_task_ids=("a",),
            ),
        ),
    )
    await store.create_task_graph(request)
    await store.complete_task("a", {})
    claim = TaskSessionClosureClaim(session_id="first", plan_id="c" * 64, task_ids=("a",))
    with pytest.raises(TaskGraphConflict):
        await store.claim_session_closure(claim)
    with pytest.raises(TaskGraphConflict):
        await store.delete_session_tasks("first", task_ids=("a",), policy=None)
    await store.fail_task("b", {"code": "test"})
    await store.claim_session_closure(claim)
    await store.delete_session_tasks("first", task_ids=("a",), policy=None)
    snapshot = await store.load_task_graph("closure")
    assert snapshot is not None and snapshot.members[0].status is TaskStatus.COMPLETED
    assert await store.create_task_graph(request) == snapshot.receipt
    with pytest.raises(TaskGraphConflict):
        await store.create_task(TaskCreate(task_id="a", type="test"))


async def test_sqlite_graph_readiness_wakeup_requires_committed_evidence(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteTaskStore(tmp_path / "wakeup.sqlite")
    observed = []
    try:
        await store.create_task_graph(graph_request())

        def notified():
            assert not store._connection.in_transaction
            observed.append(True)

        monkeypatch.setattr(store, "_publish_task_admission_broadcast", notified)
        original = store._record_schedule_transition_unlocked

        def fail_after_graph(prior, current, **kwargs):
            if current.id == "root" and current.status is TaskStatus.COMPLETED:
                raise RuntimeError("late publication failure")
            return original(prior, current, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(store, "_record_schedule_transition_unlocked", fail_after_graph)
            with pytest.raises(RuntimeError, match="late publication"):
                await store.complete_task("root", {})
        await asyncio.sleep(0)
        assert observed == []
        root = await store.load_task("root")
        assert root is not None and root.status is TaskStatus.PENDING
        await store.complete_task("root", {})
        await asyncio.sleep(0)
        assert observed == [True]
        claimed = await store.claim_task("worker")
        assert claimed is not None and claimed.id in {"a", "b"}
    finally:
        await store.close()


async def test_sqlite_reopen_and_independent_completion_release_join_once(tmp_path) -> None:
    path = tmp_path / "reopen.sqlite"
    first = SQLiteTaskStore(path)
    receipt = await first.create_task_graph(graph_request())
    await first.complete_task("root", {})
    await first.close()
    left, right = SQLiteTaskStore(path), SQLiteTaskStore(path)
    try:
        assert await right.create_task_graph(graph_request()) == receipt
        await asyncio.gather(left.complete_task("a", {}), right.complete_task("b", {}))
        claims = await asyncio.gather(left.claim_task("left"), right.claim_task("right"))
        assert [task.id for task in claims if task is not None] == ["join"]
        events = await right.list_task_graph_events("graph")
        assert (
            len(
                [
                    event
                    for event in events
                    if event.type is TaskGraphEventType.READY and event.task_id == "join"
                ]
            )
            == 1
        )
    finally:
        await left.close()
        await right.close()


async def test_sqlite_graph_readback_rejects_member_authority_substitution(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "corrupt.sqlite")
    try:
        await store.create_task_graph(graph_request())
        store._connection.execute("UPDATE cayu_tasks SET graph_id = 'other' WHERE id = 'root'")
        store._connection.commit()
        with pytest.raises(TaskGraphUnavailable):
            await store.load_task_graph("graph")
    finally:
        await store.close()


async def test_public_graph_acknowledgement_cancellation_replays_without_republication(
    store: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu import CayuApp

    committed = asyncio.Event()
    release = asyncio.Event()

    original = store.create_task_graph
    first = True

    async def delayed(request):
        nonlocal first
        receipt = await original(request)
        if first:
            first = False
            committed.set()
            await release.wait()
        return receipt

    monkeypatch.setattr(store, "create_task_graph", delayed)
    app = CayuApp(task_store=store, enable_logging=False)
    caller = asyncio.create_task(app.create_task_graph(graph_request()))
    try:
        await asyncio.wait_for(committed.wait(), 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert caller.cancelled() and caller.cancelling() == 1
        before = await app.list_task_graph_events("graph")
        replay = await app.create_task_graph(graph_request())
        assert replay.task_ids == ("a", "b", "join", "root")
        assert await app.list_task_graph_events("graph") == before
    finally:
        release.set()
        if not caller.done():
            caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.parametrize(
    "field", ["graph_id", "request_sha256", "submitted_request_sha256", "task_ids"]
)
async def test_public_graph_rejects_receipt_substitution(field: str) -> None:
    from cayu import CayuApp

    class SubstitutingStore(InMemoryTaskStore):
        async def create_task_graph(self, request):
            receipt = await super().create_task_graph(request)
            value = {
                "graph_id": "other",
                "request_sha256": "e" * 64,
                "submitted_request_sha256": "f" * 64,
                "task_ids": ("other",),
            }[field]
            return receipt.model_copy(update={field: value})

    app = CayuApp(task_store=SubstitutingStore(), enable_logging=False)
    with pytest.raises(TaskGraphConflict):
        await app.create_task_graph(graph_request())
