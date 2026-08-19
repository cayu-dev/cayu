"""Postgres TaskStore parity tests.

Mirror the conformance assertions in ``test_task_store.py`` against a real
Dockerized Postgres. They skip automatically when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

import pytest
from psycopg import errors as psycopg_errors
from pydantic import ValidationError
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)
from tests.core.task_store_conformance import (
    assert_task_claim_lost_conformance,
    assert_task_session_invocation_binding_conformance,
)
from tests.core.task_terminalization_conformance import (
    assert_task_terminalization_acknowledgement_conformance,
)
from tests.core.task_topology_conformance import (
    assert_task_topology_bounded_projection_conformance,
    assert_task_topology_store_conformance,
)

from cayu import (
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskExecutionSource,
    TaskInvocation,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalizationRetryPolicy,
    TaskTerminalKind,
    TaskTopologyQuery,
    task_create_with_execution_source,
    terminalize_task_with_retry,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    extract_durable_value_error,
)

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
    "cayu_knowledge_embeddings",
    "cayu_task_terminalization_receipts",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_revisions",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_eval_results",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
    "cayu_schema_migrations",
)


def test_postgres_task_store_replays_terminalization_and_receipt(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_terminal", type="review"))
        assert await store.claim_task("worker_a") is not None
        request = TaskTerminalizationRequest(
            task_id="task_terminal",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done", "metrics": {"changed": 2, "checked": 4}},
            idempotency_key="terminal-attempt-1",
        )

        first = await store.terminalize_task(request)
        replayed = await store.terminalize_task(request)
        receipt = await store.load_task_terminalization_receipt(
            "task_terminal", "terminal-attempt-1"
        )

        assert replayed == first
        assert type(receipt) is TaskTerminalizationReceipt
        assert receipt.task == first
        assert receipt.worker_id == "worker_a"
        assert receipt.request_sha256 == (
            "f44314f4f13d93a708c544e83a90ecb2e2dea4d6dd7f4ceb0512b2f895d364a8"
        )

    _run(postgres_dsn, ops)


def test_postgres_task_store_terminalization_acknowledgement_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_terminalization_acknowledgement_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_connection_failure_subclass_is_acknowledgement_ambiguous(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_connection_failure", type="review"))
        assert await store.claim_task("worker_a") is not None
        terminalize = store.terminalize_task
        calls = 0

        async def fail_before_commit_once(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise psycopg_errors.ConnectionFailure("acknowledgement lost")
            return await terminalize(request)

        store.terminalize_task = fail_before_commit_once
        outcome = await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="task_connection_failure",
                worker_id="worker_a",
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done"},
                idempotency_key="connection-failure",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            ),
        )

        assert outcome.attempt_count == 2
        assert calls == 2

    _run(postgres_dsn, ops)


def test_postgres_task_store_terminalization_rejects_wrong_worker_and_changed_intent(
    postgres_dsn,
):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_wrong_worker", type="review"))
        assert await store.claim_task("worker_a") is not None
        winner = TaskTerminalizationRequest(
            task_id="task_wrong_worker",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="terminal-key",
        )
        with pytest.raises(TaskClaimLost):
            await store.terminalize_task(winner.model_copy(update={"worker_id": "worker_b"}))
        assert (
            await store.load_task_terminalization_receipt("task_wrong_worker", "terminal-key")
            is None
        )

        terminal = await store.terminalize_task(winner)
        conflicts = (
            winner.model_copy(update={"worker_id": "worker_b"}),
            winner.model_copy(update={"result": {"summary": "changed"}}),
            TaskTerminalizationRequest(
                task_id="task_wrong_worker",
                worker_id="worker_a",
                kind=TaskTerminalKind.FAILED,
                error={"message": "changed"},
                idempotency_key="terminal-key",
            ),
        )
        for conflicting in conflicts:
            with pytest.raises(TaskTerminalizationConflict):
                await store.terminalize_task(conflicting)
        assert await store.load_task("task_wrong_worker") == terminal

    _run(postgres_dsn, ops)


def test_postgres_task_store_terminalization_concurrency_converges_or_conflicts(
    postgres_dsn,
):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_exact_race", type="review"))
        assert await store.claim_task("worker_a") is not None
        exact = TaskTerminalizationRequest(
            task_id="task_exact_race",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="race-key",
        )
        exact_results = await asyncio.gather(*(store.terminalize_task(exact) for _ in range(8)))
        assert all(result == exact_results[0] for result in exact_results)

        await store.create_task(TaskCreate(task_id="task_conflict_race", type="review"))
        assert await store.claim_task("worker_b") is not None
        requests = (
            TaskTerminalizationRequest(
                task_id="task_conflict_race",
                worker_id="worker_b",
                kind=TaskTerminalKind.COMPLETED,
                result={"winner": "completed"},
                idempotency_key="conflict-key",
            ),
            TaskTerminalizationRequest(
                task_id="task_conflict_race",
                worker_id="worker_b",
                kind=TaskTerminalKind.FAILED,
                error={"winner": "failed"},
                idempotency_key="conflict-key",
            ),
        )

        async def apply(request: TaskTerminalizationRequest):
            try:
                return await store.terminalize_task(request)
            except TaskTerminalizationConflict as exc:
                return exc

        outcomes = await asyncio.gather(*(apply(request) for request in requests))
        assert sum(type(outcome) is Task for outcome in outcomes) == 1
        assert sum(isinstance(outcome, TaskTerminalizationConflict) for outcome in outcomes) == 1

    _run(postgres_dsn, ops)


def test_postgres_task_store_replays_terminalization_after_reconstruction(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        request = TaskTerminalizationRequest(
            task_id="task_restart",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="restart-key",
        )
        first_store = _new_store(postgres_dsn)
        try:
            await first_store.create_task(TaskCreate(task_id="task_restart", type="review"))
            assert await first_store.claim_task("worker_a") is not None
            terminal = await first_store.terminalize_task(request)
        finally:
            await first_store.close()

        reconstructed = _new_store(postgres_dsn)
        try:
            assert await reconstructed.terminalize_task(request) == terminal
            receipt = await reconstructed.load_task_terminalization_receipt(
                "task_restart", "restart-key"
            )
            assert receipt is not None
            assert receipt.task == terminal
        finally:
            await reconstructed.close()

    asyncio.run(run())


async def _truncate(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _new_store(dsn: str, *, clock=None):
    from cayu import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    # Tests own a throwaway database and (re)create the schema each run.
    return PostgresTaskStore(
        dsn,
        min_size=1,
        max_size=4,
        clock=clock,
        schema_mode=SchemaMode.CREATE,
    )


def _run(dsn: str, coro_factory) -> object:
    async def runner():
        await _truncate(dsn)
        store = _new_store(dsn)
        try:
            return await coro_factory(store)
        finally:
            await store.close()

    return asyncio.run(runner())


def test_postgres_task_store_task_claim_lost_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_claim_lost_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_store_binds_session_identity_to_invocation(postgres_dsn):
    async def ops(store):
        await assert_task_session_invocation_binding_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_store_persists_and_inherits_invocation_provenance(postgres_dsn):
    async def ops(store):
        root = await store.create_task(
            task_create_with_execution_source(
                TaskCreate(
                    task_id="provenance-root",
                    type="webhook",
                    invocation_origin=InvocationOriginClaim(
                        subject="github:org/repo",
                        tenant="customer-a",
                    ),
                ),
                source=TaskExecutionSource.WEBHOOK,
            )
        )
        child = await store.create_task(
            TaskCreate(
                task_id="provenance-child",
                type="step",
                parent_task_id=root.id,
            )
        )
        assert root.invocation.origin.trust is InvocationOriginTrust.HOST_ASSERTED
        assert child.invocation.origin == root.invocation.origin
        assert child.invocation.root_invocation_id == root.invocation.root_invocation_id
        assert child.invocation.source is TaskExecutionSource.SDK_TASK

        snapshot = await store.load_invocation_snapshot(child.id)
        assert snapshot is not None
        assert snapshot.id == child.id
        assert snapshot.session_id == child.session_id
        assert snapshot.invocation == child.invocation
        assert await store.load_invocation_snapshot("missing") is None

        reopened = _new_store(postgres_dsn)
        try:
            loaded = await reopened.load_task(child.id)
            assert loaded is not None
            assert loaded.invocation == child.invocation
        finally:
            await reopened.close()

    _run(postgres_dsn, ops)


def test_postgres_persists_availability_and_claims_once_at_exact_boundary(
    postgres_dsn,
):
    async def run() -> None:
        await _truncate(postgres_dsn)
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _MutableClock(boundary - timedelta(microseconds=1))
        creator = _new_store(postgres_dsn, clock=clock)
        try:
            await creator.create_task(
                TaskCreate(
                    task_id="durable-future",
                    type="scheduled",
                    available_at=boundary,
                )
            )
            await creator.create_task(
                TaskCreate(
                    task_id="after-boundary",
                    type="scheduled",
                    available_at=boundary + timedelta(microseconds=1),
                )
            )
            before = await creator.aggregate_operational_snapshot()
            assert before.claimable_pending_count == 0
            assert before.scheduled_pending_count == 2
            assert await creator.claim_task("worker-before") is None
        finally:
            await creator.close()

        reconstructed = _new_store(postgres_dsn, clock=clock)
        try:
            loaded = await reconstructed.load_task("durable-future")
            assert loaded is not None
            assert loaded.available_at == boundary

            clock.value = boundary
            at_boundary = await reconstructed.aggregate_operational_snapshot()
            assert at_boundary.claimable_pending_count == 1
            assert at_boundary.scheduled_pending_count == 1
            claims = await asyncio.gather(
                *(reconstructed.claim_task(f"worker-{index}") for index in range(8))
            )
            winners = [claim for claim in claims if claim is not None]
            assert len(winners) == 1
            assert winners[0].id == "durable-future"
            assert winners[0].available_at == boundary

            clock.value = boundary + timedelta(microseconds=1)
            after = await reconstructed.claim_task("worker-after")
            assert after is not None
            assert after.id == "after-boundary"
        finally:
            await reconstructed.close()

    asyncio.run(run())


def test_postgres_injected_availability_clock_does_not_expire_new_task_lease(
    postgres_dsn,
):
    async def run() -> None:
        await _truncate(postgres_dsn)
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        store = _new_store(postgres_dsn, clock=_MutableClock(boundary))
        try:
            await store.create_task(
                TaskCreate(
                    task_id="lease-clock",
                    type="scheduled",
                    available_at=boundary,
                )
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            released = await store.release_task("lease-clock", "worker")
            assert released.status is TaskStatus.PENDING
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_production_claim_uses_database_clock(postgres_dsn):
    async def run() -> None:
        import psycopg

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await store.ensure_schema()
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute("SELECT transaction_timestamp()")
                row = await cur.fetchone()
                assert row is not None
                database_now = row[0]

            available_at = database_now + timedelta(minutes=5)
            await store.create_task(
                TaskCreate(
                    task_id="database-clock-authority",
                    type="scheduled",
                    available_at=available_at,
                )
            )

            # Simulate a worker process whose wall clock is far ahead. The
            # production claim path must not consult it for eligibility.
            store._clock = lambda: available_at + timedelta(hours=1)
            assert await store.claim_task("clock-skewed-worker") is None
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_store_task_topology_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_topology_store_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_topology_bounded_projection_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_topology_bounded_projection_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_topology_uses_canonical_id_ordering(postgres_dsn):
    async def ops(store):
        import psycopg

        await store.ensure_schema()
        invocation = TaskInvocation(
            origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
            root_invocation_id=str(uuid4()),
            root_session_id="collation-session",
            source=TaskExecutionSource.SDK_TASK,
        ).model_dump_json()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO cayu_tasks (
                        id, type, status, session_id, input, metadata,
                        created_at, updated_at, invocation
                    )
                    VALUES
                        (
                            'collation-a', 'step', 'pending', 'collation-session',
                            '{}'::jsonb, '{}'::jsonb,
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            %s::jsonb
                        ),
                        (
                            'collation-B', 'step', 'pending', 'collation-session',
                            '{}'::jsonb, '{}'::jsonb,
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            %s::jsonb
                        )
                    """,
                    (invocation, invocation),
                )
            await conn.commit()

        first = await store.query_task_topology(
            TaskTopologyQuery(
                linked_session_ids=("collation-session",),
                session_task_limit=1,
            )
        )
        first_branch = first.session_branches[0]
        assert [task.id for task in first_branch.tasks] == ["collation-B"]
        assert first_branch.has_more is True
        assert first_branch.next_cursor is not None

        continuation = await store.query_task_topology(
            TaskTopologyQuery(
                linked_session_ids=("collation-session",),
                session_cursors={"collation-session": first_branch.next_cursor},
                session_task_limit=1,
            )
        )
        continuation_branch = continuation.session_branches[0]
        assert [task.id for task in continuation_branch.tasks] == ["collation-a"]
        assert continuation_branch.has_more is False
        assert continuation_branch.next_cursor is None

    _run(postgres_dsn, ops)


def test_postgres_task_topology_branch_plan_is_bounded(postgres_dsn):
    async def ops(store):
        import psycopg

        parent = await store.create_task(
            TaskCreate(
                task_id="topology-plan-parent",
                type="workflow",
                session_id="topology-plan-session",
            )
        )
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                INSERT INTO cayu_tasks (
                    id, type, status, session_id, parent_task_id,
                    input, metadata, created_at, updated_at, invocation
                )
                SELECT
                    'topology-plan-child-' || lpad(value::text, 6, '0'),
                    'step',
                    'pending',
                    'topology-plan-session',
                    'topology-plan-parent',
                    '{}'::jsonb,
                    '{}'::jsonb,
                    TIMESTAMPTZ '2026-01-01T00:00:00Z',
                    TIMESTAMPTZ '2026-01-01T00:00:00Z',
                    %s::jsonb
                FROM generate_series(0, 99999) AS value
                """,
                (parent.invocation.model_dump_json(),),
            )
            await conn.commit()
            await cur.execute("SET LOCAL enable_seqscan = off")

            async def explain(
                scope_column: Literal["session_id", "parent_task_id"],
                branch_id: str,
            ):
                await cur.execute(
                    f"""
                    EXPLAIN (ANALYZE, COSTS OFF, FORMAT JSON)
                    WITH requested_branches AS (
                        SELECT branch_id, cursor_created_at, cursor_id,
                               candidate_limit, branch_order
                        FROM unnest(
                            %s::text[],
                            %s::timestamptz[],
                            %s::text[],
                            %s::integer[]
                        ) WITH ORDINALITY AS requested(
                            branch_id,
                            cursor_created_at,
                            cursor_id,
                            candidate_limit,
                            branch_order
                        )
                    )
                    SELECT child.*
                    FROM requested_branches AS requested
                    CROSS JOIN LATERAL (
                        SELECT id, created_at
                        FROM cayu_tasks
                        WHERE cayu_tasks.{scope_column} = requested.branch_id
                          AND (
                              requested.cursor_created_at IS NULL
                              OR cayu_tasks.created_at > requested.cursor_created_at
                              OR (
                                  cayu_tasks.created_at = requested.cursor_created_at
                                  AND cayu_tasks.id COLLATE "C" >
                                      requested.cursor_id COLLATE "C"
                              )
                          )
                        ORDER BY cayu_tasks.created_at ASC,
                                 cayu_tasks.id COLLATE "C" ASC
                        LIMIT requested.candidate_limit
                    ) AS child
                    ORDER BY requested.branch_order ASC,
                             child.created_at ASC,
                             child.id COLLATE "C" ASC
                    """,
                    ([branch_id], [None], [None], [26]),
                )
                return (await cur.fetchone())[0][0]["Plan"]

            parent_plan = await explain("parent_task_id", "topology-plan-parent")
            session_plan = await explain("session_id", "topology-plan-session")

        def plan_nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from plan_nodes(child)

        for plan, index_name in (
            (parent_plan, "idx_cayu_tasks_parent_created_id"),
            (session_plan, "idx_cayu_tasks_session_created_id"),
        ):
            index_nodes = [
                node for node in plan_nodes(plan) if node.get("Index Name") == index_name
            ]
            assert index_nodes
            assert all(node["Actual Rows"] <= 26 for node in index_nodes)

    _run(postgres_dsn, ops)


def test_postgres_task_store_create_load_and_copy_boundary(postgres_dsn):
    async def ops(store):
        request_input = {"invoice_id": "inv_123", "lines": [{"amount": 25}]}
        task = await store.create_task(
            TaskCreate(
                task_id="task_invoice",
                type="process_invoice",
                title="Process invoice",
                description="Extract and post invoice fields.",
                session_id="sess_invoice",
                assigned_agent_name="invoice_agent",
                input=request_input,
                metadata={"source": "webhook"},
            )
        )
        request_input["lines"][0]["amount"] = 999

        loaded = await store.load_task("task_invoice")
        assert loaded is not None
        assert task.status == TaskStatus.PENDING
        assert loaded.input == {"invoice_id": "inv_123", "lines": [{"amount": 25}]}
        assert loaded.metadata == {"source": "webhook"}

        loaded.input["invoice_id"] = "mutated"
        loaded_again = await store.load_task("task_invoice")
        assert loaded_again is not None
        assert loaded_again.input["invoice_id"] == "inv_123"

    _run(postgres_dsn, ops)


def test_postgres_task_store_creates_running_task_atomically(postgres_dsn):
    async def ops(store):
        running = await store.create_running_task(
            TaskCreate(
                task_id="task_atomic_run",
                type="run",
                session_id="sess_atomic_run",
                input={"prompt": "hello"},
            ),
            session_invocation=unattributed_session_invocation_binding("sess_atomic_run"),
        )

        assert running.status is TaskStatus.RUNNING
        assert running.session_id == "sess_atomic_run"
        assert running.started_at is not None
        assert running.completed_at is None
        assert await store.claim_task("worker_a") is None

        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_running_task(
                TaskCreate(
                    task_id="task_atomic_run",
                    type="duplicate",
                    session_id="sess_other",
                ),
                session_invocation=unattributed_session_invocation_binding("sess_other"),
            )
        with pytest.raises(ValueError, match="session_id is required"):
            await store.create_running_task(
                TaskCreate(task_id="task_missing_session", type="run"),
                session_invocation=unattributed_session_invocation_binding("sess_missing"),
            )
        assert await store.load_task("task_missing_session") is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_lifecycle_and_terminal_guards(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_lifecycle", type="analyze_repository"))

        running = await store.start_task(
            "task_lifecycle",
            session_id="sess_analysis",
            session_invocation=await task_backed_session_invocation(
                store, "task_lifecycle", "sess_analysis"
            ),
        )
        assert running.status == TaskStatus.RUNNING
        assert running.session_id == "sess_analysis"
        assert running.started_at is not None
        assert running.completed_at is None

        completed = await store.complete_task("task_lifecycle", {"summary": "done"})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == {"summary": "done"}
        assert completed.error is None
        assert completed.completed_at is not None

        with pytest.raises(ValueError, match="already terminal"):
            await store.fail_task("task_lifecycle", {"message": "too late"})

        with pytest.raises(KeyError, match="Task not found"):
            await store.start_task("missing_task")

    _run(postgres_dsn, ops)


def test_postgres_task_store_hold_resume_and_attention_states(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_blocked", type="review"))
        await store.create_task(TaskCreate(task_id="task_attention", type="review"))
        await store.create_task(TaskCreate(task_id="task_pause_claim", type="review"))

        blocked = await store.block_task(
            "task_blocked",
            reason="Waiting on vendor API",
            payload={"dependency": "vendor_api"},
        )
        assert blocked.status == TaskStatus.BLOCKED
        assert blocked.status_reason == "Waiting on vendor API"
        assert blocked.status_payload == {"dependency": "vendor_api"}

        attention = await store.mark_task_needs_attention(
            "task_attention",
            reason="Operator approval required",
            payload={"field": "amount"},
        )
        assert attention.status == TaskStatus.NEEDS_ATTENTION

        claimed = await store.claim_task(
            "worker_a",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
        )
        assert claimed is not None
        assert claimed.id == "task_pause_claim"

        paused = await store.pause_task("task_pause_claim", reason="Worker shutting down")
        assert paused.status == TaskStatus.PAUSED
        assert paused.worker_id is None
        assert paused.lease_expires_at is None

        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None

        resumed = await store.resume_task("task_blocked")
        assert resumed.status == TaskStatus.PENDING
        assert resumed.status_reason is None
        assert resumed.status_payload is None

        claimed_after_resume = await store.claim_task("worker_c", TaskQuery(type="review"))
        assert claimed_after_resume is not None
        assert claimed_after_resume.id == "task_blocked"

        with pytest.raises(ValueError, match="not paused, blocked, or waiting"):
            await store.resume_task("task_blocked")

        escalated = await store.block_task(
            "task_attention",
            reason="Waiting on supervisor decision",
        )
        assert escalated.status == TaskStatus.BLOCKED
        assert escalated.status_reason == "Waiting on supervisor decision"
        assert escalated.status_payload is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_does_not_hold_attached_running_tasks(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_hold", type="review"))
        await store.start_task(
            "task_attached_hold",
            session_id="sess_attached_hold",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_hold", "sess_attached_hold"
            ),
        )

        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.pause_task("task_attached_hold", reason="not allowed")
        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.block_task("task_attached_hold", reason="not allowed")
        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.mark_task_needs_attention("task_attached_hold", reason="not allowed")

        loaded = await store.load_task("task_attached_hold")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.status_reason is None
        assert loaded.status_payload is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_list_tasks_with_filters_and_pagination(postgres_dsn):
    async def ops(store):
        await store.create_task(
            TaskCreate(
                task_id="task_1",
                type="process_invoice",
                session_id="sess_1",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_2",
                type="process_invoice",
                session_id="sess_2",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_3",
                type="review_report",
                parent_task_id="task_2",
                assigned_agent_name="reviewer",
            )
        )
        await store.start_task(
            "task_1",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_1",
                "sess_1",
            ),
        )
        await store.complete_task("task_2", {"posted": True})

        invoice_tasks = await store.list_tasks(
            TaskQuery(type="process_invoice", order_by=TaskOrder.CREATED_AT_ASC)
        )
        invoice_agent_tasks = await store.list_tasks(
            TaskQuery(assigned_agent_name="invoice_agent", order_by=TaskOrder.CREATED_AT_ASC)
        )
        completed_tasks = await store.list_tasks(TaskQuery(status=TaskStatus.COMPLETED))
        child_tasks = await store.list_tasks(TaskQuery(parent_task_id="task_2"))
        search_tasks = await store.list_tasks(
            TaskQuery(q="invoice", order_by=TaskOrder.CREATED_AT_ASC)
        )
        search_parent_tasks = await store.list_tasks(
            TaskQuery(q="TASK_2", order_by=TaskOrder.CREATED_AT_ASC)
        )
        paged_tasks = await store.list_tasks(
            TaskQuery(limit=1, offset=1, order_by=TaskOrder.CREATED_AT_ASC)
        )

        assert [t.id for t in invoice_tasks] == ["task_1", "task_2"]
        assert [t.id for t in invoice_agent_tasks] == ["task_1", "task_2"]
        assert [t.id for t in completed_tasks] == ["task_2"]
        assert [t.id for t in child_tasks] == ["task_3"]
        assert [t.id for t in search_tasks] == ["task_1", "task_2"]
        assert [t.id for t in search_parent_tasks] == ["task_2", "task_3"]
        assert [t.id for t in paged_tasks] == ["task_2"]

    _run(postgres_dsn, ops)


def test_postgres_task_store_reject_duplicate_tasks_and_invalid_payloads(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="JSON-compatible"):
            await store.complete_task("task_duplicate", {"bad": object()})

        with pytest.raises(ValueError, match="JSON object"):
            await store.fail_task("task_duplicate", ["not", "an", "object"])  # type: ignore[arg-type]

    _run(postgres_dsn, ops)


def test_postgres_task_store_revalidates_portable_values_before_atomic_mutation(postgres_dsn):
    async def ops(store):
        poisoned_create = TaskCreate(
            task_id="task_poisoned_create",
            type="demo",
            input={"safe": True},
        )
        poisoned_create.input["bad"] = float("nan")
        with pytest.raises((DurableValueError, ValidationError)) as invalid_create:
            await store.create_task(poisoned_create)
        create_error = extract_durable_value_error(invalid_create.value)
        assert create_error is not None
        assert create_error.code == "non_finite_number"
        assert await store.load_task("task_poisoned_create") is None

        request = TaskCreate(
            task_id="task_portable_numbers",
            type="demo",
            input={"numbers": {"safe": True}},
            metadata={"numbers": {"safe": True}},
        )
        numbers = {
            "ordinary": 1.0,
            "negative_zero": -0.0,
            "large": 1e18,
            "fractional": 1e-7,
        }
        request.input["numbers"] = dict(numbers)
        request.metadata["numbers"] = dict(numbers)
        await store.create_task(request)

        with pytest.raises(DurableValueError) as invalid_result:
            await store.complete_task(
                "task_portable_numbers",
                {"bad": MAX_DURABLE_JSON_INTEGER + 1},
            )
        assert invalid_result.value.code == "integer_out_of_range"
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        with pytest.raises(DurableValueError) as invalid_reason:
            await store.pause_task("task_portable_numbers", reason="poisoned\x00reason")
        assert invalid_reason.value.code == "nul_character"
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        for invalid_text, code in (
            ("workload-secret-value\x00", "nul_character"),
            ("workload-secret-value\ud800", "unicode_surrogate"),
        ):
            with pytest.raises(DurableValueError) as invalid_session_id:
                await store.start_task(
                    "task_portable_numbers",
                    session_id=invalid_text,
                )
            assert invalid_session_id.value.code == code
            assert "workload-secret-value" not in str(invalid_session_id.value)

            forged_query = TaskQuery(q="safe")
            forged_query.q = invalid_text
            with pytest.raises(ValidationError) as invalid_query:
                await store.list_tasks(forged_query)
            query_error = extract_durable_value_error(invalid_query.value)
            assert query_error is not None
            assert query_error.code == code
            assert "workload-secret-value" not in str(invalid_query.value)

            pending = await store.load_task("task_portable_numbers")
            assert pending is not None
            assert pending.status is TaskStatus.PENDING
            assert pending.session_id is None

        forged_query = TaskQuery()
        forged_query.offset = MAX_DURABLE_JSON_INTEGER + 1
        with pytest.raises(ValidationError):
            await store.list_tasks(forged_query)
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        await store.complete_task("task_portable_numbers", {"numbers": numbers})
        loaded = await store.load_task("task_portable_numbers")
        assert loaded is not None
        assert loaded.result is not None
        for value in (
            loaded.input["numbers"],
            loaded.metadata["numbers"],
            loaded.result["numbers"],
        ):
            assert value == {
                "ordinary": 1,
                "negative_zero": 0,
                "large": 1_000_000_000_000_000_000,
                "fractional": 1e-7,
            }
            assert type(value["ordinary"]) is int
            assert type(value["negative_zero"]) is int
            assert type(value["large"]) is int
            assert type(value["fractional"]) is float

    _run(postgres_dsn, ops)


def test_postgres_task_store_claim_heartbeat_and_release_task(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_a", type="review"))
        await store.create_task(TaskCreate(task_id="task_b", type="review"))
        await store.create_task(
            TaskCreate(task_id="task_session_linked", type="review", session_id="sess_linked")
        )

        first = await store.claim_task(
            "worker_a",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=300,
        )
        assert first is not None
        assert first.id == "task_a"
        assert first.status == TaskStatus.CLAIMED
        assert first.worker_id == "worker_a"
        assert first.lease_expires_at is not None
        assert first.started_at is None

        second = await store.claim_task(
            "worker_b",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=300,
        )
        assert second is not None
        assert second.id == "task_b"
        assert second.worker_id == "worker_b"

        assert await store.claim_task("worker_c", TaskQuery(type="review")) is None
        linked = await store.load_task("task_session_linked")
        assert linked is not None
        assert linked.status == TaskStatus.PENDING
        assert linked.worker_id is None

        heartbeat = await store.heartbeat("task_a", "worker_a", extend_seconds=600)
        assert heartbeat.lease_expires_at is not None
        assert heartbeat.lease_expires_at > first.lease_expires_at

        with pytest.raises(ValueError, match="does not own"):
            await store.heartbeat("task_a", "worker_b")

        released = await store.release_task("task_a", "worker_a")
        assert released.status == TaskStatus.PENDING
        assert released.worker_id is None
        assert released.lease_expires_at is None

        reclaimed = await store.claim_task("worker_c", TaskQuery(type="review"))
        assert reclaimed is not None
        assert reclaimed.id == "task_a"
        assert reclaimed.worker_id == "worker_c"

        completed = await store.complete_task("task_a", {"ok": True})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.worker_id is None
        assert completed.lease_expires_at is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_attach_task_starts_claimed_task(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_claimed", type="review"))

        with pytest.raises(ValueError, match="not claimed by worker worker_a"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_unclaimed",
                session_invocation=await task_backed_session_invocation(
                    store,
                    "task_claimed",
                    "sess_unclaimed",
                ),
                worker_id="worker_a",
            )
        unclaimed = await store.load_task("task_claimed")
        assert unclaimed is not None
        assert unclaimed.status == TaskStatus.PENDING
        assert unclaimed.session_id is None

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        assert claimed.status == TaskStatus.CLAIMED
        assert claimed.worker_id == "worker_a"
        assert claimed.session_id is None
        assert claimed.lease_expires_at is not None

        with pytest.raises(ValueError, match="session_id"):
            await store.attach_task(
                "task_claimed",
                session_id="",
                session_invocation=unattributed_session_invocation_binding("unused_session"),
                worker_id="worker_a",
            )
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_claimed", session_id="sess_wrong")
        with pytest.raises(ValueError, match="does not own"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_wrong",
                session_invocation=await task_backed_session_invocation(
                    store,
                    "task_claimed",
                    "sess_wrong",
                ),
                worker_id="worker_b",
            )

        started = await store.attach_task(
            "task_claimed",
            session_id="sess_claimed",
            session_invocation=await task_backed_session_invocation(
                store, "task_claimed", "sess_claimed"
            ),
            worker_id="worker_a",
        )
        assert started.status == TaskStatus.RUNNING
        assert started.session_id == "sess_claimed"
        assert started.worker_id == "worker_a"
        assert started.lease_expires_at == claimed.lease_expires_at

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_expired_claim_handoff(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_expired_handoff", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None

        await asyncio.sleep(1.05)
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_expired_handoff", session_id="sess_expired")
        with pytest.raises(TaskClaimLost, match="lease for worker worker_a has expired"):
            await store.heartbeat("task_expired_handoff", "worker_a")

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_release_after_session_attachment(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_release", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        await store.attach_task(
            "task_attached_release",
            session_id="sess_attached",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_release", "sess_attached"
            ),
            worker_id="worker_a",
        )

        with pytest.raises(ValueError, match="already attached to session sess_attached"):
            await store.release_task("task_attached_release", "worker_a")

        loaded = await store.load_task("task_attached_release")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.session_id == "sess_attached"
        assert loaded.worker_id == "worker_a"

    _run(postgres_dsn, ops)


def test_postgres_task_store_releases_attached_worker_without_requeueing(postgres_dsn):
    async def ops(store):
        await store.create_task(
            TaskCreate(
                task_id="task_attached_handoff",
                type="review",
                assigned_agent_name="reviewer",
                metadata={"tenant": "acme"},
            )
        )
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_attached_handoff",
            session_id="sess_attached_handoff",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_handoff", "sess_attached_handoff"
            ),
            worker_id="worker_a",
        )

        released = await store.release_attached_task_worker(
            "task_attached_handoff",
            "worker_a",
        )

        assert released.status == TaskStatus.RUNNING
        assert released.session_id == "sess_attached_handoff"
        assert released.worker_id is None
        assert released.lease_expires_at is None
        assert released.assigned_agent_name == "reviewer"
        assert released.metadata == {"tenant": "acme"}
        assert released.created_at == attached.created_at
        assert released.started_at == attached.started_at
        assert released.updated_at >= attached.updated_at
        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None
        assert await store.reclaim_expired(query=TaskQuery(type="review")) == []

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_invalid_attached_worker_release(postgres_dsn):
    async def ops(store):
        with pytest.raises(KeyError, match="Task not found"):
            await store.release_attached_task_worker("missing", "worker_a")

        await store.create_task(TaskCreate(task_id="task_unattached", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        unattached_before = await store.load_task("task_unattached")
        assert unattached_before is not None
        with pytest.raises(ValueError, match="not running"):
            await store.release_attached_task_worker("task_unattached", "worker_a")

        await store.create_task(TaskCreate(task_id="task_wrong_worker", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        await store.attach_task(
            "task_wrong_worker",
            session_id="sess_wrong_worker",
            session_invocation=await task_backed_session_invocation(
                store, "task_wrong_worker", "sess_wrong_worker"
            ),
            worker_id="worker_a",
        )
        wrong_worker_before = await store.load_task("task_wrong_worker")
        assert wrong_worker_before is not None
        with pytest.raises(ValueError, match="does not own"):
            await store.release_attached_task_worker("task_wrong_worker", "worker_b")

        await store.create_task(TaskCreate(task_id="task_expired_worker", type="review"))
        await store.claim_task("worker_a", lease_seconds=1)
        await store.attach_task(
            "task_expired_worker",
            session_id="sess_expired_worker",
            session_invocation=await task_backed_session_invocation(
                store, "task_expired_worker", "sess_expired_worker"
            ),
            worker_id="worker_a",
        )
        await asyncio.sleep(1.05)
        expired_before = await store.load_task("task_expired_worker")
        assert expired_before is not None
        with pytest.raises(TaskClaimLost, match="lease for worker worker_a has expired"):
            await store.release_attached_task_worker("task_expired_worker", "worker_a")

        await store.create_task(TaskCreate(task_id="task_terminal", type="review"))
        terminal_before = await store.complete_task(
            "task_terminal",
            {"winner": "terminal-state"},
        )
        with pytest.raises(ValueError, match="running"):
            await store.release_attached_task_worker("task_terminal", "worker_a")

        unattached_after = await store.load_task("task_unattached")
        wrong_worker_after = await store.load_task("task_wrong_worker")
        expired_after = await store.load_task("task_expired_worker")
        terminal_after = await store.load_task("task_terminal")
        assert unattached_after == unattached_before
        assert wrong_worker_after == wrong_worker_before
        assert expired_after == expired_before
        assert terminal_after is not None
        assert terminal_after == terminal_before

    _run(postgres_dsn, ops)


def test_postgres_task_store_does_not_reclaim_attached_expired_leases(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_expired", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None
        await store.attach_task(
            "task_attached_expired",
            session_id="sess_attached_expired",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_expired", "sess_attached_expired"
            ),
            worker_id="worker_a",
        )

        await asyncio.sleep(1.05)
        reclaimed = await store.reclaim_expired(query=TaskQuery(type="review"))
        assert reclaimed == []

        loaded = await store.load_task("task_attached_expired")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.session_id == "sess_attached_expired"
        assert loaded.worker_id == "worker_a"

    _run(postgres_dsn, ops)


def test_postgres_task_store_reclaim_expired_leases(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_expired", type="demo"))
        await store.create_task(TaskCreate(task_id="task_waiting", type="demo"))
        await store.claim_task(
            "worker_a",
            TaskQuery(type="demo", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=1,
        )

        await asyncio.sleep(1.05)
        reclaimed = await store.reclaim_expired(
            query=TaskQuery(type="demo"),
            max_reclaims=1,
        )
        assert [task.id for task in reclaimed] == ["task_expired"]
        assert reclaimed[0].status == TaskStatus.PENDING
        assert reclaimed[0].worker_id is None
        assert reclaimed[0].lease_expires_at is None

        loaded = await store.load_task("task_expired")
        assert loaded is not None
        assert loaded.status == TaskStatus.PENDING

        assert await store.reclaim_expired(query=TaskQuery(status=TaskStatus.PENDING)) == []

    _run(postgres_dsn, ops)


def test_postgres_task_store_validate_worker_lease_inputs(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_validate_worker", type="demo"))

        with pytest.raises(ValueError, match="lease_seconds must be >= 1"):
            await store.claim_task("worker_a", lease_seconds=0)
        with pytest.raises(TypeError, match="lease_seconds must be an integer"):
            await store.claim_task("worker_a", lease_seconds=True)  # type: ignore[arg-type]

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None

        with pytest.raises(ValueError, match="extend_seconds must be >= 1"):
            await store.heartbeat("task_validate_worker", "worker_a", extend_seconds=0)
        with pytest.raises(ValueError, match="max_reclaims must be >= 1"):
            await store.reclaim_expired(max_reclaims=0)
        with pytest.raises(ValueError, match="do not support session_id"):
            await store.claim_task("worker_b", TaskQuery(session_id="sess_1"))
        with pytest.raises(ValueError, match="do not support session_id"):
            await store.reclaim_expired(query=TaskQuery(session_id="sess_1"))
        with pytest.raises(ValueError, match="do not support limit"):
            await store.claim_task("worker_b", TaskQuery(limit=2))
        with pytest.raises(ValueError, match="do not support offset"):
            await store.reclaim_expired(query=TaskQuery(offset=1))

    _run(postgres_dsn, ops)


def test_postgres_task_store_concurrent_claims_do_not_duplicate_tasks(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_a", type="review"))
        await store.create_task(TaskCreate(task_id="task_b", type="review"))

        second = _new_store(postgres_dsn)
        try:
            claimed = await asyncio.gather(
                store.claim_task(
                    "worker_a",
                    TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
                ),
                second.claim_task(
                    "worker_b",
                    TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
                ),
            )
            claimed_ids = sorted(task.id for task in claimed if task is not None)
            worker_ids = sorted(task.worker_id for task in claimed if task is not None)

            assert claimed_ids == ["task_a", "task_b"]
            assert worker_ids == ["worker_a", "worker_b"]

            loaded_a = await store.load_task("task_a")
            loaded_b = await second.load_task("task_b")
            assert loaded_a is not None
            assert loaded_b is not None
            assert {loaded_a.worker_id, loaded_b.worker_id} == {"worker_a", "worker_b"}
            assert loaded_a.id != loaded_b.id
        finally:
            await second.close()

    _run(postgres_dsn, ops)


def test_postgres_task_store_cancel_and_persistence(postgres_dsn):
    async def ops(store):
        await store.create_task(
            TaskCreate(
                task_id="task_cancel",
                type="process_invoice",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.start_task(
            "task_cancel",
            session_id="sess_cancel",
            session_invocation=await task_backed_session_invocation(
                store, "task_cancel", "sess_cancel"
            ),
        )
        cancelled = await store.cancel_task("task_cancel", {"reason": "operator stop"})
        assert cancelled.status == TaskStatus.CANCELLED
        assert cancelled.error == {"reason": "operator stop"}
        assert cancelled.started_at is not None
        assert cancelled.completed_at is not None

        # Reload from a fresh store/pool to confirm durability.
        reopened = _new_store(postgres_dsn)
        try:
            loaded = await reopened.load_task("task_cancel")
            assert loaded is not None
            assert loaded.status == TaskStatus.CANCELLED
            assert loaded.session_id == "sess_cancel"
            assert loaded.error == {"reason": "operator stop"}
        finally:
            await reopened.close()

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_stale_cross_pool_transitions(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_claim", type="demo"))

        second = _new_store(postgres_dsn)
        try:
            await store.start_task(
                "task_claim",
                session_id="session_one",
                session_invocation=await task_backed_session_invocation(
                    store, "task_claim", "session_one"
                ),
            )
            with pytest.raises(ValueError, match="cannot transition to running"):
                await second.start_task("task_claim", session_id="session_two")

            completed = await second.complete_task("task_claim", {"ok": True})
            assert completed.status == TaskStatus.COMPLETED
            assert completed.session_id == "session_one"

            with pytest.raises(ValueError, match="already terminal"):
                await store.fail_task("task_claim", {"message": "too late"})
        finally:
            await second.close()

    _run(postgres_dsn, ops)
