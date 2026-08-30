from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.core.work_context_store_conformance import (
    _create_delivery_knowledge,
    _delivery_processor,
    _process_delivery,
    assert_work_context_store_conformance,
    checkpoint,
    context,
    recall_delivery,
)

from cayu import (
    AgentRecallCheckpoint,
    AgentRecallCheckpointKey,
    AgentRecallCheckpointMode,
    AgentRecallDelivery,
    AgentRecallDeliveryConflict,
    AgentRecallDeliveryEvidenceKind,
    AgentRecallDeliveryState,
    AgentRecallProcessingRequest,
    AgentRecallSubscription,
    AgentRecallSubscriptionConflict,
    AgentRecallSubscriptionEvaluationOutcome,
    AgentRecallSubscriptionStatus,
    AgentRecallSubscriptionWakeState,
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextPublicationReceipt,
    AutomaticRecallPolicy,
    InMemoryAgentWorkContextStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    SQLiteAgentWorkContextStore,
    SQLiteKnowledgeStore,
    agent_recall_facet_aspect,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations
from cayu.storage.migrations import SchemaMode


@dataclass(frozen=True)
class _StoreCase:
    name: str
    open: Any
    reset: Any
    reopenable: bool
    clock: _ManualClock


@dataclass
class _ManualClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


async def _drop_postgres_schema(postgres_dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
            )
            for (table,) in await cursor.fetchall():
                await cursor.execute(
                    sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(str(table)))
                )
            await cursor.execute(
                "DROP FUNCTION IF EXISTS cayu_test_block_agent_work_context_head_update() CASCADE"
            )
            await cursor.execute(
                "DROP FUNCTION IF EXISTS "
                "cayu_test_block_agent_recall_checkpoint_head_update() CASCADE"
            )
            await cursor.execute(
                "DROP FUNCTION IF EXISTS "
                "cayu_test_block_agent_recall_delivery_state_insert() CASCADE"
            )
            await cursor.execute(
                "DROP FUNCTION IF EXISTS "
                "cayu_test_block_agent_recall_subscription_state_update() CASCADE"
            )
        await connection.commit()


async def _wait_for_postgres_head_lock(
    connection: Any,
    *,
    lock_key: int,
    task: asyncio.Task[Any],
) -> None:
    lock_class_id = (lock_key >> 32) & 0xFFFF_FFFF
    lock_object_id = lock_key & 0xFFFF_FFFF
    for _ in range(1_000):
        if task.done():
            await task
            raise AssertionError(
                "Postgres write completed before reaching its blocked head update."
            )
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid::bigint = %s
                      AND objid::bigint = %s
                      AND objsubid = 1
                      AND granted IS FALSE
                )
                """,
                (lock_class_id, lock_object_id),
            )
            row = await cursor.fetchone()
        if row is not None and row[0] is True:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for the Postgres head update to block.")


async def _acquire_postgres_advisory_lock(connection: Any, lock_key: int) -> None:
    await connection.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
    await connection.commit()


async def _release_postgres_advisory_lock(connection: Any, lock_key: int) -> None:
    cursor = await connection.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    row = await cursor.fetchone()
    await connection.commit()
    assert row == (True,)


@pytest.fixture(params=("memory", "sqlite", "postgres"))
def work_context_store_case(request, tmp_path: Path) -> _StoreCase:
    now = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)
    clock = _ManualClock(now)
    if request.param == "memory":

        async def open_memory():
            return InMemoryAgentWorkContextStore(clock=clock)

        async def reset_memory() -> None:
            return None

        return _StoreCase("memory", open_memory, reset_memory, False, clock)
    if request.param == "sqlite":
        location = tmp_path / "work-context.sqlite"

        async def open_sqlite():
            return SQLiteAgentWorkContextStore(location, clock=clock)

        async def reset_sqlite() -> None:
            for path in (location, Path(f"{location}-shm"), Path(f"{location}-wal")):
                path.unlink(missing_ok=True)

        return _StoreCase("sqlite", open_sqlite, reset_sqlite, True, clock)

    postgres_dsn = request.getfixturevalue("postgres_dsn")

    async def open_postgres():
        from cayu import PostgresAgentWorkContextStore

        return PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            clock=clock,
        )

    async def reset_postgres() -> None:
        await _drop_postgres_schema(postgres_dsn)

    return _StoreCase("postgres", open_postgres, reset_postgres, True, clock)


async def _close(store) -> None:
    await store.close()


def _subscription_policy(result, *, threshold: float) -> AutomaticRecallPolicy:
    assert result.recall is not None
    return AutomaticRecallPolicy(
        calibration_version="subscription-test-v1",
        fusion_strategy_version=result.recall.fusion.strategy_version,
        fusion_configuration_version=result.recall.fusion.configuration_version,
        minimum_inject_score=threshold,
        minimum_offer_score=threshold,
    )


def test_idle_recall_subscription_commits_silent_or_wake_atomically(
    work_context_store_case,
) -> None:
    async def run() -> None:
        await work_context_store_case.reset()
        clock = work_context_store_case.clock
        store = await work_context_store_case.open()
        knowledge = InMemoryKnowledgeStore()
        await _create_delivery_knowledge(knowledge, "subscription-entry")
        processor = _delivery_processor(knowledge)

        wake_context = context(
            revision=1,
            operation_id="subscription:wake-context",
            task_id="task:subscription-wake",
        )
        await store.publish_work_context(wake_context, expected_revision=None)
        wake_policy_result = await _process_delivery(
            processor,
            wake_context,
            operation_id="subscription:wake-policy",
            checkpoint_value=None,
            agent_id="agent:delivery",
        )
        wake_subscription = AgentRecallSubscription.create(
            subscription_id="subscription:wake",
            agent_id="agent:delivery",
            work_context=wake_context,
            knowledge_namespace="project:delivery",
            access_policy_sha256=wake_policy_result.access_policy_sha256,
            query="checkpoint-aware delivery evidence",
            admission_policy=_subscription_policy(wake_policy_result, threshold=0.0),
            minimum_interval_seconds=60,
            expires_at=clock.value + timedelta(days=1),
            revision=1,
            operation_id="subscription:wake-publish",
            published_by="application:test",
            published_at=clock.value - timedelta(minutes=1),
        )
        wake_result = await _process_delivery(
            processor,
            wake_context,
            operation_id="subscription:wake-processing",
            checkpoint_value=None,
            agent_id="agent:delivery",
            checkpoint_stream_id=wake_subscription.checkpoint_stream_id(),
        )
        publication = await store.publish_recall_subscription(
            wake_subscription,
            expected_revision=None,
        )
        assert publication.subscription == wake_subscription
        assert await store.load_recall_subscription(wake_subscription.subscription_id) == (
            wake_subscription
        )
        claimed = await store.claim_due_recall_subscription(
            wake_subscription.checkpoint_key(),
            claim_id="subscription:wake-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert claimed is not None and claimed.claim is not None
        evaluation = await store.commit_recall_subscription_evaluation(
            claimed.claim,
            wake_result,
            evaluation_id="subscription:z-wake-evaluation",
            delivery_id="subscription:wake-delivery",
            staged_by="runtime:inline",
            evaluated_at=clock.value,
        )
        assert evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.WAKE
        assert evaluation.delivery_id == "subscription:wake-delivery"
        assert await store.load_recall_checkpoint(wake_subscription.checkpoint_key()) == (
            wake_result.proposed_checkpoint
        )
        subscription_owned_delivery = AgentRecallDelivery.from_processing_result(
            wake_result,
            delivery_id="subscription:wake-delivery",
            expected_checkpoint_revision=None,
            staged_by="runtime:inline",
            staged_at=clock.value,
        )
        with pytest.raises(AgentRecallDeliveryConflict, match="delivery_operation_reused"):
            await store.stage_recall_delivery(subscription_owned_delivery)
        assert (
            await store.claim_recall_delivery(
                wake_subscription.checkpoint_key(),
                claim_id="subscription:not-an-active-delivery",
                worker_id="runtime:inline",
                lease_seconds=300,
            )
            is None
        )
        if work_context_store_case.reopenable:
            await store.close()
            store = await work_context_store_case.open()
            assert await store.load_recall_subscription(wake_subscription.subscription_id) == (
                wake_subscription
            )
            assert await store.load_recall_subscription_evaluation(evaluation.evaluation_id) == (
                evaluation
            )
            reopened_wake = await store.load_recall_subscription_wake(evaluation.evaluation_id)
            assert reopened_wake is not None
            assert reopened_wake.state is AgentRecallSubscriptionWakeState.PENDING
            assert reopened_wake.delivery.delivery_id == evaluation.delivery_id
        await _create_delivery_knowledge(knowledge, "subscription-parallel-entry")
        parallel_subscription = AgentRecallSubscription.create(
            subscription_id="subscription:parallel-wake",
            agent_id="agent:delivery",
            work_context=wake_context,
            knowledge_namespace="project:delivery",
            access_policy_sha256=wake_result.access_policy_sha256,
            query="checkpoint-aware delivery evidence",
            admission_policy=_subscription_policy(wake_result, threshold=0.0),
            minimum_interval_seconds=60,
            expires_at=clock.value + timedelta(days=1),
            revision=1,
            operation_id="subscription:parallel-wake-publish",
            published_by="application:test",
            published_at=clock.value,
        )
        await store.publish_recall_subscription(
            parallel_subscription,
            expected_revision=None,
        )
        parallel_claimed = await store.claim_due_recall_subscription(
            parallel_subscription.checkpoint_key(),
            claim_id="subscription:parallel-wake-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert parallel_claimed is not None and parallel_claimed.claim is not None
        parallel_result = await _process_delivery(
            processor,
            wake_context,
            operation_id="subscription:parallel-wake-processing",
            checkpoint_value=None,
            agent_id="agent:delivery",
            checkpoint_stream_id=parallel_subscription.checkpoint_stream_id(),
        )
        parallel_evaluation = await store.commit_recall_subscription_evaluation(
            parallel_claimed.claim,
            parallel_result,
            evaluation_id="subscription:a-parallel-wake-evaluation",
            delivery_id="subscription:parallel-wake-delivery",
            staged_by="runtime:inline",
            evaluated_at=clock.value,
        )
        assert parallel_evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.WAKE
        parallel_wake = await store.claim_recall_subscription_wake(
            wake_subscription.checkpoint_key(),
            claim_id="subscription:parallel-wake-handoff",
            runner_id="runtime:parallel",
            lease_seconds=300,
        )
        assert parallel_wake is not None and parallel_wake.claim is not None
        assert parallel_wake.evaluation == parallel_evaluation
        wake = await store.claim_recall_subscription_wake(
            wake_subscription.checkpoint_key(),
            claim_id="subscription:wake-handoff",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert wake is not None and wake.claim is not None
        assert wake.evaluation == evaluation
        assert await store.load_recall_subscription_wake(wake.wake_id) == wake
        parallel_wake = await store.acknowledge_recall_subscription_wake(
            parallel_wake.claim,
            acknowledgement_id="subscription:parallel-wake-accepted",
            acknowledged_at=clock.value,
        )
        assert parallel_wake.acknowledgement is not None
        await store.publish_recall_subscription(
            parallel_subscription.model_copy(
                update={
                    "revision": 2,
                    "operation_id": "subscription:parallel-wake-cancel",
                    "status": AgentRecallSubscriptionStatus.CANCELLED,
                }
            ),
            expected_revision=1,
        )
        clock.advance(timedelta(seconds=61))
        assert (
            await store.claim_due_recall_subscription(
                wake_subscription.checkpoint_key(),
                claim_id="subscription:wake-coalesced",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            is None
        )
        original_wake_claim = wake.claim
        wake = await store.renew_recall_subscription_wake(
            original_wake_claim,
            lease_seconds=300,
        )
        assert wake.claim is not None
        assert wake.claim.state_revision == original_wake_claim.state_revision + 1
        assert (
            await store.renew_recall_subscription_wake(
                original_wake_claim,
                lease_seconds=300,
            )
            == wake
        )
        released_wake = await store.release_recall_subscription_wake(
            wake.claim,
            release_id="subscription:wake-release",
            reason="scheduler process stopped before acceptance",
            released_at=clock.value,
        )
        assert released_wake.release is not None
        assert (
            await store.release_recall_subscription_wake(
                wake.claim,
                release_id="subscription:wake-release",
                reason="scheduler process stopped before acceptance",
                released_at=clock.value,
            )
            == released_wake
        )
        wake = await store.claim_recall_subscription_wake(
            wake_subscription.checkpoint_key(),
            claim_id="subscription:wake-handoff-retry",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert wake is not None and wake.claim is not None
        expired_claim = wake.claim
        clock.advance(timedelta(seconds=301))
        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="expired_wake_claim",
        ):
            await store.renew_recall_subscription_wake(
                expired_claim,
                lease_seconds=300,
            )
        wake = await store.claim_recall_subscription_wake(
            wake_subscription.checkpoint_key(),
            claim_id="subscription:wake-handoff-takeover",
            runner_id="runtime:recovery",
            lease_seconds=300,
        )
        assert wake is not None and wake.claim is not None
        assert wake.claim.attempt == expired_claim.attempt + 1
        with pytest.raises(AgentRecallSubscriptionConflict, match="stale_wake_claim"):
            await store.release_recall_subscription_wake(
                expired_claim,
                release_id="subscription:stale-wake-release",
                reason="stale scheduler must not release a newer attempt",
                released_at=clock.value,
            )
        with pytest.raises(AgentRecallSubscriptionConflict, match="stale_wake_claim"):
            await store.acknowledge_recall_subscription_wake(
                expired_claim,
                acknowledgement_id="subscription:stale-wake-acknowledgement",
                acknowledged_at=clock.value,
            )
        wake = await store.acknowledge_recall_subscription_wake(
            wake.claim,
            acknowledgement_id="subscription:wake-accepted",
            acknowledged_at=clock.value,
        )
        assert wake.acknowledgement is not None
        assert (
            await store.acknowledge_recall_subscription_wake(
                wake.claim,
                acknowledgement_id="subscription:wake-accepted",
                acknowledged_at=clock.value,
            )
            == wake
        )
        staged = await store.load_recall_delivery(wake.delivery.delivery_id)
        assert staged is not None and staged.state is AgentRecallDeliveryState.PENDING
        delivery = await store.claim_recall_delivery(
            wake_subscription.checkpoint_key(),
            claim_id="subscription:delivery-handoff",
            worker_id="runtime:inline",
            lease_seconds=300,
        )
        assert delivery is not None and delivery.claim is not None
        await store.acknowledge_recall_delivery(
            delivery.claim,
            acknowledgement_id="subscription:delivery-accepted",
            evidence_kind=AgentRecallDeliveryEvidenceKind.RECALL_RECEIPT,
            evidence_ref="recall-receipt:subscription-wake",
            acknowledged_at=clock.value,
        )

        silent_context = context(
            revision=1,
            operation_id="subscription:silent-context",
            task_id="task:subscription-silent",
        )
        await store.publish_work_context(silent_context, expected_revision=None)
        silent_policy_result = await _process_delivery(
            processor,
            silent_context,
            operation_id="subscription:silent-policy",
            checkpoint_value=None,
            agent_id="agent:delivery",
        )
        silent_subscription = AgentRecallSubscription.create(
            subscription_id="subscription:silent",
            agent_id="agent:delivery",
            work_context=silent_context,
            knowledge_namespace="project:delivery",
            access_policy_sha256=silent_policy_result.access_policy_sha256,
            query="checkpoint-aware delivery evidence",
            admission_policy=_subscription_policy(silent_policy_result, threshold=1.0),
            minimum_interval_seconds=60,
            expires_at=clock.value + timedelta(days=1),
            revision=1,
            operation_id="subscription:silent-publish",
            published_by="application:test",
            published_at=clock.value - timedelta(minutes=1),
        )
        silent_result = await _process_delivery(
            processor,
            silent_context,
            operation_id="subscription:silent-processing",
            checkpoint_value=None,
            agent_id="agent:delivery",
            checkpoint_stream_id=silent_subscription.checkpoint_stream_id(),
        )
        await store.publish_recall_subscription(silent_subscription, expected_revision=None)
        silent_claimed = await store.claim_due_recall_subscription(
            silent_subscription.checkpoint_key(),
            claim_id="subscription:silent-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert silent_claimed is not None and silent_claimed.claim is not None
        silent_evaluation = await store.commit_recall_subscription_evaluation(
            silent_claimed.claim,
            silent_result,
            evaluation_id="subscription:silent-evaluation",
            delivery_id=None,
            staged_by="runtime:inline",
            evaluated_at=clock.value,
        )
        assert silent_evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.SILENT
        assert await store.load_recall_delivery("subscription:silent-delivery") is None
        assert await store.load_recall_checkpoint(silent_subscription.checkpoint_key()) == (
            silent_result.proposed_checkpoint
        )
        clock.advance(timedelta(seconds=61))
        no_work_claimed = await store.claim_due_recall_subscription(
            silent_subscription.checkpoint_key(),
            claim_id="subscription:no-work-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert no_work_claimed is not None and no_work_claimed.claim is not None
        no_work_result = await _process_delivery(
            processor,
            silent_context,
            operation_id="subscription:no-work-processing",
            checkpoint_value=silent_result.proposed_checkpoint,
            agent_id="agent:delivery",
            checkpoint_stream_id=silent_subscription.checkpoint_stream_id(),
        )
        no_work_evaluation = await store.commit_recall_subscription_evaluation(
            no_work_claimed.claim,
            no_work_result,
            evaluation_id="subscription:no-work-evaluation",
            delivery_id=None,
            staged_by="runtime:inline",
            evaluated_at=clock.value,
        )
        assert no_work_evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.NO_WORK
        assert await store.load_recall_checkpoint(silent_subscription.checkpoint_key()) == (
            silent_result.proposed_checkpoint
        )
        await store.close()
        await work_context_store_case.reset()

    asyncio.run(run())


def test_processing_operation_identity_is_fenced_across_atomic_entrypoints(
    work_context_store_case,
) -> None:
    async def run() -> None:
        await work_context_store_case.reset()
        clock = work_context_store_case.clock
        store = await work_context_store_case.open()
        knowledge = InMemoryKnowledgeStore()
        await _create_delivery_knowledge(knowledge, "operation-fencing-entry")
        processor = _delivery_processor(knowledge)
        work_context = context(
            revision=1,
            operation_id="operation-fencing:context",
            task_id="task:operation-fencing",
        )
        await store.publish_work_context(work_context, expected_revision=None)

        evaluation_policy_result = await _process_delivery(
            processor,
            work_context,
            operation_id="operation-fencing:evaluation-policy",
            checkpoint_value=None,
        )
        evaluation_subscription = AgentRecallSubscription.create(
            subscription_id="subscription:operation-fencing:evaluation-owned",
            agent_id="agent:delivery",
            work_context=work_context,
            knowledge_namespace="project:delivery",
            access_policy_sha256=evaluation_policy_result.access_policy_sha256,
            query="checkpoint-aware delivery evidence",
            admission_policy=_subscription_policy(evaluation_policy_result, threshold=1.0),
            minimum_interval_seconds=60,
            expires_at=clock.value + timedelta(days=1),
            revision=1,
            operation_id="operation-fencing:evaluation-subscription",
            published_by="application:test",
            published_at=clock.value - timedelta(minutes=1),
        )
        evaluation_result = await _process_delivery(
            processor,
            work_context,
            operation_id="operation-fencing:evaluation-owned",
            checkpoint_value=None,
            checkpoint_stream_id=evaluation_subscription.checkpoint_stream_id(),
        )
        await store.publish_recall_subscription(
            evaluation_subscription,
            expected_revision=None,
        )
        claimed = await store.claim_due_recall_subscription(
            evaluation_subscription.checkpoint_key(),
            claim_id="operation-fencing:evaluation-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert claimed is not None and claimed.claim is not None
        evaluation = await store.commit_recall_subscription_evaluation(
            claimed.claim,
            evaluation_result,
            evaluation_id="operation-fencing:evaluation",
            delivery_id=None,
            staged_by="runtime:inline",
            evaluated_at=clock.value,
        )
        assert evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.SILENT
        assert evaluation_result.proposed_checkpoint is not None

        with pytest.raises(AgentWorkContextConflict, match="checkpoint_operation_reused"):
            await store.advance_recall_checkpoint(
                evaluation_result.proposed_checkpoint,
                expected_revision=None,
            )
        evaluation_owned_delivery = AgentRecallDelivery.from_processing_result(
            evaluation_result,
            delivery_id="operation-fencing:evaluation-owned-delivery",
            expected_checkpoint_revision=None,
            staged_by="runtime:inline",
            staged_at=clock.value,
        )
        with pytest.raises(AgentRecallDeliveryConflict, match="delivery_operation_reused"):
            await store.stage_recall_delivery(evaluation_owned_delivery)

        clock.advance(timedelta(seconds=61))
        no_work_claimed = await store.claim_due_recall_subscription(
            evaluation_subscription.checkpoint_key(),
            claim_id="operation-fencing:no-work-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert no_work_claimed is not None and no_work_claimed.claim is not None
        no_work_result = await _process_delivery(
            processor,
            work_context,
            operation_id="operation-fencing:no-work-owned",
            checkpoint_value=evaluation_result.proposed_checkpoint,
            checkpoint_stream_id=evaluation_subscription.checkpoint_stream_id(),
        )
        no_work_evaluation = await store.commit_recall_subscription_evaluation(
            no_work_claimed.claim,
            no_work_result,
            evaluation_id="operation-fencing:no-work-evaluation",
            delivery_id=None,
            staged_by="runtime:inline",
            evaluated_at=clock.value,
        )
        assert no_work_evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.NO_WORK
        reserved_no_work_checkpoint = evaluation_result.proposed_checkpoint.model_copy(
            update={
                "revision": 2,
                "processing_mode": AgentRecallCheckpointMode.DELTA,
                "processing_id": no_work_result.processing_id,
                "operation_id": no_work_result.operation_id,
                "updated_at": clock.value,
            }
        )
        with pytest.raises(AgentWorkContextConflict, match="checkpoint_operation_reused"):
            await store.advance_recall_checkpoint(
                reserved_no_work_checkpoint,
                expected_revision=1,
            )

        checkpoint_policy_result = await _process_delivery(
            processor,
            work_context,
            operation_id="operation-fencing:checkpoint-policy",
            checkpoint_value=None,
        )
        checkpoint_subscription = AgentRecallSubscription.create(
            subscription_id="subscription:operation-fencing:checkpoint-owned",
            agent_id="agent:delivery",
            work_context=work_context,
            knowledge_namespace="project:delivery",
            access_policy_sha256=checkpoint_policy_result.access_policy_sha256,
            query="checkpoint-aware delivery evidence",
            admission_policy=_subscription_policy(checkpoint_policy_result, threshold=1.0),
            minimum_interval_seconds=60,
            expires_at=clock.value + timedelta(days=1),
            revision=1,
            operation_id="operation-fencing:checkpoint-subscription",
            published_by="application:test",
            published_at=clock.value - timedelta(minutes=1),
        )
        checkpoint_result = await _process_delivery(
            processor,
            work_context,
            operation_id="operation-fencing:checkpoint-owned",
            checkpoint_value=None,
            checkpoint_stream_id=checkpoint_subscription.checkpoint_stream_id(),
        )
        await store.publish_recall_subscription(
            checkpoint_subscription,
            expected_revision=None,
        )
        checkpoint_claimed = await store.claim_due_recall_subscription(
            checkpoint_subscription.checkpoint_key(),
            claim_id="operation-fencing:checkpoint-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert checkpoint_claimed is not None and checkpoint_claimed.claim is not None
        assert checkpoint_result.proposed_checkpoint is not None
        await store.advance_recall_checkpoint(
            checkpoint_result.proposed_checkpoint,
            expected_revision=None,
        )
        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="processing_operation_reused",
        ):
            await store.commit_recall_subscription_evaluation(
                checkpoint_claimed.claim,
                checkpoint_result,
                evaluation_id="operation-fencing:checkpoint-evaluation",
                delivery_id=None,
                staged_by="runtime:inline",
                evaluated_at=clock.value,
            )

        await store.close()
        await work_context_store_case.reset()

    asyncio.run(run())


def test_recall_subscription_revision_lifecycle_facets_and_input_authority(
    work_context_store_case,
) -> None:
    async def run() -> None:
        await work_context_store_case.reset()
        clock = work_context_store_case.clock
        store = await work_context_store_case.open()
        knowledge = InMemoryKnowledgeStore()
        await _create_delivery_knowledge(knowledge, "subscription-authority-entry")
        processor = _delivery_processor(knowledge)
        work = context(
            revision=1,
            operation_id="subscription:authority-context",
            task_id="task:subscription-authority",
        )
        await store.publish_work_context(work, expected_revision=None)
        base_result = await _process_delivery(
            processor,
            work,
            operation_id="subscription:authority-base-processing",
            checkpoint_value=None,
            agent_id="agent:delivery",
        )
        policy = _subscription_policy(base_result, threshold=1.0)

        def subscription(
            subscription_id: str,
            operation_id: str,
            *,
            revision: int = 1,
            priority: int = 0,
            status: AgentRecallSubscriptionStatus = AgentRecallSubscriptionStatus.ACTIVE,
            bound_context: AgentWorkContext = work,
            query: str = "checkpoint-aware delivery evidence",
            scope_ids: tuple[str, ...] = ("repository:cayu",),
            expires_at: datetime | None = None,
        ) -> AgentRecallSubscription:
            return AgentRecallSubscription.create(
                subscription_id=subscription_id,
                agent_id="agent:delivery",
                work_context=bound_context,
                knowledge_namespace="project:delivery",
                access_policy_sha256=base_result.access_policy_sha256,
                admission_policy=policy,
                minimum_interval_seconds=60,
                expires_at=expires_at or (clock.value + timedelta(days=1)),
                revision=revision,
                operation_id=operation_id,
                published_by="application:test",
                published_at=clock.value,
                query=query,
                scope_ids=scope_ids,
                priority=priority,
                status=status,
            )

        with pytest.raises(
            ValueError,
            match="query without exact facets.*lexical search token",
        ):
            subscription(
                "subscription:tokenless-query",
                "subscription:tokenless-query-publish",
                query="!!!",
                scope_ids=(),
            )
        assert (
            subscription(
                "subscription:tokenless-faceted-query",
                "subscription:tokenless-faceted-query-publish",
                query="!!!",
            ).query
            == "!!!"
        )

        outside = subscription(
            "subscription:outside-facet",
            "subscription:outside-facet-publish",
            scope_ids=("repository:other",),
        )
        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="subscription_facet_outside_work_context",
        ):
            await store.publish_recall_subscription(outside, expected_revision=None)

        low = subscription(
            "subscription:a-low",
            "subscription:low-publish",
            priority=1,
        )
        high = subscription(
            "subscription:z-high",
            "subscription:high-publish",
            priority=100,
        )
        assert high.checkpoint_stream_id() != low.checkpoint_stream_id()
        assert (
            high.checkpoint_stream_id()
            == subscription(
                high.subscription_id,
                "subscription:high-schedule-only",
                revision=2,
                priority=high.priority,
                status=AgentRecallSubscriptionStatus.PAUSED,
            ).checkpoint_stream_id()
        )
        assert (
            high.checkpoint_stream_id()
            != subscription(
                high.subscription_id,
                "subscription:high-query-change",
                revision=2,
                priority=high.priority,
                query="different retrieval definition",
            ).checkpoint_stream_id()
        )
        assert (
            high.checkpoint_stream_id()
            != subscription(
                high.subscription_id,
                "subscription:high-facet-change",
                revision=2,
                priority=high.priority,
                scope_ids=(),
            ).checkpoint_stream_id()
        )
        low_receipt = await store.publish_recall_subscription(low, expected_revision=None)
        await store.publish_recall_subscription(high, expected_revision=None)
        assert await store.publish_recall_subscription(low, expected_revision=None) == low_receipt
        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="publication_operation_reused",
        ):
            await store.publish_recall_subscription(
                low.model_copy(update={"priority": 2}),
                expected_revision=None,
            )
        assert (
            await store.claim_due_recall_subscription(
                high.checkpoint_key().model_copy(update={"access_policy_sha256": "f" * 64}),
                claim_id="subscription:wrong-access-claim",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            is None
        )

        claimed = await store.claim_due_recall_subscription(
            high.checkpoint_key(),
            claim_id="subscription:priority-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert claimed is not None and claimed.claim is not None
        assert claimed.subscription.subscription_id == high.subscription_id

        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="evaluation_authority_mismatch",
        ):
            await store.commit_recall_subscription_evaluation(
                claimed.claim,
                base_result,
                evaluation_id="subscription:mismatched-input-evaluation",
                delivery_id=None,
                staged_by="runtime:inline",
                evaluated_at=clock.value,
            )
        scope = KnowledgeAccessScope.for_namespace("project:delivery")
        exact_result = await processor.process(
            AgentRecallProcessingRequest(
                agent_id="agent:delivery",
                work_context=work,
                situation=high.recall_situation(scope, current_time=clock.value),
                checkpoint_stream_id=high.checkpoint_stream_id(),
                checkpoint=None,
                processing_id="subscription:exact-input-processing",
                operation_id="subscription:exact-input-operation",
                updated_by="runtime:inline",
                updated_at=clock.value,
            )
        )
        evaluation = await store.commit_recall_subscription_evaluation(
            claimed.claim,
            exact_result,
            evaluation_id="subscription:exact-input-evaluation",
            delivery_id=None,
            staged_by="runtime:inline",
            evaluated_at=clock.value,
        )
        assert evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.SILENT

        paused_low = subscription(
            low.subscription_id,
            "subscription:low-pause",
            revision=2,
            priority=low.priority,
            status=AgentRecallSubscriptionStatus.PAUSED,
        )
        await store.publish_recall_subscription(paused_low, expected_revision=1)
        clock.advance(timedelta(seconds=61))
        claimed = await store.claim_due_recall_subscription(
            high.checkpoint_key(),
            claim_id="subscription:lifecycle-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert claimed is not None and claimed.claim is not None
        renewed = await store.renew_recall_subscription(
            claimed.claim,
            lease_seconds=300,
        )
        assert renewed.claim is not None
        assert await store.renew_recall_subscription(claimed.claim, lease_seconds=300) == renewed
        released = await store.release_recall_subscription(
            renewed.claim,
            release_id="subscription:lifecycle-release",
            reason="runner stopped before evaluation",
            released_at=clock.value,
        )
        assert released.release is not None
        assert (
            await store.release_recall_subscription(
                renewed.claim,
                release_id="subscription:lifecycle-release",
                reason="runner stopped before evaluation",
                released_at=clock.value,
            )
            == released
        )
        claimed_again = await store.claim_due_recall_subscription(
            high.checkpoint_key(),
            claim_id="subscription:lifecycle-reclaim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert claimed_again is not None and claimed_again.claim is not None
        stale_revision_result = await processor.process(
            AgentRecallProcessingRequest(
                agent_id="agent:delivery",
                work_context=work,
                situation=high.recall_situation(scope, current_time=clock.value),
                checkpoint_stream_id=high.checkpoint_stream_id(),
                checkpoint=exact_result.proposed_checkpoint,
                processing_id="subscription:stale-revision-processing",
                operation_id="subscription:stale-revision-operation",
                updated_by="runtime:inline",
                updated_at=clock.value,
            )
        )
        paused_high = subscription(
            high.subscription_id,
            "subscription:high-pause",
            revision=2,
            priority=high.priority,
            status=AgentRecallSubscriptionStatus.PAUSED,
        )
        await store.publish_recall_subscription(paused_high, expected_revision=1)
        with pytest.raises(AgentRecallSubscriptionConflict, match="stale_subscription_claim"):
            await store.commit_recall_subscription_evaluation(
                claimed_again.claim,
                stale_revision_result,
                evaluation_id="subscription:stale-revision-evaluation",
                delivery_id=None,
                staged_by="runtime:inline",
                evaluated_at=clock.value,
            )
        assert (
            await store.claim_due_recall_subscription(
                high.checkpoint_key(),
                claim_id="subscription:all-paused",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            is None
        )

        resumed_high = subscription(
            high.subscription_id,
            "subscription:high-resume",
            revision=3,
            priority=high.priority,
        )
        await store.publish_recall_subscription(resumed_high, expected_revision=2)
        stale_context_claim = await store.claim_due_recall_subscription(
            high.checkpoint_key(),
            claim_id="subscription:stale-context-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert stale_context_claim is not None and stale_context_claim.claim is not None
        no_work_result = await processor.process(
            AgentRecallProcessingRequest(
                agent_id="agent:delivery",
                work_context=work,
                situation=resumed_high.recall_situation(scope, current_time=clock.value),
                checkpoint_stream_id=resumed_high.checkpoint_stream_id(),
                checkpoint=exact_result.proposed_checkpoint,
                processing_id="subscription:stale-context-processing",
                operation_id="subscription:stale-context-operation",
                updated_by="runtime:inline",
                updated_at=clock.value,
            )
        )
        replacement_context = context(
            revision=2,
            operation_id="subscription:replacement-context",
            task_id=work.task_id,
            goal="Ship durable cross-agent freshness after requirements changed",
        )
        await store.publish_work_context(replacement_context, expected_revision=1)
        with pytest.raises(AgentRecallSubscriptionConflict, match="stale_work_context"):
            await store.commit_recall_subscription_evaluation(
                stale_context_claim.claim,
                no_work_result,
                evaluation_id="subscription:stale-context-evaluation",
                delivery_id=None,
                staged_by="runtime:inline",
                evaluated_at=clock.value,
            )
        assert (
            await store.claim_due_recall_subscription(
                high.checkpoint_key(),
                claim_id="subscription:stale-context-not-due",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            is None
        )

        rebound = subscription(
            high.subscription_id,
            "subscription:high-rebind",
            revision=4,
            priority=high.priority,
            bound_context=replacement_context,
        )
        await store.publish_recall_subscription(rebound, expected_revision=3)
        rebound_claim = await store.claim_due_recall_subscription(
            rebound.checkpoint_key(),
            claim_id="subscription:rebound-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert rebound_claim is not None and rebound_claim.claim is not None
        cancelled = subscription(
            high.subscription_id,
            "subscription:high-cancel",
            revision=5,
            priority=high.priority,
            status=AgentRecallSubscriptionStatus.CANCELLED,
            bound_context=replacement_context,
        )
        await store.publish_recall_subscription(cancelled, expected_revision=4)
        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="cancelled_subscription_is_terminal",
        ):
            await store.publish_recall_subscription(
                subscription(
                    high.subscription_id,
                    "subscription:invalid-cancelled-resume",
                    revision=6,
                    priority=high.priority,
                    bound_context=replacement_context,
                ),
                expected_revision=5,
            )
        assert (
            await store.claim_due_recall_subscription(
                cancelled.checkpoint_key(),
                claim_id="subscription:cancelled-not-due",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            is None
        )

        expiring = subscription(
            "subscription:expired",
            "subscription:expired-publish",
            bound_context=replacement_context,
            expires_at=clock.value + timedelta(seconds=1),
        )
        await store.publish_recall_subscription(expiring, expected_revision=None)
        expiring_claim = await store.claim_due_recall_subscription(
            expiring.checkpoint_key(),
            claim_id="subscription:expiring-claim",
            runner_id="runtime:inline",
            lease_seconds=300,
        )
        assert expiring_claim is not None and expiring_claim.claim is not None
        expiring_result = await processor.process(
            AgentRecallProcessingRequest(
                agent_id="agent:delivery",
                work_context=replacement_context,
                situation=expiring.recall_situation(scope, current_time=clock.value),
                checkpoint_stream_id=expiring.checkpoint_stream_id(),
                checkpoint=None,
                processing_id="subscription:expiring-processing",
                operation_id="subscription:expiring-operation",
                updated_by="runtime:inline",
                updated_at=clock.value,
            )
        )
        clock.advance(timedelta(seconds=2))
        with pytest.raises(AgentRecallSubscriptionConflict, match="expired_subscription"):
            await store.commit_recall_subscription_evaluation(
                expiring_claim.claim,
                expiring_result,
                evaluation_id="subscription:expired-evaluation",
                delivery_id=None,
                staged_by="runtime:inline",
                evaluated_at=clock.value - timedelta(seconds=2),
            )
        assert (
            await store.claim_due_recall_subscription(
                expiring.checkpoint_key(),
                claim_id="subscription:expired-not-due",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            is None
        )
        await store.close()
        await work_context_store_case.reset()

    asyncio.run(run())


def test_recall_subscription_concurrent_claim_and_lease_takeover(
    work_context_store_case,
) -> None:
    async def run() -> None:
        await work_context_store_case.reset()
        clock = work_context_store_case.clock
        store = await work_context_store_case.open()
        knowledge = InMemoryKnowledgeStore()
        await _create_delivery_knowledge(knowledge, "subscription-claim-entry")
        processor = _delivery_processor(knowledge)
        work = context(
            revision=1,
            operation_id="subscription:claim-context",
            task_id="task:subscription-claim",
        )
        await store.publish_work_context(work, expected_revision=None)
        result = await _process_delivery(
            processor,
            work,
            operation_id="subscription:claim-policy-processing",
            checkpoint_value=None,
            agent_id="agent:delivery",
        )
        subscription = AgentRecallSubscription.create(
            subscription_id="subscription:claim",
            agent_id="agent:delivery",
            work_context=work,
            knowledge_namespace="project:delivery",
            access_policy_sha256=result.access_policy_sha256,
            admission_policy=_subscription_policy(result, threshold=1.0),
            minimum_interval_seconds=60,
            expires_at=clock.value + timedelta(days=1),
            revision=1,
            operation_id="subscription:claim-publish",
            published_by="application:test",
            published_at=clock.value,
            query="checkpoint-aware delivery evidence",
        )
        await store.publish_recall_subscription(subscription, expected_revision=None)

        first, second = await asyncio.gather(
            store.claim_due_recall_subscription(
                subscription.checkpoint_key(),
                claim_id="subscription:concurrent-a",
                runner_id="runtime:a",
                lease_seconds=10,
            ),
            store.claim_due_recall_subscription(
                subscription.checkpoint_key(),
                claim_id="subscription:concurrent-b",
                runner_id="runtime:b",
                lease_seconds=10,
            ),
        )
        claimed = [record for record in (first, second) if record is not None]
        assert len(claimed) == 1
        original = claimed[0]
        assert original.claim is not None

        clock.advance(timedelta(seconds=11))
        takeover = await store.claim_due_recall_subscription(
            subscription.checkpoint_key(),
            claim_id="subscription:takeover",
            runner_id="runtime:takeover",
            lease_seconds=30,
        )
        assert takeover is not None and takeover.claim is not None
        assert takeover.attempt == original.attempt + 1
        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="stale_subscription_claim",
        ):
            await store.renew_recall_subscription(
                original.claim,
                lease_seconds=30,
            )
        with pytest.raises(
            AgentRecallSubscriptionConflict,
            match="stale_subscription_claim",
        ):
            await store.release_recall_subscription(
                original.claim,
                release_id="subscription:stale-release",
                reason="stale runner cannot return a replacement claim",
                released_at=clock.value,
            )
        await store.release_recall_subscription(
            takeover.claim,
            release_id="subscription:takeover-release",
            reason="takeover runner returned work",
            released_at=clock.value,
        )
        await store.close()
        await work_context_store_case.reset()

    asyncio.run(run())


def test_agent_work_context_store_shared_conformance(work_context_store_case) -> None:
    async def run() -> None:
        await work_context_store_case.reset()
        store = await work_context_store_case.open()
        try:
            await assert_work_context_store_conformance(
                store,
                advance_clock=work_context_store_case.clock.advance,
            )
        finally:
            await _close(store)
        if work_context_store_case.reopenable:
            reopened = await work_context_store_case.open()
            try:
                current = await reopened.load_work_context("task-memory-v51")
                assert current is not None
                assert current.revision == 4
                persisted_checkpoint = await reopened.load_recall_checkpoint(
                    AgentRecallCheckpointKey(
                        agent_id="agent:primary",
                        task_id="task-memory-v51",
                        knowledge_namespace="project:cayu",
                        access_policy_sha256="a" * 64,
                    )
                )
                assert persisted_checkpoint is not None
                assert persisted_checkpoint.revision == 5
                persisted_delivery = await reopened.load_recall_delivery("delivery:3")
                assert persisted_delivery is not None
                assert persisted_delivery.state is AgentRecallDeliveryState.ACKNOWLEDGED
                assert persisted_delivery.delivery.materialized_result().operation_id == (
                    "delivery:process:3"
                )
                claimed_delivery = await reopened.load_recall_delivery("delivery:4")
                released_delivery = await reopened.load_recall_delivery("delivery:5")
                pending_delivery = await reopened.load_recall_delivery("delivery:6")
                assert pending_delivery is not None
                assert pending_delivery.state is AgentRecallDeliveryState.PENDING
                assert pending_delivery.claim is None
                assert claimed_delivery is not None
                assert claimed_delivery.state is AgentRecallDeliveryState.CLAIMED
                assert claimed_delivery.claim is not None
                assert released_delivery is not None
                assert released_delivery.state is AgentRecallDeliveryState.PENDING
                assert released_delivery.release is not None
                assert released_delivery.release.reason == "durable retry evidence"
            finally:
                await _close(reopened)
        await work_context_store_case.reset()

    asyncio.run(run())


def test_agent_work_context_durable_multi_instance_cas(work_context_store_case) -> None:
    if not work_context_store_case.reopenable:
        pytest.skip("The in-memory store has no shared external durability boundary.")

    async def run() -> None:
        await work_context_store_case.reset()
        first_store = await work_context_store_case.open()
        second_store = await work_context_store_case.open()
        try:
            initial = context(revision=1, operation_id="multi-instance:context:create")
            await first_store.publish_work_context(initial, expected_revision=None)
            candidates = (
                context(
                    revision=2,
                    operation_id="multi-instance:context:a",
                    goal="Multi-instance writer A",
                ),
                context(
                    revision=2,
                    operation_id="multi-instance:context:b",
                    goal="Multi-instance writer B",
                ),
            )
            outcomes = await asyncio.gather(
                first_store.publish_work_context(candidates[0], expected_revision=1),
                second_store.publish_work_context(candidates[1], expected_revision=1),
                return_exceptions=True,
            )
            successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            assert len(successes) == 1
            assert len(failures) == 1
            assert isinstance(failures[0], AgentWorkContextConflict)
            current = await first_store.load_work_context(initial.task_id)
            assert current is not None
            assert current in candidates
            assert await second_store.load_work_context(initial.task_id) == current

            initial_checkpoint = checkpoint(
                current,
                revision=1,
                operation_id="multi-instance:checkpoint:create",
            )
            await first_store.advance_recall_checkpoint(
                initial_checkpoint,
                expected_revision=None,
            )
            checkpoint_candidates = (
                checkpoint(
                    current,
                    revision=2,
                    operation_id="multi-instance:checkpoint:a",
                    knowledge_sequence=11,
                    index_readiness_sequence=8,
                    processing_mode=AgentRecallCheckpointMode.DELTA,
                ),
                checkpoint(
                    current,
                    revision=2,
                    operation_id="multi-instance:checkpoint:b",
                    knowledge_sequence=12,
                    index_readiness_sequence=9,
                    processing_mode=AgentRecallCheckpointMode.DELTA,
                ),
            )
            checkpoint_outcomes = await asyncio.gather(
                first_store.advance_recall_checkpoint(
                    checkpoint_candidates[0],
                    expected_revision=1,
                ),
                second_store.advance_recall_checkpoint(
                    checkpoint_candidates[1],
                    expected_revision=1,
                ),
                return_exceptions=True,
            )
            checkpoint_successes = [
                outcome for outcome in checkpoint_outcomes if not isinstance(outcome, BaseException)
            ]
            checkpoint_failures = [
                outcome for outcome in checkpoint_outcomes if isinstance(outcome, BaseException)
            ]
            assert len(checkpoint_successes) == 1
            assert len(checkpoint_failures) == 1
            assert isinstance(checkpoint_failures[0], AgentWorkContextConflict)
            stored_checkpoint = await first_store.load_recall_checkpoint(initial_checkpoint.key())
            assert stored_checkpoint in checkpoint_candidates
            assert (
                await second_store.load_recall_checkpoint(initial_checkpoint.key())
                == stored_checkpoint
            )

            delivery_candidates = (
                await recall_delivery(
                    current,
                    delivery_id="multi-instance:delivery:a",
                    operation_id="multi-instance:delivery:process:a",
                    entry_ids=("multi-instance-entry",),
                ),
                await recall_delivery(
                    current,
                    delivery_id="multi-instance:delivery:b",
                    operation_id="multi-instance:delivery:process:b",
                    entry_ids=("multi-instance-entry",),
                ),
            )
            delivery_outcomes = await asyncio.gather(
                first_store.stage_recall_delivery(delivery_candidates[0]),
                second_store.stage_recall_delivery(delivery_candidates[1]),
                return_exceptions=True,
            )
            delivery_successes = [
                outcome for outcome in delivery_outcomes if not isinstance(outcome, BaseException)
            ]
            delivery_failures = [
                outcome for outcome in delivery_outcomes if isinstance(outcome, BaseException)
            ]
            assert len(delivery_successes) == 1
            assert len(delivery_failures) == 1
            assert isinstance(delivery_failures[0], AgentRecallDeliveryConflict)
            staged_delivery = delivery_successes[0]
            assert staged_delivery.delivery in delivery_candidates
            claim_outcomes = await asyncio.gather(
                first_store.claim_recall_delivery(
                    staged_delivery.delivery.key(),
                    claim_id="multi-instance:claim:a",
                    worker_id="multi-instance:worker:a",
                    lease_seconds=30,
                ),
                second_store.claim_recall_delivery(
                    staged_delivery.delivery.key(),
                    claim_id="multi-instance:claim:b",
                    worker_id="multi-instance:worker:b",
                    lease_seconds=30,
                ),
            )
            claimed = [record for record in claim_outcomes if record is not None]
            assert len(claimed) == 1
            assert claimed[0].state is AgentRecallDeliveryState.CLAIMED
        finally:
            await _close(first_store)
            await _close(second_store)
            await work_context_store_case.reset()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failing_table",
    ("cayu_agent_recall_deliveries", "cayu_agent_recall_delivery_states"),
    ids=("delivery-insert", "state-insert"),
)
def test_sqlite_recall_delivery_stage_rolls_back_every_material_boundary(
    tmp_path: Path,
    failing_table: str,
) -> None:
    async def run() -> None:
        database = tmp_path / "delivery-stage-rollback.sqlite"
        store = SQLiteAgentWorkContextStore(database)
        published = context(revision=1, operation_id="delivery-rollback:sqlite:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-rollback:sqlite",
            operation_id="delivery-rollback:sqlite:process",
            entry_ids=("delivery-rollback-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                f"""
                CREATE TRIGGER cayu_test_fail_agent_recall_delivery_insert
                BEFORE INSERT ON {failing_table}
                BEGIN
                    SELECT RAISE(ABORT, 'test delivery insert failure');
                END
                """
            )
            with pytest.raises(sqlite3.IntegrityError, match="test delivery insert failure"):
                await store.stage_recall_delivery(delivery)
            assert await store.load_recall_checkpoint(delivery.key()) is None
            assert await store.load_recall_delivery(delivery.delivery_id) is None
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                "DROP TRIGGER cayu_test_fail_agent_recall_delivery_insert"
            )
            staged = await store.stage_recall_delivery(delivery)
            assert staged.delivery == delivery
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failing_table",
    (
        "cayu_agent_recall_deliveries",
        "cayu_agent_recall_delivery_states",
        "cayu_agent_recall_subscription_evaluations",
        "cayu_agent_recall_subscription_wake_states",
        "cayu_agent_recall_subscription_states",
    ),
    ids=(
        "delivery",
        "delivery-state",
        "evaluation",
        "wake-state",
        "subscription-state",
    ),
)
def test_sqlite_recall_subscription_wake_rolls_back_every_material_boundary(
    tmp_path: Path,
    failing_table: str,
) -> None:
    async def run() -> None:
        clock = _ManualClock(datetime(2026, 8, 28, 9, 0, tzinfo=UTC))
        store = SQLiteAgentWorkContextStore(
            tmp_path / f"subscription-rollback-{failing_table}.sqlite",
            clock=clock,
        )
        knowledge = InMemoryKnowledgeStore()
        await _create_delivery_knowledge(knowledge, "subscription-rollback-entry")
        processor = _delivery_processor(knowledge)
        work = context(
            revision=1,
            operation_id="subscription:rollback-context",
            task_id="task:subscription-rollback",
        )
        try:
            await store.publish_work_context(work, expected_revision=None)
            policy_result = await _process_delivery(
                processor,
                work,
                operation_id="subscription:rollback-policy",
                checkpoint_value=None,
                agent_id="agent:delivery",
            )
            subscription = AgentRecallSubscription.create(
                subscription_id="subscription:rollback",
                agent_id="agent:delivery",
                work_context=work,
                knowledge_namespace="project:delivery",
                access_policy_sha256=policy_result.access_policy_sha256,
                admission_policy=_subscription_policy(policy_result, threshold=0.0),
                minimum_interval_seconds=60,
                expires_at=clock.value + timedelta(days=1),
                revision=1,
                operation_id="subscription:rollback-publish",
                published_by="application:test",
                published_at=clock.value,
                query="checkpoint-aware delivery evidence",
            )
            result = await _process_delivery(
                processor,
                work,
                operation_id="subscription:rollback-processing",
                checkpoint_value=None,
                agent_id="agent:delivery",
                checkpoint_stream_id=subscription.checkpoint_stream_id(),
            )
            await store.publish_recall_subscription(subscription, expected_revision=None)
            claimed = await store.claim_due_recall_subscription(
                subscription.checkpoint_key(),
                claim_id="subscription:rollback-claim",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            assert claimed is not None and claimed.claim is not None
            action = (
                "UPDATE" if failing_table == "cayu_agent_recall_subscription_states" else "INSERT"
            )
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                f"""
                CREATE TRIGGER cayu_test_fail_agent_recall_subscription_commit
                BEFORE {action} ON {failing_table}
                BEGIN
                    SELECT RAISE(ABORT, 'test subscription commit failure');
                END
                """
            )
            with pytest.raises(
                sqlite3.IntegrityError,
                match="test subscription commit failure",
            ):
                await store.commit_recall_subscription_evaluation(
                    claimed.claim,
                    result,
                    evaluation_id="subscription:rollback-evaluation",
                    delivery_id="subscription:rollback-delivery",
                    staged_by="runtime:inline",
                    evaluated_at=clock.value,
                )
            assert await store.load_recall_checkpoint(subscription.checkpoint_key()) is None
            assert await store.load_recall_delivery("subscription:rollback-delivery") is None
            assert (
                await store.load_recall_subscription_evaluation("subscription:rollback-evaluation")
                is None
            )
            replayed_claim = await store.claim_due_recall_subscription(
                subscription.checkpoint_key(),
                claim_id="subscription:rollback-claim",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            assert replayed_claim == claimed
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                "DROP TRIGGER cayu_test_fail_agent_recall_subscription_commit"
            )
            committed = await store.commit_recall_subscription_evaluation(
                claimed.claim,
                result,
                evaluation_id="subscription:rollback-evaluation",
                delivery_id="subscription:rollback-delivery",
                staged_by="runtime:inline",
                evaluated_at=clock.value,
            )
            assert committed.outcome is AgentRecallSubscriptionEvaluationOutcome.WAKE
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failing_table",
    ("cayu_agent_recall_deliveries", "cayu_agent_recall_delivery_states"),
    ids=("delivery-insert", "state-insert"),
)
def test_postgres_recall_delivery_stage_cancellation_rolls_back_every_material_boundary(
    postgres_dsn: str,
    failing_table: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        lock_key = 7_505_119_600_004
        await _drop_postgres_schema(postgres_dsn)
        store = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=1,
            schema_mode=SchemaMode.CREATE,
        )
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        pending: asyncio.Task[Any] | None = None
        held_lock = False
        published = context(revision=1, operation_id="delivery-rollback:postgres:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-rollback:postgres",
            operation_id="delivery-rollback:postgres:process",
            entry_ids=("delivery-rollback-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            async with blocker.cursor() as cursor:
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_recall_delivery_state_insert()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    f"""
                    CREATE TRIGGER cayu_test_block_agent_recall_delivery_state_insert
                    BEFORE INSERT ON {failing_table}
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_recall_delivery_state_insert()
                    """
                )
            await blocker.commit()
            await _acquire_postgres_advisory_lock(blocker, lock_key)
            held_lock = True
            pending = asyncio.create_task(store.stage_recall_delivery(delivery))
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=lock_key,
                task=pending,
            )
            pending.cancel("cancel staged recall before state publication")
            with pytest.raises(asyncio.CancelledError):
                await pending
            pending = None
            assert await store.load_recall_checkpoint(delivery.key()) is None
            assert await store.load_recall_delivery(delivery.delivery_id) is None
            await _release_postgres_advisory_lock(blocker, lock_key)
            held_lock = False
            staged = await store.stage_recall_delivery(delivery)
            assert staged.delivery == delivery
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
            if held_lock:
                await _release_postgres_advisory_lock(blocker, lock_key)
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await store.close()
            await blocker.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_recall_subscription_cancellation_rolls_back_atomic_wake(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        lock_key = 7_505_119_600_072
        clock = _ManualClock(datetime(2026, 8, 28, 9, 0, tzinfo=UTC))
        await _drop_postgres_schema(postgres_dsn)
        store = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=1,
            schema_mode=SchemaMode.CREATE,
            clock=clock,
        )
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        pending: asyncio.Task[Any] | None = None
        held_lock = False
        knowledge = InMemoryKnowledgeStore()
        await _create_delivery_knowledge(knowledge, "subscription-cancel-entry")
        processor = _delivery_processor(knowledge)
        work = context(
            revision=1,
            operation_id="subscription:cancel-context",
            task_id="task:subscription-cancel",
        )
        try:
            await store.publish_work_context(work, expected_revision=None)
            policy_result = await _process_delivery(
                processor,
                work,
                operation_id="subscription:cancel-policy",
                checkpoint_value=None,
                agent_id="agent:delivery",
            )
            subscription = AgentRecallSubscription.create(
                subscription_id="subscription:cancel",
                agent_id="agent:delivery",
                work_context=work,
                knowledge_namespace="project:delivery",
                access_policy_sha256=policy_result.access_policy_sha256,
                admission_policy=_subscription_policy(policy_result, threshold=0.0),
                minimum_interval_seconds=60,
                expires_at=clock.value + timedelta(days=1),
                revision=1,
                operation_id="subscription:cancel-publish",
                published_by="application:test",
                published_at=clock.value,
                query="checkpoint-aware delivery evidence",
            )
            result = await _process_delivery(
                processor,
                work,
                operation_id="subscription:cancel-processing",
                checkpoint_value=None,
                agent_id="agent:delivery",
                checkpoint_stream_id=subscription.checkpoint_stream_id(),
            )
            await store.publish_recall_subscription(subscription, expected_revision=None)
            claimed = await store.claim_due_recall_subscription(
                subscription.checkpoint_key(),
                claim_id="subscription:cancel-claim",
                runner_id="runtime:inline",
                lease_seconds=300,
            )
            assert claimed is not None and claimed.claim is not None
            async with blocker.cursor() as cursor:
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_recall_subscription_state_update()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    """
                    CREATE TRIGGER cayu_test_block_agent_recall_subscription_state_update
                    BEFORE UPDATE ON cayu_agent_recall_subscription_states
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_recall_subscription_state_update()
                    """
                )
            await blocker.commit()
            await _acquire_postgres_advisory_lock(blocker, lock_key)
            held_lock = True
            pending = asyncio.create_task(
                store.commit_recall_subscription_evaluation(
                    claimed.claim,
                    result,
                    evaluation_id="subscription:cancel-evaluation",
                    delivery_id="subscription:cancel-delivery",
                    staged_by="runtime:inline",
                    evaluated_at=clock.value,
                )
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=lock_key,
                task=pending,
            )
            pending.cancel("cancel atomic subscription wake before state publication")
            with pytest.raises(asyncio.CancelledError):
                await pending
            pending = None
            assert await store.load_recall_checkpoint(subscription.checkpoint_key()) is None
            assert await store.load_recall_delivery("subscription:cancel-delivery") is None
            assert (
                await store.load_recall_subscription_evaluation("subscription:cancel-evaluation")
                is None
            )
            await _release_postgres_advisory_lock(blocker, lock_key)
            held_lock = False
            committed = await store.commit_recall_subscription_evaluation(
                claimed.claim,
                result,
                evaluation_id="subscription:cancel-evaluation",
                delivery_id="subscription:cancel-delivery",
                staged_by="runtime:inline",
                evaluated_at=clock.value,
            )
            assert committed.outcome is AgentRecallSubscriptionEvaluationOutcome.WAKE
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
            if held_lock:
                await _release_postgres_advisory_lock(blocker, lock_key)
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await store.close()
            await blocker.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_sqlite_recall_delivery_rejects_corrupt_denormalized_identity(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = SQLiteAgentWorkContextStore(tmp_path / "delivery-index-corruption.sqlite")
        published = context(revision=1, operation_id="delivery-corruption:sqlite:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-corruption:sqlite",
            operation_id="delivery-corruption:sqlite:process",
            entry_ids=("delivery-corruption-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            await store.stage_recall_delivery(delivery)
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                "UPDATE cayu_agent_recall_deliveries "
                "SET processing_result_sha256 = ? WHERE delivery_id = ?",
                ("f" * 64, delivery.delivery_id),
            )
            with pytest.raises(RuntimeError, match="indexes conflict with durable state"):
                await store.load_recall_delivery(delivery.delivery_id)
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_recall_delivery_rejects_corrupt_denormalized_identity(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        store = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        published = context(revision=1, operation_id="delivery-corruption:postgres:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-corruption:postgres",
            operation_id="delivery-corruption:postgres:process",
            entry_ids=("delivery-corruption-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            await store.stage_recall_delivery(delivery)
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                await connection.execute(
                    "UPDATE cayu_agent_recall_deliveries "
                    "SET processing_result_sha256 = %s WHERE delivery_id = %s",
                    ("f" * 64, delivery.delivery_id),
                )
                await connection.commit()
            with pytest.raises(RuntimeError, match="indexes conflict with durable state"):
                await store.load_recall_delivery(delivery.delivery_id)
        finally:
            await store.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


@pytest.mark.parametrize("use_external_pool", (False, True), ids=("owned-pool", "external-pool"))
def test_postgres_cancellation_rolls_back_context_and_checkpoint_head_updates(
    postgres_dsn: str,
    use_external_pool: bool,
) -> None:
    async def run() -> None:
        import psycopg
        from psycopg_pool import AsyncConnectionPool

        from cayu import PostgresAgentWorkContextStore

        context_lock_key = 7_505_119_600_001
        checkpoint_lock_key = 7_505_119_600_002
        await _drop_postgres_schema(postgres_dsn)
        external_pool = (
            AsyncConnectionPool(
                postgres_dsn,
                min_size=1,
                max_size=1,
                open=False,
            )
            if use_external_pool
            else None
        )
        store = (
            PostgresAgentWorkContextStore(
                pool=external_pool,
                schema_mode=SchemaMode.CREATE,
            )
            if external_pool is not None
            else PostgresAgentWorkContextStore(
                postgres_dsn,
                min_size=1,
                max_size=1,
                schema_mode=SchemaMode.CREATE,
            )
        )
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        held_lock: int | None = None
        pending: asyncio.Task[Any] | None = None
        try:
            initial = context(
                revision=1,
                operation_id="postgres-cancellation:context:create",
            )
            await store.publish_work_context(initial, expected_revision=None)

            async with blocker.cursor() as cursor:
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_work_context_head_update()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({context_lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    """
                    CREATE TRIGGER cayu_test_block_agent_work_context_head_update
                    BEFORE UPDATE ON cayu_agent_work_context_heads
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_work_context_head_update()
                    """
                )
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_recall_checkpoint_head_update()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({checkpoint_lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    """
                    CREATE TRIGGER cayu_test_block_agent_recall_checkpoint_head_update
                    BEFORE UPDATE ON cayu_agent_recall_checkpoint_heads
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_recall_checkpoint_head_update()
                    """
                )
            await blocker.commit()

            await _acquire_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = context_lock_key
            successor = context(
                revision=2,
                operation_id="postgres-cancellation:context:append",
                goal="Rollback a cancelled context publication",
            )
            pending = asyncio.create_task(
                store.publish_work_context(successor, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=context_lock_key,
                task=pending,
            )
            pending.cancel("cancel context publication during head update")
            with pytest.raises(asyncio.CancelledError):
                await pending
            pending = None

            assert await store.load_work_context(initial.task_id) == initial
            assert await store.load_work_context(initial.task_id, revision=2) is None
            assert await store.load_work_context_publication(successor.operation_id) is None
            await _release_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = None
            publication = await store.publish_work_context(successor, expected_revision=1)
            assert publication.context == successor

            initial_checkpoint = checkpoint(
                successor,
                revision=1,
                operation_id="postgres-cancellation:checkpoint:create",
            )
            await store.advance_recall_checkpoint(initial_checkpoint, expected_revision=None)
            await _acquire_postgres_advisory_lock(blocker, checkpoint_lock_key)
            held_lock = checkpoint_lock_key
            successor_checkpoint = checkpoint(
                successor,
                revision=2,
                operation_id="postgres-cancellation:checkpoint:advance",
                knowledge_sequence=11,
                index_readiness_sequence=8,
                processing_mode=AgentRecallCheckpointMode.DELTA,
            )
            pending = asyncio.create_task(
                store.advance_recall_checkpoint(successor_checkpoint, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=checkpoint_lock_key,
                task=pending,
            )
            pending.cancel("cancel checkpoint advancement during head update")
            with pytest.raises(asyncio.CancelledError):
                await pending
            pending = None

            assert (
                await store.load_recall_checkpoint(initial_checkpoint.key()) == initial_checkpoint
            )
            assert await store.load_recall_checkpoint(initial_checkpoint.key(), revision=2) is None
            await _release_postgres_advisory_lock(blocker, checkpoint_lock_key)
            held_lock = None
            assert (
                await store.advance_recall_checkpoint(
                    successor_checkpoint,
                    expected_revision=1,
                )
                == successor_checkpoint
            )
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
            if held_lock is not None:
                await _release_postgres_advisory_lock(blocker, held_lock)
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await store.close()
            if external_pool is not None:
                await external_pool.close()
            await blocker.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_context_publication_fences_stale_checkpoint(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        context_lock_key = 7_505_119_600_003
        await _drop_postgres_schema(postgres_dsn)
        store = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        held_lock = False
        publication_task: asyncio.Task[Any] | None = None
        checkpoint_task: asyncio.Task[Any] | None = None
        try:
            initial = context(
                revision=1,
                operation_id="postgres-context-fence:context:create",
            )
            await store.publish_work_context(initial, expected_revision=None)
            initial_checkpoint = checkpoint(
                initial,
                revision=1,
                operation_id="postgres-context-fence:checkpoint:create",
            )
            await store.advance_recall_checkpoint(initial_checkpoint, expected_revision=None)

            async with blocker.cursor() as cursor:
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_work_context_head_update()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({context_lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    """
                    CREATE TRIGGER cayu_test_block_agent_work_context_head_update
                    BEFORE UPDATE ON cayu_agent_work_context_heads
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_work_context_head_update()
                    """
                )
                await cursor.execute(
                    "SELECT hashtextextended(%s, 0)",
                    (f"cayu-agent-work-context:task:{initial.task_id}",),
                )
                task_lock_row = await cursor.fetchone()
            await blocker.commit()
            assert task_lock_row is not None
            task_lock_key = int(task_lock_row[0])

            await _acquire_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = True
            successor = context(
                revision=2,
                operation_id="postgres-context-fence:context:append",
                goal="Publish the new current processing basis",
            )
            publication_task = asyncio.create_task(
                store.publish_work_context(successor, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=context_lock_key,
                task=publication_task,
            )

            stale_checkpoint = checkpoint(
                initial,
                revision=2,
                operation_id="postgres-context-fence:checkpoint:stale",
                knowledge_sequence=11,
                index_readiness_sequence=8,
                processing_mode=AgentRecallCheckpointMode.FULL_INDEX,
            )
            checkpoint_task = asyncio.create_task(
                store.advance_recall_checkpoint(stale_checkpoint, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=task_lock_key,
                task=checkpoint_task,
            )

            await _release_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = False
            publication = await publication_task
            publication_task = None
            assert publication.context == successor
            with pytest.raises(AgentWorkContextConflict, match="stale_work_context_revision"):
                await checkpoint_task
            checkpoint_task = None
            assert (
                await store.load_recall_checkpoint(initial_checkpoint.key()) == initial_checkpoint
            )
            assert await store.load_recall_checkpoint(initial_checkpoint.key(), revision=2) is None
        finally:
            for pending in (publication_task, checkpoint_task):
                if pending is not None and not pending.done():
                    pending.cancel()
            if held_lock:
                await _release_postgres_advisory_lock(blocker, context_lock_key)
            await asyncio.gather(
                *(
                    pending
                    for pending in (publication_task, checkpoint_task)
                    if pending is not None
                ),
                return_exceptions=True,
            )
            await store.close()
            await blocker.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_work_context_store_rejects_autocommit_pool_and_configuration_drift() -> None:
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresAgentWorkContextStore

    autocommit_pool = AsyncConnectionPool(
        "",
        open=False,
        kwargs={"autocommit": True},
    )
    with pytest.raises(TypeError, match="work-context mutations require transactional"):
        PostgresAgentWorkContextStore(pool=autocommit_pool)

    pool = AsyncConnectionPool("", open=False, kwargs={})
    store = PostgresAgentWorkContextStore(pool=pool)
    pool_kwargs = cast("dict[str, Any]", pool.kwargs)
    pool_kwargs["autocommit"] = True

    async def reject_drift() -> None:
        with pytest.raises(TypeError, match="work-context mutations require transactional"):
            await store.load_work_context("task:pool-drift")

    asyncio.run(reject_drift())


def test_agent_work_context_canonicalizes_collections_and_binds_content() -> None:
    value = AgentWorkContext.create(
        task_id="task:canonical",
        goal="Keep deterministic task state",
        revision=1,
        operation_id="operation:canonical",
        published_by="application:test",
        published_at=datetime(2026, 8, 28, tzinfo=UTC),
        scope_ids=("scope:z", "scope:a"),
        entity_ids=("entity:b", "entity:a"),
    )
    assert value.scope_ids == ("scope:a", "scope:z")
    assert value.entity_ids == ("entity:a", "entity:b")
    assert AgentWorkContext.model_validate_json(value.model_dump_json()) == value
    checkpoint_value = checkpoint(
        value,
        revision=1,
        operation_id="checkpoint:canonical",
    )
    assert (
        AgentRecallCheckpoint.model_validate_json(checkpoint_value.model_dump_json())
        == checkpoint_value
    )
    checkpoint_key = checkpoint_value.key()
    assert (
        AgentRecallCheckpointKey.model_validate_json(checkpoint_key.model_dump_json())
        == checkpoint_key
    )
    assert len(checkpoint_key.fingerprint()) == 64
    with pytest.raises(ValidationError, match="content_sha256"):
        value.model_copy(update={"goal": "Altered without a new content identity"})


def test_agent_work_context_publication_receipt_rejects_impossible_no_change() -> None:
    stored = context(revision=1, operation_id="receipt:stored-context")

    with pytest.raises(ValidationError, match="requires an expected revision"):
        AgentWorkContextPublicationReceipt(
            operation_id="receipt:impossible-create-no-change",
            request_sha256="a" * 64,
            expected_revision=None,
            requested_content_sha256=stored.content_sha256,
            changed=False,
            context=stored,
            committed_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="requires a distinct operation identity"):
        AgentWorkContextPublicationReceipt(
            operation_id=stored.operation_id,
            request_sha256="a" * 64,
            expected_revision=1,
            requested_content_sha256=stored.content_sha256,
            changed=False,
            context=stored,
            committed_at=datetime(2026, 8, 28, tzinfo=UTC),
        )


def test_agent_work_context_rejects_duplicate_and_oversized_values() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        context(
            revision=1,
            operation_id="context:duplicate",
            entity_ids=("same", "same"),
        )
    with pytest.raises(ValidationError, match="goal"):
        AgentWorkContext.create(
            task_id="task:oversized",
            goal="x" * 32_001,
            revision=1,
            operation_id="operation:oversized",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="workflow_iteration"):
        AgentWorkContext.create(
            task_id="task:iteration-overflow",
            goal="Reject unsafe JSON integer overflow",
            revision=1,
            operation_id="operation:iteration-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
            workflow_id="workflow:test",
            workflow_phase="bounded",
            workflow_iteration=9_223_372_036_854_775_808,
        )
    with pytest.raises(ValidationError, match="knowledge_sequence"):
        AgentRecallCheckpoint(
            agent_id="agent:overflow",
            task_id="task:overflow",
            knowledge_namespace="project:cayu",
            access_policy_sha256="a" * 64,
            revision=1,
            work_context_revision=1,
            work_context_sha256="b" * 64,
            knowledge_sequence=9_223_372_036_854_775_808,
            index_readiness_sequence=0,
            knowledge_high_water_sequence=9_223_372_036_854_775_808,
            index_readiness_high_water_sequence=0,
            processing_mode=AgentRecallCheckpointMode.FULL_INDEX,
            processing_id="processing:overflow",
            operation_id="operation:overflow",
            updated_by="application:test",
            updated_at=datetime(2026, 8, 28, tzinfo=UTC),
        )

    valid_context = context(revision=1, operation_id="context:frontier-bounds")
    with pytest.raises(ValidationError, match="knowledge_high_water_sequence"):
        checkpoint(
            valid_context,
            revision=1,
            operation_id="checkpoint:future-knowledge",
            knowledge_sequence=11,
            knowledge_high_water_sequence=10,
        )
    with pytest.raises(ValidationError, match="index_readiness_high_water_sequence"):
        checkpoint(
            valid_context,
            revision=1,
            operation_id="checkpoint:future-index",
            index_readiness_sequence=8,
            index_readiness_high_water_sequence=7,
        )
    with pytest.raises(ValidationError, match="task_id"):
        AgentWorkContext.create(
            task_id="x" * 513,
            goal="Reject oversized durable identity",
            revision=1,
            operation_id="operation:identity-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="cannot contain more than 128"):
        AgentWorkContext.create(
            task_id="task:collection-overflow",
            goal="Reject unbounded collections",
            revision=1,
            operation_id="operation:collection-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
            entity_ids=tuple(f"entity:{index}" for index in range(129)),
        )
    with pytest.raises(ValidationError, match="serialized byte limit"):
        AgentWorkContext.create(
            task_id="task:record-overflow",
            goal="Reject oversized aggregate content",
            revision=1,
            operation_id="operation:record-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
            entity_ids=tuple(f"entity:{index:03d}:" + "x" * 2_990 for index in range(128)),
        )

    subscription_context = context(
        revision=1,
        operation_id="context:subscription-input-bounds",
        entity_ids=("entity:a:" + "x" * 4_080, "entity:b:" + "x" * 4_080),
    )
    policy = AutomaticRecallPolicy(
        calibration_version="subscription-input-bounds-v1",
        fusion_strategy_version="subscription-input-bounds-v1",
        fusion_configuration_version="subscription-input-bounds-v1",
        minimum_inject_score=0.5,
        minimum_offer_score=0.5,
    )
    subscription_arguments = {
        "subscription_id": "subscription:input-bounds",
        "agent_id": "agent:input-bounds",
        "work_context": subscription_context,
        "knowledge_namespace": "project:input-bounds",
        "access_policy_sha256": "a" * 64,
        "admission_policy": policy,
        "minimum_interval_seconds": 60,
        "expires_at": datetime(2026, 8, 29, tzinfo=UTC),
        "revision": 1,
        "operation_id": "subscription:input-bounds:publish",
        "published_by": "application:test",
        "published_at": datetime(2026, 8, 28, tzinfo=UTC),
    }
    with pytest.raises(ValidationError, match="query.*at most 8192"):
        AgentRecallSubscription.create(
            **subscription_arguments,
            query="x" * 8_193,
        )
    facet_subscription = AgentRecallSubscription.create(
        **subscription_arguments,
        entity_ids=subscription_context.entity_ids,
    )
    assert facet_subscription.facet_aspect_groups() == (
        tuple(
            agent_recall_facet_aspect("entity_ids", value)
            for value in subscription_context.entity_ids
        ),
    )

    valid = context(revision=1, operation_id="context:strict-input")
    invalid_payload = valid.model_dump(mode="python")
    invalid_payload["revision"] = "1"
    with pytest.raises(ValidationError, match="revision"):
        AgentWorkContext.model_validate(invalid_payload)


_POST_REVISION_70_RECALL_TABLES = (
    "cayu_agent_recall_subscription_wake_states",
    "cayu_agent_recall_subscription_wake_releases",
    "cayu_agent_recall_subscription_wake_claims",
    "cayu_agent_recall_subscription_evaluations",
    "cayu_agent_recall_subscription_states",
    "cayu_agent_recall_subscription_releases",
    "cayu_agent_recall_subscription_claims",
    "cayu_agent_recall_subscription_publications",
    "cayu_agent_recall_subscription_heads",
    "cayu_agent_recall_subscription_revisions",
    "cayu_agent_recall_delivery_acknowledgements",
    "cayu_agent_recall_delivery_states",
    "cayu_agent_recall_delivery_releases",
    "cayu_agent_recall_delivery_claims",
    "cayu_agent_recall_deliveries",
)


def _without_checkpoint_stream_identity(ddl: str) -> str:
    legacy = ddl.replace(
        "            checkpoint_stream_id TEXT COLLATE BINARY NOT NULL,\n",
        "",
    ).replace(
        '            checkpoint_stream_id TEXT COLLATE "C" NOT NULL,\n',
        "",
    )
    legacy = legacy.replace(", checkpoint_stream_id", "")
    assert "checkpoint_stream_id" not in legacy
    return legacy


def _sqlite_schema_snapshot(connection: sqlite3.Connection) -> tuple[object, ...]:
    return (
        connection.execute("PRAGMA user_version").fetchone(),
        tuple(
            connection.execute(
                "SELECT revision, kind, compatible_from, checksum, applied_at "
                "FROM cayu_schema_migrations ORDER BY revision"
            )
        ),
        tuple(
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE tbl_name LIKE 'cayu_%' ORDER BY type, name"
            )
        ),
    )


@pytest.mark.parametrize("historical_revision", (69, 70))
def test_sqlite_revision_73_rejects_pre_stream_checkpoint_schema_before_mutation(
    tmp_path: Path,
    historical_revision: int,
) -> None:
    database = tmp_path / f"revision-{historical_revision}-pre-stream-checkpoint.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in _POST_REVISION_70_RECALL_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute("DROP TABLE cayu_agent_recall_checkpoint_heads")
        connection.execute("DROP TABLE cayu_agent_recall_checkpoints")
        connection.execute("DROP TABLE cayu_task_interrupted_handoff_receipts")
        connection.execute("DROP INDEX IF EXISTS idx_cayu_tasks_interrupted_handoff_recovery")
        connection.executescript(
            _without_checkpoint_stream_identity(sqlite_support._MIGRATION_STEPS[69])
        )
        if historical_revision == 70:
            connection.executescript(sqlite_support._MIGRATION_STEPS[70])
        connection.execute(
            "DELETE FROM cayu_schema_migrations WHERE revision > ?",
            (historical_revision,),
        )
        connection.execute(f"PRAGMA user_version = {historical_revision}")
        connection.commit()
        before = _sqlite_schema_snapshot(connection)
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match="independent recall checkpoint streams",
    ):
        SQLiteAgentWorkContextStore(database, schema_mode=SchemaMode.MIGRATE)

    connection = sqlite3.connect(database)
    try:
        assert _sqlite_schema_snapshot(connection) == before
        assert "checkpoint_stream_id" not in {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(cayu_agent_recall_checkpoints)")
        }
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'cayu_agent_recall_deliveries'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


async def _postgres_schema_snapshot(cursor: Any) -> tuple[object, ...]:
    await cursor.execute(
        "SELECT revision, kind, compatible_from, checksum, applied_at "
        "FROM cayu_schema_migrations ORDER BY revision"
    )
    ledger = tuple(await cursor.fetchall())
    await cursor.execute(
        "SELECT table_name, column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name LIKE 'cayu_%' "
        "ORDER BY table_name, ordinal_position"
    )
    columns = tuple(await cursor.fetchall())
    await cursor.execute(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%' "
        "ORDER BY indexname"
    )
    indexes = tuple(await cursor.fetchall())
    return ledger, columns, indexes


@pytest.mark.parametrize("historical_revision", (69, 70))
def test_postgres_revision_73_rejects_pre_stream_checkpoint_schema_before_mutation(
    postgres_dsn: str,
    historical_revision: int,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore
        from cayu.storage import postgres as postgres_storage

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                for table in _POST_REVISION_70_RECALL_TABLES:
                    await cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                await cursor.execute("DROP TABLE cayu_agent_recall_checkpoint_heads")
                await cursor.execute("DROP TABLE cayu_agent_recall_checkpoints")
                await cursor.execute("DROP TABLE cayu_task_interrupted_handoff_receipts")
                await cursor.execute(
                    "DROP INDEX IF EXISTS idx_cayu_tasks_interrupted_handoff_recovery"
                )
                for statement in postgres_storage._MIGRATION_STEPS[69]:
                    await cursor.execute(_without_checkpoint_stream_identity(statement))
                if historical_revision == 70:
                    for statement in postgres_storage._MIGRATION_STEPS[70]:
                        await cursor.execute(statement)
                    for index in postgres_storage._CONCURRENT_INDEX_MIGRATIONS[70]:
                        await cursor.execute(index.transactional_create_statement())
                await cursor.execute(
                    "DELETE FROM cayu_schema_migrations WHERE revision > %s",
                    (historical_revision,),
                )
            await connection.commit()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            before = await _postgres_schema_snapshot(cursor)

        migrator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            with pytest.raises(
                schema_migrations.SchemaTooOld,
                match="independent recall checkpoint streams",
            ):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            assert await _postgres_schema_snapshot(cursor) == before
            await cursor.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'cayu_agent_recall_checkpoints' "
                "AND column_name = 'checkpoint_stream_id'"
            )
            assert await cursor.fetchone() is None
            await cursor.execute("SELECT to_regclass('cayu_agent_recall_deliveries')")
            assert await cursor.fetchone() == (None,)
        await _drop_postgres_schema(postgres_dsn)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_postgres_schema(postgres_dsn))


def test_sqlite_revision_69_adds_empty_work_context_storage_without_backfill(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-68-to-69-populated.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(
            database,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="revision-67-entry",
                    text="Preserve data without inventing agent work context.",
                )
            )
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        for table in (
            "cayu_agent_recall_subscription_wake_states",
            "cayu_agent_recall_subscription_wake_releases",
            "cayu_agent_recall_subscription_wake_claims",
            "cayu_agent_recall_subscription_evaluations",
            "cayu_agent_recall_subscription_states",
            "cayu_agent_recall_subscription_releases",
            "cayu_agent_recall_subscription_claims",
            "cayu_agent_recall_subscription_publications",
            "cayu_agent_recall_subscription_heads",
            "cayu_agent_recall_subscription_revisions",
            "cayu_agent_recall_delivery_states",
            "cayu_agent_recall_delivery_releases",
            "cayu_agent_recall_delivery_claims",
            "cayu_agent_recall_deliveries",
            "cayu_agent_recall_checkpoint_heads",
            "cayu_agent_recall_checkpoints",
            "cayu_agent_work_context_publications",
            "cayu_agent_work_context_heads",
            "cayu_agent_work_context_revisions",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute(
            "DELETE FROM cayu_schema_migrations WHERE revision IN (69, 70, 71, 72, 73)"
        )
        connection.execute("PRAGMA user_version = 68")
        connection.commit()
    finally:
        connection.close()

    store = SQLiteAgentWorkContextStore(database, schema_mode=SchemaMode.MIGRATE)
    asyncio.run(store.close())

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            schema_migrations.LATEST_REVISION,
        )
        assert connection.execute(
            "SELECT text FROM cayu_knowledge_revisions "
            "WHERE entry_id = 'revision-67-entry' AND revision = 1"
        ).fetchone() == ("Preserve data without inventing agent work context.",)
        for table in (
            "cayu_agent_work_context_revisions",
            "cayu_agent_work_context_publications",
            "cayu_agent_recall_checkpoints",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (schema_migrations.LATEST_REVISION,)
    finally:
        connection.close()


def test_sqlite_revision_69_rejects_malformed_work_context_storage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-69-malformed-work-context.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE cayu_agent_work_context_publications")
        connection.execute(
            "CREATE TABLE cayu_agent_work_context_publications (operation_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
        SQLiteAgentWorkContextStore(database)


def test_sqlite_revision_69_primary_identities_are_non_null(tmp_path: Path) -> None:
    database = tmp_path / "revision-69-non-null-primary-identities.sqlite"

    async def seed() -> AgentWorkContext:
        store = SQLiteAgentWorkContextStore(database)
        value = context(revision=1, operation_id="sqlite-non-null:context")
        try:
            await store.publish_work_context(value, expected_revision=None)
        finally:
            await store.close()
        return value

    stored = asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            connection.execute(
                "INSERT INTO cayu_agent_work_context_heads "
                "(task_id, current_revision) VALUES (NULL, 1)"
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            connection.execute(
                "INSERT INTO cayu_agent_work_context_publications ("
                "operation_id, task_id, request_sha256, context_revision, "
                "changed, receipt_json, committed_at"
                ") VALUES (NULL, ?, ?, 1, 0, '{}', ?)",
                (
                    stored.task_id,
                    "a" * 64,
                    "2026-08-28T00:00:00+00:00",
                ),
            )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "malformation",
    (
        "nocase_identity",
        "split_foreign_key",
        "missing_revision_check",
        "nullable_primary_identity",
    ),
)
def test_sqlite_revision_69_rejects_subtle_work_context_schema_conflicts(
    tmp_path: Path,
    malformation: str,
) -> None:
    database = tmp_path / f"revision-69-{malformation}.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())
    ddl = sqlite_support._MIGRATION_STEPS[69]
    if malformation == "nocase_identity":
        malformed_ddl = ddl.replace("COLLATE BINARY", "COLLATE NOCASE")
    elif malformation == "split_foreign_key":
        malformed_ddl = ddl.replace(
            """FOREIGN KEY (task_id, current_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT""",
            """FOREIGN KEY (task_id)
                REFERENCES cayu_agent_work_context_revisions(task_id) ON DELETE RESTRICT,
            FOREIGN KEY (current_revision)
                REFERENCES cayu_agent_work_context_revisions(revision) ON DELETE RESTRICT""",
            1,
        )
    elif malformation == "missing_revision_check":
        prefix, marker, checkpoint_ddl = ddl.partition(
            "CREATE TABLE IF NOT EXISTS cayu_agent_recall_checkpoints"
        )
        assert marker
        checkpoint_ddl = checkpoint_ddl.replace(
            """revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),""",
            "revision INTEGER NOT NULL,",
            1,
        )
        malformed_ddl = prefix + marker + checkpoint_ddl
    else:
        malformed_ddl = ddl.replace(
            "task_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY",
            "task_id TEXT COLLATE BINARY PRIMARY KEY",
            1,
        )
    assert malformed_ddl != ddl

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "cayu_agent_recall_checkpoint_heads",
            "cayu_agent_recall_checkpoints",
            "cayu_agent_work_context_publications",
            "cayu_agent_work_context_heads",
            "cayu_agent_work_context_revisions",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.executescript(malformed_ddl)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
        SQLiteAgentWorkContextStore(database)


def test_postgres_revision_69_adds_empty_work_context_storage_without_backfill(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore, PostgresKnowledgeStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await creator.ensure_schema()
            await creator.create_entry(
                KnowledgeEntry(
                    id="revision-67-entry",
                    text="Preserve data without inventing agent work context.",
                )
            )
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_wake_states")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_wake_releases")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_wake_claims")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_evaluations")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_states")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_releases")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_claims")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_publications")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_heads")
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_revisions")
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_states")
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_releases")
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_claims")
                await cursor.execute("DROP TABLE cayu_agent_recall_deliveries")
                await cursor.execute("DROP TABLE cayu_agent_recall_checkpoint_heads")
                await cursor.execute("DROP TABLE cayu_agent_recall_checkpoints")
                await cursor.execute("DROP TABLE cayu_agent_work_context_publications")
                await cursor.execute("DROP TABLE cayu_agent_work_context_heads")
                await cursor.execute("DROP TABLE cayu_agent_work_context_revisions")
                await cursor.execute(
                    "DELETE FROM cayu_schema_migrations WHERE revision IN (69, 70, 71, 72, 73)"
                )
            await connection.commit()

        migrator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (schema_migrations.LATEST_REVISION,)
            await cursor.execute(
                "SELECT text FROM cayu_knowledge_revisions "
                "WHERE entry_id = 'revision-67-entry' AND revision = 1"
            )
            assert await cursor.fetchone() == (
                "Preserve data without inventing agent work context.",
            )
            for table in (
                "cayu_agent_work_context_revisions",
                "cayu_agent_work_context_publications",
                "cayu_agent_recall_checkpoints",
            ):
                await cursor.execute(f"SELECT COUNT(*) FROM {table}")
                assert await cursor.fetchone() == (0,)

        await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_69_rejects_malformed_work_context_storage(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_agent_work_context_publications")
                await cursor.execute(
                    "CREATE TABLE cayu_agent_work_context_publications "
                    "(operation_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_69_rejects_missing_checkpoint_revision_constraint(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "ALTER TABLE cayu_agent_recall_checkpoints DROP CONSTRAINT "
                    "cayu_agent_recall_checkpoints_revision_check"
                )
            await connection.commit()

        validator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_sqlite_revision_71_adds_empty_delivery_storage_without_backfill(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-70-to-71-with-checkpoint.sqlite"
    published = context(revision=1, operation_id="revision-71:sqlite:context")
    processed = checkpoint(
        published,
        revision=1,
        operation_id="revision-71:sqlite:checkpoint",
    )

    async def seed() -> None:
        store = SQLiteAgentWorkContextStore(database)
        try:
            await store.publish_work_context(published, expected_revision=None)
            await store.advance_recall_checkpoint(processed, expected_revision=None)
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "cayu_agent_recall_subscription_wake_states",
            "cayu_agent_recall_subscription_wake_releases",
            "cayu_agent_recall_subscription_wake_claims",
            "cayu_agent_recall_subscription_evaluations",
            "cayu_agent_recall_subscription_states",
            "cayu_agent_recall_subscription_releases",
            "cayu_agent_recall_subscription_claims",
            "cayu_agent_recall_subscription_publications",
            "cayu_agent_recall_subscription_heads",
            "cayu_agent_recall_subscription_revisions",
            "cayu_agent_recall_delivery_states",
            "cayu_agent_recall_delivery_releases",
            "cayu_agent_recall_delivery_claims",
            "cayu_agent_recall_deliveries",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision IN (71, 72, 73)")
        connection.execute("PRAGMA user_version = 70")
        connection.commit()
    finally:
        connection.close()

    migrated = SQLiteAgentWorkContextStore(database, schema_mode=SchemaMode.MIGRATE)

    async def verify() -> None:
        try:
            assert await migrated.load_work_context(published.task_id) == published
            assert await migrated.load_recall_checkpoint(processed.key()) == processed
        finally:
            await migrated.close()

    asyncio.run(verify())
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            schema_migrations.LATEST_REVISION,
        )
        for table in (
            "cayu_agent_recall_deliveries",
            "cayu_agent_recall_delivery_states",
            "cayu_agent_recall_delivery_claims",
            "cayu_agent_recall_delivery_releases",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
    finally:
        connection.close()


def test_sqlite_revision_71_rejects_malformed_delivery_storage(tmp_path: Path) -> None:
    database = tmp_path / "revision-71-malformed-delivery.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE cayu_agent_recall_delivery_states")
        connection.execute(
            "CREATE TABLE cayu_agent_recall_delivery_states (delivery_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="recall-delivery contract"):
        SQLiteAgentWorkContextStore(database)


def test_postgres_revision_71_adds_empty_delivery_storage_without_backfill(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        published = context(revision=1, operation_id="revision-71:postgres:context")
        processed = checkpoint(
            published,
            revision=1,
            operation_id="revision-71:postgres:checkpoint",
        )
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.publish_work_context(published, expected_revision=None)
            await creator.advance_recall_checkpoint(processed, expected_revision=None)
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                for table in (
                    "cayu_agent_recall_subscription_wake_states",
                    "cayu_agent_recall_subscription_wake_releases",
                    "cayu_agent_recall_subscription_wake_claims",
                    "cayu_agent_recall_subscription_evaluations",
                    "cayu_agent_recall_subscription_states",
                    "cayu_agent_recall_subscription_releases",
                    "cayu_agent_recall_subscription_claims",
                    "cayu_agent_recall_subscription_publications",
                    "cayu_agent_recall_subscription_heads",
                    "cayu_agent_recall_subscription_revisions",
                    "cayu_agent_recall_delivery_states",
                    "cayu_agent_recall_delivery_releases",
                    "cayu_agent_recall_delivery_claims",
                    "cayu_agent_recall_deliveries",
                ):
                    await cursor.execute(f"DROP TABLE {table}")
                await cursor.execute(
                    "DELETE FROM cayu_schema_migrations WHERE revision IN (71, 72, 73)"
                )
            await connection.commit()

        migrator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            assert await migrator.load_work_context(published.task_id) == published
            assert await migrator.load_recall_checkpoint(processed.key()) == processed
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (schema_migrations.LATEST_REVISION,)
            for table in (
                "cayu_agent_recall_deliveries",
                "cayu_agent_recall_delivery_states",
                "cayu_agent_recall_delivery_claims",
                "cayu_agent_recall_delivery_releases",
            ):
                await cursor.execute(f"SELECT COUNT(*) FROM {table}")
                assert await cursor.fetchone() == (0,)
        await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_71_rejects_malformed_delivery_storage(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_states")
                await cursor.execute(
                    "CREATE TABLE cayu_agent_recall_delivery_states (delivery_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="recall-delivery contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_sqlite_revision_73_adds_empty_subscription_storage_without_inference(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-72-to-73-with-checkpoint.sqlite"
    published = context(revision=1, operation_id="revision-73:sqlite:context")
    processed = checkpoint(
        published,
        revision=1,
        operation_id="revision-73:sqlite:checkpoint",
    )

    async def seed() -> None:
        store = SQLiteAgentWorkContextStore(database)
        try:
            await store.publish_work_context(published, expected_revision=None)
            await store.advance_recall_checkpoint(processed, expected_revision=None)
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "cayu_agent_recall_subscription_wake_states",
            "cayu_agent_recall_subscription_wake_releases",
            "cayu_agent_recall_subscription_wake_claims",
            "cayu_agent_recall_subscription_evaluations",
            "cayu_agent_recall_subscription_states",
            "cayu_agent_recall_subscription_releases",
            "cayu_agent_recall_subscription_claims",
            "cayu_agent_recall_subscription_publications",
            "cayu_agent_recall_subscription_heads",
            "cayu_agent_recall_subscription_revisions",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 73")
        connection.execute("PRAGMA user_version = 72")
        connection.commit()
    finally:
        connection.close()

    migrated = SQLiteAgentWorkContextStore(database, schema_mode=SchemaMode.MIGRATE)

    async def verify() -> None:
        try:
            assert await migrated.load_work_context(published.task_id) == published
            assert await migrated.load_recall_checkpoint(processed.key()) == processed
            assert await migrated.load_recall_subscription("not-inferred") is None
        finally:
            await migrated.close()

    asyncio.run(verify())
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (73,)
        for table in (
            "cayu_agent_recall_subscription_revisions",
            "cayu_agent_recall_subscription_publications",
            "cayu_agent_recall_subscription_states",
            "cayu_agent_recall_subscription_claims",
            "cayu_agent_recall_subscription_releases",
            "cayu_agent_recall_subscription_evaluations",
            "cayu_agent_recall_subscription_wake_claims",
            "cayu_agent_recall_subscription_wake_releases",
            "cayu_agent_recall_subscription_wake_states",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
        with pytest.raises(sqlite3.IntegrityError, match="processing_schema_version"):
            connection.execute(
                """
                INSERT INTO cayu_agent_recall_deliveries (
                        delivery_id, operation_id, agent_id, task_id,
                        knowledge_namespace, access_policy_sha256,
                        checkpoint_stream_id,
                        checkpoint_revision, processing_result_sha256,
                        delivery_json, staged_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "revision-73:sqlite:pre-contract-writer",
                    processed.operation_id,
                    processed.agent_id,
                    processed.task_id,
                    processed.knowledge_namespace,
                    processed.access_policy_sha256,
                    processed.checkpoint_stream_id,
                    processed.revision,
                    "a" * 64,
                    "{}",
                    processed.updated_at.isoformat(),
                ),
            )
        connection.rollback()
    finally:
        connection.close()


def test_sqlite_revision_73_rejects_malformed_subscription_storage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-73-malformed-subscription.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE cayu_agent_recall_subscription_wake_states")
        connection.execute(
            "CREATE TABLE cayu_agent_recall_subscription_wake_states (wake_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="recall-subscription contract"):
        SQLiteAgentWorkContextStore(database)


def test_sqlite_revision_73_rejects_populated_pre_contract_deliveries(tmp_path: Path) -> None:
    database = tmp_path / "revision-73-populated-delivery.sqlite"
    published = context(revision=1, operation_id="revision-73:legacy-delivery:context")

    async def seed() -> None:
        store = SQLiteAgentWorkContextStore(database)
        try:
            await store.publish_work_context(published, expected_revision=None)
            delivery = await recall_delivery(
                published,
                delivery_id="revision-73:legacy-delivery",
                operation_id="revision-73:legacy-delivery:processing",
                entry_ids=("revision-73-legacy-delivery-entry",),
            )
            await store.stage_recall_delivery(delivery)
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "cayu_agent_recall_subscription_wake_states",
            "cayu_agent_recall_subscription_wake_releases",
            "cayu_agent_recall_subscription_wake_claims",
            "cayu_agent_recall_subscription_evaluations",
            "cayu_agent_recall_subscription_states",
            "cayu_agent_recall_subscription_releases",
            "cayu_agent_recall_subscription_claims",
            "cayu_agent_recall_subscription_publications",
            "cayu_agent_recall_subscription_heads",
            "cayu_agent_recall_subscription_revisions",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 73")
        connection.execute("PRAGMA user_version = 72")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match="cannot migrate a populated recall-delivery database",
    ):
        SQLiteAgentWorkContextStore(database, schema_mode=SchemaMode.MIGRATE)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (72,)
        assert connection.execute(
            "SELECT COUNT(*) FROM cayu_agent_recall_deliveries"
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_postgres_revision_73_adds_empty_subscription_storage_without_inference(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        published = context(revision=1, operation_id="revision-73:postgres:context")
        processed = checkpoint(
            published,
            revision=1,
            operation_id="revision-73:postgres:checkpoint",
        )
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.publish_work_context(published, expected_revision=None)
            await creator.advance_recall_checkpoint(processed, expected_revision=None)
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                for table in (
                    "cayu_agent_recall_subscription_wake_states",
                    "cayu_agent_recall_subscription_wake_releases",
                    "cayu_agent_recall_subscription_wake_claims",
                    "cayu_agent_recall_subscription_evaluations",
                    "cayu_agent_recall_subscription_states",
                    "cayu_agent_recall_subscription_releases",
                    "cayu_agent_recall_subscription_claims",
                    "cayu_agent_recall_subscription_publications",
                    "cayu_agent_recall_subscription_heads",
                    "cayu_agent_recall_subscription_revisions",
                ):
                    await cursor.execute(f"DROP TABLE {table}")
                await cursor.execute("DELETE FROM cayu_schema_migrations WHERE revision = 73")
            await connection.commit()

        migrator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            assert await migrator.load_work_context(published.task_id) == published
            assert await migrator.load_recall_checkpoint(processed.key()) == processed
            assert await migrator.load_recall_subscription("not-inferred") is None
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (73,)
            for table in (
                "cayu_agent_recall_subscription_revisions",
                "cayu_agent_recall_subscription_publications",
                "cayu_agent_recall_subscription_states",
                "cayu_agent_recall_subscription_claims",
                "cayu_agent_recall_subscription_releases",
                "cayu_agent_recall_subscription_evaluations",
                "cayu_agent_recall_subscription_wake_claims",
                "cayu_agent_recall_subscription_wake_releases",
                "cayu_agent_recall_subscription_wake_states",
            ):
                await cursor.execute(f"SELECT COUNT(*) FROM {table}")
                assert await cursor.fetchone() == (0,)
            with pytest.raises(
                psycopg.errors.NotNullViolation,
                match="processing_schema_version",
            ):
                await cursor.execute(
                    """
                    INSERT INTO cayu_agent_recall_deliveries (
                        delivery_id, operation_id, agent_id, task_id,
                        knowledge_namespace, access_policy_sha256,
                        checkpoint_stream_id,
                        checkpoint_revision, processing_result_sha256,
                        delivery_json, staged_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                    )
                    """,
                    (
                        "revision-73:postgres:pre-contract-writer",
                        processed.operation_id,
                        processed.agent_id,
                        processed.task_id,
                        processed.knowledge_namespace,
                        processed.access_policy_sha256,
                        processed.checkpoint_stream_id,
                        processed.revision,
                        "a" * 64,
                        "{}",
                        processed.updated_at,
                    ),
                )
            await connection.rollback()
        await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_73_rejects_malformed_subscription_storage(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_agent_recall_subscription_wake_states")
                await cursor.execute(
                    "CREATE TABLE cayu_agent_recall_subscription_wake_states "
                    "(wake_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="recall-subscription contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_73_rejects_populated_pre_contract_deliveries(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        published = context(
            revision=1,
            operation_id="revision-73:postgres-legacy-delivery:context",
        )
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.publish_work_context(published, expected_revision=None)
            delivery = await recall_delivery(
                published,
                delivery_id="revision-73:postgres-legacy-delivery",
                operation_id="revision-73:postgres-legacy-delivery:processing",
                entry_ids=("revision-73-postgres-legacy-delivery-entry",),
            )
            await creator.stage_recall_delivery(delivery)
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                for table in (
                    "cayu_agent_recall_subscription_wake_states",
                    "cayu_agent_recall_subscription_wake_releases",
                    "cayu_agent_recall_subscription_wake_claims",
                    "cayu_agent_recall_subscription_evaluations",
                    "cayu_agent_recall_subscription_states",
                    "cayu_agent_recall_subscription_releases",
                    "cayu_agent_recall_subscription_claims",
                    "cayu_agent_recall_subscription_publications",
                    "cayu_agent_recall_subscription_heads",
                    "cayu_agent_recall_subscription_revisions",
                ):
                    await cursor.execute(f"DROP TABLE {table}")
                await cursor.execute("DELETE FROM cayu_schema_migrations WHERE revision = 73")
            await connection.commit()

        migrator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            with pytest.raises(
                schema_migrations.SchemaTooOld,
                match="cannot migrate a populated recall-delivery database",
            ):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (72,)
            await cursor.execute("SELECT COUNT(*) FROM cayu_agent_recall_deliveries")
            assert await cursor.fetchone() == (1,)
        await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())
