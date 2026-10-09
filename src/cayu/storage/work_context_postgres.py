"""PostgreSQL agent work contexts, recall subscriptions and delivery checkpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from cayu.storage import _postgres_base as postgres_base
from cayu.storage import _postgres_support as pg_support
from cayu.storage._phase_timing import PostgresTimingScope, timed_postgres_connection

try:
    from psycopg_pool import AsyncConnectionPool  # noqa: TC002 — keep runtime annotation resolution
except ModuleNotFoundError as exc:
    raise RuntimeError(
        'Cayu\'s Postgres stores require the optional psycopg packages. Install them with `pip install "cayu[postgres]"`.'
    ) from exc
from cayu._clock import utc_clock
from cayu.storage import migrations as schema
from cayu.storage._postgres_verified_work import (
    _require_quiescent_postgres_mutation_connection,
    _require_quiescent_postgres_mutation_pool,
)
from cayu.work_context import (
    AgentRecallCheckpoint,
    AgentRecallCheckpointKey,
    AgentRecallDelivery,
    AgentRecallDeliveryClaim,
    AgentRecallDeliveryConflict,
    AgentRecallDeliveryEvidenceKind,
    AgentRecallDeliveryRecord,
    AgentRecallDeliveryRelease,
    AgentRecallDeliveryState,
    AgentRecallSubscription,
    AgentRecallSubscriptionClaim,
    AgentRecallSubscriptionConflict,
    AgentRecallSubscriptionEvaluation,
    AgentRecallSubscriptionPublicationReceipt,
    AgentRecallSubscriptionRecord,
    AgentRecallSubscriptionRelease,
    AgentRecallSubscriptionRunState,
    AgentRecallSubscriptionWake,
    AgentRecallSubscriptionWakeClaim,
    AgentRecallSubscriptionWakeRelease,
    AgentRecallSubscriptionWakeState,
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextPublicationReceipt,
    AgentWorkContextStore,
    _acknowledge_agent_recall_delivery_record,
    _acknowledge_agent_recall_subscription_wake,
    _agent_recall_delivery_release,
    _agent_recall_subscription_release,
    _agent_recall_subscription_wake_release,
    _bounded_identity,
    _claim_agent_recall_delivery_record,
    _claim_agent_recall_subscription_record,
    _claim_agent_recall_subscription_wake,
    _positive_revision,
    _prepare_agent_recall_subscription_evaluation,
    _release_agent_recall_delivery_record,
    _release_agent_recall_subscription_record,
    _release_agent_recall_subscription_wake,
    _renew_agent_recall_delivery_record,
    _renew_agent_recall_subscription_record,
    _renew_agent_recall_subscription_wake,
    _require_replayable_delivery_claim_attempt,
    _require_replayable_subscription_wake_claim,
    _utc,
    _validate_delivery_lease_seconds,
    agent_recall_delivery_claim_request_sha256,
    agent_recall_subscription_claim_request_sha256,
    agent_recall_subscription_evaluation_request_sha256,
    agent_recall_subscription_publication_request_sha256,
    agent_recall_subscription_wake_claim_request_sha256,
    agent_work_context_publication_request_sha256,
    copy_agent_recall_checkpoint,
    copy_agent_recall_checkpoint_key,
    copy_agent_recall_delivery,
    copy_agent_recall_delivery_claim,
    copy_agent_recall_delivery_record,
    copy_agent_recall_subscription,
    copy_agent_recall_subscription_claim,
    copy_agent_recall_subscription_evaluation,
    copy_agent_recall_subscription_publication_receipt,
    copy_agent_recall_subscription_record,
    copy_agent_recall_subscription_wake,
    copy_agent_recall_subscription_wake_claim,
    copy_agent_work_context,
    copy_agent_work_context_publication_receipt,
    validate_agent_recall_checkpoint_advance,
    validate_agent_recall_checkpoint_work_context,
    validate_agent_recall_subscription_publication,
    validate_agent_work_context_publication,
)


async def _lock_agent_work_context_identity(
    cur: Any,
    category: str,
    identity: str,
) -> None:
    """Serialize one work-context identity without exposing it as a lock key."""

    await cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"cayu-agent-work-context:{category}:{identity}",),
    )


async def _lock_agent_work_context_identity_shared(
    cur: Any,
    category: str,
    identity: str,
) -> None:
    """Share one work-context identity lock with readers that publish progress."""

    await cur.execute(
        "SELECT pg_advisory_xact_lock_shared(hashtextextended(%s, 0))",
        (f"cayu-agent-work-context:{category}:{identity}",),
    )


class PostgresAgentWorkContextStore(postgres_base._PostgresStoreBase, AgentWorkContextStore):
    """PostgreSQL work-context/checkpoint store with transaction-fenced CAS."""

    _min_required_revision = 73

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if pool is not None:
            _require_quiescent_postgres_mutation_pool(
                pool,
                boundary_name="work-context",
            )
        self._clock = utc_clock(clock)
        self._clock_is_injected = clock is not None
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
        )
        self._postgres_mutation_allowed_configure = (
            None if pool is not None else postgres_base._configure_store_connection
        )
        _require_quiescent_postgres_mutation_pool(
            self._pool,
            allowed_configure=self._postgres_mutation_allowed_configure,
            boundary_name="work-context",
        )

    async def _ensure_ready(self) -> None:
        _require_quiescent_postgres_mutation_pool(
            self._pool,
            allowed_configure=self._postgres_mutation_allowed_configure,
            boundary_name="work-context",
        )
        await super()._ensure_ready()

    @asynccontextmanager
    async def _mutation_cursor(self) -> AsyncIterator[Any]:
        async with PostgresTimingScope(self._pool.connection(), raw=True) as conn:
            _require_quiescent_postgres_mutation_connection(
                conn,
                boundary_name="work-context",
            )
            async with timed_postgres_connection(conn).cursor() as cur:
                yield cur

    async def publish_work_context(
        self,
        context: AgentWorkContext,
        *,
        expected_revision: int | None,
    ) -> AgentWorkContextPublicationReceipt:
        context = copy_agent_work_context(context)
        request_sha256 = agent_work_context_publication_request_sha256(
            context,
            expected_revision,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "publication-operation",
                context.operation_id,
            )
            replay = await self._load_publication(cur, context.operation_id)
            if replay is not None:
                if replay.request_sha256 != request_sha256:
                    raise AgentWorkContextConflict("publication_operation_reused")
                return copy_agent_work_context_publication_receipt(replay)
            await _lock_agent_work_context_identity(cur, "task", context.task_id)
            current = await self._load_context(cur, context.task_id, revision=None)
            validate_agent_work_context_publication(
                context,
                expected_revision,
                current,
            )
            changed = current is None or current.content_sha256 != context.content_sha256
            result = context if changed else current
            assert result is not None
            receipt = AgentWorkContextPublicationReceipt(
                operation_id=context.operation_id,
                request_sha256=request_sha256,
                expected_revision=expected_revision,
                requested_content_sha256=context.content_sha256,
                changed=changed,
                context=result,
                committed_at=self._clock(),
            )
            if changed:
                await cur.execute(
                    """
                    INSERT INTO cayu_agent_work_context_revisions (
                        task_id, revision, content_sha256, operation_id,
                        record_json, published_at
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        context.task_id,
                        context.revision,
                        context.content_sha256,
                        context.operation_id,
                        pg_support._dumps(context.model_dump(mode="json")),
                        context.published_at,
                    ),
                )
                if expected_revision is None:
                    await cur.execute(
                        """
                        INSERT INTO cayu_agent_work_context_heads (
                            task_id, current_revision
                        ) VALUES (%s, %s)
                        """,
                        (context.task_id, context.revision),
                    )
                else:
                    await cur.execute(
                        """
                        UPDATE cayu_agent_work_context_heads
                        SET current_revision = %s
                        WHERE task_id = %s AND current_revision = %s
                        """,
                        (
                            context.revision,
                            context.task_id,
                            expected_revision,
                        ),
                    )
                    if cur.rowcount != 1:
                        raise AgentWorkContextConflict("stale_context_revision")
            await cur.execute(
                """
                INSERT INTO cayu_agent_work_context_publications (
                    operation_id, task_id, request_sha256, context_revision,
                    changed, receipt_json, committed_at
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    receipt.operation_id,
                    result.task_id,
                    receipt.request_sha256,
                    result.revision,
                    receipt.changed,
                    pg_support._dumps(receipt.model_dump(mode="json")),
                    receipt.committed_at,
                ),
            )
            return copy_agent_work_context_publication_receipt(receipt)

    async def load_work_context(
        self,
        task_id: str,
        *,
        revision: int | None = None,
    ) -> AgentWorkContext | None:
        task_id = _bounded_identity(task_id, "task_id")
        if revision is not None:
            _positive_revision(revision, "revision")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            context = await self._load_context(cur, task_id, revision=revision)
            return None if context is None else copy_agent_work_context(context)

    async def load_work_context_publication(
        self,
        operation_id: str,
    ) -> AgentWorkContextPublicationReceipt | None:
        operation_id = _bounded_identity(operation_id, "operation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            receipt = await self._load_publication(cur, operation_id)
            return None if receipt is None else copy_agent_work_context_publication_receipt(receipt)

    async def advance_recall_checkpoint(
        self,
        checkpoint: AgentRecallCheckpoint,
        *,
        expected_revision: int | None,
    ) -> AgentRecallCheckpoint:
        if expected_revision is not None:
            _positive_revision(expected_revision, "expected_revision")
        checkpoint = copy_agent_recall_checkpoint(checkpoint)
        key = checkpoint.key()
        key_identity = key.fingerprint()
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "recall-processing-operation",
                checkpoint.operation_id,
            )
            await cur.execute(
                """
                SELECT 1 FROM cayu_agent_recall_deliveries WHERE operation_id = %s
                UNION ALL
                SELECT 1 FROM cayu_agent_recall_subscription_evaluations
                WHERE processing_operation_id = %s
                LIMIT 1
                """,
                (checkpoint.operation_id, checkpoint.operation_id),
            )
            if await cur.fetchone() is not None:
                raise AgentWorkContextConflict("checkpoint_operation_reused")
            replay = await self._load_checkpoint_by_operation(
                cur,
                checkpoint.operation_id,
            )
            if replay is not None:
                replay_expected_revision = None if replay.revision == 1 else replay.revision - 1
                if replay != checkpoint or expected_revision != replay_expected_revision:
                    raise AgentWorkContextConflict("checkpoint_operation_reused")
                return copy_agent_recall_checkpoint(replay)
            await _lock_agent_work_context_identity(cur, "checkpoint", key_identity)
            await _lock_agent_work_context_identity_shared(
                cur,
                "task",
                checkpoint.task_id,
            )
            await self._advance_checkpoint_locked(
                cur,
                checkpoint,
                expected_revision=expected_revision,
            )
            return copy_agent_recall_checkpoint(checkpoint)

    async def load_recall_checkpoint(
        self,
        key: AgentRecallCheckpointKey,
        *,
        revision: int | None = None,
    ) -> AgentRecallCheckpoint | None:
        key = copy_agent_recall_checkpoint_key(key)
        if revision is not None:
            _positive_revision(revision, "revision")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            checkpoint = await self._load_checkpoint(cur, key, revision=revision)
            return None if checkpoint is None else copy_agent_recall_checkpoint(checkpoint)

    async def stage_recall_delivery(
        self,
        delivery: AgentRecallDelivery,
    ) -> AgentRecallDeliveryRecord:
        delivery = copy_agent_recall_delivery(delivery)
        key = delivery.key()
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "recall-processing-operation",
                delivery.operation_id,
            )
            await cur.execute(
                "SELECT 1 FROM cayu_agent_recall_subscription_evaluations "
                "WHERE processing_operation_id = %s",
                (delivery.operation_id,),
            )
            if await cur.fetchone() is not None:
                raise AgentRecallDeliveryConflict("delivery_operation_reused")
            await _lock_agent_work_context_identity(cur, "delivery", delivery.delivery_id)
            existing = await self._load_delivery(cur, delivery.delivery_id)
            if existing is not None:
                if existing.delivery != delivery:
                    raise AgentRecallDeliveryConflict("delivery_id_reused")
                return copy_agent_recall_delivery_record(existing)
            await _lock_agent_work_context_identity(cur, "checkpoint", key.fingerprint())
            await cur.execute(
                """
                SELECT delivery_id
                FROM cayu_agent_recall_deliveries
                WHERE agent_id = %s AND task_id = %s
                  AND knowledge_namespace = %s AND access_policy_sha256 = %s
                  AND checkpoint_stream_id = %s
                  AND checkpoint_revision = %s
                """,
                (*key.sort_key(), delivery.checkpoint.revision),
            )
            if await cur.fetchone() is not None:
                raise AgentRecallDeliveryConflict("checkpoint_delivery_exists")
            operation_delivery = await self._load_delivery_by_operation(
                cur,
                delivery.operation_id,
            )
            if operation_delivery is not None:
                raise AgentRecallDeliveryConflict("delivery_operation_reused")
            checkpoint_replay = await self._load_checkpoint_by_operation(
                cur,
                delivery.operation_id,
            )
            if checkpoint_replay is not None:
                raise AgentRecallDeliveryConflict("checkpoint_committed_without_delivery")
            if delivery.staged_at > await self._authoritative_delivery_now(cur):
                raise AgentRecallDeliveryConflict("delivery_staged_in_future")
            await _lock_agent_work_context_identity_shared(cur, "task", delivery.task_id)
            await self._advance_checkpoint_locked(
                cur,
                delivery.checkpoint,
                expected_revision=delivery.expected_checkpoint_revision,
            )
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_deliveries (
                    delivery_id, operation_id, agent_id, task_id,
                    knowledge_namespace, access_policy_sha256,
                    checkpoint_stream_id,
                    checkpoint_revision, processing_result_sha256,
                    delivery_json, staged_at, processing_schema_version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s::jsonb, %s, %s
                )
                """,
                (
                    delivery.delivery_id,
                    delivery.operation_id,
                    delivery.agent_id,
                    delivery.task_id,
                    delivery.knowledge_namespace,
                    delivery.access_policy_sha256,
                    delivery.checkpoint.checkpoint_stream_id,
                    delivery.checkpoint.revision,
                    delivery.processing_result_sha256,
                    pg_support._dumps(delivery.model_dump(mode="json")),
                    delivery.staged_at,
                    str(delivery.processing_result["schema_version"]),
                ),
            )
            record = AgentRecallDeliveryRecord(
                delivery=delivery,
                updated_at=delivery.staged_at,
            )
            await self._insert_delivery_state(cur, record)
            return copy_agent_recall_delivery_record(record)

    async def load_recall_delivery(
        self,
        delivery_id: str,
    ) -> AgentRecallDeliveryRecord | None:
        delivery_id = _bounded_identity(delivery_id, "delivery_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            record = await self._load_delivery(cur, delivery_id)
            return None if record is None else copy_agent_recall_delivery_record(record)

    async def claim_recall_delivery(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> AgentRecallDeliveryRecord | None:
        key = copy_agent_recall_checkpoint_key(key)
        request_sha256 = agent_recall_delivery_claim_request_sha256(
            key,
            claim_id=claim_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(cur, "delivery-claim", claim_id)
            await cur.execute(
                "SELECT delivery_id, worker_id, request_sha256, attempt "
                "FROM cayu_agent_recall_delivery_claims WHERE claim_id = %s",
                (claim_id,),
            )
            replay = await cur.fetchone()
            if replay is not None:
                if str(replay[2]) != request_sha256:
                    raise AgentRecallDeliveryConflict("claim_id_reused")
                record = await self._load_delivery(cur, str(replay[0]))
                if record is None:  # pragma: no cover - protected by foreign key
                    raise RuntimeError("Postgres recall-delivery claim lost its delivery.")
                _require_replayable_delivery_claim_attempt(
                    record,
                    claim_id=claim_id,
                    worker_id=str(replay[1]),
                    attempt=int(replay[3]),
                    now=await self._delivery_now(cur, record.updated_at),
                )
                return copy_agent_recall_delivery_record(record)
            await _lock_agent_work_context_identity(cur, "checkpoint", key.fingerprint())
            await cur.execute(
                """
                SELECT delivery.delivery_json::text,
                       delivery.operation_id,
                       delivery.processing_result_sha256,
                       delivery.staged_at,
                       checkpoint.record_json::text,
                       state.delivery_id, state.agent_id, state.task_id,
                       state.knowledge_namespace, state.access_policy_sha256,
                       state.checkpoint_stream_id, state.checkpoint_revision,
                       state.state, state.attempt,
                       state.state_revision, state.lease_expires_at,
                       state.release_id, state.acknowledgement_id,
                       state.state_json::text, state.updated_at,
                       delivery.processing_schema_version
                FROM cayu_agent_recall_delivery_states AS state
                JOIN cayu_agent_recall_deliveries AS delivery
                  ON delivery.delivery_id = state.delivery_id
                LEFT JOIN cayu_agent_recall_checkpoints AS checkpoint
                  ON checkpoint.agent_id = delivery.agent_id
                 AND checkpoint.task_id = delivery.task_id
                 AND checkpoint.knowledge_namespace = delivery.knowledge_namespace
                 AND checkpoint.access_policy_sha256 = delivery.access_policy_sha256
                 AND checkpoint.checkpoint_stream_id = delivery.checkpoint_stream_id
                 AND checkpoint.revision = delivery.checkpoint_revision
                 AND checkpoint.operation_id = delivery.operation_id
                WHERE state.agent_id = %s AND state.task_id = %s
                  AND state.knowledge_namespace = %s
                  AND state.access_policy_sha256 = %s
                  AND state.checkpoint_stream_id = %s
                  AND state.state != 'acknowledged'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM cayu_agent_recall_subscription_wake_states AS wake_state
                    WHERE wake_state.delivery_id = state.delivery_id
                      AND wake_state.state != 'acknowledged'
                  )
                ORDER BY state.checkpoint_revision, state.delivery_id COLLATE "C"
                LIMIT 1
                FOR UPDATE OF state
                """,
                key.sort_key(),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            current = self._delivery_record_from_row(row)
            now = await self._delivery_now(cur, current.updated_at)
            if (
                current.state is AgentRecallDeliveryState.CLAIMED
                and current.claim is not None
                and current.claim.lease_expires_at > now
            ):
                return None
            claimed = _claim_agent_recall_delivery_record(
                current,
                claim_id=claim_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                now=now,
            )
            assert claimed.claim is not None
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_delivery_claims (
                    claim_id, delivery_id, worker_id, request_sha256,
                    attempt, claimed_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    claimed.claim.claim_id,
                    claimed.delivery.delivery_id,
                    claimed.claim.worker_id,
                    request_sha256,
                    claimed.claim.attempt,
                    claimed.claim.claimed_at,
                ),
            )
            await self._update_delivery_state(cur, current, claimed)
            return copy_agent_recall_delivery_record(claimed)

    async def renew_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallDeliveryRecord:
        claim = copy_agent_recall_delivery_claim(claim)
        _validate_delivery_lease_seconds(lease_seconds)
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            current = await self._load_delivery(cur, claim.delivery_id, for_update=True)
            if current is None:
                raise AgentRecallDeliveryConflict("unknown_delivery")
            renewed = _renew_agent_recall_delivery_record(
                current,
                claim,
                lease_seconds=lease_seconds,
                now=await self._delivery_now(cur, current.updated_at),
            )
            if renewed != current:
                await self._update_delivery_state(cur, current, renewed)
            return copy_agent_recall_delivery_record(renewed)

    async def release_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallDeliveryRecord:
        claim = copy_agent_recall_delivery_claim(claim)
        requested = _agent_recall_delivery_release(
            claim,
            release_id=release_id,
            reason=reason,
            released_at=released_at,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(cur, "delivery-release", release_id)
            current = await self._load_delivery(cur, claim.delivery_id, for_update=True)
            if current is None:
                raise AgentRecallDeliveryConflict("unknown_delivery")
            await cur.execute(
                "SELECT delivery_id, release_json::text "
                "FROM cayu_agent_recall_delivery_releases WHERE release_id = %s",
                (requested.release_id,),
            )
            replay = await cur.fetchone()
            if replay is not None:
                stored = AgentRecallDeliveryRelease.model_validate_json(str(replay[1]))
                if str(replay[0]) != claim.delivery_id or stored != requested:
                    raise AgentRecallDeliveryConflict("release_id_reused")
                if current.release != stored:
                    raise AgentRecallDeliveryConflict("release_replay_superseded")
                return copy_agent_recall_delivery_record(current)
            released = _release_agent_recall_delivery_record(
                current,
                claim,
                release_id=requested.release_id,
                reason=requested.reason,
                released_at=requested.released_at,
                now=await self._delivery_now(cur, current.updated_at),
            )
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_delivery_releases (
                    release_id, delivery_id, claim_id, request_sha256,
                    release_json, released_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    requested.release_id,
                    requested.delivery_id,
                    requested.claim_id,
                    requested.fingerprint(),
                    pg_support._dumps(requested.model_dump(mode="json")),
                    requested.released_at,
                ),
            )
            await self._update_delivery_state(cur, current, released)
            return copy_agent_recall_delivery_record(released)

    async def acknowledge_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        acknowledgement_id: str,
        evidence_kind: AgentRecallDeliveryEvidenceKind,
        evidence_ref: str,
        acknowledged_at: datetime,
    ) -> AgentRecallDeliveryRecord:
        claim = copy_agent_recall_delivery_claim(claim)
        acknowledgement_id = _bounded_identity(acknowledgement_id, "acknowledgement_id")
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "delivery-acknowledgement",
                acknowledgement_id,
            )
            current = await self._load_delivery(cur, claim.delivery_id, for_update=True)
            if current is None:
                raise AgentRecallDeliveryConflict("unknown_delivery")
            await cur.execute(
                "SELECT delivery_id FROM cayu_agent_recall_delivery_states "
                "WHERE acknowledgement_id = %s",
                (acknowledgement_id,),
            )
            occupied = await cur.fetchone()
            if occupied is not None and str(occupied[0]) != claim.delivery_id:
                raise AgentRecallDeliveryConflict("acknowledgement_reused")
            acknowledged = _acknowledge_agent_recall_delivery_record(
                current,
                claim,
                acknowledgement_id=acknowledgement_id,
                evidence_kind=evidence_kind,
                evidence_ref=evidence_ref,
                acknowledged_at=acknowledged_at,
                now=await self._delivery_now(cur, current.updated_at),
            )
            if acknowledged != current:
                await self._update_delivery_state(cur, current, acknowledged)
            return copy_agent_recall_delivery_record(acknowledged)

    async def publish_recall_subscription(
        self,
        subscription: AgentRecallSubscription,
        *,
        expected_revision: int | None,
    ) -> AgentRecallSubscriptionPublicationReceipt:
        subscription = copy_agent_recall_subscription(subscription)
        request_sha256 = agent_recall_subscription_publication_request_sha256(
            subscription,
            expected_revision,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "subscription-publication-operation",
                subscription.operation_id,
            )
            replay = await self._load_subscription_publication(
                cur,
                subscription.operation_id,
            )
            if replay is not None:
                if replay.request_sha256 != request_sha256:
                    raise AgentRecallSubscriptionConflict("publication_operation_reused")
                return copy_agent_recall_subscription_publication_receipt(replay)
            await _lock_agent_work_context_identity(
                cur,
                "subscription",
                subscription.subscription_id,
            )
            await _lock_agent_work_context_identity_shared(
                cur,
                "task",
                subscription.task_id,
            )
            current = await self._load_subscription(
                cur,
                subscription.subscription_id,
                revision=None,
            )
            work_context = await self._load_context(
                cur,
                subscription.task_id,
                revision=None,
            )
            now = await self._authoritative_delivery_now(cur)
            if subscription.published_at > now:
                raise AgentRecallSubscriptionConflict("publication_from_future")
            validate_agent_recall_subscription_publication(
                subscription,
                expected_revision,
                current,
                work_context,
            )
            receipt = AgentRecallSubscriptionPublicationReceipt(
                operation_id=subscription.operation_id,
                request_sha256=request_sha256,
                expected_revision=expected_revision,
                subscription=subscription,
                committed_at=now,
            )
            prior_state = await self._load_subscription_state(
                cur,
                subscription.subscription_id,
                for_update=True,
            )
            state = AgentRecallSubscriptionRecord(
                subscription=subscription,
                state_revision=(0 if prior_state is None else prior_state.state_revision + 1),
                attempt=0 if prior_state is None else prior_state.attempt,
                next_evaluation_at=max(now, subscription.published_at),
                updated_at=now,
            )
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_subscription_revisions (
                    subscription_id, revision, operation_id, agent_id, task_id,
                    knowledge_namespace, access_policy_sha256,
                    work_context_revision, work_context_sha256, status, priority,
                    subscription_json, expires_at, published_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s::jsonb, %s, %s
                )
                """,
                (
                    subscription.subscription_id,
                    subscription.revision,
                    subscription.operation_id,
                    subscription.agent_id,
                    subscription.task_id,
                    subscription.knowledge_namespace,
                    subscription.access_policy_sha256,
                    subscription.work_context_revision,
                    subscription.work_context_sha256,
                    subscription.status.value,
                    subscription.priority,
                    pg_support._dumps(subscription.model_dump(mode="json")),
                    subscription.expires_at,
                    subscription.published_at,
                ),
            )
            if current is None:
                await cur.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_heads (
                        subscription_id, current_revision
                    ) VALUES (%s, %s)
                    """,
                    (subscription.subscription_id, subscription.revision),
                )
                await self._insert_subscription_state(cur, state)
            else:
                await cur.execute(
                    """
                    UPDATE cayu_agent_recall_subscription_heads
                    SET current_revision = %s
                    WHERE subscription_id = %s AND current_revision = %s
                    """,
                    (
                        subscription.revision,
                        subscription.subscription_id,
                        expected_revision,
                    ),
                )
                if cur.rowcount != 1:
                    raise AgentRecallSubscriptionConflict("stale_subscription_revision")
                if prior_state is None:  # pragma: no cover - foreign key invariant
                    raise RuntimeError("Postgres subscription head lost its state.")
                await self._update_subscription_state(cur, prior_state, state)
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_subscription_publications (
                    operation_id, subscription_id, subscription_revision,
                    request_sha256, receipt_json, committed_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    receipt.operation_id,
                    subscription.subscription_id,
                    subscription.revision,
                    request_sha256,
                    pg_support._dumps(receipt.model_dump(mode="json")),
                    receipt.committed_at,
                ),
            )
            return copy_agent_recall_subscription_publication_receipt(receipt)

    async def load_recall_subscription(
        self,
        subscription_id: str,
        *,
        revision: int | None = None,
    ) -> AgentRecallSubscription | None:
        subscription_id = _bounded_identity(subscription_id, "subscription_id")
        if revision is not None:
            _positive_revision(revision, "revision")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            subscription = await self._load_subscription(
                cur,
                subscription_id,
                revision=revision,
            )
            return None if subscription is None else copy_agent_recall_subscription(subscription)

    async def claim_due_recall_subscription(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        runner_id: str,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionRecord | None:
        key = copy_agent_recall_checkpoint_key(key)
        request_sha256 = agent_recall_subscription_claim_request_sha256(
            key,
            claim_id=claim_id,
            runner_id=runner_id,
            lease_seconds=lease_seconds,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "subscription-claim",
                claim_id,
            )
            await cur.execute(
                "SELECT subscription_id, runner_id, request_sha256, attempt "
                "FROM cayu_agent_recall_subscription_claims WHERE claim_id = %s",
                (claim_id,),
            )
            replay = await cur.fetchone()
            if replay is not None:
                if str(replay[2]) != request_sha256:
                    raise AgentRecallSubscriptionConflict("claim_id_reused")
                record = await self._load_subscription_state(cur, str(replay[0]))
                if record is None:  # pragma: no cover - foreign key invariant
                    raise RuntimeError("Postgres subscription claim lost its state.")
                current_claim = record.claim
                now = await self._delivery_now(cur, record.updated_at)
                if (
                    record.run_state is not AgentRecallSubscriptionRunState.CLAIMED
                    or current_claim is None
                    or current_claim.claim_id != claim_id
                    or current_claim.runner_id != str(replay[1])
                    or current_claim.attempt != int(replay[3])
                ):
                    raise AgentRecallSubscriptionConflict("claim_replay_superseded")
                if current_claim.lease_expires_at <= now:
                    raise AgentRecallSubscriptionConflict("expired_subscription_claim")
                return copy_agent_recall_subscription_record(record)
            now = await self._authoritative_delivery_now(cur)
            await cur.execute(
                """
                SELECT revision.subscription_json::text,
                       state.subscription_id, state.current_revision,
                       state.agent_id, state.task_id,
                       state.knowledge_namespace, state.access_policy_sha256,
                       state.run_state, state.attempt, state.state_revision,
                       state.lease_expires_at, state.release_id,
                       state.next_evaluation_at, state.last_evaluation_id,
                       state.state_json::text, state.updated_at
                FROM cayu_agent_recall_subscription_states AS state
                JOIN cayu_agent_recall_subscription_revisions AS revision
                  ON revision.subscription_id = state.subscription_id
                 AND revision.revision = state.current_revision
                JOIN cayu_agent_work_context_heads AS context_head
                  ON context_head.task_id = revision.task_id
                 AND context_head.current_revision = revision.work_context_revision
                JOIN cayu_agent_work_context_revisions AS context_revision
                  ON context_revision.task_id = context_head.task_id
                 AND context_revision.revision = context_head.current_revision
                 AND context_revision.content_sha256 = revision.work_context_sha256
                WHERE state.agent_id = %s AND state.task_id = %s
                  AND state.knowledge_namespace = %s
                  AND state.access_policy_sha256 = %s
                  AND revision.status = 'active'
                  AND revision.expires_at > %s
                  AND state.next_evaluation_at <= %s
                  AND (
                    state.run_state = 'due'
                    OR state.lease_expires_at <= %s
                  )
                  AND NOT EXISTS (
                    SELECT 1
                    FROM cayu_agent_recall_subscription_evaluations AS evaluation
                    JOIN cayu_agent_recall_subscription_wake_states AS wake_state
                      ON wake_state.wake_id = evaluation.evaluation_id
                    WHERE evaluation.subscription_id = state.subscription_id
                      AND wake_state.state != 'acknowledged'
                  )
                ORDER BY state.next_evaluation_at,
                         revision.priority DESC,
                         state.subscription_id COLLATE "C"
                LIMIT 1
                FOR UPDATE OF state SKIP LOCKED
                """,
                (*key.authority_sort_key(), now, now, now),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            current = self._subscription_record_from_row(row)
            claimed = _claim_agent_recall_subscription_record(
                current,
                claim_id=claim_id,
                runner_id=runner_id,
                lease_seconds=lease_seconds,
                now=max(now, current.updated_at),
            )
            assert claimed.claim is not None
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_subscription_claims (
                    claim_id, subscription_id, subscription_revision,
                    runner_id, request_sha256, attempt, claimed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    claimed.claim.claim_id,
                    claimed.subscription.subscription_id,
                    claimed.subscription.revision,
                    claimed.claim.runner_id,
                    request_sha256,
                    claimed.claim.attempt,
                    claimed.claim.claimed_at,
                ),
            )
            await self._update_subscription_state(cur, current, claimed)
            return copy_agent_recall_subscription_record(claimed)

    async def renew_recall_subscription(
        self,
        claim: AgentRecallSubscriptionClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionRecord:
        claim = copy_agent_recall_subscription_claim(claim)
        _validate_delivery_lease_seconds(lease_seconds)
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            current = await self._load_subscription_state(
                cur,
                claim.subscription_id,
                for_update=True,
            )
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_subscription")
            renewed = _renew_agent_recall_subscription_record(
                current,
                claim,
                lease_seconds=lease_seconds,
                now=await self._delivery_now(cur, current.updated_at),
            )
            if renewed != current:
                await self._update_subscription_state(cur, current, renewed)
            return copy_agent_recall_subscription_record(renewed)

    async def release_recall_subscription(
        self,
        claim: AgentRecallSubscriptionClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallSubscriptionRecord:
        claim = copy_agent_recall_subscription_claim(claim)
        requested = _agent_recall_subscription_release(
            claim,
            release_id=release_id,
            reason=reason,
            released_at=released_at,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "subscription-release",
                release_id,
            )
            current = await self._load_subscription_state(
                cur,
                claim.subscription_id,
                for_update=True,
            )
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_subscription")
            await cur.execute(
                "SELECT subscription_id, release_json::text "
                "FROM cayu_agent_recall_subscription_releases WHERE release_id = %s",
                (requested.release_id,),
            )
            replay = await cur.fetchone()
            if replay is not None:
                stored = AgentRecallSubscriptionRelease.model_validate_json(str(replay[1]))
                if str(replay[0]) != claim.subscription_id or stored != requested:
                    raise AgentRecallSubscriptionConflict("release_id_reused")
                if current.release != stored:
                    raise AgentRecallSubscriptionConflict("release_replay_superseded")
                return copy_agent_recall_subscription_record(current)
            released = _release_agent_recall_subscription_record(
                current,
                claim,
                release_id=requested.release_id,
                reason=requested.reason,
                released_at=requested.released_at,
                now=await self._delivery_now(cur, current.updated_at),
            )
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_subscription_releases (
                    release_id, subscription_id, claim_id, request_sha256,
                    release_json, released_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    requested.release_id,
                    requested.subscription_id,
                    requested.claim_id,
                    requested.fingerprint(),
                    pg_support._dumps(requested.model_dump(mode="json")),
                    requested.released_at,
                ),
            )
            await self._update_subscription_state(cur, current, released)
            return copy_agent_recall_subscription_record(released)

    async def commit_recall_subscription_evaluation(
        self,
        claim: AgentRecallSubscriptionClaim,
        result,
        *,
        evaluation_id: str,
        delivery_id: str | None,
        staged_by: str,
        evaluated_at: datetime,
    ) -> AgentRecallSubscriptionEvaluation:
        from cayu.memory.processing import AgentRecallProcessingResult

        claim = copy_agent_recall_subscription_claim(claim)
        if type(result) is not AgentRecallProcessingResult:
            raise TypeError("result must be an AgentRecallProcessingResult.")
        request_sha256 = agent_recall_subscription_evaluation_request_sha256(
            claim,
            result,
            evaluation_id=evaluation_id,
            delivery_id=delivery_id,
            staged_by=staged_by,
            evaluated_at=evaluated_at,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "subscription-evaluation",
                evaluation_id,
            )
            replay = await self._load_subscription_evaluation(cur, evaluation_id)
            if replay is not None:
                if replay.request_sha256 != request_sha256:
                    raise AgentRecallSubscriptionConflict("evaluation_id_reused")
                return copy_agent_recall_subscription_evaluation(replay)
            await _lock_agent_work_context_identity(
                cur,
                "recall-processing-operation",
                result.operation_id,
            )
            await cur.execute(
                "SELECT evaluation_id "
                "FROM cayu_agent_recall_subscription_evaluations "
                "WHERE processing_operation_id = %s",
                (result.operation_id,),
            )
            if await cur.fetchone() is not None:
                raise AgentRecallSubscriptionConflict("processing_operation_reused")
            await cur.execute(
                """
                SELECT 1 FROM cayu_agent_recall_checkpoints WHERE operation_id = %s
                UNION ALL
                SELECT 1 FROM cayu_agent_recall_deliveries WHERE operation_id = %s
                LIMIT 1
                """,
                (result.operation_id, result.operation_id),
            )
            if await cur.fetchone() is not None:
                raise AgentRecallSubscriptionConflict("processing_operation_reused")
            checkpoint = result.proposed_checkpoint
            if checkpoint is not None:
                await _lock_agent_work_context_identity(
                    cur,
                    "recall-processing-operation",
                    checkpoint.operation_id,
                )
                if delivery_id is not None:
                    await _lock_agent_work_context_identity(
                        cur,
                        "delivery",
                        delivery_id,
                    )
                await _lock_agent_work_context_identity(
                    cur,
                    "checkpoint",
                    checkpoint.key().fingerprint(),
                )
            current = await self._load_subscription_state(
                cur,
                claim.subscription_id,
                for_update=True,
            )
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_subscription")
            await _lock_agent_work_context_identity_shared(
                cur,
                "task",
                current.subscription.task_id,
            )
            current_work_context = await self._load_context(
                cur,
                current.subscription.task_id,
                revision=None,
            )
            now = await self._delivery_now(cur, current.updated_at)
            evaluation, delivery, updated = _prepare_agent_recall_subscription_evaluation(
                current,
                claim,
                result,
                current_work_context,
                evaluation_id=evaluation_id,
                delivery_id=delivery_id,
                staged_by=staged_by,
                evaluated_at=evaluated_at,
                now=now,
            )
            if delivery is not None:
                await self._insert_subscription_delivery_locked(cur, delivery)
            elif checkpoint is not None:
                await self._advance_checkpoint_locked(
                    cur,
                    checkpoint,
                    expected_revision=(
                        None if checkpoint.revision == 1 else checkpoint.revision - 1
                    ),
                )
            subscription = current.subscription
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_subscription_evaluations (
                    evaluation_id, subscription_id, subscription_revision,
                    agent_id, task_id, knowledge_namespace,
                    access_policy_sha256, claim_id, processing_operation_id,
                    request_sha256, outcome, delivery_id,
                    evaluation_json, committed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s::jsonb, %s
                )
                """,
                (
                    evaluation.evaluation_id,
                    evaluation.subscription_id,
                    evaluation.subscription_revision,
                    subscription.agent_id,
                    subscription.task_id,
                    subscription.knowledge_namespace,
                    subscription.access_policy_sha256,
                    evaluation.claim_id,
                    evaluation.processing_operation_id,
                    evaluation.request_sha256,
                    evaluation.outcome.value,
                    evaluation.delivery_id,
                    pg_support._dumps(evaluation.model_dump(mode="json")),
                    evaluation.committed_at,
                ),
            )
            if delivery is not None:
                await self._insert_subscription_wake_state(
                    cur,
                    AgentRecallSubscriptionWake(
                        wake_id=evaluation.evaluation_id,
                        subscription=subscription,
                        evaluation=evaluation,
                        delivery=delivery,
                        updated_at=evaluation.committed_at,
                    ),
                )
            await self._update_subscription_state(cur, current, updated)
            return copy_agent_recall_subscription_evaluation(evaluation)

    async def load_recall_subscription_evaluation(
        self,
        evaluation_id: str,
    ) -> AgentRecallSubscriptionEvaluation | None:
        evaluation_id = _bounded_identity(evaluation_id, "evaluation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            evaluation = await self._load_subscription_evaluation(cur, evaluation_id)
            return (
                None
                if evaluation is None
                else copy_agent_recall_subscription_evaluation(evaluation)
            )

    async def claim_recall_subscription_wake(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        runner_id: str,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionWake | None:
        key = copy_agent_recall_checkpoint_key(key)
        request_sha256 = agent_recall_subscription_wake_claim_request_sha256(
            key,
            claim_id=claim_id,
            runner_id=runner_id,
            lease_seconds=lease_seconds,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(cur, "subscription-wake-claim", claim_id)
            await cur.execute(
                "SELECT wake_id, runner_id, request_sha256, attempt "
                "FROM cayu_agent_recall_subscription_wake_claims WHERE claim_id = %s",
                (claim_id,),
            )
            replay = await cur.fetchone()
            if replay is not None:
                if str(replay[2]) != request_sha256:
                    raise AgentRecallSubscriptionConflict("wake_claim_id_reused")
                wake = await self._load_subscription_wake(cur, str(replay[0]))
                if wake is None:  # pragma: no cover - foreign key invariant
                    raise RuntimeError("Postgres subscription-wake claim lost its state.")
                _require_replayable_subscription_wake_claim(
                    wake,
                    claim_id=claim_id,
                    runner_id=str(replay[1]),
                    attempt=int(replay[3]),
                    now=await self._delivery_now(cur, wake.updated_at),
                )
                return copy_agent_recall_subscription_wake(wake)
            now = await self._authoritative_delivery_now(cur)
            await cur.execute(
                """
                SELECT revision.subscription_json::text,
                       evaluation.evaluation_json::text,
                       delivery.delivery_json::text,
                       wake_state.wake_id, wake_state.delivery_id,
                       wake_state.agent_id, wake_state.task_id,
                       wake_state.knowledge_namespace,
                       wake_state.access_policy_sha256, wake_state.state,
                       wake_state.attempt, wake_state.state_revision,
                       wake_state.claim_id, wake_state.lease_expires_at,
                       wake_state.release_id, wake_state.acknowledgement_id,
                       wake_state.state_json::text, wake_state.committed_at,
                       wake_state.updated_at
                FROM cayu_agent_recall_subscription_wake_states AS wake_state
                JOIN cayu_agent_recall_subscription_evaluations AS evaluation
                  ON evaluation.evaluation_id = wake_state.wake_id
                JOIN cayu_agent_recall_subscription_revisions AS revision
                  ON revision.subscription_id = evaluation.subscription_id
                 AND revision.revision = evaluation.subscription_revision
                JOIN cayu_agent_recall_deliveries AS delivery
                  ON delivery.delivery_id = wake_state.delivery_id
                WHERE wake_state.agent_id = %s AND wake_state.task_id = %s
                  AND wake_state.knowledge_namespace = %s
                  AND wake_state.access_policy_sha256 = %s
                  AND wake_state.state != 'acknowledged'
                  AND (
                    wake_state.state = 'pending'
                    OR wake_state.lease_expires_at <= %s
                  )
                ORDER BY wake_state.committed_at,
                         wake_state.wake_id COLLATE "C"
                LIMIT 1
                FOR UPDATE OF wake_state SKIP LOCKED
                """,
                (*key.authority_sort_key(), now),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            current = self._subscription_wake_from_row(row)
            now = max(now, current.updated_at)
            claimed = _claim_agent_recall_subscription_wake(
                current,
                claim_id=claim_id,
                runner_id=runner_id,
                lease_seconds=lease_seconds,
                now=now,
            )
            assert claimed.claim is not None
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_subscription_wake_claims (
                    claim_id, wake_id, delivery_id, runner_id, request_sha256,
                    attempt, claimed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    claimed.claim.claim_id,
                    claimed.wake_id,
                    claimed.claim.delivery_id,
                    claimed.claim.runner_id,
                    request_sha256,
                    claimed.claim.attempt,
                    claimed.claim.claimed_at,
                ),
            )
            await self._update_subscription_wake_state(cur, current, claimed)
            return copy_agent_recall_subscription_wake(claimed)

    async def load_recall_subscription_wake(
        self,
        wake_id: str,
    ) -> AgentRecallSubscriptionWake | None:
        wake_id = _bounded_identity(wake_id, "wake_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            wake = await self._load_subscription_wake(cur, wake_id)
            return None if wake is None else copy_agent_recall_subscription_wake(wake)

    async def renew_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionWake:
        claim = copy_agent_recall_subscription_wake_claim(claim)
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            current = await self._load_subscription_wake(
                cur,
                claim.wake_id,
                for_update=True,
            )
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_wake")
            renewed = _renew_agent_recall_subscription_wake(
                current,
                claim,
                lease_seconds=lease_seconds,
                now=await self._delivery_now(cur, current.updated_at),
            )
            if renewed != current:
                await self._update_subscription_wake_state(cur, current, renewed)
            return copy_agent_recall_subscription_wake(renewed)

    async def release_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallSubscriptionWake:
        claim = copy_agent_recall_subscription_wake_claim(claim)
        requested = _agent_recall_subscription_wake_release(
            claim,
            release_id=release_id,
            reason=reason,
            released_at=released_at,
        )
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "subscription-wake-release",
                requested.release_id,
            )
            current = await self._load_subscription_wake(
                cur,
                claim.wake_id,
                for_update=True,
            )
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_wake")
            await cur.execute(
                "SELECT wake_id, release_json::text "
                "FROM cayu_agent_recall_subscription_wake_releases "
                "WHERE release_id = %s",
                (requested.release_id,),
            )
            replay = await cur.fetchone()
            if replay is not None:
                stored = AgentRecallSubscriptionWakeRelease.model_validate_json(str(replay[1]))
                if str(replay[0]) != claim.wake_id or stored != requested:
                    raise AgentRecallSubscriptionConflict("wake_release_id_reused")
                if current.release != stored:
                    raise AgentRecallSubscriptionConflict("wake_release_replay_superseded")
                return copy_agent_recall_subscription_wake(current)
            released = _release_agent_recall_subscription_wake(
                current,
                claim,
                release_id=requested.release_id,
                reason=requested.reason,
                released_at=requested.released_at,
                now=await self._delivery_now(cur, current.updated_at),
            )
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_subscription_wake_releases (
                    release_id, wake_id, claim_id, request_sha256,
                    release_json, released_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    requested.release_id,
                    requested.wake_id,
                    requested.claim_id,
                    requested.fingerprint(),
                    pg_support._dumps(requested.model_dump(mode="json")),
                    requested.released_at,
                ),
            )
            await self._update_subscription_wake_state(cur, current, released)
            return copy_agent_recall_subscription_wake(released)

    async def acknowledge_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        acknowledgement_id: str,
        acknowledged_at: datetime,
    ) -> AgentRecallSubscriptionWake:
        claim = copy_agent_recall_subscription_wake_claim(claim)
        acknowledgement_id = _bounded_identity(acknowledgement_id, "acknowledgement_id")
        await self._ensure_ready()
        async with self._mutation_cursor() as cur:
            await _lock_agent_work_context_identity(
                cur,
                "subscription-wake-acknowledgement",
                acknowledgement_id,
            )
            current = await self._load_subscription_wake(
                cur,
                claim.wake_id,
                for_update=True,
            )
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_wake")
            await cur.execute(
                "SELECT wake_id FROM cayu_agent_recall_subscription_wake_states "
                "WHERE acknowledgement_id = %s",
                (acknowledgement_id,),
            )
            occupied = await cur.fetchone()
            if occupied is not None and str(occupied[0]) != claim.wake_id:
                raise AgentRecallSubscriptionConflict("wake_acknowledgement_id_reused")
            acknowledged = _acknowledge_agent_recall_subscription_wake(
                current,
                claim,
                acknowledgement_id=acknowledgement_id,
                acknowledged_at=acknowledged_at,
                now=await self._delivery_now(cur, current.updated_at),
            )
            if acknowledged != current:
                await self._update_subscription_wake_state(cur, current, acknowledged)
            return copy_agent_recall_subscription_wake(acknowledged)

    async def _advance_checkpoint_locked(
        self,
        cur: Any,
        checkpoint: AgentRecallCheckpoint,
        *,
        expected_revision: int | None,
    ) -> None:
        key = checkpoint.key()
        work_context = await self._load_context(
            cur,
            checkpoint.task_id,
            revision=checkpoint.work_context_revision,
        )
        current_work_context = await self._load_context(
            cur,
            checkpoint.task_id,
            revision=None,
        )
        validate_agent_recall_checkpoint_work_context(
            checkpoint,
            work_context,
            current_work_context,
        )
        current = await self._load_checkpoint(cur, key, revision=None)
        validate_agent_recall_checkpoint_advance(
            checkpoint,
            expected_revision,
            current,
        )
        await cur.execute(
            """
            INSERT INTO cayu_agent_recall_checkpoints (
                agent_id, task_id, knowledge_namespace, access_policy_sha256,
                checkpoint_stream_id, revision,
                work_context_revision, work_context_sha256,
                knowledge_sequence, index_readiness_sequence, processing_mode,
                knowledge_high_water_sequence,
                index_readiness_high_water_sequence,
                processing_id, operation_id, record_json, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            """,
            (
                checkpoint.agent_id,
                checkpoint.task_id,
                checkpoint.knowledge_namespace,
                checkpoint.access_policy_sha256,
                checkpoint.checkpoint_stream_id,
                checkpoint.revision,
                checkpoint.work_context_revision,
                checkpoint.work_context_sha256,
                checkpoint.knowledge_sequence,
                checkpoint.index_readiness_sequence,
                checkpoint.processing_mode.value,
                checkpoint.knowledge_high_water_sequence,
                checkpoint.index_readiness_high_water_sequence,
                checkpoint.processing_id,
                checkpoint.operation_id,
                pg_support._dumps(checkpoint.model_dump(mode="json")),
                checkpoint.updated_at,
            ),
        )
        if current is None:
            await cur.execute(
                """
                INSERT INTO cayu_agent_recall_checkpoint_heads (
                    agent_id, task_id, knowledge_namespace,
                    access_policy_sha256, checkpoint_stream_id, current_revision
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (*key.sort_key(), checkpoint.revision),
            )
            return
        await cur.execute(
            """
            UPDATE cayu_agent_recall_checkpoint_heads
            SET current_revision = %s
            WHERE agent_id = %s AND task_id = %s
              AND knowledge_namespace = %s AND access_policy_sha256 = %s
              AND checkpoint_stream_id = %s
              AND current_revision = %s
            """,
            (
                checkpoint.revision,
                *key.sort_key(),
                expected_revision,
            ),
        )
        if cur.rowcount != 1:
            raise AgentWorkContextConflict("stale_checkpoint_revision")

    @staticmethod
    async def _load_subscription(
        cur: Any,
        subscription_id: str,
        *,
        revision: int | None,
    ) -> AgentRecallSubscription | None:
        if revision is None:
            await cur.execute(
                """
                SELECT revision.subscription_json::text
                FROM cayu_agent_recall_subscription_heads AS head
                JOIN cayu_agent_recall_subscription_revisions AS revision
                  ON revision.subscription_id = head.subscription_id
                 AND revision.revision = head.current_revision
                WHERE head.subscription_id = %s
                """,
                (subscription_id,),
            )
        else:
            await cur.execute(
                "SELECT subscription_json::text "
                "FROM cayu_agent_recall_subscription_revisions "
                "WHERE subscription_id = %s AND revision = %s",
                (subscription_id, revision),
            )
        row = await cur.fetchone()
        return None if row is None else AgentRecallSubscription.model_validate_json(str(row[0]))

    @staticmethod
    async def _load_subscription_publication(
        cur: Any,
        operation_id: str,
    ) -> AgentRecallSubscriptionPublicationReceipt | None:
        await cur.execute(
            "SELECT receipt_json::text "
            "FROM cayu_agent_recall_subscription_publications "
            "WHERE operation_id = %s",
            (operation_id,),
        )
        row = await cur.fetchone()
        return (
            None
            if row is None
            else AgentRecallSubscriptionPublicationReceipt.model_validate_json(str(row[0]))
        )

    @classmethod
    async def _load_subscription_state(
        cls,
        cur: Any,
        subscription_id: str,
        *,
        for_update: bool = False,
    ) -> AgentRecallSubscriptionRecord | None:
        lock_clause = "FOR UPDATE OF state" if for_update else ""
        await cur.execute(
            f"""
            SELECT revision.subscription_json::text,
                   state.subscription_id, state.current_revision,
                   state.agent_id, state.task_id,
                   state.knowledge_namespace, state.access_policy_sha256,
                   state.run_state, state.attempt, state.state_revision,
                   state.lease_expires_at, state.release_id,
                   state.next_evaluation_at, state.last_evaluation_id,
                   state.state_json::text, state.updated_at
            FROM cayu_agent_recall_subscription_states AS state
            JOIN cayu_agent_recall_subscription_revisions AS revision
              ON revision.subscription_id = state.subscription_id
             AND revision.revision = state.current_revision
            WHERE state.subscription_id = %s
            {lock_clause}
            """,
            (subscription_id,),
        )
        row = await cur.fetchone()
        return None if row is None else cls._subscription_record_from_row(row)

    @staticmethod
    def _subscription_record_from_row(
        row: Sequence[Any],
    ) -> AgentRecallSubscriptionRecord:
        subscription = json.loads(str(row[0]))
        state = json.loads(str(row[14]))
        if type(subscription) is not dict or type(state) is not dict:
            raise RuntimeError("Postgres recall-subscription JSON must contain objects.")
        state["subscription"] = subscription
        record = AgentRecallSubscriptionRecord.model_validate_json(pg_support._dumps(state))
        claim = record.claim
        lease_expires_at = (
            claim.lease_expires_at
            if record.run_state is AgentRecallSubscriptionRunState.CLAIMED and claim is not None
            else None
        )
        release_id = None if record.release is None else record.release.release_id
        if (
            str(row[1]) != record.subscription.subscription_id
            or int(row[2]) != record.subscription.revision
            or tuple(str(value) for value in row[3:7])
            != record.subscription.checkpoint_key().authority_sort_key()
            or str(row[7]) != record.run_state.value
            or int(row[8]) != record.attempt
            or int(row[9]) != record.state_revision
            or row[10] != lease_expires_at
            or row[11] != release_id
            or row[12] != record.next_evaluation_at
            or row[13] != record.last_evaluation_id
            or row[15] != record.updated_at
        ):
            raise RuntimeError("Postgres recall-subscription indexes conflict with durable state.")
        return record

    @staticmethod
    async def _load_subscription_evaluation(
        cur: Any,
        evaluation_id: str,
    ) -> AgentRecallSubscriptionEvaluation | None:
        await cur.execute(
            "SELECT evaluation_json::text "
            "FROM cayu_agent_recall_subscription_evaluations "
            "WHERE evaluation_id = %s",
            (evaluation_id,),
        )
        row = await cur.fetchone()
        return (
            None
            if row is None
            else AgentRecallSubscriptionEvaluation.model_validate_json(str(row[0]))
        )

    @classmethod
    async def _load_subscription_wake(
        cls,
        cur: Any,
        wake_id: str,
        *,
        for_update: bool = False,
    ) -> AgentRecallSubscriptionWake | None:
        await cur.execute(
            """
            SELECT revision.subscription_json::text,
                   evaluation.evaluation_json::text,
                   delivery.delivery_json::text,
                   wake_state.wake_id, wake_state.delivery_id,
                   wake_state.agent_id, wake_state.task_id,
                   wake_state.knowledge_namespace,
                   wake_state.access_policy_sha256, wake_state.state,
                   wake_state.attempt, wake_state.state_revision,
                   wake_state.claim_id, wake_state.lease_expires_at,
                   wake_state.release_id, wake_state.acknowledgement_id,
                   wake_state.state_json::text, wake_state.committed_at,
                   wake_state.updated_at
            FROM cayu_agent_recall_subscription_wake_states AS wake_state
            JOIN cayu_agent_recall_subscription_evaluations AS evaluation
              ON evaluation.evaluation_id = wake_state.wake_id
            JOIN cayu_agent_recall_subscription_revisions AS revision
              ON revision.subscription_id = evaluation.subscription_id
             AND revision.revision = evaluation.subscription_revision
            JOIN cayu_agent_recall_deliveries AS delivery
              ON delivery.delivery_id = wake_state.delivery_id
            WHERE wake_state.wake_id = %s
            """
            + (" FOR UPDATE OF wake_state" if for_update else ""),
            (wake_id,),
        )
        row = await cur.fetchone()
        return None if row is None else cls._subscription_wake_from_row(row)

    @staticmethod
    def _subscription_wake_from_row(row: Sequence[Any]) -> AgentRecallSubscriptionWake:
        subscription = json.loads(str(row[0]))
        evaluation = json.loads(str(row[1]))
        delivery = json.loads(str(row[2]))
        state = json.loads(str(row[16]))
        if any(type(value) is not dict for value in (subscription, evaluation, delivery, state)):
            raise RuntimeError("Postgres subscription-wake JSON must contain objects.")
        state.update(
            subscription=subscription,
            evaluation=evaluation,
            delivery=delivery,
        )
        wake = AgentRecallSubscriptionWake.model_validate_json(pg_support._dumps(state))
        claim = wake.claim
        lease_expires_at = (
            claim.lease_expires_at
            if wake.state is AgentRecallSubscriptionWakeState.CLAIMED and claim is not None
            else None
        )
        release_id = None if wake.release is None else wake.release.release_id
        acknowledgement_id = (
            None if wake.acknowledgement is None else wake.acknowledgement.acknowledgement_id
        )
        if (
            str(row[3]) != wake.wake_id
            or str(row[4]) != wake.delivery.delivery_id
            or tuple(str(value) for value in row[5:9])
            != wake.subscription.checkpoint_key().authority_sort_key()
            or str(row[9]) != wake.state.value
            or int(row[10]) != wake.attempt
            or int(row[11]) != wake.state_revision
            or row[12] != (None if claim is None else claim.claim_id)
            or row[13] != lease_expires_at
            or row[14] != release_id
            or row[15] != acknowledgement_id
            or row[17] != wake.evaluation.committed_at
            or row[18] != wake.updated_at
        ):
            raise RuntimeError("Postgres subscription-wake indexes conflict with durable state.")
        return wake

    @classmethod
    async def _insert_subscription_wake_state(
        cls,
        cur: Any,
        wake: AgentRecallSubscriptionWake,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_agent_recall_subscription_wake_states (
                wake_id, delivery_id, agent_id, task_id, knowledge_namespace,
                access_policy_sha256, state, attempt, state_revision, claim_id,
                lease_expires_at, release_id, acknowledgement_id, state_json,
                committed_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s::jsonb, %s, %s
            )
            """,
            cls._subscription_wake_state_values(wake),
        )

    @classmethod
    async def _update_subscription_wake_state(
        cls,
        cur: Any,
        current: AgentRecallSubscriptionWake,
        updated: AgentRecallSubscriptionWake,
    ) -> None:
        await cur.execute(
            """
            UPDATE cayu_agent_recall_subscription_wake_states
            SET delivery_id = %s, agent_id = %s, task_id = %s,
                knowledge_namespace = %s, access_policy_sha256 = %s,
                state = %s, attempt = %s, state_revision = %s, claim_id = %s,
                lease_expires_at = %s, release_id = %s, acknowledgement_id = %s,
                state_json = %s::jsonb, committed_at = %s, updated_at = %s
            WHERE wake_id = %s AND state_revision = %s
            """,
            (
                *cls._subscription_wake_state_values(updated)[1:],
                updated.wake_id,
                current.state_revision,
            ),
        )
        if cur.rowcount != 1:
            raise AgentRecallSubscriptionConflict("stale_wake_state")

    @staticmethod
    def _subscription_wake_state_values(
        wake: AgentRecallSubscriptionWake,
    ) -> tuple[object, ...]:
        claim = wake.claim
        lease_expires_at = (
            claim.lease_expires_at
            if wake.state is AgentRecallSubscriptionWakeState.CLAIMED and claim is not None
            else None
        )
        state = wake.model_dump(
            mode="json",
            exclude={"subscription", "evaluation", "delivery"},
        )
        return (
            wake.wake_id,
            wake.delivery.delivery_id,
            wake.subscription.agent_id,
            wake.subscription.task_id,
            wake.subscription.knowledge_namespace,
            wake.subscription.access_policy_sha256,
            wake.state.value,
            wake.attempt,
            wake.state_revision,
            None if claim is None else claim.claim_id,
            lease_expires_at,
            None if wake.release is None else wake.release.release_id,
            (None if wake.acknowledgement is None else wake.acknowledgement.acknowledgement_id),
            pg_support._dumps(state),
            wake.evaluation.committed_at,
            wake.updated_at,
        )

    @classmethod
    async def _insert_subscription_state(
        cls,
        cur: Any,
        record: AgentRecallSubscriptionRecord,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_agent_recall_subscription_states (
                subscription_id, current_revision, agent_id, task_id,
                knowledge_namespace, access_policy_sha256, run_state,
                attempt, state_revision, lease_expires_at, release_id,
                next_evaluation_at, last_evaluation_id, state_json, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            """,
            cls._subscription_state_values(record),
        )

    @classmethod
    async def _update_subscription_state(
        cls,
        cur: Any,
        current: AgentRecallSubscriptionRecord,
        updated: AgentRecallSubscriptionRecord,
    ) -> None:
        await cur.execute(
            """
            UPDATE cayu_agent_recall_subscription_states
            SET current_revision = %s, agent_id = %s, task_id = %s,
                knowledge_namespace = %s, access_policy_sha256 = %s,
                run_state = %s, attempt = %s, state_revision = %s,
                lease_expires_at = %s, release_id = %s,
                next_evaluation_at = %s, last_evaluation_id = %s,
                state_json = %s::jsonb, updated_at = %s
            WHERE subscription_id = %s AND state_revision = %s
            """,
            (
                *cls._subscription_state_values(updated)[1:],
                updated.subscription.subscription_id,
                current.state_revision,
            ),
        )
        if cur.rowcount != 1:
            raise AgentRecallSubscriptionConflict("stale_subscription_state")

    @staticmethod
    def _subscription_state_values(
        record: AgentRecallSubscriptionRecord,
    ) -> tuple[object, ...]:
        claim = record.claim
        lease_expires_at = (
            claim.lease_expires_at
            if record.run_state is AgentRecallSubscriptionRunState.CLAIMED and claim is not None
            else None
        )
        subscription = record.subscription
        state = record.model_dump(mode="json", exclude={"subscription"})
        return (
            subscription.subscription_id,
            subscription.revision,
            subscription.agent_id,
            subscription.task_id,
            subscription.knowledge_namespace,
            subscription.access_policy_sha256,
            record.run_state.value,
            record.attempt,
            record.state_revision,
            lease_expires_at,
            None if record.release is None else record.release.release_id,
            record.next_evaluation_at,
            record.last_evaluation_id,
            pg_support._dumps(state),
            record.updated_at,
        )

    async def _insert_subscription_delivery_locked(
        self,
        cur: Any,
        delivery: AgentRecallDelivery,
    ) -> None:
        key = delivery.key()
        await _lock_agent_work_context_identity(
            cur,
            "recall-processing-operation",
            delivery.operation_id,
        )
        await _lock_agent_work_context_identity(cur, "delivery", delivery.delivery_id)
        if await self._load_delivery(cur, delivery.delivery_id) is not None:
            raise AgentRecallSubscriptionConflict("delivery_id_reused")
        await _lock_agent_work_context_identity(cur, "checkpoint", key.fingerprint())
        await cur.execute(
            """
            SELECT delivery_id
            FROM cayu_agent_recall_deliveries
            WHERE agent_id = %s AND task_id = %s
              AND knowledge_namespace = %s AND access_policy_sha256 = %s
              AND checkpoint_stream_id = %s
              AND checkpoint_revision = %s
            """,
            (*key.sort_key(), delivery.checkpoint.revision),
        )
        if await cur.fetchone() is not None:
            raise AgentRecallSubscriptionConflict("checkpoint_delivery_exists")
        if await self._load_delivery_by_operation(cur, delivery.operation_id) is not None:
            raise AgentRecallSubscriptionConflict("delivery_operation_reused")
        if await self._load_checkpoint_by_operation(cur, delivery.operation_id) is not None:
            raise AgentRecallSubscriptionConflict("checkpoint_committed_without_delivery")
        if delivery.staged_at > await self._authoritative_delivery_now(cur):
            raise AgentRecallSubscriptionConflict("delivery_staged_in_future")
        await _lock_agent_work_context_identity_shared(cur, "task", delivery.task_id)
        await self._advance_checkpoint_locked(
            cur,
            delivery.checkpoint,
            expected_revision=delivery.expected_checkpoint_revision,
        )
        await cur.execute(
            """
            INSERT INTO cayu_agent_recall_deliveries (
                delivery_id, operation_id, agent_id, task_id,
                knowledge_namespace, access_policy_sha256,
                checkpoint_stream_id,
                checkpoint_revision, processing_result_sha256,
                delivery_json, staged_at, processing_schema_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s::jsonb, %s, %s
            )
            """,
            (
                delivery.delivery_id,
                delivery.operation_id,
                delivery.agent_id,
                delivery.task_id,
                delivery.knowledge_namespace,
                delivery.access_policy_sha256,
                delivery.checkpoint.checkpoint_stream_id,
                delivery.checkpoint.revision,
                delivery.processing_result_sha256,
                pg_support._dumps(delivery.model_dump(mode="json")),
                delivery.staged_at,
                str(delivery.processing_result["schema_version"]),
            ),
        )
        await self._insert_delivery_state(
            cur,
            AgentRecallDeliveryRecord(
                delivery=delivery,
                updated_at=delivery.staged_at,
            ),
        )

    async def _delivery_now(self, cur: Any, floor: datetime) -> datetime:
        return max(await self._authoritative_delivery_now(cur), floor)

    async def _authoritative_delivery_now(self, cur: Any) -> datetime:
        if self._clock_is_injected:
            return _utc(self._clock(), "clock result")
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        if row is None:  # pragma: no cover - PostgreSQL always returns one row
            raise RuntimeError("Postgres did not return its recall-delivery clock.")
        return row[0]

    @classmethod
    async def _load_delivery(
        cls,
        cur: Any,
        delivery_id: str,
        *,
        for_update: bool = False,
    ) -> AgentRecallDeliveryRecord | None:
        lock_clause = "FOR UPDATE OF state" if for_update else ""
        await cur.execute(
            f"""
            SELECT delivery.delivery_json::text,
                   delivery.operation_id,
                   delivery.processing_result_sha256,
                   delivery.staged_at,
                   checkpoint.record_json::text,
                   state.delivery_id, state.agent_id, state.task_id,
                   state.knowledge_namespace, state.access_policy_sha256,
                   state.checkpoint_stream_id, state.checkpoint_revision,
                   state.state, state.attempt,
                   state.state_revision, state.lease_expires_at,
                   state.release_id, state.acknowledgement_id,
                   state.state_json::text, state.updated_at,
                   delivery.processing_schema_version
            FROM cayu_agent_recall_deliveries AS delivery
            JOIN cayu_agent_recall_delivery_states AS state
              ON state.delivery_id = delivery.delivery_id
            LEFT JOIN cayu_agent_recall_checkpoints AS checkpoint
              ON checkpoint.agent_id = delivery.agent_id
             AND checkpoint.task_id = delivery.task_id
             AND checkpoint.knowledge_namespace = delivery.knowledge_namespace
             AND checkpoint.access_policy_sha256 = delivery.access_policy_sha256
             AND checkpoint.checkpoint_stream_id = delivery.checkpoint_stream_id
             AND checkpoint.revision = delivery.checkpoint_revision
             AND checkpoint.operation_id = delivery.operation_id
            WHERE delivery.delivery_id = %s
            {lock_clause}
            """,
            (delivery_id,),
        )
        row = await cur.fetchone()
        return None if row is None else cls._delivery_record_from_row(row)

    @classmethod
    async def _load_delivery_by_operation(
        cls,
        cur: Any,
        operation_id: str,
    ) -> AgentRecallDeliveryRecord | None:
        await cur.execute(
            "SELECT delivery_id FROM cayu_agent_recall_deliveries WHERE operation_id = %s",
            (operation_id,),
        )
        row = await cur.fetchone()
        return None if row is None else await cls._load_delivery(cur, str(row[0]))

    @staticmethod
    def _delivery_record_from_row(row: Sequence[Any]) -> AgentRecallDeliveryRecord:
        if row[4] is None:
            raise RuntimeError("Postgres recall delivery conflicts with its checkpoint.")
        delivery = json.loads(str(row[0]))
        state = json.loads(str(row[18]))
        if type(delivery) is not dict or type(state) is not dict:
            raise RuntimeError("Postgres recall-delivery JSON must contain objects.")
        state["delivery"] = delivery
        record = AgentRecallDeliveryRecord.model_validate_json(pg_support._dumps(state))
        checkpoint = AgentRecallCheckpoint.model_validate_json(str(row[4]))
        lease_expires_at = (
            record.claim.lease_expires_at
            if record.state is AgentRecallDeliveryState.CLAIMED and record.claim is not None
            else None
        )
        release_id = None if record.release is None else record.release.release_id
        acknowledgement_id = (
            None if record.acknowledgement is None else record.acknowledgement.acknowledgement_id
        )
        if (
            checkpoint != record.delivery.checkpoint
            or str(row[1]) != record.delivery.operation_id
            or str(row[2]) != record.delivery.processing_result_sha256
            or row[3] != record.delivery.staged_at
            or str(row[5]) != record.delivery.delivery_id
            or tuple(str(value) for value in row[6:11]) != record.delivery.key().sort_key()
            or int(row[11]) != record.delivery.checkpoint.revision
            or str(row[12]) != record.state.value
            or int(row[13]) != record.attempt
            or int(row[14]) != record.state_revision
            or row[15] != lease_expires_at
            or row[16] != release_id
            or row[17] != acknowledgement_id
            or row[19] != record.updated_at
            or str(row[20]) != record.delivery.processing_result["schema_version"]
        ):
            raise RuntimeError("Postgres recall-delivery indexes conflict with durable state.")
        return record

    @classmethod
    async def _insert_delivery_state(
        cls,
        cur: Any,
        record: AgentRecallDeliveryRecord,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_agent_recall_delivery_states (
                delivery_id, agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id,
                checkpoint_revision, state, attempt,
                state_revision, lease_expires_at, release_id,
                acknowledgement_id, state_json, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            """,
            cls._delivery_state_values(record),
        )

    @classmethod
    async def _update_delivery_state(
        cls,
        cur: Any,
        current: AgentRecallDeliveryRecord,
        updated: AgentRecallDeliveryRecord,
    ) -> None:
        await cur.execute(
            """
            UPDATE cayu_agent_recall_delivery_states
            SET agent_id = %s, task_id = %s, knowledge_namespace = %s,
                access_policy_sha256 = %s, checkpoint_stream_id = %s,
                checkpoint_revision = %s, state = %s,
                attempt = %s, state_revision = %s, lease_expires_at = %s,
                release_id = %s, acknowledgement_id = %s,
                state_json = %s::jsonb, updated_at = %s
            WHERE delivery_id = %s AND state_revision = %s
            """,
            (
                *cls._delivery_state_values(updated)[1:],
                updated.delivery.delivery_id,
                current.state_revision,
            ),
        )
        if cur.rowcount != 1:
            raise AgentRecallDeliveryConflict("stale_delivery_state")

    @staticmethod
    def _delivery_state_values(record: AgentRecallDeliveryRecord) -> tuple[object, ...]:
        claim = record.claim
        lease_expires_at = (
            claim.lease_expires_at
            if record.state is AgentRecallDeliveryState.CLAIMED and claim is not None
            else None
        )
        state = record.model_dump(mode="json", exclude={"delivery"})
        return (
            record.delivery.delivery_id,
            record.delivery.agent_id,
            record.delivery.task_id,
            record.delivery.knowledge_namespace,
            record.delivery.access_policy_sha256,
            record.delivery.checkpoint.checkpoint_stream_id,
            record.delivery.checkpoint.revision,
            record.state.value,
            record.attempt,
            record.state_revision,
            lease_expires_at,
            None if record.release is None else record.release.release_id,
            (None if record.acknowledgement is None else record.acknowledgement.acknowledgement_id),
            pg_support._dumps(state),
            record.updated_at,
        )

    @staticmethod
    async def _load_context(
        cur: Any,
        task_id: str,
        *,
        revision: int | None,
    ) -> AgentWorkContext | None:
        if revision is None:
            await cur.execute(
                """
                SELECT revision.record_json::text
                FROM cayu_agent_work_context_heads AS head
                JOIN cayu_agent_work_context_revisions AS revision
                  ON revision.task_id = head.task_id
                 AND revision.revision = head.current_revision
                WHERE head.task_id = %s
                """,
                (task_id,),
            )
        else:
            await cur.execute(
                """
                SELECT record_json::text
                FROM cayu_agent_work_context_revisions
                WHERE task_id = %s AND revision = %s
                """,
                (task_id, revision),
            )
        row = await cur.fetchone()
        return None if row is None else AgentWorkContext.model_validate_json(str(row[0]))

    @staticmethod
    async def _load_publication(
        cur: Any,
        operation_id: str,
    ) -> AgentWorkContextPublicationReceipt | None:
        await cur.execute(
            """
            SELECT receipt_json::text
            FROM cayu_agent_work_context_publications
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        return (
            None
            if row is None
            else AgentWorkContextPublicationReceipt.model_validate_json(str(row[0]))
        )

    @staticmethod
    async def _load_checkpoint(
        cur: Any,
        key: AgentRecallCheckpointKey,
        *,
        revision: int | None,
    ) -> AgentRecallCheckpoint | None:
        if revision is None:
            await cur.execute(
                """
                SELECT checkpoint.record_json::text
                FROM cayu_agent_recall_checkpoint_heads AS head
                JOIN cayu_agent_recall_checkpoints AS checkpoint
                  ON checkpoint.agent_id = head.agent_id
                 AND checkpoint.task_id = head.task_id
                 AND checkpoint.knowledge_namespace = head.knowledge_namespace
                 AND checkpoint.access_policy_sha256 = head.access_policy_sha256
                 AND checkpoint.checkpoint_stream_id = head.checkpoint_stream_id
                 AND checkpoint.revision = head.current_revision
                WHERE head.agent_id = %s AND head.task_id = %s
                  AND head.knowledge_namespace = %s
                  AND head.access_policy_sha256 = %s
                  AND head.checkpoint_stream_id = %s
                """,
                key.sort_key(),
            )
        else:
            await cur.execute(
                """
                SELECT record_json::text
                FROM cayu_agent_recall_checkpoints
                WHERE agent_id = %s AND task_id = %s
                  AND knowledge_namespace = %s AND access_policy_sha256 = %s
                  AND checkpoint_stream_id = %s
                  AND revision = %s
                """,
                (*key.sort_key(), revision),
            )
        row = await cur.fetchone()
        return None if row is None else AgentRecallCheckpoint.model_validate_json(str(row[0]))

    @staticmethod
    async def _load_checkpoint_by_operation(
        cur: Any,
        operation_id: str,
    ) -> AgentRecallCheckpoint | None:
        await cur.execute(
            """
            SELECT record_json::text
            FROM cayu_agent_recall_checkpoints
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        return None if row is None else AgentRecallCheckpoint.model_validate_json(str(row[0]))
