"""SQLite durability for agent work contexts and recall checkpoints."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from cayu._clock import utc_clock
from cayu._validation import require_nonblank
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
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

_SQLITE_MIN_REQUIRED_REVISION = 73


def _document(value) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_context(document: str) -> AgentWorkContext:
    return AgentWorkContext.model_validate_json(document)


def _parse_publication(document: str) -> AgentWorkContextPublicationReceipt:
    return AgentWorkContextPublicationReceipt.model_validate_json(document)


def _parse_checkpoint(document: str) -> AgentRecallCheckpoint:
    return AgentRecallCheckpoint.model_validate_json(document)


def _state_document(record: AgentRecallDeliveryRecord) -> str:
    return json.dumps(
        record.model_dump(mode="json", exclude={"delivery"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_delivery_record(
    delivery_document: str,
    state_document: str,
) -> AgentRecallDeliveryRecord:
    delivery = json.loads(delivery_document)
    if type(delivery) is not dict:
        raise ValueError("SQLite recall delivery must be a JSON object.")
    state = json.loads(state_document)
    if type(state) is not dict:
        raise ValueError("SQLite recall-delivery state must be a JSON object.")
    state["delivery"] = delivery
    return AgentRecallDeliveryRecord.model_validate_json(
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _parse_subscription(document: str) -> AgentRecallSubscription:
    return AgentRecallSubscription.model_validate_json(document)


def _parse_subscription_publication(
    document: str,
) -> AgentRecallSubscriptionPublicationReceipt:
    return AgentRecallSubscriptionPublicationReceipt.model_validate_json(document)


def _subscription_state_document(record: AgentRecallSubscriptionRecord) -> str:
    return json.dumps(
        record.model_dump(mode="json", exclude={"subscription"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_subscription_record(
    subscription_document: str,
    state_document: str,
) -> AgentRecallSubscriptionRecord:
    subscription = json.loads(subscription_document)
    state = json.loads(state_document)
    if type(subscription) is not dict or type(state) is not dict:
        raise ValueError("SQLite recall-subscription JSON must contain objects.")
    state["subscription"] = subscription
    return AgentRecallSubscriptionRecord.model_validate_json(
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _parse_subscription_evaluation(document: str) -> AgentRecallSubscriptionEvaluation:
    return AgentRecallSubscriptionEvaluation.model_validate_json(document)


def _subscription_wake_state_document(wake: AgentRecallSubscriptionWake) -> str:
    return json.dumps(
        wake.model_dump(
            mode="json",
            exclude={"subscription", "evaluation", "delivery"},
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_subscription_wake(
    subscription_document: str,
    evaluation_document: str,
    delivery_document: str,
    state_document: str,
) -> AgentRecallSubscriptionWake:
    subscription = json.loads(subscription_document)
    evaluation = json.loads(evaluation_document)
    delivery = json.loads(delivery_document)
    state = json.loads(state_document)
    if any(type(value) is not dict for value in (subscription, evaluation, delivery, state)):
        raise ValueError("SQLite subscription-wake JSON must contain objects.")
    state.update(
        subscription=subscription,
        evaluation=evaluation,
        delivery=delivery,
    )
    return AgentRecallSubscriptionWake.model_validate_json(
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


class SQLiteAgentWorkContextStore(AgentWorkContextStore):
    """SQLite implementation with transaction-fenced current pointers."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        clock=None,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteAgentWorkContextStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._clock = utc_clock(clock)
        self._lock = asyncio.Lock()
        self._connection = sqlite_support.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection,
                schema_mode,
                app_min_supported=_SQLITE_MIN_REQUIRED_REVISION,
            )
        except BaseException:
            self._connection.close()
            raise

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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                replay = self._load_publication_unlocked(context.operation_id)
                if replay is not None:
                    if replay.request_sha256 != request_sha256:
                        raise AgentWorkContextConflict("publication_operation_reused")
                    return copy_agent_work_context_publication_receipt(replay)
                current = self._load_context_unlocked(context.task_id, revision=None)
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
                    self._insert_context_unlocked(context, expected_revision=expected_revision)
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_work_context_publications (
                        operation_id, task_id, request_sha256, context_revision,
                        changed, receipt_json, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.operation_id,
                        result.task_id,
                        receipt.request_sha256,
                        result.revision,
                        int(receipt.changed),
                        _document(receipt),
                        receipt.committed_at.isoformat(),
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
        async with self._lock:
            context = self._load_context_unlocked(task_id, revision=revision)
            return None if context is None else copy_agent_work_context(context)

    async def load_work_context_publication(
        self,
        operation_id: str,
    ) -> AgentWorkContextPublicationReceipt | None:
        operation_id = _bounded_identity(operation_id, "operation_id")
        async with self._lock:
            receipt = self._load_publication_unlocked(operation_id)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                occupied = self._connection.execute(
                    """
                    SELECT 1 FROM cayu_agent_recall_deliveries WHERE operation_id = ?
                    UNION ALL
                    SELECT 1 FROM cayu_agent_recall_subscription_evaluations
                    WHERE processing_operation_id = ?
                    LIMIT 1
                    """,
                    (checkpoint.operation_id, checkpoint.operation_id),
                ).fetchone()
                if occupied is not None:
                    raise AgentWorkContextConflict("checkpoint_operation_reused")
                replay_row = self._connection.execute(
                    """
                    SELECT record_json
                    FROM cayu_agent_recall_checkpoints
                    WHERE operation_id = ?
                    """,
                    (checkpoint.operation_id,),
                ).fetchone()
                if replay_row is not None:
                    replay = _parse_checkpoint(str(replay_row["record_json"]))
                    replay_expected_revision = None if replay.revision == 1 else replay.revision - 1
                    if replay != checkpoint or expected_revision != replay_expected_revision:
                        raise AgentWorkContextConflict("checkpoint_operation_reused")
                    return copy_agent_recall_checkpoint(replay)
                self._advance_checkpoint_unlocked(
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
        async with self._lock:
            checkpoint = self._load_checkpoint_unlocked(key, revision=revision)
            return None if checkpoint is None else copy_agent_recall_checkpoint(checkpoint)

    async def stage_recall_delivery(
        self,
        delivery: AgentRecallDelivery,
    ) -> AgentRecallDeliveryRecord:
        delivery = copy_agent_recall_delivery(delivery)
        key = delivery.key()
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                evaluation_row = self._connection.execute(
                    "SELECT 1 FROM cayu_agent_recall_subscription_evaluations "
                    "WHERE processing_operation_id = ?",
                    (delivery.operation_id,),
                ).fetchone()
                if evaluation_row is not None:
                    raise AgentRecallDeliveryConflict("delivery_operation_reused")
                existing = self._load_delivery_unlocked(delivery.delivery_id)
                if existing is not None:
                    if existing.delivery != delivery:
                        raise AgentRecallDeliveryConflict("delivery_id_reused")
                    return copy_agent_recall_delivery_record(existing)
                occupied = self._connection.execute(
                    """
                    SELECT delivery_id
                    FROM cayu_agent_recall_deliveries
                    WHERE agent_id = ? AND task_id = ?
                      AND knowledge_namespace = ? AND access_policy_sha256 = ?
                      AND checkpoint_stream_id = ?
                      AND checkpoint_revision = ?
                    """,
                    (*key.sort_key(), delivery.checkpoint.revision),
                ).fetchone()
                if occupied is not None:
                    raise AgentRecallDeliveryConflict("checkpoint_delivery_exists")
                operation_row = self._connection.execute(
                    "SELECT delivery_id FROM cayu_agent_recall_deliveries WHERE operation_id = ?",
                    (delivery.operation_id,),
                ).fetchone()
                if operation_row is not None:
                    raise AgentRecallDeliveryConflict("delivery_operation_reused")
                checkpoint_row = self._connection.execute(
                    "SELECT 1 FROM cayu_agent_recall_checkpoints WHERE operation_id = ?",
                    (delivery.operation_id,),
                ).fetchone()
                if checkpoint_row is not None:
                    raise AgentRecallDeliveryConflict("checkpoint_committed_without_delivery")
                if delivery.staged_at > _utc(self._clock(), "clock result"):
                    raise AgentRecallDeliveryConflict("delivery_staged_in_future")
                self._advance_checkpoint_unlocked(
                    delivery.checkpoint,
                    expected_revision=delivery.expected_checkpoint_revision,
                )
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_deliveries (
                        delivery_id, operation_id, agent_id, task_id,
                        knowledge_namespace, access_policy_sha256,
                        checkpoint_stream_id,
                        checkpoint_revision, processing_result_sha256,
                        delivery_json, staged_at, processing_schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        _document(delivery),
                        delivery.staged_at.isoformat(),
                        str(delivery.processing_result["schema_version"]),
                    ),
                )
                record = AgentRecallDeliveryRecord(
                    delivery=delivery,
                    updated_at=delivery.staged_at,
                )
                self._insert_delivery_state_unlocked(record)
                return copy_agent_recall_delivery_record(record)

    async def load_recall_delivery(
        self,
        delivery_id: str,
    ) -> AgentRecallDeliveryRecord | None:
        delivery_id = _bounded_identity(delivery_id, "delivery_id")
        async with self._lock:
            record = self._load_delivery_unlocked(delivery_id)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                replay = self._connection.execute(
                    "SELECT delivery_id, worker_id, request_sha256, attempt "
                    "FROM cayu_agent_recall_delivery_claims WHERE claim_id = ?",
                    (claim_id,),
                ).fetchone()
                if replay is not None:
                    if replay["request_sha256"] != request_sha256:
                        raise AgentRecallDeliveryConflict("claim_id_reused")
                    record = self._load_delivery_unlocked(str(replay["delivery_id"]))
                    if record is None:  # pragma: no cover - protected by foreign key
                        raise RuntimeError("SQLite recall-delivery claim lost its delivery.")
                    _require_replayable_delivery_claim_attempt(
                        record,
                        claim_id=claim_id,
                        worker_id=str(replay["worker_id"]),
                        attempt=int(replay["attempt"]),
                        now=max(_utc(self._clock(), "clock result"), record.updated_at),
                    )
                    return copy_agent_recall_delivery_record(record)
                row = self._connection.execute(
                    """
                    SELECT delivery.delivery_json,
                           delivery.operation_id AS delivery_operation_id,
                           delivery.processing_result_sha256 AS delivery_result_sha256,
                           delivery.staged_at AS delivery_staged_at,
                           delivery.processing_schema_version AS delivery_processing_schema_version,
                           checkpoint.record_json AS checkpoint_json,
                           state.*
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
                    WHERE state.agent_id = ? AND state.task_id = ?
                      AND state.knowledge_namespace = ?
                      AND state.access_policy_sha256 = ?
                      AND state.checkpoint_stream_id = ?
                      AND state.state != 'acknowledged'
                      AND NOT EXISTS (
                        SELECT 1
                        FROM cayu_agent_recall_subscription_wake_states AS wake_state
                        WHERE wake_state.delivery_id = state.delivery_id
                          AND wake_state.state != 'acknowledged'
                      )
                    ORDER BY state.checkpoint_revision, state.delivery_id COLLATE BINARY
                    LIMIT 1
                    """,
                    key.sort_key(),
                ).fetchone()
                if row is None:
                    return None
                current = self._delivery_record_from_row(row)
                now = max(_utc(self._clock(), "clock result"), current.updated_at)
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
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_delivery_claims (
                        claim_id, delivery_id, worker_id, request_sha256,
                        attempt, claimed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claimed.claim.claim_id,
                        claimed.delivery.delivery_id,
                        claimed.claim.worker_id,
                        request_sha256,
                        claimed.claim.attempt,
                        claimed.claim.claimed_at.isoformat(),
                    ),
                )
                self._update_delivery_state_unlocked(current, claimed)
                return copy_agent_recall_delivery_record(claimed)

    async def renew_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallDeliveryRecord:
        claim = copy_agent_recall_delivery_claim(claim)
        _validate_delivery_lease_seconds(lease_seconds)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_delivery_unlocked(claim.delivery_id)
                if current is None:
                    raise AgentRecallDeliveryConflict("unknown_delivery")
                renewed = _renew_agent_recall_delivery_record(
                    current,
                    claim,
                    lease_seconds=lease_seconds,
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                if renewed != current:
                    self._update_delivery_state_unlocked(current, renewed)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_delivery_unlocked(claim.delivery_id)
                if current is None:
                    raise AgentRecallDeliveryConflict("unknown_delivery")
                replay = self._connection.execute(
                    "SELECT delivery_id, release_json "
                    "FROM cayu_agent_recall_delivery_releases WHERE release_id = ?",
                    (requested.release_id,),
                ).fetchone()
                if replay is not None:
                    stored = AgentRecallDeliveryRelease.model_validate_json(
                        str(replay["release_json"])
                    )
                    if replay["delivery_id"] != claim.delivery_id or stored != requested:
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
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_delivery_releases (
                        release_id, delivery_id, claim_id, request_sha256,
                        release_json, released_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        requested.release_id,
                        requested.delivery_id,
                        requested.claim_id,
                        requested.fingerprint(),
                        _document(requested),
                        requested.released_at.isoformat(),
                    ),
                )
                self._update_delivery_state_unlocked(current, released)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_delivery_unlocked(claim.delivery_id)
                if current is None:
                    raise AgentRecallDeliveryConflict("unknown_delivery")
                occupied = self._connection.execute(
                    "SELECT delivery_id FROM cayu_agent_recall_delivery_states "
                    "WHERE acknowledgement_id = ?",
                    (acknowledgement_id,),
                ).fetchone()
                if occupied is not None and occupied["delivery_id"] != claim.delivery_id:
                    raise AgentRecallDeliveryConflict("acknowledgement_reused")
                acknowledged = _acknowledge_agent_recall_delivery_record(
                    current,
                    claim,
                    acknowledgement_id=acknowledgement_id,
                    evidence_kind=evidence_kind,
                    evidence_ref=evidence_ref,
                    acknowledged_at=acknowledged_at,
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                if acknowledged != current:
                    self._update_delivery_state_unlocked(current, acknowledged)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                replay = self._load_subscription_publication_unlocked(subscription.operation_id)
                if replay is not None:
                    if replay.request_sha256 != request_sha256:
                        raise AgentRecallSubscriptionConflict("publication_operation_reused")
                    return copy_agent_recall_subscription_publication_receipt(replay)
                current = self._load_subscription_unlocked(
                    subscription.subscription_id,
                    revision=None,
                )
                work_context = self._load_context_unlocked(
                    subscription.task_id,
                    revision=None,
                )
                now = _utc(self._clock(), "clock result")
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
                prior_state = self._load_subscription_state_unlocked(subscription.subscription_id)
                state = AgentRecallSubscriptionRecord(
                    subscription=subscription,
                    state_revision=(0 if prior_state is None else prior_state.state_revision + 1),
                    attempt=0 if prior_state is None else prior_state.attempt,
                    next_evaluation_at=max(now, subscription.published_at),
                    updated_at=now,
                )
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_revisions (
                        subscription_id, revision, operation_id, agent_id, task_id,
                        knowledge_namespace, access_policy_sha256,
                        work_context_revision, work_context_sha256, status, priority,
                        subscription_json, expires_at, published_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        _document(subscription),
                        subscription.expires_at.isoformat(),
                        subscription.published_at.isoformat(),
                    ),
                )
                if current is None:
                    self._connection.execute(
                        "INSERT INTO cayu_agent_recall_subscription_heads "
                        "(subscription_id, current_revision) VALUES (?, ?)",
                        (subscription.subscription_id, subscription.revision),
                    )
                    self._insert_subscription_state_unlocked(state)
                else:
                    head = self._connection.execute(
                        "UPDATE cayu_agent_recall_subscription_heads "
                        "SET current_revision = ? "
                        "WHERE subscription_id = ? AND current_revision = ?",
                        (
                            subscription.revision,
                            subscription.subscription_id,
                            expected_revision,
                        ),
                    )
                    if head.rowcount != 1:
                        raise AgentRecallSubscriptionConflict("stale_subscription_revision")
                    assert prior_state is not None
                    self._update_subscription_state_unlocked(prior_state, state)
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_publications (
                        operation_id, subscription_id, subscription_revision,
                        request_sha256, receipt_json, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.operation_id,
                        subscription.subscription_id,
                        subscription.revision,
                        request_sha256,
                        _document(receipt),
                        receipt.committed_at.isoformat(),
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
        async with self._lock:
            subscription = self._load_subscription_unlocked(
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                replay = self._connection.execute(
                    "SELECT subscription_id, runner_id, request_sha256, attempt "
                    "FROM cayu_agent_recall_subscription_claims WHERE claim_id = ?",
                    (claim_id,),
                ).fetchone()
                if replay is not None:
                    if replay["request_sha256"] != request_sha256:
                        raise AgentRecallSubscriptionConflict("claim_id_reused")
                    record = self._load_subscription_state_unlocked(str(replay["subscription_id"]))
                    if record is None:  # pragma: no cover - foreign key invariant
                        raise RuntimeError("SQLite subscription claim lost its state.")
                    current_claim = record.claim
                    now = max(_utc(self._clock(), "clock result"), record.updated_at)
                    if (
                        record.run_state is not AgentRecallSubscriptionRunState.CLAIMED
                        or current_claim is None
                        or current_claim.claim_id != claim_id
                        or current_claim.runner_id != str(replay["runner_id"])
                        or current_claim.attempt != int(replay["attempt"])
                    ):
                        raise AgentRecallSubscriptionConflict("claim_replay_superseded")
                    if current_claim.lease_expires_at <= now:
                        raise AgentRecallSubscriptionConflict("expired_subscription_claim")
                    return copy_agent_recall_subscription_record(record)
                now = _utc(self._clock(), "clock result")
                row = self._connection.execute(
                    """
                    SELECT revision.subscription_json, state.*
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
                    WHERE state.agent_id = ? AND state.task_id = ?
                      AND state.knowledge_namespace = ?
                      AND state.access_policy_sha256 = ?
                      AND revision.status = 'active'
                      AND revision.expires_at > ?
                      AND state.next_evaluation_at <= ?
                      AND (
                        state.run_state = 'due'
                        OR state.lease_expires_at <= ?
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
                             state.subscription_id COLLATE BINARY
                    LIMIT 1
                    """,
                    (
                        *key.authority_sort_key(),
                        now.isoformat(),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                ).fetchone()
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
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_claims (
                        claim_id, subscription_id, subscription_revision,
                        runner_id, request_sha256, attempt, claimed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claimed.claim.claim_id,
                        claimed.subscription.subscription_id,
                        claimed.subscription.revision,
                        claimed.claim.runner_id,
                        request_sha256,
                        claimed.claim.attempt,
                        claimed.claim.claimed_at.isoformat(),
                    ),
                )
                self._update_subscription_state_unlocked(current, claimed)
                return copy_agent_recall_subscription_record(claimed)

    async def renew_recall_subscription(
        self,
        claim: AgentRecallSubscriptionClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionRecord:
        claim = copy_agent_recall_subscription_claim(claim)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_subscription_state_unlocked(claim.subscription_id)
                if current is None:
                    raise AgentRecallSubscriptionConflict("unknown_subscription")
                renewed = _renew_agent_recall_subscription_record(
                    current,
                    claim,
                    lease_seconds=lease_seconds,
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                self._update_subscription_state_unlocked(current, renewed)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_subscription_state_unlocked(claim.subscription_id)
                if current is None:
                    raise AgentRecallSubscriptionConflict("unknown_subscription")
                replay = self._connection.execute(
                    "SELECT subscription_id, release_json "
                    "FROM cayu_agent_recall_subscription_releases WHERE release_id = ?",
                    (requested.release_id,),
                ).fetchone()
                if replay is not None:
                    stored = AgentRecallSubscriptionRelease.model_validate_json(
                        str(replay["release_json"])
                    )
                    if replay["subscription_id"] != claim.subscription_id or stored != requested:
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
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_releases (
                        release_id, subscription_id, claim_id, request_sha256,
                        release_json, released_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        requested.release_id,
                        requested.subscription_id,
                        requested.claim_id,
                        requested.fingerprint(),
                        _document(requested),
                        requested.released_at.isoformat(),
                    ),
                )
                self._update_subscription_state_unlocked(current, released)
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
        from cayu.recall_processing import AgentRecallProcessingResult

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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                replay = self._load_subscription_evaluation_unlocked(evaluation_id)
                if replay is not None:
                    if replay.request_sha256 != request_sha256:
                        raise AgentRecallSubscriptionConflict("evaluation_id_reused")
                    return copy_agent_recall_subscription_evaluation(replay)
                occupied = self._connection.execute(
                    "SELECT evaluation_id "
                    "FROM cayu_agent_recall_subscription_evaluations "
                    "WHERE processing_operation_id = ?",
                    (result.operation_id,),
                ).fetchone()
                if occupied is not None:
                    raise AgentRecallSubscriptionConflict("processing_operation_reused")
                occupied_checkpoint = self._connection.execute(
                    "SELECT 1 FROM cayu_agent_recall_checkpoints WHERE operation_id = ?",
                    (result.operation_id,),
                ).fetchone()
                occupied_delivery = self._connection.execute(
                    "SELECT 1 FROM cayu_agent_recall_deliveries WHERE operation_id = ?",
                    (result.operation_id,),
                ).fetchone()
                if occupied_checkpoint is not None or occupied_delivery is not None:
                    raise AgentRecallSubscriptionConflict("processing_operation_reused")
                current = self._load_subscription_state_unlocked(claim.subscription_id)
                if current is None:
                    raise AgentRecallSubscriptionConflict("unknown_subscription")
                evaluation, delivery, updated = _prepare_agent_recall_subscription_evaluation(
                    current,
                    claim,
                    result,
                    self._load_context_unlocked(
                        current.subscription.task_id,
                        revision=None,
                    ),
                    evaluation_id=evaluation_id,
                    delivery_id=delivery_id,
                    staged_by=staged_by,
                    evaluated_at=evaluated_at,
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                if delivery is not None:
                    self._insert_subscription_delivery_unlocked(delivery)
                elif result.proposed_checkpoint is not None:
                    checkpoint = result.proposed_checkpoint
                    self._advance_checkpoint_unlocked(
                        checkpoint,
                        expected_revision=(
                            None if checkpoint.revision == 1 else checkpoint.revision - 1
                        ),
                    )
                subscription = current.subscription
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_evaluations (
                        evaluation_id, subscription_id, subscription_revision,
                        agent_id, task_id, knowledge_namespace,
                        access_policy_sha256, claim_id, processing_operation_id,
                        request_sha256, outcome, delivery_id,
                        evaluation_json, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        _document(evaluation),
                        evaluation.committed_at.isoformat(),
                    ),
                )
                if delivery is not None:
                    self._insert_subscription_wake_state_unlocked(
                        AgentRecallSubscriptionWake(
                            wake_id=evaluation.evaluation_id,
                            subscription=subscription,
                            evaluation=evaluation,
                            delivery=delivery,
                            updated_at=evaluation.committed_at,
                        )
                    )
                self._update_subscription_state_unlocked(current, updated)
                return copy_agent_recall_subscription_evaluation(evaluation)

    async def load_recall_subscription_evaluation(
        self,
        evaluation_id: str,
    ) -> AgentRecallSubscriptionEvaluation | None:
        evaluation_id = _bounded_identity(evaluation_id, "evaluation_id")
        async with self._lock:
            evaluation = self._load_subscription_evaluation_unlocked(evaluation_id)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                replay = self._connection.execute(
                    "SELECT wake_id, runner_id, request_sha256, attempt "
                    "FROM cayu_agent_recall_subscription_wake_claims WHERE claim_id = ?",
                    (claim_id,),
                ).fetchone()
                if replay is not None:
                    if replay["request_sha256"] != request_sha256:
                        raise AgentRecallSubscriptionConflict("wake_claim_id_reused")
                    wake = self._load_subscription_wake_unlocked(str(replay["wake_id"]))
                    if wake is None:  # pragma: no cover - foreign key invariant
                        raise RuntimeError("SQLite subscription-wake claim lost its state.")
                    _require_replayable_subscription_wake_claim(
                        wake,
                        claim_id=claim_id,
                        runner_id=str(replay["runner_id"]),
                        attempt=int(replay["attempt"]),
                        now=max(_utc(self._clock(), "clock result"), wake.updated_at),
                    )
                    return copy_agent_recall_subscription_wake(wake)
                now = _utc(self._clock(), "clock result")
                row = self._connection.execute(
                    """
                    SELECT revision.subscription_json, evaluation.evaluation_json,
                           delivery.delivery_json, wake_state.*
                    FROM cayu_agent_recall_subscription_wake_states AS wake_state
                    JOIN cayu_agent_recall_subscription_evaluations AS evaluation
                      ON evaluation.evaluation_id = wake_state.wake_id
                    JOIN cayu_agent_recall_subscription_revisions AS revision
                      ON revision.subscription_id = evaluation.subscription_id
                     AND revision.revision = evaluation.subscription_revision
                    JOIN cayu_agent_recall_deliveries AS delivery
                      ON delivery.delivery_id = wake_state.delivery_id
                    WHERE wake_state.agent_id = ? AND wake_state.task_id = ?
                      AND wake_state.knowledge_namespace = ?
                      AND wake_state.access_policy_sha256 = ?
                      AND wake_state.state != 'acknowledged'
                      AND (
                        wake_state.state = 'pending'
                        OR wake_state.lease_expires_at <= ?
                      )
                    ORDER BY wake_state.committed_at,
                             wake_state.wake_id COLLATE BINARY
                    LIMIT 1
                    """,
                    (*key.authority_sort_key(), now.isoformat()),
                ).fetchone()
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
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_wake_claims (
                        claim_id, wake_id, delivery_id, runner_id, request_sha256,
                        attempt, claimed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claimed.claim.claim_id,
                        claimed.wake_id,
                        claimed.claim.delivery_id,
                        claimed.claim.runner_id,
                        request_sha256,
                        claimed.claim.attempt,
                        claimed.claim.claimed_at.isoformat(),
                    ),
                )
                self._update_subscription_wake_state_unlocked(current, claimed)
                return copy_agent_recall_subscription_wake(claimed)

    async def load_recall_subscription_wake(
        self,
        wake_id: str,
    ) -> AgentRecallSubscriptionWake | None:
        wake_id = _bounded_identity(wake_id, "wake_id")
        async with self._lock:
            wake = self._load_subscription_wake_unlocked(wake_id)
            return None if wake is None else copy_agent_recall_subscription_wake(wake)

    async def renew_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionWake:
        claim = copy_agent_recall_subscription_wake_claim(claim)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_subscription_wake_unlocked(claim.wake_id)
                if current is None:
                    raise AgentRecallSubscriptionConflict("unknown_wake")
                renewed = _renew_agent_recall_subscription_wake(
                    current,
                    claim,
                    lease_seconds=lease_seconds,
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                self._update_subscription_wake_state_unlocked(current, renewed)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_subscription_wake_unlocked(claim.wake_id)
                if current is None:
                    raise AgentRecallSubscriptionConflict("unknown_wake")
                replay = self._connection.execute(
                    "SELECT wake_id, release_json "
                    "FROM cayu_agent_recall_subscription_wake_releases "
                    "WHERE release_id = ?",
                    (requested.release_id,),
                ).fetchone()
                if replay is not None:
                    stored = AgentRecallSubscriptionWakeRelease.model_validate_json(
                        str(replay["release_json"])
                    )
                    if replay["wake_id"] != claim.wake_id or stored != requested:
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
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                self._connection.execute(
                    """
                    INSERT INTO cayu_agent_recall_subscription_wake_releases (
                        release_id, wake_id, claim_id, request_sha256,
                        release_json, released_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        requested.release_id,
                        requested.wake_id,
                        requested.claim_id,
                        requested.fingerprint(),
                        _document(requested),
                        requested.released_at.isoformat(),
                    ),
                )
                self._update_subscription_wake_state_unlocked(current, released)
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
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current = self._load_subscription_wake_unlocked(claim.wake_id)
                if current is None:
                    raise AgentRecallSubscriptionConflict("unknown_wake")
                occupied = self._connection.execute(
                    "SELECT wake_id FROM cayu_agent_recall_subscription_wake_states "
                    "WHERE acknowledgement_id = ?",
                    (acknowledgement_id,),
                ).fetchone()
                if occupied is not None and occupied["wake_id"] != claim.wake_id:
                    raise AgentRecallSubscriptionConflict("wake_acknowledgement_id_reused")
                acknowledged = _acknowledge_agent_recall_subscription_wake(
                    current,
                    claim,
                    acknowledgement_id=acknowledgement_id,
                    acknowledged_at=acknowledged_at,
                    now=max(_utc(self._clock(), "clock result"), current.updated_at),
                )
                self._update_subscription_wake_state_unlocked(current, acknowledged)
                return copy_agent_recall_subscription_wake(acknowledged)

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _load_subscription_unlocked(
        self,
        subscription_id: str,
        *,
        revision: int | None,
    ) -> AgentRecallSubscription | None:
        if revision is None:
            row = self._connection.execute(
                """
                SELECT revision.subscription_json
                FROM cayu_agent_recall_subscription_heads AS head
                JOIN cayu_agent_recall_subscription_revisions AS revision
                  ON revision.subscription_id = head.subscription_id
                 AND revision.revision = head.current_revision
                WHERE head.subscription_id = ?
                """,
                (subscription_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT subscription_json "
                "FROM cayu_agent_recall_subscription_revisions "
                "WHERE subscription_id = ? AND revision = ?",
                (subscription_id, revision),
            ).fetchone()
        return None if row is None else _parse_subscription(str(row["subscription_json"]))

    def _load_subscription_publication_unlocked(
        self,
        operation_id: str,
    ) -> AgentRecallSubscriptionPublicationReceipt | None:
        row = self._connection.execute(
            "SELECT receipt_json "
            "FROM cayu_agent_recall_subscription_publications "
            "WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        return None if row is None else _parse_subscription_publication(str(row["receipt_json"]))

    def _load_subscription_state_unlocked(
        self,
        subscription_id: str,
    ) -> AgentRecallSubscriptionRecord | None:
        row = self._connection.execute(
            """
            SELECT revision.subscription_json, state.*
            FROM cayu_agent_recall_subscription_states AS state
            JOIN cayu_agent_recall_subscription_revisions AS revision
              ON revision.subscription_id = state.subscription_id
             AND revision.revision = state.current_revision
            WHERE state.subscription_id = ?
            """,
            (subscription_id,),
        ).fetchone()
        return None if row is None else self._subscription_record_from_row(row)

    @staticmethod
    def _subscription_record_from_row(row) -> AgentRecallSubscriptionRecord:
        record = _parse_subscription_record(
            str(row["subscription_json"]),
            str(row["state_json"]),
        )
        claim = record.claim
        lease_expires_at = (
            claim.lease_expires_at.isoformat()
            if record.run_state is AgentRecallSubscriptionRunState.CLAIMED and claim is not None
            else None
        )
        release_id = None if record.release is None else record.release.release_id
        if (
            row["subscription_id"] != record.subscription.subscription_id
            or row["current_revision"] != record.subscription.revision
            or (
                row["agent_id"],
                row["task_id"],
                row["knowledge_namespace"],
                row["access_policy_sha256"],
            )
            != record.subscription.checkpoint_key().authority_sort_key()
            or row["run_state"] != record.run_state.value
            or row["attempt"] != record.attempt
            or row["state_revision"] != record.state_revision
            or row["lease_expires_at"] != lease_expires_at
            or row["release_id"] != release_id
            or row["next_evaluation_at"] != record.next_evaluation_at.isoformat()
            or row["last_evaluation_id"] != record.last_evaluation_id
            or row["updated_at"] != record.updated_at.isoformat()
        ):
            raise RuntimeError("SQLite recall-subscription indexes conflict with durable state.")
        return record

    def _load_subscription_evaluation_unlocked(
        self,
        evaluation_id: str,
    ) -> AgentRecallSubscriptionEvaluation | None:
        row = self._connection.execute(
            "SELECT evaluation_json "
            "FROM cayu_agent_recall_subscription_evaluations "
            "WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
        return None if row is None else _parse_subscription_evaluation(str(row["evaluation_json"]))

    def _load_subscription_wake_unlocked(
        self,
        wake_id: str,
    ) -> AgentRecallSubscriptionWake | None:
        row = self._connection.execute(
            """
            SELECT revision.subscription_json, evaluation.evaluation_json,
                   delivery.delivery_json, wake_state.*
            FROM cayu_agent_recall_subscription_wake_states AS wake_state
            JOIN cayu_agent_recall_subscription_evaluations AS evaluation
              ON evaluation.evaluation_id = wake_state.wake_id
            JOIN cayu_agent_recall_subscription_revisions AS revision
              ON revision.subscription_id = evaluation.subscription_id
             AND revision.revision = evaluation.subscription_revision
            JOIN cayu_agent_recall_deliveries AS delivery
              ON delivery.delivery_id = wake_state.delivery_id
            WHERE wake_state.wake_id = ?
            """,
            (wake_id,),
        ).fetchone()
        return None if row is None else self._subscription_wake_from_row(row)

    @staticmethod
    def _subscription_wake_from_row(row) -> AgentRecallSubscriptionWake:
        wake = _parse_subscription_wake(
            str(row["subscription_json"]),
            str(row["evaluation_json"]),
            str(row["delivery_json"]),
            str(row["state_json"]),
        )
        claim = wake.claim
        lease_expires_at = (
            claim.lease_expires_at.isoformat()
            if wake.state is AgentRecallSubscriptionWakeState.CLAIMED and claim is not None
            else None
        )
        release_id = None if wake.release is None else wake.release.release_id
        acknowledgement_id = (
            None if wake.acknowledgement is None else wake.acknowledgement.acknowledgement_id
        )
        if (
            row["wake_id"] != wake.wake_id
            or row["delivery_id"] != wake.delivery.delivery_id
            or (
                row["agent_id"],
                row["task_id"],
                row["knowledge_namespace"],
                row["access_policy_sha256"],
            )
            != wake.subscription.checkpoint_key().authority_sort_key()
            or row["state"] != wake.state.value
            or row["attempt"] != wake.attempt
            or row["state_revision"] != wake.state_revision
            or row["claim_id"] != (None if claim is None else claim.claim_id)
            or row["lease_expires_at"] != lease_expires_at
            or row["release_id"] != release_id
            or row["acknowledgement_id"] != acknowledgement_id
            or row["committed_at"] != wake.evaluation.committed_at.isoformat()
            or row["updated_at"] != wake.updated_at.isoformat()
        ):
            raise RuntimeError("SQLite subscription-wake indexes conflict with durable state.")
        return wake

    def _insert_subscription_wake_state_unlocked(
        self,
        wake: AgentRecallSubscriptionWake,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_agent_recall_subscription_wake_states (
                wake_id, delivery_id, agent_id, task_id, knowledge_namespace,
                access_policy_sha256, state, attempt, state_revision, claim_id,
                lease_expires_at, release_id, acknowledgement_id, state_json,
                committed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._subscription_wake_state_values(wake),
        )

    def _update_subscription_wake_state_unlocked(
        self,
        current: AgentRecallSubscriptionWake,
        updated: AgentRecallSubscriptionWake,
    ) -> None:
        cursor = self._connection.execute(
            """
            UPDATE cayu_agent_recall_subscription_wake_states
            SET delivery_id = ?, agent_id = ?, task_id = ?,
                knowledge_namespace = ?, access_policy_sha256 = ?, state = ?,
                attempt = ?, state_revision = ?, claim_id = ?,
                lease_expires_at = ?, release_id = ?, acknowledgement_id = ?,
                state_json = ?, committed_at = ?, updated_at = ?
            WHERE wake_id = ? AND state_revision = ?
            """,
            (
                *self._subscription_wake_state_values(updated)[1:],
                updated.wake_id,
                current.state_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise AgentRecallSubscriptionConflict("stale_wake_state")

    @staticmethod
    def _subscription_wake_state_values(
        wake: AgentRecallSubscriptionWake,
    ) -> tuple[object, ...]:
        claim = wake.claim
        lease_expires_at = (
            claim.lease_expires_at.isoformat()
            if wake.state is AgentRecallSubscriptionWakeState.CLAIMED and claim is not None
            else None
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
            _subscription_wake_state_document(wake),
            wake.evaluation.committed_at.isoformat(),
            wake.updated_at.isoformat(),
        )

    def _insert_subscription_state_unlocked(
        self,
        record: AgentRecallSubscriptionRecord,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_agent_recall_subscription_states (
                subscription_id, current_revision, agent_id, task_id,
                knowledge_namespace, access_policy_sha256, run_state,
                attempt, state_revision, lease_expires_at, release_id,
                next_evaluation_at, last_evaluation_id, state_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._subscription_state_values(record),
        )

    def _update_subscription_state_unlocked(
        self,
        current: AgentRecallSubscriptionRecord,
        updated: AgentRecallSubscriptionRecord,
    ) -> None:
        cursor = self._connection.execute(
            """
            UPDATE cayu_agent_recall_subscription_states
            SET current_revision = ?, agent_id = ?, task_id = ?,
                knowledge_namespace = ?, access_policy_sha256 = ?, run_state = ?,
                attempt = ?, state_revision = ?, lease_expires_at = ?,
                release_id = ?, next_evaluation_at = ?, last_evaluation_id = ?,
                state_json = ?, updated_at = ?
            WHERE subscription_id = ? AND state_revision = ?
            """,
            (
                *self._subscription_state_values(updated)[1:],
                updated.subscription.subscription_id,
                current.state_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise AgentRecallSubscriptionConflict("stale_subscription_state")

    @staticmethod
    def _subscription_state_values(
        record: AgentRecallSubscriptionRecord,
    ) -> tuple[object, ...]:
        claim = record.claim
        lease_expires_at = (
            claim.lease_expires_at.isoformat()
            if record.run_state is AgentRecallSubscriptionRunState.CLAIMED and claim is not None
            else None
        )
        subscription = record.subscription
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
            record.next_evaluation_at.isoformat(),
            record.last_evaluation_id,
            _subscription_state_document(record),
            record.updated_at.isoformat(),
        )

    def _insert_subscription_delivery_unlocked(self, delivery: AgentRecallDelivery) -> None:
        key = delivery.key()
        if self._load_delivery_unlocked(delivery.delivery_id) is not None:
            raise AgentRecallSubscriptionConflict("delivery_id_reused")
        occupied = self._connection.execute(
            "SELECT delivery_id FROM cayu_agent_recall_deliveries "
            "WHERE agent_id = ? AND task_id = ? AND knowledge_namespace = ? "
            "AND access_policy_sha256 = ? AND checkpoint_stream_id = ? "
            "AND checkpoint_revision = ?",
            (*key.sort_key(), delivery.checkpoint.revision),
        ).fetchone()
        if occupied is not None:
            raise AgentRecallSubscriptionConflict("checkpoint_delivery_exists")
        operation = self._connection.execute(
            "SELECT delivery_id FROM cayu_agent_recall_deliveries WHERE operation_id = ?",
            (delivery.operation_id,),
        ).fetchone()
        if operation is not None:
            raise AgentRecallSubscriptionConflict("delivery_operation_reused")
        checkpoint = self._connection.execute(
            "SELECT 1 FROM cayu_agent_recall_checkpoints WHERE operation_id = ?",
            (delivery.operation_id,),
        ).fetchone()
        if checkpoint is not None:
            raise AgentRecallSubscriptionConflict("checkpoint_committed_without_delivery")
        self._advance_checkpoint_unlocked(
            delivery.checkpoint,
            expected_revision=delivery.expected_checkpoint_revision,
        )
        self._connection.execute(
            """
            INSERT INTO cayu_agent_recall_deliveries (
                delivery_id, operation_id, agent_id, task_id,
                knowledge_namespace, access_policy_sha256,
                checkpoint_stream_id,
                checkpoint_revision, processing_result_sha256,
                delivery_json, staged_at, processing_schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _document(delivery),
                delivery.staged_at.isoformat(),
                str(delivery.processing_result["schema_version"]),
            ),
        )
        self._insert_delivery_state_unlocked(
            AgentRecallDeliveryRecord(delivery=delivery, updated_at=delivery.staged_at)
        )

    def _load_context_unlocked(
        self,
        task_id: str,
        *,
        revision: int | None,
    ) -> AgentWorkContext | None:
        if revision is None:
            row = self._connection.execute(
                """
                SELECT revision.record_json
                FROM cayu_agent_work_context_heads AS head
                JOIN cayu_agent_work_context_revisions AS revision
                  ON revision.task_id = head.task_id
                 AND revision.revision = head.current_revision
                WHERE head.task_id = ?
                """,
                (task_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT record_json
                FROM cayu_agent_work_context_revisions
                WHERE task_id = ? AND revision = ?
                """,
                (task_id, revision),
            ).fetchone()
        return None if row is None else _parse_context(str(row["record_json"]))

    def _load_publication_unlocked(
        self,
        operation_id: str,
    ) -> AgentWorkContextPublicationReceipt | None:
        row = self._connection.execute(
            """
            SELECT receipt_json
            FROM cayu_agent_work_context_publications
            WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        return None if row is None else _parse_publication(str(row["receipt_json"]))

    def _load_checkpoint_unlocked(
        self,
        key: AgentRecallCheckpointKey,
        *,
        revision: int | None,
    ) -> AgentRecallCheckpoint | None:
        if revision is None:
            row = self._connection.execute(
                """
                SELECT checkpoint.record_json
                FROM cayu_agent_recall_checkpoint_heads AS head
                JOIN cayu_agent_recall_checkpoints AS checkpoint
                  ON checkpoint.agent_id = head.agent_id
                 AND checkpoint.task_id = head.task_id
                 AND checkpoint.knowledge_namespace = head.knowledge_namespace
                 AND checkpoint.access_policy_sha256 = head.access_policy_sha256
                 AND checkpoint.checkpoint_stream_id = head.checkpoint_stream_id
                 AND checkpoint.revision = head.current_revision
                WHERE head.agent_id = ? AND head.task_id = ?
                  AND head.knowledge_namespace = ? AND head.access_policy_sha256 = ?
                  AND head.checkpoint_stream_id = ?
                """,
                key.sort_key(),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT record_json
                FROM cayu_agent_recall_checkpoints
                WHERE agent_id = ? AND task_id = ?
                  AND knowledge_namespace = ? AND access_policy_sha256 = ?
                  AND checkpoint_stream_id = ?
                  AND revision = ?
                """,
                (*key.sort_key(), revision),
            ).fetchone()
        return None if row is None else _parse_checkpoint(str(row["record_json"]))

    def _load_delivery_unlocked(
        self,
        delivery_id: str,
    ) -> AgentRecallDeliveryRecord | None:
        row = self._connection.execute(
            """
            SELECT delivery.delivery_json,
                   delivery.operation_id AS delivery_operation_id,
                   delivery.processing_result_sha256 AS delivery_result_sha256,
                   delivery.staged_at AS delivery_staged_at,
                   delivery.processing_schema_version AS delivery_processing_schema_version,
                   checkpoint.record_json AS checkpoint_json,
                   state.*
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
            WHERE delivery.delivery_id = ?
            """,
            (delivery_id,),
        ).fetchone()
        return None if row is None else self._delivery_record_from_row(row)

    @staticmethod
    def _delivery_record_from_row(row) -> AgentRecallDeliveryRecord:
        if row["checkpoint_json"] is None:
            raise RuntimeError("SQLite recall delivery conflicts with its checkpoint.")
        record = _parse_delivery_record(
            str(row["delivery_json"]),
            str(row["state_json"]),
        )
        checkpoint = _parse_checkpoint(str(row["checkpoint_json"]))
        lease_expires_at = (
            record.claim.lease_expires_at.isoformat()
            if record.state is AgentRecallDeliveryState.CLAIMED and record.claim is not None
            else None
        )
        release_id = None if record.release is None else record.release.release_id
        acknowledgement_id = (
            None if record.acknowledgement is None else record.acknowledgement.acknowledgement_id
        )
        if (
            checkpoint != record.delivery.checkpoint
            or row["delivery_operation_id"] != record.delivery.operation_id
            or row["delivery_result_sha256"] != record.delivery.processing_result_sha256
            or row["delivery_staged_at"] != record.delivery.staged_at.isoformat()
            or row["delivery_processing_schema_version"]
            != record.delivery.processing_result["schema_version"]
            or row["delivery_id"] != record.delivery.delivery_id
            or (
                row["agent_id"],
                row["task_id"],
                row["knowledge_namespace"],
                row["access_policy_sha256"],
                row["checkpoint_stream_id"],
            )
            != record.delivery.key().sort_key()
            or row["checkpoint_revision"] != record.delivery.checkpoint.revision
            or row["state"] != record.state.value
            or row["attempt"] != record.attempt
            or row["state_revision"] != record.state_revision
            or row["lease_expires_at"] != lease_expires_at
            or row["release_id"] != release_id
            or row["acknowledgement_id"] != acknowledgement_id
            or row["updated_at"] != record.updated_at.isoformat()
        ):
            raise RuntimeError("SQLite recall-delivery indexes conflict with durable state.")
        return record

    def _advance_checkpoint_unlocked(
        self,
        checkpoint: AgentRecallCheckpoint,
        *,
        expected_revision: int | None,
    ) -> None:
        key = checkpoint.key()
        work_context = self._load_context_unlocked(
            checkpoint.task_id,
            revision=checkpoint.work_context_revision,
        )
        current_work_context = self._load_context_unlocked(
            checkpoint.task_id,
            revision=None,
        )
        validate_agent_recall_checkpoint_work_context(
            checkpoint,
            work_context,
            current_work_context,
        )
        current = self._load_checkpoint_unlocked(key, revision=None)
        validate_agent_recall_checkpoint_advance(
            checkpoint,
            expected_revision,
            current,
        )
        self._connection.execute(
            """
            INSERT INTO cayu_agent_recall_checkpoints (
                agent_id, task_id, knowledge_namespace, access_policy_sha256,
                checkpoint_stream_id, revision,
                work_context_revision, work_context_sha256,
                knowledge_sequence, index_readiness_sequence, processing_mode,
                knowledge_high_water_sequence,
                index_readiness_high_water_sequence,
                processing_id, operation_id, record_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _document(checkpoint),
                checkpoint.updated_at.isoformat(),
            ),
        )
        if current is None:
            self._connection.execute(
                """
                INSERT INTO cayu_agent_recall_checkpoint_heads (
                    agent_id, task_id, knowledge_namespace,
                    access_policy_sha256, checkpoint_stream_id, current_revision
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (*key.sort_key(), checkpoint.revision),
            )
            return
        cursor = self._connection.execute(
            """
            UPDATE cayu_agent_recall_checkpoint_heads
            SET current_revision = ?
            WHERE agent_id = ? AND task_id = ?
              AND knowledge_namespace = ? AND access_policy_sha256 = ?
              AND checkpoint_stream_id = ?
              AND current_revision = ?
            """,
            (
                checkpoint.revision,
                *key.sort_key(),
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise AgentWorkContextConflict("stale_checkpoint_revision")

    def _insert_delivery_state_unlocked(self, record: AgentRecallDeliveryRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_agent_recall_delivery_states (
                delivery_id, agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id,
                checkpoint_revision, state, attempt,
                state_revision, lease_expires_at, release_id,
                acknowledgement_id, state_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._delivery_state_values(record),
        )

    def _update_delivery_state_unlocked(
        self,
        current: AgentRecallDeliveryRecord,
        updated: AgentRecallDeliveryRecord,
    ) -> None:
        cursor = self._connection.execute(
            """
            UPDATE cayu_agent_recall_delivery_states
            SET agent_id = ?, task_id = ?, knowledge_namespace = ?,
                access_policy_sha256 = ?, checkpoint_stream_id = ?,
                checkpoint_revision = ?, state = ?,
                attempt = ?, state_revision = ?, lease_expires_at = ?,
                release_id = ?, acknowledgement_id = ?, state_json = ?,
                updated_at = ?
            WHERE delivery_id = ? AND state_revision = ?
            """,
            (
                *self._delivery_state_values(updated)[1:],
                updated.delivery.delivery_id,
                current.state_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise AgentRecallDeliveryConflict("stale_delivery_state")

    @staticmethod
    def _delivery_state_values(record: AgentRecallDeliveryRecord) -> tuple[object, ...]:
        claim = record.claim
        lease_expires_at = (
            claim.lease_expires_at.isoformat()
            if record.state is AgentRecallDeliveryState.CLAIMED and claim is not None
            else None
        )
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
            _state_document(record),
            record.updated_at.isoformat(),
        )

    def _insert_context_unlocked(
        self,
        context: AgentWorkContext,
        *,
        expected_revision: int | None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_agent_work_context_revisions (
                task_id, revision, content_sha256, operation_id,
                record_json, published_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                context.task_id,
                context.revision,
                context.content_sha256,
                context.operation_id,
                _document(context),
                context.published_at.isoformat(),
            ),
        )
        if expected_revision is None:
            self._connection.execute(
                """
                INSERT INTO cayu_agent_work_context_heads (
                    task_id, current_revision
                ) VALUES (?, ?)
                """,
                (context.task_id, context.revision),
            )
            return
        cursor = self._connection.execute(
            """
            UPDATE cayu_agent_work_context_heads
            SET current_revision = ?
            WHERE task_id = ? AND current_revision = ?
            """,
            (
                context.revision,
                context.task_id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise AgentWorkContextConflict("stale_context_revision")


__all__ = ["SQLiteAgentWorkContextStore"]
