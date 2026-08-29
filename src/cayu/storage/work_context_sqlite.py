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
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextPublicationReceipt,
    AgentWorkContextStore,
    _acknowledge_agent_recall_delivery_record,
    _agent_recall_delivery_release,
    _bounded_identity,
    _claim_agent_recall_delivery_record,
    _positive_revision,
    _release_agent_recall_delivery_record,
    _renew_agent_recall_delivery_record,
    _require_replayable_delivery_claim_attempt,
    _utc,
    _validate_delivery_lease_seconds,
    agent_recall_delivery_claim_request_sha256,
    agent_work_context_publication_request_sha256,
    copy_agent_recall_checkpoint,
    copy_agent_recall_checkpoint_key,
    copy_agent_recall_delivery,
    copy_agent_recall_delivery_claim,
    copy_agent_recall_delivery_record,
    copy_agent_work_context,
    copy_agent_work_context_publication_receipt,
    validate_agent_recall_checkpoint_advance,
    validate_agent_recall_checkpoint_work_context,
    validate_agent_work_context_publication,
)

_SQLITE_MIN_REQUIRED_REVISION = 71


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
                        checkpoint_revision, processing_result_sha256,
                        delivery_json, staged_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        delivery.delivery_id,
                        delivery.operation_id,
                        delivery.agent_id,
                        delivery.task_id,
                        delivery.knowledge_namespace,
                        delivery.access_policy_sha256,
                        delivery.checkpoint.revision,
                        delivery.processing_result_sha256,
                        _document(delivery),
                        delivery.staged_at.isoformat(),
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
                     AND checkpoint.revision = delivery.checkpoint_revision
                     AND checkpoint.operation_id = delivery.operation_id
                    WHERE state.agent_id = ? AND state.task_id = ?
                      AND state.knowledge_namespace = ?
                      AND state.access_policy_sha256 = ?
                      AND state.state != 'acknowledged'
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

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

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
                 AND checkpoint.revision = head.current_revision
                WHERE head.agent_id = ? AND head.task_id = ?
                  AND head.knowledge_namespace = ? AND head.access_policy_sha256 = ?
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
            or row["delivery_id"] != record.delivery.delivery_id
            or (
                row["agent_id"],
                row["task_id"],
                row["knowledge_namespace"],
                row["access_policy_sha256"],
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
                revision, work_context_revision, work_context_sha256,
                knowledge_sequence, index_readiness_sequence, processing_mode,
                knowledge_high_water_sequence,
                index_readiness_high_water_sequence,
                processing_id, operation_id, record_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.agent_id,
                checkpoint.task_id,
                checkpoint.knowledge_namespace,
                checkpoint.access_policy_sha256,
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
                    access_policy_sha256, current_revision
                ) VALUES (?, ?, ?, ?, ?)
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
                access_policy_sha256, checkpoint_revision, state, attempt,
                state_revision, lease_expires_at, release_id,
                acknowledgement_id, state_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                access_policy_sha256 = ?, checkpoint_revision = ?, state = ?,
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
