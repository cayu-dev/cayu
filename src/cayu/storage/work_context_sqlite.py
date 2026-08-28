"""SQLite durability for agent work contexts and recall checkpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from cayu._clock import utc_clock
from cayu._validation import require_nonblank
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.work_context import (
    AgentRecallCheckpoint,
    AgentRecallCheckpointKey,
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextPublicationReceipt,
    AgentWorkContextStore,
    _bounded_identity,
    _positive_revision,
    agent_work_context_publication_request_sha256,
    copy_agent_recall_checkpoint,
    copy_agent_recall_checkpoint_key,
    copy_agent_work_context,
    copy_agent_work_context_publication_receipt,
    validate_agent_recall_checkpoint_advance,
    validate_agent_recall_checkpoint_work_context,
    validate_agent_work_context_publication,
)

_SQLITE_MIN_REQUIRED_REVISION = 69


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
        key = checkpoint.key()
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
                else:
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
