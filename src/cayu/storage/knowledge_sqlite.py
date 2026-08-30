from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from cayu.knowledge_maintenance_persistence import (
        KnowledgeMaintenanceAcceptedPlan,
        KnowledgeMaintenanceProposalPublication,
        KnowledgeMaintenanceProposalPublicationReceipt,
    )

from cayu._clock import utc_clock
from cayu._validation import (
    copy_label_map,
    require_nonblank,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KNOWLEDGE_CHUNK_TEXT_PROJECTION,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActivationAuthority,
    KnowledgeActivationConflict,
    KnowledgeActivationReceipt,
    KnowledgeActivationSource,
    KnowledgeActorType,
    KnowledgeChange,
    KnowledgeChangeBatch,
    KnowledgeChangeClaim,
    KnowledgeChangeConsumerConflict,
    KnowledgeChangeConsumerState,
    KnowledgeChangeKind,
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEmbeddingIdentity,
    KnowledgeEntry,
    KnowledgeEntryReadLimitExceeded,
    KnowledgeEvidence,
    KnowledgeEvidenceConflict,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceResult,
    KnowledgeEvidenceRole,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeIndexReadiness,
    KnowledgeIndexReadinessBatch,
    KnowledgeIndexReadinessConflict,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    KnowledgeLineageCurrentness,
    KnowledgeLineageQuery,
    KnowledgeLineageResult,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeMaintenanceConflict,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceOutcome,
    KnowledgeMaintenanceProposal,
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    KnowledgeQuery,
    KnowledgeRelation,
    KnowledgeRelationConflict,
    KnowledgeRelationDirection,
    KnowledgeRelationKind,
    KnowledgeRelationPublicationReceipt,
    KnowledgeRelationQuery,
    KnowledgeRelationResult,
    KnowledgeReviewApproval,
    KnowledgeRevisionConflict,
    KnowledgeRevisionRef,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    _activation_receipt_matches,
    _bounded_knowledge_evidence,
    _bounded_knowledge_index_identity,
    _bounded_knowledge_lineage_result,
    _bounded_knowledge_relation_result,
    _copy_chunks_for_revision,
    _copy_entry_evidence,
    _copy_evidence_for_revision,
    _decode_knowledge_lineage_cursor,
    _decode_knowledge_relation_cursor,
    _initialize_knowledge_change_consumer_state,
    _knowledge_access_scope_sha256,
    _knowledge_access_snapshot,
    _knowledge_access_snapshot_json,
    _knowledge_activation_receipt_json,
    _knowledge_activation_retirement,
    _knowledge_activation_retirement_json,
    _knowledge_change_audiences,
    _knowledge_change_claim_sha256,
    _knowledge_change_identity,
    _knowledge_change_lease_seconds,
    _knowledge_change_now,
    _knowledge_chunk_content_hash,
    _knowledge_embedding_identity_sha256,
    _knowledge_entry_id,
    _knowledge_index_readiness_update_sha256,
    _knowledge_lineage_link,
    _knowledge_lineage_query_fingerprint,
    _knowledge_maintenance_access_snapshot,
    _knowledge_maintenance_access_snapshot_json,
    _knowledge_maintenance_identity,
    _knowledge_maintenance_successors,
    _knowledge_publication_operation_id,
    _knowledge_relation_access_snapshot,
    _knowledge_relation_access_snapshot_json,
    _knowledge_relation_change_audiences,
    _knowledge_relation_identity,
    _knowledge_relation_query_fingerprint,
    _knowledge_scope_allows_activation_receipt,
    _knowledge_scope_allows_lineage_endpoint,
    _knowledge_scope_allows_maintenance_access_snapshot,
    _knowledge_scope_allows_relation_access_snapshot,
    _knowledge_scope_allows_snapshot,
    _KnowledgeActivationRetirement,
    _KnowledgeMaintenanceAccessSnapshot,
    _KnowledgeRelationAccessSnapshot,
    _next_knowledge_revision,
    _parse_knowledge_access_snapshot_json,
    _parse_knowledge_activation_retirement_json,
    _parse_knowledge_maintenance_access_snapshot_json,
    _parse_knowledge_relation_access_snapshot_json,
    _prepare_review_approval_receipts,
    _replay_review_approval_from_receipts,
    _require_knowledge_activation_retirement_access,
    _require_knowledge_activation_retirement_capacity,
    _require_knowledge_entry_access,
    _require_knowledge_maintenance_current_entries,
    _require_knowledge_maintenance_current_replacement,
    _require_knowledge_maintenance_publication_boundary,
    _require_knowledge_maintenance_source_evidence,
    _require_knowledge_successor_access,
    _validate_activation_publication_material,
    _validate_knowledge_change_limit,
    _validate_knowledge_change_sequence,
    _validate_knowledge_index_readiness_limit,
    _validate_knowledge_index_readiness_transition,
    _validate_knowledge_index_sequence,
    _validate_knowledge_maintenance_record,
    _validate_knowledge_maintenance_replay,
    _validate_knowledge_publication_replay,
    _validate_knowledge_relation_publication_replay,
    _validate_knowledge_revision,
    _validate_knowledge_search_frontier,
    _validate_review_approval_authority,
    _validate_review_approval_scope,
    _validate_revision_append,
    _validate_revision_successor,
    copy_knowledge_access_scope,
    copy_knowledge_activation_authority,
    copy_knowledge_activation_receipt,
    copy_knowledge_change_claim,
    copy_knowledge_change_consumer_state,
    copy_knowledge_chunk,
    copy_knowledge_embedding_identity,
    copy_knowledge_entry,
    copy_knowledge_index_readiness_update,
    copy_knowledge_lineage_query,
    copy_knowledge_list_query,
    copy_knowledge_maintenance_decision,
    copy_knowledge_maintenance_decision_receipt,
    copy_knowledge_maintenance_proposal,
    copy_knowledge_publication_receipt,
    copy_knowledge_query,
    copy_knowledge_relation_publication_receipt,
    copy_knowledge_relation_query,
    copy_knowledge_revision_refs,
    knowledge_entry_payload_bytes,
    prepare_knowledge_maintenance_decision,
    prepare_knowledge_publication,
    prepare_knowledge_relations,
)

_SEARCH_TOKEN_RE = re.compile(r"\w+")
_SEARCH_PAGE_SIZE = 500
_KNOWLEDGE_FTS_TABLE = "cayu_knowledge_chunks_fts"
_EXACT_REVISION_FTS_TABLE = "cayu_knowledge_exact_revision_fts"
_CHUNK_ID_LOOKUP_BATCH_SIZE = 400
_EVIDENCE_ID_LOOKUP_BATCH_SIZE = 400
_MAINTENANCE_REJECTED_REPLACEMENT_RETIREMENT_TRANSITIONS = frozenset(
    {
        (KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED),
        (KnowledgeStatus.PENDING, KnowledgeStatus.DELETED),
        (KnowledgeStatus.ARCHIVED, KnowledgeStatus.DELETED),
    }
)
_SQLITE_MIN_REQUIRED_REVISION = 75


class SQLiteKnowledgeStore(KnowledgeStore):
    """SQLite-backed durable knowledge store with FTS5 keyword search."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        access_scope: KnowledgeAccessScope | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteKnowledgeStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._default_access_scope = (
            None if access_scope is None else copy_knowledge_access_scope(access_scope)
        )
        self._clock = utc_clock(clock)
        self._schema_mode = schema_mode
        self._lock = asyncio.Lock()
        self._connection = sqlite_support.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection,
                schema_mode,
                app_min_supported=_SQLITE_MIN_REQUIRED_REVISION,
            )
            self._connection.execute(
                f"""
                CREATE VIRTUAL TABLE temp.{_EXACT_REVISION_FTS_TABLE}
                USING fts5(
                    entry_id UNINDEXED,
                    entry_revision UNINDEXED,
                    chunk_id UNINDEXED,
                    title,
                    text
                )
                """
            )
        except BaseException:
            self._connection.close()
            raise

    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        scope = self._operation_access_scope(access_scope)
        entry = copy_knowledge_entry(entry)
        _validate_revision_append(entry, expected_revision=None)
        _require_knowledge_entry_access(scope, entry, operation="create_entry")
        copied_chunks = (
            [_default_chunk_for_entry(entry)]
            if chunks is None
            else _copy_entry_chunks(entry.id, entry.revision, chunks)
        )
        copied_evidence = _copy_entry_evidence(
            entry.id,
            entry.revision,
            evidence or [],
            chunks=copied_chunks,
        )
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                existing_entry = self._load_entry_unlocked(entry.id)
                if existing_entry is not None:
                    _require_knowledge_entry_access(
                        scope,
                        existing_entry,
                        operation="create_entry",
                    )
                    raise KnowledgeRevisionConflict(
                        entry.id,
                        expected_revision=None,
                        actual_revision=existing_entry.revision,
                    )
                retirement = self._load_activation_retirement_unlocked(entry.id)
                if retirement is not None:
                    _require_knowledge_activation_retirement_access(
                        scope,
                        retirement,
                        operation="create_entry",
                    )
                    raise KnowledgePublicationConflict("entry_retired")
                self._require_chunk_ids_available_unlocked(
                    copied_chunks,
                    access_scope=scope,
                    operation="create_entry",
                )
                self._require_evidence_ids_available_unlocked(
                    copied_evidence,
                    access_scope=scope,
                    operation="create_entry",
                )
                self._insert_entry_unlocked(entry)
                self._insert_chunks_unlocked(entry, copied_chunks)
                self._insert_evidence_unlocked(copied_evidence)
                self._insert_change_unlocked(
                    before_entry=None,
                    after_entry=entry,
                    kind=KnowledgeChangeKind.CREATED,
                )
            return copy_knowledge_entry(entry)

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        scope = self._operation_access_scope(access_scope)
        entry = copy_knowledge_entry(entry)
        _validate_revision_append(entry, expected_revision=expected_revision)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                self._append_revision_unlocked(
                    entry,
                    expected_revision=expected_revision,
                    chunks=chunks,
                    evidence=evidence,
                    access_scope=scope,
                    operation="append_entry_revision",
                    change_kind=KnowledgeChangeKind.REVISION_APPENDED,
                    inherit_evidence=False,
                )
        return copy_knowledge_entry(entry)

    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if max_bytes is not None:
            _validate_positive_int(max_bytes, "max_bytes")
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                access_now = datetime.now(UTC)
                if max_bytes is None:
                    entry = self._load_entry_in_scope_unlocked(
                        clean_id,
                        scope,
                        revision=revision,
                        access_now=access_now,
                    )
                    return None if entry is None else copy_knowledge_entry(entry)
                payload_descriptor = self._load_entry_payload_bytes_in_scope_unlocked(
                    clean_id,
                    scope,
                    revision=revision,
                    access_now=access_now,
                )
                if payload_descriptor is None:
                    return None
                selected_revision, stored_payload_bytes = payload_descriptor
                if stored_payload_bytes > max_bytes:
                    raise KnowledgeEntryReadLimitExceeded(
                        clean_id,
                        revision=selected_revision,
                        payload_bytes=stored_payload_bytes,
                        max_bytes=max_bytes,
                    )
                entry = self._load_entry_in_scope_unlocked(
                    clean_id,
                    scope,
                    revision=revision,
                    access_now=access_now,
                )
                if entry is None:
                    raise RuntimeError(
                        "Knowledge entry disappeared inside a stable SQLite read snapshot."
                    )
                actual_bytes = knowledge_entry_payload_bytes(entry)
                if actual_bytes != stored_payload_bytes:
                    raise RuntimeError(
                        "Stored knowledge entry payload size does not match canonical content."
                    )
                return copy_knowledge_entry(entry)

    async def transition_entry_status(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        from_status: KnowledgeStatus,
        to_status: KnowledgeStatus,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeEntry:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if not isinstance(from_status, KnowledgeStatus):
            raise ValueError("from_status must be a KnowledgeStatus.")
        if not isinstance(to_status, KnowledgeStatus):
            raise ValueError("to_status must be a KnowledgeStatus.")
        expected_namespace = (
            require_clean_nonblank(expected_namespace, "expected_namespace")
            if expected_namespace is not None
            else None
        )
        expected_labels = copy_label_map(expected_labels or {}, "expected_labels")
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                entry = self._load_entry_unlocked(clean_id)
                if entry is None:
                    raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
                _require_knowledge_entry_access(
                    scope,
                    entry,
                    operation="transition_entry_status",
                )
                if entry.revision != expected_revision:
                    raise KnowledgeRevisionConflict(
                        clean_id,
                        expected_revision=expected_revision,
                        actual_revision=entry.revision,
                    )
                if expected_namespace is not None and entry.namespace != expected_namespace:
                    raise ValueError(
                        f"Knowledge entry {clean_id!r} does not match expected namespace."
                    )
                for key, value in expected_labels.items():
                    if entry.labels.get(key) != value:
                        raise ValueError(
                            f"Knowledge entry {clean_id!r} does not match expected labels."
                        )
                if entry.status is not from_status:
                    raise ValueError(
                        f"Knowledge entry {clean_id!r} is {entry.status.value!r}, "
                        f"not {from_status.value!r}."
                    )
                target = entry.model_copy(
                    update={
                        "revision": _next_knowledge_revision(expected_revision),
                        "status": to_status,
                        "updated_at": max(
                            datetime.now(UTC),
                            entry.created_at,
                            entry.updated_at,
                        ),
                    }
                )
                self._append_revision_unlocked(
                    target,
                    expected_revision=expected_revision,
                    chunks=None,
                    evidence=None,
                    access_scope=scope,
                    operation="transition_entry_status",
                    change_kind=(
                        KnowledgeChangeKind.TOMBSTONED
                        if to_status is KnowledgeStatus.DELETED
                        else KnowledgeChangeKind.STATUS_TRANSITIONED
                    ),
                    inherit_evidence=True,
                )
                return copy_knowledge_entry(target)

    async def delete_entry(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if type(hard) is not bool:
            raise ValueError("`hard` must be a boolean.")
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                entry = self._load_entry_unlocked(clean_id)
                if entry is None:
                    if not hard:
                        return None
                    retirement = self._load_activation_retirement_unlocked(clean_id)
                    if retirement is None:
                        return None
                    _require_knowledge_activation_retirement_access(
                        scope,
                        retirement,
                        operation="delete_entry",
                    )
                    if retirement.entry_revision != expected_revision:
                        raise KnowledgeRevisionConflict(
                            clean_id,
                            expected_revision=expected_revision,
                            actual_revision=retirement.entry_revision,
                        )
                    receipt_count = int(
                        self._connection.execute(
                            "SELECT COUNT(*) FROM cayu_knowledge_activation_receipts "
                            "WHERE entry_id = ?",
                            (clean_id,),
                        ).fetchone()[0]
                    )
                    if receipt_count < 1:
                        raise KnowledgeActivationConflict("malformed_retirement")
                    self._connection.execute(
                        "DELETE FROM cayu_knowledge_activation_receipts WHERE entry_id = ?",
                        (clean_id,),
                    )
                    self._connection.execute(
                        "DELETE FROM cayu_knowledge_activation_retirements WHERE entry_id = ?",
                        (clean_id,),
                    )
                    return None
                if self._load_activation_retirement_unlocked(clean_id) is not None:
                    raise KnowledgeActivationConflict("malformed_retirement")
                _require_knowledge_entry_access(scope, entry, operation="delete_entry")
                if entry.revision != expected_revision:
                    raise KnowledgeRevisionConflict(
                        clean_id,
                        expected_revision=expected_revision,
                        actual_revision=entry.revision,
                    )
                if hard:
                    self._require_maintenance_replacement_mutation_allowed_unlocked(
                        entry_id=clean_id,
                        entry_revision=entry.revision,
                        preserve_history=True,
                    )
                    self._insert_change_unlocked(
                        before_entry=entry,
                        after_entry=None,
                        kind=KnowledgeChangeKind.HARD_DELETED,
                    )
                    self._connection.execute(
                        "DELETE FROM cayu_knowledge_activation_receipts WHERE entry_id = ?",
                        (clean_id,),
                    )
                    self._delete_chunks_unlocked(clean_id)
                    self._connection.execute(
                        "DELETE FROM cayu_knowledge_entries WHERE id = ?",
                        (clean_id,),
                    )
                    return copy_knowledge_entry(entry)
                target = entry.model_copy(
                    update={
                        "revision": _next_knowledge_revision(expected_revision),
                        "status": KnowledgeStatus.DELETED,
                        "updated_at": max(
                            datetime.now(UTC),
                            entry.created_at,
                            entry.updated_at,
                        ),
                    }
                )
                self._append_revision_unlocked(
                    target,
                    expected_revision=expected_revision,
                    chunks=None,
                    evidence=None,
                    access_scope=scope,
                    operation="delete_entry",
                    change_kind=KnowledgeChangeKind.TOMBSTONED,
                    inherit_evidence=True,
                )
                return copy_knowledge_entry(target)

    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        scope = self._operation_access_scope(access_scope)
        cutoff = _knowledge_change_now(now)
        access_sql, access_params = _knowledge_access_scope_filter_sql(
            scope,
            now=cutoff,
        )
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                rows = self._connection.execute(
                    "SELECT id FROM cayu_knowledge_current_entries "
                    "AS e WHERE expires_at IS NOT NULL AND expires_at <= ? "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM cayu_knowledge_maintenance_proposals AS proposal "
                    "WHERE proposal.replacement_entry_id = e.id"
                    ") "
                    f"{access_sql} ORDER BY e.id COLLATE BINARY",
                    [sqlite_support.format_datetime(cutoff), *access_params],
                ).fetchall()
                expired_ids = [str(row["id"]) for row in rows]
                if not expired_ids:
                    return 0
                # FTS is a virtual table (no FK cascade), so clear chunks/FTS explicitly; the
                # entries DELETE then cascades to labels/aspects/impact_targets.
                retired_at = datetime.now(UTC)
                for entry_id in expired_ids:
                    entry = self._load_entry_unlocked(entry_id)
                    if entry is None:
                        raise RuntimeError(
                            "SQLite knowledge entry disappeared during expiration pruning."
                        )
                    self._insert_change_unlocked(
                        before_entry=entry,
                        after_entry=None,
                        kind=KnowledgeChangeKind.EXPIRED,
                    )
                    has_activation_receipt = bool(
                        self._connection.execute(
                            "SELECT EXISTS("
                            "SELECT 1 FROM cayu_knowledge_activation_receipts "
                            "WHERE entry_id = ? LIMIT 1)",
                            (entry_id,),
                        ).fetchone()[0]
                    )
                    if has_activation_receipt:
                        if self._load_activation_retirement_unlocked(entry_id) is not None:
                            raise KnowledgeActivationConflict("malformed_retirement")
                        self._insert_activation_retirement_unlocked(
                            _knowledge_activation_retirement(entry, retired_at=retired_at)
                        )
                    self._delete_chunks_unlocked(entry_id)
                self._connection.executemany(
                    "DELETE FROM cayu_knowledge_entries WHERE id = ?",
                    [(entry_id,) for entry_id in expired_ids],
                )
            return len(expired_ids)

    async def publish_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
        activation_authority: KnowledgeActivationAuthority | None = None,
    ) -> KnowledgePublicationReceipt:
        scope = self._operation_access_scope(access_scope)
        (
            operation_id,
            copied_entry,
            copied_chunks,
            copied_evidence,
            request_sha256,
        ) = prepare_knowledge_publication(
            entry,
            chunks,
            evidence=evidence,
            operation_id=operation_id,
            expected_revision=expected_revision,
            activation_authority=activation_authority,
        )
        _require_knowledge_entry_access(scope, copied_entry, operation="publish_entry_revision")
        copied_authority = (
            None
            if activation_authority is None
            else copy_knowledge_activation_authority(activation_authority)
        )
        if copied_authority is not None:
            _validate_activation_publication_material(
                copied_authority,
                operation_id=operation_id,
                entry=copied_entry,
                chunks=copied_chunks,
                evidence=copied_evidence,
                expected_revision=expected_revision,
                access_scope=scope,
            )
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                existing_receipt = self._load_publication_receipt_unlocked(
                    operation_id,
                    access_scope=scope,
                )
                if existing_receipt is not None:
                    existing_activation = self._load_activation_receipt_unlocked(
                        operation_id,
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if (
                        existing_activation is not None
                        and existing_activation.authority.request.source
                        is KnowledgeActivationSource.REVIEW_APPROVAL
                    ):
                        raise KnowledgePublicationConflict("operation_occupied")
                    _validate_knowledge_publication_replay(
                        existing_receipt,
                        entry=copied_entry,
                        chunks=copied_chunks,
                        evidence=copied_evidence,
                        expected_revision=expected_revision,
                        request_sha256=request_sha256,
                        activation_authority=copied_authority,
                    )
                    if copied_authority is None:
                        if existing_activation is not None:
                            raise KnowledgePublicationConflict("activation_mismatch")
                    elif existing_activation is None or not _activation_receipt_matches(
                        existing_activation,
                        authority=copied_authority,
                        publication_request_sha256=request_sha256,
                        publication_committed_at=existing_receipt.committed_at,
                    ):
                        raise KnowledgePublicationConflict("activation_mismatch")
                    return copy_knowledge_publication_receipt(
                        existing_receipt,
                        replayed=True,
                    )
                existing_activation = self._load_activation_receipt_unlocked(
                    operation_id,
                    access_scope=scope,
                    deny_inaccessible=True,
                )
                if existing_activation is not None:
                    raise KnowledgePublicationConflict("operation_occupied")
                existing_entry = self._load_entry_unlocked(copied_entry.id)
                actual_revision = None if existing_entry is None else existing_entry.revision
                if existing_entry is not None:
                    _require_knowledge_entry_access(
                        scope,
                        existing_entry,
                        operation="publish_entry_revision",
                    )
                retirement = self._load_activation_retirement_unlocked(copied_entry.id)
                if retirement is not None:
                    _require_knowledge_activation_retirement_access(
                        scope,
                        retirement,
                        operation="publish_entry_revision",
                    )
                    raise KnowledgePublicationConflict("entry_retired")
                if actual_revision != expected_revision:
                    raise KnowledgeRevisionConflict(
                        copied_entry.id,
                        expected_revision=expected_revision,
                        actual_revision=actual_revision,
                    )
                if existing_entry is not None:
                    self._require_maintenance_replacement_mutation_allowed_unlocked(
                        entry_id=existing_entry.id,
                        entry_revision=existing_entry.revision,
                        current_status=existing_entry.status,
                        successor_status=copied_entry.status,
                        operation="publish_entry_revision",
                    )
                    _validate_revision_successor(existing_entry, copied_entry)
                    if copied_authority is None and self._has_activation_receipts_unlocked(
                        copied_entry.id
                    ):
                        _require_knowledge_activation_retirement_capacity(copied_entry)
                self._require_chunk_ids_available_unlocked(
                    copied_chunks,
                    access_scope=scope,
                    operation="publish_entry_revision",
                )
                self._require_evidence_ids_available_unlocked(
                    copied_evidence,
                    access_scope=scope,
                    operation="publish_entry_revision",
                )
                committed_at = datetime.now(UTC)
                receipt = KnowledgePublicationReceipt(
                    operation_id=operation_id,
                    entry_id=copied_entry.id,
                    entry_revision=copied_entry.revision,
                    expected_revision=expected_revision,
                    request_sha256=request_sha256,
                    entry_created_at=copied_entry.created_at,
                    entry_updated_at=copied_entry.updated_at,
                    committed_at=committed_at,
                )
                activation_receipt = (
                    None
                    if copied_authority is None
                    else KnowledgeActivationReceipt(
                        operation_id=operation_id,
                        entry_id=copied_entry.id,
                        entry_revision=copied_entry.revision,
                        expected_revision=expected_revision,
                        publication_request_sha256=request_sha256,
                        authority=copied_authority,
                        committed_at=committed_at,
                    )
                )
                if existing_entry is None:
                    self._insert_entry_unlocked(copied_entry)
                else:
                    assert expected_revision is not None
                    self._insert_revision_unlocked(copied_entry)
                    self._advance_current_revision_unlocked(
                        copied_entry,
                        expected_revision=expected_revision,
                    )
                self._insert_chunks_unlocked(copied_entry, copied_chunks)
                self._insert_evidence_unlocked(copied_evidence)
                self._insert_change_unlocked(
                    before_entry=existing_entry,
                    after_entry=copied_entry,
                    kind=(
                        KnowledgeChangeKind.CREATED
                        if existing_entry is None
                        else KnowledgeChangeKind.REVISION_APPENDED
                    ),
                    operation_id=operation_id,
                    committed_at=receipt.committed_at,
                )
                self._insert_publication_receipt_unlocked(receipt, copied_entry)
                if activation_receipt is not None:
                    self._insert_activation_receipt_unlocked(
                        activation_receipt,
                        access_entry=copied_entry,
                    )
            return copy_knowledge_publication_receipt(receipt)

    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        async with self._lock:
            receipt = self._load_publication_receipt_in_scope_unlocked(operation_id, scope)
        return None if receipt is None else copy_knowledge_publication_receipt(receipt)

    async def load_activation_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeActivationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                receipt = self._load_activation_receipt_unlocked(
                    operation_id,
                    access_scope=scope,
                    deny_inaccessible=False,
                )
        return None if receipt is None else copy_knowledge_activation_receipt(receipt)

    async def approve_pending_entry(
        self,
        authority: KnowledgeActivationAuthority,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeReviewApproval:
        scope = self._operation_access_scope(access_scope)
        authority = copy_knowledge_activation_authority(authority)
        request = authority.request
        _validate_review_approval_authority(authority, access_scope=scope)
        expected_namespace = (
            require_clean_nonblank(expected_namespace, "expected_namespace")
            if expected_namespace is not None
            else None
        )
        expected_labels = copy_label_map(expected_labels or {}, "expected_labels")
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                existing_receipt = self._load_activation_receipt_unlocked(
                    request.operation_id,
                    access_scope=scope,
                    deny_inaccessible=True,
                )
                if existing_receipt is not None:
                    publication = self._load_publication_receipt_unlocked(
                        request.operation_id,
                        access_scope=scope,
                    )
                    if publication is None:
                        raise KnowledgeActivationConflict("malformed_receipt")
                    approval = _replay_review_approval_from_receipts(
                        publication,
                        existing_receipt,
                        authority=authority,
                    )
                    if approval is None:
                        raise KnowledgeActivationConflict("operation_mismatch")
                    _validate_review_approval_scope(
                        approval.entry,
                        expected_namespace=expected_namespace,
                        expected_labels=expected_labels,
                    )
                    return approval
                if (
                    self._load_publication_receipt_unlocked(
                        request.operation_id,
                        access_scope=scope,
                    )
                    is not None
                ):
                    raise KnowledgeActivationConflict("operation_occupied")
                current = self._load_entry_unlocked(request.candidate_entry.id)
                if current is None:
                    raise KeyError(
                        f"Knowledge entry {request.candidate_entry.id!r} does not exist."
                    )
                _require_knowledge_entry_access(scope, current, operation="approve_pending_entry")
                if current.revision != request.expected_revision:
                    raise KnowledgeRevisionConflict(
                        current.id,
                        expected_revision=request.expected_revision,
                        actual_revision=current.revision,
                    )
                if current != request.candidate_entry:
                    raise KnowledgeActivationConflict("candidate_material_mismatch")
                if current.status is not KnowledgeStatus.PENDING:
                    raise ValueError("Reviewed approval requires a pending entry.")
                _validate_review_approval_scope(
                    current,
                    expected_namespace=expected_namespace,
                    expected_labels=expected_labels,
                )
                current_chunks = self._load_chunks_unlocked(
                    current.id,
                    revision=current.revision,
                )
                current_evidence = self._load_evidence_unlocked(
                    current.id,
                    revision=current.revision,
                )
                if (
                    list(request.chunks) != current_chunks
                    or list(request.evidence) != current_evidence
                ):
                    raise KnowledgeActivationConflict("candidate_material_mismatch")
                activated = current.model_copy(
                    update={
                        "revision": request.target_revision,
                        "status": KnowledgeStatus.ACTIVE,
                        "updated_at": max(
                            datetime.now(UTC),
                            current.created_at,
                            current.updated_at,
                        ),
                    }
                )
                _require_knowledge_activation_retirement_capacity(activated)
                target_chunks = (
                    [_default_chunk_for_entry(activated)]
                    if _has_only_default_chunk(current, current_chunks)
                    else _copy_chunks_for_revision(current_chunks, activated)
                )
                target_evidence = _copy_evidence_for_revision(
                    current_evidence,
                    entry=activated,
                    previous_chunks=current_chunks,
                    chunks=target_chunks,
                )
                committed_at = datetime.now(UTC)
                publication_receipt, receipt = _prepare_review_approval_receipts(
                    current,
                    activated,
                    target_chunks,
                    target_evidence,
                    authority,
                    committed_at=committed_at,
                )
                self._append_revision_unlocked(
                    activated,
                    expected_revision=current.revision,
                    chunks=None,
                    evidence=None,
                    access_scope=scope,
                    operation="approve_pending_entry",
                    change_kind=KnowledgeChangeKind.STATUS_TRANSITIONED,
                    inherit_evidence=True,
                    change_operation_id=request.operation_id,
                    committed_at=committed_at,
                )
                self._insert_publication_receipt_unlocked(publication_receipt, activated)
                self._insert_activation_receipt_unlocked(
                    receipt,
                    access_entry=activated,
                )
                return KnowledgeReviewApproval(entry=activated, receipt=receipt)

    async def publish_relations(
        self,
        relations: list[KnowledgeRelation],
        *,
        operation_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationPublicationReceipt:
        scope = self._operation_access_scope(access_scope)
        operation_id, copied_relations, request_sha256 = prepare_knowledge_relations(
            relations,
            operation_id=operation_id,
        )
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                existing_receipt = self._load_relation_receipt_unlocked(
                    operation_id,
                    access_scope=scope,
                    deny_inaccessible=True,
                )
                if existing_receipt is not None:
                    _validate_knowledge_relation_publication_replay(
                        existing_receipt,
                        relations=copied_relations,
                        request_sha256=request_sha256,
                    )
                    return copy_knowledge_relation_publication_receipt(
                        existing_receipt,
                        replayed=True,
                    )

                endpoint_entries: list[tuple[KnowledgeEntry, KnowledgeEntry]] = []
                for relation in copied_relations:
                    endpoints: list[KnowledgeEntry] = []
                    for reference in (relation.subject, relation.object):
                        entry = self._load_entry_in_scope_unlocked(
                            reference.entry_id,
                            scope,
                            revision=reference.revision,
                        )
                        if entry is None:
                            existing = self._load_entry_unlocked(
                                reference.entry_id,
                                revision=reference.revision,
                            )
                            if existing is None:
                                raise KnowledgeRelationConflict("endpoint_missing")
                            raise KnowledgeAccessDenied("publish_relations")
                        endpoints.append(entry)
                    endpoint_entries.append((endpoints[0], endpoints[1]))
                current_entries = self._load_entries_unlocked(
                    [
                        reference.entry_id
                        for relation in copied_relations
                        for reference in (relation.subject, relation.object)
                    ]
                )
                try:
                    endpoint_access = [
                        _knowledge_relation_access_snapshot(
                            subject_exact=subject,
                            subject_current=current_entries[relation.subject.entry_id],
                            object_exact=object_,
                            object_current=current_entries[relation.object.entry_id],
                        )
                        for relation, (subject, object_) in zip(
                            copied_relations,
                            endpoint_entries,
                            strict=True,
                        )
                    ]
                except KeyError:
                    raise KnowledgeRelationConflict("endpoint_missing") from None

                for relation in copied_relations:
                    row = self._connection.execute(
                        """
                        SELECT *
                        FROM cayu_knowledge_relations
                        WHERE id = ? OR (
                            kind = ?
                            AND subject_entry_id = ?
                            AND subject_revision = ?
                            AND object_entry_id = ?
                            AND object_revision = ?
                        )
                        LIMIT 1
                        """,
                        (
                            relation.id,
                            *_relation_semantic_row_values(relation),
                        ),
                    ).fetchone()
                    if row is not None:
                        occupied = _relation_from_row(row)
                        if not self._relation_endpoints_in_scope_unlocked(occupied, scope):
                            raise KnowledgeAccessDenied("publish_relations")
                        raise KnowledgeRelationConflict("relation_exists")
                    historic_change = self._connection.execute(
                        "SELECT sequence FROM cayu_knowledge_changes WHERE relation_id = ?",
                        (relation.id,),
                    ).fetchone()
                    if historic_change is not None:
                        if (
                            self._load_change_in_scope_unlocked(
                                int(historic_change["sequence"]),
                                scope,
                            )
                            is None
                        ):
                            raise KnowledgeAccessDenied("publish_relations")
                        raise KnowledgeRelationConflict("relation_exists")

                committed_at = self._clock()
                receipt = KnowledgeRelationPublicationReceipt(
                    operation_id=operation_id,
                    relation_ids=[relation.id for relation in copied_relations],
                    request_sha256=request_sha256,
                    committed_at=committed_at,
                )
                try:
                    self._insert_relations_unlocked(copied_relations)
                    for relation, access_snapshot in zip(
                        copied_relations,
                        endpoint_access,
                        strict=True,
                    ):
                        self._insert_relation_change_unlocked(
                            relation,
                            access_snapshot=access_snapshot,
                            operation_id=operation_id,
                            committed_at=committed_at,
                        )
                    self._insert_relation_receipt_unlocked(
                        receipt,
                        access_snapshots=endpoint_access,
                    )
                except sqlite3.IntegrityError:
                    raise KnowledgeRelationConflict("relation_exists") from None
            return copy_knowledge_relation_publication_receipt(receipt)

    async def load_relation_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationPublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_relation_identity(operation_id, "operation_id")
        async with self._lock:
            receipt = self._load_relation_receipt_unlocked(
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if receipt is None else copy_knowledge_relation_publication_receipt(receipt)

    async def read_relations(
        self,
        query: KnowledgeRelationQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationResult | None:
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_relation_query(query)
        fingerprint = _knowledge_relation_query_fingerprint(query, scope)
        cursor = _decode_knowledge_relation_cursor(query.cursor, fingerprint=fingerprint)
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                reference = self._load_entry_in_scope_unlocked(
                    query.reference.entry_id,
                    scope,
                    revision=query.reference.revision,
                )
                if reference is None:
                    return None
                relation_sql, relation_params = _sqlite_relation_query_filter_sql(query)
                access_sql, access_params = _sqlite_relation_access_scope_filter_sql(scope)
                cursor_sql = ""
                cursor_params: list[object] = []
                if cursor is not None:
                    cursor_sql = (
                        " AND (relation.created_at > ? OR "
                        "(relation.created_at = ? AND relation.id COLLATE BINARY > ?))"
                    )
                    created_at = sqlite_support.format_datetime(cursor.created_at)
                    cursor_params.extend([created_at, created_at, cursor.relation_id])
                rows = self._connection.execute(
                    f"""
                    SELECT relation.*
                    FROM cayu_knowledge_relations AS relation
                    WHERE 1 = 1
                    {relation_sql}
                    {cursor_sql}
                    {access_sql}
                    ORDER BY relation.created_at ASC, relation.id COLLATE BINARY ASC
                    LIMIT ?
                    """,
                    (
                        *relation_params,
                        *cursor_params,
                        *access_params,
                        query.limit + 1,
                    ),
                ).fetchall()
        return _bounded_knowledge_relation_result(
            query,
            [_relation_from_row(row) for row in rows],
            fingerprint=fingerprint,
        )

    async def inspect_lineage(
        self,
        query: KnowledgeLineageQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeLineageResult | None:
        return await self._inspect_lineage(
            query,
            access_scope=access_scope,
            through_sequence=None,
        )

    async def _inspect_lineage_at_change_sequence(
        self,
        query: KnowledgeLineageQuery,
        *,
        through_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeLineageResult | None:
        _validate_knowledge_change_sequence(through_sequence, "through_sequence")
        return await self._inspect_lineage(
            query,
            access_scope=access_scope,
            through_sequence=through_sequence,
        )

    async def _inspect_lineage(
        self,
        query: KnowledgeLineageQuery,
        *,
        access_scope: KnowledgeAccessScope | None,
        through_sequence: int | None,
    ) -> KnowledgeLineageResult | None:
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_lineage_query(query)
        fingerprint = _knowledge_lineage_query_fingerprint(
            query,
            scope,
            through_change_sequence=through_sequence,
        )
        cursor = _decode_knowledge_lineage_cursor(query.cursor, fingerprint=fingerprint)
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                access_now = datetime.now(UTC)
                reference_exact = self._load_entry_unlocked(
                    query.reference.entry_id,
                    revision=query.reference.revision,
                )
                reference_live = self._load_entry_unlocked(query.reference.entry_id)
                reference_current = (
                    reference_live
                    if through_sequence is None
                    else self._load_entry_at_change_sequence_unlocked(
                        query.reference.entry_id,
                        through_sequence=through_sequence,
                    )
                )
                if (
                    reference_exact is None
                    or reference_live is None
                    or reference_current is None
                    or not _knowledge_scope_allows_lineage_endpoint(
                        scope,
                        reference_exact,
                        reference_live,
                        now=access_now,
                    )
                    or not _knowledge_scope_allows_lineage_endpoint(
                        scope,
                        reference_exact,
                        reference_current,
                        now=access_now,
                    )
                ):
                    return None
                relation_sql, relation_params = _sqlite_relation_query_filter_sql(query)
                lineage_sql, lineage_params = _sqlite_lineage_filter_sql(query)
                access_sql, access_params = _sqlite_relation_access_scope_filter_sql(
                    scope,
                    allow_archived_current=True,
                    now=access_now,
                    through_change_sequence=through_sequence,
                )
                cursor_sql = ""
                cursor_params: list[object] = []
                if cursor is not None:
                    cursor_sql = (
                        " AND (relation.created_at > ? OR "
                        "(relation.created_at = ? AND relation.id COLLATE BINARY > ?))"
                    )
                    created_at = sqlite_support.format_datetime(cursor.created_at)
                    cursor_params.extend([created_at, created_at, cursor.relation_id])
                frontier_sql = ""
                frontier_params: list[object] = []
                current_join_sql = """
                    JOIN cayu_knowledge_current_entries AS subject_current
                      ON subject_current.id = relation.subject_entry_id
                    JOIN cayu_knowledge_current_entries AS object_current
                      ON object_current.id = relation.object_entry_id
                """
                current_join_params: list[object] = []
                if through_sequence is not None:
                    frontier_sql = """
                        AND EXISTS (
                            SELECT 1
                            FROM cayu_knowledge_changes AS boundary_change
                            WHERE boundary_change.relation_id = relation.id
                              AND boundary_change.sequence <= ?
                        )
                    """
                    frontier_params.append(through_sequence)
                    current_join_sql = """
                        JOIN cayu_knowledge_changes AS subject_current_change
                          ON subject_current_change.entry_id = relation.subject_entry_id
                         AND subject_current_change.kind <> 'relation_published'
                         AND subject_current_change.sequence = (
                             SELECT MAX(subject_boundary.sequence)
                             FROM cayu_knowledge_changes AS subject_boundary
                             WHERE subject_boundary.entry_id = relation.subject_entry_id
                               AND subject_boundary.kind <> 'relation_published'
                               AND subject_boundary.sequence <= ?
                         )
                         AND subject_current_change.sequence = (
                             SELECT MAX(subject_materialization.sequence)
                             FROM cayu_knowledge_changes AS subject_materialization
                             WHERE subject_materialization.entry_id =
                                       subject_current_change.entry_id
                               AND subject_materialization.entry_revision =
                                       subject_current_change.entry_revision
                               AND subject_materialization.kind <> 'relation_published'
                         )
                        JOIN cayu_knowledge_revisions AS subject_current
                          ON subject_current.entry_id = subject_current_change.entry_id
                         AND subject_current.revision = subject_current_change.entry_revision
                        JOIN cayu_knowledge_changes AS object_current_change
                          ON object_current_change.entry_id = relation.object_entry_id
                         AND object_current_change.kind <> 'relation_published'
                         AND object_current_change.sequence = (
                             SELECT MAX(object_boundary.sequence)
                             FROM cayu_knowledge_changes AS object_boundary
                             WHERE object_boundary.entry_id = relation.object_entry_id
                               AND object_boundary.kind <> 'relation_published'
                               AND object_boundary.sequence <= ?
                         )
                         AND object_current_change.sequence = (
                             SELECT MAX(object_materialization.sequence)
                             FROM cayu_knowledge_changes AS object_materialization
                             WHERE object_materialization.entry_id =
                                       object_current_change.entry_id
                               AND object_materialization.entry_revision =
                                       object_current_change.entry_revision
                               AND object_materialization.kind <> 'relation_published'
                         )
                        JOIN cayu_knowledge_revisions AS object_current
                          ON object_current.entry_id = object_current_change.entry_id
                         AND object_current.revision = object_current_change.entry_revision
                    """
                    current_join_params.extend((through_sequence, through_sequence))
                rows = self._connection.execute(
                    f"""
                    SELECT relation.id, relation.subject_entry_id,
                           relation.subject_revision, relation.object_entry_id,
                           relation.object_revision, relation.kind,
                           relation.created_at,
                           subject_current.revision AS subject_current_revision,
                           subject_current.status AS subject_current_status,
                           object_current.revision AS object_current_revision,
                           object_current.status AS object_current_status
                    FROM cayu_knowledge_relations AS relation
                    {current_join_sql}
                    WHERE 1 = 1
                    {relation_sql}
                    {lineage_sql}
                    {cursor_sql}
                    {frontier_sql}
                    {access_sql}
                    ORDER BY relation.created_at ASC, relation.id COLLATE BINARY ASC
                    LIMIT ?
                    """,
                    (
                        *current_join_params,
                        *relation_params,
                        *lineage_params,
                        *cursor_params,
                        *frontier_params,
                        *access_params,
                        query.limit + 1,
                    ),
                ).fetchall()
        links = [
            _knowledge_lineage_link(
                relation_id=str(row["id"]),
                kind=KnowledgeRelationKind(str(row["kind"])),
                subject=KnowledgeRevisionRef(
                    entry_id=str(row["subject_entry_id"]),
                    revision=int(row["subject_revision"]),
                ),
                object_=KnowledgeRevisionRef(
                    entry_id=str(row["object_entry_id"]),
                    revision=int(row["object_revision"]),
                ),
                created_at=sqlite_support.parse_datetime(str(row["created_at"])),
                reference=query.reference,
                subject_current=KnowledgeRevisionRef(
                    entry_id=str(row["subject_entry_id"]),
                    revision=int(row["subject_current_revision"]),
                ),
                subject_status=KnowledgeStatus(str(row["subject_current_status"])),
                object_current=KnowledgeRevisionRef(
                    entry_id=str(row["object_entry_id"]),
                    revision=int(row["object_current_revision"]),
                ),
                object_status=KnowledgeStatus(str(row["object_current_status"])),
            )
            for row in rows
        ]
        return _bounded_knowledge_lineage_result(
            query,
            reference_current=KnowledgeRevisionRef(
                entry_id=reference_current.id,
                revision=reference_current.revision,
            ),
            reference_status=reference_current.status,
            candidates=links,
            fingerprint=fingerprint,
        )

    async def publish_maintenance_proposal(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        evidence: list[KnowledgeEvidence],
        proposal: KnowledgeMaintenanceProposal,
        accepted_plan: KnowledgeMaintenanceAcceptedPlan,
        operation_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposalPublicationReceipt:
        from cayu.knowledge_maintenance_persistence import (
            KnowledgeMaintenanceProposalPublicationConflict,
            KnowledgeMaintenanceProposalPublicationReceipt,
            copy_knowledge_maintenance_proposal_publication_receipt,
            prepare_knowledge_maintenance_proposal_publication,
            validate_knowledge_maintenance_proposal_publication_replay,
        )

        scope = self._operation_access_scope(access_scope)
        (
            operation_id,
            copied_entry,
            copied_chunks,
            copied_evidence,
            copied_proposal,
            copied_plan,
            request_sha256,
        ) = prepare_knowledge_maintenance_proposal_publication(
            entry,
            chunks,
            evidence=evidence,
            proposal=proposal,
            accepted_plan=accepted_plan,
            operation_id=operation_id,
        )
        operation = "publish_maintenance_proposal"
        _require_knowledge_entry_access(scope, copied_entry, operation=operation)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                existing = self._load_maintenance_proposal_record_unlocked(
                    operation_id,
                    access_scope=scope,
                    deny_inaccessible=True,
                )
                if existing is not None:
                    stored_proposal, stored_plan, receipt, _ = existing
                    validate_knowledge_maintenance_proposal_publication_replay(
                        receipt,
                        operation_id=operation_id,
                        proposal=copied_proposal,
                        accepted_plan=copied_plan,
                        entry=copied_entry,
                        request_sha256=request_sha256,
                    )
                    if stored_proposal != copied_proposal or stored_plan != copied_plan:
                        raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
                    return copy_knowledge_maintenance_proposal_publication_receipt(
                        receipt,
                        replayed=True,
                    )

                occupied = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                    "WHERE proposal_id = ?",
                    (copied_proposal.id,),
                ).fetchone()
                if occupied is not None:
                    self._load_maintenance_proposal_record_unlocked(
                        str(occupied["operation_id"]),
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    raise KnowledgeMaintenanceProposalPublicationConflict("proposal_id_reuse")
                decided = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                    "WHERE proposal_id = ?",
                    (copied_proposal.id,),
                ).fetchone()
                if decided is not None:
                    self._load_maintenance_record_unlocked(
                        str(decided["operation_id"]),
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    raise KnowledgeMaintenanceProposalPublicationConflict(
                        "proposal_already_decided"
                    )

                source_entries = self._load_entries_unlocked(
                    [source.entry_id for source in copied_proposal.sources]
                )
                current_entries = dict(source_entries)
                current_entries[copied_entry.id] = copied_entry
                replacement, sources = _require_knowledge_maintenance_current_entries(
                    copied_proposal,
                    current_entries,
                    access_scope=scope,
                    operation=operation,
                )
                _require_knowledge_maintenance_publication_boundary(replacement, sources)
                _require_knowledge_maintenance_source_evidence(copied_evidence, sources)
                occupied_entry = self._load_entry_unlocked(copied_entry.id)
                if occupied_entry is not None:
                    _require_knowledge_entry_access(scope, occupied_entry, operation=operation)
                    raise KnowledgeMaintenanceProposalPublicationConflict("replacement_id_reuse")
                self._require_chunk_ids_available_unlocked(
                    copied_chunks,
                    access_scope=scope,
                    operation=operation,
                )
                self._require_evidence_ids_available_unlocked(
                    copied_evidence,
                    access_scope=scope,
                    operation=operation,
                )
                committed_at = max(self._clock(), copied_proposal.created_at)
                receipt = KnowledgeMaintenanceProposalPublicationReceipt(
                    operation_id=operation_id,
                    proposal_id=copied_proposal.id,
                    proposal_fingerprint=copied_proposal.fingerprint,
                    accepted_plan_fingerprint=copied_plan.fingerprint,
                    request_sha256=request_sha256,
                    replacement=copied_proposal.replacement,
                    committed_at=committed_at,
                )
                snapshot = _knowledge_maintenance_access_snapshot([replacement, *sources])
                self._insert_entry_unlocked(copied_entry)
                self._insert_chunks_unlocked(copied_entry, copied_chunks)
                self._insert_evidence_unlocked(copied_evidence)
                self._insert_change_unlocked(
                    before_entry=None,
                    after_entry=copied_entry,
                    kind=KnowledgeChangeKind.CREATED,
                    operation_id=operation_id,
                    committed_at=committed_at,
                )
                self._insert_maintenance_proposal_record_unlocked(
                    copied_proposal,
                    copied_plan,
                    receipt,
                    access_snapshot=snapshot,
                )
            return copy_knowledge_maintenance_proposal_publication_receipt(receipt)

    async def load_maintenance_proposal_publication(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposalPublication | None:
        from cayu.knowledge_maintenance_persistence import (
            KnowledgeMaintenanceProposalPublication,
            KnowledgeMaintenanceProposalPublicationConflict,
            KnowledgeMaintenanceProposalPublicationOutcome,
            copy_knowledge_maintenance_proposal_publication_receipt,
        )

        scope = self._operation_access_scope(access_scope)
        proposal_id = _knowledge_maintenance_identity(proposal_id, "proposal_id")
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                row = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                    "WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone()
                if row is None:
                    return None
                record = self._load_maintenance_proposal_record_unlocked(
                    str(row["operation_id"]),
                    access_scope=scope,
                    deny_inaccessible=False,
                )
                if record is None:
                    return None
                proposal, accepted_plan, receipt, publication_snapshot = record
                replacement = self._load_entry_unlocked(
                    proposal.replacement.entry_id,
                    revision=proposal.replacement.revision,
                )
                if replacement is None:
                    raise KnowledgeMaintenanceProposalPublicationConflict("replacement_missing")
                decision_row = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                    "WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone()
                if decision_row is not None:
                    try:
                        decision_record = self._load_maintenance_record_unlocked(
                            str(decision_row["operation_id"]),
                            access_scope=scope,
                            deny_inaccessible=True,
                        )
                    except (KnowledgeAccessDenied, KnowledgeMaintenanceConflict):
                        raise KnowledgeMaintenanceProposalPublicationConflict(
                            "malformed_receipt"
                        ) from None
                    if (
                        decision_record is None
                        or decision_record[0] != proposal
                        or decision_record[3] != publication_snapshot
                    ):
                        raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
                decided = decision_row is not None
        return KnowledgeMaintenanceProposalPublication(
            proposal=proposal,
            accepted_plan=accepted_plan,
            replacement=replacement,
            receipt=copy_knowledge_maintenance_proposal_publication_receipt(
                receipt,
                replayed=True,
            ),
            outcome=(
                KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_DECIDED
                if decided
                else KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
            ),
        )

    async def apply_maintenance_decision(
        self,
        proposal: KnowledgeMaintenanceProposal,
        decision: KnowledgeMaintenanceDecision,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecisionReceipt:
        scope = self._operation_access_scope(access_scope)
        proposal, decision, request_sha256 = prepare_knowledge_maintenance_decision(
            proposal,
            decision,
        )
        operation = "apply_maintenance_decision"
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                publication_rows = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                    "WHERE proposal_id = ? OR replacement_entry_id = ? "
                    "ORDER BY operation_id",
                    (proposal.id, proposal.replacement.entry_id),
                ).fetchall()
                publication_snapshot: _KnowledgeMaintenanceAccessSnapshot | None = None
                for publication_row in publication_rows:
                    publication = self._load_maintenance_proposal_record_unlocked(
                        str(publication_row["operation_id"]),
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if publication is None or publication[0] != proposal:
                        raise KnowledgeMaintenanceConflict("proposal_publication_mismatch")
                    if publication_snapshot is not None and publication_snapshot != publication[3]:
                        raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
                    publication_snapshot = publication[3]
                existing = self._load_maintenance_record_unlocked(
                    decision.operation_id,
                    access_scope=scope,
                    deny_inaccessible=True,
                )
                if existing is not None:
                    stored_proposal, stored_decision, receipt, _ = existing
                    _validate_knowledge_maintenance_replay(
                        stored_proposal,
                        stored_decision,
                        receipt,
                        proposal=proposal,
                        decision=decision,
                        request_sha256=request_sha256,
                    )
                    return copy_knowledge_maintenance_decision_receipt(
                        receipt,
                        replayed=True,
                    )

                prior = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                    "WHERE proposal_id = ?",
                    (proposal.id,),
                ).fetchone()
                if prior is not None:
                    self._load_maintenance_record_unlocked(
                        str(prior["operation_id"]),
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    raise KnowledgeMaintenanceConflict("proposal_already_decided")
                current_entries = self._load_entries_unlocked(
                    [
                        proposal.replacement.entry_id,
                        *(source.entry_id for source in proposal.sources),
                    ]
                )
                if (
                    decision.kind is KnowledgeMaintenanceDecisionKind.REJECT
                    and publication_snapshot is not None
                ):
                    replacement = _require_knowledge_maintenance_current_replacement(
                        proposal,
                        current_entries,
                        access_scope=scope,
                        operation=operation,
                    )
                    sources: list[KnowledgeEntry] = []
                    decision_snapshot = publication_snapshot
                else:
                    replacement, sources = _require_knowledge_maintenance_current_entries(
                        proposal,
                        current_entries,
                        access_scope=scope,
                        operation=operation,
                    )
                    decision_snapshot = publication_snapshot or (
                        _knowledge_maintenance_access_snapshot([replacement, *sources])
                    )
                committed_at = max(self._clock(), proposal.created_at, decision.decided_at)
                if decision.kind is KnowledgeMaintenanceDecisionKind.REJECT:
                    receipt = KnowledgeMaintenanceDecisionReceipt(
                        operation_id=decision.operation_id,
                        proposal_id=proposal.id,
                        proposal_fingerprint=proposal.fingerprint,
                        request_sha256=request_sha256,
                        outcome=KnowledgeMaintenanceOutcome.REJECTED,
                        committed_at=committed_at,
                    )
                    self._insert_maintenance_record_unlocked(
                        proposal,
                        decision,
                        receipt,
                        access_snapshot=decision_snapshot,
                    )
                    return copy_knowledge_maintenance_decision_receipt(receipt)

                active_replacement, archived_sources = _knowledge_maintenance_successors(
                    proposal,
                    replacement,
                    sources,
                    access_scope=scope,
                    committed_at=committed_at,
                    operation=operation,
                )
                for relation in proposal.relations:
                    row = self._connection.execute(
                        """
                        SELECT *
                        FROM cayu_knowledge_relations
                        WHERE id = ? OR (
                            kind = ?
                            AND subject_entry_id = ?
                            AND subject_revision = ?
                            AND object_entry_id = ?
                            AND object_revision = ?
                        )
                        LIMIT 1
                        """,
                        (relation.id, *_relation_semantic_row_values(relation)),
                    ).fetchone()
                    if row is not None:
                        occupied = _relation_from_row(row)
                        if not self._relation_endpoints_in_scope_unlocked(occupied, scope):
                            raise KnowledgeAccessDenied(operation)
                        raise KnowledgeMaintenanceConflict("relation_exists")
                    historic = self._connection.execute(
                        "SELECT sequence FROM cayu_knowledge_changes WHERE relation_id = ?",
                        (relation.id,),
                    ).fetchone()
                    if historic is not None:
                        if (
                            self._load_change_in_scope_unlocked(
                                int(historic["sequence"]),
                                scope,
                            )
                            is None
                        ):
                            raise KnowledgeAccessDenied(operation)
                        raise KnowledgeMaintenanceConflict("relation_exists")

                for successor in [active_replacement, *archived_sources]:
                    self._append_revision_unlocked(
                        successor,
                        expected_revision=current_entries[successor.id].revision,
                        chunks=None,
                        evidence=None,
                        access_scope=scope,
                        operation=operation,
                        change_kind=KnowledgeChangeKind.STATUS_TRANSITIONED,
                        inherit_evidence=True,
                        change_operation_id=decision.operation_id,
                        committed_at=committed_at,
                        allow_pending_maintenance_replacement=True,
                    )

                post_current = self._load_entries_unlocked(list(current_entries))
                relation_access: list[_KnowledgeRelationAccessSnapshot] = []
                for relation in proposal.relations:
                    subject_exact = self._load_entry_unlocked(
                        relation.subject.entry_id,
                        revision=relation.subject.revision,
                    )
                    object_exact = self._load_entry_unlocked(
                        relation.object.entry_id,
                        revision=relation.object.revision,
                    )
                    if subject_exact is None or object_exact is None:
                        raise KnowledgeMaintenanceConflict("relation_endpoint")
                    relation_access.append(
                        _knowledge_relation_access_snapshot(
                            subject_exact=subject_exact,
                            subject_current=post_current[relation.subject.entry_id],
                            object_exact=object_exact,
                            object_current=post_current[relation.object.entry_id],
                        )
                    )
                self._insert_relations_unlocked(proposal.relations)
                for relation, snapshot in zip(
                    proposal.relations,
                    relation_access,
                    strict=True,
                ):
                    self._insert_relation_change_unlocked(
                        relation,
                        access_snapshot=snapshot,
                        operation_id=decision.operation_id,
                        committed_at=committed_at,
                    )
                receipt = KnowledgeMaintenanceDecisionReceipt(
                    operation_id=decision.operation_id,
                    proposal_id=proposal.id,
                    proposal_fingerprint=proposal.fingerprint,
                    request_sha256=request_sha256,
                    outcome=KnowledgeMaintenanceOutcome.APPLIED,
                    replacement=KnowledgeRevisionRef(
                        entry_id=active_replacement.id,
                        revision=active_replacement.revision,
                    ),
                    archived_revisions=[
                        KnowledgeRevisionRef(entry_id=entry.id, revision=entry.revision)
                        for entry in archived_sources
                    ],
                    relation_ids=[relation.id for relation in proposal.relations],
                    committed_at=committed_at,
                )
                self._insert_maintenance_record_unlocked(
                    proposal,
                    decision,
                    receipt,
                    access_snapshot=decision_snapshot,
                )
                return copy_knowledge_maintenance_decision_receipt(receipt)

    async def load_maintenance_proposal(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposal | None:
        scope = self._operation_access_scope(access_scope)
        proposal_id = _knowledge_maintenance_identity(proposal_id, "proposal_id")
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                publication_row = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                    "WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone()
                if publication_row is not None:
                    publication = self._load_maintenance_proposal_record_unlocked(
                        str(publication_row["operation_id"]),
                        access_scope=scope,
                        deny_inaccessible=False,
                    )
                    return (
                        None
                        if publication is None
                        else copy_knowledge_maintenance_proposal(publication[0])
                    )
                row = self._connection.execute(
                    "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                    "WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone()
                if row is None:
                    return None
                record = self._load_maintenance_record_unlocked(
                    str(row["operation_id"]),
                    access_scope=scope,
                    deny_inaccessible=False,
                )
        return None if record is None else copy_knowledge_maintenance_proposal(record[0])

    async def load_maintenance_decision(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecision | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        async with self._lock:
            record = self._load_maintenance_record_unlocked(
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if record is None else copy_knowledge_maintenance_decision(record[1])

    async def load_maintenance_decision_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecisionReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        async with self._lock:
            record = self._load_maintenance_record_unlocked(
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if record is None else copy_knowledge_maintenance_decision_receipt(record[2])

    async def read_evidence(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        max_records: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> KnowledgeEvidenceResult | None:
        scope = self._operation_access_scope(access_scope)
        entry_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        _validate_positive_int(max_records, "max_records")
        _validate_positive_int(max_bytes, "max_bytes")
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                entry = self._load_entry_in_scope_unlocked(
                    entry_id,
                    scope,
                    revision=revision,
                )
                if entry is None:
                    return None
                total_evidence_known = self._count_evidence_unlocked(
                    entry.id,
                    revision=entry.revision,
                )
                stored = self._load_evidence_unlocked(
                    entry.id,
                    revision=entry.revision,
                    limit=max_records,
                )
        selected = _bounded_knowledge_evidence(
            stored,
            max_records=max_records,
            max_bytes=max_bytes,
        )
        return KnowledgeEvidenceResult(
            entry_id=entry.id,
            entry_revision=entry.revision,
            evidence=selected,
            truncated=len(selected) < total_evidence_known,
            limit=max_records,
            max_bytes=max_bytes,
            total_evidence_known=total_evidence_known,
        )

    async def read_changes(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeBatch:
        scope = self._operation_access_scope(access_scope)
        _validate_knowledge_change_sequence(after_sequence, "after_sequence")
        _validate_knowledge_change_limit(limit)
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                high_water = self._accessible_change_high_water_unlocked(scope)
                if after_sequence > high_water:
                    row = self._connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) AS current_sequence "
                        "FROM cayu_knowledge_changes"
                    ).fetchone()
                    current_sequence = 0 if row is None else int(row["current_sequence"])
                    if after_sequence > current_sequence:
                        raise ValueError(
                            "`after_sequence` cannot exceed the current knowledge change sequence."
                        )
                rows = self._load_accessible_change_rows_unlocked(
                    scope,
                    after_sequence=after_sequence,
                    through_sequence=high_water,
                    limit=limit + 1,
                )
        changes = [_change_from_row(row) for row in rows[:limit]]
        truncated = len(rows) > limit
        next_after = changes[-1].sequence if truncated else max(after_sequence, high_water)
        return KnowledgeChangeBatch(
            changes=changes,
            after_sequence=after_sequence,
            next_after_sequence=next_after,
            high_water_sequence=high_water,
            truncated=truncated,
            limit=limit,
        )

    async def claim_change(
        self,
        consumer_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeClaim | None:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        worker_id = _knowledge_change_identity(worker_id, "worker_id")
        lease_seconds = _knowledge_change_lease_seconds(lease_seconds)
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current_time = self._clock()
                state = self._load_change_consumer_unlocked(consumer_id)
                if state is None:
                    state = KnowledgeChangeConsumerState(
                        consumer_id=consumer_id,
                        access_scope_sha256=scope_sha256,
                        updated_at=current_time,
                    )
                elif state.access_scope_sha256 != scope_sha256:
                    raise KnowledgeChangeConsumerConflict("access_scope_mismatch")
                if state.pending_change_sequence is not None:
                    stored_change = self._load_change_in_scope_unlocked(
                        state.pending_change_sequence,
                        scope,
                    )
                    assert state.lease_expires_at is not None
                    if stored_change is not None and state.lease_expires_at > current_time:
                        if state.pending_worker_id != worker_id:
                            self._save_change_consumer_unlocked(state)
                            return None
                        assert state.pending_claim_id is not None
                        assert state.claimed_at is not None
                        self._save_change_consumer_unlocked(state)
                        return KnowledgeChangeClaim(
                            consumer_id=consumer_id,
                            worker_id=worker_id,
                            claim_id=state.pending_claim_id,
                            change=stored_change,
                            attempt=state.pending_attempt,
                            claimed_at=state.claimed_at,
                            lease_expires_at=state.lease_expires_at,
                        )
                    state = state.model_copy(
                        update={
                            "pending_change_sequence": None,
                            "pending_claim_id": None,
                            "pending_worker_id": None,
                            "claimed_at": None,
                            "lease_expires_at": None,
                            "pending_attempt": (
                                state.pending_attempt if stored_change is not None else 0
                            ),
                            "updated_at": current_time,
                        }
                    )
                change = self._next_accessible_change_unlocked(
                    scope,
                    after_sequence=state.cursor_sequence,
                )
                if change is None:
                    self._save_change_consumer_unlocked(state)
                    return None
                claim_id = f"kclaim_{uuid4().hex}"
                claimed_at = current_time
                lease_expires_at = claimed_at + timedelta(seconds=lease_seconds)
                attempt = state.pending_attempt + 1
                state = state.model_copy(
                    update={
                        "pending_change_sequence": change.sequence,
                        "pending_claim_id": claim_id,
                        "pending_worker_id": worker_id,
                        "pending_attempt": attempt,
                        "claimed_at": claimed_at,
                        "lease_expires_at": lease_expires_at,
                        "updated_at": current_time,
                    }
                )
                self._save_change_consumer_unlocked(state)
                return KnowledgeChangeClaim(
                    consumer_id=consumer_id,
                    worker_id=worker_id,
                    claim_id=claim_id,
                    change=change,
                    attempt=attempt,
                    claimed_at=claimed_at,
                    lease_expires_at=lease_expires_at,
                )

    async def initialize_change_consumer(
        self,
        consumer_id: str,
        *,
        baseline_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        _validate_knowledge_change_sequence(baseline_sequence, "baseline_sequence")
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current_time = self._clock()
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) AS current_sequence "
                    "FROM cayu_knowledge_changes"
                ).fetchone()
                current_sequence = 0 if row is None else int(row["current_sequence"])
                if baseline_sequence > current_sequence:
                    raise ValueError(
                        "`baseline_sequence` cannot exceed the current knowledge change sequence."
                    )
                state = _initialize_knowledge_change_consumer_state(
                    self._load_change_consumer_unlocked(consumer_id),
                    consumer_id=consumer_id,
                    access_scope_sha256=_knowledge_access_scope_sha256(scope),
                    baseline_sequence=baseline_sequence,
                    now=current_time,
                )
                self._save_change_consumer_unlocked(state)
                return copy_knowledge_change_consumer_state(state)

    async def acknowledge_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        claim = copy_knowledge_change_claim(claim)
        claim_sha256 = _knowledge_change_claim_sha256(claim)
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current_time = self._clock()
                state = self._load_change_consumer_unlocked(claim.consumer_id)
                if state is None or state.access_scope_sha256 != scope_sha256:
                    raise KnowledgeChangeConsumerConflict("unknown_consumer")
                acknowledged = self._load_change_acknowledgement_unlocked(
                    claim.consumer_id,
                    claim.claim_id,
                )
                if acknowledged is not None:
                    if acknowledged != (claim_sha256, claim.change.sequence):
                        raise KnowledgeChangeConsumerConflict("stale_claim")
                    if state.cursor_sequence < claim.change.sequence:
                        raise RuntimeError(
                            "Knowledge change acknowledgement is ahead of its consumer."
                        )
                    return copy_knowledge_change_consumer_state(state)
                self._require_live_change_claim_unlocked(state, claim, now=current_time)
                state = state.model_copy(
                    update={
                        "cursor_sequence": claim.change.sequence,
                        "pending_change_sequence": None,
                        "pending_claim_id": None,
                        "pending_worker_id": None,
                        "pending_attempt": 0,
                        "claimed_at": None,
                        "lease_expires_at": None,
                        "last_acknowledged_claim_id": claim.claim_id,
                        "updated_at": current_time,
                    }
                )
                self._save_change_consumer_unlocked(state)
                self._insert_change_acknowledgement_unlocked(
                    claim,
                    claim_sha256=claim_sha256,
                    acknowledged_at=current_time,
                )
                return copy_knowledge_change_consumer_state(state)

    async def release_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        claim = copy_knowledge_change_claim(claim)
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                current_time = self._clock()
                state = self._load_change_consumer_unlocked(claim.consumer_id)
                if state is None or state.access_scope_sha256 != scope_sha256:
                    raise KnowledgeChangeConsumerConflict("unknown_consumer")
                self._require_live_change_claim_unlocked(
                    state,
                    claim,
                    now=current_time,
                )
                state = state.model_copy(
                    update={
                        "pending_change_sequence": None,
                        "pending_claim_id": None,
                        "pending_worker_id": None,
                        "claimed_at": None,
                        "lease_expires_at": None,
                        "updated_at": current_time,
                    }
                )
                self._save_change_consumer_unlocked(state)
                return copy_knowledge_change_consumer_state(state)

    async def load_change_consumer_state(
        self,
        consumer_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState | None:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        async with self._lock:
            state = self._load_change_consumer_unlocked(consumer_id)
        if state is None or state.access_scope_sha256 != scope_sha256:
            return None
        return copy_knowledge_change_consumer_state(state)

    async def publish_index_readiness(
        self,
        update: KnowledgeIndexReadinessUpdate,
        *,
        expected_sequence: int | None,
        operation_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness:
        scope = self._operation_access_scope(access_scope)
        update = copy_knowledge_index_readiness_update(update)
        operation_id = _bounded_knowledge_index_identity(operation_id, "operation_id")
        if expected_sequence is not None:
            _validate_knowledge_index_sequence(
                expected_sequence,
                "expected_sequence",
                allow_zero=False,
            )
        identity_sha256 = _knowledge_embedding_identity_sha256(update.identity)
        update_sha256 = _knowledge_index_readiness_update_sha256(update)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                replay_row = self._connection.execute(
                    "SELECT * FROM cayu_knowledge_index_readiness_events WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if replay_row is not None:
                    if str(replay_row["update_sha256"]) != update_sha256:
                        raise KnowledgeIndexReadinessConflict("operation_reuse")
                    if not self._index_identity_is_accessible_unlocked(
                        scope,
                        update.identity,
                    ):
                        raise KnowledgeAccessDenied("publish_index_readiness")
                    return _index_readiness_from_row(replay_row)
                if not self._index_identity_is_accessible_unlocked(
                    scope,
                    update.identity,
                    require_current=True,
                ):
                    raise KnowledgeIndexReadinessConflict("stale_identity")
                current_row = self._connection.execute(
                    """
                    SELECT event.*
                    FROM cayu_knowledge_index_readiness_current AS current
                    JOIN cayu_knowledge_index_readiness_events AS event
                      ON event.sequence = current.sequence
                     AND event.identity_sha256 = current.identity_sha256
                    WHERE current.identity_sha256 = ?
                    """,
                    (identity_sha256,),
                ).fetchone()
                current = None if current_row is None else _index_readiness_from_row(current_row)
                _validate_knowledge_index_readiness_transition(
                    current,
                    update,
                    expected_sequence=expected_sequence,
                )
                published_at = self._clock()
                cursor = self._connection.execute(
                    """
                    INSERT INTO cayu_knowledge_index_readiness_events (
                        identity_sha256,
                        entry_id,
                        entry_revision,
                        chunk_id,
                        projection_type,
                        projection_content_hash,
                        embedding_model,
                        dimensions,
                        preprocessing_version,
                        generator,
                        generator_version,
                        index_representation_version,
                        state,
                        attempt_id,
                        failure_code,
                        operation_id,
                        update_sha256,
                        published_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity_sha256,
                        update.identity.entry_id,
                        update.identity.entry_revision,
                        update.identity.chunk_id,
                        update.identity.projection_type,
                        update.identity.projection_content_hash,
                        update.identity.embedding_model,
                        update.identity.dimensions,
                        update.identity.preprocessing_version,
                        update.identity.generator,
                        update.identity.generator_version,
                        update.identity.index_representation_version,
                        str(update.state),
                        update.attempt_id,
                        update.failure_code,
                        operation_id,
                        update_sha256,
                        sqlite_support.format_datetime(published_at),
                    ),
                )
                if cursor.lastrowid is None:  # pragma: no cover - sqlite invariant
                    raise RuntimeError("SQLite did not return an index readiness sequence.")
                sequence = cursor.lastrowid
                if current is None:
                    self._connection.execute(
                        "INSERT INTO cayu_knowledge_index_readiness_current "
                        "(identity_sha256, sequence) VALUES (?, ?)",
                        (identity_sha256, sequence),
                    )
                else:
                    cursor = self._connection.execute(
                        """
                        UPDATE cayu_knowledge_index_readiness_current
                        SET sequence = ?
                        WHERE identity_sha256 = ? AND sequence = ?
                        """,
                        (sequence, identity_sha256, current.sequence),
                    )
                    if cursor.rowcount != 1:  # pragma: no cover - writer lock invariant
                        raise KnowledgeIndexReadinessConflict("stale_sequence")
                return KnowledgeIndexReadiness(
                    sequence=sequence,
                    identity=update.identity,
                    state=update.state,
                    attempt_id=update.attempt_id,
                    failure_code=update.failure_code,
                    operation_id=operation_id,
                    published_at=published_at,
                )

    async def load_index_readiness(
        self,
        identity: KnowledgeEmbeddingIdentity,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness | None:
        scope = self._operation_access_scope(access_scope)
        identity = copy_knowledge_embedding_identity(identity)
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                if not self._index_identity_is_accessible_unlocked(scope, identity):
                    return None
                row = self._connection.execute(
                    """
                    SELECT event.*
                    FROM cayu_knowledge_index_readiness_current AS current
                JOIN cayu_knowledge_index_readiness_events AS event
                  ON event.sequence = current.sequence
                 AND event.identity_sha256 = current.identity_sha256
                    WHERE current.identity_sha256 = ?
                    """,
                    (_knowledge_embedding_identity_sha256(identity),),
                ).fetchone()
        if row is None:
            return None
        readiness = _index_readiness_from_row(row)
        if readiness.identity != identity:
            raise RuntimeError("Knowledge index readiness identity digest collision.")
        return readiness

    async def read_index_readiness(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadinessBatch:
        scope = self._operation_access_scope(access_scope)
        _validate_knowledge_index_sequence(after_sequence, "after_sequence")
        _validate_knowledge_index_readiness_limit(limit)
        exact_access_sql, exact_access_params = _knowledge_access_scope_filter_sql(
            scope,
            entry_alias="e",
        )
        current_access_sql, current_access_params = _knowledge_access_scope_filter_sql(
            scope,
            entry_alias="current_entry",
        )
        accessible_from = """
            FROM cayu_knowledge_index_readiness_events AS event
            JOIN (
                SELECT logical.id, logical.namespace, revision.*
                FROM cayu_knowledge_entries AS logical
                JOIN cayu_knowledge_revisions AS revision
                  ON revision.entry_id = logical.id
            ) AS e
              ON e.id = event.entry_id AND e.revision = event.entry_revision
            JOIN cayu_knowledge_current_entries AS current_entry
              ON current_entry.id = event.entry_id
            WHERE TRUE
        """
        access_params = [*exact_access_params, *current_access_params]
        async with self._lock:
            with sqlite_support._transaction(self._connection, begin_immediate=False):
                high_water_row = self._connection.execute(
                    "SELECT COALESCE(MAX(event.sequence), 0) AS high_water "
                    + accessible_from
                    + exact_access_sql
                    + current_access_sql,
                    access_params,
                ).fetchone()
                high_water = 0 if high_water_row is None else int(high_water_row["high_water"])
                if after_sequence > high_water:
                    current_row = self._connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) AS current_sequence "
                        "FROM cayu_knowledge_index_readiness_events"
                    ).fetchone()
                    current_sequence = (
                        0 if current_row is None else int(current_row["current_sequence"])
                    )
                    if after_sequence > current_sequence:
                        raise ValueError(
                            "`after_sequence` cannot exceed the current knowledge "
                            "index readiness sequence."
                        )
                rows = self._connection.execute(
                    "SELECT event.* "
                    + accessible_from
                    + " AND event.sequence > ? AND event.sequence <= ?"
                    + exact_access_sql
                    + current_access_sql
                    + " ORDER BY event.sequence LIMIT ?",
                    [after_sequence, high_water, *access_params, limit + 1],
                ).fetchall()
        readiness = [_index_readiness_from_row(row) for row in rows[:limit]]
        truncated = len(rows) > limit
        next_after = readiness[-1].sequence if truncated else max(after_sequence, high_water)
        return KnowledgeIndexReadinessBatch(
            readiness=readiness,
            after_sequence=after_sequence,
            next_after_sequence=next_after,
            high_water_sequence=high_water,
            truncated=truncated,
            limit=limit,
        )

    def _index_identity_is_accessible_unlocked(
        self,
        scope: KnowledgeAccessScope,
        identity: KnowledgeEmbeddingIdentity,
        *,
        require_current: bool = False,
    ) -> bool:
        current = self._load_entry_in_scope_unlocked(identity.entry_id, scope)
        if current is None:
            return False
        if require_current and current.revision != identity.entry_revision:
            return False
        revision = self._load_entry_in_scope_unlocked(
            identity.entry_id,
            scope,
            revision=identity.entry_revision,
        )
        if revision is None:
            return False
        if identity.chunk_id is None:
            return True
        row = self._connection.execute(
            """
            SELECT id, entry_id, entry_revision, chunk_index, text,
                   content_hash, source_uri, metadata_json
            FROM cayu_knowledge_chunks
            WHERE id = ? AND entry_id = ? AND entry_revision = ?
            """,
            (identity.chunk_id, identity.entry_id, identity.entry_revision),
        ).fetchone()
        if row is None:
            return False
        if identity.projection_type == KNOWLEDGE_CHUNK_TEXT_PROJECTION:
            return identity.projection_content_hash == _knowledge_chunk_content_hash(
                _chunk_from_row(row)
            )
        return True

    async def read_chunks(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if chunk_index is not None:
            _validate_nonnegative_int(chunk_index, "chunk_index")
        _validate_nonnegative_int(around, "around")
        if chunk_index is None and around != 0:
            raise ValueError("`around` requires `chunk_index`.")
        _validate_positive_int(max_chunks, "max_chunks")
        _validate_positive_int(max_bytes, "max_bytes")
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                entry = self._load_entry_in_scope_unlocked(
                    clean_id,
                    scope,
                    revision=revision,
                )
                if entry is None:
                    return []
                chunks = self._load_chunks_unlocked(clean_id, revision=entry.revision)
        if chunk_index is not None:
            chunks = _center_chunk_window(chunks, chunk_index=chunk_index, max_chunks=max_chunks)
        start_index = 0 if chunk_index is None else max(0, chunk_index - around)
        end_index = None if chunk_index is None else chunk_index + around
        return _bounded_chunks(
            chunks,
            start_index=start_index,
            end_index=end_index,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )

    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_query(query)
        return await self._search(
            knowledge_query,
            scope,
            revision_refs=None,
            through_change_sequence=None,
        )

    async def search_at_frontier(
        self,
        query: KnowledgeQuery,
        *,
        knowledge_sequence: int,
        index_readiness_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_query(query)
        _validate_knowledge_change_sequence(knowledge_sequence, "knowledge_sequence")
        _validate_knowledge_index_sequence(
            index_readiness_sequence,
            "index_readiness_sequence",
        )
        return await self._search(
            knowledge_query,
            scope,
            revision_refs=None,
            through_change_sequence=knowledge_sequence,
        )

    async def search_revisions(
        self,
        query: KnowledgeQuery,
        revision_refs: Sequence[KnowledgeRevisionRef],
        *,
        knowledge_sequence: int | None = None,
        index_readiness_sequence: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_query(query)
        references = copy_knowledge_revision_refs(revision_refs)
        _validate_knowledge_search_frontier(
            knowledge_sequence,
            index_readiness_sequence,
        )
        return await self._search(
            knowledge_query,
            scope,
            revision_refs=references,
            through_change_sequence=knowledge_sequence,
        )

    async def _search(
        self,
        knowledge_query: KnowledgeQuery,
        scope: KnowledgeAccessScope,
        *,
        revision_refs: tuple[KnowledgeRevisionRef, ...] | None,
        through_change_sequence: int | None,
    ) -> KnowledgeSearchResult:
        if knowledge_query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("SQLiteKnowledgeStore supports only auto and keyword search modes.")
        fts_query, preview_terms = _sqlite_knowledge_fts_query(knowledge_query)
        none_fts_query = _sqlite_knowledge_none_fts_query(knowledge_query)
        if revision_refs is not None:
            async with self._lock:
                with sqlite_support._transaction(
                    self._connection,
                    begin_immediate=False,
                ):
                    return self._search_exact_revisions_unlocked(
                        knowledge_query,
                        scope,
                        revision_refs,
                        through_change_sequence=through_change_sequence,
                        fts_query=fts_query,
                        none_fts_query=none_fts_query,
                        preview_terms=preview_terms,
                    )
        where_sql, params = _knowledge_filter_sql(knowledge_query)
        if through_change_sequence is not None:
            where_sql += """
                AND (
                    SELECT MAX(boundary_change.sequence)
                    FROM cayu_knowledge_changes AS boundary_change
                    WHERE boundary_change.entry_id = e.id
                      AND boundary_change.entry_revision = e.revision
                      AND boundary_change.kind <> 'relation_published'
                ) <= ?
            """
            params.append(through_change_sequence)
        access_sql, access_params = _knowledge_access_scope_filter_sql(scope)
        where_sql += access_sql
        params.extend(access_params)
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                total_hits_known = self._count_search_hits_unlocked(
                    fts_query,
                    none_fts_query,
                    where_sql,
                    params,
                )
                unique_rows = self._search_unique_rows_unlocked(
                    fts_query=fts_query,
                    none_fts_query=none_fts_query,
                    where_sql=where_sql,
                    params=params,
                    limit=knowledge_query.limit,
                )
                hits, byte_truncated = self._hits_from_search_rows_unlocked(
                    unique_rows,
                    knowledge_query,
                    preview_terms,
                )
        return KnowledgeSearchResult(
            query=knowledge_query,
            hits=hits,
            truncated=byte_truncated or len(hits) < total_hits_known,
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_hits_known=total_hits_known,
        )

    def _search_exact_revisions_unlocked(
        self,
        knowledge_query: KnowledgeQuery,
        scope: KnowledgeAccessScope,
        revision_refs: tuple[KnowledgeRevisionRef, ...],
        *,
        through_change_sequence: int | None,
        fts_query: str | None,
        none_fts_query: str | None,
        preview_terms: list[str],
    ) -> KnowledgeSearchResult:
        """Search only caller-authorized current entries from the bounded exact set.

        A connection-local FTS5 table preserves the durable index's tokenizer and
        query semantics without making delta cost depend on the complete corpus.
        """

        entry_ids = sorted({reference.entry_id for reference in revision_refs})
        access_now = datetime.now(UTC)
        entries = self._load_entries_unlocked(
            entry_ids,
            access_scope=scope,
            access_now=access_now,
        )
        materialization_sequences = (
            self._load_revision_materialization_sequences_unlocked(revision_refs)
            if through_change_sequence is not None
            else None
        )
        current_refs_list: list[KnowledgeRevisionRef] = []
        for reference in revision_refs:
            entry = entries.get(reference.entry_id)
            if entry is None or entry.revision != reference.revision:
                continue
            if through_change_sequence is not None:
                assert materialization_sequences is not None
                sequence = materialization_sequences.get((reference.entry_id, reference.revision))
                if sequence is None or sequence > through_change_sequence:
                    continue
            current_refs_list.append(reference)
        current_refs = tuple(current_refs_list)
        chunks_by_revision = self._load_chunks_for_revision_refs_unlocked(current_refs)
        fts_rows = [
            (
                entry.id,
                entry.revision,
                chunk.id,
                entry.title or "",
                _fts_text_for_entry_chunk(entry, chunk),
            )
            for reference in current_refs
            if (entry := entries.get(reference.entry_id)) is not None
            for chunk in chunks_by_revision.get((entry.id, entry.revision), [])
        ]
        if not fts_rows:
            return KnowledgeSearchResult(
                query=knowledge_query,
                hits=[],
                truncated=False,
                limit=knowledge_query.limit,
                max_bytes=knowledge_query.max_bytes,
                total_hits_known=0,
            )

        table = _EXACT_REVISION_FTS_TABLE
        self._connection.execute(f"DELETE FROM temp.{table}")
        try:
            self._connection.executemany(
                f"""
                INSERT INTO temp.{table} (
                    entry_id, entry_revision, chunk_id, title, text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                fts_rows,
            )
            where_sql, params = _knowledge_filter_sql(knowledge_query)
            total_hits_known = self._count_exact_search_hits_unlocked(
                fts_query,
                none_fts_query,
                where_sql,
                params,
            )
            unique_rows = self._search_exact_unique_rows_unlocked(
                fts_query=fts_query,
                none_fts_query=none_fts_query,
                where_sql=where_sql,
                params=params,
                limit=knowledge_query.limit,
            )
            hits, byte_truncated = self._hits_from_search_rows_unlocked(
                unique_rows,
                knowledge_query,
                preview_terms,
            )
        finally:
            self._connection.execute(f"DELETE FROM temp.{table}")
        return KnowledgeSearchResult(
            query=knowledge_query,
            hits=hits,
            truncated=byte_truncated or len(hits) < total_hits_known,
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_hits_known=total_hits_known,
        )

    def _count_exact_search_hits_unlocked(
        self,
        fts_query: str | None,
        none_fts_query: str | None,
        where_sql: str,
        params: list[object],
    ) -> int:
        return self._count_search_hits_unlocked(
            fts_query,
            none_fts_query,
            where_sql,
            params,
            fts_table=_EXACT_REVISION_FTS_TABLE,
            temporary_fts=True,
        )

    def _search_exact_unique_rows_unlocked(
        self,
        *,
        fts_query: str | None,
        none_fts_query: str | None,
        where_sql: str,
        params: list[object],
        limit: int,
    ) -> list[sqlite3.Row]:
        return self._search_unique_rows_unlocked(
            fts_query=fts_query,
            none_fts_query=none_fts_query,
            where_sql=where_sql,
            params=params,
            limit=limit,
            fts_table=_EXACT_REVISION_FTS_TABLE,
            temporary_fts=True,
        )

    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_list_query(query)
        where_sql, params = _knowledge_list_filter_sql(knowledge_query)
        access_sql, access_params = _knowledge_access_scope_filter_sql(scope)
        where_sql += access_sql
        params.extend(access_params)
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                total_entries_known = self._count_list_entries_unlocked(where_sql, params)
                rows = self._connection.execute(
                    f"""
                    SELECT e.id
                    FROM cayu_knowledge_current_entries AS e
                    WHERE 1 = 1
                    {where_sql}
                    ORDER BY COALESCE(e.importance, 0.0) DESC,
                             e.updated_at DESC,
                             e.id ASC
                    LIMIT ?
                    """,
                    [*params, knowledge_query.limit],
                ).fetchall()
                entry_map = self._load_entries_unlocked([str(row["id"]) for row in rows])
                entries = [
                    entry for row in rows if (entry := entry_map.get(str(row["id"]))) is not None
                ]
                facets, facets_truncated = self._list_facets_unlocked(
                    knowledge_query,
                    where_sql,
                    params,
                )
                items, byte_truncated = self._list_items_unlocked(entries, knowledge_query)
        return KnowledgeListResult(
            query=knowledge_query,
            entries=items,
            facets=facets,
            facets_truncated=facets_truncated,
            truncated=byte_truncated or len(items) < total_entries_known or facets_truncated,
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_entries_known=total_entries_known,
        )

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _count_search_hits_unlocked(
        self,
        fts_query: str | None,
        none_fts_query: str | None,
        where_sql: str,
        params: list[object],
        *,
        fts_table: str = _KNOWLEDGE_FTS_TABLE,
        temporary_fts: bool = False,
    ) -> int:
        none_sql, none_params = _sqlite_knowledge_none_filter_sql(
            none_fts_query,
            fts_table=fts_table,
            temporary_fts=temporary_fts,
        )
        if fts_query is None and not temporary_fts:
            row = self._connection.execute(
                f"""
                SELECT COUNT(*)
                FROM cayu_knowledge_current_entries AS e
                WHERE EXISTS (
                    SELECT 1
                    FROM cayu_knowledge_chunks AS available_chunk
                    WHERE available_chunk.entry_id = e.id
                      AND available_chunk.entry_revision = e.revision
                )
                {none_sql}
                {where_sql}
                """,
                [*none_params, *params],
            ).fetchone()
            return 0 if row is None else int(row[0])
        from_table = f"temp.{fts_table}" if temporary_fts else fts_table
        match_sql = "1 = 1" if fts_query is None else f"{fts_table} MATCH ?"
        match_params: list[object] = [] if fts_query is None else [fts_query]
        row = self._connection.execute(
            f"""
            SELECT COUNT(DISTINCT e.id)
            FROM {from_table}
            JOIN cayu_knowledge_chunks AS c
              ON c.id = {fts_table}.chunk_id
             AND c.entry_id = {fts_table}.entry_id
             AND c.entry_revision = {fts_table}.entry_revision
            JOIN cayu_knowledge_current_entries AS e
                ON e.id = c.entry_id AND e.revision = c.entry_revision
            WHERE {match_sql}
            {none_sql}
            {where_sql}
            """,
            [*match_params, *none_params, *params],
        ).fetchone()
        return 0 if row is None else int(row[0])

    def _search_unique_rows_unlocked(
        self,
        *,
        fts_query: str | None,
        none_fts_query: str | None,
        where_sql: str,
        params: list[object],
        limit: int,
        fts_table: str = _KNOWLEDGE_FTS_TABLE,
        temporary_fts: bool = False,
    ) -> list[sqlite3.Row]:
        none_sql, none_params = _sqlite_knowledge_none_filter_sql(
            none_fts_query,
            fts_table=fts_table,
            temporary_fts=temporary_fts,
        )
        if fts_query is None and not temporary_fts:
            return list(
                self._connection.execute(
                    f"""
                    SELECT
                        e.id AS entry_id,
                        (
                            SELECT available_chunk.id
                            FROM cayu_knowledge_chunks AS available_chunk
                            WHERE available_chunk.entry_id = e.id
                              AND available_chunk.entry_revision = e.revision
                            ORDER BY available_chunk.chunk_index ASC,
                                     available_chunk.id ASC
                            LIMIT 1
                        ) AS chunk_id,
                        -1.0 AS fts_score
                    FROM cayu_knowledge_current_entries AS e
                    WHERE EXISTS (
                        SELECT 1
                        FROM cayu_knowledge_chunks AS available_chunk
                        WHERE available_chunk.entry_id = e.id
                          AND available_chunk.entry_revision = e.revision
                    )
                    {none_sql}
                    {where_sql}
                    ORDER BY COALESCE(e.importance, 0.0) DESC,
                             e.updated_at DESC,
                             e.id ASC
                    LIMIT ?
                    """,
                    [*none_params, *params, limit],
                ).fetchall()
            )
        from_table = f"temp.{fts_table}" if temporary_fts else fts_table
        unique_rows: list[sqlite3.Row] = []
        seen_entry_ids: set[str] = set()
        offset = 0
        while len(unique_rows) < limit:
            match_sql = "1 = 1" if fts_query is None else f"{fts_table} MATCH ?"
            match_params: list[object] = [] if fts_query is None else [fts_query]
            score_sql = "-1.0" if fts_query is None else f"bm25({fts_table})"
            rows = self._connection.execute(
                f"""
                SELECT
                    e.id AS entry_id,
                    c.id AS chunk_id,
                    {score_sql} AS fts_score
                FROM {from_table}
                JOIN cayu_knowledge_chunks AS c
                  ON c.id = {fts_table}.chunk_id
                 AND c.entry_id = {fts_table}.entry_id
                 AND c.entry_revision = {fts_table}.entry_revision
                JOIN cayu_knowledge_current_entries AS e
                    ON e.id = c.entry_id AND e.revision = c.entry_revision
                WHERE {match_sql}
                {none_sql}
                {where_sql}
                ORDER BY fts_score ASC,
                         COALESCE(e.importance, 0.0) DESC,
                         e.updated_at DESC,
                         e.id ASC,
                         c.chunk_index ASC
                LIMIT ? OFFSET ?
                """,
                [*match_params, *none_params, *params, _SEARCH_PAGE_SIZE, offset],
            ).fetchall()
            if not rows:
                break
            for row in rows:
                entry_id = str(row["entry_id"])
                if entry_id in seen_entry_ids:
                    continue
                seen_entry_ids.add(entry_id)
                unique_rows.append(row)
                if len(unique_rows) >= limit:
                    break
            if len(rows) < _SEARCH_PAGE_SIZE:
                break
            offset += _SEARCH_PAGE_SIZE
        return unique_rows

    def _hits_from_search_rows_unlocked(
        self,
        rows: list[sqlite3.Row],
        query: KnowledgeQuery,
        terms: list[str],
    ) -> tuple[list[KnowledgeHit], bool]:
        entries = self._load_entries_unlocked([str(row["entry_id"]) for row in rows])
        chunks = self._load_chunks_by_ids_unlocked([str(row["chunk_id"]) for row in rows])
        hits: list[KnowledgeHit] = []
        remaining = query.max_bytes
        truncated = False
        for row in rows:
            if remaining <= 0:
                truncated = True
                break
            entry = entries.get(str(row["entry_id"]))
            chunk = chunks.get(str(row["chunk_id"]))
            if entry is None or chunk is None:
                continue
            filter_only = not terms
            reason, preview_text = (
                ("exact aspect filter", chunk.text)
                if filter_only
                else _preview_for_match(entry, chunk, terms)
            )
            preview_bytes = len(preview_text.encode("utf-8"))
            preview = _truncate_text_to_bytes(preview_text, remaining)
            if not preview:
                truncated = True
                break
            returned_bytes = len(preview.encode("utf-8"))
            preview_complete = returned_bytes == preview_bytes
            if not preview_complete:
                truncated = True
            remaining -= returned_bytes
            hits.append(
                KnowledgeHit(
                    entry=entry,
                    chunk=chunk,
                    score=-float(row["fts_score"]),
                    score_kind="exact_metadata" if filter_only else "sqlite_fts5_bm25",
                    rank=len(hits) + 1,
                    reason=reason,
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return hits, truncated

    def _count_list_entries_unlocked(self, where_sql: str, params: list[object]) -> int:
        row = self._connection.execute(
            f"""
            SELECT COUNT(*)
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            """,
            params,
        ).fetchone()
        return 0 if row is None else int(row[0])

    def _list_items_unlocked(
        self,
        entries: list[KnowledgeEntry],
        query: KnowledgeListQuery,
    ) -> tuple[list[KnowledgeListItem], bool]:
        chunk_counts = self._count_chunks_by_entry_unlocked([entry.id for entry in entries])
        items: list[KnowledgeListItem] = []
        remaining = query.max_bytes
        truncated = False
        for entry in entries:
            if remaining <= 0:
                truncated = True
                break
            preview_source = entry.title or entry.text
            preview_bytes = len(preview_source.encode("utf-8"))
            preview = _truncate_text_to_bytes(preview_source, remaining)
            if not preview:
                truncated = True
                break
            returned_bytes = len(preview.encode("utf-8"))
            preview_complete = returned_bytes == preview_bytes
            if not preview_complete:
                truncated = True
            remaining -= returned_bytes
            items.append(
                KnowledgeListItem(
                    entry=entry,
                    chunk_count=chunk_counts.get(entry.id, 0),
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return items, truncated

    def _list_facets_unlocked(
        self,
        query: KnowledgeListQuery,
        where_sql: str,
        params: list[object],
    ) -> tuple[list[KnowledgeFacet], bool]:
        if query.group_by is None:
            return [], False
        rows = self._connection.execute(
            *_sqlite_list_facet_sql(
                query.group_by,
                where_sql,
                params,
                limit=query.limit + 1,
            )
        ).fetchall()
        facets = [
            KnowledgeFacet(
                field=query.group_by,
                key=str(row["key"]) if row["key"] is not None else None,
                value=str(row["value"]),
                count=int(row["count"]),
            )
            for row in rows[: query.limit]
        ]
        return facets, len(rows) > query.limit

    def _insert_entry_unlocked(self, entry: KnowledgeEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id,
                namespace,
                current_revision,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.namespace,
                entry.revision,
                sqlite_support.format_datetime(entry.created_at),
                sqlite_support.format_datetime(entry.updated_at),
            ),
        )
        self._insert_revision_unlocked(entry)

    def _insert_revision_unlocked(self, entry: KnowledgeEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_revisions (
                entry_id,
                revision,
                text,
                kind,
                visibility,
                status,
                created_by_type,
                created_by,
                created_at,
                updated_at,
                source_type,
                source_uri,
                source_id,
                source_hash,
                importance,
                importance_source,
                confidence,
                last_used_at,
                expires_at,
                title,
                metadata_json,
                payload_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _entry_row_values(entry),
        )
        if entry.labels:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_labels (entry_id, entry_revision, key, value)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (entry.id, entry.revision, key, value)
                    for key, value in sorted(entry.labels.items())
                ],
            )
        if entry.aspects:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_aspects (entry_id, entry_revision, aspect)
                VALUES (?, ?, ?)
                """,
                [(entry.id, entry.revision, aspect) for aspect in entry.aspects],
            )
        if entry.impact_targets:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_impact_targets (
                    entry_id, entry_revision, impact_target
                )
                VALUES (?, ?, ?)
                """,
                [(entry.id, entry.revision, target) for target in entry.impact_targets],
            )

    def _advance_current_revision_unlocked(
        self,
        entry: KnowledgeEntry,
        *,
        expected_revision: int,
    ) -> None:
        cursor = self._connection.execute(
            """
            UPDATE cayu_knowledge_entries
            SET current_revision = ?, updated_at = ?
            WHERE id = ? AND current_revision = ?
            """,
            (
                entry.revision,
                sqlite_support.format_datetime(entry.updated_at),
                entry.id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            current = self._load_entry_unlocked(entry.id)
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=None if current is None else current.revision,
            )

    def _append_revision_unlocked(
        self,
        entry: KnowledgeEntry,
        *,
        expected_revision: int,
        chunks: list[KnowledgeChunk] | None,
        evidence: list[KnowledgeEvidence] | None,
        access_scope: KnowledgeAccessScope,
        operation: str,
        change_kind: KnowledgeChangeKind,
        inherit_evidence: bool,
        change_operation_id: str | None = None,
        committed_at: datetime | None = None,
        allow_pending_maintenance_replacement: bool = False,
    ) -> None:
        _validate_revision_append(entry, expected_revision=expected_revision)
        current = self._load_entry_unlocked(entry.id)
        if current is None:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=None,
            )
        _require_knowledge_entry_access(access_scope, current, operation=operation)
        if current.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=current.revision,
            )
        if not allow_pending_maintenance_replacement:
            self._require_maintenance_replacement_mutation_allowed_unlocked(
                entry_id=current.id,
                entry_revision=current.revision,
                current_status=current.status,
                successor_status=entry.status,
                operation=operation,
            )
        _validate_revision_successor(current, entry)
        _require_knowledge_successor_access(access_scope, entry, operation=operation)
        if self._has_activation_receipts_unlocked(entry.id):
            _require_knowledge_activation_retirement_capacity(entry)
        previous_chunks = self._load_chunks_unlocked(
            entry.id,
            revision=current.revision,
        )
        if chunks is not None:
            copied_chunks = _copy_entry_chunks(entry.id, entry.revision, chunks)
        elif _has_only_default_chunk(current, previous_chunks):
            copied_chunks = [_default_chunk_for_entry(entry)]
        else:
            copied_chunks = _copy_chunks_for_revision(previous_chunks, entry)
        if inherit_evidence:
            if evidence is not None:
                raise ValueError("Lifecycle evidence inheritance cannot accept evidence.")
            copied_evidence = _copy_evidence_for_revision(
                self._load_evidence_unlocked(entry.id, revision=current.revision),
                entry=entry,
                previous_chunks=previous_chunks,
                chunks=copied_chunks,
            )
        else:
            copied_evidence = _copy_entry_evidence(
                entry.id,
                entry.revision,
                evidence or [],
                chunks=copied_chunks,
            )
        self._require_chunk_ids_available_unlocked(
            copied_chunks,
            access_scope=access_scope,
            operation=operation,
        )
        self._require_evidence_ids_available_unlocked(
            copied_evidence,
            access_scope=access_scope,
            operation=operation,
        )
        self._insert_revision_unlocked(entry)
        self._insert_chunks_unlocked(entry, copied_chunks)
        self._insert_evidence_unlocked(copied_evidence)
        self._advance_current_revision_unlocked(
            entry,
            expected_revision=expected_revision,
        )
        self._insert_change_unlocked(
            before_entry=current,
            after_entry=entry,
            kind=change_kind,
            operation_id=change_operation_id,
            committed_at=committed_at,
        )

    def _require_maintenance_replacement_mutation_allowed_unlocked(
        self,
        *,
        entry_id: str,
        entry_revision: int,
        current_status: KnowledgeStatus | None = None,
        successor_status: KnowledgeStatus | None = None,
        operation: str | None = None,
        preserve_history: bool = False,
    ) -> None:
        row = self._connection.execute(
            "SELECT proposal.proposal_id, proposal.replacement_revision, "
            "proposal.proposal_fingerprint, "
            "decision.operation_id AS decision_operation_id, "
            "decision.proposal_json AS decision_proposal_json, "
            "decision.decision_json, decision.receipt_json "
            "FROM cayu_knowledge_maintenance_proposals AS proposal "
            "LEFT JOIN cayu_knowledge_maintenance_decisions AS decision "
            "ON decision.proposal_id = proposal.proposal_id "
            "WHERE proposal.replacement_entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            return
        if preserve_history:
            raise KnowledgeMaintenanceConflict("maintenance_replacement_history_owned")
        replacement_revision = int(row["replacement_revision"])
        decision_operation_id = row["decision_operation_id"]
        if decision_operation_id is None:
            raise KnowledgeMaintenanceConflict("pending_replacement_lifecycle_owned")
        try:
            proposal = KnowledgeMaintenanceProposal.model_validate_json(
                row["decision_proposal_json"]
            )
            decision = KnowledgeMaintenanceDecision.model_validate_json(row["decision_json"])
            receipt = KnowledgeMaintenanceDecisionReceipt.model_validate_json(row["receipt_json"])
            if (
                proposal.id != str(row["proposal_id"])
                or proposal.fingerprint != str(row["proposal_fingerprint"])
                or proposal.replacement.entry_id != entry_id
                or proposal.replacement.revision != replacement_revision
                or decision.operation_id != str(decision_operation_id)
                or decision.proposal_id != str(row["proposal_id"])
                or receipt.operation_id != decision.operation_id
                or receipt.proposal_id != decision.proposal_id
            ):
                raise ValueError("Maintenance decision binding is inconsistent.")
            _validate_knowledge_maintenance_record(proposal, decision, receipt)
        except Exception:
            raise KnowledgeMaintenanceConflict("malformed_proposal_publication") from None
        if decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE:
            if entry_revision > replacement_revision:
                return
            raise KnowledgeMaintenanceConflict("pending_replacement_lifecycle_owned")
        if (
            operation in {"delete_entry", "transition_entry_status"}
            and (current_status, successor_status)
            in _MAINTENANCE_REJECTED_REPLACEMENT_RETIREMENT_TRANSITIONS
        ):
            return
        raise KnowledgeMaintenanceConflict("rejected_replacement_lifecycle_owned")

    def _insert_chunks_unlocked(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_chunks (
                id,
                entry_id,
                entry_revision,
                chunk_index,
                text,
                content_hash,
                source_uri,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_chunk_row_values(chunk) for chunk in chunks],
        )
        self._insert_entry_fts_unlocked(entry, chunks)

    def _insert_evidence_unlocked(self, evidence: list[KnowledgeEvidence]) -> None:
        if not evidence:
            return
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_evidence (
                id,
                entry_id,
                entry_revision,
                chunk_id,
                role,
                source_type,
                source_id,
                source_uri,
                source_revision,
                source_hash,
                locator_json,
                disposition,
                created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_evidence_row_values(item) for item in evidence],
        )

    def _load_evidence_unlocked(
        self,
        entry_id: str,
        *,
        revision: int,
        limit: int | None = None,
    ) -> list[KnowledgeEvidence]:
        limit_sql = "" if limit is None else " LIMIT ?"
        params: tuple[object, ...] = (
            (entry_id, revision) if limit is None else (entry_id, revision, limit)
        )
        rows = self._connection.execute(
            f"""
            SELECT
                id,
                entry_id,
                entry_revision,
                chunk_id,
                role,
                source_type,
                source_id,
                source_uri,
                source_revision,
                source_hash,
                locator_json,
                disposition,
                created_at,
                metadata_json
            FROM cayu_knowledge_evidence
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY id COLLATE BINARY
            {limit_sql}
            """,
            params,
        ).fetchall()
        return [_evidence_from_row(row) for row in rows]

    def _count_evidence_unlocked(self, entry_id: str, *, revision: int) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS evidence_count
            FROM cayu_knowledge_evidence
            WHERE entry_id = ? AND entry_revision = ?
            """,
            (entry_id, revision),
        ).fetchone()
        return 0 if row is None else int(row["evidence_count"])

    def _require_evidence_ids_available_unlocked(
        self,
        evidence: list[KnowledgeEvidence],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = sorted({item.id for item in evidence})
        occupied_entry_ids: set[str] = set()
        for offset in range(0, len(proposed_ids), _EVIDENCE_ID_LOOKUP_BATCH_SIZE):
            batch = proposed_ids[offset : offset + _EVIDENCE_ID_LOOKUP_BATCH_SIZE]
            placeholders = ", ".join("?" for _ in batch)
            rows = self._connection.execute(
                f"""
                SELECT DISTINCT entry_id
                FROM cayu_knowledge_evidence
                WHERE id IN ({placeholders})
                ORDER BY entry_id
                """,
                batch,
            ).fetchall()
            occupied_entry_ids.update(str(row["entry_id"]) for row in rows)
        for occupied_entry_id in sorted(occupied_entry_ids):
            owner = self._load_entry_unlocked(occupied_entry_id)
            if owner is None:
                raise KnowledgeEvidenceConflict(operation)
            _require_knowledge_entry_access(
                access_scope,
                owner,
                operation=operation,
            )
        if occupied_entry_ids:
            raise KnowledgeEvidenceConflict(operation)

    def _insert_change_unlocked(
        self,
        *,
        before_entry: KnowledgeEntry | None,
        after_entry: KnowledgeEntry | None,
        kind: KnowledgeChangeKind,
        operation_id: str | None = None,
        committed_at: datetime | None = None,
    ) -> KnowledgeChange:
        entry = after_entry if after_entry is not None else before_entry
        if entry is None:
            raise ValueError("A knowledge change requires a before or after entry.")
        change_id = f"kchg_{uuid4().hex}"
        committed_at = datetime.now(UTC) if committed_at is None else committed_at
        before_requires_include_expired: bool | None = None
        if (
            before_entry is not None
            and before_entry.expires_at is not None
            and before_entry.expires_at <= committed_at
        ):
            audience_row = self._connection.execute(
                """
                SELECT audience.requires_include_expired
                FROM cayu_knowledge_changes AS change_record
                JOIN cayu_knowledge_change_audiences AS audience
                  ON audience.change_sequence = change_record.sequence
                 AND audience.audience_kind = 'after'
                WHERE change_record.entry_id = ?
                  AND change_record.entry_revision = ?
                ORDER BY change_record.sequence DESC
                LIMIT 1
                """,
                (before_entry.id, before_entry.revision),
            ).fetchone()
            if audience_row is not None:
                before_requires_include_expired = bool(audience_row["requires_include_expired"])
            else:
                baseline_row = self._connection.execute(
                    "SELECT applied_at FROM cayu_schema_migrations WHERE revision = 43"
                ).fetchone()
                if baseline_row is None:
                    raise RuntimeError("SQLite knowledge outbox baseline is missing.")
                before_requires_include_expired = (
                    before_entry.expires_at
                    <= sqlite_support.parse_datetime(baseline_row["applied_at"])
                )
        cursor = self._connection.execute(
            """
            INSERT INTO cayu_knowledge_changes (
                id,
                kind,
                entry_id,
                entry_revision,
                committed_at,
                operation_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                change_id,
                kind.value,
                entry.id,
                entry.revision,
                sqlite_support.format_datetime(committed_at),
                operation_id,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a knowledge change sequence.")
        sequence = int(cursor.lastrowid)
        change = KnowledgeChange(
            id=change_id,
            sequence=sequence,
            kind=kind,
            entry_id=entry.id,
            entry_revision=entry.revision,
            committed_at=committed_at,
            operation_id=operation_id,
        )
        audiences = _knowledge_change_audiences(
            change,
            before_entry=before_entry,
            after_entry=after_entry,
            before_requires_include_expired=before_requires_include_expired,
        )
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_change_audiences (
                change_sequence,
                audience_kind,
                namespace,
                visibility,
                source_type,
                source_id,
                status,
                requires_include_expired
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    sequence,
                    audience.kind,
                    audience.snapshot.namespace,
                    audience.snapshot.visibility.value,
                    audience.snapshot.source_type,
                    audience.snapshot.source_id,
                    audience.snapshot.status.value,
                    int(audience.requires_include_expired),
                )
                for audience in audiences
            ],
        )
        label_rows = [
            (sequence, audience.kind, key, value)
            for audience in audiences
            for key, value in sorted(audience.snapshot.labels.items())
        ]
        if label_rows:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_change_labels (
                    change_sequence, audience_kind, key, value
                )
                VALUES (?, ?, ?, ?)
                """,
                label_rows,
            )
        return change

    def _insert_relations_unlocked(self, relations: list[KnowledgeRelation]) -> None:
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_relations (
                id,
                subject_entry_id,
                subject_revision,
                object_entry_id,
                object_revision,
                kind,
                created_by_type,
                created_by,
                policy_id,
                created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_relation_row_values(relation) for relation in relations],
        )

    def _insert_relation_change_unlocked(
        self,
        relation: KnowledgeRelation,
        *,
        access_snapshot: _KnowledgeRelationAccessSnapshot,
        operation_id: str,
        committed_at: datetime,
    ) -> KnowledgeChange:
        change_id = f"kchg_{uuid4().hex}"
        cursor = self._connection.execute(
            """
            INSERT INTO cayu_knowledge_changes (
                id,
                kind,
                entry_id,
                entry_revision,
                committed_at,
                operation_id,
                relation_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                change_id,
                KnowledgeChangeKind.RELATION_PUBLISHED.value,
                relation.subject.entry_id,
                relation.subject.revision,
                sqlite_support.format_datetime(committed_at),
                operation_id,
                relation.id,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a knowledge change sequence.")
        change = KnowledgeChange(
            id=change_id,
            sequence=int(cursor.lastrowid),
            kind=KnowledgeChangeKind.RELATION_PUBLISHED,
            entry_id=relation.subject.entry_id,
            entry_revision=relation.subject.revision,
            committed_at=committed_at,
            operation_id=operation_id,
            relation_id=relation.id,
        )
        audiences = _knowledge_relation_change_audiences(
            change,
            access_snapshot=access_snapshot,
        )
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_change_audiences (
                change_sequence,
                audience_kind,
                namespace,
                visibility,
                source_type,
                source_id,
                status,
                requires_include_expired
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    change.sequence,
                    audience.kind,
                    audience.snapshot.namespace,
                    audience.snapshot.visibility.value,
                    audience.snapshot.source_type,
                    audience.snapshot.source_id,
                    audience.snapshot.status.value,
                    int(audience.requires_include_expired),
                )
                for audience in audiences
            ],
        )
        labels = [
            (change.sequence, audience.kind, key, value)
            for audience in audiences
            for key, value in sorted(audience.snapshot.labels.items())
        ]
        if labels:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_change_labels (
                    change_sequence, audience_kind, key, value
                )
                VALUES (?, ?, ?, ?)
                """,
                labels,
            )
        return change

    def _relation_endpoints_in_scope_unlocked(
        self,
        relation: KnowledgeRelation,
        access_scope: KnowledgeAccessScope,
    ) -> bool:
        return all(
            self._load_entry_in_scope_unlocked(
                reference.entry_id,
                access_scope,
                revision=reference.revision,
            )
            is not None
            for reference in (relation.subject, relation.object)
        )

    def _load_relation_receipt_unlocked(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> KnowledgeRelationPublicationReceipt | None:
        row = self._connection.execute(
            """
            SELECT operation_id, relation_ids_json, request_sha256,
                   committed_at, access_snapshots_json
            FROM cayu_knowledge_relation_publication_receipts
            WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            relation_ids = json.loads(row["relation_ids_json"])
            raw_snapshots = json.loads(row["access_snapshots_json"])
            receipt = KnowledgeRelationPublicationReceipt(
                operation_id=row["operation_id"],
                relation_ids=relation_ids,
                request_sha256=row["request_sha256"],
                committed_at=sqlite_support.parse_datetime(row["committed_at"]),
            )
            if type(raw_snapshots) is not list or len(raw_snapshots) != len(receipt.relation_ids):
                raise ValueError("Relation receipt access snapshots are malformed.")
            snapshots = [
                _parse_knowledge_relation_access_snapshot_json(json.dumps(raw_snapshot))
                for raw_snapshot in raw_snapshots
            ]
        except Exception:
            raise KnowledgeRelationConflict("malformed_receipt") from None
        authorized = all(
            _knowledge_scope_allows_relation_access_snapshot(access_scope, snapshot)
            for snapshot in snapshots
        )
        if not authorized:
            if deny_inaccessible:
                raise KnowledgeAccessDenied("publish_relations")
            return None
        return receipt

    def _insert_relation_receipt_unlocked(
        self,
        receipt: KnowledgeRelationPublicationReceipt,
        *,
        access_snapshots: list[_KnowledgeRelationAccessSnapshot],
    ) -> None:
        snapshots = [
            json.loads(_knowledge_relation_access_snapshot_json(snapshot))
            for snapshot in access_snapshots
        ]
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_relation_publication_receipts (
                operation_id,
                relation_ids_json,
                request_sha256,
                committed_at,
                access_snapshots_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                receipt.operation_id,
                json.dumps(receipt.relation_ids, ensure_ascii=False, separators=(",", ":")),
                receipt.request_sha256,
                sqlite_support.format_datetime(receipt.committed_at),
                json.dumps(snapshots, ensure_ascii=False, separators=(",", ":")),
            ),
        )

    def _load_maintenance_proposal_record_unlocked(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> (
        tuple[
            KnowledgeMaintenanceProposal,
            KnowledgeMaintenanceAcceptedPlan,
            KnowledgeMaintenanceProposalPublicationReceipt,
            _KnowledgeMaintenanceAccessSnapshot,
        ]
        | None
    ):
        from cayu.knowledge_maintenance_persistence import (
            KnowledgeMaintenanceAcceptedPlan,
            KnowledgeMaintenanceProposalPublicationConflict,
            KnowledgeMaintenanceProposalPublicationReceipt,
            prepare_knowledge_maintenance_proposal_publication,
        )

        access_row = self._connection.execute(
            "SELECT access_snapshot_json FROM cayu_knowledge_maintenance_proposals "
            "WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if access_row is None:
            return None
        try:
            snapshot = _parse_knowledge_maintenance_access_snapshot_json(
                access_row["access_snapshot_json"]
            )
        except Exception:
            raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt") from None
        if not _knowledge_scope_allows_maintenance_access_snapshot(access_scope, snapshot):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("publish_maintenance_proposal")
            return None

        row = self._connection.execute(
            """
            SELECT proposal_id, replacement_entry_id, replacement_revision,
                   proposal_fingerprint, accepted_plan_fingerprint, request_sha256,
                   committed_at, proposal_json, accepted_plan_json, receipt_json
            FROM cayu_knowledge_maintenance_proposals
            WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            proposal = KnowledgeMaintenanceProposal.model_validate_json(row["proposal_json"])
            accepted_plan = KnowledgeMaintenanceAcceptedPlan.model_validate_json(
                row["accepted_plan_json"]
            )
            receipt = KnowledgeMaintenanceProposalPublicationReceipt.model_validate_json(
                row["receipt_json"]
            )
            replacement = self._load_entry_unlocked(
                proposal.replacement.entry_id,
                revision=proposal.replacement.revision,
            )
            if replacement is None:
                raise ValueError("Published replacement is missing.")
            chunks = self._load_chunks_unlocked(
                replacement.id,
                revision=replacement.revision,
            )
            evidence = self._load_evidence_unlocked(
                replacement.id,
                revision=replacement.revision,
            )
            (
                prepared_operation,
                prepared_entry,
                prepared_chunks,
                prepared_evidence,
                prepared_proposal,
                prepared_plan,
                prepared_sha256,
            ) = prepare_knowledge_maintenance_proposal_publication(
                replacement,
                chunks,
                evidence=evidence,
                proposal=proposal,
                accepted_plan=accepted_plan,
                operation_id=operation_id,
            )
            if (
                prepared_operation != operation_id
                or prepared_entry != replacement
                or prepared_chunks != chunks
                or prepared_evidence != evidence
                or prepared_proposal != proposal
                or prepared_plan != accepted_plan
                or proposal.id != row["proposal_id"]
                or proposal.replacement.entry_id != row["replacement_entry_id"]
                or proposal.replacement.revision != row["replacement_revision"]
                or proposal.fingerprint != row["proposal_fingerprint"]
                or accepted_plan.fingerprint != row["accepted_plan_fingerprint"]
                or prepared_sha256 != row["request_sha256"]
                or receipt.operation_id != operation_id
                or receipt.proposal_id != proposal.id
                or receipt.proposal_fingerprint != proposal.fingerprint
                or receipt.accepted_plan_fingerprint != accepted_plan.fingerprint
                or receipt.request_sha256 != prepared_sha256
                or receipt.replacement != proposal.replacement
                or receipt.committed_at != sqlite_support.parse_datetime(row["committed_at"])
                or receipt.replayed
            ):
                raise ValueError("Proposal publication indexes conflict with content.")
        except KnowledgeMaintenanceProposalPublicationConflict:
            raise
        except Exception:
            raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt") from None
        return proposal, accepted_plan, receipt, snapshot

    def _insert_maintenance_proposal_record_unlocked(
        self,
        proposal: KnowledgeMaintenanceProposal,
        accepted_plan: KnowledgeMaintenanceAcceptedPlan,
        receipt: KnowledgeMaintenanceProposalPublicationReceipt,
        *,
        access_snapshot: _KnowledgeMaintenanceAccessSnapshot,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_maintenance_proposals (
                operation_id,
                proposal_id,
                replacement_entry_id,
                replacement_revision,
                proposal_fingerprint,
                accepted_plan_fingerprint,
                request_sha256,
                committed_at,
                proposal_json,
                accepted_plan_json,
                receipt_json,
                access_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.operation_id,
                receipt.proposal_id,
                proposal.replacement.entry_id,
                proposal.replacement.revision,
                receipt.proposal_fingerprint,
                receipt.accepted_plan_fingerprint,
                receipt.request_sha256,
                sqlite_support.format_datetime(receipt.committed_at),
                proposal.model_dump_json(warnings=False),
                accepted_plan.model_dump_json(warnings=False),
                receipt.model_dump_json(warnings=False),
                _knowledge_maintenance_access_snapshot_json(access_snapshot),
            ),
        )

    def _load_maintenance_record_unlocked(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> (
        tuple[
            KnowledgeMaintenanceProposal,
            KnowledgeMaintenanceDecision,
            KnowledgeMaintenanceDecisionReceipt,
            _KnowledgeMaintenanceAccessSnapshot,
        ]
        | None
    ):
        row = self._connection.execute(
            """
            SELECT proposal_id, proposal_fingerprint, request_sha256, committed_at,
                   proposal_json, decision_json, receipt_json, access_snapshot_json
            FROM cayu_knowledge_maintenance_decisions
            WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            proposal = KnowledgeMaintenanceProposal.model_validate_json(row["proposal_json"])
            decision = KnowledgeMaintenanceDecision.model_validate_json(row["decision_json"])
            receipt = KnowledgeMaintenanceDecisionReceipt.model_validate_json(row["receipt_json"])
            snapshot = _parse_knowledge_maintenance_access_snapshot_json(
                row["access_snapshot_json"]
            )
            if (
                decision.operation_id != operation_id
                or receipt.operation_id != operation_id
                or proposal.id != row["proposal_id"]
                or proposal.id != decision.proposal_id
                or proposal.id != receipt.proposal_id
                or proposal.fingerprint != row["proposal_fingerprint"]
                or proposal.fingerprint != decision.proposal_fingerprint
                or proposal.fingerprint != receipt.proposal_fingerprint
                or receipt.request_sha256 != row["request_sha256"]
                or receipt.committed_at != sqlite_support.parse_datetime(row["committed_at"])
            ):
                raise ValueError("Maintenance record indexes conflict with content.")
            _validate_knowledge_maintenance_record(proposal, decision, receipt)
        except KnowledgeMaintenanceConflict:
            raise
        except Exception:
            raise KnowledgeMaintenanceConflict("malformed_receipt") from None
        if not _knowledge_scope_allows_maintenance_access_snapshot(access_scope, snapshot):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("apply_maintenance_decision")
            return None
        return proposal, decision, receipt, snapshot

    def _insert_maintenance_record_unlocked(
        self,
        proposal: KnowledgeMaintenanceProposal,
        decision: KnowledgeMaintenanceDecision,
        receipt: KnowledgeMaintenanceDecisionReceipt,
        *,
        access_snapshot: _KnowledgeMaintenanceAccessSnapshot,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_maintenance_decisions (
                operation_id,
                proposal_id,
                proposal_fingerprint,
                request_sha256,
                committed_at,
                proposal_json,
                decision_json,
                receipt_json,
                access_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.operation_id,
                receipt.proposal_id,
                receipt.proposal_fingerprint,
                receipt.request_sha256,
                sqlite_support.format_datetime(receipt.committed_at),
                proposal.model_dump_json(warnings=False),
                decision.model_dump_json(warnings=False),
                receipt.model_dump_json(warnings=False),
                _knowledge_maintenance_access_snapshot_json(access_snapshot),
            ),
        )

    def _accessible_change_high_water_unlocked(
        self,
        scope: KnowledgeAccessScope,
    ) -> int:
        access_sql, access_params = _sqlite_change_access_scope_filter_sql(
            scope,
            alias="change_record",
        )
        row = self._connection.execute(
            "SELECT COALESCE(MAX(change_record.sequence), 0) AS high_water "
            "FROM cayu_knowledge_changes AS change_record "
            f"WHERE 1 = 1 {access_sql}",
            access_params,
        ).fetchone()
        return 0 if row is None else int(row["high_water"])

    def _load_accessible_change_rows_unlocked(
        self,
        scope: KnowledgeAccessScope,
        *,
        after_sequence: int,
        through_sequence: int,
        limit: int,
    ) -> list[sqlite3.Row]:
        access_sql, access_params = _sqlite_change_access_scope_filter_sql(
            scope,
            alias="change_record",
        )
        return self._connection.execute(
            """
            SELECT
                change_record.id,
                change_record.sequence,
                change_record.kind,
                change_record.entry_id,
                change_record.entry_revision,
                change_record.committed_at,
                change_record.operation_id,
                change_record.relation_id
            FROM cayu_knowledge_changes AS change_record
            WHERE change_record.sequence > ?
              AND change_record.sequence <= ?
            """
            f"{access_sql} "
            "ORDER BY change_record.sequence LIMIT ?",
            (after_sequence, through_sequence, *access_params, limit),
        ).fetchall()

    def _load_change_in_scope_unlocked(
        self,
        sequence: int,
        scope: KnowledgeAccessScope,
    ) -> KnowledgeChange | None:
        access_sql, access_params = _sqlite_change_access_scope_filter_sql(
            scope,
            alias="change_record",
        )
        row = self._connection.execute(
            """
            SELECT
                change_record.id,
                change_record.sequence,
                change_record.kind,
                change_record.entry_id,
                change_record.entry_revision,
                change_record.committed_at,
                change_record.operation_id,
                change_record.relation_id
            FROM cayu_knowledge_changes AS change_record
            WHERE change_record.sequence = ?
            """
            f"{access_sql}",
            (sequence, *access_params),
        ).fetchone()
        return None if row is None else _change_from_row(row)

    def _next_accessible_change_unlocked(
        self,
        scope: KnowledgeAccessScope,
        *,
        after_sequence: int,
    ) -> KnowledgeChange | None:
        high_water = self._accessible_change_high_water_unlocked(scope)
        rows = self._load_accessible_change_rows_unlocked(
            scope,
            after_sequence=after_sequence,
            through_sequence=high_water,
            limit=1,
        )
        return None if not rows else _change_from_row(rows[0])

    def _load_change_consumer_unlocked(
        self,
        consumer_id: str,
    ) -> KnowledgeChangeConsumerState | None:
        row = self._connection.execute(
            """
            SELECT
                consumer_id,
                access_scope_sha256,
                cursor_sequence,
                pending_change_sequence,
                pending_claim_id,
                pending_worker_id,
                pending_attempt,
                claimed_at,
                lease_expires_at,
                last_acknowledged_claim_id,
                updated_at
            FROM cayu_knowledge_change_consumers
            WHERE consumer_id = ?
            """,
            (consumer_id,),
        ).fetchone()
        return None if row is None else _change_consumer_from_row(row)

    def _save_change_consumer_unlocked(self, state: KnowledgeChangeConsumerState) -> None:
        state = copy_knowledge_change_consumer_state(state)
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_change_consumers (
                consumer_id,
                access_scope_sha256,
                cursor_sequence,
                pending_change_sequence,
                pending_claim_id,
                pending_worker_id,
                pending_attempt,
                claimed_at,
                lease_expires_at,
                last_acknowledged_claim_id,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (consumer_id) DO UPDATE SET
                access_scope_sha256 = excluded.access_scope_sha256,
                cursor_sequence = excluded.cursor_sequence,
                pending_change_sequence = excluded.pending_change_sequence,
                pending_claim_id = excluded.pending_claim_id,
                pending_worker_id = excluded.pending_worker_id,
                pending_attempt = excluded.pending_attempt,
                claimed_at = excluded.claimed_at,
                lease_expires_at = excluded.lease_expires_at,
                last_acknowledged_claim_id = excluded.last_acknowledged_claim_id,
                updated_at = excluded.updated_at
            """,
            (
                state.consumer_id,
                state.access_scope_sha256,
                state.cursor_sequence,
                state.pending_change_sequence,
                state.pending_claim_id,
                state.pending_worker_id,
                state.pending_attempt,
                (
                    None
                    if state.claimed_at is None
                    else sqlite_support.format_datetime(state.claimed_at)
                ),
                (
                    None
                    if state.lease_expires_at is None
                    else sqlite_support.format_datetime(state.lease_expires_at)
                ),
                state.last_acknowledged_claim_id,
                sqlite_support.format_datetime(state.updated_at),
            ),
        )

    def _load_change_acknowledgement_unlocked(
        self,
        consumer_id: str,
        claim_id: str,
    ) -> tuple[str, int] | None:
        row = self._connection.execute(
            """
            SELECT claim_sha256, change_sequence
            FROM cayu_knowledge_change_acknowledgements
            WHERE consumer_id = ? AND claim_id = ?
            """,
            (consumer_id, claim_id),
        ).fetchone()
        if row is None:
            return None
        return str(row["claim_sha256"]), int(row["change_sequence"])

    def _insert_change_acknowledgement_unlocked(
        self,
        claim: KnowledgeChangeClaim,
        *,
        claim_sha256: str,
        acknowledged_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_change_acknowledgements (
                consumer_id,
                claim_id,
                claim_sha256,
                change_sequence,
                acknowledged_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                claim.consumer_id,
                claim.claim_id,
                claim_sha256,
                claim.change.sequence,
                sqlite_support.format_datetime(acknowledged_at),
            ),
        )

    def _require_matching_change_claim_unlocked(
        self,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
    ) -> None:
        row = self._connection.execute(
            """
            SELECT id, sequence, kind, entry_id, entry_revision,
                   committed_at, operation_id, relation_id
            FROM cayu_knowledge_changes
            WHERE sequence = ?
            """,
            (claim.change.sequence,),
        ).fetchone()
        stored_change = None if row is None else _change_from_row(row)
        if (
            state.pending_change_sequence != claim.change.sequence
            or state.pending_claim_id != claim.claim_id
            or state.pending_worker_id != claim.worker_id
            or state.pending_attempt != claim.attempt
            or stored_change != claim.change
        ):
            raise KnowledgeChangeConsumerConflict("stale_claim")

    def _require_live_change_claim_unlocked(
        self,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
        *,
        now: datetime,
    ) -> None:
        self._require_matching_change_claim_unlocked(state, claim)
        if state.lease_expires_at is None or state.lease_expires_at <= now:
            raise KnowledgeChangeConsumerConflict("expired_claim")

    def _require_chunk_ids_available_unlocked(
        self,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = sorted({chunk.id for chunk in chunks})
        occupied_entry_ids: set[str] = set()
        for offset in range(0, len(proposed_ids), _CHUNK_ID_LOOKUP_BATCH_SIZE):
            batch = proposed_ids[offset : offset + _CHUNK_ID_LOOKUP_BATCH_SIZE]
            placeholders = ", ".join("?" for _ in batch)
            rows = self._connection.execute(
                f"""
                SELECT DISTINCT entry_id
                FROM cayu_knowledge_chunks
                WHERE id IN ({placeholders})
                ORDER BY entry_id
                """,
                batch,
            ).fetchall()
            occupied_entry_ids.update(str(row["entry_id"]) for row in rows)
        for occupied_entry_id in sorted(occupied_entry_ids):
            owner = self._load_entry_unlocked(occupied_entry_id)
            if owner is None:
                raise KnowledgeChunkConflict(operation)
            _require_knowledge_entry_access(
                access_scope,
                owner,
                operation=operation,
            )
        if occupied_entry_ids:
            raise KnowledgeChunkConflict(operation)

    def _load_publication_receipt_unlocked(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgePublicationReceipt | None:
        row = self._connection.execute(
            """
            SELECT
                operation_id,
                entry_id,
                entry_revision,
                expected_revision,
                request_sha256,
                entry_created_at,
                entry_updated_at,
                committed_at,
                access_snapshot_json
            FROM cayu_knowledge_publication_receipts
            WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row["access_snapshot_json"])
            receipt = KnowledgePublicationReceipt(
                operation_id=row["operation_id"],
                entry_id=row["entry_id"],
                entry_revision=row["entry_revision"],
                expected_revision=row["expected_revision"],
                request_sha256=row["request_sha256"],
                entry_created_at=sqlite_support.parse_datetime(row["entry_created_at"]),
                entry_updated_at=sqlite_support.parse_datetime(row["entry_updated_at"]),
                committed_at=sqlite_support.parse_datetime(row["committed_at"]),
            )
        except Exception:
            raise KnowledgePublicationConflict("malformed_receipt") from None
        if not _knowledge_scope_allows_snapshot(access_scope, snapshot):
            raise KnowledgeAccessDenied("publish_entry_revision")
        return receipt

    def _load_publication_receipt_in_scope_unlocked(
        self,
        operation_id: str,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgePublicationReceipt | None:
        row = self._connection.execute(
            """
            SELECT
                receipt.operation_id,
                receipt.entry_id,
                receipt.entry_revision,
                receipt.expected_revision,
                receipt.request_sha256,
                receipt.entry_created_at,
                receipt.entry_updated_at,
                receipt.committed_at,
                receipt.access_snapshot_json
            FROM cayu_knowledge_publication_receipts AS receipt
            WHERE receipt.operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row["access_snapshot_json"])
            if not _knowledge_scope_allows_snapshot(access_scope, snapshot):
                return None
            return KnowledgePublicationReceipt(
                operation_id=row["operation_id"],
                entry_id=row["entry_id"],
                entry_revision=row["entry_revision"],
                expected_revision=row["expected_revision"],
                request_sha256=row["request_sha256"],
                entry_created_at=sqlite_support.parse_datetime(row["entry_created_at"]),
                entry_updated_at=sqlite_support.parse_datetime(row["entry_updated_at"]),
                committed_at=sqlite_support.parse_datetime(row["committed_at"]),
            )
        except Exception:
            raise KnowledgePublicationConflict("malformed_receipt") from None

    def _insert_publication_receipt_unlocked(
        self,
        receipt: KnowledgePublicationReceipt,
        entry: KnowledgeEntry,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_publication_receipts (
                operation_id,
                entry_id,
                entry_revision,
                expected_revision,
                request_sha256,
                entry_created_at,
                entry_updated_at,
                committed_at,
                access_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.operation_id,
                receipt.entry_id,
                receipt.entry_revision,
                receipt.expected_revision,
                receipt.request_sha256,
                sqlite_support.format_datetime(receipt.entry_created_at),
                sqlite_support.format_datetime(receipt.entry_updated_at),
                sqlite_support.format_datetime(receipt.committed_at),
                _knowledge_access_snapshot_json(_knowledge_access_snapshot(entry)),
            ),
        )

    def _has_activation_receipts_unlocked(self, entry_id: str) -> bool:
        return bool(
            self._connection.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM cayu_knowledge_activation_receipts "
                "WHERE entry_id = ? LIMIT 1)",
                (entry_id,),
            ).fetchone()[0]
        )

    def _load_activation_receipt_unlocked(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> KnowledgeActivationReceipt | None:
        row = self._connection.execute(
            """
            SELECT operation_id, entry_id, entry_revision, expected_revision,
                   publication_request_sha256, committed_at,
                   receipt_json, access_snapshot_json
            FROM cayu_knowledge_activation_receipts
            WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row["access_snapshot_json"])
        except Exception:
            raise KnowledgeActivationConflict("malformed_receipt") from None
        cutoff = datetime.now(UTC)
        if not _knowledge_scope_allows_snapshot(access_scope, snapshot, now=cutoff):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("load_activation_receipt")
            return None
        access_sql, access_params = _knowledge_access_scope_filter_sql(
            access_scope,
            now=cutoff,
        )
        current_access = self._connection.execute(
            f"""
            SELECT
                EXISTS (
                    SELECT 1
                    FROM cayu_knowledge_current_entries AS e
                    WHERE e.id = ?
                ) AS entry_exists,
                EXISTS (
                    SELECT 1
                    FROM cayu_knowledge_current_entries AS e
                    WHERE e.id = ?
                    {access_sql}
                ) AS entry_allowed
            """,
            (row["entry_id"], row["entry_id"], *access_params),
        ).fetchone()
        if current_access is None:
            raise KnowledgeActivationConflict("malformed_receipt")
        current_exists = bool(current_access["entry_exists"])
        retirement = self._load_activation_retirement_unlocked(str(row["entry_id"]))
        if current_exists:
            if retirement is not None:
                raise KnowledgeActivationConflict("malformed_retirement")
            current_allowed = bool(current_access["entry_allowed"])
        else:
            current_allowed = _knowledge_scope_allows_activation_receipt(
                access_scope,
                snapshot,
                None,
                retirement=retirement,
                entry_id=str(row["entry_id"]),
                entry_revision=int(row["entry_revision"]),
                now=cutoff,
            )
        if not current_allowed:
            if deny_inaccessible:
                raise KnowledgeAccessDenied("load_activation_receipt")
            return None
        try:
            receipt = KnowledgeActivationReceipt.model_validate_json(row["receipt_json"])
            if (
                receipt.operation_id != row["operation_id"]
                or receipt.entry_id != row["entry_id"]
                or receipt.entry_revision != row["entry_revision"]
                or receipt.expected_revision != row["expected_revision"]
                or receipt.publication_request_sha256 != row["publication_request_sha256"]
                or receipt.committed_at != sqlite_support.parse_datetime(row["committed_at"])
            ):
                raise ValueError("Activation receipt columns disagree with its JSON envelope.")
        except Exception:
            raise KnowledgeActivationConflict("malformed_receipt") from None
        return receipt

    def _load_activation_retirement_unlocked(
        self,
        entry_id: str,
    ) -> _KnowledgeActivationRetirement | None:
        row = self._connection.execute(
            "SELECT entry_id, entry_revision, retired_at, retirement_json "
            "FROM cayu_knowledge_activation_retirements WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            retirement = _parse_knowledge_activation_retirement_json(row["retirement_json"])
            if (
                retirement.entry_id != row["entry_id"]
                or retirement.entry_revision != row["entry_revision"]
                or retirement.retired_at != sqlite_support.parse_datetime(row["retired_at"])
            ):
                raise ValueError("Activation retirement columns disagree with its envelope.")
        except Exception:
            raise KnowledgeActivationConflict("malformed_retirement") from None
        return retirement

    def _insert_activation_retirement_unlocked(
        self,
        retirement: _KnowledgeActivationRetirement,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_activation_retirements (
                entry_id, entry_revision, retired_at, retirement_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                retirement.entry_id,
                retirement.entry_revision,
                sqlite_support.format_datetime(retirement.retired_at),
                _knowledge_activation_retirement_json(retirement),
            ),
        )

    def _insert_activation_receipt_unlocked(
        self,
        receipt: KnowledgeActivationReceipt,
        *,
        access_entry: KnowledgeEntry,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_activation_receipts (
                operation_id,
                entry_id,
                entry_revision,
                expected_revision,
                publication_request_sha256,
                committed_at,
                receipt_json,
                access_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.operation_id,
                receipt.entry_id,
                receipt.entry_revision,
                receipt.expected_revision,
                receipt.publication_request_sha256,
                sqlite_support.format_datetime(receipt.committed_at),
                _knowledge_activation_receipt_json(receipt),
                _knowledge_access_snapshot_json(_knowledge_access_snapshot(access_entry)),
            ),
        )

    def _delete_chunks_unlocked(self, entry_id: str) -> None:
        self._delete_entry_fts_unlocked(entry_id)
        self._connection.execute(
            "DELETE FROM cayu_knowledge_chunks WHERE entry_id = ?",
            (entry_id,),
        )

    def _delete_entry_fts_unlocked(self, entry_id: str) -> None:
        rowids = self._load_chunk_fts_rowids_unlocked(entry_id)
        if not rowids:
            return
        self._connection.executemany(
            "DELETE FROM cayu_knowledge_chunks_fts WHERE rowid = ?",
            [(rowid,) for rowid in rowids.values()],
        )

    def _insert_entry_fts_unlocked(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> None:
        rowids = self._load_chunk_fts_rowids_unlocked(entry.id, entry.revision)
        expected_chunk_ids = {chunk.id for chunk in chunks}
        if rowids.keys() != expected_chunk_ids:
            raise RuntimeError("SQLite knowledge chunks changed while preparing their FTS rows.")
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_chunks_fts (
                rowid, entry_id, entry_revision, chunk_id, title, text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rowids[chunk.id],
                    entry.id,
                    entry.revision,
                    chunk.id,
                    entry.title or "",
                    _fts_text_for_entry_chunk(entry, chunk),
                )
                for chunk in chunks
            ],
        )

    def _load_chunk_fts_rowids_unlocked(
        self,
        entry_id: str,
        revision: int | None = None,
    ) -> dict[str, int]:
        revision_sql = "" if revision is None else " AND entry_revision = ?"
        params: tuple[object, ...] = (entry_id,) if revision is None else (entry_id, revision)
        rows = self._connection.execute(
            f"""
            SELECT id, fts_rowid
            FROM cayu_knowledge_chunks
            WHERE entry_id = ?
            {revision_sql}
            ORDER BY chunk_index ASC
            """,
            params,
        ).fetchall()
        return {str(row["id"]): int(row["fts_rowid"]) for row in rows}

    def _load_entry_unlocked(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
    ) -> KnowledgeEntry | None:
        if revision is None:
            row = self._connection.execute(
                "SELECT * FROM cayu_knowledge_current_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT
                    logical.id AS id,
                    revision.revision AS revision,
                    logical.namespace AS namespace,
                    revision.*
                FROM cayu_knowledge_entries AS logical
                JOIN cayu_knowledge_revisions AS revision
                  ON revision.entry_id = logical.id
                WHERE logical.id = ? AND revision.revision = ?
                """,
                (entry_id, revision),
            ).fetchone()
        if row is None:
            return None
        selected_revision = int(row["revision"])
        return _entry_from_row(
            row,
            labels=self._load_labels_unlocked(entry_id, selected_revision),
            aspects=self._load_aspects_unlocked(entry_id, selected_revision),
            impact_targets=self._load_impact_targets_unlocked(entry_id, selected_revision),
        )

    def _load_entry_at_change_sequence_unlocked(
        self,
        entry_id: str,
        *,
        through_sequence: int,
    ) -> KnowledgeEntry | None:
        row = self._connection.execute(
            """
            SELECT candidate.entry_revision
            FROM cayu_knowledge_changes AS candidate
            WHERE candidate.entry_id = ?
              AND candidate.kind <> 'relation_published'
              AND candidate.sequence <= ?
              AND candidate.sequence = (
                  SELECT MAX(materialization.sequence)
                  FROM cayu_knowledge_changes AS materialization
                  WHERE materialization.entry_id = candidate.entry_id
                    AND materialization.entry_revision = candidate.entry_revision
                    AND materialization.kind <> 'relation_published'
              )
            ORDER BY candidate.sequence DESC
            LIMIT 1
            """,
            (entry_id, through_sequence),
        ).fetchone()
        if row is None:
            return None
        return self._load_entry_unlocked(
            entry_id,
            revision=int(row["entry_revision"]),
        )

    def _load_entry_in_scope_unlocked(
        self,
        entry_id: str,
        access_scope: KnowledgeAccessScope,
        *,
        revision: int | None = None,
        access_now: datetime | None = None,
    ) -> KnowledgeEntry | None:
        if access_now is None:
            access_now = datetime.now(UTC)
        access_sql, access_params = _knowledge_access_scope_filter_sql(
            access_scope,
            now=access_now,
        )
        if revision is None:
            row = self._connection.execute(
                f"""
                SELECT e.*
                FROM cayu_knowledge_current_entries AS e
                WHERE e.id = ?
                {access_sql}
                """,
                [entry_id, *access_params],
            ).fetchone()
        else:
            current_access_sql, current_access_params = _knowledge_access_scope_filter_sql(
                access_scope,
                entry_alias="current_entry",
                now=access_now,
            )
            row = self._connection.execute(
                f"""
                SELECT e.*
                FROM (
                    SELECT
                        logical.id AS id,
                        stored.revision AS revision,
                        logical.namespace AS namespace,
                        stored.text AS text,
                        stored.kind AS kind,
                        stored.visibility AS visibility,
                        stored.status AS status,
                        stored.created_by_type AS created_by_type,
                        stored.created_by AS created_by,
                        stored.created_at AS created_at,
                        stored.updated_at AS updated_at,
                        stored.source_type AS source_type,
                        stored.source_uri AS source_uri,
                        stored.source_id AS source_id,
                        stored.source_hash AS source_hash,
                        stored.importance AS importance,
                        stored.importance_source AS importance_source,
                        stored.confidence AS confidence,
                        stored.last_used_at AS last_used_at,
                        stored.expires_at AS expires_at,
                        stored.title AS title,
                        stored.metadata_json AS metadata_json
                    FROM cayu_knowledge_entries AS logical
                    JOIN cayu_knowledge_revisions AS stored
                      ON stored.entry_id = logical.id
                    WHERE logical.id = ? AND stored.revision = ?
                ) AS e
                JOIN cayu_knowledge_current_entries AS current_entry
                  ON current_entry.id = e.id
                WHERE TRUE
                {access_sql}
                {current_access_sql}
                """,
                [
                    entry_id,
                    revision,
                    *access_params,
                    *current_access_params,
                ],
            ).fetchone()
        if row is None:
            return None
        selected_revision = int(row["revision"])
        return _entry_from_row(
            row,
            labels=self._load_labels_unlocked(entry_id, selected_revision),
            aspects=self._load_aspects_unlocked(entry_id, selected_revision),
            impact_targets=self._load_impact_targets_unlocked(entry_id, selected_revision),
        )

    def _load_entry_payload_bytes_in_scope_unlocked(
        self,
        entry_id: str,
        access_scope: KnowledgeAccessScope,
        *,
        revision: int | None = None,
        access_now: datetime,
    ) -> tuple[int, int] | None:
        access_sql, access_params = _knowledge_access_scope_filter_sql(
            access_scope,
            now=access_now,
        )
        if revision is None:
            row = self._connection.execute(
                f"""
                SELECT e.revision, e.payload_bytes
                FROM cayu_knowledge_current_entries AS e
                WHERE e.id = ?
                {access_sql}
                """,
                [entry_id, *access_params],
            ).fetchone()
        else:
            current_access_sql, current_access_params = _knowledge_access_scope_filter_sql(
                access_scope,
                entry_alias="current_entry",
                now=access_now,
            )
            row = self._connection.execute(
                f"""
                SELECT e.revision, e.payload_bytes
                FROM (
                    SELECT
                        logical.id AS id,
                        stored.revision AS revision,
                        logical.namespace AS namespace,
                        stored.visibility AS visibility,
                        stored.status AS status,
                        stored.source_type AS source_type,
                        stored.source_id AS source_id,
                        stored.expires_at AS expires_at,
                        stored.payload_bytes AS payload_bytes
                    FROM cayu_knowledge_entries AS logical
                    JOIN cayu_knowledge_revisions AS stored
                      ON stored.entry_id = logical.id
                    WHERE logical.id = ? AND stored.revision = ?
                ) AS e
                JOIN cayu_knowledge_current_entries AS current_entry
                  ON current_entry.id = e.id
                WHERE TRUE
                {access_sql}
                {current_access_sql}
                """,
                [
                    entry_id,
                    revision,
                    *access_params,
                    *current_access_params,
                ],
            ).fetchone()
        if row is None:
            return None
        return int(row["revision"]), int(row["payload_bytes"])

    def _load_chunks_unlocked(
        self,
        entry_id: str,
        *,
        revision: int,
    ) -> list[KnowledgeChunk]:
        rows = self._connection.execute(
            """
            SELECT *
            FROM cayu_knowledge_chunks
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY chunk_index ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def _load_chunks_for_revision_refs_unlocked(
        self,
        revision_refs: tuple[KnowledgeRevisionRef, ...],
    ) -> dict[tuple[str, int], list[KnowledgeChunk]]:
        if not revision_refs:
            return {}
        values = ", ".join("(?, ?)" for _ in revision_refs)
        params: list[object] = []
        for reference in revision_refs:
            params.extend((reference.entry_id, reference.revision))
        rows = self._connection.execute(
            f"""
            SELECT chunk.*
            FROM cayu_knowledge_chunks AS chunk
            WHERE (chunk.entry_id, chunk.entry_revision) IN (VALUES {values})
            ORDER BY chunk.entry_id ASC,
                     chunk.entry_revision ASC,
                     chunk.chunk_index ASC
            """,
            params,
        ).fetchall()
        chunks_by_revision: dict[tuple[str, int], list[KnowledgeChunk]] = {}
        for row in rows:
            key = (str(row["entry_id"]), int(row["entry_revision"]))
            chunks_by_revision.setdefault(key, []).append(_chunk_from_row(row))
        return chunks_by_revision

    def _load_revision_materialization_sequences_unlocked(
        self,
        revision_refs: tuple[KnowledgeRevisionRef, ...],
    ) -> dict[tuple[str, int], int]:
        if not revision_refs:
            return {}
        values = ", ".join("(?, ?)" for _ in revision_refs)
        params: list[object] = []
        for reference in revision_refs:
            params.extend((reference.entry_id, reference.revision))
        rows = self._connection.execute(
            f"""
            SELECT
                change_record.entry_id,
                change_record.entry_revision,
                MAX(change_record.sequence) AS materialization_sequence
            FROM cayu_knowledge_changes AS change_record
            WHERE change_record.kind <> 'relation_published'
              AND (change_record.entry_id, change_record.entry_revision)
                  IN (VALUES {values})
            GROUP BY change_record.entry_id, change_record.entry_revision
            """,
            params,
        ).fetchall()
        return {
            (str(row["entry_id"]), int(row["entry_revision"])): int(row["materialization_sequence"])
            for row in rows
        }

    def _load_entries_unlocked(
        self,
        entry_ids: list[str],
        *,
        access_scope: KnowledgeAccessScope | None = None,
        access_now: datetime | None = None,
    ) -> dict[str, KnowledgeEntry]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        access_sql, access_params = (
            ("", [])
            if access_scope is None
            else _knowledge_access_scope_filter_sql(access_scope, now=access_now)
        )
        rows = self._connection.execute(
            f"""
            SELECT e.*
            FROM cayu_knowledge_current_entries AS e
            WHERE e.id IN ({placeholders})
            {access_sql}
            """,
            [*unique_ids, *access_params],
        ).fetchall()
        loaded_ids = [str(row["id"]) for row in rows]
        labels = self._load_labels_for_entries_unlocked(loaded_ids)
        aspects = self._load_aspects_for_entries_unlocked(loaded_ids)
        impact_targets = self._load_impact_targets_for_entries_unlocked(loaded_ids)
        return {
            row["id"]: _entry_from_row(
                row,
                labels=labels.get(row["id"], {}),
                aspects=aspects.get(row["id"], []),
                impact_targets=impact_targets.get(row["id"], []),
            )
            for row in rows
        }

    def _load_chunks_by_ids_unlocked(self, chunk_ids: list[str]) -> dict[str, KnowledgeChunk]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self._connection.execute(
            f"SELECT * FROM cayu_knowledge_chunks WHERE id IN ({placeholders})",
            unique_ids,
        ).fetchall()
        return {row["id"]: _chunk_from_row(row) for row in rows}

    def _count_chunks_by_entry_unlocked(self, entry_ids: list[str]) -> dict[str, int]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self._connection.execute(
            f"""
            SELECT chunk.entry_id, COUNT(*) AS chunk_count
            FROM cayu_knowledge_chunks AS chunk
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = chunk.entry_id
             AND logical.current_revision = chunk.entry_revision
            WHERE chunk.entry_id IN ({placeholders})
            GROUP BY chunk.entry_id
            """,
            unique_ids,
        ).fetchall()
        return {row["entry_id"]: int(row["chunk_count"]) for row in rows}

    def _load_labels_for_entries_unlocked(
        self,
        entry_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        if not entry_ids:
            return {}
        placeholders = ", ".join("?" for _ in entry_ids)
        rows = self._connection.execute(
            f"""
            SELECT label.entry_id, label.key, label.value
            FROM cayu_knowledge_labels AS label
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = label.entry_id
             AND logical.current_revision = label.entry_revision
            WHERE label.entry_id IN ({placeholders})
            ORDER BY label.entry_id ASC, label.key ASC
            """,
            entry_ids,
        ).fetchall()
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            result.setdefault(row["entry_id"], {})[row["key"]] = row["value"]
        return result

    def _load_aspects_for_entries_unlocked(
        self,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        placeholders = ", ".join("?" for _ in entry_ids)
        rows = self._connection.execute(
            f"""
            SELECT aspect.entry_id, aspect.aspect
            FROM cayu_knowledge_aspects AS aspect
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = aspect.entry_id
             AND logical.current_revision = aspect.entry_revision
            WHERE aspect.entry_id IN ({placeholders})
            ORDER BY aspect.entry_id ASC, aspect.aspect ASC
            """,
            entry_ids,
        ).fetchall()
        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(row["entry_id"], []).append(row["aspect"])
        return result

    def _load_impact_targets_for_entries_unlocked(
        self,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        placeholders = ", ".join("?" for _ in entry_ids)
        rows = self._connection.execute(
            f"""
            SELECT target.entry_id, target.impact_target
            FROM cayu_knowledge_impact_targets AS target
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = target.entry_id
             AND logical.current_revision = target.entry_revision
            WHERE target.entry_id IN ({placeholders})
            ORDER BY target.entry_id ASC, target.impact_target ASC
            """,
            entry_ids,
        ).fetchall()
        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(row["entry_id"], []).append(row["impact_target"])
        return result

    def _load_labels_unlocked(self, entry_id: str, revision: int) -> dict[str, str]:
        rows = self._connection.execute(
            """
            SELECT key, value
            FROM cayu_knowledge_labels
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY key ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _load_aspects_unlocked(self, entry_id: str, revision: int) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT aspect
            FROM cayu_knowledge_aspects
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY aspect ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return [row["aspect"] for row in rows]

    def _load_impact_targets_unlocked(self, entry_id: str, revision: int) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT impact_target
            FROM cayu_knowledge_impact_targets
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY impact_target ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return [row["impact_target"] for row in rows]


def _knowledge_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    return _knowledge_metadata_filter_sql(
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        aspect_groups=query.aspect_groups,
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _knowledge_list_filter_sql(query: KnowledgeListQuery) -> tuple[str, list[object]]:
    return _knowledge_metadata_filter_sql(
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        aspect_groups=[],
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _knowledge_access_scope_filter_sql(
    scope: KnowledgeAccessScope,
    *,
    entry_alias: str = "e",
    now: datetime | None = None,
) -> tuple[str, list[object]]:
    if entry_alias not in {"e", "current_entry"}:
        raise ValueError("Unsupported knowledge access-filter alias.")
    clauses: list[str] = []
    params: list[object] = []
    if not scope.allow_all_namespaces:
        placeholders = ", ".join("?" for _ in scope.allowed_namespaces)
        clauses.append(f"{entry_alias}.namespace IN ({placeholders})")
        params.extend(scope.allowed_namespaces)
    for key, value in scope.required_labels.items():
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_labels AS access_label
                WHERE access_label.entry_id = {entry_alias}.id
                  AND access_label.entry_revision = {entry_alias}.revision
                  AND access_label.key = ?
                  AND access_label.value = ?
            )
            """
        )
        params.extend([key, value])
    visibility_placeholders = ", ".join("?" for _ in scope.allowed_visibilities)
    clauses.append(f"{entry_alias}.visibility IN ({visibility_placeholders})")
    params.extend(str(visibility) for visibility in scope.allowed_visibilities)
    status_placeholders = ", ".join("?" for _ in scope.allowed_statuses)
    clauses.append(f"{entry_alias}.status IN ({status_placeholders})")
    params.extend(str(status) for status in scope.allowed_statuses)
    if scope.allowed_source_types is not None:
        if scope.allowed_source_types:
            placeholders = ", ".join("?" for _ in scope.allowed_source_types)
            clauses.append(f"{entry_alias}.source_type IN ({placeholders})")
            params.extend(scope.allowed_source_types)
        else:
            clauses.append("0")
    if scope.allowed_source_ids is not None:
        if scope.allowed_source_ids:
            placeholders = ", ".join("?" for _ in scope.allowed_source_ids)
            clauses.append(f"{entry_alias}.source_id IN ({placeholders})")
            params.extend(scope.allowed_source_ids)
        else:
            clauses.append("0")
    if not scope.include_expired:
        clauses.append(f"({entry_alias}.expires_at IS NULL OR {entry_alias}.expires_at > ?)")
        params.append(sqlite_support.format_datetime(datetime.now(UTC) if now is None else now))
    return " AND " + " AND ".join(clauses), params


def _knowledge_metadata_filter_sql(
    *,
    namespace: str | None,
    labels: dict[str, str],
    kinds: list[str] | None,
    statuses: list[KnowledgeStatus],
    visibilities: list[KnowledgeVisibility] | None,
    aspects: list[str],
    aspect_groups: list[list[str]],
    impact_targets: list[str],
    source_type: str | None,
    source_id: str | None,
    include_expired: bool,
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if namespace is not None:
        clauses.append("e.namespace = ?")
        params.append(namespace)
    for key, value in labels.items():
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_labels AS label
                WHERE label.entry_id = e.id
                  AND label.entry_revision = e.revision
                  AND label.key = ?
                  AND label.value = ?
            )
            """
        )
        params.extend([key, value])
    if kinds is not None:
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            clauses.append(f"e.kind IN ({placeholders})")
            params.extend(kinds)
        else:
            clauses.append("0")
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        clauses.append(f"e.status IN ({placeholders})")
        params.extend(str(status) for status in statuses)
    if visibilities is not None:
        placeholders = ", ".join("?" for _ in visibilities)
        clauses.append(f"e.visibility IN ({placeholders})")
        params.extend(str(visibility) for visibility in visibilities)
    if source_type is not None:
        clauses.append("e.source_type = ?")
        params.append(source_type)
    if source_id is not None:
        clauses.append("e.source_id = ?")
        params.append(source_id)
    if aspects:
        placeholders = ", ".join("?" for _ in aspects)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_aspects AS aspect
                WHERE aspect.entry_id = e.id
                  AND aspect.entry_revision = e.revision
                  AND aspect.aspect IN ({placeholders})
            )
            """
        )
        params.extend(aspects)
    for group in aspect_groups:
        placeholders = ", ".join("?" for _ in group)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_aspects AS grouped_aspect
                WHERE grouped_aspect.entry_id = e.id
                  AND grouped_aspect.entry_revision = e.revision
                  AND grouped_aspect.aspect IN ({placeholders})
            )
            """
        )
        params.extend(group)
    if impact_targets:
        placeholders = ", ".join("?" for _ in impact_targets)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_impact_targets AS target
                WHERE target.entry_id = e.id
                  AND target.entry_revision = e.revision
                  AND target.impact_target IN ({placeholders})
            )
            """
        )
        params.extend(impact_targets)
    if not include_expired:
        clauses.append("(e.expires_at IS NULL OR e.expires_at > ?)")
        params.append(sqlite_support.format_datetime(datetime.now(UTC)))
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _sqlite_knowledge_fts_query(query: KnowledgeQuery) -> tuple[str | None, list[str]]:
    any_terms = _dedupe_search_tokens(
        [
            *_expand_search_tokens(_tokenize_search_text(query.text or "")),
            *(
                token
                for term in query.any_terms
                for group in _structured_search_token_groups(term)
                for token in group
            ),
        ]
    )
    all_groups = _dedupe_search_token_groups(
        [group for term in query.all_terms for group in _structured_search_token_groups(term)]
    )
    phrases = [phrase.casefold() for phrase in query.phrases]
    positive_parts: list[str] = []
    if any_terms:
        positive_parts.append(
            "(" + " OR ".join(_sqlite_fts_quote(term) for term in any_terms) + ")"
        )
    positive_parts.extend(
        "(" + " OR ".join(_sqlite_fts_quote(term) for term in group) + ")" for group in all_groups
    )
    if phrases:
        positive_parts.append(
            "(" + " OR ".join(_sqlite_fts_quote(phrase) for phrase in phrases) + ")"
        )
    if not positive_parts:
        return None, []
    fts_query = " AND ".join(positive_parts)
    preview_terms = _dedupe_search_tokens(
        [
            *any_terms,
            *(term for group in all_groups for term in group),
            *_tokenize_search_text(" ".join(phrases)),
        ]
    )
    return fts_query, preview_terms


def _sqlite_knowledge_none_fts_query(query: KnowledgeQuery) -> str | None:
    none_terms = _dedupe_search_tokens(
        [
            token
            for term in query.none_terms
            for group in _structured_search_token_groups(term)
            for token in group
        ]
    )
    if not none_terms:
        return None
    return " OR ".join(_sqlite_fts_quote(term) for term in none_terms)


def _sqlite_knowledge_none_filter_sql(
    none_fts_query: str | None,
    *,
    fts_table: str = _KNOWLEDGE_FTS_TABLE,
    temporary_fts: bool = False,
) -> tuple[str, list[object]]:
    if none_fts_query is None:
        return "", []
    from_table = f"temp.{fts_table}" if temporary_fts else fts_table
    return (
        f"""
        AND e.id NOT IN (
            SELECT DISTINCT {fts_table}.entry_id
            FROM {from_table}
            JOIN cayu_knowledge_current_entries AS negative_entry
              ON negative_entry.id = {fts_table}.entry_id
             AND negative_entry.revision = {fts_table}.entry_revision
            WHERE {fts_table} MATCH ?
        )
        """,
        [none_fts_query],
    )


def _sqlite_list_facet_sql(
    group_by: KnowledgeListGroup,
    where_sql: str,
    params: list[object],
    *,
    limit: int,
) -> tuple[str, list[object]]:
    limited_params = [*params, limit]
    if group_by is KnowledgeListGroup.KIND:
        return (
            f"""
            SELECT NULL AS key, e.kind AS value, COUNT(*) AS count
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            GROUP BY e.kind
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.NAMESPACE:
        return (
            f"""
            SELECT NULL AS key, e.namespace AS value, COUNT(*) AS count
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            GROUP BY e.namespace
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.LABEL:
        return (
            f"""
            SELECT label.key AS key, label.value AS value, COUNT(DISTINCT e.id) AS count
            FROM cayu_knowledge_current_entries AS e
            JOIN cayu_knowledge_labels AS label
              ON label.entry_id = e.id AND label.entry_revision = e.revision
            WHERE 1 = 1
            {where_sql}
            GROUP BY label.key, label.value
            ORDER BY count DESC, key ASC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.ASPECT:
        return (
            f"""
            SELECT NULL AS key, aspect.aspect AS value, COUNT(DISTINCT e.id) AS count
            FROM cayu_knowledge_current_entries AS e
            JOIN cayu_knowledge_aspects AS aspect
              ON aspect.entry_id = e.id AND aspect.entry_revision = e.revision
            WHERE 1 = 1
            {where_sql}
            GROUP BY aspect.aspect
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.IMPACT_TARGET:
        return (
            f"""
            SELECT NULL AS key, target.impact_target AS value, COUNT(DISTINCT e.id) AS count
            FROM cayu_knowledge_current_entries AS e
            JOIN cayu_knowledge_impact_targets AS target
              ON target.entry_id = e.id AND target.entry_revision = e.revision
            WHERE 1 = 1
            {where_sql}
            GROUP BY target.impact_target
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.VISIBILITY:
        return (
            f"""
            SELECT NULL AS key, e.visibility AS value, COUNT(*) AS count
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            GROUP BY e.visibility
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    return (
        f"""
        SELECT NULL AS key, e.source_type AS value, COUNT(*) AS count
        FROM cayu_knowledge_current_entries AS e
        WHERE e.source_type IS NOT NULL
        {where_sql}
        GROUP BY e.source_type
        ORDER BY count DESC, value ASC
        LIMIT ?
        """,
        limited_params,
    )


def _structured_search_token_groups(value: str) -> list[list[str]]:
    tokens = _tokenize_search_text(value)
    if not tokens:
        raise ValueError("Structured knowledge search terms must contain at least one token.")
    return [_search_token_variants(token) for token in tokens]


def _sqlite_fts_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _dedupe_search_tokens(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_search_token_groups(groups: list[list[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            result.append(group)
            seen.add(key)
    return result


def _entry_row_values(entry: KnowledgeEntry) -> tuple[object, ...]:
    return (
        entry.id,
        entry.revision,
        entry.text,
        entry.kind,
        str(entry.visibility),
        str(entry.status),
        str(entry.created_by_type),
        entry.created_by,
        sqlite_support.format_datetime(entry.created_at),
        sqlite_support.format_datetime(entry.updated_at),
        entry.source_type,
        entry.source_uri,
        entry.source_id,
        entry.source_hash,
        entry.importance,
        entry.importance_source,
        entry.confidence,
        sqlite_support.format_optional_datetime(entry.last_used_at),
        sqlite_support.format_optional_datetime(entry.expires_at),
        entry.title,
        sqlite_support.json_dumps(entry.metadata),
        knowledge_entry_payload_bytes(entry),
    )


def _entry_from_row(
    row: sqlite3.Row,
    *,
    labels: dict[str, str],
    aspects: list[str],
    impact_targets: list[str],
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=row["id"],
        revision=row["revision"],
        text=row["text"],
        namespace=row["namespace"],
        labels=labels,
        kind=row["kind"],
        visibility=KnowledgeVisibility(row["visibility"]),
        status=KnowledgeStatus(row["status"]),
        created_by_type=KnowledgeActorType(row["created_by_type"]),
        created_by=row["created_by"],
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        source_type=row["source_type"],
        source_uri=row["source_uri"],
        source_id=row["source_id"],
        source_hash=row["source_hash"],
        importance=row["importance"],
        importance_source=row["importance_source"],
        confidence=row["confidence"],
        last_used_at=sqlite_support.parse_optional_datetime(row["last_used_at"]),
        expires_at=sqlite_support.parse_optional_datetime(row["expires_at"]),
        title=row["title"],
        aspects=aspects,
        impact_targets=impact_targets,
        metadata=json.loads(row["metadata_json"]),
    )


def _chunk_row_values(chunk: KnowledgeChunk) -> tuple[object, ...]:
    return (
        chunk.id,
        chunk.entry_id,
        chunk.entry_revision,
        chunk.chunk_index,
        chunk.text,
        chunk.content_hash,
        chunk.source_uri,
        sqlite_support.json_dumps(chunk.metadata),
    )


def _chunk_from_row(row: sqlite3.Row) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=row["id"],
        entry_id=row["entry_id"],
        entry_revision=row["entry_revision"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        content_hash=row["content_hash"],
        source_uri=row["source_uri"],
        metadata=json.loads(row["metadata_json"]),
    )


def _index_readiness_from_row(row: sqlite3.Row) -> KnowledgeIndexReadiness:
    identity = KnowledgeEmbeddingIdentity(
        entry_id=str(row["entry_id"]),
        entry_revision=int(row["entry_revision"]),
        chunk_id=None if row["chunk_id"] is None else str(row["chunk_id"]),
        projection_type=str(row["projection_type"]),
        projection_content_hash=str(row["projection_content_hash"]),
        embedding_model=str(row["embedding_model"]),
        dimensions=int(row["dimensions"]),
        preprocessing_version=str(row["preprocessing_version"]),
        generator=str(row["generator"]),
        generator_version=str(row["generator_version"]),
        index_representation_version=str(row["index_representation_version"]),
    )
    if _knowledge_embedding_identity_sha256(identity) != str(row["identity_sha256"]):
        raise RuntimeError("SQLite knowledge index readiness identity is inconsistent.")
    return KnowledgeIndexReadiness(
        sequence=int(row["sequence"]),
        identity=identity,
        state=KnowledgeIndexState(str(row["state"])),
        attempt_id=str(row["attempt_id"]),
        failure_code=(None if row["failure_code"] is None else str(row["failure_code"])),
        operation_id=str(row["operation_id"]),
        published_at=sqlite_support.parse_datetime(str(row["published_at"])),
    )


def _evidence_row_values(evidence: KnowledgeEvidence) -> tuple[object, ...]:
    return (
        evidence.id,
        evidence.entry_id,
        evidence.entry_revision,
        evidence.chunk_id,
        evidence.role.value,
        evidence.source_type,
        evidence.source_id,
        evidence.source_uri,
        evidence.source_revision,
        evidence.source_hash,
        sqlite_support.json_dumps(evidence.locator),
        evidence.disposition.value,
        sqlite_support.format_datetime(evidence.created_at),
        sqlite_support.json_dumps(evidence.metadata),
    )


def _evidence_from_row(row: sqlite3.Row) -> KnowledgeEvidence:
    return KnowledgeEvidence(
        id=row["id"],
        entry_id=row["entry_id"],
        entry_revision=row["entry_revision"],
        chunk_id=row["chunk_id"],
        role=KnowledgeEvidenceRole(row["role"]),
        source_type=row["source_type"],
        source_id=row["source_id"],
        source_uri=row["source_uri"],
        source_revision=row["source_revision"],
        source_hash=row["source_hash"],
        locator=json.loads(row["locator_json"]),
        disposition=KnowledgeEvidenceDisposition(row["disposition"]),
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _relation_semantic_row_values(relation: KnowledgeRelation) -> tuple[object, ...]:
    return (
        relation.kind.value,
        relation.subject.entry_id,
        relation.subject.revision,
        relation.object.entry_id,
        relation.object.revision,
    )


def _relation_row_values(relation: KnowledgeRelation) -> tuple[object, ...]:
    return (
        relation.id,
        relation.subject.entry_id,
        relation.subject.revision,
        relation.object.entry_id,
        relation.object.revision,
        relation.kind.value,
        relation.created_by_type.value,
        relation.created_by,
        relation.policy_id,
        sqlite_support.format_datetime(relation.created_at),
        sqlite_support.json_dumps(relation.metadata),
    )


def _relation_from_row(row: sqlite3.Row) -> KnowledgeRelation:
    return KnowledgeRelation(
        id=row["id"],
        subject=KnowledgeRevisionRef(
            entry_id=row["subject_entry_id"],
            revision=row["subject_revision"],
        ),
        object=KnowledgeRevisionRef(
            entry_id=row["object_entry_id"],
            revision=row["object_revision"],
        ),
        kind=KnowledgeRelationKind(row["kind"]),
        created_by_type=KnowledgeActorType(row["created_by_type"]),
        created_by=row["created_by"],
        policy_id=row["policy_id"],
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _change_from_row(row: sqlite3.Row) -> KnowledgeChange:
    return KnowledgeChange(
        id=row["id"],
        sequence=row["sequence"],
        kind=KnowledgeChangeKind(row["kind"]),
        entry_id=row["entry_id"],
        entry_revision=row["entry_revision"],
        committed_at=sqlite_support.parse_datetime(row["committed_at"]),
        operation_id=row["operation_id"],
        relation_id=row["relation_id"],
    )


def _change_consumer_from_row(row: sqlite3.Row) -> KnowledgeChangeConsumerState:
    return KnowledgeChangeConsumerState(
        consumer_id=row["consumer_id"],
        access_scope_sha256=row["access_scope_sha256"],
        cursor_sequence=row["cursor_sequence"],
        pending_change_sequence=row["pending_change_sequence"],
        pending_claim_id=row["pending_claim_id"],
        pending_worker_id=row["pending_worker_id"],
        pending_attempt=row["pending_attempt"],
        claimed_at=sqlite_support.parse_optional_datetime(row["claimed_at"]),
        lease_expires_at=sqlite_support.parse_optional_datetime(row["lease_expires_at"]),
        last_acknowledged_claim_id=row["last_acknowledged_claim_id"],
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
    )


def _sqlite_relation_query_filter_sql(
    query: KnowledgeRelationQuery | KnowledgeLineageQuery,
) -> tuple[str, list[object]]:
    entry_id = query.reference.entry_id
    revision = query.reference.revision
    either = (
        "((relation.subject_entry_id = ? AND relation.subject_revision = ?) "
        "OR (relation.object_entry_id = ? AND relation.object_revision = ?))"
    )
    clauses: list[str]
    params: list[object]
    if query.direction is KnowledgeRelationDirection.BOTH:
        clauses = [either]
        params = [entry_id, revision, entry_id, revision]
    elif query.direction is KnowledgeRelationDirection.OUTGOING:
        clauses = [
            "((relation.kind = 'contradicts' AND "
            f"{either}) OR (relation.kind <> 'contradicts' "
            "AND relation.subject_entry_id = ? AND relation.subject_revision = ?))"
        ]
        params = [entry_id, revision, entry_id, revision, entry_id, revision]
    else:
        clauses = [
            "((relation.kind = 'contradicts' AND "
            f"{either}) OR (relation.kind <> 'contradicts' "
            "AND relation.object_entry_id = ? AND relation.object_revision = ?))"
        ]
        params = [entry_id, revision, entry_id, revision, entry_id, revision]
    if query.kinds:
        placeholders = ", ".join("?" for _ in query.kinds)
        clauses.append(f"relation.kind IN ({placeholders})")
        params.extend(kind.value for kind in query.kinds)
    return " AND " + " AND ".join(clauses), params


def _sqlite_relation_access_scope_filter_sql(
    scope: KnowledgeAccessScope,
    *,
    allow_archived_current: bool = False,
    now: datetime | None = None,
    through_change_sequence: int | None = None,
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    access_now = datetime.now(UTC) if now is None else now
    for entry_column, revision_column in (
        ("subject_entry_id", "subject_revision"),
        ("object_entry_id", "object_revision"),
    ):
        exact_access_sql, exact_access_params = _knowledge_access_scope_filter_sql(
            scope,
            now=access_now,
        )
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM (
                    SELECT
                        logical.id AS id,
                        stored.revision AS revision,
                        logical.namespace AS namespace,
                        stored.visibility AS visibility,
                        stored.status AS status,
                        stored.source_type AS source_type,
                        stored.source_id AS source_id,
                        stored.expires_at AS expires_at
                    FROM cayu_knowledge_entries AS logical
                    JOIN cayu_knowledge_revisions AS stored
                      ON stored.entry_id = logical.id
                ) AS e
                WHERE e.id = relation.{entry_column}
                  AND e.revision = relation.{revision_column}
                {exact_access_sql}
            )
            """
        )
        params.extend(exact_access_params)
        current_scope = (
            scope.model_copy(
                update={
                    "allowed_statuses": sorted(
                        {*scope.allowed_statuses, KnowledgeStatus.ARCHIVED},
                        key=str,
                    )
                }
            )
            if allow_archived_current
            else scope
        )
        current_access_sql, current_access_params = _knowledge_access_scope_filter_sql(
            current_scope,
            now=access_now,
        )
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_current_entries AS e
                WHERE e.id = relation.{entry_column}
                {current_access_sql}
            )
            """
        )
        params.extend(current_access_params)
        if through_change_sequence is not None:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM (
                        SELECT
                            logical.id AS id,
                            stored.revision AS revision,
                            logical.namespace AS namespace,
                            stored.visibility AS visibility,
                            stored.status AS status,
                            stored.source_type AS source_type,
                            stored.source_id AS source_id,
                            stored.expires_at AS expires_at
                        FROM cayu_knowledge_entries AS logical
                        JOIN cayu_knowledge_changes AS current_change
                          ON current_change.entry_id = logical.id
                         AND current_change.kind <> 'relation_published'
                         AND current_change.sequence = (
                             SELECT MAX(boundary_change.sequence)
                             FROM cayu_knowledge_changes AS boundary_change
                             WHERE boundary_change.entry_id = logical.id
                               AND boundary_change.kind <> 'relation_published'
                               AND boundary_change.sequence <= ?
                         )
                         AND current_change.sequence = (
                             SELECT MAX(materialization.sequence)
                             FROM cayu_knowledge_changes AS materialization
                             WHERE materialization.entry_id = current_change.entry_id
                               AND materialization.entry_revision =
                                       current_change.entry_revision
                               AND materialization.kind <> 'relation_published'
                         )
                        JOIN cayu_knowledge_revisions AS stored
                          ON stored.entry_id = current_change.entry_id
                         AND stored.revision = current_change.entry_revision
                    ) AS e
                    WHERE e.id = relation.{entry_column}
                    {current_access_sql}
                )
                """
            )
            params.extend((through_change_sequence, *current_access_params))
    return " AND " + " AND ".join(clauses), params


def _sqlite_lineage_filter_sql(query: KnowledgeLineageQuery) -> tuple[str, list[object]]:
    subject_is_anchor = "(relation.subject_entry_id = ? AND relation.subject_revision = ?)"
    counterpart_status = (
        f"CASE WHEN {subject_is_anchor} THEN object_current.status ELSE subject_current.status END"
    )
    current_relation = (
        "(relation.subject_revision = subject_current.revision "
        "AND relation.object_revision = object_current.revision)"
    )
    clauses: list[str] = []
    params: list[object] = []
    placeholders = ", ".join("?" for _ in query.counterpart_statuses)
    clauses.append(f"{counterpart_status} IN ({placeholders})")
    params.extend(
        [
            query.reference.entry_id,
            query.reference.revision,
            *(status.value for status in query.counterpart_statuses),
        ]
    )
    current_values = set(query.currentnesses)
    if len(current_values) == 1:
        clauses.append(
            current_relation
            if KnowledgeLineageCurrentness.CURRENT in current_values
            else f"NOT {current_relation}"
        )
    if query.unresolved_only:
        clauses.extend(
            [
                "relation.kind = 'contradicts'",
                current_relation,
                "subject_current.status = 'active'",
                "object_current.status = 'active'",
            ]
        )
    return " AND " + " AND ".join(clauses), params


def _sqlite_change_access_scope_filter_sql(
    scope: KnowledgeAccessScope,
    *,
    alias: str,
) -> tuple[str, list[object]]:
    if alias != "change_record":
        raise ValueError("Unsupported knowledge change access-filter alias.")
    audience_alias = "access_audience"
    clauses: list[str] = []
    params: list[object] = []
    if not scope.allow_all_namespaces:
        placeholders = ", ".join("?" for _ in scope.allowed_namespaces)
        clauses.append(f"{audience_alias}.namespace IN ({placeholders})")
        params.extend(scope.allowed_namespaces)
    for key, value in scope.required_labels.items():
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_change_labels AS access_label
                WHERE access_label.change_sequence = {alias}.sequence
                  AND access_label.audience_kind = {audience_alias}.audience_kind
                  AND access_label.key = ?
                  AND access_label.value = ?
            )
            """
        )
        params.extend([key, value])
    placeholders = ", ".join("?" for _ in scope.allowed_visibilities)
    clauses.append(f"{audience_alias}.visibility IN ({placeholders})")
    params.extend(visibility.value for visibility in scope.allowed_visibilities)
    placeholders = ", ".join("?" for _ in scope.allowed_statuses)
    clauses.append(f"{audience_alias}.status IN ({placeholders})")
    params.extend(status.value for status in scope.allowed_statuses)
    if scope.allowed_source_types is not None:
        if scope.allowed_source_types:
            placeholders = ", ".join("?" for _ in scope.allowed_source_types)
            clauses.append(f"{audience_alias}.source_type IN ({placeholders})")
            params.extend(scope.allowed_source_types)
        else:
            clauses.append("0")
    if scope.allowed_source_ids is not None:
        if scope.allowed_source_ids:
            placeholders = ", ".join("?" for _ in scope.allowed_source_ids)
            clauses.append(f"{audience_alias}.source_id IN ({placeholders})")
            params.extend(scope.allowed_source_ids)
        else:
            clauses.append("0")
    if not scope.include_expired:
        clauses.append(f"{audience_alias}.requires_include_expired = 0")
    audience_filter = " AND ".join(clauses)
    return (
        " AND (SELECT CASE "
        f"WHEN {alias}.kind = 'relation_published' THEN CASE "
        "WHEN COUNT(*) = 4 "
        "AND SUM(CASE WHEN candidate.audience_kind IN "
        "('subject_exact', 'subject_current', 'object_exact', 'object_current') "
        "THEN 1 ELSE 0 END) = 4 "
        "AND MIN(candidate.allowed) = 1 THEN 1 ELSE 0 END "
        "ELSE COALESCE(MAX(candidate.allowed), 0) END FROM ("
        f"SELECT {audience_alias}.audience_kind, CASE WHEN "
        f"{audience_filter} THEN 1 ELSE 0 END AS allowed "
        "FROM cayu_knowledge_change_audiences AS access_audience "
        f"WHERE {audience_alias}.change_sequence = {alias}.sequence"
        ") AS candidate) = 1",
        params,
    )


def _copy_entry_chunks(
    entry_id: str,
    entry_revision: int,
    chunks: list[KnowledgeChunk],
) -> list[KnowledgeChunk]:
    if type(chunks) is not list:
        raise ValueError("`chunks` must be a list.")
    if not chunks:
        raise ValueError("`chunks` cannot be empty.")
    copied_chunks = [copy_knowledge_chunk(chunk) for chunk in chunks]
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for chunk in copied_chunks:
        if chunk.entry_id != entry_id:
            raise ValueError("Knowledge chunks must belong to the entry.")
        if chunk.entry_revision != entry_revision:
            raise ValueError("Knowledge chunks must belong to the exact entry revision.")
        if chunk.id in seen_ids:
            raise ValueError("Knowledge chunk ids must be unique within an entry.")
        if chunk.chunk_index in seen_indexes:
            raise ValueError("Knowledge chunk indexes must be unique within an entry.")
        seen_ids.add(chunk.id)
        seen_indexes.add(chunk.chunk_index)
    return sorted(copied_chunks, key=lambda chunk: chunk.chunk_index)


def _center_chunk_window(
    chunks: list[KnowledgeChunk],
    *,
    chunk_index: int,
    max_chunks: int,
) -> list[KnowledgeChunk]:
    if len(chunks) <= max_chunks:
        return chunks
    closest = sorted(
        chunks, key=lambda chunk: (abs(chunk.chunk_index - chunk_index), chunk.chunk_index)
    )
    return sorted(closest[:max_chunks], key=lambda chunk: chunk.chunk_index)


def _bounded_chunks(
    chunks: list[KnowledgeChunk],
    *,
    start_index: int,
    end_index: int | None,
    max_chunks: int,
    max_bytes: int,
) -> list[KnowledgeChunk]:
    selected: list[KnowledgeChunk] = []
    remaining = max_bytes
    for chunk in chunks:
        if chunk.chunk_index < start_index:
            continue
        if end_index is not None and chunk.chunk_index > end_index:
            continue
        if len(selected) >= max_chunks or remaining <= 0:
            break
        copied = copy_knowledge_chunk(chunk)
        chunk_bytes = len(copied.text.encode("utf-8"))
        if chunk_bytes > remaining:
            truncated_text = _truncate_text_to_bytes(copied.text, remaining)
            if not truncated_text:
                break
            selected.append(
                KnowledgeChunk(
                    id=copied.id,
                    entry_id=copied.entry_id,
                    entry_revision=copied.entry_revision,
                    text=truncated_text,
                    chunk_index=copied.chunk_index,
                    content_hash=None,
                    source_uri=copied.source_uri,
                    metadata=copied.metadata,
                )
            )
            break
        selected.append(copied)
        remaining -= chunk_bytes
    return selected


def _preview_for_match(
    entry: KnowledgeEntry,
    chunk: KnowledgeChunk,
    terms: list[str],
) -> tuple[str, str]:
    if entry.title is not None:
        title_terms = set(_tokenize_search_text(entry.title))
        if any(term in title_terms for term in terms):
            return "title match", entry.title
    entry_terms = set(_tokenize_search_text(entry.text))
    if any(term in entry_terms for term in terms):
        return "entry text match", entry.text
    return "chunk text match", chunk.text


def _fts_text_for_entry_chunk(entry: KnowledgeEntry, chunk: KnowledgeChunk) -> str:
    if chunk.text == entry.text:
        return chunk.text
    return f"{entry.text}\n{chunk.text}"


def _default_chunk_for_entry(entry: KnowledgeEntry) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=f"{entry.id}:r{entry.revision}:0",
        entry_id=entry.id,
        entry_revision=entry.revision,
        text=entry.text,
        chunk_index=0,
        content_hash=sha256(entry.text.encode("utf-8")).hexdigest(),
        source_uri=entry.source_uri,
    )


def _has_only_default_chunk(entry: KnowledgeEntry, chunks: list[KnowledgeChunk]) -> bool:
    if len(chunks) != 1:
        return False
    default_chunk = _default_chunk_for_entry(entry)
    chunk = chunks[0]
    return (
        chunk.id == default_chunk.id
        and chunk.entry_id == default_chunk.entry_id
        and chunk.entry_revision == default_chunk.entry_revision
        and chunk.text == default_chunk.text
        and chunk.chunk_index == default_chunk.chunk_index
        and chunk.content_hash == default_chunk.content_hash
        and chunk.source_uri == default_chunk.source_uri
        and chunk.metadata == default_chunk.metadata
    )


def _tokenize_search_text(text: str) -> list[str]:
    return _SEARCH_TOKEN_RE.findall(text.casefold())


def _expand_search_tokens(tokens: list[str]) -> list[str]:
    return [variant for token in tokens for variant in _search_token_variants(token)]


def _search_token_variants(token: str) -> list[str]:
    variants = [token]
    if len(token) < 3 or not token.isalpha():
        return variants
    if token.endswith("ies") and len(token) > 4:
        variants.append(token[:-3] + "y")
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        variants.append(token[:-1])
    else:
        variants.append(_plural_search_token(token))
    return _dedupe_search_tokens(variants)


def _plural_search_token(token: str) -> str:
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    return token + "s"


def _truncate_text_to_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{field_name}` must be greater than 0.")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{field_name}` must be greater than or equal to 0.")
