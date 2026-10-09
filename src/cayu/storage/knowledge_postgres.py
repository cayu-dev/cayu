"""PostgreSQL knowledge persistence, publication, revisions and bounded queries."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any, LiteralString, cast
from uuid import uuid4

from cayu.knowledge._activation_rules import (
    _activation_receipt_matches,
    _prepare_review_approval_receipts,
    _replay_review_approval_from_receipts,
    _validate_review_approval_authority,
    _validate_review_approval_scope,
)
from cayu.knowledge.access import runtime_knowledge_operation
from cayu.storage import _postgres_base as postgres_base
from cayu.storage import _postgres_support as pg_support
from cayu.storage._phase_timing import PostgresTimingScope

if TYPE_CHECKING:
    from cayu.knowledge.maintenance_governance import (
        KnowledgeMaintenanceGovernanceAuthority,
        KnowledgeMaintenanceGovernanceReceipt,
    )
    from cayu.knowledge.maintenance_persistence import (
        KnowledgeMaintenanceAcceptedPlan,
        KnowledgeMaintenanceProposalPublication,
        KnowledgeMaintenanceProposalPublicationReceipt,
    )
    from cayu.knowledge.semantic_watch import (
        KnowledgeSemanticWatchAuthority,
        KnowledgeSemanticWatchReceipt,
    )
try:
    from psycopg.errors import ForeignKeyViolation, UniqueViolation
    from psycopg_pool import AsyncConnectionPool  # noqa: TC002 — keep runtime annotation resolution
except ModuleNotFoundError as exc:
    raise RuntimeError(
        'Cayu\'s Postgres stores require the optional psycopg packages. Install them with `pip install "cayu[postgres]"`.'
    ) from exc
from cayu._clock import utc_clock
from cayu._validation import copy_label_map
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge._access_rules import (
    _knowledge_change_audiences,
    _knowledge_maintenance_access_snapshot,
    _knowledge_maintenance_access_snapshot_json,
    _knowledge_relation_access_snapshot,
    _knowledge_relation_access_snapshot_json,
    _knowledge_relation_change_audiences,
    _knowledge_scope_allows_activation_receipt,
    _knowledge_scope_allows_entry,
    _knowledge_scope_allows_lineage_endpoint,
    _knowledge_scope_allows_maintenance_access_snapshot,
    _knowledge_scope_allows_relation_access_snapshot,
    _knowledge_scope_allows_snapshot,
    _KnowledgeMaintenanceAccessSnapshot,
    _KnowledgeRelationAccessSnapshot,
    _parse_knowledge_maintenance_access_snapshot_json,
    _parse_knowledge_relation_access_snapshot_json,
    _require_knowledge_activation_retirement_access,
    _require_knowledge_entry_access,
    _require_knowledge_successor_access,
)
from cayu.knowledge._maintenance_rules import (
    _knowledge_maintenance_successors,
    _require_knowledge_maintenance_current_entries,
    _require_knowledge_maintenance_current_replacement,
    _require_knowledge_maintenance_publication_boundary,
    _require_knowledge_maintenance_source_evidence,
)
from cayu.knowledge._query_rules import _validate_knowledge_search_frontier
from cayu.knowledge._relation_queries import (
    _bounded_knowledge_lineage_result,
    _bounded_knowledge_relation_result,
    _decode_knowledge_lineage_cursor,
    _decode_knowledge_relation_cursor,
    _knowledge_lineage_link,
    _knowledge_lineage_query_fingerprint,
    _knowledge_relation_query_fingerprint,
)
from cayu.knowledge._retrieval_results import _bounded_knowledge_evidence
from cayu.knowledge._revision_rules import (
    _copy_chunks_for_revision,
    _copy_evidence_for_revision,
    _validate_revision_successor,
)
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationAuthority,
    KnowledgeActivationConflict,
    KnowledgeActivationReceipt,
    KnowledgeActivationSource,
    KnowledgeReviewApproval,
    _knowledge_activation_receipt_json,
    _knowledge_activation_retirement,
    _knowledge_activation_retirement_json,
    _KnowledgeActivationRetirement,
    _parse_knowledge_activation_retirement_json,
    _require_knowledge_activation_retirement_capacity,
    copy_knowledge_activation_authority,
    copy_knowledge_activation_receipt,
)
from cayu.knowledge.base import KnowledgeStore
from cayu.knowledge.changes import (
    KnowledgeChange,
    KnowledgeChangeBatch,
    KnowledgeChangeClaim,
    KnowledgeChangeConsumerConflict,
    KnowledgeChangeConsumerState,
    KnowledgeChangeKind,
    _initialize_knowledge_change_consumer_state,
    _knowledge_change_claim_sha256,
    _knowledge_change_identity,
    _knowledge_change_lease_seconds,
    _knowledge_change_now,
    _validate_knowledge_change_limit,
    _validate_knowledge_change_sequence,
    copy_knowledge_change_claim,
    copy_knowledge_change_consumer_state,
)
from cayu.knowledge.indexing import (
    KNOWLEDGE_CHUNK_TEXT_PROJECTION,
    KnowledgeEmbeddingIdentity,
    KnowledgeIndexReadiness,
    KnowledgeIndexReadinessBatch,
    KnowledgeIndexReadinessConflict,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    _bounded_knowledge_index_identity,
    _knowledge_chunk_content_hash,
    _knowledge_embedding_identity_sha256,
    _knowledge_index_readiness_update_sha256,
    _validate_knowledge_index_readiness_limit,
    _validate_knowledge_index_readiness_transition,
    _validate_knowledge_index_sequence,
    copy_knowledge_embedding_identity,
    copy_knowledge_index_readiness_update,
)
from cayu.knowledge.maintenance_contracts import (
    KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY,
    KnowledgeMaintenanceConflict,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceOutcome,
    KnowledgeMaintenanceProposal,
    _knowledge_maintenance_identity,
    _validate_knowledge_maintenance_record,
    _validate_knowledge_maintenance_replay,
    copy_knowledge_maintenance_decision,
    copy_knowledge_maintenance_decision_receipt,
    copy_knowledge_maintenance_proposal,
    prepare_knowledge_maintenance_decision,
)
from cayu.knowledge.publication_contracts import (
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    _validate_activation_publication_material,
    _validate_knowledge_publication_replay,
    _validate_revision_append,
    copy_knowledge_publication_receipt,
    prepare_knowledge_publication,
)
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEntry,
    KnowledgeEntryReadLimitExceeded,
    KnowledgeEvidence,
    KnowledgeEvidenceConflict,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceResult,
    KnowledgeEvidenceRole,
    KnowledgeRevisionConflict,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeVisibility,
    _copy_entry_evidence,
    _knowledge_entry_id,
    _knowledge_publication_operation_id,
    _knowledge_semantic_watch_identity,
    _next_knowledge_revision,
    _validate_knowledge_revision,
    _validate_positive_int,
    copy_knowledge_chunk,
    copy_knowledge_entry,
    copy_knowledge_revision_refs,
    knowledge_entry_payload_bytes,
)
from cayu.knowledge.relations import (
    KnowledgeLineageCurrentness,
    KnowledgeLineageQuery,
    KnowledgeLineageResult,
    KnowledgeRelation,
    KnowledgeRelationConflict,
    KnowledgeRelationDirection,
    KnowledgeRelationKind,
    KnowledgeRelationPublicationReceipt,
    KnowledgeRelationQuery,
    KnowledgeRelationResult,
    _knowledge_relation_identity,
    _knowledge_relation_semantic_key,
    _validate_knowledge_relation_publication_replay,
    copy_knowledge_lineage_query,
    copy_knowledge_relation_publication_receipt,
    copy_knowledge_relation_query,
    prepare_knowledge_relations,
)
from cayu.knowledge.scopes import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    _knowledge_access_scope_sha256,
    _knowledge_access_snapshot,
    _knowledge_access_snapshot_json,
    _parse_knowledge_access_snapshot_json,
    copy_knowledge_access_scope,
)
from cayu.knowledge.search import (
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    copy_knowledge_list_query,
    copy_knowledge_query,
)
from cayu.storage import migrations as schema
from cayu.storage._knowledge_closure import (
    KnowledgeClosureInventory,
    KnowledgeClosureQuery,
    copy_knowledge_closure_query,
)

_MAINTENANCE_REJECTED_REPLACEMENT_RETIREMENT_TRANSITIONS = frozenset(
    {
        (KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED),
        (KnowledgeStatus.PENDING, KnowledgeStatus.DELETED),
        (KnowledgeStatus.ARCHIVED, KnowledgeStatus.DELETED),
    }
)


_KNOWLEDGE_SEARCH_PAGE_SIZE = 500
_KNOWLEDGE_SEARCH_TOKEN_RE = re.compile(r"\w+")


async def _lock_knowledge_entry(cur: Any, entry_id: str) -> None:
    """Serialize every Cayu mutation of one knowledge identity."""

    await _lock_knowledge_write_identities(cur, entry_ids=(entry_id,))


async def _lock_knowledge_write_identities(
    cur: Any,
    *,
    entry_ids: tuple[str, ...] = (),
    chunk_ids: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    operation_ids: tuple[str, ...] = (),
    relation_ids: tuple[str, ...] = (),
    relation_semantics: tuple[str, ...] = (),
    maintenance_proposal_ids: tuple[str, ...] = (),
) -> None:
    """Serialize overlapping knowledge writes in one global lock order."""

    identities = {
        *(f"knowledge-entry:{entry_id}" for entry_id in entry_ids),
        *(f"knowledge-chunk:{chunk_id}" for chunk_id in chunk_ids),
        *(f"knowledge-evidence:{evidence_id}" for evidence_id in evidence_ids),
        *(f"knowledge-operation:{operation_id}" for operation_id in operation_ids),
        *(f"knowledge-relation:{relation_id}" for relation_id in relation_ids),
        *(f"knowledge-relation-semantic:{semantic}" for semantic in relation_semantics),
        *(
            f"knowledge-maintenance-proposal:{proposal_id}"
            for proposal_id in maintenance_proposal_ids
        ),
    }
    if not identities:
        return
    await cur.execute(
        """
        WITH lock_keys AS MATERIALIZED (
            SELECT DISTINCT hashtextextended(lock_identity, 0) AS lock_key
            FROM unnest(%s::text[]) AS requested(lock_identity)
        )
        SELECT pg_advisory_xact_lock(lock_key)
        FROM lock_keys
        ORDER BY lock_key
        """,
        (sorted(identities),),
    )


async def _lock_knowledge_relation_write_identities(
    cur: Any,
    *,
    operation_id: str,
    relations: list[KnowledgeRelation],
) -> None:
    """Lock relation publication identities in the canonical write-category order."""

    await _lock_knowledge_write_identities(cur, operation_ids=(operation_id,))
    await _lock_knowledge_write_identities(
        cur,
        entry_ids=tuple(
            reference.entry_id
            for relation in relations
            for reference in (relation.subject, relation.object)
        ),
    )
    await _lock_knowledge_write_identities(
        cur,
        relation_ids=tuple(relation.id for relation in relations),
        relation_semantics=tuple(
            sha256(
                pg_support._dumps(list(_knowledge_relation_semantic_key(relation))).encode("utf-8")
            ).hexdigest()
            for relation in relations
        ),
    )


async def _lock_knowledge_semantic_watch_write_identities(
    cur: Any,
    *,
    operation_id: str,
    entry_ids: tuple[str, ...],
) -> None:
    """Serialize a watch outcome in the canonical write-category order."""

    await _lock_knowledge_write_identities(cur, operation_ids=(operation_id,))
    await _lock_knowledge_write_identities(cur, entry_ids=entry_ids)


async def _lock_knowledge_maintenance_write_identities(
    cur: Any,
    *,
    proposal: KnowledgeMaintenanceProposal,
    decision: KnowledgeMaintenanceDecision,
) -> None:
    """Serialize every identity that one reviewed decision may mutate."""

    await _lock_knowledge_write_identities(
        cur,
        operation_ids=(decision.operation_id,),
        maintenance_proposal_ids=(proposal.id,),
    )
    await _lock_knowledge_write_identities(
        cur,
        entry_ids=(
            proposal.replacement.entry_id,
            *(source.entry_id for source in proposal.sources),
        ),
    )
    await _lock_knowledge_write_identities(
        cur,
        relation_ids=tuple(relation.id for relation in proposal.relations),
        relation_semantics=tuple(
            sha256(
                pg_support._dumps(list(_knowledge_relation_semantic_key(relation))).encode("utf-8")
            ).hexdigest()
            for relation in proposal.relations
        ),
    )


async def _begin_knowledge_read_snapshot(cur: Any) -> None:
    """Keep access filtering and result hydration on one database snapshot."""

    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")


async def _lock_knowledge_change_sequence(cur: Any) -> None:
    """Serialize sequence allocation so visible changes cannot later develop gaps."""

    await cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended('cayu-knowledge-change-sequence', 0))"
    )


class PostgresKnowledgeStore(postgres_base._PostgresStoreBase, KnowledgeStore):
    """Postgres-backed durable knowledge store with full-text search."""

    resource_knowledge_access_version = 1

    _min_required_revision = 78

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        read_only: bool = False,
        access_scope: KnowledgeAccessScope | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._default_access_scope = (
            None if access_scope is None else copy_knowledge_access_scope(access_scope)
        )
        self._clock = utc_clock(clock)
        self._clock_is_injected = clock is not None
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
            read_only=read_only,
        )

    @runtime_knowledge_operation("create")
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
            else _copy_knowledge_entry_chunks(entry.id, entry.revision, chunks)
        )
        copied_evidence = _copy_entry_evidence(
            entry.id,
            entry.revision,
            evidence or [],
            chunks=copied_chunks,
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    # Every mutation acquires identity categories in the same
                    # order: operation (when present), entry, chunks, evidence, then change.
                    # Appends sometimes discover inherited chunk ids only after
                    # locking/loading the current entry, so combining categories
                    # here would permit an entry/chunk advisory-lock deadlock.
                    await _lock_knowledge_entry(cur, entry.id)
                    await _lock_knowledge_write_identities(
                        cur, chunk_ids=tuple(chunk.id for chunk in copied_chunks)
                    )
                    await _lock_knowledge_write_identities(
                        cur,
                        evidence_ids=tuple(item.id for item in copied_evidence),
                    )
                    existing_entry = await self._load_entry(cur, entry.id)
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
                    retirement = await self._load_activation_retirement(cur, entry.id)
                    if retirement is not None:
                        _require_knowledge_activation_retirement_access(
                            scope,
                            retirement,
                            operation="create_entry",
                        )
                        raise KnowledgePublicationConflict("entry_retired")
                    await self._require_chunk_ids_available(
                        cur,
                        copied_chunks,
                        access_scope=scope,
                        operation="create_entry",
                    )
                    await self._require_evidence_ids_available(
                        cur,
                        copied_evidence,
                        access_scope=scope,
                        operation="create_entry",
                    )
                    await self._insert_entry(cur, entry)
                    await self._insert_chunks(cur, entry, copied_chunks)
                    await self._insert_evidence(cur, copied_evidence)
                    await self._insert_change(
                        cur,
                        before_entry=None,
                        after_entry=entry,
                        kind=KnowledgeChangeKind.CREATED,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return copy_knowledge_entry(entry)

    @runtime_knowledge_operation("modify")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_entry(cur, entry.id)
                    await self._append_revision(
                        cur,
                        entry,
                        expected_revision=expected_revision,
                        chunks=chunks,
                        evidence=evidence,
                        access_scope=scope,
                        operation="append_entry_revision",
                        change_kind=KnowledgeChangeKind.REVISION_APPENDED,
                        inherit_evidence=False,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return copy_knowledge_entry(entry)

    @runtime_knowledge_operation("read")
    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        entry_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if max_bytes is not None:
            _validate_positive_int(max_bytes, "max_bytes")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            access_now = datetime.now(UTC)
            if max_bytes is None:
                return await self._load_entry_in_scope(
                    cur,
                    entry_id,
                    scope,
                    revision=revision,
                    access_now=access_now,
                )
            descriptor = await self._load_entry_payload_bytes_in_scope(
                cur,
                entry_id,
                scope,
                revision=revision,
                access_now=access_now,
            )
            if descriptor is None:
                return None
            selected_revision, stored_payload_bytes = descriptor
            if stored_payload_bytes > max_bytes:
                raise KnowledgeEntryReadLimitExceeded(
                    entry_id=entry_id,
                    revision=selected_revision,
                    payload_bytes=stored_payload_bytes,
                    max_bytes=max_bytes,
                )
            entry = await self._load_entry_in_scope(
                cur,
                entry_id,
                scope,
                revision=revision,
                access_now=access_now,
            )
            if entry is None:
                raise RuntimeError("Knowledge entry disappeared inside a repeatable-read snapshot.")
            actual_payload_bytes = knowledge_entry_payload_bytes(entry)
            if actual_payload_bytes != stored_payload_bytes:
                raise RuntimeError(
                    "Knowledge entry payload size metadata does not match its canonical payload."
                )
            return entry

    @runtime_knowledge_operation("modify")
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
        entry_id = _knowledge_entry_id(entry_id)
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_entry(cur, entry_id)
                    entry = await self._load_entry(cur, entry_id)
                    if entry is None:
                        raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
                    _require_knowledge_entry_access(
                        scope,
                        entry,
                        operation="transition_entry_status",
                    )
                    if entry.revision != expected_revision:
                        raise KnowledgeRevisionConflict(
                            entry_id,
                            expected_revision=expected_revision,
                            actual_revision=entry.revision,
                        )
                    if expected_namespace is not None and entry.namespace != expected_namespace:
                        raise ValueError(
                            f"Knowledge entry {entry_id!r} does not match expected namespace."
                        )
                    for key, value in expected_labels.items():
                        if entry.labels.get(key) != value:
                            raise ValueError(
                                f"Knowledge entry {entry_id!r} does not match expected labels."
                            )
                    if entry.status is not from_status:
                        raise ValueError(
                            f"Knowledge entry {entry_id!r} is {entry.status.value!r}, "
                            f"not {from_status.value!r}."
                        )
                    loaded = entry.model_copy(
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
                    await self._append_revision(
                        cur,
                        loaded,
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
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        if loaded is None:
            raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
        return loaded

    @runtime_knowledge_operation("delete")
    async def delete_entry(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        entry_id = _knowledge_entry_id(entry_id)
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if type(hard) is not bool:
            raise ValueError("`hard` must be a boolean.")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_entry(cur, entry_id)
                    entry = await self._load_entry(cur, entry_id)
                    if entry is None:
                        if not hard:
                            await conn.commit()
                            return None
                        retirement = await self._load_activation_retirement(cur, entry_id)
                        if retirement is None:
                            await conn.commit()
                            return None
                        _require_knowledge_activation_retirement_access(
                            scope,
                            retirement,
                            operation="delete_entry",
                        )
                        if retirement.entry_revision != expected_revision:
                            raise KnowledgeRevisionConflict(
                                entry_id,
                                expected_revision=expected_revision,
                                actual_revision=retirement.entry_revision,
                            )
                        await cur.execute(
                            "SELECT COUNT(*) FROM cayu_knowledge_activation_receipts "
                            "WHERE entry_id = %s",
                            (entry_id,),
                        )
                        receipt_count_row = await cur.fetchone()
                        if receipt_count_row is None or int(receipt_count_row[0]) < 1:
                            raise KnowledgeActivationConflict("malformed_retirement")
                        await cur.execute(
                            "DELETE FROM cayu_knowledge_activation_receipts WHERE entry_id = %s",
                            (entry_id,),
                        )
                        await cur.execute(
                            "DELETE FROM cayu_knowledge_activation_retirements WHERE entry_id = %s",
                            (entry_id,),
                        )
                        await conn.commit()
                        return None
                    if await self._load_activation_retirement(cur, entry_id) is not None:
                        raise KnowledgeActivationConflict("malformed_retirement")
                    _require_knowledge_entry_access(scope, entry, operation="delete_entry")
                    if entry.revision != expected_revision:
                        raise KnowledgeRevisionConflict(
                            entry_id,
                            expected_revision=expected_revision,
                            actual_revision=entry.revision,
                        )
                    if hard:
                        await self._require_maintenance_replacement_mutation_allowed(
                            cur,
                            entry_id=entry_id,
                            entry_revision=entry.revision,
                            preserve_history=True,
                        )
                        await self._insert_change(
                            cur,
                            before_entry=entry,
                            after_entry=None,
                            kind=KnowledgeChangeKind.HARD_DELETED,
                        )
                        await cur.execute(
                            "DELETE FROM cayu_knowledge_activation_receipts WHERE entry_id = %s",
                            (entry_id,),
                        )
                        await cur.execute(
                            "DELETE FROM cayu_knowledge_entries WHERE id = %s",
                            (entry_id,),
                        )
                        await conn.commit()
                        return copy_knowledge_entry(entry)
                    loaded = entry.model_copy(
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
                    await self._append_revision(
                        cur,
                        loaded,
                        expected_revision=expected_revision,
                        chunks=None,
                        evidence=None,
                        access_scope=scope,
                        operation="delete_entry",
                        change_kind=KnowledgeChangeKind.TOMBSTONED,
                        inherit_evidence=True,
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        if loaded is None:
            raise KeyError(f"Knowledge entry {entry_id!r} does not exist.")
        return loaded

    @runtime_knowledge_operation("modify")
    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        scope = self._operation_access_scope(access_scope)
        cutoff = _knowledge_change_now(now)
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(
            scope,
            now=cutoff,
        )
        maintenance_exclusion_sql = (
            "AND NOT EXISTS ("
            "SELECT 1 FROM cayu_knowledge_maintenance_proposals AS proposal "
            "WHERE proposal.replacement_entry_id = e.id"
            ") "
            if self._min_required_revision >= 67
            else ""
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        cast(
                            "LiteralString",
                            "SELECT e.id FROM cayu_knowledge_current_entries AS e "
                            "WHERE TRUE "
                            "AND e.expires_at IS NOT NULL AND e.expires_at <= %s "
                            f"{maintenance_exclusion_sql}"
                            f"{access_sql}",
                        ),
                        (cutoff, *access_params),
                    )
                    candidate_ids = sorted(str(row[0]) for row in await cur.fetchall())
                    await _lock_knowledge_write_identities(
                        cur,
                        entry_ids=tuple(candidate_ids),
                    )
                    entries = await self._load_entries(cur, candidate_ids)
                    eligible = [
                        entry
                        for entry_id in candidate_ids
                        if (entry := entries.get(entry_id)) is not None
                        and entry.expires_at is not None
                        and entry.expires_at <= cutoff
                        and _knowledge_scope_allows_snapshot(
                            scope,
                            _knowledge_access_snapshot(entry),
                            now=cutoff,
                        )
                    ]
                    governed_entry_ids: set[str] = set()
                    if eligible and self._min_required_revision >= 75:
                        await cur.execute(
                            "SELECT DISTINCT entry_id "
                            "FROM cayu_knowledge_activation_receipts "
                            "WHERE entry_id = ANY(%s)",
                            ([entry.id for entry in eligible],),
                        )
                        governed_entry_ids = {str(row[0]) for row in await cur.fetchall()}
                    retired_at = datetime.now(UTC)
                    for entry in eligible:
                        await self._insert_change(
                            cur,
                            before_entry=entry,
                            after_entry=None,
                            kind=KnowledgeChangeKind.EXPIRED,
                        )
                        if entry.id in governed_entry_ids:
                            if await self._load_activation_retirement(cur, entry.id) is not None:
                                raise KnowledgeActivationConflict("malformed_retirement")
                            await self._insert_activation_retirement(
                                cur,
                                _knowledge_activation_retirement(entry, retired_at=retired_at),
                            )
                    if eligible:
                        await cur.execute(
                            "DELETE FROM cayu_knowledge_entries WHERE id = ANY(%s)",
                            ([entry.id for entry in eligible],),
                        )
                    pruned = len(eligible)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return pruned

    @runtime_knowledge_operation("modify")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_write_identities(cur, operation_ids=(operation_id,))
                    await _lock_knowledge_entry(cur, copied_entry.id)
                    await _lock_knowledge_write_identities(
                        cur, chunk_ids=tuple(chunk.id for chunk in copied_chunks)
                    )
                    await _lock_knowledge_write_identities(
                        cur,
                        evidence_ids=tuple(item.id for item in copied_evidence),
                    )
                    existing_receipt = await self._load_publication_receipt(
                        cur,
                        operation_id,
                        access_scope=scope,
                    )
                    if existing_receipt is not None:
                        existing_activation = await self._load_activation_receipt(
                            cur,
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
                        await conn.commit()
                        return copy_knowledge_publication_receipt(
                            existing_receipt,
                            replayed=True,
                        )
                    existing_activation = await self._load_activation_receipt(
                        cur,
                        operation_id,
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if existing_activation is not None:
                        raise KnowledgePublicationConflict("operation_occupied")
                    existing_entry = await self._load_entry(cur, copied_entry.id)
                    if existing_entry is not None:
                        _require_knowledge_entry_access(
                            scope,
                            existing_entry,
                            operation="publish_entry_revision",
                        )
                    retirement = await self._load_activation_retirement(cur, copied_entry.id)
                    if retirement is not None:
                        _require_knowledge_activation_retirement_access(
                            scope,
                            retirement,
                            operation="publish_entry_revision",
                        )
                        raise KnowledgePublicationConflict("entry_retired")
                    actual_revision = None if existing_entry is None else existing_entry.revision
                    if actual_revision != expected_revision:
                        raise KnowledgeRevisionConflict(
                            copied_entry.id,
                            expected_revision=expected_revision,
                            actual_revision=actual_revision,
                        )
                    if existing_entry is not None:
                        await self._require_maintenance_replacement_mutation_allowed(
                            cur,
                            entry_id=existing_entry.id,
                            entry_revision=existing_entry.revision,
                            current_status=existing_entry.status,
                            successor_status=copied_entry.status,
                            operation="publish_entry_revision",
                        )
                        _validate_revision_successor(existing_entry, copied_entry)
                        if (
                            self._min_required_revision >= 75
                            and copied_authority is None
                            and await self._has_activation_receipts(cur, copied_entry.id)
                        ):
                            _require_knowledge_activation_retirement_capacity(copied_entry)
                    await self._require_chunk_ids_available(
                        cur,
                        copied_chunks,
                        access_scope=scope,
                        operation="publish_entry_revision",
                    )
                    await self._require_evidence_ids_available(
                        cur,
                        copied_evidence,
                        access_scope=scope,
                        operation="publish_entry_revision",
                    )
                    if existing_entry is None:
                        await self._insert_entry(cur, copied_entry)
                    else:
                        assert expected_revision is not None
                        await self._insert_revision(cur, copied_entry)
                        await self._advance_current_revision(
                            cur,
                            copied_entry,
                            expected_revision=expected_revision,
                        )
                    await self._insert_chunks(cur, copied_entry, copied_chunks)
                    await self._insert_evidence(cur, copied_evidence)
                    await _lock_knowledge_change_sequence(cur)
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
                    await self._insert_change(
                        cur,
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
                    await self._insert_publication_receipt(cur, receipt, copied_entry)
                    if activation_receipt is not None:
                        await self._insert_activation_receipt(
                            cur,
                            activation_receipt,
                            access_entry=copied_entry,
                        )
                await conn.commit()
                return copy_knowledge_publication_receipt(receipt)
            except UniqueViolation:
                await conn.rollback()
                raise KnowledgePublicationConflict("concurrent_occupancy") from None
            except Exception:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            receipt = await self._load_publication_receipt_in_scope(cur, operation_id, scope)
        return None if receipt is None else copy_knowledge_publication_receipt(receipt)

    @runtime_knowledge_operation("read")
    async def load_activation_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeActivationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            receipt = await self._load_activation_receipt(
                cur,
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if receipt is None else copy_knowledge_activation_receipt(receipt)

    @runtime_knowledge_operation("modify")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_write_identities(
                        cur,
                        operation_ids=(request.operation_id,),
                    )
                    await _lock_knowledge_entry(cur, request.candidate_entry.id)
                    existing_receipt = await self._load_activation_receipt(
                        cur,
                        request.operation_id,
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if existing_receipt is not None:
                        publication = await self._load_publication_receipt(
                            cur,
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
                        await conn.commit()
                        return approval
                    if (
                        await self._load_publication_receipt(
                            cur,
                            request.operation_id,
                            access_scope=scope,
                        )
                        is not None
                    ):
                        raise KnowledgeActivationConflict("operation_occupied")
                    current = await self._load_entry(cur, request.candidate_entry.id)
                    if current is None:
                        raise KeyError(
                            f"Knowledge entry {request.candidate_entry.id!r} does not exist."
                        )
                    _require_knowledge_entry_access(
                        scope,
                        current,
                        operation="approve_pending_entry",
                    )
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
                    current_chunks = await self._load_chunks(
                        cur,
                        current.id,
                        revision=current.revision,
                    )
                    current_evidence = await self._load_evidence(
                        cur,
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
                        if _knowledge_has_only_default_chunk(current, current_chunks)
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
                    await self._append_revision(
                        cur,
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
                    await self._insert_publication_receipt(cur, publication_receipt, activated)
                    await self._insert_activation_receipt(
                        cur,
                        receipt,
                        access_entry=activated,
                    )
                await conn.commit()
                return KnowledgeReviewApproval(entry=activated, receipt=receipt)
            except UniqueViolation:
                await conn.rollback()
                raise KnowledgeActivationConflict("concurrent_occupancy") from None
            except Exception:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("modify")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_relation_write_identities(
                        cur,
                        operation_id=operation_id,
                        relations=copied_relations,
                    )
                    existing_receipt = await self._load_relation_receipt(
                        cur,
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
                        await conn.commit()
                        return copy_knowledge_relation_publication_receipt(
                            existing_receipt,
                            replayed=True,
                        )

                    endpoint_entries: list[tuple[KnowledgeEntry, KnowledgeEntry]] = []
                    for relation in copied_relations:
                        endpoints: list[KnowledgeEntry] = []
                        for reference in (relation.subject, relation.object):
                            entry = await self._load_entry_in_scope(
                                cur,
                                reference.entry_id,
                                scope,
                                revision=reference.revision,
                            )
                            if entry is None:
                                existing = await self._load_entry(
                                    cur,
                                    reference.entry_id,
                                    revision=reference.revision,
                                )
                                if existing is None:
                                    raise KnowledgeRelationConflict("endpoint_missing")
                                raise KnowledgeAccessDenied("publish_relations")
                            endpoints.append(entry)
                        endpoint_entries.append((endpoints[0], endpoints[1]))
                    current_entries = await self._load_entries(
                        cur,
                        [
                            reference.entry_id
                            for relation in copied_relations
                            for reference in (relation.subject, relation.object)
                        ],
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
                        await cur.execute(
                            """
                            SELECT id, subject_entry_id, subject_revision,
                                   object_entry_id, object_revision, kind,
                                   created_by_type, created_by, policy_id,
                                   created_at, metadata
                            FROM cayu_knowledge_relations
                            WHERE id = %s OR (
                                kind = %s
                                AND subject_entry_id = %s
                                AND subject_revision = %s
                                AND object_entry_id = %s
                                AND object_revision = %s
                            )
                            LIMIT 1
                            """,
                            (relation.id, *_postgres_relation_semantic_row_values(relation)),
                        )
                        row = await cur.fetchone()
                        if row is not None:
                            occupied = _knowledge_relation_from_row(row)
                            if not await self._relation_endpoints_in_scope(
                                cur,
                                occupied,
                                scope,
                            ):
                                raise KnowledgeAccessDenied("publish_relations")
                            raise KnowledgeRelationConflict("relation_exists")
                        await cur.execute(
                            "SELECT sequence FROM cayu_knowledge_changes WHERE relation_id = %s",
                            (relation.id,),
                        )
                        historic_change = await cur.fetchone()
                        if historic_change is not None:
                            access_sql, access_params = (
                                _postgres_knowledge_change_access_scope_filter_sql(scope)
                            )
                            await cur.execute(
                                cast(
                                    "LiteralString",
                                    "SELECT 1 FROM cayu_knowledge_changes AS "
                                    "change_record WHERE change_record.sequence = %s" + access_sql,
                                ),
                                (int(historic_change[0]), *access_params),
                            )
                            if await cur.fetchone() is None:
                                raise KnowledgeAccessDenied("publish_relations")
                            raise KnowledgeRelationConflict("relation_exists")

                    committed_at = self._clock()
                    receipt = KnowledgeRelationPublicationReceipt(
                        operation_id=operation_id,
                        relation_ids=[relation.id for relation in copied_relations],
                        request_sha256=request_sha256,
                        committed_at=committed_at,
                    )
                    await self._insert_relations(cur, copied_relations)
                    for relation, access_snapshot in zip(
                        copied_relations,
                        endpoint_access,
                        strict=True,
                    ):
                        await self._insert_relation_change(
                            cur,
                            relation,
                            access_snapshot=access_snapshot,
                            operation_id=operation_id,
                            committed_at=committed_at,
                        )
                    await self._insert_relation_receipt(
                        cur,
                        receipt,
                        access_snapshots=endpoint_access,
                    )
                await conn.commit()
                return copy_knowledge_relation_publication_receipt(receipt)
            except ForeignKeyViolation:
                await conn.rollback()
                raise KnowledgeRelationConflict("endpoint_missing") from None
            except UniqueViolation:
                await conn.rollback()
                raise KnowledgeRelationConflict("relation_exists") from None
            except BaseException:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_relation_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationPublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_relation_identity(operation_id, "operation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            receipt = await self._load_relation_receipt(
                cur,
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if receipt is None else copy_knowledge_relation_publication_receipt(receipt)

    @runtime_knowledge_operation("read")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            reference = await self._load_entry_in_scope(
                cur,
                query.reference.entry_id,
                scope,
                revision=query.reference.revision,
            )
            if reference is None:
                return None
            relation_sql, relation_params = _postgres_relation_query_filter_sql(query)
            access_sql, access_params = _postgres_relation_access_scope_filter_sql(scope)
            cursor_sql = ""
            cursor_params: list[object] = []
            if cursor is not None:
                cursor_sql = ' AND (relation.created_at, relation.id COLLATE "C") > (%s, %s)'
                cursor_params.extend([cursor.created_at, cursor.relation_id])
            await cur.execute(
                cast(
                    "LiteralString",
                    """
                    SELECT relation.id, relation.subject_entry_id,
                           relation.subject_revision, relation.object_entry_id,
                           relation.object_revision, relation.kind,
                           relation.created_by_type, relation.created_by,
                           relation.policy_id, relation.created_at, relation.metadata
                    FROM cayu_knowledge_relations AS relation
                    WHERE TRUE
                    """
                    + relation_sql
                    + cursor_sql
                    + access_sql
                    + ' ORDER BY relation.created_at, relation.id COLLATE "C" LIMIT %s',
                ),
                (
                    *relation_params,
                    *cursor_params,
                    *access_params,
                    query.limit + 1,
                ),
            )
            rows = await cur.fetchall()
        return _bounded_knowledge_relation_result(
            query,
            [_knowledge_relation_from_row(row) for row in rows],
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            access_now = datetime.now(UTC)
            reference_exact = await self._load_entry(
                cur,
                query.reference.entry_id,
                revision=query.reference.revision,
            )
            reference_live = await self._load_entry(cur, query.reference.entry_id)
            reference_current = (
                reference_live
                if through_sequence is None
                else await self._load_entry_at_change_sequence(
                    cur,
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
            relation_sql, relation_params = _postgres_relation_query_filter_sql(query)
            lineage_sql, lineage_params = _postgres_lineage_filter_sql(query)
            access_sql, access_params = _postgres_relation_access_scope_filter_sql(
                scope,
                allow_archived_current=True,
                now=access_now,
                through_change_sequence=through_sequence,
            )
            cursor_sql = ""
            cursor_params: list[object] = []
            if cursor is not None:
                cursor_sql = ' AND (relation.created_at, relation.id COLLATE "C") > (%s, %s)'
                cursor_params.extend([cursor.created_at, cursor.relation_id])
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
                          AND boundary_change.sequence <= %s
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
                           AND subject_boundary.sequence <= %s
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
                           AND object_boundary.sequence <= %s
                     )
                     AND object_current_change.sequence = (
                         SELECT MAX(object_materialization.sequence)
                         FROM cayu_knowledge_changes AS object_materialization
                         WHERE object_materialization.entry_id = object_current_change.entry_id
                           AND object_materialization.entry_revision =
                                   object_current_change.entry_revision
                           AND object_materialization.kind <> 'relation_published'
                     )
                    JOIN cayu_knowledge_revisions AS object_current
                      ON object_current.entry_id = object_current_change.entry_id
                     AND object_current.revision = object_current_change.entry_revision
                """
                current_join_params.extend((through_sequence, through_sequence))
            await cur.execute(
                cast(
                    "LiteralString",
                    """
                    SELECT relation.id, relation.subject_entry_id,
                           relation.subject_revision, relation.object_entry_id,
                           relation.object_revision, relation.kind,
                           relation.created_at,
                           subject_current.revision, subject_current.status,
                           object_current.revision, object_current.status
                    FROM cayu_knowledge_relations AS relation
                    """
                    + current_join_sql
                    + """
                    WHERE TRUE
                    """
                    + relation_sql
                    + lineage_sql
                    + cursor_sql
                    + frontier_sql
                    + access_sql
                    + ' ORDER BY relation.created_at, relation.id COLLATE "C" LIMIT %s',
                ),
                (
                    *current_join_params,
                    *relation_params,
                    *lineage_params,
                    *cursor_params,
                    *frontier_params,
                    *access_params,
                    query.limit + 1,
                ),
            )
            rows = await cur.fetchall()
        links = [
            _knowledge_lineage_link(
                relation_id=str(row[0]),
                kind=KnowledgeRelationKind(str(row[5])),
                subject=KnowledgeRevisionRef(
                    entry_id=str(row[1]),
                    revision=int(row[2]),
                ),
                object_=KnowledgeRevisionRef(
                    entry_id=str(row[3]),
                    revision=int(row[4]),
                ),
                created_at=pg_support.to_utc(row[6]),
                reference=query.reference,
                subject_current=KnowledgeRevisionRef(
                    entry_id=str(row[1]),
                    revision=int(row[7]),
                ),
                subject_status=KnowledgeStatus(str(row[8])),
                object_current=KnowledgeRevisionRef(
                    entry_id=str(row[3]),
                    revision=int(row[9]),
                ),
                object_status=KnowledgeStatus(str(row[10])),
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

    @runtime_knowledge_operation("modify")
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
        from cayu.knowledge.maintenance_persistence import (
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_write_identities(
                        cur,
                        operation_ids=(operation_id,),
                        maintenance_proposal_ids=(copied_proposal.id,),
                    )
                    await _lock_knowledge_write_identities(
                        cur,
                        entry_ids=(
                            copied_entry.id,
                            *(source.entry_id for source in copied_proposal.sources),
                        ),
                    )
                    await _lock_knowledge_write_identities(
                        cur,
                        chunk_ids=tuple(chunk.id for chunk in copied_chunks),
                        evidence_ids=tuple(item.id for item in copied_evidence),
                    )
                    existing = await self._load_maintenance_proposal_record(
                        cur,
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
                            raise KnowledgeMaintenanceProposalPublicationConflict(
                                "malformed_receipt"
                            )
                        await conn.commit()
                        return copy_knowledge_maintenance_proposal_publication_receipt(
                            receipt,
                            replayed=True,
                        )

                    await cur.execute(
                        "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                        "WHERE proposal_id = %s",
                        (copied_proposal.id,),
                    )
                    occupied = await cur.fetchone()
                    if occupied is not None:
                        await self._load_maintenance_proposal_record(
                            cur,
                            str(occupied[0]),
                            access_scope=scope,
                            deny_inaccessible=True,
                        )
                        raise KnowledgeMaintenanceProposalPublicationConflict("proposal_id_reuse")
                    await cur.execute(
                        "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                        "WHERE proposal_id = %s",
                        (copied_proposal.id,),
                    )
                    decided = await cur.fetchone()
                    if decided is not None:
                        await self._load_maintenance_record(
                            cur,
                            str(decided[0]),
                            access_scope=scope,
                            deny_inaccessible=True,
                        )
                        raise KnowledgeMaintenanceProposalPublicationConflict(
                            "proposal_already_decided"
                        )

                    source_entries = await self._load_entries(
                        cur,
                        [source.entry_id for source in copied_proposal.sources],
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
                    occupied_entry = await self._load_entry(cur, copied_entry.id)
                    if occupied_entry is not None:
                        _require_knowledge_entry_access(
                            scope,
                            occupied_entry,
                            operation=operation,
                        )
                        raise KnowledgeMaintenanceProposalPublicationConflict(
                            "replacement_id_reuse"
                        )
                    await self._require_chunk_ids_available(
                        cur,
                        copied_chunks,
                        access_scope=scope,
                        operation=operation,
                    )
                    await self._require_evidence_ids_available(
                        cur,
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
                    await self._insert_entry(cur, copied_entry)
                    await self._insert_chunks(cur, copied_entry, copied_chunks)
                    await self._insert_evidence(cur, copied_evidence)
                    await _lock_knowledge_change_sequence(cur)
                    await self._insert_change(
                        cur,
                        before_entry=None,
                        after_entry=copied_entry,
                        kind=KnowledgeChangeKind.CREATED,
                        operation_id=operation_id,
                        committed_at=committed_at,
                    )
                    await self._insert_maintenance_proposal_record(
                        cur,
                        copied_proposal,
                        copied_plan,
                        receipt,
                        access_snapshot=snapshot,
                    )
                await conn.commit()
                return copy_knowledge_maintenance_proposal_publication_receipt(receipt)
            except (ForeignKeyViolation, UniqueViolation):
                await conn.rollback()
                raise KnowledgeMaintenanceProposalPublicationConflict(
                    "concurrent_occupancy"
                ) from None
            except BaseException:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_maintenance_proposal_publication(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposalPublication | None:
        from cayu.knowledge.maintenance_persistence import (
            KnowledgeMaintenanceProposalPublication,
            KnowledgeMaintenanceProposalPublicationConflict,
            KnowledgeMaintenanceProposalPublicationOutcome,
            copy_knowledge_maintenance_proposal_publication_receipt,
        )

        scope = self._operation_access_scope(access_scope)
        proposal_id = _knowledge_maintenance_identity(proposal_id, "proposal_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            await cur.execute(
                "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                "WHERE proposal_id = %s",
                (proposal_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            record = await self._load_maintenance_proposal_record(
                cur,
                str(row[0]),
                access_scope=scope,
                deny_inaccessible=False,
            )
            if record is None:
                return None
            proposal, accepted_plan, receipt, publication_snapshot = record
            replacement = await self._load_entry(
                cur,
                proposal.replacement.entry_id,
                revision=proposal.replacement.revision,
            )
            if replacement is None:
                raise KnowledgeMaintenanceProposalPublicationConflict("replacement_missing")
            await cur.execute(
                "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                "WHERE proposal_id = %s",
                (proposal_id,),
            )
            decision_row = await cur.fetchone()
            if decision_row is not None:
                try:
                    decision_record = await self._load_maintenance_record(
                        cur,
                        str(decision_row[0]),
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

    @runtime_knowledge_operation("modify")
    async def record_maintenance_governance_route(
        self,
        authority: KnowledgeMaintenanceGovernanceAuthority,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceGovernanceReceipt:
        from cayu.knowledge.maintenance_governance import (
            KnowledgeMaintenanceGovernanceAuthority,
            KnowledgeMaintenanceGovernanceDisposition,
            KnowledgeMaintenanceGovernanceReceipt,
            copy_knowledge_maintenance_governance_authority,
            copy_knowledge_maintenance_governance_receipt,
            require_knowledge_maintenance_governance_authority_records,
        )

        if type(authority) is not KnowledgeMaintenanceGovernanceAuthority:
            raise TypeError("authority must be a KnowledgeMaintenanceGovernanceAuthority.")
        copied = copy_knowledge_maintenance_governance_authority(authority)
        if (
            copied.decision.disposition
            is not KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW
        ):
            raise ValueError("Only route-to-review authority can use this store operation.")
        scope = self._operation_access_scope(access_scope)
        if scope != copied.request.access_scope:
            raise KnowledgeAccessDenied("record_maintenance_governance_route")
        proposal = copied.request.proposal
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_write_identities(
                        cur,
                        operation_ids=(copied.request.operation_id,),
                        maintenance_proposal_ids=(proposal.id,),
                    )
                    publication = await self._load_maintenance_proposal_record(
                        cur,
                        copied.request.publication_operation_id,
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if publication is None:
                        raise KnowledgeMaintenanceConflict("proposal_publication_mismatch")
                    stored_proposal, accepted_plan, publication_receipt, snapshot = publication
                    if (
                        stored_proposal != proposal
                        or stored_proposal.fingerprint != proposal.fingerprint
                        or accepted_plan.fingerprint != copied.request.accepted_plan_fingerprint
                        or publication_receipt.request_sha256
                        != copied.request.publication_request_sha256
                    ):
                        raise KnowledgeMaintenanceConflict("proposal_publication_mismatch")
                    copied = require_knowledge_maintenance_governance_authority_records(
                        copied,
                        stored_proposal,
                        accepted_plan,
                        publication_receipt,
                    )

                    existing = await self._load_maintenance_governance_route(
                        cur,
                        copied.request.operation_id,
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if existing is not None:
                        if existing.authority != copied:
                            raise KnowledgeMaintenanceConflict("governance_operation_reuse")
                        await conn.commit()
                        return copy_knowledge_maintenance_governance_receipt(
                            existing,
                            replayed=True,
                        )
                    await cur.execute(
                        "SELECT 1 FROM cayu_knowledge_maintenance_decisions "
                        "WHERE operation_id = %s",
                        (copied.request.operation_id,),
                    )
                    if await cur.fetchone() is not None:
                        raise KnowledgeMaintenanceConflict("governance_operation_reuse")
                    await cur.execute(
                        "SELECT operation_id FROM "
                        "cayu_knowledge_maintenance_governance_routes "
                        "WHERE proposal_id = %s",
                        (proposal.id,),
                    )
                    prior_route = await cur.fetchone()
                    if prior_route is not None:
                        await self._load_maintenance_governance_route(
                            cur,
                            str(prior_route[0]),
                            access_scope=scope,
                            deny_inaccessible=True,
                        )
                        raise KnowledgeMaintenanceConflict("proposal_already_governed")
                    await cur.execute(
                        "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                        "WHERE proposal_id = %s",
                        (proposal.id,),
                    )
                    prior_decision = await cur.fetchone()
                    if prior_decision is not None:
                        await self._load_maintenance_record(
                            cur,
                            str(prior_decision[0]),
                            access_scope=scope,
                            deny_inaccessible=True,
                        )
                        raise KnowledgeMaintenanceConflict("proposal_already_decided")

                    committed_at = max(
                        self._clock(),
                        proposal.created_at,
                        publication_receipt.committed_at,
                    )
                    receipt = KnowledgeMaintenanceGovernanceReceipt(
                        operation_id=copied.request.operation_id,
                        proposal_id=proposal.id,
                        proposal_fingerprint=proposal.fingerprint,
                        authority=copied,
                        committed_at=committed_at,
                    )
                    await cur.execute(
                        """
                        INSERT INTO cayu_knowledge_maintenance_governance_routes (
                            operation_id, proposal_id, proposal_fingerprint,
                            request_sha256, committed_at, receipt_json,
                            access_snapshot
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            receipt.operation_id,
                            receipt.proposal_id,
                            receipt.proposal_fingerprint,
                            copied.request.fingerprint,
                            pg_support.to_utc(receipt.committed_at),
                            receipt.model_dump_json(warnings=False),
                            _knowledge_maintenance_access_snapshot_json(snapshot),
                        ),
                    )
                await conn.commit()
                return copy_knowledge_maintenance_governance_receipt(receipt)
            except UniqueViolation:
                await conn.rollback()
                raise KnowledgeMaintenanceConflict("concurrent_occupancy") from None
            except BaseException:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_maintenance_governance_route(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceGovernanceReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            return await self._load_maintenance_governance_route(
                cur,
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )

    @runtime_knowledge_operation("modify")
    async def record_semantic_watch_outcome(
        self,
        authority: KnowledgeSemanticWatchAuthority,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSemanticWatchReceipt:
        from cayu.knowledge.semantic_watch import (
            KnowledgeSemanticWatchAuthority,
            KnowledgeSemanticWatchConflict,
            KnowledgeSemanticWatchReceipt,
            copy_knowledge_semantic_watch_authority,
            copy_knowledge_semantic_watch_receipt,
            require_knowledge_semantic_watch_authority_records,
        )

        if type(authority) is not KnowledgeSemanticWatchAuthority:
            raise TypeError("authority must be a KnowledgeSemanticWatchAuthority.")
        copied = copy_knowledge_semantic_watch_authority(authority)
        scope = self._operation_access_scope(access_scope)
        if scope != copied.invocation.access_scope:
            raise KnowledgeAccessDenied("record_semantic_watch_outcome")
        operation_id = copied.invocation.operation_id
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_semantic_watch_write_identities(
                        cur,
                        operation_id=operation_id,
                        entry_ids=tuple(
                            candidate.reference.entry_id for candidate in copied.evidence.candidates
                        ),
                    )
                    existing = await self._load_semantic_watch_receipt(
                        cur,
                        operation_id,
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if existing is not None:
                        if existing.authority.invocation != copied.invocation:
                            raise KnowledgeSemanticWatchConflict("operation_reuse")
                        await conn.commit()
                        return copy_knowledge_semantic_watch_receipt(existing, replayed=True)
                    validation_now = self._clock()
                    records = []
                    references = {candidate.reference for candidate in copied.evidence.candidates}
                    for reference in sorted(
                        references,
                        key=lambda item: (item.entry_id, item.revision),
                    ):
                        entry = await self._load_entry(cur, reference.entry_id)
                        if entry is None or entry.revision != reference.revision:
                            raise KnowledgeSemanticWatchConflict("candidate_stale")
                        if not _knowledge_scope_allows_entry(
                            scope,
                            entry,
                            now=validation_now,
                        ):
                            raise KnowledgeAccessDenied("record_semantic_watch_outcome")
                        records.append(
                            (
                                entry,
                                await self._load_chunks(
                                    cur,
                                    entry.id,
                                    revision=entry.revision,
                                ),
                            )
                        )
                    copied = require_knowledge_semantic_watch_authority_records(
                        copied,
                        records,
                        now=validation_now,
                    )
                    receipt = KnowledgeSemanticWatchReceipt(
                        operation_id=operation_id,
                        invocation_sha256=copied.invocation.fingerprint,
                        request_sha256=copied.decision.request_sha256,
                        authority=copied,
                        committed_at=validation_now,
                    )
                    await cur.execute(
                        """
                        INSERT INTO cayu_knowledge_semantic_watch_receipts (
                            operation_id, invocation_sha256, request_sha256,
                            committed_at, receipt_json, access_scope
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            receipt.operation_id,
                            receipt.invocation_sha256,
                            receipt.request_sha256,
                            pg_support.to_utc(receipt.committed_at),
                            receipt.model_dump_json(warnings=False),
                            scope.model_dump_json(warnings=False),
                        ),
                    )
                await conn.commit()
                return copy_knowledge_semantic_watch_receipt(receipt)
            except UniqueViolation:
                await conn.rollback()
                raise KnowledgeSemanticWatchConflict("operation_reuse") from None
            except BaseException:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_semantic_watch_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSemanticWatchReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_semantic_watch_identity(operation_id, "operation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            return await self._load_semantic_watch_receipt(
                cur,
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )

    @runtime_knowledge_operation("modify")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await _lock_knowledge_maintenance_write_identities(
                        cur,
                        proposal=proposal,
                        decision=decision,
                    )
                    await cur.execute(
                        "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                        "WHERE proposal_id = %s OR replacement_entry_id = %s "
                        "ORDER BY operation_id",
                        (proposal.id, proposal.replacement.entry_id),
                    )
                    publication_rows = await cur.fetchall()
                    publication_snapshot: _KnowledgeMaintenanceAccessSnapshot | None = None
                    governance_publication = None
                    for publication_row in publication_rows:
                        publication = await self._load_maintenance_proposal_record(
                            cur,
                            str(publication_row[0]),
                            access_scope=scope,
                            deny_inaccessible=True,
                        )
                        if publication is None or publication[0] != proposal:
                            raise KnowledgeMaintenanceConflict("proposal_publication_mismatch")
                        if (
                            publication_snapshot is not None
                            and publication_snapshot != publication[3]
                        ):
                            raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
                        publication_snapshot = publication[3]
                        governance_publication = publication
                    if KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY in decision.metadata:
                        from cayu.knowledge.maintenance_governance import (
                            governance_authority_from_maintenance_records,
                        )

                        if governance_publication is None:
                            raise KnowledgeMaintenanceConflict(
                                "governance_requires_published_proposal"
                            )
                        governance_authority_from_maintenance_records(
                            governance_publication[0],
                            governance_publication[1],
                            governance_publication[2],
                            decision,
                        )
                    routed = await self._load_maintenance_governance_route(
                        cur,
                        decision.operation_id,
                        access_scope=scope,
                        deny_inaccessible=True,
                    )
                    if routed is not None:
                        raise KnowledgeMaintenanceConflict("operation_reuse")
                    if KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY in decision.metadata:
                        await cur.execute(
                            "SELECT operation_id FROM "
                            "cayu_knowledge_maintenance_governance_routes "
                            "WHERE proposal_id = %s",
                            (proposal.id,),
                        )
                        prior_route = await cur.fetchone()
                        if prior_route is not None:
                            await self._load_maintenance_governance_route(
                                cur,
                                str(prior_route[0]),
                                access_scope=scope,
                                deny_inaccessible=True,
                            )
                            raise KnowledgeMaintenanceConflict("proposal_already_governed")
                    existing = await self._load_maintenance_record(
                        cur,
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
                        await conn.commit()
                        return copy_knowledge_maintenance_decision_receipt(
                            receipt,
                            replayed=True,
                        )

                    await cur.execute(
                        "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                        "WHERE proposal_id = %s",
                        (proposal.id,),
                    )
                    prior = await cur.fetchone()
                    if prior is not None:
                        await self._load_maintenance_record(
                            cur,
                            str(prior[0]),
                            access_scope=scope,
                            deny_inaccessible=True,
                        )
                        raise KnowledgeMaintenanceConflict("proposal_already_decided")
                    current_entries = await self._load_entries(
                        cur,
                        [
                            proposal.replacement.entry_id,
                            *(source.entry_id for source in proposal.sources),
                        ],
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
                        await self._insert_maintenance_record(
                            cur,
                            proposal,
                            decision,
                            receipt,
                            access_snapshot=decision_snapshot,
                        )
                        await conn.commit()
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
                        await cur.execute(
                            """
                            SELECT id, subject_entry_id, subject_revision,
                                   object_entry_id, object_revision, kind,
                                   created_by_type, created_by, policy_id,
                                   created_at, metadata
                            FROM cayu_knowledge_relations
                            WHERE id = %s OR (
                                kind = %s
                                AND subject_entry_id = %s
                                AND subject_revision = %s
                                AND object_entry_id = %s
                                AND object_revision = %s
                            )
                            LIMIT 1
                            """,
                            (relation.id, *_postgres_relation_semantic_row_values(relation)),
                        )
                        row = await cur.fetchone()
                        if row is not None:
                            occupied = _knowledge_relation_from_row(row)
                            if not await self._relation_endpoints_in_scope(cur, occupied, scope):
                                raise KnowledgeAccessDenied(operation)
                            raise KnowledgeMaintenanceConflict("relation_exists")
                        await cur.execute(
                            "SELECT sequence FROM cayu_knowledge_changes WHERE relation_id = %s",
                            (relation.id,),
                        )
                        historic = await cur.fetchone()
                        if historic is not None:
                            access_sql, access_params = (
                                _postgres_knowledge_change_access_scope_filter_sql(scope)
                            )
                            await cur.execute(
                                cast(
                                    "LiteralString",
                                    "SELECT 1 FROM cayu_knowledge_changes AS change_record "
                                    "WHERE change_record.sequence = %s" + access_sql,
                                ),
                                (int(historic[0]), *access_params),
                            )
                            if await cur.fetchone() is None:
                                raise KnowledgeAccessDenied(operation)
                            raise KnowledgeMaintenanceConflict("relation_exists")

                    for successor in [active_replacement, *archived_sources]:
                        await self._append_revision(
                            cur,
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

                    post_current = await self._load_entries(cur, list(current_entries))
                    relation_access: list[_KnowledgeRelationAccessSnapshot] = []
                    for relation in proposal.relations:
                        subject_exact = await self._load_entry(
                            cur,
                            relation.subject.entry_id,
                            revision=relation.subject.revision,
                        )
                        object_exact = await self._load_entry(
                            cur,
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
                    await self._insert_relations(cur, proposal.relations)
                    for relation, snapshot in zip(
                        proposal.relations,
                        relation_access,
                        strict=True,
                    ):
                        await self._insert_relation_change(
                            cur,
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
                    await self._insert_maintenance_record(
                        cur,
                        proposal,
                        decision,
                        receipt,
                        access_snapshot=decision_snapshot,
                    )
                await conn.commit()
                return copy_knowledge_maintenance_decision_receipt(receipt)
            except ForeignKeyViolation:
                await conn.rollback()
                raise KnowledgeMaintenanceConflict("relation_endpoint") from None
            except UniqueViolation:
                await conn.rollback()
                raise KnowledgeMaintenanceConflict("identity_conflict") from None
            except BaseException:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_maintenance_proposal(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposal | None:
        scope = self._operation_access_scope(access_scope)
        proposal_id = _knowledge_maintenance_identity(proposal_id, "proposal_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            await cur.execute(
                "SELECT operation_id FROM cayu_knowledge_maintenance_proposals "
                "WHERE proposal_id = %s",
                (proposal_id,),
            )
            publication_row = await cur.fetchone()
            if publication_row is not None:
                publication = await self._load_maintenance_proposal_record(
                    cur,
                    str(publication_row[0]),
                    access_scope=scope,
                    deny_inaccessible=False,
                )
                return (
                    None
                    if publication is None
                    else copy_knowledge_maintenance_proposal(publication[0])
                )
            await cur.execute(
                "SELECT operation_id FROM cayu_knowledge_maintenance_decisions "
                "WHERE proposal_id = %s",
                (proposal_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            record = await self._load_maintenance_record(
                cur,
                str(row[0]),
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if record is None else copy_knowledge_maintenance_proposal(record[0])

    @runtime_knowledge_operation("read")
    async def load_maintenance_decision(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecision | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            record = await self._load_maintenance_record(
                cur,
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if record is None else copy_knowledge_maintenance_decision(record[1])

    @runtime_knowledge_operation("read")
    async def load_maintenance_decision_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecisionReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            record = await self._load_maintenance_record(
                cur,
                operation_id,
                access_scope=scope,
                deny_inaccessible=False,
            )
        return None if record is None else copy_knowledge_maintenance_decision_receipt(record[2])

    async def inspect_closure_sources(self, query: KnowledgeClosureQuery) -> dict[str, object]:
        query = copy_knowledge_closure_query(query)
        inventory = KnowledgeClosureInventory(query)
        revisions: set[tuple[str, int]] = set()
        count = 0
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            for start in range(0, max(len(query.sources), len(query.source_uris)), 100):
                batch = query.sources[start : start + 100]
                uri_batch = query.source_uris[start : start + 100]
                source_types = [pair[0] for pair in batch]
                source_ids = [pair[1] for pair in batch]
                uri_types = [pair[0] for pair in uri_batch]
                source_uris = [pair[1] for pair in uri_batch]
                await cur.execute(
                    """
                    SELECT COUNT(*), MAX(octet_length(e.locator::text) + octet_length(e.metadata::text))
                    FROM cayu_knowledge_evidence e
                    WHERE EXISTS (
                        SELECT 1 FROM unnest(%s::text[], %s::text[]) AS selected(source_type, source_id)
                        WHERE e.source_type = selected.source_type AND e.source_id = selected.source_id
                    ) OR EXISTS (
                        SELECT 1 FROM unnest(%s::text[], %s::text[]) AS selected(source_type, source_uri)
                        WHERE e.source_type = selected.source_type AND e.source_uri = selected.source_uri
                    )
                    """,
                    (source_types, source_ids, uri_types, source_uris),
                )
                sizes = await cur.fetchone()
                if (
                    sizes is None
                    or sizes[0] > query.max_records
                    or (sizes[1] or 0) > query.max_bytes
                ):
                    raise ValueError("Knowledge closure inventory exceeds its bounds.")
                await cur.execute(
                    """
                    SELECT e.id, e.entry_id, e.entry_revision, e.chunk_id, e.role,
                           e.source_type, e.source_id, e.source_uri, e.source_revision,
                           e.source_hash, e.locator, e.disposition, e.created_at, e.metadata
                    FROM cayu_knowledge_evidence e
                    WHERE EXISTS (
                        SELECT 1 FROM unnest(%s::text[], %s::text[]) AS selected(source_type, source_id)
                        WHERE e.source_type = selected.source_type AND e.source_id = selected.source_id
                    ) OR EXISTS (
                        SELECT 1 FROM unnest(%s::text[], %s::text[]) AS selected(source_type, source_uri)
                        WHERE e.source_type = selected.source_type AND e.source_uri = selected.source_uri
                    )
                    ORDER BY e.id LIMIT %s
                    """,
                    (
                        source_types,
                        source_ids,
                        uri_types,
                        source_uris,
                        query.max_records + 1,
                    ),
                )
                while rows := await cur.fetchmany(100):
                    for row in rows:
                        revisions.add(inventory.add_evidence(_knowledge_evidence_from_row(row)))
                await cur.execute(
                    """
                    SELECT e.entry_id, e.revision, e.source_type, e.source_id, e.source_uri, e.source_hash
                    FROM cayu_knowledge_revisions e
                    WHERE EXISTS (
                        SELECT 1 FROM unnest(%s::text[], %s::text[]) AS selected(source_type, source_id)
                        WHERE e.source_type = selected.source_type AND e.source_id = selected.source_id
                    ) OR EXISTS (
                        SELECT 1 FROM unnest(%s::text[], %s::text[]) AS selected(source_type, source_uri)
                        WHERE e.source_type = selected.source_type AND e.source_uri = selected.source_uri
                    )
                    ORDER BY e.entry_id, e.revision LIMIT %s
                    """,
                    (source_types, source_ids, uri_types, source_uris, query.max_records + 1),
                )
                while rows := await cur.fetchmany(100):
                    for row in rows:
                        revisions.add(inventory.add_revision(*row))
            ordered_revisions = sorted(revisions)
            for start in range(0, len(ordered_revisions), 100):
                batch = ordered_revisions[start : start + 100]
                await cur.execute(
                    """
                    SELECT event.* FROM cayu_knowledge_index_readiness_events event
                    JOIN unnest(%s::text[], %s::bigint[]) AS selected(entry_id, revision)
                      ON event.entry_id = selected.entry_id
                     AND event.entry_revision = selected.revision
                    ORDER BY event.sequence LIMIT %s
                    """,
                    (
                        [pair[0] for pair in batch],
                        [pair[1] for pair in batch],
                        query.max_records + 1,
                    ),
                )
                while rows := await cur.fetchmany(100):
                    for row in rows:
                        inventory.add_readiness(_knowledge_index_readiness_from_row(row))
            count = inventory.count
            # A plain knowledge-store handle can share a database with a vector
            # store. Inspect the actual durable table, not this handle's class.
            await cur.execute("SELECT to_regclass('cayu_knowledge_embeddings')")
            table = await cur.fetchone()
            if table is not None and table[0] is not None:
                ordered_revisions = sorted(revisions)
                for start in range(0, len(ordered_revisions), 100):
                    batch = ordered_revisions[start : start + 100]
                    await cur.execute(
                        """
                        SELECT e.entry_id, e.entry_revision, e.chunk_id, e.projection_type,
                               e.projection_content_hash, e.embedding_model, e.dimensions,
                               e.preprocessing_version, e.generator, e.generator_version,
                               e.index_representation_version, e.attempt_id, e.readiness_sequence,
                               e.embedding_sha256
                        FROM cayu_knowledge_embeddings e
                        JOIN unnest(%s::text[], %s::bigint[]) AS selected(entry_id, revision)
                          ON e.entry_id = selected.entry_id AND e.entry_revision = selected.revision
                        ORDER BY e.identity_sha256, e.readiness_sequence LIMIT %s
                        """,
                        (
                            [pair[0] for pair in batch],
                            [pair[1] for pair in batch],
                            query.max_records - count + 1,
                        ),
                    )
                    while rows := await cur.fetchmany(100):
                        for row in rows:
                            count += 1
                            if count > query.max_records:
                                raise ValueError("Knowledge closure inventory exceeds its bounds.")
                            identity = KnowledgeEmbeddingIdentity(
                                **dict(
                                    zip(
                                        (
                                            "entry_id",
                                            "entry_revision",
                                            "chunk_id",
                                            "projection_type",
                                            "projection_content_hash",
                                            "embedding_model",
                                            "dimensions",
                                            "preprocessing_version",
                                            "generator",
                                            "generator_version",
                                            "index_representation_version",
                                        ),
                                        row[:11],
                                        strict=True,
                                    )
                                )
                            )
                            inventory.add(
                                "knowledge_projections",
                                {
                                    "identity": identity.model_dump(mode="json"),
                                    "attempt_id": row[11],
                                    "readiness_sequence": row[12],
                                    "vector_sha256": row[13],
                                },
                            )
            return inventory.document()

    @runtime_knowledge_operation("read")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            entry = await self._load_entry_in_scope(
                cur,
                entry_id,
                scope,
                revision=revision,
            )
            if entry is None:
                return None
            await cur.execute(
                """
                SELECT COUNT(*)
                FROM cayu_knowledge_evidence
                WHERE entry_id = %s AND entry_revision = %s
                """,
                (entry.id, entry.revision),
            )
            total_row = await cur.fetchone()
            total_evidence_known = 0 if total_row is None else int(total_row[0])
            stored = await self._load_evidence(
                cur,
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

    @runtime_knowledge_operation("read")
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
        access_sql, access_params = _postgres_knowledge_change_access_scope_filter_sql(
            scope,
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            await cur.execute(
                cast(
                    "LiteralString",
                    "SELECT COALESCE(MAX(change_record.sequence), 0) "
                    "FROM cayu_knowledge_changes AS change_record "
                    f"WHERE TRUE {access_sql}",
                ),
                access_params,
            )
            high_water_row = await cur.fetchone()
            high_water = 0 if high_water_row is None else int(high_water_row[0])
            if after_sequence > high_water:
                await cur.execute("SELECT COALESCE(MAX(sequence), 0) FROM cayu_knowledge_changes")
                current_sequence_row = await cur.fetchone()
                current_sequence = (
                    0 if current_sequence_row is None else int(current_sequence_row[0])
                )
                if after_sequence > current_sequence:
                    raise ValueError(
                        "`after_sequence` cannot exceed the current knowledge change sequence."
                    )
            await cur.execute(
                cast(
                    "LiteralString",
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
                    WHERE change_record.sequence > %s
                      AND change_record.sequence <= %s
                    """
                    + access_sql
                    + " ORDER BY change_record.sequence LIMIT %s",
                ),
                (after_sequence, high_water, *access_params, limit + 1),
            )
            rows = await cur.fetchall()
        changes = [_knowledge_change_from_row(row) for row in rows[:limit]]
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

    @runtime_knowledge_operation("modify")
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
        access_sql, access_params = _postgres_knowledge_change_access_scope_filter_sql(
            scope,
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    state = await self._lock_or_create_change_consumer(
                        cur,
                        consumer_id,
                        access_scope_sha256=scope_sha256,
                    )
                    current_time = await self._change_consumer_now(cur)
                    if state.access_scope_sha256 != scope_sha256:
                        raise KnowledgeChangeConsumerConflict("access_scope_mismatch")
                    if state.pending_change_sequence is not None:
                        await cur.execute(
                            cast(
                                "LiteralString",
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
                                WHERE change_record.sequence = %s
                                """
                                + access_sql,
                            ),
                            (state.pending_change_sequence, *access_params),
                        )
                        row = await cur.fetchone()
                        stored_change = None if row is None else _knowledge_change_from_row(row)
                        assert state.lease_expires_at is not None
                        if stored_change is not None and state.lease_expires_at > current_time:
                            if state.pending_worker_id != worker_id:
                                await conn.commit()
                                return None
                            assert state.pending_claim_id is not None
                            assert state.claimed_at is not None
                            claim = KnowledgeChangeClaim(
                                consumer_id=consumer_id,
                                worker_id=worker_id,
                                claim_id=state.pending_claim_id,
                                change=stored_change,
                                attempt=state.pending_attempt,
                                claimed_at=state.claimed_at,
                                lease_expires_at=state.lease_expires_at,
                            )
                            await conn.commit()
                            return claim
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
                    await cur.execute(
                        cast(
                            "LiteralString",
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
                            WHERE change_record.sequence > %s
                            """
                            + access_sql
                            + " ORDER BY change_record.sequence LIMIT 1",
                        ),
                        (state.cursor_sequence, *access_params),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await self._save_change_consumer(cur, state)
                        await conn.commit()
                        return None
                    change = _knowledge_change_from_row(row)
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
                    await self._save_change_consumer(cur, state)
                    claim = KnowledgeChangeClaim(
                        consumer_id=consumer_id,
                        worker_id=worker_id,
                        claim_id=claim_id,
                        change=change,
                        attempt=attempt,
                        claimed_at=claimed_at,
                        lease_expires_at=lease_expires_at,
                    )
                await conn.commit()
                return claim
            except Exception:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("modify")
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
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM cayu_knowledge_changes"
                    )
                    row = await cur.fetchone()
                    current_sequence = 0 if row is None else int(row[0])
                    if baseline_sequence > current_sequence:
                        raise ValueError(
                            "`baseline_sequence` cannot exceed the current knowledge "
                            "change sequence."
                        )
                    state = await self._lock_or_create_change_consumer(
                        cur,
                        consumer_id,
                        access_scope_sha256=scope_sha256,
                    )
                    current_time = await self._change_consumer_now(cur)
                    state = _initialize_knowledge_change_consumer_state(
                        state,
                        consumer_id=consumer_id,
                        access_scope_sha256=scope_sha256,
                        baseline_sequence=baseline_sequence,
                        now=current_time,
                    )
                    await self._save_change_consumer(cur, state)
                await conn.commit()
                return copy_knowledge_change_consumer_state(state)
            except Exception:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("modify")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    state = await self._load_change_consumer(
                        cur,
                        claim.consumer_id,
                        for_update=True,
                    )
                    if state is None or state.access_scope_sha256 != scope_sha256:
                        raise KnowledgeChangeConsumerConflict("unknown_consumer")
                    acknowledged = await self._load_change_acknowledgement(
                        cur,
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
                        await conn.commit()
                        return copy_knowledge_change_consumer_state(state)
                    current_time = await self._change_consumer_now(cur)
                    await self._require_live_change_claim(
                        cur,
                        state,
                        claim,
                        now=current_time,
                    )
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
                    await self._save_change_consumer(cur, state)
                    await self._insert_change_acknowledgement(
                        cur,
                        claim,
                        claim_sha256=claim_sha256,
                        acknowledged_at=current_time,
                    )
                await conn.commit()
                return copy_knowledge_change_consumer_state(state)
            except Exception:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("modify")
    async def release_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        claim = copy_knowledge_change_claim(claim)
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    state = await self._load_change_consumer(
                        cur,
                        claim.consumer_id,
                        for_update=True,
                    )
                    if state is None or state.access_scope_sha256 != scope_sha256:
                        raise KnowledgeChangeConsumerConflict("unknown_consumer")
                    current_time = await self._change_consumer_now(cur)
                    await self._require_live_change_claim(
                        cur,
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
                    await self._save_change_consumer(cur, state)
                await conn.commit()
                return copy_knowledge_change_consumer_state(state)
            except Exception:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_change_consumer_state(
        self,
        consumer_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState | None:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            state = await self._load_change_consumer(
                cur,
                consumer_id,
                for_update=False,
            )
        if state is None or state.access_scope_sha256 != scope_sha256:
            return None
        return copy_knowledge_change_consumer_state(state)

    @runtime_knowledge_operation("modify")
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"knowledge-index-operation:{operation_id}",),
                    )
                    await cur.execute(
                        "SELECT * FROM cayu_knowledge_index_readiness_events "
                        "WHERE operation_id = %s",
                        (operation_id,),
                    )
                    replay_row = await cur.fetchone()
                    if replay_row is not None:
                        if str(replay_row[17]) != update_sha256:
                            raise KnowledgeIndexReadinessConflict("operation_reuse")
                        if not await self._index_identity_is_accessible(
                            cur,
                            scope,
                            update.identity,
                        ):
                            raise KnowledgeAccessDenied("publish_index_readiness")
                        readiness = _knowledge_index_readiness_from_row(replay_row)
                        await conn.commit()
                        return readiness
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"knowledge-index-identity:{identity_sha256}",),
                    )
                    if not await self._index_identity_is_accessible(
                        cur,
                        scope,
                        update.identity,
                        require_current=True,
                    ):
                        raise KnowledgeIndexReadinessConflict("stale_identity")
                    await cur.execute(
                        """
                        SELECT event.*
                        FROM cayu_knowledge_index_readiness_current AS current
                        JOIN cayu_knowledge_index_readiness_events AS event
                          ON event.sequence = current.sequence
                         AND event.identity_sha256 = current.identity_sha256
                        WHERE current.identity_sha256 = %s
                        """,
                        (identity_sha256,),
                    )
                    current_row = await cur.fetchone()
                    current = (
                        None
                        if current_row is None
                        else _knowledge_index_readiness_from_row(current_row)
                    )
                    _validate_knowledge_index_readiness_transition(
                        current,
                        update,
                        expected_sequence=expected_sequence,
                    )
                    await cur.execute("SELECT clock_timestamp()")
                    now_row = await cur.fetchone()
                    if now_row is None:  # pragma: no cover - database invariant
                        raise RuntimeError("Postgres did not return its current time.")
                    published_at = pg_support.to_utc(now_row[0])
                    await cur.execute(
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
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s
                        )
                        RETURNING sequence
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
                            published_at,
                        ),
                    )
                    sequence_row = await cur.fetchone()
                    if sequence_row is None:  # pragma: no cover - database invariant
                        raise RuntimeError("Postgres did not return readiness sequence.")
                    sequence = int(sequence_row[0])
                    if current is None:
                        await cur.execute(
                            """
                            INSERT INTO cayu_knowledge_index_readiness_current (
                                identity_sha256, sequence
                            ) VALUES (%s, %s)
                            """,
                            (identity_sha256, sequence),
                        )
                    else:
                        await cur.execute(
                            """
                            UPDATE cayu_knowledge_index_readiness_current
                            SET sequence = %s
                            WHERE identity_sha256 = %s AND sequence = %s
                            """,
                            (sequence, identity_sha256, current.sequence),
                        )
                        if cur.rowcount != 1:  # pragma: no cover - advisory-lock invariant
                            raise KnowledgeIndexReadinessConflict("stale_sequence")
                await conn.commit()
                return KnowledgeIndexReadiness(
                    sequence=sequence,
                    identity=update.identity,
                    state=update.state,
                    attempt_id=update.attempt_id,
                    failure_code=update.failure_code,
                    operation_id=operation_id,
                    published_at=published_at,
                )
            except Exception:
                await conn.rollback()
                raise

    @runtime_knowledge_operation("read")
    async def load_index_readiness(
        self,
        identity: KnowledgeEmbeddingIdentity,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness | None:
        scope = self._operation_access_scope(access_scope)
        identity = copy_knowledge_embedding_identity(identity)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            if not await self._index_identity_is_accessible(cur, scope, identity):
                return None
            await cur.execute(
                """
                SELECT event.*
                FROM cayu_knowledge_index_readiness_current AS current
                JOIN cayu_knowledge_index_readiness_events AS event
                  ON event.sequence = current.sequence
                 AND event.identity_sha256 = current.identity_sha256
                WHERE current.identity_sha256 = %s
                """,
                (_knowledge_embedding_identity_sha256(identity),),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        readiness = _knowledge_index_readiness_from_row(row)
        if readiness.identity != identity:
            raise RuntimeError("Knowledge index readiness identity digest collision.")
        return readiness

    @runtime_knowledge_operation("read")
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
        exact_access_sql, exact_access_params = _postgres_knowledge_access_scope_filter_sql(
            scope,
            entry_alias="e",
        )
        current_access_sql, current_access_params = _postgres_knowledge_access_scope_filter_sql(
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            await cur.execute(
                cast(
                    "LiteralString",
                    "SELECT COALESCE(MAX(event.sequence), 0) "
                    + accessible_from
                    + exact_access_sql
                    + current_access_sql,
                ),
                access_params,
            )
            high_water_row = await cur.fetchone()
            high_water = 0 if high_water_row is None else int(high_water_row[0])
            if after_sequence > high_water:
                await cur.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM cayu_knowledge_index_readiness_events"
                )
                current_row = await cur.fetchone()
                current_sequence = 0 if current_row is None else int(current_row[0])
                if after_sequence > current_sequence:
                    raise ValueError(
                        "`after_sequence` cannot exceed the current knowledge "
                        "index readiness sequence."
                    )
            await cur.execute(
                cast(
                    "LiteralString",
                    "SELECT event.* "
                    + accessible_from
                    + " AND event.sequence > %s AND event.sequence <= %s"
                    + exact_access_sql
                    + current_access_sql
                    + " ORDER BY event.sequence LIMIT %s",
                ),
                (after_sequence, high_water, *access_params, limit + 1),
            )
            rows = await cur.fetchall()
        readiness = [_knowledge_index_readiness_from_row(row) for row in rows[:limit]]
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

    async def _index_identity_is_accessible(
        self,
        cur: Any,
        scope: KnowledgeAccessScope,
        identity: KnowledgeEmbeddingIdentity,
        *,
        require_current: bool = False,
    ) -> bool:
        current = await self._load_entry_in_scope(cur, identity.entry_id, scope)
        if current is None:
            return False
        if require_current and current.revision != identity.entry_revision:
            return False
        revision = await self._load_entry_in_scope(
            cur,
            identity.entry_id,
            scope,
            revision=identity.entry_revision,
        )
        if revision is None:
            return False
        if identity.chunk_id is None:
            return True
        chunk = await self._load_chunk(cur, identity.chunk_id)
        if (
            chunk is None
            or chunk.entry_id != identity.entry_id
            or chunk.entry_revision != identity.entry_revision
        ):
            return False
        if identity.projection_type == KNOWLEDGE_CHUNK_TEXT_PROJECTION:
            return identity.projection_content_hash == _knowledge_chunk_content_hash(chunk)
        return True

    @runtime_knowledge_operation("read")
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
        entry_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if chunk_index is not None:
            _validate_knowledge_nonnegative_int(chunk_index, "chunk_index")
        _validate_knowledge_nonnegative_int(around, "around")
        if chunk_index is None and around != 0:
            raise ValueError("`around` requires `chunk_index`.")
        _validate_knowledge_positive_int(max_chunks, "max_chunks")
        _validate_knowledge_positive_int(max_bytes, "max_bytes")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            entry = await self._load_entry_in_scope(
                cur,
                entry_id,
                scope,
                revision=revision,
            )
            if entry is None:
                return []
            chunks = await self._load_chunks(cur, entry_id, revision=entry.revision)
        if chunk_index is not None:
            chunks = _center_knowledge_chunk_window(
                chunks,
                chunk_index=chunk_index,
                max_chunks=max_chunks,
            )
        start_index = 0 if chunk_index is None else max(0, chunk_index - around)
        end_index = None if chunk_index is None else chunk_index + around
        return _bounded_knowledge_chunks(
            chunks,
            start_index=start_index,
            end_index=end_index,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )

    @runtime_knowledge_operation("read")
    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_query(query)
        if query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("PostgresKnowledgeStore supports only auto and keyword search modes.")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            return await self._keyword_search_in_snapshot(
                cur,
                query,
                access_scope=scope,
                revision_refs=None,
                through_change_sequence=None,
            )

    @runtime_knowledge_operation("read")
    async def search_at_frontier(
        self,
        query: KnowledgeQuery,
        *,
        knowledge_sequence: int,
        index_readiness_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_query(query)
        _validate_knowledge_change_sequence(knowledge_sequence, "knowledge_sequence")
        _validate_knowledge_index_sequence(
            index_readiness_sequence,
            "index_readiness_sequence",
        )
        if query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("PostgresKnowledgeStore supports only auto and keyword search modes.")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            return await self._keyword_search_in_snapshot(
                cur,
                query,
                access_scope=scope,
                revision_refs=None,
                through_change_sequence=knowledge_sequence,
            )

    @runtime_knowledge_operation("read")
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
        query = copy_knowledge_query(query)
        references = copy_knowledge_revision_refs(revision_refs)
        _validate_knowledge_search_frontier(
            knowledge_sequence,
            index_readiness_sequence,
        )
        if query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("PostgresKnowledgeStore supports only auto and keyword search modes.")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            return await self._keyword_search_in_snapshot(
                cur,
                query,
                access_scope=scope,
                revision_refs=references,
                through_change_sequence=knowledge_sequence,
            )

    async def _keyword_search_in_snapshot(
        self,
        cur: Any,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope,
        revision_refs: tuple[KnowledgeRevisionRef, ...] | None = None,
        through_change_sequence: int | None = None,
    ) -> KnowledgeSearchResult:
        """Run keyword retrieval inside a caller-owned read snapshot."""

        ts_query, preview_terms = _postgres_knowledge_ts_query(query)
        search_filter_sql, search_filter_params = _postgres_knowledge_search_filter_sql(query)
        where_sql, params = _postgres_knowledge_filter_sql(query)
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(access_scope)
        where_sql += access_sql
        params.extend(access_params)
        revision_sql, revision_params = _postgres_knowledge_revision_refs_filter_sql(revision_refs)
        where_sql += revision_sql
        params.extend(revision_params)
        frontier_sql, frontier_params = _postgres_knowledge_frontier_filter_sql(
            through_change_sequence
        )
        where_sql += frontier_sql
        params.extend(frontier_params)
        total_hits_known = await self._count_search_hits(
            cur,
            search_filter_sql,
            [*search_filter_params, *params],
            where_sql,
            metadata_only=ts_query is None,
        )
        rows = await self._search_unique_rows(
            cur,
            ts_query=ts_query,
            search_filter_sql=search_filter_sql,
            where_sql=where_sql,
            params=[*search_filter_params, *params],
            limit=query.limit,
        )
        hits, byte_truncated = await self._hits_from_search_rows(
            cur,
            rows,
            query,
            preview_terms,
        )
        return KnowledgeSearchResult(
            query=query,
            hits=hits,
            truncated=byte_truncated or len(hits) < total_hits_known,
            limit=query.limit,
            max_bytes=query.max_bytes,
            total_hits_known=total_hits_known,
        )

    @runtime_knowledge_operation("read")
    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_list_query(query)
        where_sql, params = _postgres_knowledge_list_filter_sql(query)
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(scope)
        where_sql += access_sql
        params.extend(access_params)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await _begin_knowledge_read_snapshot(cur)
            total_entries_known = await self._count_list_entries(cur, where_sql, params)
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT e.id
                    FROM cayu_knowledge_current_entries AS e
                    WHERE TRUE
                    {where_sql}
                    ORDER BY COALESCE(e.importance, 0.0) DESC,
                             e.updated_at DESC,
                             e.id ASC
                    LIMIT %s
                    """,
                ),
                [*params, query.limit],
            )
            rows = await cur.fetchall()
            entry_map = await self._load_entries(cur, [str(row[0]) for row in rows])
            entries = [entry for row in rows if (entry := entry_map.get(str(row[0]))) is not None]
            facets, facets_truncated = await self._list_facets(cur, query, where_sql, params)
            items, byte_truncated = await self._list_items(cur, entries, query)
        return KnowledgeListResult(
            query=query,
            entries=items,
            facets=facets,
            facets_truncated=facets_truncated,
            truncated=byte_truncated or len(items) < total_entries_known or facets_truncated,
            limit=query.limit,
            max_bytes=query.max_bytes,
            total_entries_known=total_entries_known,
        )

    async def _insert_entry(self, cur: Any, entry: KnowledgeEntry) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id,
                namespace,
                current_revision,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                entry.id,
                entry.namespace,
                entry.revision,
                pg_support.to_utc(entry.created_at),
                pg_support.to_utc(entry.updated_at),
            ),
        )
        await self._insert_revision(cur, entry)

    async def _insert_revision(self, cur: Any, entry: KnowledgeEntry) -> None:
        await cur.execute(
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
                metadata,
                payload_bytes
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            _knowledge_entry_row_values(entry),
        )
        if entry.labels:
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_labels (entry_id, entry_revision, key, value)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (entry.id, entry.revision, key, value)
                    for key, value in sorted(entry.labels.items())
                ],
            )
        if entry.aspects:
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_aspects (entry_id, entry_revision, aspect)
                VALUES (%s, %s, %s)
                """,
                [(entry.id, entry.revision, aspect) for aspect in entry.aspects],
            )
        if entry.impact_targets:
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_impact_targets (
                    entry_id, entry_revision, impact_target
                )
                VALUES (%s, %s, %s)
                """,
                [(entry.id, entry.revision, target) for target in entry.impact_targets],
            )

    async def _advance_current_revision(
        self,
        cur: Any,
        entry: KnowledgeEntry,
        *,
        expected_revision: int,
    ) -> None:
        await cur.execute(
            """
            UPDATE cayu_knowledge_entries
            SET current_revision = %s, updated_at = %s
            WHERE id = %s AND current_revision = %s
            """,
            (
                entry.revision,
                pg_support.to_utc(entry.updated_at),
                entry.id,
                expected_revision,
            ),
        )
        if cur.rowcount != 1:
            current = await self._load_entry(cur, entry.id)
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=None if current is None else current.revision,
            )

    async def _append_revision(
        self,
        cur: Any,
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
        current = await self._load_entry(cur, entry.id)
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
            await self._require_maintenance_replacement_mutation_allowed(
                cur,
                entry_id=current.id,
                entry_revision=current.revision,
                current_status=current.status,
                successor_status=entry.status,
                operation=operation,
            )
        _validate_revision_successor(current, entry)
        _require_knowledge_successor_access(access_scope, entry, operation=operation)
        if self._min_required_revision >= 75 and await self._has_activation_receipts(
            cur,
            entry.id,
        ):
            _require_knowledge_activation_retirement_capacity(entry)
        previous_chunks = await self._load_chunks(
            cur,
            entry.id,
            revision=current.revision,
        )
        if chunks is not None:
            copied_chunks = _copy_knowledge_entry_chunks(
                entry.id,
                entry.revision,
                chunks,
            )
        elif _knowledge_has_only_default_chunk(current, previous_chunks):
            copied_chunks = [_default_chunk_for_entry(entry)]
        else:
            copied_chunks = _copy_chunks_for_revision(previous_chunks, entry)
        if inherit_evidence:
            if evidence is not None:
                raise ValueError("Lifecycle evidence inheritance cannot accept evidence.")
            copied_evidence = _copy_evidence_for_revision(
                await self._load_evidence(cur, entry.id, revision=current.revision),
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
        await _lock_knowledge_write_identities(
            cur,
            chunk_ids=tuple(chunk.id for chunk in copied_chunks),
        )
        await _lock_knowledge_write_identities(
            cur,
            evidence_ids=tuple(item.id for item in copied_evidence),
        )
        await self._require_chunk_ids_available(
            cur,
            copied_chunks,
            access_scope=access_scope,
            operation=operation,
        )
        await self._require_evidence_ids_available(
            cur,
            copied_evidence,
            access_scope=access_scope,
            operation=operation,
        )
        await self._insert_revision(cur, entry)
        await self._insert_chunks(cur, entry, copied_chunks)
        await self._insert_evidence(cur, copied_evidence)
        await self._advance_current_revision(
            cur,
            entry,
            expected_revision=expected_revision,
        )
        await self._insert_change(
            cur,
            before_entry=current,
            after_entry=entry,
            kind=change_kind,
            operation_id=change_operation_id,
            committed_at=committed_at,
        )

    async def _require_maintenance_replacement_mutation_allowed(
        self,
        cur: Any,
        *,
        entry_id: str,
        entry_revision: int,
        current_status: KnowledgeStatus | None = None,
        successor_status: KnowledgeStatus | None = None,
        operation: str | None = None,
        preserve_history: bool = False,
    ) -> None:
        if self._min_required_revision < 67:
            return
        await cur.execute(
            "SELECT proposal.proposal_id, proposal.replacement_revision, "
            "proposal.proposal_fingerprint, decision.operation_id, "
            "decision.proposal::text, decision.decision::text, decision.receipt::text "
            "FROM cayu_knowledge_maintenance_proposals AS proposal "
            "LEFT JOIN cayu_knowledge_maintenance_decisions AS decision "
            "ON decision.proposal_id = proposal.proposal_id "
            "WHERE proposal.replacement_entry_id = %s",
            (entry_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return
        if preserve_history:
            raise KnowledgeMaintenanceConflict("maintenance_replacement_history_owned")
        proposal_id = str(row[0])
        replacement_revision = int(row[1])
        decision_operation_id = row[3]
        if decision_operation_id is None:
            raise KnowledgeMaintenanceConflict("pending_replacement_lifecycle_owned")
        try:
            proposal = KnowledgeMaintenanceProposal.model_validate_json(row[4])
            decision = KnowledgeMaintenanceDecision.model_validate_json(row[5])
            receipt = KnowledgeMaintenanceDecisionReceipt.model_validate_json(row[6])
            if (
                proposal.id != proposal_id
                or proposal.fingerprint != str(row[2])
                or proposal.replacement.entry_id != entry_id
                or proposal.replacement.revision != replacement_revision
                or decision.operation_id != str(decision_operation_id)
                or decision.proposal_id != proposal_id
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

    async def _load_publication_receipt(
        self,
        cur: Any,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgePublicationReceipt | None:
        await cur.execute(
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
                access_snapshot::text
            FROM cayu_knowledge_publication_receipts
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row[8])
            receipt = KnowledgePublicationReceipt(
                operation_id=row[0],
                entry_id=row[1],
                entry_revision=row[2],
                expected_revision=row[3],
                request_sha256=row[4],
                entry_created_at=row[5],
                entry_updated_at=row[6],
                committed_at=row[7],
            )
        except Exception:
            raise KnowledgePublicationConflict("malformed_receipt") from None
        if not _knowledge_scope_allows_snapshot(access_scope, snapshot):
            raise KnowledgeAccessDenied("publish_entry_revision")
        return receipt

    async def _load_publication_receipt_in_scope(
        self,
        cur: Any,
        operation_id: str,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgePublicationReceipt | None:
        await cur.execute(
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
                access_snapshot::text
            FROM cayu_knowledge_publication_receipts
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row[8])
            if not _knowledge_scope_allows_snapshot(access_scope, snapshot):
                return None
            return KnowledgePublicationReceipt(
                operation_id=row[0],
                entry_id=row[1],
                entry_revision=row[2],
                expected_revision=row[3],
                request_sha256=row[4],
                entry_created_at=row[5],
                entry_updated_at=row[6],
                committed_at=row[7],
            )
        except Exception:
            raise KnowledgePublicationConflict("malformed_receipt") from None

    async def _insert_publication_receipt(
        self,
        cur: Any,
        receipt: KnowledgePublicationReceipt,
        entry: KnowledgeEntry,
    ) -> None:
        await cur.execute(
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
                access_snapshot
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                receipt.operation_id,
                receipt.entry_id,
                receipt.entry_revision,
                receipt.expected_revision,
                receipt.request_sha256,
                receipt.entry_created_at,
                receipt.entry_updated_at,
                receipt.committed_at,
                _knowledge_access_snapshot_json(_knowledge_access_snapshot(entry)),
            ),
        )

    @staticmethod
    async def _has_activation_receipts(cur: Any, entry_id: str) -> bool:
        await cur.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM cayu_knowledge_activation_receipts "
            "WHERE entry_id = %s LIMIT 1)",
            (entry_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KnowledgeActivationConflict("malformed_receipt")
        return bool(row[0])

    async def _load_activation_receipt(
        self,
        cur: Any,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> KnowledgeActivationReceipt | None:
        await cur.execute(
            """
            SELECT operation_id, entry_id, entry_revision, expected_revision,
                   publication_request_sha256, committed_at,
                   receipt_json, access_snapshot::text
            FROM cayu_knowledge_activation_receipts
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row[7])
        except Exception:
            raise KnowledgeActivationConflict("malformed_receipt") from None
        cutoff = datetime.now(UTC)
        if not _knowledge_scope_allows_snapshot(access_scope, snapshot, now=cutoff):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("load_activation_receipt")
            return None
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(
            access_scope,
            now=cutoff,
        )
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                SELECT
                    EXISTS (
                        SELECT 1
                        FROM cayu_knowledge_current_entries AS e
                        WHERE e.id = %s
                    ),
                    EXISTS (
                        SELECT 1
                        FROM cayu_knowledge_current_entries AS e
                        WHERE e.id = %s
                        {access_sql}
                    )
                """,
            ),
            (row[1], row[1], *access_params),
        )
        current_access = await cur.fetchone()
        if current_access is None:
            raise KnowledgeActivationConflict("malformed_receipt")
        current_exists = bool(current_access[0])
        retirement = await self._load_activation_retirement(cur, str(row[1]))
        if current_exists:
            if retirement is not None:
                raise KnowledgeActivationConflict("malformed_retirement")
            current_allowed = bool(current_access[1])
        else:
            current_allowed = _knowledge_scope_allows_activation_receipt(
                access_scope,
                snapshot,
                None,
                retirement=retirement,
                entry_id=str(row[1]),
                entry_revision=int(row[2]),
                now=cutoff,
            )
        if not current_allowed:
            if deny_inaccessible:
                raise KnowledgeAccessDenied("load_activation_receipt")
            return None
        try:
            receipt = KnowledgeActivationReceipt.model_validate_json(row[6])
            if (
                receipt.operation_id != row[0]
                or receipt.entry_id != row[1]
                or receipt.entry_revision != row[2]
                or receipt.expected_revision != row[3]
                or receipt.publication_request_sha256 != row[4]
                or receipt.committed_at != row[5]
            ):
                raise ValueError("Activation receipt columns disagree with its JSON envelope.")
        except Exception:
            raise KnowledgeActivationConflict("malformed_receipt") from None
        return receipt

    async def _load_activation_retirement(
        self,
        cur: Any,
        entry_id: str,
    ) -> _KnowledgeActivationRetirement | None:
        await cur.execute(
            "SELECT entry_id, entry_revision, retired_at, retirement_json "
            "FROM cayu_knowledge_activation_retirements WHERE entry_id = %s",
            (entry_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            retirement = _parse_knowledge_activation_retirement_json(row[3])
            if (
                retirement.entry_id != row[0]
                or retirement.entry_revision != row[1]
                or retirement.retired_at != row[2]
            ):
                raise ValueError("Activation retirement columns disagree with its envelope.")
        except Exception:
            raise KnowledgeActivationConflict("malformed_retirement") from None
        return retirement

    async def _insert_activation_retirement(
        self,
        cur: Any,
        retirement: _KnowledgeActivationRetirement,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_activation_retirements (
                entry_id, entry_revision, retired_at, retirement_json
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                retirement.entry_id,
                retirement.entry_revision,
                retirement.retired_at,
                _knowledge_activation_retirement_json(retirement),
            ),
        )

    async def _insert_activation_receipt(
        self,
        cur: Any,
        receipt: KnowledgeActivationReceipt,
        *,
        access_entry: KnowledgeEntry,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_activation_receipts (
                operation_id,
                entry_id,
                entry_revision,
                expected_revision,
                publication_request_sha256,
                committed_at,
                receipt_json,
                access_snapshot
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                receipt.operation_id,
                receipt.entry_id,
                receipt.entry_revision,
                receipt.expected_revision,
                receipt.publication_request_sha256,
                receipt.committed_at,
                _knowledge_activation_receipt_json(receipt),
                _knowledge_access_snapshot_json(_knowledge_access_snapshot(access_entry)),
            ),
        )

    async def _insert_chunks(
        self,
        cur: Any,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> None:
        await cur.executemany(
            """
            INSERT INTO cayu_knowledge_chunks (
                id,
                entry_id,
                entry_revision,
                chunk_index,
                text,
                content_hash,
                source_uri,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [_knowledge_chunk_row_values(chunk) for chunk in chunks],
        )

    async def _insert_evidence(
        self,
        cur: Any,
        evidence: list[KnowledgeEvidence],
    ) -> None:
        if not evidence:
            return
        await cur.executemany(
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
                locator,
                disposition,
                created_at,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s::jsonb, %s, %s, %s::jsonb
            )
            """,
            [_knowledge_evidence_row_values(item) for item in evidence],
        )

    async def _load_evidence(
        self,
        cur: Any,
        entry_id: str,
        *,
        revision: int,
        limit: int | None = None,
    ) -> list[KnowledgeEvidence]:
        limit_sql = "" if limit is None else " LIMIT %s"
        params: tuple[object, ...] = (
            (entry_id, revision) if limit is None else (entry_id, revision, limit)
        )
        await cur.execute(
            cast(
                "LiteralString",
                """
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
                locator,
                disposition,
                created_at,
                metadata
            FROM cayu_knowledge_evidence
            WHERE entry_id = %s AND entry_revision = %s
            ORDER BY id COLLATE "C"
            """
                + limit_sql,
            ),
            params,
        )
        return [_knowledge_evidence_from_row(row) for row in await cur.fetchall()]

    async def _require_evidence_ids_available(
        self,
        cur: Any,
        evidence: list[KnowledgeEvidence],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = sorted({item.id for item in evidence})
        if not proposed_ids:
            return
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(access_scope)
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                WITH occupied AS (
                    SELECT DISTINCT entry_id
                    FROM cayu_knowledge_evidence
                    WHERE id = ANY(%s)
                )
                SELECT
                    occupied.entry_id,
                    EXISTS (
                        SELECT 1
                        FROM cayu_knowledge_current_entries AS e
                        WHERE e.id = occupied.entry_id
                        {access_sql}
                    ) AS authorized
                FROM occupied
                ORDER BY occupied.entry_id
                """,
            ),
            (proposed_ids, *access_params),
        )
        occupied = [(str(row[0]), bool(row[1])) for row in await cur.fetchall()]
        if any(not authorized for _, authorized in occupied):
            raise KnowledgeAccessDenied(operation)
        if occupied:
            raise KnowledgeEvidenceConflict(operation)

    async def _insert_change(
        self,
        cur: Any,
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
        await _lock_knowledge_change_sequence(cur)
        change_id = f"kchg_{uuid4().hex}"
        committed_at = datetime.now(UTC) if committed_at is None else committed_at
        before_requires_include_expired: bool | None = None
        if (
            before_entry is not None
            and before_entry.expires_at is not None
            and before_entry.expires_at <= committed_at
        ):
            await cur.execute(
                """
                SELECT audience.requires_include_expired
                FROM cayu_knowledge_changes AS change_record
                JOIN cayu_knowledge_change_audiences AS audience
                  ON audience.change_sequence = change_record.sequence
                 AND audience.audience_kind = 'after'
                WHERE change_record.entry_id = %s
                  AND change_record.entry_revision = %s
                ORDER BY change_record.sequence DESC
                LIMIT 1
                """,
                (before_entry.id, before_entry.revision),
            )
            audience_row = await cur.fetchone()
            if audience_row is not None:
                before_requires_include_expired = bool(audience_row[0])
            else:
                await cur.execute(
                    "SELECT applied_at FROM cayu_schema_migrations WHERE revision = 43"
                )
                baseline_row = await cur.fetchone()
                if baseline_row is None:
                    raise RuntimeError("Postgres knowledge outbox baseline is missing.")
                before_requires_include_expired = before_entry.expires_at <= baseline_row[0]
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_changes (
                id,
                kind,
                entry_id,
                entry_revision,
                committed_at,
                operation_id
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING sequence
            """,
            (
                change_id,
                kind.value,
                entry.id,
                entry.revision,
                pg_support.to_utc(committed_at),
                operation_id,
            ),
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Postgres did not return a knowledge change sequence.")
        sequence = int(row[0])
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
        await cur.executemany(
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                    audience.requires_include_expired,
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
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_change_labels (
                    change_sequence, audience_kind, key, value
                )
                VALUES (%s, %s, %s, %s)
                """,
                label_rows,
            )
        return change

    async def _insert_relations(
        self,
        cur: Any,
        relations: list[KnowledgeRelation],
    ) -> None:
        await cur.executemany(
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
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            [_knowledge_relation_row_values(relation) for relation in relations],
        )

    async def _insert_relation_change(
        self,
        cur: Any,
        relation: KnowledgeRelation,
        *,
        access_snapshot: _KnowledgeRelationAccessSnapshot,
        operation_id: str,
        committed_at: datetime,
    ) -> KnowledgeChange:
        await _lock_knowledge_change_sequence(cur)
        change_id = f"kchg_{uuid4().hex}"
        await cur.execute(
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
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING sequence
            """,
            (
                change_id,
                KnowledgeChangeKind.RELATION_PUBLISHED.value,
                relation.subject.entry_id,
                relation.subject.revision,
                pg_support.to_utc(committed_at),
                operation_id,
                relation.id,
            ),
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Postgres did not return a knowledge change sequence.")
        change = KnowledgeChange(
            id=change_id,
            sequence=int(row[0]),
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
        await cur.executemany(
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                    audience.requires_include_expired,
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
            await cur.executemany(
                """
                INSERT INTO cayu_knowledge_change_labels (
                    change_sequence, audience_kind, key, value
                )
                VALUES (%s, %s, %s, %s)
                """,
                labels,
            )
        return change

    async def _relation_endpoints_in_scope(
        self,
        cur: Any,
        relation: KnowledgeRelation,
        access_scope: KnowledgeAccessScope,
    ) -> bool:
        for reference in (relation.subject, relation.object):
            if (
                await self._load_entry_in_scope(
                    cur,
                    reference.entry_id,
                    access_scope,
                    revision=reference.revision,
                )
                is None
            ):
                return False
        return True

    async def _load_relation_receipt(
        self,
        cur: Any,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> KnowledgeRelationPublicationReceipt | None:
        await cur.execute(
            """
            SELECT operation_id, relation_ids, request_sha256,
                   committed_at, access_snapshots
            FROM cayu_knowledge_relation_publication_receipts
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            relation_ids = _json_list(row[1])
            raw_snapshots = _json_list(row[4])
            receipt = KnowledgeRelationPublicationReceipt(
                operation_id=str(row[0]),
                relation_ids=relation_ids,
                request_sha256=str(row[2]),
                committed_at=pg_support.to_utc(row[3]),
            )
            if len(raw_snapshots) != len(receipt.relation_ids):
                raise ValueError("Relation receipt access snapshots are malformed.")
            snapshots = [
                _parse_knowledge_relation_access_snapshot_json(pg_support._dumps(raw_snapshot))
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

    async def _insert_relation_receipt(
        self,
        cur: Any,
        receipt: KnowledgeRelationPublicationReceipt,
        *,
        access_snapshots: list[_KnowledgeRelationAccessSnapshot],
    ) -> None:
        snapshots = [
            json.loads(_knowledge_relation_access_snapshot_json(snapshot))
            for snapshot in access_snapshots
        ]
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_relation_publication_receipts (
                operation_id,
                relation_ids,
                request_sha256,
                committed_at,
                access_snapshots
            )
            VALUES (%s, %s::jsonb, %s, %s, %s::jsonb)
            """,
            (
                receipt.operation_id,
                pg_support._dumps(receipt.relation_ids),
                receipt.request_sha256,
                pg_support.to_utc(receipt.committed_at),
                pg_support._dumps(snapshots),
            ),
        )

    async def _load_maintenance_proposal_record(
        self,
        cur: Any,
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
        from cayu.knowledge.maintenance_persistence import (
            KnowledgeMaintenanceAcceptedPlan,
            KnowledgeMaintenanceProposalPublicationConflict,
            KnowledgeMaintenanceProposalPublicationReceipt,
            prepare_knowledge_maintenance_proposal_publication,
        )

        await cur.execute(
            "SELECT access_snapshot::text "
            "FROM cayu_knowledge_maintenance_proposals WHERE operation_id = %s",
            (operation_id,),
        )
        access_row = await cur.fetchone()
        if access_row is None:
            return None
        try:
            snapshot = _parse_knowledge_maintenance_access_snapshot_json(access_row[0])
        except Exception:
            raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt") from None
        if not _knowledge_scope_allows_maintenance_access_snapshot(access_scope, snapshot):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("publish_maintenance_proposal")
            return None

        await cur.execute(
            """
            SELECT proposal_id, replacement_entry_id, replacement_revision,
                   proposal_fingerprint, accepted_plan_fingerprint, request_sha256,
                   committed_at, proposal::text, accepted_plan::text, receipt::text
            FROM cayu_knowledge_maintenance_proposals
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            proposal = KnowledgeMaintenanceProposal.model_validate_json(row[7])
            accepted_plan = KnowledgeMaintenanceAcceptedPlan.model_validate_json(row[8])
            receipt = KnowledgeMaintenanceProposalPublicationReceipt.model_validate_json(row[9])
            replacement = await self._load_entry(
                cur,
                proposal.replacement.entry_id,
                revision=proposal.replacement.revision,
            )
            if replacement is None:
                raise ValueError("Published replacement is missing.")
            chunks = await self._load_chunks(
                cur,
                replacement.id,
                revision=replacement.revision,
            )
            evidence = await self._load_evidence(
                cur,
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
                or proposal.id != str(row[0])
                or proposal.replacement.entry_id != str(row[1])
                or proposal.replacement.revision != int(row[2])
                or proposal.fingerprint != str(row[3])
                or accepted_plan.fingerprint != str(row[4])
                or prepared_sha256 != str(row[5])
                or receipt.operation_id != operation_id
                or receipt.proposal_id != proposal.id
                or receipt.proposal_fingerprint != proposal.fingerprint
                or receipt.accepted_plan_fingerprint != accepted_plan.fingerprint
                or receipt.request_sha256 != prepared_sha256
                or receipt.replacement != proposal.replacement
                or receipt.committed_at != pg_support.to_utc(row[6])
                or receipt.replayed
            ):
                raise ValueError("Proposal publication indexes conflict with content.")
        except KnowledgeMaintenanceProposalPublicationConflict:
            raise
        except Exception:
            raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt") from None
        return proposal, accepted_plan, receipt, snapshot

    async def _insert_maintenance_proposal_record(
        self,
        cur: Any,
        proposal: KnowledgeMaintenanceProposal,
        accepted_plan: KnowledgeMaintenanceAcceptedPlan,
        receipt: KnowledgeMaintenanceProposalPublicationReceipt,
        *,
        access_snapshot: _KnowledgeMaintenanceAccessSnapshot,
    ) -> None:
        await cur.execute(
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
                proposal,
                accepted_plan,
                receipt,
                access_snapshot
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
            )
            """,
            (
                receipt.operation_id,
                receipt.proposal_id,
                proposal.replacement.entry_id,
                proposal.replacement.revision,
                receipt.proposal_fingerprint,
                receipt.accepted_plan_fingerprint,
                receipt.request_sha256,
                pg_support.to_utc(receipt.committed_at),
                pg_support._dumps(proposal.model_dump(mode="json")),
                pg_support._dumps(accepted_plan.model_dump(mode="json")),
                pg_support._dumps(receipt.model_dump(mode="json")),
                _knowledge_maintenance_access_snapshot_json(access_snapshot),
            ),
        )

    async def _load_maintenance_governance_route(
        self,
        cur: Any,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> KnowledgeMaintenanceGovernanceReceipt | None:
        from cayu.knowledge.maintenance_governance import (
            KnowledgeMaintenanceGovernanceDisposition,
            KnowledgeMaintenanceGovernanceReceipt,
            copy_knowledge_maintenance_governance_receipt,
        )

        await cur.execute(
            """
            SELECT proposal_id, proposal_fingerprint, request_sha256,
                   committed_at, receipt_json, access_snapshot::text
            FROM cayu_knowledge_maintenance_governance_routes
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            receipt = KnowledgeMaintenanceGovernanceReceipt.model_validate_json(row[4])
            snapshot = _parse_knowledge_maintenance_access_snapshot_json(row[5])
            if (
                receipt.operation_id != operation_id
                or receipt.proposal_id != str(row[0])
                or receipt.proposal_fingerprint != str(row[1])
                or receipt.authority.request.fingerprint != str(row[2])
                or receipt.committed_at != pg_support.to_utc(row[3])
                or receipt.replayed
                or receipt.authority.decision.disposition
                is not KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW
            ):
                raise ValueError("Governance route indexes conflict with content.")
        except KnowledgeMaintenanceConflict:
            raise
        except Exception:
            raise KnowledgeMaintenanceConflict("malformed_governance_receipt") from None
        if not _knowledge_scope_allows_maintenance_access_snapshot(access_scope, snapshot):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("record_maintenance_governance_route")
            return None
        return copy_knowledge_maintenance_governance_receipt(receipt)

    async def _load_semantic_watch_receipt(
        self,
        cur: Any,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
        deny_inaccessible: bool,
    ) -> KnowledgeSemanticWatchReceipt | None:
        from cayu.knowledge.semantic_watch import (
            KnowledgeSemanticWatchConflict,
            KnowledgeSemanticWatchReceipt,
            copy_knowledge_semantic_watch_receipt,
        )

        await cur.execute(
            """
            SELECT invocation_sha256, request_sha256, committed_at,
                   receipt_json, access_scope::text
            FROM cayu_knowledge_semantic_watch_receipts
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            receipt = KnowledgeSemanticWatchReceipt.model_validate_json(row[3])
            stored_scope = KnowledgeAccessScope.model_validate_json(row[4])
            if (
                receipt.operation_id != operation_id
                or receipt.invocation_sha256 != str(row[0])
                or receipt.request_sha256 != str(row[1])
                or receipt.committed_at != pg_support.to_utc(row[2])
                or receipt.replayed
                or receipt.authority.invocation.access_scope != stored_scope
            ):
                raise ValueError("Semantic-watch receipt indexes conflict with content.")
        except Exception:
            raise KnowledgeSemanticWatchConflict("malformed_receipt") from None
        if stored_scope != access_scope:
            if deny_inaccessible:
                raise KnowledgeAccessDenied("record_semantic_watch_outcome")
            return None
        return copy_knowledge_semantic_watch_receipt(receipt)

    async def _load_maintenance_record(
        self,
        cur: Any,
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
        await cur.execute(
            """
            SELECT proposal_id, proposal_fingerprint, request_sha256,
                   committed_at, proposal::text, decision::text,
                   receipt::text, access_snapshot::text
            FROM cayu_knowledge_maintenance_decisions
            WHERE operation_id = %s
            """,
            (operation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            proposal = KnowledgeMaintenanceProposal.model_validate_json(row[4])
            decision = KnowledgeMaintenanceDecision.model_validate_json(row[5])
            receipt = KnowledgeMaintenanceDecisionReceipt.model_validate_json(row[6])
            snapshot = _parse_knowledge_maintenance_access_snapshot_json(row[7])
            if (
                decision.operation_id != operation_id
                or receipt.operation_id != operation_id
                or proposal.id != str(row[0])
                or proposal.id != decision.proposal_id
                or proposal.id != receipt.proposal_id
                or proposal.fingerprint != str(row[1])
                or proposal.fingerprint != decision.proposal_fingerprint
                or proposal.fingerprint != receipt.proposal_fingerprint
                or receipt.request_sha256 != str(row[2])
                or receipt.committed_at != pg_support.to_utc(row[3])
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

    async def _insert_maintenance_record(
        self,
        cur: Any,
        proposal: KnowledgeMaintenanceProposal,
        decision: KnowledgeMaintenanceDecision,
        receipt: KnowledgeMaintenanceDecisionReceipt,
        *,
        access_snapshot: _KnowledgeMaintenanceAccessSnapshot,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_maintenance_decisions (
                operation_id,
                proposal_id,
                proposal_fingerprint,
                request_sha256,
                committed_at,
                proposal,
                decision,
                receipt,
                access_snapshot
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
            """,
            (
                receipt.operation_id,
                receipt.proposal_id,
                receipt.proposal_fingerprint,
                receipt.request_sha256,
                pg_support.to_utc(receipt.committed_at),
                proposal.model_dump_json(warnings=False),
                decision.model_dump_json(warnings=False),
                receipt.model_dump_json(warnings=False),
                _knowledge_maintenance_access_snapshot_json(access_snapshot),
            ),
        )

    async def _load_change(
        self,
        cur: Any,
        sequence: int,
    ) -> KnowledgeChange | None:
        await cur.execute(
            """
            SELECT id, sequence, kind, entry_id, entry_revision,
                   committed_at, operation_id, relation_id
            FROM cayu_knowledge_changes
            WHERE sequence = %s
            """,
            (sequence,),
        )
        row = await cur.fetchone()
        return None if row is None else _knowledge_change_from_row(row)

    async def _load_change_consumer(
        self,
        cur: Any,
        consumer_id: str,
        *,
        for_update: bool,
    ) -> KnowledgeChangeConsumerState | None:
        lock_sql = " FOR UPDATE" if for_update else ""
        await cur.execute(
            cast(
                "LiteralString",
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
                WHERE consumer_id = %s
                """
                + lock_sql,
            ),
            (consumer_id,),
        )
        row = await cur.fetchone()
        return None if row is None else _knowledge_change_consumer_from_row(row)

    async def _save_change_consumer(
        self,
        cur: Any,
        state: KnowledgeChangeConsumerState,
    ) -> None:
        state = copy_knowledge_change_consumer_state(state)
        await cur.execute(
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (consumer_id) DO UPDATE SET
                access_scope_sha256 = EXCLUDED.access_scope_sha256,
                cursor_sequence = EXCLUDED.cursor_sequence,
                pending_change_sequence = EXCLUDED.pending_change_sequence,
                pending_claim_id = EXCLUDED.pending_claim_id,
                pending_worker_id = EXCLUDED.pending_worker_id,
                pending_attempt = EXCLUDED.pending_attempt,
                claimed_at = EXCLUDED.claimed_at,
                lease_expires_at = EXCLUDED.lease_expires_at,
                last_acknowledged_claim_id = EXCLUDED.last_acknowledged_claim_id,
                updated_at = EXCLUDED.updated_at
            """,
            (
                state.consumer_id,
                state.access_scope_sha256,
                state.cursor_sequence,
                state.pending_change_sequence,
                state.pending_claim_id,
                state.pending_worker_id,
                state.pending_attempt,
                pg_support.to_utc_optional(state.claimed_at),
                pg_support.to_utc_optional(state.lease_expires_at),
                state.last_acknowledged_claim_id,
                pg_support.to_utc(state.updated_at),
            ),
        )

    async def _lock_or_create_change_consumer(
        self,
        cur: Any,
        consumer_id: str,
        *,
        access_scope_sha256: str,
    ) -> KnowledgeChangeConsumerState:
        state = await self._load_change_consumer(cur, consumer_id, for_update=True)
        if state is None:
            if self._clock_is_injected:
                await cur.execute(
                    """
                    INSERT INTO cayu_knowledge_change_consumers (
                        consumer_id, access_scope_sha256, updated_at
                    )
                    VALUES (%s, %s, %s)
                    ON CONFLICT (consumer_id) DO NOTHING
                    """,
                    (consumer_id, access_scope_sha256, self._clock()),
                )
            else:
                await cur.execute(
                    """
                    INSERT INTO cayu_knowledge_change_consumers (
                        consumer_id, access_scope_sha256, updated_at
                    )
                    VALUES (%s, %s, clock_timestamp())
                    ON CONFLICT (consumer_id) DO NOTHING
                    """,
                    (consumer_id, access_scope_sha256),
                )
            state = await self._load_change_consumer(cur, consumer_id, for_update=True)
            if state is None:
                raise RuntimeError("Postgres did not persist a knowledge change consumer.")
        return state

    async def _change_consumer_now(self, cur: Any) -> datetime:
        if self._clock_is_injected:
            return self._clock()
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Postgres did not return its knowledge consumer clock.")
        return pg_support.to_utc(row[0])

    async def _load_change_acknowledgement(
        self,
        cur: Any,
        consumer_id: str,
        claim_id: str,
    ) -> tuple[str, int] | None:
        await cur.execute(
            """
            SELECT claim_sha256, change_sequence
            FROM cayu_knowledge_change_acknowledgements
            WHERE consumer_id = %s AND claim_id = %s
            """,
            (consumer_id, claim_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return str(row[0]), int(row[1])

    async def _insert_change_acknowledgement(
        self,
        cur: Any,
        claim: KnowledgeChangeClaim,
        *,
        claim_sha256: str,
        acknowledged_at: datetime,
    ) -> None:
        await cur.execute(
            """
            INSERT INTO cayu_knowledge_change_acknowledgements (
                consumer_id,
                claim_id,
                claim_sha256,
                change_sequence,
                acknowledged_at
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                claim.consumer_id,
                claim.claim_id,
                claim_sha256,
                claim.change.sequence,
                acknowledged_at,
            ),
        )

    async def _require_matching_change_claim(
        self,
        cur: Any,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
    ) -> None:
        stored_change = await self._load_change(cur, claim.change.sequence)
        if (
            state.pending_change_sequence != claim.change.sequence
            or state.pending_claim_id != claim.claim_id
            or state.pending_worker_id != claim.worker_id
            or state.pending_attempt != claim.attempt
            or stored_change != claim.change
        ):
            raise KnowledgeChangeConsumerConflict("stale_claim")

    async def _require_live_change_claim(
        self,
        cur: Any,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
        *,
        now: datetime,
    ) -> None:
        await self._require_matching_change_claim(cur, state, claim)
        if state.lease_expires_at is None or state.lease_expires_at <= now:
            raise KnowledgeChangeConsumerConflict("expired_claim")

    async def _require_chunk_ids_available(
        self,
        cur: Any,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = sorted({chunk.id for chunk in chunks})
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(access_scope)
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                WITH occupied AS (
                    SELECT DISTINCT entry_id
                    FROM cayu_knowledge_chunks
                    WHERE id = ANY(%s)
                )
                SELECT
                    occupied.entry_id,
                    EXISTS (
                        SELECT 1
                        FROM cayu_knowledge_current_entries AS e
                        WHERE e.id = occupied.entry_id
                        {access_sql}
                    ) AS authorized
                FROM occupied
                ORDER BY occupied.entry_id
                """,
            ),
            (proposed_ids, *access_params),
        )
        occupied = [(str(row[0]), bool(row[1])) for row in await cur.fetchall()]
        if any(not authorized for _, authorized in occupied):
            raise KnowledgeAccessDenied(operation)
        if occupied:
            raise KnowledgeChunkConflict(operation)

    async def _load_entry(
        self,
        cur: Any,
        entry_id: str,
        *,
        revision: int | None = None,
    ) -> KnowledgeEntry | None:
        if revision is None:
            await cur.execute(
                """
            SELECT
                id,
                revision,
                namespace,
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
                metadata
            FROM cayu_knowledge_current_entries
            WHERE id = %s
                """,
                (entry_id,),
            )
        else:
            await cur.execute(
                """
                SELECT
                    logical.id,
                    revision.revision,
                    logical.namespace,
                    revision.text,
                    revision.kind,
                    revision.visibility,
                    revision.status,
                    revision.created_by_type,
                    revision.created_by,
                    revision.created_at,
                    revision.updated_at,
                    revision.source_type,
                    revision.source_uri,
                    revision.source_id,
                    revision.source_hash,
                    revision.importance,
                    revision.importance_source,
                    revision.confidence,
                    revision.last_used_at,
                    revision.expires_at,
                    revision.title,
                    revision.metadata
                FROM cayu_knowledge_entries AS logical
                JOIN cayu_knowledge_revisions AS revision
                  ON revision.entry_id = logical.id
                WHERE logical.id = %s AND revision.revision = %s
                """,
                (entry_id, revision),
            )
        row = await cur.fetchone()
        if row is None:
            return None
        selected_revision = int(row[1])
        return _knowledge_entry_from_row(
            row,
            labels=await self._load_labels(cur, entry_id, selected_revision),
            aspects=await self._load_aspects(cur, entry_id, selected_revision),
            impact_targets=await self._load_impact_targets(
                cur,
                entry_id,
                selected_revision,
            ),
        )

    async def _load_entry_at_change_sequence(
        self,
        cur: Any,
        entry_id: str,
        *,
        through_sequence: int,
    ) -> KnowledgeEntry | None:
        await cur.execute(
            """
            SELECT candidate.entry_revision
            FROM cayu_knowledge_changes AS candidate
            WHERE candidate.entry_id = %s
              AND candidate.kind <> 'relation_published'
              AND candidate.sequence <= %s
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
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return await self._load_entry(cur, entry_id, revision=int(row[0]))

    async def _load_entry_in_scope(
        self,
        cur: Any,
        entry_id: str,
        access_scope: KnowledgeAccessScope,
        *,
        revision: int | None = None,
        access_now: datetime | None = None,
    ) -> KnowledgeEntry | None:
        if access_now is None:
            access_now = datetime.now(UTC)
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(
            access_scope,
            now=access_now,
        )
        if revision is None:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT e.*
                    FROM cayu_knowledge_current_entries AS e
                    WHERE e.id = %s
                    {access_sql}
                    """,
                ),
                (entry_id, *access_params),
            )
        else:
            current_access_sql, current_access_params = _postgres_knowledge_access_scope_filter_sql(
                access_scope,
                entry_alias="current_entry",
                now=access_now,
            )
            await cur.execute(
                cast(
                    "LiteralString",
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
                            stored.metadata AS metadata
                        FROM cayu_knowledge_entries AS logical
                        JOIN cayu_knowledge_revisions AS stored
                          ON stored.entry_id = logical.id
                        WHERE logical.id = %s AND stored.revision = %s
                    ) AS e
                    JOIN cayu_knowledge_current_entries AS current_entry
                      ON current_entry.id = e.id
                    WHERE TRUE
                    {access_sql}
                    {current_access_sql}
                    """,
                ),
                (
                    entry_id,
                    revision,
                    *access_params,
                    *current_access_params,
                ),
            )
        row = await cur.fetchone()
        if row is None:
            return None
        selected_revision = int(row[1])
        return _knowledge_entry_from_row(
            row,
            labels=await self._load_labels(cur, entry_id, selected_revision),
            aspects=await self._load_aspects(cur, entry_id, selected_revision),
            impact_targets=await self._load_impact_targets(
                cur,
                entry_id,
                selected_revision,
            ),
        )

    async def _load_entry_payload_bytes_in_scope(
        self,
        cur: Any,
        entry_id: str,
        access_scope: KnowledgeAccessScope,
        *,
        revision: int | None = None,
        access_now: datetime,
    ) -> tuple[int, int] | None:
        access_sql, access_params = _postgres_knowledge_access_scope_filter_sql(
            access_scope,
            now=access_now,
        )
        if revision is None:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT e.revision, e.payload_bytes
                    FROM cayu_knowledge_current_entries AS e
                    WHERE e.id = %s
                    {access_sql}
                    """,
                ),
                (entry_id, *access_params),
            )
        else:
            current_access_sql, current_access_params = _postgres_knowledge_access_scope_filter_sql(
                access_scope,
                entry_alias="current_entry",
                now=access_now,
            )
            await cur.execute(
                cast(
                    "LiteralString",
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
                        WHERE logical.id = %s AND stored.revision = %s
                    ) AS e
                    JOIN cayu_knowledge_current_entries AS current_entry
                      ON current_entry.id = e.id
                    WHERE TRUE
                    {access_sql}
                    {current_access_sql}
                    """,
                ),
                (
                    entry_id,
                    revision,
                    *access_params,
                    *current_access_params,
                ),
            )
        row = await cur.fetchone()
        if row is None:
            return None
        return int(row[0]), int(row[1])

    async def _load_chunk(self, cur: Any, chunk_id: str) -> KnowledgeChunk | None:
        await cur.execute(
            """
            SELECT id, entry_id, entry_revision, chunk_index, text, content_hash, source_uri, metadata
            FROM cayu_knowledge_chunks
            WHERE id = %s
            """,
            (chunk_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return _knowledge_chunk_from_row(row)

    async def _load_chunks(
        self,
        cur: Any,
        entry_id: str,
        *,
        revision: int,
    ) -> list[KnowledgeChunk]:
        await cur.execute(
            """
            SELECT id, entry_id, entry_revision, chunk_index, text, content_hash, source_uri, metadata
            FROM cayu_knowledge_chunks
            WHERE entry_id = %s AND entry_revision = %s
            ORDER BY chunk_index ASC
            """,
            (entry_id, revision),
        )
        return [_knowledge_chunk_from_row(row) for row in await cur.fetchall()]

    async def _load_labels(self, cur: Any, entry_id: str, revision: int) -> dict[str, str]:
        await cur.execute(
            """
            SELECT key, value
            FROM cayu_knowledge_labels
            WHERE entry_id = %s AND entry_revision = %s
            ORDER BY key ASC
            """,
            (entry_id, revision),
        )
        return {row[0]: row[1] for row in await cur.fetchall()}

    async def _load_aspects(self, cur: Any, entry_id: str, revision: int) -> list[str]:
        await cur.execute(
            """
            SELECT aspect
            FROM cayu_knowledge_aspects
            WHERE entry_id = %s AND entry_revision = %s
            ORDER BY aspect ASC
            """,
            (entry_id, revision),
        )
        return [row[0] for row in await cur.fetchall()]

    async def _load_impact_targets(
        self,
        cur: Any,
        entry_id: str,
        revision: int,
    ) -> list[str]:
        await cur.execute(
            """
            SELECT impact_target
            FROM cayu_knowledge_impact_targets
            WHERE entry_id = %s AND entry_revision = %s
            ORDER BY impact_target ASC
            """,
            (entry_id, revision),
        )
        return [row[0] for row in await cur.fetchall()]

    async def _load_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, KnowledgeEntry]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        await cur.execute(
            """
            SELECT
                id,
                revision,
                namespace,
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
                metadata
            FROM cayu_knowledge_current_entries
            WHERE id = ANY(%s)
            """,
            (unique_ids,),
        )
        rows = await cur.fetchall()
        labels = await self._load_labels_for_entries(cur, unique_ids)
        aspects = await self._load_aspects_for_entries(cur, unique_ids)
        impact_targets = await self._load_impact_targets_for_entries(cur, unique_ids)
        return {
            row[0]: _knowledge_entry_from_row(
                row,
                labels=labels.get(row[0], {}),
                aspects=aspects.get(row[0], []),
                impact_targets=impact_targets.get(row[0], []),
            )
            for row in rows
        }

    async def _load_chunks_by_ids(
        self,
        cur: Any,
        chunk_ids: list[str],
    ) -> dict[str, KnowledgeChunk]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        await cur.execute(
            """
            SELECT id, entry_id, entry_revision, chunk_index, text, content_hash, source_uri, metadata
            FROM cayu_knowledge_chunks
            WHERE id = ANY(%s)
            """,
            (unique_ids,),
        )
        return {row[0]: _knowledge_chunk_from_row(row) for row in await cur.fetchall()}

    async def _count_chunks_by_entry(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, int]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        await cur.execute(
            """
            SELECT chunk.entry_id, COUNT(*)
            FROM cayu_knowledge_chunks AS chunk
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = chunk.entry_id
             AND logical.current_revision = chunk.entry_revision
            WHERE chunk.entry_id = ANY(%s)
            GROUP BY chunk.entry_id
            """,
            (unique_ids,),
        )
        return {row[0]: int(row[1]) for row in await cur.fetchall()}

    async def _load_labels_for_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        if not entry_ids:
            return {}
        await cur.execute(
            """
            SELECT label.entry_id, label.key, label.value
            FROM cayu_knowledge_labels AS label
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = label.entry_id
             AND logical.current_revision = label.entry_revision
            WHERE label.entry_id = ANY(%s)
            ORDER BY label.entry_id ASC, label.key ASC
            """,
            (entry_ids,),
        )
        result: dict[str, dict[str, str]] = {}
        for row in await cur.fetchall():
            result.setdefault(row[0], {})[row[1]] = row[2]
        return result

    async def _load_aspects_for_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        await cur.execute(
            """
            SELECT aspect.entry_id, aspect.aspect
            FROM cayu_knowledge_aspects AS aspect
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = aspect.entry_id
             AND logical.current_revision = aspect.entry_revision
            WHERE aspect.entry_id = ANY(%s)
            ORDER BY aspect.entry_id ASC, aspect.aspect ASC
            """,
            (entry_ids,),
        )
        result: dict[str, list[str]] = {}
        for row in await cur.fetchall():
            result.setdefault(row[0], []).append(row[1])
        return result

    async def _load_impact_targets_for_entries(
        self,
        cur: Any,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        await cur.execute(
            """
            SELECT target.entry_id, target.impact_target
            FROM cayu_knowledge_impact_targets AS target
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = target.entry_id
             AND logical.current_revision = target.entry_revision
            WHERE target.entry_id = ANY(%s)
            ORDER BY target.entry_id ASC, target.impact_target ASC
            """,
            (entry_ids,),
        )
        result: dict[str, list[str]] = {}
        for row in await cur.fetchall():
            result.setdefault(row[0], []).append(row[1])
        return result

    async def _count_search_hits(
        self,
        cur: Any,
        search_filter_sql: str,
        params: list[object],
        where_sql: str,
        *,
        metadata_only: bool,
    ) -> int:
        if metadata_only:
            await cur.execute(
                f"""
                SELECT COUNT(*)
                FROM cayu_knowledge_current_entries AS e
                WHERE {search_filter_sql}
                  AND EXISTS (
                      SELECT 1
                      FROM cayu_knowledge_chunks AS available_chunk
                      WHERE available_chunk.entry_id = e.id
                        AND available_chunk.entry_revision = e.revision
                  )
                {where_sql}
                """,
                params,
            )
            row = await cur.fetchone()
            return 0 if row is None else int(row[0])
        await cur.execute(
            f"""
            SELECT COUNT(DISTINCT e.id)
            FROM cayu_knowledge_chunks AS c
            JOIN cayu_knowledge_current_entries AS e
              ON e.id = c.entry_id AND e.revision = c.entry_revision
            WHERE {search_filter_sql}
            {where_sql}
            """,
            params,
        )
        row = await cur.fetchone()
        return 0 if row is None else int(row[0])

    async def _search_unique_rows(
        self,
        cur: Any,
        *,
        ts_query: str | None,
        search_filter_sql: str,
        where_sql: str,
        params: list[object],
        limit: int,
    ) -> list[tuple[Any, ...]]:
        if ts_query is None:
            await cur.execute(
                f"""
                SELECT
                    e.id AS entry_id,
                    available_chunk.id AS chunk_id,
                    1.0 AS score
                FROM cayu_knowledge_current_entries AS e
                JOIN LATERAL (
                    SELECT candidate_chunk.id
                    FROM cayu_knowledge_chunks AS candidate_chunk
                    WHERE candidate_chunk.entry_id = e.id
                      AND candidate_chunk.entry_revision = e.revision
                    ORDER BY candidate_chunk.chunk_index ASC,
                             candidate_chunk.id COLLATE "C" ASC
                    LIMIT 1
                ) AS available_chunk ON TRUE
                WHERE {search_filter_sql}
                {where_sql}
                ORDER BY COALESCE(e.importance, 0.0) DESC,
                         e.updated_at DESC,
                         e.id COLLATE "C" ASC
                LIMIT %s
                """,
                [*params, limit],
            )
            return list(await cur.fetchall())
        unique_rows: list[tuple[Any, ...]] = []
        seen_entry_ids: set[str] = set()
        offset = 0
        while len(unique_rows) < limit:
            score_sql = (
                "1.0"
                if ts_query is None
                else f"ts_rank_cd({_postgres_entry_search_vector_sql()}, to_tsquery('simple', %s))"
            )
            score_params: list[object] = [] if ts_query is None else [ts_query]
            await cur.execute(
                f"""
                SELECT
                    e.id AS entry_id,
                    c.id AS chunk_id,
                    {score_sql} AS score
                FROM cayu_knowledge_chunks AS c
                JOIN cayu_knowledge_current_entries AS e
                  ON e.id = c.entry_id AND e.revision = c.entry_revision
                WHERE {search_filter_sql}
                {where_sql}
                ORDER BY score DESC,
                         COALESCE(e.importance, 0.0) DESC,
                         e.updated_at DESC,
                         e.id ASC,
                         c.chunk_index ASC
                LIMIT %s OFFSET %s
                """,
                [
                    *score_params,
                    *params,
                    _KNOWLEDGE_SEARCH_PAGE_SIZE,
                    offset,
                ],
            )
            rows = await cur.fetchall()
            if not rows:
                break
            for row in rows:
                entry_id = str(row[0])
                if entry_id in seen_entry_ids:
                    continue
                seen_entry_ids.add(entry_id)
                unique_rows.append(row)
                if len(unique_rows) >= limit:
                    break
            if len(rows) < _KNOWLEDGE_SEARCH_PAGE_SIZE:
                break
            offset += _KNOWLEDGE_SEARCH_PAGE_SIZE
        return unique_rows

    async def _hits_from_search_rows(
        self,
        cur: Any,
        rows: list[tuple[Any, ...]],
        query: KnowledgeQuery,
        terms: list[str],
    ) -> tuple[list[KnowledgeHit], bool]:
        entries = await self._load_entries(cur, [str(row[0]) for row in rows])
        chunks = await self._load_chunks_by_ids(cur, [str(row[1]) for row in rows])
        hits: list[KnowledgeHit] = []
        remaining = query.max_bytes
        truncated = False
        for row in rows:
            if remaining <= 0:
                truncated = True
                break
            entry = entries.get(str(row[0]))
            chunk = chunks.get(str(row[1]))
            if entry is None or chunk is None:
                continue
            filter_only = not terms
            reason, preview_text = (
                ("exact aspect filter", chunk.text)
                if filter_only
                else _knowledge_preview_for_match(entry, chunk, terms)
            )
            preview_bytes = len(preview_text.encode("utf-8"))
            preview = _truncate_knowledge_text_to_bytes(preview_text, remaining)
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
                    score=float(row[2]),
                    score_kind=("exact_metadata" if filter_only else "postgres_full_text"),
                    rank=len(hits) + 1,
                    reason=reason,
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return hits, truncated

    async def _count_list_entries(
        self,
        cur: Any,
        where_sql: str,
        params: list[object],
    ) -> int:
        await cur.execute(
            f"""
            SELECT COUNT(*)
            FROM cayu_knowledge_current_entries AS e
            WHERE TRUE
            {where_sql}
            """,
            params,
        )
        row = await cur.fetchone()
        return 0 if row is None else int(row[0])

    async def _list_items(
        self,
        cur: Any,
        entries: list[KnowledgeEntry],
        query: KnowledgeListQuery,
    ) -> tuple[list[KnowledgeListItem], bool]:
        chunk_counts = await self._count_chunks_by_entry(cur, [entry.id for entry in entries])
        items: list[KnowledgeListItem] = []
        remaining = query.max_bytes
        truncated = False
        for entry in entries:
            if remaining <= 0:
                truncated = True
                break
            preview_source = entry.title or entry.text
            preview_bytes = len(preview_source.encode("utf-8"))
            preview = _truncate_knowledge_text_to_bytes(preview_source, remaining)
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

    async def _list_facets(
        self,
        cur: Any,
        query: KnowledgeListQuery,
        where_sql: str,
        params: list[object],
    ) -> tuple[list[KnowledgeFacet], bool]:
        if query.group_by is None:
            return [], False
        sql, facet_params = _postgres_list_facet_sql(
            query.group_by,
            where_sql,
            params,
            limit=query.limit + 1,
        )
        await cur.execute(sql, facet_params)
        rows = await cur.fetchall()
        return [
            KnowledgeFacet(
                field=query.group_by,
                key=str(row[0]) if row[0] is not None else None,
                value=str(row[1]),
                count=int(row[2]),
            )
            for row in rows[: query.limit]
        ], len(rows) > query.limit


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if type(value) is not list:
        raise TypeError("Expected a JSON array.")
    return value


def _knowledge_entry_row_values(entry: KnowledgeEntry) -> tuple[object, ...]:
    return (
        entry.id,
        entry.revision,
        entry.text,
        entry.kind,
        str(entry.visibility),
        str(entry.status),
        str(entry.created_by_type),
        entry.created_by,
        pg_support.to_utc(entry.created_at),
        pg_support.to_utc(entry.updated_at),
        entry.source_type,
        entry.source_uri,
        entry.source_id,
        entry.source_hash,
        entry.importance,
        entry.importance_source,
        entry.confidence,
        pg_support.to_utc_optional(entry.last_used_at),
        pg_support.to_utc_optional(entry.expires_at),
        entry.title,
        pg_support._dumps(entry.metadata),
        knowledge_entry_payload_bytes(entry),
    )


def _knowledge_entry_from_row(
    row: tuple[Any, ...],
    *,
    labels: dict[str, str],
    aspects: list[str],
    impact_targets: list[str],
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=row[0],
        revision=row[1],
        namespace=row[2],
        text=row[3],
        kind=row[4],
        visibility=KnowledgeVisibility(row[5]),
        status=KnowledgeStatus(row[6]),
        created_by_type=KnowledgeActorType(row[7]),
        created_by=row[8],
        created_at=pg_support.to_utc(row[9]),
        updated_at=pg_support.to_utc(row[10]),
        source_type=row[11],
        source_uri=row[12],
        source_id=row[13],
        source_hash=row[14],
        importance=row[15],
        importance_source=row[16],
        confidence=row[17],
        last_used_at=pg_support.to_utc_optional(row[18]),
        expires_at=pg_support.to_utc_optional(row[19]),
        title=row[20],
        labels=labels,
        aspects=aspects,
        impact_targets=impact_targets,
        metadata=pg_support._json_obj(row[21]),
    )


def _knowledge_chunk_row_values(chunk: KnowledgeChunk) -> tuple[object, ...]:
    return (
        chunk.id,
        chunk.entry_id,
        chunk.entry_revision,
        chunk.chunk_index,
        chunk.text,
        chunk.content_hash,
        chunk.source_uri,
        pg_support._dumps(chunk.metadata),
    )


def _knowledge_chunk_from_row(row: tuple[Any, ...]) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=row[0],
        entry_id=row[1],
        entry_revision=row[2],
        chunk_index=row[3],
        text=row[4],
        content_hash=row[5],
        source_uri=row[6],
        metadata=pg_support._json_obj(row[7]),
    )


def _knowledge_index_readiness_from_row(row: tuple[Any, ...]) -> KnowledgeIndexReadiness:
    identity = KnowledgeEmbeddingIdentity(
        entry_id=str(row[2]),
        entry_revision=int(row[3]),
        chunk_id=None if row[4] is None else str(row[4]),
        projection_type=str(row[5]),
        projection_content_hash=str(row[6]),
        embedding_model=str(row[7]),
        dimensions=int(row[8]),
        preprocessing_version=str(row[9]),
        generator=str(row[10]),
        generator_version=str(row[11]),
        index_representation_version=str(row[12]),
    )
    if _knowledge_embedding_identity_sha256(identity) != str(row[1]):
        raise RuntimeError("Postgres knowledge index readiness identity is inconsistent.")
    return KnowledgeIndexReadiness(
        sequence=int(row[0]),
        identity=identity,
        state=KnowledgeIndexState(str(row[13])),
        attempt_id=str(row[14]),
        failure_code=None if row[15] is None else str(row[15]),
        operation_id=str(row[16]),
        published_at=pg_support.to_utc(row[18]),
    )


def _knowledge_evidence_row_values(evidence: KnowledgeEvidence) -> tuple[object, ...]:
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
        pg_support._dumps(evidence.locator),
        evidence.disposition.value,
        pg_support.to_utc(evidence.created_at),
        pg_support._dumps(evidence.metadata),
    )


def _knowledge_evidence_from_row(row: tuple[Any, ...]) -> KnowledgeEvidence:
    return KnowledgeEvidence(
        id=row[0],
        entry_id=row[1],
        entry_revision=row[2],
        chunk_id=row[3],
        role=KnowledgeEvidenceRole(row[4]),
        source_type=row[5],
        source_id=row[6],
        source_uri=row[7],
        source_revision=row[8],
        source_hash=row[9],
        locator=pg_support._json_obj(row[10]),
        disposition=KnowledgeEvidenceDisposition(row[11]),
        created_at=pg_support.to_utc(row[12]),
        metadata=pg_support._json_obj(row[13]),
    )


def _postgres_relation_semantic_row_values(
    relation: KnowledgeRelation,
) -> tuple[object, ...]:
    return (
        relation.kind.value,
        relation.subject.entry_id,
        relation.subject.revision,
        relation.object.entry_id,
        relation.object.revision,
    )


def _knowledge_relation_row_values(relation: KnowledgeRelation) -> tuple[object, ...]:
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
        pg_support.to_utc(relation.created_at),
        pg_support._dumps(relation.metadata),
    )


def _knowledge_relation_from_row(row: tuple[Any, ...]) -> KnowledgeRelation:
    return KnowledgeRelation(
        id=str(row[0]),
        subject=KnowledgeRevisionRef(entry_id=str(row[1]), revision=int(row[2])),
        object=KnowledgeRevisionRef(entry_id=str(row[3]), revision=int(row[4])),
        kind=KnowledgeRelationKind(str(row[5])),
        created_by_type=KnowledgeActorType(str(row[6])),
        created_by=str(row[7]),
        policy_id=None if row[8] is None else str(row[8]),
        created_at=pg_support.to_utc(row[9]),
        metadata=pg_support._json_obj(row[10]),
    )


def _knowledge_change_from_row(row: tuple[Any, ...]) -> KnowledgeChange:
    return KnowledgeChange(
        id=row[0],
        sequence=row[1],
        kind=KnowledgeChangeKind(row[2]),
        entry_id=row[3],
        entry_revision=row[4],
        committed_at=pg_support.to_utc(row[5]),
        operation_id=row[6],
        relation_id=row[7],
    )


def _knowledge_change_consumer_from_row(
    row: tuple[Any, ...],
) -> KnowledgeChangeConsumerState:
    return KnowledgeChangeConsumerState(
        consumer_id=row[0],
        access_scope_sha256=row[1],
        cursor_sequence=row[2],
        pending_change_sequence=row[3],
        pending_claim_id=row[4],
        pending_worker_id=row[5],
        pending_attempt=row[6],
        claimed_at=pg_support.to_utc_optional(row[7]),
        lease_expires_at=pg_support.to_utc_optional(row[8]),
        last_acknowledged_claim_id=row[9],
        updated_at=pg_support.to_utc(row[10]),
    )


def _copy_knowledge_entry_chunks(
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


def _postgres_knowledge_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    return _postgres_knowledge_metadata_filter_sql(
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


def _postgres_knowledge_revision_refs_filter_sql(
    revision_refs: tuple[KnowledgeRevisionRef, ...] | None,
) -> tuple[str, list[object]]:
    if revision_refs is None:
        return "", []
    if not revision_refs:
        return " AND FALSE", []
    values = ", ".join("(%s, %s)" for _ in revision_refs)
    params: list[object] = []
    for reference in revision_refs:
        params.extend((reference.entry_id, reference.revision))
    return f" AND (e.id, e.revision) IN ({values})", params


def _postgres_knowledge_frontier_filter_sql(
    through_change_sequence: int | None,
) -> tuple[str, list[object]]:
    if through_change_sequence is None:
        return "", []
    return (
        """
        AND (
            SELECT MAX(boundary_change.sequence)
            FROM cayu_knowledge_changes AS boundary_change
            WHERE boundary_change.entry_id = e.id
              AND boundary_change.entry_revision = e.revision
              AND boundary_change.kind <> 'relation_published'
        ) <= %s
        """,
        [through_change_sequence],
    )


def _postgres_knowledge_list_filter_sql(
    query: KnowledgeListQuery,
) -> tuple[str, list[object]]:
    return _postgres_knowledge_metadata_filter_sql(
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


def _postgres_knowledge_access_scope_filter_sql(
    scope: KnowledgeAccessScope,
    *,
    entry_alias: str = "e",
    now: datetime | None = None,
) -> tuple[str, list[object]]:
    if entry_alias not in {"e", "current_entry"}:
        raise ValueError("Unsupported knowledge access-filter alias.")
    clauses: list[str] = []
    params: list[object] = []
    from cayu.knowledge.access import predicate_sql

    resource_sql, resource_params = predicate_sql(
        scope,
        postgres=True,
        table="cayu_knowledge_labels",
        correlation=f"resource_label.entry_id = {entry_alias}.id AND resource_label.entry_revision = {entry_alias}.revision",
    )
    clauses.append(resource_sql)
    params.extend(resource_params)
    if not scope.allow_all_namespaces:
        clauses.append(f"{entry_alias}.namespace = ANY(%s)")
        params.append(list(scope.allowed_namespaces))
    for key, value in scope.required_labels.items():
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_labels AS access_label
                WHERE access_label.entry_id = {entry_alias}.id
                  AND access_label.entry_revision = {entry_alias}.revision
                  AND access_label.key = %s
                  AND access_label.value = %s
            )
            """
        )
        params.extend([key, value])
    clauses.append(f"{entry_alias}.visibility = ANY(%s)")
    params.append([str(visibility) for visibility in scope.allowed_visibilities])
    clauses.append(f"{entry_alias}.status = ANY(%s)")
    params.append([str(status) for status in scope.allowed_statuses])
    if scope.allowed_source_types is not None:
        clauses.append(f"{entry_alias}.source_type = ANY(%s)")
        params.append(list(scope.allowed_source_types))
    if scope.allowed_source_ids is not None:
        clauses.append(f"{entry_alias}.source_id = ANY(%s)")
        params.append(list(scope.allowed_source_ids))
    if not scope.include_expired:
        if now is None:
            clauses.append(
                f"({entry_alias}.expires_at IS NULL OR {entry_alias}.expires_at > NOW())"
            )
        else:
            clauses.append(f"({entry_alias}.expires_at IS NULL OR {entry_alias}.expires_at > %s)")
            params.append(now)
    return " AND " + " AND ".join(clauses), params


def _postgres_relation_query_filter_sql(
    query: KnowledgeRelationQuery | KnowledgeLineageQuery,
) -> tuple[str, list[object]]:
    entry_id = query.reference.entry_id
    revision = query.reference.revision
    either = (
        "((relation.subject_entry_id = %s AND relation.subject_revision = %s) "
        "OR (relation.object_entry_id = %s AND relation.object_revision = %s))"
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
            "AND relation.subject_entry_id = %s AND relation.subject_revision = %s))"
        ]
        params = [entry_id, revision, entry_id, revision, entry_id, revision]
    else:
        clauses = [
            "((relation.kind = 'contradicts' AND "
            f"{either}) OR (relation.kind <> 'contradicts' "
            "AND relation.object_entry_id = %s AND relation.object_revision = %s))"
        ]
        params = [entry_id, revision, entry_id, revision, entry_id, revision]
    if query.kinds:
        clauses.append("relation.kind = ANY(%s)")
        params.append([kind.value for kind in query.kinds])
    return " AND " + " AND ".join(clauses), params


def _postgres_relation_access_scope_filter_sql(
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
        exact_access_sql, exact_access_params = _postgres_knowledge_access_scope_filter_sql(
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
        current_access_sql, current_access_params = _postgres_knowledge_access_scope_filter_sql(
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
                               AND boundary_change.sequence <= %s
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


def _postgres_lineage_filter_sql(query: KnowledgeLineageQuery) -> tuple[str, list[object]]:
    subject_is_anchor = "(relation.subject_entry_id = %s AND relation.subject_revision = %s)"
    counterpart_status = (
        f"CASE WHEN {subject_is_anchor} THEN object_current.status ELSE subject_current.status END"
    )
    current_relation = (
        "(relation.subject_revision = subject_current.revision "
        "AND relation.object_revision = object_current.revision)"
    )
    clauses = [f"{counterpart_status} = ANY(%s)"]
    params: list[object] = [
        query.reference.entry_id,
        query.reference.revision,
        [status.value for status in query.counterpart_statuses],
    ]
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


def _postgres_knowledge_change_access_scope_filter_sql(
    scope: KnowledgeAccessScope,
) -> tuple[str, list[object]]:
    alias = "change_record"
    audience_alias = "access_audience"
    clauses: list[str] = []
    params: list[object] = []
    from cayu.knowledge.access import predicate_sql

    resource_sql, resource_params = predicate_sql(
        scope,
        postgres=True,
        table="cayu_knowledge_change_labels",
        correlation=f"resource_label.change_sequence = {alias}.sequence AND resource_label.audience_kind = {audience_alias}.audience_kind",
    )
    clauses.append(resource_sql)
    params.extend(resource_params)
    if not scope.allow_all_namespaces:
        clauses.append(f"{audience_alias}.namespace = ANY(%s)")
        params.append(list(scope.allowed_namespaces))
    for key, value in scope.required_labels.items():
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_change_labels AS access_label
                WHERE access_label.change_sequence = {alias}.sequence
                  AND access_label.audience_kind = {audience_alias}.audience_kind
                  AND access_label.key = %s
                  AND access_label.value = %s
            )
            """
        )
        params.extend([key, value])
    clauses.append(f"{audience_alias}.visibility = ANY(%s)")
    params.append([visibility.value for visibility in scope.allowed_visibilities])
    clauses.append(f"{audience_alias}.status = ANY(%s)")
    params.append([status.value for status in scope.allowed_statuses])
    if scope.allowed_source_types is not None:
        clauses.append(f"{audience_alias}.source_type = ANY(%s)")
        params.append(list(scope.allowed_source_types))
    if scope.allowed_source_ids is not None:
        clauses.append(f"{audience_alias}.source_id = ANY(%s)")
        params.append(list(scope.allowed_source_ids))
    if not scope.include_expired:
        clauses.append(f"NOT {audience_alias}.requires_include_expired")
    audience_filter = " AND ".join(clauses)
    return (
        " AND (SELECT CASE "
        f"WHEN {alias}.kind = 'relation_published' THEN "
        "COUNT(*) = 4 "
        "AND BOOL_AND(candidate.audience_kind IN "
        "('subject_exact', 'subject_current', 'object_exact', 'object_current')) "
        "AND BOOL_AND(COALESCE(candidate.allowed, FALSE)) "
        "ELSE COALESCE(BOOL_OR(candidate.allowed), FALSE) END FROM ("
        f"SELECT {audience_alias}.audience_kind, ("
        f"{audience_filter}) AS allowed "
        "FROM cayu_knowledge_change_audiences AS access_audience "
        f"WHERE {audience_alias}.change_sequence = {alias}.sequence"
        ") AS candidate)",
        params,
    )


def _postgres_knowledge_metadata_filter_sql(
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
        clauses.append("e.namespace = %s")
        params.append(namespace)
    for key, value in labels.items():
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_labels AS label
                WHERE label.entry_id = e.id
                  AND label.entry_revision = e.revision
                  AND label.key = %s
                  AND label.value = %s
            )
            """
        )
        params.extend([key, value])
    if kinds is not None:
        if kinds:
            clauses.append("e.kind = ANY(%s)")
            params.append(kinds)
        else:
            clauses.append("FALSE")
    if statuses:
        clauses.append("e.status = ANY(%s)")
        params.append([str(status) for status in statuses])
    if visibilities is not None:
        clauses.append("e.visibility = ANY(%s)")
        params.append([str(visibility) for visibility in visibilities])
    if source_type is not None:
        clauses.append("e.source_type = %s")
        params.append(source_type)
    if source_id is not None:
        clauses.append("e.source_id = %s")
        params.append(source_id)
    if aspects:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_aspects AS aspect
                WHERE aspect.entry_id = e.id
                  AND aspect.entry_revision = e.revision
                  AND aspect.aspect = ANY(%s)
            )
            """
        )
        params.append(aspects)
    for group in aspect_groups:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_aspects AS grouped_aspect
                WHERE grouped_aspect.entry_id = e.id
                  AND grouped_aspect.entry_revision = e.revision
                  AND grouped_aspect.aspect = ANY(%s)
            )
            """
        )
        params.append(group)
    if impact_targets:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_impact_targets AS target
                WHERE target.entry_id = e.id
                  AND target.entry_revision = e.revision
                  AND target.impact_target = ANY(%s)
            )
            """
        )
        params.append(impact_targets)
    if not include_expired:
        clauses.append("(e.expires_at IS NULL OR e.expires_at > %s)")
        params.append(datetime.now(UTC))
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _postgres_knowledge_ts_query(query: KnowledgeQuery) -> tuple[str | None, list[str]]:
    any_terms = _dedupe_knowledge_search_tokens(
        [
            *_expand_knowledge_search_tokens(_tokenize_knowledge_search_text(query.text or "")),
            *(
                token
                for term in query.any_terms
                for group in _structured_knowledge_search_token_groups(term)
                for token in group
            ),
        ]
    )
    all_groups = _dedupe_knowledge_search_token_groups(
        [
            group
            for term in query.all_terms
            for group in _structured_knowledge_search_token_groups(term)
        ]
    )
    phrase_queries = [_postgres_phrase_query(phrase) for phrase in query.phrases]
    phrase_terms = _dedupe_knowledge_search_tokens(
        [term for phrase in query.phrases for term in _tokenize_knowledge_search_text(phrase)]
    )
    positive_parts: list[str] = []
    if any_terms:
        positive_parts.append("(" + " | ".join(any_terms) + ")")
    if all_groups:
        positive_parts.append(" & ".join("(" + " | ".join(group) + ")" for group in all_groups))
    if phrase_queries:
        positive_parts.append("(" + " | ".join(phrase_queries) + ")")
    if not positive_parts:
        return None, []
    ts_query = " & ".join(positive_parts)
    preview_terms = _dedupe_knowledge_search_tokens(
        [*any_terms, *(term for group in all_groups for term in group), *phrase_terms]
    )
    return ts_query, preview_terms


def _postgres_knowledge_search_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    any_terms = _dedupe_knowledge_search_tokens(
        [
            *_expand_knowledge_search_tokens(_tokenize_knowledge_search_text(query.text or "")),
            *(
                token
                for term in query.any_terms
                for group in _structured_knowledge_search_token_groups(term)
                for token in group
            ),
        ]
    )
    all_groups = _dedupe_knowledge_search_token_groups(
        [
            group
            for term in query.all_terms
            for group in _structured_knowledge_search_token_groups(term)
        ]
    )
    phrase_queries = [_postgres_phrase_query(phrase) for phrase in query.phrases]
    clauses: list[str] = []
    params: list[object] = []
    if any_terms:
        clause, clause_params = _postgres_document_match_clause("(" + " | ".join(any_terms) + ")")
        clauses.append(clause)
        params.extend(clause_params)
    for group in all_groups:
        clause, clause_params = _postgres_document_match_clause("(" + " | ".join(group) + ")")
        clauses.append(clause)
        params.extend(clause_params)
    if phrase_queries:
        phrase_clauses: list[str] = []
        for phrase_query in phrase_queries:
            clause, clause_params = _postgres_document_match_clause(phrase_query)
            phrase_clauses.append(clause)
            params.extend(clause_params)
        clauses.append("(" + " OR ".join(phrase_clauses) + ")")
    if not any_terms and not all_groups and not phrase_queries:
        none_sql, none_params = _postgres_knowledge_none_filter_sql(query)
        return cast("LiteralString", "TRUE" + none_sql), none_params
    none_sql, none_params = _postgres_knowledge_none_filter_sql(query)
    return cast("LiteralString", " AND ".join(clauses) + none_sql), [*params, *none_params]


def _postgres_knowledge_none_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    none_terms = _dedupe_knowledge_search_tokens(
        [
            token
            for term in query.none_terms
            for group in _structured_knowledge_search_token_groups(term)
            for token in group
        ]
    )
    if not none_terms:
        return "", []
    none_ts_query = "(" + " | ".join(none_terms) + ")"
    return (
        cast(
            "LiteralString",
            """
            AND NOT (
                to_tsvector('simple', COALESCE(e.title, '')) @@ to_tsquery('simple', %s)
                OR to_tsvector('simple', e.text) @@ to_tsquery('simple', %s)
                OR EXISTS (
                    SELECT 1
                    FROM cayu_knowledge_chunks AS excluded_chunk
                    WHERE excluded_chunk.entry_id = e.id
                      AND excluded_chunk.entry_revision = e.revision
                      AND to_tsvector('simple', excluded_chunk.text)
                          @@ to_tsquery('simple', %s)
                )
            )
            """,
        ),
        [none_ts_query, none_ts_query, none_ts_query],
    )


def _postgres_document_match_clause(ts_query: str) -> tuple[LiteralString, list[object]]:
    return (
        cast(
            "LiteralString",
            """
            (
                to_tsvector('simple', COALESCE(e.title, '')) @@ to_tsquery('simple', %s)
                OR to_tsvector('simple', e.text) @@ to_tsquery('simple', %s)
                OR (
                    c.text <> e.text
                    AND to_tsvector('simple', c.text) @@ to_tsquery('simple', %s)
                )
            )
            """,
        ),
        [ts_query, ts_query, ts_query],
    )


def _postgres_list_facet_sql(
    group_by: KnowledgeListGroup,
    where_sql: str,
    params: list[object],
    *,
    limit: int,
) -> tuple[LiteralString, list[object]]:
    limited_params = [*params, limit]
    if group_by is KnowledgeListGroup.KIND:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, e.kind AS value, COUNT(*) AS count
                FROM cayu_knowledge_current_entries AS e
                WHERE TRUE
                {where_sql}
                GROUP BY e.kind
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.NAMESPACE:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, e.namespace AS value, COUNT(*) AS count
                FROM cayu_knowledge_current_entries AS e
                WHERE TRUE
                {where_sql}
                GROUP BY e.namespace
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.LABEL:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT label.key AS key, label.value AS value, COUNT(DISTINCT e.id) AS count
                FROM cayu_knowledge_current_entries AS e
                JOIN cayu_knowledge_labels AS label
                  ON label.entry_id = e.id AND label.entry_revision = e.revision
                WHERE TRUE
                {where_sql}
                GROUP BY label.key, label.value
                ORDER BY count DESC, key ASC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.ASPECT:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, aspect.aspect AS value, COUNT(DISTINCT e.id) AS count
                FROM cayu_knowledge_current_entries AS e
                JOIN cayu_knowledge_aspects AS aspect
                  ON aspect.entry_id = e.id AND aspect.entry_revision = e.revision
                WHERE TRUE
                {where_sql}
                GROUP BY aspect.aspect
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.IMPACT_TARGET:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, target.impact_target AS value, COUNT(DISTINCT e.id) AS count
                FROM cayu_knowledge_current_entries AS e
                JOIN cayu_knowledge_impact_targets AS target
                  ON target.entry_id = e.id AND target.entry_revision = e.revision
                WHERE TRUE
                {where_sql}
                GROUP BY target.impact_target
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    if group_by is KnowledgeListGroup.VISIBILITY:
        return (
            cast(
                "LiteralString",
                f"""
                SELECT NULL AS key, e.visibility AS value, COUNT(*) AS count
                FROM cayu_knowledge_current_entries AS e
                WHERE TRUE
                {where_sql}
                GROUP BY e.visibility
                ORDER BY count DESC, value ASC
                LIMIT %s
                """,
            ),
            limited_params,
        )
    return (
        cast(
            "LiteralString",
            f"""
            SELECT NULL AS key, e.source_type AS value, COUNT(*) AS count
            FROM cayu_knowledge_current_entries AS e
            WHERE e.source_type IS NOT NULL
            {where_sql}
            GROUP BY e.source_type
            ORDER BY count DESC, value ASC
            LIMIT %s
            """,
        ),
        limited_params,
    )


def _structured_knowledge_search_token_groups(value: str) -> list[list[str]]:
    tokens = _tokenize_knowledge_search_text(value)
    if not tokens:
        raise ValueError("Structured knowledge search terms must contain at least one token.")
    return [_knowledge_search_token_variants(token) for token in tokens]


def _postgres_phrase_query(value: str) -> str:
    tokens = _tokenize_knowledge_search_text(value)
    if not tokens:
        raise ValueError("Structured knowledge search phrases must contain at least one token.")
    return " <-> ".join(tokens)


def _dedupe_knowledge_search_tokens(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_knowledge_search_token_groups(groups: list[list[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            result.append(group)
            seen.add(key)
    return result


def _postgres_entry_search_vector_sql() -> LiteralString:
    return cast(
        "LiteralString",
        """
        setweight(to_tsvector('simple', COALESCE(e.title, '')), 'A')
        || setweight(to_tsvector('simple', e.text), 'B')
        || to_tsvector(
               'simple',
               CASE WHEN c.text = e.text THEN '' ELSE c.text END
           )
        """,
    )


def _center_knowledge_chunk_window(
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


def _bounded_knowledge_chunks(
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
            truncated_text = _truncate_knowledge_text_to_bytes(copied.text, remaining)
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


def _knowledge_preview_for_match(
    entry: KnowledgeEntry,
    chunk: KnowledgeChunk,
    terms: list[str],
) -> tuple[str, str]:
    if entry.title is not None:
        title_terms = set(_tokenize_knowledge_search_text(entry.title))
        if any(term in title_terms for term in terms):
            return "title match", entry.title
    entry_terms = set(_tokenize_knowledge_search_text(entry.text))
    if any(term in entry_terms for term in terms):
        return "entry text match", entry.text
    return "chunk text match", chunk.text


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


def _knowledge_has_only_default_chunk(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
) -> bool:
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


def _tokenize_knowledge_search_text(text: str) -> list[str]:
    return _KNOWLEDGE_SEARCH_TOKEN_RE.findall(text.casefold())


def _expand_knowledge_search_tokens(tokens: list[str]) -> list[str]:
    return [variant for token in tokens for variant in _knowledge_search_token_variants(token)]


def _knowledge_search_token_variants(token: str) -> list[str]:
    variants = [token]
    if len(token) < 3 or not token.isalpha():
        return variants
    if token.endswith("ies") and len(token) > 4:
        variants.append(token[:-3] + "y")
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        variants.append(token[:-1])
    else:
        variants.append(_plural_knowledge_search_token(token))
    return _dedupe_knowledge_search_tokens(variants)


def _plural_knowledge_search_token(token: str) -> str:
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    return token + "s"


def _truncate_knowledge_text_to_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _validate_knowledge_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{field_name}` must be greater than or equal to 0.")


def _validate_knowledge_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 1:
        raise ValueError(f"`{field_name}` must be greater than or equal to 1.")
