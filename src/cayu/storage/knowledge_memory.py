"""Ordinary in-memory knowledge storage and backend-local list helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from cayu._clock import utc_clock
from cayu._validation import copy_label_map
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge import (
    _activation_rules,
    _maintenance_rules,
    _query_rules,
    _relation_queries,
    _retrieval_results,
    _revision_rules,
    _search_scoring,
)
from cayu.knowledge._access_rules import (
    _knowledge_change_audiences,
    _knowledge_maintenance_access_snapshot,
    _knowledge_relation_access_snapshot,
    _knowledge_relation_change_audiences,
    _knowledge_scope_allows_activation_receipt,
    _knowledge_scope_allows_change,
    _knowledge_scope_allows_entry,
    _knowledge_scope_allows_lineage_endpoint,
    _knowledge_scope_allows_maintenance_access_snapshot,
    _knowledge_scope_allows_relation_access_snapshot,
    _knowledge_scope_allows_snapshot,
    _KnowledgeChangeAudience,
    _KnowledgeMaintenanceAccessSnapshot,
    _KnowledgeRelationAccessSnapshot,
    _require_knowledge_activation_retirement_access,
    _require_knowledge_entry_access,
    _require_knowledge_successor_access,
)
from cayu.knowledge.access import runtime_knowledge_operation
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationAuthority,
    KnowledgeActivationConflict,
    KnowledgeActivationReceipt,
    KnowledgeActivationSource,
    KnowledgeReviewApproval,
    _knowledge_activation_retirement,
    _KnowledgeActivationRetirement,
    _require_knowledge_activation_retirement_capacity,
    copy_knowledge_activation_authority,
    copy_knowledge_activation_receipt,
)
from cayu.knowledge.base import KnowledgeStore
from cayu.knowledge.changes import (
    MAX_KNOWLEDGE_CHANGE_SEQUENCE,
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
    copy_knowledge_change,
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
    _bounded_knowledge_index_identity,
    _knowledge_chunk_content_hash,
    _knowledge_embedding_identity_sha256,
    _knowledge_index_readiness_update_sha256,
    _validate_knowledge_index_readiness_limit,
    _validate_knowledge_index_readiness_transition,
    _validate_knowledge_index_sequence,
    copy_knowledge_embedding_identity,
    copy_knowledge_index_readiness,
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
    KnowledgeMaintenanceStale,
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
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEntry,
    KnowledgeEntryReadLimitExceeded,
    KnowledgeEvidence,
    KnowledgeEvidenceConflict,
    KnowledgeEvidenceResult,
    KnowledgeRevisionConflict,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    _copy_entry_chunks,
    _copy_entry_evidence,
    _knowledge_entry_id,
    _knowledge_publication_operation_id,
    _knowledge_semantic_watch_identity,
    _next_knowledge_revision,
    _validate_knowledge_revision,
    _validate_nonnegative_int,
    _validate_positive_int,
    copy_knowledge_entry,
    copy_knowledge_revision_refs,
    knowledge_entry_payload_bytes,
)
from cayu.knowledge.relations import (
    KnowledgeLineageLink,
    KnowledgeLineageQuery,
    KnowledgeLineageResult,
    KnowledgeRelation,
    KnowledgeRelationConflict,
    KnowledgeRelationPublicationReceipt,
    KnowledgeRelationQuery,
    KnowledgeRelationResult,
    _knowledge_lineage_link_matches_query,
    _knowledge_relation_identity,
    _knowledge_relation_matches_query,
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
    _KnowledgeAccessSnapshot,
    copy_knowledge_access_scope,
)
from cayu.knowledge.search import (
    KnowledgeFacet,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    _knowledge_query_terms,
    _query_terms_have_positive_terms,
    copy_knowledge_list_query,
    copy_knowledge_query,
)
from cayu.storage._knowledge_closure import (
    KnowledgeClosureInventory,
    KnowledgeClosureQuery,
    copy_knowledge_closure_query,
)

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

_KNOWLEDGE_REJECTED_REPLACEMENT_RETIREMENT_TRANSITIONS = frozenset(
    {
        (KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED),
        (KnowledgeStatus.PENDING, KnowledgeStatus.DELETED),
        (KnowledgeStatus.ARCHIVED, KnowledgeStatus.DELETED),
    }
)


class InMemoryKnowledgeStore(KnowledgeStore):
    """In-memory knowledge store for tests, demos, and single-process apps."""

    resource_knowledge_access_version = 1

    async def inspect_closure_sources(self, query: KnowledgeClosureQuery) -> dict[str, object]:
        query = copy_knowledge_closure_query(query)
        inventory = KnowledgeClosureInventory(query)
        sources = frozenset(query.sources)
        source_uris = frozenset(query.source_uris)
        revisions: set[tuple[str, int]] = set()
        for entry_id, versions in self._entries.items():
            for revision, entry in versions.items():
                if (
                    type(entry) is not KnowledgeEntry
                    or (entry.source_type is not None and type(entry.source_type) is not str)
                    or (entry.source_id is not None and type(entry.source_id) is not str)
                    or (entry.source_uri is not None and type(entry.source_uri) is not str)
                ):
                    raise ValueError("Knowledge closure revision source is malformed.")
                if (entry.source_type, entry.source_id) not in sources and (
                    entry.source_type,
                    entry.source_uri,
                ) not in source_uris:
                    continue
                if inventory.add_revision(
                    entry.id,
                    entry.revision,
                    entry.source_type,
                    entry.source_id,
                    entry.source_uri,
                    entry.source_hash,
                ) != (entry_id, revision):
                    raise ValueError("Knowledge closure revision ownership conflicts.")
                revisions.add((entry_id, revision))
        for revision, records in self._evidence.items():
            for evidence in records:
                # Validate decision-bearing scalars before hashing or membership
                # lookup; do not serialize an extension-mutated model.
                if (
                    type(evidence) is not KnowledgeEvidence
                    or type(evidence.source_type) is not str
                    or (evidence.source_id is not None and type(evidence.source_id) is not str)
                    or (evidence.source_uri is not None and type(evidence.source_uri) is not str)
                ):
                    raise ValueError("Knowledge closure source evidence is malformed.")
                if (evidence.source_type, evidence.source_id) not in sources and (
                    evidence.source_type,
                    evidence.source_uri,
                ) not in source_uris:
                    continue
                if inventory.add_evidence(evidence) != revision:
                    raise ValueError("Knowledge closure revision ownership conflicts.")
                revisions.add(revision)
        self._add_closure_projections(inventory, revisions)
        for readiness in self._index_readiness:
            if (
                type(readiness) is not KnowledgeIndexReadiness
                or type(readiness.identity) is not KnowledgeEmbeddingIdentity
                or type(readiness.identity.entry_id) is not str
                or type(readiness.identity.entry_revision) is not int
            ):
                raise ValueError("Knowledge closure readiness identity is malformed.")
            if (readiness.identity.entry_id, readiness.identity.entry_revision) in revisions:
                inventory.add_readiness(readiness)
        return inventory.document()

    def _add_closure_projections(
        self, inventory: KnowledgeClosureInventory, revisions: set[tuple[str, int]]
    ) -> None:
        # Keyword-only memory has no stored embedding projections.
        return None

    def __init__(
        self,
        entries: list[KnowledgeEntry] | None = None,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._default_access_scope = (
            None if access_scope is None else copy_knowledge_access_scope(access_scope)
        )
        self._clock = utc_clock(clock)
        self._entries: dict[str, dict[int, KnowledgeEntry]] = {}
        self._entry_payload_bytes: dict[tuple[str, int], int] = {}
        self._current_revisions: dict[str, int] = {}
        self._chunks: dict[tuple[str, int], list[KnowledgeChunk]] = {}
        self._evidence: dict[tuple[str, int], list[KnowledgeEvidence]] = {}
        self._publication_receipts: dict[str, KnowledgePublicationReceipt] = {}
        self._publication_access: dict[str, _KnowledgeAccessSnapshot] = {}
        self._activation_receipts: dict[str, KnowledgeActivationReceipt] = {}
        self._activation_access: dict[str, _KnowledgeAccessSnapshot] = {}
        self._activation_entry_ids: set[str] = set()
        self._activation_retirements: dict[str, _KnowledgeActivationRetirement] = {}
        self._relations: dict[str, KnowledgeRelation] = {}
        self._relation_ids_by_endpoint: dict[tuple[str, int], set[str]] = {}
        self._relation_semantics: dict[tuple[str, str, int, str, int], str] = {}
        self._relation_publication_receipts: dict[str, KnowledgeRelationPublicationReceipt] = {}
        self._relation_publication_access: dict[
            str, tuple[_KnowledgeRelationAccessSnapshot, ...]
        ] = {}
        self._relation_change_sequences: dict[str, int] = {}
        self._revision_materialization_sequences: dict[tuple[str, int], int] = {}
        self._maintenance_proposals: dict[str, KnowledgeMaintenanceProposal] = {}
        self._maintenance_proposal_publications: dict[
            str,
            tuple[
                KnowledgeMaintenanceProposal,
                KnowledgeMaintenanceAcceptedPlan,
                KnowledgeMaintenanceProposalPublicationReceipt,
            ],
        ] = {}
        self._maintenance_proposal_operation_by_id: dict[str, str] = {}
        self._maintenance_proposal_replacement_revisions: dict[str, int] = {}
        self._maintenance_proposal_id_by_replacement_entry: dict[str, str] = {}
        self._maintenance_proposal_publication_access: dict[
            str, _KnowledgeMaintenanceAccessSnapshot
        ] = {}
        self._maintenance_decisions: dict[str, KnowledgeMaintenanceDecision] = {}
        self._maintenance_receipts: dict[str, KnowledgeMaintenanceDecisionReceipt] = {}
        self._maintenance_operation_by_proposal: dict[str, str] = {}
        self._maintenance_access: dict[str, _KnowledgeMaintenanceAccessSnapshot] = {}
        self._maintenance_governance_routes: dict[str, KnowledgeMaintenanceGovernanceReceipt] = {}
        self._maintenance_governance_route_access: dict[
            str, _KnowledgeMaintenanceAccessSnapshot
        ] = {}
        self._maintenance_governance_route_by_proposal: dict[str, str] = {}
        self._semantic_watch_receipts: dict[str, KnowledgeSemanticWatchReceipt] = {}
        self._semantic_watch_receipt_access: dict[str, KnowledgeAccessScope] = {}
        self._changes: list[KnowledgeChange] = []
        self._changes_by_sequence: dict[int, KnowledgeChange] = {}
        self._change_access: dict[int, tuple[_KnowledgeChangeAudience, ...]] = {}
        self._revision_change_expiration_access: dict[tuple[str, int], bool] = {}
        self._next_change_sequence = 1
        self._change_consumers: dict[str, KnowledgeChangeConsumerState] = {}
        self._acknowledged_change_claims: dict[tuple[str, str], tuple[str, int]] = {}
        self._index_readiness: list[KnowledgeIndexReadiness] = []
        self._index_readiness_by_identity: dict[str, KnowledgeIndexReadiness] = {}
        self._index_readiness_history_by_identity: dict[str, list[KnowledgeIndexReadiness]] = {}
        self._index_readiness_operations: dict[
            str,
            tuple[str, KnowledgeIndexReadiness],
        ] = {}
        self._next_index_readiness_sequence = 1
        if entries:
            for entry in entries:
                copied = copy_knowledge_entry(entry)
                if copied.revision != 1:
                    raise ValueError("Initial knowledge entries must be revision 1.")
                if copied.id in self._entries:
                    raise ValueError(f"Duplicate knowledge entry id {copied.id!r}.")
                payload_bytes = knowledge_entry_payload_bytes(copied)
                self._entries[copied.id] = {1: copied}
                self._entry_payload_bytes[(copied.id, 1)] = payload_bytes
                self._current_revisions[copied.id] = 1
                self._chunks[(copied.id, 1)] = [_revision_rules._default_chunk_for_entry(copied)]
                self._evidence[(copied.id, 1)] = []
                change = self._prepare_change(copied, kind=KnowledgeChangeKind.CREATED)
                self._record_change(change, before_entry=None, after_entry=copied)

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
        existing = self._current_entry(entry.id)
        if existing is not None:
            _require_knowledge_entry_access(scope, existing, operation="create_entry")
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=None,
                actual_revision=existing.revision,
            )
        retirement = self._activation_retirements.get(entry.id)
        if retirement is not None:
            _require_knowledge_activation_retirement_access(
                scope,
                retirement,
                operation="create_entry",
            )
            raise KnowledgePublicationConflict("entry_retired")
        copied_chunks = self._revision_chunks(entry, chunks)
        copied_evidence = _copy_entry_evidence(
            entry.id,
            entry.revision,
            evidence or [],
            chunks=copied_chunks,
        )
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=scope,
            operation="create_entry",
        )
        self._require_evidence_ids_available(
            copied_evidence,
            access_scope=scope,
            operation="create_entry",
        )
        payload_bytes = knowledge_entry_payload_bytes(entry)
        change = self._prepare_change(entry, kind=KnowledgeChangeKind.CREATED)
        self._entries[entry.id] = {1: entry}
        self._entry_payload_bytes[(entry.id, 1)] = payload_bytes
        self._current_revisions[entry.id] = 1
        self._chunks[(entry.id, 1)] = copied_chunks
        self._evidence[(entry.id, 1)] = copied_evidence
        self._record_change(change, before_entry=None, after_entry=entry)
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
        return self._append_revision(
            entry,
            chunks=chunks,
            evidence=evidence,
            expected_revision=expected_revision,
            access_scope=scope,
            operation="append_entry_revision",
            change_kind=KnowledgeChangeKind.REVISION_APPENDED,
            inherit_evidence=False,
        )

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
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if max_bytes is not None:
            _validate_positive_int(max_bytes, "max_bytes")
        if revision is not None:
            current = self._current_entry(clean_id)
            if current is None or not _knowledge_scope_allows_entry(scope, current):
                return None
        entry = self._entry_revision(clean_id, revision)
        if entry is None or not _knowledge_scope_allows_entry(scope, entry):
            return None
        if max_bytes is not None:
            try:
                payload_bytes = self._entry_payload_bytes[(clean_id, entry.revision)]
            except KeyError as exc:
                raise RuntimeError("Knowledge entry payload size metadata is missing.") from exc
            if payload_bytes > max_bytes:
                raise KnowledgeEntryReadLimitExceeded(
                    clean_id,
                    revision=entry.revision,
                    payload_bytes=payload_bytes,
                    max_bytes=max_bytes,
                )
        return copy_knowledge_entry(entry)

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
        entry = self._current_entry(clean_id)
        if entry is None:
            raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
        _require_knowledge_entry_access(scope, entry, operation="transition_entry_status")
        if entry.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                clean_id,
                expected_revision=expected_revision,
                actual_revision=entry.revision,
            )
        if entry.status is not from_status:
            raise ValueError(
                f"Knowledge entry {clean_id!r} is {entry.status.value!r}, "
                f"not {from_status.value!r}."
            )
        if expected_namespace is not None and entry.namespace != expected_namespace:
            raise ValueError(f"Knowledge entry {clean_id!r} does not match expected namespace.")
        for key, value in expected_labels.items():
            if entry.labels.get(key) != value:
                raise ValueError(f"Knowledge entry {clean_id!r} does not match expected labels.")
        updated = entry.model_copy(
            update={
                "revision": _next_knowledge_revision(expected_revision),
                "status": to_status,
                "updated_at": _next_updated_at(entry),
            }
        )
        return self._append_revision(
            updated,
            chunks=None,
            evidence=None,
            expected_revision=expected_revision,
            access_scope=scope,
            operation="transition_entry_status",
            change_kind=(
                KnowledgeChangeKind.TOMBSTONED
                if to_status is KnowledgeStatus.DELETED
                else KnowledgeChangeKind.STATUS_TRANSITIONED
            ),
            inherit_evidence=True,
        )

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
        clean_id = _knowledge_entry_id(entry_id)
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if type(hard) is not bool:
            raise ValueError("`hard` must be a boolean.")
        entry = self._current_entry(clean_id)
        if entry is None:
            if not hard:
                return None
            retirement = self._activation_retirements.get(clean_id)
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
            activation_operation_ids = [
                operation_id
                for operation_id, receipt in self._activation_receipts.items()
                if receipt.entry_id == clean_id
            ]
            if not activation_operation_ids:
                raise KnowledgeActivationConflict("malformed_retirement")
            for operation_id in activation_operation_ids:
                self._activation_receipts.pop(operation_id, None)
                self._activation_access.pop(operation_id, None)
            self._activation_entry_ids.discard(clean_id)
            self._activation_retirements.pop(clean_id)
            return None
        if clean_id in self._activation_retirements:
            raise KnowledgeActivationConflict("malformed_retirement")
        _require_knowledge_entry_access(scope, entry, operation="delete_entry")
        if entry.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                clean_id,
                expected_revision=expected_revision,
                actual_revision=entry.revision,
            )
        if hard:
            self._require_maintenance_replacement_history_preserved(clean_id)
            change = self._prepare_change(entry, kind=KnowledgeChangeKind.HARD_DELETED)
            self._drop_relations_for_entry(clean_id)
            activation_operation_ids = [
                operation_id
                for operation_id, receipt in self._activation_receipts.items()
                if receipt.entry_id == clean_id
            ]
            for operation_id in activation_operation_ids:
                self._activation_receipts.pop(operation_id, None)
                self._activation_access.pop(operation_id, None)
            self._activation_entry_ids.discard(clean_id)
            self._entries.pop(clean_id, None)
            self._current_revisions.pop(clean_id, None)
            for key in [key for key in self._entry_payload_bytes if key[0] == clean_id]:
                self._entry_payload_bytes.pop(key, None)
            for key in [key for key in self._chunks if key[0] == clean_id]:
                self._chunks.pop(key, None)
            for key in [key for key in self._evidence if key[0] == clean_id]:
                self._evidence.pop(key, None)
            self._record_change(change, before_entry=entry, after_entry=None)
            return copy_knowledge_entry(entry)
        return await self.transition_entry_status(
            clean_id,
            expected_revision=expected_revision,
            access_scope=scope,
            from_status=entry.status,
            to_status=KnowledgeStatus.DELETED,
        )

    @runtime_knowledge_operation("modify")
    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        scope = self._operation_access_scope(access_scope)
        cutoff = _knowledge_change_now(now)
        expired_entries = sorted(
            (
                entry
                for entry_id in self._entries
                if (entry := self._current_entry(entry_id)) is not None
                if entry.expires_at is not None
                and entry.expires_at <= cutoff
                and entry.id not in self._maintenance_proposal_replacement_revisions
                and _knowledge_scope_allows_entry(scope, entry, now=cutoff)
            ),
            key=lambda entry: entry.id,
        )
        prepared_changes = [
            (entry, self._prepare_change(entry, kind=KnowledgeChangeKind.EXPIRED))
            for entry in expired_entries
        ]
        prepared_retirements: dict[str, _KnowledgeActivationRetirement] = {}
        governed_entry_ids = {receipt.entry_id for receipt in self._activation_receipts.values()}
        for entry, change in prepared_changes:
            if entry.id not in governed_entry_ids:
                continue
            if entry.id in self._activation_retirements:
                raise KnowledgeActivationConflict("malformed_retirement")
            prepared_retirements[entry.id] = _knowledge_activation_retirement(
                entry,
                retired_at=change.committed_at,
            )
        for entry, change in prepared_changes:
            entry_id = entry.id
            self._drop_relations_for_entry(entry_id)
            self._entries.pop(entry_id, None)
            self._current_revisions.pop(entry_id, None)
            for key in [key for key in self._entry_payload_bytes if key[0] == entry_id]:
                self._entry_payload_bytes.pop(key, None)
            for key in [key for key in self._chunks if key[0] == entry_id]:
                self._chunks.pop(key, None)
            for key in [key for key in self._evidence if key[0] == entry_id]:
                self._evidence.pop(key, None)
            retirement = prepared_retirements.get(entry_id)
            if retirement is not None:
                self._activation_retirements[entry_id] = retirement
            self._record_change(change, before_entry=entry, after_entry=None)
        return len(expired_entries)

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
        existing_receipt = self._publication_receipts.get(operation_id)
        if existing_receipt is not None:
            receipt_access = self._publication_access.get(operation_id)
            if receipt_access is None:
                raise KnowledgePublicationConflict("malformed_receipt")
            if not _knowledge_scope_allows_snapshot(scope, receipt_access):
                raise KnowledgeAccessDenied("publish_entry_revision")
            existing_activation = self._load_activation_receipt_in_scope(
                operation_id,
                scope,
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
            elif existing_activation is None or not _activation_rules._activation_receipt_matches(
                existing_activation,
                authority=copied_authority,
                publication_request_sha256=request_sha256,
                publication_committed_at=existing_receipt.committed_at,
            ):
                raise KnowledgePublicationConflict("activation_mismatch")
            return copy_knowledge_publication_receipt(existing_receipt, replayed=True)
        if (
            self._load_activation_receipt_in_scope(
                operation_id,
                scope,
                deny_inaccessible=True,
            )
            is not None
        ):
            raise KnowledgePublicationConflict("operation_occupied")
        existing_entry = self._current_entry(copied_entry.id)
        actual_revision = None if existing_entry is None else existing_entry.revision
        if existing_entry is not None:
            _require_knowledge_entry_access(
                scope, existing_entry, operation="publish_entry_revision"
            )
        retirement = self._activation_retirements.get(copied_entry.id)
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
            self._require_maintenance_replacement_mutation_allowed(
                existing_entry,
                successor=copied_entry,
                operation="publish_entry_revision",
            )
            _revision_rules._validate_revision_successor(existing_entry, copied_entry)
            if copied_authority is None and copied_entry.id in self._activation_entry_ids:
                _require_knowledge_activation_retirement_capacity(copied_entry)
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=scope,
            operation="publish_entry_revision",
        )
        self._require_evidence_ids_available(
            copied_evidence,
            access_scope=scope,
            operation="publish_entry_revision",
        )
        payload_bytes = knowledge_entry_payload_bytes(copied_entry)
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
        change = self._prepare_change(
            copied_entry,
            kind=(
                KnowledgeChangeKind.CREATED
                if existing_entry is None
                else KnowledgeChangeKind.REVISION_APPENDED
            ),
            operation_id=operation_id,
            committed_at=receipt.committed_at,
        )
        self._entries.setdefault(copied_entry.id, {})[copied_entry.revision] = copied_entry
        self._entry_payload_bytes[(copied_entry.id, copied_entry.revision)] = payload_bytes
        self._chunks[(copied_entry.id, copied_entry.revision)] = copied_chunks
        self._evidence[(copied_entry.id, copied_entry.revision)] = copied_evidence
        self._current_revisions[copied_entry.id] = copied_entry.revision
        self._publication_receipts[operation_id] = receipt
        self._publication_access[operation_id] = _knowledge_access_snapshot(copied_entry)
        if activation_receipt is not None:
            self._activation_receipts[operation_id] = activation_receipt
            self._activation_access[operation_id] = _knowledge_access_snapshot(copied_entry)
            self._activation_entry_ids.add(copied_entry.id)
        self._record_change(
            change,
            before_entry=existing_entry,
            after_entry=copied_entry,
        )
        return copy_knowledge_publication_receipt(receipt)

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
        existing_record = self._maintenance_proposal_publications.get(operation_id)
        if existing_record is not None:
            snapshot = self._maintenance_proposal_publication_access.get(operation_id)
            if snapshot is None:
                raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
            if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
                raise KnowledgeAccessDenied(operation)
            stored_proposal, stored_plan, receipt = existing_record
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

        occupied_operation = self._maintenance_proposal_operation_by_id.get(copied_proposal.id)
        if occupied_operation is not None:
            snapshot = self._maintenance_proposal_publication_access.get(occupied_operation)
            if snapshot is None:
                raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
            if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
                raise KnowledgeAccessDenied(operation)
            raise KnowledgeMaintenanceProposalPublicationConflict("proposal_id_reuse")
        decided_operation = self._maintenance_operation_by_proposal.get(copied_proposal.id)
        if decided_operation is not None:
            snapshot = self._maintenance_access.get(decided_operation)
            if snapshot is None:
                raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
            if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
                raise KnowledgeAccessDenied(operation)
            raise KnowledgeMaintenanceProposalPublicationConflict("proposal_already_decided")

        current_entries = {
            source.entry_id: current
            for source in copied_proposal.sources
            if (current := self._current_entry(source.entry_id)) is not None
        }
        current_entries[copied_entry.id] = copied_entry
        replacement, sources = _maintenance_rules._require_knowledge_maintenance_current_entries(
            copied_proposal,
            current_entries,
            access_scope=scope,
            operation=operation,
        )
        _maintenance_rules._require_knowledge_maintenance_publication_boundary(replacement, sources)
        _maintenance_rules._require_knowledge_maintenance_source_evidence(copied_evidence, sources)
        occupied_entry = self._current_entry(copied_entry.id)
        if occupied_entry is not None:
            _require_knowledge_entry_access(scope, occupied_entry, operation=operation)
            raise KnowledgeMaintenanceProposalPublicationConflict("replacement_id_reuse")
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=scope,
            operation=operation,
        )
        self._require_evidence_ids_available(
            copied_evidence,
            access_scope=scope,
            operation=operation,
        )
        payload_bytes = knowledge_entry_payload_bytes(copied_entry)
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
        change = self._prepare_change(
            copied_entry,
            kind=KnowledgeChangeKind.CREATED,
            operation_id=operation_id,
            committed_at=committed_at,
        )
        snapshot = _knowledge_maintenance_access_snapshot([replacement, *sources])
        self._entries[copied_entry.id] = {copied_entry.revision: copied_entry}
        self._entry_payload_bytes[(copied_entry.id, copied_entry.revision)] = payload_bytes
        self._chunks[(copied_entry.id, copied_entry.revision)] = copied_chunks
        self._evidence[(copied_entry.id, copied_entry.revision)] = copied_evidence
        self._current_revisions[copied_entry.id] = copied_entry.revision
        self._maintenance_proposals[copied_proposal.id] = copied_proposal
        self._maintenance_proposal_replacement_revisions[copied_entry.id] = copied_entry.revision
        self._maintenance_proposal_id_by_replacement_entry[copied_entry.id] = copied_proposal.id
        self._maintenance_proposal_publications[operation_id] = (
            copied_proposal,
            copied_plan,
            receipt,
        )
        self._maintenance_proposal_operation_by_id[copied_proposal.id] = operation_id
        self._maintenance_proposal_publication_access[operation_id] = snapshot
        self._record_change(change, before_entry=None, after_entry=copied_entry)
        return copy_knowledge_maintenance_proposal_publication_receipt(receipt)

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
        operation_id = self._maintenance_proposal_operation_by_id.get(proposal_id)
        if operation_id is None:
            return None
        snapshot = self._maintenance_proposal_publication_access.get(operation_id)
        record = self._maintenance_proposal_publications.get(operation_id)
        if snapshot is None or record is None:
            raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
        if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
            return None
        proposal, accepted_plan, receipt = record
        replacement = self._entry_revision(
            proposal.replacement.entry_id,
            proposal.replacement.revision,
        )
        if replacement is None:
            raise KnowledgeMaintenanceProposalPublicationConflict("replacement_missing")
        decision_operation = self._maintenance_operation_by_proposal.get(proposal_id)
        if decision_operation is not None:
            decision_snapshot = self._maintenance_access.get(decision_operation)
            stored_proposal = self._maintenance_proposals.get(proposal_id)
            stored_decision = self._maintenance_decisions.get(decision_operation)
            stored_receipt = self._maintenance_receipts.get(decision_operation)
            try:
                if stored_proposal is None or stored_decision is None or stored_receipt is None:
                    raise ValueError("Published decision authority is incomplete.")
                if decision_snapshot != snapshot or stored_proposal != proposal:
                    raise ValueError("Published decision authority is inconsistent.")
                _validate_knowledge_maintenance_record(
                    stored_proposal,
                    stored_decision,
                    stored_receipt,
                )
            except Exception:
                raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt") from None
        decided = decision_operation is not None
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

    def _require_maintenance_replacement_mutation_allowed(
        self,
        current: KnowledgeEntry,
        *,
        successor: KnowledgeEntry,
        operation: str,
    ) -> None:
        replacement_revision = self._maintenance_proposal_replacement_revisions.get(current.id)
        if replacement_revision is None:
            return
        proposal_id = self._maintenance_proposal_id_by_replacement_entry.get(current.id)
        decision_operation = (
            None
            if proposal_id is None
            else self._maintenance_operation_by_proposal.get(proposal_id)
        )
        decision = (
            None
            if decision_operation is None
            else self._maintenance_decisions.get(decision_operation)
        )
        proposal = None if proposal_id is None else self._maintenance_proposals.get(proposal_id)
        receipt = (
            None
            if decision_operation is None
            else self._maintenance_receipts.get(decision_operation)
        )
        if decision is not None:
            if proposal is None or receipt is None:
                raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
            try:
                if (
                    proposal.replacement.entry_id != current.id
                    or proposal.replacement.revision != replacement_revision
                ):
                    raise ValueError("Maintenance proposal replacement binding is inconsistent.")
                _validate_knowledge_maintenance_record(proposal, decision, receipt)
            except Exception:
                raise KnowledgeMaintenanceConflict("malformed_proposal_publication") from None
        if decision is not None and decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE:
            if current.revision > replacement_revision:
                return
        elif decision is not None and decision.kind is KnowledgeMaintenanceDecisionKind.REJECT:
            if (
                operation in {"delete_entry", "transition_entry_status"}
                and (current.status, successor.status)
                in _KNOWLEDGE_REJECTED_REPLACEMENT_RETIREMENT_TRANSITIONS
            ):
                return
            raise KnowledgeMaintenanceConflict("rejected_replacement_lifecycle_owned")
        raise KnowledgeMaintenanceConflict("pending_replacement_lifecycle_owned")

    def _require_maintenance_replacement_history_preserved(self, entry_id: str) -> None:
        if entry_id in self._maintenance_proposal_replacement_revisions:
            raise KnowledgeMaintenanceConflict("maintenance_replacement_history_owned")

    def _append_revision(
        self,
        entry: KnowledgeEntry,
        *,
        chunks: list[KnowledgeChunk] | None,
        evidence: list[KnowledgeEvidence] | None,
        expected_revision: int,
        access_scope: KnowledgeAccessScope,
        operation: str,
        change_kind: KnowledgeChangeKind,
        inherit_evidence: bool,
    ) -> KnowledgeEntry:
        _validate_revision_append(entry, expected_revision=expected_revision)
        current = self._current_entry(entry.id)
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
        self._require_maintenance_replacement_mutation_allowed(
            current,
            successor=entry,
            operation=operation,
        )
        _revision_rules._validate_revision_successor(current, entry)
        _require_knowledge_successor_access(access_scope, entry, operation=operation)
        if entry.id in self._activation_entry_ids:
            _require_knowledge_activation_retirement_capacity(entry)
        previous_chunks = self._chunks.get((current.id, current.revision), [])
        copied_chunks = self._revision_chunks(entry, chunks, previous=current)
        if inherit_evidence:
            if evidence is not None:
                raise ValueError("Lifecycle evidence inheritance cannot accept evidence.")
            copied_evidence = _revision_rules._copy_evidence_for_revision(
                self._evidence.get((current.id, current.revision), []),
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
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=access_scope,
            operation=operation,
        )
        self._require_evidence_ids_available(
            copied_evidence,
            access_scope=access_scope,
            operation=operation,
        )
        payload_bytes = knowledge_entry_payload_bytes(entry)
        change = self._prepare_change(entry, kind=change_kind)
        self._entries[entry.id][entry.revision] = entry
        self._entry_payload_bytes[(entry.id, entry.revision)] = payload_bytes
        self._chunks[(entry.id, entry.revision)] = copied_chunks
        self._evidence[(entry.id, entry.revision)] = copied_evidence
        self._current_revisions[entry.id] = entry.revision
        self._record_change(change, before_entry=current, after_entry=entry)
        return copy_knowledge_entry(entry)

    def _require_evidence_ids_available(
        self,
        evidence: list[KnowledgeEvidence],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = {item.id for item in evidence}
        occupied_entry_ids = {
            entry_id
            for (entry_id, _), stored in self._evidence.items()
            if any(item.id in proposed_ids for item in stored)
        }
        for occupied_entry_id in sorted(occupied_entry_ids):
            owner = self._current_entry(occupied_entry_id)
            if owner is None:
                raise KnowledgeEvidenceConflict(operation)
            _require_knowledge_entry_access(
                access_scope,
                owner,
                operation=operation,
            )
        if occupied_entry_ids:
            raise KnowledgeEvidenceConflict(operation)

    def _prepare_change(
        self,
        entry: KnowledgeEntry,
        *,
        kind: KnowledgeChangeKind,
        operation_id: str | None = None,
        committed_at: datetime | None = None,
    ) -> KnowledgeChange:
        sequence = self._next_change_sequence
        if sequence > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
            raise RuntimeError("Knowledge change sequence is exhausted.")
        self._next_change_sequence += 1
        return KnowledgeChange(
            id=f"kchg_{uuid4().hex}",
            sequence=sequence,
            kind=kind,
            entry_id=entry.id,
            entry_revision=entry.revision,
            committed_at=(datetime.now(UTC) if committed_at is None else committed_at),
            operation_id=operation_id,
        )

    def _record_change(
        self,
        change: KnowledgeChange,
        *,
        before_entry: KnowledgeEntry | None,
        after_entry: KnowledgeEntry | None,
    ) -> None:
        copied = copy_knowledge_change(change)
        self._changes.append(copied)
        self._changes_by_sequence[copied.sequence] = copied
        if after_entry is not None and copied.kind is not KnowledgeChangeKind.RELATION_PUBLISHED:
            self._revision_materialization_sequences[(after_entry.id, after_entry.revision)] = (
                copied.sequence
            )
        self._change_access[change.sequence] = _knowledge_change_audiences(
            copied,
            before_entry=before_entry,
            after_entry=after_entry,
            before_requires_include_expired=(
                None
                if before_entry is None
                else self._revision_change_expiration_access.get(
                    (before_entry.id, before_entry.revision)
                )
            ),
        )
        if after_entry is not None:
            after_audience = next(
                audience
                for audience in self._change_access[change.sequence]
                if audience.kind == "after"
            )
            self._revision_change_expiration_access[(after_entry.id, after_entry.revision)] = (
                after_audience.requires_include_expired
            )

    def _change_by_sequence(self, sequence: int) -> KnowledgeChange | None:
        return self._changes_by_sequence.get(sequence)

    def _accessible_changes(
        self,
        scope: KnowledgeAccessScope,
        *,
        after_sequence: int,
        limit: int | None = None,
    ) -> list[KnowledgeChange]:
        result: list[KnowledgeChange] = []
        for change in self._changes:
            audiences = self._change_access.get(change.sequence, ())
            authorized = _knowledge_scope_allows_change(scope, change, audiences)
            if change.sequence <= after_sequence or not authorized:
                continue
            result.append(change)
            if limit is not None and len(result) >= limit:
                break
        return result

    def _require_chunk_ids_available(
        self,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = {chunk.id for chunk in chunks}
        occupied_entry_ids: set[str] = set()
        for (existing_entry_id, _), existing_chunks in self._chunks.items():
            if any(chunk.id in proposed_ids for chunk in existing_chunks):
                occupied_entry_ids.add(existing_entry_id)
        for occupied_entry_id in sorted(occupied_entry_ids):
            owner = self._current_entry(occupied_entry_id)
            if owner is None:
                raise KnowledgeChunkConflict(operation)
            _require_knowledge_entry_access(
                access_scope,
                owner,
                operation=operation,
            )
        if occupied_entry_ids:
            raise KnowledgeChunkConflict(operation)

    def _current_entry(self, entry_id: str) -> KnowledgeEntry | None:
        revision = self._current_revisions.get(entry_id)
        if revision is None:
            return None
        return self._entries[entry_id][revision]

    def _entry_at_change_sequence(
        self,
        entry_id: str,
        through_sequence: int,
    ) -> KnowledgeEntry | None:
        """Return the logical current entry at one captured change sequence."""

        candidates = (
            (sequence, entry)
            for revision, entry in self._entries.get(entry_id, {}).items()
            if (sequence := self._revision_materialization_sequences.get((entry_id, revision)))
            is not None
            and sequence <= through_sequence
        )
        return max(candidates, key=lambda item: item[0], default=(0, None))[1]

    def _drop_relations_for_entry(self, entry_id: str) -> None:
        relation_ids: set[str] = set()
        for revision in self._entries.get(entry_id, {}):
            relation_ids.update(self._relation_ids_by_endpoint.get((entry_id, revision), ()))
        for relation_id in sorted(relation_ids):
            self._drop_relation(relation_id)

    def _index_relation(self, relation: KnowledgeRelation) -> None:
        for endpoint in (relation.subject, relation.object):
            key = (endpoint.entry_id, endpoint.revision)
            self._relation_ids_by_endpoint.setdefault(key, set()).add(relation.id)

    def _drop_relation(self, relation_id: str) -> None:
        relation = self._relations[relation_id]
        indexed_endpoints: list[tuple[tuple[str, int], set[str]]] = []
        for endpoint in (relation.subject, relation.object):
            key = (endpoint.entry_id, endpoint.revision)
            indexed = self._relation_ids_by_endpoint.get(key)
            if indexed is None or relation_id not in indexed:
                raise RuntimeError("In-memory knowledge relation endpoint index is inconsistent.")
            indexed_endpoints.append((key, indexed))

        self._relations.pop(relation_id)
        self._relation_semantics.pop(_knowledge_relation_semantic_key(relation), None)
        for key, indexed in indexed_endpoints:
            indexed.remove(relation_id)
            if not indexed:
                self._relation_ids_by_endpoint.pop(key)

    def _entry_revision(
        self,
        entry_id: str,
        revision: int | None,
    ) -> KnowledgeEntry | None:
        selected_revision = self._current_revisions.get(entry_id) if revision is None else revision
        if selected_revision is None:
            return None
        return self._entries.get(entry_id, {}).get(selected_revision)

    def _revision_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None,
        *,
        previous: KnowledgeEntry | None = None,
    ) -> list[KnowledgeChunk]:
        if chunks is not None:
            return _copy_entry_chunks(entry.id, entry.revision, chunks)
        if previous is None:
            return [_revision_rules._default_chunk_for_entry(entry)]
        previous_chunks = self._chunks.get((previous.id, previous.revision), [])
        if _revision_rules._has_only_default_chunk(previous, previous_chunks):
            return [_revision_rules._default_chunk_for_entry(entry)]
        return _revision_rules._copy_chunks_for_revision(previous_chunks, entry)

    def _load_activation_receipt_in_scope(
        self,
        operation_id: str,
        scope: KnowledgeAccessScope,
        *,
        deny_inaccessible: bool,
    ) -> KnowledgeActivationReceipt | None:
        receipt = self._activation_receipts.get(operation_id)
        if receipt is None:
            return None
        snapshot = self._activation_access.get(operation_id)
        if snapshot is None:
            raise KnowledgeActivationConflict("malformed_receipt")
        cutoff = datetime.now(UTC)
        if not _knowledge_scope_allows_snapshot(scope, snapshot, now=cutoff):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("load_activation_receipt")
            return None
        current = self._current_entry(receipt.entry_id)
        retirement = self._activation_retirements.get(receipt.entry_id)
        if not _knowledge_scope_allows_activation_receipt(
            scope,
            snapshot,
            current,
            retirement=retirement,
            entry_id=receipt.entry_id,
            entry_revision=receipt.entry_revision,
            now=cutoff,
        ):
            if deny_inaccessible:
                raise KnowledgeAccessDenied("load_activation_receipt")
            return None
        return receipt

    @runtime_knowledge_operation("read")
    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        receipt = self._publication_receipts.get(operation_id)
        if receipt is None:
            return None
        snapshot = self._publication_access.get(operation_id)
        if snapshot is None or not _knowledge_scope_allows_snapshot(scope, snapshot):
            return None
        return copy_knowledge_publication_receipt(receipt)

    @runtime_knowledge_operation("read")
    async def load_activation_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeActivationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        receipt = self._load_activation_receipt_in_scope(
            operation_id,
            scope,
            deny_inaccessible=False,
        )
        if receipt is None:
            return None
        return copy_knowledge_activation_receipt(receipt)

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
        _activation_rules._validate_review_approval_authority(authority, access_scope=scope)
        expected_namespace = (
            require_clean_nonblank(expected_namespace, "expected_namespace")
            if expected_namespace is not None
            else None
        )
        expected_labels = copy_label_map(expected_labels or {}, "expected_labels")
        existing_receipt = self._load_activation_receipt_in_scope(
            request.operation_id,
            scope,
            deny_inaccessible=True,
        )
        if existing_receipt is not None:
            publication = self._publication_receipts.get(request.operation_id)
            publication_snapshot = self._publication_access.get(request.operation_id)
            if publication is None or publication_snapshot is None:
                raise KnowledgeActivationConflict("malformed_receipt")
            if not _knowledge_scope_allows_snapshot(scope, publication_snapshot):
                raise KnowledgeAccessDenied("approve_pending_entry")
            approval = _activation_rules._replay_review_approval_from_receipts(
                publication,
                existing_receipt,
                authority=authority,
            )
            if approval is None:
                raise KnowledgeActivationConflict("operation_mismatch")
            _activation_rules._validate_review_approval_scope(
                approval.entry,
                expected_namespace=expected_namespace,
                expected_labels=expected_labels,
            )
            return approval
        if request.operation_id in self._publication_receipts:
            raise KnowledgeActivationConflict("operation_occupied")
        current = self._current_entry(request.candidate_entry.id)
        if current is None:
            raise KeyError(f"Knowledge entry {request.candidate_entry.id!r} does not exist.")
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
        _activation_rules._validate_review_approval_scope(
            current,
            expected_namespace=expected_namespace,
            expected_labels=expected_labels,
        )
        current_chunks = self._chunks.get((current.id, current.revision), [])
        current_evidence = self._evidence.get((current.id, current.revision), [])
        if list(request.chunks) != current_chunks or list(request.evidence) != current_evidence:
            raise KnowledgeActivationConflict("candidate_material_mismatch")
        activated = current.model_copy(
            update={
                "revision": request.target_revision,
                "status": KnowledgeStatus.ACTIVE,
                "updated_at": _next_updated_at(current),
            }
        )
        _require_knowledge_activation_retirement_capacity(activated)
        self._require_maintenance_replacement_mutation_allowed(
            current,
            successor=activated,
            operation="approve_pending_entry",
        )
        _require_knowledge_successor_access(
            scope,
            activated,
            operation="approve_pending_entry",
        )
        target_chunks = self._revision_chunks(activated, None, previous=current)
        target_evidence = _revision_rules._copy_evidence_for_revision(
            current_evidence,
            entry=activated,
            previous_chunks=current_chunks,
            chunks=target_chunks,
        )
        self._require_chunk_ids_available(
            target_chunks,
            access_scope=scope,
            operation="approve_pending_entry",
        )
        self._require_evidence_ids_available(
            target_evidence,
            access_scope=scope,
            operation="approve_pending_entry",
        )
        committed_at = datetime.now(UTC)
        publication_receipt, receipt = _activation_rules._prepare_review_approval_receipts(
            current,
            activated,
            target_chunks,
            target_evidence,
            authority,
            committed_at=committed_at,
        )
        change = self._prepare_change(
            activated,
            kind=KnowledgeChangeKind.STATUS_TRANSITIONED,
            operation_id=request.operation_id,
            committed_at=committed_at,
        )
        self._entries[activated.id][activated.revision] = activated
        self._entry_payload_bytes[(activated.id, activated.revision)] = (
            knowledge_entry_payload_bytes(activated)
        )
        self._chunks[(activated.id, activated.revision)] = target_chunks
        self._evidence[(activated.id, activated.revision)] = target_evidence
        self._current_revisions[activated.id] = activated.revision
        self._publication_receipts[request.operation_id] = publication_receipt
        self._publication_access[request.operation_id] = _knowledge_access_snapshot(activated)
        self._activation_receipts[request.operation_id] = receipt
        self._activation_access[request.operation_id] = _knowledge_access_snapshot(activated)
        self._activation_entry_ids.add(activated.id)
        self._record_change(change, before_entry=current, after_entry=activated)
        return KnowledgeReviewApproval(
            entry=copy_knowledge_entry(activated),
            receipt=copy_knowledge_activation_receipt(receipt),
        )

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
        existing_receipt = self._relation_publication_receipts.get(operation_id)
        if existing_receipt is not None:
            snapshots = self._relation_publication_access.get(operation_id)
            if snapshots is None or not all(
                _knowledge_scope_allows_relation_access_snapshot(scope, snapshot)
                for snapshot in snapshots
            ):
                raise KnowledgeAccessDenied("publish_relations")
            _validate_knowledge_relation_publication_replay(
                existing_receipt,
                relations=copied_relations,
                request_sha256=request_sha256,
            )
            return copy_knowledge_relation_publication_receipt(
                existing_receipt,
                replayed=True,
            )

        endpoint_access: list[_KnowledgeRelationAccessSnapshot] = []
        for relation in copied_relations:
            subject_exact, subject_current, object_exact, object_current = (
                self._require_relation_endpoints(
                    relation,
                    scope,
                    operation="publish_relations",
                )
            )
            endpoint_access.append(
                _knowledge_relation_access_snapshot(
                    subject_exact=subject_exact,
                    subject_current=subject_current,
                    object_exact=object_exact,
                    object_current=object_current,
                )
            )
            occupied_id = self._relations.get(relation.id)
            occupied_semantic_id = self._relation_semantics.get(
                _knowledge_relation_semantic_key(relation)
            )
            if occupied_id is None:
                historic_sequence = self._relation_change_sequences.get(relation.id)
                if historic_sequence is not None:
                    historic_change = self._changes_by_sequence[historic_sequence]
                    audiences = self._change_access.get(historic_sequence, ())
                    if not _knowledge_scope_allows_change(
                        scope,
                        historic_change,
                        audiences,
                    ):
                        raise KnowledgeAccessDenied("publish_relations")
                    raise KnowledgeRelationConflict("relation_exists")
            for occupied_relation_id in (
                relation.id if occupied_id else None,
                occupied_semantic_id,
            ):
                if occupied_relation_id is None:
                    continue
                occupied = self._relations.get(occupied_relation_id)
                if occupied is not None:
                    self._require_relation_endpoints(
                        occupied,
                        scope,
                        operation="publish_relations",
                    )
                raise KnowledgeRelationConflict("relation_exists")

        if self._next_change_sequence + len(copied_relations) - 1 > (MAX_KNOWLEDGE_CHANGE_SEQUENCE):
            raise RuntimeError("Knowledge change sequence is exhausted.")
        committed_at = self._clock()
        receipt = KnowledgeRelationPublicationReceipt(
            operation_id=operation_id,
            relation_ids=[relation.id for relation in copied_relations],
            request_sha256=request_sha256,
            committed_at=committed_at,
        )
        prepared_changes = []
        for index, (relation, access_snapshot) in enumerate(
            zip(copied_relations, endpoint_access, strict=True)
        ):
            change = copy_knowledge_change(
                self._prepare_relation_change(
                    relation,
                    sequence=self._next_change_sequence + index,
                    operation_id=operation_id,
                    committed_at=committed_at,
                )
            )
            prepared_changes.append(
                (
                    change,
                    _knowledge_relation_change_audiences(
                        change,
                        access_snapshot=access_snapshot,
                    ),
                )
            )
        publication_access = tuple(snapshot.model_copy(deep=True) for snapshot in endpoint_access)
        for relation, (change, audiences) in zip(
            copied_relations,
            prepared_changes,
            strict=True,
        ):
            self._relations[relation.id] = relation
            self._index_relation(relation)
            self._relation_semantics[_knowledge_relation_semantic_key(relation)] = relation.id
            self._changes.append(change)
            self._changes_by_sequence[change.sequence] = change
            self._change_access[change.sequence] = audiences
            assert change.relation_id is not None
            self._relation_change_sequences[change.relation_id] = change.sequence
        self._next_change_sequence += len(prepared_changes)
        self._relation_publication_receipts[operation_id] = receipt
        self._relation_publication_access[operation_id] = publication_access
        return copy_knowledge_relation_publication_receipt(receipt)

    @runtime_knowledge_operation("read")
    async def load_relation_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationPublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_relation_identity(operation_id, "operation_id")
        receipt = self._relation_publication_receipts.get(operation_id)
        if receipt is None:
            return None
        snapshots = self._relation_publication_access.get(operation_id)
        if snapshots is None or not all(
            _knowledge_scope_allows_relation_access_snapshot(scope, snapshot)
            for snapshot in snapshots
        ):
            return None
        return copy_knowledge_relation_publication_receipt(receipt)

    @runtime_knowledge_operation("read")
    async def read_relations(
        self,
        query: KnowledgeRelationQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationResult | None:
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_relation_query(query)
        fingerprint = _relation_queries._knowledge_relation_query_fingerprint(query, scope)
        cursor = _relation_queries._decode_knowledge_relation_cursor(
            query.cursor, fingerprint=fingerprint
        )
        reference_entry = self._entry_revision(
            query.reference.entry_id,
            query.reference.revision,
        )
        reference_current = self._current_entry(query.reference.entry_id)
        if (
            reference_entry is None
            or reference_current is None
            or not _knowledge_scope_allows_entry(scope, reference_entry)
            or not _knowledge_scope_allows_entry(scope, reference_current)
        ):
            return None
        candidates: list[KnowledgeRelation] = []
        endpoint = (query.reference.entry_id, query.reference.revision)
        for relation_id in self._relation_ids_by_endpoint.get(endpoint, ()):
            relation = self._relations.get(relation_id)
            if relation is None:
                raise RuntimeError("In-memory knowledge relation endpoint index is inconsistent.")
            if query.kinds and relation.kind not in query.kinds:
                continue
            if not _knowledge_relation_matches_query(relation, query):
                continue
            if cursor is not None and (relation.created_at, relation.id) <= (
                cursor.created_at,
                cursor.relation_id,
            ):
                continue
            try:
                self._require_relation_endpoints(
                    relation,
                    scope,
                    operation="read_relations",
                )
            except (KnowledgeAccessDenied, KnowledgeRelationConflict):
                continue
            candidates.append(relation)
        candidates.sort(key=lambda item: (item.created_at, item.id))
        return _relation_queries._bounded_knowledge_relation_result(
            query,
            candidates,
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
        fingerprint = _relation_queries._knowledge_lineage_query_fingerprint(
            query,
            scope,
            through_change_sequence=through_sequence,
        )
        cursor = _relation_queries._decode_knowledge_lineage_cursor(
            query.cursor, fingerprint=fingerprint
        )
        access_now = datetime.now(UTC)
        reference_exact = self._entry_revision(
            query.reference.entry_id,
            query.reference.revision,
        )
        reference_live = self._current_entry(query.reference.entry_id)
        reference_current = (
            reference_live
            if through_sequence is None
            else self._entry_at_change_sequence(
                query.reference.entry_id,
                through_sequence,
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
        candidates: list[KnowledgeLineageLink] = []
        endpoint = (query.reference.entry_id, query.reference.revision)
        for relation_id in self._relation_ids_by_endpoint.get(endpoint, ()):
            if through_sequence is not None and (
                self._relation_change_sequences.get(relation_id) is None
                or self._relation_change_sequences[relation_id] > through_sequence
            ):
                continue
            relation = self._relations.get(relation_id)
            if relation is None:
                raise RuntimeError("In-memory knowledge relation endpoint index is inconsistent.")
            if not _relation_queries._knowledge_relation_matches_lineage_query(relation, query):
                continue
            if cursor is not None and (relation.created_at, relation.id) <= (
                cursor.created_at,
                cursor.relation_id,
            ):
                continue
            subject_exact = self._entry_revision(
                relation.subject.entry_id,
                relation.subject.revision,
            )
            subject_live = self._current_entry(relation.subject.entry_id)
            subject_current = (
                subject_live
                if through_sequence is None
                else self._entry_at_change_sequence(
                    relation.subject.entry_id,
                    through_sequence,
                )
            )
            object_exact = self._entry_revision(
                relation.object.entry_id,
                relation.object.revision,
            )
            object_live = self._current_entry(relation.object.entry_id)
            object_current = (
                object_live
                if through_sequence is None
                else self._entry_at_change_sequence(
                    relation.object.entry_id,
                    through_sequence,
                )
            )
            if any(
                item is None
                for item in (
                    subject_exact,
                    subject_live,
                    subject_current,
                    object_exact,
                    object_live,
                    object_current,
                )
            ):
                raise RuntimeError("In-memory knowledge relation endpoint is missing.")
            assert subject_exact is not None
            assert subject_live is not None
            assert subject_current is not None
            assert object_exact is not None
            assert object_live is not None
            assert object_current is not None
            if (
                not _knowledge_scope_allows_lineage_endpoint(
                    scope,
                    subject_exact,
                    subject_live,
                    now=access_now,
                )
                or not _knowledge_scope_allows_lineage_endpoint(
                    scope,
                    subject_exact,
                    subject_current,
                    now=access_now,
                )
                or not _knowledge_scope_allows_lineage_endpoint(
                    scope,
                    object_exact,
                    object_live,
                    now=access_now,
                )
                or not _knowledge_scope_allows_lineage_endpoint(
                    scope,
                    object_exact,
                    object_current,
                    now=access_now,
                )
            ):
                continue
            link = _relation_queries._knowledge_lineage_link(
                relation_id=relation.id,
                kind=relation.kind,
                subject=relation.subject,
                object_=relation.object,
                created_at=relation.created_at,
                reference=query.reference,
                subject_current=KnowledgeRevisionRef(
                    entry_id=subject_current.id,
                    revision=subject_current.revision,
                ),
                subject_status=subject_current.status,
                object_current=KnowledgeRevisionRef(
                    entry_id=object_current.id,
                    revision=object_current.revision,
                ),
                object_status=object_current.status,
            )
            if _knowledge_lineage_link_matches_query(link, query):
                candidates.append(link)
        candidates.sort(key=lambda item: (item.created_at, item.relation_id))
        return _relation_queries._bounded_knowledge_lineage_result(
            query,
            reference_current=KnowledgeRevisionRef(
                entry_id=reference_current.id,
                revision=reference_current.revision,
            ),
            reference_status=reference_current.status,
            candidates=candidates,
            fingerprint=fingerprint,
        )

    def _require_relation_endpoints(
        self,
        relation: KnowledgeRelation,
        scope: KnowledgeAccessScope,
        *,
        operation: str,
    ) -> tuple[KnowledgeEntry, KnowledgeEntry, KnowledgeEntry, KnowledgeEntry]:
        result: list[tuple[KnowledgeEntry, KnowledgeEntry]] = []
        for reference in (relation.subject, relation.object):
            exact = self._entry_revision(reference.entry_id, reference.revision)
            current = self._current_entry(reference.entry_id)
            if exact is None or current is None:
                raise KnowledgeRelationConflict("endpoint_missing")
            if not _knowledge_scope_allows_entry(
                scope,
                exact,
            ) or not _knowledge_scope_allows_entry(scope, current):
                raise KnowledgeAccessDenied(operation)
            result.append((exact, current))
        return result[0][0], result[0][1], result[1][0], result[1][1]

    def _prepare_relation_change(
        self,
        relation: KnowledgeRelation,
        *,
        sequence: int,
        operation_id: str,
        committed_at: datetime,
    ) -> KnowledgeChange:
        return KnowledgeChange(
            id=f"kchg_{uuid4().hex}",
            sequence=sequence,
            kind=KnowledgeChangeKind.RELATION_PUBLISHED,
            entry_id=relation.subject.entry_id,
            entry_revision=relation.subject.revision,
            committed_at=committed_at,
            operation_id=operation_id,
            relation_id=relation.id,
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
        publication_operation = self._maintenance_proposal_operation_by_id.get(proposal.id)
        if publication_operation != copied.request.publication_operation_id:
            raise KnowledgeMaintenanceConflict("proposal_publication_mismatch")
        assert publication_operation is not None
        publication = self._maintenance_proposal_publications.get(publication_operation)
        snapshot = self._maintenance_proposal_publication_access.get(publication_operation)
        if publication is None or snapshot is None:
            raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
        stored_proposal, accepted_plan, publication_receipt = publication
        if (
            stored_proposal != proposal
            or stored_proposal.fingerprint != proposal.fingerprint
            or accepted_plan.fingerprint != copied.request.accepted_plan_fingerprint
            or publication_receipt.request_sha256 != copied.request.publication_request_sha256
            or not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot)
        ):
            raise KnowledgeMaintenanceConflict("proposal_publication_mismatch")
        copied = require_knowledge_maintenance_governance_authority_records(
            copied,
            stored_proposal,
            accepted_plan,
            publication_receipt,
        )

        existing = self._maintenance_governance_routes.get(copied.request.operation_id)
        if existing is not None:
            existing_snapshot = self._maintenance_governance_route_access.get(
                copied.request.operation_id
            )
            if existing_snapshot is None or not _knowledge_scope_allows_maintenance_access_snapshot(
                scope,
                existing_snapshot,
            ):
                raise KnowledgeAccessDenied("record_maintenance_governance_route")
            if type(existing) is not KnowledgeMaintenanceGovernanceReceipt:
                raise KnowledgeMaintenanceConflict("malformed_governance_receipt")
            if existing.authority != copied:
                raise KnowledgeMaintenanceConflict("governance_operation_reuse")
            return copy_knowledge_maintenance_governance_receipt(existing, replayed=True)
        if copied.request.operation_id in self._maintenance_receipts:
            raise KnowledgeMaintenanceConflict("governance_operation_reuse")
        prior_route = self._maintenance_governance_route_by_proposal.get(proposal.id)
        if prior_route is not None:
            raise KnowledgeMaintenanceConflict("proposal_already_governed")
        if proposal.id in self._maintenance_operation_by_proposal:
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
        self._maintenance_governance_routes[copied.request.operation_id] = receipt
        self._maintenance_governance_route_access[copied.request.operation_id] = snapshot
        self._maintenance_governance_route_by_proposal[proposal.id] = copied.request.operation_id
        return copy_knowledge_maintenance_governance_receipt(receipt)

    @runtime_knowledge_operation("read")
    async def load_maintenance_governance_route(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceGovernanceReceipt | None:
        from cayu.knowledge.maintenance_governance import (
            KnowledgeMaintenanceGovernanceReceipt,
            copy_knowledge_maintenance_governance_receipt,
        )

        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        snapshot = self._maintenance_governance_route_access.get(operation_id)
        receipt = self._maintenance_governance_routes.get(operation_id)
        if receipt is None:
            if snapshot is not None:
                raise KnowledgeMaintenanceConflict("malformed_governance_receipt")
            return None
        if snapshot is None:
            raise KnowledgeMaintenanceConflict("malformed_governance_receipt")
        if type(receipt) is not KnowledgeMaintenanceGovernanceReceipt:
            raise KnowledgeMaintenanceConflict("malformed_governance_receipt")
        if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
            return None
        return copy_knowledge_maintenance_governance_receipt(receipt)

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
        existing = self._semantic_watch_receipts.get(operation_id)
        if existing is not None:
            stored_scope = self._semantic_watch_receipt_access.get(operation_id)
            try:
                if stored_scope is None or type(existing) is not KnowledgeSemanticWatchReceipt:
                    raise ValueError("Semantic-watch receipt storage is incomplete.")
                stored_scope = copy_knowledge_access_scope(stored_scope)
                existing = copy_knowledge_semantic_watch_receipt(existing)
                if existing.replayed or existing.authority.invocation.access_scope != stored_scope:
                    raise ValueError("Semantic-watch receipt indexes conflict with content.")
            except Exception:
                raise KnowledgeSemanticWatchConflict("malformed_receipt") from None
            if stored_scope != scope:
                raise KnowledgeAccessDenied("record_semantic_watch_outcome")
            if existing.authority.invocation != copied.invocation:
                raise KnowledgeSemanticWatchConflict("operation_reuse")
            return copy_knowledge_semantic_watch_receipt(existing, replayed=True)
        if operation_id in self._semantic_watch_receipt_access:
            raise KnowledgeSemanticWatchConflict("malformed_receipt")
        validation_now = self._clock()
        records = []
        references = {candidate.reference for candidate in copied.evidence.candidates}
        for reference in sorted(references, key=lambda item: (item.entry_id, item.revision)):
            entry = self._current_entry(reference.entry_id)
            if entry is None or entry.revision != reference.revision:
                raise KnowledgeSemanticWatchConflict("candidate_stale")
            if not _knowledge_scope_allows_entry(scope, entry, now=validation_now):
                raise KnowledgeAccessDenied("record_semantic_watch_outcome")
            records.append(
                (
                    entry,
                    self._chunks.get((entry.id, entry.revision), ()),
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
        self._semantic_watch_receipts[operation_id] = receipt
        self._semantic_watch_receipt_access[operation_id] = copy_knowledge_access_scope(scope)
        return copy_knowledge_semantic_watch_receipt(receipt)

    @runtime_knowledge_operation("read")
    async def load_semantic_watch_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSemanticWatchReceipt | None:
        from cayu.knowledge.semantic_watch import (
            KnowledgeSemanticWatchConflict,
            KnowledgeSemanticWatchReceipt,
            copy_knowledge_semantic_watch_receipt,
        )

        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_semantic_watch_identity(operation_id, "operation_id")
        receipt = self._semantic_watch_receipts.get(operation_id)
        stored_scope = self._semantic_watch_receipt_access.get(operation_id)
        if receipt is None:
            if stored_scope is not None:
                raise KnowledgeSemanticWatchConflict("malformed_receipt")
            return None
        try:
            if stored_scope is None or type(receipt) is not KnowledgeSemanticWatchReceipt:
                raise ValueError("Semantic-watch receipt storage is incomplete.")
            stored_scope = copy_knowledge_access_scope(stored_scope)
            receipt = copy_knowledge_semantic_watch_receipt(receipt)
            if receipt.replayed or receipt.authority.invocation.access_scope != stored_scope:
                raise ValueError("Semantic-watch receipt indexes conflict with content.")
        except Exception:
            raise KnowledgeSemanticWatchConflict("malformed_receipt") from None
        if stored_scope != scope:
            return None
        return receipt

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
        publication_operations: set[str] = set()
        publication_operation = self._maintenance_proposal_operation_by_id.get(proposal.id)
        if publication_operation is not None:
            publication_operations.add(publication_operation)
        owner_proposal_id = self._maintenance_proposal_id_by_replacement_entry.get(
            proposal.replacement.entry_id
        )
        if owner_proposal_id is not None:
            owner_operation = self._maintenance_proposal_operation_by_id.get(owner_proposal_id)
            if owner_operation is None:
                raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
            publication_operations.add(owner_operation)

        publication_snapshot: _KnowledgeMaintenanceAccessSnapshot | None = None
        for owned_operation in sorted(publication_operations):
            snapshot = self._maintenance_proposal_publication_access.get(owned_operation)
            record = self._maintenance_proposal_publications.get(owned_operation)
            if snapshot is None or record is None:
                raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
            if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
                raise KnowledgeAccessDenied(operation)
            stored_proposal = record[0]
            if stored_proposal != proposal or stored_proposal.fingerprint != proposal.fingerprint:
                raise KnowledgeMaintenanceConflict("proposal_publication_mismatch")
            if publication_snapshot is not None and publication_snapshot != snapshot:
                raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
            publication_snapshot = snapshot
        if KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY in decision.metadata:
            from cayu.knowledge.maintenance_governance import (
                governance_authority_from_maintenance_records,
            )

            if publication_operation is None:
                raise KnowledgeMaintenanceConflict("governance_requires_published_proposal")
            governed_record = self._maintenance_proposal_publications.get(publication_operation)
            if governed_record is None:
                raise KnowledgeMaintenanceConflict("malformed_proposal_publication")
            governance_authority_from_maintenance_records(
                governed_record[0],
                governed_record[1],
                governed_record[2],
                decision,
            )
        routed_receipt = self._maintenance_governance_routes.get(decision.operation_id)
        if routed_receipt is not None:
            routed_snapshot = self._maintenance_governance_route_access.get(decision.operation_id)
            if routed_snapshot is None or not _knowledge_scope_allows_maintenance_access_snapshot(
                scope,
                routed_snapshot,
            ):
                raise KnowledgeAccessDenied(operation)
            raise KnowledgeMaintenanceConflict("operation_reuse")
        prior_governance_route = self._maintenance_governance_route_by_proposal.get(proposal.id)
        if (
            prior_governance_route is not None
            and KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY in decision.metadata
        ):
            routed_snapshot = self._maintenance_governance_route_access.get(prior_governance_route)
            if routed_snapshot is None or not _knowledge_scope_allows_maintenance_access_snapshot(
                scope,
                routed_snapshot,
            ):
                raise KnowledgeAccessDenied(operation)
            raise KnowledgeMaintenanceConflict("proposal_already_governed")
        existing_receipt = self._maintenance_receipts.get(decision.operation_id)
        if existing_receipt is not None:
            snapshot = self._maintenance_access.get(decision.operation_id)
            if snapshot is None:
                raise KnowledgeMaintenanceConflict("malformed_receipt")
            if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
                raise KnowledgeAccessDenied(operation)
            stored_proposal = self._maintenance_proposals.get(existing_receipt.proposal_id)
            stored_decision = self._maintenance_decisions.get(decision.operation_id)
            if stored_proposal is None or stored_decision is None:
                raise KnowledgeMaintenanceConflict("malformed_receipt")
            _validate_knowledge_maintenance_replay(
                stored_proposal,
                stored_decision,
                existing_receipt,
                proposal=proposal,
                decision=decision,
                request_sha256=request_sha256,
            )
            return copy_knowledge_maintenance_decision_receipt(
                existing_receipt,
                replayed=True,
            )

        prior_operation = self._maintenance_operation_by_proposal.get(proposal.id)
        if prior_operation is not None:
            snapshot = self._maintenance_access.get(prior_operation)
            if snapshot is None:
                raise KnowledgeMaintenanceConflict("malformed_receipt")
            if not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot):
                raise KnowledgeAccessDenied(operation)
            raise KnowledgeMaintenanceConflict("proposal_already_decided")
        current_entries = {
            entry_id: entry
            for entry_id in [
                proposal.replacement.entry_id,
                *(source.entry_id for source in proposal.sources),
            ]
            if (entry := self._current_entry(entry_id)) is not None
        }
        if (
            decision.kind is KnowledgeMaintenanceDecisionKind.REJECT
            and publication_snapshot is not None
        ):
            replacement = _maintenance_rules._require_knowledge_maintenance_current_replacement(
                proposal,
                current_entries,
                access_scope=scope,
                operation=operation,
            )
            sources: list[KnowledgeEntry] = []
            decision_snapshot = publication_snapshot
        else:
            replacement, sources = (
                _maintenance_rules._require_knowledge_maintenance_current_entries(
                    proposal,
                    current_entries,
                    access_scope=scope,
                    operation=operation,
                )
            )
            decision_snapshot = _knowledge_maintenance_access_snapshot([replacement, *sources])
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
            self._maintenance_proposals[proposal.id] = proposal
            self._maintenance_decisions[decision.operation_id] = decision
            self._maintenance_receipts[decision.operation_id] = receipt
            self._maintenance_operation_by_proposal[proposal.id] = decision.operation_id
            self._maintenance_access[decision.operation_id] = decision_snapshot
            return copy_knowledge_maintenance_decision_receipt(receipt)

        active_replacement, archived_sources = _maintenance_rules._knowledge_maintenance_successors(
            proposal,
            replacement,
            sources,
            access_scope=scope,
            committed_at=committed_at,
            operation=operation,
        )
        successor_by_id = {entry.id: entry for entry in [active_replacement, *archived_sources]}
        lifecycle_material: list[
            tuple[KnowledgeEntry, KnowledgeEntry, list[KnowledgeChunk], list[KnowledgeEvidence]]
        ] = []
        successor_payload_bytes: dict[tuple[str, int], int] = {}
        all_chunk_ids: list[str] = []
        all_evidence_ids: list[str] = []
        for successor in [active_replacement, *archived_sources]:
            current = current_entries[successor.id]
            previous_chunks = self._chunks.get((current.id, current.revision), [])
            chunks = self._revision_chunks(successor, None, previous=current)
            evidence = _revision_rules._copy_evidence_for_revision(
                self._evidence.get((current.id, current.revision), []),
                entry=successor,
                previous_chunks=previous_chunks,
                chunks=chunks,
            )
            self._require_chunk_ids_available(
                chunks,
                access_scope=scope,
                operation=operation,
            )
            self._require_evidence_ids_available(
                evidence,
                access_scope=scope,
                operation=operation,
            )
            all_chunk_ids.extend(chunk.id for chunk in chunks)
            all_evidence_ids.extend(item.id for item in evidence)
            successor_payload_bytes[(successor.id, successor.revision)] = (
                knowledge_entry_payload_bytes(successor)
            )
            lifecycle_material.append((current, successor, chunks, evidence))
        if len(set(all_chunk_ids)) != len(all_chunk_ids):
            raise KnowledgeChunkConflict(operation)
        if len(set(all_evidence_ids)) != len(all_evidence_ids):
            raise KnowledgeEvidenceConflict(operation)

        for relation in proposal.relations:
            occupied_id = self._relations.get(relation.id)
            occupied_semantic_id = self._relation_semantics.get(
                _knowledge_relation_semantic_key(relation)
            )
            for occupied_relation_id in (
                relation.id if occupied_id else None,
                occupied_semantic_id,
            ):
                if occupied_relation_id is None:
                    continue
                occupied = self._relations.get(occupied_relation_id)
                if occupied is not None:
                    self._require_relation_endpoints(occupied, scope, operation=operation)
                raise KnowledgeMaintenanceConflict("relation_exists")
            historic_sequence = self._relation_change_sequences.get(relation.id)
            if historic_sequence is not None:
                historic_change = self._changes_by_sequence[historic_sequence]
                audiences = self._change_access.get(historic_sequence, ())
                if not _knowledge_scope_allows_change(scope, historic_change, audiences):
                    raise KnowledgeAccessDenied(operation)
                raise KnowledgeMaintenanceConflict("relation_exists")

        change_count = len(lifecycle_material) + len(proposal.relations)
        if self._next_change_sequence + change_count - 1 > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
            raise RuntimeError("Knowledge change sequence is exhausted.")
        prepared_lifecycle_changes: list[
            tuple[KnowledgeChange, KnowledgeEntry, KnowledgeEntry]
        ] = []
        sequence = self._next_change_sequence
        for current, successor, _, _ in lifecycle_material:
            prepared_lifecycle_changes.append(
                (
                    KnowledgeChange(
                        id=f"kchg_{uuid4().hex}",
                        sequence=sequence,
                        kind=KnowledgeChangeKind.STATUS_TRANSITIONED,
                        entry_id=successor.id,
                        entry_revision=successor.revision,
                        committed_at=committed_at,
                        operation_id=decision.operation_id,
                    ),
                    current,
                    successor,
                )
            )
            sequence += 1

        post_current_entries = dict(current_entries)
        post_current_entries.update(successor_by_id)
        prepared_relation_changes: list[
            tuple[KnowledgeRelation, KnowledgeChange, tuple[_KnowledgeChangeAudience, ...]]
        ] = []
        for relation in proposal.relations:
            subject_exact = (
                active_replacement
                if relation.subject
                == KnowledgeRevisionRef(
                    entry_id=active_replacement.id,
                    revision=active_replacement.revision,
                )
                else self._entry_revision(
                    relation.subject.entry_id,
                    relation.subject.revision,
                )
            )
            object_exact = (
                active_replacement
                if relation.object
                == KnowledgeRevisionRef(
                    entry_id=active_replacement.id,
                    revision=active_replacement.revision,
                )
                else self._entry_revision(
                    relation.object.entry_id,
                    relation.object.revision,
                )
            )
            if subject_exact is None or object_exact is None:
                raise KnowledgeMaintenanceStale("relation_endpoint")
            snapshot = _knowledge_relation_access_snapshot(
                subject_exact=subject_exact,
                subject_current=post_current_entries[relation.subject.entry_id],
                object_exact=object_exact,
                object_current=post_current_entries[relation.object.entry_id],
            )
            change = self._prepare_relation_change(
                relation,
                sequence=sequence,
                operation_id=decision.operation_id,
                committed_at=committed_at,
            )
            prepared_relation_changes.append(
                (
                    relation,
                    change,
                    _knowledge_relation_change_audiences(
                        change,
                        access_snapshot=snapshot,
                    ),
                )
            )
            sequence += 1

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
        maintenance_access = _knowledge_maintenance_access_snapshot([replacement, *sources])

        for _, successor, chunks, evidence in lifecycle_material:
            self._entries[successor.id][successor.revision] = successor
            self._entry_payload_bytes[(successor.id, successor.revision)] = successor_payload_bytes[
                (successor.id, successor.revision)
            ]
            self._chunks[(successor.id, successor.revision)] = chunks
            self._evidence[(successor.id, successor.revision)] = evidence
            self._current_revisions[successor.id] = successor.revision
        for change, current, successor in prepared_lifecycle_changes:
            self._record_change(change, before_entry=current, after_entry=successor)
        for relation, change, audiences in prepared_relation_changes:
            self._relations[relation.id] = relation
            self._index_relation(relation)
            self._relation_semantics[_knowledge_relation_semantic_key(relation)] = relation.id
            self._changes.append(change)
            self._changes_by_sequence[change.sequence] = change
            self._change_access[change.sequence] = audiences
            self._relation_change_sequences[relation.id] = change.sequence
        self._next_change_sequence = sequence
        self._maintenance_proposals[proposal.id] = proposal
        self._maintenance_decisions[decision.operation_id] = decision
        self._maintenance_receipts[decision.operation_id] = receipt
        self._maintenance_operation_by_proposal[proposal.id] = decision.operation_id
        self._maintenance_access[decision.operation_id] = maintenance_access
        return copy_knowledge_maintenance_decision_receipt(receipt)

    @runtime_knowledge_operation("read")
    async def load_maintenance_proposal(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposal | None:
        scope = self._operation_access_scope(access_scope)
        proposal_id = _knowledge_maintenance_identity(proposal_id, "proposal_id")
        publication_operation = self._maintenance_proposal_operation_by_id.get(proposal_id)
        if publication_operation is not None:
            snapshot = self._maintenance_proposal_publication_access.get(publication_operation)
            proposal = self._maintenance_proposals.get(proposal_id)
            if (
                snapshot is None
                or proposal is None
                or not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot)
            ):
                return None
            return copy_knowledge_maintenance_proposal(proposal)
        operation_id = self._maintenance_operation_by_proposal.get(proposal_id)
        if operation_id is None:
            return None
        snapshot = self._maintenance_access.get(operation_id)
        proposal = self._maintenance_proposals.get(proposal_id)
        if (
            snapshot is None
            or proposal is None
            or not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot)
        ):
            return None
        return copy_knowledge_maintenance_proposal(proposal)

    @runtime_knowledge_operation("read")
    async def load_maintenance_decision(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecision | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        snapshot = self._maintenance_access.get(operation_id)
        decision = self._maintenance_decisions.get(operation_id)
        if (
            snapshot is None
            or decision is None
            or not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot)
        ):
            return None
        return copy_knowledge_maintenance_decision(decision)

    @runtime_knowledge_operation("read")
    async def load_maintenance_decision_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecisionReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_maintenance_identity(operation_id, "operation_id")
        snapshot = self._maintenance_access.get(operation_id)
        receipt = self._maintenance_receipts.get(operation_id)
        if (
            snapshot is None
            or receipt is None
            or not _knowledge_scope_allows_maintenance_access_snapshot(scope, snapshot)
        ):
            return None
        return copy_knowledge_maintenance_decision_receipt(receipt)

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
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        _validate_positive_int(max_records, "max_records")
        _validate_positive_int(max_bytes, "max_bytes")
        access_now = datetime.now(UTC)
        if revision is not None:
            current = self._current_entry(clean_id)
            if current is None or not _knowledge_scope_allows_entry(
                scope,
                current,
                now=access_now,
            ):
                return None
        entry = self._entry_revision(clean_id, revision)
        if entry is None or not _knowledge_scope_allows_entry(
            scope,
            entry,
            now=access_now,
        ):
            return None
        stored = self._evidence.get((clean_id, entry.revision), [])
        selected = _retrieval_results._bounded_knowledge_evidence(
            stored,
            max_records=max_records,
            max_bytes=max_bytes,
        )
        return KnowledgeEvidenceResult(
            entry_id=entry.id,
            entry_revision=entry.revision,
            evidence=selected,
            truncated=len(selected) < len(stored),
            limit=max_records,
            max_bytes=max_bytes,
            total_evidence_known=len(stored),
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
        current_sequence = self._next_change_sequence - 1
        if after_sequence > current_sequence:
            raise ValueError(
                "`after_sequence` cannot exceed the current knowledge change sequence."
            )
        selected: list[KnowledgeChange] = []
        high_water = 0
        for change in self._changes:
            audiences = self._change_access.get(change.sequence, ())
            if not _knowledge_scope_allows_change(scope, change, audiences):
                continue
            high_water = max(high_water, change.sequence)
            if change.sequence > after_sequence and len(selected) <= limit:
                selected.append(change)
        truncated = len(selected) > limit
        selected = selected[:limit]
        next_after = selected[-1].sequence if truncated else max(after_sequence, high_water)
        return KnowledgeChangeBatch(
            changes=selected,
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
        current_time = self._clock()
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        state = self._change_consumers.get(consumer_id)
        if state is None:
            state = KnowledgeChangeConsumerState(
                consumer_id=consumer_id,
                access_scope_sha256=scope_sha256,
                updated_at=current_time,
            )
        elif state.access_scope_sha256 != scope_sha256:
            raise KnowledgeChangeConsumerConflict("access_scope_mismatch")

        if state.pending_change_sequence is not None:
            stored_change = self._change_by_sequence(state.pending_change_sequence)
            audiences = self._change_access.get(state.pending_change_sequence, ())
            still_allowed = stored_change is not None and _knowledge_scope_allows_change(
                scope,
                stored_change,
                audiences,
            )
            assert state.lease_expires_at is not None
            if still_allowed and state.lease_expires_at > current_time:
                if state.pending_worker_id != worker_id:
                    self._change_consumers[consumer_id] = state
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
                self._change_consumers[consumer_id] = state
                return claim
            state = state.model_copy(
                update={
                    "pending_change_sequence": None,
                    "pending_claim_id": None,
                    "pending_worker_id": None,
                    "claimed_at": None,
                    "lease_expires_at": None,
                    "pending_attempt": (state.pending_attempt if still_allowed else 0),
                    "updated_at": current_time,
                }
            )

        candidates = self._accessible_changes(
            scope,
            after_sequence=state.cursor_sequence,
            limit=1,
        )
        if not candidates:
            self._change_consumers[consumer_id] = state
            return None
        change = candidates[0]
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
        self._change_consumers[consumer_id] = state
        return KnowledgeChangeClaim(
            consumer_id=consumer_id,
            worker_id=worker_id,
            claim_id=claim_id,
            change=change,
            attempt=attempt,
            claimed_at=claimed_at,
            lease_expires_at=lease_expires_at,
        )

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
        current_time = self._clock()
        current_sequence = self._next_change_sequence - 1
        if baseline_sequence > current_sequence:
            raise ValueError(
                "`baseline_sequence` cannot exceed the current knowledge change sequence."
            )
        state = _initialize_knowledge_change_consumer_state(
            self._change_consumers.get(consumer_id),
            consumer_id=consumer_id,
            access_scope_sha256=_knowledge_access_scope_sha256(scope),
            baseline_sequence=baseline_sequence,
            now=current_time,
        )
        self._change_consumers[consumer_id] = state
        return copy_knowledge_change_consumer_state(state)

    @runtime_knowledge_operation("modify")
    async def acknowledge_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        claim = copy_knowledge_change_claim(claim)
        current_time = self._clock()
        state = self._change_consumers.get(claim.consumer_id)
        if state is None or state.access_scope_sha256 != _knowledge_access_scope_sha256(scope):
            raise KnowledgeChangeConsumerConflict("unknown_consumer")
        claim_sha256 = _knowledge_change_claim_sha256(claim)
        acknowledged = self._acknowledged_change_claims.get((claim.consumer_id, claim.claim_id))
        if acknowledged is not None:
            if acknowledged != (claim_sha256, claim.change.sequence):
                raise KnowledgeChangeConsumerConflict("stale_claim")
            if state.cursor_sequence < claim.change.sequence:
                raise RuntimeError("Knowledge change acknowledgement is ahead of its consumer.")
            return copy_knowledge_change_consumer_state(state)
        self._require_live_change_claim(state, claim, now=current_time)
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
        self._change_consumers[claim.consumer_id] = state
        self._acknowledged_change_claims[(claim.consumer_id, claim.claim_id)] = (
            claim_sha256,
            claim.change.sequence,
        )
        return copy_knowledge_change_consumer_state(state)

    @runtime_knowledge_operation("modify")
    async def release_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        claim = copy_knowledge_change_claim(claim)
        current_time = self._clock()
        state = self._change_consumers.get(claim.consumer_id)
        if state is None or state.access_scope_sha256 != _knowledge_access_scope_sha256(scope):
            raise KnowledgeChangeConsumerConflict("unknown_consumer")
        self._require_live_change_claim(state, claim, now=current_time)
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
        self._change_consumers[claim.consumer_id] = state
        return copy_knowledge_change_consumer_state(state)

    @runtime_knowledge_operation("read")
    async def load_change_consumer_state(
        self,
        consumer_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState | None:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        state = self._change_consumers.get(consumer_id)
        if state is None:
            return None
        if state.access_scope_sha256 != _knowledge_access_scope_sha256(scope):
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
        update_sha256 = _knowledge_index_readiness_update_sha256(update)
        replay = self._index_readiness_operations.get(operation_id)
        if replay is not None:
            stored_sha256, readiness = replay
            if stored_sha256 != update_sha256:
                raise KnowledgeIndexReadinessConflict("operation_reuse")
            if not self._index_identity_is_accessible(scope, update.identity):
                raise KnowledgeAccessDenied("publish_index_readiness")
            return copy_knowledge_index_readiness(readiness)
        if not self._index_identity_is_accessible(
            scope,
            update.identity,
            require_current=True,
        ):
            raise KnowledgeIndexReadinessConflict("stale_identity")
        identity_sha256 = _knowledge_embedding_identity_sha256(update.identity)
        current = self._index_readiness_by_identity.get(identity_sha256)
        _validate_knowledge_index_readiness_transition(
            current,
            update,
            expected_sequence=expected_sequence,
        )
        sequence = self._next_index_readiness_sequence
        if sequence > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
            raise OverflowError("Knowledge index readiness sequence is exhausted.")
        readiness = KnowledgeIndexReadiness(
            sequence=sequence,
            identity=update.identity,
            state=update.state,
            attempt_id=update.attempt_id,
            failure_code=update.failure_code,
            operation_id=operation_id,
            published_at=self._clock(),
        )
        self._next_index_readiness_sequence += 1
        self._index_readiness.append(readiness)
        self._index_readiness_by_identity[identity_sha256] = readiness
        self._index_readiness_history_by_identity.setdefault(identity_sha256, []).append(readiness)
        self._index_readiness_operations[operation_id] = (update_sha256, readiness)
        return copy_knowledge_index_readiness(readiness)

    @runtime_knowledge_operation("read")
    async def load_index_readiness(
        self,
        identity: KnowledgeEmbeddingIdentity,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness | None:
        scope = self._operation_access_scope(access_scope)
        identity = copy_knowledge_embedding_identity(identity)
        if not self._index_identity_is_accessible(scope, identity):
            return None
        readiness = self._index_readiness_by_identity.get(
            _knowledge_embedding_identity_sha256(identity)
        )
        if readiness is None:
            return None
        return copy_knowledge_index_readiness(readiness)

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
        current_sequence = self._next_index_readiness_sequence - 1
        if after_sequence > current_sequence:
            raise ValueError(
                "`after_sequence` cannot exceed the current knowledge index readiness sequence."
            )
        accessible = [
            item
            for item in self._index_readiness
            if self._index_identity_is_accessible(scope, item.identity)
        ]
        high_water = max((item.sequence for item in accessible), default=0)
        selected = [item for item in accessible if item.sequence > after_sequence]
        truncated = len(selected) > limit
        selected = selected[:limit]
        next_after = selected[-1].sequence if truncated else max(after_sequence, high_water)
        return KnowledgeIndexReadinessBatch(
            readiness=selected,
            after_sequence=after_sequence,
            next_after_sequence=next_after,
            high_water_sequence=high_water,
            truncated=truncated,
            limit=limit,
        )

    def _index_identity_is_accessible(
        self,
        scope: KnowledgeAccessScope,
        identity: KnowledgeEmbeddingIdentity,
        *,
        require_current: bool = False,
    ) -> bool:
        current = self._current_entry(identity.entry_id)
        if current is None or not _knowledge_scope_allows_entry(scope, current):
            return False
        if require_current and current.revision != identity.entry_revision:
            return False
        revision = self._entry_revision(identity.entry_id, identity.entry_revision)
        if revision is None or not _knowledge_scope_allows_entry(scope, revision):
            return False
        if identity.chunk_id is None:
            return True
        chunk = next(
            (
                candidate
                for candidate in self._chunks.get(
                    (identity.entry_id, identity.entry_revision),
                    [],
                )
                if candidate.id == identity.chunk_id
            ),
            None,
        )
        if chunk is None:
            return False
        if identity.projection_type == KNOWLEDGE_CHUNK_TEXT_PROJECTION:
            return identity.projection_content_hash == _knowledge_chunk_content_hash(chunk)
        return True

    def _require_matching_change_claim(
        self,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
    ) -> None:
        stored_change = self._change_by_sequence(claim.change.sequence)
        if (
            state.pending_change_sequence != claim.change.sequence
            or state.pending_claim_id != claim.claim_id
            or state.pending_worker_id != claim.worker_id
            or state.pending_attempt != claim.attempt
            or stored_change != claim.change
        ):
            raise KnowledgeChangeConsumerConflict("stale_claim")

    def _require_live_change_claim(
        self,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
        *,
        now: datetime,
    ) -> None:
        self._require_matching_change_claim(state, claim)
        if state.lease_expires_at is None or state.lease_expires_at <= now:
            raise KnowledgeChangeConsumerConflict("expired_claim")

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
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if revision is not None:
            current = self._current_entry(clean_id)
            if current is None or not _knowledge_scope_allows_entry(scope, current):
                return []
        entry = self._entry_revision(clean_id, revision)
        if entry is None or not _knowledge_scope_allows_entry(scope, entry):
            return []
        if chunk_index is not None:
            _validate_nonnegative_int(chunk_index, "chunk_index")
        _validate_nonnegative_int(around, "around")
        if chunk_index is None and around != 0:
            raise ValueError("`around` requires `chunk_index`.")
        _validate_positive_int(max_chunks, "max_chunks")
        _validate_positive_int(max_bytes, "max_bytes")
        start_index = 0 if chunk_index is None else max(0, chunk_index - around)
        end_index = None if chunk_index is None else chunk_index + around
        chunks = self._chunks.get((clean_id, entry.revision), [])
        if chunk_index is not None:
            chunks = _retrieval_results._center_chunk_window(
                chunks, chunk_index=chunk_index, max_chunks=max_chunks
            )
        return _retrieval_results._bounded_chunks(
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
        knowledge_query = copy_knowledge_query(query)
        return self._keyword_search(
            knowledge_query,
            scope,
            revision_keys=None,
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
        knowledge_query = copy_knowledge_query(query)
        _validate_knowledge_change_sequence(knowledge_sequence, "knowledge_sequence")
        _validate_knowledge_index_sequence(
            index_readiness_sequence,
            "index_readiness_sequence",
        )
        return self._keyword_search(
            knowledge_query,
            scope,
            revision_keys=None,
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
        knowledge_query = copy_knowledge_query(query)
        references = copy_knowledge_revision_refs(revision_refs)
        _query_rules._validate_knowledge_search_frontier(
            knowledge_sequence,
            index_readiness_sequence,
        )
        return self._keyword_search(
            knowledge_query,
            scope,
            revision_keys={(item.entry_id, item.revision) for item in references},
            through_change_sequence=knowledge_sequence,
        )

    def _keyword_search(
        self,
        knowledge_query: KnowledgeQuery,
        scope: KnowledgeAccessScope,
        *,
        revision_keys: set[tuple[str, int]] | None,
        through_change_sequence: int | None,
    ) -> KnowledgeSearchResult:
        if knowledge_query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("InMemoryKnowledgeStore supports only auto and keyword search modes.")
        terms = _knowledge_query_terms(knowledge_query)
        scored: list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str]] = []
        entry_ids = (
            self._entries
            if revision_keys is None
            else sorted({entry_id for entry_id, _ in revision_keys})
        )
        for entry_id in entry_ids:
            entry = self._current_entry(entry_id)
            if entry is None:  # pragma: no cover - internal invariant
                continue
            if revision_keys is not None and (entry.id, entry.revision) not in revision_keys:
                continue
            if through_change_sequence is not None and (
                self._revision_materialization_sequences.get((entry.id, entry.revision)) is None
                or self._revision_materialization_sequences[(entry.id, entry.revision)]
                > through_change_sequence
            ):
                continue
            if not _knowledge_scope_allows_entry(scope, entry):
                continue
            if not _query_rules._entry_matches_query(entry, knowledge_query):
                continue
            chunks = self._chunks.get((entry.id, entry.revision), [])
            if _search_scoring._entry_matches_none_terms(entry, chunks, terms):
                continue
            score, chunk, reason, preview_text = _search_scoring._score_entry(
                entry, chunks, knowledge_query
            )
            if not _query_terms_have_positive_terms(terms):
                chunk = chunks[0] if chunks else None
                score = 1.0
                reason = "exact aspect filter"
                preview_text = entry.text if chunk is None else chunk.text
            elif score <= 0:
                continue
            scored.append((score, entry, chunk, reason, preview_text))
        scored.sort(
            key=lambda item: (
                -item[0],
                -(item[1].importance or 0.0),
                -item[1].updated_at.timestamp(),
                item[1].id,
            )
        )
        return _retrieval_results._keyword_search_result_from_scored(
            scored,
            knowledge_query,
            score_kind=(
                "inmemory_keyword" if _query_terms_have_positive_terms(terms) else "exact_metadata"
            ),
        )

    @runtime_knowledge_operation("read")
    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_list_query(query)
        entries = [
            entry
            for entry_id in self._entries
            if (entry := self._current_entry(entry_id)) is not None
            if _knowledge_scope_allows_entry(scope, entry)
            if _query_rules._entry_matches_list_query(entry, knowledge_query)
        ]
        entries.sort(
            key=lambda entry: (
                -(entry.importance or 0.0),
                -entry.updated_at.timestamp(),
                entry.id,
            )
        )
        facets, facets_truncated = _knowledge_facets(
            entries,
            knowledge_query.group_by,
            limit=knowledge_query.limit,
        )
        items: list[KnowledgeListItem] = []
        remaining = knowledge_query.max_bytes
        truncated = False
        for entry in entries[: knowledge_query.limit]:
            if remaining <= 0:
                truncated = True
                break
            preview_source = entry.title or entry.text
            preview = _retrieval_results._truncate_text_to_bytes(preview_source, remaining)
            if not preview:
                truncated = True
                break
            preview_complete = len(preview.encode("utf-8")) == len(preview_source.encode("utf-8"))
            if not preview_complete:
                truncated = True
            remaining -= len(preview.encode("utf-8"))
            items.append(
                KnowledgeListItem(
                    entry=entry,
                    chunk_count=len(self._chunks.get((entry.id, entry.revision), [])),
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return KnowledgeListResult(
            query=knowledge_query,
            entries=items,
            facets=facets,
            facets_truncated=facets_truncated,
            truncated=truncated or len(items) < len(entries) or facets_truncated,
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_entries_known=len(entries),
        )


def _knowledge_facets(
    entries: list[KnowledgeEntry],
    group_by: KnowledgeListGroup | None,
    *,
    limit: int,
) -> tuple[list[KnowledgeFacet], bool]:
    if group_by is None:
        return [], False
    counter: Counter[tuple[str | None, str]] = Counter()
    for entry in entries:
        if group_by is KnowledgeListGroup.KIND:
            counter[(None, entry.kind)] += 1
        elif group_by is KnowledgeListGroup.LABEL:
            for key, value in entry.labels.items():
                counter[(key, value)] += 1
        elif group_by is KnowledgeListGroup.ASPECT:
            for aspect in entry.aspects:
                counter[(None, aspect)] += 1
        elif group_by is KnowledgeListGroup.IMPACT_TARGET:
            for target in entry.impact_targets:
                counter[(None, target)] += 1
        elif group_by is KnowledgeListGroup.VISIBILITY:
            counter[(None, entry.visibility.value)] += 1
        elif group_by is KnowledgeListGroup.SOURCE_TYPE and entry.source_type is not None:
            counter[(None, entry.source_type)] += 1
        elif group_by is KnowledgeListGroup.NAMESPACE:
            counter[(None, entry.namespace)] += 1
    facets = [
        KnowledgeFacet(field=group_by, key=key, value=value, count=count)
        for (key, value), count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    return facets[:limit], len(facets) > limit


def _next_updated_at(entry: KnowledgeEntry) -> datetime:
    return max(datetime.now(UTC), entry.created_at, entry.updated_at)
