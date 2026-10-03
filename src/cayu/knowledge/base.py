"""Backend-independent knowledge store interface and shared access-scope defaults."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from cayu.knowledge.activation_contracts import (
    KnowledgeActivationAuthority,
    KnowledgeActivationReceipt,
    KnowledgeReviewApproval,
)
from cayu.knowledge.changes import (
    KnowledgeChangeBatch,
    KnowledgeChangeClaim,
    KnowledgeChangeConsumerState,
)
from cayu.knowledge.indexing import (
    DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
    KnowledgeEmbeddingBackfillResult,
    KnowledgeEmbeddingIdentity,
    KnowledgeEmbeddingProjection,
    KnowledgeEmbeddingProjectionWriteResult,
    KnowledgeIndexReadiness,
    KnowledgeIndexReadinessBatch,
    KnowledgeIndexReadinessUpdate,
)
from cayu.knowledge.maintenance_contracts import (
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceProposal,
)
from cayu.knowledge.publication_contracts import KnowledgePublicationReceipt
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceResult,
    KnowledgeRevisionRef,
    KnowledgeStatus,
)
from cayu.knowledge.relations import (
    KnowledgeLineageQuery,
    KnowledgeLineageResult,
    KnowledgeRelation,
    KnowledgeRelationPublicationReceipt,
    KnowledgeRelationQuery,
    KnowledgeRelationResult,
)
from cayu.knowledge.scopes import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    copy_knowledge_access_scope,
)
from cayu.knowledge.search import (
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
)
from cayu.storage._knowledge_closure import KnowledgeClosureQuery

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


def _intersect_resource_knowledge_scope(scope):
    from cayu.knowledge.access import intersect_scope

    return intersect_scope(scope)


class KnowledgeStore(ABC):
    """Searchable knowledge contract."""

    async def inspect_closure_sources(self, query: KnowledgeClosureQuery) -> dict[str, object]:
        """Inventory retained evidence and projections for exact source identities.

        This administrative seam does not expose source payloads, knowledge
        content, metadata, or vectors. Unsupported stores must refuse rather
        than acknowledge an empty inventory.
        """
        raise NotImplementedError("Knowledge store does not support closure source inventory.")

    _default_access_scope: KnowledgeAccessScope | None = None

    def bound_access_scope(self) -> KnowledgeAccessScope | None:
        """Return the explicitly bound single-principal scope, if configured."""

        if self._default_access_scope is None:
            return None
        return copy_knowledge_access_scope(self._default_access_scope)

    def _operation_access_scope(
        self,
        access_scope: KnowledgeAccessScope | None,
    ) -> KnowledgeAccessScope:
        default_scope = self._default_access_scope
        if access_scope is None:
            if default_scope is None:
                raise TypeError("knowledge operation requires `access_scope`.")
            return _intersect_resource_knowledge_scope(copy_knowledge_access_scope(default_scope))
        explicit_scope = copy_knowledge_access_scope(access_scope)
        if default_scope is not None and explicit_scope != default_scope:
            raise KnowledgeAccessDenied("access_scope_override")
        return _intersect_resource_knowledge_scope(explicit_scope)

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        """Return search modes this store can execute directly."""

        return (KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD)

    @abstractmethod
    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        """Create revision 1 of a previously unoccupied logical entry id."""

    @abstractmethod
    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        """Append exactly one revision using compare-and-swap."""

    @abstractmethod
    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        """Load one revision, optionally refusing its content before copying it."""

    async def search_revisions(
        self,
        query: KnowledgeQuery,
        revision_refs: Sequence[KnowledgeRevisionRef],
        *,
        knowledge_sequence: int | None = None,
        index_readiness_sequence: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        """Search referenced current revisions, optionally at one captured frontier.

        This narrow operation exists for delta retrieval. Implementations must apply
        revision eligibility before ranking; searching globally and filtering the
        returned top-k candidates is not conformant. When frontier sequences are
        supplied, both must be supplied and neither authoritative materialization nor
        semantic readiness may cross them.
        """

        raise NotImplementedError("This KnowledgeStore does not support exact-revision search.")

    async def search_at_frontier(
        self,
        query: KnowledgeQuery,
        *,
        knowledge_sequence: int,
        index_readiness_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        """Search current records whose authoritative inputs existed at a captured frontier.

        This narrow operation exists for full-index checkpoint recall. Implementations must
        exclude current revisions materialized after ``knowledge_sequence`` and semantic
        readiness published after ``index_readiness_sequence``. Falling back to an ordinary
        current search is not conformant because it can cross the captured processing frontier.
        """

        raise NotImplementedError("This KnowledgeStore does not support frontier-bounded search.")

    async def _inspect_lineage_at_change_sequence(
        self,
        query: KnowledgeLineageQuery,
        *,
        through_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeLineageResult | None:
        """Inspect lineage without crossing one captured knowledge-change sequence."""

        raise NotImplementedError(
            "This KnowledgeStore does not support frontier-bounded lineage inspection."
        )

    @abstractmethod
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
        """Append one lifecycle-only revision using compare-and-swap."""

    @abstractmethod
    async def delete_entry(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        """Append a tombstone by default, or physically erase after a CAS check.

        Hard deletion also erases governed activation history retained after
        expiration pruning. In that case the canonical entry is already absent,
        so a successful, idempotent audit purge returns ``None``.
        """

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
        """Publish one create/append exactly once with immutable replay evidence.

        Implementations commit the revision, chunks, evidence, current pointer,
        metadata-only change, and receipt atomically. ``expected_revision=None``
        creates revision 1; a positive value appends exactly its successor.
        Use :func:`prepare_knowledge_publication` to copy and bind the canonical
        authority tuple before entering the store transaction.
        """

        raise NotImplementedError(
            "This KnowledgeStore does not support owned revision publication."
        )

    async def approve_pending_entry(
        self,
        authority: KnowledgeActivationAuthority,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeReviewApproval:
        """Atomically append one active successor under explicit reviewed authority."""

        raise NotImplementedError(
            "This KnowledgeStore does not support attributed knowledge review approval."
        )

    async def load_activation_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeActivationReceipt | None:
        """Load immutable activation attribution for one exact operation."""

        raise NotImplementedError(
            "This KnowledgeStore does not support knowledge activation receipts."
        )

    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None:
        """Load immutable publication evidence for one operation id."""

        raise NotImplementedError(
            "This KnowledgeStore does not support knowledge publication receipts."
        )

    async def publish_relations(
        self,
        relations: list[KnowledgeRelation],
        *,
        operation_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationPublicationReceipt:
        """Atomically publish one bounded immutable relation batch exactly once."""

        raise NotImplementedError(
            "This KnowledgeStore does not support revision-bound knowledge relations."
        )

    async def load_relation_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationPublicationReceipt | None:
        """Load immutable replay evidence for one relation publication."""

        raise NotImplementedError(
            "This KnowledgeStore does not support knowledge relation receipts."
        )

    async def read_relations(
        self,
        query: KnowledgeRelationQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeRelationResult | None:
        """Read one bounded page around an exact authorized revision."""

        raise NotImplementedError(
            "This KnowledgeStore does not support revision-bound knowledge relations."
        )

    async def inspect_lineage(
        self,
        query: KnowledgeLineageQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeLineageResult | None:
        """Inspect safe lineage around one exact authorized revision."""

        raise NotImplementedError(
            "This KnowledgeStore does not support knowledge lineage inspection."
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
        """Atomically persist one accepted plan as a pending review artifact."""

        raise NotImplementedError(
            "This KnowledgeStore does not support maintenance proposal publication."
        )

    async def load_maintenance_proposal_publication(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposalPublication | None:
        """Load one exact pending or decided maintenance proposal artifact."""

        raise NotImplementedError(
            "This KnowledgeStore does not support maintenance proposal publication."
        )

    async def apply_maintenance_decision(
        self,
        proposal: KnowledgeMaintenanceProposal,
        decision: KnowledgeMaintenanceDecision,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecisionReceipt:
        """Apply one exact reviewed maintenance decision atomically."""

        raise NotImplementedError(
            "This KnowledgeStore does not support reviewed knowledge maintenance."
        )

    async def record_maintenance_governance_route(
        self,
        authority: KnowledgeMaintenanceGovernanceAuthority,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceGovernanceReceipt:
        """Atomically retain route-to-review authority without lifecycle changes."""

        raise NotImplementedError(
            "This KnowledgeStore does not support maintenance governance routing."
        )

    async def load_maintenance_governance_route(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceGovernanceReceipt | None:
        """Load immutable route-to-review attribution in scope."""

        raise NotImplementedError(
            "This KnowledgeStore does not support maintenance governance routing."
        )

    async def record_semantic_watch_outcome(
        self,
        authority: KnowledgeSemanticWatchAuthority,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSemanticWatchReceipt:
        """Atomically retain one exact policy-governed semantic-watch outcome."""

        raise NotImplementedError("This KnowledgeStore does not support semantic-watch outcomes.")

    async def load_semantic_watch_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSemanticWatchReceipt | None:
        """Load immutable semantic-watch attribution for one scoped operation."""

        raise NotImplementedError("This KnowledgeStore does not support semantic-watch outcomes.")

    async def load_maintenance_proposal(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceProposal | None:
        """Load one durably published or decided maintenance proposal in scope."""

        raise NotImplementedError(
            "This KnowledgeStore does not support reviewed knowledge maintenance."
        )

    async def load_maintenance_decision(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecision | None:
        """Load one durable maintenance decision in scope."""

        raise NotImplementedError(
            "This KnowledgeStore does not support reviewed knowledge maintenance."
        )

    async def load_maintenance_decision_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceDecisionReceipt | None:
        """Load immutable maintenance application evidence in scope."""

        raise NotImplementedError(
            "This KnowledgeStore does not support reviewed knowledge maintenance."
        )

    @abstractmethod
    async def read_evidence(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        max_records: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> KnowledgeEvidenceResult | None:
        """Read evidence for the current or one exact authorized revision."""

    @abstractmethod
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
        """Read bounded chunks for the current or one exact historical revision."""

    @abstractmethod
    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        """Search knowledge and return a bounded result envelope."""

    @abstractmethod
    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        """List entries/facets for discovery without requiring a lexical search term."""

    @abstractmethod
    async def read_changes(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeBatch:
        """Read one bounded ordered page of accessible canonical changes."""

    @abstractmethod
    async def initialize_change_consumer(
        self,
        consumer_id: str,
        *,
        baseline_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        """Bind a consumer and its cursor to a captured full-scan high-water mark."""

    @abstractmethod
    async def claim_change(
        self,
        consumer_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeClaim | None:
        """Lease the consumer's next accessible change with at-least-once semantics."""

    @abstractmethod
    async def acknowledge_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        """Fenced acknowledgement that advances one consumer cursor."""

    @abstractmethod
    async def release_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        """Release a live claim without advancing its consumer cursor."""

    @abstractmethod
    async def load_change_consumer_state(
        self,
        consumer_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState | None:
        """Load one scope-bound consumer cursor and lease state."""

    async def publish_index_readiness(
        self,
        update: KnowledgeIndexReadinessUpdate,
        *,
        expected_sequence: int | None,
        operation_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness:
        """Publish one fenced derived-index state transition.

        This hook is intentionally optional so lexical-only custom stores do not
        have to pretend they own derived indexes. Implementations that advertise
        semantic retrieval must provide equivalent durable semantics.
        """

        raise NotImplementedError(
            "This KnowledgeStore does not support index readiness publication."
        )

    async def load_index_readiness(
        self,
        identity: KnowledgeEmbeddingIdentity,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness | None:
        """Load the latest accessible readiness for one exact identity."""

        raise NotImplementedError("This KnowledgeStore does not support index readiness reads.")

    async def store_embedding_projections(
        self,
        projections: list[KnowledgeEmbeddingProjection],
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEmbeddingProjectionWriteResult:
        """Persist already-computed vectors for exact current pending attempts.

        Stale, superseded, and unauthorized projections are omitted from the
        returned accepted identities. Readiness remains a separate fenced
        publication step so a vector cannot become searchable prematurely.
        """

        raise NotImplementedError(
            "This KnowledgeStore does not support embedding projection persistence."
        )

    async def backfill_embeddings(
        self,
        query: KnowledgeListQuery | None = None,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        limit: int = DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
        refresh_existing: bool = False,
        cursor: str | None = None,
    ) -> KnowledgeEmbeddingBackfillResult:
        """Repair one bounded page of current embedding projections.

        Pass the previous result's ``next_cursor`` to continue the exact same
        query, scope, projection configuration, and refresh mode.
        """

        raise NotImplementedError("This KnowledgeStore does not support embedding backfill.")

    async def read_index_readiness(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadinessBatch:
        """Read accessible readiness events through a captured high-water mark."""

        raise NotImplementedError("This KnowledgeStore does not support index readiness reads.")

    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        """Hard-delete entries whose ``expires_at`` is at or before ``now`` (default: current UTC).

        Returns the count removed. The read-time filter (:func:`_entry_is_expired`) only *hides*
        expired entries; this reclaims their storage. Hosts call it on a schedule or opportunistically.
        ``now`` is injectable for deterministic tests.

        Default raises ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This KnowledgeStore does not support prune_expired.")
