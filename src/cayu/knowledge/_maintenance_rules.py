"""Shared validation and revision transitions for reviewed knowledge maintenance."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge import _revision_rules
from cayu.knowledge._access_rules import (
    _require_knowledge_entry_access,
    _require_knowledge_successor_access,
)
from cayu.knowledge.maintenance_contracts import (
    KnowledgeMaintenanceProposal,
    KnowledgeMaintenanceStale,
)
from cayu.knowledge.records import (
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceRole,
    KnowledgeStatus,
    _next_knowledge_revision,
)
from cayu.knowledge.relations import KnowledgeRelationKind
from cayu.knowledge.scopes import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    copy_knowledge_access_scope,
)


def _require_knowledge_maintenance_current_entries(
    proposal: KnowledgeMaintenanceProposal,
    current_entries: dict[str, KnowledgeEntry],
    *,
    access_scope: KnowledgeAccessScope,
    operation: str,
) -> tuple[KnowledgeEntry, list[KnowledgeEntry]]:
    replacement = _require_knowledge_maintenance_current_replacement(
        proposal,
        current_entries,
        access_scope=access_scope,
        operation=operation,
    )

    sources: list[KnowledgeEntry] = []
    for reference in proposal.sources:
        source = current_entries.get(reference.entry_id)
        if source is None:
            raise KnowledgeMaintenanceStale("source_missing")
        _require_knowledge_entry_access(access_scope, source, operation=operation)
        if source.revision != reference.revision:
            raise KnowledgeMaintenanceStale("source_revision")
        if source.status is not KnowledgeStatus.ACTIVE:
            raise KnowledgeMaintenanceStale("source_status")
        sources.append(source)
    return replacement, sources


def _require_knowledge_maintenance_current_replacement(
    proposal: KnowledgeMaintenanceProposal,
    current_entries: dict[str, KnowledgeEntry],
    *,
    access_scope: KnowledgeAccessScope,
    operation: str,
) -> KnowledgeEntry:
    if copy_knowledge_access_scope(access_scope) != proposal.access_scope:
        raise KnowledgeAccessDenied(operation)
    replacement = current_entries.get(proposal.replacement.entry_id)
    if replacement is None:
        raise KnowledgeMaintenanceStale("replacement_missing")
    _require_knowledge_entry_access(access_scope, replacement, operation=operation)
    if replacement.revision != proposal.replacement.revision:
        raise KnowledgeMaintenanceStale("replacement_revision")
    if replacement.status is not KnowledgeStatus.PENDING:
        raise KnowledgeMaintenanceStale("replacement_status")
    return replacement


def _require_knowledge_maintenance_publication_boundary(
    replacement: KnowledgeEntry,
    sources: list[KnowledgeEntry],
) -> None:
    """Prevent a generated replacement from widening any source boundary."""

    boundary = (replacement.namespace, replacement.labels, replacement.visibility)
    if any((source.namespace, source.labels, source.visibility) != boundary for source in sources):
        raise ValueError(
            "A maintenance proposal replacement and every source must have identical "
            "namespace, labels, and visibility."
        )


def _require_knowledge_maintenance_source_evidence(
    evidence: list[KnowledgeEvidence],
    sources: list[KnowledgeEntry],
) -> None:
    """Bind one immutable evidence record to every reviewed source revision."""

    by_source: dict[tuple[str, int], KnowledgeEvidence] = {}
    for item in evidence:
        if (
            item.source_type != "knowledge_revision"
            or item.source_id is None
            or item.source_revision is None
            or item.source_hash is None
            or item.chunk_id is not None
            or item.role is not KnowledgeEvidenceRole.ORIGIN
            or item.disposition is not KnowledgeEvidenceDisposition.LIVE
        ):
            raise ValueError(
                "Maintenance proposal evidence must identify one live exact knowledge revision."
            )
        try:
            source_revision = int(item.source_revision)
        except ValueError:
            raise ValueError(
                "Maintenance proposal evidence source revisions must be canonical integers."
            ) from None
        if str(source_revision) != item.source_revision:
            raise ValueError(
                "Maintenance proposal evidence source revisions must be canonical integers."
            )
        key = (item.source_id, source_revision)
        if key in by_source:
            raise ValueError("Maintenance proposal evidence cannot repeat a source revision.")
        by_source[key] = item

    expected = {(source.id, source.revision) for source in sources}
    if set(by_source) != expected:
        raise ValueError("Maintenance proposal evidence must exactly cover every source revision.")
    for source in sources:
        item = by_source[(source.id, source.revision)]
        expected_hash = sha256(
            canonical_durable_json_bytes(
                source.model_dump(mode="json"),
                "maintenance source revision",
            )
        ).hexdigest()
        if item.source_hash != expected_hash or item.locator != {
            "entry_id": source.id,
            "revision": source.revision,
        }:
            raise ValueError("Maintenance proposal evidence does not bind its source revision.")


def _knowledge_maintenance_successors(
    proposal: KnowledgeMaintenanceProposal,
    replacement: KnowledgeEntry,
    sources: list[KnowledgeEntry],
    *,
    access_scope: KnowledgeAccessScope,
    committed_at: datetime,
    operation: str,
) -> tuple[KnowledgeEntry, list[KnowledgeEntry]]:
    active_replacement = replacement.model_copy(
        update={
            "revision": _next_knowledge_revision(replacement.revision),
            "status": KnowledgeStatus.ACTIVE,
            "updated_at": max(committed_at, replacement.created_at, replacement.updated_at),
        }
    )
    _revision_rules._validate_revision_successor(replacement, active_replacement)
    _require_knowledge_successor_access(
        access_scope,
        active_replacement,
        operation=operation,
    )
    superseded = {
        (relation.object.entry_id, relation.object.revision)
        for relation in proposal.relations
        if relation.kind is KnowledgeRelationKind.SUPERSEDES
    }
    archived_sources: list[KnowledgeEntry] = []
    for source in sources:
        if (source.id, source.revision) not in superseded:
            continue
        archived = source.model_copy(
            update={
                "revision": _next_knowledge_revision(source.revision),
                "status": KnowledgeStatus.ARCHIVED,
                "updated_at": max(committed_at, source.created_at, source.updated_at),
            }
        )
        _revision_rules._validate_revision_successor(source, archived)
        _require_knowledge_successor_access(access_scope, archived, operation=operation)
        archived_sources.append(archived)
    archived_sources.sort(key=lambda entry: entry.id)
    return active_replacement, archived_sources
