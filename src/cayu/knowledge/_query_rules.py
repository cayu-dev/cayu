"""Shared knowledge query matching, semantic text and frontier validation."""

from __future__ import annotations

from datetime import UTC, datetime

from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.knowledge.changes import _validate_knowledge_change_sequence
from cayu.knowledge.indexing import _validate_knowledge_index_sequence
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus, KnowledgeVisibility
from cayu.knowledge.search import KnowledgeListQuery, KnowledgeQuery


def _entry_matches_query(entry: KnowledgeEntry, query: KnowledgeQuery) -> bool:
    return _entry_matches_metadata(
        entry,
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


def _entry_matches_list_query(entry: KnowledgeEntry, query: KnowledgeListQuery) -> bool:
    return _entry_matches_metadata(
        entry,
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


def _entry_matches_metadata(
    entry: KnowledgeEntry,
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
) -> bool:
    if namespace is not None and entry.namespace != namespace:
        return False
    for key, value in labels.items():
        if entry.labels.get(key) != value:
            return False
    if kinds is not None and entry.kind not in set(kinds):
        return False
    if entry.status not in set(statuses):
        return False
    if visibilities is not None and entry.visibility not in set(visibilities):
        return False
    if source_type is not None and entry.source_type != source_type:
        return False
    if source_id is not None and entry.source_id != source_id:
        return False
    if aspects and not set(aspects).intersection(entry.aspects):
        return False
    entry_aspects = set(entry.aspects)
    if any(not entry_aspects.intersection(group) for group in aspect_groups):
        return False
    if impact_targets and not set(impact_targets).intersection(entry.impact_targets):
        return False
    return not _entry_is_expired(entry, include_expired=include_expired)


def _entry_is_expired(entry: KnowledgeEntry, *, include_expired: bool) -> bool:
    return (
        not include_expired
        and entry.expires_at is not None
        and entry.expires_at <= datetime.now(UTC)
    )


def _semantic_query_text(query: KnowledgeQuery) -> str:
    parts: list[str] = []
    if query.text is not None:
        parts.append(query.text)
    parts.extend(query.any_terms)
    parts.extend(query.all_terms)
    parts.extend(query.phrases)
    return require_nonblank(" ".join(parts), "semantic query text")


def _validate_knowledge_search_frontier(
    knowledge_sequence: int | None,
    index_readiness_sequence: int | None,
) -> None:
    if (knowledge_sequence is None) != (index_readiness_sequence is None):
        raise ValueError(
            "`knowledge_sequence` and `index_readiness_sequence` must be supplied together."
        )
    if knowledge_sequence is None:
        return
    assert index_readiness_sequence is not None
    _validate_knowledge_change_sequence(knowledge_sequence, "knowledge_sequence")
    _validate_knowledge_index_sequence(
        index_readiness_sequence,
        "index_readiness_sequence",
    )
