"""Grouping pending knowledge proposals by similarity and ordering them by corroboration."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.review_clusters import cluster_knowledge_proposals
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.similarity import LexicalKnowledgeSimilarity
from cayu.storage.knowledge_review import KnowledgeReviewWorkflow
from cayu.storage.memory import InMemoryKnowledgeStore

FIXTURE = Path(__file__).parents[1] / "fixtures" / "knowledge_duplicate_pairs.json"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _sources(count: int) -> dict:
    return {"cayu:corroboration": {"source_count": count, "sources": []}}


def _fixture() -> tuple[list[KnowledgeEntry], list[KnowledgeEntry], dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    proposals, kept = [], []
    for index, row in enumerate(data["entries"]):
        entry = KnowledgeEntry(
            id=row["id"],
            title=row["title"],
            text=row["text"],
            labels={"module": row["module"]},
            status=KnowledgeStatus.ACTIVE if row["status"] == "kept" else KnowledgeStatus.PENDING,
            created_at=_NOW + timedelta(minutes=index),
            updated_at=_NOW + timedelta(minutes=index),
        )
        (kept if row["status"] == "kept" else proposals).append(entry)
    return proposals, kept, data


def test_clustering_the_labelled_fixture_groups_duplicates_and_attaches_kept_ones():
    proposals, kept, data = _fixture()
    clusters = asyncio.run(
        cluster_knowledge_proposals(
            proposals,
            similarity=LexicalKnowledgeSimilarity(scope_labels=("module",)),
            kept=kept,
        )
    )
    cluster_of = {
        member.id: number for number, cluster in enumerate(clusters) for member in cluster.members
    }
    resembles = {
        number: {entry.id for entry in cluster.similar_kept}
        for number, cluster in enumerate(clusters)
    }
    # Every proposal is in exactly one cluster; kept entries are never members.
    assert sorted(cluster_of) == sorted(entry.id for entry in proposals)
    assert sum(len(cluster.members) for cluster in clusters) == len(proposals)
    kept_ids = {entry.id for entry in kept}

    def grouped(a: str, b: str) -> bool:
        if a in kept_ids:
            a, b = b, a
        if b in kept_ids:
            return b in resembles[cluster_of[a]]
        return cluster_of[a] == cluster_of[b]

    reviewed = [pair for pair in data["distinct"] if not set(pair) <= kept_ids]
    together = [pair for pair in data["duplicates"] if grouped(*pair)]
    apart = [pair for pair in reviewed if grouped(*pair)]
    # Measured: 25 of 27 labelled duplicates end up together (in one cluster, or a kept
    # entry listed with the proposal's cluster) and none of the 222 distinct pairs with a
    # proposal in them. Clusters join transitively, so leave room for one stray link.
    assert len(reviewed) == 222
    assert len(together) >= 24
    assert len(apart) <= 5


def test_clusters_are_ordered_by_corroboration():
    text = "Queued delivery retries keep the same delivery identity after acknowledgement loss."
    proposals = [
        KnowledgeEntry(
            id="lonely", text="Branch storage snapshots stay immutable.", status="pending"
        ),
        KnowledgeEntry(id="copy-a", text=text, status="pending", created_at=_NOW),
        KnowledgeEntry(
            id="copy-b",
            text=text + " Verified.",
            status="pending",
            metadata=_sources(3),
            created_at=_NOW + timedelta(days=1),
        ),
        KnowledgeEntry(
            id="corroborated",
            text="Worker leases renew only after re-reading the task row.",
            status="pending",
            metadata=_sources(5),
        ),
    ]
    clusters = asyncio.run(
        cluster_knowledge_proposals(proposals, similarity=LexicalKnowledgeSimilarity())
    )
    assert [[member.id for member in cluster.members] for cluster in clusters] == [
        ["corroborated"],
        ["copy-b", "copy-a"],
        ["lonely"],
    ]
    assert [cluster.sources for cluster in clusters] == [5, 4, 1]


def test_grouped_review_reads_the_workflow_scope_and_admits_entries_first():
    text = "Queued delivery retries keep the same delivery identity after acknowledgement loss."
    entries = [
        KnowledgeEntry(id="p1", text=text, status="pending", labels={"team": "a"}),
        KnowledgeEntry(id="p2", text=text + " Again.", status="pending", labels={"team": "a"}),
        KnowledgeEntry(id="hidden", text=text, status="pending", labels={"team": "a"}),
        KnowledgeEntry(id="other-team", text=text, status="pending", labels={"team": "b"}),
        KnowledgeEntry(id="kept", text=text, labels={"team": "a"}),
        KnowledgeEntry(id="dropped", text=text, status="archived", labels={"team": "a"}),
    ]
    store = InMemoryKnowledgeStore(entries, access_scope=KnowledgeAccessScope.privileged())
    workflow = KnowledgeReviewWorkflow(store, labels={"team": "a"})

    async def admit(entry: KnowledgeEntry) -> bool:
        return entry.id != "hidden"

    grouped = asyncio.run(
        workflow.list_pending_clusters(similarity=LexicalKnowledgeSimilarity(), admit=admit)
    )
    assert [[m.id for m in cluster.members] for cluster in grouped.clusters] == [["p1", "p2"]]
    assert [entry.id for entry in grouped.clusters[0].similar_kept] == ["kept"]
    assert (grouped.pending_compared, grouped.kept_compared, grouped.truncated) == (2, 1, False)

    first = asyncio.run(
        workflow.list_pending_clusters(similarity=LexicalKnowledgeSimilarity(), max_entries=1)
    )
    assert first.pending_compared == 1 and first.truncated
    with pytest.raises(ValueError, match="max_entries"):
        asyncio.run(
            workflow.list_pending_clusters(similarity=LexicalKnowledgeSimilarity(), max_entries=0)
        )


def test_entries_admit_leaves_out_still_count_toward_the_read_budget():
    entries = [KnowledgeEntry(id=f"hidden-{i}", text="x", status="pending") for i in range(250)]
    store = InMemoryKnowledgeStore(entries, access_scope=KnowledgeAccessScope.privileged())
    reads = 0
    list_entries = store.list_entries

    async def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return await list_entries(*args, **kwargs)

    store.list_entries = counted  # type: ignore[method-assign]

    async def admit(entry: KnowledgeEntry) -> bool:
        return False

    grouped = asyncio.run(
        KnowledgeReviewWorkflow(store).list_pending_clusters(
            similarity=LexicalKnowledgeSimilarity(), max_entries=1, admit=admit
        )
    )
    assert grouped.clusters == [] and grouped.truncated
    # One page of 100 covers the 4-entry budget, so the scan stops after one read.
    assert reads == 1


def test_clustering_the_fixture_matches_scoring_every_pair():
    """The lexical part's indexed comparison groups exactly as scoring every pair does."""

    class _EveryPair(LexicalKnowledgeSimilarity):
        # Overriding `score` opts out of the indexed comparison.
        async def score(self, subject, candidates):
            return await super().score(subject, candidates)

    proposals, kept, _ = _fixture()

    def grouped(similarity) -> list[tuple[list[str], list[str]]]:
        clusters = asyncio.run(
            cluster_knowledge_proposals(proposals, similarity=similarity, kept=kept)
        )
        return [
            ([m.id for m in cluster.members], [e.id for e in cluster.similar_kept])
            for cluster in clusters
        ]

    for scope_labels in ((), ("module",)):
        assert grouped(LexicalKnowledgeSimilarity(scope_labels=scope_labels)) == grouped(
            _EveryPair(scope_labels=scope_labels)
        )
