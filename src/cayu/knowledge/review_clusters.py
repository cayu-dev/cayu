"""Group pending knowledge proposals that say the same thing, and rank them for review.

``cluster_knowledge_proposals`` joins proposals a ``KnowledgeSimilarity`` part calls
similar (within its scope) into clusters, transitively, attaches the kept entries each
cluster resembles, and orders the clusters by corroboration: the number of sources their
proposals record (``knowledge_source_count``), so a fact several sessions proposed
independently comes first. Every proposal is in exactly one cluster; one that resembles
nothing is a cluster of one.

Grouping only orders the review queue. Keeping or dropping stays a reviewer's decision,
recorded per entry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cayu.knowledge._sources import knowledge_source_count
from cayu.knowledge.records import KnowledgeEntry
from cayu.knowledge.similarity import (
    KnowledgeComparable,
    KnowledgeSimilarity,
)

#: How many kept entries a cluster lists as resembling it, best first.
DEFAULT_SIMILAR_KEPT_PER_CLUSTER = 3


@dataclass(frozen=True)
class KnowledgeProposalCluster:
    """Similar pending proposals, most corroborated first, and kept entries like them."""

    members: tuple[KnowledgeEntry, ...]
    similar_kept: tuple[KnowledgeEntry, ...] = ()

    @property
    def sources(self) -> int:
        """The sources the cluster's proposals record, together."""

        return sum(knowledge_source_count(member) for member in self.members)


async def cluster_knowledge_proposals(
    proposals: Sequence[KnowledgeEntry],
    *,
    similarity: KnowledgeSimilarity,
    kept: Sequence[KnowledgeEntry] = (),
    similar_kept_per_cluster: int = DEFAULT_SIMILAR_KEPT_PER_CLUSTER,
) -> list[KnowledgeProposalCluster]:
    """Cluster ``proposals`` by similarity and order the clusters by corroboration.

    Clusters come most sources first, then larger, then most recently updated. Within a
    cluster, members come most sources first, then oldest. ``kept`` entries are only
    compared, never clustered.
    """

    if type(similar_kept_per_cluster) is not int or similar_kept_per_cluster < 0:
        raise ValueError("similar_kept_per_cluster must be a non-negative integer.")
    unique = list({entry.id: entry for entry in proposals}.values())
    items = [KnowledgeComparable.from_entry(entry) for entry in unique]
    parent = list(range(len(unique)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    kept = list(kept)
    compared_kept = kept if similar_kept_per_cluster else []
    # One call compares the proposals with each other and with the kept entries, so a
    # part prepares each entry once (``KnowledgeSimilarity.matches_many``).
    found = await similarity.matches_many(
        items, [KnowledgeComparable.from_entry(entry) for entry in compared_kept]
    )
    resembles: list[dict[int, float]] = [{} for _ in unique]
    for index, matches in enumerate(found):
        for match in matches:
            if match.index < len(unique):
                a, b = root(index), root(match.index)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                resembles[index][match.index - len(unique)] = match.score.score

    groups: dict[int, list[int]] = {}
    for index in range(len(unique)):
        groups.setdefault(root(index), []).append(index)

    clusters: list[KnowledgeProposalCluster] = []
    for indexes in groups.values():
        members = sorted(
            (unique[index] for index in indexes),
            key=lambda entry: (-knowledge_source_count(entry), entry.created_at, entry.id),
        )
        clusters.append(
            KnowledgeProposalCluster(
                members=tuple(members),
                similar_kept=_similar_kept(indexes, kept, resembles, similar_kept_per_cluster),
            )
        )
    clusters.sort(
        key=lambda cluster: (
            -cluster.sources,
            -len(cluster.members),
            -max(member.updated_at.timestamp() for member in cluster.members),
            cluster.members[0].id,
        )
    )
    return clusters


def _similar_kept(
    indexes: Sequence[int],
    kept: Sequence[KnowledgeEntry],
    resembles: Sequence[dict[int, float]],
    limit: int,
) -> tuple[KnowledgeEntry, ...]:
    best: dict[int, float] = {}
    for index in indexes:
        for position, score in resembles[index].items():
            best[position] = max(best.get(position, 0.0), score)
    ranked = sorted(best, key=lambda position: (-best[position], position))[:limit]
    return tuple(kept[position] for position in ranked)
