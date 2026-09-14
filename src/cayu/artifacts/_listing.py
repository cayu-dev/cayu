"""Bounded newest-first selection for streaming artifact inventories."""

from datetime import datetime
from heapq import heappush, heapreplace

from cayu.artifacts.base import ArtifactListResult, ArtifactMetadata


class BoundedArtifactListing:
    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._selected: list[tuple[datetime, int, ArtifactMetadata]] = []
        self._total_count = 0

    def add(self, artifact: ArtifactMetadata) -> None:
        # Preserve the existing stable ordering for equal creation timestamps.
        entry = (artifact.created_at, -self._total_count, artifact)
        self._total_count += 1
        if self._limit is None:
            self._selected.append(entry)
        elif len(self._selected) < self._limit:
            heappush(self._selected, entry)
        elif entry[:2] > self._selected[0][:2]:
            heapreplace(self._selected, entry)

    def result(self) -> ArtifactListResult:
        selected = tuple(entry[2] for entry in sorted(self._selected, reverse=True))
        return ArtifactListResult(
            artifacts=selected,
            total_count=self._total_count,
            truncated=len(selected) < self._total_count,
        )
