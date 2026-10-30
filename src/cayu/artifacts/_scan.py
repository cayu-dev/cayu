"""Bounded metadata batches for complete operator inventory reads."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from cayu._validation import require_clean_nonblank
from cayu.artifacts.base import ArtifactMetadata, ArtifactScope, ArtifactStoreUnavailableError

ARTIFACT_SCAN_BATCH_MAX = 500
ARTIFACT_SCAN_RECORD_MAX = 100_000
ArtifactBatchConsumer = Callable[[tuple[ArtifactMetadata, ...]], Awaitable[None]]


@dataclass(frozen=True)
class ArtifactScanFilter:
    scope: ArtifactScope | None = None
    session_id: str | None = None
    agent_name: str | None = None
    environment_name: str | None = None

    def __post_init__(self) -> None:
        if self.scope is not None and type(self.scope) is not ArtifactScope:
            raise TypeError("Artifact scope must be an ArtifactScope.")
        for name in ("session_id", "agent_name", "environment_name"):
            value = getattr(self, name)
            if value is not None:
                require_clean_nonblank(value, name)

    def matches(self, artifact: ArtifactMetadata) -> bool:
        return all(
            getattr(self, name) is None or getattr(self, name) == getattr(artifact, name)
            for name in ("scope", "session_id", "agent_name", "environment_name")
        )


def validate_scan_bounds(batch_size: int, max_records: int) -> None:
    if type(batch_size) is not int or not 1 <= batch_size <= ARTIFACT_SCAN_BATCH_MAX:
        raise ValueError("Artifact scan batch size must be between 1 and 500.")
    if type(max_records) is not int or not 1 <= max_records <= ARTIFACT_SCAN_RECORD_MAX:
        raise ValueError("Artifact scan record bound must be between 1 and 100000.")


class ArtifactScan:
    def __init__(
        self,
        consume: ArtifactBatchConsumer,
        filters: ArtifactScanFilter,
        *,
        batch_size: int,
        max_records: int,
    ) -> None:
        validate_scan_bounds(batch_size, max_records)
        if not callable(consume):
            raise TypeError("Artifact scan requires a batch consumer.")
        self.consume = consume
        self.filters = filters
        self.batch_size = batch_size
        self.max_records = max_records
        self.scanned = 0
        self.delivered = 0
        self.pending: list[ArtifactMetadata] = []

    async def add(self, artifact: ArtifactMetadata) -> None:
        self.scanned += 1
        if self.scanned > self.max_records:
            raise ArtifactStoreUnavailableError("Artifact inventory exceeds the scan record bound.")
        if self.filters.matches(artifact):
            self.pending.append(artifact)
            if len(self.pending) == self.batch_size:
                await self.flush()

    async def flush(self) -> None:
        if self.pending:
            batch = tuple(self.pending)
            self.pending.clear()
            await self.consume(batch)
            self.delivered += len(batch)
