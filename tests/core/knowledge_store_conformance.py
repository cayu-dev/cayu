from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from cayu.storage import (
    KnowledgeAccessScope,
    KnowledgeChangeKind,
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEmbeddingProjection,
    KnowledgeEntry,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeRevisionConflict,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeStore,
    knowledge_chunk_embedding_identity,
)

CapabilityState = Literal["supported", "not_applicable"]
KnowledgeStoreLifecycle = Literal["process_bound", "reopenable"]
KnowledgeStoreDurability = Literal["ephemeral", "durable"]
KnowledgeStoreScenario = Literal[
    "revision_cas",
    "result_isolation",
    "access_scope",
    "atomic_write",
    "change_outbox",
    "change_page",
    "stable_ordering",
    "lifecycle_guard",
    "projection_readiness",
    "embedding_space",
]


class KnowledgeStoreConformanceFailure(AssertionError):
    """A shared knowledge scenario observed a backend contract violation."""


def require_knowledge_conformance(
    condition: bool,
    *,
    adapter: str,
    scenario: KnowledgeStoreScenario,
    observed: object,
) -> None:
    if condition:
        return
    raise KnowledgeStoreConformanceFailure(
        "KnowledgeStore conformance failed: "
        f"adapter={adapter} scenario={scenario} observed={observed!r}"
    )


@dataclass(frozen=True)
class KnowledgeStoreCapabilityClaim:
    state: CapabilityState
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {"supported", "not_applicable"}:
            raise ValueError(f"Unknown knowledge-store capability state {self.state!r}.")
        if self.state == "supported" and self.reason is not None:
            raise ValueError("Supported knowledge-store capabilities cannot define a reason.")
        if self.state == "not_applicable" and not (self.reason and self.reason.strip()):
            raise ValueError("Not-applicable knowledge-store capabilities require a reason.")

    @classmethod
    def supported(cls) -> KnowledgeStoreCapabilityClaim:
        return cls("supported")

    @classmethod
    def not_applicable(cls, reason: str) -> KnowledgeStoreCapabilityClaim:
        return cls("not_applicable", reason)


@dataclass(frozen=True)
class KnowledgeStoreCapabilities:
    owned_publication: KnowledgeStoreCapabilityClaim
    change_outbox: KnowledgeStoreCapabilityClaim
    index_readiness: KnowledgeStoreCapabilityClaim
    embedding_projections: KnowledgeStoreCapabilityClaim


KnowledgeClock = Callable[[], datetime]
KnowledgeStoreFactory = Callable[
    [KnowledgeAccessScope | None, KnowledgeClock | None],
    Awaitable[KnowledgeStore],
]
KnowledgeStoreReset = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class KnowledgeStoreConformanceRegistration:
    name: str
    store_type: type[KnowledgeStore]
    factory: KnowledgeStoreFactory
    reset: KnowledgeStoreReset
    lifecycle: KnowledgeStoreLifecycle
    durability: KnowledgeStoreDurability
    capabilities: KnowledgeStoreCapabilities

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Knowledge-store conformance registration name must be nonblank.")
        if not isinstance(self.store_type, type) or not issubclass(self.store_type, KnowledgeStore):
            raise TypeError(
                "Knowledge-store conformance registration type must implement KnowledgeStore."
            )
        if not callable(self.factory) or not callable(self.reset):
            raise TypeError(
                "Knowledge-store conformance factories and reset hooks must be callable."
            )
        if self.lifecycle not in {"process_bound", "reopenable"}:
            raise ValueError(f"Unknown knowledge-store lifecycle {self.lifecycle!r}.")
        if self.durability not in {"ephemeral", "durable"}:
            raise ValueError(f"Unknown knowledge-store durability {self.durability!r}.")
        if (self.lifecycle == "reopenable") != (self.durability == "durable"):
            raise ValueError(
                "Reopenable knowledge stores must declare durable storage, and "
                "durable stores must run reopen conformance."
            )
        if not isinstance(self.capabilities, KnowledgeStoreCapabilities):
            raise TypeError("Knowledge-store conformance capabilities must be declared explicitly.")

    @property
    def durable(self) -> bool:
        return self.durability == "durable"

    async def open(
        self,
        *,
        access_scope: KnowledgeAccessScope | None,
        clock: KnowledgeClock | None = None,
    ) -> KnowledgeStore:
        store = await self.factory(access_scope, clock)
        if not isinstance(store, self.store_type):
            raise TypeError(
                f"Knowledge-store registration {self.name!r} returned "
                f"{type(store).__name__}, expected {self.store_type.__name__}."
            )
        return store


def _entry(
    entry_id: str,
    *,
    text: str,
    revision: int = 1,
    namespace: str = "default",
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> KnowledgeEntry:
    effective_timestamp = timestamp or datetime.now(UTC)
    return KnowledgeEntry(
        id=entry_id,
        revision=revision,
        text=text,
        namespace=namespace,
        status=status,
        metadata={} if metadata is None else metadata,
        created_at=effective_timestamp,
        updated_at=effective_timestamp,
    )


def _chunk(entry: KnowledgeEntry, *, chunk_id: str | None = None) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id or f"{entry.id}:r{entry.revision}:0",
        entry_id=entry.id,
        entry_revision=entry.revision,
        chunk_index=0,
        text=entry.text,
    )


async def verify_revision_cas(store: KnowledgeStore, *, adapter: str) -> None:
    entry = _entry("conformance-cas", text="revision one")
    created = await store.create_entry(entry, [_chunk(entry)])
    successors = [
        created.model_copy(
            update={
                "revision": 2,
                "text": text,
            }
        )
        for text in ("revision two alpha", "revision two beta")
    ]
    outcomes = await asyncio.gather(
        *(
            store.append_entry_revision(
                successor,
                [_chunk(successor)],
                expected_revision=1,
            )
            for successor in successors
        ),
        return_exceptions=True,
    )
    successes = [outcome for outcome in outcomes if isinstance(outcome, KnowledgeEntry)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, KnowledgeRevisionConflict)]
    current = await store.get_entry(entry.id)
    require_knowledge_conformance(
        len(successes) == 1 and len(conflicts) == 1 and current == successes[0],
        adapter=adapter,
        scenario="revision_cas",
        observed=(outcomes, current),
    )


async def verify_result_isolation(store: KnowledgeStore, *, adapter: str) -> None:
    entry = _entry(
        "conformance-isolation",
        text="owned snapshot",
        metadata={"nested": {"value": 1}},
    )
    await store.create_entry(entry, [_chunk(entry)])
    loaded = await store.get_entry(entry.id)
    require_knowledge_conformance(
        loaded is not None,
        adapter=adapter,
        scenario="result_isolation",
        observed=loaded,
    )
    assert loaded is not None
    loaded.metadata["nested"]["value"] = 99  # type: ignore[index]
    reloaded = await store.get_entry(entry.id)
    require_knowledge_conformance(
        reloaded is not None and reloaded.metadata == {"nested": {"value": 1}},
        adapter=adapter,
        scenario="result_isolation",
        observed=reloaded,
    )


async def verify_access_scope(store: KnowledgeStore, *, adapter: str) -> None:
    privileged = KnowledgeAccessScope.privileged()
    narrow = KnowledgeAccessScope.for_namespace("allowed")
    entry = _entry(
        "conformance-access",
        text="private namespace",
        namespace="denied",
    )
    await store.create_entry(entry, [_chunk(entry)], access_scope=privileged)
    loaded = await store.get_entry(entry.id, access_scope=narrow)
    require_knowledge_conformance(
        loaded is None,
        adapter=adapter,
        scenario="access_scope",
        observed=loaded,
    )


async def verify_atomic_invalid_write(store: KnowledgeStore, *, adapter: str) -> None:
    owner = _entry("conformance-chunk-owner", text="owns shared chunk")
    target = _entry("conformance-atomic", text="revision one")
    await store.create_entry(owner, [_chunk(owner, chunk_id="conformance-shared-chunk")])
    await store.create_entry(target, [_chunk(target)])
    successor = target.model_copy(update={"revision": 2, "text": "revision two"})
    chunks = [
        _chunk(successor),
        KnowledgeChunk(
            id="conformance-shared-chunk",
            entry_id=successor.id,
            entry_revision=2,
            chunk_index=1,
            text="must conflict",
        ),
    ]
    failure: Exception | None = None
    try:
        await store.append_entry_revision(successor, chunks, expected_revision=1)
    except Exception as error:
        failure = error
    current = await store.get_entry(target.id)
    current_chunks = await store.read_chunks(target.id)
    require_knowledge_conformance(
        isinstance(failure, KnowledgeChunkConflict)
        and current is not None
        and current.revision == 1
        and [chunk.id for chunk in current_chunks] == [f"{target.id}:r1:0"],
        adapter=adapter,
        scenario="atomic_write",
        observed=(failure, current, current_chunks),
    )


async def verify_change_outbox(store: KnowledgeStore, *, adapter: str) -> None:
    before = await store.read_changes()
    entry = _entry("conformance-outbox", text="canonical mutation")
    await store.create_entry(entry, [_chunk(entry)])
    page = await store.read_changes(after_sequence=before.high_water_sequence)
    matching = [change for change in page.changes if change.entry_id == entry.id]
    require_knowledge_conformance(
        len(matching) == 1
        and matching[0].kind is KnowledgeChangeKind.CREATED
        and page.high_water_sequence > before.high_water_sequence,
        adapter=adapter,
        scenario="change_outbox",
        observed=page,
    )


async def verify_change_page(store: KnowledgeStore, *, adapter: str) -> None:
    before = await store.read_changes()
    for suffix in ("a", "b"):
        entry = _entry(f"conformance-page-{suffix}", text=f"page {suffix}")
        await store.create_entry(entry, [_chunk(entry)])
    page = await store.read_changes(after_sequence=before.high_water_sequence, limit=1)
    require_knowledge_conformance(
        len(page.changes) == 1
        and page.truncated
        and page.next_after_sequence == page.changes[0].sequence
        and page.next_after_sequence < page.high_water_sequence,
        adapter=adapter,
        scenario="change_page",
        observed=page,
    )


async def verify_stable_ordering(store: KnowledgeStore, *, adapter: str) -> None:
    timestamp = datetime(2026, 8, 21, 8, 0, tzinfo=UTC)
    for suffix in ("z", "a"):
        entry = _entry(
            f"conformance-order-{suffix}",
            text=f"ordering {suffix}",
            timestamp=timestamp,
        )
        await store.create_entry(entry, [_chunk(entry)])
    result = await store.list_entries(KnowledgeListQuery(limit=100))
    selected = [
        item.entry.id for item in result.entries if item.entry.id.startswith("conformance-order-")
    ]
    require_knowledge_conformance(
        selected == ["conformance-order-a", "conformance-order-z"],
        adapter=adapter,
        scenario="stable_ordering",
        observed=selected,
    )


async def verify_lifecycle_guard(store: KnowledgeStore, *, adapter: str) -> None:
    entry = _entry("conformance-lifecycle", text="active entry")
    await store.create_entry(entry, [_chunk(entry)])
    failure: Exception | None = None
    try:
        await store.transition_entry_status(
            entry.id,
            expected_revision=1,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ARCHIVED,
        )
    except Exception as error:
        failure = error
    current = await store.get_entry(entry.id)
    require_knowledge_conformance(
        type(failure) is ValueError
        and current is not None
        and current.revision == 1
        and current.status is KnowledgeStatus.ACTIVE,
        adapter=adapter,
        scenario="lifecycle_guard",
        observed=(failure, current),
    )


async def verify_projection_readiness(store: KnowledgeStore, *, adapter: str) -> None:
    entry = _entry("conformance-readiness", text="github credential proxy")
    created = await store.create_entry(entry, [_chunk(entry)])
    await store.backfill_embeddings()
    query = KnowledgeQuery(
        text="github credential",
        mode=KnowledgeSearchMode.SEMANTIC,
        min_score=0.0,
    )
    before = await store.search(query)
    successor = created.model_copy(
        update={
            "revision": 2,
            "text": "invoice approval procedure",
        }
    )
    await store.append_entry_revision(
        successor,
        [_chunk(successor)],
        expected_revision=1,
    )
    after = await store.search(query)
    require_knowledge_conformance(
        [hit.entry.revision for hit in before.hits] == [1]
        and after.hits == []
        and len(after.index_coverage) == 1
        and after.index_coverage[0].pending_records == 1,
        adapter=adapter,
        scenario="projection_readiness",
        observed=(before, after),
    )


async def verify_embedding_space_isolation(store: KnowledgeStore, *, adapter: str) -> None:
    entry = _entry("conformance-embedding-space", text="github proxy")
    chunk = _chunk(entry)
    await store.create_entry(entry, [chunk])
    identity = knowledge_chunk_embedding_identity(
        chunk,
        embedding_model="incompatible-model",
        dimensions=3,
    )
    pending = await store.publish_index_readiness(
        KnowledgeIndexReadinessUpdate(
            identity=identity,
            state=KnowledgeIndexState.PENDING,
            attempt_id="conformance-incompatible-space",
        ),
        expected_sequence=None,
        operation_id="conformance-incompatible-space:pending",
    )
    failure: Exception | None = None
    try:
        await store.store_embedding_projections(
            [
                KnowledgeEmbeddingProjection(
                    identity=identity,
                    readiness_sequence=pending.sequence,
                    attempt_id=pending.attempt_id,
                    vector=[1.0, 0.0, 0.0],
                )
            ]
        )
    except Exception as error:
        failure = error
    require_knowledge_conformance(
        type(failure) is ValueError,
        adapter=adapter,
        scenario="embedding_space",
        observed=failure,
    )


CORE_KNOWLEDGE_STORE_SCENARIOS = (
    verify_revision_cas,
    verify_result_isolation,
    verify_atomic_invalid_write,
    verify_change_outbox,
    verify_change_page,
    verify_stable_ordering,
    verify_lifecycle_guard,
)
