from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from tests.core.knowledge_store_conformance import (
    KnowledgeStoreConformanceFailure,
    verify_access_scope,
    verify_atomic_invalid_write,
    verify_bounded_entry_read,
    verify_change_outbox,
    verify_change_page,
    verify_embedding_space_isolation,
    verify_lifecycle_guard,
    verify_projection_readiness,
    verify_result_isolation,
    verify_revision_cas,
    verify_stable_ordering,
)

from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.storage import (
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeRevisionConflict,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
)

_CoreDefect = Literal[
    "lost-update",
    "mutation-leakage",
    "access-widening",
    "partial-write",
    "missing-outbox",
    "dishonest-counts",
    "unstable-ordering",
    "invalid-lifecycle",
]


class _SeededBrokenKnowledgeStore(InMemoryKnowledgeStore):
    """Plausible broken adapters proving each canonical scenario is discriminating."""

    def __init__(self, defect: _CoreDefect) -> None:
        super().__init__(
            access_scope=(
                None if defect == "access-widening" else KnowledgeAccessScope.privileged()
            )
        )
        self.defect = defect

    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        created = await super().create_entry(
            entry,
            chunks,
            evidence=evidence,
            access_scope=access_scope,
        )
        if self.defect == "missing-outbox":
            change = self._changes.pop()
            self._changes_by_sequence.pop(change.sequence)
            self._change_access.pop(change.sequence)
        return created

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        if self.defect == "partial-write" and chunks is not None and len(chunks) > 1:
            await super().append_entry_revision(
                entry,
                chunks[:1],
                expected_revision=expected_revision,
                evidence=evidence,
                access_scope=access_scope,
            )
            raise ValueError("simulated adapter failure after a partial commit")
        try:
            return await super().append_entry_revision(
                entry,
                chunks,
                expected_revision=expected_revision,
                evidence=evidence,
                access_scope=access_scope,
            )
        except KnowledgeRevisionConflict:
            if self.defect != "lost-update":
                raise
            current = await super().get_entry(entry.id, access_scope=access_scope)
            assert current is not None
            return current

    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        if self.defect == "mutation-leakage":
            return self._entry_revision(entry_id, revision)
        if self.defect == "access-widening":
            return await super().get_entry(
                entry_id,
                revision=revision,
                max_bytes=max_bytes,
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return await super().get_entry(
            entry_id,
            revision=revision,
            max_bytes=max_bytes,
            access_scope=access_scope,
        )

    async def read_changes(self, **kwargs):
        result = await super().read_changes(**kwargs)
        if self.defect != "dishonest-counts" or not result.truncated:
            return result
        return result.model_copy(
            update={
                "truncated": False,
                "next_after_sequence": result.high_water_sequence,
            }
        )

    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        result = await super().list_entries(query, access_scope=access_scope)
        if self.defect != "unstable-ordering":
            return result
        return result.model_copy(update={"entries": list(reversed(result.entries))})

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
        if self.defect == "invalid-lifecycle":
            current = await super().get_entry(entry_id, access_scope=access_scope)
            assert current is not None
            from_status = current.status
        return await super().transition_entry_status(
            entry_id,
            expected_revision=expected_revision,
            access_scope=access_scope,
            from_status=from_status,
            to_status=to_status,
            expected_namespace=expected_namespace,
            expected_labels=expected_labels,
        )


class _KeywordEmbeddingProvider(TextEmbeddingProvider):
    name = "knowledge-conformance"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(
                    index=index,
                    vector=(
                        [1.0, 0.0, 0.0]
                        if "github" in text.casefold() or "credential" in text.casefold()
                        else [0.0, 1.0, 0.0]
                    ),
                )
                for index, text in enumerate(request.texts)
            ],
        )


_EmbeddingDefect = Literal["stale-projection-hit", "mixed-embedding-space"]


class _SeededBrokenEmbeddingKnowledgeStore(InMemoryEmbeddingKnowledgeStore):
    def __init__(self, defect: _EmbeddingDefect) -> None:
        super().__init__(
            embedding_provider=_KeywordEmbeddingProvider(),
            embedding_model="conformance-model",
            embedding_dimensions=3,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        self.defect = defect
        self._cached_semantic_result: KnowledgeSearchResult | None = None

    async def search(
        self,
        query,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        if (
            self.defect == "stale-projection-hit"
            and query.mode is KnowledgeSearchMode.SEMANTIC
            and self._cached_semantic_result is not None
        ):
            return self._cached_semantic_result
        result = await super().search(query, access_scope=access_scope)
        if self.defect == "stale-projection-hit" and query.mode is KnowledgeSearchMode.SEMANTIC:
            self._cached_semantic_result = result
        return result

    def _embedding_identity_matches_configuration(self, identity) -> bool:
        if self.defect == "mixed-embedding-space":
            return True
        return super()._embedding_identity_matches_configuration(identity)

    def _embedding_identity_is_current(self, identity) -> bool:
        if self.defect == "mixed-embedding-space":
            return True
        return super()._embedding_identity_is_current(identity)


@pytest.mark.parametrize(
    ("defect", "scenario"),
    (
        ("lost-update", verify_revision_cas),
        ("mutation-leakage", verify_result_isolation),
        ("mutation-leakage", verify_bounded_entry_read),
        ("access-widening", verify_access_scope),
        ("partial-write", verify_atomic_invalid_write),
        ("missing-outbox", verify_change_outbox),
        ("dishonest-counts", verify_change_page),
        ("unstable-ordering", verify_stable_ordering),
        ("invalid-lifecycle", verify_lifecycle_guard),
    ),
)
def test_seeded_broken_knowledge_store_is_rejected(defect, scenario) -> None:
    store = _SeededBrokenKnowledgeStore(defect)
    with pytest.raises(KnowledgeStoreConformanceFailure):
        asyncio.run(scenario(store, adapter=f"broken-{defect}"))


@pytest.mark.parametrize(
    ("defect", "scenario"),
    (
        ("stale-projection-hit", verify_projection_readiness),
        ("mixed-embedding-space", verify_embedding_space_isolation),
    ),
)
def test_seeded_broken_embedding_store_is_rejected(defect, scenario) -> None:
    store = _SeededBrokenEmbeddingKnowledgeStore(defect)
    with pytest.raises(KnowledgeStoreConformanceFailure):
        asyncio.run(scenario(store, adapter=f"broken-{defect}"))


def test_reference_embedding_store_passes_projection_scenarios() -> None:
    async def run() -> None:
        readiness_store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=_KeywordEmbeddingProvider(),
            embedding_model="conformance-model",
            embedding_dimensions=3,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        await verify_projection_readiness(readiness_store, adapter="memory-embedding")

        space_store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=_KeywordEmbeddingProvider(),
            embedding_model="conformance-model",
            embedding_dimensions=3,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        await verify_embedding_space_isolation(space_store, adapter="memory-embedding")

    asyncio.run(run())
