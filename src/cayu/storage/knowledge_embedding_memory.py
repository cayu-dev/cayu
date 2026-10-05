"""Optional embedding search and indexing for in-memory knowledge storage."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime
from itertools import chain
from math import sqrt
from typing import TypedDict
from uuid import uuid4

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.embeddings import TextEmbeddingProvider, TextEmbeddingRequest, copy_text_embedding_result
from cayu.knowledge import (
    _embedding_backfill,
    _query_rules,
    _retrieval_results,
    _search_scoring,
)
from cayu.knowledge._access_rules import (
    _knowledge_scope_allows_entry,
)
from cayu.knowledge.access import runtime_knowledge_operation
from cayu.knowledge.changes import (
    KnowledgeChangeConsumerConflict,
    KnowledgeChangeKind,
    _validate_knowledge_change_limit,
    _validate_knowledge_change_sequence,
)
from cayu.knowledge.indexing import (
    DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
    KNOWLEDGE_CHUNK_TEXT_GENERATOR,
    KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
    KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
    KNOWLEDGE_CHUNK_TEXT_PROJECTION,
    KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
    KnowledgeEmbeddingBackfillResult,
    KnowledgeEmbeddingIdentity,
    KnowledgeEmbeddingProjection,
    KnowledgeEmbeddingProjectionConflict,
    KnowledgeEmbeddingProjectionWriteResult,
    KnowledgeEmbeddingWorkerResult,
    KnowledgeIndexCoverage,
    KnowledgeIndexReadiness,
    KnowledgeIndexReadinessConflict,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    _copy_knowledge_embedding_projections,
    _knowledge_embedding_identity_sha256,
    _knowledge_embedding_vector_sha256,
    _validate_knowledge_embedding_work_record_limit,
    _validate_knowledge_index_sequence,
    copy_knowledge_embedding_identity,
    knowledge_chunk_embedding_identity,
)
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_LIMIT,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    _validate_positive_int,
    copy_knowledge_chunk,
    copy_knowledge_entry,
    copy_knowledge_revision_refs,
)
from cayu.knowledge.scopes import (
    KnowledgeAccessScope,
)
from cayu.knowledge.search import (
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    _knowledge_query_terms,
    _query_terms_have_positive_terms,
    _validate_nonnegative_float,
    _validate_unit_float,
    copy_knowledge_list_query,
    copy_knowledge_query,
)
from cayu.storage._knowledge_closure import (
    KnowledgeClosureInventory,
)
from cayu.storage.knowledge_memory import InMemoryKnowledgeStore


class _StoredChunkEmbedding(TypedDict):
    identity: KnowledgeEmbeddingIdentity
    vector: list[float]
    vector_sha256: str
    readiness_sequence: int
    attempt_id: str


class InMemoryEmbeddingKnowledgeStore(InMemoryKnowledgeStore):
    """In-memory knowledge store with opt-in embedding search.

    This backend is intended for tests, demos, and small single-process apps. It
    keeps vectors in memory and does not persist them. Durable production vector
    search should use a store with a real vector index.
    """

    resource_knowledge_access_version = 1

    def _add_closure_projections(
        self, inventory: KnowledgeClosureInventory, revisions: set[tuple[str, int]]
    ) -> None:
        # Historical attempts remain real stored derivatives. Current/history
        # overlap is identity-deduplicated by the inventory collector.
        for history in chain((self._chunk_embeddings,), self._chunk_embedding_history.values()):
            for stored in history.values():
                identity = copy_knowledge_embedding_identity(stored["identity"])
                if (identity.entry_id, identity.entry_revision) not in revisions:
                    continue
                inventory.add(
                    "knowledge_projections",
                    {
                        "identity": identity.model_dump(mode="json"),
                        "attempt_id": stored["attempt_id"],
                        "readiness_sequence": stored["readiness_sequence"],
                        "vector_sha256": stored["vector_sha256"],
                    },
                )

    def __init__(
        self,
        *,
        embedding_provider: TextEmbeddingProvider,
        embedding_model: str,
        embedding_dimensions: int,
        entries: list[KnowledgeEntry] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        clock: Callable[[], datetime] | None = None,
        hybrid_keyword_weight: float = 0.35,
        semantic_min_score: float = 0.55,
    ) -> None:
        if not isinstance(embedding_provider, TextEmbeddingProvider):
            raise TypeError("embedding_provider must implement TextEmbeddingProvider.")
        self.embedding_provider = embedding_provider
        self.embedding_model = require_clean_nonblank(embedding_model, "embedding_model")
        _validate_positive_int(embedding_dimensions, "embedding_dimensions")
        self.embedding_dimensions = embedding_dimensions
        self.hybrid_keyword_weight = _validate_nonnegative_float(
            hybrid_keyword_weight,
            "hybrid_keyword_weight",
        )
        self.semantic_min_score = _validate_unit_float(
            semantic_min_score,
            "semantic_min_score",
        )
        self._chunk_embeddings: dict[str, _StoredChunkEmbedding] = {}
        self._chunk_embedding_history: dict[str, dict[str, _StoredChunkEmbedding]] = {}
        super().__init__(entries, access_scope=access_scope, clock=clock)

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        return (
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.KEYWORD,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        )

    @runtime_knowledge_operation("modify")
    async def process_embedding_changes(
        self,
        consumer_id: str,
        worker_id: str,
        *,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        record_limit: int = DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
        lease_seconds: float = 300.0,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEmbeddingWorkerResult:
        """Consume a bounded page of canonical changes into the embedding index."""

        _validate_knowledge_change_limit(limit)
        _validate_knowledge_embedding_work_record_limit(record_limit)
        scope = self._operation_access_scope(access_scope)
        claimed_changes = 0
        acknowledged_changes = 0
        indexed_records = 0
        failed_records = 0
        removed_records = 0
        processed_records = 0
        for _ in range(limit):
            claim = await self.claim_change(
                consumer_id,
                worker_id,
                lease_seconds=lease_seconds,
                access_scope=scope,
            )
            if claim is None:
                break
            claimed_changes += 1
            try:
                if claim.change.kind is KnowledgeChangeKind.RELATION_PUBLISHED:
                    await self.acknowledge_change(claim, access_scope=scope)
                    acknowledged_changes += 1
                    continue
                current = self._current_entry(claim.change.entry_id)
                remaining = record_limit - processed_records
                if current is None or current.status is KnowledgeStatus.DELETED:
                    removed, cleanup_truncated = self._drop_entry_embeddings(
                        claim.change.entry_id,
                        limit=remaining,
                    )
                    removed_records += removed
                    processed_records += removed
                elif (
                    current.revision != claim.change.entry_revision
                    or not _knowledge_scope_allows_entry(scope, current)
                ):
                    removed, cleanup_truncated = self._drop_stale_entry_embeddings(
                        current.id,
                        limit=remaining,
                    )
                    removed_records += removed
                    processed_records += removed
                else:
                    chunks = [
                        copy_knowledge_chunk(chunk)
                        for chunk in self._chunks.get(
                            (current.id, current.revision),
                            [],
                        )
                    ]
                    chunks, truncated = self._embedding_work_candidates(
                        chunks,
                        limit=record_limit - processed_records,
                    )
                    indexed, failed = await self._index_chunks_with_readiness(
                        chunks,
                        attempt_id=claim.claim_id,
                        operation_prefix=f"kidx:{claim.claim_id}",
                        access_scope=scope,
                    )
                    indexed_records += indexed
                    failed_records += failed
                    processed_records += len(chunks)
                    removed, cleanup_truncated = self._drop_stale_entry_embeddings(
                        current.id,
                        limit=record_limit - processed_records,
                    )
                    removed_records += removed
                    processed_records += removed
                    cleanup_truncated = truncated or cleanup_truncated
                if cleanup_truncated:
                    await self.release_change(claim, access_scope=scope)
                    break
                await self.acknowledge_change(claim, access_scope=scope)
                acknowledged_changes += 1
                if processed_records >= record_limit:
                    break
            except Exception:
                with suppress(KnowledgeChangeConsumerConflict):
                    await self.release_change(claim, access_scope=scope)
                raise
        return KnowledgeEmbeddingWorkerResult(
            consumer_id=consumer_id,
            worker_id=worker_id,
            claimed_changes=claimed_changes,
            acknowledged_changes=acknowledged_changes,
            indexed_records=indexed_records,
            failed_records=failed_records,
            removed_records=removed_records,
            limit=limit,
            processed_records=processed_records,
            record_limit=record_limit,
        )

    def _embedding_work_candidates(
        self,
        chunks: list[KnowledgeChunk],
        *,
        limit: int,
    ) -> tuple[list[KnowledgeChunk], bool]:
        candidates: list[KnowledgeChunk] = []
        for chunk in chunks:
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            readiness = self._index_readiness_by_identity.get(identity_sha256)
            stored = self._chunk_embeddings.get(identity_sha256)
            if (
                readiness is not None
                and readiness.state is KnowledgeIndexState.READY
                and stored is not None
                and stored["identity"] == identity
            ):
                continue
            candidates.append(copy_knowledge_chunk(chunk))
            if len(candidates) > limit:
                break
        return candidates[:limit], len(candidates) > limit

    @runtime_knowledge_operation("modify")
    async def backfill_embeddings(
        self,
        query: KnowledgeListQuery | None = None,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        limit: int = DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
        refresh_existing: bool = False,
        cursor: str | None = None,
    ) -> KnowledgeEmbeddingBackfillResult:
        """Repair or refresh one bounded deterministic page of current chunk vectors."""

        _validate_knowledge_embedding_work_record_limit(limit, field_name="limit")
        if type(refresh_existing) is not bool:
            raise ValueError("`refresh_existing` must be a boolean.")
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_list_query(query or KnowledgeListQuery())
        fingerprint = _embedding_backfill._knowledge_embedding_backfill_fingerprint(
            knowledge_query,
            scope,
            refresh_existing=refresh_existing,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
        )
        after = _embedding_backfill._decode_knowledge_embedding_backfill_cursor(
            cursor,
            fingerprint=fingerprint,
        )
        after_key = (
            None
            if after is None
            else _embedding_backfill._knowledge_embedding_backfill_sort_key(
                importance=after.importance,
                updated_at=after.updated_at,
                entry_id=after.entry_id,
                chunk_index=after.chunk_index,
                chunk_id=after.chunk_id,
            )
        )
        entries = [
            entry
            for entry_id in self._entries
            if (entry := self._current_entry(entry_id)) is not None
            if _knowledge_scope_allows_entry(scope, entry)
            if _query_rules._entry_matches_list_query(entry, knowledge_query)
        ]
        entries.sort(
            key=lambda entry: _embedding_backfill._knowledge_embedding_backfill_sort_key(
                importance=entry.importance or 0.0,
                updated_at=entry.updated_at,
                entry_id=entry.id,
                chunk_index=0,
                chunk_id="",
            )
        )
        candidates: list[tuple[KnowledgeEntry, KnowledgeChunk]] = []
        for entry in entries:
            for chunk in sorted(
                self._chunks.get((entry.id, entry.revision), []),
                key=lambda item: (item.chunk_index, item.id),
            ):
                candidate_key = _embedding_backfill._knowledge_embedding_backfill_sort_key(
                    importance=entry.importance or 0.0,
                    updated_at=entry.updated_at,
                    entry_id=entry.id,
                    chunk_index=chunk.chunk_index,
                    chunk_id=chunk.id,
                )
                if after_key is not None and candidate_key <= after_key:
                    continue
                identity = knowledge_chunk_embedding_identity(
                    chunk,
                    embedding_model=self.embedding_model,
                    dimensions=self.embedding_dimensions,
                )
                identity_sha256 = _knowledge_embedding_identity_sha256(identity)
                readiness = self._index_readiness_by_identity.get(identity_sha256)
                stored = self._chunk_embeddings.get(identity_sha256)
                if (
                    not refresh_existing
                    and readiness is not None
                    and readiness.state is KnowledgeIndexState.READY
                    and stored is not None
                    and stored["identity"] == identity
                ):
                    continue
                candidates.append((entry, copy_knowledge_chunk(chunk)))
                if len(candidates) > limit:
                    break
            if len(candidates) > limit:
                break
        page = candidates[:limit]
        chunks = [chunk for _, chunk in page]
        next_cursor = (
            _embedding_backfill._encode_knowledge_embedding_backfill_cursor(
                fingerprint=fingerprint,
                importance=page[-1][0].importance or 0.0,
                updated_at=page[-1][0].updated_at,
                chunk=page[-1][1],
            )
            if len(candidates) > limit and page
            else None
        )
        attempt_id = f"kbackfill_{uuid4().hex}"
        indexed, failed = await self._index_chunks_with_readiness(
            chunks,
            attempt_id=attempt_id,
            operation_prefix=f"kidx:{attempt_id}",
            access_scope=scope,
            refresh_existing=refresh_existing,
        )
        return KnowledgeEmbeddingBackfillResult(
            scanned_records=len(chunks),
            indexed_records=indexed,
            failed_records=failed,
            skipped_records=len(chunks) - indexed - failed,
            limit=limit,
            refresh_existing=refresh_existing,
            next_cursor=next_cursor,
        )

    @runtime_knowledge_operation("modify")
    async def store_embedding_projections(
        self,
        projections: list[KnowledgeEmbeddingProjection],
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEmbeddingProjectionWriteResult:
        """Persist vectors only while their exact authorized attempt is pending."""

        scope = self._operation_access_scope(access_scope)
        copied = _copy_knowledge_embedding_projections(projections)
        accepted: list[tuple[str, KnowledgeEmbeddingProjection, str]] = []
        writes: list[tuple[str, KnowledgeEmbeddingProjection, str]] = []
        for projection in copied:
            identity = projection.identity
            if not self._embedding_identity_matches_configuration(identity):
                raise ValueError("Embedding projection identity does not match this store.")
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            readiness = self._index_readiness_by_identity.get(identity_sha256)
            if (
                readiness is None
                or readiness.sequence != projection.readiness_sequence
                or readiness.state is not KnowledgeIndexState.PENDING
                or readiness.attempt_id != projection.attempt_id
                or not self._index_identity_is_accessible(scope, identity)
                or not self._embedding_identity_is_current(identity)
            ):
                continue
            vector_sha256 = _knowledge_embedding_vector_sha256(projection.vector)
            stored = self._chunk_embeddings.get(identity_sha256)
            if (
                stored is not None
                and stored["readiness_sequence"] == projection.readiness_sequence
                and stored["attempt_id"] == projection.attempt_id
            ):
                if stored["vector_sha256"] != vector_sha256:
                    raise KnowledgeEmbeddingProjectionConflict("attempt_vector_conflict")
            else:
                writes.append((identity_sha256, projection, vector_sha256))
            accepted.append((identity_sha256, projection, vector_sha256))
        for identity_sha256, projection, vector_sha256 in writes:
            stored_embedding: _StoredChunkEmbedding = {
                "identity": copy_knowledge_embedding_identity(projection.identity),
                "vector": list(projection.vector),
                "vector_sha256": vector_sha256,
                "readiness_sequence": projection.readiness_sequence,
                "attempt_id": projection.attempt_id,
            }
            self._chunk_embeddings[identity_sha256] = stored_embedding
            self._chunk_embedding_history.setdefault(identity_sha256, {})[projection.attempt_id] = (
                stored_embedding
            )
        return KnowledgeEmbeddingProjectionWriteResult(
            submitted_records=len(copied),
            stored_identities=[projection.identity for _, projection, _ in accepted],
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
        return await self._embedding_search(
            knowledge_query,
            scope,
            revision_keys=None,
            knowledge_sequence=None,
            index_readiness_sequence=None,
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
        return await self._embedding_search(
            knowledge_query,
            scope,
            revision_keys=None,
            knowledge_sequence=knowledge_sequence,
            index_readiness_sequence=index_readiness_sequence,
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
        return await self._embedding_search(
            knowledge_query,
            scope,
            revision_keys={(item.entry_id, item.revision) for item in references},
            knowledge_sequence=knowledge_sequence,
            index_readiness_sequence=index_readiness_sequence,
        )

    async def _embedding_search(
        self,
        knowledge_query: KnowledgeQuery,
        scope: KnowledgeAccessScope,
        *,
        revision_keys: set[tuple[str, int]] | None,
        knowledge_sequence: int | None,
        index_readiness_sequence: int | None,
    ) -> KnowledgeSearchResult:
        terms = _knowledge_query_terms(knowledge_query)
        if knowledge_query.mode is KnowledgeSearchMode.KEYWORD or (
            knowledge_query.mode is KnowledgeSearchMode.AUTO
            and not _query_terms_have_positive_terms(terms)
        ):
            return self._keyword_search(
                knowledge_query,
                scope,
                revision_keys=revision_keys,
                through_change_sequence=knowledge_sequence,
            )
        if knowledge_query.mode not in {
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        }:
            raise ValueError(
                "InMemoryEmbeddingKnowledgeStore supports auto, keyword, semantic, and "
                "hybrid search modes."
            )
        candidates: list[tuple[KnowledgeEntry, list[KnowledgeChunk]]] = []
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
            if knowledge_sequence is not None and (
                self._revision_materialization_sequences.get((entry.id, entry.revision)) is None
                or self._revision_materialization_sequences[(entry.id, entry.revision)]
                > knowledge_sequence
            ):
                continue
            chunks = self._chunks.get((entry.id, entry.revision), [])
            if not _knowledge_scope_allows_entry(scope, entry):
                continue
            if not _query_rules._entry_matches_query(entry, knowledge_query):
                continue
            if _search_scoring._entry_matches_none_terms(entry, chunks, terms):
                continue
            candidates.append(
                (
                    copy_knowledge_entry(entry),
                    [copy_knowledge_chunk(chunk) for chunk in chunks],
                )
            )
        candidate_chunks = [chunk for _, chunks in candidates for chunk in chunks]
        candidate_embeddings, coverage = self._ready_embeddings_and_coverage(
            candidate_chunks,
            access_scope=scope,
            through_sequence=index_readiness_sequence,
        )
        if not candidates:
            return KnowledgeSearchResult(
                query=knowledge_query,
                hits=[],
                truncated=False,
                limit=knowledge_query.limit,
                max_bytes=knowledge_query.max_bytes,
                total_hits_known=0,
                index_coverage=[coverage],
            )
        semantic_query_text = _query_rules._semantic_query_text(knowledge_query)
        query_vector = (
            await self._embed_query(knowledge_query, semantic_query_text)
            if candidate_embeddings
            else None
        )
        semantic_min_score = (
            self.semantic_min_score
            if knowledge_query.min_score is None
            else knowledge_query.min_score
        )
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]
        ] = []
        for entry, chunks in candidates:
            semantic_score, chunk = (
                (None, None)
                if query_vector is None
                else self._best_semantic_score(
                    chunks,
                    candidate_embeddings,
                    query_vector,
                )
            )
            semantic_matched = False
            score = 0.0
            semantic_reason = "semantic projection not ready"
            reason = semantic_reason
            preview_text = entry.text
            score_normalized: float | None = None
            if semantic_score is not None:
                normalized_semantic = _normalize_cosine_similarity(semantic_score)
                semantic_matched = normalized_semantic >= semantic_min_score
                score = normalized_semantic if semantic_matched else 0.0
                semantic_reason = (
                    "semantic chunk match" if chunk is not None else "semantic entry match"
                )
                reason = semantic_reason
                preview_text = chunk.text if chunk is not None else entry.text
                score_normalized = normalized_semantic if semantic_matched else None
            if knowledge_query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
                keyword_score, keyword_chunk, keyword_reason, keyword_preview = (
                    _search_scoring._score_entry(
                        entry,
                        chunks,
                        knowledge_query,
                    )
                )
                if keyword_score > 0:
                    keyword_boost = min(keyword_score, 10.0) / 10.0
                    score += self.hybrid_keyword_weight * keyword_boost
                    if keyword_chunk is not None:
                        chunk = keyword_chunk
                    reason = (
                        f"hybrid {semantic_reason}; {keyword_reason}"
                        if semantic_matched
                        else f"hybrid keyword match; {keyword_reason}"
                    )
                    preview_text = keyword_preview
            elif not semantic_matched:
                continue
            if score <= 0:
                continue
            scored.append((score, entry, chunk, reason, preview_text, score_normalized, True))
        scored.sort(
            key=lambda item: (
                -item[0],
                -(item[1].importance or 0.0),
                -item[1].updated_at.timestamp(),
                item[1].id,
            )
        )
        score_kind = (
            "inmemory_semantic"
            if knowledge_query.mode is KnowledgeSearchMode.SEMANTIC
            else "inmemory_hybrid"
        )
        return _retrieval_results._search_result_from_scored_embeddings(
            scored,
            knowledge_query,
            score_kind=score_kind,
            index_coverage=[coverage],
        )

    async def _index_chunks_with_readiness(
        self,
        chunks: list[KnowledgeChunk],
        *,
        attempt_id: str,
        operation_prefix: str,
        access_scope: KnowledgeAccessScope,
        refresh_existing: bool = False,
    ) -> tuple[int, int]:
        """Index exact current identities and fence every visible vector with readiness."""

        pending: list[
            tuple[KnowledgeChunk, KnowledgeEmbeddingIdentity, KnowledgeIndexReadiness]
        ] = []
        indexed = 0
        for chunk in chunks:
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            current = await self.load_index_readiness(
                identity,
                access_scope=access_scope,
            )
            stored = self._chunk_embeddings.get(identity_sha256)
            if (
                not refresh_existing
                and current is not None
                and current.state is KnowledgeIndexState.READY
                and stored is not None
                and stored["identity"] == identity
            ):
                continue
            if (
                current is not None
                and current.state is KnowledgeIndexState.PENDING
                and current.attempt_id == attempt_id
            ):
                readiness = current
            else:
                readiness = await self.publish_index_readiness(
                    KnowledgeIndexReadinessUpdate(
                        identity=identity,
                        state=KnowledgeIndexState.PENDING,
                        attempt_id=attempt_id,
                    ),
                    expected_sequence=None if current is None else current.sequence,
                    operation_id=f"{operation_prefix}:{identity_sha256}:pending",
                    access_scope=access_scope,
                )
            if not refresh_existing and stored is not None and stored["identity"] == identity:
                if not self._chunk_is_current(chunk):
                    continue
                await self.publish_index_readiness(
                    KnowledgeIndexReadinessUpdate(
                        identity=identity,
                        state=KnowledgeIndexState.READY,
                        attempt_id=readiness.attempt_id,
                    ),
                    expected_sequence=readiness.sequence,
                    operation_id=f"{operation_prefix}:{identity_sha256}:ready",
                    access_scope=access_scope,
                )
                indexed += 1
                continue
            pending.append((chunk, identity, readiness))
        if not pending:
            return indexed, 0

        try:
            from cayu.resource_access import require_dispatch

            await require_dispatch()
            result = copy_text_embedding_result(
                await self.embedding_provider.embed_texts(
                    TextEmbeddingRequest(
                        model=self.embedding_model,
                        texts=[chunk.text for chunk, _, _ in pending],
                        dimensions=self.embedding_dimensions,
                    )
                )
            )
            if result.model != self.embedding_model:
                raise ValueError("Embedding provider returned an unexpected model identity.")
            if len(result.embeddings) != len(pending):
                raise ValueError("Embedding provider returned a different number of embeddings.")
            by_index = {embedding.index: embedding for embedding in result.embeddings}
            if len(by_index) != len(result.embeddings):
                raise ValueError("Embedding provider returned duplicate indexes.")
            for index in range(len(pending)):
                embedding = by_index.get(index)
                if embedding is None:
                    raise ValueError("Embedding provider did not return every requested index.")
                self._validate_embedding_dimension(embedding.vector)
        except Exception:
            failed = 0
            for chunk, identity, readiness in pending:
                if not self._chunk_is_current(chunk):
                    continue
                identity_sha256 = _knowledge_embedding_identity_sha256(identity)
                try:
                    await self.publish_index_readiness(
                        KnowledgeIndexReadinessUpdate(
                            identity=identity,
                            state=KnowledgeIndexState.FAILED,
                            attempt_id=readiness.attempt_id,
                            failure_code="embedding_provider_error",
                        ),
                        expected_sequence=readiness.sequence,
                        operation_id=f"{operation_prefix}:{identity_sha256}:failed",
                        access_scope=access_scope,
                    )
                except KnowledgeIndexReadinessConflict as conflict:
                    if conflict.reason not in {"stale_identity", "stale_sequence"}:
                        raise
                else:
                    failed += 1
            return indexed, failed

        projection_result = await self.store_embedding_projections(
            [
                KnowledgeEmbeddingProjection(
                    identity=identity,
                    readiness_sequence=readiness.sequence,
                    attempt_id=readiness.attempt_id,
                    vector=by_index[index].vector,
                )
                for index, (_, identity, readiness) in enumerate(pending)
            ],
            access_scope=access_scope,
        )
        stored_identity_sha256s = {
            _knowledge_embedding_identity_sha256(identity)
            for identity in projection_result.stored_identities
        }
        for _, identity, readiness in pending:
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            if identity_sha256 not in stored_identity_sha256s:
                continue
            try:
                await self.publish_index_readiness(
                    KnowledgeIndexReadinessUpdate(
                        identity=identity,
                        state=KnowledgeIndexState.READY,
                        attempt_id=readiness.attempt_id,
                    ),
                    expected_sequence=readiness.sequence,
                    operation_id=f"{operation_prefix}:{identity_sha256}:ready",
                    access_scope=access_scope,
                )
            except KnowledgeIndexReadinessConflict as conflict:
                if conflict.reason not in {"stale_identity", "stale_sequence"}:
                    raise
                continue
            indexed += 1
        return indexed, 0

    def _ready_embeddings_and_coverage(
        self,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope,
        through_sequence: int | None = None,
    ) -> tuple[dict[str, list[float]], KnowledgeIndexCoverage]:
        embeddings: dict[str, list[float]] = {}
        ready = 0
        failed = 0
        eligible_identity_sha256s: set[str] = set()
        for chunk in chunks:
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            eligible_identity_sha256s.add(identity_sha256)
            readiness = self._readiness_at_sequence(identity_sha256, through_sequence)
            stored = (
                self._chunk_embeddings.get(identity_sha256)
                if through_sequence is None or readiness is None
                else max(
                    (
                        embedding
                        for embedding in self._chunk_embedding_history.get(
                            identity_sha256, {}
                        ).values()
                        if embedding["readiness_sequence"] <= readiness.sequence
                    ),
                    key=lambda embedding: embedding["readiness_sequence"],
                    default=None,
                )
            )
            if (
                readiness is not None
                and (through_sequence is None or readiness.sequence <= through_sequence)
                and readiness.state is KnowledgeIndexState.READY
                and stored is not None
                and stored["identity"] == identity
            ):
                embeddings[chunk.id] = list(stored["vector"])
                ready += 1
            elif (
                readiness is not None
                and (through_sequence is None or readiness.sequence <= through_sequence)
                and readiness.state is KnowledgeIndexState.FAILED
            ):
                failed += 1
        eligible = len(chunks)
        high_water = max(
            (
                item.sequence
                for item in self._index_readiness
                if _knowledge_embedding_identity_sha256(item.identity) in eligible_identity_sha256s
                and (through_sequence is None or item.sequence <= through_sequence)
                and self._index_identity_is_accessible(access_scope, item.identity)
            ),
            default=0,
        )
        pending = eligible - ready - failed
        return embeddings, KnowledgeIndexCoverage(
            projection_type=KNOWLEDGE_CHUNK_TEXT_PROJECTION,
            embedding_model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            preprocessing_version=KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
            generator=KNOWLEDGE_CHUNK_TEXT_GENERATOR,
            generator_version=KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
            index_representation_version=KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
            eligible_records=eligible,
            ready_records=ready,
            pending_records=pending,
            failed_records=failed,
            high_water_sequence=high_water,
            complete=ready == eligible and pending == 0 and failed == 0,
        )

    def _readiness_at_sequence(
        self,
        identity_sha256: str,
        through_sequence: int | None,
    ) -> KnowledgeIndexReadiness | None:
        if through_sequence is None:
            return self._index_readiness_by_identity.get(identity_sha256)
        history = self._index_readiness_history_by_identity.get(identity_sha256, [])
        position = bisect_right(
            history,
            through_sequence,
            key=lambda readiness: readiness.sequence,
        )
        return None if position == 0 else history[position - 1]

    async def _embed_query(self, query: KnowledgeQuery, text: str) -> list[float]:
        from cayu.resource_access import require_dispatch

        await require_dispatch()
        result = copy_text_embedding_result(
            await self.embedding_provider.embed_texts(
                TextEmbeddingRequest(
                    model=self.embedding_model,
                    texts=[text],
                    dimensions=self.embedding_dimensions,
                )
            )
        )
        if result.model != self.embedding_model:
            raise ValueError("Embedding provider returned an unexpected model identity.")
        if len(result.embeddings) != 1:
            raise ValueError("Embedding provider returned an unexpected query result count.")
        embedding = next((item for item in result.embeddings if item.index == 0), None)
        if embedding is None:
            raise ValueError("Embedding provider did not return query embedding index 0.")
        self._validate_embedding_dimension(embedding.vector)
        return list(embedding.vector)

    def _validate_embedding_dimension(self, vector: list[float]) -> None:
        if len(vector) != self.embedding_dimensions:
            raise ValueError("Embedding provider returned a vector with unexpected dimension.")

    def _best_semantic_score(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: dict[str, list[float]],
        query_vector: list[float],
    ) -> tuple[float | None, KnowledgeChunk | None]:
        best_score: float | None = None
        best_chunk: KnowledgeChunk | None = None
        for chunk in chunks:
            vector = embeddings.get(chunk.id)
            if vector is None:
                continue
            score = _cosine_similarity(query_vector, vector)
            if best_score is None or score > best_score:
                best_score = score
                best_chunk = chunk
        return best_score, best_chunk

    def _drop_entry_embeddings(self, entry_id: str, *, limit: int) -> tuple[int, bool]:
        stale_ids = sorted(
            identity_sha256
            for identity_sha256, embedding in self._chunk_embeddings.items()
            if embedding["identity"].entry_id == entry_id
        )
        selected = stale_ids[:limit]
        for identity_sha256 in selected:
            self._chunk_embeddings.pop(identity_sha256, None)
            self._chunk_embedding_history.pop(identity_sha256, None)
        return len(selected), len(stale_ids) > limit

    def _drop_stale_entry_embeddings(
        self,
        entry_id: str,
        *,
        limit: int,
    ) -> tuple[int, bool]:
        current = self._current_entry(entry_id)
        current_identity_sha256 = (
            set()
            if current is None
            else {
                _knowledge_embedding_identity_sha256(
                    knowledge_chunk_embedding_identity(
                        chunk,
                        embedding_model=self.embedding_model,
                        dimensions=self.embedding_dimensions,
                    )
                )
                for chunk in self._chunks.get((entry_id, current.revision), [])
            }
        )
        stale_ids = sorted(
            identity_sha256
            for identity_sha256, embedding in self._chunk_embeddings.items()
            if embedding["identity"].entry_id == entry_id
            and identity_sha256 not in current_identity_sha256
        )
        selected = stale_ids[:limit]
        for identity_sha256 in selected:
            self._chunk_embeddings.pop(identity_sha256, None)
            self._chunk_embedding_history.pop(identity_sha256, None)
        return len(selected), len(stale_ids) > limit

    def _embedding_identity_matches_configuration(
        self,
        identity: KnowledgeEmbeddingIdentity,
    ) -> bool:
        return (
            identity.chunk_id is not None
            and identity.projection_type == KNOWLEDGE_CHUNK_TEXT_PROJECTION
            and identity.embedding_model == self.embedding_model
            and identity.dimensions == self.embedding_dimensions
            and identity.preprocessing_version == KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION
            and identity.generator == KNOWLEDGE_CHUNK_TEXT_GENERATOR
            and identity.generator_version == KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION
            and identity.index_representation_version
            == KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION
        )

    def _embedding_identity_is_current(self, identity: KnowledgeEmbeddingIdentity) -> bool:
        current = self._current_entry(identity.entry_id)
        if current is None or current.revision != identity.entry_revision:
            return False
        chunk = next(
            (
                item
                for item in self._chunks.get((identity.entry_id, identity.entry_revision), [])
                if item.id == identity.chunk_id
            ),
            None,
        )
        if chunk is None:
            return False
        return (
            knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            == identity
        )

    def _chunk_is_current(self, chunk: KnowledgeChunk) -> bool:
        current = self._current_entry(chunk.entry_id)
        if current is None or current.revision != chunk.entry_revision:
            return False
        return any(
            stored == chunk
            for stored in self._chunks.get((chunk.entry_id, chunk.entry_revision), [])
        )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension.")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(
        left_item * right_item for left_item, right_item in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)


def _normalize_cosine_similarity(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))
