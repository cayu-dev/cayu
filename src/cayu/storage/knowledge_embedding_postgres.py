"""PostgreSQL embedding projection, pgvector indexes and semantic retrieval."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, LiteralString, NoReturn, cast
from uuid import uuid4

from cayu.knowledge.access import runtime_knowledge_operation
from cayu.storage import _postgres_support as pg_support
from cayu.storage._phase_timing import PostgresTimingScope

try:
    from psycopg import sql
    from psycopg_pool import AsyncConnectionPool  # noqa: TC002 — keep runtime annotation resolution
except ModuleNotFoundError as exc:
    raise RuntimeError(
        'Cayu\'s Postgres stores require the optional psycopg packages. Install them with `pip install "cayu[postgres]"`.'
    ) from exc
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.embeddings import TextEmbeddingProvider, TextEmbeddingRequest, copy_text_embedding_result
from cayu.knowledge._embedding_backfill import (
    _decode_knowledge_embedding_backfill_cursor,
    _encode_knowledge_embedding_backfill_cursor,
    _knowledge_embedding_backfill_fingerprint,
)
from cayu.knowledge._query_rules import _semantic_query_text, _validate_knowledge_search_frontier
from cayu.knowledge._retrieval_results import _search_result_from_scored_embeddings
from cayu.knowledge._search_scoring import _score_entry
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
    knowledge_chunk_embedding_identity,
)
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_LIMIT,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    _validate_positive_int,
    copy_knowledge_revision_refs,
)
from cayu.knowledge.scopes import KnowledgeAccessScope
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
from cayu.storage import knowledge_postgres
from cayu.storage import migrations as schema
from cayu.storage.knowledge_postgres import PostgresKnowledgeStore

_PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS = 2000
_PGVECTOR_SEMANTIC_CANDIDATE_MULTIPLIER = 8
# Preserve the logging namespace used by existing application filters.
logger = logging.getLogger("cayu.storage.postgres")
_PGVECTOR_SCHEMA_ADVISORY_LOCK_KEY = 0x6361_7975_7665_6374 & 0x7FFF_FFFF_FFFF_FFFF


def _warn_if_embedding_dims_exceed_hnsw(dimensions: int) -> None:
    """Warn (do not reject) when embedding dimensions exceed pgvector's HNSW cap.

    pgvector's HNSW index supports at most 2000 dimensions. Larger models (e.g. 3072-dim) are still
    allowed — the store just can't build the index, so semantic search falls back to an exact O(n)
    brute-force scan. Surface that loudly instead of failing silently.
    """
    if dimensions > _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS:
        logger.warning(
            "Embedding dimensions (%d) exceed pgvector's HNSW limit (%d); the HNSW index will not be "
            "created and semantic search will fall back to an exact brute-force scan (O(n) per query).",
            dimensions,
            _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS,
        )


class PostgresEmbeddingKnowledgeStore(PostgresKnowledgeStore):
    """Postgres knowledge store with pgvector-backed semantic chunk search."""

    resource_knowledge_access_version = 1

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        embedding_provider: TextEmbeddingProvider,
        embedding_model: str,
        embedding_dimensions: int,
        access_scope: KnowledgeAccessScope | None = None,
        clock: Callable[[], datetime] | None = None,
        hybrid_keyword_weight: float = 0.35,
        semantic_min_score: float = 0.55,
    ) -> None:
        if not isinstance(embedding_provider, TextEmbeddingProvider):
            raise TypeError("embedding_provider must implement TextEmbeddingProvider.")
        _validate_positive_int(embedding_dimensions, "embedding_dimensions")
        _warn_if_embedding_dims_exceed_hnsw(embedding_dimensions)
        self.embedding_provider = embedding_provider
        self.embedding_model = require_clean_nonblank(embedding_model, "embedding_model")
        self.embedding_dimensions = embedding_dimensions
        self.hybrid_keyword_weight = _validate_nonnegative_float(
            hybrid_keyword_weight,
            "hybrid_keyword_weight",
        )
        self.semantic_min_score = _validate_unit_float(
            semantic_min_score,
            "semantic_min_score",
        )
        self._embedding_schema_ready = False
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
            access_scope=access_scope,
            clock=clock,
        )

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        return (
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.KEYWORD,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        )

    async def _ensure_ready(self) -> None:
        await super()._ensure_ready()
        if self._embedding_schema_ready:
            return
        async with self._open_lock:
            if self._embedding_schema_ready:
                return
            await self._reconcile_embedding_schema()
            self._embedding_schema_ready = True

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
        """Consume a bounded page of canonical changes into the vector index."""

        _validate_knowledge_change_limit(limit)
        _validate_knowledge_embedding_work_record_limit(record_limit)
        scope = self._operation_access_scope(access_scope)
        await self._ensure_ready()
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
                async with (
                    PostgresTimingScope(self._pool.connection()) as conn,
                    conn.cursor() as cur,
                ):
                    current = await self._load_entry(cur, claim.change.entry_id)
                    current_allowed = (
                        current is not None
                        and await self._load_entry_in_scope(
                            cur,
                            current.id,
                            scope,
                        )
                        is not None
                    )
                remaining = record_limit - processed_records
                if current is None or current.status is KnowledgeStatus.DELETED:
                    removed, cleanup_truncated = await self._drop_entry_embeddings(
                        claim.change.entry_id,
                        expected_deleted_revision=(None if current is None else current.revision),
                        limit=remaining,
                    )
                    removed_records += removed
                    processed_records += removed
                elif current.revision != claim.change.entry_revision or not current_allowed:
                    removed, cleanup_truncated = await self._drop_stale_entry_embeddings(
                        current.id,
                        limit=remaining,
                    )
                    removed_records += removed
                    processed_records += removed
                else:
                    chunks, truncated = await self._embedding_change_candidate_chunks(
                        current.id,
                        current.revision,
                        limit=record_limit - processed_records,
                        access_scope=scope,
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
                    removed, cleanup_truncated = await self._drop_stale_entry_embeddings(
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
        """Embed a bounded batch of existing chunks matching knowledge filters.

        By default this only fills missing or stale embedding rows. Set
        ``refresh_existing=True`` to re-embed current rows for the configured
        model and dimensions.
        """

        _validate_knowledge_embedding_work_record_limit(limit, field_name="limit")
        if type(refresh_existing) is not bool:
            raise ValueError("`refresh_existing` must be a boolean.")
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_list_query(query or KnowledgeListQuery())
        await self._ensure_ready()
        chunks, next_cursor = await self._backfill_candidate_chunks(
            query,
            limit,
            access_scope=scope,
            refresh_existing=refresh_existing,
            cursor=cursor,
        )
        attempt_id = f"kbackfill_{uuid4().hex}"
        embedded_chunks, failed_chunks = await self._index_chunks_with_readiness(
            chunks,
            attempt_id=attempt_id,
            operation_prefix=f"kidx:{attempt_id}",
            access_scope=scope,
            refresh_existing=refresh_existing,
        )
        return KnowledgeEmbeddingBackfillResult(
            scanned_records=len(chunks),
            indexed_records=embedded_chunks,
            failed_records=failed_chunks,
            skipped_records=len(chunks) - embedded_chunks - failed_chunks,
            limit=limit,
            refresh_existing=refresh_existing,
            next_cursor=next_cursor,
        )

    @runtime_knowledge_operation("read")
    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        query = copy_knowledge_query(query)
        return await self._embedding_search(
            query,
            scope,
            revision_refs=None,
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
        query = copy_knowledge_query(query)
        _validate_knowledge_change_sequence(knowledge_sequence, "knowledge_sequence")
        _validate_knowledge_index_sequence(
            index_readiness_sequence,
            "index_readiness_sequence",
        )
        return await self._embedding_search(
            query,
            scope,
            revision_refs=None,
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
        query = copy_knowledge_query(query)
        references = copy_knowledge_revision_refs(revision_refs)
        _validate_knowledge_search_frontier(
            knowledge_sequence,
            index_readiness_sequence,
        )
        return await self._embedding_search(
            query,
            scope,
            revision_refs=references,
            knowledge_sequence=knowledge_sequence,
            index_readiness_sequence=index_readiness_sequence,
        )

    async def _embedding_search(
        self,
        query: KnowledgeQuery,
        scope: KnowledgeAccessScope,
        *,
        revision_refs: tuple[KnowledgeRevisionRef, ...] | None,
        knowledge_sequence: int | None,
        index_readiness_sequence: int | None,
    ) -> KnowledgeSearchResult:
        terms = _knowledge_query_terms(query)
        if query.mode is KnowledgeSearchMode.KEYWORD or (
            query.mode is KnowledgeSearchMode.AUTO and not _query_terms_have_positive_terms(terms)
        ):
            # The public entry point already resolved and intersected the scope.
            # Re-entering it would treat resource constraints as a caller override
            # of a store's bound default scope.
            await self._ensure_ready()
            async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
                await knowledge_postgres._begin_knowledge_read_snapshot(cur)
                return await self._keyword_search_in_snapshot(
                    cur,
                    query,
                    access_scope=scope,
                    revision_refs=revision_refs,
                    through_change_sequence=knowledge_sequence,
                )
        if query.mode not in {
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        }:
            raise ValueError(
                "PostgresEmbeddingKnowledgeStore supports auto, keyword, semantic, "
                "and hybrid search modes."
            )
        await self._ensure_ready()
        if revision_refs == ():
            return KnowledgeSearchResult(
                query=query,
                hits=[],
                truncated=False,
                limit=query.limit,
                max_bytes=query.max_bytes,
                total_hits_known=0,
                index_coverage=[
                    KnowledgeIndexCoverage(
                        projection_type=KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                        embedding_model=self.embedding_model,
                        dimensions=self.embedding_dimensions,
                        preprocessing_version=KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                        generator=KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                        generator_version=KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                        index_representation_version=(
                            KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION
                        ),
                        eligible_records=0,
                        ready_records=0,
                        pending_records=0,
                        failed_records=0,
                        high_water_sequence=0,
                        complete=True,
                    )
                ],
            )
        if revision_refs is not None or knowledge_sequence is not None:
            no_ready_result = await self._exact_search_without_ready_semantic_candidates(
                query,
                scope,
                revision_refs,
                knowledge_sequence=knowledge_sequence,
                index_readiness_sequence=index_readiness_sequence,
            )
            if no_ready_result is not None:
                return no_ready_result
        semantic_query_text = _semantic_query_text(query)
        query_vector = await self._embed_query(query, semantic_query_text)
        keyword_result: KnowledgeSearchResult | None = None
        async with (
            PostgresTimingScope(self._pool.connection()) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await knowledge_postgres._begin_knowledge_read_snapshot(cur)
            coverage = await self._index_coverage_in_snapshot(
                cur,
                query,
                access_scope=scope,
                revision_refs=revision_refs,
                through_change_sequence=knowledge_sequence,
                through_index_readiness_sequence=index_readiness_sequence,
            )
            (
                rows,
                candidate_limit_reached,
                semantic_total_hits_known_floor,
            ) = await self._semantic_search_rows_in_snapshot(
                cur,
                query,
                query_vector,
                access_scope=scope,
                ready_records=coverage.ready_records,
                revision_refs=revision_refs,
                through_change_sequence=knowledge_sequence,
                through_index_readiness_sequence=index_readiness_sequence,
            )
            scored, byte_truncated = await self._scored_semantic_rows(
                cur,
                rows,
                query,
                access_scope=scope,
            )
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
                keyword_query = query.model_copy(update={"mode": KnowledgeSearchMode.KEYWORD})
                keyword_result = await self._keyword_search_in_snapshot(
                    cur,
                    keyword_query,
                    access_scope=scope,
                    revision_refs=revision_refs,
                    through_change_sequence=knowledge_sequence,
                )
        return self._finalize_embedding_search(
            query,
            coverage=coverage,
            scored=scored,
            keyword_result=keyword_result,
            byte_truncated=byte_truncated,
            candidate_limit_reached=candidate_limit_reached,
            semantic_total_hits_known_floor=semantic_total_hits_known_floor,
        )

    async def _exact_search_without_ready_semantic_candidates(
        self,
        query: KnowledgeQuery,
        scope: KnowledgeAccessScope,
        revision_refs: tuple[KnowledgeRevisionRef, ...] | None,
        *,
        knowledge_sequence: int | None,
        index_readiness_sequence: int | None,
    ) -> KnowledgeSearchResult | None:
        """Return a bounded result without a provider call when no vector is searchable."""

        async with (
            PostgresTimingScope(self._pool.connection()) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await knowledge_postgres._begin_knowledge_read_snapshot(cur)
            coverage = await self._index_coverage_in_snapshot(
                cur,
                query,
                access_scope=scope,
                revision_refs=revision_refs,
                through_change_sequence=knowledge_sequence,
                through_index_readiness_sequence=index_readiness_sequence,
            )
            if coverage.ready_records > 0:
                return None
            keyword_result: KnowledgeSearchResult | None = None
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
                keyword_result = await self._keyword_search_in_snapshot(
                    cur,
                    query.model_copy(update={"mode": KnowledgeSearchMode.KEYWORD}),
                    access_scope=scope,
                    revision_refs=revision_refs,
                    through_change_sequence=knowledge_sequence,
                )
        return self._finalize_embedding_search(
            query,
            coverage=coverage,
            scored=[],
            keyword_result=keyword_result,
            byte_truncated=False,
            candidate_limit_reached=False,
            semantic_total_hits_known_floor=0,
        )

    def _finalize_embedding_search(
        self,
        query: KnowledgeQuery,
        *,
        coverage: KnowledgeIndexCoverage,
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]
        ],
        keyword_result: KnowledgeSearchResult | None,
        byte_truncated: bool,
        candidate_limit_reached: bool,
        semantic_total_hits_known_floor: int,
    ) -> KnowledgeSearchResult:
        total_hits_known_floor = len(scored)
        if query.mode is KnowledgeSearchMode.SEMANTIC:
            total_hits_known_floor = max(
                total_hits_known_floor,
                semantic_total_hits_known_floor,
            )
        if keyword_result is not None:
            scored = self._merge_keyword_hits(scored, keyword_result)
            byte_truncated = byte_truncated or keyword_result.truncated
            keyword_total_hits_known = keyword_result.total_hits_known
            keyword_hits_floor = (
                keyword_total_hits_known
                if keyword_total_hits_known is not None
                else len(keyword_result.hits)
            )
            total_hits_known_floor = max(
                total_hits_known_floor,
                keyword_hits_floor,
            )
        score_kind = (
            "postgres_semantic" if query.mode is KnowledgeSearchMode.SEMANTIC else "postgres_hybrid"
        )
        result = _search_result_from_scored_embeddings(
            scored,
            query,
            score_kind=score_kind,
            index_coverage=[coverage],
        )
        total_hits_known = max(
            result.total_hits_known if result.total_hits_known is not None else len(result.hits),
            total_hits_known_floor,
        )
        return KnowledgeSearchResult(
            query=result.query,
            hits=result.hits,
            truncated=(
                byte_truncated
                or result.truncated
                or candidate_limit_reached
                or len(result.hits) < total_hits_known
            ),
            limit=result.limit,
            max_bytes=result.max_bytes,
            total_hits_known=total_hits_known,
            index_coverage=result.index_coverage,
        )

    async def _reconcile_embedding_schema(self) -> None:
        mode = self._schema_mode
        async with PostgresTimingScope(self._pool.connection()) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)", (_PGVECTOR_SCHEMA_ADVISORY_LOCK_KEY,)
                )
                if mode in {schema.SchemaMode.CREATE, schema.SchemaMode.MIGRATE}:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            CREATE TABLE IF NOT EXISTS cayu_knowledge_embeddings (
                                identity_sha256 TEXT NOT NULL
                                    CHECK (identity_sha256 ~ '^[0-9a-f]{{64}}$'),
                                entry_id TEXT NOT NULL,
                                entry_revision INTEGER NOT NULL
                                    CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
                                chunk_id TEXT NOT NULL,
                                projection_type TEXT NOT NULL,
                                projection_content_hash TEXT NOT NULL,
                                embedding_model TEXT NOT NULL,
                                dimensions INTEGER NOT NULL,
                                preprocessing_version TEXT NOT NULL,
                                generator TEXT NOT NULL,
                                generator_version TEXT NOT NULL,
                                index_representation_version TEXT NOT NULL,
                                readiness_sequence BIGINT NOT NULL
                                    CHECK (readiness_sequence > 0),
                                attempt_id TEXT NOT NULL,
                                current_projection BOOLEAN NOT NULL,
                                embedding_sha256 TEXT NOT NULL
                                    CHECK (embedding_sha256 ~ '^[0-9a-f]{{64}}$'),
                                embedding vector({self.embedding_dimensions}) NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL,
                                updated_at TIMESTAMPTZ NOT NULL,
                                PRIMARY KEY (identity_sha256, readiness_sequence),
                                UNIQUE (identity_sha256, attempt_id),
                                FOREIGN KEY (chunk_id, entry_id, entry_revision)
                                    REFERENCES cayu_knowledge_chunks(
                                        id, entry_id, entry_revision
                                    ) ON DELETE CASCADE
                            )
                            """,
                        )
                    )
                    await self._validate_embedding_schema(cur, require_indexes=False)
                    await cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_embeddings_entry
                        ON cayu_knowledge_embeddings(entry_id, entry_revision)
                        """
                    )
                    await cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_embeddings_model_dims
                        ON cayu_knowledge_embeddings(embedding_model, dimensions)
                        """
                    )
                    await cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS
                            idx_cayu_knowledge_embeddings_current_identity
                        ON cayu_knowledge_embeddings(identity_sha256)
                        WHERE current_projection
                        """
                    )
                    # HNSW tops out at 2000 dims; above the cap no index is built and semantic search
                    # falls back to an exact brute-force scan (the constructor warns — see
                    # _warn_if_embedding_dims_exceed_hnsw).
                    if self.embedding_dimensions <= _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS:
                        await self._create_embedding_hnsw_index(cur)
                        await self._create_embedding_history_hnsw_index(cur)
                elif mode is schema.SchemaMode.VALIDATE:
                    await cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                    row = await cur.fetchone()
                    if row is None or not bool(row[0]):
                        raise RuntimeError(
                            "PostgresEmbeddingKnowledgeStore requires the pgvector extension. "
                            "Use schema_mode=CREATE/MIGRATE or create extension vector manually."
                        )
                    await cur.execute("SELECT to_regclass('cayu_knowledge_embeddings')")
                    row = await cur.fetchone()
                    if row is None or row[0] is None:
                        raise RuntimeError(
                            "Missing Postgres knowledge embedding schema. "
                            "Run with schema_mode=CREATE or MIGRATE first."
                        )
                await self._validate_embedding_schema(cur)
                await self._validate_embedding_hnsw_indexes(cur)
            await conn.commit()

    async def _validate_embedding_schema(
        self,
        cur: Any,
        *,
        require_indexes: bool = True,
    ) -> None:
        await cur.execute(
            """
            SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
            FROM pg_attribute AS a
            WHERE a.attrelid = 'cayu_knowledge_embeddings'::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
            """
        )
        actual = [(str(row[0]), str(row[1]), bool(row[2])) for row in await cur.fetchall()]
        expected = [
            ("identity_sha256", "text", True),
            ("entry_id", "text", True),
            ("entry_revision", "integer", True),
            ("chunk_id", "text", True),
            ("projection_type", "text", True),
            ("projection_content_hash", "text", True),
            ("embedding_model", "text", True),
            ("dimensions", "integer", True),
            ("preprocessing_version", "text", True),
            ("generator", "text", True),
            ("generator_version", "text", True),
            ("index_representation_version", "text", True),
            ("readiness_sequence", "bigint", True),
            ("attempt_id", "text", True),
            ("current_projection", "boolean", True),
            ("embedding_sha256", "text", True),
            ("embedding", f"vector({self.embedding_dimensions})", True),
            ("created_at", "timestamp with time zone", True),
            ("updated_at", "timestamp with time zone", True),
        ]
        if actual != expected:
            self._raise_embedding_schema_error("cayu_knowledge_embeddings")

        await cur.execute(
            """
            SELECT constraint_record.contype,
                   constraint_record.convalidated,
                   pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = 'cayu_knowledge_embeddings'
            """
        )
        constraints = [
            (str(kind), bool(validated), " ".join(str(definition).lower().split()))
            for kind, validated, definition in await cur.fetchall()
        ]
        required_constraints = (
            ("p", ("primary key (identity_sha256, readiness_sequence)",)),
            ("u", ("unique (identity_sha256, attempt_id)",)),
            ("c", ("identity_sha256", "[0-9a-f]{64}")),
            ("c", ("entry_revision > 0", "entry_revision <= 2147483647")),
            ("c", ("readiness_sequence > 0",)),
            ("c", ("embedding_sha256", "[0-9a-f]{64}")),
            (
                "f",
                (
                    "foreign key (chunk_id, entry_id, entry_revision)",
                    "references cayu_knowledge_chunks(id, entry_id, entry_revision)",
                    "on delete cascade",
                ),
            ),
        )
        for kind, fragments in required_constraints:
            if not any(
                candidate_kind == kind
                and validated
                and all(fragment in definition for fragment in fragments)
                for candidate_kind, validated, definition in constraints
            ):
                self._raise_embedding_schema_error("cayu_knowledge_embeddings")

        if not require_indexes:
            return
        expected_indexes = {
            "idx_cayu_knowledge_embeddings_entry": (
                "cayu_knowledge_embeddings",
                "using btree (entry_id, entry_revision)",
                False,
                None,
            ),
            "idx_cayu_knowledge_embeddings_model_dims": (
                "cayu_knowledge_embeddings",
                "using btree (embedding_model, dimensions)",
                False,
                None,
            ),
            "idx_cayu_knowledge_embeddings_current_identity": (
                "cayu_knowledge_embeddings",
                "using btree (identity_sha256)",
                True,
                "current_projection",
            ),
        }
        await cur.execute(
            """
            SELECT table_record.relname, index_record.relname,
                   index_state.indisvalid, index_state.indisready,
                   index_state.indisunique, index_state.indpred IS NULL,
                   pg_get_indexdef(index_record.oid),
                   pg_get_expr(index_state.indpred, index_state.indrelid)
            FROM pg_catalog.pg_index AS index_state
            JOIN pg_catalog.pg_class AS index_record
              ON index_record.oid = index_state.indexrelid
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = index_state.indrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND index_record.relname = ANY(%s)
            """,
            (list(expected_indexes),),
        )
        indexes = {
            str(index): (
                str(table),
                bool(valid),
                bool(ready),
                bool(unique),
                bool(unqualified),
                " ".join(str(definition).lower().split()),
                " ".join(str(predicate or "").lower().split()),
            )
            for table, index, valid, ready, unique, unqualified, definition, predicate in (
                await cur.fetchall()
            )
        }
        for name, (table, definition, unique, predicate) in expected_indexes.items():
            value = indexes.get(name)
            if (
                value is None
                or value[0] != table
                or not value[1]
                or not value[2]
                or value[3] != unique
                or value[4] != (predicate is None)
                or definition not in value[5]
                or value[6] != ("" if predicate is None else predicate)
            ):
                self._raise_embedding_schema_error(name)

    @staticmethod
    def _raise_embedding_schema_error(name: str) -> NoReturn:
        raise RuntimeError(
            "Postgres knowledge embedding schema does not match Cayu's "
            f"revision-bound projection contract at {name!r}. Drop the derived "
            "cayu_knowledge_embeddings table, restart with schema_mode=CREATE or MIGRATE, "
            "and rebuild its projections from canonical knowledge entries."
        )

    def _embedding_hnsw_index_name(self) -> str:
        identity = pg_support._dumps(
            [
                KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                self.embedding_model,
                self.embedding_dimensions,
                KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
            ]
        )
        suffix = sha256(identity.encode("utf-8")).hexdigest()[:28]
        return f"idx_cayu_knowledge_embeddings_hnsw_{suffix}"

    def _embedding_history_hnsw_index_name(self) -> str:
        suffix = self._embedding_hnsw_index_name().rsplit("_", 1)[-1]
        return f"idx_cayu_knowledge_embeddings_history_{suffix[:24]}"

    async def _create_embedding_hnsw_index(self, cur: Any) -> None:
        await cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON cayu_knowledge_embeddings USING hnsw (embedding vector_cosine_ops)
                WHERE current_projection
                  AND projection_type = {projection_type}
                  AND embedding_model = {embedding_model}
                  AND dimensions = {dimensions}
                  AND preprocessing_version = {preprocessing_version}
                  AND generator = {generator}
                  AND generator_version = {generator_version}
                  AND index_representation_version = {index_representation_version}
                """
            ).format(
                index=sql.Identifier(self._embedding_hnsw_index_name()),
                projection_type=sql.Literal(KNOWLEDGE_CHUNK_TEXT_PROJECTION),
                embedding_model=sql.Literal(self.embedding_model),
                dimensions=sql.Literal(self.embedding_dimensions),
                preprocessing_version=sql.Literal(KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION),
                generator=sql.Literal(KNOWLEDGE_CHUNK_TEXT_GENERATOR),
                generator_version=sql.Literal(KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION),
                index_representation_version=sql.Literal(
                    KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION
                ),
            )
        )

    async def _create_embedding_history_hnsw_index(self, cur: Any) -> None:
        await cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON cayu_knowledge_embeddings USING hnsw (embedding vector_cosine_ops)
                WHERE projection_type = {projection_type}
                  AND embedding_model = {embedding_model}
                  AND dimensions = {dimensions}
                  AND preprocessing_version = {preprocessing_version}
                  AND generator = {generator}
                  AND generator_version = {generator_version}
                  AND index_representation_version = {index_representation_version}
                """
            ).format(
                index=sql.Identifier(self._embedding_history_hnsw_index_name()),
                projection_type=sql.Literal(KNOWLEDGE_CHUNK_TEXT_PROJECTION),
                embedding_model=sql.Literal(self.embedding_model),
                dimensions=sql.Literal(self.embedding_dimensions),
                preprocessing_version=sql.Literal(KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION),
                generator=sql.Literal(KNOWLEDGE_CHUNK_TEXT_GENERATOR),
                generator_version=sql.Literal(KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION),
                index_representation_version=sql.Literal(
                    KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION
                ),
            )
        )

    async def _validate_embedding_hnsw_indexes(self, cur: Any) -> None:
        await cur.execute(
            """
            SELECT index_record.relname,
                   index_state.indisvalid,
                   index_state.indisready,
                   pg_get_indexdef(index_record.oid),
                   pg_get_expr(index_state.indpred, index_state.indrelid),
                   quote_literal(%s), quote_literal(%s), quote_literal(%s),
                   quote_literal(%s), quote_literal(%s), quote_literal(%s)
            FROM pg_catalog.pg_index AS index_state
            JOIN pg_catalog.pg_class AS index_record
              ON index_record.oid = index_state.indexrelid
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = index_state.indrelid
            JOIN pg_catalog.pg_am AS access_method
              ON access_method.oid = index_record.relam
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = 'cayu_knowledge_embeddings'
              AND access_method.amname = 'hnsw'
            """,
            (
                KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                self.embedding_model,
                KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
            ),
        )
        rows = await cur.fetchall()
        required_predicate_fields = (
            "projection_type",
            "embedding_model",
            "dimensions",
            "preprocessing_version",
            "generator",
            "generator_version",
            "index_representation_version",
        )
        expected_names = {
            self._embedding_hnsw_index_name(): True,
            self._embedding_history_hnsw_index_name(): False,
        }
        expected_found: set[str] = set()
        for row in rows:
            name = str(row[0])
            definition = " ".join(str(row[3]).lower().split())
            predicate = str(row[4] or "").lower()
            predicate_shape = re.sub(r"'(?:''|[^'])*'(?:::text)?", "?", predicate)
            if (
                not bool(row[1])
                or not bool(row[2])
                or "using hnsw (embedding vector_cosine_ops)" not in definition
                or any(
                    re.search(rf"\b{field}\s*=", predicate_shape) is None
                    for field in required_predicate_fields
                )
                or re.search(r"\bor\b", predicate_shape) is not None
                or re.search(r"\bany\s*\(", predicate_shape) is not None
                or re.search(r"\bnot\b", predicate_shape) is not None
            ):
                raise RuntimeError(
                    "Postgres knowledge embedding HNSW indexes must isolate one "
                    "complete compatible projection space. Drop the conflicting derived "
                    "embedding index before restarting with schema_mode=CREATE or MIGRATE."
                )
            current_only = expected_names.get(name)
            if current_only is None:
                continue
            expected_fragments = (
                f"projection_type = {str(row[5]).lower()}::text",
                f"embedding_model = {str(row[6]).lower()}::text",
                f"dimensions = {self.embedding_dimensions}",
                f"preprocessing_version = {str(row[7]).lower()}::text",
                f"generator = {str(row[8]).lower()}::text",
                f"generator_version = {str(row[9]).lower()}::text",
                f"index_representation_version = {str(row[10]).lower()}::text",
            )
            remaining_predicate = predicate
            exact_projection_space = True
            for fragment in expected_fragments:
                if remaining_predicate.count(fragment) != 1:
                    exact_projection_space = False
                    break
                remaining_predicate = remaining_predicate.replace(fragment, "", 1)
            current_projection_occurrences = len(
                re.findall(r"\bcurrent_projection\b", remaining_predicate)
            )
            if current_only:
                if current_projection_occurrences != 1:
                    exact_projection_space = False
                else:
                    remaining_predicate = re.sub(
                        r"\bcurrent_projection\b",
                        "",
                        remaining_predicate,
                        count=1,
                    )
            elif current_projection_occurrences != 0:
                exact_projection_space = False
            remaining_predicate = re.sub(r"\band\b", "", remaining_predicate)
            remaining_predicate = re.sub(r"[\s()]", "", remaining_predicate)
            if not exact_projection_space or remaining_predicate:
                raise RuntimeError(
                    "Postgres knowledge embedding HNSW index conflicts with the "
                    "configured projection space. Drop the conflicting derived embedding "
                    "index before restarting with schema_mode=CREATE or MIGRATE."
                )
            expected_found.add(name)
        if (
            self.embedding_dimensions <= _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS
            and expected_found != set(expected_names)
        ):
            raise RuntimeError(
                "Missing Postgres knowledge embedding HNSW index for the configured "
                "projection space. Restart with schema_mode=CREATE or MIGRATE."
            )

    async def _semantic_search_rows(
        self,
        query: KnowledgeQuery,
        query_vector: list[float],
        *,
        access_scope: KnowledgeAccessScope,
    ) -> tuple[list[tuple[str, str, float]], bool, int]:
        async with (
            PostgresTimingScope(self._pool.connection()) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await knowledge_postgres._begin_knowledge_read_snapshot(cur)
            coverage = await self._index_coverage_in_snapshot(
                cur,
                query,
                access_scope=access_scope,
            )
            return await self._semantic_search_rows_in_snapshot(
                cur,
                query,
                query_vector,
                access_scope=access_scope,
                ready_records=coverage.ready_records,
            )

    async def _semantic_search_rows_in_snapshot(
        self,
        cur: Any,
        query: KnowledgeQuery,
        query_vector: list[float],
        *,
        access_scope: KnowledgeAccessScope,
        ready_records: int,
        force_exact: bool = False,
        revision_refs: tuple[KnowledgeRevisionRef, ...] | None = None,
        through_change_sequence: int | None = None,
        through_index_readiness_sequence: int | None = None,
    ) -> tuple[list[tuple[str, str, float]], bool, int]:
        where_sql, params = knowledge_postgres._postgres_knowledge_filter_sql(query)
        access_sql, access_params = knowledge_postgres._postgres_knowledge_access_scope_filter_sql(
            access_scope
        )
        where_sql += access_sql
        params.extend(access_params)
        revision_sql, revision_params = (
            knowledge_postgres._postgres_knowledge_revision_refs_filter_sql(revision_refs)
        )
        where_sql += revision_sql
        params.extend(revision_params)
        frontier_sql, frontier_params = knowledge_postgres._postgres_knowledge_frontier_filter_sql(
            through_change_sequence
        )
        where_sql += frontier_sql
        params.extend(frontier_params)
        none_sql, none_params = knowledge_postgres._postgres_knowledge_none_filter_sql(query)
        if through_index_readiness_sequence is None:
            readiness_join_sql = """
                JOIN cayu_knowledge_index_readiness_current AS readiness_current
                  ON readiness_current.identity_sha256 = emb.identity_sha256
                JOIN cayu_knowledge_index_readiness_events AS readiness
                  ON readiness.sequence = readiness_current.sequence
                 AND readiness.identity_sha256 = emb.identity_sha256
                 AND readiness.state = 'ready'
            """
            readiness_join_params: list[object] = []
            projection_selection_sql = " AND emb.current_projection"
        else:
            readiness_join_sql = """
                JOIN cayu_knowledge_index_readiness_events AS readiness
                  ON readiness.identity_sha256 = emb.identity_sha256
                 AND readiness.state = 'ready'
                 AND readiness.sequence = (
                     SELECT MAX(boundary.sequence)
                     FROM cayu_knowledge_index_readiness_events AS boundary
                     WHERE boundary.identity_sha256 = emb.identity_sha256
                       AND boundary.sequence <= %s
                 )
            """
            readiness_join_params = [through_index_readiness_sequence]
            projection_selection_sql = """
                AND emb.readiness_sequence = (
                    SELECT MAX(projection.readiness_sequence)
                    FROM cayu_knowledge_embeddings AS projection
                    WHERE projection.identity_sha256 = emb.identity_sha256
                      AND projection.readiness_sequence <= readiness.sequence
                )
            """
        vector_literal = _postgres_vector_literal(query_vector)
        candidate_limit = max(
            query.limit,
            query.limit * _PGVECTOR_SEMANTIC_CANDIDATE_MULTIPLIER,
        )
        semantic_min_score = self.semantic_min_score if query.min_score is None else query.min_score
        min_score_sql = (
            ""
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}
            else "WHERE normalized_score >= %s"
        )
        min_score_params: list[object] = (
            []
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}
            else [semantic_min_score]
        )
        exact_scan = (
            force_exact
            or revision_refs is not None
            or bool(query.none_terms)
            or self.embedding_dimensions > _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS
        )
        if exact_scan:
            # pgvector applies WHERE filters after its bounded approximate
            # HNSW scan. A dense set of nearer excluded entries could
            # therefore consume the complete ANN candidate budget before a
            # valid lower-ranked entry is visited. Use an exact vector scan
            # for bounded exact-revision sets and entry-wide negative filters
            # so those predicates are authoritative before Cayu's candidate
            # limit. Frontier predicates retain HNSW and use the underfill
            # fallback below; forcing them exact would make every initial or
            # changed-context checkpoint recall scan the complete vector table.
            # These settings are transaction-local so ordinary semantic
            # searches retain HNSW and pooled connections cannot leak the
            # exact-search policy to later requests.
            await cur.execute("SET LOCAL enable_indexscan = off")
            await cur.execute("SET LOCAL enable_seqscan = on")
        elif self.embedding_dimensions <= _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS:
            # The HNSW graph is partial to one complete projection space. Force
            # a custom plan so PostgreSQL can prove that the bound identity
            # parameters imply that partial-index predicate even after psycopg
            # begins preparing this frequently executed query.
            await cur.execute("SET LOCAL plan_cache_mode = force_custom_plan")
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                WITH nearest_chunks AS (
                    SELECT
                        e.id AS entry_id,
                        c.id AS chunk_id,
                        c.chunk_index AS chunk_index,
                        emb.embedding <=> %s::vector AS distance,
                        (1.0 + (1.0 - (emb.embedding <=> %s::vector))) / 2.0 AS normalized_score,
                        COALESCE(e.importance, 0.0) AS importance,
                        e.updated_at AS updated_at
                    FROM cayu_knowledge_embeddings AS emb
                    JOIN cayu_knowledge_chunks AS c
                      ON c.id = emb.chunk_id
                     AND c.entry_id = emb.entry_id
                     AND c.entry_revision = emb.entry_revision
                    JOIN cayu_knowledge_current_entries AS e
                      ON e.id = emb.entry_id AND e.revision = c.entry_revision
                    {readiness_join_sql}
                    WHERE emb.projection_type = %s
                      AND emb.projection_content_hash =
                          'sha256:' || encode(sha256(convert_to(c.text, 'UTF8')), 'hex')
                      AND emb.embedding_model = %s
                      AND emb.dimensions = %s
                      AND emb.preprocessing_version = %s
                      AND emb.generator = %s
                      AND emb.generator_version = %s
                      AND emb.index_representation_version = %s
                    {projection_selection_sql}
                    {where_sql}
                    {none_sql}
                    ORDER BY emb.embedding <=> %s::vector
                    LIMIT %s
                ),
                best_entries AS (
                    SELECT DISTINCT ON (entry_id)
                        entry_id,
                        chunk_id,
                        normalized_score,
                        importance,
                        updated_at
                    FROM nearest_chunks
                    ORDER BY entry_id, distance ASC, chunk_index ASC
                ),
                filtered_entries AS (
                    SELECT *
                    FROM best_entries
                    {min_score_sql}
                )
                SELECT
                    entry_id,
                    chunk_id,
                    normalized_score,
                    (SELECT COUNT(*) FROM nearest_chunks) AS candidate_chunk_count,
                    (SELECT COUNT(*) FROM filtered_entries) AS candidate_entry_count
                FROM filtered_entries
                ORDER BY normalized_score DESC,
                         importance DESC,
                         updated_at DESC,
                         entry_id ASC
                LIMIT %s
                """,
            ),
            [
                vector_literal,
                vector_literal,
                *readiness_join_params,
                KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                self.embedding_model,
                self.embedding_dimensions,
                KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
                *params,
                *none_params,
                vector_literal,
                candidate_limit,
                *min_score_params,
                query.limit,
            ],
        )
        raw_rows = await cur.fetchall()
        candidate_chunk_count = 0 if not raw_rows else int(raw_rows[0][3])
        candidate_entry_count = 0 if not raw_rows else int(raw_rows[0][4])
        expected_candidate_chunks = min(candidate_limit, ready_records)
        if not exact_scan and candidate_chunk_count < expected_candidate_chunks:
            # pgvector's approximate scan can apply namespace, lifecycle, and
            # access predicates after walking a bounded HNSW candidate set.
            # If eligible READY rows underfill that set, rerun exactly inside
            # this same repeatable-read snapshot. This keeps ANN fast on the
            # ordinary path without allowing filtered false negatives.
            return await self._semantic_search_rows_in_snapshot(
                cur,
                query,
                query_vector,
                access_scope=access_scope,
                ready_records=ready_records,
                force_exact=True,
                revision_refs=revision_refs,
                through_change_sequence=through_change_sequence,
                through_index_readiness_sequence=through_index_readiness_sequence,
            )
        candidate_limit_reached = candidate_chunk_count >= candidate_limit
        return (
            [(str(row[0]), str(row[1]), float(row[2])) for row in raw_rows if row[0] is not None],
            candidate_limit_reached,
            candidate_entry_count,
        )

    async def _index_coverage_in_snapshot(
        self,
        cur: Any,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope,
        revision_refs: tuple[KnowledgeRevisionRef, ...] | None = None,
        through_change_sequence: int | None = None,
        through_index_readiness_sequence: int | None = None,
    ) -> KnowledgeIndexCoverage:
        where_sql, params = knowledge_postgres._postgres_knowledge_filter_sql(query)
        access_sql, access_params = knowledge_postgres._postgres_knowledge_access_scope_filter_sql(
            access_scope
        )
        where_sql += access_sql
        params.extend(access_params)
        revision_sql, revision_params = (
            knowledge_postgres._postgres_knowledge_revision_refs_filter_sql(revision_refs)
        )
        where_sql += revision_sql
        params.extend(revision_params)
        frontier_sql, frontier_params = knowledge_postgres._postgres_knowledge_frontier_filter_sql(
            through_change_sequence
        )
        where_sql += frontier_sql
        params.extend(frontier_params)
        none_sql, none_params = knowledge_postgres._postgres_knowledge_none_filter_sql(query)
        if through_index_readiness_sequence is None:
            readiness_current_join_sql = """
                JOIN cayu_knowledge_index_readiness_current AS current
                  ON current.identity_sha256 = event.identity_sha256
                 AND current.sequence = event.sequence
            """
            readiness_frontier_sql = ""
            readiness_order_sql = ""
            readiness_frontier_params: list[object] = []
            embedding_selection_sql = " AND embedding.current_projection"
        else:
            readiness_current_join_sql = ""
            readiness_frontier_sql = " AND event.sequence <= %s"
            readiness_order_sql = "ORDER BY event.sequence DESC"
            readiness_frontier_params = [through_index_readiness_sequence]
            embedding_selection_sql = """
                AND embedding.readiness_sequence = (
                    SELECT MAX(projection.readiness_sequence)
                    FROM cayu_knowledge_embeddings AS projection
                    WHERE projection.identity_sha256 = readiness.identity_sha256
                      AND projection.readiness_sequence <= readiness.sequence
                )
            """
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                WITH eligible AS (
                    SELECT c.*
                    FROM cayu_knowledge_chunks AS c
                    JOIN cayu_knowledge_current_entries AS e
                      ON e.id = c.entry_id AND e.revision = c.entry_revision
                    WHERE TRUE
                    {where_sql}
                    {none_sql}
                ), classified AS (
                    SELECT
                        readiness.sequence,
                        CASE
                            WHEN readiness.state = 'ready'
                             AND embedding.identity_sha256 IS NOT NULL THEN 'ready'
                            WHEN readiness.state = 'failed' THEN 'failed'
                            ELSE 'pending'
                        END AS state
                    FROM eligible AS c
                    LEFT JOIN LATERAL (
                        SELECT event.*
                        FROM cayu_knowledge_index_readiness_events AS event
                        {readiness_current_join_sql}
                        WHERE event.entry_id = c.entry_id
                          AND event.entry_revision = c.entry_revision
                          AND event.chunk_id = c.id
                          AND event.projection_type = %s
                          AND event.projection_content_hash =
                              'sha256:' || encode(sha256(convert_to(c.text, 'UTF8')), 'hex')
                          AND event.embedding_model = %s
                          AND event.dimensions = %s
                          AND event.preprocessing_version = %s
                          AND event.generator = %s
                          AND event.generator_version = %s
                          AND event.index_representation_version = %s
                        {readiness_frontier_sql}
                        {readiness_order_sql}
                        LIMIT 1
                    ) AS readiness ON TRUE
                    LEFT JOIN cayu_knowledge_embeddings AS embedding
                      ON embedding.identity_sha256 = readiness.identity_sha256
                     AND embedding.entry_id = c.entry_id
                     AND embedding.entry_revision = c.entry_revision
                     AND embedding.chunk_id = c.id
                     AND embedding.projection_type = readiness.projection_type
                     AND embedding.projection_content_hash =
                         readiness.projection_content_hash
                     AND embedding.embedding_model = readiness.embedding_model
                     AND embedding.dimensions = readiness.dimensions
                     AND embedding.preprocessing_version = readiness.preprocessing_version
                     AND embedding.generator = readiness.generator
                     AND embedding.generator_version = readiness.generator_version
                     AND embedding.index_representation_version =
                         readiness.index_representation_version
                    {embedding_selection_sql}
                )
                SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE state = 'ready'),
                    COUNT(*) FILTER (WHERE state = 'pending'),
                    COUNT(*) FILTER (WHERE state = 'failed'),
                    COALESCE(MAX(sequence), 0)
                FROM classified
                """,
            ),
            [
                *params,
                *none_params,
                KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                self.embedding_model,
                self.embedding_dimensions,
                KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
                *readiness_frontier_params,
            ],
        )
        row = await cur.fetchone()
        if row is None:  # pragma: no cover - aggregate invariant
            raise RuntimeError("Postgres did not return knowledge index coverage.")
        eligible = int(row[0])
        ready = int(row[1])
        pending = int(row[2])
        failed = int(row[3])
        return KnowledgeIndexCoverage(
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
            high_water_sequence=int(row[4]),
            complete=ready == eligible and pending == 0 and failed == 0,
        )

    async def _backfill_candidate_chunks(
        self,
        query: KnowledgeListQuery,
        limit: int,
        *,
        access_scope: KnowledgeAccessScope,
        refresh_existing: bool,
        cursor: str | None,
        search_query: KnowledgeQuery | None = None,
    ) -> tuple[list[KnowledgeChunk], str | None]:
        fingerprint = _knowledge_embedding_backfill_fingerprint(
            query,
            access_scope,
            refresh_existing=refresh_existing,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
        )
        after = _decode_knowledge_embedding_backfill_cursor(
            cursor,
            fingerprint=fingerprint,
        )
        where_sql, params = knowledge_postgres._postgres_knowledge_list_filter_sql(query)
        access_sql, access_params = knowledge_postgres._postgres_knowledge_access_scope_filter_sql(
            access_scope
        )
        where_sql += access_sql
        params.extend(access_params)
        none_sql, none_params = (
            ("", [])
            if search_query is None
            else knowledge_postgres._postgres_knowledge_none_filter_sql(search_query)
        )
        missing_embedding_filter_sql = ""
        current_embedding_params: list[object] = []
        if not refresh_existing:
            missing_embedding_filter_sql = """
                AND NOT EXISTS (
                    SELECT 1
                    FROM cayu_knowledge_index_readiness_events AS readiness
                    JOIN cayu_knowledge_index_readiness_current AS current
                      ON current.identity_sha256 = readiness.identity_sha256
                     AND current.sequence = readiness.sequence
                    JOIN cayu_knowledge_embeddings AS embedding
                      ON embedding.identity_sha256 = readiness.identity_sha256
                     AND embedding.entry_id = readiness.entry_id
                     AND embedding.entry_revision = readiness.entry_revision
                     AND embedding.chunk_id = readiness.chunk_id
                     AND embedding.projection_type = readiness.projection_type
                     AND embedding.projection_content_hash =
                         readiness.projection_content_hash
                     AND embedding.embedding_model = readiness.embedding_model
                     AND embedding.dimensions = readiness.dimensions
                     AND embedding.preprocessing_version = readiness.preprocessing_version
                     AND embedding.generator = readiness.generator
                     AND embedding.generator_version = readiness.generator_version
                     AND embedding.index_representation_version =
                         readiness.index_representation_version
                     AND embedding.current_projection
                    WHERE readiness.entry_id = c.entry_id
                      AND readiness.entry_revision = c.entry_revision
                      AND readiness.chunk_id = c.id
                      AND readiness.projection_type = %s
                      AND readiness.state = 'ready'
                      AND readiness.projection_content_hash =
                          'sha256:' || encode(sha256(convert_to(c.text, 'UTF8')), 'hex')
                      AND readiness.embedding_model = %s
                      AND readiness.dimensions = %s
                      AND readiness.preprocessing_version = %s
                      AND readiness.generator = %s
                      AND readiness.generator_version = %s
                      AND readiness.index_representation_version = %s
                )
                """
            current_embedding_params = [
                KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                self.embedding_model,
                self.embedding_dimensions,
                KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
            ]
        cursor_sql = ""
        cursor_params: list[object] = []
        if after is not None:
            cursor_sql = """
                AND (
                    COALESCE(e.importance, 0.0) < %s
                    OR (
                        COALESCE(e.importance, 0.0) = %s
                        AND e.updated_at < %s
                    )
                    OR (
                        COALESCE(e.importance, 0.0) = %s
                        AND e.updated_at = %s
                        AND e.id > %s
                    )
                    OR (
                        COALESCE(e.importance, 0.0) = %s
                        AND e.updated_at = %s
                        AND e.id = %s
                        AND c.chunk_index > %s
                    )
                    OR (
                        COALESCE(e.importance, 0.0) = %s
                        AND e.updated_at = %s
                        AND e.id = %s
                        AND c.chunk_index = %s
                        AND c.id > %s
                    )
                )
            """
            cursor_params = [
                after.importance,
                after.importance,
                after.updated_at,
                after.importance,
                after.updated_at,
                after.entry_id,
                after.importance,
                after.updated_at,
                after.entry_id,
                after.chunk_index,
                after.importance,
                after.updated_at,
                after.entry_id,
                after.chunk_index,
                after.chunk_id,
            ]
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT c.id, c.entry_id, c.entry_revision, c.chunk_index,
                           c.text, c.content_hash, c.source_uri, c.metadata,
                           COALESCE(e.importance, 0.0), e.updated_at
                    FROM cayu_knowledge_chunks AS c
                    JOIN cayu_knowledge_current_entries AS e
                      ON e.id = c.entry_id AND e.revision = c.entry_revision
                    WHERE TRUE
                    {where_sql}
                    {none_sql}
                    {missing_embedding_filter_sql}
                    {cursor_sql}
                    ORDER BY COALESCE(e.importance, 0.0) DESC,
                             e.updated_at DESC,
                             e.id ASC,
                             c.chunk_index ASC,
                             c.id ASC
                    LIMIT %s
                    """,
                ),
                [
                    *params,
                    *none_params,
                    *current_embedding_params,
                    *cursor_params,
                    limit + 1,
                ],
            )
            rows = await cur.fetchall()
        page = rows[:limit]
        chunks = [knowledge_postgres._knowledge_chunk_from_row(row) for row in page]
        next_cursor = (
            _encode_knowledge_embedding_backfill_cursor(
                fingerprint=fingerprint,
                importance=float(page[-1][8]),
                updated_at=page[-1][9],
                chunk=chunks[-1],
            )
            if len(rows) > limit and page
            else None
        )
        return chunks, next_cursor

    async def _embedding_change_candidate_chunks(
        self,
        entry_id: str,
        entry_revision: int,
        *,
        limit: int,
        access_scope: KnowledgeAccessScope,
    ) -> tuple[list[KnowledgeChunk], bool]:
        _validate_positive_int(limit, "record_limit")
        access_sql, access_params = knowledge_postgres._postgres_knowledge_access_scope_filter_sql(
            access_scope,
            entry_alias="e",
        )
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT c.id, c.entry_id, c.entry_revision, c.chunk_index,
                           c.text, c.content_hash, c.source_uri, c.metadata
                    FROM cayu_knowledge_chunks AS c
                    JOIN cayu_knowledge_current_entries AS e
                      ON e.id = c.entry_id AND e.revision = c.entry_revision
                    WHERE c.entry_id = %s
                      AND c.entry_revision = %s
                    {access_sql}
                      AND NOT EXISTS (
                          SELECT 1
                          FROM cayu_knowledge_index_readiness_events AS readiness
                          JOIN cayu_knowledge_index_readiness_current AS current
                            ON current.identity_sha256 = readiness.identity_sha256
                           AND current.sequence = readiness.sequence
                          LEFT JOIN cayu_knowledge_embeddings AS embedding
                            ON embedding.identity_sha256 = readiness.identity_sha256
                           AND embedding.entry_id = readiness.entry_id
                           AND embedding.entry_revision = readiness.entry_revision
                           AND embedding.chunk_id = readiness.chunk_id
                           AND embedding.projection_type = readiness.projection_type
                           AND embedding.projection_content_hash =
                               readiness.projection_content_hash
                           AND embedding.embedding_model = readiness.embedding_model
                           AND embedding.dimensions = readiness.dimensions
                           AND embedding.preprocessing_version =
                               readiness.preprocessing_version
                           AND embedding.generator = readiness.generator
                           AND embedding.generator_version = readiness.generator_version
                           AND embedding.index_representation_version =
                               readiness.index_representation_version
                           AND embedding.current_projection
                          WHERE readiness.entry_id = c.entry_id
                            AND readiness.entry_revision = c.entry_revision
                            AND readiness.chunk_id = c.id
                            AND readiness.projection_type = %s
                            AND readiness.projection_content_hash =
                                'sha256:' || encode(
                                    sha256(convert_to(c.text, 'UTF8')), 'hex'
                                )
                            AND readiness.embedding_model = %s
                            AND readiness.dimensions = %s
                            AND readiness.preprocessing_version = %s
                            AND readiness.generator = %s
                            AND readiness.generator_version = %s
                            AND readiness.index_representation_version = %s
                            AND readiness.state = 'ready'
                            AND embedding.identity_sha256 IS NOT NULL
                      )
                    ORDER BY c.chunk_index, c.id
                    LIMIT %s
                    """,
                ),
                [
                    entry_id,
                    entry_revision,
                    *access_params,
                    KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                    self.embedding_model,
                    self.embedding_dimensions,
                    KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                    KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                    KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                    KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
                    limit + 1,
                ],
            )
            rows = await cur.fetchall()
        return (
            [knowledge_postgres._knowledge_chunk_from_row(row) for row in rows[:limit]],
            len(rows) > limit,
        )

    async def _scored_semantic_rows(
        self,
        cur: Any,
        rows: list[tuple[str, str, float]],
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> tuple[
        list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]],
        bool,
    ]:
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]
        ] = []
        byte_truncated = False
        scope = self._operation_access_scope(access_scope)
        semantic_min_score = self.semantic_min_score if query.min_score is None else query.min_score
        for entry_id, chunk_id, normalized_score in rows:
            entry = await self._load_entry_in_scope(cur, entry_id, scope)
            chunk = await self._load_chunk(cur, chunk_id)
            if entry is None or chunk is None or chunk.entry_id != entry.id:
                continue
            semantic_matched = normalized_score >= semantic_min_score
            score = normalized_score if semantic_matched else 0.0
            reason = "semantic chunk match"
            preview_text = chunk.text
            score_normalized = normalized_score if semantic_matched else None
            if query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
                chunks = await self._load_chunks(
                    cur,
                    entry.id,
                    revision=entry.revision,
                )
                keyword_score, keyword_chunk, keyword_reason, keyword_preview = _score_entry(
                    entry,
                    chunks,
                    query,
                )
                if keyword_score > 0:
                    keyword_boost = min(keyword_score, 10.0) / 10.0
                    score += self.hybrid_keyword_weight * keyword_boost
                    if keyword_chunk is not None:
                        chunk = keyword_chunk
                    reason = (
                        f"hybrid semantic chunk match; {keyword_reason}"
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
        return scored, byte_truncated

    def _merge_keyword_hits(
        self,
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]
        ],
        keyword_result: KnowledgeSearchResult,
    ) -> list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]]:
        merged = list(scored)
        seen_entry_ids = {entry.id for _, entry, _, _, _, _, _ in merged}
        for hit in keyword_result.hits:
            if hit.entry.id in seen_entry_ids:
                continue
            if hit.score is None:
                continue
            keyword_boost = min(float(hit.score), 10.0) / 10.0
            score = self.hybrid_keyword_weight * keyword_boost
            if score <= 0:
                continue
            text_preview = hit.text_preview
            if text_preview is None:
                continue
            seen_entry_ids.add(hit.entry.id)
            merged.append(
                (
                    score,
                    hit.entry,
                    hit.chunk,
                    f"hybrid keyword match; {hit.reason or 'keyword match'}",
                    hit.text_preview or hit.entry.title or hit.entry.id,
                    None,
                    hit.text_preview_complete,
                )
            )
        merged.sort(
            key=lambda item: (
                -item[0],
                -(item[1].importance or 0.0),
                -item[1].updated_at.timestamp(),
                item[1].id,
            )
        )
        return merged

    async def _index_chunks_with_readiness(
        self,
        chunks: list[KnowledgeChunk],
        *,
        attempt_id: str,
        operation_prefix: str,
        access_scope: KnowledgeAccessScope,
        refresh_existing: bool = False,
    ) -> tuple[int, int]:
        await self._ensure_ready()
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
            stored = await self._embedding_identity_exists(identity)
            if (
                not refresh_existing
                and current is not None
                and current.state is KnowledgeIndexState.READY
                and stored
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
            if stored and not refresh_existing:
                if not await self._chunk_identity_is_current(identity):
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
            for _, identity, readiness in pending:
                if not await self._chunk_identity_is_current(identity):
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
        stored_identities = {
            _knowledge_embedding_identity_sha256(identity)
            for identity in projection_result.stored_identities
        }
        for _, identity, readiness in pending:
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            if identity_sha256 not in stored_identities:
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
        for projection in copied:
            if not self._embedding_identity_matches_configuration(projection.identity):
                raise ValueError("Embedding projection identity does not match this store.")
        if not copied:
            return KnowledgeEmbeddingProjectionWriteResult(
                submitted_records=0,
                stored_identities=[],
            )
        await self._ensure_ready()
        now = datetime.now(UTC)
        access_sql, access_params = knowledge_postgres._postgres_knowledge_access_scope_filter_sql(
            scope,
            entry_alias="e",
        )
        rows: list[tuple[object, ...]] = []
        identity_by_sha256: dict[str, KnowledgeEmbeddingIdentity] = {}
        requested_markers: dict[str, tuple[int, str, str]] = {}
        for projection in copied:
            identity = projection.identity
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            vector_sha256 = _knowledge_embedding_vector_sha256(projection.vector)
            identity_by_sha256[identity_sha256] = identity
            requested_markers[identity_sha256] = (
                projection.readiness_sequence,
                projection.attempt_id,
                vector_sha256,
            )
            rows.append(
                (
                    identity_sha256,
                    identity.entry_id,
                    identity.entry_revision,
                    identity.chunk_id,
                    identity.projection_type,
                    identity.projection_content_hash,
                    identity.embedding_model,
                    identity.dimensions,
                    identity.preprocessing_version,
                    identity.generator,
                    identity.generator_version,
                    identity.index_representation_version,
                    projection.readiness_sequence,
                    projection.attempt_id,
                    False,
                    vector_sha256,
                    _postgres_vector_literal(projection.vector),
                    now,
                    now,
                    identity_sha256,
                    projection.readiness_sequence,
                    projection.attempt_id,
                    identity.chunk_id,
                    identity.entry_id,
                    identity.entry_revision,
                    identity.projection_content_hash,
                    *access_params,
                )
            )
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            # Keep the readiness pointer stable while accepting and activating
            # projection attempts. Readiness publication updates the same rows,
            # so this ordered row lock serializes a batch without adding a
            # second lock namespace or one advisory-lock round trip per vector.
            await self._lock_embedding_projection_readiness(
                cur,
                list(identity_by_sha256),
            )
            await cur.executemany(
                cast(
                    "LiteralString",
                    f"""
                INSERT INTO cayu_knowledge_embeddings (
                    identity_sha256,
                    entry_id,
                    entry_revision,
                    chunk_id,
                    projection_type,
                    projection_content_hash,
                    embedding_model,
                    dimensions,
                    preprocessing_version,
                    generator,
                    generator_version,
                    index_representation_version,
                    readiness_sequence,
                    attempt_id,
                    current_projection,
                    embedding_sha256,
                    embedding,
                    created_at,
                    updated_at
                )
                SELECT
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s
                FROM cayu_knowledge_chunks AS c
                JOIN cayu_knowledge_current_entries AS e
                  ON e.id = c.entry_id
                 AND e.revision = c.entry_revision
                JOIN cayu_knowledge_index_readiness_current AS readiness_current
                  ON readiness_current.identity_sha256 = %s
                 AND readiness_current.sequence = %s
                JOIN cayu_knowledge_index_readiness_events AS readiness
                  ON readiness.sequence = readiness_current.sequence
                 AND readiness.identity_sha256 = readiness_current.identity_sha256
                 AND readiness.state = 'pending'
                 AND readiness.attempt_id = %s
                WHERE c.id = %s
                  AND c.entry_id = %s
                  AND c.entry_revision = %s
                  AND 'sha256:' || encode(sha256(convert_to(c.text, 'UTF8')), 'hex') = %s
                  {access_sql}
                ON CONFLICT (identity_sha256, readiness_sequence) DO NOTHING
                """,
                ),
                rows,
            )
            await cur.execute(
                """
                WITH latest AS (
                    SELECT identity_sha256, MAX(readiness_sequence) AS readiness_sequence
                    FROM cayu_knowledge_embeddings
                    WHERE identity_sha256 = ANY(%s)
                    GROUP BY identity_sha256
                )
                SELECT latest.identity_sha256, latest.readiness_sequence
                FROM latest
                LEFT JOIN cayu_knowledge_embeddings AS current
                  ON current.identity_sha256 = latest.identity_sha256
                 AND current.current_projection
                WHERE current.readiness_sequence IS DISTINCT FROM latest.readiness_sequence
                ORDER BY latest.identity_sha256
                """,
                (list(identity_by_sha256),),
            )
            activation_rows = await cur.fetchall()
            if activation_rows:
                activation_identity_sha256s = [str(row[0]) for row in activation_rows]
                activation_sequences = [int(row[1]) for row in activation_rows]
                await cur.execute(
                    """
                    UPDATE cayu_knowledge_embeddings
                    SET current_projection = FALSE
                    WHERE identity_sha256 = ANY(%s)
                      AND current_projection
                    """,
                    (activation_identity_sha256s,),
                )
                await cur.execute(
                    """
                    UPDATE cayu_knowledge_embeddings AS embedding
                    SET current_projection = TRUE
                    FROM unnest(%s::text[], %s::bigint[])
                         AS target(identity_sha256, readiness_sequence)
                    WHERE embedding.identity_sha256 = target.identity_sha256
                      AND embedding.readiness_sequence = target.readiness_sequence
                      AND NOT embedding.current_projection
                    """,
                    (activation_identity_sha256s, activation_sequences),
                )
                if cur.rowcount != len(activation_rows):
                    raise RuntimeError(
                        "Postgres embedding projection activation lost a selected identity."
                    )
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                SELECT embedding.identity_sha256,
                       embedding.readiness_sequence,
                       embedding.attempt_id,
                       embedding.embedding_sha256
                FROM cayu_knowledge_embeddings AS embedding
                JOIN cayu_knowledge_chunks AS c
                  ON c.id = embedding.chunk_id
                 AND c.entry_id = embedding.entry_id
                 AND c.entry_revision = embedding.entry_revision
                JOIN cayu_knowledge_current_entries AS e
                  ON e.id = c.entry_id
                 AND e.revision = c.entry_revision
                JOIN cayu_knowledge_index_readiness_current AS readiness_current
                  ON readiness_current.identity_sha256 = embedding.identity_sha256
                 AND readiness_current.sequence = embedding.readiness_sequence
                JOIN cayu_knowledge_index_readiness_events AS readiness
                  ON readiness.sequence = readiness_current.sequence
                 AND readiness.identity_sha256 = readiness_current.identity_sha256
                 AND readiness.state = 'pending'
                 AND readiness.attempt_id = embedding.attempt_id
                WHERE embedding.identity_sha256 = ANY(%s)
                  AND embedding.projection_content_hash =
                      'sha256:' || encode(sha256(convert_to(c.text, 'UTF8')), 'hex')
                  {access_sql}
                """,
                ),
                [list(identity_by_sha256), *access_params],
            )
            accepted_sha256s: set[str] = set()
            conflicting_sha256s: set[str] = set()
            for row in await cur.fetchall():
                identity_sha256 = str(row[0])
                requested = requested_markers.get(identity_sha256)
                if requested is None or requested[:2] != (int(row[1]), str(row[2])):
                    continue
                if requested[2] != str(row[3]):
                    conflicting_sha256s.add(identity_sha256)
                else:
                    accepted_sha256s.add(identity_sha256)
            if conflicting_sha256s:
                await conn.rollback()
                raise KnowledgeEmbeddingProjectionConflict("attempt_vector_conflict")
            await conn.commit()
        return KnowledgeEmbeddingProjectionWriteResult(
            submitted_records=len(copied),
            stored_identities=[
                identity_by_sha256[identity_sha256]
                for identity_sha256 in identity_by_sha256
                if identity_sha256 in accepted_sha256s
            ],
        )

    @staticmethod
    async def _lock_embedding_projection_readiness(
        cur: Any,
        identity_sha256s: list[str],
    ) -> None:
        await cur.execute(
            """
            SELECT identity_sha256
            FROM cayu_knowledge_index_readiness_current
            WHERE identity_sha256 = ANY(%s)
            ORDER BY identity_sha256
            FOR UPDATE
            """,
            (identity_sha256s,),
        )
        await cur.fetchall()

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

    async def _embedding_identity_exists(self, identity: KnowledgeEmbeddingIdentity) -> bool:
        identity_sha256 = _knowledge_embedding_identity_sha256(identity)
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT entry_id, entry_revision, chunk_id, projection_type,
                       projection_content_hash, embedding_model, dimensions,
                       preprocessing_version, generator, generator_version,
                       index_representation_version
                FROM cayu_knowledge_embeddings
                WHERE identity_sha256 = %s
                """,
                (identity_sha256,),
            )
            row = await cur.fetchone()
        if row is None:
            return False
        actual = (
            str(row[0]),
            int(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[5]),
            int(row[6]),
            str(row[7]),
            str(row[8]),
            str(row[9]),
            str(row[10]),
        )
        expected = (
            identity.entry_id,
            identity.entry_revision,
            identity.chunk_id,
            identity.projection_type,
            identity.projection_content_hash,
            identity.embedding_model,
            identity.dimensions,
            identity.preprocessing_version,
            identity.generator,
            identity.generator_version,
            identity.index_representation_version,
        )
        if actual != expected:
            raise RuntimeError("Postgres knowledge embedding identity digest collision.")
        return True

    async def _chunk_identity_is_current(self, identity: KnowledgeEmbeddingIdentity) -> bool:
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT c.text
                FROM cayu_knowledge_chunks AS c
                JOIN cayu_knowledge_entries AS logical
                  ON logical.id = c.entry_id
                 AND logical.current_revision = c.entry_revision
                WHERE c.id = %s AND c.entry_id = %s AND c.entry_revision = %s
                """,
                (identity.chunk_id, identity.entry_id, identity.entry_revision),
            )
            row = await cur.fetchone()
        if row is None:
            return False
        content_hash = f"sha256:{sha256(str(row[0]).encode('utf-8')).hexdigest()}"
        return content_hash == identity.projection_content_hash

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

    async def _drop_stale_entry_embeddings(
        self,
        entry_id: str,
        *,
        limit: int,
    ) -> tuple[int, bool]:
        if isinstance(limit, bool) or type(limit) is not int or limit < 0:
            raise ValueError("`limit` must be a nonnegative integer.")
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT DISTINCT embedding.identity_sha256
                FROM cayu_knowledge_embeddings AS embedding
                WHERE embedding.entry_id = %s
                  AND NOT EXISTS (
                      SELECT 1
                      FROM cayu_knowledge_chunks AS current_chunk
                      JOIN cayu_knowledge_entries AS logical
                        ON logical.id = current_chunk.entry_id
                       AND logical.current_revision = current_chunk.entry_revision
                      WHERE current_chunk.id = embedding.chunk_id
                        AND current_chunk.entry_id = embedding.entry_id
                        AND current_chunk.entry_revision = embedding.entry_revision
                  )
                ORDER BY embedding.identity_sha256
                LIMIT %s
                """,
                (entry_id, limit + 1),
            )
            stale_ids = [str(row[0]) for row in await cur.fetchall()]
            selected = stale_ids[:limit]
            if selected:
                await cur.execute(
                    """
                    DELETE FROM cayu_knowledge_embeddings AS embedding
                    WHERE embedding.identity_sha256 = ANY(%s)
                      AND embedding.entry_id = %s
                      AND NOT EXISTS (
                          SELECT 1
                          FROM cayu_knowledge_chunks AS current_chunk
                          JOIN cayu_knowledge_entries AS logical
                            ON logical.id = current_chunk.entry_id
                           AND logical.current_revision = current_chunk.entry_revision
                          WHERE current_chunk.id = embedding.chunk_id
                            AND current_chunk.entry_id = embedding.entry_id
                            AND current_chunk.entry_revision = embedding.entry_revision
                      )
                    """,
                    (selected, entry_id),
                )
                removed = len(selected)
            else:
                removed = 0
            await conn.commit()
        return removed, len(stale_ids) > limit

    async def _drop_entry_embeddings(
        self,
        entry_id: str,
        *,
        expected_deleted_revision: int | None,
        limit: int,
    ) -> tuple[int, bool]:
        if isinstance(limit, bool) or type(limit) is not int or limit < 0:
            raise ValueError("`limit` must be a nonnegative integer.")
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            if expected_deleted_revision is None:
                await cur.execute(
                    """
                    SELECT DISTINCT embedding.identity_sha256
                    FROM cayu_knowledge_embeddings AS embedding
                    WHERE embedding.entry_id = %s
                      AND NOT EXISTS (
                          SELECT 1
                          FROM cayu_knowledge_entries AS logical
                          WHERE logical.id = embedding.entry_id
                      )
                    ORDER BY embedding.identity_sha256
                    LIMIT %s
                    """,
                    (entry_id, limit + 1),
                )
            else:
                await cur.execute(
                    """
                    SELECT DISTINCT embedding.identity_sha256
                    FROM cayu_knowledge_embeddings AS embedding
                    WHERE embedding.entry_id = %s
                      AND EXISTS (
                          SELECT 1
                          FROM cayu_knowledge_entries AS logical
                          JOIN cayu_knowledge_revisions AS revision
                            ON revision.entry_id = logical.id
                           AND revision.revision = logical.current_revision
                          WHERE logical.id = embedding.entry_id
                            AND logical.current_revision = %s
                            AND revision.status = 'deleted'
                      )
                    ORDER BY embedding.identity_sha256
                    LIMIT %s
                    """,
                    (entry_id, expected_deleted_revision, limit + 1),
                )
            stale_ids = [str(row[0]) for row in await cur.fetchall()]
            selected = stale_ids[:limit]
            if selected:
                if expected_deleted_revision is None:
                    await cur.execute(
                        """
                        DELETE FROM cayu_knowledge_embeddings AS embedding
                        WHERE embedding.identity_sha256 = ANY(%s)
                          AND embedding.entry_id = %s
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_knowledge_entries AS logical
                              WHERE logical.id = embedding.entry_id
                          )
                        """,
                        (selected, entry_id),
                    )
                else:
                    await cur.execute(
                        """
                        DELETE FROM cayu_knowledge_embeddings AS embedding
                        WHERE embedding.identity_sha256 = ANY(%s)
                          AND embedding.entry_id = %s
                          AND EXISTS (
                              SELECT 1
                              FROM cayu_knowledge_entries AS logical
                              JOIN cayu_knowledge_revisions AS revision
                                ON revision.entry_id = logical.id
                               AND revision.revision = logical.current_revision
                              WHERE logical.id = embedding.entry_id
                                AND logical.current_revision = %s
                                AND revision.status = 'deleted'
                          )
                        """,
                        (selected, entry_id, expected_deleted_revision),
                    )
                removed = len(selected)
            else:
                removed = 0
            await conn.commit()
        return removed, len(stale_ids) > limit


def _postgres_vector_literal(vector: list[float]) -> str:
    values: list[str] = []
    for index, item in enumerate(vector):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"embedding vector item {index} must be a number.")
        number = float(item)
        if number != number or number in {float("inf"), float("-inf")}:
            raise ValueError(f"embedding vector item {index} must be finite.")
        values.append(repr(number))
    if not values:
        raise ValueError("embedding vector cannot be empty.")
    return "[" + ",".join(values) + "]"
