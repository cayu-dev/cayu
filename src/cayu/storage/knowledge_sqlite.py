from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from cayu._validation import (
    copy_label_map,
    require_nonblank,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    KnowledgeQuery,
    KnowledgeRevisionConflict,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    _copy_chunks_for_revision,
    _knowledge_access_snapshot,
    _knowledge_access_snapshot_json,
    _knowledge_publication_operation_id,
    _knowledge_scope_allows_snapshot,
    _next_knowledge_revision,
    _parse_knowledge_access_snapshot_json,
    _require_knowledge_entry_access,
    _require_knowledge_successor_access,
    _validate_knowledge_publication_replay,
    _validate_knowledge_revision,
    _validate_revision_append,
    _validate_revision_successor,
    copy_knowledge_access_scope,
    copy_knowledge_chunk,
    copy_knowledge_entry,
    copy_knowledge_list_query,
    copy_knowledge_publication_receipt,
    copy_knowledge_query,
    prepare_knowledge_publication,
)

_SEARCH_TOKEN_RE = re.compile(r"\w+")
_SEARCH_PAGE_SIZE = 500
_CHUNK_ID_LOOKUP_BATCH_SIZE = 400
_SQLITE_MIN_REQUIRED_REVISION = 42


class SQLiteKnowledgeStore(KnowledgeStore):
    """SQLite-backed durable knowledge store with FTS5 keyword search."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteKnowledgeStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._default_access_scope = (
            None if access_scope is None else copy_knowledge_access_scope(access_scope)
        )
        self._schema_mode = schema_mode
        self._lock = asyncio.Lock()
        self._connection = sqlite_support.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection,
                schema_mode,
                app_min_supported=_SQLITE_MIN_REQUIRED_REVISION,
            )
        except BaseException:
            self._connection.close()
            raise

    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        if evidence:
            raise NotImplementedError(
                "SQLiteKnowledgeStore does not support knowledge evidence yet."
            )
        scope = self._operation_access_scope(access_scope)
        entry = copy_knowledge_entry(entry)
        _validate_revision_append(entry, expected_revision=None)
        _require_knowledge_entry_access(scope, entry, operation="create_entry")
        copied_chunks = (
            [_default_chunk_for_entry(entry)]
            if chunks is None
            else _copy_entry_chunks(entry.id, entry.revision, chunks)
        )
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                existing_entry = self._load_entry_unlocked(entry.id)
                if existing_entry is not None:
                    _require_knowledge_entry_access(
                        scope,
                        existing_entry,
                        operation="create_entry",
                    )
                    raise KnowledgeRevisionConflict(
                        entry.id,
                        expected_revision=None,
                        actual_revision=existing_entry.revision,
                    )
                self._require_chunk_ids_available_unlocked(
                    copied_chunks,
                    access_scope=scope,
                    operation="create_entry",
                )
                self._insert_entry_unlocked(entry)
                self._insert_chunks_unlocked(entry, copied_chunks)
            return copy_knowledge_entry(entry)

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        if evidence:
            raise NotImplementedError(
                "SQLiteKnowledgeStore does not support knowledge evidence yet."
            )
        scope = self._operation_access_scope(access_scope)
        entry = copy_knowledge_entry(entry)
        _validate_revision_append(entry, expected_revision=expected_revision)
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                self._append_revision_unlocked(
                    entry,
                    expected_revision=expected_revision,
                    chunks=chunks,
                    access_scope=scope,
                    operation="append_entry_revision",
                )
        return copy_knowledge_entry(entry)

    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                entry = self._load_entry_in_scope_unlocked(
                    clean_id,
                    scope,
                    revision=revision,
                )
                return None if entry is None else copy_knowledge_entry(entry)

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
        scope = self._operation_access_scope(access_scope)
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if not isinstance(from_status, KnowledgeStatus):
            raise ValueError("from_status must be a KnowledgeStatus.")
        if not isinstance(to_status, KnowledgeStatus):
            raise ValueError("to_status must be a KnowledgeStatus.")
        expected_namespace = (
            require_clean_nonblank(expected_namespace, "expected_namespace")
            if expected_namespace is not None
            else None
        )
        expected_labels = copy_label_map(expected_labels or {}, "expected_labels")
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                entry = self._load_entry_unlocked(clean_id)
                if entry is None:
                    raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
                _require_knowledge_entry_access(
                    scope,
                    entry,
                    operation="transition_entry_status",
                )
                if entry.revision != expected_revision:
                    raise KnowledgeRevisionConflict(
                        clean_id,
                        expected_revision=expected_revision,
                        actual_revision=entry.revision,
                    )
                if expected_namespace is not None and entry.namespace != expected_namespace:
                    raise ValueError(
                        f"Knowledge entry {clean_id!r} does not match expected namespace."
                    )
                for key, value in expected_labels.items():
                    if entry.labels.get(key) != value:
                        raise ValueError(
                            f"Knowledge entry {clean_id!r} does not match expected labels."
                        )
                if entry.status is not from_status:
                    raise ValueError(
                        f"Knowledge entry {clean_id!r} is {entry.status.value!r}, "
                        f"not {from_status.value!r}."
                    )
                target = entry.model_copy(
                    update={
                        "revision": _next_knowledge_revision(expected_revision),
                        "status": to_status,
                        "updated_at": max(
                            datetime.now(UTC),
                            entry.created_at,
                            entry.updated_at,
                        ),
                    }
                )
                self._append_revision_unlocked(
                    target,
                    expected_revision=expected_revision,
                    chunks=None,
                    access_scope=scope,
                    operation="transition_entry_status",
                )
                return copy_knowledge_entry(target)

    async def delete_entry(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if type(hard) is not bool:
            raise ValueError("`hard` must be a boolean.")
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                entry = self._load_entry_unlocked(clean_id)
                if entry is None:
                    return None
                _require_knowledge_entry_access(scope, entry, operation="delete_entry")
                if entry.revision != expected_revision:
                    raise KnowledgeRevisionConflict(
                        clean_id,
                        expected_revision=expected_revision,
                        actual_revision=entry.revision,
                    )
                if hard:
                    self._delete_chunks_unlocked(clean_id)
                    self._connection.execute(
                        "DELETE FROM cayu_knowledge_entries WHERE id = ?",
                        (clean_id,),
                    )
                    return copy_knowledge_entry(entry)
                target = entry.model_copy(
                    update={
                        "revision": _next_knowledge_revision(expected_revision),
                        "status": KnowledgeStatus.DELETED,
                        "updated_at": max(
                            datetime.now(UTC),
                            entry.created_at,
                            entry.updated_at,
                        ),
                    }
                )
                self._append_revision_unlocked(
                    target,
                    expected_revision=expected_revision,
                    chunks=None,
                    access_scope=scope,
                    operation="delete_entry",
                )
                return copy_knowledge_entry(target)

    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        scope = self._operation_access_scope(access_scope)
        cutoff = datetime.now(UTC) if now is None else now
        access_sql, access_params = _knowledge_access_scope_filter_sql(
            scope,
            now=cutoff,
        )
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                rows = self._connection.execute(
                    "SELECT id FROM cayu_knowledge_current_entries "
                    "AS e WHERE expires_at IS NOT NULL AND expires_at <= ? "
                    f"{access_sql}",
                    [sqlite_support.format_datetime(cutoff), *access_params],
                ).fetchall()
                expired_ids = [str(row["id"]) for row in rows]
                if not expired_ids:
                    return 0
                # FTS is a virtual table (no FK cascade), so clear chunks/FTS explicitly; the
                # entries DELETE then cascades to labels/aspects/impact_targets.
                for entry_id in expired_ids:
                    self._delete_chunks_unlocked(entry_id)
                self._connection.executemany(
                    "DELETE FROM cayu_knowledge_entries WHERE id = ?",
                    [(entry_id,) for entry_id in expired_ids],
                )
            return len(expired_ids)

    async def publish_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> KnowledgePublicationReceipt:
        scope = self._operation_access_scope(access_scope)
        (
            operation_id,
            copied_entry,
            copied_chunks,
            copied_evidence,
            request_sha256,
        ) = prepare_knowledge_publication(
            entry,
            chunks,
            evidence=evidence,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )
        if copied_evidence:
            raise NotImplementedError(
                "SQLiteKnowledgeStore does not support knowledge evidence yet."
            )
        _require_knowledge_entry_access(scope, copied_entry, operation="publish_entry_revision")
        async with self._lock:
            with sqlite_support._transaction(self._connection):
                existing_receipt = self._load_publication_receipt_unlocked(
                    operation_id,
                    access_scope=scope,
                )
                if existing_receipt is not None:
                    _validate_knowledge_publication_replay(
                        existing_receipt,
                        entry=copied_entry,
                        chunks=copied_chunks,
                        evidence=copied_evidence,
                        expected_revision=expected_revision,
                        request_sha256=request_sha256,
                    )
                    return copy_knowledge_publication_receipt(
                        existing_receipt,
                        replayed=True,
                    )
                existing_entry = self._load_entry_unlocked(copied_entry.id)
                actual_revision = None if existing_entry is None else existing_entry.revision
                if existing_entry is not None:
                    _require_knowledge_entry_access(
                        scope,
                        existing_entry,
                        operation="publish_entry_revision",
                    )
                if actual_revision != expected_revision:
                    raise KnowledgeRevisionConflict(
                        copied_entry.id,
                        expected_revision=expected_revision,
                        actual_revision=actual_revision,
                    )
                if existing_entry is not None:
                    _validate_revision_successor(existing_entry, copied_entry)
                self._require_chunk_ids_available_unlocked(
                    copied_chunks,
                    access_scope=scope,
                    operation="publish_entry_revision",
                )
                receipt = KnowledgePublicationReceipt(
                    operation_id=operation_id,
                    entry_id=copied_entry.id,
                    entry_revision=copied_entry.revision,
                    expected_revision=expected_revision,
                    request_sha256=request_sha256,
                    entry_created_at=copied_entry.created_at,
                    entry_updated_at=copied_entry.updated_at,
                    committed_at=datetime.now(UTC),
                )
                if existing_entry is None:
                    self._insert_entry_unlocked(copied_entry)
                else:
                    assert expected_revision is not None
                    self._insert_revision_unlocked(copied_entry)
                    self._advance_current_revision_unlocked(
                        copied_entry,
                        expected_revision=expected_revision,
                    )
                self._insert_chunks_unlocked(copied_entry, copied_chunks)
                self._insert_publication_receipt_unlocked(receipt, copied_entry)
            return copy_knowledge_publication_receipt(receipt)

    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        async with self._lock:
            receipt = self._load_publication_receipt_in_scope_unlocked(operation_id, scope)
        return None if receipt is None else copy_knowledge_publication_receipt(receipt)

    async def read_chunks(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        scope = self._operation_access_scope(access_scope)
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if chunk_index is not None:
            _validate_nonnegative_int(chunk_index, "chunk_index")
        _validate_nonnegative_int(around, "around")
        if chunk_index is None and around != 0:
            raise ValueError("`around` requires `chunk_index`.")
        _validate_positive_int(max_chunks, "max_chunks")
        _validate_positive_int(max_bytes, "max_bytes")
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                entry = self._load_entry_in_scope_unlocked(
                    clean_id,
                    scope,
                    revision=revision,
                )
                if entry is None:
                    return []
                chunks = self._load_chunks_unlocked(clean_id, revision=entry.revision)
        if chunk_index is not None:
            chunks = _center_chunk_window(chunks, chunk_index=chunk_index, max_chunks=max_chunks)
        start_index = 0 if chunk_index is None else max(0, chunk_index - around)
        end_index = None if chunk_index is None else chunk_index + around
        return _bounded_chunks(
            chunks,
            start_index=start_index,
            end_index=end_index,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )

    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_query(query)
        if knowledge_query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("SQLiteKnowledgeStore supports only auto and keyword search modes.")
        fts_query, preview_terms = _sqlite_knowledge_fts_query(knowledge_query)
        none_fts_query = _sqlite_knowledge_none_fts_query(knowledge_query)
        where_sql, params = _knowledge_filter_sql(knowledge_query)
        access_sql, access_params = _knowledge_access_scope_filter_sql(scope)
        where_sql += access_sql
        params.extend(access_params)
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                total_hits_known = self._count_search_hits_unlocked(
                    fts_query,
                    none_fts_query,
                    where_sql,
                    params,
                )
                unique_rows = self._search_unique_rows_unlocked(
                    fts_query=fts_query,
                    none_fts_query=none_fts_query,
                    where_sql=where_sql,
                    params=params,
                    limit=knowledge_query.limit,
                )
                hits, byte_truncated = self._hits_from_search_rows_unlocked(
                    unique_rows,
                    knowledge_query,
                    preview_terms,
                )
        return KnowledgeSearchResult(
            query=knowledge_query,
            hits=hits,
            truncated=byte_truncated or len(hits) < total_hits_known,
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_hits_known=total_hits_known,
        )

    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_list_query(query)
        where_sql, params = _knowledge_list_filter_sql(knowledge_query)
        access_sql, access_params = _knowledge_access_scope_filter_sql(scope)
        where_sql += access_sql
        params.extend(access_params)
        async with self._lock:
            with sqlite_support._transaction(
                self._connection,
                begin_immediate=False,
            ):
                total_entries_known = self._count_list_entries_unlocked(where_sql, params)
                rows = self._connection.execute(
                    f"""
                    SELECT e.id
                    FROM cayu_knowledge_current_entries AS e
                    WHERE 1 = 1
                    {where_sql}
                    ORDER BY COALESCE(e.importance, 0.0) DESC,
                             e.updated_at DESC,
                             e.id ASC
                    LIMIT ?
                    """,
                    [*params, knowledge_query.limit],
                ).fetchall()
                entry_map = self._load_entries_unlocked([str(row["id"]) for row in rows])
                entries = [
                    entry for row in rows if (entry := entry_map.get(str(row["id"]))) is not None
                ]
                facets, facets_truncated = self._list_facets_unlocked(
                    knowledge_query,
                    where_sql,
                    params,
                )
                items, byte_truncated = self._list_items_unlocked(entries, knowledge_query)
        return KnowledgeListResult(
            query=knowledge_query,
            entries=items,
            facets=facets,
            facets_truncated=facets_truncated,
            truncated=byte_truncated or len(items) < total_entries_known or facets_truncated,
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_entries_known=total_entries_known,
        )

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _count_search_hits_unlocked(
        self,
        fts_query: str,
        none_fts_query: str | None,
        where_sql: str,
        params: list[object],
    ) -> int:
        none_sql, none_params = _sqlite_knowledge_none_filter_sql(none_fts_query)
        row = self._connection.execute(
            f"""
            SELECT COUNT(DISTINCT e.id)
            FROM cayu_knowledge_chunks_fts
            JOIN cayu_knowledge_chunks AS c
                ON c.fts_rowid = cayu_knowledge_chunks_fts.rowid
            JOIN cayu_knowledge_current_entries AS e
                ON e.id = c.entry_id AND e.revision = c.entry_revision
            WHERE cayu_knowledge_chunks_fts MATCH ?
            {none_sql}
            {where_sql}
            """,
            [fts_query, *none_params, *params],
        ).fetchone()
        return 0 if row is None else int(row[0])

    def _search_unique_rows_unlocked(
        self,
        *,
        fts_query: str,
        none_fts_query: str | None,
        where_sql: str,
        params: list[object],
        limit: int,
    ) -> list[sqlite3.Row]:
        none_sql, none_params = _sqlite_knowledge_none_filter_sql(none_fts_query)
        unique_rows: list[sqlite3.Row] = []
        seen_entry_ids: set[str] = set()
        offset = 0
        while len(unique_rows) < limit:
            rows = self._connection.execute(
                f"""
                SELECT
                    e.id AS entry_id,
                    c.id AS chunk_id,
                    bm25(cayu_knowledge_chunks_fts) AS fts_score
                FROM cayu_knowledge_chunks_fts
                JOIN cayu_knowledge_chunks AS c
                    ON c.fts_rowid = cayu_knowledge_chunks_fts.rowid
                JOIN cayu_knowledge_current_entries AS e
                    ON e.id = c.entry_id AND e.revision = c.entry_revision
                WHERE cayu_knowledge_chunks_fts MATCH ?
                {none_sql}
                {where_sql}
                ORDER BY fts_score ASC,
                         COALESCE(e.importance, 0.0) DESC,
                         e.updated_at DESC,
                         e.id ASC,
                         c.chunk_index ASC
                LIMIT ? OFFSET ?
                """,
                [fts_query, *none_params, *params, _SEARCH_PAGE_SIZE, offset],
            ).fetchall()
            if not rows:
                break
            for row in rows:
                entry_id = str(row["entry_id"])
                if entry_id in seen_entry_ids:
                    continue
                seen_entry_ids.add(entry_id)
                unique_rows.append(row)
                if len(unique_rows) >= limit:
                    break
            if len(rows) < _SEARCH_PAGE_SIZE:
                break
            offset += _SEARCH_PAGE_SIZE
        return unique_rows

    def _hits_from_search_rows_unlocked(
        self,
        rows: list[sqlite3.Row],
        query: KnowledgeQuery,
        terms: list[str],
    ) -> tuple[list[KnowledgeHit], bool]:
        entries = self._load_entries_unlocked([str(row["entry_id"]) for row in rows])
        chunks = self._load_chunks_by_ids_unlocked([str(row["chunk_id"]) for row in rows])
        hits: list[KnowledgeHit] = []
        remaining = query.max_bytes
        truncated = False
        for row in rows:
            if remaining <= 0:
                truncated = True
                break
            entry = entries.get(str(row["entry_id"]))
            chunk = chunks.get(str(row["chunk_id"]))
            if entry is None or chunk is None:
                continue
            reason, preview_text = _preview_for_match(entry, chunk, terms)
            preview_bytes = len(preview_text.encode("utf-8"))
            preview = _truncate_text_to_bytes(preview_text, remaining)
            if not preview:
                truncated = True
                break
            returned_bytes = len(preview.encode("utf-8"))
            preview_complete = returned_bytes == preview_bytes
            if not preview_complete:
                truncated = True
            remaining -= returned_bytes
            hits.append(
                KnowledgeHit(
                    entry=entry,
                    chunk=chunk,
                    score=-float(row["fts_score"]),
                    score_kind="sqlite_fts5_bm25",
                    rank=len(hits) + 1,
                    reason=reason,
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return hits, truncated

    def _count_list_entries_unlocked(self, where_sql: str, params: list[object]) -> int:
        row = self._connection.execute(
            f"""
            SELECT COUNT(*)
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            """,
            params,
        ).fetchone()
        return 0 if row is None else int(row[0])

    def _list_items_unlocked(
        self,
        entries: list[KnowledgeEntry],
        query: KnowledgeListQuery,
    ) -> tuple[list[KnowledgeListItem], bool]:
        chunk_counts = self._count_chunks_by_entry_unlocked([entry.id for entry in entries])
        items: list[KnowledgeListItem] = []
        remaining = query.max_bytes
        truncated = False
        for entry in entries:
            if remaining <= 0:
                truncated = True
                break
            preview_source = entry.title or entry.text
            preview_bytes = len(preview_source.encode("utf-8"))
            preview = _truncate_text_to_bytes(preview_source, remaining)
            if not preview:
                truncated = True
                break
            returned_bytes = len(preview.encode("utf-8"))
            preview_complete = returned_bytes == preview_bytes
            if not preview_complete:
                truncated = True
            remaining -= returned_bytes
            items.append(
                KnowledgeListItem(
                    entry=entry,
                    chunk_count=chunk_counts.get(entry.id, 0),
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return items, truncated

    def _list_facets_unlocked(
        self,
        query: KnowledgeListQuery,
        where_sql: str,
        params: list[object],
    ) -> tuple[list[KnowledgeFacet], bool]:
        if query.group_by is None:
            return [], False
        rows = self._connection.execute(
            *_sqlite_list_facet_sql(
                query.group_by,
                where_sql,
                params,
                limit=query.limit + 1,
            )
        ).fetchall()
        facets = [
            KnowledgeFacet(
                field=query.group_by,
                key=str(row["key"]) if row["key"] is not None else None,
                value=str(row["value"]),
                count=int(row["count"]),
            )
            for row in rows[: query.limit]
        ]
        return facets, len(rows) > query.limit

    def _insert_entry_unlocked(self, entry: KnowledgeEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id,
                namespace,
                current_revision,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.namespace,
                entry.revision,
                sqlite_support.format_datetime(entry.created_at),
                sqlite_support.format_datetime(entry.updated_at),
            ),
        )
        self._insert_revision_unlocked(entry)

    def _insert_revision_unlocked(self, entry: KnowledgeEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_revisions (
                entry_id,
                revision,
                text,
                kind,
                visibility,
                status,
                created_by_type,
                created_by,
                created_at,
                updated_at,
                source_type,
                source_uri,
                source_id,
                source_hash,
                importance,
                importance_source,
                confidence,
                last_used_at,
                expires_at,
                title,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _entry_row_values(entry),
        )
        if entry.labels:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_labels (entry_id, entry_revision, key, value)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (entry.id, entry.revision, key, value)
                    for key, value in sorted(entry.labels.items())
                ],
            )
        if entry.aspects:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_aspects (entry_id, entry_revision, aspect)
                VALUES (?, ?, ?)
                """,
                [(entry.id, entry.revision, aspect) for aspect in entry.aspects],
            )
        if entry.impact_targets:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_impact_targets (
                    entry_id, entry_revision, impact_target
                )
                VALUES (?, ?, ?)
                """,
                [(entry.id, entry.revision, target) for target in entry.impact_targets],
            )

    def _advance_current_revision_unlocked(
        self,
        entry: KnowledgeEntry,
        *,
        expected_revision: int,
    ) -> None:
        cursor = self._connection.execute(
            """
            UPDATE cayu_knowledge_entries
            SET current_revision = ?, updated_at = ?
            WHERE id = ? AND current_revision = ?
            """,
            (
                entry.revision,
                sqlite_support.format_datetime(entry.updated_at),
                entry.id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            current = self._load_entry_unlocked(entry.id)
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=None if current is None else current.revision,
            )

    def _append_revision_unlocked(
        self,
        entry: KnowledgeEntry,
        *,
        expected_revision: int,
        chunks: list[KnowledgeChunk] | None,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        _validate_revision_append(entry, expected_revision=expected_revision)
        current = self._load_entry_unlocked(entry.id)
        if current is None:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=None,
            )
        _require_knowledge_entry_access(access_scope, current, operation=operation)
        if current.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=current.revision,
            )
        _validate_revision_successor(current, entry)
        _require_knowledge_successor_access(access_scope, entry, operation=operation)
        previous_chunks = self._load_chunks_unlocked(
            entry.id,
            revision=current.revision,
        )
        if chunks is not None:
            copied_chunks = _copy_entry_chunks(entry.id, entry.revision, chunks)
        elif _has_only_default_chunk(current, previous_chunks):
            copied_chunks = [_default_chunk_for_entry(entry)]
        else:
            copied_chunks = _copy_chunks_for_revision(previous_chunks, entry)
        self._require_chunk_ids_available_unlocked(
            copied_chunks,
            access_scope=access_scope,
            operation=operation,
        )
        self._insert_revision_unlocked(entry)
        self._insert_chunks_unlocked(entry, copied_chunks)
        self._advance_current_revision_unlocked(
            entry,
            expected_revision=expected_revision,
        )

    def _insert_chunks_unlocked(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_chunks (
                id,
                entry_id,
                entry_revision,
                chunk_index,
                text,
                content_hash,
                source_uri,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_chunk_row_values(chunk) for chunk in chunks],
        )
        self._insert_entry_fts_unlocked(entry, chunks)

    def _require_chunk_ids_available_unlocked(
        self,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = sorted({chunk.id for chunk in chunks})
        occupied_entry_ids: set[str] = set()
        for offset in range(0, len(proposed_ids), _CHUNK_ID_LOOKUP_BATCH_SIZE):
            batch = proposed_ids[offset : offset + _CHUNK_ID_LOOKUP_BATCH_SIZE]
            placeholders = ", ".join("?" for _ in batch)
            rows = self._connection.execute(
                f"""
                SELECT DISTINCT entry_id
                FROM cayu_knowledge_chunks
                WHERE id IN ({placeholders})
                ORDER BY entry_id
                """,
                batch,
            ).fetchall()
            occupied_entry_ids.update(str(row["entry_id"]) for row in rows)
        for occupied_entry_id in sorted(occupied_entry_ids):
            owner = self._load_entry_unlocked(occupied_entry_id)
            if owner is None:
                raise KnowledgeChunkConflict(operation)
            _require_knowledge_entry_access(
                access_scope,
                owner,
                operation=operation,
            )
        if occupied_entry_ids:
            raise KnowledgeChunkConflict(operation)

    def _load_publication_receipt_unlocked(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgePublicationReceipt | None:
        row = self._connection.execute(
            """
            SELECT
                operation_id,
                entry_id,
                entry_revision,
                expected_revision,
                request_sha256,
                entry_created_at,
                entry_updated_at,
                committed_at,
                access_snapshot_json
            FROM cayu_knowledge_publication_receipts
            WHERE operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row["access_snapshot_json"])
            receipt = KnowledgePublicationReceipt(
                operation_id=row["operation_id"],
                entry_id=row["entry_id"],
                entry_revision=row["entry_revision"],
                expected_revision=row["expected_revision"],
                request_sha256=row["request_sha256"],
                entry_created_at=sqlite_support.parse_datetime(row["entry_created_at"]),
                entry_updated_at=sqlite_support.parse_datetime(row["entry_updated_at"]),
                committed_at=sqlite_support.parse_datetime(row["committed_at"]),
            )
        except Exception:
            raise KnowledgePublicationConflict("malformed_receipt") from None
        if not _knowledge_scope_allows_snapshot(access_scope, snapshot):
            raise KnowledgeAccessDenied("publish_entry_revision")
        return receipt

    def _load_publication_receipt_in_scope_unlocked(
        self,
        operation_id: str,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgePublicationReceipt | None:
        row = self._connection.execute(
            """
            SELECT
                receipt.operation_id,
                receipt.entry_id,
                receipt.entry_revision,
                receipt.expected_revision,
                receipt.request_sha256,
                receipt.entry_created_at,
                receipt.entry_updated_at,
                receipt.committed_at,
                receipt.access_snapshot_json
            FROM cayu_knowledge_publication_receipts AS receipt
            WHERE receipt.operation_id = ?
            """,
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            snapshot = _parse_knowledge_access_snapshot_json(row["access_snapshot_json"])
            if not _knowledge_scope_allows_snapshot(access_scope, snapshot):
                return None
            return KnowledgePublicationReceipt(
                operation_id=row["operation_id"],
                entry_id=row["entry_id"],
                entry_revision=row["entry_revision"],
                expected_revision=row["expected_revision"],
                request_sha256=row["request_sha256"],
                entry_created_at=sqlite_support.parse_datetime(row["entry_created_at"]),
                entry_updated_at=sqlite_support.parse_datetime(row["entry_updated_at"]),
                committed_at=sqlite_support.parse_datetime(row["committed_at"]),
            )
        except Exception:
            raise KnowledgePublicationConflict("malformed_receipt") from None

    def _insert_publication_receipt_unlocked(
        self,
        receipt: KnowledgePublicationReceipt,
        entry: KnowledgeEntry,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_publication_receipts (
                operation_id,
                entry_id,
                entry_revision,
                expected_revision,
                request_sha256,
                entry_created_at,
                entry_updated_at,
                committed_at,
                access_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.operation_id,
                receipt.entry_id,
                receipt.entry_revision,
                receipt.expected_revision,
                receipt.request_sha256,
                sqlite_support.format_datetime(receipt.entry_created_at),
                sqlite_support.format_datetime(receipt.entry_updated_at),
                sqlite_support.format_datetime(receipt.committed_at),
                _knowledge_access_snapshot_json(_knowledge_access_snapshot(entry)),
            ),
        )

    def _delete_chunks_unlocked(self, entry_id: str) -> None:
        self._delete_entry_fts_unlocked(entry_id)
        self._connection.execute(
            "DELETE FROM cayu_knowledge_chunks WHERE entry_id = ?",
            (entry_id,),
        )

    def _delete_entry_fts_unlocked(self, entry_id: str) -> None:
        rowids = self._load_chunk_fts_rowids_unlocked(entry_id)
        if not rowids:
            return
        self._connection.executemany(
            "DELETE FROM cayu_knowledge_chunks_fts WHERE rowid = ?",
            [(rowid,) for rowid in rowids.values()],
        )

    def _insert_entry_fts_unlocked(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> None:
        rowids = self._load_chunk_fts_rowids_unlocked(entry.id, entry.revision)
        expected_chunk_ids = {chunk.id for chunk in chunks}
        if rowids.keys() != expected_chunk_ids:
            raise RuntimeError("SQLite knowledge chunks changed while preparing their FTS rows.")
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_chunks_fts (
                rowid, entry_id, entry_revision, chunk_id, title, text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rowids[chunk.id],
                    entry.id,
                    entry.revision,
                    chunk.id,
                    entry.title or "",
                    _fts_text_for_entry_chunk(entry, chunk),
                )
                for chunk in chunks
            ],
        )

    def _load_chunk_fts_rowids_unlocked(
        self,
        entry_id: str,
        revision: int | None = None,
    ) -> dict[str, int]:
        revision_sql = "" if revision is None else " AND entry_revision = ?"
        params: tuple[object, ...] = (entry_id,) if revision is None else (entry_id, revision)
        rows = self._connection.execute(
            f"""
            SELECT id, fts_rowid
            FROM cayu_knowledge_chunks
            WHERE entry_id = ?
            {revision_sql}
            ORDER BY chunk_index ASC
            """,
            params,
        ).fetchall()
        return {str(row["id"]): int(row["fts_rowid"]) for row in rows}

    def _load_entry_unlocked(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
    ) -> KnowledgeEntry | None:
        if revision is None:
            row = self._connection.execute(
                "SELECT * FROM cayu_knowledge_current_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT
                    logical.id AS id,
                    revision.revision AS revision,
                    logical.namespace AS namespace,
                    revision.*
                FROM cayu_knowledge_entries AS logical
                JOIN cayu_knowledge_revisions AS revision
                  ON revision.entry_id = logical.id
                WHERE logical.id = ? AND revision.revision = ?
                """,
                (entry_id, revision),
            ).fetchone()
        if row is None:
            return None
        selected_revision = int(row["revision"])
        return _entry_from_row(
            row,
            labels=self._load_labels_unlocked(entry_id, selected_revision),
            aspects=self._load_aspects_unlocked(entry_id, selected_revision),
            impact_targets=self._load_impact_targets_unlocked(entry_id, selected_revision),
        )

    def _load_entry_in_scope_unlocked(
        self,
        entry_id: str,
        access_scope: KnowledgeAccessScope,
        *,
        revision: int | None = None,
    ) -> KnowledgeEntry | None:
        access_now = datetime.now(UTC)
        access_sql, access_params = _knowledge_access_scope_filter_sql(
            access_scope,
            now=access_now,
        )
        if revision is None:
            row = self._connection.execute(
                f"""
                SELECT e.*
                FROM cayu_knowledge_current_entries AS e
                WHERE e.id = ?
                {access_sql}
                """,
                [entry_id, *access_params],
            ).fetchone()
        else:
            current_access_sql, current_access_params = _knowledge_access_scope_filter_sql(
                access_scope,
                entry_alias="current_entry",
                now=access_now,
            )
            row = self._connection.execute(
                f"""
                SELECT e.*
                FROM (
                    SELECT
                        logical.id AS id,
                        stored.revision AS revision,
                        logical.namespace AS namespace,
                        stored.text AS text,
                        stored.kind AS kind,
                        stored.visibility AS visibility,
                        stored.status AS status,
                        stored.created_by_type AS created_by_type,
                        stored.created_by AS created_by,
                        stored.created_at AS created_at,
                        stored.updated_at AS updated_at,
                        stored.source_type AS source_type,
                        stored.source_uri AS source_uri,
                        stored.source_id AS source_id,
                        stored.source_hash AS source_hash,
                        stored.importance AS importance,
                        stored.importance_source AS importance_source,
                        stored.confidence AS confidence,
                        stored.last_used_at AS last_used_at,
                        stored.expires_at AS expires_at,
                        stored.title AS title,
                        stored.metadata_json AS metadata_json
                    FROM cayu_knowledge_entries AS logical
                    JOIN cayu_knowledge_revisions AS stored
                      ON stored.entry_id = logical.id
                    WHERE logical.id = ? AND stored.revision = ?
                ) AS e
                JOIN cayu_knowledge_current_entries AS current_entry
                  ON current_entry.id = e.id
                WHERE TRUE
                {access_sql}
                {current_access_sql}
                """,
                [
                    entry_id,
                    revision,
                    *access_params,
                    *current_access_params,
                ],
            ).fetchone()
        if row is None:
            return None
        selected_revision = int(row["revision"])
        return _entry_from_row(
            row,
            labels=self._load_labels_unlocked(entry_id, selected_revision),
            aspects=self._load_aspects_unlocked(entry_id, selected_revision),
            impact_targets=self._load_impact_targets_unlocked(entry_id, selected_revision),
        )

    def _load_chunks_unlocked(
        self,
        entry_id: str,
        *,
        revision: int,
    ) -> list[KnowledgeChunk]:
        rows = self._connection.execute(
            """
            SELECT *
            FROM cayu_knowledge_chunks
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY chunk_index ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def _load_entries_unlocked(self, entry_ids: list[str]) -> dict[str, KnowledgeEntry]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self._connection.execute(
            f"SELECT * FROM cayu_knowledge_current_entries WHERE id IN ({placeholders})",
            unique_ids,
        ).fetchall()
        labels = self._load_labels_for_entries_unlocked(unique_ids)
        aspects = self._load_aspects_for_entries_unlocked(unique_ids)
        impact_targets = self._load_impact_targets_for_entries_unlocked(unique_ids)
        return {
            row["id"]: _entry_from_row(
                row,
                labels=labels.get(row["id"], {}),
                aspects=aspects.get(row["id"], []),
                impact_targets=impact_targets.get(row["id"], []),
            )
            for row in rows
        }

    def _load_chunks_by_ids_unlocked(self, chunk_ids: list[str]) -> dict[str, KnowledgeChunk]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self._connection.execute(
            f"SELECT * FROM cayu_knowledge_chunks WHERE id IN ({placeholders})",
            unique_ids,
        ).fetchall()
        return {row["id"]: _chunk_from_row(row) for row in rows}

    def _count_chunks_by_entry_unlocked(self, entry_ids: list[str]) -> dict[str, int]:
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self._connection.execute(
            f"""
            SELECT chunk.entry_id, COUNT(*) AS chunk_count
            FROM cayu_knowledge_chunks AS chunk
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = chunk.entry_id
             AND logical.current_revision = chunk.entry_revision
            WHERE chunk.entry_id IN ({placeholders})
            GROUP BY chunk.entry_id
            """,
            unique_ids,
        ).fetchall()
        return {row["entry_id"]: int(row["chunk_count"]) for row in rows}

    def _load_labels_for_entries_unlocked(
        self,
        entry_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        if not entry_ids:
            return {}
        placeholders = ", ".join("?" for _ in entry_ids)
        rows = self._connection.execute(
            f"""
            SELECT label.entry_id, label.key, label.value
            FROM cayu_knowledge_labels AS label
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = label.entry_id
             AND logical.current_revision = label.entry_revision
            WHERE label.entry_id IN ({placeholders})
            ORDER BY label.entry_id ASC, label.key ASC
            """,
            entry_ids,
        ).fetchall()
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            result.setdefault(row["entry_id"], {})[row["key"]] = row["value"]
        return result

    def _load_aspects_for_entries_unlocked(
        self,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        placeholders = ", ".join("?" for _ in entry_ids)
        rows = self._connection.execute(
            f"""
            SELECT aspect.entry_id, aspect.aspect
            FROM cayu_knowledge_aspects AS aspect
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = aspect.entry_id
             AND logical.current_revision = aspect.entry_revision
            WHERE aspect.entry_id IN ({placeholders})
            ORDER BY aspect.entry_id ASC, aspect.aspect ASC
            """,
            entry_ids,
        ).fetchall()
        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(row["entry_id"], []).append(row["aspect"])
        return result

    def _load_impact_targets_for_entries_unlocked(
        self,
        entry_ids: list[str],
    ) -> dict[str, list[str]]:
        if not entry_ids:
            return {}
        placeholders = ", ".join("?" for _ in entry_ids)
        rows = self._connection.execute(
            f"""
            SELECT target.entry_id, target.impact_target
            FROM cayu_knowledge_impact_targets AS target
            JOIN cayu_knowledge_entries AS logical
              ON logical.id = target.entry_id
             AND logical.current_revision = target.entry_revision
            WHERE target.entry_id IN ({placeholders})
            ORDER BY target.entry_id ASC, target.impact_target ASC
            """,
            entry_ids,
        ).fetchall()
        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(row["entry_id"], []).append(row["impact_target"])
        return result

    def _load_labels_unlocked(self, entry_id: str, revision: int) -> dict[str, str]:
        rows = self._connection.execute(
            """
            SELECT key, value
            FROM cayu_knowledge_labels
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY key ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _load_aspects_unlocked(self, entry_id: str, revision: int) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT aspect
            FROM cayu_knowledge_aspects
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY aspect ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return [row["aspect"] for row in rows]

    def _load_impact_targets_unlocked(self, entry_id: str, revision: int) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT impact_target
            FROM cayu_knowledge_impact_targets
            WHERE entry_id = ? AND entry_revision = ?
            ORDER BY impact_target ASC
            """,
            (entry_id, revision),
        ).fetchall()
        return [row["impact_target"] for row in rows]


def _knowledge_filter_sql(query: KnowledgeQuery) -> tuple[str, list[object]]:
    return _knowledge_metadata_filter_sql(
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _knowledge_list_filter_sql(query: KnowledgeListQuery) -> tuple[str, list[object]]:
    return _knowledge_metadata_filter_sql(
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _knowledge_access_scope_filter_sql(
    scope: KnowledgeAccessScope,
    *,
    entry_alias: str = "e",
    now: datetime | None = None,
) -> tuple[str, list[object]]:
    if entry_alias not in {"e", "current_entry"}:
        raise ValueError("Unsupported knowledge access-filter alias.")
    clauses: list[str] = []
    params: list[object] = []
    if not scope.allow_all_namespaces:
        placeholders = ", ".join("?" for _ in scope.allowed_namespaces)
        clauses.append(f"{entry_alias}.namespace IN ({placeholders})")
        params.extend(scope.allowed_namespaces)
    for key, value in scope.required_labels.items():
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_labels AS access_label
                WHERE access_label.entry_id = {entry_alias}.id
                  AND access_label.entry_revision = {entry_alias}.revision
                  AND access_label.key = ?
                  AND access_label.value = ?
            )
            """
        )
        params.extend([key, value])
    visibility_placeholders = ", ".join("?" for _ in scope.allowed_visibilities)
    clauses.append(f"{entry_alias}.visibility IN ({visibility_placeholders})")
    params.extend(str(visibility) for visibility in scope.allowed_visibilities)
    status_placeholders = ", ".join("?" for _ in scope.allowed_statuses)
    clauses.append(f"{entry_alias}.status IN ({status_placeholders})")
    params.extend(str(status) for status in scope.allowed_statuses)
    if scope.allowed_source_types is not None:
        if scope.allowed_source_types:
            placeholders = ", ".join("?" for _ in scope.allowed_source_types)
            clauses.append(f"{entry_alias}.source_type IN ({placeholders})")
            params.extend(scope.allowed_source_types)
        else:
            clauses.append("0")
    if scope.allowed_source_ids is not None:
        if scope.allowed_source_ids:
            placeholders = ", ".join("?" for _ in scope.allowed_source_ids)
            clauses.append(f"{entry_alias}.source_id IN ({placeholders})")
            params.extend(scope.allowed_source_ids)
        else:
            clauses.append("0")
    if not scope.include_expired:
        clauses.append(f"({entry_alias}.expires_at IS NULL OR {entry_alias}.expires_at > ?)")
        params.append(sqlite_support.format_datetime(datetime.now(UTC) if now is None else now))
    return " AND " + " AND ".join(clauses), params


def _knowledge_metadata_filter_sql(
    *,
    namespace: str | None,
    labels: dict[str, str],
    kinds: list[str] | None,
    statuses: list[KnowledgeStatus],
    visibilities: list[KnowledgeVisibility] | None,
    aspects: list[str],
    impact_targets: list[str],
    source_type: str | None,
    source_id: str | None,
    include_expired: bool,
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if namespace is not None:
        clauses.append("e.namespace = ?")
        params.append(namespace)
    for key, value in labels.items():
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_labels AS label
                WHERE label.entry_id = e.id
                  AND label.entry_revision = e.revision
                  AND label.key = ?
                  AND label.value = ?
            )
            """
        )
        params.extend([key, value])
    if kinds is not None:
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            clauses.append(f"e.kind IN ({placeholders})")
            params.extend(kinds)
        else:
            clauses.append("0")
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        clauses.append(f"e.status IN ({placeholders})")
        params.extend(str(status) for status in statuses)
    if visibilities is not None:
        placeholders = ", ".join("?" for _ in visibilities)
        clauses.append(f"e.visibility IN ({placeholders})")
        params.extend(str(visibility) for visibility in visibilities)
    if source_type is not None:
        clauses.append("e.source_type = ?")
        params.append(source_type)
    if source_id is not None:
        clauses.append("e.source_id = ?")
        params.append(source_id)
    if aspects:
        placeholders = ", ".join("?" for _ in aspects)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_aspects AS aspect
                WHERE aspect.entry_id = e.id
                  AND aspect.entry_revision = e.revision
                  AND aspect.aspect IN ({placeholders})
            )
            """
        )
        params.extend(aspects)
    if impact_targets:
        placeholders = ", ".join("?" for _ in impact_targets)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM cayu_knowledge_impact_targets AS target
                WHERE target.entry_id = e.id
                  AND target.entry_revision = e.revision
                  AND target.impact_target IN ({placeholders})
            )
            """
        )
        params.extend(impact_targets)
    if not include_expired:
        clauses.append("(e.expires_at IS NULL OR e.expires_at > ?)")
        params.append(sqlite_support.format_datetime(datetime.now(UTC)))
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _sqlite_knowledge_fts_query(query: KnowledgeQuery) -> tuple[str, list[str]]:
    any_terms = _dedupe_search_tokens(
        [
            *_expand_search_tokens(_tokenize_search_text(query.text or "")),
            *(
                token
                for term in query.any_terms
                for group in _structured_search_token_groups(term)
                for token in group
            ),
        ]
    )
    all_groups = _dedupe_search_token_groups(
        [group for term in query.all_terms for group in _structured_search_token_groups(term)]
    )
    phrases = [phrase.casefold() for phrase in query.phrases]
    positive_parts: list[str] = []
    if any_terms:
        positive_parts.append(
            "(" + " OR ".join(_sqlite_fts_quote(term) for term in any_terms) + ")"
        )
    positive_parts.extend(
        "(" + " OR ".join(_sqlite_fts_quote(term) for term in group) + ")" for group in all_groups
    )
    if phrases:
        positive_parts.append(
            "(" + " OR ".join(_sqlite_fts_quote(phrase) for phrase in phrases) + ")"
        )
    if not positive_parts:
        raise ValueError("Knowledge query requires positive search terms.")
    fts_query = " AND ".join(positive_parts)
    preview_terms = _dedupe_search_tokens(
        [
            *any_terms,
            *(term for group in all_groups for term in group),
            *_tokenize_search_text(" ".join(phrases)),
        ]
    )
    return fts_query, preview_terms


def _sqlite_knowledge_none_fts_query(query: KnowledgeQuery) -> str | None:
    none_terms = _dedupe_search_tokens(
        [
            token
            for term in query.none_terms
            for group in _structured_search_token_groups(term)
            for token in group
        ]
    )
    if not none_terms:
        return None
    return " OR ".join(_sqlite_fts_quote(term) for term in none_terms)


def _sqlite_knowledge_none_filter_sql(
    none_fts_query: str | None,
) -> tuple[str, list[object]]:
    if none_fts_query is None:
        return "", []
    return (
        """
        AND e.id NOT IN (
            SELECT DISTINCT cayu_knowledge_chunks_fts.entry_id
            FROM cayu_knowledge_chunks_fts
            JOIN cayu_knowledge_current_entries AS negative_entry
              ON negative_entry.id = cayu_knowledge_chunks_fts.entry_id
             AND negative_entry.revision = cayu_knowledge_chunks_fts.entry_revision
            WHERE cayu_knowledge_chunks_fts MATCH ?
        )
        """,
        [none_fts_query],
    )


def _sqlite_list_facet_sql(
    group_by: KnowledgeListGroup,
    where_sql: str,
    params: list[object],
    *,
    limit: int,
) -> tuple[str, list[object]]:
    limited_params = [*params, limit]
    if group_by is KnowledgeListGroup.KIND:
        return (
            f"""
            SELECT NULL AS key, e.kind AS value, COUNT(*) AS count
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            GROUP BY e.kind
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.NAMESPACE:
        return (
            f"""
            SELECT NULL AS key, e.namespace AS value, COUNT(*) AS count
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            GROUP BY e.namespace
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.LABEL:
        return (
            f"""
            SELECT label.key AS key, label.value AS value, COUNT(DISTINCT e.id) AS count
            FROM cayu_knowledge_current_entries AS e
            JOIN cayu_knowledge_labels AS label
              ON label.entry_id = e.id AND label.entry_revision = e.revision
            WHERE 1 = 1
            {where_sql}
            GROUP BY label.key, label.value
            ORDER BY count DESC, key ASC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.ASPECT:
        return (
            f"""
            SELECT NULL AS key, aspect.aspect AS value, COUNT(DISTINCT e.id) AS count
            FROM cayu_knowledge_current_entries AS e
            JOIN cayu_knowledge_aspects AS aspect
              ON aspect.entry_id = e.id AND aspect.entry_revision = e.revision
            WHERE 1 = 1
            {where_sql}
            GROUP BY aspect.aspect
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.IMPACT_TARGET:
        return (
            f"""
            SELECT NULL AS key, target.impact_target AS value, COUNT(DISTINCT e.id) AS count
            FROM cayu_knowledge_current_entries AS e
            JOIN cayu_knowledge_impact_targets AS target
              ON target.entry_id = e.id AND target.entry_revision = e.revision
            WHERE 1 = 1
            {where_sql}
            GROUP BY target.impact_target
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    if group_by is KnowledgeListGroup.VISIBILITY:
        return (
            f"""
            SELECT NULL AS key, e.visibility AS value, COUNT(*) AS count
            FROM cayu_knowledge_current_entries AS e
            WHERE 1 = 1
            {where_sql}
            GROUP BY e.visibility
            ORDER BY count DESC, value ASC
            LIMIT ?
            """,
            limited_params,
        )
    return (
        f"""
        SELECT NULL AS key, e.source_type AS value, COUNT(*) AS count
        FROM cayu_knowledge_current_entries AS e
        WHERE e.source_type IS NOT NULL
        {where_sql}
        GROUP BY e.source_type
        ORDER BY count DESC, value ASC
        LIMIT ?
        """,
        limited_params,
    )


def _structured_search_token_groups(value: str) -> list[list[str]]:
    tokens = _tokenize_search_text(value)
    if not tokens:
        raise ValueError("Structured knowledge search terms must contain at least one token.")
    return [_search_token_variants(token) for token in tokens]


def _sqlite_fts_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _dedupe_search_tokens(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_search_token_groups(groups: list[list[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            result.append(group)
            seen.add(key)
    return result


def _entry_row_values(entry: KnowledgeEntry) -> tuple[object, ...]:
    return (
        entry.id,
        entry.revision,
        entry.text,
        entry.kind,
        str(entry.visibility),
        str(entry.status),
        str(entry.created_by_type),
        entry.created_by,
        sqlite_support.format_datetime(entry.created_at),
        sqlite_support.format_datetime(entry.updated_at),
        entry.source_type,
        entry.source_uri,
        entry.source_id,
        entry.source_hash,
        entry.importance,
        entry.importance_source,
        entry.confidence,
        sqlite_support.format_optional_datetime(entry.last_used_at),
        sqlite_support.format_optional_datetime(entry.expires_at),
        entry.title,
        sqlite_support.json_dumps(entry.metadata),
    )


def _entry_from_row(
    row: sqlite3.Row,
    *,
    labels: dict[str, str],
    aspects: list[str],
    impact_targets: list[str],
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=row["id"],
        revision=row["revision"],
        text=row["text"],
        namespace=row["namespace"],
        labels=labels,
        kind=row["kind"],
        visibility=KnowledgeVisibility(row["visibility"]),
        status=KnowledgeStatus(row["status"]),
        created_by_type=KnowledgeActorType(row["created_by_type"]),
        created_by=row["created_by"],
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        source_type=row["source_type"],
        source_uri=row["source_uri"],
        source_id=row["source_id"],
        source_hash=row["source_hash"],
        importance=row["importance"],
        importance_source=row["importance_source"],
        confidence=row["confidence"],
        last_used_at=sqlite_support.parse_optional_datetime(row["last_used_at"]),
        expires_at=sqlite_support.parse_optional_datetime(row["expires_at"]),
        title=row["title"],
        aspects=aspects,
        impact_targets=impact_targets,
        metadata=json.loads(row["metadata_json"]),
    )


def _chunk_row_values(chunk: KnowledgeChunk) -> tuple[object, ...]:
    return (
        chunk.id,
        chunk.entry_id,
        chunk.entry_revision,
        chunk.chunk_index,
        chunk.text,
        chunk.content_hash,
        chunk.source_uri,
        sqlite_support.json_dumps(chunk.metadata),
    )


def _chunk_from_row(row: sqlite3.Row) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=row["id"],
        entry_id=row["entry_id"],
        entry_revision=row["entry_revision"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        content_hash=row["content_hash"],
        source_uri=row["source_uri"],
        metadata=json.loads(row["metadata_json"]),
    )


def _copy_entry_chunks(
    entry_id: str,
    entry_revision: int,
    chunks: list[KnowledgeChunk],
) -> list[KnowledgeChunk]:
    if type(chunks) is not list:
        raise ValueError("`chunks` must be a list.")
    if not chunks:
        raise ValueError("`chunks` cannot be empty.")
    copied_chunks = [copy_knowledge_chunk(chunk) for chunk in chunks]
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for chunk in copied_chunks:
        if chunk.entry_id != entry_id:
            raise ValueError("Knowledge chunks must belong to the entry.")
        if chunk.entry_revision != entry_revision:
            raise ValueError("Knowledge chunks must belong to the exact entry revision.")
        if chunk.id in seen_ids:
            raise ValueError("Knowledge chunk ids must be unique within an entry.")
        if chunk.chunk_index in seen_indexes:
            raise ValueError("Knowledge chunk indexes must be unique within an entry.")
        seen_ids.add(chunk.id)
        seen_indexes.add(chunk.chunk_index)
    return sorted(copied_chunks, key=lambda chunk: chunk.chunk_index)


def _center_chunk_window(
    chunks: list[KnowledgeChunk],
    *,
    chunk_index: int,
    max_chunks: int,
) -> list[KnowledgeChunk]:
    if len(chunks) <= max_chunks:
        return chunks
    closest = sorted(
        chunks, key=lambda chunk: (abs(chunk.chunk_index - chunk_index), chunk.chunk_index)
    )
    return sorted(closest[:max_chunks], key=lambda chunk: chunk.chunk_index)


def _bounded_chunks(
    chunks: list[KnowledgeChunk],
    *,
    start_index: int,
    end_index: int | None,
    max_chunks: int,
    max_bytes: int,
) -> list[KnowledgeChunk]:
    selected: list[KnowledgeChunk] = []
    remaining = max_bytes
    for chunk in chunks:
        if chunk.chunk_index < start_index:
            continue
        if end_index is not None and chunk.chunk_index > end_index:
            continue
        if len(selected) >= max_chunks or remaining <= 0:
            break
        copied = copy_knowledge_chunk(chunk)
        chunk_bytes = len(copied.text.encode("utf-8"))
        if chunk_bytes > remaining:
            truncated_text = _truncate_text_to_bytes(copied.text, remaining)
            if not truncated_text:
                break
            selected.append(
                KnowledgeChunk(
                    id=copied.id,
                    entry_id=copied.entry_id,
                    entry_revision=copied.entry_revision,
                    text=truncated_text,
                    chunk_index=copied.chunk_index,
                    content_hash=None,
                    source_uri=copied.source_uri,
                    metadata=copied.metadata,
                )
            )
            break
        selected.append(copied)
        remaining -= chunk_bytes
    return selected


def _preview_for_match(
    entry: KnowledgeEntry,
    chunk: KnowledgeChunk,
    terms: list[str],
) -> tuple[str, str]:
    if entry.title is not None:
        title_terms = set(_tokenize_search_text(entry.title))
        if any(term in title_terms for term in terms):
            return "title match", entry.title
    entry_terms = set(_tokenize_search_text(entry.text))
    if any(term in entry_terms for term in terms):
        return "entry text match", entry.text
    return "chunk text match", chunk.text


def _fts_text_for_entry_chunk(entry: KnowledgeEntry, chunk: KnowledgeChunk) -> str:
    if chunk.text == entry.text:
        return chunk.text
    return f"{entry.text}\n{chunk.text}"


def _default_chunk_for_entry(entry: KnowledgeEntry) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=f"{entry.id}:r{entry.revision}:0",
        entry_id=entry.id,
        entry_revision=entry.revision,
        text=entry.text,
        chunk_index=0,
        content_hash=sha256(entry.text.encode("utf-8")).hexdigest(),
        source_uri=entry.source_uri,
    )


def _has_only_default_chunk(entry: KnowledgeEntry, chunks: list[KnowledgeChunk]) -> bool:
    if len(chunks) != 1:
        return False
    default_chunk = _default_chunk_for_entry(entry)
    chunk = chunks[0]
    return (
        chunk.id == default_chunk.id
        and chunk.entry_id == default_chunk.entry_id
        and chunk.entry_revision == default_chunk.entry_revision
        and chunk.text == default_chunk.text
        and chunk.chunk_index == default_chunk.chunk_index
        and chunk.content_hash == default_chunk.content_hash
        and chunk.source_uri == default_chunk.source_uri
        and chunk.metadata == default_chunk.metadata
    )


def _tokenize_search_text(text: str) -> list[str]:
    return _SEARCH_TOKEN_RE.findall(text.casefold())


def _expand_search_tokens(tokens: list[str]) -> list[str]:
    return [variant for token in tokens for variant in _search_token_variants(token)]


def _search_token_variants(token: str) -> list[str]:
    variants = [token]
    if len(token) < 3 or not token.isalpha():
        return variants
    if token.endswith("ies") and len(token) > 4:
        variants.append(token[:-3] + "y")
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        variants.append(token[:-1])
    else:
        variants.append(_plural_search_token(token))
    return _dedupe_search_tokens(variants)


def _plural_search_token(token: str) -> str:
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    return token + "s"


def _truncate_text_to_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{field_name}` must be greater than 0.")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{field_name}` must be greater than or equal to 0.")
