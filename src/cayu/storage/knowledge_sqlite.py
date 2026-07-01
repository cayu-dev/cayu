from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    copy_knowledge_chunk,
    copy_knowledge_entry,
    copy_knowledge_list_query,
    copy_knowledge_query,
)

_SEARCH_TOKEN_RE = re.compile(r"\w+")
_SEARCH_PAGE_SIZE = 500


class SQLiteKnowledgeStore(KnowledgeStore):
    """SQLite-backed durable knowledge store with FTS5 keyword search."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
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
        self._schema_mode = schema_mode
        self._lock = asyncio.Lock()
        self._connection = sqlite_support.connect(db_path)
        sqlite_support.reconcile_schema(self._connection, schema_mode)

    async def put_entry(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        entry = copy_knowledge_entry(entry)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing_entry = self._load_entry_unlocked(entry.id)
                existing_chunks = self._load_chunks_unlocked(entry.id)
                self._upsert_entry_unlocked(entry)
                if (
                    existing_entry is None
                    or not existing_chunks
                    or _has_only_default_chunk(existing_entry, existing_chunks)
                ):
                    self._replace_chunks_unlocked(entry.id, [_default_chunk_for_entry(entry)])
                self._connection.commit()
                return copy_knowledge_entry(entry)
            except Exception:
                self._connection.rollback()
                raise

    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        async with self._lock:
            entry = self._load_entry_unlocked(clean_id)
            return None if entry is None else copy_knowledge_entry(entry)

    async def update_entry_status(
        self,
        entry_id: str,
        status: KnowledgeStatus,
    ) -> KnowledgeEntry:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        if not isinstance(status, KnowledgeStatus):
            raise ValueError("status must be a KnowledgeStatus.")
        async with self._lock:
            entry = self._load_entry_unlocked(clean_id)
            if entry is None:
                raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
            updated_at = max(datetime.now(UTC), entry.created_at, entry.updated_at)
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE cayu_knowledge_entries
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        str(status),
                        sqlite_support.format_datetime(updated_at),
                        clean_id,
                    ),
                )
            loaded = self._load_entry_unlocked(clean_id)
            if loaded is None:
                raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
            return loaded

    async def delete_entry(
        self,
        entry_id: str,
        *,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        async with self._lock:
            entry = self._load_entry_unlocked(clean_id)
            if entry is None:
                return None
            if hard:
                with self._connection:
                    self._delete_chunks_unlocked(clean_id)
                    self._connection.execute(
                        "DELETE FROM cayu_knowledge_entries WHERE id = ?",
                        (clean_id,),
                    )
                return copy_knowledge_entry(entry)
        return await self.update_entry_status(clean_id, KnowledgeStatus.DELETED)

    async def replace_chunks(
        self,
        entry_id: str,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        copied_chunks = _copy_entry_chunks(clean_id, chunks)
        async with self._lock:
            if self._load_entry_unlocked(clean_id) is None:
                raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
            with self._connection:
                self._replace_chunks_unlocked(clean_id, copied_chunks)
            return [copy_knowledge_chunk(chunk) for chunk in copied_chunks]

    async def put_entry_with_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> KnowledgeEntry:
        copied_entry = copy_knowledge_entry(entry)
        copied_chunks = _copy_entry_chunks(copied_entry.id, chunks)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._upsert_entry_unlocked(copied_entry)
                self._replace_chunks_unlocked(copied_entry.id, copied_chunks)
                self._connection.commit()
                return copy_knowledge_entry(copied_entry)
            except Exception:
                self._connection.rollback()
                raise

    async def read_chunks(
        self,
        entry_id: str,
        *,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        if chunk_index is not None:
            _validate_nonnegative_int(chunk_index, "chunk_index")
        _validate_nonnegative_int(around, "around")
        if chunk_index is None and around != 0:
            raise ValueError("`around` requires `chunk_index`.")
        _validate_positive_int(max_chunks, "max_chunks")
        _validate_positive_int(max_bytes, "max_bytes")
        async with self._lock:
            if self._load_entry_unlocked(clean_id) is None:
                return []
            chunks = self._load_chunks_unlocked(clean_id)
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

    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        knowledge_query = copy_knowledge_query(query)
        if knowledge_query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("SQLiteKnowledgeStore supports only auto and keyword search modes.")
        fts_query, preview_terms = _sqlite_knowledge_fts_query(knowledge_query)
        where_sql, params = _knowledge_filter_sql(knowledge_query)
        async with self._lock:
            total_hits_known = self._count_search_hits_unlocked(fts_query, where_sql, params)
            unique_rows = self._search_unique_rows_unlocked(
                fts_query=fts_query,
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

    async def list_entries(self, query: KnowledgeListQuery) -> KnowledgeListResult:
        knowledge_query = copy_knowledge_list_query(query)
        where_sql, params = _knowledge_list_filter_sql(knowledge_query)
        async with self._lock:
            total_entries_known = self._count_list_entries_unlocked(where_sql, params)
            rows = self._connection.execute(
                f"""
                SELECT e.id
                FROM cayu_knowledge_entries AS e
                WHERE 1 = 1
                {where_sql}
                ORDER BY COALESCE(e.importance, 0.0) DESC,
                         e.updated_at DESC,
                         e.id ASC
                LIMIT ?
                """,
                [*params, knowledge_query.limit],
            ).fetchall()
            entries = [
                entry
                for row in rows
                if (entry := self._load_entry_unlocked(str(row["id"]))) is not None
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
        where_sql: str,
        params: list[object],
    ) -> int:
        row = self._connection.execute(
            f"""
            SELECT COUNT(DISTINCT e.id)
            FROM cayu_knowledge_chunks_fts
            JOIN cayu_knowledge_chunks AS c
                ON c.id = cayu_knowledge_chunks_fts.chunk_id
            JOIN cayu_knowledge_entries AS e
                ON e.id = c.entry_id
            WHERE cayu_knowledge_chunks_fts MATCH ?
            {where_sql}
            """,
            [fts_query, *params],
        ).fetchone()
        return 0 if row is None else int(row[0])

    def _search_unique_rows_unlocked(
        self,
        *,
        fts_query: str,
        where_sql: str,
        params: list[object],
        limit: int,
    ) -> list[sqlite3.Row]:
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
                    ON c.id = cayu_knowledge_chunks_fts.chunk_id
                JOIN cayu_knowledge_entries AS e
                    ON e.id = c.entry_id
                WHERE cayu_knowledge_chunks_fts MATCH ?
                {where_sql}
                ORDER BY fts_score ASC,
                         COALESCE(e.importance, 0.0) DESC,
                         e.updated_at DESC,
                         e.id ASC,
                         c.chunk_index ASC
                LIMIT ? OFFSET ?
                """,
                [fts_query, *params, _SEARCH_PAGE_SIZE, offset],
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
        hits: list[KnowledgeHit] = []
        remaining = query.max_bytes
        truncated = False
        for row in rows:
            if remaining <= 0:
                truncated = True
                break
            entry = self._load_entry_unlocked(row["entry_id"])
            chunk = self._load_chunk_unlocked(row["chunk_id"])
            if entry is None or chunk is None:
                continue
            reason, preview_text = _preview_for_match(entry, chunk, terms)
            preview_bytes = len(preview_text.encode("utf-8"))
            preview = _truncate_text_to_bytes(preview_text, remaining)
            if not preview:
                truncated = True
                break
            returned_bytes = len(preview.encode("utf-8"))
            if returned_bytes < preview_bytes:
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
                )
            )
        return hits, truncated

    def _count_list_entries_unlocked(self, where_sql: str, params: list[object]) -> int:
        row = self._connection.execute(
            f"""
            SELECT COUNT(*)
            FROM cayu_knowledge_entries AS e
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
            if returned_bytes < preview_bytes:
                truncated = True
            remaining -= returned_bytes
            items.append(
                KnowledgeListItem(
                    entry=entry,
                    chunk_count=len(self._load_chunks_unlocked(entry.id)),
                    text_preview=preview,
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

    def _upsert_entry_unlocked(self, entry: KnowledgeEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO cayu_knowledge_entries (
                id,
                namespace,
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
            ON CONFLICT(id) DO UPDATE SET
                namespace = excluded.namespace,
                text = excluded.text,
                kind = excluded.kind,
                visibility = excluded.visibility,
                status = excluded.status,
                created_by_type = excluded.created_by_type,
                created_by = excluded.created_by,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                source_type = excluded.source_type,
                source_uri = excluded.source_uri,
                source_id = excluded.source_id,
                source_hash = excluded.source_hash,
                importance = excluded.importance,
                importance_source = excluded.importance_source,
                confidence = excluded.confidence,
                last_used_at = excluded.last_used_at,
                expires_at = excluded.expires_at,
                title = excluded.title,
                metadata_json = excluded.metadata_json
            """,
            _entry_row_values(entry),
        )
        self._replace_entry_lists_unlocked(entry)
        self._refresh_entry_fts_unlocked(entry.id)

    def _replace_entry_lists_unlocked(self, entry: KnowledgeEntry) -> None:
        for table in (
            "cayu_knowledge_labels",
            "cayu_knowledge_aspects",
            "cayu_knowledge_impact_targets",
        ):
            self._connection.execute(f"DELETE FROM {table} WHERE entry_id = ?", (entry.id,))
        if entry.labels:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_labels (entry_id, key, value)
                VALUES (?, ?, ?)
                """,
                [(entry.id, key, value) for key, value in sorted(entry.labels.items())],
            )
        if entry.aspects:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_aspects (entry_id, aspect)
                VALUES (?, ?)
                """,
                [(entry.id, aspect) for aspect in entry.aspects],
            )
        if entry.impact_targets:
            self._connection.executemany(
                """
                INSERT INTO cayu_knowledge_impact_targets (entry_id, impact_target)
                VALUES (?, ?)
                """,
                [(entry.id, target) for target in entry.impact_targets],
            )

    def _replace_chunks_unlocked(self, entry_id: str, chunks: list[KnowledgeChunk]) -> None:
        self._delete_chunks_unlocked(entry_id)
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_chunks (
                id,
                entry_id,
                chunk_index,
                text,
                content_hash,
                source_uri,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [_chunk_row_values(chunk) for chunk in chunks],
        )
        self._refresh_entry_fts_unlocked(entry_id)

    def _delete_chunks_unlocked(self, entry_id: str) -> None:
        self._connection.execute(
            "DELETE FROM cayu_knowledge_chunks_fts WHERE entry_id = ?",
            (entry_id,),
        )
        self._connection.execute(
            "DELETE FROM cayu_knowledge_chunks WHERE entry_id = ?",
            (entry_id,),
        )

    def _refresh_entry_fts_unlocked(self, entry_id: str) -> None:
        entry = self._load_entry_unlocked(entry_id)
        if entry is None:
            return
        self._connection.execute(
            "DELETE FROM cayu_knowledge_chunks_fts WHERE entry_id = ?",
            (entry_id,),
        )
        chunks = self._load_chunks_unlocked(entry_id)
        if not chunks:
            return
        self._connection.executemany(
            """
            INSERT INTO cayu_knowledge_chunks_fts (entry_id, chunk_id, title, text)
            VALUES (?, ?, ?, ?)
            """,
            [
                (entry.id, chunk.id, entry.title or "", _fts_text_for_entry_chunk(entry, chunk))
                for chunk in chunks
            ],
        )

    def _load_entry_unlocked(self, entry_id: str) -> KnowledgeEntry | None:
        row = self._connection.execute(
            "SELECT * FROM cayu_knowledge_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            return None
        return _entry_from_row(
            row,
            labels=self._load_labels_unlocked(entry_id),
            aspects=self._load_aspects_unlocked(entry_id),
            impact_targets=self._load_impact_targets_unlocked(entry_id),
        )

    def _load_chunk_unlocked(self, chunk_id: str) -> KnowledgeChunk | None:
        row = self._connection.execute(
            "SELECT * FROM cayu_knowledge_chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return _chunk_from_row(row)

    def _load_chunks_unlocked(self, entry_id: str) -> list[KnowledgeChunk]:
        rows = self._connection.execute(
            """
            SELECT *
            FROM cayu_knowledge_chunks
            WHERE entry_id = ?
            ORDER BY chunk_index ASC
            """,
            (entry_id,),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def _load_labels_unlocked(self, entry_id: str) -> dict[str, str]:
        rows = self._connection.execute(
            """
            SELECT key, value
            FROM cayu_knowledge_labels
            WHERE entry_id = ?
            ORDER BY key ASC
            """,
            (entry_id,),
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _load_aspects_unlocked(self, entry_id: str) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT aspect
            FROM cayu_knowledge_aspects
            WHERE entry_id = ?
            ORDER BY aspect ASC
            """,
            (entry_id,),
        ).fetchall()
        return [row["aspect"] for row in rows]

    def _load_impact_targets_unlocked(self, entry_id: str) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT impact_target
            FROM cayu_knowledge_impact_targets
            WHERE entry_id = ?
            ORDER BY impact_target ASC
            """,
            (entry_id,),
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
    none_terms = _dedupe_search_tokens(
        [
            token
            for term in query.none_terms
            for group in _structured_search_token_groups(term)
            for token in group
        ]
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
    for term in none_terms:
        fts_query += f" NOT {_sqlite_fts_quote(term)}"
    preview_terms = _dedupe_search_tokens(
        [
            *any_terms,
            *(term for group in all_groups for term in group),
            *_tokenize_search_text(" ".join(phrases)),
        ]
    )
    return fts_query, preview_terms


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
            FROM cayu_knowledge_entries AS e
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
            FROM cayu_knowledge_entries AS e
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
            FROM cayu_knowledge_entries AS e
            JOIN cayu_knowledge_labels AS label ON label.entry_id = e.id
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
            FROM cayu_knowledge_entries AS e
            JOIN cayu_knowledge_aspects AS aspect ON aspect.entry_id = e.id
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
            FROM cayu_knowledge_entries AS e
            JOIN cayu_knowledge_impact_targets AS target ON target.entry_id = e.id
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
            FROM cayu_knowledge_entries AS e
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
        FROM cayu_knowledge_entries AS e
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
        entry.namespace,
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
        chunk_index=row["chunk_index"],
        text=row["text"],
        content_hash=row["content_hash"],
        source_uri=row["source_uri"],
        metadata=json.loads(row["metadata_json"]),
    )


def _copy_entry_chunks(entry_id: str, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
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
        id=f"{entry.id}:0",
        entry_id=entry.id,
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
