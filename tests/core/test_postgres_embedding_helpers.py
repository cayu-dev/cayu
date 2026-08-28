"""DB-free unit tests for the pgvector embedding-store module helpers (MEM-08)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cayu.storage import KnowledgeAccessScope, KnowledgeQuery, KnowledgeSearchMode
from cayu.storage.postgres import (
    _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS,
    PostgresEmbeddingKnowledgeStore,
    _warn_if_embedding_dims_exceed_hnsw,
)

if TYPE_CHECKING:
    import pytest


def test_warn_if_embedding_dims_exceed_hnsw_warns_above_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # MEM-08: >2000 dims cannot get an HNSW index; the store warns rather than failing silently.
    with caplog.at_level(logging.WARNING, logger="cayu.storage.postgres"):
        _warn_if_embedding_dims_exceed_hnsw(_PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS + 1)

    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "brute-force" in warnings[0].getMessage()


def test_warn_if_embedding_dims_exceed_hnsw_silent_within_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="cayu.storage.postgres"):
        _warn_if_embedding_dims_exceed_hnsw(_PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS)

    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_frontier_semantic_search_keeps_the_hnsw_fast_path() -> None:
    class RecordingCursor:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def execute(self, statement, params=None) -> None:
            self.statements.append(str(statement))

        async def fetchall(self) -> list[tuple[object, ...]]:
            return []

    async def run() -> tuple[tuple[list[tuple[str, str, float]], bool, int], list[str]]:
        store = object.__new__(PostgresEmbeddingKnowledgeStore)
        store.embedding_model = "frontier-fast-path-test"
        store.embedding_dimensions = 3
        store.semantic_min_score = 0.0
        cursor = RecordingCursor()
        result = await store._semantic_search_rows_in_snapshot(
            cursor,
            KnowledgeQuery(
                text="frontier indexed semantic recall",
                namespace="frontier-fast-path",
                mode=KnowledgeSearchMode.SEMANTIC,
                limit=5,
            ),
            [1.0, 0.0, 0.0],
            access_scope=KnowledgeAccessScope.for_namespace("frontier-fast-path"),
            ready_records=0,
            through_change_sequence=17,
            through_index_readiness_sequence=23,
        )
        return result, cursor.statements

    result, statements = asyncio.run(run())

    assert result == ([], False, 0)
    assert "SET LOCAL plan_cache_mode = force_custom_plan" in statements
    assert "SET LOCAL enable_indexscan = off" not in statements
    assert "SET LOCAL enable_seqscan = on" not in statements
