"""DB-free unit tests for the pgvector embedding-store module helpers (MEM-08)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cayu.storage.postgres import (
    _PGVECTOR_HNSW_VECTOR_MAX_DIMENSIONS,
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
