"""Shared embedding-backfill query identity, cursor validation and page ordering."""

from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import UTC, datetime
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import canonical_durable_json_bytes
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.indexing import (
    KNOWLEDGE_CHUNK_TEXT_GENERATOR,
    KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
    KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
    KNOWLEDGE_CHUNK_TEXT_PROJECTION,
    KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
    _bounded_knowledge_embedding_backfill_cursor,
)
from cayu.knowledge.records import (
    MAX_KNOWLEDGE_CHUNK_INDEX,
    KnowledgeChunk,
    _knowledge_chunk_id,
    _knowledge_entry_id,
    _validate_nonnegative_int,
)
from cayu.knowledge.scopes import (
    KnowledgeAccessScope,
    _knowledge_access_scope_sha256,
    copy_knowledge_access_scope,
)
from cayu.knowledge.search import (
    KnowledgeListQuery,
    _validate_unit_float,
    copy_knowledge_list_query,
)

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")

_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_VERSION = 1


class _KnowledgeEmbeddingBackfillCursor(BaseModel):
    """Validated keyset state carried inside an opaque backfill cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    version: int
    fingerprint: str
    importance: float
    updated_at: datetime
    entry_id: str
    chunk_index: int
    chunk_id: str

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if type(value) is not int or value != _KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_VERSION:
            raise ValueError("Unsupported knowledge embedding backfill cursor version.")
        return value

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        value = require_clean_nonblank(value, "fingerprint")
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("Backfill cursor fingerprint must be a SHA-256 digest.")
        return value

    @field_validator("importance", mode="before")
    @classmethod
    def validate_importance(cls, value) -> float:
        return _validate_unit_float(value, "importance")

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`updated_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value, "entry_id")

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(cls, value: int) -> int:
        _validate_nonnegative_int(value, "chunk_index")
        if value > MAX_KNOWLEDGE_CHUNK_INDEX:
            raise ValueError(
                f"`chunk_index` must be less than or equal to {MAX_KNOWLEDGE_CHUNK_INDEX}."
            )
        return value

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        return _knowledge_chunk_id(value, "chunk_id")


def _knowledge_embedding_backfill_fingerprint(
    query: KnowledgeListQuery,
    access_scope: KnowledgeAccessScope,
    *,
    refresh_existing: bool,
    embedding_model: str,
    embedding_dimensions: int,
) -> str:
    query = copy_knowledge_list_query(query)
    access_scope = copy_knowledge_access_scope(access_scope)
    material = {
        "query": query.model_dump(mode="json"),
        "access_scope_sha256": _knowledge_access_scope_sha256(access_scope),
        "refresh_existing": refresh_existing,
        "projection_type": KNOWLEDGE_CHUNK_TEXT_PROJECTION,
        "embedding_model": require_clean_nonblank(embedding_model, "embedding_model"),
        "dimensions": embedding_dimensions,
        "preprocessing_version": KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
        "generator": KNOWLEDGE_CHUNK_TEXT_GENERATOR,
        "generator_version": KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
        "index_representation_version": KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
    }
    return sha256(
        canonical_durable_json_bytes(material, "knowledge embedding backfill query")
    ).hexdigest()


def _encode_knowledge_embedding_backfill_cursor(
    *,
    fingerprint: str,
    importance: float,
    updated_at: datetime,
    chunk: KnowledgeChunk,
) -> str:
    cursor = _KnowledgeEmbeddingBackfillCursor(
        version=_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_VERSION,
        fingerprint=fingerprint,
        importance=importance,
        updated_at=updated_at,
        entry_id=chunk.entry_id,
        chunk_index=chunk.chunk_index,
        chunk_id=chunk.id,
    )
    raw = canonical_durable_json_bytes(
        cursor.model_dump(mode="json"),
        "knowledge embedding backfill cursor",
    )
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _bounded_knowledge_embedding_backfill_cursor(encoded, "next_cursor")


def _decode_knowledge_embedding_backfill_cursor(
    cursor: str | None,
    *,
    fingerprint: str,
) -> _KnowledgeEmbeddingBackfillCursor | None:
    if cursor is None:
        return None
    cursor = _bounded_knowledge_embedding_backfill_cursor(cursor, "cursor")
    try:
        encoded = cursor.encode("ascii")
        padding = b"=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).rstrip(b"=") != encoded:
            raise ValueError("Non-canonical backfill cursor encoding.")
        decoded = json.loads(raw.decode("utf-8"))
        parsed = _KnowledgeEmbeddingBackfillCursor.model_validate(decoded)
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise ValueError("Invalid knowledge embedding backfill cursor.") from exc
    if parsed.fingerprint != fingerprint:
        raise ValueError(
            "Knowledge embedding backfill cursor does not match this query, scope, "
            "projection configuration, and refresh mode."
        )
    return parsed


def _knowledge_embedding_backfill_sort_key(
    *,
    importance: float,
    updated_at: datetime,
    entry_id: str,
    chunk_index: int,
    chunk_id: str,
) -> tuple[float, int, str, int, str]:
    updated_at = updated_at.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = updated_at - epoch
    updated_at_microseconds = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return (
        -importance,
        -updated_at_microseconds,
        entry_id,
        chunk_index,
        chunk_id,
    )
