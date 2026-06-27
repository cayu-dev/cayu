from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_finite,
    require_nonblank,
)
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_KIND,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    copy_knowledge_chunk,
    copy_knowledge_entry,
)

DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES = 4_000
DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES = 400
DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS = 1_000
MIN_KNOWLEDGE_TEXT_BYTES = 4
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class KnowledgeIndexRequest(BaseModel):
    """Request to deterministically index text into an entry and chunks."""

    model_config = ConfigDict(extra="forbid")

    text: str
    entry_id: str | None = None
    namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE
    labels: dict[str, str] = Field(default_factory=dict)
    kind: str = DEFAULT_KNOWLEDGE_KIND
    visibility: KnowledgeVisibility = KnowledgeVisibility.GLOBAL
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    created_by_type: KnowledgeActorType = KnowledgeActorType.APP
    created_by: str = "app"
    source_type: str | None = None
    source_uri: str | None = None
    source_id: str | None = None
    aspects: list[str] = Field(default_factory=list)
    impact_targets: list[str] = Field(default_factory=list)
    importance: float | None = None
    importance_source: str | None = None
    confidence: float | None = None
    expires_at: datetime | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_metadata: dict[str, Any] = Field(default_factory=dict)
    entry_text_max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES
    chunk_target_bytes: int = DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES
    chunk_overlap_bytes: int = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES
    max_chunks: int = DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS
    skip_unchanged: bool = True

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("metadata", "chunk_metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_json_value(value, info.field_name)

    @field_validator("namespace", "kind", "created_by")
    @classmethod
    def validate_clean_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "entry_id",
        "source_type",
        "source_uri",
        "source_id",
        "importance_source",
        "title",
    )
    @classmethod
    def validate_optional_clean_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("aspects", "impact_targets", mode="before")
    @classmethod
    def copy_string_list(cls, value, info) -> list[str]:
        if value is None:
            return []
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"`{info.field_name}` must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`{info.field_name}[{index}]` must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return list(dict.fromkeys(result))

    @field_validator("importance", "confidence", mode="before")
    @classmethod
    def validate_optional_unit_interval(cls, value, info) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"`{info.field_name}` must be a number.")
        value = require_finite(float(value), info.field_name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"`{info.field_name}` must be between 0.0 and 1.0.")
        return value

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator(
        "entry_text_max_bytes",
        "chunk_target_bytes",
        "chunk_overlap_bytes",
        "max_chunks",
    )
    @classmethod
    def validate_nonnegative_ints(cls, value: int, info) -> int:
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if info.field_name == "chunk_overlap_bytes":
            if value < 0:
                raise ValueError(f"`{info.field_name}` must be greater than or equal to 0.")
            return value
        if info.field_name in {"entry_text_max_bytes", "chunk_target_bytes"}:
            if value < MIN_KNOWLEDGE_TEXT_BYTES:
                raise ValueError(
                    f"`{info.field_name}` must be at least {MIN_KNOWLEDGE_TEXT_BYTES}."
                )
            return value
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> KnowledgeIndexRequest:
        if self.chunk_overlap_bytes >= self.chunk_target_bytes:
            raise ValueError("`chunk_overlap_bytes` must be less than `chunk_target_bytes`.")
        if self.chunk_overlap_bytes > self.chunk_target_bytes // 2:
            raise ValueError("`chunk_overlap_bytes` must be at most half of `chunk_target_bytes`.")
        return self


class KnowledgeIndexResult(BaseModel):
    """Deterministic output from a knowledge indexing run."""

    model_config = ConfigDict(extra="forbid")

    entry: KnowledgeEntry
    chunks: list[KnowledgeChunk]
    source_hash: str
    text_bytes: int
    chunk_count: int
    truncated: bool = False
    written: bool = False
    unchanged: bool = False

    @field_validator("entry")
    @classmethod
    def copy_entry(cls, value):
        return copy_knowledge_entry(value)

    @field_validator("chunks")
    @classmethod
    def copy_chunks(cls, value):
        chunks = [copy_knowledge_chunk(chunk) for chunk in value]
        if not chunks:
            raise ValueError("`chunks` cannot be empty.")
        return chunks

    @field_validator("source_hash")
    @classmethod
    def validate_source_hash(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("text_bytes", "chunk_count")
    @classmethod
    def validate_nonnegative_int(cls, value: int, info) -> int:
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be greater than or equal to 0.")
        return value

    @model_validator(mode="after")
    def validate_chunk_count(self) -> KnowledgeIndexResult:
        if self.chunk_count != len(self.chunks):
            raise ValueError("`chunk_count` must match `chunks` length.")
        if self.entry.source_hash != self.source_hash:
            raise ValueError("`source_hash` must match `entry.source_hash`.")
        if self.text_bytes < len(self.entry.text.encode("utf-8")):
            raise ValueError("`text_bytes` cannot be less than `entry.text` bytes.")
        for chunk in self.chunks:
            if chunk.entry_id != self.entry.id:
                raise ValueError("`chunks` must belong to `entry`.")
            if chunk.content_hash != _hash_text(chunk.text):
                raise ValueError("`chunks[].content_hash` must match chunk text.")
            if chunk.metadata.get("source_hash") != self.source_hash:
                raise ValueError("`chunks[].metadata.source_hash` must match `source_hash`.")
        if self.written and self.unchanged:
            raise ValueError("`written` and `unchanged` cannot both be true.")
        return self


class KnowledgeIndexer:
    """Deterministic text indexer for KnowledgeStore entries and chunks."""

    def __init__(self, store: KnowledgeStore | None = None) -> None:
        if store is not None and not isinstance(store, KnowledgeStore):
            raise TypeError("store must implement KnowledgeStore.")
        self.store = store

    def build(self, request: KnowledgeIndexRequest) -> KnowledgeIndexResult:
        request = copy_knowledge_index_request(request)
        source_hash = _hash_text(request.text)
        entry_id = request.entry_id or _generated_entry_id(request, source_hash)
        entry_text = _bounded_entry_text(request.text, request.entry_text_max_bytes)
        entry = KnowledgeEntry(
            id=entry_id,
            text=entry_text,
            namespace=request.namespace,
            labels=request.labels,
            kind=request.kind,
            visibility=request.visibility,
            status=request.status,
            created_by_type=request.created_by_type,
            created_by=request.created_by,
            source_type=request.source_type,
            source_uri=request.source_uri,
            source_id=request.source_id,
            source_hash=source_hash,
            aspects=request.aspects,
            impact_targets=request.impact_targets,
            importance=request.importance,
            importance_source=request.importance_source,
            confidence=request.confidence,
            expires_at=request.expires_at,
            title=request.title,
            metadata=request.metadata,
        )
        chunks, truncated = _build_chunks(entry_id, request, source_hash=source_hash)
        return KnowledgeIndexResult(
            entry=entry,
            chunks=chunks,
            source_hash=source_hash,
            text_bytes=len(request.text.encode("utf-8")),
            chunk_count=len(chunks),
            truncated=truncated,
        )

    async def index_text(self, request: KnowledgeIndexRequest) -> KnowledgeIndexResult:
        request = copy_knowledge_index_request(request)
        result = self.build(request)
        if self.store is None:
            return result
        existing = await self.store.get_entry(result.entry.id)
        if (
            request.skip_unchanged
            and existing is not None
            and existing.source_hash == result.source_hash
            and _same_indexed_entry(existing, result.entry)
            and _same_indexed_chunks(
                await self.store.read_chunks(
                    result.entry.id,
                    max_chunks=len(result.chunks) + 1,
                    max_bytes=_chunk_comparison_max_bytes(result.chunks),
                ),
                result.chunks,
            )
        ):
            return result.model_copy(update={"unchanged": True})
        await self.store.put_entry_with_chunks(result.entry, result.chunks)
        return result.model_copy(update={"written": True})


def copy_knowledge_index_request(request: KnowledgeIndexRequest) -> KnowledgeIndexRequest:
    if type(request) is not KnowledgeIndexRequest:
        raise TypeError("KnowledgeIndexRequest instances must not be subclasses.")
    return KnowledgeIndexRequest(
        text=request.text,
        entry_id=request.entry_id,
        namespace=request.namespace,
        labels=copy_label_map(request.labels, "labels"),
        kind=request.kind,
        visibility=request.visibility,
        status=request.status,
        created_by_type=request.created_by_type,
        created_by=request.created_by,
        source_type=request.source_type,
        source_uri=request.source_uri,
        source_id=request.source_id,
        aspects=list(request.aspects),
        impact_targets=list(request.impact_targets),
        importance=request.importance,
        importance_source=request.importance_source,
        confidence=request.confidence,
        expires_at=request.expires_at,
        title=request.title,
        metadata=copy_json_value(request.metadata, "metadata"),
        chunk_metadata=copy_json_value(request.chunk_metadata, "chunk_metadata"),
        entry_text_max_bytes=request.entry_text_max_bytes,
        chunk_target_bytes=request.chunk_target_bytes,
        chunk_overlap_bytes=request.chunk_overlap_bytes,
        max_chunks=request.max_chunks,
        skip_unchanged=request.skip_unchanged,
    )


def copy_knowledge_index_result(result: KnowledgeIndexResult) -> KnowledgeIndexResult:
    if type(result) is not KnowledgeIndexResult:
        raise TypeError("KnowledgeIndexResult instances must not be subclasses.")
    return KnowledgeIndexResult(
        entry=copy_knowledge_entry(result.entry),
        chunks=[copy_knowledge_chunk(chunk) for chunk in result.chunks],
        source_hash=result.source_hash,
        text_bytes=result.text_bytes,
        chunk_count=result.chunk_count,
        truncated=result.truncated,
        written=result.written,
        unchanged=result.unchanged,
    )


@dataclass(frozen=True)
class _TextBlock:
    text: str
    heading_path: tuple[str, ...]


def _build_chunks(
    entry_id: str,
    request: KnowledgeIndexRequest,
    *,
    source_hash: str,
) -> tuple[list[KnowledgeChunk], bool]:
    blocks = _split_markdownish_blocks(request.text)
    chunks: list[KnowledgeChunk] = []
    current_parts: list[str] = []
    current_heading_paths: list[tuple[str, ...]] = []
    current_overlap_bytes = 0
    overlap_text = ""
    truncated = False

    def flush_current() -> None:
        nonlocal current_parts, current_heading_paths, current_overlap_bytes, overlap_text
        text = "\n\n".join(part for part in current_parts if part.strip())
        if not text:
            current_parts = []
            current_heading_paths = []
            current_overlap_bytes = 0
            return
        chunk_index = len(chunks)
        chunks.append(
            KnowledgeChunk(
                id=f"{entry_id}:{chunk_index}",
                entry_id=entry_id,
                text=text,
                chunk_index=chunk_index,
                content_hash=_hash_text(text),
                source_uri=request.source_uri,
                metadata={
                    **request.chunk_metadata,
                    "source_hash": source_hash,
                    "heading_paths": _heading_paths_metadata(current_heading_paths),
                    "overlap_from_previous_bytes": current_overlap_bytes,
                },
            )
        )
        overlap_text = _suffix_by_bytes(text, request.chunk_overlap_bytes)
        current_parts = []
        current_heading_paths = []
        current_overlap_bytes = 0

    for block in blocks:
        for piece in _block_pieces(block, request.chunk_target_bytes, request.chunk_overlap_bytes):
            if len(chunks) >= request.max_chunks:
                truncated = True
                break
            candidate_parts = [*current_parts, piece.text]
            candidate_text = "\n\n".join(candidate_parts)
            if current_parts and len(candidate_text.encode("utf-8")) > request.chunk_target_bytes:
                flush_current()
                if len(chunks) >= request.max_chunks:
                    truncated = True
                    break
                current_parts, current_overlap_bytes = _parts_with_bounded_overlap(
                    overlap_text,
                    piece.text,
                    request.chunk_target_bytes,
                    heading_path=piece.heading_path,
                )
                current_heading_paths = [piece.heading_path]
            else:
                current_parts.append(piece.text)
                current_heading_paths.append(piece.heading_path)
        if truncated:
            break

    if not truncated and current_parts:
        if len(chunks) >= request.max_chunks:
            truncated = True
        else:
            flush_current()

    if not chunks:
        chunk_text = _bounded_entry_text(request.text, request.chunk_target_bytes)
        chunks.append(
            KnowledgeChunk(
                id=f"{entry_id}:0",
                entry_id=entry_id,
                text=chunk_text,
                chunk_index=0,
                content_hash=_hash_text(chunk_text),
                source_uri=request.source_uri,
                metadata={**request.chunk_metadata, "source_hash": source_hash},
            )
        )
    return chunks, truncated


def _split_markdownish_blocks(text: str) -> list[_TextBlock]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_stack: list[str] = []
    blocks: list[_TextBlock] = []
    paragraph_lines: list[str] = []
    paragraph_heading_path: tuple[str, ...] = ()

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_heading_path
        paragraph = "\n".join(paragraph_lines).strip()
        if paragraph:
            blocks.append(_TextBlock(text=paragraph, heading_path=paragraph_heading_path))
        paragraph_lines = []
        paragraph_heading_path = ()

    for line in lines:
        heading_match = _MARKDOWN_HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading = require_nonblank(heading_match.group(2), "heading").strip()
            heading_stack = [*heading_stack[: level - 1], heading]
            continue
        if not line.strip():
            flush_paragraph()
            continue
        if not paragraph_lines:
            paragraph_heading_path = tuple(heading_stack)
        paragraph_lines.append(line.rstrip())
    flush_paragraph()
    if blocks:
        return blocks
    return [_TextBlock(text=require_nonblank(text, "text").strip(), heading_path=())]


def _block_pieces(
    block: _TextBlock,
    chunk_target_bytes: int,
    chunk_overlap_bytes: int,
) -> list[_TextBlock]:
    text = _text_with_heading_context(block)
    if len(text.encode("utf-8")) <= chunk_target_bytes:
        return [_TextBlock(text=text, heading_path=block.heading_path)]
    if block.heading_path:
        heading = " > ".join(block.heading_path)
        prefix = f"{heading}\n\n"
        prefix_bytes = len(prefix.encode("utf-8"))
        body_target_bytes = chunk_target_bytes - prefix_bytes
        if body_target_bytes >= MIN_KNOWLEDGE_TEXT_BYTES:
            body_overlap_bytes = min(chunk_overlap_bytes, body_target_bytes // 2)
            return [
                _TextBlock(text=f"{prefix}{piece}", heading_path=block.heading_path)
                for piece in _split_text_by_bytes(block.text, body_target_bytes, body_overlap_bytes)
                if piece.strip()
            ]
    return [
        _TextBlock(text=piece, heading_path=block.heading_path)
        for piece in _split_text_by_bytes(text, chunk_target_bytes, chunk_overlap_bytes)
        if piece.strip()
    ]


def _text_with_heading_context(block: _TextBlock) -> str:
    if not block.heading_path:
        return block.text
    heading = " > ".join(block.heading_path)
    if block.text == heading or block.text.startswith(f"{heading}\n\n"):
        return block.text
    return f"{heading}\n\n{block.text}"


def _split_text_by_bytes(text: str, target_bytes: int, overlap_bytes: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        total = 0
        end = start
        while end < len(text):
            char_bytes = len(text[end].encode("utf-8"))
            if total and total + char_bytes > target_bytes:
                break
            total += char_bytes
            end += 1
            if total >= target_bytes:
                break
        if end <= start:
            end = start + 1
        pieces.append(text[start:end])
        if end >= len(text):
            break
        next_start = _overlap_start(text, start=start, end=end, overlap_bytes=overlap_bytes)
        start = end if next_start <= start else next_start
    return pieces


def _overlap_start(text: str, *, start: int, end: int, overlap_bytes: int) -> int:
    if overlap_bytes <= 0:
        return end
    total = 0
    index = end
    while index > start:
        char_bytes = len(text[index - 1].encode("utf-8"))
        if total + char_bytes > overlap_bytes:
            break
        index -= 1
        total += char_bytes
    return index


def _suffix_by_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    total = 0
    index = len(text)
    while index > 0:
        char_bytes = len(text[index - 1].encode("utf-8"))
        if total + char_bytes > max_bytes:
            break
        index -= 1
        total += char_bytes
    return text[index:]


def _parts_with_bounded_overlap(
    overlap_text: str,
    piece_text: str,
    target_bytes: int,
    *,
    heading_path: tuple[str, ...],
) -> tuple[list[str], int]:
    if not overlap_text:
        return [piece_text], 0
    separator_bytes = len(b"\n\n")
    if heading_path:
        heading_prefix = f"{' > '.join(heading_path)}\n\n"
        if piece_text.startswith(heading_prefix):
            piece_body = piece_text.removeprefix(heading_prefix)
            if not piece_body.strip():
                return [piece_text], 0
            piece_bytes = len(piece_text.encode("utf-8"))
            overlap_budget = target_bytes - piece_bytes - separator_bytes
            if overlap_budget <= 0:
                return [piece_text], 0
            overlap = _suffix_by_bytes(overlap_text, overlap_budget)
            if not overlap.strip():
                return [piece_text], 0
            return [f"{heading_prefix}{overlap}", piece_body], len(overlap.encode("utf-8"))

    piece_bytes = len(piece_text.encode("utf-8"))
    overlap_budget = target_bytes - piece_bytes - separator_bytes
    if overlap_budget <= 0:
        return [piece_text], 0
    overlap = _suffix_by_bytes(overlap_text, overlap_budget)
    if not overlap.strip():
        return [piece_text], 0
    return [overlap, piece_text], len(overlap.encode("utf-8"))


def _heading_paths_metadata(paths: list[tuple[str, ...]]) -> list[list[str]]:
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        unique.append(list(path))
    return unique


def _same_indexed_entry(existing: KnowledgeEntry, indexed: KnowledgeEntry) -> bool:
    return (
        existing.text == indexed.text
        and existing.namespace == indexed.namespace
        and existing.labels == indexed.labels
        and existing.kind == indexed.kind
        and existing.visibility == indexed.visibility
        and existing.status == indexed.status
        and existing.created_by_type == indexed.created_by_type
        and existing.created_by == indexed.created_by
        and existing.source_type == indexed.source_type
        and existing.source_uri == indexed.source_uri
        and existing.source_id == indexed.source_id
        and existing.source_hash == indexed.source_hash
        and existing.aspects == indexed.aspects
        and existing.impact_targets == indexed.impact_targets
        and existing.importance == indexed.importance
        and existing.importance_source == indexed.importance_source
        and existing.confidence == indexed.confidence
        and existing.expires_at == indexed.expires_at
        and existing.title == indexed.title
        and existing.metadata == indexed.metadata
    )


def _same_indexed_chunks(existing: list[KnowledgeChunk], indexed: list[KnowledgeChunk]) -> bool:
    if len(existing) != len(indexed):
        return False
    return all(
        _same_indexed_chunk(left, right) for left, right in zip(existing, indexed, strict=True)
    )


def _same_indexed_chunk(existing: KnowledgeChunk, indexed: KnowledgeChunk) -> bool:
    return (
        existing.id == indexed.id
        and existing.entry_id == indexed.entry_id
        and existing.text == indexed.text
        and existing.chunk_index == indexed.chunk_index
        and existing.content_hash == indexed.content_hash
        and existing.source_uri == indexed.source_uri
        and existing.metadata == indexed.metadata
    )


def _chunk_comparison_max_bytes(chunks: list[KnowledgeChunk]) -> int:
    chunk_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
    separator_bytes = max(len(chunks) - 1, 0) * len(b"\n\n")
    return max(chunk_bytes + separator_bytes + 1, 1)


def _bounded_entry_text(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    if truncated.strip():
        return truncated
    return text.strip()[0]


def _generated_entry_id(request: KnowledgeIndexRequest, source_hash: str) -> str:
    if request.source_uri is not None or request.source_id is not None:
        basis = "\0".join(
            [
                request.namespace,
                request.source_type or "",
                request.source_uri or "",
                request.source_id or "",
            ]
        )
    else:
        basis = "\0".join([request.namespace, request.kind, source_hash])
    return f"knowledge_{sha256(basis.encode('utf-8')).hexdigest()[:32]}"


def _hash_text(text: str) -> str:
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"
