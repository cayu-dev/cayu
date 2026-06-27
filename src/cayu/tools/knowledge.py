from __future__ import annotations

from typing import Any

from cayu._validation import copy_label_map, require_clean_nonblank, require_nonblank
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeChunk,
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeVisibility,
)

DEFAULT_KNOWLEDGE_TOOL_LIMIT = DEFAULT_KNOWLEDGE_LIMIT
MAX_KNOWLEDGE_TOOL_LIMIT = 25
DEFAULT_KNOWLEDGE_TOOL_MAX_BYTES = DEFAULT_KNOWLEDGE_MAX_BYTES
MAX_KNOWLEDGE_TOOL_MAX_BYTES = 128 * 1024
DEFAULT_READ_KNOWLEDGE_MAX_CHUNKS = 5
MAX_READ_KNOWLEDGE_MAX_CHUNKS = 50
DEFAULT_READ_KNOWLEDGE_AROUND = 0
MAX_READ_KNOWLEDGE_AROUND = 10
_KNOWLEDGE_STORE_METHODS = (
    "put_entry",
    "get_entry",
    "update_entry_status",
    "delete_entry",
    "replace_chunks",
    "put_entry_with_chunks",
    "read_chunks",
    "search",
)


class SearchKnowledgeTool(Tool):
    spec = ToolSpec(
        name="search_knowledge",
        description=(
            "Search the active knowledge store for reusable facts, procedures, skills, "
            "documents, warnings, decisions, or other durable context. Use this when "
            "relevant information may exist outside the current conversation."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the knowledge store.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional knowledge namespace. Defaults to `default`.",
                },
                "labels": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Optional exact-match labels such as project or user scope.",
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional entry kinds to include.",
                },
                "visibilities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [visibility.value for visibility in KnowledgeVisibility],
                    },
                    "description": "Optional visibility scopes to include.",
                },
                "aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional exact-match aspect filters.",
                },
                "impact_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional exact-match impact target filters.",
                },
                "source_type": {
                    "type": "string",
                    "description": "Optional exact source type filter.",
                },
                "source_id": {
                    "type": "string",
                    "description": "Optional exact source id filter.",
                },
                "include_expired": {
                    "type": "boolean",
                    "description": "Include expired entries when true.",
                },
                "mode": {
                    "type": "string",
                    "enum": [mode.value for mode in KnowledgeSearchMode],
                    "description": (
                        "Search mode. The active knowledge store decides which modes it supports."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_TOOL_LIMIT,
                    "description": "Maximum number of hits to return.",
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_TOOL_MAX_BYTES,
                    "description": "Maximum total preview bytes to return.",
                },
            },
            "required": ["query"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(ctx)
        if store is None:
            return _missing_knowledge_store_result()
        query = KnowledgeQuery(
            text=_require_arg_string(args, "query"),
            namespace=_optional_arg_string(args, "namespace") or "default",
            labels=_optional_labels(args, "labels"),
            kinds=_optional_string_list(args, "kinds"),
            visibilities=_optional_visibilities(args, "visibilities"),
            aspects=_optional_string_list(args, "aspects") or [],
            impact_targets=_optional_string_list(args, "impact_targets") or [],
            source_type=_optional_arg_string(args, "source_type"),
            source_id=_optional_arg_string(args, "source_id"),
            include_expired=_optional_bool(args, "include_expired", default=False),
            mode=_optional_search_mode(args, "mode"),
            limit=_optional_positive_int(
                args,
                "limit",
                default=DEFAULT_KNOWLEDGE_TOOL_LIMIT,
                maximum=MAX_KNOWLEDGE_TOOL_LIMIT,
            ),
            max_bytes=_optional_positive_int(
                args,
                "max_bytes",
                default=DEFAULT_KNOWLEDGE_TOOL_MAX_BYTES,
                maximum=MAX_KNOWLEDGE_TOOL_MAX_BYTES,
            ),
        )
        result = await store.search(query)
        hits = [_knowledge_hit_payload(hit) for hit in result.hits]
        content = "No knowledge results found." if not hits else _format_search_hits(result.hits)
        return ToolResult(
            content=content,
            structured={
                "query": result.query.model_dump(mode="json"),
                "hits": hits,
                "truncated": result.truncated,
                "limit": result.limit,
                "max_bytes": result.max_bytes,
                "total_hits_known": result.total_hits_known,
            },
        )


class ReadKnowledgeTool(Tool):
    spec = ToolSpec(
        name="read_knowledge",
        description=(
            "Read bounded chunks from a knowledge entry returned by search_knowledge. "
            "Use entry_id with an optional chunk_index and around window to expand context."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "Knowledge entry id to read.",
                },
                "chunk_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional chunk index to center the read around.",
                },
                "around": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_READ_KNOWLEDGE_AROUND,
                    "description": "Number of neighboring chunks to include around chunk_index.",
                },
                "max_chunks": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_READ_KNOWLEDGE_MAX_CHUNKS,
                    "description": "Maximum chunks to return.",
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_TOOL_MAX_BYTES,
                    "description": "Maximum bytes of chunk text to return.",
                },
            },
            "required": ["entry_id"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(ctx)
        if store is None:
            return _missing_knowledge_store_result()
        entry_id = _require_arg_string(args, "entry_id")
        chunk_index = _optional_nonnegative_int(args, "chunk_index")
        around = _optional_nonnegative_int(
            args,
            "around",
            default=DEFAULT_READ_KNOWLEDGE_AROUND,
            maximum=MAX_READ_KNOWLEDGE_AROUND,
        )
        max_chunks = _optional_positive_int(
            args,
            "max_chunks",
            default=DEFAULT_READ_KNOWLEDGE_MAX_CHUNKS,
            maximum=MAX_READ_KNOWLEDGE_MAX_CHUNKS,
        )
        max_bytes = _optional_positive_int(
            args,
            "max_bytes",
            default=DEFAULT_KNOWLEDGE_TOOL_MAX_BYTES,
            maximum=MAX_KNOWLEDGE_TOOL_MAX_BYTES,
        )
        chunks = await store.read_chunks(
            entry_id,
            chunk_index=chunk_index,
            around=around,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )
        chunk_payloads = [_knowledge_chunk_payload(chunk) for chunk in chunks]
        if not chunk_payloads:
            content = f"No knowledge chunks found for entry_id {entry_id!r}."
        else:
            content = _format_chunks(entry_id, chunks)
        return ToolResult(
            content=content,
            structured={
                "entry_id": entry_id,
                "chunk_index": chunk_index,
                "around": around,
                "max_chunks": max_chunks,
                "max_bytes": max_bytes,
                "chunks": chunk_payloads,
            },
        )


def _require_knowledge_store(ctx: ToolContext) -> Any:
    if ctx.knowledge_store is None:
        return None
    if not _is_knowledge_store(ctx.knowledge_store):
        raise TypeError("Tool context knowledge_store must implement KnowledgeStore.")
    return ctx.knowledge_store


def _is_knowledge_store(value: Any) -> bool:
    return all(
        callable(getattr(value, method_name, None)) for method_name in _KNOWLEDGE_STORE_METHODS
    )


def _missing_knowledge_store_result() -> ToolResult:
    return ToolResult(
        content="No knowledge store configured for this tool call.",
        structured={"error": "missing_knowledge_store"},
        is_error=True,
    )


def _require_arg_string(args: dict, key: str) -> str:
    value = args.get(key)
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    return require_nonblank(value, key)


def _optional_arg_string(args: dict, key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    return require_clean_nonblank(value, key)


def _optional_labels(args: dict, key: str) -> dict[str, str]:
    value = args.get(key)
    if value is None:
        return {}
    return copy_label_map(value, key)


def _optional_string_list(args: dict, key: str) -> list[str] | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is not list:
        raise ValueError(f"Tool argument `{key}` must be a list.")
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"Tool argument `{key}[{index}]` must be a string.")
        result.append(require_clean_nonblank(item, f"{key}[{index}]"))
    return list(dict.fromkeys(result))


def _optional_visibilities(args: dict, key: str) -> list[KnowledgeVisibility] | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is not list:
        raise ValueError(f"Tool argument `{key}` must be a list.")
    visibilities: list[KnowledgeVisibility] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"Tool argument `{key}[{index}]` must be a string.")
        try:
            visibilities.append(KnowledgeVisibility(item))
        except ValueError as exc:
            raise ValueError(f"Tool argument `{key}[{index}]` is not a valid visibility.") from exc
    if not visibilities:
        raise ValueError(f"Tool argument `{key}` cannot be empty.")
    return list(dict.fromkeys(visibilities))


def _optional_search_mode(args: dict, key: str) -> KnowledgeSearchMode:
    value = args.get(key, KnowledgeSearchMode.AUTO.value)
    if isinstance(value, KnowledgeSearchMode):
        return value
    else:
        if type(value) is not str:
            raise ValueError(f"Tool argument `{key}` must be a string.")
        try:
            return KnowledgeSearchMode(value)
        except ValueError as exc:
            raise ValueError(f"Tool argument `{key}` is not a valid search mode.") from exc


def _optional_bool(args: dict, key: str, *, default: bool) -> bool:
    value = args.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"Tool argument `{key}` must be a boolean.")
    return value


def _optional_positive_int(
    args: dict,
    key: str,
    *,
    default: int,
    maximum: int,
) -> int:
    value = args.get(key, default)
    if type(value) is not int:
        raise ValueError(f"Tool argument `{key}` must be an integer.")
    if value <= 0:
        raise ValueError(f"Tool argument `{key}` must be greater than zero.")
    if value > maximum:
        raise ValueError(f"Tool argument `{key}` must be at most {maximum}.")
    return value


def _optional_nonnegative_int(
    args: dict,
    key: str,
    *,
    default: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = args.get(key, default)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"Tool argument `{key}` must be an integer.")
    if value < 0:
        raise ValueError(f"Tool argument `{key}` must be greater than or equal to zero.")
    if maximum is not None and value > maximum:
        raise ValueError(f"Tool argument `{key}` must be at most {maximum}.")
    return value


def _knowledge_hit_payload(hit: KnowledgeHit) -> dict[str, Any]:
    entry = hit.entry
    return {
        "entry_id": entry.id,
        "namespace": entry.namespace,
        "kind": entry.kind,
        "visibility": entry.visibility.value,
        "status": entry.status.value,
        "title": entry.title,
        "labels": dict(entry.labels),
        "aspects": list(entry.aspects),
        "impact_targets": list(entry.impact_targets),
        "source_type": entry.source_type,
        "source_uri": entry.source_uri,
        "source_id": entry.source_id,
        "importance": entry.importance,
        "confidence": entry.confidence,
        "chunk_id": hit.chunk.id if hit.chunk is not None else None,
        "chunk_index": hit.chunk.chunk_index if hit.chunk is not None else None,
        "score": hit.score,
        "rank": hit.rank,
        "score_kind": hit.score_kind,
        "score_normalized": hit.score_normalized,
        "reason": hit.reason,
        "text_preview": hit.text_preview,
    }


def _knowledge_chunk_payload(chunk: KnowledgeChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "entry_id": chunk.entry_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "source_uri": chunk.source_uri,
        "metadata": dict(chunk.metadata),
    }


def _format_search_hits(hits: list[KnowledgeHit]) -> str:
    lines = ["Knowledge results:"]
    for index, hit in enumerate(hits, start=1):
        entry = hit.entry
        title = f" title={entry.title!r}" if entry.title else ""
        chunk = ""
        if hit.chunk is not None:
            chunk = f" chunk_index={hit.chunk.chunk_index}"
        score = f" score={hit.score:.4f}" if hit.score is not None else ""
        lines.append(f"{index}. entry_id={entry.id!r} kind={entry.kind!r}{title}{chunk}{score}")
        if hit.text_preview:
            lines.append(hit.text_preview)
    lines.append("Use read_knowledge with entry_id and optional chunk_index to expand a hit.")
    return "\n".join(lines)


def _format_chunks(entry_id: str, chunks: list[KnowledgeChunk]) -> str:
    lines = [f"Knowledge chunks for entry_id {entry_id!r}:"]
    for chunk in chunks:
        lines.append(f"[chunk_index={chunk.chunk_index}]")
        lines.append(chunk.text)
    return "\n".join(lines)
