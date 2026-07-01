from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from cayu._validation import copy_label_map, require_clean_nonblank, require_nonblank
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeChunk,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeVisibility,
)

DEFAULT_KNOWLEDGE_TOOL_LIMIT = DEFAULT_KNOWLEDGE_LIMIT
MAX_KNOWLEDGE_TOOL_LIMIT = 25
DEFAULT_KNOWLEDGE_TOOL_MAX_BYTES = DEFAULT_KNOWLEDGE_MAX_BYTES
MAX_KNOWLEDGE_TOOL_MAX_BYTES = 128 * 1024
DEFAULT_SEARCH_KNOWLEDGE_PREVIEW_BYTES = 320
DEFAULT_LIST_KNOWLEDGE_PREVIEW_BYTES = 240
MAX_KNOWLEDGE_TOOL_PREVIEW_BYTES = 4 * 1024
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
    "list_entries",
)


class SearchKnowledgeTool(Tool):
    spec = ToolSpec(
        name="search_knowledge",
        description=(
            "Search the active knowledge store for reusable facts, procedures, skills, "
            "documents, warnings, decisions, or other durable context. Use this when "
            "relevant information may exist outside the current conversation. Prefer a "
            "broad keyword query before applying exact facet filters such as aspects, "
            "labels, kinds, or source fields. Do not use a truncated facet value as a "
            "hard filter unless it is clearly relevant; over-filtering can hide the "
            "right knowledge."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Simple search query. Tokenized as broad any-term keyword search.",
                },
                "any": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "minItems": 1,
                    "description": "Optional terms where at least one should match.",
                },
                "all": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "minItems": 1,
                    "description": "Optional terms that must all match.",
                },
                "none": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "minItems": 1,
                    "description": (
                        "Optional terms that must not match. Use only after prior "
                        "results show those terms are irrelevant; this can hide "
                        "otherwise relevant knowledge."
                    ),
                },
                "phrases": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "minItems": 1,
                    "description": "Optional exact phrases where at least one should match.",
                },
                "namespace": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Optional knowledge namespace. Defaults to `default`.",
                },
                "labels": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": "\\S",
                    },
                    "description": "Optional exact-match labels such as project or user scope.",
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
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
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "description": (
                        "Optional exact-match aspect filters. Use only when a prior "
                        "untruncated discovery result or app context shows the aspect is "
                        "clearly relevant; otherwise search without this filter first."
                    ),
                },
                "impact_targets": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "description": "Optional exact-match impact target filters.",
                },
                "source_type": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Optional exact source type filter.",
                },
                "source_id": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
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
                        "Search mode. Use auto by default. Use semantic or hybrid "
                        "only when app instructions or prior tool results indicate "
                        "the active knowledge store supports semantic search, "
                        "especially for conceptual recall where exact keywords or "
                        "facets may miss relevant knowledge."
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
                "preview_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_TOOL_PREVIEW_BYTES,
                    "description": (
                        "Maximum bytes of preview text per hit. Use read_knowledge "
                        "to expand a hit instead of raising this for broad searches."
                    ),
                },
            },
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(ctx)
        if store is None:
            return _missing_knowledge_store_result()
        try:
            query = KnowledgeQuery(
                text=_optional_nonblank_string(args, "query"),
                any_terms=_optional_string_list(args, "any") or [],
                all_terms=_optional_string_list(args, "all") or [],
                none_terms=_optional_string_list(args, "none") or [],
                phrases=_optional_string_list(args, "phrases") or [],
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
            preview_bytes = _optional_positive_int(
                args,
                "preview_bytes",
                default=DEFAULT_SEARCH_KNOWLEDGE_PREVIEW_BYTES,
                maximum=MAX_KNOWLEDGE_TOOL_PREVIEW_BYTES,
            )
        except (ValidationError, ValueError) as exc:
            return _invalid_knowledge_arguments_result(exc)
        result = await store.search(query)
        hits = [_knowledge_hit_payload(hit, preview_bytes=preview_bytes) for hit in result.hits]
        search_modes = _knowledge_search_modes_payload(store)
        content = (
            "No knowledge results found."
            if not hits
            else _format_search_hits(result.hits, preview_bytes=preview_bytes)
        )
        return ToolResult(
            content=content,
            structured={
                "query": _search_query_payload(result.query),
                "hits": hits,
                "truncated": result.truncated,
                "limit": result.limit,
                "max_bytes": result.max_bytes,
                "preview_bytes": preview_bytes,
                "total_hits_known": result.total_hits_known,
                "search_modes": search_modes,
            },
        )


class ListKnowledgeTool(Tool):
    spec = ToolSpec(
        name="list_knowledge",
        description=(
            "Discover what active knowledge exists without guessing exact search terms. "
            "Use this to list entries or facets such as kinds, labels, aspects, namespaces, "
            "or source types before calling search_knowledge/read_knowledge. For large "
            "stores, call group_by first and leave include_entries false; request entry "
            "previews only for small or already-filtered result sets. If the result says "
            "facets were truncated, increase limit or narrow filters before relying on "
            "a missing or low-ranked facet value; otherwise use broad search_knowledge "
            "without that exact facet filter."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "namespace": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Optional namespace filter. Omit to list across namespaces.",
                },
                "labels": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": "\\S",
                    },
                    "description": "Optional exact-match labels such as project or user scope.",
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
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
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "description": (
                        "Optional exact-match aspect filters. Use only for already-known "
                        "aspects; broad discovery with group_by can reveal available "
                        "aspect values."
                    ),
                },
                "impact_targets": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "description": "Optional exact-match impact target filters.",
                },
                "source_type": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Optional exact source type filter.",
                },
                "source_id": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Optional exact source id filter.",
                },
                "include_expired": {
                    "type": "boolean",
                    "description": "Include expired entries when true.",
                },
                "group_by": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [group.value for group in KnowledgeListGroup],
                    },
                    "minItems": 1,
                    "maxItems": len(KnowledgeListGroup),
                    "description": (
                        "Optional facet fields to count instead of relying on entry "
                        "previews. Use this first for large knowledge stores. If facets "
                        "are truncated, raise limit or narrow filters before choosing a "
                        "facet value as a hard search filter."
                    ),
                },
                "include_entries": {
                    "type": "boolean",
                    "description": (
                        "Whether to include entry previews along with facets. Defaults "
                        "to false when group_by is set, true otherwise. Keep false for "
                        "broad discovery; use true only for small or filtered lists."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_TOOL_LIMIT,
                    "description": (
                        "Maximum number of entries or facet values to return per facet "
                        "group. Use a higher value for broad facet discovery."
                    ),
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_TOOL_MAX_BYTES,
                    "description": "Maximum total preview bytes to return.",
                },
                "preview_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_TOOL_PREVIEW_BYTES,
                    "description": (
                        "Maximum bytes of preview text per listed entry. Use "
                        "search_knowledge/read_knowledge to inspect content."
                    ),
                },
            },
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(ctx)
        if store is None:
            return _missing_knowledge_store_result()
        try:
            query = KnowledgeListQuery(
                namespace=_optional_arg_string(args, "namespace"),
                labels=_optional_labels(args, "labels"),
                kinds=_optional_string_list(args, "kinds"),
                visibilities=_optional_visibilities(args, "visibilities"),
                aspects=_optional_string_list(args, "aspects") or [],
                impact_targets=_optional_string_list(args, "impact_targets") or [],
                source_type=_optional_arg_string(args, "source_type"),
                source_id=_optional_arg_string(args, "source_id"),
                include_expired=_optional_bool(args, "include_expired", default=False),
                group_by=None,
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
            group_by = _optional_list_groups(args, "group_by")
            preview_bytes = _optional_positive_int(
                args,
                "preview_bytes",
                default=DEFAULT_LIST_KNOWLEDGE_PREVIEW_BYTES,
                maximum=MAX_KNOWLEDGE_TOOL_PREVIEW_BYTES,
            )
            include_entries = _optional_bool(
                args,
                "include_entries",
                default=not group_by,
            )
        except (ValidationError, ValueError) as exc:
            return _invalid_knowledge_arguments_result(exc)
        result = await store.list_entries(
            _list_query_with_group(query, group_by[0] if group_by else None)
        )
        all_facets = list(result.facets)
        facets_truncated = bool(getattr(result, "facets_truncated", False))
        for group in group_by[1:]:
            grouped_result = await store.list_entries(_list_query_with_group(query, group))
            all_facets.extend(grouped_result.facets)
            facets_truncated = facets_truncated or bool(
                getattr(grouped_result, "facets_truncated", False)
            )
        exposed_entries = result.entries if include_entries else []
        entries = [
            _knowledge_list_item_payload(item, preview_bytes=preview_bytes)
            for item in exposed_entries
        ]
        facets = [_knowledge_facet_payload(facet) for facet in all_facets]
        search_modes = _knowledge_search_modes_payload(store)
        content = _format_knowledge_list(
            exposed_entries,
            all_facets,
            total_entries_known=result.total_entries_known,
            include_entries=include_entries,
            preview_bytes=preview_bytes,
            facets_truncated=facets_truncated,
            search_modes=search_modes,
        )
        return ToolResult(
            content=content,
            structured={
                "query": _list_query_payload(query, group_by),
                "entries": entries,
                "facets": facets,
                "facet_groups": _knowledge_facet_groups_payload(all_facets),
                "facets_truncated": facets_truncated,
                "truncated": result.truncated or facets_truncated,
                "limit": result.limit,
                "max_bytes": result.max_bytes,
                "preview_bytes": preview_bytes,
                "include_entries": include_entries,
                "total_entries_known": result.total_entries_known,
                "search_modes": search_modes,
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
                    "minLength": 1,
                    "pattern": "\\S",
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
        try:
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
        except (ValidationError, ValueError) as exc:
            return _invalid_knowledge_arguments_result(exc)
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


def _invalid_knowledge_arguments_result(exc: Exception) -> ToolResult:
    return ToolResult(
        content=str(exc),
        structured={"error": "invalid_arguments"},
        is_error=True,
    )


def _require_arg_string(args: dict, key: str) -> str:
    value = args.get(key)
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    return require_nonblank(value, key)


def _optional_nonblank_string(args: dict, key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
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


def _optional_list_groups(args: dict, key: str) -> list[KnowledgeListGroup]:
    value = args.get(key)
    if value is None:
        return []
    if isinstance(value, KnowledgeListGroup):
        return [value]
    raw_groups: list[str]
    if type(value) is str:
        raw_groups = [value]
    elif type(value) is list:
        raw_groups = []
        for index, item in enumerate(value):
            if type(item) is not str:
                raise ValueError(f"Tool argument `{key}[{index}]` must be a string.")
            raw_groups.append(item)
        if not raw_groups:
            raise ValueError(f"Tool argument `{key}` cannot be empty.")
    else:
        raise ValueError(f"Tool argument `{key}` must be a string or list of strings.")

    groups: list[KnowledgeListGroup] = []
    for index, raw_group in enumerate(raw_groups):
        try:
            groups.append(KnowledgeListGroup(raw_group))
        except ValueError as exc:
            raise ValueError(
                f"Tool argument `{key}[{index}]` is not a valid knowledge list group."
            ) from exc
    return list(dict.fromkeys(groups))


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


def _knowledge_hit_payload(hit: KnowledgeHit, *, preview_bytes: int) -> dict[str, Any]:
    entry = hit.entry
    text_preview, preview_truncated = _bounded_preview(hit.text_preview, preview_bytes)
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
        "text_preview": text_preview,
        "text_preview_truncated": preview_truncated,
    }


def _knowledge_list_item_payload(
    item: KnowledgeListItem,
    *,
    preview_bytes: int,
) -> dict[str, Any]:
    entry = item.entry
    text_preview, preview_truncated = _bounded_preview(item.text_preview, preview_bytes)
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
        "chunk_count": item.chunk_count,
        "text_preview": text_preview,
        "text_preview_truncated": preview_truncated,
    }


def _knowledge_facet_payload(facet: KnowledgeFacet) -> dict[str, Any]:
    return {
        "field": facet.field.value,
        "key": facet.key,
        "value": facet.value,
        "count": facet.count,
    }


def _knowledge_facet_groups_payload(
    facets: list[KnowledgeFacet],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for facet in facets:
        groups.setdefault(facet.field.value, []).append(_knowledge_facet_payload(facet))
    return groups


def _knowledge_search_modes_payload(store: Any) -> list[str]:
    supported = getattr(store, "supported_search_modes", None)
    if callable(supported):
        modes = supported()
    else:
        modes = (KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD)
    return [KnowledgeSearchMode(mode).value for mode in modes]


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


def _list_query_with_group(
    query: KnowledgeListQuery,
    group_by: KnowledgeListGroup | None,
) -> KnowledgeListQuery:
    return KnowledgeListQuery(
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
        group_by=group_by,
        limit=query.limit,
        max_bytes=query.max_bytes,
    )


def _list_query_payload(
    query: KnowledgeListQuery,
    group_by: list[KnowledgeListGroup],
) -> dict[str, Any]:
    return {
        "namespace": query.namespace,
        "labels": dict(query.labels),
        "kinds": list(query.kinds) if query.kinds is not None else None,
        "visibilities": (
            [visibility.value for visibility in query.visibilities]
            if query.visibilities is not None
            else None
        ),
        "aspects": list(query.aspects),
        "impact_targets": list(query.impact_targets),
        "source_type": query.source_type,
        "source_id": query.source_id,
        "include_expired": query.include_expired,
        "group_by": [group.value for group in group_by] if group_by else None,
        "limit": query.limit,
        "max_bytes": query.max_bytes,
    }


def _search_query_payload(query: KnowledgeQuery) -> dict[str, Any]:
    return {
        "query": query.text,
        "any": list(query.any_terms),
        "all": list(query.all_terms),
        "none": list(query.none_terms),
        "phrases": list(query.phrases),
        "namespace": query.namespace,
        "labels": dict(query.labels),
        "kinds": list(query.kinds) if query.kinds is not None else None,
        "visibilities": (
            [visibility.value for visibility in query.visibilities]
            if query.visibilities is not None
            else None
        ),
        "aspects": list(query.aspects),
        "impact_targets": list(query.impact_targets),
        "source_type": query.source_type,
        "source_id": query.source_id,
        "include_expired": query.include_expired,
        "mode": query.mode.value,
        "limit": query.limit,
        "max_bytes": query.max_bytes,
    }


def _format_search_hits(hits: list[KnowledgeHit], *, preview_bytes: int) -> str:
    lines = ["Knowledge results:"]
    for index, hit in enumerate(hits, start=1):
        entry = hit.entry
        title = f" title={entry.title!r}" if entry.title else ""
        chunk = ""
        if hit.chunk is not None:
            chunk = f" chunk_index={hit.chunk.chunk_index}"
        score = f" score={hit.score:.4f}" if hit.score is not None else ""
        lines.append(f"{index}. entry_id={entry.id!r} kind={entry.kind!r}{title}{chunk}{score}")
        text_preview, preview_truncated = _bounded_preview(hit.text_preview, preview_bytes)
        if text_preview:
            suffix = " [preview truncated]" if preview_truncated else ""
            lines.append(f"{text_preview}{suffix}")
    lines.append("Use read_knowledge with entry_id and optional chunk_index to expand a hit.")
    return "\n".join(lines)


def _format_knowledge_list(
    entries: list[KnowledgeListItem],
    facets: list[KnowledgeFacet],
    *,
    total_entries_known: int | None,
    include_entries: bool,
    preview_bytes: int,
    facets_truncated: bool,
    search_modes: list[str],
) -> str:
    header = [
        "Knowledge discovery:",
        "Search modes: " + ", ".join(search_modes),
    ]
    if not entries and not facets:
        if not include_entries and total_entries_known:
            return "\n".join(
                [
                    *header,
                    (
                        "Knowledge discovery found matching entries, but no entry previews were "
                        "requested and no facets matched the selected group. Use include_entries=true "
                        "for a bounded entry sample, or choose a different group_by field."
                    ),
                ]
            )
        return "\n".join([*header, "No knowledge entries found."])
    lines = list(header)
    if facets:
        lines.append("Facets:")
        for facet in facets:
            key = f"{facet.key}=" if facet.key is not None else ""
            lines.append(f"- {facet.field.value}: {key}{facet.value} ({facet.count})")
        if facets_truncated:
            lines.append(
                "Facet list truncated. Increase limit or narrow filters before choosing "
                "a facet value that may be hidden."
            )
    if entries:
        lines.append("Entries:")
        for index, item in enumerate(entries, start=1):
            entry = item.entry
            title = f" title={entry.title!r}" if entry.title else ""
            lines.append(
                f"{index}. entry_id={entry.id!r} namespace={entry.namespace!r} "
                f"kind={entry.kind!r}{title} chunks={item.chunk_count}"
            )
            text_preview, preview_truncated = _bounded_preview(
                item.text_preview,
                preview_bytes,
            )
            if text_preview:
                suffix = " [preview truncated]" if preview_truncated else ""
                lines.append(f"{text_preview}{suffix}")
    lines.append("Use search_knowledge for targeted recall, then read_knowledge to expand a hit.")
    return "\n".join(lines)


def _format_chunks(entry_id: str, chunks: list[KnowledgeChunk]) -> str:
    lines = [f"Knowledge chunks for entry_id {entry_id!r}:"]
    for chunk in chunks:
        lines.append(f"[chunk_index={chunk.chunk_index}]")
        lines.append(chunk.text)
    return "\n".join(lines)


def _bounded_preview(text: str | None, max_bytes: int) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip(), True
