from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.storage.knowledge_indexer import (
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    content_knowledge_entry_id,
    knowledge_source_hash,
)
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeVisibility,
)
from cayu.tools._errors import invalid_tool_arguments_result

DEFAULT_KNOWLEDGE_TOOL_LIMIT = DEFAULT_KNOWLEDGE_LIMIT
MAX_KNOWLEDGE_TOOL_LIMIT = 25
DEFAULT_KNOWLEDGE_TOOL_MAX_BYTES = DEFAULT_KNOWLEDGE_MAX_BYTES
MAX_KNOWLEDGE_TOOL_MAX_BYTES = 128 * 1024
DEFAULT_SEARCH_KNOWLEDGE_PREVIEW_BYTES = 320
DEFAULT_LIST_KNOWLEDGE_PREVIEW_BYTES = 240
MAX_KNOWLEDGE_TOOL_PREVIEW_BYTES = 4 * 1024
DEFAULT_AUTO_SEMANTIC_MIN_SCORE = 0.75
_MIN_SCORE_INPUT_SCHEMA = {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "description": (
        "Optional normalized relevance threshold for scored semantic hits. "
        "This is an application-owned retrieval policy override; set 0 to "
        "inspect all returned hits."
    ),
}
DEFAULT_READ_KNOWLEDGE_MAX_CHUNKS = 5
MAX_READ_KNOWLEDGE_MAX_CHUNKS = 50
DEFAULT_READ_KNOWLEDGE_AROUND = 0
MAX_READ_KNOWLEDGE_AROUND = 10
DEFAULT_REMEMBER_KNOWLEDGE_MAX_BYTES = 64 * 1024
MAX_REMEMBER_KNOWLEDGE_MAX_BYTES = 512 * 1024
DEFAULT_REMEMBER_KNOWLEDGE_CHUNK_TARGET_BYTES = 4_000
MAX_REMEMBER_KNOWLEDGE_CHUNK_TARGET_BYTES = 32 * 1024
DEFAULT_REMEMBER_KNOWLEDGE_MAX_CHUNKS = 100
MAX_REMEMBER_KNOWLEDGE_MAX_CHUNKS = 1_000
# Each knowledge tool only requires the store methods it actually calls, so
# read-only stores can back the read tools without implementing the write API.
_SEARCH_KNOWLEDGE_STORE_METHODS = ("search",)
_LIST_KNOWLEDGE_STORE_METHODS = ("list_entries",)
_READ_KNOWLEDGE_STORE_METHODS = ("read_chunks",)
_REMEMBER_KNOWLEDGE_STORE_METHODS = (
    "get_entry",
    "put_entry_with_chunks",
    "read_chunks",
    "delete_entry",
)


class RememberKnowledgePolicy(BaseModel):
    """Application policy for model-authored knowledge writes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_status: KnowledgeStatus = KnowledgeStatus.PENDING
    allow_active_writes: bool = False
    default_namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE
    default_visibility: KnowledgeVisibility = KnowledgeVisibility.GLOBAL
    allowed_kinds: tuple[str, ...] | None = None
    default_kind: str = "fact"
    default_created_by: str = "model"
    require_labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("default_namespace", "default_kind", "default_created_by")
    @classmethod
    def validate_clean_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("allowed_kinds", mode="before")
    @classmethod
    def validate_allowed_kinds(cls, value) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str | bytes):
            raise TypeError("allowed_kinds must be an iterable of strings.")
        kinds: list[str] = []
        try:
            items = list(value)
        except TypeError as exc:
            raise TypeError("allowed_kinds must be an iterable of strings.") from exc
        for index, item in enumerate(items):
            if type(item) is not str:
                raise ValueError(f"`allowed_kinds[{index}]` must be a string.")
            kinds.append(require_clean_nonblank(item, f"allowed_kinds[{index}]"))
        if not kinds:
            raise ValueError("allowed_kinds cannot be empty.")
        return tuple(dict.fromkeys(kinds))

    @field_validator("require_labels", mode="before")
    @classmethod
    def copy_required_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "require_labels")

    @model_validator(mode="after")
    def validate_status_policy(self) -> RememberKnowledgePolicy:
        if self.default_status is KnowledgeStatus.ACTIVE and not self.allow_active_writes:
            raise ValueError("default_status='active' requires allow_active_writes=True.")
        if self.default_status not in {KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE}:
            raise ValueError("default_status must be pending or active.")
        if self.allowed_kinds is not None and self.default_kind not in self.allowed_kinds:
            raise ValueError("default_kind must be included in allowed_kinds.")
        return self


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
                        "Search mode. Use auto by default. If the active store "
                        "supports semantic search, auto may use embedding-backed hybrid "
                        "recall. Use semantic or hybrid, or keyword, only when app "
                        "instructions or prior tool results indicate the active store "
                        "supports that mode or a specific mode is required."
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

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        allow_score_override: bool = False,
        auto_min_score: float | None = DEFAULT_AUTO_SEMANTIC_MIN_SCORE,
    ) -> None:
        super().__init__(spec=spec)
        self._allow_score_override = allow_score_override
        self._auto_min_score = _validate_optional_unit_float(
            auto_min_score,
            "auto_min_score",
        )
        if allow_score_override:
            schema = self.spec.input_schema
            schema.setdefault("properties", {})["min_score"] = dict(_MIN_SCORE_INPUT_SCHEMA)
            self.spec = self.spec.model_copy(update={"input_schema": schema})
            self._validate_spec()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(
            ctx,
            methods=_SEARCH_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
        if store is None:
            return _missing_knowledge_store_result()
        search_modes = _knowledge_search_modes_payload(store)
        try:
            if "min_score" in args and not self._allow_score_override:
                raise ValueError(
                    "Tool argument `min_score` is not enabled for this search_knowledge tool."
                )
            mode = _optional_search_mode(args, "mode")
            min_score = (
                _optional_unit_float(args, "min_score") if self._allow_score_override else None
            )
            effective_min_score = _effective_search_min_score(
                mode=mode,
                search_modes=search_modes,
                min_score=min_score,
                auto_min_score=self._auto_min_score,
            )
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
                mode=mode,
                min_score=effective_min_score,
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
        filtered_hits = _filter_search_hits(result.hits, min_score=effective_min_score)
        hits = [_knowledge_hit_payload(hit, preview_bytes=preview_bytes) for hit in filtered_hits]
        filtered_count = len(result.hits) - len(filtered_hits)
        min_score_applied = _min_score_applied(
            result.hits,
            min_score=effective_min_score,
            store_can_apply=_search_can_apply_min_score(search_modes, result.query.mode),
        )
        content = (
            "No knowledge results found."
            if not hits
            else _format_search_hits(filtered_hits, preview_bytes=preview_bytes)
        )
        if min_score_applied is False:
            content += (
                f"\nNote: min_score {effective_min_score} was not applied because the "
                "store returned no normalized-scored hits."
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
                "min_score": effective_min_score,
                "min_score_applied": min_score_applied,
                "filtered_hits": filtered_count,
            },
        )


class RememberKnowledgeTool(Tool):
    spec = ToolSpec(
        name="remember_knowledge",
        # Writes to the knowledge store; never overlaps other tools in a round.
        parallel_safe=False,
        description=(
            "Propose new durable knowledge for the active knowledge store. Use this only "
            "for stable facts, preferences, procedures, warnings, decisions, or lessons "
            "that should be reusable beyond the current turn. Model-authored knowledge "
            "is policy-controlled and defaults to pending review unless the application "
            "explicitly allows active writes. This tool creates new entries only; it does "
            "not edit, archive, or delete existing knowledge. Remembering identical text "
            "with the same kind again returns the existing entry instead of writing a "
            "duplicate."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": (
                        "Concise knowledge text to remember. Store one stable fact, "
                        "preference, procedure, warning, decision, or lesson per call; "
                        "do not paste large documents, transcripts, or raw tool output."
                    ),
                },
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Optional short title for human review and list previews.",
                },
                "kind": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": (
                        "Optional knowledge kind such as fact, preference, procedure, "
                        "instruction, skill, document, example, warning, decision, event, "
                        "or summary. Policy may restrict accepted kinds."
                    ),
                },
                "aspects": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "description": "Optional controlled aspects for retrieval routing.",
                },
            },
            "required": ["text"],
        },
    )

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        policy: RememberKnowledgePolicy | dict[str, Any] | None = None,
        max_text_bytes: int = DEFAULT_REMEMBER_KNOWLEDGE_MAX_BYTES,
        chunk_target_bytes: int = DEFAULT_REMEMBER_KNOWLEDGE_CHUNK_TARGET_BYTES,
        max_chunks: int = DEFAULT_REMEMBER_KNOWLEDGE_MAX_CHUNKS,
    ) -> None:
        super().__init__(spec=spec)
        self._policy = (
            RememberKnowledgePolicy()
            if policy is None
            else RememberKnowledgePolicy.model_validate(policy)
        )
        if self._policy.allowed_kinds is not None:
            schema = copy_json_value(self.spec.input_schema, "input_schema")
            schema["properties"]["kind"] = {
                **schema["properties"]["kind"],
                "enum": list(self._policy.allowed_kinds),
                "description": (
                    "Optional knowledge kind. Choose one of: "
                    f"{', '.join(self._policy.allowed_kinds)}. If omitted, policy "
                    f"uses {self._policy.default_kind}."
                ),
            }
            self.spec = self.spec.model_copy(update={"input_schema": schema})
            self._validate_spec()
        self._max_text_bytes = _validate_bounded_positive_int(
            max_text_bytes,
            "max_text_bytes",
            maximum=MAX_REMEMBER_KNOWLEDGE_MAX_BYTES,
        )
        self._chunk_target_bytes = _validate_bounded_positive_int(
            chunk_target_bytes,
            "chunk_target_bytes",
            minimum=DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES * 2,
            maximum=MAX_REMEMBER_KNOWLEDGE_CHUNK_TARGET_BYTES,
        )
        self._max_chunks = _validate_bounded_positive_int(
            max_chunks,
            "max_chunks",
            maximum=MAX_REMEMBER_KNOWLEDGE_MAX_CHUNKS,
        )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(
            ctx,
            methods=_REMEMBER_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
        if store is None:
            return _missing_knowledge_store_result()
        try:
            text = _require_arg_string(args, "text")
            if len(text.encode("utf-8")) > self._max_text_bytes:
                raise ValueError(f"`text` must be at most {self._max_text_bytes} bytes.")
            kind = _optional_arg_string(args, "kind") or self._policy.default_kind
            self._validate_kind(kind)
            metadata = _remember_metadata(ctx)
        except (ValidationError, ValueError) as exc:
            return _invalid_knowledge_arguments_result(exc)
        source_hash = knowledge_source_hash(text)
        entry_id = content_knowledge_entry_id(
            namespace=self._policy.default_namespace,
            kind=kind,
            source_hash=source_hash,
        )
        try:
            existing_entry = await store.get_entry(entry_id)
        except Exception:
            # Deduplication is best-effort; fall through to the normal write
            # path, which has its own failure handling.
            existing_entry = None
        if existing_entry is not None:
            if existing_entry.source_hash == source_hash and _remember_existing_entry_is_live(
                existing_entry
            ):
                return _remember_knowledge_already_known_result(existing_entry)
            if existing_entry.source_hash == source_hash:
                replacement_entry = await _remember_live_or_next_replacement_entry(
                    store,
                    entry_id=entry_id,
                    source_hash=source_hash,
                )
                if isinstance(replacement_entry, KnowledgeEntry):
                    return _remember_knowledge_already_known_result(replacement_entry)
                entry_id = replacement_entry
            elif existing_entry.source_hash != source_hash:
                # A different payload occupies the content-derived id (for
                # example a truncated-hash collision); write under a unique id
                # instead of overwriting the existing entry.
                entry_id = f"knowledge_{uuid4().hex}"
        try:
            request = KnowledgeIndexRequest(
                text=text,
                entry_id=entry_id,
                namespace=self._policy.default_namespace,
                labels=self._policy.require_labels,
                kind=kind,
                visibility=self._policy.default_visibility,
                status=self._policy.default_status,
                created_by_type=KnowledgeActorType.MODEL,
                created_by=ctx.agent_name or self._policy.default_created_by,
                source_type="tool",
                source_uri=f"cayu://sessions/{ctx.session_id}",
                source_id=ctx.session_id,
                aspects=_optional_string_list(args, "aspects") or [],
                title=_optional_arg_string(args, "title"),
                metadata=metadata,
                chunk_metadata=metadata,
                chunk_target_bytes=self._chunk_target_bytes,
                max_chunks=self._max_chunks,
                skip_unchanged=False,
            )
            result = KnowledgeIndexer().build(request)
            if result.truncated:
                raise ValueError("`text` exceeds the configured remember_knowledge chunk capacity.")
        except (ValidationError, ValueError) as exc:
            return _invalid_knowledge_arguments_result(exc)
        try:
            stored_entry = await store.put_entry_with_chunks(result.entry, result.chunks)
        except Exception as exc:
            inspection_error = None
            try:
                stored_entry = await store.get_entry(result.entry.id)
                stored_chunks = await store.read_chunks(
                    result.entry.id,
                    max_chunks=len(result.chunks) + 1,
                    max_bytes=_remember_chunk_read_max_bytes(result.chunks),
                )
            except Exception as inspect_exc:
                stored_entry = None
                stored_chunks = []
                inspection_error = str(inspect_exc)
            if not _remember_write_matches(result, stored_entry, stored_chunks):
                cleanup_error = None
                try:
                    await store.delete_entry(result.entry.id, hard=True)
                except Exception as cleanup_exc:
                    cleanup_error = str(cleanup_exc)
                return _knowledge_write_failed_result(
                    exc,
                    entry_id=result.entry.id,
                    cleanup_error=cleanup_error,
                    inspection_error=inspection_error,
                )
            result = result.model_copy(update={"entry": stored_entry, "written": True})
            return _remember_knowledge_success_result(
                result,
                post_write_error=str(exc),
            )
        result = result.model_copy(update={"entry": stored_entry, "written": True})
        return _remember_knowledge_success_result(result)

    def _validate_kind(self, kind: str) -> None:
        if self._policy.allowed_kinds is not None and kind not in self._policy.allowed_kinds:
            allowed = ", ".join(self._policy.allowed_kinds)
            raise ValueError(f"`kind` must be one of: {allowed}.")


def _remember_knowledge_success_result(
    result: Any,
    *,
    post_write_error: str | None = None,
) -> ToolResult:
    entry = result.entry
    status_note = (
        "It is active for normal retrieval."
        if entry.status is KnowledgeStatus.ACTIVE
        else "It is pending review and normal searches exclude it by default."
    )
    content = f"Knowledge stored as {entry.status.value}: {entry.id}. {status_note}"
    structured: dict[str, Any] = {
        "entry": _remembered_entry_payload(entry),
        "chunk_count": result.chunk_count,
        "written": result.written,
        "already_known": False,
        "source_hash": result.source_hash,
        "status": entry.status.value,
    }
    if post_write_error is not None:
        structured["post_write_error"] = post_write_error
    return ToolResult(
        content=content,
        structured=structured,
    )


def _remember_existing_entry_is_live(entry: KnowledgeEntry) -> bool:
    if entry.status not in {KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING}:
        return False
    return entry.expires_at is None or entry.expires_at > datetime.now(UTC)


async def _remember_live_or_next_replacement_entry(
    store: Any,
    *,
    entry_id: str,
    source_hash: str,
) -> KnowledgeEntry | str:
    for index in range(1, 11):
        replacement_entry_id = _remember_stale_replacement_entry_id(entry_id, index)
        try:
            replacement_entry = await store.get_entry(replacement_entry_id)
        except Exception:
            return replacement_entry_id
        if replacement_entry is None:
            return replacement_entry_id
        if replacement_entry.source_hash == source_hash and _remember_existing_entry_is_live(
            replacement_entry
        ):
            return replacement_entry
    return f"knowledge_{uuid4().hex}"


def _remember_stale_replacement_entry_id(entry_id: str, index: int) -> str:
    suffix = "_live" if index == 1 else f"_live_{index}"
    return f"{entry_id}{suffix}"


def _remember_knowledge_already_known_result(entry: KnowledgeEntry) -> ToolResult:
    status_note = (
        "It is active for normal retrieval."
        if entry.status is KnowledgeStatus.ACTIVE
        else f"Its status is {entry.status.value}."
    )
    content = (
        f"Knowledge already known as {entry.status.value}: {entry.id}. "
        f"{status_note} No new entry was written."
    )
    return ToolResult(
        content=content,
        structured={
            "entry": _remembered_entry_payload(entry),
            "written": False,
            "already_known": True,
            "source_hash": entry.source_hash,
            "status": entry.status.value,
        },
    )


def _remember_chunk_read_max_bytes(chunks: list[KnowledgeChunk]) -> int:
    chunk_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
    separator_bytes = max(len(chunks) - 1, 0) * len(b"\n\n")
    return max(chunk_bytes + separator_bytes + 1, 1)


def _remember_write_matches(
    result: Any,
    stored_entry: KnowledgeEntry | None,
    stored_chunks: list[KnowledgeChunk],
) -> bool:
    if stored_entry is None:
        return False
    expected_entry = result.entry
    if _remember_entry_write_payload(stored_entry) != _remember_entry_write_payload(expected_entry):
        return False
    if len(stored_chunks) != len(result.chunks):
        return False
    for expected_chunk, stored_chunk in zip(result.chunks, stored_chunks, strict=True):
        if stored_chunk.id != expected_chunk.id:
            return False
        if stored_chunk.entry_id != expected_chunk.entry_id:
            return False
        if stored_chunk.chunk_index != expected_chunk.chunk_index:
            return False
        if stored_chunk.content_hash != expected_chunk.content_hash:
            return False
        if stored_chunk.text != expected_chunk.text:
            return False
    return True


def _remember_entry_write_payload(entry: KnowledgeEntry) -> dict[str, Any]:
    """Fields that must survive a create write before recovery can treat it as stored."""

    return {
        "id": entry.id,
        "text": entry.text,
        "namespace": entry.namespace,
        "labels": entry.labels,
        "kind": entry.kind,
        "visibility": entry.visibility,
        "status": entry.status,
        "created_by_type": entry.created_by_type,
        "created_by": entry.created_by,
        "source_type": entry.source_type,
        "source_uri": entry.source_uri,
        "source_id": entry.source_id,
        "source_hash": entry.source_hash,
        "aspects": entry.aspects,
        "impact_targets": entry.impact_targets,
        "importance": entry.importance,
        "importance_source": entry.importance_source,
        "confidence": entry.confidence,
        "expires_at": entry.expires_at,
        "title": entry.title,
        "metadata": entry.metadata,
    }


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
        store = _require_knowledge_store(
            ctx,
            methods=_LIST_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
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
        store = _require_knowledge_store(
            ctx,
            methods=_READ_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
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


def _require_knowledge_store(
    ctx: ToolContext,
    *,
    methods: tuple[str, ...],
    tool_name: str,
) -> Any:
    if ctx.knowledge_store is None:
        return None
    missing = [
        method_name
        for method_name in methods
        if not callable(getattr(ctx.knowledge_store, method_name, None))
    ]
    if missing:
        raise TypeError(
            f"Tool context knowledge_store is missing methods required by "
            f"{tool_name}: {', '.join(missing)}."
        )
    return ctx.knowledge_store


def _missing_knowledge_store_result() -> ToolResult:
    return ToolResult(
        content="No knowledge store configured for this tool call.",
        structured={"error": "missing_knowledge_store"},
        is_error=True,
    )


def _invalid_knowledge_arguments_result(exc: Exception) -> ToolResult:
    return invalid_tool_arguments_result(exc)


def _knowledge_write_failed_result(
    exc: Exception,
    *,
    entry_id: str,
    cleanup_error: str | None,
    inspection_error: str | None = None,
) -> ToolResult:
    structured: dict[str, Any] = {
        "error": "knowledge_write_failed",
        "entry_id": entry_id,
        "cleanup": "failed" if cleanup_error is not None else "completed",
    }
    if inspection_error is not None:
        structured["inspection_error"] = inspection_error
    if cleanup_error is not None:
        structured["cleanup_error"] = cleanup_error
    return ToolResult(
        content=f"Failed to store knowledge: {exc}",
        structured=structured,
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


def _validate_bounded_positive_int(
    value: int,
    key: str,
    *,
    minimum: int = 1,
    maximum: int,
) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{key}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{key}` must be at least {minimum}.")
    if value > maximum:
        raise ValueError(f"`{key}` must be at most {maximum}.")
    return value


def _optional_unit_float(args: dict, key: str) -> float | None:
    value = args.get(key)
    if value is None:
        return None
    return _validate_optional_unit_float(value, key)


def _validate_optional_unit_float(value: float | None, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"`{key}` must be a number.")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"`{key}` must be between 0.0 and 1.0.")
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


def _remembered_entry_payload(entry: KnowledgeEntry) -> dict[str, Any]:
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
        "source_hash": entry.source_hash,
        "created_by_type": entry.created_by_type.value,
        "created_by": entry.created_by,
        "importance": entry.importance,
        "importance_source": entry.importance_source,
        "confidence": entry.confidence,
        "metadata": copy_json_value(entry.metadata, "metadata"),
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


def _effective_search_min_score(
    *,
    mode: KnowledgeSearchMode,
    search_modes: list[str],
    min_score: float | None,
    auto_min_score: float | None,
) -> float | None:
    if min_score is not None:
        return min_score
    if auto_min_score is None:
        return None
    if mode is not KnowledgeSearchMode.AUTO:
        return None
    if (
        KnowledgeSearchMode.SEMANTIC.value not in search_modes
        and KnowledgeSearchMode.HYBRID.value not in search_modes
    ):
        return None
    return auto_min_score


def _min_score_applied(
    hits: list[KnowledgeHit],
    *,
    min_score: float | None,
    store_can_apply: bool,
) -> bool | None:
    """Whether the requested score threshold could actually take effect.

    Returns None when no threshold was in force, True when the active store can
    apply scored semantic thresholds, and False when a threshold was requested
    against an unscored keyword-only result shape.
    """

    if min_score is None or min_score <= 0:
        return None
    if store_can_apply:
        return True
    return not hits or any(hit.score_normalized is not None for hit in hits)


def _filter_search_hits(hits: list[KnowledgeHit], *, min_score: float | None) -> list[KnowledgeHit]:
    if min_score is None or min_score <= 0:
        return list(hits)
    semantic_scored_hits = [hit for hit in hits if hit.score_normalized is not None]
    if not semantic_scored_hits:
        return list(hits)
    return [
        hit for hit in hits if hit.score_normalized is None or hit.score_normalized >= min_score
    ]


def _search_can_apply_min_score(
    search_modes: list[str],
    mode: KnowledgeSearchMode,
) -> bool:
    if mode is KnowledgeSearchMode.SEMANTIC:
        return KnowledgeSearchMode.SEMANTIC.value in search_modes
    if mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
        return (
            KnowledgeSearchMode.SEMANTIC.value in search_modes
            or KnowledgeSearchMode.HYBRID.value in search_modes
        )
    return False


def _remember_metadata(ctx: ToolContext) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tool_name": RememberKnowledgeTool.spec.name,
        "session_id": ctx.session_id,
    }
    if ctx.agent_name is not None:
        metadata["agent_name"] = ctx.agent_name
    if ctx.environment_name is not None:
        metadata["environment_name"] = ctx.environment_name
    if ctx.workspace_id is not None:
        metadata["workspace_id"] = ctx.workspace_id
    return metadata


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
        "min_score": query.min_score,
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
