from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._exception_groups import exception_group_children
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_finite,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.storage.knowledge_indexer import (
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    content_knowledge_entry_id,
    copy_knowledge_index_result,
    knowledge_source_hash,
)
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    MAX_KNOWLEDGE_REVISION,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEntry,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    KnowledgeQuery,
    KnowledgeRevisionConflict,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    _knowledge_publication_operation_id,
    _next_knowledge_revision,
    copy_knowledge_access_scope,
    copy_knowledge_entry,
    copy_knowledge_publication_receipt,
    prepare_knowledge_publication,
)
from cayu.tools._errors import structured_invalid_arguments, tool_argument_validation
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    await_invocation_operation,
)
from cayu.tools._redaction import (
    await_revision_stable_secret_output,
    record_ambiguous_secret_output,
    unstable_secret_redaction_result,
)
from cayu.vaults import SecretRedactor

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
MAX_RETAINED_REMEMBER_KNOWLEDGE_PUBLICATIONS = 64
MAX_RETAINED_REMEMBER_KNOWLEDGE_READS = 64
# Each knowledge tool only requires the store methods it actually calls, so
# read-only stores can back the read tools without implementing the write API.
_SEARCH_KNOWLEDGE_STORE_METHODS = ("search",)
_LIST_KNOWLEDGE_STORE_METHODS = ("list_entries",)
_READ_KNOWLEDGE_STORE_METHODS = ("read_chunks",)
_REMEMBER_KNOWLEDGE_STORE_METHODS = (
    "get_entry",
    "publish_entry_revision",
    "load_entry_publication_receipt",
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


def _copy_remember_knowledge_policy(
    policy: RememberKnowledgePolicy | dict[str, Any],
) -> RememberKnowledgePolicy:
    """Revalidate and detach application-owned publication authority."""

    if type(policy) is RememberKnowledgePolicy:
        return RememberKnowledgePolicy(
            default_status=policy.default_status,
            allow_active_writes=policy.allow_active_writes,
            default_namespace=policy.default_namespace,
            default_visibility=policy.default_visibility,
            allowed_kinds=policy.allowed_kinds,
            default_kind=policy.default_kind,
            default_created_by=policy.default_created_by,
            # Let the model's field validator perform the defensive copy so
            # malformed forged instances retain the public ValidationError
            # contract instead of leaking a helper-level ValueError.
            require_labels=policy.require_labels,
        )
    if type(policy) is not dict:
        raise TypeError("policy must be a RememberKnowledgePolicy or dictionary.")
    return RememberKnowledgePolicy.model_validate(policy)


class SearchKnowledgeTool(Tool):
    spec = ToolSpec(
        name="search_knowledge",
        effect=ToolEffect.NONE,
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

    def _execution_profile_material(self) -> dict[str, Any]:
        """Return the bounded configuration that governs knowledge search."""

        return {
            "allow_score_override": self._allow_score_override,
            "auto_min_score": self._auto_min_score,
        }

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

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(
            ctx,
            methods=_SEARCH_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
        if store is None:
            return _missing_knowledge_store_result()
        search_modes = _knowledge_search_modes_payload(store)
        with tool_argument_validation():
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

        async def search(_redactor: SecretRedactor):
            return await store.search(query)

        captured = await await_revision_stable_secret_output(ctx, search)
        if captured is None:
            return unstable_secret_redaction_result()
        result, capture_snapshot = captured
        redactor = capture_snapshot.redactor
        filtered_hits = _filter_search_hits(result.hits, min_score=effective_min_score)
        hits = [
            _knowledge_hit_payload(
                hit,
                preview_bytes=preview_bytes,
                redactor=redactor,
            )
            for hit in filtered_hits
        ]
        filtered_count = len(result.hits) - len(filtered_hits)
        min_score_applied = _min_score_applied(
            result.hits,
            min_score=effective_min_score,
            store_can_apply=_search_can_apply_min_score(search_modes, result.query.mode),
        )
        content = (
            "No knowledge results found."
            if not hits
            else _format_search_hits(
                filtered_hits,
                preview_bytes=preview_bytes,
                redactor=redactor,
            )
        )
        if min_score_applied is False:
            content += (
                f"\nNote: min_score {effective_min_score} was not applied because the "
                "store returned no normalized-scored hits."
            )
        if any(
            source.text_preview is not None
            and (not source.text_preview_complete or payload["text_preview_truncated"])
            for source, payload in zip(filtered_hits, hits, strict=True)
        ):
            record_ambiguous_secret_output(ctx, capture_snapshot)
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
        effect=ToolEffect.EXTERNAL,
        description=(
            "Propose new durable knowledge for the active knowledge store. Use this only "
            "for stable facts, preferences, procedures, warnings, decisions, or lessons "
            "that should be reusable beyond the current turn. Model-authored knowledge "
            "is policy-controlled and defaults to pending review unless the application "
            "explicitly allows active writes. This tool creates logical entries and never "
            "mutates a stored revision; it cannot edit, archive, or delete one. If identical "
            "archived, deleted, or expired knowledge already owns the logical id, it appends "
            "a reviewed successor revision. "
            "Remembering identical live text with the same kind returns the existing entry "
            "instead of writing a duplicate."
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

    def _execution_profile_material(self) -> dict[str, Any] | None:
        """Return material only when the write policy is Cayu's public default."""

        policy = self._policy
        if policy != RememberKnowledgePolicy():
            # Application policy contains exact storage scope and attribution
            # values. Even an unkeyed hash of that low-entropy material would
            # provide a durable offline guessing oracle. Callers that need
            # cross-process portability must explicitly version the behavior
            # through ToolSpec.execution_profile_identity.
            return None
        return {
            "policy": "cayu_default",
            "max_text_bytes": self._max_text_bytes,
            "chunk_target_bytes": self._chunk_target_bytes,
            "max_chunks": self._max_chunks,
        }

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
            RememberKnowledgePolicy() if policy is None else _copy_remember_knowledge_policy(policy)
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
        # Opaque stores are not required to abort a dispatched mutation when
        # their awaiting task is cancelled. Retain those exact operations so a
        # caller timeout/cancellation can finish without abandoning ownership,
        # and so an identical retry joins the original mutation instead of
        # dispatching a competing write.
        self._owned_publications: dict[str, _OwnedKnowledgePublication] = {}
        # Read-only receipt and entry lookups may be abandoned after authentic
        # caller cancellation even when an opaque adapter ignores cancellation.
        # Retain every dispatched lookup in a bounded registry until it settles.
        self._read_operations = BoundedInvocationOperationRegistry(
            max_operations=MAX_RETAINED_REMEMBER_KNOWLEDGE_READS
        )

    @property
    def _publish_arguments(self) -> bool:
        """Keep knowledge material out of terminal audit/transcript projections."""

        return False

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(
            ctx,
            methods=_REMEMBER_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
        if store is None:
            return _missing_knowledge_store_result()
        with tool_argument_validation():
            policy = _copy_remember_knowledge_policy(self._policy)
            text = _require_arg_string(args, "text")
            if len(text.encode("utf-8")) > self._max_text_bytes:
                raise ValueError(f"`text` must be at most {self._max_text_bytes} bytes.")
            kind = _optional_arg_string(args, "kind") or policy.default_kind
            self._validate_kind(kind, policy=policy)
            aspects = _optional_string_list(args, "aspects") or []
            title = _optional_arg_string(args, "title")
            metadata = _remember_metadata(ctx)
        source_hash = knowledge_source_hash(text)
        with tool_argument_validation():
            operation_id = _knowledge_publication_operation_id(
                ctx.idempotency_key or f"remember_{uuid4().hex}"
            )
            intent_sha256 = _remember_request_intent_sha256(
                text=text,
                namespace=policy.default_namespace,
                labels=policy.require_labels,
                kind=kind,
                visibility=policy.default_visibility,
                status=policy.default_status,
                created_by=ctx.agent_name or policy.default_created_by,
                session_id=ctx.session_id,
                aspects=aspects,
                title=title,
                metadata=metadata,
                chunk_target_bytes=self._chunk_target_bytes,
                max_chunks=self._max_chunks,
            )
        owned = self._owned_publications.get(operation_id)
        if owned is not None:
            if owned.intent_sha256 != intent_sha256:
                return _knowledge_write_failed_result(
                    entry_id=None,
                    outcome="operation_conflict",
                )
            return await asyncio.shield(owned.task)
        if not _remember_store_supports_owned_publication(store):
            return _knowledge_write_failed_result(
                entry_id=None,
                outcome="owned_publication_unsupported",
            )
        try:
            prior_receipt = await _remember_load_publication_receipt(
                store,
                operation_id,
                operation_registry=self._read_operations,
            )
        except NotImplementedError:
            return _knowledge_write_failed_result(
                entry_id=None,
                outcome="owned_publication_unsupported",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _knowledge_write_failed_result(
                entry_id=None,
                outcome="receipt_read_failed",
            )
        entry_id = (
            prior_receipt.entry_id
            if prior_receipt is not None
            else content_knowledge_entry_id(
                namespace=policy.default_namespace,
                kind=kind,
                source_hash=source_hash,
            )
        )
        revision_parent: KnowledgeEntry | None = None
        if prior_receipt is None:
            # Deduplication is best-effort, but the extension read still needs
            # an owned task boundary so store-originated cancellation cannot
            # impersonate cancellation of this run.
            existing_entry = await _remember_load_entry(
                store,
                entry_id,
                operation_registry=self._read_operations,
            )
            if existing_entry is not None:
                if _remember_entry_matches_material_and_scope(
                    existing_entry,
                    expected_entry_id=entry_id,
                    text=text,
                    source_hash=source_hash,
                    namespace=policy.default_namespace,
                    kind=kind,
                    visibility=policy.default_visibility,
                    required_labels=policy.require_labels,
                ):
                    if _remember_existing_entry_is_live(existing_entry):
                        return _remember_knowledge_already_known_result(existing_entry)
                    revision_parent = existing_entry
                elif existing_entry.source_hash == source_hash:
                    return _knowledge_write_failed_result(
                        entry_id=entry_id,
                        outcome="publication_conflict",
                    )
                elif existing_entry.source_hash != source_hash:
                    # A different payload occupies the content-derived id (for
                    # example a truncated-hash collision). Use an operation-bound
                    # fallback so concurrent delivery of one exact operation
                    # converges instead of choosing independent random identities.
                    entry_id = _remember_operation_fallback_entry_id(
                        entry_id,
                        operation_id,
                    )
        with tool_argument_validation():
            request = KnowledgeIndexRequest(
                text=text,
                entry_id=entry_id,
                namespace=policy.default_namespace,
                labels=policy.require_labels,
                kind=kind,
                visibility=policy.default_visibility,
                status=policy.default_status,
                created_by_type=KnowledgeActorType.MODEL,
                created_by=ctx.agent_name or policy.default_created_by,
                source_type="tool",
                source_uri=f"cayu://sessions/{ctx.session_id}",
                source_id=ctx.session_id,
                aspects=aspects,
                title=title,
                metadata=metadata,
                chunk_metadata=metadata,
                chunk_target_bytes=self._chunk_target_bytes,
                max_chunks=self._max_chunks,
                skip_unchanged=False,
            )
            result = KnowledgeIndexer().build(request)
            if revision_parent is not None:
                result = _remember_result_for_successor(result, revision_parent)
            if result.truncated:
                raise ValueError("`text` exceeds the configured remember_knowledge chunk capacity.")
        if prior_receipt is not None:
            result = _remember_result_with_receipt_identity(result, prior_receipt)
            observation = await _remember_observe_owned_publication(
                store,
                result=result,
                operation_id=operation_id,
                receipt=prior_receipt,
                operation_registry=self._read_operations,
            )
            if observation.confirmed:
                return _remember_knowledge_replayed_result(result)
            return _knowledge_write_failed_result(
                # The receipt is extension-owned and has not authenticated its
                # identity against the requested publication. Keep its entry id
                # out of model-visible failure evidence, but do not turn an
                # unavailable confirmation read into an identity conflict.
                entry_id=None,
                outcome=(
                    "receipt_conflict"
                    if observation.receipt_incompatible
                    else "receipt_read_failed"
                ),
            )

        return await self._await_owned_publication(
            store=store,
            result=result,
            operation_id=operation_id,
            source_hash=source_hash,
            namespace=policy.default_namespace,
            kind=kind,
            visibility=policy.default_visibility,
            required_labels=policy.require_labels,
            intent_sha256=intent_sha256,
        )

    async def _await_owned_publication(
        self,
        *,
        store: Any,
        result: Any,
        operation_id: str,
        source_hash: str,
        namespace: str,
        kind: str,
        visibility: KnowledgeVisibility,
        required_labels: dict[str, str],
        intent_sha256: str,
    ) -> ToolResult:
        owned = self._owned_publications.get(operation_id)
        if owned is not None and owned.intent_sha256 != intent_sha256:
            return _knowledge_write_failed_result(
                entry_id=None,
                outcome="operation_conflict",
            )
        if owned is None:
            if len(self._owned_publications) >= MAX_RETAINED_REMEMBER_KNOWLEDGE_PUBLICATIONS:
                return _knowledge_write_failed_result(
                    entry_id=None,
                    outcome="publication_capacity_exhausted",
                )
            task = asyncio.create_task(
                _remember_publish_owned(
                    store,
                    result=result,
                    operation_id=operation_id,
                    source_hash=source_hash,
                    namespace=namespace,
                    kind=kind,
                    visibility=visibility,
                    required_labels=required_labels,
                    operation_registry=self._read_operations,
                )
            )
            owned = _OwnedKnowledgePublication(
                intent_sha256=intent_sha256,
                task=task,
            )
            self._owned_publications[operation_id] = owned
            task.add_done_callback(
                lambda completed, operation_id=operation_id, owned=owned: (
                    self._release_owned_publication(
                        operation_id,
                        owned,
                        completed,
                    )
                )
            )
        # Shield only the retained task. The invoking task remains normally
        # cancellable, while the exact store operation keeps one owner until it
        # settles and consumes its result through the completion callback.
        return await asyncio.shield(owned.task)

    def _release_owned_publication(
        self,
        operation_id: str,
        owned: _OwnedKnowledgePublication,
        completed: asyncio.Task[ToolResult],
    ) -> None:
        if self._owned_publications.get(operation_id) is owned:
            self._owned_publications.pop(operation_id, None)
        # An invocation may have timed out or been cancelled. Always observe a
        # detached task's terminal state so extension failures do not become
        # unhandled event-loop diagnostics.
        if not completed.cancelled():
            completed.exception()

    def _validate_kind(self, kind: str, *, policy: RememberKnowledgePolicy) -> None:
        if policy.allowed_kinds is not None and kind not in policy.allowed_kinds:
            allowed = ", ".join(policy.allowed_kinds)
            raise ValueError(f"`kind` must be one of: {allowed}.")


@dataclass(frozen=True)
class _OwnedKnowledgePublication:
    intent_sha256: str
    task: asyncio.Task[ToolResult]


@dataclass(frozen=True)
class _RememberPublicationObservation:
    confirmed: bool = False
    receipt_present: bool = False
    receipt_absent: bool = False
    receipt_incompatible: bool = False


def _remember_request_intent_sha256(
    *,
    text: str,
    namespace: str,
    labels: dict[str, str],
    kind: str,
    visibility: KnowledgeVisibility,
    status: KnowledgeStatus,
    created_by: str,
    session_id: str,
    aspects: list[str],
    title: str | None,
    metadata: dict[str, Any],
    chunk_target_bytes: int,
    max_chunks: int,
) -> str:
    """Fingerprint caller authority before any retry-side store operation."""

    material = {
        "contract": "cayu.remember_knowledge.request_intent.v1",
        "text": text,
        "namespace": namespace,
        "labels": labels,
        "kind": kind,
        "visibility": visibility.value,
        "status": status.value,
        "created_by_type": KnowledgeActorType.MODEL.value,
        "created_by": created_by,
        "source_type": "tool",
        "source_uri": f"cayu://sessions/{session_id}",
        "source_id": session_id,
        "aspects": aspects,
        "title": title,
        "metadata": metadata,
        "chunk_metadata": metadata,
        "chunk_target_bytes": chunk_target_bytes,
        "max_chunks": max_chunks,
    }
    return sha256(
        canonical_durable_json_bytes(material, "remember knowledge publication intent")
    ).hexdigest()


async def _remember_load_publication_receipt(
    store: Any,
    operation_id: str,
    *,
    operation_registry: BoundedInvocationOperationRegistry,
) -> KnowledgePublicationReceipt | None:
    async def operation_factory(
        store: Any = store,
        operation_id: str = operation_id,
    ) -> KnowledgePublicationReceipt | None:
        receipt = await store.load_entry_publication_receipt(operation_id)
        return None if receipt is None else copy_knowledge_publication_receipt(receipt)

    pending = await_invocation_operation(
        operation_factory,
        request_child_cancellation=False,
        abandon_on_caller_cancellation=True,
        operation_registry=operation_registry,
    )
    del operation_factory
    outcome = await pending
    del pending
    uncontainable_failure = _remember_uncontainable_store_failure(outcome.error)
    if uncontainable_failure is not None:
        raise uncontainable_failure
    if outcome.cancellation is not None:
        cause = _remember_cancellation_cause(outcome.error, settled=False)
        raise outcome.cancellation from cause
    if isinstance(outcome.error, asyncio.CancelledError):
        raise RuntimeError(
            "Knowledge receipt lookup was cancelled without caller cancellation."
        ) from None
    if isinstance(outcome.error, BaseExceptionGroup):
        raise RuntimeError("Knowledge receipt lookup reported multiple failures.") from None
    if outcome.error is not None:
        raise outcome.error
    return outcome.result


async def _remember_load_entry(
    store: Any,
    entry_id: str,
    *,
    operation_registry: BoundedInvocationOperationRegistry,
) -> KnowledgeEntry | None:
    """Load one entry without trusting extension-owned cancellation or output."""

    async def operation_factory(
        store: Any = store,
        entry_id: str = entry_id,
    ) -> KnowledgeEntry | None:
        entry = await store.get_entry(entry_id)
        return None if entry is None else copy_knowledge_entry(entry)

    pending = await_invocation_operation(
        operation_factory,
        request_child_cancellation=False,
        abandon_on_caller_cancellation=True,
        operation_registry=operation_registry,
    )
    del operation_factory
    outcome = await pending
    del pending
    uncontainable_failure = _remember_uncontainable_store_failure(outcome.error)
    if uncontainable_failure is not None:
        raise uncontainable_failure
    if outcome.cancellation is not None:
        raise outcome.cancellation from None
    if outcome.error is not None or outcome.result is None:
        # These reads only optimize deduplication and conflict recovery. A
        # bounded extension failure must not abort or redefine publication.
        return None
    return outcome.result


def _remember_store_supports_owned_publication(store: Any) -> bool:
    if type(store) is _ScopedKnowledgeStoreHandle:
        store = store._store
    store_type = type(store)
    return (
        getattr(store_type, "publish_entry_revision", None)
        is not KnowledgeStore.publish_entry_revision
        and getattr(store_type, "load_entry_publication_receipt", None)
        is not KnowledgeStore.load_entry_publication_receipt
    )


def _remember_publication_conflict(
    failure: BaseException | None,
) -> tuple[bool, str | None]:
    """Detach bounded conflict evidence from an extension-owned exception."""

    # Subclasses can override attribute access. Exact-type admission plus
    # object-level state lookup prevents extension code from running while the
    # runtime classifies the failure.
    if type(failure) is KnowledgeRevisionConflict:
        return True, "revision_conflict"
    if type(failure) is not KnowledgePublicationConflict:
        return False, None
    try:
        state = object.__getattribute__(failure, "__dict__")
    except BaseException:
        return False, None
    if type(state) is not dict:
        return False, None
    reason = state.get("reason")
    if type(reason) is not str:
        return False, None
    try:
        reason = require_clean_nonblank(reason, "reason")
    except (TypeError, ValueError):
        return False, None
    return True, reason


def _remember_deterministic_publication_failure(
    failure: BaseException | None,
) -> str | None:
    """Classify exact built-in failures that prove publication did not commit."""

    # Keep this exact-type boundary aligned with `_remember_publication_conflict`.
    # An extension-owned subclass can redefine exception behavior and does not
    # carry the built-in store's atomic no-commit guarantee.
    if type(failure) is KnowledgeAccessDenied:
        return "access_denied"
    if type(failure) is KnowledgeChunkConflict:
        return "publication_conflict"
    if type(failure) is KnowledgeRevisionConflict:
        return "publication_conflict"
    return None


async def _remember_publish_owned(
    store: Any,
    *,
    result: Any,
    operation_id: str,
    source_hash: str,
    namespace: str,
    kind: str,
    visibility: KnowledgeVisibility,
    required_labels: dict[str, str],
    operation_registry: BoundedInvocationOperationRegistry,
) -> ToolResult:
    # Keep reconciliation authority detached from mutable objects handed to an
    # extension store. The public store hook receives its own normalized copies,
    # so in-place adapter normalization cannot redefine the operation Cayu later
    # authenticates from the receipt.
    result = copy_knowledge_index_result(result)
    _, publication_entry, publication_chunks, _ = prepare_knowledge_publication(
        result.entry,
        result.chunks,
        operation_id=operation_id,
        expected_revision=_remember_expected_revision(result.entry),
    )

    async def operation_factory(
        store: Any = store,
        entry: KnowledgeEntry = publication_entry,
        chunks: list[KnowledgeChunk] = publication_chunks,
        operation_id: str = operation_id,
    ) -> KnowledgePublicationReceipt:
        return copy_knowledge_publication_receipt(
            await store.publish_entry_revision(
                entry,
                chunks,
                operation_id=operation_id,
                expected_revision=_remember_expected_revision(entry),
            )
        )

    pending = await_invocation_operation(
        operation_factory,
        request_child_cancellation=False,
    )
    del operation_factory
    outcome = await pending
    del pending
    uncontainable_failure = _remember_uncontainable_store_failure(outcome.error)
    if uncontainable_failure is not None:
        raise uncontainable_failure
    if outcome.cancellation is not None:
        settled, cancellation = await _remember_confirm_owned_publication_after_cancellation(
            store,
            result=result,
            operation_id=operation_id,
            cancellation=outcome.cancellation,
            operation_registry=operation_registry,
        )
        cause = _remember_cancellation_cause(outcome.error, settled=settled)
        raise cancellation from cause

    failure = (
        RuntimeError("Knowledge publication reported multiple failures.")
        if isinstance(outcome.error, BaseExceptionGroup)
        else outcome.error
    )
    failure_is_conflict, conflict_reason = _remember_publication_conflict(failure)
    deterministic_failure_outcome = _remember_deterministic_publication_failure(failure)
    returned_receipt: KnowledgePublicationReceipt | None = None
    if failure is None:
        try:
            if not isinstance(outcome.result, KnowledgePublicationReceipt):
                raise TypeError("Publication result is not a knowledge receipt.")
            returned_receipt = copy_knowledge_publication_receipt(outcome.result)
        except (TypeError, ValueError):
            failure = RuntimeError("Knowledge store returned an invalid publication receipt.")
    confirmed = False
    operation_receipt_present = False
    operation_receipt_absent = False
    operation_receipt_conflict = False
    try:
        observation = await _remember_observe_owned_publication(
            store,
            result=result,
            operation_id=operation_id,
            receipt=returned_receipt,
            operation_registry=operation_registry,
        )
        confirmed = observation.confirmed
        operation_receipt_present = observation.receipt_present
        operation_receipt_absent = observation.receipt_absent
        if not confirmed:
            try:
                durable_receipt = await _remember_load_publication_receipt(
                    store,
                    operation_id,
                    operation_registry=operation_registry,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                durable_receipt = None
            else:
                if durable_receipt is None:
                    operation_receipt_absent = True
                else:
                    operation_receipt_present = True
            if durable_receipt is not None:
                reconciled_result = _remember_result_with_receipt_identity(
                    result,
                    durable_receipt,
                )
                observation = await _remember_observe_owned_publication(
                    store,
                    result=reconciled_result,
                    operation_id=operation_id,
                    receipt=durable_receipt,
                    operation_registry=operation_registry,
                )
                confirmed = observation.confirmed
                operation_receipt_present |= observation.receipt_present
                operation_receipt_absent |= observation.receipt_absent
                # The first pass may differ only because a concurrent exact
                # publication supplied the durable entry identity. Only this
                # receipt-reconstructed pass can prove an operation conflict.
                operation_receipt_conflict = observation.receipt_incompatible
                if confirmed:
                    result = reconciled_result
    except asyncio.CancelledError as cancellation:
        (
            settled,
            retained_cancellation,
        ) = await _remember_confirm_owned_publication_after_cancellation(
            store,
            result=result,
            operation_id=operation_id,
            cancellation=cancellation,
            operation_registry=operation_registry,
        )
        cause = _remember_cancellation_cause(failure, settled=settled)
        raise retained_cancellation from cause
    if confirmed:
        replayed_publication = (returned_receipt is not None and returned_receipt.replayed) or (
            failure_is_conflict and conflict_reason == "operation_mismatch"
        )
        if replayed_publication:
            return _remember_knowledge_replayed_result(result)
        return _remember_knowledge_success_result(
            result.model_copy(update={"written": True}),
            post_write_error=(
                None
                if failure is None or failure_is_conflict
                else "publication_acknowledgement_lost"
            ),
        )
    publication_absence_confirmed = operation_receipt_absent and not operation_receipt_present
    if failure_is_conflict and operation_receipt_conflict:
        outcome_code = "operation_conflict"
    elif failure_is_conflict and conflict_reason in {
        "entry_occupied",
        "concurrent_occupancy",
        "revision_conflict",
    }:
        winner = await _remember_live_matching_entry(
            store,
            entry_id=result.entry.id,
            text=result.entry.text,
            source_hash=source_hash,
            namespace=namespace,
            kind=kind,
            visibility=visibility,
            required_labels=required_labels,
            operation_registry=operation_registry,
        )
        if winner is not None:
            return _remember_knowledge_already_known_result(winner)
        outcome_code = (
            "publication_conflict" if publication_absence_confirmed else "ambiguous_publication"
        )
    elif failure_is_conflict and publication_absence_confirmed:
        outcome_code = "publication_conflict"
    elif failure_is_conflict:
        outcome_code = "ambiguous_publication"
    elif deterministic_failure_outcome is not None and publication_absence_confirmed:
        outcome_code = deterministic_failure_outcome
    elif isinstance(failure, asyncio.CancelledError):
        outcome_code = "store_cancelled"
    elif failure is None and (operation_receipt_conflict or publication_absence_confirmed):
        outcome_code = "invalid_publication_result"
    else:
        outcome_code = "ambiguous_publication"
    return _knowledge_write_failed_result(
        entry_id=result.entry.id,
        outcome=outcome_code,
    )


async def _remember_confirm_owned_publication(
    store: Any,
    *,
    result: Any,
    operation_id: str,
    receipt: KnowledgePublicationReceipt | None = None,
    operation_registry: BoundedInvocationOperationRegistry,
) -> bool:
    observation = await _remember_observe_owned_publication(
        store,
        result=result,
        operation_id=operation_id,
        receipt=receipt,
        operation_registry=operation_registry,
    )
    return observation.confirmed


async def _remember_observe_owned_publication(
    store: Any,
    *,
    result: Any,
    operation_id: str,
    receipt: KnowledgePublicationReceipt | None = None,
    operation_registry: BoundedInvocationOperationRegistry,
) -> _RememberPublicationObservation:
    try:
        durable_receipt = await _remember_load_publication_receipt(
            store,
            operation_id,
            operation_registry=operation_registry,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return _RememberPublicationObservation()
    if durable_receipt is None:
        return _RememberPublicationObservation(receipt_absent=True)
    try:
        if receipt is not None and _remember_receipt_identity(
            receipt
        ) != _remember_receipt_identity(durable_receipt):
            return _RememberPublicationObservation(
                receipt_present=True,
                receipt_incompatible=True,
            )
        _, expected_entry, _, request_sha256 = prepare_knowledge_publication(
            result.entry,
            result.chunks,
            operation_id=operation_id,
            expected_revision=_remember_expected_revision(result.entry),
        )
        confirmed = (
            durable_receipt.operation_id == operation_id
            and durable_receipt.entry_id == expected_entry.id
            and durable_receipt.entry_revision == expected_entry.revision
            and durable_receipt.expected_revision == _remember_expected_revision(expected_entry)
            and durable_receipt.request_sha256 == request_sha256
            and durable_receipt.entry_created_at == expected_entry.created_at
            and durable_receipt.entry_updated_at == expected_entry.updated_at
        )
        return _RememberPublicationObservation(
            confirmed=confirmed,
            receipt_present=True,
            receipt_incompatible=not confirmed,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return _RememberPublicationObservation(receipt_present=True)


async def _remember_confirm_owned_publication_after_cancellation(
    store: Any,
    *,
    result: Any,
    operation_id: str,
    cancellation: asyncio.CancelledError,
    operation_registry: BoundedInvocationOperationRegistry,
) -> tuple[bool, asyncio.CancelledError]:
    """Finish receipt reconciliation while retaining the first caller signal."""

    def operation_factory(
        store: Any = store,
        result: Any = result,
        operation_id: str = operation_id,
    ):
        return _remember_reconcile_owned_publication(
            store,
            result=result,
            operation_id=operation_id,
            operation_registry=operation_registry,
        )

    pending = await_invocation_operation(
        operation_factory,
        request_child_cancellation=False,
        cancellation=cancellation,
    )
    del operation_factory
    outcome = await pending
    del pending
    uncontainable_failure = _remember_uncontainable_store_failure(outcome.error)
    if uncontainable_failure is not None:
        raise uncontainable_failure
    retained_cancellation = outcome.cancellation or cancellation
    if outcome.error is not None or type(outcome.result) is not bool:
        return False, retained_cancellation
    return outcome.result, retained_cancellation


async def _remember_reconcile_owned_publication(
    store: Any,
    *,
    result: Any,
    operation_id: str,
    operation_registry: BoundedInvocationOperationRegistry,
) -> bool:
    """Confirm current or receipt-reconstructed publication authority."""

    if await _remember_confirm_owned_publication(
        store,
        result=result,
        operation_id=operation_id,
        operation_registry=operation_registry,
    ):
        return True
    try:
        durable_receipt = await _remember_load_publication_receipt(
            store,
            operation_id,
            operation_registry=operation_registry,
        )
        if durable_receipt is None:
            return False
        reconciled_result = _remember_result_with_receipt_identity(result, durable_receipt)
        return await _remember_confirm_owned_publication(
            store,
            result=reconciled_result,
            operation_id=operation_id,
            receipt=durable_receipt,
            operation_registry=operation_registry,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return False


def _remember_uncontainable_store_failure(
    error: BaseException | None,
) -> BaseException | None:
    """Return a fatal/unclassifiable extension signal, if one is present."""

    if error is None:
        return None
    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            return error
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is None:
                return error
            pending.extend(children)
        elif not isinstance(candidate, Exception | asyncio.CancelledError):
            return error
    return None


def _remember_receipt_identity(receipt: KnowledgePublicationReceipt) -> tuple[object, ...]:
    copied = copy_knowledge_publication_receipt(receipt)
    return (
        copied.operation_id,
        copied.entry_id,
        copied.entry_revision,
        copied.expected_revision,
        copied.request_sha256,
        copied.entry_created_at,
        copied.entry_updated_at,
        copied.committed_at,
    )


def _remember_result_with_receipt_identity(
    result: Any,
    receipt: KnowledgePublicationReceipt,
) -> Any:
    replay_entry = copy_knowledge_entry(
        result.entry.model_copy(
            update={
                "id": receipt.entry_id,
                "revision": receipt.entry_revision,
                "created_at": receipt.entry_created_at,
                "updated_at": receipt.entry_updated_at,
            }
        )
    )
    replay_chunks = [
        chunk.model_copy(
            update={
                "id": (f"{receipt.entry_id}:r{receipt.entry_revision}:{chunk.chunk_index}"),
                "entry_id": receipt.entry_id,
                "entry_revision": receipt.entry_revision,
            }
        )
        for chunk in result.chunks
    ]
    return copy_knowledge_index_result(
        result.model_copy(
            update={
                "entry": replay_entry,
                "chunks": replay_chunks,
            }
        )
    )


def _remember_result_for_successor(result: Any, parent: KnowledgeEntry) -> Any:
    result = copy_knowledge_index_result(result)
    if result.entry.id != parent.id or result.entry.namespace != parent.namespace:
        raise ValueError("Remembered revision must preserve logical identity and namespace.")
    revision = _next_knowledge_revision(parent.revision)
    entry = result.entry.model_copy(
        update={
            "revision": revision,
            "created_at": parent.created_at,
            "updated_at": max(datetime.now(UTC), parent.updated_at),
        }
    )
    chunks = [
        chunk.model_copy(
            update={
                "id": f"{entry.id}:r{revision}:{chunk.chunk_index}",
                "entry_revision": revision,
            }
        )
        for chunk in result.chunks
    ]
    return copy_knowledge_index_result(
        result.model_copy(
            update={
                "entry": entry,
                "chunks": chunks,
                "chunk_count": len(chunks),
            }
        )
    )


def _remember_expected_revision(entry: KnowledgeEntry) -> int | None:
    return None if entry.revision == 1 else entry.revision - 1


async def _remember_live_matching_entry(
    store: Any,
    *,
    entry_id: str,
    text: str,
    source_hash: str,
    namespace: str,
    kind: str,
    visibility: KnowledgeVisibility,
    required_labels: dict[str, str],
    operation_registry: BoundedInvocationOperationRegistry,
) -> KnowledgeEntry | None:
    entry = await _remember_load_entry(
        store,
        entry_id,
        operation_registry=operation_registry,
    )
    if (
        entry is not None
        and _remember_entry_matches_material_and_scope(
            entry,
            expected_entry_id=entry_id,
            text=text,
            source_hash=source_hash,
            namespace=namespace,
            kind=kind,
            visibility=visibility,
            required_labels=required_labels,
        )
        and _remember_existing_entry_is_live(entry)
    ):
        return copy_knowledge_entry(entry)
    return None


def _remember_cancellation_cause(
    failure: BaseException | None,
    *,
    settled: bool,
) -> RuntimeError | None:
    if failure is None:
        return None
    if settled and isinstance(failure, asyncio.CancelledError):
        return None
    if settled:
        return RuntimeError("Knowledge publication committed before cancellation settled.")
    return RuntimeError("Knowledge publication outcome remained ambiguous after cancellation.")


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


def _remember_knowledge_replayed_result(result: Any) -> ToolResult:
    entry = result.entry
    return ToolResult(
        content=(
            f"Knowledge publication previously committed: {entry.id}. "
            "Its current lifecycle state was not checked."
        ),
        structured={
            "entry": {"entry_id": entry.id, "revision": entry.revision},
            "chunk_count": result.chunk_count,
            "written": False,
            "already_known": None,
            "source_hash": result.source_hash,
            "status": None,
            "publication_replayed": True,
        },
    )


def _remember_existing_entry_is_live(entry: KnowledgeEntry) -> bool:
    if entry.status not in {KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING}:
        return False
    return entry.expires_at is None or entry.expires_at > datetime.now(UTC)


def _remember_operation_fallback_entry_id(entry_id: str, operation_id: str) -> str:
    material = f"cayu-remember-operation-entry-v1\0{entry_id}\0{operation_id}".encode()
    return f"knowledge_{sha256(material).hexdigest()}"


def _remember_entry_matches_material_and_scope(
    entry: KnowledgeEntry,
    *,
    expected_entry_id: str,
    text: str,
    source_hash: str,
    namespace: str,
    kind: str,
    visibility: KnowledgeVisibility,
    required_labels: dict[str, str],
) -> bool:
    return (
        entry.id == expected_entry_id
        and entry.text == text
        and entry.source_hash == source_hash
        and knowledge_source_hash(entry.text) == source_hash
        and entry.namespace == namespace
        and entry.kind == kind
        and entry.visibility is visibility
        and all(entry.labels.get(key) == value for key, value in required_labels.items())
    )


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


class ListKnowledgeTool(Tool):
    spec = ToolSpec(
        name="list_knowledge",
        effect=ToolEffect.NONE,
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

    def _execution_profile_material(self) -> dict[str, object]:
        return {}

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(
            ctx,
            methods=_LIST_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
        if store is None:
            return _missing_knowledge_store_result()
        with tool_argument_validation():
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

        async def list_entries(_redactor: SecretRedactor):
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
            return result, all_facets, facets_truncated

        captured = await await_revision_stable_secret_output(ctx, list_entries)
        if captured is None:
            return unstable_secret_redaction_result()
        (result, all_facets, facets_truncated), capture_snapshot = captured
        redactor = capture_snapshot.redactor
        exposed_entries = result.entries if include_entries else []
        entries = [
            _knowledge_list_item_payload(
                item,
                preview_bytes=preview_bytes,
                redactor=redactor,
            )
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
            redactor=redactor,
            facets_truncated=facets_truncated,
            search_modes=search_modes,
        )
        if any(
            source.text_preview is not None
            and (not source.text_preview_complete or payload["text_preview_truncated"])
            for source, payload in zip(exposed_entries, entries, strict=True)
        ):
            record_ambiguous_secret_output(ctx, capture_snapshot)
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
        effect=ToolEffect.NONE,
        description=(
            "Read bounded chunks from any knowledge entry returned by automatic "
            "knowledge candidates, search_knowledge, or list_knowledge. Use entry_id "
            "with an optional revision, chunk_index, and around window to expand context."
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
                "revision": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_KNOWLEDGE_REVISION,
                    "description": (
                        "Optional exact historical revision. Defaults to the current revision."
                    ),
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

    def _execution_profile_material(self) -> dict[str, object]:
        return {}

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(
            ctx,
            methods=_READ_KNOWLEDGE_STORE_METHODS,
            tool_name=self.spec.name,
        )
        if store is None:
            return _missing_knowledge_store_result()
        with tool_argument_validation():
            entry_id = require_unicode_scalar_text(
                require_clean_nonblank(_require_arg_string(args, "entry_id"), "entry_id"),
                "entry_id",
            )
            revision = _optional_nonnegative_int(
                args,
                "revision",
                maximum=MAX_KNOWLEDGE_REVISION,
            )
            if revision == 0:
                raise ValueError("Tool argument `revision` must be greater than zero.")
            chunk_index = _optional_nonnegative_int(args, "chunk_index")
            around = _optional_nonnegative_int(
                args,
                "around",
                default=DEFAULT_READ_KNOWLEDGE_AROUND,
                maximum=MAX_READ_KNOWLEDGE_AROUND,
            )
            if chunk_index is None and around != 0:
                raise ValueError("`around` requires `chunk_index`.")
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
            revision=revision,
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
                "revision": chunks[0].entry_revision if chunks else revision,
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
    access_scope = ctx.knowledge_access_scope
    if access_scope is None:
        bound_scope = getattr(ctx.knowledge_store, "bound_access_scope", None)
        if callable(bound_scope):
            access_scope = bound_scope()
    if access_scope is None:
        raise PermissionError(
            f"Tool context knowledge_store for {tool_name} is missing an access scope."
        )
    return _ScopedKnowledgeStoreHandle(
        ctx.knowledge_store,
        copy_knowledge_access_scope(access_scope),
    )


class _ScopedKnowledgeStoreHandle:
    """Bind one trusted ToolContext scope without exposing it to model arguments."""

    def __init__(self, store: Any, access_scope: KnowledgeAccessScope) -> None:
        self._store = store
        self._access_scope = copy_knowledge_access_scope(access_scope)
        bound_scope = getattr(store, "bound_access_scope", None)
        self._scope_is_store_bound = callable(bound_scope) and bound_scope() == self._access_scope

    def supported_search_modes(self):
        supported = getattr(self._store, "supported_search_modes", None)
        if not callable(supported):
            return (KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD)
        return supported()

    async def search(self, query):
        if self._scope_is_store_bound:
            return await self._store.search(query)
        return await self._store.search(query, access_scope=self._access_scope)

    async def list_entries(self, query):
        if self._scope_is_store_bound:
            return await self._store.list_entries(query)
        return await self._store.list_entries(query, access_scope=self._access_scope)

    async def read_chunks(self, entry_id, **kwargs):
        if self._scope_is_store_bound:
            return await self._store.read_chunks(entry_id, **kwargs)
        return await self._store.read_chunks(
            entry_id,
            access_scope=self._access_scope,
            **kwargs,
        )

    async def get_entry(self, entry_id, **kwargs):
        if self._scope_is_store_bound:
            return await self._store.get_entry(entry_id, **kwargs)
        return await self._store.get_entry(
            entry_id,
            access_scope=self._access_scope,
            **kwargs,
        )

    async def publish_entry_revision(
        self,
        entry,
        chunks,
        *,
        operation_id,
        expected_revision=None,
    ):
        if self._scope_is_store_bound:
            return await self._store.publish_entry_revision(
                entry,
                chunks,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )
        return await self._store.publish_entry_revision(
            entry,
            chunks,
            access_scope=self._access_scope,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )

    async def load_entry_publication_receipt(self, operation_id):
        if self._scope_is_store_bound:
            return await self._store.load_entry_publication_receipt(operation_id)
        return await self._store.load_entry_publication_receipt(
            operation_id,
            access_scope=self._access_scope,
        )


def _missing_knowledge_store_result() -> ToolResult:
    return ToolResult(
        content="No knowledge store configured for this tool call.",
        structured={"error": "missing_knowledge_store"},
        is_error=True,
    )


def _knowledge_write_failed_result(
    *,
    entry_id: str | None,
    outcome: str,
) -> ToolResult:
    structured: dict[str, Any] = {
        "error": "knowledge_write_failed",
        "outcome": require_clean_nonblank(outcome, "outcome"),
        "cleanup": "not_attempted_unowned",
    }
    if entry_id is not None:
        structured["entry_id"] = require_clean_nonblank(entry_id, "entry_id")
    return ToolResult(
        content="Failed to store knowledge safely. No unowned cleanup was attempted.",
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
    value = require_finite(float(value), key)
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


def _knowledge_hit_payload(
    hit: KnowledgeHit,
    *,
    preview_bytes: int,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    entry = hit.entry
    text_preview, preview_truncated = _bounded_preview(
        hit.text_preview,
        preview_bytes,
        redactor=redactor,
        source_complete=hit.text_preview_complete,
    )
    return {
        "entry_id": entry.id,
        "revision": entry.revision,
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
    """Project only operational entry identity/status into the model-visible result."""

    return {
        "entry_id": entry.id,
        "revision": entry.revision,
        "status": entry.status.value,
    }


def _knowledge_list_item_payload(
    item: KnowledgeListItem,
    *,
    preview_bytes: int,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    entry = item.entry
    text_preview, preview_truncated = _bounded_preview(
        item.text_preview,
        preview_bytes,
        redactor=redactor,
        source_complete=item.text_preview_complete,
    )
    return {
        "entry_id": entry.id,
        "revision": entry.revision,
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
        "entry_revision": chunk.entry_revision,
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


def _format_search_hits(
    hits: list[KnowledgeHit],
    *,
    preview_bytes: int,
    redactor: SecretRedactor,
) -> str:
    lines = ["Knowledge results:"]
    for index, hit in enumerate(hits, start=1):
        entry = hit.entry
        title = f" title={entry.title!r}" if entry.title else ""
        chunk = ""
        if hit.chunk is not None:
            chunk = f" chunk_index={hit.chunk.chunk_index}"
        score = f" score={hit.score:.4f}" if hit.score is not None else ""
        lines.append(f"{index}. entry_id={entry.id!r} kind={entry.kind!r}{title}{chunk}{score}")
        text_preview, preview_truncated = _bounded_preview(
            hit.text_preview,
            preview_bytes,
            redactor=redactor,
            source_complete=hit.text_preview_complete,
        )
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
    redactor: SecretRedactor,
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
                redactor=redactor,
                source_complete=item.text_preview_complete,
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


def _bounded_preview(
    text: str | None,
    max_bytes: int,
    *,
    redactor: SecretRedactor,
    source_complete: bool,
) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    preview, truncated = redactor.redact_utf8_head(
        text.encode("utf-8"),
        max_bytes=max_bytes,
        source_complete=source_complete,
    )
    return (preview.rstrip() if truncated else preview), truncated
