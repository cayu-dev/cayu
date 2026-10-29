"""Scoped kept-entry discovery and native lifecycle retirement, separate from review."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import StringConstraints, ValidationError

from cayu.knowledge._retrieval_results import _truncate_text_to_bytes as truncate_text_to_bytes
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationConflict,
    KnowledgeActivationSource,
)
from cayu.knowledge.base import KnowledgeStore
from cayu.knowledge.records import (
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
)
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeListQuery, encode_knowledge_list_cursor
from cayu.server._inbox_access import InboxAccessGate
from cayu.server._inbox_cursor import InboxCursorSeal
from cayu.server._knowledge_preview import knowledge_preview
from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    KeptKnowledgeDetailResponse,
    KeptKnowledgeListResponse,
    RetireKnowledgeRequest,
    RetireKnowledgeResponse,
)
from cayu.sessions.event_queries import EventQueryResultTooLarge
from cayu.sessions.inspection import SessionInspectionIdentity
from cayu.sessions.knowledge_usage import retained_knowledge_usage

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.storage.knowledge_review import KnowledgeReviewWorkflow


NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# Chunks are read whole up to this many bytes, then redacted, then bounded for the caller.
_CHUNK_READ_BYTES = 1_048_576


def register_kept_knowledge(
    router: APIRouter,
    *,
    app: CayuApp,
    protected: list[Any],
    workflow: Callable[[], KnowledgeReviewWorkflow],
    gate: InboxAccessGate,
    seal: InboxCursorSeal,
    optional_auth_context: Any,
    serialize_item: Callable[[Any], dict[str, Any]],
    serialize_detail: Callable[[KnowledgeEntry], dict[str, Any]],
    serialize_chunk: Callable[[Any], dict[str, Any]],
    parse_labels: Callable[[list[str] | None], dict[str, str]],
    parse_strings: Callable[[list[str] | None, str], list[str]],
) -> None:
    auth_default = cast("AuthContext | None", optional_auth_context)

    def kept_filters(
        owner: KnowledgeReviewWorkflow,
        namespace: str | None,
        kind: list[str] | None,
        label: list[str] | None,
    ) -> tuple[str | None, dict[str, str], list[str] | None]:
        """The namespace, labels and kinds to list, within the review scope."""

        requested = parse_strings(None if namespace is None else [namespace], "namespace")
        if requested and owner.namespace is not None and requested[0] != owner.namespace:
            raise HTTPException(
                422,
                f"Kept knowledge here is in namespace {owner.namespace!r}; "
                "the requested namespace is outside it.",
            )
        labels = dict(owner.labels)
        for key, value in parse_labels(label).items():
            if labels.get(key, value) != value:
                raise HTTPException(
                    422, f"Kept knowledge here has label {key}={labels[key]!r}; filter by that."
                )
            labels[key] = value
        kinds = parse_strings(kind, "kind") if kind is not None else None
        return (requested[0] if requested else owner.namespace), labels, kinds

    def specialist_of(entry: KnowledgeEntry) -> str | None:
        return (
            entry.created_by
            if entry.created_by_type is KnowledgeActorType.MODEL
            and entry.created_by in app.list_agents()
            else None
        )

    async def tool_source_session(
        entry: KnowledgeEntry,
        specialist: str | None,
        *,
        limited: bool,
        identities: dict[str, SessionInspectionIdentity | None],
    ) -> str | None:
        """The exposed id of the session that produced a tool entry, when it is verified.

        Only for a caller the inbox access rule doesn't limit, and only when the session
        belongs to the specialist that created the entry.
        """

        if entry.source_type != "tool" or not entry.source_id or not specialist or limited:
            return None
        if entry.source_id not in identities:
            try:
                identities[entry.source_id] = await app.session_store.inspect_identity(
                    entry.source_id
                )
            except KeyError:
                identities[entry.source_id] = None
        source = identities[entry.source_id]
        if source is None or source.agent_name != specialist:
            return None
        return app.project_session_id_for_exposure(source.id)

    async def attribution(
        store: KnowledgeStore, entry: KnowledgeEntry, scope: KnowledgeAccessScope
    ):
        try:
            receipt = await store.load_entry_activation_receipt(
                entry.id, revision=entry.revision, access_scope=scope
            )
        except NotImplementedError:
            return None, None, None, "unsupported"
        if receipt is None:
            return None, None, None, "not_recorded"
        if receipt.entry_id != entry.id or receipt.entry_revision != entry.revision:
            raise KnowledgeActivationConflict("operation_mismatch")
        kind = (
            "reviewer"
            if receipt.authority.request.source is KnowledgeActivationSource.REVIEW_APPROVAL
            else "policy"
        )
        return receipt.authority.decision.policy_identity, receipt.committed_at, kind, None

    @router.get("/knowledge", response_model=KeptKnowledgeListResponse, dependencies=protected)
    async def kept_knowledge(
        cursor: str | None = Query(default=None, max_length=6144),
        limit: int = Query(default=50, ge=1, le=100),
        max_bytes: int = Query(default=100_000, ge=1, le=1_048_576),
        namespace: str | None = None,
        kind: Annotated[list[str] | None, Query()] = None,
        label: Annotated[list[str] | None, Query()] = None,
        auth_context: AuthContext | None = auth_default,
    ):
        owner = workflow()
        store = cast("KnowledgeStore", owner.store)
        limited = gate.applies(auth_context)
        listed_namespace, listed_labels, kinds = kept_filters(owner, namespace, kind, label)
        binding: dict[str, Any] = {
            "subject": None if auth_context is None else auth_context.subject,
            "tenant": None if auth_context is None else auth_context.tenant,
            "namespace": listed_namespace,
            "labels": listed_labels,
            "scope": owner.access_scope.model_dump(mode="json"),
        }
        if kinds is not None:
            binding["kinds"] = kinds
        position = None if cursor is None else seal.open("kept-knowledge", cursor, binding)
        try:
            query = KnowledgeListQuery(
                namespace=listed_namespace,
                labels=listed_labels,
                kinds=kinds,
                statuses=[KnowledgeStatus.ACTIVE],
                cursor=position,
                limit=limit,
                max_bytes=max_bytes,
            )
        except ValidationError as exc:
            fields = sorted({str(error["loc"][0]) for error in exc.errors() if error.get("loc")})
            raise HTTPException(
                422, f"Invalid knowledge filters: {', '.join(fields) or 'query'}."
            ) from None
        try:
            listed = await store.list_entries(query, access_scope=owner.access_scope)
            more = listed.truncated and (
                listed.total_entries_known is None
                or listed.total_entries_known > len(listed.entries)
            )
            after = None
            if more:
                if not listed.entries:
                    raise HTTPException(413, "A kept entry exceeds the requested page byte bound.")
                after = encode_knowledge_list_cursor(listed.entries[-1].entry)
                if after == position:
                    raise HTTPException(409, "Knowledge listing could not advance.")
            visible = [
                item
                for item in listed.entries
                if await gate.permits(auth_context, gate.knowledge_scope(item.entry))
            ]
            reason = "access_limited" if limited else None
            totals, through = {}, None
            if not limited and visible:
                try:
                    totals, through = await retained_knowledge_usage(
                        app.session_store,
                        {(item.entry.namespace, item.entry.id) for item in visible},
                    )
                except NotImplementedError:
                    reason = "unsupported"
                except (EventQueryResultTooLarge, ValueError):
                    reason = "source_incomplete"
            rows = []
            source_identities: dict[str, SessionInspectionIdentity | None] = {}
            for item in visible:
                entry = item.entry
                kept_by, kept_at, kept_kind, unavailable = await attribution(
                    store, entry, owner.access_scope
                )
                specialist = specialist_of(entry)
                source_session = await tool_source_session(
                    entry, specialist, limited=limited, identities=source_identities
                )
                row = serialize_item(item)
                # Store previews may already truncate across a secret boundary.
                # Project complete source first, then apply the public preview cap.
                row["text_preview"] = knowledge_preview(entry.title or entry.text, app.redact_json)
                if entry.source_type == "tool":
                    row["source_id"] = source_session
                count, last = totals.get((entry.namespace, entry.id), (None, None))
                row.update(
                    specialist=specialist,
                    source_session_id=source_session,
                    kept_by=kept_by,
                    kept_at=kept_at.isoformat() if kept_at else None,
                    kept_by_kind=kept_kind,
                    attribution_unavailable_reason=unavailable,
                    lookups=str(count) if count is not None else None,
                    last_used_at=last.isoformat() if last else None,
                )
                rows.append(app.redact_json(row))
            if len(json.dumps(rows, ensure_ascii=False).encode("utf-8")) > 1_048_576:
                raise HTTPException(
                    413, "Kept entry metadata exceeds the public response byte bound."
                )
            return {
                "entries": rows,
                "next_cursor": seal.seal("kept-knowledge", after, binding)
                if after is not None
                else None,
                "total_count": None,
                "usage_through_sequence": through,
                "usage_unavailable_reason": reason,
            }
        except NotImplementedError:
            raise HTTPException(501, "This store does not support kept knowledge reads.") from None
        except PermissionError:
            raise HTTPException(403, "Knowledge access denied.") from None
        except KnowledgeActivationConflict:
            raise HTTPException(409, "Knowledge attribution changed or is inconsistent.") from None

    @router.get(
        "/knowledge/kept/{entry_id}",
        response_model=KeptKnowledgeDetailResponse,
        dependencies=protected,
    )
    async def kept_knowledge_entry(
        entry_id: NonBlankString,
        max_chunks: int = Query(default=50, ge=1, le=50),
        max_bytes: int = Query(default=128_000, ge=1, le=128_000),
        auth_context: AuthContext | None = auth_default,
    ):
        """One kept (active) entry with its text and chunks; 404 for anything else."""

        owner = workflow()
        store = cast("KnowledgeStore", owner.store)
        try:
            entry = await owner.get_entry(entry_id)
            if entry is None or entry.status is not KnowledgeStatus.ACTIVE:
                raise HTTPException(404, "Knowledge entry not found.")
            await gate.require_knowledge(auth_context, entry)
            # Read whole chunks and redact them before applying the caller's byte bound,
            # so a truncation can't split a secret the redactor would have matched.
            read = await store.read_chunks(
                entry.id,
                revision=entry.revision,
                access_scope=owner.access_scope,
                max_chunks=max_chunks,
                max_bytes=_CHUNK_READ_BYTES,
            )
        except NotImplementedError:
            raise HTTPException(501, "This store does not support kept knowledge reads.") from None
        except PermissionError:
            # Outside the review scope reads the same as absent.
            raise HTTPException(404, "Knowledge entry not found.") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        # Stores cut UTF-8 at a character boundary, so a cut chunk can end up to three bytes
        # short of the bound; anything that close may have been cut.
        if sum(len(chunk.text.encode("utf-8")) for chunk in read) > _CHUNK_READ_BYTES - 4:
            # The store may have cut the last chunk before redaction; leave it out.
            read = read[:-1]
        # Redact each whole chunk, then apply the caller's bounds to the redacted text.
        chunks: list[dict[str, Any]] = []
        remaining = max_bytes
        for chunk in read[:max_chunks]:
            if remaining <= 0:
                break
            row = serialize_chunk(chunk)
            text = app.redact_json(chunk.text)
            if text != chunk.text:
                row["content_hash"] = None
            if len(text.encode("utf-8")) > remaining:
                text = truncate_text_to_bytes(text, remaining)
                row["content_hash"] = None
                remaining = 0
                if not text.strip():
                    break
            else:
                remaining -= len(text.encode("utf-8"))
            row["text"] = text
            chunks.append(row)
        detail = serialize_detail(entry)
        if entry.source_type == "tool":
            # Like the list: the verified, exposed source session, or nothing.
            detail["source_id"] = await tool_source_session(
                entry,
                specialist_of(entry),
                limited=gate.applies(auth_context),
                identities={},
            )
        return app.redact_json(
            {
                **detail,
                "chunks": chunks,
                "chunk_limit": max_chunks,
                "chunk_max_bytes": max_bytes,
            }
        )

    @router.post(
        "/knowledge/{entry_id}/retire",
        response_model=RetireKnowledgeResponse,
        dependencies=protected,
    )
    async def retire_knowledge(
        entry_id: NonBlankString,
        body: RetireKnowledgeRequest,
        auth_context: AuthContext | None = auth_default,
    ):
        owner = workflow()
        store = cast("KnowledgeStore", owner.store)
        try:
            entry = await owner.get_entry(entry_id)
            if entry is None:
                raise HTTPException(404, "Knowledge entry not found.")
            await gate.require_knowledge(auth_context, entry)
            # Preserve hidden-item responses before binding the authorized revision.
            if entry.revision != body.expected_revision:
                raise HTTPException(
                    409, "Knowledge retirement conflicts with its current revision."
                )
            retired = await store.transition_entry_status(
                entry_id,
                expected_revision=entry.revision,
                access_scope=owner.access_scope,
                from_status=KnowledgeStatus.ACTIVE,
                to_status=KnowledgeStatus.ARCHIVED,
                expected_namespace=owner.namespace,
                expected_labels=owner.labels,
            )
            return {"entry_id": entry_id, "revision": retired.revision, "status": "archived"}
        except NotImplementedError:
            raise HTTPException(501, "This store does not support knowledge retirement.") from None
        except KeyError:
            raise HTTPException(404, "Knowledge entry not found.") from None
        except PermissionError:
            raise HTTPException(403, "Knowledge access denied.") from None
        except (KnowledgeRevisionConflict, KnowledgeActivationConflict, ValueError):
            raise HTTPException(
                409, "Knowledge retirement conflicts with its current revision or scope."
            ) from None
