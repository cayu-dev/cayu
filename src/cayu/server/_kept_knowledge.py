"""Scoped kept-entry discovery and native lifecycle retirement, separate from review."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import StringConstraints

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
) -> None:
    auth_default = cast("AuthContext | None", optional_auth_context)

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
        auth_context: AuthContext | None = auth_default,
    ):
        owner = workflow()
        store = cast("KnowledgeStore", owner.store)
        limited = gate.applies(auth_context)
        binding = {
            "subject": None if auth_context is None else auth_context.subject,
            "tenant": None if auth_context is None else auth_context.tenant,
            "namespace": owner.namespace,
            "labels": owner.labels,
            "scope": owner.access_scope.model_dump(mode="json"),
        }
        position = None if cursor is None else seal.open("kept-knowledge", cursor, binding)
        try:
            listed = await store.list_entries(
                KnowledgeListQuery(
                    namespace=owner.namespace,
                    labels=owner.labels,
                    statuses=[KnowledgeStatus.ACTIVE],
                    cursor=position,
                    limit=limit,
                    max_bytes=max_bytes,
                ),
                access_scope=owner.access_scope,
            )
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
                specialist = (
                    entry.created_by
                    if entry.created_by_type is KnowledgeActorType.MODEL
                    and entry.created_by in app.list_agents()
                    else None
                )
                source_session = None
                if entry.source_type == "tool" and entry.source_id and specialist and not limited:
                    if entry.source_id not in source_identities:
                        try:
                            source_identities[
                                entry.source_id
                            ] = await app.session_store.inspect_identity(entry.source_id)
                        except KeyError:
                            source_identities[entry.source_id] = None
                    source = source_identities[entry.source_id]
                    if source is not None and source.agent_name == specialist:
                        source_session = app.project_session_id_for_exposure(source.id)
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
