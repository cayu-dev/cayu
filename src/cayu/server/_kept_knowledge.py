"""Scoped kept-entry discovery and native lifecycle retirement, separate from review."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import StringConstraints, ValidationError

from cayu.knowledge._retrieval_results import _truncate_text_to_bytes as truncate_text_to_bytes
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationConflict,
    KnowledgeActivationReceipt,
    KnowledgeActivationSource,
)
from cayu.knowledge.base import KnowledgeStore
from cayu.knowledge.publication_contracts import (
    KnowledgePublicationConflict,
    copy_knowledge_publication_receipt,
)
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
    ArchivedKnowledgeDetailResponse,
    ArchivedKnowledgeListResponse,
    KeptKnowledgeDetailResponse,
    KeptKnowledgeListResponse,
    PendingKnowledgeDetailResponse,
    RetireKnowledgeRequest,
    RetireKnowledgeResponse,
)
from cayu.sessions.event_queries import EventQueryResultTooLarge
from cayu.sessions.inspection import SessionInspectionIdentity
from cayu.sessions.knowledge_usage import retained_knowledge_usage
from cayu.storage._knowledge_rejection import KnowledgeReviewMaterialTooLarge

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.storage.knowledge_review import KnowledgeReviewWorkflow


#: The reviewer version a knowledge decision made through these routes records.
KNOWLEDGE_REVIEWER_VERSION = "1"
NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# Chunks are read whole up to this many bytes, then redacted, then bounded for the caller.
_CHUNK_READ_BYTES = 1_048_576


def register_kept_knowledge(
    router: APIRouter,
    *,
    app: CayuApp,
    pending_router: APIRouter,
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

    def source_session_to_verify(
        entry: KnowledgeEntry, specialist: str | None, *, limited: bool
    ) -> str | None:
        """The session a tool entry names, when this caller may see it verified."""

        if entry.source_type != "tool" or not entry.source_id or not specialist or limited:
            return None
        return entry.source_id

    async def page_source_identities(
        entries: list[KnowledgeEntry], *, limited: bool
    ) -> dict[str, SessionInspectionIdentity | None]:
        """The source sessions a page's tool entries name, in one store read."""

        wanted = list(
            dict.fromkeys(
                session_id
                for entry in entries
                if (
                    session_id := source_session_to_verify(
                        entry, specialist_of(entry), limited=limited
                    )
                )
                is not None
            )
        )
        if not wanted:
            return {}
        found = await app.session_store.inspect_identities(wanted)
        return {session_id: found.get(session_id) for session_id in wanted}

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

        source_id = source_session_to_verify(entry, specialist, limited=limited)
        if source_id is None:
            return None
        if source_id not in identities:
            try:
                identities[source_id] = await app.session_store.inspect_identity(source_id)
            except KeyError:
                identities[source_id] = None
        source = identities[source_id]
        if source is None or source.agent_name != specialist:
            return None
        return app.project_session_id_for_exposure(source.id)

    async def page_receipts(
        store: KnowledgeStore, entries: list[KnowledgeEntry], scope: KnowledgeAccessScope
    ) -> dict[tuple[str, int], KnowledgeActivationReceipt | None] | None:
        """The page's activation receipts in one store read; None when unsupported."""

        if not entries:
            return {}
        try:
            return await store.load_entry_activation_receipts(
                [(entry.id, entry.revision) for entry in entries], access_scope=scope
            )
        except NotImplementedError:
            return None

    def attribution(
        entry: KnowledgeEntry,
        receipts: dict[tuple[str, int], KnowledgeActivationReceipt | None] | None,
    ):
        if receipts is None:
            return None, None, None, "unsupported"
        receipt = receipts.get((entry.id, entry.revision))
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
                count_total=False,
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
            page_entries = [item.entry for item in visible]
            receipts = await page_receipts(store, page_entries, owner.access_scope)
            source_identities = await page_source_identities(page_entries, limited=limited)
            for item in visible:
                entry = item.entry
                kept_by, kept_at, kept_kind, unavailable = attribution(entry, receipts)
                specialist = specialist_of(entry)
                source_session = await tool_source_session(
                    entry, specialist, limited=limited, identities=source_identities
                )
                row = serialize_item(item)
                # Store previews may already truncate across a secret boundary.
                # Project complete source first, then apply the public preview cap.
                row["text_preview"] = knowledge_preview(entry.text, app.redact_json)
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

        return await knowledge_detail(
            entry_id, max_chunks, max_bytes, auth_context, KnowledgeStatus.ACTIVE
        )

    @pending_router.get(
        "/knowledge/pending/{entry_id}",
        response_model=PendingKnowledgeDetailResponse,
    )
    async def get_pending_knowledge(
        entry_id: NonBlankString,
        max_chunks: int = Query(default=50, ge=1, le=50),
        max_bytes: int = Query(default=128_000, ge=1, le=128_000),
        auth_context: AuthContext | None = auth_default,
    ):
        return await knowledge_detail(
            entry_id, max_chunks, max_bytes, auth_context, KnowledgeStatus.PENDING
        )

    @router.get(
        "/knowledge/archived",
        response_model=ArchivedKnowledgeListResponse,
        dependencies=protected,
    )
    async def archived_knowledge(
        cursor: str | None = Query(default=None, max_length=6144),
        limit: int = Query(default=50, ge=1, le=100),
        max_bytes: int = Query(default=100_000, ge=1, le=1_048_576),
        namespace: str | None = None,
        kind: Annotated[list[str] | None, Query()] = None,
        label: Annotated[list[str] | None, Query()] = None,
        auth_context: AuthContext | None = auth_default,
    ):
        """Archived entries: kept entries archived and proposals dropped.

        Read through the review scope, so a server whose knowledge scope admits only
        active and pending entries still lists what it archived. Full access, as retiring.
        """

        owner = workflow()
        listed_namespace, listed_labels, kinds = kept_filters(owner, namespace, kind, label)
        binding: dict[str, Any] = {
            "subject": None if auth_context is None else auth_context.subject,
            "tenant": None if auth_context is None else auth_context.tenant,
            "namespace": listed_namespace,
            "labels": listed_labels,
            "kinds": kinds,
            "scope": owner.review_access_scope.model_dump(mode="json"),
        }
        position = None if cursor is None else seal.open("archived-knowledge", cursor, binding)
        try:
            listed = await owner.list_archived(
                namespace=listed_namespace,
                labels=listed_labels,
                kinds=kinds,
                cursor=position,
                limit=limit,
                max_bytes=max_bytes,
            )
        except ValidationError as exc:
            fields = sorted({str(error["loc"][0]) for error in exc.errors() if error.get("loc")})
            raise HTTPException(
                422, f"Invalid knowledge filters: {', '.join(fields) or 'query'}."
            ) from None
        except NotImplementedError:
            raise HTTPException(501, "This store does not support knowledge reads.") from None
        except PermissionError:
            raise HTTPException(403, "Knowledge access denied.") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        more = listed.truncated and (
            listed.total_entries_known is None or listed.total_entries_known > len(listed.entries)
        )
        after = None
        if more:
            if not listed.entries:
                raise HTTPException(413, "An archived entry exceeds the requested page byte bound.")
            after = encode_knowledge_list_cursor(listed.entries[-1].entry)
            if after == position:
                raise HTTPException(409, "Knowledge listing could not advance.")
        limited = gate.applies(auth_context)
        visible = [
            item
            for item in listed.entries
            if await gate.permits(auth_context, gate.knowledge_scope(item.entry))
        ]
        source_identities = await page_source_identities(
            [item.entry for item in visible], limited=limited
        )
        rows = []
        for item in visible:
            entry = item.entry
            row = serialize_item(item)
            # As the kept list: redact the whole text before the preview cuts it.
            row["text_preview"] = knowledge_preview(entry.text, app.redact_json)
            if entry.source_type == "tool":
                # As the kept list: the verified, exposed source session, or nothing.
                row["source_id"] = await tool_source_session(
                    entry, specialist_of(entry), limited=limited, identities=source_identities
                )
            row.update(await owner.archived_decision(entry))
            rows.append(app.redact_json(row))
        if len(json.dumps(rows, ensure_ascii=False).encode("utf-8")) > 1_048_576:
            raise HTTPException(
                413, "Archived entry metadata exceeds the public response byte bound."
            )
        return {
            "entries": rows,
            "next_cursor": seal.seal("archived-knowledge", after, binding)
            if after is not None
            else None,
        }

    @router.get(
        "/knowledge/archived/{entry_id}",
        response_model=ArchivedKnowledgeDetailResponse,
        dependencies=protected,
    )
    async def archived_knowledge_entry(
        entry_id: NonBlankString,
        max_chunks: int = Query(default=50, ge=1, le=50),
        max_bytes: int = Query(default=128_000, ge=1, le=128_000),
        auth_context: AuthContext | None = auth_default,
    ):
        """One archived entry with its text, chunks and the decision that archived it; 404
        for anything else. Read through the review scope, as the archived list."""

        return await knowledge_detail(
            entry_id, max_chunks, max_bytes, auth_context, KnowledgeStatus.ARCHIVED
        )

    async def knowledge_detail(
        entry_id: str,
        max_chunks: int,
        max_bytes: int,
        auth_context: AuthContext | None,
        status: KnowledgeStatus,
    ):
        owner = workflow()
        store = cast("KnowledgeStore", owner.store)
        # Archived entries are read, chunks too, through the review scope.
        scope = (
            owner.review_access_scope if status is KnowledgeStatus.ARCHIVED else owner.access_scope
        )
        if status is KnowledgeStatus.PENDING:
            try:
                entry = await owner.get_pending(entry_id)
            except NotImplementedError:
                raise HTTPException(501, "This store does not support knowledge reads.") from None
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(403, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
        else:
            try:
                entry = (
                    await owner.get_archived(entry_id)
                    if status is KnowledgeStatus.ARCHIVED
                    else await owner.get_entry(entry_id)
                )
            except NotImplementedError:
                raise HTTPException(501, "This store does not support knowledge reads.") from None
            except PermissionError:
                raise HTTPException(404, "Knowledge entry not found.") from None
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
            if entry is None or entry.status is not status:
                raise HTTPException(404, "Knowledge entry not found.")
        await gate.require_knowledge(auth_context, entry)
        try:
            # Redact whole chunks before the caller's byte bound can split a secret.
            read = await store.read_chunks(
                entry.id,
                revision=entry.revision,
                access_scope=scope,
                max_chunks=max_chunks,
                max_bytes=_CHUNK_READ_BYTES,
            )
        except NotImplementedError:
            raise HTTPException(501, "This store does not support knowledge reads.") from None
        except PermissionError:
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
            row = {
                key: app.redact_json(value)
                for key, value in serialize_chunk(chunk).items()
                if key != "text"
            }
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
        return {
            **{key: app.redact_json(value) for key, value in detail.items()},
            "content_redacted": True,
            "chunks": chunks,
            "chunk_limit": max_chunks,
            "chunk_max_bytes": max_bytes,
        }

    @router.post(
        "/knowledge/{entry_id}/retire",
        response_model=RetireKnowledgeResponse,
        dependencies=protected,
    )
    async def retire_knowledge(
        entry_id: NonBlankString,
        body: RetireKnowledgeRequest,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", min_length=1, max_length=256),
        ] = None,
        auth_context: AuthContext | None = auth_default,
    ):
        """Archive one kept entry, so knowledge search no longer finds it.

        With a ``reason`` or an ``Idempotency-Key`` the retirement records its reviewer and
        reason on the archived revision, and a retry under the same key replays it.
        """

        from cayu.server.routes import _knowledge_reviewer_attribution

        reviewed = body.reason is not None or idempotency_key is not None
        if not reviewed and body.expected_revision is None:
            raise HTTPException(
                422, "Name expected_revision, or give a reason or an Idempotency-Key."
            )
        owner = workflow()
        store = cast("KnowledgeStore", owner.store)
        operation_id = idempotency_key or f"server-knowledge-retire-{uuid4().hex}"
        try:
            entry = (
                await reviewed_decision_entry(owner, entry_id, operation_id)
                if reviewed
                else await owner.get_entry(entry_id)
            )
            if entry is None:
                raise HTTPException(404, "Knowledge entry not found.")
            await gate.require_knowledge(auth_context, entry)
            if reviewed:
                revision = await reviewed_decision_revision(
                    owner,
                    entry,
                    operation_id,
                    expected_revision=body.expected_revision,
                    from_status=KnowledgeStatus.ACTIVE,
                )
                identity, annotations = _knowledge_reviewer_attribution(auth_context)
                archived = await owner.archive_reviewed(
                    entry_id,
                    expected_revision=revision,
                    operation_id=operation_id,
                    reviewer_identity=identity,
                    reviewer_version=KNOWLEDGE_REVIEWER_VERSION,
                    reason=app.redact_json(body.reason),
                    annotations=annotations,
                )
                return {
                    "entry_id": entry_id,
                    "revision": archived.entry.revision,
                    "status": "archived",
                    "reason": app.redact_json(archived.decision.reason),
                }
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
        except KnowledgeReviewMaterialTooLarge:
            raise HTTPException(
                413,
                "This entry is too large to retire with a recorded reason; retire it by "
                "expected_revision alone.",
            ) from None
        except KeyError:
            raise HTTPException(404, "Knowledge entry not found.") from None
        except PermissionError:
            raise HTTPException(403, "Knowledge access denied.") from None
        except (
            KnowledgeRevisionConflict,
            KnowledgeActivationConflict,
            KnowledgePublicationConflict,
            ValueError,
        ):
            raise HTTPException(
                409, "Knowledge retirement conflicts with its current revision or scope."
            ) from None


async def reviewed_decision_entry(
    owner: KnowledgeReviewWorkflow, entry_id: str, operation_id: str
) -> KnowledgeEntry | None:
    """The entry a reviewed drop or retirement under ``operation_id`` decides.

    The current entry, as the workflow's access scope reads it. Once ``operation_id`` has
    archived it, the archived entry, read through the review scope, so a retry replays
    the decision even when the access scope can't read archived entries; an archived entry
    no decision of this operation wrote stays as hidden as before.
    """

    entry = await owner.get_entry(entry_id)
    if entry is not None:
        return entry
    try:
        archived = await owner.get_archived(entry_id)
    except PermissionError:
        # Outside the review namespace or labels: as hidden as it was.
        return None
    if archived is None:
        return None
    raw = await owner.load_review_publication(operation_id)
    recorded = raw is not None and copy_knowledge_publication_receipt(raw).entry_id == entry_id
    return archived if recorded else None


async def reviewed_decision_revision(
    owner: KnowledgeReviewWorkflow,
    entry: KnowledgeEntry,
    operation_id: str,
    *,
    expected_revision: int | None,
    from_status: KnowledgeStatus,
) -> int:
    """The revision a reviewed drop or retirement under ``operation_id`` archives.

    The revision an earlier attempt under the same operation recorded, so a retry replays
    it instead of conflicting with the archived revision it wrote; otherwise the entry's
    current revision, which must be in ``from_status``. ``expected_revision`` must match
    it when given. Raises ``KnowledgePublicationConflict`` otherwise.
    """

    raw = await owner.load_review_publication(operation_id)
    if raw is not None:
        receipt = copy_knowledge_publication_receipt(raw)
        if receipt.entry_id != entry.id or receipt.expected_revision is None:
            raise KnowledgePublicationConflict("operation_mismatch")
        revision = receipt.expected_revision
    elif entry.status is not from_status:
        raise KnowledgePublicationConflict("entry_unavailable")
    else:
        revision = entry.revision
    if expected_revision is not None and expected_revision != revision:
        raise KnowledgePublicationConflict("revision_mismatch")
    return revision
