"""`POST /api/knowledge`: an operator adds one knowledge entry, kept or pending review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from cayu.knowledge.activation_contracts import (
    MAX_KNOWLEDGE_ACTIVATION_CHUNKS,
    KnowledgeActivationConflict,
    KnowledgeActivationSource,
    KnowledgeGovernanceMode,
    prepare_knowledge_activation_request,
)
from cayu.knowledge.base import KnowledgeStore
from cayu.knowledge.publication_contracts import KnowledgePublicationConflict
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_KIND,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
)
from cayu.server._inbox_access import InboxAccessGate
from cayu.server._knowledge_preview import knowledge_preview
from cayu.server.auth import AuthContext
from cayu.server.contracts import AddKnowledgeRequest, AddKnowledgeResponse
from cayu.storage.knowledge_indexer import (
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    knowledge_source_hash,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.storage.knowledge_review import KnowledgeReviewWorkflow


def register_knowledge_addition(
    router: APIRouter,
    *,
    app: CayuApp,
    protected: list[Any],
    workflow: Callable[[], KnowledgeReviewWorkflow],
    gate: InboxAccessGate,
    optional_auth_context: Any,
    serialize_entry: Callable[[KnowledgeEntry], dict[str, Any]],
) -> None:
    auth_default = cast("AuthContext | None", optional_auth_context)

    @router.post(
        "/knowledge",
        response_model=AddKnowledgeResponse,
        dependencies=protected,
    )
    async def add_knowledge(
        body: AddKnowledgeRequest,
        auth_context: AuthContext | None = auth_default,
    ):
        """Add one knowledge entry, kept (``active``) or as a proposal (``pending``).

        The entry's id comes from its namespace, kind, labels and text, so sending the same
        request again returns the entry it added (``created: false``) instead of a copy.
        """

        from cayu.server.routes import _knowledge_reviewer_attribution

        owner = workflow()
        store = cast("KnowledgeStore", owner.store)
        namespace = _namespace(owner, body.namespace)
        labels = dict(body.labels)
        for key, value in owner.labels.items():
            if labels.setdefault(key, value) != value:
                raise HTTPException(
                    422, f"Knowledge here has label {key}={value!r}; the entry can't name another."
                )
        kind = DEFAULT_KNOWLEDGE_KIND if body.kind is None else body.kind
        identity, _ = _knowledge_reviewer_attribution(auth_context)
        try:
            built = KnowledgeIndexer().build(
                KnowledgeIndexRequest(
                    text=body.text,
                    entry_id=_entry_id(namespace, kind, labels, body.text),
                    namespace=namespace,
                    labels=labels,
                    kind=kind,
                    status=KnowledgeStatus(body.status),
                    created_by_type=KnowledgeActorType.USER,
                    created_by=identity,
                    source_uri=body.source_uri,
                    title=body.title,
                )
            )
        except ValueError as exc:
            raise HTTPException(422, _first_line(exc)) from None
        if built.truncated or built.entry_text_truncated:
            raise HTTPException(413, "The text is too long for one knowledge entry.")
        entry = built.entry
        _require_session_labels(owner, entry)
        if entry.status is KnowledgeStatus.PENDING:
            if auth_context is None:
                # Without authentication every caller, the reviewer who would keep it included, is
                # this one identity, and keeping refuses the one who added a proposal.
                raise HTTPException(
                    422,
                    "This server has no authentication, so a proposal added here could never "
                    "be kept: `cayu knowledge keep` refuses the one who added it. Add the entry as active.",
                )
            _require_reviewable(owner, entry, built.chunks)
        if gate.applies(auth_context) and not await gate.permits(
            auth_context, gate.knowledge_scope(entry)
        ):
            raise HTTPException(403, "Knowledge access denied for these labels.")
        try:
            existing = await owner.get_review_entry(entry.id)
            created = existing is None
            if existing is None:
                try:
                    await store.create_entry(
                        entry, chunks=built.chunks, access_scope=owner.access_scope
                    )
                except KnowledgeRevisionConflict:
                    # Another request added it first; it answers below like a repeat.
                    existing = await owner.get_review_entry(entry.id)
                    if existing is None:
                        raise
                    created = False
            if existing is not None:
                if not _same_entry(existing, entry):
                    raise HTTPException(409, _occupied(existing))
                entry = existing
        except NotImplementedError:
            raise HTTPException(501, "This store does not support adding knowledge.") from None
        except PermissionError as exc:
            # The store's own reason, such as a label the scope's session_label_keys require.
            raise HTTPException(403, _first_line(exc)) from None
        except (
            KnowledgeRevisionConflict,
            KnowledgeActivationConflict,
            KnowledgePublicationConflict,
        ):
            raise HTTPException(
                409,
                f"Knowledge entry {entry.id} was used before and can't be added again; "
                "change the text to add it as a new entry.",
            ) from None
        except ValueError as exc:
            raise HTTPException(422, _first_line(exc)) from None
        return {
            **{key: app.redact_json(value) for key, value in serialize_entry(entry).items()},
            "chunk_count": built.chunk_count,
            "text_preview": knowledge_preview(entry.text, app.redact_json),
            "created": created,
        }


# Keeping uses its Idempotency-Key as the operation id (at most 256 characters) or a shorter
# generated one; the check below sizes the request with the longest.
_LONGEST_REVIEW_OPERATION_ID = "o" * 256


def _require_reviewable(
    owner: KnowledgeReviewWorkflow, entry: KnowledgeEntry, chunks: list[KnowledgeChunk]
) -> None:
    """Refuse a proposal that couldn't be kept: its activation request, which carries the
    text and every chunk, must fit the activation request's byte bound."""

    if len(chunks) > MAX_KNOWLEDGE_ACTIVATION_CHUNKS:
        raise HTTPException(413, _TOO_LARGE_TO_REVIEW)
    try:
        prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=owner.access_scope,
            operation_id=_LONGEST_REVIEW_OPERATION_ID,
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.REVIEW_APPROVAL,
            expected_revision=entry.revision,
            forbidden_authority_identities=(entry.created_by,),
        )
    except ValueError:
        raise HTTPException(413, _TOO_LARGE_TO_REVIEW) from None


_TOO_LARGE_TO_REVIEW = (
    "The text is too long to keep through review; add it as active, or split it into "
    "several entries."
)


def _require_session_labels(owner: KnowledgeReviewWorkflow, entry: KnowledgeEntry) -> None:
    """Refuse an entry without a label the scope's ``session_label_keys`` pin, as the store
    would, in words a command line can act on (the store's own names Python's ``labels=``)."""

    scope = owner.access_scope
    if not scope.session_label_keys or scope.resource_constraints:
        return
    missing = [key for key in scope.session_label_keys if key not in entry.labels]
    if not missing:
        return
    named = ", ".join(repr(key) for key in missing)
    raise HTTPException(
        422,
        f"The entry has no {named} label. This agent's knowledge is pinned to each "
        f"conversation's {named} (session_label_keys), so no conversation could read it; "
        f"add the {named} label of the one it belongs to.",
    )


def _namespace(owner: KnowledgeReviewWorkflow, requested: str | None) -> str:
    """The namespace to add to: the review namespace, the one asked for, or the scope's only one."""

    if owner.namespace is not None:
        if requested is not None and requested != owner.namespace:
            raise HTTPException(
                422,
                f"Knowledge here is in namespace {owner.namespace!r}; "
                "the requested namespace is outside it.",
            )
        return owner.namespace
    if requested is not None:
        return requested
    scope = owner.access_scope
    if scope.allow_all_namespaces:
        return DEFAULT_KNOWLEDGE_NAMESPACE
    if len(scope.allowed_namespaces) == 1:
        return scope.allowed_namespaces[0]
    raise HTTPException(
        422,
        "This application's knowledge has several namespaces; name one ("
        + ", ".join(scope.allowed_namespaces)
        + ").",
    )


def _entry_id(namespace: str, kind: str, labels: dict[str, str], text: str) -> str:
    """The entry's id, from what makes it the same entry: its place, labels and text."""

    basis = json.dumps(
        ["operator", namespace, kind, sorted(labels.items()), knowledge_source_hash(text)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"knowledge_{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:32]}"


def _same_entry(existing: KnowledgeEntry, entry: KnowledgeEntry) -> bool:
    return (
        existing.status is entry.status
        and existing.source_hash == entry.source_hash
        and existing.namespace == entry.namespace
        and existing.kind == entry.kind
        and existing.labels == entry.labels
        and existing.title == entry.title
        and existing.source_uri == entry.source_uri
    )


def _occupied(existing: KnowledgeEntry) -> str:
    if existing.status is KnowledgeStatus.ARCHIVED:
        return (
            f"Knowledge entry {existing.id} with this text and labels was archived or "
            "dropped; change the text to add it again."
        )
    return (
        f"Knowledge entry {existing.id} already has this text and labels, as a "
        f"{existing.status.value} entry with other fields; inspect it with "
        "`cayu knowledge show`."
    )


def _first_line(exc: BaseException) -> str:
    """A refusal's reason in one line: a validation error's rules, or the message's first line."""

    if isinstance(exc, ValidationError):
        return "Invalid knowledge entry: " + "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'entry'}: {error['msg']}"
            for error in exc.errors()[:5]
        )
    text = str(exc).strip()
    return text.splitlines()[0] if text else "Invalid knowledge entry."
