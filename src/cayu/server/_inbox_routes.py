"""HTTP routes for the agent's inbox: list, read and settle what waits on a person.

An item id is a locator, never authority: it names the session (by its public
alias), the source event of the pause and its human-attention identity, or a
knowledge entry and revision. Every read re-derives the exact pending action
or proposal from durable state, and every settle goes through the Runtime's
exact-action APIs with the verified caller as the decider.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from fastapi import APIRouter, HTTPException, Path, Query, Response
from pydantic import ValidationError

from cayu._event_schema import private_event_linkage_value
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.approvals.review import HumanReviewDenied
from cayu.approvals.revisions import approval_revision_metadata
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import (
    USER_INPUT_DECISION_REASON_KEY,
    UserInputResponse,
    user_input_lifecycle_authority_from_checkpoint,
    user_input_resolution_request_digest,
)
from cayu.events import EventType
from cayu.knowledge.activation_contracts import KnowledgeActivationConflict
from cayu.knowledge.publication_contracts import KnowledgePublicationConflict
from cayu.knowledge.records import KnowledgeRevisionConflict, KnowledgeStatus
from cayu.knowledge.search import decode_knowledge_list_cursor, encode_knowledge_list_cursor
from cayu.runtime._approval_support import approval_resolution_request_digest
from cayu.runtime._inbox_continuation import controlled_inbox_continuation
from cayu.runtime.human_attention import (
    AttentionKind,
    HumanAttentionReference,
    HumanAttentionRequest,
    manual_recovery_settlement,
    observe_human_attention,
)
from cayu.server._inbox_access import PENDING_KINDS, InboxAccessGate
from cayu.server._inbox_counts import count_inbox
from cayu.server._inbox_cursor import InboxCursorSeal
from cayu.server._knowledge_preview import knowledge_preview
from cayu.server.contracts import ApiErrorResponse
from cayu.server.inbox import (
    INBOX_DECISIONS,
    InboxConflict,
    InboxConflictCode,
    InboxConflictResponse,
    InboxCounts,
    InboxDecider,
    InboxIssue,
    InboxItem,
    InboxItemKind,
    InboxItemState,
    InboxItemStatus,
    InboxKnowledgeEntry,
    InboxPage,
    InboxSettledDecision,
    InboxSettlement,
    InboxSettleRequest,
    InboxSettleResponse,
)
from cayu.sessions._pending_approval_reader import approval_resolution_intent_from_checkpoint
from cayu.sessions._terminal_evidence import _INTERRUPTION_TYPE_USER_INPUT_REQUIRED
from cayu.sessions.checkpoints import CheckpointCompatibilityError
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.pending_action_contracts import (
    PendingActionQuery,
    PendingActionRecord,
    PendingActionResultTooLarge,
)
from cayu.sessions.pending_actions import manual_recovery_action_id, manual_recovery_identity
from cayu.sessions.queries import decode_session_cursor
from cayu.sessions.records import EventRecord, Session

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.knowledge.records import KnowledgeEntry
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.server.auth import AuthContext
    from cayu.sessions.store import SessionStore
    from cayu.storage.knowledge_review import KnowledgeReviewWorkflow

_ITEM_ID_PREFIX = "inbox_"
_CURSOR_PREFIX = "inboxcursor_"
_LOCATOR_VERSION = 1
_SEALED = "sealed"
_ITEM_ID_PATTERN = r"^inbox_[A-Za-z0-9_-]{1,8192}$"
_KIND_CODES: dict[InboxItemKind, str] = {
    "approval": "a",
    "question": "q",
    "recovery": "r",
    "knowledge": "k",
}
_KINDS_BY_CODE = {code: kind for kind, code in _KIND_CODES.items()}
_ATTENTION_KINDS: dict[InboxItemKind, AttentionKind] = {
    "approval": "tool_approval",
    "question": "user_input",
    "recovery": "manual_recovery",
}
_ACTION_LINKAGE = {"approval": "approval_id", "question": "input_id"}
_RESOLVED_DECISIONS: dict[str, InboxSettledDecision] = {
    "approval": "approve",
    "question": "answer",
    "recovery": "recover",
}
# A pause is resolved once, so the resumption that claimed it is the first
# matching one after the pause; a bounded scan finds it without reading history.
_CLAIM_SCAN_PAGE = 50
_CLAIM_SCAN_PAGES = 20
_PENDING_SCAN_LIMIT = 200
_KNOWLEDGE_PAGE_MAX_BYTES = 262_144
_KNOWLEDGE_SCAN_PAGES = 10
_KNOWLEDGE_REVIEWER_VERSION = "1"
_NO_STORE = {"Cache-Control": "private, no-store"}
_NOT_FOUND = "Inbox item not found."

ItemIdPath = Annotated[str, Path(pattern=_ITEM_ID_PATTERN)]


@dataclass(frozen=True, slots=True)
class _SessionLocator:
    kind: InboxItemKind
    public_session_id: str
    sequence: int
    attention_id: str


@dataclass(frozen=True, slots=True)
class _KnowledgeLocator:
    entry_id: str
    revision: int


@dataclass(frozen=True, slots=True)
class _SessionItem:
    """One pause re-derived from durable state, with its exact attention reference."""

    kind: InboxItemKind
    session: Session
    reference: HumanAttentionReference
    pending: PendingActionRecord | None


def _encode(prefix: str, document: list[Any]) -> str:
    raw = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return prefix + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode(prefix: str, value: str) -> list[Any] | None:
    if not value.startswith(prefix):
        return None
    encoded = value.removeprefix(prefix)
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        document = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if type(document) is not list or not document or document[0] != _LOCATOR_VERSION:
        return None
    return document


def _session_item_id(locator: _SessionLocator) -> str:
    return _encode(
        _ITEM_ID_PREFIX,
        [
            _LOCATOR_VERSION,
            _KIND_CODES[locator.kind],
            locator.public_session_id,
            locator.sequence,
            locator.attention_id,
        ],
    )


def _knowledge_item_id(locator: _KnowledgeLocator) -> str:
    return _encode(
        _ITEM_ID_PREFIX,
        [_LOCATOR_VERSION, _KIND_CODES["knowledge"], locator.entry_id, locator.revision],
    )


def _decode_item_id(value: str) -> _SessionLocator | _KnowledgeLocator | None:
    document = _decode(_ITEM_ID_PREFIX, value)
    if document is None or len(document) < 2 or type(document[1]) is not str:
        return None
    kind = _KINDS_BY_CODE.get(document[1])
    if kind == "knowledge" and len(document) == 4:
        entry_id, revision = document[2], document[3]
        if type(entry_id) is str and entry_id and type(revision) is int and revision >= 1:
            return _KnowledgeLocator(entry_id, revision)
        return None
    if kind is None or kind == "knowledge" or len(document) != 5:
        return None
    public_session_id, sequence, attention_id = document[2], document[3], document[4]
    if (
        type(public_session_id) is str
        and public_session_id
        and type(sequence) is int
        and sequence >= 1
        and type(attention_id) is str
    ):
        return _SessionLocator(kind, public_session_id, sequence, attention_id)
    return None


_Phase = Literal["actions", "knowledge"]


def _encode_cursor(phase: _Phase, after: str | None, *, sealed: bool = False) -> str:
    """A plain position, or with ``sealed`` one only this server's key can open."""

    return _encode(
        _CURSOR_PREFIX, [_LOCATOR_VERSION, phase, after, *((_SEALED,) if sealed else ())]
    )


def _decode_cursor(value: str) -> tuple[_Phase, str | None, bool]:
    """The phase, its position and whether that position is sealed."""

    document = _decode(_CURSOR_PREFIX, value)
    if document is not None and len(document) == 4 and document[3] == _SEALED:
        phase, after = document[1], document[2]
        # A decoded document can hold any JSON value, so check the type before
        # set membership: a list phase is unhashable.
        if type(phase) is str and phase in {"actions", "knowledge"} and type(after) is str:
            return cast("_Phase", phase), after, True
    elif document is not None and len(document) == 3:
        phase, after = document[1], document[2]
        try:
            if phase == "knowledge" and after is None:
                return "knowledge", None, False
            if phase == "knowledge" and type(after) is str:
                decode_knowledge_list_cursor(after)
                return "knowledge", after, False
            if phase == "actions" and type(after) is str:
                decode_session_cursor(after)
                return "actions", after, False
        except ValueError:
            pass
    raise HTTPException(
        status_code=422,
        detail="Invalid or expired inbox cursor. Restart the inbox listing.",
        headers=_NO_STORE,
    )


def _filter_fingerprint(filters: list[str | None]) -> str:
    # Keep the envelope bounded even for long valid filter values.
    return hashlib.sha256(json.dumps(filters, ensure_ascii=False).encode("utf-8")).hexdigest()


def _filtered_cursor(cursor: str | None, filters: list[str | None]) -> str | None:
    if cursor is None or not any(filters):
        return cursor
    document = _decode(_CURSOR_PREFIX, cursor)
    assert document is not None
    return _encode(_CURSOR_PREFIX, [*document, {"filters": _filter_fingerprint(filters)}])


def _unfilter_cursor(cursor: str, filters: list[str | None]) -> str:
    document = _decode(_CURSOR_PREFIX, cursor)
    if document is not None:
        bound = document[-1] if type(document[-1]) is dict else None
        if bound == {"filters": _filter_fingerprint(filters)}:
            return _encode(_CURSOR_PREFIX, document[:-1])
        if bound is None and not any(filters):
            return cursor
    raise HTTPException(
        status_code=422,
        detail="Invalid or expired inbox cursor. Restart the inbox listing.",
        headers=_NO_STORE,
    )


def _knowledge_operation_id(locator: _KnowledgeLocator, decision: str = "keep") -> str:
    """One reviewed approval per exact proposal revision, so a retry replays it."""

    digest = hashlib.sha256(
        json.dumps(
            [f"cayu.inbox.knowledge-{decision}.v1", locator.entry_id, locator.revision]
        ).encode("utf-8")
    ).hexdigest()
    return f"cayu-inbox-{decision}-{digest}"


def _decider(value: object) -> InboxDecider | None:
    if not isinstance(value, dict):
        return None
    try:
        actor = ResolutionActor.model_validate(value)
    except ValidationError:
        return None
    return InboxDecider(subject=actor.subject, tenant=actor.tenant, source=actor.source)


def _knowledge_decider(annotations: dict[str, Any], identity: str) -> InboxDecider:
    """Recover the reviewer that the protected HTTP attribution recorded for a keep."""

    subject = identity
    if annotations.get("subject_encoding") == "base64-utf8":
        subject = base64.b64decode(str(annotations["subject_b64"])).decode("utf-8")
    tenant = annotations.get("tenant")
    if annotations.get("tenant_encoding") == "base64-utf8":
        tenant = base64.b64decode(str(annotations["tenant_b64"])).decode("utf-8")
    return InboxDecider(
        subject=subject,
        tenant=tenant if type(tenant) is str else None,
        source=(
            ResolutionActorSource.HTTP_AUTH
            if annotations.get("identity_source") == "http_auth"
            else None
        ),
    )


def _inbox_actor(auth_context: AuthContext | None) -> ResolutionActor | None:
    """The verified caller without claims, so a refreshed token repeats the same request."""

    if auth_context is None:
        return None
    try:
        return ResolutionActor(
            subject=auth_context.subject,
            tenant=auth_context.tenant,
            source=ResolutionActorSource.HTTP_AUTH,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _conflict(code: InboxConflictCode, message: str, item: InboxItemState | None) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=InboxConflict(code=code, message=message, item=item).model_dump(mode="json"),
        headers=_NO_STORE,
    )


def _state_conflict(state: InboxItemState) -> HTTPException:
    if state.settlement is not None and state.settlement.decision == "expire":
        return _conflict("expired", "The approval expired before anyone decided it.", state)
    if state.settlement is not None:
        return _conflict("already_settled", "This item was already settled.", state)
    if state.status == "superseded":
        return _conflict("superseded", "This item was replaced before it was settled.", state)
    return _conflict(
        "unavailable", "The item's current state can't be read right now. Try again.", state
    )


def _knowledge_item(entry: KnowledgeEntry, cayu_app: Any) -> InboxItem:
    preview = knowledge_preview(entry.text, cayu_app.redact_json)
    return InboxItem(
        id=_knowledge_item_id(_KnowledgeLocator(entry.id, entry.revision)),
        kind="knowledge",
        created_at=entry.updated_at,
        title=cayu_app.redact_json(entry.title) or "Knowledge proposal",
        detail=preview or None,
        knowledge=InboxKnowledgeEntry(entry_id=entry.id, revision=entry.revision, kind=entry.kind),
        decisions=list(INBOX_DECISIONS["knowledge"]),
    )


def _inbox_item_matches_query(item: InboxItem, q: str | None) -> bool:
    """Match only the redacted public projection, never quarantined source fields."""
    if q is None:
        return True
    values = [
        item.kind,
        item.session_id,
        item.agent_name,
        item.title,
        item.detail,
        item.tool_name,
        item.question,
        *item.options,
    ]
    if item.knowledge is not None:
        values.extend((item.knowledge.entry_id, item.knowledge.kind))
    folded_query = q.casefold()
    return any(value is not None and folded_query in value.casefold() for value in values)


def _listed_kind(action: PendingActionRecord) -> InboxItemKind | None:
    # A delegated parent is navigation to its child, which is listed itself, and a
    # legacy action without an incarnation has no exact identity to settle.
    kind = PENDING_KINDS.get(action.kind)
    return None if kind is None or action.attention_id is None else kind


def _event_reference(
    kind: InboxItemKind, session: Session, record: EventRecord, attention_id: str
) -> HumanAttentionReference | None:
    """The exact reference of a pause that is no longer pending, from its source event.

    A recovery item is read with the recovery-specific fields, as pending-action
    reads project it: a sibling call that crashed in an ``ask_user`` round names
    its own call, not the question's.
    """

    assert session.instance_id is not None
    event = record.event
    if kind == "recovery":
        identity = manual_recovery_identity(event.payload)
        if identity is None:
            return None
        round_id, tool_call_id = identity.round_id, identity.tool_call_id
        action_id = manual_recovery_action_id(session.id, record, identity)
    else:
        action_id = private_event_linkage_value(event, field_name=_ACTION_LINKAGE[kind])
        round_id = private_event_linkage_value(event, field_name="tool_round_id")
        tool_call_id = private_event_linkage_value(event, field_name="tool_call_id")
        if action_id is None:
            return None
    try:
        return HumanAttentionReference(
            attention_id=attention_id,
            session_id=session.id,
            session_instance_id=session.instance_id,
            kind=_ATTENTION_KINDS[kind],
            action_id=action_id,
            round_id=round_id,
            tool_call_id=tool_call_id,
            source_sequence=record.sequence,
        )
    except ValidationError:
        # The locator names another incarnation of the session or another action.
        return None


def _proposal_content(entry: KnowledgeEntry) -> dict[str, Any]:
    """Everything a status change leaves as it was."""

    return entry.model_dump(exclude={"revision", "status", "updated_at"})


def _pending_state(item: InboxItem) -> InboxItemState:
    return InboxItemState(
        id=item.id,
        kind=item.kind,
        status="pending",
        session_id=item.session_id,
        agent_name=item.agent_name,
        item=item,
    )


def register_inbox_routes(
    router: APIRouter,
    bounded_router: APIRouter,
    *,
    cayu_app: Any,
    session_store: SessionStore,
    knowledge_review_workflow: Callable[[], KnowledgeReviewWorkflow] | None,
    continuation_loop_policies: Callable[[str], Awaitable[tuple[LoopPolicy, ...]]],
    optional_auth_context: Depends,
    inbox_gate: InboxAccessGate,
    inbox_cursor: InboxCursorSeal,
) -> None:
    """Register the inbox list, read and settle routes."""

    from cayu.server.routes import (
        _accepted_detached_mutation,
        _clean_optional_query_value,
        _knowledge_reviewer_attribution,
        _serialize_pending_action,
        _serialize_session_base,
        _serialize_session_cursor,
    )

    inbox_auth_context = cast("AuthContext | None", optional_auth_context)

    def pending_item(kind: InboxItemKind, action: PendingActionRecord) -> InboxItem:
        assert action.attention_id is not None
        projected = _serialize_pending_action(cayu_app, action)
        locator = _SessionLocator(
            kind, projected["session"]["id"], action.event.sequence, action.attention_id
        )
        return InboxItem(
            id=_session_item_id(locator),
            kind=kind,
            attention_id=action.attention_id,
            session_id=locator.public_session_id,
            agent_name=projected["session"]["agent_name"],
            created_at=action.event.event.timestamp,
            title=projected["title"],
            detail=projected["detail"],
            tool_name=projected["tool_name"],
            arguments_unavailable_reason=(
                None
                if kind not in {"approval", "recovery"}
                else "quarantined"
                if cayu_app.human_review_configured
                else "no_review_policy"
            ),
            question=projected["question"],
            options=projected["options"],
            decisions=list(INBOX_DECISIONS[kind]),
            execution_deadline=action.session.execution_deadline,
        )

    # ── Approvals, questions and recovery items ──────────────────────────────

    async def resolve_session_item(locator: _SessionLocator) -> _SessionItem | None:
        try:
            session_id = await cayu_app._resolve_public_session_id(locator.public_session_id)
        except ValueError:
            return None
        session = await session_store.load(session_id)
        if session is None or session.instance_id is None:
            return None
        try:
            page = await session_store.query_pending_actions(
                PendingActionQuery(session_id=session.id, limit=_PENDING_SCAN_LIMIT)
            )
        except (CheckpointCompatibilityError, PendingActionResultTooLarge):
            page = None
        for action in () if page is None else page.actions:
            if (
                action.attention_id == locator.attention_id
                and action.event.sequence == locator.sequence
                and PENDING_KINDS.get(action.kind) == locator.kind
            ):
                request = HumanAttentionRequest.from_pending_action(action)
                assert request is not None
                return _SessionItem(locator.kind, session, request.reference, action)
        records = await session_store.query_events(
            EventQuery(session_id=session.id, after_sequence=locator.sequence - 1, limit=1)
        )
        if not records or records[0].sequence != locator.sequence:
            return None
        reference = _event_reference(locator.kind, session, records[0], locator.attention_id)
        if reference is None:
            return None
        return _SessionItem(locator.kind, session, reference, None)

    async def claimed_settlement(item: _SessionItem) -> InboxSettlement | None:
        """The durable decision event and its attribution, including recovered sibling calls."""

        reference = item.reference
        after = reference.source_sequence
        event_types = (
            (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED)
            if item.kind == "recovery"
            else (EventType.SESSION_RESUMED,)
        )
        for _page in range(_CLAIM_SCAN_PAGES):
            records = await session_store.query_events(
                EventQuery(
                    session_id=item.session.id,
                    event_types=event_types,
                    after_sequence=after,
                    limit=_CLAIM_SCAN_PAGE,
                )
            )
            for record in records:
                payload = record.event.payload
                decision: InboxSettledDecision | None = None
                if item.kind == "approval" and payload.get("approval_id") == reference.action_id:
                    if payload.get("expired") is True:
                        decision = "expire"
                    else:
                        decision = "approve" if payload.get("decision") == "approve" else "deny"
                elif (
                    item.kind == "question"
                    and payload.get("input_id") == reference.action_id
                    and payload.get("interruption_type") == _INTERRUPTION_TYPE_USER_INPUT_REQUIRED
                ) or (
                    item.kind == "recovery"
                    and payload.get("tool_round_id") == reference.round_id
                    and payload.get("tool_call_id") == reference.tool_call_id
                    and manual_recovery_settlement(record.event) is not None
                ):
                    decision = _RESOLVED_DECISIONS[item.kind]
                if decision is not None:
                    return InboxSettlement(
                        decision=decision,
                        decided_by=_decider(cayu_app.redact_json(payload.get("resolved_by"))),
                        decided_at=record.event.timestamp,
                        reason=cayu_app.redact_json(payload.get("reason"))
                        if type(payload.get("reason")) is str
                        else None,
                    )
            if len(records) < _CLAIM_SCAN_PAGE:
                return None
            after = records[-1].sequence
        return None

    async def session_item_state(item: _SessionItem) -> InboxItemState:
        if item.pending is not None:
            return _pending_state(pending_item(item.kind, item.pending))
        public_session_id = cayu_app.project_session_id_for_exposure(item.session.id)
        locator = _SessionLocator(
            item.kind,
            public_session_id,
            item.reference.source_sequence,
            item.reference.attention_id,
        )

        def state(
            status: InboxItemStatus, settlement: InboxSettlement | None = None
        ) -> InboxItemState:
            return InboxItemState(
                id=_session_item_id(locator),
                kind=item.kind,
                status=status,
                session_id=public_session_id,
                agent_name=_serialize_session_base(cayu_app, item.session)["agent_name"],
                settlement=settlement,
            )

        settlement = await claimed_settlement(item)
        if settlement is not None:
            return state("settled", settlement)
        observation = await observe_human_attention(session_store, item.reference)
        observed: dict[str, InboxSettledDecision] = {
            "resolved": _RESOLVED_DECISIONS[item.kind],
            "cancelled": "recover" if item.kind == "recovery" else "deny",
            "expired": "expire",
        }
        if observation.state in observed:
            return state("settled", InboxSettlement(decision=observed[observation.state]))
        if observation.state == "superseded":
            return state("superseded")
        return state("unavailable")

    async def same_request_recorded(
        item: _SessionItem, request: ToolApprovalRequest | UserInputResponse
    ) -> bool:
        """Whether the Runtime already holds exactly this decision, by digest."""

        session_id = item.session.id
        action_id = item.reference.action_id
        if isinstance(request, ToolApprovalRequest):
            close_kind = "approval-close"
            digest = approval_resolution_request_digest(request)
        else:
            close_kind = "user-input-close"
            digest = user_input_resolution_request_digest(request)
        receipt = await session_store.load_runtime_publication_receipt(
            session_id, f"{close_kind}:{action_id}"
        )
        if receipt is not None:
            return (
                receipt.kind == close_kind
                and receipt.intent.get("resolution_request_digest") == digest
            )
        # Still carrying on: the claim's intent holds the digest until closure.
        try:
            checkpoint = await session_store.load_checkpoint(session_id)
            if isinstance(request, ToolApprovalRequest):
                intent = approval_resolution_intent_from_checkpoint(checkpoint)
                claimed = None if intent is None else (intent.approval_id, intent)
            else:
                _pause, input_intent = user_input_lifecycle_authority_from_checkpoint(checkpoint)
                claimed = None if input_intent is None else (input_intent.input_id, input_intent)
        except (TypeError, ValueError):
            return False
        return (
            claimed is not None
            and claimed[0] == action_id
            and claimed[1].resolution_request_digest == digest
        )

    async def settled_answer(
        item: _SessionItem, request: ToolApprovalRequest | UserInputResponse
    ) -> InboxSettleResponse:
        """Answer a settle of an item that is no longer pending."""

        state = await session_item_state(item)
        if (
            state.settlement is not None
            and state.settlement.decision != "expire"
            and await same_request_recorded(item, request)
        ):
            return InboxSettleResponse(replayed=True, item=state)
        raise _state_conflict(state)

    def settlement_request(
        item: _SessionItem,
        body: InboxSettleRequest,
        auth_context: AuthContext | None,
        loop_policies: tuple[LoopPolicy, ...],
    ) -> ToolApprovalRequest | UserInputResponse:
        """The exact Runtime request for a decision, with private identities."""

        reference = item.reference
        actor = _inbox_actor(auth_context)
        if item.kind == "question":
            assert body.answer is not None
            return UserInputResponse(
                review_reference=body.review_reference,
                session_id=item.session.id,
                task_worker_id=body.task_worker_id,
                task_handoff_id=body.task_handoff_id,
                input_id=reference.action_id,
                answer=body.answer,
                metadata={}
                if body.reason is None
                else {USER_INPUT_DECISION_REASON_KEY: body.reason},
                resolved_by=actor,
                loop_policies=loop_policies,
            )
        assert reference.round_id is not None and reference.tool_call_id is not None
        return ToolApprovalRequest(
            review_reference=body.review_reference,
            session_id=item.session.id,
            task_worker_id=body.task_worker_id,
            task_handoff_id=body.task_handoff_id,
            approval_id=reference.action_id,
            tool_round_id=reference.round_id,
            tool_call_id=reference.tool_call_id,
            decision=ToolApprovalDecision(body.decision),
            arguments=body.arguments,
            reason=body.reason,
            metadata=(
                {}
                if body.approval_revision is None
                else approval_revision_metadata(
                    proposed_digest=body.approval_revision.proposed_digest,
                    approved_digest=body.approval_revision.approved_digest,
                )
            ),
            resolved_by=actor,
            loop_policies=loop_policies,
        )

    async def settle_session_item(
        item: _SessionItem,
        body: InboxSettleRequest,
        auth_context: AuthContext | None,
    ) -> InboxSettleResponse:
        if item.kind == "recovery":
            raise _conflict(
                "not_settleable",
                "Recovery items are settled in the Developer dashboard, with the evidence "
                "of what happened.",
                await session_item_state(item),
            )
        if body.decision not in INBOX_DECISIONS[item.kind]:
            raise HTTPException(
                status_code=422,
                detail=f"A {item.kind} accepts {' or '.join(INBOX_DECISIONS[item.kind])}.",
            )
        if item.kind == "approval" and (
            item.reference.round_id is None or item.reference.tool_call_id is None
        ):
            raise _conflict(
                "unavailable",
                "This approval has no durable round identity. Open it in the Developer dashboard.",
                await session_item_state(item),
            )
        if body.review_reference is not None and (
            auth_context is None
            or body.review_reference.context.recipient != auth_context.subject
            or body.review_reference.context.tenant != auth_context.tenant
        ):
            raise HTTPException(
                status_code=403, detail="Human-review recipient does not match authentication."
            )
        try:
            await cayu_app.require_human_review_resolution_authority(
                item.session.id, body.review_reference
            )
        except HumanReviewDenied:
            raise HTTPException(
                status_code=403, detail="Human-review decision is not authorized."
            ) from None
        request = settlement_request(
            item, body, auth_context, await continuation_loop_policies(item.session.id)
        )
        if item.pending is None:
            return await settled_answer(item, request)
        stream = (
            controlled_inbox_continuation(cayu_app, request, item.reference)
            if body.task_continuation == "claim"
            else cayu_app.resolve_tool_approval(request)
            if isinstance(request, ToolApprovalRequest)
            else cayu_app.resolve_user_input(request)
        )
        try:
            await _accepted_detached_mutation(
                stream,
                cayu_app=cayu_app,
                session_id=item.session.id,
                # The resumption is the durable claim that names the decision and the person.
                until=lambda event: event.type == EventType.SESSION_RESUMED,
                conflict_error_types=(RuntimeError, TimeoutError, ValueError),
            )
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            # Another decision won the claim, the pause changed first, or the
            # Runtime refused this decision and the item is still pending.
            current = await resolve_session_item(
                _SessionLocator(
                    item.kind,
                    cayu_app.project_session_id_for_exposure(item.session.id),
                    item.reference.source_sequence,
                    item.reference.attention_id,
                )
            )
            if current is None:
                raise HTTPException(status_code=404, detail=_NOT_FOUND) from exc
            if current.pending is not None:
                raise _conflict(
                    "conflict", str(exc.detail), await session_item_state(current)
                ) from exc
            return await settled_answer(current, request)
        settled = await session_item_state(replace(item, pending=None))
        if settled.settlement is None or settled.settlement.decision == "expire":
            # The Runtime stopped before recording the decision, or recorded expiry instead.
            raise _state_conflict(settled)
        return InboxSettleResponse(replayed=False, item=settled)

    # ── Knowledge proposals ──────────────────────────────────────────────────

    async def knowledge_item_state(
        workflow: KnowledgeReviewWorkflow, locator: _KnowledgeLocator
    ) -> tuple[InboxItemState, KnowledgeEntry] | None:
        """What happened to one exact proposal revision, from the revision after it.

        Returns the state with the proposal revision itself, which describes the
        item to the inbox access rule.
        """

        try:
            proposal = await workflow.get_entry(locator.entry_id, revision=locator.revision)
            successor = (
                None
                if proposal is None
                else await workflow.get_entry(locator.entry_id, revision=locator.revision + 1)
            )
        except (PermissionError, ValueError):
            return None
        if proposal is None or proposal.status is not KnowledgeStatus.PENDING:
            return None
        item_id = _knowledge_item_id(locator)
        if successor is None:
            try:
                current = await workflow.get_entry(locator.entry_id)
            except (PermissionError, ValueError):
                return None
            if current is not None and current.revision == locator.revision:
                return _pending_state(_knowledge_item(current, cayu_app)), proposal
            return InboxItemState(id=item_id, kind="knowledge", status="unavailable"), proposal
        # Ordinary decisions preserve proposal content. Reviewed drops additionally
        # carry annotations authenticated by the exact atomic publication receipt.
        lifecycle_only = _proposal_content(successor) == _proposal_content(proposal)
        if lifecycle_only and successor.status is KnowledgeStatus.ACTIVE:
            receipt = await workflow.load_review_approval(_knowledge_operation_id(locator))
            decision = (
                None
                if receipt is None
                or receipt.authority.request.candidate_entry.revision != locator.revision
                else receipt.authority.decision
            )
            settlement = InboxSettlement(
                decision="keep",
                decided_by=(
                    None
                    if decision is None
                    else _knowledge_decider(decision.annotations, decision.policy_identity)
                ),
                decided_at=successor.updated_at,
                reason=None
                if decision is None or type(decision.annotations.get("reason")) is not str
                else cayu_app.redact_json(decision.annotations["reason"]),
            )
        elif successor.status is KnowledgeStatus.ARCHIVED:
            try:
                reviewed = await workflow.load_review_rejection(
                    locator.entry_id,
                    expected_revision=locator.revision,
                    operation_id=_knowledge_operation_id(locator, "drop"),
                )
            except PermissionError:
                return None
            except (KnowledgePublicationConflict, TypeError, ValueError):
                return InboxItemState(id=item_id, kind="knowledge", status="unavailable"), proposal
            if reviewed is not None:
                settlement = InboxSettlement(
                    decision="drop",
                    decided_at=reviewed.receipt.committed_at,
                    decided_by=_knowledge_decider(
                        reviewed.decision.annotations, reviewed.decision.reviewer_identity
                    ),
                    reason=cayu_app.redact_json(reviewed.decision.reason),
                )
            elif lifecycle_only:
                settlement = InboxSettlement(decision="drop", decided_at=successor.updated_at)
            else:
                return InboxItemState(id=item_id, kind="knowledge", status="superseded"), proposal
        elif successor.status is KnowledgeStatus.DELETED:
            return InboxItemState(id=item_id, kind="knowledge", status="unavailable"), proposal
        else:
            return InboxItemState(id=item_id, kind="knowledge", status="superseded"), proposal
        settled = InboxItemState(
            id=item_id, kind="knowledge", status="settled", settlement=settlement
        )
        return settled, proposal

    async def settle_knowledge_item(
        workflow: KnowledgeReviewWorkflow,
        locator: _KnowledgeLocator,
        state: InboxItemState,
        body: InboxSettleRequest,
        auth_context: AuthContext | None,
    ) -> InboxSettleResponse:
        if body.decision not in INBOX_DECISIONS["knowledge"]:
            raise HTTPException(
                status_code=422, detail="A knowledge proposal accepts keep or drop."
            )
        settled = state.settlement
        if state.status in {"superseded", "unavailable"} or (
            settled is not None and settled.decision != body.decision
        ):
            raise _state_conflict(state)
        try:
            identity, annotations = _knowledge_reviewer_attribution(auth_context)
            if body.decision == "keep":
                if body.reason is not None:
                    annotations["reason"] = cayu_app.redact_json(body.reason)
                # A repeated keep by the same reviewer replays the recorded approval.
                await workflow.approve(
                    locator.entry_id,
                    operation_id=_knowledge_operation_id(locator),
                    reviewer_identity=identity,
                    reviewer_version=_KNOWLEDGE_REVIEWER_VERSION,
                    annotations=annotations,
                    expected_revision=locator.revision,
                )
            else:
                await workflow.reject_reviewed(
                    locator.entry_id,
                    expected_revision=locator.revision,
                    operation_id=_knowledge_operation_id(locator, "drop"),
                    reviewer_identity=identity,
                    reviewer_version=_KNOWLEDGE_REVIEWER_VERSION,
                    reason=cayu_app.redact_json(body.reason),
                    annotations=annotations,
                )
        except (KeyError, PermissionError):
            raise HTTPException(status_code=404, detail=_NOT_FOUND) from None
        except NotImplementedError:
            raise HTTPException(
                status_code=501, detail="Audited knowledge decisions are unsupported by this store."
            ) from None
        except (
            KnowledgeActivationConflict,
            KnowledgePublicationConflict,
            KnowledgeRevisionConflict,
            ValueError,
        ) as exc:
            current = await knowledge_item_state(workflow, locator)
            if current is None:
                raise HTTPException(status_code=404, detail=_NOT_FOUND) from exc
            if current[0].status == "pending":
                raise _conflict("conflict", str(exc), current[0]) from exc
            raise _state_conflict(current[0]) from exc
        current = await knowledge_item_state(workflow, locator)
        if current is None:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
        return InboxSettleResponse(replayed=settled is not None, item=current[0])

    # ── Listing ──────────────────────────────────────────────────────────────

    def principal(auth_context: AuthContext | None) -> dict[str, str | None]:
        return {
            "subject": None if auth_context is None else auth_context.subject,
            "tenant": None if auth_context is None else auth_context.tenant,
        }

    async def session_items(
        auth_context: AuthContext | None,
        after: str | None,
        sealed: bool,
        limit: int,
        query: PendingActionQuery,
        filters: list[str | None],
        q: str | None,
    ) -> tuple[list[InboxItem], list[InboxIssue], str | None]:
        """One page of session items: items, issues, and the cursor that continues
        them, ``None`` once the page ended the session items.

        A caller the inbox access rule limits gets only items it may see, paged
        from its own items, and none of the store's issues, which name sessions it
        can't see. When its scan ends among other people's items, the cursor seals
        that position, so it can keep paging without learning where they are.
        """

        if sealed:
            assert after is not None
            after = inbox_cursor.open(
                "actions", after, {**principal(auth_context), "filters": filters}
            )
        try:
            if inbox_gate.applies(auth_context):
                permitted = await inbox_gate.scan_pending(
                    auth_context, query.model_copy(update={"cursor": after, "limit": limit})
                )
                items = [
                    pending_item(kind, action)
                    for action in permitted.actions
                    if (kind := _listed_kind(action)) is not None
                ]
                if permitted.next_cursor is not None:
                    next_cursor = _encode_cursor(
                        "actions", _serialize_session_cursor(cayu_app, permitted.next_cursor)
                    )
                elif permitted.hidden_after is not None:
                    next_cursor = _encode_cursor(
                        "actions",
                        inbox_cursor.seal(
                            "actions",
                            permitted.hidden_after,
                            {**principal(auth_context), "filters": filters},
                        ),
                        sealed=True,
                    )
                else:
                    next_cursor = None
                return (
                    [item for item in items if _inbox_item_matches_query(item, q)],
                    [],
                    next_cursor,
                )
            page = await session_store.query_pending_actions(
                query.model_copy(update={"cursor": after, "limit": limit})
            )
        except CheckpointCompatibilityError as exc:
            raise HTTPException(status_code=409, detail=exc.safe_evidence()) from exc
        except PendingActionResultTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        items = [
            pending_item(kind, action)
            for action in page.actions
            if (kind := _listed_kind(action)) is not None
        ]
        issues = [
            InboxIssue(
                code=issue.code.value,
                detail=cayu_app.redact_json(issue.detail),
                session_id=cayu_app.project_session_id_for_exposure(issue.session_id),
                agent_name=cayu_app.redact_json(issue.agent_name),
            )
            for issue in page.issues
        ]
        if page.has_more and page.next_cursor is None:
            # Do not turn an unfinished source scan into a complete public page.
            # Custom/wrapped stores may lack continuation evidence; count readers
            # must see incompleteness rather than publish the visible subtotal.
            issues.append(
                InboxIssue(
                    code="inbox_truncated",
                    detail="The pending-action source could not provide a continuation.",
                )
            )
        next_cursor = (
            _encode_cursor("actions", _serialize_session_cursor(cayu_app, page.next_cursor))
            if page.has_more and page.next_cursor is not None
            else None
        )
        return [item for item in items if _inbox_item_matches_query(item, q)], issues, next_cursor

    async def knowledge_items(
        auth_context: AuthContext | None,
        limit: int,
        after: str | None = None,
        sealed: bool = False,
        filters: list[str | None] | None = None,
        q: str | None = None,
    ) -> tuple[list[InboxItem], str | None]:
        """Scan bounded store pages, continuing after hidden positions too.

        A caller no access rule limits may see every proposal, so its cursor is a
        plain position. A limited caller's cursor is sealed: it can name a hidden
        proposal.
        """

        assert knowledge_review_workflow is not None
        workflow = knowledge_review_workflow()
        limited = inbox_gate.applies(auth_context)
        binding = {
            **principal(auth_context),
            "filters": filters or [None, None, None, None],
            "query": {
                "status": "pending",
                "namespace": workflow.namespace,
                "labels": workflow.labels,
                "scope": workflow.access_scope.model_dump(mode="json"),
            },
        }
        position = after
        if sealed:
            assert after is not None
            position = inbox_cursor.open("knowledge", after, binding)

        def continuation(position: str) -> str:
            if not limited:
                return _encode_cursor("knowledge", position)
            return _encode_cursor(
                "knowledge", inbox_cursor.seal("knowledge", position, binding), sealed=True
            )

        items: list[InboxItem] = []
        for _page in range(_KNOWLEDGE_SCAN_PAGES):
            listed = await workflow.list_pending(
                cursor=position, limit=_PENDING_SCAN_LIMIT, max_bytes=_KNOWLEDGE_PAGE_MAX_BYTES
            )
            more = listed.truncated and (
                listed.total_entries_known is None
                or listed.total_entries_known > len(listed.entries)
            )
            for index, listed_entry in enumerate(listed.entries):
                entry = listed_entry.entry
                position = encode_knowledge_list_cursor(entry)
                if await inbox_gate.permits(auth_context, inbox_gate.knowledge_scope(entry)):
                    item = _knowledge_item(entry, cayu_app)
                    if _inbox_item_matches_query(item, q):
                        items.append(item)
                if len(items) == limit:
                    more = more or index + 1 < len(listed.entries)
                    return items, continuation(position) if more else None
            if not more:
                return items, None
            if not listed.entries:
                # A custom store that cannot return even one bounded entry must
                # not return a cursor that repeats forever.
                raise HTTPException(
                    status_code=409,
                    detail="Knowledge inbox scan could not advance.",
                    headers=_NO_STORE,
                )
        assert position is not None
        return items, continuation(position)

    # ── Routes ───────────────────────────────────────────────────────────────

    async def visible_item(
        item_id: str, auth_context: AuthContext | None
    ) -> tuple[InboxItemState, _SessionItem | tuple[KnowledgeReviewWorkflow, _KnowledgeLocator]]:
        """Resolve an item the caller may see; anything else is 404, like an unknown id."""

        locator = _decode_item_id(item_id)
        if isinstance(locator, _SessionLocator):
            item = await resolve_session_item(locator)
            if item is not None and await inbox_gate.permits(
                auth_context,
                inbox_gate.item_scope(
                    item.session, item.kind, attention_id=item.reference.attention_id
                ),
            ):
                return await session_item_state(item), item
        elif isinstance(locator, _KnowledgeLocator) and knowledge_review_workflow is not None:
            workflow = knowledge_review_workflow()
            known = await knowledge_item_state(workflow, locator)
            if known is not None and await inbox_gate.permits(
                auth_context, inbox_gate.knowledge_scope(known[1])
            ):
                return known[0], (workflow, locator)
        raise HTTPException(status_code=404, detail=_NOT_FOUND, headers=_NO_STORE)

    @router.get(
        "/inbox",
        response_model=InboxPage,
        responses={
            409: {"model": ApiErrorResponse},
            413: {"model": ApiErrorResponse, "description": "Content Too Large"},
            422: {"model": ApiErrorResponse, "description": "Unprocessable Content"},
        },
        description=(
            "List what is waiting on a person: approvals, questions and recovery items "
            "from this agent's sessions, then knowledge proposals on their own pages. A "
            "configured inbox access rule hides items the caller may not see and settle, "
            "here and on every other route that lists or settles them. Follow "
            "`next_cursor` until it is null."
        ),
    )
    async def list_inbox(
        response: Response,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: Annotated[str | None, Query(min_length=1, max_length=8192)] = None,
        session_id: Annotated[str | None, Query(max_length=4096)] = None,
        agent_name: Annotated[str | None, Query(max_length=4096)] = None,
        kind: InboxItemKind | None = None,
        q: Annotated[str | None, Query(max_length=4096)] = None,
        auth_context: AuthContext | None = inbox_auth_context,
    ) -> InboxPage:
        response.headers.update(_NO_STORE)
        return await inbox_page(auth_context, limit, cursor, session_id, agent_name, kind, q)

    async def inbox_page(
        auth_context: AuthContext | None,
        limit: int,
        cursor: str | None,
        session_id: str | None,
        agent_name: str | None,
        kind: InboxItemKind | None,
        q: str | None,
    ) -> InboxPage:
        public_session = _clean_optional_query_value(session_id, "session_id")
        agent_name = _clean_optional_query_value(agent_name, "agent_name")
        q = _clean_optional_query_value(q, "q")
        filters = [public_session, agent_name, kind, q]
        phase, after, sealed = (
            ("actions", None, False)
            if cursor is None
            else _decode_cursor(_unfilter_cursor(cursor, filters))
        )
        knowledge = (
            knowledge_review_workflow is not None
            and kind in (None, "knowledge")
            and public_session is None
            and agent_name is None
        )
        private_session = public_session
        if public_session is not None:
            try:
                private_session = await cayu_app._resolve_public_session_id(public_session)
            except ValueError:
                # Unknown public aliases must not reveal private session identity.
                return InboxPage(items=[], next_cursor=None, has_more=False)
        query = PendingActionQuery(
            session_id=private_session,
            agent_name=agent_name,
            kind=next(
                (pending for pending, listed in PENDING_KINDS.items() if listed == kind), None
            ),
            limit=limit,
        )
        if phase == "knowledge":
            if not knowledge:
                return InboxPage(items=[], next_cursor=None, has_more=False)
            items, next_cursor = await knowledge_items(
                auth_context, limit, after, sealed, filters, q
            )
            return InboxPage(
                items=items,
                next_cursor=_filtered_cursor(next_cursor, filters),
                has_more=next_cursor is not None,
            )
        if kind == "knowledge":
            items, issues, next_cursor = [], [], None
        else:
            items, issues, next_cursor = await session_items(
                auth_context, after, sealed, limit, query, filters, q
            )
        if next_cursor is None and knowledge:
            if items:
                next_cursor = _encode_cursor("knowledge", None)
            else:
                proposals, next_cursor = await knowledge_items(
                    auth_context, limit, filters=filters, q=q
                )
                items.extend(proposals)
        return InboxPage(
            items=items,
            next_cursor=_filtered_cursor(next_cursor, filters),
            has_more=next_cursor is not None,
            issues=issues,
        )

    @router.get(
        "/inbox/counts",
        response_model=InboxCounts,
        responses={
            409: {"model": ApiErrorResponse},
            413: {"model": ApiErrorResponse, "description": "Content Too Large"},
        },
        description=(
            "Count every pending inbox kind independently of list page size. "
            "Limited callers receive unavailable counts, never hidden totals. "
            "Reads bounded source pages to exhaustion; source issues yield unavailable counts."
        ),
    )
    async def inbox_counts(
        response: Response,
        session_id: Annotated[str | None, Query(max_length=4096)] = None,
        agent_name: Annotated[str | None, Query(max_length=4096)] = None,
        q: Annotated[str | None, Query(max_length=4096)] = None,
        auth_context: AuthContext | None = inbox_auth_context,
    ) -> InboxCounts:
        response.headers.update(_NO_STORE)
        session_id = _clean_optional_query_value(session_id, "session_id")
        agent_name = _clean_optional_query_value(agent_name, "agent_name")
        q = _clean_optional_query_value(q, "q")
        if inbox_gate.applies(auth_context):
            return InboxCounts.unavailable("access_limited")

        async def read_page(cursor: str | None) -> InboxPage:
            return await inbox_page(
                auth_context, _PENDING_SCAN_LIMIT, cursor, session_id, agent_name, None, q
            )

        return await count_inbox(read_page)

    @router.get(
        "/inbox/{item_id}",
        response_model=InboxItemState,
        responses={404: {"model": ApiErrorResponse}},
        description=(
            "Read one item's current state: the item while it is pending, and who "
            "settled it, how and when once it is settled, wherever that happened."
        ),
    )
    async def get_inbox_item(
        item_id: ItemIdPath,
        response: Response,
        auth_context: AuthContext | None = inbox_auth_context,
    ) -> InboxItemState:
        response.headers.update(_NO_STORE)
        state, _source = await visible_item(item_id, auth_context)
        return state

    @bounded_router.post(
        "/inbox/{item_id}/settle",
        response_model=InboxSettleResponse,
        responses={
            400: {"model": ApiErrorResponse},
            403: {"model": ApiErrorResponse},
            404: {"model": ApiErrorResponse},
            409: {"model": InboxConflictResponse},
        },
        description=(
            "Settle one item as the authenticated caller: approve (optionally with "
            "changed arguments) or deny an approval, answer a question, keep or drop a "
            "knowledge proposal. Returns once the Runtime durably accepted the decision; "
            "the agent carries on in the background. Repeating the exact decision returns "
            "the recorded settlement with `replayed: true`. Any other settle of a settled, "
            "replaced or expired item gets 409 with the item's current state."
        ),
    )
    async def settle_inbox_item(
        item_id: ItemIdPath,
        body: InboxSettleRequest,
        response: Response,
        auth_context: AuthContext | None = inbox_auth_context,
    ) -> InboxSettleResponse:
        response.headers.update(_NO_STORE)
        state, source = await visible_item(item_id, auth_context)
        if isinstance(source, _SessionItem):
            return await settle_session_item(source, body, auth_context)
        workflow, locator = source
        return await settle_knowledge_item(workflow, locator, state, body, auth_context)
