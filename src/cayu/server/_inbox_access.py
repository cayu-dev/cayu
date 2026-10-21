"""One decision point for who may see and settle an inbox item.

The deployment's ``InboxAccess`` rule is applied wherever an approval, question,
recovery item or knowledge proposal can be listed or settled: the inbox routes,
pending actions, the resolve and recover routes, human review and human
attention reads, supervision events, knowledge review and tool authorization
grants. Every route describes its item through this gate, which builds each
session item's ``InboxItemScope`` in one place from the item's exact
human-attention identity, the same identity the inbox lists it under. Without a
rule, or for an unauthenticated trusted-local caller, the gate does no work and
every route behaves as it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from cayu.events import EventType
from cayu.server.inbox import InboxItemKind, InboxItemScope, InboxKnowledgeEntry
from cayu.sessions.checkpoints import CheckpointCompatibilityError
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.pending_action_contracts import (
    PendingActionQuery,
    PendingActionResultTooLarge,
    human_attention_id,
)
from cayu.sessions.pending_actions import manual_recovery_action_id, manual_recovery_identity
from cayu.sessions.queries import SessionOrder, encode_session_cursor
from cayu.sessions.records import PendingActionKind, PendingActionSession, Session

if TYPE_CHECKING:
    from cayu.knowledge.records import KnowledgeEntry
    from cayu.server.auth import AuthContext
    from cayu.server.inbox import InboxAccess
    from cayu.sessions.pending_action_contracts import PendingActionRecord
    from cayu.sessions.store import SessionStore

PENDING_KINDS: dict[PendingActionKind, InboxItemKind] = {
    PendingActionKind.TOOL_APPROVAL: "approval",
    PendingActionKind.USER_INPUT: "question",
    PendingActionKind.MANUAL_RECOVERY: "recovery",
}
_ATTENTION_KINDS: dict[InboxItemKind, PendingActionKind] = {
    kind: pending for pending, kind in PENDING_KINDS.items()
}
_IDENTITY_FIELDS = ("approval_id", "input_id", "round_id", "tool_call_id")
_NOT_FOUND = "Pending action not found."
# Store pages one request scans for the items a limited caller may see.
_SCAN_PAGES = 10
_SESSION_PENDING_LIMIT = 200
# A manual-recovery handoff is a rare event; a bounded newest-first scan finds it.
_HANDOFF_SCAN_PAGE = 50
_HANDOFF_SCAN_PAGES = 20


@dataclass(frozen=True, slots=True)
class PermittedPendingPage:
    """The pending actions a limited caller may see, paged from its own actions.

    ``next_cursor`` continues right after the last permitted action's session, so it
    never names a hidden session. ``truncated`` means the scan budget ran out among
    hidden actions before finding one the caller may see, so there is no position to
    continue from.
    """

    actions: list[PendingActionRecord]
    next_cursor: str | None
    truncated: bool


class InboxAccessGate:
    """Applies one deployment rule to every route that lists or settles items."""

    def __init__(self, cayu_app: Any, session_store: SessionStore, rule: InboxAccess | None):
        self._cayu_app = cayu_app
        self._session_store = session_store
        self._rule = rule

    def applies(self, person: AuthContext | None) -> bool:
        """Whether this caller's view is limited by the rule."""

        return self._rule is not None and person is not None

    async def permits(self, person: AuthContext | None, scope: InboxItemScope) -> bool:
        if self._rule is None or person is None:
            return True
        return await self._rule(person, scope) is True

    # ── Building scopes ──────────────────────────────────────────────────────

    def item_scope(
        self, session: Session, kind: InboxItemKind, attention_id: str
    ) -> InboxItemScope:
        """The one constructor of a session item's scope, from its exact attention identity."""

        return InboxItemScope(
            kind=kind,
            agent_name=session.agent_name,
            session_id=self._cayu_app.project_session_id_for_exposure(session.id),
            attention_id=attention_id,
            origin_tenant=session.invocation.origin.tenant,
        )

    @staticmethod
    def knowledge_scope(entry: KnowledgeEntry) -> InboxItemScope:
        return InboxItemScope(
            kind="knowledge",
            knowledge=InboxKnowledgeEntry(
                entry_id=entry.id, revision=entry.revision, kind=entry.kind
            ),
        )

    async def _session(
        self, session_id: str, sessions: dict[str, Session | None]
    ) -> Session | None:
        if session_id not in sessions:
            sessions[session_id] = await self._session_store.load(session_id)
        return sessions[session_id]

    async def pending_scope(
        self, action: PendingActionRecord, sessions: dict[str, Session | None]
    ) -> InboxItemScope | None:
        """Describe a pending action as the inbox lists it.

        ``None`` for a delegated parent (navigation to its child, which has its own
        row), an action without an exact identity, or a session that is gone.
        """

        kind = PENDING_KINDS.get(action.kind)
        if kind is None or action.attention_id is None:
            return None
        session = await self._session(action.session.id, sessions)
        return None if session is None else self.item_scope(session, kind, action.attention_id)

    async def locate_scope(
        self,
        session_id: str,
        kinds: tuple[InboxItemKind, ...],
        *,
        approval_id: str | None = None,
        input_id: str | None = None,
        round_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> InboxItemScope | None:
        """Describe the item a resolve or recover request names, from durable state.

        ``session_id`` and the identities are private values. While the item is
        pending, it is the matching pending action, described exactly as the inbox
        lists it. Afterwards, for a retry, it is the item those identities opened:
        a manual-recovery handoff of that call, else the approval or question. An
        item that can't be identified is ``None``, never a partial scope.
        """

        session = await self._session_store.load(session_id)
        if session is None or session.instance_id is None:
            return None
        identity = {
            field: value
            for field, value in zip(
                _IDENTITY_FIELDS, (approval_id, input_id, round_id, tool_call_id), strict=True
            )
            if value is not None
        }
        if not identity:
            return None
        try:
            page = await self._session_store.query_pending_actions(
                PendingActionQuery(session_id=session.id, limit=_SESSION_PENDING_LIMIT)
            )
        except (CheckpointCompatibilityError, PendingActionResultTooLarge):
            page = None
        for action in () if page is None else page.actions:
            kind = PENDING_KINDS.get(action.kind)
            if (
                kind in kinds
                and action.attention_id is not None
                and all(getattr(action, field) == value for field, value in identity.items())
            ):
                return self.item_scope(session, kind, action.attention_id)
        if "recovery" in kinds and tool_call_id is not None:
            handoff = await self._recovery_handoff_attention(session, identity)
            if handoff is not None:
                return self.item_scope(session, "recovery", handoff)
        for kind, field in (("approval", "approval_id"), ("question", "input_id")):
            if kind in kinds and field in identity:
                return self.item_scope(
                    session,
                    kind,
                    human_attention_id(
                        session_id=session.id,
                        session_instance_id=session.instance_id,
                        kind=_ATTENTION_KINDS[kind],
                        discriminator=identity[field],
                        tool_call_id=None,
                    ),
                )
        return None

    async def _recovery_handoff_attention(
        self, session: Session, identity: dict[str, str]
    ) -> str | None:
        """The attention identity of the manual-recovery handoff of the named call."""

        assert session.instance_id is not None
        before: int | None = None
        for _page in range(_HANDOFF_SCAN_PAGES):
            records = await self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_types=(EventType.SESSION_INTERRUPTED, EventType.SESSION_FAILED),
                    before_sequence=before,
                    limit=_HANDOFF_SCAN_PAGE,
                    order_by=EventOrder.SEQUENCE_DESC,
                )
            )
            for record in records:
                if record.event.payload.get("manual_recovery_required") is not True:
                    continue
                handoff = manual_recovery_identity(record.event.payload)
                if handoff is not None and all(
                    getattr(handoff, field) == value for field, value in identity.items()
                ):
                    return human_attention_id(
                        session_id=session.id,
                        session_instance_id=session.instance_id,
                        kind=PendingActionKind.MANUAL_RECOVERY,
                        discriminator=manual_recovery_action_id(session.id, record, handoff),
                        tool_call_id=handoff.tool_call_id,
                    )
            if len(records) < _HANDOFF_SCAN_PAGE:
                return None
            before = records[-1].sequence
        return None

    # ── Decisions ────────────────────────────────────────────────────────────

    async def permits_pending(
        self,
        person: AuthContext | None,
        action: PendingActionRecord,
        sessions: dict[str, Session | None],
    ) -> bool:
        if not self.applies(person):
            return True
        scope = await self.pending_scope(action, sessions)
        return scope is not None and await self.permits(person, scope)

    async def permits_attention(
        self,
        person: AuthContext | None,
        session_id: str,
        kind: InboxItemKind,
        attention_id: str | None,
        sessions: dict[str, Session | None],
    ) -> bool:
        """Whether the rule lets this caller see the item with this attention identity."""

        if not self.applies(person):
            return True
        if attention_id is None:
            return False
        session = await self._session(session_id, sessions)
        return session is not None and await self.permits(
            person, self.item_scope(session, kind, attention_id)
        )

    async def permits_item(
        self,
        person: AuthContext | None,
        session_id: str,
        kinds: tuple[InboxItemKind, ...],
        **identity: str | None,
    ) -> bool:
        """Whether the rule lets this caller see and settle the item a request names."""

        if not self.applies(person):
            return True
        scope = await self.locate_scope(session_id, kinds, **identity)
        return scope is not None and await self.permits(person, scope)

    async def require_item(
        self,
        person: AuthContext | None,
        session_id: str,
        kinds: tuple[InboxItemKind, ...],
        **identity: str | None,
    ) -> None:
        """Refuse a route acting on an item the rule hides, as if it were absent."""

        if not await self.permits_item(person, session_id, kinds, **identity):
            raise HTTPException(status_code=404, detail=_NOT_FOUND)

    async def require_knowledge(self, person: AuthContext | None, entry: KnowledgeEntry) -> None:
        if self.applies(person) and not await self.permits(person, self.knowledge_scope(entry)):
            raise HTTPException(status_code=404, detail="Knowledge entry not found.")

    # ── Listing ──────────────────────────────────────────────────────────────

    async def scan_pending(
        self, person: AuthContext | None, query: PendingActionQuery
    ) -> PermittedPendingPage:
        """Page the pending actions this caller may see, up to ``query.limit`` of them.

        The store pages across every session, so its cursor and issues name sessions
        a limited caller can't see; this page carries neither. A session's actions
        are kept together, so a page can end slightly past ``query.limit``.
        """

        permitted: list[PendingActionRecord] = []
        last: PendingActionSession | None = None
        sessions: dict[str, Session | None] = {}
        cursor = query.cursor
        for _page in range(_SCAN_PAGES):
            page = await self._session_store.query_pending_actions(
                query.model_copy(update={"cursor": cursor})
            )
            for action in page.actions:
                if (
                    len(permitted) >= query.limit
                    and last is not None
                    and (action.session.id != last.id)
                ):
                    return PermittedPendingPage(permitted, _cursor_after(last), False)
                if await self.permits_pending(person, action, sessions):
                    permitted.append(action)
                    last = action.session
            if not page.has_more:
                return PermittedPendingPage(permitted, None, False)
            cursor = page.next_cursor
        if last is not None:
            return PermittedPendingPage(permitted, _cursor_after(last), False)
        return PermittedPendingPage(permitted, None, True)


def _cursor_after(session: PendingActionSession) -> str:
    """The store position right after one session, in pending-action order."""

    return encode_session_cursor(session, SessionOrder.UPDATED_AT_DESC)
