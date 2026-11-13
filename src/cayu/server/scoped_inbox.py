"""One person's inbox, scoped by the application's resource access policy.

``(await app.access(subject)).inbox`` is a ``ScopedInboxAccess``: the approvals and
questions waiting in sessions that subject may continue, as
``ScopedCayuAccess.resume`` requires (the session's stored resource binding names
the subject under the policy's authority, and the subject's current decision
covers the session's labels for ``execute``). In b2b that is the conversation's
owner. Knowledge proposals and recovery items stay with operators.

Reads and settles go through the same inbox code as the Developer dashboard's
``/api/inbox``, with the subject as the decider, so the item ids, bodies and
conflicts are the same. ``create_resource_router(..., inbox=True)`` serves it over
HTTP, ``cayu inbox --as SUBJECT`` from the command line, and
``resource_access_rule(app)`` applies the same rule to ``/api``.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any, Final

from fastapi import HTTPException

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.approvals.review import (
    HumanReviewConflict,
    HumanReviewContext,
    HumanReviewDenied,
    HumanReviewLongFieldView,
    HumanReviewReference,
    HumanReviewView,
)
from cayu.server._inbox_access import InboxAccessGate
from cayu.server._inbox_cursor import app_inbox_cursor_seal
from cayu.server._inbox_routes import _inbox_item_matches_query, _listed_kind, inbox_operations
from cayu.server.auth import AuthContext
from cayu.server.inbox import (
    InboxAccess,
    InboxConflict,
    InboxCounts,
    InboxIssue,
    InboxItem,
    InboxItemKind,
    InboxItemScope,
    InboxItemState,
    InboxPage,
    InboxSettleRequest,
    InboxSettleResponse,
)
from cayu.sessions.checkpoints import CheckpointCompatibilityError
from cayu.sessions.pending_action_contracts import (
    PendingActionQuery,
    PendingActionResultTooLarge,
)
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from cayu.resource_access import ScopedCayuAccess
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.sessions.records import Session

__all__ = [
    "MEMBER_INBOX_KINDS",
    "InboxRefused",
    "InboxSettleConflict",
    "ScopedInboxAccess",
    "resource_access_rule",
]

#: The kinds a person decides through the scoped inbox; operators decide the rest.
MEMBER_INBOX_KINDS: Final[tuple[InboxItemKind, ...]] = ("approval", "question")
#: Settle fields only an operator uses. A person approves or denies the call as the
#: agent proposed it: changed ``arguments`` and an ``approval_revision`` are execution
#: overrides, which ``/api`` also refuses outside a full-access role
#: (``cayu.server._decision_only``). Task continuation authority and profile adoption
#: are operator decisions.
_OPERATOR_FIELDS: Final = (
    "arguments",
    "approval_revision",
    "task_worker_id",
    "task_handoff_id",
    "task_continuation",
    "profile_adoption",
)
_OPERATOR_DECISIONS: Final = frozenset({"dismiss", "keep", "drop"})
_MAX_LIMIT: Final = 200
_PENDING_LIMIT: Final = 200
# Session pages one list call reads, and one count reads before giving up.
_LIST_SESSION_PAGES: Final = 10
_COUNT_SESSION_PAGES: Final = 20
_COUNT_PAGE_SIZE: Final = 200
_REVIEW_REFUSED: Final = "Human-review access is not authorized."


class InboxRefused(Exception):
    """A scoped inbox call refused as the HTTP inbox would answer it.

    ``status_code`` and ``detail`` are that answer: 422 for a decision only an
    operator makes, 403 for a review the person may not see, 409 for a review
    that changed. An item the person may not see raises ``ResourceAccessDenied``.
    """

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(detail if isinstance(detail, str) else f"Refused ({status_code}).")
        self.status_code = status_code
        self.detail = detail


class InboxSettleConflict(InboxRefused):
    """A settle refused with 409: settled already, replaced, expired, and so on."""

    def __init__(self, conflict: InboxConflict) -> None:
        super().__init__(409, conflict.model_dump(mode="json"))
        self.conflict = conflict


async def may_decide(app: Any, access: ScopedCayuAccess, scope: InboxItemScope) -> bool:
    """Whether ``access``'s subject may see and settle the item ``scope`` describes."""

    if scope.kind not in MEMBER_INBOX_KINDS or scope.session_id is None:
        return False
    try:
        session_id = await app._resolve_public_session_id(scope.session_id)
        session = await access.sessions.load(session_id)
        await access.require_continuation(session)
    except (ResourceAccessDenied, ValueError, KeyError):
        return False
    return True


def resource_access_rule(app: Any) -> InboxAccess:
    """An ``InboxAccess`` rule that applies the scoped inbox's rule on ``/api``.

    A caller may see and settle an approval or question when it may continue the
    item's session under ``app.resource_access_policy``; knowledge proposals and
    recovery items are hidden from everyone it limits. Use it as
    ``[tool.cayu.serve] inbox_access = "cayu.server.inbox:resource_access_rule"``
    or pass ``resource_access_rule(app)`` as ``create_server(inbox_access=...)``.
    Raises ``ValueError`` when the application has no resource access policy.
    """

    if getattr(app, "resource_access_policy", None) is None:
        raise ValueError("resource_access_rule needs an application with a resource_access_policy.")

    async def rule(person: AuthContext, scope: InboxItemScope) -> bool:
        if scope.kind not in MEMBER_INBOX_KINDS:
            return False
        try:
            access = await app.access(person.subject)
        except ResourceAccessDenied:
            return False
        return await may_decide(app, access, scope)

    return rule


async def _no_loop_policies(_session_id: str) -> tuple[LoopPolicy, ...]:
    return ()


class ScopedInboxAccess:
    """One subject's inbox; get it as ``(await app.access(subject)).inbox``.

    ``continuation_loop_policies`` is what a public service gives the sessions it
    started when they continue, as ``create_router`` takes it.
    """

    def __init__(
        self,
        app: Any,
        access: ScopedCayuAccess,
        *,
        continuation_loop_policies: Callable[[str], Awaitable[tuple[LoopPolicy, ...]]]
        | None = None,
    ) -> None:
        self._app = app
        self._access = access
        self._store = app._runtime_session_store
        self._gate = InboxAccessGate(app, self._store, self._permits)
        self._cursor_seal = app_inbox_cursor_seal(app)
        self._operations = inbox_operations(
            cayu_app=app,
            session_store=self._store,
            knowledge_review_workflow=None,
            continuation_loop_policies=continuation_loop_policies or _no_loop_policies,
            inbox_gate=self._gate,
            inbox_cursor=self._cursor_seal,
        )

    @property
    def subject(self) -> str:
        return self._access.subject

    def _person(self, tenant: str | None) -> AuthContext:
        return AuthContext(subject=self._access.subject, tenant=tenant or None)

    async def _permits(self, person: AuthContext, scope: InboxItemScope) -> bool:
        return person.subject == self._access.subject and await may_decide(
            self._app, self._access, scope
        )

    async def _continuable(self, session: Session) -> bool:
        try:
            await self._access.require_continuation(session)
        except ResourceAccessDenied:
            return False
        return True

    async def _session_items(
        self, session: Session, kind: InboxItemKind | None, q: str | None
    ) -> tuple[builtins.list[InboxItem], builtins.list[InboxIssue]]:
        if not await self._continuable(session):
            return [], []
        public_id = self._app.project_session_id_for_exposure(session.id)
        try:
            page = await self._store.query_pending_actions(
                PendingActionQuery(session_id=session.id, limit=_PENDING_LIMIT)
            )
        except PendingActionResultTooLarge:
            return [], [_issue("source_too_large", public_id, session)]
        except CheckpointCompatibilityError:
            return [], [_issue("source_invalid", public_id, session)]
        items = []
        for action in page.actions:
            listed = _listed_kind(action)
            if listed not in MEMBER_INBOX_KINDS or (kind is not None and listed != kind):
                continue
            item = self._operations.pending_item(listed, action)
            if _inbox_item_matches_query(item, q):
                items.append(item)
        issues = (
            [_issue("source_too_large", public_id, session)] if page.has_more or page.issues else []
        )
        return items, issues

    async def list(
        self,
        kind: InboxItemKind | None = None,
        session_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        *,
        agent_name: str | None = None,
        q: str | None = None,
    ) -> InboxPage:
        """One page of the items this subject may decide, from its own waiting sessions.

        Items of one session stay together, so a page can end slightly past ``limit``.
        Follow ``next_cursor`` until it is ``None``. It is sealed, as the Developer
        dashboard's inbox seals a limited caller's, because the store's position names
        a private session id: it opens only for this subject with the same filters,
        for 15 minutes, and any other cursor raises ``HTTPException`` (422). The key is
        the deployment's ``inbox_cursor_key`` that ``create_router`` (``create_server``,
        ``mount_cayu``) or ``create_resource_router`` was given for this app, else one
        this process makes.
        """

        if type(limit) is not int or not 1 <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be from 1 to {_MAX_LIMIT}.")
        if kind is not None and kind not in MEMBER_INBOX_KINDS:
            return InboxPage(items=[], next_cursor=None, has_more=False)
        if session_id is not None:
            try:
                private_id = await self._app._resolve_public_session_id(session_id)
                session = await self._access.sessions.load(private_id)
            except (ResourceAccessDenied, ValueError, KeyError):
                return InboxPage(items=[], next_cursor=None, has_more=False)
            if agent_name is not None and session.agent_name != agent_name:
                return InboxPage(items=[], next_cursor=None, has_more=False)
            items, issues = await self._session_items(session, kind, q)
            return InboxPage(items=items, next_cursor=None, has_more=False, issues=issues)
        items: list[InboxItem] = []
        issues: list[InboxIssue] = []
        binding = {
            "subject": self._access.subject,
            "filters": [kind, agent_name, q],
        }
        position = (
            None if cursor is None else self._cursor_seal.open("member-inbox", cursor, binding)
        )
        for _page in range(_LIST_SESSION_PAGES):
            page = await self._access.sessions.list_sessions(
                SessionQuery(
                    status=SessionStatus.INTERRUPTED,
                    agent_name=agent_name,
                    cursor=position,
                    limit=limit - len(items),
                )
            )
            for session in page.sessions:
                found, problems = await self._session_items(session, kind, q)
                items.extend(found)
                issues.extend(problems)
            position = page.next_cursor
            if position is None or len(items) >= limit:
                break
        next_cursor = (
            None if position is None else self._cursor_seal.seal("member-inbox", position, binding)
        )
        return InboxPage(
            items=items, next_cursor=next_cursor, has_more=position is not None, issues=issues
        )

    async def counts(
        self,
        *,
        session_id: str | None = None,
        agent_name: str | None = None,
        q: str | None = None,
    ) -> InboxCounts:
        """How many approvals and questions wait on this subject, matching ``q`` as ``list`` does.

        ``unavailable_reason`` is ``source_incomplete`` when a session's actions
        couldn't be read or the scan budget ran out before the last session.
        """

        counts: dict[InboxItemKind, int] = {kind: 0 for kind in MEMBER_INBOX_KINDS}
        if session_id is not None:
            page = await self.list(
                session_id=session_id, limit=_MAX_LIMIT, agent_name=agent_name, q=q
            )
            if page.issues:
                return InboxCounts.unavailable("source_incomplete")
            for item in page.items:
                counts[item.kind] += 1
            return InboxCounts(counts=counts, total_count=sum(counts.values()), complete=True)
        position: str | None = None
        for _page in range(_COUNT_SESSION_PAGES):
            page = await self._access.sessions.list_sessions(
                SessionQuery(
                    status=SessionStatus.INTERRUPTED,
                    agent_name=agent_name,
                    cursor=position,
                    limit=_COUNT_PAGE_SIZE,
                )
            )
            for session in page.sessions:
                items, issues = await self._session_items(session, None, q)
                if issues:
                    return InboxCounts.unavailable("source_incomplete")
                for item in items:
                    counts[item.kind] += 1
            position = page.next_cursor
            if position is None:
                return InboxCounts(counts=counts, total_count=sum(counts.values()), complete=True)
        return InboxCounts.unavailable("source_incomplete")

    async def get(self, item_id: str) -> InboxItemState:
        """One item's current state; ``ResourceAccessDenied`` when this subject can't see it."""

        return await self._call(self._operations.state(item_id, self._person(None)))

    async def settle(
        self, item_id: str, request: InboxSettleRequest, *, tenant: str | None = None
    ) -> InboxSettleResponse:
        """Settle an approval or question as this subject, with ``tenant`` as provenance.

        The decision is recorded with this subject as ``decided_by``, and the session
        continues under its stored resource binding. Raises ``InboxRefused`` (422) for
        a decision only an operator makes, including approving with changed
        ``arguments`` or an ``approval_revision``, ``InboxSettleConflict`` (409) as
        ``/api`` does, and ``ResourceAccessDenied`` for an item this subject can't see.
        """

        if not isinstance(request, InboxSettleRequest):
            raise TypeError("settle takes an InboxSettleRequest.")
        operator_only = [name for name in _OPERATOR_FIELDS if getattr(request, name) is not None]
        if operator_only or request.decision in _OPERATOR_DECISIONS:
            named = ", ".join(operator_only) or f"decision={request.decision}"
            raise InboxRefused(
                422,
                f"An operator decides with {named}, in the Developer dashboard or `cayu inbox`.",
            )
        return await self._call(self._operations.settle(item_id, request, self._person(tenant)))

    async def review(
        self, session_id: str, purpose: str, *, tenant: str | None = None
    ) -> HumanReviewView:
        """The human-review view of a session's pending decision, issued to this subject."""

        await self._require_session(session_id)
        try:
            view = await self._app.inspect_human_review(
                session_id,
                context=HumanReviewContext(
                    recipient=self._access.subject, tenant=tenant or None, purpose=purpose
                ),
            )
        except (HumanReviewDenied, KeyError, ValueError):
            raise InboxRefused(403, _REVIEW_REFUSED) from None
        if view.kind is not None and not await self._review_permitted(session_id, view):
            raise InboxRefused(403, _REVIEW_REFUSED)
        return view

    async def review_long_field(
        self,
        session_id: str,
        reference: HumanReviewReference,
        key: str,
        *,
        tenant: str | None = None,
    ) -> HumanReviewLongFieldView:
        """One long field of the current review, whole, for the review issued to this subject."""

        if reference.context.recipient != self._access.subject or reference.context.tenant != (
            tenant or None
        ):
            raise InboxRefused(403, "Human-review recipient does not match authentication.")
        await self._require_session(session_id)
        try:
            current = await self._app.inspect_human_review(session_id, context=reference.context)
            if current.kind is None:
                raise HumanReviewConflict()
            if not await self._review_permitted(session_id, current):
                raise HumanReviewDenied()
            return await self._app.read_human_review_long_field(
                session_id, reference=reference, key=key
            )
        except HumanReviewConflict:
            raise InboxRefused(
                409, "Human-review content is no longer current; inspect again."
            ) from None
        except (HumanReviewDenied, KeyError, ValueError):
            raise InboxRefused(403, _REVIEW_REFUSED) from None

    async def _require_session(self, session_id: str) -> None:
        try:
            private_id = await self._app._resolve_public_session_id(session_id)
        except (ValueError, KeyError):
            raise ResourceAccessDenied() from None
        await self._access.require_continuation(await self._access.sessions.load(private_id))

    async def _review_permitted(self, session_id: str, view: HumanReviewView) -> bool:
        approval = view.kind == "tool_approval"
        return await self._gate.permits_item(
            self._person(None),
            await self._app._resolve_public_session_id(session_id),
            ("approval",) if approval else ("question",),
            **{"approval_id" if approval else "input_id": view.interaction_id},
        )

    async def _call(self, operation: Awaitable[Any]) -> Any:
        try:
            return await operation
        except HTTPException as exc:
            if exc.status_code == 404:
                raise ResourceAccessDenied() from None
            if exc.status_code == 409 and isinstance(exc.detail, dict):
                try:
                    conflict = InboxConflict.model_validate(exc.detail)
                except ValueError:
                    pass
                else:
                    raise InboxSettleConflict(conflict) from None
            raise InboxRefused(exc.status_code, exc.detail) from None


def _issue(code: Any, public_id: str, session: Session) -> InboxIssue:
    return InboxIssue(
        code=code,
        detail="This session's pending actions couldn't be read in full.",
        session_id=public_id,
        agent_name=session.agent_name,
    )
