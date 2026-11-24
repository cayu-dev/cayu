"""Delete every session a query matches, closing the unfinished ones first when asked.

``delete_sessions(app, SessionQuery(labels={"person": "ana"}), close_reason="Reset")``
composes the existing parts. Matches below another match are left to it. Each remaining
match that hasn't finished is closed once with ``close_reason``, as
``CayuApp.close_session`` closes one (which closes its open child sessions too); without a
reason, or when the close is refused, it is reported and nothing below it is touched.
Then it is deleted with ``CayuApp.delete_session(..., child_policy=RECURSIVE)``, the
dependent-first closure that takes its child sessions, events and records under its own
bounds and checks. Every refusal is one of those checks'.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from cayu.runtime.session_closure import SessionClosureChildPolicy
from cayu.sessions.queries import SessionOrder, SessionQuery, encode_session_cursor
from cayu.sessions.records import SessionStatus

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.approvals.tools import ResolutionActor

#: The most sessions one call deletes; the rest are left, and ``has_more`` says so.
MAX_BULK_SESSION_DELETIONS = 1000
_FINISHED = frozenset({SessionStatus.COMPLETED, SessionStatus.FAILED})
_PAGE = 500
# The deepest ancestry walked to find a match below another match.
_MAX_ANCESTRY = 64


class SessionDeletionRefusal(BaseModel):
    """A matching session that was not deleted, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    reason: str


class SessionBulkDeletion(BaseModel):
    """What ``delete_sessions`` did to the sessions its query matched."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Matching sessions that are gone, including ones a deleted match's closure took.
    deleted: tuple[str, ...] = ()
    #: Sessions closed with ``close_reason`` before they were deleted.
    closed: tuple[str, ...] = ()
    refused: tuple[SessionDeletionRefusal, ...] = ()
    #: More sessions match after the ones this call examined; call again with
    #: ``after=next_after`` for them. Refused sessions aren't examined again.
    has_more: bool = False
    #: The position after the last session this call examined, or ``None``.
    next_after: str | None = None

    @property
    def done(self) -> bool:
        """Whether every matching session is gone."""
        return not self.refused and not self.has_more


async def _matching(
    app: CayuApp, query: SessionQuery, max_sessions: int, after: str | None
) -> tuple[list, bool]:
    # Read every match of this call before deleting any, so a deletion can't move one
    # across pages. The keyset position names values, not rows, so it stays valid
    # after the sessions before it are deleted.
    sessions: list = []
    page_query = query.model_copy(
        update={
            "limit": _PAGE,
            "offset": 0,
            "cursor": after,
            "include_total_count": False,
            "order_by": SessionOrder.CREATED_AT_ASC,
        }
    )
    while True:
        page = await app.session_store.list_sessions(page_query)
        for session in page.sessions:
            if len(sessions) == max_sessions:
                return sessions, True
            sessions.append(session)
        if page.next_cursor is None:
            return sessions, False
        page_query = page_query.model_copy(update={"cursor": page.next_cursor})


async def delete_sessions(
    app: CayuApp,
    query: SessionQuery,
    *,
    close_reason: str | None = None,
    closed_by: ResolutionActor | None = None,
    max_sessions: int = MAX_BULK_SESSION_DELETIONS,
    after: str | None = None,
) -> SessionBulkDeletion:
    """Delete each session ``query`` matches; see the module docstring.

    ``query``'s paging and order are ignored: matches are examined oldest first, up to
    ``max_sessions``, starting after ``after`` (a previous call's ``next_after``).
    ``closed_by`` is recorded on each session it closes.
    """

    if type(query) is not SessionQuery:
        raise TypeError("delete_sessions takes a SessionQuery.")
    if not 1 <= max_sessions <= MAX_BULK_SESSION_DELETIONS:
        raise ValueError(f"max_sessions must be 1 to {MAX_BULK_SESSION_DELETIONS}.")
    if close_reason is not None and not close_reason.strip():
        raise ValueError("close_reason must not be blank.")
    sessions, has_more = await _matching(app, query, max_sessions, after)
    matched = {session.id: session for session in sessions}
    deleted: list[str] = []
    closed: list[str] = []
    refused: list[SessionDeletionRefusal] = []
    # Each match's nearest matching ancestor, if any: its closure takes the match.
    owner: dict[str, str] = {}
    for session in sessions:
        parent_id, depth = session.parent_session_id, 0
        while parent_id is not None and depth < _MAX_ANCESTRY:
            if parent_id in matched:
                owner[session.id] = parent_id
                break
            parent = await app.session_store.load(parent_id)
            parent_id, depth = (None if parent is None else parent.parent_session_id), depth + 1
    gone: set[str] = set()
    for session in sessions:
        if session.id in owner:
            continue
        refusal = await _delete_root(app, session.id, close_reason, closed_by, closed)
        if refusal is None:
            deleted.append(session.id)
            gone.add(session.id)
        else:
            refused.append(SessionDeletionRefusal(session_id=session.id, reason=refusal))
    for session in sessions:
        root = owner.get(session.id)
        while root is not None and root in owner:
            root = owner[root]
        if root is None:
            continue
        if root in gone and await app.session_store.load(session.id) is None:
            deleted.append(session.id)
        else:
            refused.append(
                SessionDeletionRefusal(
                    session_id=session.id, reason="A matching session above it was not deleted."
                )
            )
    return SessionBulkDeletion(
        deleted=tuple(deleted),
        closed=tuple(closed),
        refused=tuple(refused),
        has_more=has_more,
        next_after=(
            encode_session_cursor(sessions[-1], SessionOrder.CREATED_AT_ASC)
            if has_more and sessions
            else None
        ),
    )


async def _delete_root(
    app: CayuApp,
    session_id: str,
    close_reason: str | None,
    closed_by: ResolutionActor | None,
    closed: list[str],
) -> str | None:
    """Close and delete one match with everything below it; the refusal, or ``None``."""

    current = await app.session_store.load(session_id)
    if current is None:
        return None
    # The recursive closure erases descendants whatever their state, so an unfinished
    # one needs the same close a matching session does. They are found as the closure
    # finds them; when it can't, its own deletion below refuses.
    descendants = await app.session_descendant_ids(session_id)
    unfinished = await _unfinished(app, descendants or ())
    if close_reason is None and (current.status not in _FINISHED or unfinished):
        which = (
            f"The session is {current.status.value}"
            if current.status not in _FINISHED
            else f"A session below it is {unfinished[0][1].value}"
        )
        return f"{which}; give a close reason."
    if current.status not in _FINISHED:
        # Its close also closes its open child sessions.
        refusal = await _close(app, session_id, close_reason, closed_by, closed)
        if refusal is not None:
            return refusal
    # Any still open below it, deepest first.
    for descendant_id, _status in reversed(await _unfinished(app, descendants or ())):
        refusal = await _close(app, descendant_id, close_reason, closed_by, closed)
        if refusal is not None:
            return refusal
    try:
        await app.delete_session(
            session_id, finished_only=False, child_policy=SessionClosureChildPolicy.RECURSIVE
        )
    except (ValueError, NotImplementedError) as exc:
        return str(app.redact_json(str(exc)))
    return None


async def _unfinished(app: CayuApp, session_ids) -> list[tuple[str, SessionStatus]]:
    found = []
    for session_id in session_ids:
        session = await app.session_store.load(session_id)
        if session is not None and session.status not in _FINISHED:
            found.append((session_id, session.status))
    return found


async def _close(
    app: CayuApp,
    session_id: str,
    close_reason: str | None,
    closed_by: ResolutionActor | None,
    closed: list[str],
) -> str | None:
    assert close_reason is not None
    try:
        await app.close_session(session_id, reason=close_reason, closed_by=closed_by)
    except KeyError:
        return None
    except (RuntimeError, ValueError) as exc:
        return str(app.redact_json(str(exc)))
    closed.append(session_id)
    return None


__all__ = [
    "MAX_BULK_SESSION_DELETIONS",
    "SessionBulkDeletion",
    "SessionDeletionRefusal",
    "delete_sessions",
]
