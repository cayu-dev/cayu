"""Content-free browser usage from the native runtime publication receipt."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cayu.sessions._browser_usage import BROWSER_USAGE_KEY, browser_usage_receipt

if TYPE_CHECKING:
    from cayu.sessions.records import Session
    from cayu.sessions.store import SessionStore


async def has_retained_browser_usage(store: SessionStore, session: Session) -> bool:
    # Generic browser-session JSON and tool events are caller-writable and are
    # deliberately not evidence. The reserved receipt survives native close.
    record = await store.load_session_operation(session.id, BROWSER_USAGE_KEY)
    return record == browser_usage_receipt(session.id, session.instance_id)


async def retained_browser_usage(
    store: SessionStore, sessions: Sequence[Session]
) -> dict[str, bool]:
    """``has_retained_browser_usage`` for several sessions with one store read.

    Raises ``KeyError`` for a session the store no longer has, as the single
    read does.
    """

    if not sessions:
        return {}
    records = await store.load_session_operations(
        tuple(session.id for session in sessions), BROWSER_USAGE_KEY
    )
    usage: dict[str, bool] = {}
    for session in sessions:
        if session.id not in records:
            raise KeyError(f"Session not found: {session.id}")
        usage[session.id] = records[session.id] == browser_usage_receipt(
            session.id, session.instance_id
        )
    return usage
