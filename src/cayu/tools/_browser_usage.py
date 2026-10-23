"""Content-free browser usage from the native runtime publication receipt."""

from __future__ import annotations

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
