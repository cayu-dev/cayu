"""Application-owned durable inbox and bounded reconciliation.

SQLite is the test notification destination. Replace its accept operation with a
service that durably commits before acknowledging. This code owns no Runtime
pause state and can neither answer questions nor grant approvals.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from cayu import CayuApp, Event, HumanAttentionReference, HumanAttentionRequest, PendingActionQuery
from cayu.observability.events import EventSink

REFRESH_EVENTS = frozenset(
    {
        "session.awaiting_user_input",
        "session.interrupted",
        "session.delegated_action.updated",
        "session.checkpointed",
        "session.resumed",
        "session.completed",
        "session.failed",
        "tool.call.approval_requested",
        "tool.call.approved",
        "tool.call.approval_denied",
        "tool.call.approval_expired",
        "tool.call.completed",
        "tool.call.blocked",
        "tool.call.failed",
    }
)
TERMINAL = ("resolved", "cancelled", "superseded", "expired")


class Inbox:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS wakes (
                    session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(session_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS notifications (
                    attention_id TEXT PRIMARY KEY, reference_json TEXT NOT NULL,
                    kind TEXT NOT NULL, summary TEXT NOT NULL, state TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    def accept_hint(self, session_id: str, event_id: str):
        # Event identity is deduplication data only. No callback payload is trusted
        # to identify an executable action or change a notification state.
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO wakes(session_id, event_id) VALUES (?, ?)",
                (session_id, event_id),
            )

    def accept(self, request: HumanAttentionRequest, *, crash_after_accept: bool = False):
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO notifications VALUES (?, ?, ?, ?, ?)",
                (
                    request.reference.attention_id,
                    request.reference.model_dump_json(),
                    request.reference.kind,
                    request.summary,
                    "active",
                ),
            )
        # Simulate durable destination acceptance followed by lost acknowledgement.
        if crash_after_accept:
            os._exit(18)

    def set_state(self, attention_id: str, state: str):
        # Concurrent stale refreshes can never reopen a terminal notification.
        with self.connect() as connection:
            connection.execute(
                """UPDATE notifications SET state = ? WHERE attention_id = ?
                                  AND state NOT IN ('resolved', 'cancelled', 'superseded', 'expired')""",
                (state, attention_id),
            )

    def references(self, after: str = "", limit: int = 200):
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT attention_id, reference_json FROM notifications
                WHERE attention_id > ? AND state NOT IN ('resolved', 'cancelled', 'superseded', 'expired')
                ORDER BY attention_id LIMIT ?""",
                (after, limit),
            ).fetchall()
        return [(key, HumanAttentionReference.model_validate_json(value)) for key, value in rows]

    def inspect(self):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT attention_id, kind, summary, state FROM notifications ORDER BY attention_id"
            ).fetchall()
            wakes = connection.execute("SELECT count(*), sum(processed = 0) FROM wakes").fetchone()
        return {
            "notifications": [
                dict(zip(("attention_id", "kind", "summary", "state"), row, strict=True))
                for row in rows
            ],
            "accepted_event_hints": wakes[0],
            "pending_event_hints": wakes[1] or 0,
        }


class DurableAttentionSink(EventSink):
    def __init__(
        self, inbox: Inbox, *, crash_before_delivery: bool = False, fail_after_accept: bool = False
    ):
        self.inbox = inbox
        self.crash_before_delivery = crash_before_delivery
        self.fail_after_accept = fail_after_accept

    async def emit(self, event: Event) -> None:
        if event.type not in REFRESH_EVENTS:
            return
        if self.crash_before_delivery and event.type == "session.interrupted":
            os._exit(17)  # The pause boundary is already committed by Runtime.
        await asyncio.to_thread(self.inbox.accept_hint, event.session_id or "", event.id)
        if self.fail_after_accept and event.type == "session.interrupted":
            self.fail_after_accept = False
            raise OSError("Simulated lost destination acknowledgement")


async def reconcile(
    app: CayuApp,
    inbox: Inbox,
    *,
    max_pages: int = 32,
    crash_after_accept: bool = False,
    session_id: str | None = None,
):
    """Enroll all current actions in the configured scope, including old pauses.

    Each read/transaction is bounded. An incomplete scan leaves wake receipts
    unacknowledged; absence from a page is never a resolution signal. Production
    applications must bind session/tenant scope from trusted configuration.
    """
    if type(max_pages) is not int or not 1 <= max_pages <= 1000:
        raise ValueError("max_pages must be between 1 and 1000.")
    with inbox.connect() as connection:
        wakes = connection.execute(
            "SELECT session_id, event_id FROM wakes WHERE processed = 0 LIMIT 200"
        ).fetchall()
    cursor = None
    complete = True
    for _ in range(max_pages):
        page = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id, cursor=cursor, limit=200)
        )
        complete = complete and not page.issues
        for action in page.actions:
            request = HumanAttentionRequest.from_pending_action(action)
            if request is None:
                if action.kind != "delegated_action":
                    complete = False
                # Delegated rows point to a child; the all-session scan discovers
                # that child's one actionable request. Never answer through parent.
                continue
            observation = await app.get_human_attention_state(request.reference)
            if observation.state == "active":
                await asyncio.to_thread(
                    inbox.accept, request, crash_after_accept=crash_after_accept
                )
            elif observation.state == "unavailable":
                complete = False
        cursor = page.next_cursor
        if cursor is None:
            complete = complete and not page.has_more
            break
    else:
        complete = False

    # Inspect exact enrolled references. Missing global rows, failed pages, and
    # callbacks received out of order cannot settle a notification by themselves.
    after = ""
    for _ in range(max_pages):
        references = await asyncio.to_thread(inbox.references, after)
        for key, reference in references:
            observation = await app.get_human_attention_state(reference)
            if observation.state == "unavailable":
                complete = False
            await asyncio.to_thread(inbox.set_state, key, observation.state)
        if not references or len(references) < 200:
            break
        after = references[-1][0]
    else:
        complete = False
    if complete:
        with inbox.connect() as connection:
            connection.executemany(
                "UPDATE wakes SET processed = 1 WHERE session_id = ? AND event_id = ?", wakes
            )
    return {"complete": complete, **inbox.inspect()}
