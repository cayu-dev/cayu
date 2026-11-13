"""Inbox counts come from stored counts, agree with the list, and answer 304 when unchanged.

SQLite and Postgres count approvals, questions and recovery items from the
checkpoint pending-action flags, and pending knowledge from the knowledge store;
the in-memory store has no native count, so the route reads list pages as before.
Every store must give the counts the list shows.
"""

from __future__ import annotations

import asyncio

import httpx
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.server.test_inbox_failed_session_recovery import SESSION, _fail_while_the_call_ran
from tests.server.test_server_inbox import _app, _as, _authenticate

from cayu.messages import Message
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]


async def _pause(app, session_id: str, agent: str) -> None:
    async for _event in app.run(
        RunRequest(agent_name=agent, session_id=session_id, messages=[Message.text("user", "go")])
    ):
        pass


async def _listed_counts(client: httpx.AsyncClient) -> dict[str, int]:
    counts = {"approval": 0, "question": 0, "recovery": 0, "knowledge": 0}
    cursor = None
    for _ in range(20):
        params = {"limit": 1, **({"cursor": cursor} if cursor else {})}
        page = (await client.get("/api/inbox", params=params, headers=_as("priya"))).json()
        for item in page["items"]:
            counts[item["kind"]] += 1
        cursor = page["next_cursor"]
        if cursor is None:
            return counts
    raise AssertionError("The inbox did not end.")


def test_stored_counts_match_the_list_and_answer_304_when_unchanged(session_store_case) -> None:
    async def run():
        store = await _open_store(session_store_case)
        try:
            app = _app(session_store=store)
            await _pause(app, "approval-session", "approver")
            await _pause(app, "question-session", "asker")
            # A failed session whose started call never finished: a recovery item.
            await _fail_while_the_call_ran(store, environment=None)
            # A completed session's leftover round is an issue, never a recovery item.
            await _fail_while_the_call_ran(store, environment=None, session_id="leftover-session")
            await store.update_status("leftover-session", SessionStatus.COMPLETED)
            transport = httpx.ASGITransport(
                app=create_server(app, config=ServerConfig.protected(_authenticate))
            )
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                listed = await _listed_counts(client)
                first = await client.get("/api/inbox/counts", headers=_as("priya"))
                tag = first.headers["etag"]
                unchanged = await client.get(
                    "/api/inbox/counts", headers={**_as("priya"), "If-None-Match": tag}
                )
                filtered = await client.get(
                    "/api/inbox/counts",
                    params={"session_id": SESSION},
                    headers=_as("priya"),
                )
                await _pause(app, "second-approval", "approver")
                changed = await client.get(
                    "/api/inbox/counts", headers={**_as("priya"), "If-None-Match": tag}
                )
            return listed, first, unchanged, filtered, changed
        finally:
            await _close_store(store)

    listed, first, unchanged, filtered, changed = asyncio.run(run())

    assert listed == {"approval": 1, "question": 1, "recovery": 1, "knowledge": 0}
    assert first.status_code == 200, first.text
    if session_store_case[0] == "memory":
        # Without stored counts the route reads the list, where the leftover round
        # is a source issue, so it publishes no total.
        assert first.json()["unavailable_reason"] == "source_incomplete"
        assert unchanged.status_code == 304
        return
    assert first.json() == {
        "counts": listed,
        "total_count": 3,
        "complete": True,
        "unavailable_reason": None,
    }
    assert first.headers["etag"].startswith('W/"')
    assert first.headers["cache-control"] == "private, no-store"
    assert unchanged.status_code == 304
    assert unchanged.content == b""
    assert unchanged.headers["etag"] == first.headers["etag"]
    assert filtered.json()["counts"] == {
        "approval": 0,
        "question": 0,
        "recovery": 1,
        "knowledge": 0,
    }
    assert changed.status_code == 200
    assert changed.json()["counts"]["approval"] == 2
    assert changed.headers["etag"] != first.headers["etag"]
