"""Paging a large inbox: every item once, in a stable order, counted, with bounded store reads."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from tests.core.pending_action_conformance import (
    _approval_checkpoint,
    _approval_request_event,
    _pending_call,
)
from tests.server.test_server_inbox import _as, _authenticate

from cayu import AgentSpec, CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeStatus,
    SQLiteSessionStore,
)

RECOVERIES = 7
APPROVALS = 4
PROPOSALS = 5


def _round(index: int) -> dict[str, str]:
    digits = f"{index:032x}"
    return {
        "model_step_id": f"mstep_{digits}",
        "model_attempt_id": f"matt_{digits}",
        "tool_round_id": f"tround_{digits}",
    }


async def _seed(store: SQLiteSessionStore) -> None:
    """Sessions a worker died in while a tool ran, and sessions waiting on an approval."""

    for index in range(RECOVERIES + APPROVALS):
        recovery = index < RECOVERIES
        session_id = f"{'lost' if recovery else 'asking'}-{index}"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        call = f"call-{index}"
        if recovery:
            identity = _round(index)
            await store.append_events(
                session_id,
                [
                    Event(
                        type=EventType.TOOL_CALL_STARTED,
                        session_id=session_id,
                        tool_name="charge",
                        payload={**identity, "tool_call_id": call, "arguments": {}},
                    ),
                    Event(
                        type=EventType.SESSION_FAILED,
                        session_id=session_id,
                        payload={
                            **identity,
                            "manual_recovery_required": True,
                            "tool_call_id": call,
                        },
                    ),
                ],
            )
            await store.checkpoint(
                session_id,
                {
                    "pending_tool_round": {
                        **identity,
                        "agent_name": "assistant",
                        "tool_calls": [_pending_call(call, "charge")],
                    }
                },
            )
            await store.update_status(session_id, SessionStatus.FAILED)
        else:
            approval = f"approval-{index}"
            await store.append_events(
                session_id,
                [
                    _approval_request_event(
                        event_id=f"asked-{index}",
                        session_id=session_id,
                        approval_id=approval,
                        tool_call_id=call,
                        tool_name="deploy",
                    )
                ],
            )
            await store.checkpoint(session_id, _approval_checkpoint(approval, call, "deploy"))
            await store.update_status(session_id, SessionStatus.INTERRUPTED)


def large_inbox(tmp_path) -> CayuApp:
    """An app whose inbox holds approvals, recovery items and knowledge proposals."""

    path = tmp_path / "inbox.sqlite"

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        await _seed(store)
        await store.close()

    asyncio.run(seed())
    scope = KnowledgeAccessScope.privileged()
    proposals = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id=f"proposal-{index}",
                title=f"Proposal {index}",
                text=f"What the agent learned, number {index}.",
                status=KnowledgeStatus.PENDING,
                created_by_type=KnowledgeActorType.MODEL,
                created_by="assistant",
            )
            for index in range(PROPOSALS)
        ],
        access_scope=scope,
    )
    app = CayuApp(
        session_store=SQLiteSessionStore(path),
        knowledge_store=proposals,
        knowledge_access_scope=scope,
        enable_logging=False,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


def _pages(client: TestClient, limit: int, **filters: str) -> list[list[dict]]:
    pages: list[list[dict]] = []
    cursor = None
    for _ in range(50):
        response = client.get(
            "/api/inbox",
            params={**filters, "limit": limit, **({"cursor": cursor} if cursor else {})},
            headers=_as("priya"),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["issues"] == []
        pages.append(body["items"])
        cursor = body["next_cursor"]
        assert body["has_more"] is (cursor is not None)
        if cursor is None:
            return pages
    raise AssertionError("The inbox listing never ended.")


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (None, {"approval": APPROVALS, "recovery": RECOVERIES, "knowledge": PROPOSALS}),
        ("approval", {"approval": APPROVALS}),
        ("recovery", {"recovery": RECOVERIES}),
        ("knowledge", {"knowledge": PROPOSALS}),
    ],
)
def test_each_kind_and_the_combined_view_page_every_item_once_in_a_stable_order(
    tmp_path, kind, expected
) -> None:
    app = large_inbox(tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        filters = {} if kind is None else {"kind": kind}
        pages = _pages(client, 3, **filters)
        assert len(pages) > 1 and all(len(page) <= 3 for page in pages)
        listed = [item for page in pages for item in page]
        ids = [item["id"] for item in listed]
        assert len(ids) == len(set(ids))
        found = {name: sum(item["kind"] == name for item in listed) for name in expected}
        assert found == expected
        # The same order on every read, and whatever the page size.
        assert [item["id"] for page in _pages(client, 3, **filters) for item in page] == ids
        assert [item["id"] for page in _pages(client, 200, **filters) for item in page] == ids
        if kind is None:
            # Approvals, questions and recovery items come first, then knowledge proposals.
            kinds = [item["kind"] for item in listed]
            assert kinds.index("knowledge") == len(kinds) - PROPOSALS

        counts = client.get("/api/inbox/counts", headers=_as("priya")).json()
        assert counts["complete"] is True
        assert counts["counts"] == {
            "approval": APPROVALS,
            "question": 0,
            "recovery": RECOVERIES,
            "knowledge": PROPOSALS,
        }
        assert counts["total_count"] == APPROVALS + RECOVERIES + PROPOSALS


async def _everyone(person, scope) -> bool:
    return True


@pytest.mark.parametrize("limited", [False, True], ids=["open", "access_rule"])
def test_listing_recovery_items_reads_the_store_per_page_not_per_item(
    tmp_path, monkeypatch, limited
) -> None:
    """However many items wait, a page costs a pending-action read or two, never one per item."""

    app = large_inbox(tmp_path)
    store = app.session_store
    reads: dict[str, int] = {}
    for name in ("load", "load_sessions", "query_events", "query_pending_actions"):
        original = getattr(store, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            reads[_name] = reads.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, counted)
    server = create_server(
        app,
        config=ServerConfig.protected(_authenticate),
        inbox_access=_everyone if limited else None,
    )
    with TestClient(server) as client:
        reads.clear()
        pages = _pages(client, 3, kind="recovery")
        assert sum(len(page) for page in pages) == RECOVERIES
        assert reads.get("load", 0) == 0
        assert reads.get("query_events", 0) == 0
        # A limited caller's page may read one store page past its last item.
        assert reads["query_pending_actions"] <= (2 if limited else 1) * len(pages)
        # Its items are checked against the rule with each store page's sessions read together.
        assert reads.get("load_sessions", 0) <= (reads["query_pending_actions"] if limited else 0)
        assert reads.get("load_sessions", 0) >= (len(pages) if limited else 0)
