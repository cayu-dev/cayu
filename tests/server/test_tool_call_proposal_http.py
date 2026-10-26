"""A proposed call is listed and settled like any approval, and served watchers react."""

from __future__ import annotations

import time
from functools import partial

from fastapi.testclient import TestClient
from tests.core.test_tool_call_proposals import FILED, _build

from cayu.approvals import ToolCallProposalStatus
from cayu.events import EventType
from cayu.observability import EventWatcher
from cayu.server import ServerConfig, create_server
from cayu.sessions.event_queries import EventQuery


def test_dashboard_lists_and_denies_a_proposal_and_the_served_watcher_reacts() -> None:
    FILED.clear()
    outcomes: list[tuple[str, str]] = []
    app = _build()
    app.register_event_watcher(
        EventWatcher(
            name="approval-outcomes",
            query=EventQuery(
                event_types=(EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_APPROVAL_DENIED)
            ),
            handler=lambda context: outcomes.append(
                (context.record.event.type.value, context.record.event.session_id)
            ),
        )
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        pending = client.portal.call(
            partial(app.propose_tool_call, "scout", "file_issue", {"title": "Stored title"})
        )
        assert pending.status is ToolCallProposalStatus.PENDING_APPROVAL
        response = client.get("/api/pending-actions")
        assert response.status_code == 200, response.text
        [action] = response.json()["actions"]
        assert action["session"]["id"] == pending.session_id
        response = client.post(
            "/api/tool-approvals/resolve",
            json={
                "session_id": action["session"]["id"],
                "approval_id": action["approval_id"],
                "tool_round_id": action["round_id"],
                "tool_call_id": action["tool_call_id"],
                "decision": "deny",
                "reason": "Duplicate.",
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 10
        while not outcomes and time.monotonic() < deadline:
            time.sleep(0.05)

    assert outcomes == [("tool.call.approval_denied", pending.session_id)]
    assert FILED == []
