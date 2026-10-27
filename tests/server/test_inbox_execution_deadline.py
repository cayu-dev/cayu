"""The inbox and pending actions show a paused session's execution deadline."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.core.test_continuation_deadlines import _app, _pause_in_handler, _run
from tests.core.test_continuation_deadlines import skip_ahead as skip_ahead
from tests.server.test_server_inbox import _as, _authenticate, _items, _settle, _wait_for_status

from cayu.deadlines import ExecutionDeadline
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore


def _call(client: TestClient, work) -> None:
    portal = client.portal
    assert portal is not None
    portal.call(work)


def test_an_approval_raised_in_a_task_handler_can_be_settled_after_the_handler_limit(
    skip_ahead,
) -> None:
    app = _app(InMemorySessionStore())
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _call(client, lambda: _pause_in_handler(app, "filing", "approval"))

        async def stored_expiry() -> None:
            session = await app.session_store.load("filing")
            stored_expiry.value = session.execution_deadline.expires_at

        _call(client, stored_expiry)
        [item] = _items(client, "priya")
        [action] = client.get("/api/pending-actions", headers=_as("tom")).json()["actions"]
        for shown in (item["execution_deadline"], action["execution_deadline"]):
            assert shown["origin"] == "run_default"
            assert shown["source"] == "task_worker"
            assert shown["waiting_counts"] is False
            assert shown["expires_at"].startswith(
                stored_expiry.value.isoformat(timespec="seconds").replace("+00:00", "")
            )

        skip_ahead(4 * 3600)
        settled = _settle(client, item["id"], "priya", decision="approve")
        assert settled.status_code == 200, settled.text
        _wait_for_status(client, app, "filing")


def test_a_settle_after_an_explicit_deadline_says_which_deadline_passed(skip_ahead) -> None:
    app = _app(InMemorySessionStore())
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:

        async def pause() -> None:
            await _run(app, "filing", "user_input", execution_deadline=ExecutionDeadline.after(60))

        _call(client, pause)
        [item] = _items(client, "priya")
        assert item["execution_deadline"]["origin"] == "explicit"
        assert item["execution_deadline"]["waiting_counts"] is True

        skip_ahead(120)
        refused = _settle(client, item["id"], "priya", decision="answer", answer="staging")
        assert refused.status_code == 409, refused.text
        detail = refused.json()["detail"]
        assert detail["code"] == "conflict"
        assert "Execution deadline expired" in detail["message"]
        assert "set explicitly with execution_deadline" in detail["message"]
        assert item["execution_deadline"]["expires_at"][:19] in detail["message"]
        assert detail["item"]["status"] == "pending"
