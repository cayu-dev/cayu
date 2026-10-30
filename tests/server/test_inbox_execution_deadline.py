"""The inbox and pending actions show a paused session's execution deadline."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from tests.core.test_continuation_deadlines import _app, _pause_in_handler, _run, _store
from tests.core.test_continuation_deadlines import skip_ahead as skip_ahead
from tests.server.test_server_inbox import _as, _authenticate, _items, _settle, _wait_for_status

from cayu import SessionStatus
from cayu.deadlines import ExecutionDeadline, task_event_deadline_scope
from cayu.events import EventType
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


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("enclosing", [False, True])
def test_an_approval_past_its_deadline_can_only_be_denied_and_the_denial_ends_it(
    skip_ahead,
    enclosing,
    backend,
    tmp_path,
    request,
) -> None:
    session_id = f"filing-{uuid4().hex[:12]}"
    app = _app(_store(backend, tmp_path, request))
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:

        async def pause() -> None:
            if enclosing:
                async with task_event_deadline_scope(ExecutionDeadline.after(180)):
                    await _pause_in_handler(app, session_id, "approval")
            else:
                await _run(
                    app, session_id, "approval", execution_deadline=ExecutionDeadline.after(60)
                )

        _call(client, pause)
        [item] = _items(client, "priya")
        assert item["decisions"] == ["approve", "deny"]

        skip_ahead(181)
        [item] = _items(client, "priya")
        assert item["decisions"] == ["deny"]
        refused = _settle(client, item["id"], "priya", decision="approve")
        assert refused.status_code == 409, refused.text
        assert "Execution deadline expired" in refused.json()["detail"]["message"]

        denied = _settle(client, item["id"], "priya", decision="deny", reason="Too late.")
        assert denied.status_code == 200, denied.text
        settlement = denied.json()["item"]["settlement"]
        assert (settlement["decision"], settlement["reason"]) == ("deny", "Too late.")
        assert settlement["decided_by"]["subject"] == "priya"
        assert _items(client, "priya") == []
        again = _settle(client, item["id"], "priya", decision="deny", reason="Too late.")
        assert again.status_code == 200, again.text
        assert again.json()["replayed"] is True

        async def ended() -> None:
            session = await app.session_store.load(session_id)
            assert session.status is SessionStatus.FAILED
            events = await app.session_store.load_events(session_id)
            [failed] = [event for event in events if event.type is EventType.SESSION_FAILED]
            assert failed.payload["reason"] == "tool_approval_denied_after_deadline"
            assert "execution deadline passed" in failed.payload["error"]
            assert [event.type for event in events].count(EventType.TOOL_CALL_APPROVAL_DENIED) == 1
            assert not any(event.type is EventType.TOOL_CALL_STARTED for event in events)

        _call(client, ended)
