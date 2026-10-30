"""Dismissing a recovery item abandons its session; the unknown call never runs again."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from tests.core._event_projection_support import private_events_for_public_events
from tests.core.test_abandoned_session_resume import _child, _crash_child, _Proposal, _proposal_app
from tests.core.test_user_input import (
    _CountingTool,
    _crashed_user_input_resume_events,
    _ScriptedProvider,
)
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.server.test_server_inbox import _app, _authenticate, _items, _pause, _portal, _settle

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus, ToolEffect
from cayu.approvals.user_input import UserInputResponse
from cayu.server import ServerConfig, create_server
from cayu.storage import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool


@pytest.fixture
def recovery_inbox(request, tmp_path):
    path = tmp_path / "recovery.sqlite"
    receipt = tmp_path / "proposal-receipt"
    sid = "lost-" + uuid4().hex
    # A worker died while the tool ran: it started, and nothing recorded how it ended.
    asyncio.run(
        _crash_child(
            _child(
                SQLiteSessionStore(path),
                "sqlite",
                request,
                receipt,
                sid,
                ToolEffect.EXTERNAL,
                "crash",
            )
        )
    )
    app = _proposal_app(
        SQLiteSessionStore(path),
        _Proposal(ToolEffect.EXTERNAL, receipt),
        VersionedFakeProvider([]),
        lease_seconds=1.5,
    )
    config = (
        ServerConfig.local_development()
        if getattr(request, "param", None) == "open"
        else ServerConfig.protected(_authenticate)
    )
    with TestClient(create_server(app, config=config)) as client:

        async def recover() -> None:
            await app.recover_abandoned_sessions(inactive_for_seconds=0)

        _portal(client).call(recover)
        yield client, app, sid, receipt


@pytest.mark.parametrize("interrupt_publication", [False, True])
def test_dismissing_a_recovery_item_abandons_its_session(
    recovery_inbox, monkeypatch, interrupt_publication
) -> None:
    client, app, sid, receipt = recovery_inbox
    if interrupt_publication:
        original = app._session_recovery.recover_incomplete_session
        calls = 0

        async def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Publication interrupted")
            return await original(*args, **kwargs)

        monkeypatch.setattr(app._session_recovery, "recover_incomplete_session", interrupted)
        [item] = _items(client, "priya")
        failed = _settle(client, item["id"], "priya", decision="dismiss", reason="Run ended.")
        assert failed.status_code == 409
        assert _portal(client).call(app.session_store.load, sid).status is SessionStatus.FAILED
        assert not any(
            event.type is EventType.SESSION_FAILED
            for event in _portal(client).call(app.session_store.load_events, sid)
        )
    else:
        [item] = _items(client, "priya")
    assert (item["kind"], item["decisions"]) == ("recovery", ["dismiss"])

    unexplained = _settle(client, item["id"], "priya", decision="dismiss")
    assert unexplained.status_code == 422, unexplained.text
    dismissed = _settle(client, item["id"], "priya", decision="dismiss", reason="Run ended.")
    assert dismissed.status_code == 200, dismissed.text
    settlement = dismissed.json()["item"]["settlement"]
    assert (settlement["decision"], settlement["reason"]) == ("dismiss", "Run ended.")
    assert settlement["decided_by"] == {
        "subject": "priya",
        "tenant": None,
        "source": "http_auth",
    }
    assert _items(client, "priya") == []
    state = client.get(f"/api/inbox/{item['id']}", headers={"Authorization": "Bearer tom-token"})
    assert state.status_code == 200, state.text
    assert state.json()["settlement"] == settlement

    again = _settle(client, item["id"], "priya", decision="dismiss", reason="Run ended.")
    assert again.status_code == 200, again.text
    assert again.json()["replayed"] is True
    other = _settle(client, item["id"], "tom", decision="dismiss", reason="Run ended.")
    assert other.status_code == 409, other.text
    assert other.json()["detail"]["code"] == "already_settled"

    async def abandoned() -> None:
        session = await app.session_store.load(sid)
        assert session.status is SessionStatus.FAILED
        events = await app.session_store.load_events(sid)
        [failed] = [event for event in events if event.type is EventType.SESSION_FAILED]
        assert (failed.payload["error_type"], failed.payload["reason"]) == (
            "SessionAbandoned",
            "inbox_dismissed",
        )
        assert failed.payload["metadata"]["reason"] == "Run ended."

    _portal(client).call(abandoned)
    # The call ran once, before the crash, and never again.
    assert receipt.with_suffix(".calls").read_text().count("\n") == 1


def test_a_recovery_item_whose_session_cant_be_abandoned_stays_pending() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        _ScriptedProvider(
            [("sibling", "count", {}), ("question", "ask_user", {"question": "Go?"})]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="asker", model="fake-model"), tools=[_CountingTool(), UserInputTool()]
    )
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:

        async def crash_while_answered() -> None:
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="asker",
                        session_id="asked",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            await app.session_store.append_events(
                "asked",
                _crashed_user_input_resume_events(
                    await private_events_for_public_events(app.session_store, paused),
                    session_id="asked",
                    tool_call_id="sibling",
                ),
            )
            async for _event in app.resolve_user_input(
                UserInputResponse(
                    session_id="asked", input_id=awaiting.payload["input_id"], answer="yes"
                )
            ):
                pass

        _portal(client).call(crash_while_answered)
        [item] = _items(client, "priya")
        assert item["kind"] == "recovery"
        refused = _settle(client, item["id"], "priya", decision="dismiss", reason="Stale.")
        assert refused.status_code == 409, refused.text
        detail = refused.json()["detail"]
        assert detail["code"] == "not_settleable"
        assert "awaiting user input" in detail["message"]
        assert detail["item"]["status"] == "pending"
        assert [listed["id"] for listed in _items(client, "priya")] == [item["id"]]


def test_only_a_recovery_item_can_be_dismissed() -> None:
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "booking", "approver")
        [item] = _items(client, "priya")
        refused = _settle(client, item["id"], "priya", decision="dismiss", reason="No.")
        assert refused.status_code == 422, refused.text


def test_racing_dismissal_cannot_claim_another_persons_settlement(recovery_inbox, monkeypatch):
    client, app, _sid, _receipt = recovery_inbox
    [item] = _items(client, "priya")
    original = app.abandon_session

    async def competing(session_id, **options):
        await original(
            session_id,
            reason="inbox_dismissed",
            metadata={
                "reason": "Other decision",
                "resolved_by": {"subject": "tom", "tenant": None, "source": "http_auth"},
            },
        )
        return await original(session_id, **options)

    monkeypatch.setattr(app, "abandon_session", competing)
    response = _settle(client, item["id"], "priya", decision="dismiss", reason="Run ended.")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "already_settled"
    settlement = response.json()["detail"]["item"]["settlement"]
    assert settlement["decided_by"]["subject"] == "tom"
    assert settlement["reason"] == "Other decision"


@pytest.mark.parametrize("recovery_inbox", ["open"], indirect=True)
def test_open_access_dismissal_succeeds_and_replays_without_attribution(recovery_inbox):
    client, _app, _sid, _receipt = recovery_inbox
    [item] = client.get("/api/inbox").json()["items"]
    body = {"decision": "dismiss", "reason": "Run ended."}
    first = client.post(f"/api/inbox/{item['id']}/settle", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["replayed"] is False
    assert first.json()["item"]["settlement"]["decided_by"] is None
    again = client.post(f"/api/inbox/{item['id']}/settle", json=body)
    assert again.status_code == 200, again.text
    assert again.json()["replayed"] is True
    assert again.json()["item"] == first.json()["item"]
    other = client.post(
        f"/api/inbox/{item['id']}/settle", json={**body, "reason": "Different reason"}
    )
    assert other.status_code == 409
    assert other.json()["detail"]["code"] == "already_settled"
