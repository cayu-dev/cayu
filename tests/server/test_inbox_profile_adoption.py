"""The inbox and the approval route can settle an approval across an upgrade."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.core.test_tool_approval_profile_adoption import (
    _AdoptExplicitRequests,
    _app,
    _FileBug,
    _pause,
    _RedescribedFileBug,
    _still_pending,
    _use_cayu,
)
from tests.server.test_server_inbox import _as, _authenticate, _items, _settle

from cayu.events import EventType
from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery
from cayu.runtime.execution_profiles import ExecutionProfilePolicyAction
from cayu.server import ServerConfig, create_server
from cayu.sessions.memory import InMemorySessionStore


def _upgraded(monkeypatch, config: ServerConfig):
    """Pause under Cayu 0.15, then serve the same store from 0.16 with a new tool version."""

    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    tool = _FileBug(behavior_version="2")
    app = _app(store, tool=tool, profile_policy=_AdoptExplicitRequests())
    return store, tool, requested, TestClient(create_server(app, config=config))


def _decided(store) -> list[dict]:
    events = asyncio.run(store.load_events("bug-report"))
    return [e.payload for e in events if e.type == EventType.SESSION_EXECUTION_PROFILE_DECIDED]


def test_the_inbox_settles_an_approval_with_the_settling_person_as_adopter(monkeypatch) -> None:
    store, tool, _requested, client = _upgraded(monkeypatch, ServerConfig.protected(_authenticate))
    with client:
        [item] = _items(client, "priya")

        refused = _settle(client, item["id"], "priya", decision="approve")
        assert refused.status_code == 409, refused.text
        assert refused.json()["detail"]["code"] == "profile_changed"
        assert "profile_adoption" in refused.json()["detail"]["message"]
        assert refused.json()["detail"]["item"]["status"] == "pending"
        assert tool.calls == []

        forged = _settle(
            client,
            item["id"],
            "priya",
            decision="approve",
            profile_adoption={"idempotency_key": "k", "reason": "r", "requested_by": "tom"},
        )
        assert forged.status_code == 400, forged.text

        settled = _settle(
            client,
            item["id"],
            "priya",
            decision="approve",
            profile_adoption={"idempotency_key": "adopt-0.16", "reason": "Upgraded to 0.16."},
        )
        assert settled.status_code == 200, settled.text
        assert settled.json()["item"]["status"] == "settled"

    [decided] = _decided(store)
    assert decided["actor"]["subject"] == "priya"
    assert decided["actor"]["source"] == "http_auth"
    assert decided["reason"] == "Upgraded to 0.16."
    assert tool.calls == [{"title": "Crash on save"}]


def test_profile_adoption_is_only_for_approvals(monkeypatch) -> None:
    _store, _tool, _requested, client = _upgraded(
        monkeypatch, ServerConfig.protected(_authenticate)
    )
    with client:
        [item] = _items(client, "priya")
        response = _settle(
            client,
            item["id"],
            "priya",
            decision="answer",
            answer="yes",
            profile_adoption={"idempotency_key": "k", "reason": "r"},
        )
    assert response.status_code == 422, response.text


def test_the_approval_route_carries_an_adoption_intent(monkeypatch) -> None:
    store, tool, requested, client = _upgraded(monkeypatch, ServerConfig.local_development())
    body = {
        "session_id": "bug-report",
        "approval_id": requested["approval_id"],
        "tool_round_id": requested["tool_round_id"],
        "tool_call_id": requested["tool_call_id"],
        "decision": "approve",
    }
    with client:
        missing_actor = client.post(
            "/api/tool-approvals/resolve",
            json={**body, "profile_adoption": {"idempotency_key": "k", "reason": "r"}},
            headers=_as("priya"),
        )
        assert missing_actor.status_code == 400, missing_actor.text
        assert _still_pending(store)

        with client.stream(
            "POST",
            "/api/tool-approvals/resolve",
            json={
                **body,
                "profile_adoption": {
                    "idempotency_key": "adopt-0.16",
                    "reason": "Upgraded to 0.16.",
                    "requested_by": {"subject": "operator"},
                },
            },
        ) as response:
            lines = list(response.iter_lines())
        assert response.status_code == 200, "\n".join(lines)

    [decided] = _decided(store)
    assert (decided["actor"]["subject"], decided["actor"]["source"]) == ("operator", "request")
    assert tool.calls == [{"title": "Crash on save"}]


def test_an_unauthenticated_inbox_records_the_adopter_as_local_development(monkeypatch) -> None:
    store, tool, _requested, client = _upgraded(monkeypatch, ServerConfig.local_development())
    with client:
        [item] = client.get("/api/inbox").json()["items"]
        refused = client.post(f"/api/inbox/{item['id']}/settle", json={"decision": "approve"})
        assert refused.json()["detail"]["code"] == "profile_changed"
        # The dashboard's retry body: the same decision plus a key and a reason.
        settled = client.post(
            f"/api/inbox/{item['id']}/settle",
            json={
                "decision": "approve",
                "profile_adoption": {"idempotency_key": "adopt-0.16", "reason": "Upgraded."},
            },
        )
        assert settled.status_code == 200, settled.text

    [decided] = _decided(store)
    assert (decided["actor"]["subject"], decided["actor"]["source"]) == (
        "unauthenticated-local-development",
        "request",
    )
    assert tool.calls == [{"title": "Crash on save"}]


def test_a_refused_adoption_is_a_plain_conflict(monkeypatch) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    app = _app(
        store,
        tool=_FileBug(behavior_version="2"),
        profile_policy=_AdoptExplicitRequests(ExecutionProfilePolicyAction.REJECT),
    )
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        refused = _settle(
            client,
            item["id"],
            "priya",
            decision="approve",
            profile_adoption={"idempotency_key": "adopt-0.16", "reason": "Upgraded."},
        )
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "conflict"
    assert refused.json()["detail"]["item"]["status"] == "pending"


def test_a_changed_tool_catalogue_is_a_conflict_not_an_offer_to_adopt(monkeypatch) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    app = _app(store, tool=_RedescribedFileBug(), profile_policy=_AdoptExplicitRequests())
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        refused = _settle(client, item["id"], "priya", decision="approve")
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "conflict"
    assert "direct_tools" in refused.json()["detail"]["message"]


@pytest.mark.parametrize("tool", [_FileBug(behavior_version="2"), _RedescribedFileBug()])
def test_the_inbox_settles_a_denial_the_changed_profile_wont_continue(monkeypatch, tool) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    tool.calls.clear()
    app = _app(store, tool=tool)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        settled = _settle(client, item["id"], "priya", decision="deny", reason="Not a bug.")
        assert settled.status_code == 200, settled.text
        assert settled.json()["replayed"] is False
        settlement = settled.json()["item"]["settlement"]
        assert (settlement["decision"], settlement["reason"]) == ("deny", "Not a bug.")
        assert settlement["decided_by"]["subject"] == "priya"
        assert _items(client, "priya") == []

        again = _settle(client, item["id"], "priya", decision="deny", reason="Not a bug.")
        assert again.status_code == 200, again.text
        assert again.json()["replayed"] is True
        other = _settle(client, item["id"], "priya", decision="approve")
        assert other.status_code == 409, other.text

    assert tool.calls == []
    assert not _still_pending(store)


def test_a_retried_denial_writes_the_events_a_crash_left_out(monkeypatch) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    publish = IncompleteSessionRecovery._publish_stranded_approval_denial

    async def crash_after_ending(self, session, denial):
        # The session ended with the denial recorded; the process stops before its events.
        raise RuntimeError("Worker stopped before the denial events were written.")

    monkeypatch.setattr(
        IncompleteSessionRecovery, "_publish_stranded_approval_denial", crash_after_ending
    )
    app = _app(store, tool=_FileBug(behavior_version="2"))
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        assert (
            _settle(client, item["id"], "priya", decision="deny", reason="Not a bug.").status_code
            == 409
        )
        assert not _still_pending(store)
        events = asyncio.run(store.load_events("bug-report"))
        assert EventType.TOOL_CALL_APPROVAL_DENIED not in [e.type for e in events]

        monkeypatch.setattr(IncompleteSessionRecovery, "_publish_stranded_approval_denial", publish)
        again = _settle(client, item["id"], "priya", decision="deny", reason="Not a bug.")
        assert again.status_code == 200, again.text
        assert again.json()["replayed"] is True
        settlement = again.json()["item"]["settlement"]
        assert (settlement["decision"], settlement["decided_by"]["subject"]) == ("deny", "priya")

    events = asyncio.run(store.load_events("bug-report"))
    assert [e.type for e in events[-2:]] == [
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_FAILED,
    ]
