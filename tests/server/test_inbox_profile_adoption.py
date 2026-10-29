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
        refused = _settle(client, item["id"], "priya", decision="deny")
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "conflict"
    assert "direct_tools" in refused.json()["detail"]["message"]
