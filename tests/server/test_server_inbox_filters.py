"""Inbox query and totals contract through authenticated HTTP and durable records."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.server.test_server_inbox import (
    _as,
    _authenticate,
    _durable_app,
    _own_tenant_and_not_tom,
    _pause,
)

from cayu.server import ServerConfig, create_server


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"session_id": "first"}, {"first"}),
        ({"agent_name": "asker"}, {"second"}),
        ({"kind": "approval"}, {"first", "third"}),
        ({"agent_name": "asker", "kind": "approval"}, set()),
    ],
)
def test_inbox_filters_page_exact_items(store_kind, tmp_path, filters, expected):
    app = _durable_app(store_kind, tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        for session, agent in [("first", "approver"), ("second", "asker"), ("third", "approver")]:
            _pause(client, app, session, agent)
        seen = set()
        cursor = None
        for _ in range(10):
            response = client.get(
                "/api/inbox",
                params={**filters, "limit": 1, **({"cursor": cursor} if cursor else {})},
                headers=_as("priya"),
            )
            assert response.status_code == 200, response.text
            body = response.json()
            seen.update(item["session_id"] for item in body["items"])
            cursor = body["next_cursor"]
            if cursor is None:
                break
        assert cursor is None
        assert seen == expected


def test_filtered_sealed_cursor_binds_filters_and_preserves_empty_page(tmp_path):
    app = _durable_app("memory", tmp_path)
    with TestClient(
        create_server(
            app, config=ServerConfig.protected(_authenticate), inbox_access=_own_tenant_and_not_tom
        )
    ) as client:
        _pause(client, app, "visible-old", "approver", tenant="acme")
        for index in range(11):
            _pause(client, app, f"hidden-{index}", "approver", tenant="globex")
        query = {"kind": "approval", "limit": 1}
        first = client.get("/api/inbox", params=query, headers=_as("acme")).json()
        assert first["items"] == []
        assert first["next_cursor"] is not None
        changed = client.get(
            "/api/inbox",
            params={**query, "kind": "question", "cursor": first["next_cursor"]},
            headers=_as("acme"),
        )
        assert changed.status_code == 422
        second = client.get(
            "/api/inbox", params={**query, "cursor": first["next_cursor"]}, headers=_as("acme")
        ).json()
        assert [item["session_id"] for item in second["items"]] == ["visible-old"]
        assert second["issues"] == []


@pytest.mark.parametrize(
    "field,value", [("kind", "future"), ("agent_name", " "), ("session_id", " ")]
)
def test_inbox_rejects_invalid_filters(field, value, tmp_path):
    app = _durable_app("memory", tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        response = client.get("/api/inbox", params={field: value}, headers=_as("priya"))
        assert response.status_code == 422


def test_counts_cover_both_sources_and_filters_independent_of_page_size():
    from tests.server.test_server_inbox import _knowledge_app, _settle

    app, _store = _knowledge_app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "approval-session", "approver")
        _pause(client, app, "question-session", "asker")
        first = client.get("/api/inbox", params={"limit": 1}, headers=_as("priya")).json()
        assert len(first["items"]) == 1
        response = client.get("/api/inbox/counts", headers=_as("priya"))
        assert response.status_code == 200, response.text
        assert response.json() == {
            "counts": {"approval": 1, "question": 1, "recovery": 0, "knowledge": 2},
            "total_count": 4,
            "complete": True,
            "unavailable_reason": None,
        }
        for filters in ({"session_id": "approval-session"}, {"agent_name": "approver"}):
            filtered = client.get("/api/inbox/counts", params=filters, headers=_as("priya"))
            assert filtered.json()["counts"] == {
                "approval": 1,
                "question": 0,
                "recovery": 0,
                "knowledge": 0,
            }
            listed = client.get("/api/inbox", params=filters, headers=_as("priya")).json()
            assert len(listed["items"]) == 1
        approvals = client.get(
            "/api/inbox", params={"kind": "approval"}, headers=_as("priya")
        ).json()
        assert (
            _settle(client, approvals["items"][0]["id"], "priya", decision="deny").status_code
            == 200
        )
        assert client.get("/api/inbox/counts", headers=_as("tom")).json()["total_count"] == 3
        proposals = client.get(
            "/api/inbox", params={"kind": "knowledge"}, headers=_as("priya")
        ).json()
        assert len(proposals["items"]) == 2
        assert all(item["kind"] == "knowledge" for item in proposals["items"])


def test_limited_counts_are_unavailable_without_hidden_total(tmp_path):
    app = _durable_app("memory", tmp_path)
    with TestClient(
        create_server(
            app, config=ServerConfig.protected(_authenticate), inbox_access=_own_tenant_and_not_tom
        )
    ) as client:
        _pause(client, app, "secret-session", "approver", tenant="globex")
        response = client.get("/api/inbox/counts", headers=_as("acme"))
        assert response.status_code == 200, response.text
        assert response.json() == {
            "counts": None,
            "total_count": None,
            "complete": False,
            "unavailable_reason": "access_limited",
        }
        assert response.headers["cache-control"] == "private, no-store"
        assert "secret-session" not in response.text


def test_approval_argument_display_explicitly_withheld_in_list_and_detail(tmp_path):
    app = _durable_app("memory", tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "approval", "approver")
        [item] = client.get("/api/inbox", headers=_as("priya")).json()["items"]
        assert item["arguments"] is None
        assert item["arguments_unavailable_reason"] == "quarantined"
        detail = client.get(f"/api/inbox/{item['id']}", headers=_as("tom")).json()["item"]
        assert detail == item
        assert "proposed" not in str(item)


@pytest.mark.parametrize("role", ["admin", "developer", "member"])
def test_roles_read_filtered_counts_and_settle_same_runtime_record(role, tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _auth, _token
    from tests.server.test_proxy_identity import _as as signed_headers
    from tests.server.test_server_inbox import _portal, _reference_params

    from cayu.runtime import HumanAttentionRequest
    from cayu.sessions import PendingActionQuery

    key = ec.generate_private_key(ec.SECP256R1())
    app = _durable_app("memory", tmp_path)
    headers = signed_headers(_token(key, role=role))
    with TestClient(
        create_server(app, config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        _pause(client, app, "approval", "approver")
        response = client.get("/api/inbox/counts", headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["total_count"] == 1
        item = client.get("/api/inbox", params={"kind": "approval"}, headers=headers).json()[
            "items"
        ][0]
        attention, _ = _reference_params(app, client, "approval")
        assert item["attention_id"] == attention
        if role == "member":
            denied = client.post(
                f"/api/inbox/{item['id']}/settle",
                headers=headers,
                json={"decision": "approve", "arguments": {"note": "changed"}},
            )
            assert denied.status_code == 403
        settled = client.post(
            f"/api/inbox/{item['id']}/settle",
            headers=headers,
            json={"decision": "deny", "reason": "Reviewed"},
        )
        assert settled.status_code == 200, settled.text
        assert client.get("/api/inbox/counts", headers=headers).json()["total_count"] == 0

        async def pending():
            page = await app.session_store.query_pending_actions(
                PendingActionQuery(session_id="approval")
            )
            return [HumanAttentionRequest.from_pending_action(action) for action in page.actions]

        assert _portal(client).call(pending) == []
        assert client.get(f"/api/inbox/{item['id']}", headers=headers).json()["status"] == "settled"


def test_counts_follow_more_than_ten_bounded_pages_without_capped_totals(monkeypatch, tmp_path):
    import cayu.server._inbox_routes as routes

    app = _durable_app("memory", tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        for index in range(12):
            _pause(client, app, f"approval-{index}", "approver")
        monkeypatch.setattr(routes, "_PENDING_SCAN_LIMIT", 1)
        response = client.get("/api/inbox/counts", headers=_as("priya"))
        assert response.status_code == 200, response.text
        assert response.json()["total_count"] == 12
        assert response.json()["counts"]["approval"] == 12


def test_count_request_cancellation_propagates_without_partial_publication(monkeypatch, tmp_path):
    import asyncio

    import httpx
    from tests.server.test_server_inbox import _portal

    app = _durable_app("memory", tmp_path)
    server = create_server(app, config=ServerConfig.protected(_authenticate))
    with TestClient(server) as client:
        _pause(client, app, "approval", "approver")

        async def exercise():
            entered = asyncio.Event()
            cleaned = asyncio.Event()
            calls = 0

            async def blocked_query(query):
                nonlocal calls
                calls += 1
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

            monkeypatch.setattr(app._runtime_session_store, "query_pending_actions", blocked_query)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as http:
                task = asyncio.create_task(http.get("/api/inbox/counts", headers=_as("priya")))
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                assert task.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()
                assert cleaned.is_set()
                assert calls == 1

        _portal(client).call(exercise)


@pytest.mark.parametrize("mode", ["duplicate", "conflict", "issue", "stalled"])
def test_counts_reconcile_overlap_and_incomplete_sources(monkeypatch, tmp_path, mode):
    from cayu.sessions.pending_action_contracts import PendingActionIssue
    from cayu.sessions.queries import SessionOrder, encode_session_cursor

    app = _durable_app("memory", tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "older", "approver")
        _pause(client, app, "newer", "asker")
        store = app._runtime_session_store
        original = store.query_pending_actions
        first = None

        async def overlapping(query):
            nonlocal first
            if first is None:
                first = await original(query)
                after = encode_session_cursor(
                    first.actions[0].session, SessionOrder.UPDATED_AT_DESC
                )
                return first.model_copy(update={"has_more": True, "next_cursor": after})
            if mode == "issue":
                return first.model_copy(
                    update={
                        "issues": [PendingActionIssue.source_invalid(first.actions[0].session)],
                    }
                )
            if mode == "conflict":
                return first.model_copy(
                    update={
                        "actions": [first.actions[0].model_copy(update={"title": "changed"})],
                    }
                )
            if mode == "stalled":
                after = encode_session_cursor(
                    first.actions[0].session, SessionOrder.UPDATED_AT_DESC
                )
                return first.model_copy(update={"has_more": True, "next_cursor": after})
            return first

        monkeypatch.setattr(store, "query_pending_actions", overlapping)
        response = client.get("/api/inbox/counts", headers=_as("priya"))
        if mode == "stalled":
            assert response.status_code == 409
        else:
            assert response.status_code == 200, response.text
            if mode == "duplicate":
                assert response.json()["total_count"] == 2
            else:
                assert response.json() == {
                    "counts": None,
                    "total_count": None,
                    "complete": False,
                    "unavailable_reason": "source_incomplete",
                }


@pytest.mark.parametrize(
    "later,empty", [(False, False), (False, True), (True, False), (True, True)]
)
def test_count_incomplete_source_without_cursor_never_publishes_subtotal(
    monkeypatch, tmp_path, later, empty
):
    from cayu.sessions.queries import SessionOrder, encode_session_cursor

    app = _durable_app("memory", tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "older", "approver")
        _pause(client, app, "newer", "asker")
        store = app._runtime_session_store
        original = store.query_pending_actions
        calls = 0

        async def incomplete(query):
            nonlocal calls
            calls += 1
            page = await original(query)
            if later and calls == 1:
                after = encode_session_cursor(page.actions[0].session, SessionOrder.UPDATED_AT_DESC)
                return page.model_copy(
                    update={
                        "actions": page.actions[:1],
                        "has_more": True,
                        "next_cursor": after,
                    }
                )
            return page.model_copy(
                update={
                    "actions": [] if empty else page.actions,
                    "has_more": True,
                    "next_cursor": None,
                }
            )

        monkeypatch.setattr(store, "query_pending_actions", incomplete)
        response = client.get("/api/inbox/counts", headers=_as("priya"))
        assert response.status_code == 200, response.text
        assert response.json() == {
            "counts": None,
            "total_count": None,
            "complete": False,
            "unavailable_reason": "source_incomplete",
        }
        assert calls == (2 if later else 1)
