"""Decision reasons require positive static secret-scope evidence before publication."""

from __future__ import annotations

import json
import warnings

import pytest
from fastapi.testclient import TestClient
from tests.server.test_server_inbox import (
    _as,
    _authenticate,
    _durable_app,
    _items,
    _pause,
    _portal,
    _settle,
    _supervision,
    _wait_for_status,
)

from cayu.server import ServerConfig, create_server

CANARY = "unresolved-operator-secret-canary"


def configure_scope(app, scope):
    if scope == "dynamic":
        from cayu.environments import Environment, EnvironmentSpec
        from cayu.vaults.static import StaticVault

        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"), vault=StaticVault({"unused": "different-secret"})
            ),
            default=True,
        )


async def retain_unknown_scope(app, session_id):
    checkpoint = await app.session_store.load_checkpoint(session_id)

    def withhold(value):
        if isinstance(value, dict):
            return {
                k: "unknown" if k == "secret_resolution_scope" else withhold(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [withhold(v) for v in value]
        return value

    await app.session_store.checkpoint(session_id, withhold(checkpoint))


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("scope", ["static", "dynamic", "unknown"])
@pytest.mark.parametrize("decision", ["approve", "deny", "answer"])
def test_inbox_reason_scope_gates_durable_and_public_views(
    backend, scope, decision, tmp_path, caplog, capsys
):
    app = _durable_app(backend, tmp_path)
    configure_scope(app, scope)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "reason-scope", "asker" if decision == "answer" else "approver")
        if scope == "unknown":
            _portal(client).call(retain_unknown_scope, app, "reason-scope")
        [item] = _items(client, "priya")
        body = {"decision": decision, "reason": CANARY}
        if decision == "answer":
            body["answer"] = "staging"
        with warnings.catch_warnings(record=True) as captured:
            response = _settle(client, item["id"], "priya", **body)
            assert response.status_code == 200, response.text
            _wait_for_status(client, app, "reason-scope")
            expected = CANARY if scope == "static" else None
            assert response.json()["item"]["settlement"]["reason"] == expected
            state = client.get(f"/api/inbox/{item['id']}", headers=_as("priya"))
            assert state.json()["settlement"]["reason"] == expected
            replay = _settle(client, item["id"], "priya", **body)
            assert replay.status_code == 200 and replay.json()["replayed"]
            assert replay.json()["item"]["settlement"]["reason"] == expected
            changed = _settle(client, item["id"], "priya", **{**body, "reason": "different"})
            assert changed.status_code == 409
            durable = _portal(client).call(app.session_store.load_events, "reason-scope")
            events = client.get("/api/sessions/reason-scope/events", headers=_as("priya"))
            from cayu.server.contracts import SESSION_FOLLOW_PATH

            follow = client.get(
                "/api" + SESSION_FOLLOW_PATH.replace("{session_id}", "reason-scope"),
                headers=_as("priya"),
            )
            assert events.status_code == follow.status_code == 200
            supervision = _supervision(client, "priya")
            if scope == "static":
                assert any(touch["reason"] == expected for touch in supervision)
            if scope != "static":
                output = capsys.readouterr()
                assert CANARY not in repr(durable) + events.text + follow.text + json.dumps(
                    supervision
                )
                assert CANARY not in caplog.text + output.out + output.err + repr(captured)


@pytest.mark.parametrize("scope", ["static", "dynamic", "unknown"])
def test_direct_denial_projects_operator_metadata_without_altering_private_request(scope, tmp_path):
    from cayu.events import EventType

    app = _durable_app("memory", tmp_path)
    configure_scope(app, scope)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "direct-denial", "approver")
        if scope == "unknown":
            _portal(client).call(retain_unknown_scope, app, "direct-denial")
        [action] = client.get("/api/pending-actions", headers=_as("priya")).json()["actions"]
        response = client.post(
            "/api/tool-approvals/resolve",
            headers=_as("priya"),
            json={
                "session_id": action["session"]["id"],
                "approval_id": action["approval_id"],
                "tool_round_id": action["round_id"],
                "tool_call_id": action["tool_call_id"],
                "decision": "deny",
                "reason": CANARY,
                "metadata": {"provided": CANARY},
            },
        )
        assert response.status_code == 200, response.text
        events = _portal(client).call(app.session_store.load_events, "direct-denial")
        denied = next(
            event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_DENIED
        )
        assert denied.payload["reason"] == (CANARY if scope == "static" else None)
        assert denied.payload["metadata"] == ({"provided": CANARY} if scope == "static" else {})
        if scope != "static":
            assert CANARY not in repr(events) + response.text
