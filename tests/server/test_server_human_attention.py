from __future__ import annotations

import asyncio

from examples.human_attention.app import build
from examples.human_attention.server import create_demo_server
from fastapi.testclient import TestClient

from cayu import HumanAttentionRequest, Message, PendingActionQuery, RunRequest


def test_optional_server_attention_uses_protected_current_resolution(tmp_path, monkeypatch):
    async def pause():
        app, store, inbox = build(tmp_path, phase="pause")
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="attention-demo",
                        session_id="attention-example",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            action = (await store.query_pending_actions(PendingActionQuery())).actions[0]
            request = HumanAttentionRequest.from_pending_action(action)
            inbox.accept(request)
            return request
        finally:
            await store.close()

    request = asyncio.run(pause())
    monkeypatch.setenv("ATTENTION_STATE", str(tmp_path))
    monkeypatch.setenv("ATTENTION_SERVER_USERNAME", "operator")
    monkeypatch.setenv("ATTENTION_SERVER_PASSWORD", "local-fixture-password")
    with TestClient(create_demo_server()) as client:
        assert client.get("/api/pending-actions").status_code == 401
        response = client.get("/api/pending-actions", auth=("operator", "local-fixture-password"))
        assert response.status_code == 200, response.text
        action = response.json()["actions"][0]
        assert action["attention_id"] == request.reference.attention_id
        assert action["arguments"] is None
        answer = {
            "session_id": action["session"]["id"],
            "input_id": action["input_id"],
            "answer": "staging",
        }
        assert client.post("/api/user-input/resolve", json=answer).status_code == 401
        resolved = client.post(
            "/api/user-input/resolve", json=answer, auth=("operator", "local-fixture-password")
        )
        assert resolved.status_code == 200, resolved.text
        assert "session.completed" in resolved.text

    async def observe():
        app, store, _inbox = build(tmp_path, phase="consume")
        try:
            assert (await app.get_human_attention_state(request.reference)).state == "resolved"
        finally:
            await store.close()

    asyncio.run(observe())
