"""Source audit survives the authenticated entrance and public event projection."""

import asyncio

import pytest
from tests.server.test_session_message_lifecycle_api import (
    CONTENT,
    HEADERS,
    application,
    client_for,
    sse_events,
)

from cayu.vaults import SecretRedactor


@pytest.mark.parametrize("collision", ["session_id", "transcript_sha256", "checkpoint_sha256"])
def test_source_schema_survives_enqueue_withdraw_and_exact_http_replay(collision):
    app, store, _ = application(redactor=SecretRedactor(collision))
    with client_for(app) as client:
        response = client.post(
            "/api/sessions/source/messages/source-snapshot",
            headers=HEADERS,
            json={"include_transcript_digest": True, "include_checkpoint_digest": True},
        )
        assert response.status_code == 200, response.text
        source = response.json()
        response = client.post(
            "/api/sessions/target/messages",
            headers=HEADERS,
            json={
                "idempotency_key": "source-audit",
                "content": CONTENT,
                "delivery_mode": "next_turn",
                "conditions": {"source": source},
            },
        )
        assert response.status_code == 200, response.text
        accepted = next(
            event for event in sse_events(response) if event["type"] == "session.message.queued"
        )
        assert accepted["payload"]["source"] == source
        response = client.get("/api/sessions/target/messages", headers=HEADERS)
        assert response.status_code == 200, response.text
        page = response.json()
        record = page["records"][0]
        path = f"/api/sessions/target/messages/{record['queue_id']}/withdraw"
        body = {
            "session_instance_id": page["session_instance_id"],
            "idempotency_key": "withdraw-source-audit",
            "expected_revision": record["revision"],
        }
        first = client.post(path, json=body, headers=HEADERS)
        assert first.status_code == 200, first.text
        assert first.json()["event"]["payload"]["source"] == source
        replay = client.post(path, json=body, headers=HEADERS)
        assert replay.status_code == 200, replay.text
        assert replay.json()["event"] == first.json()["event"]
    events = asyncio.run(store.load_events("target"))
    lifecycle = [event for event in events if event.type.startswith("session.message.")]
    assert len(lifecycle) == 2
    assert all(event.payload["source"] == source for event in lifecycle)
    assert all(CONTENT not in str(event.payload) for event in lifecycle)
