"""Real HTTP follow-up omission, override, and lifecycle contracts."""

from __future__ import annotations

import json

import pytest
from tests.core.test_session_followups import FollowupProvider, app_for, spec_for

from cayu.sessions.memory import InMemorySessionStore


@pytest.mark.parametrize("strategy", ["tool", "native"])
@pytest.mark.parametrize("first_state", ["completed", "provider_failed"])
@pytest.mark.parametrize("override", ["omitted", "same", "null", "changed"])
def test_http_followup_preserves_structured_contract(strategy, first_state, override):
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    from fastapi.testclient import TestClient

    from cayu.server import ServerConfig, create_server

    store = InMemorySessionStore()
    session_id = "http-followup"
    provider = FollowupProvider(store, session_id, strategy, first_state)
    app = app_for(store, provider)
    spec = spec_for(strategy).model_dump(mode="json")
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        started = client.post(
            "/api/run",
            json={"session_id": session_id, "prompt": "First", "structured_output": spec},
        )
        assert started.status_code == 200, started.text
        assert (
            f'"session.{"completed" if first_state == "completed" else "failed"}"' in started.text
        )
        body = {"session_id": session_id, "prompt": "Again"}
        if override != "omitted":
            body["structured_output"] = (
                None
                if override == "null"
                else {**spec, **({"max_retries": 1} if override == "changed" else {})}
            )
        resumed = client.post("/api/resume", json=body)
        if override in {"omitted", "same"}:
            assert resumed.status_code == 200, resumed.text
            events = [
                json.loads(line.removeprefix("data:").strip())
                for line in resumed.text.splitlines()
                if line.startswith("data:")
            ]
            assert events[-1]["type"] == "session.completed"
            assert len(provider.requests) == 2
            assert provider.observations[-1].ended_at is None
        else:
            assert resumed.status_code == 409, resumed.text
            assert "structured_output" in resumed.json()["detail"]
            assert len(provider.requests) == 1
            assert client.portal is not None
            transcript = client.portal.call(store.load_transcript, session_id)
            assert all(
                part.text != "Again"
                for message in transcript
                for part in message.content
                if part.type == "text"
            )


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/resume", {"session_id": "missing", "prompt": "Again"}),
        (
            "/api/sessions/missing/messages",
            {"idempotency_key": "next", "content": "Again", "delivery_mode": "next_turn"},
        ),
    ],
)
def test_member_followup_is_role_forbidden_before_session_lookup(path, body):
    from cryptography.hazmat.primitives.asymmetric import ec
    from fastapi.testclient import TestClient
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    from cayu import CayuApp
    from cayu.server import ServerConfig, create_server

    key = ec.generate_private_key(ec.SECP256R1())
    with TestClient(
        create_server(
            CayuApp(enable_logging=False), config=ServerConfig.protected(_auth(FakeJwks(key)))
        )
    ) as client:
        response = client.post(path, json=body, headers=_as(_token(key, role="member")))
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "role_forbidden"


def test_followup_openapi_includes_role_forbidden_shape():
    from cayu import CayuApp
    from cayu.server import ServerConfig, create_server

    schema = create_server(
        CayuApp(enable_logging=False), config=ServerConfig.local_development()
    ).openapi()
    for path in ("/api/resume", "/api/sessions/{session_id}/messages"):
        response = schema["paths"][path]["post"]["responses"]["403"]["content"]["application/json"][
            "schema"
        ]
        assert {item["$ref"].rsplit("/", 1)[-1] for item in response["anyOf"]} == {
            "ApiErrorResponse",
            "RoleForbiddenResponse",
        }
    assert (
        schema["components"]["schemas"]["RoleForbiddenDetail"]["properties"]["code"]["const"]
        == "role_forbidden"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("strategy", ["tool", "native"])
async def test_running_http_followup_and_withdraw_use_durable_queue(strategy):
    import asyncio

    from httpx import ASGITransport, AsyncClient
    from tests.core.test_session_followups import BlockingFollowupProvider
    from tests.server.test_session_message_lifecycle_api import (
        HEADERS,
        OwnershipPolicy,
        authenticate,
    )

    from cayu import AgentSpec, CayuApp, Message, RunRequest
    from cayu.server import ServerConfig, create_server

    store = InMemorySessionStore()
    policy = OwnershipPolicy()
    session_id = "http-queued"
    provider = BlockingFollowupProvider(store, session_id, strategy)
    app = CayuApp(session_store=store, session_message_access_policy=policy, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "First")],
                    structured_output=spec_for(strategy),
                )
            )
        ]

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(provider.started.wait(), 10)
        session = await store.load(session_id)
        for action in ("enqueue", "inspect", "withdraw"):
            policy.grants.add(("alice", "tenant-a", session_id, session.instance_id, action))
        server = create_server(app, config=ServerConfig.protected(authenticate))
        async with AsyncClient(
            transport=ASGITransport(app=server), base_url="http://test", headers=HEADERS
        ) as client:
            for content in ("Again", "Withdraw me"):
                response = await client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={
                        "idempotency_key": content,
                        "content": content,
                        "delivery_mode": "next_turn",
                    },
                )
                assert response.status_code == 200, response.text
                assert "session.message.queued" in response.text
                assert content not in response.text
            inspection = await client.get(f"/api/sessions/{session_id}/messages")
            assert inspection.status_code == 200, inspection.text
            body = inspection.json()
            row = next(row for row in body["records"] if row["message"]["content"] == "Withdraw me")
            withdrawn = await client.post(
                f"/api/sessions/{session_id}/messages/{row['queue_id']}/withdraw",
                json={
                    "session_instance_id": body["session_instance_id"],
                    "idempotency_key": "withdraw",
                    "expected_revision": row["revision"],
                },
            )
            assert withdrawn.status_code == 200, withdrawn.text
            assert withdrawn.json()["record"]["status"] == "withdrawn"
            provider.release.set()
            events = await asyncio.wait_for(task, 15)
            assert events[-1].type.value == "session.completed"
            assert len(provider.requests) == 2
            transcript = await store.load_transcript(session_id)
            texts = [
                part.text
                for message in transcript
                for part in message.content
                if part.type == "text"
            ]
            assert texts.count("Again") == 1
            assert "Withdraw me" not in texts
    finally:
        provider.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_contract_advertises_followups_without_granting_queue_access():
    from fastapi.testclient import TestClient

    from cayu import CayuApp
    from cayu.server import ServerConfig, create_server

    with TestClient(
        create_server(CayuApp(enable_logging=False), config=ServerConfig.local_development())
    ) as client:
        capability = client.get("/api/contract").json()["capabilities"]["surfaces"][
            "session_followups"
        ]
        assert capability["configured"] is True
        assert capability["mutate"]["enabled"] is True
        denied = client.post(
            "/api/sessions/missing/messages",
            json={"idempotency_key": "next", "content": "Again", "delivery_mode": "next_turn"},
        )
        assert denied.status_code == 403


def test_role_forbidden_projection_does_not_reflect_auth_diagnostics():
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from cayu import CayuApp
    from cayu.server import ServerConfig, create_server

    secret = "private-auth-diagnostic-canary"

    async def deny(_request):
        raise HTTPException(
            403,
            detail={"code": "role_forbidden", "message": secret, "private": secret},
            headers={"X-Private": secret},
        )

    with TestClient(
        create_server(CayuApp(enable_logging=False), config=ServerConfig.protected(deny))
    ) as client:
        response = client.post(
            "/api/sessions/missing/messages",
            json={"idempotency_key": "next", "content": "Again", "delivery_mode": "next_turn"},
        )
        assert response.status_code == 403
        assert response.json() == {
            "detail": {"code": "role_forbidden", "message": "Your role can't use this."}
        }
        assert secret not in response.text
        assert secret not in str(response.headers)
        assert response.headers["cache-control"] == "private, no-store"
