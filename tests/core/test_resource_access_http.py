from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI
from tests.core.test_resource_execution_access import Policy

from cayu import CayuApp, RunRequest, SessionIdentity
from cayu.server.resource_access import create_resource_router


def test_http_authentication_is_host_owned_and_foreign_ids_are_hidden():
    async def run():
        app = CayuApp(resource_access_policy=Policy())
        for session_id, organization in [("own", "acme"), ("foreign", "other")]:
            await app.session_store.create(
                RunRequest(
                    agent_name="shared",
                    session_id=session_id,
                    messages=[],
                    labels={"organization": organization},
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )

        async def authenticate(request):
            # Deliberately ignore every caller-supplied identity and tenant field.
            return (
                "alice" if request.headers.get("authorization") == "Bearer verified-token" else ""
            )

        server = FastAPI()
        server.include_router(create_resource_router(app, authenticate=authenticate))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            assert (await client.get("/resources/sessions/own")).status_code == 401
            headers = {
                "authorization": "Bearer verified-token",
                "x-tenant": "other",
                "x-subject": "operator",
            }
            assert (await client.get("/resources/sessions/own", headers=headers)).status_code == 200
            foreign = await client.get("/resources/sessions/foreign", headers=headers)
            absent = await client.get("/resources/sessions/missing", headers=headers)
            assert foreign.status_code == absent.status_code == 404
            assert foreign.json() == absent.json()
            page = await client.post(
                "/resources/sessions/query", json={"include_total_count": True}, headers=headers
            )
            assert page.status_code == 200
            assert page.json()["total_count"] == 1
            forged = await client.post(
                "/resources/runs",
                json={
                    "agent_name": "shared",
                    "messages": [],
                    "resource_access": {"authority": "operator"},
                },
                headers=headers,
            )
            assert forged.status_code == 422
            assert (
                await client.get("/resources/sessions/foreign/records/events", headers=headers)
            ).status_code == 404
            assert (
                await client.delete("/resources/sessions/foreign", headers=headers)
            ).status_code == 404
            assert await app.session_store.load("foreign") is not None

    asyncio.run(run())


def test_http_stream_revalidates_buffered_first_event():
    from cayu.events import Event, EventType
    from cayu.server.resource_access import _stream
    from cayu.sessions.access import SessionAccessScope

    async def run():
        policy = Policy()
        app = CayuApp(resource_access_policy=policy)
        access = await app.access("alice")
        session = await access.sessions.create(
            RunRequest(
                agent_name="shared", session_id="own", messages=[], labels={"organization": "acme"}
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        assert session.invocation.resource_access.subject == "alice"
        closed = []

        async def events():
            try:
                yield Event(
                    type=EventType.MODEL_TEXT_DELTA, session_id="own", payload={"text": "protected"}
                )
            finally:
                closed.append(True)

        response = await _stream(events(), access.revalidate_delivery)
        policy.current = SessionAccessScope()
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 1 and "access_revoked" in chunks[0]
        assert "protected" not in chunks[0]
        assert closed == [True]

    asyncio.run(run())


def test_scoped_http_run_delivers_authorized_events():
    from tests.core.test_resource_execution_access import app_for

    from cayu.sessions.base import InMemorySessionStore

    async def run():
        app = app_for(InMemorySessionStore(), Policy())

        async def authenticate(request):
            return "alice"

        server = FastAPI()
        server.include_router(create_resource_router(app, authenticate=authenticate))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            response = await client.post(
                "/resources/runs",
                json={
                    "agent_name": "shared",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                    "labels": {"organization": "acme"},
                },
            )
            assert response.status_code == 200, response.text
            assert "session.completed" in response.text
            assert "access_revoked" not in response.text

    asyncio.run(run())
