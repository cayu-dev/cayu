"""Overview failure boundaries through real control-plane and Cloud HTTP reads."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _delegating_app, _start
from tests.server.test_cloud_reader_http import TOKEN, _bearer, _digest

from cayu.server import ServerConfig, create_server
from cayu.server.cloud_reporting import CLOUD_READER_TOKEN_SHA256_VARIABLE
from cayu.sessions.pending_action_contracts import PendingActionIssue, PendingActionSession
from cayu.sessions.queries import SessionOrder, encode_session_cursor
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]


@pytest.mark.store_semantics
@pytest.mark.parametrize("mode", ["delete", "replace", "unrelated"])
def test_browser_enrichment_reconciles_only_confirmed_identity_loss(
    session_store_case, monkeypatch, mode
):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "gone", "writer")
            await _start(app, "kept", "writer")
            original = store.load_session_operation
            reached = asyncio.Event()
            release = asyncio.Event()

            async def paused_read(session_id, operation_id):
                if session_id == "gone":
                    reached.set()
                    await release.wait()
                    if mode == "unrelated":
                        raise KeyError("unrelated-store-failure")
                return await original(session_id, operation_id)

            monkeypatch.setattr(store, "load_session_operation", paused_read)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development()),
                    raise_app_exceptions=False,
                ),
                base_url="http://test",
            ) as client:
                request = asyncio.create_task(
                    client.get("/api/sessions", params={"include": "overview"})
                )
                await asyncio.wait_for(reached.wait(), 3)
                if mode != "unrelated":
                    await store.delete_session("gone")
                    if mode == "replace":
                        await _start(app, "gone", "writer")
                release.set()
                response = await request
                if mode == "unrelated":
                    assert response.status_code == 500
                else:
                    assert response.status_code == 200, response.text
                    assert [row["id"] for row in response.json()["sessions"]] == ["kept"]
                    assert [row["session_id"] for row in response.json()["overviews"]] == ["kept"]
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("delegated", [False, True])
def test_cloud_advances_past_pending_issues_while_control_plane_refuses_partial_state(
    monkeypatch, delegated
):
    monkeypatch.setenv(CLOUD_READER_TOKEN_SHA256_VARIABLE, _digest(TOKEN))

    async def scenario():
        app = _delegating_app() if delegated else _app()
        await _start(app, "root" if delegated else "bad", "root" if delegated else "writer")
        store = app.session_store
        original = store.query_pending_actions

        async def incomplete(query=None, **kwargs):
            page = await original(query, **kwargs)
            session = await store.load(query.session_id)
            if session is not None and session.agent_name == ("leaf" if delegated else "writer"):
                issue = PendingActionIssue.source_invalid(
                    PendingActionSession.from_session(session)
                )
                return page.model_copy(update={"issues": [issue]})
            return page

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_server(app, config=ServerConfig.local_development()),
                raise_app_exceptions=False,
            ),
            base_url="http://test",
        ) as client:
            path = "/api/cloud/v1/activity"
            normal = await client.get(path, headers=_bearer(TOKEN))
            assert normal.status_code == 200
            monkeypatch.setattr(store, "query_pending_actions", incomplete)
            response = await client.get(path, headers=_bearer(TOKEN))
            assert response.status_code == 200, response.text
            assert response.json() == normal.json()
            cursor = response.json()["next_cursor"]
            caught_up = await client.get(path, params={"cursor": cursor}, headers=_bearer(TOKEN))
            assert caught_up.status_code == 200
            assert caught_up.json()["runs"] == []
            assert caught_up.json()["next_cursor"] == cursor
            overview = await client.get(
                "/api/sessions",
                params={"include": "overview", "ids": "root" if delegated else "bad"},
            )
            assert overview.status_code == 503
            counts = await client.get("/api/sessions/overview/counts")
            assert counts.status_code == 503
            assert "total_count" not in counts.json()

    asyncio.run(scenario())


def test_counts_stop_at_upper_creation_boundary_before_enrichment(monkeypatch):
    async def scenario():
        app = _app()
        store = app.session_store

        async def pending(name):
            return await store.create(
                RunRequest(session_id=name, agent_name="writer", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )

        await pending("older-running")
        await _start(app, "inside", "writer")
        inside = await store.load("inside")
        boundary = await pending("boundary")
        await pending("newer")
        original = store.list_sessions
        load = store.load
        calls = []
        enriched = []

        async def growing_pages(query=None):
            calls.append(query.cursor)
            assert query.order_by is SessionOrder.CREATED_AT_ASC
            assert len(calls) <= 2, "must not chase sessions created beyond until"
            page = await original(query.model_copy(update={"limit": 2}))
            if query.cursor is not None:
                return page.model_copy(
                    update={
                        "next_cursor": encode_session_cursor(
                            page.sessions[-1], SessionOrder.CREATED_AT_ASC
                        )
                    }
                )
            return page

        async def bounded_load(session_id):
            enriched.append(session_id)
            assert session_id not in {"boundary", "newer"}, (
                "outside population must not be enriched"
            )
            return await load(session_id)

        monkeypatch.setattr(store, "list_sessions", growing_pages)
        monkeypatch.setattr(store, "load", bounded_load)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_server(app, config=ServerConfig.local_development()),
                raise_app_exceptions=False,
            ),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/sessions/overview/counts",
                params={
                    "since": inside.created_at.isoformat(),
                    "until": boundary.created_at.isoformat(),
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["total_count"] == "1"
            assert response.json()["running_now"] == "1"
            assert set(enriched) == {"older-running", "inside"}
            assert len(calls) == 2

    asyncio.run(scenario())
