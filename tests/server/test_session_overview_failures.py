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
@pytest.mark.parametrize("read_kind", ["browser", "outcomes"])
def test_browser_enrichment_reconciles_only_confirmed_identity_loss(
    session_store_case, monkeypatch, mode, read_kind
):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "gone", "writer")
            await _start(app, "kept", "writer")
            method = "load_session_operations" if read_kind == "browser" else "summarize_outcomes"
            original = getattr(store, method)
            reached = asyncio.Event()
            release = asyncio.Event()

            # The page reads every row's browser receipt at once.
            async def paused_read(session_ids, *args, **kwargs):
                if "gone" in session_ids:
                    reached.set()
                    await release.wait()
                    if mode == "unrelated":
                        raise KeyError("unrelated-store-failure")
                    if read_kind == "outcomes":
                        raise KeyError("Session not found: gone")
                return await original(session_ids, *args, **kwargs)

            monkeypatch.setattr(store, method, paused_read)
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


@pytest.mark.store_semantics
@pytest.mark.parametrize("mode", ["delete", "replace"])
@pytest.mark.parametrize(
    # One read before each fence: the reader's, the projection's and the route's.
    "method",
    ["query_latest_events", "count_child_runs", "query_first_transcript_records"],
)
def test_overview_drops_a_session_replaced_after_it_was_listed(
    session_store_case, monkeypatch, mode, method
):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "gone", "writer")
            await _start(app, "kept", "writer")
            original = getattr(store, method)
            reached = asyncio.Event()
            release = asyncio.Event()

            async def paused_read(sessions, *args, **kwargs):
                reached.set()
                await release.wait()
                return await original(sessions, *args, **kwargs)

            monkeypatch.setattr(store, method, paused_read)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                request = asyncio.create_task(
                    client.get("/api/sessions", params={"include": ["overview", "first_message"]})
                )
                await asyncio.wait_for(reached.wait(), 3)
                await store.delete_session("gone")
                if mode == "replace":
                    await _start(app, "gone", "writer")
                release.set()
                response = await request
            assert response.status_code == 200, response.text
            assert [row["id"] for row in response.json()["sessions"]] == ["kept"]
            assert [row["session_id"] for row in response.json()["overviews"]] == ["kept"]
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_listed_sessions_are_summarized_only_while_still_current():
    from cayu.sessions.overview import SessionOverviewReader

    async def scenario():
        app = _app()
        store = app.session_store
        await _start(app, "replaced", "writer")
        await _start(app, "kept", "writer")
        listed = (await store.list_sessions()).sessions
        replaced = next(session for session in listed if session.id == "replaced")
        await store.delete_session("replaced")
        await _start(app, "replaced", "writer")
        rows = await SessionOverviewReader(store).read_listed(listed)
        assert [row.session.id for row in rows] == ["kept"]
        assert rows[0].session is next(session for session in listed if session.id == "kept")
        # Read by ID, the session's new incarnation is summarized instead.
        (current,) = await SessionOverviewReader(store).read(("replaced",))
        assert current.session.instance_id != replaced.instance_id

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

        original_page = store.query_pending_actions_for_sessions
        flagged = "leaf" if delegated else "writer"

        async def issues(session_ids):
            sessions = [await store.load(session_id) for session_id in session_ids]
            return [
                PendingActionIssue.source_invalid(PendingActionSession.from_session(session))
                for session in sessions
                if session is not None and session.agent_name == flagged
            ]

        async def incomplete(query=None, **kwargs):
            page = await original(query, **kwargs)
            if flagged_issues := await issues([query.session_id]):
                return page.model_copy(update={"issues": flagged_issues})
            return page

        async def incomplete_page(session_ids, query=None, **kwargs):
            page = await original_page(session_ids, query, **kwargs)
            if flagged_issues := await issues(session_ids):
                return page.model_copy(update={"issues": [*page.issues, *flagged_issues]})
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
            monkeypatch.setattr(store, "query_pending_actions_for_sessions", incomplete_page)
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


@pytest.mark.store_semantics
def test_pending_pagination_rechecks_sessions_that_move_before_its_cursor(
    session_store_case, monkeypatch
):
    from cayu.sessions.overview import SessionOverviewReader, SessionOverviewStatus

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "older", "approver")
            await _start(app, "newer", "approver")
            original = store.query_pending_actions_for_sessions
            moved = False

            async def move_after_first_page(session_ids, query, **kwargs):
                nonlocal moved
                page = await original(session_ids, query.model_copy(update={"limit": 1}), **kwargs)
                if not moved and page.has_more:
                    moved = True
                    seen = {action.session.id for action in page.actions}
                    remaining = next(sid for sid in session_ids if sid not in seen)
                    await store.update_metadata(remaining, {"changed_during_page": True})
                return page

            monkeypatch.setattr(store, "query_pending_actions_for_sessions", move_after_first_page)
            rows = await SessionOverviewReader(store).read(("older", "newer"))
            assert moved
            assert len(rows) == 2
            assert all(row.status is SessionOverviewStatus.WAITING_APPROVAL for row in rows)
            assert all(row.pending_complete for row in rows)
        finally:
            await _close_store(store)

    asyncio.run(scenario())
