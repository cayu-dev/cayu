"""Public trace cursor and concurrent native-store lifetime regressions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.core.test_tool_completion import FinalTool, app_for, call
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

from cayu import CayuApp, Message, RunRequest
from cayu.events import Event, EventType
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.storage import SQLiteSessionStore


def _store(tmp_path, backend):
    return (
        InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(tmp_path / "trace.db")
    )


def _request():
    return RunRequest(
        session_id="boundary", agent_name="support", messages=[Message.text("user", "read")]
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("populated", [False, True])
def test_trace_cursor_past_tail_or_requested_cutoff_is_empty(tmp_path, backend, populated):
    store = _store(tmp_path, backend)

    async def seed():
        await store.create(
            _request(), identity=SessionIdentity(provider_name="fake", model="fake-model")
        )
        if populated:
            await store.append_event(
                "boundary",
                Event(id="knowledge-event", type=EventType.KNOWLEDGE_LOOKUP, session_id="boundary"),
            )

    with TestClient(
        create_server(CayuApp(session_store=store), config=ServerConfig.local_development()),
        raise_server_exceptions=False,
    ) as client:
        client.portal.call(seed)
        for params in (
            {"after_sequence": 5},
            {"after_sequence": 5, "through_sequence": 0},
            {"after_sequence": 5, "through_sequence": 2},
        ):
            response = client.get("/api/sessions/boundary/trace", params=params)
            assert response.status_code == 200, response.text
            page = response.json()
            assert page["events"] == []
            assert page["has_more"] is False
            assert page["next_sequence"] == 5
            assert page["through_sequence"] == min(
                int(populated), params.get("through_sequence", 5)
            )
        if backend == "sqlite" and populated:

            async def prune():
                claim = await store.claim_persisted_event_side_effect(
                    session_id="boundary", event_id="knowledge-event"
                )
                assert claim is not None
                await store.mark_persisted_event_side_effect_delivered(claim)
                return await store.prune_events(
                    before=datetime.now(UTC) + timedelta(days=1), session_id="boundary"
                )

            removed = client.portal.call(prune)
            assert removed == 1
            response = client.get("/api/sessions/boundary/trace", params={"after_sequence": 1})
            assert response.status_code == 200, response.text
            assert response.json()["through_sequence"] == 0
            assert response.json()["events"] == []
            assert response.json()["next_sequence"] == 1
            assert response.json()["has_more"] is False


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("change", ["delete", "replace", "unrelated"])
def test_trace_transcript_read_reconciles_only_actual_incarnation_loss(tmp_path, backend, change):
    async def scenario():
        store = _store(tmp_path, backend)
        app = app_for(
            store,
            _TwoCallProvider(
                [call(), [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
            ),
            FinalTool(),
        )
        events = [event async for event in app.run(_request())]
        assert events[-1].type == "session.completed"
        entered, release = asyncio.Event(), asyncio.Event()
        original = store.query_transcript

        async def blocked(query):
            entered.set()
            await release.wait()
            if change == "unrelated":
                raise KeyError("private-unrelated-store-canary")
            return await original(query)

        store.query_transcript = blocked
        server = create_server(app, config=ServerConfig.local_development())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                reading = asyncio.create_task(client.get("/api/sessions/boundary/trace"))
                await asyncio.wait_for(entered.wait(), timeout=5)
                try:
                    if change != "unrelated":
                        await store.delete_session("boundary")
                    if change == "replace":
                        await store.create(
                            _request(),
                            identity=SessionIdentity(provider_name="fake", model="fake-model"),
                        )
                finally:
                    release.set()
                response = await reading
                assert response.status_code == (500 if change == "unrelated" else 409), (
                    response.text
                )
                assert "spans" not in response.text
                assert "private-unrelated-store-canary" not in response.text
                store.query_transcript = original
                fresh = await client.get("/api/sessions/boundary/trace")
                assert fresh.status_code == (404 if change == "delete" else 200)
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


def test_trace_cancellation_during_missing_record_reconciliation_stays_cancelled():
    async def scenario():
        store = InMemorySessionStore()
        app = app_for(
            store,
            _TwoCallProvider(
                [call(), [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
            ),
            FinalTool(),
        )
        events = [event async for event in app.run(_request())]
        assert events[-1].type == "session.completed"
        original_load, original_transcript = store.load, store.query_transcript
        entered = asyncio.Event()
        reconciling = False

        async def missing(query):
            nonlocal reconciling
            reconciling = True
            raise KeyError("missing-record")

        async def load(session_id):
            if reconciling:
                entered.set()
                await asyncio.Event().wait()
            return await original_load(session_id)

        store.query_transcript, store.load = missing, load
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            reading = asyncio.create_task(client.get("/api/sessions/boundary/trace"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert reading.cancel()
            assert reading.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await reading
            assert reading.cancelled()
            store.load, store.query_transcript = original_load, original_transcript
            assert (await client.get("/api/sessions/boundary/trace")).status_code == 200

    asyncio.run(scenario())
