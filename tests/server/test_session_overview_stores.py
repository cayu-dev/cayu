"""The new HTTP overview reconstructs the same facts on supported stores."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    _reopen_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _start

from cayu.cloud_reporting import read_cloud_activity
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]


@pytest.mark.store_semantics
def test_overview_survives_store_reconstruction(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "persisted", "asker")
            before = await read_cloud_activity(app)
            store = await _reopen_store(session_store_case, store)
            app = _app(store)
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get("/api/sessions", params={"include": "overview"})
                assert response.status_code == 200
                (row,) = response.json()["overviews"]
                assert row["run_id"] == before.runs[0].run_id
                assert row["state"] == "waiting_input"
                assert row["tokens"]["input"] == "1000"
                counts = await client.get("/api/sessions/overview/counts")
                assert counts.status_code == 200
                assert counts.json()["counts_by_state"]["waiting_input"] == "1"
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_id_batch_omits_absent_aliases_after_delete_and_reopen(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            for session_id in ("raw-kept", "alias-kept", "deleted"):
                await _start(app, session_id, "writer")
            codec = store.public_authority_alias_codec
            aliases = {
                name: codec.encode(name, field_name="session_id")
                for name in ("alias-kept", "deleted", "never-created")
            }
            for name in ("alias-kept", "deleted"):
                await store.register_public_authority_alias(
                    aliases[name], field_name="session_id", private_value=name
                )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                before = await client.get(
                    "/api/sessions", params={"ids": aliases["deleted"], "include": "overview"}
                )
                assert before.status_code == 200, before.text
                assert (
                    await client.delete(f"/api/sessions/{aliases['deleted']}")
                ).status_code == 204
            store = await _reopen_store(session_store_case, store)
            app = _app(store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                ids = [
                    aliases["never-created"],
                    "raw-kept",
                    aliases["alias-kept"],
                    "missing-raw",
                    aliases["deleted"],
                ]
                for selected in (ids, list(reversed(ids))):
                    for includes in ([], [("include", "overview")]):
                        response = await client.get(
                            "/api/sessions",
                            params=[*(("ids", value) for value in selected), *includes],
                        )
                        assert response.status_code == 200, response.text
                        body = response.json()
                        assert {row["id"] for row in body["sessions"]} == {"raw-kept", "alias-kept"}
                        assert body["total_count"] == 2
                        if includes:
                            assert {row["session_id"] for row in body["overviews"]} == {
                                "raw-kept",
                                "alias-kept",
                            }
                before_reuse = await client.get(
                    "/api/sessions", params={"ids": aliases["alias-kept"], "include": "overview"}
                )
                old_run = before_reuse.json()["overviews"][0]["run_id"]
                # Native deletion permits reuse; HTTP closure intentionally retains its fence.
                await store.delete_session("alias-kept")
                await _start(app, "alias-kept", "writer")
                await store.register_public_authority_alias(
                    aliases["alias-kept"], field_name="session_id", private_value="alias-kept"
                )
                replacement = await client.get(
                    "/api/sessions", params={"ids": aliases["alias-kept"], "include": "overview"}
                )
                assert replacement.status_code == 200, replacement.text
                assert replacement.json()["overviews"][0]["run_id"] != old_run
                # A raw identity colliding with a registered alias remains ambiguous.
                await store.create(
                    RunRequest(session_id=aliases["alias-kept"], agent_name="writer", messages=[]),
                    identity=SessionIdentity(provider_name="scripted", model="test"),
                )
                for invalid in ("cayu_authority_malformed", aliases["alias-kept"]):
                    rejected = await client.get(
                        "/api/sessions",
                        params=[("ids", "raw-kept"), ("ids", invalid), ("include", "overview")],
                    )
                    assert rejected.status_code == 409, rejected.text
                    assert "sessions" not in rejected.json()
                assert "ambiguous" in rejected.json()["detail"]
        finally:
            await _close_store(store)

    asyncio.run(scenario())
