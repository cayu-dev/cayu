from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from tests.core.test_startup_recovery_isolation import _app, _seed, _state

from cayu import InMemorySessionStore, SessionStatus
from cayu.server import (
    AuthContext,
    AuthenticatedAccess,
    DashboardConfig,
    ServerConfig,
    ServerLifecycleConfig,
    create_server,
    mount_cayu,
)


async def _authenticate(request):
    if request.headers.get("Authorization") != "Bearer startup-test-operator":
        raise HTTPException(status_code=401)
    return AuthContext(subject="startup-operator")


def _server(app, mode):
    if mode == "mounted":
        server = FastAPI()
        mount_cayu(
            server,
            app,
            dashboard=False,
            access=AuthenticatedAccess(dependency=_authenticate),
            interruption_recovery_inactive_after_seconds=0,
        )
        return server, "/cayu/api"
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTING},
                recovery_inactive_after_seconds=0,
            ),
        ),
    )
    return server, "/api"


@pytest.mark.parametrize("mode", ["mounted", "standalone"])
def test_server_starts_reports_blocked_root_and_recovers_later_roots(mode):
    async def scenario():
        store = InMemorySessionStore()
        old = _app(store, "3")
        replacement = _app(store, "4")
        await _seed(store, old, "blocked-startup")
        await _seed(store, replacement, "later-startup-interrupting")
        await _seed(store, replacement, "later-startup-interrupted", SessionStatus.INTERRUPTED)
        before = await _state(store, "blocked-startup")
        server, prefix = _server(replacement, mode)
        async with server.router.lifespan_context(server):
            assert await replacement.drain_background_interruptions(timeout_s=5)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://localhost"
            ) as client:
                assert (await client.get(prefix + "/recovery/startup")).status_code == 401
                response = await client.get(
                    prefix + "/recovery/startup",
                    headers={"Authorization": "Bearer startup-test-operator"},
                )
                assert response.status_code == 200
                result = response.json()
                assert result["completed"]
                assert result["blocked_sessions"] == [
                    {
                        "session_id": "blocked-startup",
                        "blocker_codes": ["registration_incompatible"],
                    }
                ]
                assert "startup-private-canary" not in response.text
            assert before == await _state(store, "blocked-startup")
            for session_id in ("later-startup-interrupting", "later-startup-interrupted"):
                assert (await store.load(session_id)).status == SessionStatus.INTERRUPTED
                assert "pending_interruption_cascade" not in await store.load_checkpoint(session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["mounted", "standalone"])
def test_store_discovery_failure_still_fails_server_startup(mode):
    class BrokenStore(InMemorySessionStore):
        async def list_sessions_with_pending_interruption_cascade(self, query=None):
            raise OSError("store is unavailable")

    async def scenario():
        app = _app(BrokenStore(), "3")
        server, _prefix = _server(app, mode)
        with pytest.raises(OSError, match="store is unavailable"):
            async with server.router.lifespan_context(server):
                pytest.fail("Unavailable store made the server ready")

    asyncio.run(scenario())
