"""Cloud registration is owned by both supported server lifetimes."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from cayu import CayuApp
from cayu.cloud_inventory import CLOUD_REGISTRY_CONTRACT, CLOUD_REGISTRY_TOKEN_ENV
from cayu.releases import ReleaseIdentity
from cayu.server import OpenAccess, ServerConfig, create_server, mount_cayu


@pytest.mark.parametrize("mounted", [False, True])
def test_server_registers_on_start_and_cancels_reporter_before_close(monkeypatch, mounted):
    monkeypatch.setenv(CLOUD_REGISTRY_TOKEN_ENV, "cayu_reg_registry-only-token")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://cloud.example")
    monkeypatch.setenv("CAYU_CLOUD_APPLICATION_ID", "app-1")
    original_client = httpx.AsyncClient

    async def scenario():
        registered = asyncio.Event()
        clients = []
        requests = []

        async def receive(request):
            requests.append(request)
            assert request.method == "PUT"
            assert request.url.path == "/v1/applications/app-1/agent-registry/releases/release-test"
            assert request.headers["authorization"] == "Bearer cayu_reg_registry-only-token"
            assert json.loads(request.content)["contract"] == CLOUD_REGISTRY_CONTRACT
            registered.set()
            return httpx.Response(200, json={"usage_cursor": 0})

        def client(**options):
            assert options == {"timeout": 10.0, "trust_env": False}
            value = original_client(transport=httpx.MockTransport(receive), **options)
            clients.append(value)
            return value

        monkeypatch.setattr("cayu.cloud_inventory.httpx.AsyncClient", client)
        app = CayuApp(release_identity=ReleaseIdentity(id="release-test"), enable_logging=False)
        if mounted:
            server = FastAPI()
            mount_cayu(server, app, access=OpenAccess(), dashboard=False)
        else:
            server = create_server(app, config=ServerConfig.local_development())
        async with server.router.lifespan_context(server):
            await asyncio.wait_for(registered.wait(), timeout=10)
            reporters = [
                task for task in asyncio.all_tasks() if task.get_name() == "cayu-cloud-inventory"
            ]
            assert len(reporters) == 1
        assert reporters[0].done()
        assert clients[0].is_closed
        assert app.lifecycle_state == "closed"
        assert len(requests) == 1

    asyncio.run(scenario())
