"""Recording download authority is separate from login and capture authority."""

import asyncio

import httpx
from fastapi import FastAPI
from tests.core.test_browser_recording import png, prepared

from cayu.server.auth import BasicAuth
from cayu.server.browser_recording import BrowserRecordingServer


def test_recording_playback_and_download_have_distinct_permissions(tmp_path):
    async def scenario():
        store, policy, _, recording, owner = await prepared(tmp_path)
        await store.append(
            recording,
            policy=policy,
            owner=owner,
            sequence=0,
            page_id="page",
            elapsed_ms=0,
            png=png(),
        )
        await store.finish(recording, owner=owner)
        permissions = {"view": True, "download": False}

        async def authorize(principal, manifest, purpose):
            assert principal.subject == "operator" and manifest.recording_id == recording
            return permissions[purpose]

        app = FastAPI()
        server = BrowserRecordingServer(store=store, authorize=authorize)
        app.include_router(server.router(auth=BasicAuth(username="operator", password="password")))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://operator.test"
        ) as client:
            url = f"/browser-recordings/{recording}/segments/0"
            assert (await client.get(url)).status_code == 401
            client.auth = ("operator", "password")
            assert (await client.get("/browser-recordings/sessions/session")).json()[0][
                "recording_id"
            ] == recording
            viewed = await client.get(url)
            assert viewed.status_code == 200 and viewed.headers["cache-control"] == "no-store"
            assert (await client.get(url + "?download=true")).status_code == 404
            ranged = await client.get(url, headers={"Range": "bytes=0-3"})
            assert ranged.status_code == 206 and ranged.content == viewed.content[:4]
            permissions["view"] = False
            assert (await client.get(url)).status_code == 404
            assert (await client.get("/browser-recordings/sessions/session")).json() == []
            permissions["download"] = True
            downloaded = await client.get(url + "?download=true")
            assert (
                downloaded.status_code == 200
                and "attachment" in downloaded.headers["content-disposition"]
            )

    asyncio.run(scenario())


def test_recording_capability_matches_server_wiring(tmp_path):
    from fastapi.testclient import TestClient

    from cayu import CayuApp
    from cayu.server import ServerConfig, create_server

    async def deny(principal, manifest, purpose):
        return False

    store, _, _, _, _ = asyncio.run(prepared(tmp_path))
    recording_server = BrowserRecordingServer(store=store, authorize=deny)
    for configured in (False, True):
        server = create_server(
            CayuApp(),
            config=ServerConfig.protected(BasicAuth(username="operator", password="password")),
            browser_recordings=recording_server if configured else None,
        )
        with TestClient(server) as client:
            client.auth = ("operator", "password")
            contract = client.get("/api/contract").json()
            capability = contract["capabilities"]["surfaces"]["browser_recordings"]
            assert capability["configured"] is configured
            assert capability["read"]["enabled"] is configured
            assert capability["mutate"] == {
                "enabled": False,
                "unavailable_reason": "unsupported",
            }
            response = client.get("/api/browser-recordings/sessions/session")
            assert response.status_code == (200 if configured else 404)
            if configured:
                # Discovery does not bypass the per-recording access decision.
                assert response.json() == []
