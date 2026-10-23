"""Opt-in actual-browser acceptance for bounded public Sessions search."""

from __future__ import annotations

import asyncio
import os
import socket
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("playwright.async_api")

import uvicorn
from playwright.async_api import async_playwright, expect
from tests.core.test_cloud_reporting import _app

from cayu import Message
from cayu.server import DashboardConfig, ServerConfig, create_server
from cayu.sessions import RunRequest, SessionIdentity

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_SESSION_SEARCH_LIVE") != "1", reason="Opt-in packaged Chromium acceptance."
)


@asynccontextmanager
async def search_browser(*, pause=False):
    runtime = _app()
    for name in ["old-needle", *(f"filler-{index}" for index in range(101)), "fresh-match"]:
        await runtime.session_store.create(
            RunRequest(session_id=name, agent_name="writer", messages=[Message.text("user", "go")]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
    app = create_server(
        runtime,
        config=ServerConfig.local_development(
            dashboard=DashboardConfig(directory=os.environ.get("CAYU_SESSION_SEARCH_DIRECTORY"))
        ),
    )
    reached, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.middleware("http")
    async def observe(request, call_next):
        if request.url.path == "/api/sessions/summary":
            requests.append(dict(request.query_params))
            if (
                pause
                and request.query_params.get("q") == "old-needle"
                and request.query_params.get("cursor")
            ):
                reached.set()
                await release.wait()
        return await call_next(request)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        server = uvicorn.Server(uvicorn.Config(app, lifespan="on", log_level="error"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    await asyncio.sleep(0.01)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
                )
                try:
                    page = await browser.new_page()
                    page.set_default_timeout(5_000)
                    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
                    yield page, base, requests, reached, release
                finally:
                    release.set()
                    await browser.close()
        finally:
            release.set()
            server.should_exit = True
            await asyncio.wait_for(task, 10)


def test_session_search_finds_match_beyond_first_hundred_and_proves_exhaustion():
    async def run():
        async with search_browser() as (page, base, requests, _, _):
            await page.goto(f"{base}/cayu/sessions?q=old-needle")
            await expect(page.get_by_role("link", name="old-needle", exact=True)).to_be_visible()
            assert len([row for row in requests if row.get("q") == "old-needle"]) == 2
            await page.get_by_placeholder(
                "Search sessions, agents, providers, models, labels...", exact=True
            ).fill("absent")
            await expect(
                page.get_by_text("No sessions match the current filters.", exact=True)
            ).to_be_visible()
            assert len([row for row in requests if row.get("q") == "absent"]) == 2
            assert parse_qs(urlsplit(page.url).query)["q"] == ["absent"]

    asyncio.run(run())


def test_query_change_cancels_empty_page_search_and_ignores_late_old_result():
    async def run():
        async with search_browser(pause=True) as (page, base, requests, reached, release):
            failed = []
            page.on("requestfailed", lambda request: failed.append(request.url))
            await page.goto(f"{base}/cayu/sessions?q=old-needle")
            await asyncio.wait_for(reached.wait(), 5)
            await expect(
                page.get_by_text("Searching sessions...", exact=True).first
            ).to_be_visible()
            await expect(
                page.get_by_text("No sessions match the current filters.", exact=True)
            ).to_have_count(0)
            await page.get_by_placeholder(
                "Search sessions, agents, providers, models, labels...", exact=True
            ).fill("fresh-match")
            await expect(page.get_by_role("link", name="fresh-match", exact=True)).to_be_visible()
            async with asyncio.timeout(3):
                while not any("cursor=" in url and "q=old-needle" in url for url in failed):
                    await asyncio.sleep(0.01)
            release.set()
            await page.wait_for_timeout(200)
            await expect(page.get_by_role("link", name="old-needle", exact=True)).to_have_count(0)
            assert len([row for row in requests if row.get("q") == "old-needle"]) == 2
            assert parse_qs(urlsplit(page.url).query)["q"] == ["fresh-match"]

    asyncio.run(run())
