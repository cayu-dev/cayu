"""Opt-in browser acceptance for the registered dashboard navigation boundary."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from urllib.parse import parse_qs, quote, urlsplit

import pytest
from tests.server.test_dashboard_session_search import search_browser
from tests.server.test_server_inbox import _app

from cayu import Message
from cayu.sessions import RunRequest, SessionIdentity
from cayu.tasks import InMemoryTaskStore, TaskCreate

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_SESSION_SEARCH_LIVE") != "1",
    reason="Opt-in packaged Chromium navigation acceptance.",
)


def test_registered_links_keep_dynamic_identity_nested_mount_and_browser_navigation(monkeypatch):
    from fastapi import FastAPI
    from playwright.async_api import expect
    from sse_starlette.sse import AppStatus

    monkeypatch.setattr(AppStatus, "should_exit", False)

    def nested(app):
        parent = FastAPI()
        parent.mount("/product", app)
        return parent

    async def run():
        session_id = "session café?100%"
        task_id = "task café?100%"
        tasks = InMemoryTaskStore()
        runtime = _app(task_store=tasks)
        await runtime.session_store.create(
            RunRequest(
                session_id=session_id,
                agent_name="writer",
                title="Encoded navigation session",
                messages=[Message.text("user", "Inspect this session")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        slash_id = "session/encoded%2Ftail"
        await runtime.session_store.create(
            RunRequest(
                session_id=slash_id, agent_name="writer", title="Slash in identity", messages=[]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await tasks.create_task(TaskCreate(task_id=task_id, type="work", title="Encoded task"))
        try:
            async with search_browser(runtime=runtime, wrapper=nested) as (page, base, _, _, _):
                prefix = base + "/product/cayu"
                session_path = "/product/cayu/sessions/" + quote(session_id, safe="")
                task_path = "/product/cayu/tasks/" + quote(task_id, safe="")
                await page.goto(prefix + "/sessions")
                link = page.get_by_role("link", name="Encoded navigation session", exact=True)
                await expect(link).to_have_attribute("href", session_path)
                await expect(
                    page.get_by_role("link", name="Slash in identity", exact=True)
                ).to_have_attribute("href", "/product/cayu/sessions/" + quote(slash_id, safe=""))
                list_url = page.url
                async with page.context.expect_page() as opened:
                    await link.click(modifiers=["ControlOrMeta"])
                child = await opened.value
                await child.wait_for_url(base + session_path)
                assert page.url == list_url
                await child.close()
                # Modified row clicks belong to the browser; plain row clicks navigate.
                row = link.locator("xpath=ancestor::tr")
                await row.locator("td").last.click(modifiers=["ControlOrMeta"])
                assert page.url == list_url
                await row.locator("td").last.click()
                await expect(page).to_have_url(base + session_path)
                await expect(
                    page.get_by_role("tab", name="Conversation", exact=True)
                ).to_be_visible()
                await page.go_back()
                await expect(link).to_be_visible()
                await link.focus()
                await page.keyboard.press("Enter")
                await expect(page).to_have_url(base + session_path)

                await page.goto(prefix + "/tasks")
                task_link = page.get_by_role("link", name="Encoded task", exact=True)
                await expect(task_link).to_have_attribute("href", task_path)
                await task_link.click()
                await expect(page).to_have_url(base + task_path)

                await page.goto(prefix + "/tasks?task_id=" + quote(task_id, safe=""))
                await expect(page).to_have_url(base + task_path)
                await page.goto(base + session_path + "/workflow?event_type=session.failed")
                await page.wait_for_url(
                    lambda url: (
                        urlsplit(url).path == session_path and "view=trace" in urlsplit(url).query
                    )
                )
                query = parse_qs(urlsplit(page.url).query)
                assert query["view"] == ["trace"]
                assert query["workflow"] == ["1"]
                await expect(page.get_by_role("tab", name="Trace", exact=True)).to_be_visible()

                run_selector = (
                    base64.urlsafe_b64encode(b'["run","legacy-run"]').decode().rstrip("=")
                )
                await page.goto(prefix + "/evals?run=legacy-run")
                await expect(page).to_have_url(prefix + "/evals/runs/" + run_selector)
                revision = "sha256:" + "a" * 64
                identity = [
                    "suite",
                    {
                        "source": "corpus",
                        "target_key": "writer",
                        "revision": revision,
                        "suite_id": "suite",
                    },
                ]
                suite_selector = (
                    base64.urlsafe_b64encode(json.dumps(identity, separators=(",", ":")).encode())
                    .decode()
                    .rstrip("=")
                )
                await page.goto(prefix + "/evals?corpus=" + revision + "&suite=suite&target=writer")
                await expect(page).to_have_url(prefix + "/evals/suites/" + suite_selector)

                for old, new in [
                    ("/pending-actions", "/inbox"),
                    ("/agents?q=writer", "/specialists?q=writer"),
                    ("/run", "/?start=1"),
                ]:
                    await page.goto(prefix + old)
                    await expect(page).to_have_url(prefix + new)
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_retained_prototype_uses_its_own_navigation_search_contract():
    import functools
    import http.server
    import threading
    from pathlib import Path

    from playwright.async_api import async_playwright, expect

    directory = Path(__file__).resolve().parents[2] / "dashboard" / "dist" / "prototype"
    if not (directory / "prototype.html").is_file():
        pytest.skip("Build the retained prototype before its browser acceptance.")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(directory))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def run():
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
            )
            try:
                page = await browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                base = f"http://127.0.0.1:{server.server_port}/prototype.html"
                await page.goto(base + "#/evals")
                suite = page.locator('a[href*="#/evals/suites/"]').first
                await expect(suite).to_be_visible()
                await suite.click()
                await page.wait_for_url("**#/evals/suites/*")
                run_link = page.locator('table a[href*="#/evals/runs/"]').first
                await expect(run_link).to_be_visible()
                await run_link.click()
                await page.wait_for_url("**#/evals/runs/*")
                case_filter = page.get_by_role(
                    "button", name=re.compile(r"^(Regressed|Improved|Failing|Flaky) ")
                ).first
                await expect(case_filter).to_be_visible()
                filter_name = (await case_filter.inner_text()).split()[0].lower()
                await case_filter.click()
                await page.wait_for_url(lambda url: f"filter={filter_name}" in url)
                await page.goto(base + "#/pending-actions")
                await page.wait_for_url("**#/inbox")
                assert not errors
            finally:
                await browser.close()

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
