"""Real Chromium admission: denied pixels never reach encoding or disk."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from tests.egress._browser_visual_container import run_visual_container

from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_RECORDING_BROWSER_ACCEPTANCE") != "1",
        reason="Set CAYU_RUN_RECORDING_BROWSER_ACCEPTANCE=1 with the pinned Docker browser.",
    ),
]

_PROGRAM = r"""
import asyncio
import sys
from types import SimpleNamespace
sys.path.insert(0, "/repo/src/cayu/tools")
from _browser_recording_guest import capture_recording_frame, RecordingCaptureDenied
from playwright.async_api import async_playwright

async def main():
    mode = sys.argv[1]
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            context = await browser.new_context(viewport={"width":1280,"height":720})
            html = '<html><body style="background:green"><h1>Public page</h1></body></html>'
            if mode == "password": html += '<input type="password" value="prohibited">'
            if mode == "dom-race": html += '<div id="entry"></div>'
            if mode == "iframe": html += '<iframe src="https://other.test/"></iframe>'
            if mode == "shadow": html += '<div id="host"></div><script>host.attachShadow({mode:"closed"}).innerHTML="<input type=password value=prohibited>"</script>'
            if mode == "session-storage": html += '<script>sessionStorage.setItem("token","prohibited")</script>'
            parser_release = asyncio.Event()
            if mode == "delayed-parser":
                html = '<html><body><h1>Public page</h1><script src="https://public.test/slow.js"></script><input type="password" value="prohibited"></body></html>'
            async def serve(route):
                if route.request.url.endswith("slow.js"):
                    await parser_release.wait()
                    await route.fulfill(status=200, content_type="text/javascript", body="")
                else:
                    await route.fulfill(status=200, content_type="text/html", body=html)
            await context.route("**/*", serve)
            page = await context.new_page()
            await page.goto("https://public.test/", wait_until="commit" if mode == "delayed-parser" else "load")
            await page.locator("h1").wait_for()
            if mode == "cookie": await context.add_cookies([{"name":"auth","value":"prohibited","url":"https://public.test/"}])
            cdp = await context.new_cdp_session(page)
            state = SimpleNamespace(page_id="page",page=page,cdp=cdp,lifecycle="active",configured=True,
                limit_exceeded=False,denied_code=None,access_evidence=None,navigation_epoch=0)
            daemon = SimpleNamespace(pages={"page":state},active_page_id="page",context=context,
                closing=False,close_requested=asyncio.Event(),profile_output_values=None,control=SimpleNamespace(capture_restricted=False,sensitive_entry=False))
            if mode == "sensitive": daemon.control.sensitive_entry=True
            if mode == "profile": daemon.profile_output_values=()
            count = 0
            original = cdp.send
            async def send(method, params=None):
                nonlocal count
                if method != "Page.captureScreenshot":
                    return await original(method, params)
                count += 1
                if mode in {"timer", "timer-cancel"}:
                    await asyncio.sleep(0.3)
                    pending = await original("Runtime.evaluate", {"expression": "window.fired", "returnByValue": True})
                    assert pending["result"]["value"] == 0
                    if mode == "timer-cancel":
                        asyncio.current_task().cancel()
                        await asyncio.sleep(0)
                if mode == "delayed-parser":
                    # Freezing scripts does not freeze the parser. Complete its
                    # pending resource after admission, before taking pixels.
                    parser_release.set()
                    await page.wait_for_load_state("load")
                if mode == "dom-race":
                    # Inject a native DOM change without enabling page scripts,
                    # proving the post-capture guard checks content as well as
                    # loader identity and the Runtime navigation counter.
                    tree = await cdp.send("DOM.getDocument")
                    entry = await cdp.send("DOM.querySelector", {"nodeId": tree["root"]["nodeId"], "selector": "#entry"})
                    await cdp.send("DOM.setOuterHTML", {"nodeId": entry["nodeId"], "outerHTML": '<input type="password">'})
                pixels = await original(method, params)
                if mode == "navigation-race": state.navigation_epoch += 1
                return pixels
            cdp.send = send
            if mode in {"timer", "timer-cancel"}:
                await page.evaluate("window.fired = 0; setTimeout(() => { window.fired += 1 }, 100)")
            policy = {"allowed_origins":["https://different.test" if mode == "origin" else "https://public.test"],
                      "max_width":1280,"max_height":720}
            try:
                page_id, pixels = await capture_recording_frame(daemon,policy)
                assert mode in {"allowed", "timer"}, mode
                assert page_id == "page" and pixels.startswith(b"\x89PNG")
            except RecordingCaptureDenied:
                assert mode not in {"allowed", "timer"}
                assert count == (1 if mode in {"navigation-race", "dom-race"} else 0), (mode,count)
            except asyncio.CancelledError:
                assert mode == "timer-cancel"
            if mode == "delayed-parser":
                parser_release.set()
                await page.wait_for_load_state("load")
                assert await page.locator("input[type=password]").count() == 1
            # Capture restores execution even after refusal.
            assert await page.evaluate("1+1") == 2
            if mode in {"timer", "timer-cancel"}:
                await page.wait_for_function("window.fired === 1")
                await asyncio.sleep(0.2)
                assert await page.evaluate("window.fired") == 1
        finally:
            await browser.close()
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "mode",
    [
        "allowed",
        "timer",
        "timer-cancel",
        "password",
        "iframe",
        "shadow",
        "cookie",
        "session-storage",
        "sensitive",
        "profile",
        "origin",
        "navigation-race",
        "delayed-parser",
        "dom-race",
    ],
)
def test_recording_admission_before_persistence(mode):
    root = Path(__file__).resolve().parents[2]
    run_visual_container(
        [
            "--rm",
            "--network",
            "none",
            "--volume",
            f"{root}:/repo:ro",
            PINNED_BROWSER_SESSION_WORKLOAD.image,
            "python3",
            "-c",
            _PROGRAM,
            mode,
        ]
    )
