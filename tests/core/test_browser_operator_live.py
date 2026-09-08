"""Opt-in credential-free native screenshot/input proof with an isolated browser.

This does not replace the final protected HTTP/WSS-to-browser acceptance campaign.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace

import pytest
from tests.core.test_browser_control_guest import takeover_material
from tests.core.test_browser_session import _interactive_request

from cayu.tools._browser_control_guest import GuestControlFailure
from cayu.tools._browser_guest import (
    _GuestFailure,
    _InteractiveDaemon,
    _InteractivePage,
    _wait_for_interactive_shutdown,
)


@pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1",
    reason="Opt-in isolated local browser smoke test.",
)
def test_real_viewport_frame_and_sensitive_input():
    from playwright.async_api import async_playwright

    async def scenario():
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"),
                headless=True,
            )
            idle_watch = None
            try:
                context = await browser.new_context(viewport={"width": 320, "height": 240})
                page = await context.new_page()
                await context.route(
                    "https://manual.test/**",
                    lambda route: route.fulfill(
                        body="""<form onsubmit="event.preventDefault();
                          if (!document.body.dataset.mfa) {
                            document.body.dataset.mfa='required';
                            document.getElementById('code').value='';
                            document.getElementById('code').focus();
                          } else { document.body.dataset.submitted='yes'; }
                        "><label>Code <input id='code' type='password'></label>
                        <button>Continue</button></form>""",
                        content_type="text/html",
                    ),
                )
                await page.goto("https://manual.test/")
                daemon = _InteractiveDaemon("bs_live")
                daemon.context = context
                daemon.configuration_limits = _interactive_request("observe").limits
                daemon.pages["page"] = _InteractivePage(
                    page=page,
                    session_id="bs_live",
                    page_id="page",
                    lifecycle="active",
                    revision="observed",
                    configured=True,
                )
                daemon.active_page_id = "page"
                daemon.pages["page"].cdp = await context.new_cdp_session(page)
                daemon.claim_operator_channel("private-channel")
                await daemon.bind_operator_control("a" * 64)
                material = takeover_material(daemon)
                await daemon.acquire_operator_control(**material)
                daemon.last_activity = (
                    asyncio.get_running_loop().time() - daemon.idle_timeout_seconds - 1
                )
                idle_watch = asyncio.create_task(_wait_for_interactive_shutdown(daemon))
                await asyncio.sleep(0)
                assert not idle_watch.done()
                view_id = "bv_" + "a" * 32
                await daemon.grant_operator_view(
                    binding_sha256="a" * 64,
                    view_id=view_id,
                    epoch=2,
                    until_ms=int(time.time() * 1000) + 10_000,
                )
                frames = []

                async def send(generation, metadata, frame):
                    frames.append((generation, metadata, frame))

                await daemon.capture_operator_frame(
                    view_id=view_id, epoch=2, page_id="page", page_epoch=1, send=send
                )
                assert len(frames) == 1
                assert frames[0][1]["width"] == 320
                assert frames[0][1]["height"] == 240
                assert frames[0][2].startswith(b"\x89PNG\r\n\x1a\n")
                assert not daemon.operations and daemon.total_artifacts == 0
                await daemon.enter_operator_sensitive_entry(
                    request_id=material["request_id"], epoch=2
                )
                await daemon.operator_key_input(
                    request_id=material["request_id"],
                    epoch=2,
                    sequence=1,
                    page_id="page",
                    page_epoch=1,
                    key="tab",
                )
                result = await daemon.operator_text_input(
                    request_id=material["request_id"],
                    epoch=2,
                    sequence=2,
                    page_id="page",
                    page_epoch=1,
                    text="local-only-canary",
                )
                assert await page.locator("#code").input_value() == "local-only-canary"
                assert "canary" not in str(result)
                await daemon.operator_key_input(
                    request_id=material["request_id"],
                    epoch=2,
                    sequence=3,
                    page_id="page",
                    page_epoch=1,
                    key="enter",
                )
                assert await page.locator("body").get_attribute("data-mfa") == "required"
                assert await page.locator("body").get_attribute("data-submitted") is None
                assert await page.locator("#code").input_value() == ""
                mfa_result = await daemon.operator_text_input(
                    request_id=material["request_id"],
                    epoch=2,
                    sequence=4,
                    page_id="page",
                    page_epoch=1,
                    text="local-mfa-canary",
                )
                assert "canary" not in str(mfa_result)
                await daemon.operator_key_input(
                    request_id=material["request_id"],
                    epoch=2,
                    sequence=5,
                    page_id="page",
                    page_epoch=1,
                    key="enter",
                )
                assert await page.locator("body").get_attribute("data-submitted") == "yes"
                with pytest.raises(GuestControlFailure):
                    await daemon.capture_operator_frame(
                        view_id=view_id,
                        epoch=2,
                        page_id="page",
                        page_epoch=daemon.pages["page"].control_epoch,
                        send=send,
                    )
                assert len(frames) == 1
                await page.evaluate(
                    "document.title = 'local-only-canary'; localStorage.setItem('session', 'storage-only-canary')"
                )
                handed_back = await daemon.handback_operator_control(
                    request_id=material["request_id"], epoch=2
                )
                assert handed_back["control_epoch"] == 3
                assert not idle_watch.done() and not daemon.close_requested.is_set()
                assert daemon.control.fresh_observation_required
                request = replace(
                    _interactive_request("observe"),
                    session_id="bs_live",
                    page_id="page",
                    invocation_control_epoch=3,
                )
                # The actual guest dispatch gate rejects stale operator/model
                # authority and actions before a protected observation settles.
                with pytest.raises(GuestControlFailure):
                    await daemon.operator_text_input(
                        request_id=material["request_id"],
                        epoch=2,
                        sequence=6,
                        page_id="page",
                        page_epoch=1,
                        text="must-not-arrive",
                    )
                assert await page.locator("#code").input_value() == "local-mfa-canary"
                with pytest.raises(_GuestFailure):
                    await daemon.execute(replace(request, invocation_control_epoch=2))
                with pytest.raises(_GuestFailure):
                    await daemon.execute(
                        replace(request, operation="click", operation_id="blocked")
                    )
                assert not daemon.operations
                response = await daemon.execute(request)
                assert response["kind"] == "success", response
                assert response["profile_output_protected"] is True
                assert not daemon.control.fresh_observation_required
                assert daemon.operations[request.operation_id].response == response
                observation = response["observation"]
                assert observation["title"] is None and observation["refs"] == []
                assert "local-only-canary" not in str(response)
                assert "storage-only-canary" not in str(response)
                assert "local-mfa-canary" not in str(response)
                observation_count = daemon.pages["page"].observation_count
                assert await daemon.execute(request) == response
                assert daemon.pages["page"].observation_count == observation_count
                assert daemon.profile_allowed_origins is None
                assert "local-only-canary" not in str(daemon._page_set_payload())
                closed = await daemon.execute(
                    replace(
                        _interactive_request("close"),
                        session_id="bs_live",
                        invocation_control_epoch=3,
                    )
                )
                assert closed["kind"] == "closed", closed
                assert closed["allocation_disposition"] == "retired"
                assert daemon.control.state == "closed" and page.is_closed()
                assert "canary" not in repr(closed)
            finally:
                try:
                    if idle_watch is not None:
                        daemon.close_requested.set()
                        async with asyncio.timeout(2):
                            await idle_watch
                finally:
                    await browser.close()

    asyncio.run(scenario())
