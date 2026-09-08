"""Native page navigation during private-frame pacing stays caller-local."""

import asyncio
import json
import os
import time

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_frames import BrowserViewUnavailable
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime.browser_control import BrowserControlPage, BrowserControlPrincipal
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage


@pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1",
    reason="Opt-in isolated native browser regression.",
)
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_navigation_during_pacing_refuses_only_the_stale_view(tmp_path, monkeypatch, backend):
    from playwright.async_api import async_playwright

    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            before = await store.load_checkpoint(record.identity.session_id)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
                )
                release = asyncio.Event()
                pending = []
                try:
                    context = await browser.new_context(viewport={"width": 320, "height": 240})
                    await context.route(
                        "https://pacing.test/**",
                        lambda route: route.fulfill(
                            body="<p>Local navigation fixture</p>", content_type="text/html"
                        ),
                    )
                    page = await context.new_page()
                    await page.goto("https://pacing.test/first")
                    daemon = _InteractiveDaemon(record.identity.browser_session_id)
                    daemon.context = context
                    daemon.visual_worker_instance = record.identity.worker_instance_id
                    daemon.control = GuestControlFence(
                        worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
                    )
                    state = _InteractivePage(
                        page=page,
                        session_id=daemon.session_id,
                        page_id="page",
                        lifecycle="active",
                        revision="revision",
                        configured=True,
                    )
                    daemon.pages["page"] = state
                    daemon.active_page_id = "page"
                    page.on(
                        "framenavigated", lambda frame: daemon._mark_page_navigated(state, frame)
                    )
                    channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
                    channel._binding = "b" * 64
                    daemon.claim_operator_channel(channel._nonce)
                    await daemon.bind_operator_control("b" * 64)
                    replies = asyncio.Queue()

                    class Sink:
                        async def send(self, value):
                            await replies.put(value)

                    class Connection:
                        async def send(self, raw):
                            result = await channel._command(json.loads(raw), Sink())
                            await replies.put(
                                json.dumps(
                                    {
                                        "kind": "settled",
                                        "channel_id": channel._nonce,
                                        "sequence": channel._sequence,
                                        **result,
                                    }
                                )
                            )

                        async def recv(self):
                            return await replies.get()

                        async def recv_frame(self):
                            return await replies.get()

                    owner = BrowserGuestCommandOwner(
                        coordinator=coordinator(store, Policy(True)),
                        connection=Connection(),
                        bound=BoundBrowserGuest(record, channel._nonce, "b" * 64),
                    )
                    waiting = asyncio.Event()
                    sleep = asyncio.sleep
                    screenshot = page.screenshot
                    screenshots = []

                    async def held_pacing(delay, *args, **kwargs):
                        if 0 < delay <= 0.5 and not waiting.is_set():
                            waiting.set()
                            await release.wait()
                        return await sleep(delay, *args, **kwargs)

                    async def observed_screenshot(**kwargs):
                        screenshots.append(1)
                        return await screenshot(**kwargs)

                    monkeypatch.setattr(asyncio, "sleep", held_pacing)
                    monkeypatch.setattr(page, "screenshot", observed_screenshot)

                    async def request_view():
                        return await owner.request_view(
                            principal=BrowserControlPrincipal(subject="operator"),
                            operator_session_id="continuity",
                            page=BrowserControlPage(
                                page_id="page",
                                revision="revision",
                                control_epoch=state.control_epoch,
                            ),
                            until_ms=4000,
                        )

                    daemon._operator_frame_started = time.monotonic()
                    stale = asyncio.create_task(request_view())
                    pending.append(stale)
                    await sleep(0)
                    drive = asyncio.create_task(owner.step())
                    pending.append(drive)
                    await asyncio.wait_for(waiting.wait(), 3)
                    assert daemon.operator_frames.task is None
                    await page.goto("https://pacing.test/next")
                    assert state.control_epoch == 2
                    release.set()
                    await drive
                    with pytest.raises(BrowserViewUnavailable):
                        await stale
                    assert screenshots == [] and daemon._operator_frame_count == 0
                    assert not owner._closed and daemon.operator_frames.task is None
                    assert daemon.control.state == "agent_controlled"
                    await owner.step()  # Shared control still handles its heartbeat.
                    fresh = asyncio.create_task(request_view())
                    pending.append(fresh)
                    await sleep(0)
                    await owner.step()
                    assert (await fresh).png.startswith(b"\x89PNG")
                    assert screenshots == [1]
                    assert await store.load_checkpoint(record.identity.session_id) == before
                    assert await owner.coordinator.drain()
                finally:
                    release.set()
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    await browser.close()

    asyncio.run(scenario())
