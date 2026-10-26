"""Rendered consistency of the recently ported dashboard read controls."""

import asyncio
import os
import re

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_KNOWLEDGE_LIVE") != "1",
    reason="Opt-in real-server dashboard design acceptance.",
)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_read_controls_share_rendered_style_and_remain_keyboard_operable(theme):
    async def scenario():
        async with _home_browser(knowledge=True, decisions=True) as (page, base, _, _responses):
            await page.emulate_media(color_scheme=theme)
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            reference = page.get_by_role("button", name="Refresh proposals", exact=True)

            async def appearance(button):
                await expect(button).to_be_visible()
                await page.mouse.move(0, 0)
                return await button.evaluate(
                    """e => { const s = getComputedStyle(e); return {
                    height: e.getBoundingClientRect().height, radius: s.borderRadius,
                    border: s.borderColor, background: s.backgroundColor,
                    weight: s.fontWeight, font: s.fontSize, padding: s.padding
                    }}"""
                )

            expected = await appearance(reference)
            differences = []

            async def check(name):
                button = page.get_by_role("button", name=name, exact=True)
                actual = await appearance(button)
                if actual != expected:
                    differences.append(f"{name}: {actual!r} != {expected!r}")
                await button.focus()
                await expect(button).to_be_focused()
                await button.blur()

            await check("Refresh knowledge")
            await page.goto(base + "/cayu/inbox", wait_until="networkidle")
            await check("Refresh actions")
            await check("Load more")
            refresh = await page.get_by_role("button", name="Refresh actions").bounding_box()
            more = await page.get_by_role("button", name="Load more", exact=True).bounding_box()
            assert refresh and more
            if (
                abs(refresh["y"] - more["y"]) < 1
                and more["x"] - refresh["x"] - refresh["width"] < 7
            ):
                differences.append("Adjacent Inbox paging controls need a visible gap")

            async def unavailable(route):
                await route.fulfill(status=503, json={"detail": "Temporarily unavailable"})

            await page.route(re.compile(r"/api/inbox(?:\?|$)"), unavailable)
            await page.goto(base + "/cayu/sessions/question", wait_until="networkidle")
            await check("Refresh actions")
            await page.route(re.compile(r"/api/tools/calls\?"), unavailable)
            await page.goto(base + "/cayu/tools", wait_until="networkidle")
            await page.get_by_role("button", name="Inspect record", exact=True).click()
            await check("Refresh calls")
            assert not differences, differences

    asyncio.run(scenario())


def test_proposal_refresh_exposes_busy_state_and_one_recovery_control():
    async def scenario():
        async with _home_browser(knowledge=True) as (page, base, _, _responses):
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            entered, release = asyncio.Event(), asyncio.Event()

            async def delayed_failure(route):
                entered.set()
                await release.wait()
                await route.fulfill(status=503, json={"detail": "Proposals unavailable"})

            await page.route(re.compile(r"/api/inbox(?:\?|$)"), delayed_failure)
            refresh = page.get_by_role("button", name="Refresh proposals", exact=True)
            await refresh.focus()
            await page.keyboard.press("Enter")
            await asyncio.wait_for(entered.wait(), timeout=5)
            try:
                await expect(refresh).to_be_disabled()
            finally:
                release.set()
            await expect(page.get_by_role("alert")).to_contain_text("Proposals unavailable")
            await expect(refresh).to_be_enabled()
            await expect(
                page.get_by_role("button", name="Retry proposals", exact=True)
            ).to_have_count(0)
            await page.unroute(re.compile(r"/api/inbox(?:\?|$)"), delayed_failure)
            await refresh.click()
            await expect(page.get_by_role("alert")).to_have_count(0)
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_have_count(2)

    asyncio.run(scenario())
