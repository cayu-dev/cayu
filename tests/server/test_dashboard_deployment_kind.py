"""Deployment labels and protected snapshot ownership through the real dashboard."""

import asyncio
import os
from pathlib import Path

import pytest
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser

from cayu.server import ServerConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_DEPLOYMENT_LIVE") != "1",
    reason="Opt-in deployment-kind Chromium acceptance.",
)


@pytest.mark.parametrize("kind,mobile", [("local", False), ("cloud", False), ("embedded", True)])
def test_system_displays_declared_kind_from_signed_http(monkeypatch, kind, mobile):
    original = ServerConfig.protected
    monkeypatch.setattr(
        ServerConfig,
        "protected",
        lambda *args, **kwargs: original(*args, deployment_kind=kind, **kwargs),
    )

    async def exercise():
        async with _home_browser(mobile=mobile, nested=True) as (page, base, requests, responses):
            await page.goto(f"{base}/cayu/system", wait_until="networkidle")
            row = page.locator("dl > div").filter(
                has=page.get_by_text("Deployment kind", exact=True)
            )
            await expect(row.locator("dd")).to_have_text(kind.capitalize())
            assert await row.evaluate("el => el.getBoundingClientRect().right <= innerWidth + 1")
            await page.get_by_text("Debug", exact=True).click()
            await expect(page.get_by_text(f"kind={kind}", exact=False)).to_be_visible()
            screenshots = Path(__file__).resolve().parents[2] / ".agents/screenshots"
            screenshots.mkdir(parents=True, exist_ok=True)
            await row.scroll_into_view_if_needed()
            await page.screenshot(path=str(screenshots / f"deployment-{kind}.png"))
            assert any("/api/system/diagnostics" in path for path in requests)
            assert not [response for response in responses if response.status >= 400]

    asyncio.run(exercise())


@pytest.mark.parametrize("status", [401, 403])
def test_system_denial_discards_setup_until_authorized_recovery(status):
    async def exercise():
        async with _home_browser() as (page, base, _requests, _responses):
            await page.goto(f"{base}/cayu/system", wait_until="networkidle")
            await expect(
                page.locator("main dd").filter(has_text="Shell acceptance").first
            ).to_be_visible()

            async def reject(route):
                await route.fulfill(status=status, json={"detail": "Access denied"})

            await page.route("**/api/system/diagnostics", reject)
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(
                page.locator("main dd").filter(has_text="Shell acceptance").first
            ).to_have_count(0)
            await page.unroute("**/api/system/diagnostics", reject)

            async def transient(route):
                await route.fulfill(status=503, json={"detail": "Temporarily unavailable"})

            await page.route("**/api/system/diagnostics", transient)
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(page.get_by_text("Temporarily unavailable", exact=True)).to_be_visible()
            await expect(page.get_by_text("Deployment kind", exact=True)).to_have_count(0)
            await page.goto(f"{base}/cayu/", wait_until="networkidle")
            await page.goto(f"{base}/cayu/system", wait_until="networkidle")
            await expect(page.get_by_text("Deployment kind", exact=True)).to_have_count(0)
            await page.unroute("**/api/system/diagnostics", transient)
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(
                page.locator("main dd").filter(has_text="Shell acceptance").first
            ).to_be_visible()
            await expect(page.get_by_text("Deployment kind", exact=True)).to_be_visible()

    asyncio.run(exercise())


def test_system_member_does_not_request_deployment_snapshot():
    async def exercise():
        async with _home_browser(role="member") as (page, base, requests, _responses):
            await page.goto(f"{base}/cayu/system", wait_until="networkidle")
            await expect(
                page.get_by_text(
                    "Your role can read and decide, but this action requires full access.",
                    exact=True,
                )
            ).to_be_visible()
            assert not any("/api/system/diagnostics" in path for path in requests)
            response = await page.request.get(f"{base}/api/system/diagnostics")
            assert response.status == 403

    asyncio.run(exercise())


def test_system_transient_failure_preserves_authorized_snapshot():
    async def exercise():
        async with _home_browser() as (page, base, _requests, _responses):
            await page.goto(f"{base}/cayu/system", wait_until="networkidle")
            await expect(page.get_by_text("Deployment kind", exact=True)).to_be_visible()

            async def transient(route):
                await route.fulfill(status=503, json={"detail": "Temporarily unavailable"})

            await page.route("**/api/system/diagnostics", transient)
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(
                page.get_by_text("Couldn't refresh; showing what was loaded before.", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("Deployment kind", exact=True)).to_be_visible()

    asyncio.run(exercise())
