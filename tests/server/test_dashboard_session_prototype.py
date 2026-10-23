"""The Sessions port remains usable in the current fake-data harness."""

from __future__ import annotations

import os

import pytest
from tests.server.test_dashboard_prototype_links import prototype_url

__all__ = ["prototype_url"]
pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_PROTOTYPE_LIVE") != "1",
    reason="Opt-in prototype browser acceptance.",
)


def test_prototype_session_population_filters_and_native_links(prototype_url):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(f"{prototype_url}#/sessions")
            playwright.expect(page.get_by_label("Include forks", exact=True)).not_to_be_checked()
            playwright.expect(page.get_by_role("table")).to_be_visible()
            playwright.expect(
                page.get_by_text("close-eval-7f3a91c2d4e0", exact=True)
            ).to_have_count(0)
            page.get_by_label("Include eval trials", exact=True).check()
            playwright.expect(
                page.get_by_text("close-eval-7f3a91c2d4e0", exact=True)
            ).to_be_visible()
            page.get_by_label("State", exact=True).select_option("waiting")
            playwright.expect(
                page.get_by_text("close-eval-7f3a91c2d4e0", exact=True)
            ).to_have_count(0)
            link = page.get_by_role("table").get_by_role("link").first
            playwright.expect(link).to_be_visible()
            href = link.get_attribute("href")
            assert href and "#/sessions/" in href
            with context.expect_page() as opened:
                link.click(modifiers=["ControlOrMeta"])
            opened.value.wait_for_load_state()
            assert "#/sessions/" in opened.value.url
            opened.value.close()
            playwright.expect(page.get_by_label("State", exact=True)).to_have_value("waiting")
        finally:
            browser.close()
