"""Usage native inputs and complete cost values through the real production page."""

import asyncio
import io
import os
from pathlib import Path
from urllib.parse import urlencode

import pytest

pytest.importorskip("playwright.async_api")
from PIL import Image
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser
from tests.server.test_dashboard_usage_costs import cost_runtime, prices

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_USAGE_LAYOUT_LIVE") != "1",
    reason="Opt-in focused Usage layout Chromium acceptance.",
)


def _nodes(node):
    yield node
    for child in node.get("children", []) + node.get("shadowRoots", []):
        yield from _nodes(child)


def _attrs(node):
    attributes = node.get("attributes", [])
    return dict(zip(attributes[::2], attributes[1::2], strict=True))


async def _native_input_boxes(page, selector):
    """Inspect rendered native segments, not a class or computed-style snapshot."""
    cdp = await page.context.new_cdp_session(page)
    try:
        document = await cdp.send("DOM.getDocument", {"depth": -1, "pierce": True})
        field = next(
            node for node in _nodes(document["root"]) if _attrs(node).get("id") == selector
        )
        boxes = {}
        for node in _nodes(field):
            attrs = _attrs(node)
            pseudo = attrs.get("pseudo", "")
            if pseudo in {
                "-webkit-datetime-edit",
                "-webkit-datetime-edit-fields-wrapper",
                "-webkit-calendar-picker-indicator",
            }:
                model = await cdp.send("DOM.getBoxModel", {"nodeId": node["nodeId"]})
                boxes[pseudo] = model["model"]["border"]
        return boxes
    finally:
        await cdp.detach()


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("amount", ["large", "tiny"])
def test_usage_custom_input_alignment_and_complete_estimate(tmp_path, width, amount):
    async def run():
        price_book = prices()
        for index, entry in enumerate(price_book["prices"]):
            pricing = entry["schedules"][0]["pricing"]
            pricing["standard"][0]["input_per_million"] = (
                "123456789012345.123456789012345"
                if amount == "large"
                else "0.000000123456789012345"
            )
            if index:
                pricing["currency"] = "EUR"
        window = {
            "start_at": "2026-09-30T23:59:59.123456Z",
            "end_at": "2026-10-02T00:00:00.654321Z",
        }
        async with _home_browser(
            seeded_runtime=await cost_runtime(), dashboard_config={"priceBook": price_book}
        ) as (page, base, _requests, _responses):
            await page.set_viewport_size({"width": width, "height": 1000})
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **window})}",
                wait_until="networkidle",
            )
            card = page.get_by_text("Estimated cost", exact=True).locator("..")
            value = card.locator(":scope > div").nth(1)
            await expect(value).to_contain_text("EUR")
            rendered = " ".join((await value.inner_text()).split())
            assert rendered == (
                "EUR 1,234,567,890.12 USD 1,358,024,679,135.80"
                if amount == "large"
                else "<EUR 0.0001 <USD 0.0001"
            )
            # Every representative currency/amount remains one readable line, even on mobile.
            assert await value.locator("span").evaluate_all(
                "nodes => nodes.length === 2 && nodes.every(n => n.getBoundingClientRect().height <= 29)"
            )
            captures = Path(os.environ.get("CAYU_USAGE_LAYOUT_CAPTURES", str(tmp_path)))
            captures.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(captures / f"usage-{amount}-{width}.png"))
            violations = []
            geometry = await value.evaluate(
                """el => {
                    const box = el.getBoundingClientRect();
                    const range = document.createRange(); range.selectNodeContents(el);
                    return {width: el.clientWidth, scroll: el.scrollWidth,
                        contained: [...range.getClientRects()].every(r =>
                            r.left >= box.left - 1 && r.right <= box.right + 1 &&
                            r.top >= box.top - 1 && r.bottom <= box.bottom + 1)};
                }"""
            )
            if geometry["scroll"] > geometry["width"] + 1 or not geometry["contained"]:
                violations.append(f"Estimated cost is clipped: {geometry}")
            for selector in ("usage-start", "usage-end"):
                boxes = await _native_input_boxes(page, selector)
                editor = boxes["-webkit-datetime-edit"]
                segments = boxes["-webkit-datetime-edit-fields-wrapper"]
                picker = boxes["-webkit-calendar-picker-indicator"]
                if abs((editor[0] + editor[2]) / 2 - (segments[0] + segments[2]) / 2) > 1:
                    violations.append(f"{selector} native text is not horizontally centered")
                outer = await page.locator(f"#{selector}").bounding_box()
                assert outer is not None
                assert (
                    abs((segments[1] + segments[5]) / 2 - (outer["y"] + outer["height"] / 2)) <= 1
                )
                assert editor[0] <= segments[0] and segments[2] <= picker[0] + 1
                assert picker[2] <= outer["x"] + outer["width"] and picker[2] > picker[0]
            await page.get_by_role("button", name="Show", exact=True).click()
            async with page.expect_request(
                lambda request: (
                    request.method == "POST" and request.url.endswith("/api/usage/rollup")
                )
            ) as requested:
                await page.get_by_role("button", name="Refresh", exact=True).click()
            body = (await requested.value).post_data_json
            assert isinstance(body, dict)
            assert {key: body[key] for key in window} == window
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not violations, "; ".join(violations)

    asyncio.run(run())


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize(
    "browser_name",
    [
        "chromium",
        pytest.param(
            "webkit",
            marks=pytest.mark.skipif(
                os.environ.get("CAYU_DASHBOARD_USAGE_WEBKIT_LIVE") != "1",
                reason="Requires explicitly installed WebKit.",
            ),
        ),
    ],
)
def test_usage_datetime_glyphs_are_vertically_centered_in_light_theme(
    tmp_path, width, browser_name
):
    async def run():
        window = {
            "start_at": "2026-09-30T23:59:59.123456Z",
            "end_at": "2026-10-02T00:00:00.654321Z",
        }
        async with _home_browser(
            browser_name=browser_name, seeded_runtime=await cost_runtime()
        ) as (page, base, _requests, _responses):
            await page.context.add_init_script(
                "localStorage.setItem('cayu-dashboard-theme', 'light')"
            )
            await page.set_viewport_size({"width": width, "height": 1000})
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **window})}",
                wait_until="networkidle",
            )
            captures = Path(os.environ.get("CAYU_USAGE_LAYOUT_CAPTURES", str(tmp_path)))
            captures.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(captures / f"usage-light-{browser_name}-{width}.png"))
            offsets = []
            for selector in ("usage-start", "usage-end"):
                shot = await page.locator(f"#{selector}").screenshot()
                (captures / f"{selector}-{browser_name}-{width}.png").write_bytes(shot)
                pixels = Image.open(io.BytesIO(shot)).convert("RGB")
                # Light-theme native input: dark glyphs, pale border/background. Exclude
                # edges and Chromium's picker so this measures text, not its container.
                rows = [
                    y
                    for y in range(2, pixels.height - 2)
                    for x in range(12, pixels.width - 32)
                    if max(pixels.getpixel((x, y))) < 180
                ]
                assert rows, "Expected visible native date/time glyphs"
                offset = (min(rows) + max(rows)) / 2 - (pixels.height - 1) / 2
                offsets.append((selector, offset))
            await page.get_by_role("button", name="Show", exact=True).click()
            async with page.expect_request(
                lambda request: (
                    request.method == "POST" and request.url.endswith("/api/usage/rollup")
                )
            ) as requested:
                await page.get_by_role("button", name="Refresh", exact=True).click()
            body = (await requested.value).post_data_json
            assert isinstance(body, dict)
            assert {key: body[key] for key in window} == window
            assert all(abs(offset) <= 2 for _, offset in offsets), offsets

    asyncio.run(run())
