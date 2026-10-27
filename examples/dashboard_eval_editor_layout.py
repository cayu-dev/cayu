"""Real panel geometry checks for the full-page Evals suite editor."""

from playwright.async_api import expect


async def exercise_editor_layout(page, sheet) -> None:
    """Check viewport and independently constrained panel widths through live controls."""
    await expect(sheet).to_be_visible()
    for viewport, panel in (
        (1440, None),
        (900, None),
        (1280, None),
        (1920, None),
        (390, None),
        (320, None),
        (1440, 640),
        (1440, 380),
    ):
        await page.set_viewport_size({"width": viewport, "height": 900})
        await sheet.evaluate(
            """(node, width) => {
                node.style.removeProperty('width');
                node.style.removeProperty('max-width');
                if (width) {
                    node.style.setProperty('width', `${width}px`, 'important');
                    node.style.setProperty('max-width', `${width}px`, 'important');
                }
            }""",
            panel,
        )
        # Wait for responsive layout and browser geometry to settle.
        await page.wait_for_timeout(250)
        bounds = await sheet.bounding_box()
        assert bounds and bounds["width"] <= viewport, (viewport, panel, bounds)
        assert await sheet.evaluate("node => node.scrollWidth <= node.clientWidth + 1"), (
            viewport,
            panel,
            "editor content overflows horizontally",
        )
        controls = sheet.locator("input, textarea, select, button")
        for control in await controls.all():
            if not await control.is_visible():
                continue
            await control.evaluate(
                "node => node.scrollIntoView({block: 'center', inline: 'nearest'})"
            )
            box = await control.bounding_box()
            assert box, (viewport, panel, await control.get_attribute("id"))
            assert bounds["x"] - 1 <= box["x"], (viewport, panel, bounds, box)
            assert box["x"] + box["width"] <= bounds["x"] + bounds["width"] + 1, (
                viewport,
                panel,
                await control.get_attribute("id"),
                bounds,
                box,
            )
            clipped = await control.evaluate(
                """node => {
                    const control = node.getBoundingClientRect();
                    for (let owner = node.parentElement; owner; owner = owner.parentElement) {
                        const style = getComputedStyle(owner);
                        const clip = owner.getBoundingClientRect();
                        if (style.overflowX !== 'visible' &&
                            (control.left < clip.left - 1 || control.right > clip.right + 1))
                            return {axis: "horizontal", owner: owner.className, control: control.toJSON(), clip: clip.toJSON()};
                        if (style.overflowY !== 'visible' &&
                            (control.top < clip.top - 1 || control.bottom > clip.bottom + 1))
                            return {axis: "vertical", owner: owner.className, control: control.toJSON(), clip: clip.toJSON()};
                    }
                    return null;
                }"""
            )
            assert clipped is None, (viewport, panel, await control.get_attribute("id"), clipped)
            assert box["y"] >= 0 and box["y"] + box["height"] <= 901, (
                viewport,
                panel,
                box,
            )
        await sheet.get_by_test_id("authored-suite-name").fill("Layout-preserved draft")
        await expect(sheet.get_by_test_id("authored-suite-name")).to_have_value(
            "Layout-preserved draft"
        )
        print(f"EDITOR PANEL GEOMETRY PASS viewport={viewport} panel={panel}", flush=True)
    await sheet.evaluate(
        "node => { node.style.removeProperty('width'); node.style.removeProperty('max-width'); }"
    )
    await page.set_viewport_size({"width": 1440, "height": 900})
