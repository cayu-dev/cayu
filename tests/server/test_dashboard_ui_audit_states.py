"""Focused browser regressions for audited dashboard presentation and keyboard states."""

import asyncio
import os
import re

import pytest
from playwright.async_api import expect
from tests.server.test_dashboard_session_search import search_browser
from tests.server.test_dashboard_ui_parity import _runtime

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_UI_PARITY_LIVE") != "1", reason="Opt-in packaged UI acceptance."
)


@pytest.mark.parametrize("width", [320, 390, 1440])
def test_expanded_summary_stays_inside_conversation_scroll_owner(width):
    async def run():
        async with search_browser(
            runtime=await _runtime(
                ("Release notes are ready. Review before publishing. " * 100)[:4096]
            )
        ) as (page, base, *_):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.goto(base + "/cayu/sessions/visual-session")
            await page.get_by_text("Show full summary", exact=True).click()
            assert await page.evaluate("document.documentElement.scrollHeight <= innerHeight")
            scroll = page.locator("main .overflow-y-auto").first
            assert await scroll.evaluate("e=>e.scrollHeight>e.clientHeight")
            await scroll.evaluate("e=>e.scrollTop=0")
            await expect(page.get_by_text("Collapse summary", exact=True)).to_be_attached()
            assert (
                await page.locator("main header").evaluate("e=>e.getBoundingClientRect().top") == 0
            )

    asyncio.run(run())


@pytest.mark.parametrize("width", [320, 1440])
def test_inbox_error_wraps_full_message_inside_alert(width):
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": width, "height": 900})
            message = "Unavailable: https://example.invalid/" + ("x" * 240)
            await page.route(
                re.compile(r"/api/inbox(?:\?|$)"),
                lambda route: route.fulfill(status=503, json={"detail": message}),
            )
            await page.goto(base + "/cayu/inbox")
            alert = page.get_by_role("alert").filter(has_text=message)
            await expect(alert).to_be_visible()
            assert await alert.evaluate("e=>e.scrollWidth<=e.clientWidth")
            assert message in await alert.inner_text()

    asyncio.run(run())


def test_inbox_announces_initial_pending_read():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            entered, release = asyncio.Event(), asyncio.Event()

            async def held(route):
                entered.set()
                await release.wait()
                await route.continue_()

            await page.route(re.compile(r"/api/inbox(?:\?|$)"), held)
            try:
                await page.goto(base + "/cayu/inbox")
                await asyncio.wait_for(entered.wait(), 5)
                await expect(
                    page.get_by_role("status").filter(has_text="Loading inbox")
                ).to_be_visible()
            finally:
                release.set()

    asyncio.run(run())


def test_mobile_keyboard_global_search_returns_visible_focus():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 390, "height": 900})
            await page.goto(base + "/cayu/sessions")
            await expect(page.locator("main header")).to_be_visible()
            await page.keyboard.press("Control+k")
            search = page.get_by_role("dialog", name="Sidebar").get_by_role(
                "textbox", name="Search sessions", exact=True
            )
            await expect(search).to_be_focused()
            await search.fill("release")
            await search.press("Enter")
            await expect(page.get_by_role("dialog", name="Sidebar")).not_to_be_visible()
            await expect(
                page.get_by_role("button", name="Open sidebar", exact=True)
            ).to_be_focused()

    asyncio.run(run())


def test_mobile_details_owns_keyboard_and_restores_opener():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 390, "height": 900})
            await page.goto(base + "/cayu/sessions/visual-session")
            opener = page.get_by_role("button", name="Session details", exact=True)
            await opener.click()
            panel = page.get_by_role("dialog", name="Session details")
            await expect(panel).to_be_visible()
            await expect(
                panel.get_by_role("button", name="Close details", exact=True)
            ).to_be_focused()
            for _ in range(12):
                await expect(panel.locator(":focus")).to_have_count(1)
                assert await panel.evaluate("e=>e.contains(document.activeElement)"), (
                    await page.evaluate("document.activeElement.outerHTML")
                )
                await page.keyboard.press("Tab")
            await expect(panel.locator(":focus")).to_have_count(1)
            await page.keyboard.press("Escape")
            await expect(panel).not_to_be_visible()
            await expect(opener).to_be_focused()
            await opener.click()
            await panel.get_by_role("button", name="Close details", exact=True).click()
            await expect(opener).to_be_focused()

    asyncio.run(run())


def test_session_actions_escape_from_child_restores_trigger():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.goto(base + "/cayu/sessions/visual-session")
            trigger = page.get_by_role("button", name="More actions", exact=True)
            await trigger.click()
            await page.keyboard.press("Tab")
            await page.keyboard.press("Escape")
            await expect(trigger).to_have_attribute("aria-expanded", "false")
            await expect(trigger).to_be_focused()
            await trigger.click()
            await page.locator("main header h1").click()
            await expect(trigger).to_have_attribute("aria-expanded", "false")

    asyncio.run(run())


async def _tasks_runtime():
    from tests.server.test_server_inbox import _app

    from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskTypeSpec

    store = InMemoryTaskStore()
    runtime = _app(task_store=store)
    runtime.register_task_type(
        TaskTypeSpec(
            name="report",
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        )
    )
    await store.create_task(
        TaskCreate(
            task_id="audit-task", type="report", title="Review report", input={"topic": "Release"}
        )
    )
    return runtime


@pytest.mark.parametrize("check", ["search", "close", "menu"])
def test_task_mobile_controls_and_keyboard_ownership(check):
    from tests.server.test_dashboard_home_inbox import _home_browser

    async def run():
        async with _home_browser(seeded_runtime=await _tasks_runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 320, "height": 900})
            await page.goto(base + "/cayu/tasks" + ("/audit-task" if check == "menu" else ""))
            if check == "search":
                field = page.get_by_role("textbox", name="Search tasks", exact=True)
                await expect(field).to_be_visible()
                box = await field.bounding_box()
                assert box and box["width"] >= 220
            elif check == "close":
                opener = page.get_by_role("button", name="Create a task", exact=True)
                await opener.click()
                await page.get_by_role("button", name="Close", exact=True).click()
                await expect(opener).to_be_focused()
            else:
                trigger = page.get_by_role("button", name="More actions", exact=True)
                await trigger.click()
                await page.get_by_role("button", name="Pause", exact=True).focus()
                await page.keyboard.press("Escape")
                await expect(trigger).to_have_attribute("aria-expanded", "false")
                await expect(trigger).to_be_focused()
                await trigger.click()
                await page.locator("main header h1").click()
                await expect(trigger).to_have_attribute("aria-expanded", "false")

    asyncio.run(run())


def test_child_action_search_matches_control_geometry():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 320, "height": 900})
            await page.goto(base + "/cayu/inbox")
            field = page.get_by_label("Search child actions", exact=True)
            await expect(field).to_be_visible()
            assert await field.evaluate("e=>e.getBoundingClientRect().height") == 32
            assert await field.evaluate("e=>getComputedStyle(e).borderRadius") == "10px"

    asyncio.run(run())


def test_details_resize_preserves_draft_and_editor_escape_precedes_panel():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 390, "height": 900})
            await page.goto(base + "/cayu/sessions/visual-session")
            opener = page.get_by_role("button", name="Session details", exact=True)
            await opener.click()
            panel = page.locator('aside[aria-label="Session details"]')
            await panel.get_by_role("button", name="Edit labels", exact=True).click()
            editor = panel.get_by_label("Session labels JSON")
            await editor.fill('{"team":"draft-preserved"}')
            await page.set_viewport_size({"width": 1440, "height": 900})
            await expect(panel).to_have_attribute("role", "complementary")
            await expect(editor).to_have_value('{"team":"draft-preserved"}')
            await page.get_by_role("tab", name="Conversation", exact=True).click()
            await expect(panel).to_be_visible()
            await page.set_viewport_size({"width": 390, "height": 900})
            await expect(panel).to_have_attribute("role", "dialog")
            await expect(editor).to_have_value('{"team":"draft-preserved"}')
            await editor.focus()
            await page.keyboard.press("Escape")
            await expect(editor).not_to_be_visible()
            await expect(panel).to_be_visible()
            await expect(
                panel.get_by_role("button", name="Edit labels", exact=True)
            ).to_be_visible()
            await panel.get_by_role("button", name="Close details", exact=True).focus()
            await page.keyboard.press("Escape")
            await expect(panel).not_to_be_visible()
            await expect(opener).to_be_focused()

    asyncio.run(run())


def test_session_view_tabs_support_arrow_navigation_without_losing_route():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.goto(base + "/cayu/sessions/visual-session")
            conversation = page.get_by_role("tab", name="Conversation", exact=True)
            trace = page.get_by_role("tab", name="Trace", exact=True)
            await conversation.focus()
            await page.keyboard.press("ArrowRight")
            await expect(trace).to_be_focused()
            await expect(trace).to_have_attribute("aria-selected", "true")
            await expect(page).to_have_url(re.compile(r"view=trace"))
            await page.keyboard.press("Home")
            await expect(conversation).to_be_focused()
            await expect(conversation).to_have_attribute("aria-selected", "true")
            assert "/sessions/visual-session" in page.url

    asyncio.run(run())


@pytest.mark.parametrize("route", ["sessions", "tasks"])
def test_search_remains_readable_through_dense_tablet_width_sweep(route):
    from tests.server.test_dashboard_home_inbox import _home_browser

    async def run():
        async with _home_browser(seeded_runtime=await _tasks_runtime()) as (page, base, *_):
            await page.goto(base + "/cayu/" + route)
            field = page.locator("main").get_by_role("textbox", name="Search " + route, exact=True)
            await expect(field).to_be_visible()
            widths = sorted(set(range(640, 1280, 8)) | {767, 768, 1023, 1024, 1279})
            for query in ["", "readable query"]:
                await field.fill(query)
                if route == "sessions" and query:
                    await expect(
                        page.get_by_role("button", name="Clear filters", exact=True)
                    ).to_be_visible()
                measurements = []
                for width in widths:
                    await page.set_viewport_size({"width": width, "height": 900})
                    box = await field.bounding_box()
                    measurements.append((width, box["width"] if box else 0))
                    assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                assert min(size for _, size in measurements) >= 170, measurements
                await expect(field).to_have_value(query)

    asyncio.run(run())


@pytest.mark.parametrize("target", ["composer", "trace", "menu"])
def test_desktop_details_does_not_own_escape_outside_panel(target):
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 1440, "height": 900})
            await page.goto(base + "/cayu/sessions/visual-session")
            await page.get_by_role("button", name="Session details", exact=True).click()
            panel = page.locator('aside[aria-label="Session details"]')
            await panel.get_by_role("button", name="Edit labels", exact=True).click()
            editor = panel.get_by_label("Session labels JSON")
            await editor.fill('{"draft":"preserved"}')
            if target == "composer":
                focused = page.locator("textarea[placeholder]").first
            elif target == "trace":
                focused = page.get_by_role("tab", name="Trace", exact=True)
                await focused.click()
                await page.get_by_role("tab", name="Event log", exact=True).click()
                focused = page.get_by_label("Event type", exact=True)
            else:
                focused = page.get_by_role("button", name="More actions", exact=True)
                await focused.click()
                await page.keyboard.press("Tab")
            if target != "menu":
                await focused.focus()
                await page.evaluate("""() => {
                    window.auditEscape = null;
                    window.addEventListener('keydown', e => {
                        if (e.key === 'Escape') window.auditEscape = {prevented: e.defaultPrevented};
                    }, {once: true});
                }""")
            await page.keyboard.press("Escape")
            if target != "menu":
                assert await page.evaluate("window.auditEscape") == {"prevented": False}
            await expect(panel).to_be_visible()
            await expect(editor).to_have_value('{"draft":"preserved"}')
            await expect(focused).to_be_focused()
            if target == "menu":
                await expect(focused).to_have_attribute("aria-expanded", "false")

    asyncio.run(run())
