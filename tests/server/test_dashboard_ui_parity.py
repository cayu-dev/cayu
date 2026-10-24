"""Opt-in geometry and interaction acceptance for the integrated dashboard shell."""

from __future__ import annotations

import asyncio
import os
import re

import pytest
from playwright.async_api import expect
from tests.core.test_cloud_reporting import _app, _drain
from tests.server.test_dashboard_session_search import search_browser

from cayu import Message
from cayu.sessions import RunRequest

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_UI_PARITY_LIVE") != "1", reason="Opt-in packaged UI acceptance."
)


async def _runtime(summary: str = "Release notes are ready"):
    runtime = _app()
    await _drain(
        runtime.run(
            RunRequest(
                session_id="visual-session",
                agent_name="writer",
                title="Draft release notes",
                labels={"team": "operations"},
                messages=[Message.text("user", "Draft release notes")],
            )
        )
    )
    await runtime.session_store.set_summary("visual-session", summary, tone="attention")
    return runtime


@pytest.mark.parametrize("width", [390, 1440])
def test_migrated_pages_have_one_header_and_reachable_sidebar(width):
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": width, "height": 900})
            for path in [
                "",
                "inbox",
                "tasks",
                "sessions",
                "sessions/visual-session",
                "sessions/visual-session?view=trace",
            ]:
                await page.goto(f"{base}/cayu/{path}")
                await expect(page.locator("main header h1")).to_be_visible()
                await expect(page.locator("main header")).to_have_count(1)
                assert (
                    await page.locator("main header").evaluate("e => e.getBoundingClientRect().top")
                    == 0
                )
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            if width == 390:
                opener = page.get_by_role("button", name="Open sidebar", exact=True)
                await opener.click()
                await expect(page.get_by_role("dialog", name="Sidebar")).to_be_visible()
                await page.keyboard.press("Escape")
                await expect(opener).to_be_focused()
            else:
                await page.get_by_role("button", name="Hide sidebar", exact=True).click()
                await page.get_by_role("button", name="Show sidebar", exact=True).click()
                await expect(page.get_by_role("complementary", name="Sidebar")).to_be_visible()
            await page.goto(f"{base}/cayu/agents")
            await expect(page.locator("main header")).to_have_count(1)
            await expect(
                page.get_by_role("heading", name="Specialists", exact=True)
            ).to_be_visible()

    asyncio.run(run())


@pytest.mark.parametrize("width", [390, 1440])
@pytest.mark.parametrize("length", [23, 4096])
def test_summary_badges_stay_compact_and_full_summary_is_reachable(width, length):
    summary = ("Release notes are ready. " * 180)[:length]

    async def run():
        async with search_browser(runtime=await _runtime(summary)) as (page, base, *_):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.goto(f"{base}/cayu/sessions?q=release")
            await expect(
                page.get_by_role("link", name="Draft release notes", exact=True)
            ).to_be_visible()
            badge = page.locator("tbody tr span.rounded-full").first
            await expect(badge).to_be_visible()
            box = await badge.bounding_box()
            assert box and box["height"] <= 24
            assert box["width"] <= 240
            await page.get_by_role("link", name="Draft release notes", exact=True).click()
            await expect(page.get_by_text(summary, exact=True)).to_have_count(1)
            if length == 4096:
                summary_text = page.get_by_text(summary, exact=True)
                box = await summary_text.bounding_box()
                assert box and box["height"] <= 64
                await page.get_by_text("Show full summary", exact=True).click()
                expanded = await summary_text.bounding_box()
                assert expanded and expanded["height"] > box["height"]
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")

    asyncio.run(run())


def test_session_details_use_single_column_annotation_owner():
    async def run():
        async with search_browser(runtime=await _runtime("Detailed release summary. " * 100)) as (
            page,
            base,
            *_,
        ):
            await page.goto(f"{base}/cayu/sessions/visual-session?panel=overview")
            panel = page.get_by_role("complementary", name="Session details")
            await expect(panel).to_be_visible()
            summary = panel.get_by_text("Detailed release summary. " * 100, exact=True)
            box = await summary.bounding_box()
            assert box and box["height"] <= 64
            await panel.get_by_text("Show full summary", exact=True).click()
            expanded = await summary.bounding_box()
            assert expanded and expanded["height"] > box["height"]
            await panel.get_by_text("Collapse summary", exact=True).click()
            grid = panel.get_by_test_id("session-annotations").locator('[data-slot="card-content"]')
            columns = await grid.evaluate(
                "e => getComputedStyle(e).gridTemplateColumns.split(' ').length"
            )
            assert columns == 1
            await expect(panel.get_by_text("team", exact=True)).to_have_count(1)
            await panel.get_by_role("button", name="Edit labels", exact=True).click()
            editor = panel.get_by_label("Session labels JSON")
            await expect(editor).to_be_visible()
            outer = await panel.bounding_box()
            inner = await editor.bounding_box()
            assert outer and inner and inner["x"] + inner["width"] <= outer["x"] + outer["width"]

    asyncio.run(run())


@pytest.mark.parametrize("area", ["sessions", "inbox", "trace"])
def test_secondary_filters_are_disclosed_without_removing_controls(area):
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            path = "sessions/visual-session?view=trace&log=1" if area == "trace" else area
            await page.goto(f"{base}/cayu/{path}")
            await expect(page.locator("main h1")).to_be_visible()
            label = {"sessions": "Raw status", "inbox": "Session", "trace": "Environment"}[area]
            control = page.get_by_label(label, exact=True)
            await expect(control).to_be_hidden()
            await page.get_by_text("Advanced filters", exact=True).click()
            await expect(control).to_be_visible()
            if area == "sessions":
                await control.select_option("completed")
                await expect(page).to_have_url(re.compile("status=completed"))
            else:
                await control.fill("visual-session" if area == "inbox" else "example-env")
                await page.get_by_role(
                    "button", name="Filter" if area == "inbox" else "Apply filters", exact=True
                ).click()
                await expect(page).to_have_url(
                    re.compile(
                        "session_id=visual-session"
                        if area == "inbox"
                        else "environment_name=example-env"
                    )
                )

    asyncio.run(run())


@pytest.mark.parametrize("width", [390, 1440])
def test_task_pages_share_the_single_header_and_member_gates(width):
    from tests.server.test_dashboard_home_inbox import _home_browser
    from tests.server.test_server_inbox import _app as inbox_app

    from cayu.tasks import InMemoryTaskStore, TaskCreate

    async def run():
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(task_id="visual-task", type="report", title="Review report")
        )
        async with _home_browser(role="member", seeded_runtime=inbox_app(task_store=store)) as (
            page,
            base,
            *_,
        ):
            await page.set_viewport_size({"width": width, "height": 900})
            for route in ["tasks", "tasks/visual-task"]:
                await page.goto(f"{base}/cayu/{route}")
                await expect(page.locator("main header h1")).to_be_visible()
                await expect(page.locator("main header")).to_have_count(1)
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await expect(
                    page.get_by_role("button", name="Create a task", exact=True)
                ).to_have_count(0)
                if route == "tasks":
                    search = page.get_by_role("textbox", name="Search tasks", exact=True)
                    group = search.locator("xpath=..")
                    border = await group.evaluate("e => getComputedStyle(e).borderColor")
                    await search.focus()
                    assert await search.evaluate("e => getComputedStyle(e).outlineStyle") == "none"
                    assert await group.evaluate("e => getComputedStyle(e).borderColor") != border
                if width == 390:
                    await page.get_by_role("button", name="Open sidebar", exact=True).click()
                    await expect(page.get_by_role("dialog", name="Sidebar")).to_be_visible()
                    await page.keyboard.press("Escape")
            await expect(page.get_by_role("button", name="Run again", exact=True)).to_have_count(0)

    asyncio.run(run())


def test_pending_migrated_page_retains_mobile_navigation():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 390, "height": 900})
            reached, release = asyncio.Event(), asyncio.Event()

            async def delay(route):
                reached.set()
                await release.wait()
                await route.continue_()

            await page.route("**/assets/sessions-entry-*.js", delay)
            try:
                await page.goto(f"{base}/cayu/sessions", wait_until="domcontentloaded")
                await asyncio.wait_for(reached.wait(), timeout=5)
                await expect(
                    page.get_by_role("button", name="Open sidebar", exact=True)
                ).to_be_visible()
                await page.get_by_role("button", name="Open sidebar", exact=True).click()
                await expect(page.get_by_role("dialog", name="Sidebar")).to_be_visible()
                await page.keyboard.press("Escape")
            finally:
                release.set()
            await expect(page.get_by_role("heading", name="Sessions", exact=True)).to_be_visible()

    asyncio.run(run())


@pytest.mark.parametrize("width", [390, 1440])
@pytest.mark.parametrize("route", ["", "inbox", "sessions", "tasks", "tasks/scroll-task"])
def test_tall_migrated_page_scrolls_body_without_moving_header(width, route):
    from tests.server.test_dashboard_home_inbox import _home_browser
    from tests.server.test_server_inbox import _app as inbox_app

    from cayu.tasks import InMemoryTaskStore, TaskCreate

    async def run():
        tasks = InMemoryTaskStore()
        for index in range(18):
            await tasks.create_task(
                TaskCreate(
                    task_id="scroll-task" if index == 0 else f"task-{index}",
                    type="report",
                    title=f"Review report {index}",
                    description="Review all source material. " * 100,
                )
            )
        runtime = inbox_app(task_store=tasks)
        for index in range(12):
            await _drain(
                runtime.run(
                    RunRequest(
                        session_id=f"scroll-session-{index}",
                        agent_name="asker",
                        title=f"Question {index}",
                        messages=[Message.text("user", "Ask a question")],
                    )
                )
            )
        async with _home_browser(seeded_runtime=runtime) as (page, base, *_):
            await page.set_viewport_size({"width": width, "height": 360})
            await page.goto(f"{base}/cayu/{route}", wait_until="networkidle")
            await expect(page.locator("main header h1")).to_be_visible()
            body = page.locator("main div.overflow-y-auto").first
            assert await page.evaluate("document.documentElement.scrollHeight <= innerHeight")
            dimensions = await body.evaluate("e => [e.scrollHeight,e.clientHeight]")
            assert dimensions[0] > dimensions[1]
            await body.evaluate("e => {e.scrollTop = 200}")
            assert await body.evaluate("e => e.scrollTop") > 0
            assert (
                await page.locator("main header").evaluate("e => e.getBoundingClientRect().top")
                == 0
            )

    asyncio.run(run())


def test_sidebar_focus_handoff_tracks_visible_opener_across_routes_and_resize():
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": 1440, "height": 900})
            await page.goto(f"{base}/cayu/sessions")
            hide = page.get_by_role("button", name="Hide sidebar", exact=True)
            await hide.focus()
            await page.keyboard.press("Enter")
            show = page.get_by_role("button", name="Show sidebar", exact=True)
            await expect(show).to_be_focused()
            await page.keyboard.press("Enter")
            await hide.focus()
            await page.keyboard.press("Enter")
            await page.set_viewport_size({"width": 390, "height": 900})
            opener = page.get_by_role("button", name="Open sidebar", exact=True)
            await opener.click()
            await page.keyboard.press("Escape")
            await expect(opener).to_be_focused()
            await page.goto(f"{base}/cayu/agents")
            await opener.click()
            await page.keyboard.press("Escape")
            await expect(opener).to_be_focused()
            await page.set_viewport_size({"width": 1440, "height": 900})
            await hide.focus()
            await page.keyboard.press("Enter")
            await expect(opener).to_be_focused()
            await page.goto(f"{base}/cayu/sessions")
            await expect(
                page.get_by_role("button", name="Hide sidebar", exact=True)
            ).to_be_visible()

    asyncio.run(run())


@pytest.mark.parametrize("width", [390, 1440])
@pytest.mark.parametrize("field", ["sidebar", "list"])
def test_session_search_controls_match_prototype_geometry_and_focus(width, field):
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.goto(f"{base}/cayu/sessions")
            await expect(page.get_by_role("heading", name="Sessions", exact=True)).to_be_visible()
            if field == "sidebar":
                if width == 390:
                    await page.get_by_role("button", name="Open sidebar", exact=True).click()
                search = page.locator("#sidebar-search")
                group = search.locator("xpath=ancestor::form")
                assert await group.evaluate("e => e.getBoundingClientRect().height") == 32
                await search.focus()
                assert await search.evaluate("e => getComputedStyle(e).outlineStyle") == "none"
                assert await group.evaluate("e => getComputedStyle(e).backgroundColor") not in [
                    "rgba(0, 0, 0, 0)",
                    "transparent",
                ]
                assert (
                    await group.locator("kbd").evaluate("e => getComputedStyle(e).borderTopWidth")
                    == "1px"
                )
                await search.fill("release")
                await search.press("Enter")
                await expect(page).to_have_url(re.compile(r"[?&]q=release"))
            else:
                search = page.locator("main").get_by_role(
                    "textbox", name="Search sessions", exact=True
                )
                group = search.locator("xpath=..")
                box = await group.bounding_box()
                assert box and box["height"] == 32
                if width == 1440:
                    assert box["width"] <= 384
                else:
                    assert box["width"] >= 300
                assert await search.evaluate("e => getComputedStyle(e).fontSize") == "14px"
                unfocused_border = await group.evaluate("e => getComputedStyle(e).borderColor")
                await search.focus()
                assert (
                    await group.evaluate("e => getComputedStyle(e).borderColor") != unfocused_border
                )
                assert await search.evaluate("e => getComputedStyle(e).outlineStyle") == "none"
                assert await search.evaluate("e => getComputedStyle(e).boxShadow") == "none"
                await page.keyboard.type("rele")
                await expect(page).to_have_url(re.compile(r"[?&]q=rele"))
                await expect(search).to_be_focused()
                await page.keyboard.type("ase")
                await expect(page).to_have_url(re.compile(r"[?&]q=release"))
                await expect(search).to_be_focused()
                details = page.get_by_role("button", name="Clear filters", exact=True)
                await details.focus()
                await expect(details).to_have_css("outline-style", "solid")
                await expect(details).to_have_css("outline-width", "2px")

    asyncio.run(run())


@pytest.mark.parametrize("width", [390, 1440])
@pytest.mark.parametrize("area", ["inbox", "trace", "conversation"])
def test_added_filter_controls_follow_prototype_field_geometry(width, area):
    async def run():
        async with search_browser(runtime=await _runtime()) as (page, base, *_):
            await page.set_viewport_size({"width": width, "height": 900})
            path = {
                "inbox": "inbox",
                "trace": "sessions/visual-session?view=trace&log=1",
                "conversation": "sessions/visual-session",
            }[area]
            await page.goto(f"{base}/cayu/{path}")
            label = {"inbox": "Search Inbox", "trace": "Event ID", "conversation": "Message role"}[
                area
            ]
            field = page.get_by_label(label, exact=True)
            await expect(field).to_be_visible()
            height = 28 if area == "trace" else 32
            assert await field.evaluate("e => e.getBoundingClientRect().height") == height
            assert await field.evaluate("e => getComputedStyle(e).borderTopLeftRadius") == (
                "8px" if area == "trace" else "10px"
            )

    asyncio.run(run())
