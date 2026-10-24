"""Opt-in native browser acceptance for the additive Trace component plus Session activation."""

from __future__ import annotations

import asyncio
import os
from urllib.parse import parse_qs, urlsplit

import pytest
from tests.core.test_cloud_reporting import _app
from tests.server.test_dashboard_prototype_links import prototype_url
from tests.server.test_dashboard_session_search import search_browser

__all__ = ["prototype_url"]

from cayu import Message
from cayu.events import Event, EventType
from cayu.sessions import RunRequest, SessionIdentity

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_TRACE_PAGE_LIVE") != "1",
    reason="Opt-in Chromium acceptance against the combined Trace/Session activation build.",
)


@pytest.mark.parametrize("member", [False, True])
def test_trace_pages_entire_retained_history_and_filters_exact_events(member):
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        session_id = "trace-history"
        await runtime.session_store.create(
            RunRequest(
                session_id=session_id, agent_name="writer", messages=[Message.text("user", "Trace")]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        total = 10_005
        async with search_browser(runtime=runtime, member=member) as (page, base, _, _, _):
            for index in range(total):
                await runtime.session_store.append_event(
                    session_id,
                    Event(
                        type=EventType.KNOWLEDGE_LOOKUP,
                        session_id=session_id,
                        interaction_id="turn",
                        agent_name="writer",
                        payload={
                            "query": f"lookup-{index}",
                            "reference": "[PRIVATE_EVENT_AUTHORITY]",
                        },
                    ),
                )
            reads = []
            page.on(
                "request",
                lambda request: (
                    reads.append(request.url) if f"/{session_id}/trace?" in request.url else None
                ),
            )
            await page.goto(f"{base}/cayu/sessions/{session_id}?view=trace")
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            await expect(page.get_by_text("1,000 loaded events", exact=False)).to_be_visible()
            for count in range(1, 11):
                await page.get_by_role("button", name="Load more trace history", exact=True).click()
                expected = min((count + 1) * 1000, total)
                await expect(
                    page.get_by_text(f"{expected:,} loaded events", exact=False)
                ).to_be_visible(timeout=15_000)
            await expect(
                page.get_by_role("button", name="Load more trace history", exact=True)
            ).to_have_count(0)
            assert all(
                int(parse_qs(urlsplit(url).query).get("limit", ["1000"])[0]) <= 1000
                for url in reads
            )
            assert len(reads) >= 11
            await page.get_by_role("tab", name="Event log", exact=True).click()
            await page.get_by_label("Event ID", exact=True).fill(f"cayu_event_{total}")
            await page.get_by_role("button", name="Apply filters", exact=True).click()
            await expect(
                page.get_by_text("1 visible of 1 loaded events", exact=False)
            ).to_be_visible()
            await page.get_by_role("button", name=f"{total} ", exact=False).click()
            await expect(page.get_by_role("complementary", name="Event detail")).to_be_visible()
            await expect(
                page.get_by_text("Hidden by the server", exact=False).first
            ).to_be_visible()
            assert "[PRIVATE_EVENT_AUTHORITY]" not in await page.locator("body").inner_text()
            assert parse_qs(urlsplit(page.url).query)["event_id"] == [f"cayu_event_{total}"]
            await expect(
                page.get_by_role("button", name="Retry background interruption", exact=True)
            ).to_have_count(0)

    asyncio.run(run())


def test_trace_exact_transcript_round_trip_for_reused_call_ids():
    from playwright.async_api import expect
    from tests.core.test_tool_completion import FinalTool, app_for, call
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

    from cayu.providers import ModelStreamEvent
    from cayu.sessions import InMemorySessionStore

    async def run():
        runtime = app_for(
            InMemorySessionStore(),
            _TwoCallProvider(
                [
                    call(),
                    call(),
                    [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
                ]
            ),
            FinalTool(),
        )
        _ = [
            event
            async for event in runtime.run(
                RunRequest(
                    session_id="exact-trace",
                    agent_name="support",
                    messages=[Message.text("user", "Help")],
                )
            )
        ]
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            response = await page.request.get(f"{base}/api/sessions/exact-trace/trace?limit=1000")
            body = await response.json()
            tools = [span for span in body["spans"] if span["kind"] == "tool"]
            assert len(tools) == 2
            target = tools[1]
            position = target["transcript_links"][0]
            await page.goto(f"{base}/cayu/sessions/exact-trace?view=trace&span={target['id']}")
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            await page.get_by_role("button", name="Show in conversation", exact=True).click()
            row = page.locator(
                f"#transcript-tool-{position['message_index']}-{position['part_index']}"
            )
            await expect(row).to_be_visible()
            await row.get_by_role("button", name="View in trace", exact=True).click()
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            await expect(
                page.get_by_role("complementary", name="Span detail").get_by_role(
                    "button", name="Show in conversation", exact=True
                )
            ).to_be_visible()
            assert "Linked by event order" not in await page.locator("body").inner_text()

    asyncio.run(run())


def test_trace_shared_live_follower_aborts_hidden_reads_and_drains_terminal_events():
    from contextlib import suppress

    from playwright.async_api import expect
    from tests.core.test_tool_completion import FinalTool, app_for
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

    from cayu.providers import ModelStreamEvent
    from cayu.sessions import InMemorySessionStore

    async def run():
        entered, release, long_poll, aborted = (asyncio.Event() for _ in range(4))

        class Provider(_TwoCallProvider):
            async def stream(self, request):
                entered.set()
                await release.wait()
                async for event in super().stream(request):
                    yield event

        runtime = app_for(
            InMemorySessionStore(),
            Provider([[ModelStreamEvent.text_delta("Finished"), ModelStreamEvent.completed()]]),
            FinalTool(),
        )

        async def execute():
            return [
                event
                async for event in runtime.run(
                    RunRequest(
                        session_id="live-trace",
                        agent_name="support",
                        messages=[Message.text("user", "Wait")],
                    )
                )
            ]

        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            running = asyncio.create_task(execute())
            try:
                await asyncio.wait_for(entered.wait(), 10)
                requests = []

                def observe(request):
                    query = parse_qs(urlsplit(request.url).query)
                    if "/events?" in request.url and query.get("wait_seconds") == ["10"]:
                        requests.append(request.url)
                        long_poll.set()

                page.on("request", observe)
                page.on(
                    "requestfailed",
                    lambda request: aborted.set() if request.url in requests else None,
                )
                await page.goto(f"{base}/cayu/sessions/live-trace?view=trace")
                await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
                await asyncio.wait_for(long_poll.wait(), 10)
                await page.evaluate("""() => {
                    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
                    document.dispatchEvent(new Event("visibilitychange"));
                }""")
                await asyncio.wait_for(aborted.wait(), 10)
                count = len(requests)
                await runtime.session_store.append_event(
                    "live-trace",
                    Event(
                        type=EventType.KNOWLEDGE_LOOKUP,
                        session_id="live-trace",
                        payload={"query": "while-hidden"},
                    ),
                )
                await asyncio.sleep(0.2)
                assert len(requests) == count
                await page.evaluate("""() => {
                    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "visible" });
                    document.dispatchEvent(new Event("visibilitychange"));
                }""")
                await expect(
                    page.get_by_role("treeitem").filter(has_text="Knowledge lookup")
                ).to_be_visible()
                release.set()
                await asyncio.wait_for(running, 15)
                response = await page.request.get(
                    f"{base}/api/sessions/live-trace/trace?limit=1000"
                )
                final = await response.json()
                await expect(
                    page.get_by_text(f"{len(final['events']):,} loaded events", exact=False)
                ).to_be_visible(timeout=15_000)
                await expect(
                    page.get_by_role("complementary", name="Span detail").get_by_text(
                        "OK", exact=True
                    )
                ).to_be_visible()
            finally:
                release.set()
                if not running.done():
                    running.cancel()
                with suppress(asyncio.CancelledError):
                    await running

    asyncio.run(run())


def test_trace_prototype_keeps_fake_data_without_real_contract_context(prototype_url):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            failures = []
            page.on("pageerror", lambda error: failures.append(str(error)))
            prototype_trace_url = f"{prototype_url}#/sessions/close-00a4224687bc?view=trace"
            page.goto(prototype_trace_url)
            expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            page.get_by_role("tab", name="Event log", exact=True).click()
            expect(page.get_by_label("Event type", exact=True)).to_be_visible()
            assert failures == []
        finally:
            browser.close()


def test_trace_workflow_redirect_retains_deep_tree_and_causal_usage():
    from playwright.async_api import expect

    from cayu import SessionStatus

    async def run():
        runtime = _app()
        for name, parent in [
            ("trace-root", None),
            ("trace-child", "trace-root"),
            ("trace-grandchild", "trace-child"),
        ]:
            await runtime.session_store.create(
                RunRequest(
                    session_id=name,
                    agent_name="writer",
                    parent_session_id=parent,
                    causal_budget_id="trace-budget",
                    messages=[Message.text("user", name)],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await runtime.session_store.append_event(
                name,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=name,
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake",
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        }
                    },
                ),
            )
            await runtime.session_store.update_status(name, SessionStatus.COMPLETED)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(
                f"{base}/cayu/sessions/trace-root/workflow?expanded_session_id=trace-child"
            )
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            await expect(
                page.get_by_text("Session tree and causal usage", exact=True)
            ).to_be_visible()
            topology = page.get_by_role("list", name="Loaded Workflow topology")
            await expect(topology.get_by_text("trace-grandchild", exact=True)).to_be_visible()
            await expect(page.get_by_text("Causal-budget usage", exact=True)).to_be_visible()
            assert urlsplit(page.url).path == "/cayu/sessions/trace-root"
            assert parse_qs(urlsplit(page.url).query)["expanded_session_id"] == ["trace-child"]
            assert parse_qs(urlsplit(page.url).query)["view"] == ["trace"]

    asyncio.run(run())


def test_trace_paused_event_tail_still_allows_explicit_history_and_filter_reads(monkeypatch):
    import re
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))

    from examples.dashboard_behavior_live import _exercise_trace
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="paused-log", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            for index in range(205):
                await runtime.session_store.append_event(
                    "paused-log",
                    Event(
                        type=EventType.KNOWLEDGE_LOOKUP,
                        session_id="paused-log",
                        payload={"query": str(index)},
                    ),
                )
            await _exercise_trace(page, base, "paused-log")
            await expect(page.get_by_role("button", name=re.compile(r"^205\s"))).to_be_visible()
            await page.get_by_role("button", name="Older events", exact=True).click()
            await expect(page.get_by_role("button", name=re.compile(r"^105\s"))).to_be_visible()
            await page.get_by_label("Event ID", exact=True).fill("cayu_event_1")
            await page.get_by_role("button", name="Apply filters", exact=True).click()
            await expect(
                page.get_by_text("1 visible of 1 loaded events", exact=False)
            ).to_be_visible()
            await expect(page.get_by_role("button", name=re.compile(r"^1\s"))).to_be_visible()
            await expect(
                page.get_by_role("button", name="Resume live tail", exact=True)
            ).to_be_visible()
            await page.get_by_role("button", name="Clear filters", exact=True).click()
            await expect(
                page.get_by_role("button", name="Resume live tail", exact=True)
            ).to_be_visible()

    asyncio.run(run())


def test_trace_exact_noisy_event_filter_keeps_matching_row_and_detail():
    import re

    from playwright.async_api import expect

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="noisy-trace", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await runtime.session_store.append_event(
            "noisy-trace",
            Event(
                type=EventType.MODEL_THINKING_DELTA,
                session_id="noisy-trace",
                payload={"text": "Thinking"},
            ),
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/noisy-trace?view=trace&log=1")
            await expect(
                page.get_by_text("0 visible of 1 loaded events", exact=False)
            ).to_be_visible()
            await page.get_by_label("Event ID", exact=True).fill("cayu_event_1")
            await page.get_by_role("button", name="Apply filters", exact=True).click()
            await expect(page.get_by_role("button", name=re.compile(r"^1\s"))).to_be_visible()
            await expect(page.get_by_role("complementary", name="Event detail")).to_be_visible()
            await page.get_by_role("button", name="Clear filters", exact=True).click()
            await page.get_by_label("Event type", exact=True).fill("model.thinking.delta")
            await page.get_by_role("button", name="Apply filters", exact=True).click()
            await expect(page.get_by_role("button", name=re.compile(r"^1\s"))).to_be_visible()

    asyncio.run(run())


@pytest.mark.parametrize("event_sequence", [1, 1001])
def test_trace_event_relationship_rejects_recreated_session(event_sequence):
    import re

    from playwright.async_api import expect

    from cayu.server import DashboardConfig, ServerConfig, create_server

    class SwitchableService:
        def __init__(self, target):
            self.target = target

        async def __call__(self, scope, receive, send):
            await self.target(scope, receive, send)

    async def run():
        runtime = _app()
        switch = None

        def wrap(app):
            nonlocal switch
            switch = SwitchableService(app)
            return switch

        async def seed(app, label):
            await app.session_store.create(
                RunRequest(session_id="recreated-trace", agent_name="writer", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            for index in range(max(event_sequence, 2)):
                await app.session_store.append_event(
                    "recreated-trace",
                    Event(
                        type=EventType.KNOWLEDGE_LOOKUP,
                        session_id="recreated-trace",
                        payload={"query": f"{label}-{index}"},
                    ),
                )

        await seed(runtime, "original")
        async with search_browser(runtime=runtime, wrapper=wrap) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/recreated-trace?view=trace")
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            await page.get_by_role("tab", name="Event log", exact=True).click()
            # Positive same-incarnation relation, with another selected-event
            # cache key so the replacement observation is a fresh public read.
            initial_sequence = 2 if event_sequence == 1 else 1
            await page.get_by_label("Event ID", exact=True).fill(f"cayu_event_{initial_sequence}")
            await page.get_by_role("button", name="Apply filters", exact=True).click()
            await expect(page.get_by_text("(show in waterfall)", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Clear filters", exact=True).click()
            await expect(page.get_by_label("Event ID", exact=True)).to_have_value("")
            await page.get_by_role("button", name="Pause live tail", exact=True).click()
            await runtime.session_store.delete_session("recreated-trace")
            # Replace the actual service/store at the same origin, as after a
            # development service restart. Native global event sequences restart
            # with the fresh store; no private sequence counters are modified.
            replacement = _app()
            await seed(replacement, "replacement")
            server = create_server(
                replacement,
                config=ServerConfig.local_development(
                    dashboard=DashboardConfig(
                        directory=os.environ.get("CAYU_SESSION_SEARCH_DIRECTORY")
                    )
                ),
            )
            async with server.router.lifespan_context(server):
                switch.target = server
                await page.get_by_label("Event ID", exact=True).fill(f"cayu_event_{event_sequence}")
                await page.get_by_role("button", name="Apply filters", exact=True).click()
                row = page.get_by_role("button", name=re.compile(rf"^{event_sequence}\s"))
                await expect(row).to_be_visible()
                await row.click()
                detail = page.get_by_role("complementary", name="Event detail")
                await expect(
                    detail.get_by_text(f"replacement-{event_sequence - 1}", exact=True)
                ).to_be_visible()
                await expect(page.get_by_text("(show in waterfall)", exact=True)).to_have_count(0)
                await expect(page.get_by_text("incarnation", exact=False).first).to_be_visible()

    asyncio.run(run())


def test_trace_loaded_span_selection_does_not_drain_historical_pages():
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="bounded-selection", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            for index in range(2001):
                await runtime.session_store.append_event(
                    "bounded-selection",
                    Event(
                        type=EventType.KNOWLEDGE_LOOKUP,
                        session_id="bounded-selection",
                        payload={"query": str(index)},
                    ),
                )
            await page.goto(f"{base}/cayu/sessions/bounded-selection?view=trace")
            await expect(page.get_by_text("1,000 loaded events", exact=False)).to_be_visible()
            historical = []

            async def observe(route):
                if "through_sequence" in parse_qs(urlsplit(route.request.url).query):
                    historical.append(route.request.url)
                    await route.fulfill(
                        status=503,
                        content_type="application/json",
                        body='{"detail":"unexpected historical read"}',
                    )
                else:
                    await route.continue_()

            await page.route("**/api/sessions/bounded-selection/trace?*", observe)
            async with page.expect_response(
                lambda response: "/trace?" in response.url and "limit=1&" in response.url
            ):
                await (
                    page.get_by_role("tree", name="Session trace")
                    .get_by_role("treeitem")
                    .nth(1)
                    .click()
                )
            await page.wait_for_timeout(200)
            assert historical == []
            await expect(page.get_by_text("1,000 loaded events", exact=False)).to_be_visible()

    asyncio.run(run())


def test_trace_transient_live_read_failure_retains_loaded_tree_and_retries():
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="retry-trace", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await runtime.session_store.append_event(
            "retry-trace",
            Event(
                type=EventType.KNOWLEDGE_LOOKUP,
                session_id="retry-trace",
                payload={"query": "retained"},
            ),
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/retry-trace?view=trace")
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()

            async def fail_once(route):
                await route.fulfill(
                    status=503,
                    content_type="application/json",
                    body='{"detail":"temporary trace failure"}',
                )

            await page.route("**/api/sessions/retry-trace/trace?*", fail_once, times=1)
            await runtime.session_store.append_event(
                "retry-trace",
                Event(
                    type=EventType.KNOWLEDGE_LOOKUP,
                    session_id="retry-trace",
                    payload={"query": "new progress"},
                ),
            )
            await expect(
                page.get_by_role("alert").filter(has_text="temporary trace failure")
            ).to_be_visible(timeout=20_000)
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()
            await expect(page.get_by_text("1 loaded events", exact=False)).to_be_visible()
            await page.get_by_role("button", name="Retry trace update", exact=True).click()
            await expect(
                page.get_by_role("alert").filter(has_text="temporary trace failure")
            ).to_have_count(0)
            await expect(page.get_by_role("tree", name="Session trace")).to_be_visible()

    asyncio.run(run())


def test_trace_embedded_workflow_resets_when_switching_to_cached_session():
    from playwright.async_api import expect

    from cayu.sessions import SessionStatus

    async def run():
        runtime = _app()
        for name, parent in [
            ("workflow-a", None),
            ("child-a", "workflow-a"),
            ("workflow-b", None),
            ("child-b", "workflow-b"),
        ]:
            await runtime.session_store.create(
                RunRequest(
                    session_id=name, agent_name="writer", messages=[], parent_session_id=parent
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await runtime.session_store.update_status(name, SessionStatus.COMPLETED)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):

            async def focus(name):
                await page.evaluate(
                    "url => window.history.pushState({}, '', url)",
                    f"/cayu/sessions/workflow-{name}?view=trace&workflow=1",
                )
                await expect(
                    page.get_by_role("list", name="Loaded Workflow topology").get_by_text(
                        f"child-{name}", exact=True
                    )
                ).to_be_visible()

            await page.goto(f"{base}/cayu/sessions/workflow-b?view=trace&workflow=1")
            await expect(
                page.get_by_role("list", name="Loaded Workflow topology").get_by_text(
                    "child-b", exact=True
                )
            ).to_be_visible()
            await focus("a")
            await focus("b")
            await expect(
                page.get_by_role("list", name="Loaded Workflow topology").get_by_text(
                    "child-a", exact=True
                )
            ).to_have_count(0)

    asyncio.run(run())
