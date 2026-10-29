"""Usage's batched reads through the production page and signed member/admin roles."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.core.test_cloud_reporting import _app
from tests.server.test_dashboard_home_inbox import _home_browser

from cayu import RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_USAGE_SERIES_LIVE") != "1",
    reason="Opt-in focused Usage series Chromium acceptance.",
)


async def _usage_runtime():
    now = datetime.now(UTC)
    clock = now - timedelta(days=10)
    store = InMemorySessionStore(ownership_clock=lambda: clock)
    runtime = _app(store)
    runtime.register_environment(Environment(EnvironmentSpec(name="Review / λ")))
    identity = SessionIdentity(provider_name="configured", model="configured-model")
    for name in ("old", "recent"):
        await store.create(
            RunRequest(
                session_id=name, agent_name="writer", environment_name="Review / λ", messages=[]
            ),
            identity=identity,
        )
        await store.append_event(
            name,
            Event(
                id=f"activity-{name}",
                session_id=name,
                type=EventType.MODEL_COMPLETED,
                timestamp=now,
                payload={
                    "usage_metrics": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
                },
            ),
        )
        clock = now
    await store.update_status("recent", SessionStatus.FAILED)
    return runtime, now


@pytest.mark.parametrize("role", ["admin", "member"])
def test_usage_native_series_share_window_filters_and_distinct_populations(role):
    async def exercise():
        runtime, now = await _usage_runtime()
        async with _home_browser(
            role=role, nested=True, mobile=role == "member", seeded_runtime=runtime
        ) as (page, base, _, _responses):
            reads = []
            page.on("request", lambda request: reads.append(request))
            start = (now - timedelta(days=1)).isoformat()
            end = (now + timedelta(seconds=1)).isoformat()
            search = {
                "range": "custom",
                "start_at": start,
                "end_at": end,
                "agent_name": "writer",
                "provider_name": "configured",
                "model": "configured-model",
                "environment_name": "Review / λ",
            }
            await page.goto(f"{base}/cayu/usage?{urlencode(search)}", wait_until="networkidle")
            rollups = [
                r for r in reads if r.method == "POST" and r.url.endswith("/api/usage/rollup")
            ]
            assert len(rollups) == 1
            body = rollups[0].post_data_json
            assert isinstance(body, dict)
            assert body["series"]["unit"] == "hour"
            assert body["series"]["timezone"] == await page.evaluate(
                "Intl.DateTimeFormat().resolvedOptions().timeZone"
            )
            charts = [
                r
                for r in reads
                if "/sessions/overview/counts?" in r.url and "bucket_unit=" in r.url
            ]
            assert len(charts) == 1
            query = parse_qs(urlsplit(charts[0].url).query)
            assert query["since"] == [start] and query["until"] == [end]
            for key in ("agent_name", "provider_name", "model", "environment_name"):
                assert query[key] == [search[key]]
            assert query["timezone"] == [body["series"]["timezone"]]
            await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
            table = page.get_by_role("table", name="Sessions started per hour")
            await expect(table).to_contain_text("14 activity tokens")
            assert await table.locator("tbody tr").count() >= 24
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            # Existing unsupported bookmark filters remain on the headline but never broaden bars.
            reads.clear()
            await page.goto(
                f"{base}/cayu/usage?{urlencode({**search, 'label': 'team=qa'})}",
                wait_until="networkidle",
            )
            await expect(
                page.get_by_text(
                    "Session counts are unavailable for label, parent or budget filters.",
                    exact=True,
                )
            ).to_be_visible()
            assert not [r for r in reads if "bucket_unit=" in r.url]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "usage_enabled,counts_enabled", [(False, True), (True, False), (False, False)]
)
def test_optional_series_capabilities_do_not_disable_headline_or_trigger_fallback_reads(
    usage_enabled, counts_enabled
):
    async def exercise():
        runtime, now = await _usage_runtime()
        async with _home_browser(role="member", seeded_runtime=runtime) as (
            page,
            base,
            _,
            _responses,
        ):
            reads = []

            async def contract(route):
                response = await route.fetch()
                body = await response.json()
                surfaces = body["capabilities"]["surfaces"]
                for name, enabled in (
                    ("usage_series", usage_enabled),
                    ("session_overview_series", counts_enabled),
                ):
                    if not enabled:
                        surfaces.pop(name, None)
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", contract)
            page.on("request", lambda request: reads.append(request))
            search = {
                "range": "custom",
                "start_at": (now - timedelta(days=1)).isoformat(),
                "end_at": (now + timedelta(seconds=1)).isoformat(),
                "agent_name": "writer",
            }
            await page.goto(f"{base}/cayu/usage?{urlencode(search)}", wait_until="networkidle")
            body = next(r.post_data_json for r in reads if r.url.endswith("/api/usage/rollup"))
            assert ("series" in body) is usage_enabled
            chart_requests = [
                r
                for r in reads
                if "/sessions/overview/counts?" in r.url and "bucket_unit=" in r.url
            ]
            assert len(chart_requests) == int(counts_enabled)
            # Home/sidebar observations are separate; no old per-bar window requests remain.
            count_reads = [r for r in reads if "/sessions/overview/counts?" in r.url]
            assert len(count_reads) <= 2
            await expect(page.get_by_text("Model steps", exact=True)).to_be_visible()
            if counts_enabled:
                await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
                await expect(
                    page.get_by_text(
                        "Activity token detail is unavailable for this range.", exact=True
                    )
                ).to_be_visible()
            else:
                await expect(
                    page.get_by_text("Session counts are unavailable for this agent.", exact=True)
                ).to_be_visible()

    asyncio.run(exercise())


def test_series_refresh_failures_and_superseded_filters_preserve_query_identity():
    async def exercise():
        runtime, now = await _usage_runtime()
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, _responses):
            search = {
                "range": "custom",
                "start_at": (now - timedelta(days=1)).isoformat(),
                "end_at": (now + timedelta(seconds=1)).isoformat(),
                "agent_name": "writer",
            }
            await page.goto(f"{base}/cayu/usage?{urlencode(search)}", wait_until="networkidle")
            await expect(page.get_by_text("1 started", exact=True)).to_be_visible()

            async def unavailable(route):
                await route.fulfill(status=501, json={"detail": "Injected unavailable read."})

            await page.route("**/api/usage/rollup", unavailable)
            await page.route("**/api/sessions/overview/counts?*", unavailable)
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(
                page.get_by_text("Couldn't refresh; showing the last numbers loaded.", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_text(
                    "Couldn't refresh session counts; showing the last counts loaded.", exact=True
                )
            ).to_be_visible()
            await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
            await page.unroute("**/api/usage/rollup", unavailable)
            await page.unroute("**/api/sessions/overview/counts?*", unavailable)
            started, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
            failures = []
            page.on("requestfailed", lambda request: failures.append(request))

            async def slow(route, request):
                query = parse_qs(urlsplit(request.url).query)
                if query.get("agent_name") != ["asker"] or "bucket_unit" not in query:
                    await route.continue_()
                    return
                started.set()
                await release.wait()
                try:
                    await route.continue_()
                except Exception:
                    pass  # A genuinely aborted fetch can dispose its intercepted route.
                finally:
                    settled.set()

            await page.route("**/api/sessions/overview/counts?*", slow)
            try:
                await page.get_by_label("Specialist", exact=True).select_option("asker")
                await asyncio.wait_for(started.wait(), 5)
                await expect(page.get_by_text("1 started", exact=True)).to_have_count(0)
                await page.get_by_label("Specialist", exact=True).select_option("writer")
                await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
                release.set()
                await asyncio.wait_for(settled.wait(), 5)
                await page.wait_for_load_state("networkidle")
                assert any(
                    "agent_name=asker" in request.url
                    and "bucket_unit=" in request.url
                    and request.failure == "net::ERR_ABORTED"
                    for request in failures
                )
                await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
            finally:
                release.set()
                await page.unroute("**/api/sessions/overview/counts?*", slow)

    asyncio.run(exercise())


def test_relative_refresh_retains_exact_histogram_window_until_replacement_succeeds():
    async def exercise():
        runtime, now = await _usage_runtime()
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, _responses):
            await page.clock.set_fixed_time(now + timedelta(seconds=1))
            rollups = []
            charts = []

            def record(request):
                if request.url.endswith("/api/usage/rollup"):
                    rollups.append(request.post_data_json)
                if "bucket_unit=" in request.url:
                    charts.append(parse_qs(urlsplit(request.url).query))

            page.on("request", record)
            await page.goto(
                f"{base}/cayu/usage?range=7d&agent_name=writer", wait_until="networkidle"
            )
            await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
            table = page.get_by_role("table", name="Sessions started per day")
            original_periods = await table.locator("tbody tr td:first-child").all_text_contents()
            original = charts[-1]
            assert len(original_periods) == 7
            started, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def delayed_failure(route, request):
                query = parse_qs(urlsplit(request.url).query)
                if "bucket_unit" not in query or query.get("until") == original["until"]:
                    await route.continue_()
                    return
                started.set()
                await release.wait()
                try:
                    await route.fulfill(status=501, json={"detail": "Injected unavailable read."})
                finally:
                    settled.set()

            await page.route("**/api/sessions/overview/counts?*", delayed_failure)
            try:
                await page.clock.set_fixed_time(now + timedelta(hours=25))
                await page.get_by_role("button", name="Refresh", exact=True).click()
                await asyncio.wait_for(started.wait(), 5)
                assert rollups[-1]["end_at"] != original["until"][0]
                # Successful new headline must not replace the old histogram while it loads.
                await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
                await expect(
                    page.get_by_text(
                        "Refreshing session counts; showing the last counts loaded.", exact=True
                    )
                ).to_be_visible()
                assert (
                    await table.locator("tbody tr td:first-child").all_text_contents()
                    == original_periods
                )
                note = page.get_by_role("status").filter(has_text="Session counts cover")
                await expect(note).to_contain_text("Usage totals above cover the refreshed range.")
                assert await note.locator("time").evaluate_all(
                    "nodes => nodes.map(node => node.dateTime)"
                ) == [original["since"][0], original["until"][0]]
                await expect(table).not_to_contain_text("activity tokens")
                release.set()
                await expect(
                    page.get_by_text(
                        "Couldn't refresh session counts; showing the last counts loaded.",
                        exact=True,
                    )
                ).to_be_visible()
                await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
                assert (
                    await table.locator("tbody tr td:first-child").all_text_contents()
                    == original_periods
                )
                # A different filter cannot borrow the retained writer histogram.
                await page.get_by_label("Specialist", exact=True).select_option("asker")
                await expect(page.get_by_text("1 started", exact=True)).to_have_count(0)
                await expect(table).to_have_count(0)
                await expect(note).to_have_count(0)
                await page.wait_for_load_state("networkidle")
                # Restore the scope and complete a later refresh: the warning and old bounds go.
                await page.unroute("**/api/sessions/overview/counts?*", delayed_failure)
                await page.get_by_label("Specialist", exact=True).select_option("writer")
                await page.get_by_role("button", name="Refresh", exact=True).click()
                await expect(page.get_by_text("1 started", exact=True)).to_be_visible()
                await expect(table).to_contain_text("14 activity tokens")
                await expect(note).to_have_count(0)
                assert (
                    await table.locator("tbody tr td:first-child").all_text_contents()
                    != original_periods
                )
            finally:
                release.set()
                if started.is_set():
                    await asyncio.wait_for(settled.wait(), 5)
                await page.unroute("**/api/sessions/overview/counts?*", delayed_failure)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "start,end,unit,hours",
    [
        ("2026-11-01T05:30:00.000001Z", "2026-11-01T07:30:00.000002Z", "hour", [0.5, 1, 0.5]),
        ("2026-03-07T05:00:00Z", "2026-03-10T04:00:00Z", "day", [24, 23, 24]),
    ],
)
def test_viewer_timezone_uses_native_dst_boundaries(monkeypatch, start, end, unit, hours):
    from playwright.async_api import Browser

    create_context = Browser.new_context

    async def in_zone(browser, **kwargs):
        return await create_context(browser, timezone_id="America/New_York", **kwargs)

    monkeypatch.setattr(Browser, "new_context", in_zone)

    async def exercise():
        runtime, _ = await _usage_runtime()
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, responses):
            search = {"range": "custom", "start_at": start, "end_at": end}
            await page.goto(f"{base}/cayu/usage?{urlencode(search)}", wait_until="networkidle")
            response = next(
                r
                for r in responses
                if "/sessions/overview/counts?" in r.url and "bucket_unit=" in r.url
            )
            query = parse_qs(urlsplit(response.url).query)
            assert query["since"] == [start] and query["until"] == [end]
            assert query["timezone"] == ["America/New_York"]
            data = await response.json()
            buckets = data["series"]["buckets"]
            assert data["series"]["unit"] == unit
            actual = [
                (
                    datetime.fromisoformat(b["end_at"]) - datetime.fromisoformat(b["start_at"])
                ).total_seconds()
                / 3600
                for b in buckets
            ]
            assert actual == pytest.approx(hours)
            table = page.get_by_role("table", name=f"Sessions started per {unit}")
            await expect(table.locator("tbody tr")).to_have_count(len(hours))

    asyncio.run(exercise())
