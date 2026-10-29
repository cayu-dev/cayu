"""Usage cost presentation through the production page and signed viewer roles."""

import asyncio
import os
from urllib.parse import urlencode

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser
from tests.server.test_usage_cost_details import START, WINDOW, prices

from cayu import CayuApp, RunRequest
from cayu.events import Event, EventType
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_USAGE_COSTS_LIVE") != "1",
    reason="Opt-in focused Usage cost Chromium acceptance.",
)


async def cost_runtime():
    store = InMemorySessionStore()
    for index in range(12):
        costly = index == 11
        session_id = f"cost-session-{index}"
        await store.create(
            RunRequest(
                session_id=session_id,
                agent_name="writer",
                title="Costly report" if costly else f"Routine report {index}",
                messages=[],
            ),
            identity=SessionIdentity(
                provider_name="fake", model="expensive" if costly else "cheap"
            ),
        )
        await store.append_event(
            session_id,
            Event(
                session_id=session_id,
                timestamp=START,
                type=EventType.MODEL_COMPLETED,
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "expensive" if costly else "cheap",
                        "input_tokens": 10 if costly else 1000,
                        "total_tokens": 10 if costly else 1000,
                    }
                },
            ),
        )
    return CayuApp(session_store=store, enable_logging=False)


@pytest.mark.parametrize("role", ["admin", "member"])
def test_usage_costs_use_server_order_and_metadata_without_row_reads(role):
    async def run():
        runtime = await cost_runtime()
        async with _home_browser(
            role=role,
            nested=True,
            mobile=role == "member",
            seeded_runtime=runtime,
            dashboard_config={"priceBook": prices()},
        ) as (page, base, requests, responses):
            rollups = []
            page.on(
                "request",
                lambda r: rollups.append(r) if r.url.endswith("/api/usage/rollup") else None,
            )
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **WINDOW})}",
                wait_until="networkidle",
            )
            assert len(rollups) == 1
            assert rollups[0].post_data_json.get("session_order") == "cost"
            await expect(
                page.get_by_role("heading", name="Top sessions by cost", exact=True)
            ).to_be_visible()
            table = page.get_by_role("table", name="Top sessions by cost", exact=True)
            await expect(table.locator("tbody tr").first).to_contain_text("Costly report")
            await expect(table.locator("tbody tr").first).to_contain_text("writer")
            await expect(table.locator("tbody tr")).to_have_count(10)
            await expect(
                page.get_by_text("2 more sessions used 2K tokens.", exact=False)
            ).to_be_visible()
            await expect(
                page.get_by_role("table", name="Cost and usage by provider and model")
            ).to_contain_text("USD")
            assert not [
                p
                for p in requests
                if "/sessions/cost-session-" in p or p.endswith("/sessions/overview")
            ]
            assert not [r for r in responses if r.status in (401, 403)]
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert await page.locator("table td span").evaluate_all(
                "nodes => nodes.filter(n => n.textContent.includes('USD')).every(n => n.getBoundingClientRect().height <= 24)"
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode",
    ["no-pricing", "partial", "mixed", "unsupported", "malformed-capability", "malformed-ranking"],
)
def test_usage_cost_fallbacks_are_explicit_and_do_not_fetch_rows(mode):
    async def run():
        runtime = await cost_runtime()
        price_book = prices()
        if mode == "partial":
            price_book["prices"] = price_book["prices"][:1]
        elif mode == "mixed":
            price_book["prices"][1]["schedules"][0]["pricing"]["currency"] = "EUR"
        config = {} if mode == "no-pricing" else {"priceBook": price_book}
        async with _home_browser(
            role="member", seeded_runtime=runtime, dashboard_config=config
        ) as (page, base, requests, _responses):
            rollups = []
            page.on(
                "request",
                lambda r: rollups.append(r) if r.url.endswith("/api/usage/rollup") else None,
            )
            if mode in ("unsupported", "malformed-capability"):

                async def capability(route):
                    response = await route.fetch()
                    data = await response.json()
                    data["capabilities"]["surfaces"]["usage_cost_details"]["read"]["enabled"] = (
                        False if mode == "unsupported" else "true"
                    )
                    await route.fulfill(response=response, json=data)

                await page.route("**/api/contract", capability)
            elif mode == "malformed-ranking":

                async def corrupt(route):
                    response = await route.fetch()
                    data = await response.json()
                    data["session_ranking"] = None
                    await route.fulfill(response=response, json=data)

                await page.route("**/api/usage/rollup", corrupt)
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **WINDOW})}",
                wait_until="networkidle",
            )
            if mode == "malformed-ranking":
                await expect(
                    page.get_by_text(
                        "The server did not return a valid session cost ranking.", exact=True
                    )
                ).to_be_visible(timeout=15000)
                await expect(page.get_by_role("table", name="Top sessions by cost")).to_have_count(
                    0
                )
            else:
                await expect(
                    page.get_by_role("heading", name="Top sessions by tokens", exact=True)
                ).to_be_visible()
                explanation = {
                    "no-pricing": "Sessions are ranked by tokens because no price book is configured.",
                    "partial": "Sessions are ranked by tokens because some usage has no complete price.",
                    "mixed": "Sessions are ranked by tokens because costs use different currencies.",
                    "unsupported": "Cost ranking is not supported by this agent. Sessions are ranked by tokens.",
                    "malformed-capability": "Cost ranking is not supported by this agent. Sessions are ranked by tokens.",
                }[mode]
                await expect(page.get_by_text(explanation, exact=True)).to_be_visible()
                if mode == "partial":
                    table = page.get_by_role("table", name="Cost and usage by provider and model")
                    await expect(table).to_contain_text("partial")
                    await expect(table).to_contain_text("Unavailable")
                if mode == "mixed":
                    table = page.get_by_role("table", name="Cost and usage by provider and model")
                    await expect(table).to_contain_text("EUR")
                    await expect(table).to_contain_text("USD")
            assert rollups
            assert all(
                (r.post_data_json.get("session_order") is None)
                == (mode in ("unsupported", "malformed-capability"))
                for r in rollups
            )
            assert not [
                p
                for p in requests
                if "/sessions/cost-session-" in p or p.endswith("/sessions/overview")
            ]

    asyncio.run(run())


def test_usage_cost_refresh_retains_snapshot_and_scope_changes_replace_it():
    async def run():
        runtime = await cost_runtime()
        async with _home_browser(
            seeded_runtime=runtime, dashboard_config={"priceBook": prices()}
        ) as (page, base, requests, _responses):
            bodies = []
            fail = False

            async def transport(route):
                bodies.append(route.request.post_data_json)
                if fail:
                    await route.fulfill(status=503, json={"detail": "Unavailable"})
                else:
                    response = await route.fetch()
                    data = await response.json()
                    data["session_breakdown"]["groups"][0]["title"] = None
                    data["session_breakdown"]["groups"][0]["agent_name"] = None
                    await route.fulfill(response=response, json=data)

            await page.route("**/api/usage/rollup", transport)
            query = urlencode({"range": "custom", **WINDOW})
            await page.goto(f"{base}/cayu/usage?{query}", wait_until="networkidle")
            table = page.get_by_role("table", name="Top sessions by cost", exact=True)
            await expect(table.locator("tbody tr").first).to_contain_text("Untitled session")
            await expect(table.locator("tbody tr").first).to_contain_text("Unknown specialist")
            assert len([p for p in requests if p.endswith("/sessions/overview/counts")]) == 1
            fail = True
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(
                page.get_by_text("Couldn't refresh; showing the last numbers loaded.", exact=True)
            ).to_be_visible(timeout=15000)
            await expect(table.locator("tbody tr").first).to_contain_text("Untitled session")
            assert all(
                body["start_at"] == bodies[0]["start_at"] and body["end_at"] == bodies[0]["end_at"]
                for body in bodies
            )
            fail = False
            await page.goto(f"{base}/cayu/usage?{query}&model=cheap", wait_until="networkidle")
            assert bodies[-1]["session_filter"]["model"] == "cheap"
            await expect(
                page.get_by_role("table", name="Cost and usage by provider and model")
            ).not_to_contain_text("expensive")
            assert not [
                p
                for p in requests
                if "/sessions/cost-session-" in p or p.endswith("/sessions/overview")
            ]

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["zero", "empty", "sampled-hosted"])
def test_usage_cost_zero_empty_and_uncertain_projections_remain_distinct(mode):
    async def run():
        runtime = CayuApp(enable_logging=False) if mode == "empty" else await cost_runtime()
        price_book = prices()
        if mode == "zero":
            for price in price_book["prices"]:
                for tier in price["schedules"][0]["pricing"]["standard"]:
                    tier["input_per_million"] = "0"
                    tier["output_per_million"] = "0"
        async with _home_browser(
            seeded_runtime=runtime, mobile=True, dashboard_config={"priceBook": price_book}
        ) as (page, base, _, _responses):
            if mode == "sampled-hosted":

                async def projection(route):
                    response = await route.fetch()
                    data = await response.json()
                    data["session_ranking"].update(
                        effective_order="tokens",
                        unavailable_reason="pricing_projection_incomplete",
                        currency=None,
                    )
                    data["session_breakdown"]["order"] = "tokens"
                    data["session_breakdown"]["accuracy"] = {
                        "kind": "sampled",
                        "reason": "Fixture sampling",
                    }
                    data["session_breakdown"]["remainder"] = None
                    data["session_breakdown"]["groups"][1]["title"] = (
                        "A long human session title " * 9
                    )
                    data["session_breakdown"]["groups"][1]["agent_name"] = "long-specialist-" * 16
                    for field in (
                        "provider_cost_breakdown",
                        "model_cost_breakdown",
                        "session_cost_breakdown",
                    ):
                        data[field]["accuracy"] = {"kind": "sampled", "reason": "Fixture sampling"}
                        for group in data[field]["groups"]:
                            cost = group["cost"]
                            cost["unpriced_reasons"] = [
                                {
                                    "reason": "Hosted resource has no price",
                                    "unpriced_reason": "missing_pricing",
                                    "model_steps": "0",
                                }
                            ]
                        data[field]["groups"][0]["cost"]["currencies"] = []
                    await route.fulfill(response=response, json=data)

                await page.route("**/api/usage/rollup", projection)
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **WINDOW})}",
                wait_until="networkidle",
            )
            if mode == "empty":
                await expect(
                    page.get_by_text("No matching sessions in this range", exact=True)
                ).to_be_visible()
            else:
                table = page.get_by_role(
                    "table",
                    name="Top sessions by tokens"
                    if mode == "sampled-hosted"
                    else "Top sessions by cost",
                    exact=True,
                )
                if mode == "zero":
                    await expect(table).to_contain_text("USD")
                    await expect(table).to_contain_text("0.00")
                    await expect(table).not_to_contain_text("Unavailable")
                else:
                    await expect(table).to_contain_text("Unavailable")
                    await expect(table).to_contain_text("partial")
                    await expect(
                        page.get_by_text(
                            "Session detail is limited; see Debug for coverage.", exact=False
                        )
                    ).to_be_visible()
                    await expect(
                        page.get_by_text(
                            "Sessions are ranked by tokens because complete cost data is unavailable for this range. Narrow the filters or range.",
                            exact=True,
                        )
                    ).to_be_visible()
                    await expect(
                        page.get_by_role("table", name="Cost and usage by provider and model")
                    ).to_contain_text("Unavailable")
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")

    asyncio.run(run())


@pytest.mark.parametrize(
    "endpoint,status", [("rollup", 401), ("rollup", 403), ("counts", 401), ("counts", 403)]
)
def test_usage_denial_discards_report_through_transient_errors_and_remount(endpoint, status):
    async def run():
        async with _home_browser(
            seeded_runtime=await cost_runtime(), dashboard_config={"priceBook": prices()}
        ) as (page, base, _, _responses):
            phase = "success"
            denied = asyncio.Event()

            async def transport(route):
                target = "rollup" if route.request.url.endswith("/usage/rollup") else "counts"
                if phase == "success":
                    await route.continue_()
                else:
                    code = status if phase == "denied" and target == endpoint else 503
                    await route.fulfill(
                        status=code,
                        json={"detail": "Access denied" if code != 503 else "Unavailable"},
                    )
                    if code == status:
                        denied.set()

            await page.route("**/api/usage/rollup", transport)
            await page.route("**/api/sessions/overview/counts?*", transport)
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **WINDOW})}",
                wait_until="networkidle",
            )
            await expect(
                page.get_by_role("table", name="Top sessions by cost", exact=True)
            ).to_be_visible()
            await page.get_by_text("Debug", exact=True).click()
            await expect(page.get_by_text('"session_ranking":', exact=False)).to_be_visible()
            phase = "denied"
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await asyncio.wait_for(denied.wait(), 10)
            await expect(
                page.get_by_role("table", name="Top sessions by cost", exact=True)
            ).to_have_count(0)
            await expect(page.get_by_text('"session_ranking":', exact=False)).to_have_count(0)
            await expect(page.get_by_text("0 started", exact=True)).to_have_count(0)
            phase = "transient"
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(page.get_by_role("button", name="Refresh", exact=True)).to_be_enabled(
                timeout=15000
            )
            await expect(
                page.get_by_role("table", name="Top sessions by cost", exact=True)
            ).to_have_count(0)
            await page.get_by_role("link", name="Home", exact=True).last.click()
            await page.go_back()
            await expect(page.get_by_role("button", name="Refresh", exact=True)).to_be_enabled(
                timeout=15000
            )
            await expect(
                page.get_by_role("table", name="Top sessions by cost", exact=True)
            ).to_have_count(0)
            phase = "success"
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(
                page.get_by_role("heading", name="Top sessions by cost", exact=True)
            ).to_be_visible()

    asyncio.run(run())


def test_usage_denial_fences_delayed_pre_denial_success():
    async def run():
        async with _home_browser(
            seeded_runtime=await cost_runtime(), dashboard_config={"priceBook": prices()}
        ) as (page, base, _, _responses):
            phase = "success"
            pending = asyncio.Event()
            release = asyncio.Event()
            finished = asyncio.Event()

            async def rollup(route):
                if phase != "hold":
                    await route.continue_()
                    return
                response = await route.fetch()
                pending.set()
                await release.wait()
                try:
                    await route.fulfill(response=response)
                except Exception as error:
                    # An aborted browser read may already have disposed its route.
                    assert (
                        "closed" in str(error)
                        or "handled" in str(error)
                        or "canceled" in str(error)
                    )
                finally:
                    finished.set()

            async def counts(route):
                if phase != "hold":
                    await route.continue_()
                else:
                    await pending.wait()
                    await route.fulfill(status=403, json={"detail": "Access denied"})

            await page.route("**/api/usage/rollup", rollup)
            await page.route("**/api/sessions/overview/counts?*", counts)
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **WINDOW})}",
                wait_until="networkidle",
            )
            await expect(
                page.get_by_role("table", name="Top sessions by cost", exact=True)
            ).to_be_visible()
            phase = "hold"
            try:
                await page.get_by_role("button", name="Refresh", exact=True).click()
                await asyncio.wait_for(pending.wait(), 10)
                await expect(
                    page.get_by_role("table", name="Top sessions by cost", exact=True)
                ).to_have_count(0)
                release.set()
                await asyncio.wait_for(finished.wait(), 10)
                await expect(
                    page.get_by_role("table", name="Top sessions by cost", exact=True)
                ).to_have_count(0)
                phase = "success"
                await page.get_by_role("button", name="Refresh", exact=True).click()
                await expect(
                    page.get_by_role("table", name="Top sessions by cost", exact=True)
                ).to_be_visible()
            finally:
                release.set()

    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 403])
def test_usage_try_again_recovers_report_and_histogram_in_same_custom_window(status):
    async def run():
        async with _home_browser(
            seeded_runtime=await cost_runtime(), dashboard_config={"priceBook": prices()}
        ) as (page, base, _, _responses):
            denied = False
            reads = []

            async def transport(route):
                reads.append(route.request)
                if denied:
                    code = status if route.request.url.endswith("/usage/rollup") else 503
                    await route.fulfill(
                        status=code,
                        json={"detail": "Access denied" if code != 503 else "Unavailable"},
                    )
                else:
                    await route.continue_()

            await page.route("**/api/usage/rollup", transport)
            await page.route("**/api/sessions/overview/counts?*", transport)
            await page.goto(
                f"{base}/cayu/usage?{urlencode({'range': 'custom', **WINDOW})}",
                wait_until="networkidle",
            )
            await expect(page.get_by_text("0 started", exact=True)).to_be_visible()
            denied = True
            await page.get_by_role("button", name="Refresh", exact=True).click()
            await expect(page.get_by_role("button", name="Try again", exact=True)).to_be_visible()
            await expect(
                page.get_by_role("table", name="Top sessions by cost", exact=True)
            ).to_have_count(0)
            await expect(page.get_by_text("0 started", exact=True)).to_have_count(0)
            denied = False
            recovery_start = len(reads)
            await page.get_by_role("button", name="Try again", exact=True).click()
            await expect(
                page.get_by_role("table", name="Top sessions by cost", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("0 started", exact=True)).to_be_visible()
            assert any("/sessions/overview/counts?" in r.url for r in reads[recovery_start:])
            bodies = [r.post_data_json for r in reads if r.url.endswith("/usage/rollup")]
            assert all(
                b["start_at"] == bodies[0]["start_at"] and b["end_at"] == bodies[0]["end_at"]
                for b in bodies
            )

    asyncio.run(run())
