"""Opt-in actual-browser acceptance for bounded public Sessions search."""

from __future__ import annotations

import asyncio
import os
import socket
from contextlib import asynccontextmanager, suppress
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("playwright.async_api")

import uvicorn
from cryptography.hazmat.primitives.asymmetric import ec
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright, expect
from tests.core.test_cloud_reporting import PRICE, _app, _drain
from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

from cayu import Message
from cayu.server import DashboardConfig, ServerConfig, create_server
from cayu.sessions import ForkSessionRequest, RunRequest, SessionIdentity
from cayu.sessions._run_identity import eval_trial_scope

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_SESSION_SEARCH_LIVE") != "1", reason="Opt-in packaged Chromium acceptance."
)


@asynccontextmanager
async def search_browser(
    *, pause=False, presentation=False, member=False, runtime=None, wrapper=None, priced=False
):
    provided_runtime = runtime is not None
    runtime = runtime or _app()
    for name in (
        []
        if presentation or provided_runtime
        else ["old-needle", *(f"filler-{index}" for index in range(101)), "fresh-match"]
    ):
        await runtime.session_store.create(
            RunRequest(
                session_id=name,
                agent_name="writer",
                title=f"Title {name}",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
    if presentation:
        for name, title, labels, prompt, agent in [
            ("authored", "Audit the ledger", {}, "Repeated preamble", "writer"),
            ("labels", None, {"team": "finance"}, "Repeated preamble", "writer"),
            ("message", None, {}, "Inspect the invoice", "writer"),
            ("waiting", "Answer needed", {}, "Question", "asker"),
        ]:
            await _drain(
                runtime.run(
                    RunRequest(
                        session_id=name,
                        agent_name=agent,
                        title=title,
                        labels=labels,
                        messages=[Message.text("user", prompt)],
                    )
                )
            )
        await runtime.session_store.set_summary(
            "authored", "One finding needs review", tone="attention"
        )
        await runtime.session_store.set_summary("labels", "One finding handled", tone="neutral")
        await runtime.session_store.create(
            RunRequest(session_id="id-only", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name=PRICE.provider_name, model=PRICE.model),
        )
        async with eval_trial_scope():
            await _drain(
                runtime.run(
                    RunRequest(
                        session_id="eval-only",
                        agent_name="writer",
                        title="Evaluation trial",
                        messages=[Message.text("user", "go")],
                    )
                )
            )
        await _drain(
            runtime.fork_session(
                ForkSessionRequest(source_session_id="message", session_id="fork-only")
            )
        )
    key = ec.generate_private_key(ec.SECP256R1()) if member else None
    from cayu.budgets.pricing import default_price_book

    dashboard = DashboardConfig(
        directory=os.environ.get("CAYU_SESSION_SEARCH_DIRECTORY"),
        runtime_config={"priceBook": default_price_book()} if priced else {},
    )
    config = (
        ServerConfig.protected(_auth(FakeJwks(key)), dashboard=dashboard)
        if key
        else ServerConfig.local_development(dashboard=dashboard)
    )
    app = create_server(runtime, config=config)
    reached, release = asyncio.Event(), asyncio.Event()
    requests = []

    @app.middleware("http")
    async def observe(request, call_next):
        if request.url.path == "/api/sessions":
            requests.append(dict(request.query_params))
            if (
                pause
                and request.query_params.get("q") == "old-needle"
                and request.query_params.get("cursor")
            ):
                reached.set()
                await release.wait()
        return await call_next(request)

    if wrapper is not None:
        app = wrapper(app)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        server = uvicorn.Server(uvicorn.Config(app, lifespan="on", log_level="error"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    await asyncio.sleep(0.01)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
                )
                try:
                    page = await browser.new_page(
                        extra_http_headers=_as(_token(key, role="member")) if key else {},
                        viewport={"width": 390 if member else 1440, "height": 900},
                        color_scheme="dark",
                    )
                    page.set_default_timeout(5_000)
                    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
                    yield page, base, requests, reached, release
                finally:
                    release.set()
                    await browser.close()
        finally:
            release.set()
            server.should_exit = True
            await asyncio.wait_for(task, 10)


def test_session_search_finds_match_beyond_first_hundred_and_proves_exhaustion():
    async def run():
        async with search_browser() as (page, base, requests, _, _):
            await page.goto(f"{base}/cayu/sessions?q=old-needle")
            await expect(
                page.get_by_role("link", name="Title old-needle", exact=True)
            ).to_be_visible()
            assert len([row for row in requests if row.get("q") == "old-needle"]) == 2
            await page.get_by_placeholder("Search by title, ID or label", exact=True).fill("absent")
            await expect(
                page.get_by_text("No sessions match these filters", exact=True)
            ).to_be_visible()
            assert len([row for row in requests if row.get("q") == "absent"]) == 2
            assert parse_qs(urlsplit(page.url).query)["q"] == ["absent"]

    asyncio.run(run())


def test_query_change_cancels_empty_page_search_and_ignores_late_old_result():
    async def run():
        async with search_browser(pause=True) as (page, base, requests, reached, release):
            failed = []
            page.on("requestfailed", lambda request: failed.append(request.url))
            await page.goto(f"{base}/cayu/sessions?q=old-needle")
            await asyncio.wait_for(reached.wait(), 5)
            await expect(page.get_by_text("Searching sessions…", exact=True).first).to_be_visible()
            await expect(
                page.get_by_text("No sessions match these filters", exact=True)
            ).to_have_count(0)
            await page.get_by_placeholder("Search by title, ID or label", exact=True).fill(
                "fresh-match"
            )
            await expect(
                page.get_by_role("link", name="Title fresh-match", exact=True)
            ).to_be_visible()
            async with asyncio.timeout(3):
                while not any("cursor=" in url and "q=old-needle" in url for url in failed):
                    await asyncio.sleep(0.01)
            release.set()
            await page.wait_for_timeout(200)
            await expect(
                page.get_by_role("link", name="Title old-needle", exact=True)
            ).to_have_count(0)
            assert len([row for row in requests if row.get("q") == "old-needle"]) == 2
            assert parse_qs(urlsplit(page.url).query)["q"] == ["fresh-match"]

    asyncio.run(run())


@pytest.mark.parametrize("member", [False, True], ids=["admin", "member-mobile"])
def test_session_list_canonical_presentation_roles_and_population_toggles(member):
    async def run():
        async with search_browser(presentation=True, member=member) as (page, base, requests, _, _):
            await page.goto(f"{base}/cayu/sessions")
            await expect(
                page.get_by_role("link", name="Audit the ledger", exact=True)
            ).to_be_visible()
            await expect(page.get_by_role("link", name="team=finance", exact=True)).to_be_visible()
            await expect(
                page.get_by_role("link", name="Inspect the invoice", exact=True)
            ).to_be_visible()
            await expect(page.get_by_role("link", name="id-only", exact=True)).to_be_visible()
            neutral = page.get_by_text("One finding handled", exact=True)
            await expect(neutral).to_be_visible()
            assert "text-warning" not in (await neutral.get_attribute("class") or "")
            summary = page.get_by_text("One finding needs review", exact=True)
            await expect(summary).to_be_visible()
            assert "text-warning" in (await summary.get_attribute("class") or "")
            await expect(
                page.get_by_role("table").get_by_text("Needs your answer", exact=True)
            ).to_be_visible()
            await expect(
                page.get_by_role("columnheader", name="Provider", exact=True)
            ).to_have_count(0)
            await expect(page.get_by_role("columnheader", name="State", exact=True)).to_be_visible()
            await expect(
                page.get_by_role("link", name="Evaluation trial", exact=True)
            ).to_have_count(0)
            await expect(page.get_by_text("fork-only", exact=True)).to_have_count(0)
            if screenshots := os.environ.get("CAYU_SESSION_SCREENSHOTS"):
                await page.screenshot(
                    path=f"{screenshots}/sessions-{'member-mobile' if member else 'admin'}.png",
                    full_page=True,
                )
            if member:
                await expect(
                    page.get_by_role("button", name="Start a session", exact=True)
                ).to_have_count(0)
                results = page.get_by_role("region", name="Session results", exact=True)
                await results.focus()
                await expect(results).to_be_focused()
                await page.keyboard.press("ArrowRight")
                async with asyncio.timeout(2):
                    while await results.evaluate("element => element.scrollLeft") == 0:
                        await asyncio.sleep(0.01)
            else:
                await expect(
                    page.get_by_role("button", name="Start a session", exact=True)
                ).to_be_visible()
            await page.get_by_label("Include eval trials", exact=True).check()
            await expect(
                page.get_by_role("link", name="Evaluation trial", exact=True)
            ).to_be_visible()
            await page.get_by_label("Include forks", exact=True).check()
            await expect(page.get_by_text("fork-only", exact=True)).to_be_visible()
            await expect(page.get_by_text("Fork", exact=True)).to_be_visible()
            await page.get_by_label("State", exact=True).select_option("waiting")
            await expect(page.get_by_role("link", name="Answer needed", exact=True)).to_be_visible()
            await expect(page.locator("tbody tr")).to_have_count(1)
            assert requests[-1]["include_eval"] == "true"
            assert requests[-1]["include_forks"] == "true"
            assert requests[-1]["include"] == "overview"
            assert "loaded on this page" in await page.locator("main").inner_text()

    asyncio.run(run())


def test_session_list_preserves_existing_discovery_journey(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("dashboard_behavior_live")

    async def run():
        runtime, provider, _, _, directory = await example._seed_app()
        faults = None

        def wrap(app):
            nonlocal faults
            faults = example.MutationDisconnectFaults(app, provider)
            return faults

        try:
            async with search_browser(runtime=runtime, wrapper=wrap) as (
                page,
                base,
                requests,
                _,
                _,
            ):
                await page.goto(f"{base}/cayu/sessions")
                await example._exercise_session_discovery(page, faults)
                assert requests
                assert all(row.get("include") == "overview" for row in requests)
                assert all(row.get("include_eval") == "false" for row in requests)
                assert all(row.get("include_forks") == "false" for row in requests)
        finally:
            await runtime.aclose()
            directory.cleanup()

    asyncio.run(run())


def test_session_list_capability_and_read_errors_do_not_look_like_empty_results():
    async def run():
        async with search_browser(presentation=True) as (page, base, requests, _, _):

            async def unavailable(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["surfaces"]["session_overview_queries"]["read"] = {
                    "enabled": False,
                    "unavailable_reason": "not_configured",
                }
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", unavailable)
            await page.goto(f"{base}/cayu/sessions")
            await expect(page.get_by_test_id("dashboard-capability-unavailable")).to_be_visible()
            assert requests == []
            await page.unroute("**/api/contract", unavailable)
            await page.route(
                "**/api/sessions?**",
                lambda route: route.fulfill(status=422, json={"detail": "Invalid session cursor."}),
            )
            await page.reload()
            await expect(page.get_by_text("Invalid session cursor.", exact=True)).to_be_visible()
            await expect(page.get_by_text("No sessions yet", exact=True)).to_have_count(0)
            await expect(page.get_by_role("button", name="Retry", exact=True)).to_be_visible()
            await page.unroute("**/api/sessions?**")
            await page.get_by_role("button", name="Retry", exact=True).click()
            await expect(
                page.get_by_role("link", name="Audit the ledger", exact=True)
            ).to_be_visible()

    asyncio.run(run())


def test_session_list_uses_canonical_currency_and_zero_priced_cost():
    async def run():
        async with search_browser(presentation=True, priced=True) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions")
            await expect(page.get_by_role("columnheader", name="Cost", exact=True)).to_be_visible()
            await expect(page.get_by_role("columnheader", name="Tokens", exact=True)).to_have_count(
                0
            )
            response = await page.request.get(
                f"{base}/api/sessions?include=overview&ids=authored&ids=id-only"
            )
            assert response.ok
            overviews = {row["session_id"]: row for row in (await response.json())["overviews"]}
            from decimal import Decimal

            assert Decimal(overviews["authored"]["cost"]["amount"]) > 0
            assert Decimal(overviews["id-only"]["cost"]["amount"]) == 0
            for session_id, title in [("authored", "Audit the ledger"), ("id-only", "id-only")]:
                row = page.get_by_role("link", name=title, exact=True).locator("xpath=ancestor::tr")
                await expect(row).to_contain_text(overviews[session_id]["cost"]["currency"])
                await expect(row.get_by_text("Partial cost", exact=True)).to_have_count(0)

    asyncio.run(run())


def test_session_start_preserves_unknown_outcome_and_reobserves_without_resubmission():
    import json

    async def run():
        async with search_browser(presentation=True) as (page, base, _, _, _):
            submissions = []
            run_requests = []
            page.on(
                "request",
                lambda request: (
                    run_requests.append(request)
                    if urlsplit(request.url).path == "/api/run"
                    else None
                ),
            )

            async def lose_observation(route):
                submissions.append(route.request.post_data_json)
                response = await route.fetch()
                assert response.ok
                session_id = submissions[-1]["session_id"]
                error = {
                    "type": "stream.error",
                    "kind": "observer",
                    "code": "event_frame_too_large",
                    "error": "Injected observation loss",
                    "error_type": "ObserverFailure",
                    "retryable": False,
                    "session_id": session_id,
                }
                await route.fulfill(
                    status=200,
                    content_type="text/event-stream",
                    body=f"event: error\ndata: {json.dumps(error)}\n\n",
                )

            await page.route("**/api/run", lose_observation)
            await page.route(
                "**/api/sessions/*/events?**",
                lambda route: route.fulfill(
                    status=422, json={"detail": "Injected event observation unavailable"}
                ),
            )
            await page.goto(f"{base}/cayu/sessions")
            await page.get_by_role("button", name="Start a session", exact=True).click()
            await expect(page.get_by_role("tab", name="Run a specialist")).to_have_attribute(
                "aria-selected", "true"
            )
            await expect(page.get_by_role("tab", name="Create a task")).to_have_attribute(
                "aria-disabled", "true"
            )
            await page.locator("#start-specialist").select_option("writer")
            await page.get_by_label("Prompt", exact=True).fill("Start from the Sessions page")
            await page.get_by_role("button", name="Start", exact=True).click()
            await expect(
                page.locator('[data-mutation-transport-phase="transport_failed"]')
            ).to_be_visible(timeout=25_000)
            assert urlsplit(page.url).path == "/cayu/sessions"
            await expect(page.get_by_role("button", name="Start", exact=True)).to_be_disabled()
            assert len(submissions) == 1
            await page.unroute("**/api/sessions/*/events?**")
            await page.unroute("**/api/run", lose_observation)
            await page.get_by_role("button", name="Retry observation", exact=True).click()
            await page.wait_for_url(f"**/sessions/{submissions[0]['session_id']}", timeout=15_000)
            assert len(submissions) == 1
            assert sum("last-event-id" not in request.headers for request in run_requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize("close_while_observing", [False, True])
def test_session_start_success_and_close_keep_dispatched_run(close_while_observing):
    async def run():
        async with search_browser(presentation=True) as (page, base, _, _, _):
            dispatched, release = asyncio.Event(), asyncio.Event()
            submitted = []

            async def barrier(route):
                submitted.append(route.request.post_data_json)
                response = await route.fetch()
                assert response.ok
                dispatched.set()
                await release.wait()
                try:
                    await route.fulfill(response=response)
                except PlaywrightError:
                    if not close_while_observing:
                        raise

            await page.route("**/api/run", barrier)
            try:
                await page.goto(f"{base}/cayu/sessions")
                await page.get_by_role("button", name="Start a session", exact=True).click()
                await page.locator("#start-specialist").select_option("writer")
                await page.get_by_label("Prompt", exact=True).fill("Durable submitted run")
                await page.get_by_role("button", name="Start", exact=True).click()
                await asyncio.wait_for(dispatched.wait(), 10)
                if close_while_observing:
                    await page.get_by_role("button", name="Close", exact=True).click()
                    await expect(page.get_by_label("Prompt", exact=True)).to_have_count(0)
                release.set()
                session_id = submitted[0]["session_id"]
                if close_while_observing:
                    # A subsequent read and interaction happen after dispatch and close;
                    # observation cancellation must not become a runtime interruption.
                    response = await page.request.get(f"{base}/api/sessions/{session_id}")
                    assert response.ok
                    assert (await response.json())["status"] == "completed"
                    await page.get_by_role("button", name="Start a session", exact=True).click()
                    await expect(page.get_by_label("Prompt", exact=True)).to_have_value("")
                    assert urlsplit(page.url).path == "/cayu/sessions"
                else:
                    await page.wait_for_url(f"**/sessions/{session_id}", timeout=10_000)
                assert len(submitted) == 1
            finally:
                release.set()

    asyncio.run(run())


def test_session_search_keeps_keyboard_focus_across_commits_and_syncs_history():
    async def run():
        async with search_browser(presentation=True) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions?include_eval=true")
            search = page.locator("main").get_by_role("textbox", name="Search sessions", exact=True)
            await page.get_by_label("State", exact=True).select_option("completed")
            await page.wait_for_url("**state=completed*")
            history_length = await page.evaluate("history.length")
            await search.focus()
            await page.keyboard.type("Audit")
            await page.evaluate("document.body.style.minHeight = '2000px'; window.scrollTo(0, 100)")
            await page.wait_for_url("**q=Audit*")
            assert await page.evaluate("window.scrollY") == 100
            await expect(search).to_be_focused()
            await page.keyboard.type(" the ledger")
            await expect(search).to_have_value("Audit the ledger")
            await expect(
                page.get_by_role("link", name="Audit the ledger", exact=True)
            ).to_be_visible()
            async with asyncio.timeout(5):
                while parse_qs(urlsplit(page.url).query).get("q") != ["Audit the ledger"]:
                    await asyncio.sleep(0.01)
            await expect(search).to_be_focused()
            assert await page.evaluate("history.length") == history_length
            await page.go_back()
            await expect(search).to_have_value("")
            await page.go_forward()
            await expect(search).to_have_value("Audit the ledger")
            await search.focus()
            await page.keyboard.type(" discarded")
            assert await page.evaluate("history.length") == history_length
            await page.go_back()
            await expect(search).to_have_value("")
            await page.wait_for_timeout(400)
            assert "q" not in parse_qs(urlsplit(page.url).query)
            await page.go_forward()
            await expect(search).to_have_value("Audit the ledger")
            await page.get_by_role("button", name="Clear filters", exact=True).click()
            await expect(search).to_have_value("")
            await expect(search).not_to_be_focused()
            await page.wait_for_timeout(400)
            assert "q" not in parse_qs(urlsplit(page.url).query)

    asyncio.run(run())


def test_start_rejected_request_can_be_corrected_without_losing_draft():
    async def run():
        async with search_browser(presentation=True) as (page, base, _, _, _):
            requests = []

            async def reject_first(route):
                import json

                body = route.request.post_data_json
                requests.append(body)
                response = (
                    await route.fetch(post_data=json.dumps({**body, "agent": "removed-agent"}))
                    if len(requests) == 1
                    else await route.fetch()
                )
                await route.fulfill(response=response)

            await page.route("**/api/run", reject_first)
            await page.goto(f"{base}/cayu/sessions")
            await page.get_by_role("button", name="Start a session", exact=True).click()
            await page.locator("#start-specialist").select_option("writer")
            await page.get_by_label("Prompt", exact=True).fill("Preserve my draft")
            start = page.get_by_role("button", name="Start", exact=True)
            await start.click()
            await expect(page.get_by_text("Request rejected", exact=True)).to_be_visible()
            await expect(start).to_be_enabled()
            await expect(page.get_by_label("Prompt", exact=True)).to_have_value("Preserve my draft")
            await start.click()
            await page.wait_for_url("**/sessions/*", timeout=10_000)
            assert len(requests) == 2
            assert requests[0]["session_id"] != requests[1]["session_id"]
            assert requests[1]["prompt"] == "Preserve my draft"

    asyncio.run(run())


def test_start_runtime_failure_opens_existing_failed_session():
    from tests.core.test_cloud_reporting import _Provider

    from cayu import AgentSpec, CayuApp

    def fail(_):
        raise RuntimeError("Deliberate provider failure")

    async def run():
        runtime = CayuApp(enable_logging=False)
        runtime.register_provider(
            _Provider(name=PRICE.provider_name, response_factory=fail), default=True
        )
        runtime.register_agent(AgentSpec(name="writer", model=PRICE.model))
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            submitted = []
            page.on(
                "request",
                lambda request: (
                    submitted.append(request.post_data_json)
                    if urlsplit(request.url).path == "/api/run"
                    else None
                ),
            )

            async def runtime_error_only(route):
                import json

                response = await route.fetch()
                assert response.status in (200, 500)
                session_id = route.request.post_data_json["session_id"]
                durable = await page.request.get(f"{base}/api/sessions/{session_id}")
                assert (await durable.json())["status"] == "failed"
                # Deliver the runtime error without its preceding terminal-event frame.
                # The failed durable session still owns the reported outcome.
                error = {
                    "type": "stream.error",
                    "kind": "runtime",
                    "code": "runtime_failed",
                    "error": "Deliberate provider failure",
                    "error_type": "RuntimeError",
                    "retryable": False,
                    "session_id": session_id,
                }
                await route.fulfill(
                    status=200,
                    content_type="text/event-stream",
                    body=f"event: error\ndata: {json.dumps(error)}\n\n",
                )

            await page.route("**/api/run", runtime_error_only)
            await page.goto(f"{base}/cayu/sessions")
            await page.get_by_role("button", name="Start a session", exact=True).click()
            await page.get_by_label("Prompt", exact=True).fill("Fail durably")
            await page.get_by_role("button", name="Start", exact=True).click()
            async with asyncio.timeout(10):
                while not submitted:
                    await asyncio.sleep(0.01)
            await page.wait_for_url(f"**/sessions/{submitted[0]['session_id']}", timeout=15_000)
            response = await page.request.get(f"{base}/api/sessions/{submitted[0]['session_id']}")
            assert (await response.json())["status"] == "failed"
            assert len(submitted) == 1

    asyncio.run(run())


def test_clear_filters_cancels_draft_when_committed_query_is_unchanged():
    async def run():
        async with search_browser(presentation=True) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions?status=failed")
            search = page.locator("main").get_by_role("textbox", name="Search sessions", exact=True)
            await search.focus()
            await page.keyboard.type("pending")
            await page.get_by_role("button", name="Clear filters", exact=True).click()
            await expect(search).to_have_value("")
            await page.wait_for_timeout(400)
            assert "q" not in parse_qs(urlsplit(page.url).query)

    asyncio.run(run())


def test_overlapping_list_reads_do_not_cancel_shared_title_fallback():
    async def run():
        async with search_browser(presentation=True) as (page, base, requests, _, _):
            reached, release = asyncio.Event(), asyncio.Event()
            await page.add_init_script("""(() => {
                const original = window.fetch.bind(window);
                window.__releaseTitleAbort = () => {};
                window.fetch = async (input, init) => {
                    try { return await original(input, init); }
                    catch (error) {
                        if (String(input).includes('/message/transcript') && error.name === 'AbortError')
                            await new Promise(resolve => { window.__releaseTitleAbort = resolve; });
                        throw error;
                    }
                };
            })()""")

            async def pause_title(route):
                response = await route.fetch()
                reached.set()
                await release.wait()
                # The old list's read may already have been cancelled.
                with suppress(PlaywrightError):
                    await route.fulfill(response=response)

            await page.route("**/api/sessions/message/transcript?**", pause_title)
            try:
                await page.goto(f"{base}/cayu/sessions")
                await asyncio.wait_for(reached.wait(), 5)
                await page.get_by_label("Include forks", exact=True).check()
                async with asyncio.timeout(5):
                    while not any(row.get("include_forks") == "true" for row in requests):
                        await asyncio.sleep(0.01)
                # Let the second list response join the still-pending shared title query.
                await page.wait_for_timeout(100)
                await page.evaluate("window.__releaseTitleAbort()")
                release.set()
                await expect(
                    page.get_by_text("message", exact=True)
                    .locator("xpath=ancestor::tr")
                    .get_by_role("link", name="Inspect the invoice", exact=True)
                ).to_be_visible()
                await expect(
                    page.get_by_text("Message title unavailable", exact=True)
                ).to_have_count(0)
            finally:
                release.set()
                await page.evaluate("window.__releaseTitleAbort()")

    asyncio.run(run())
