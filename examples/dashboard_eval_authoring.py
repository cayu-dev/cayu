"""Focused browser contract for the full-page suite and case editor."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from playwright.async_api import async_playwright, expect

from _live_checks import is_superseded_browser_read_abort
from cayu import AgentSpec, CayuApp, RunRequest
from cayu.evals import CorpusTarget, EvaluationEvidencePolicySpec, ModelJudgeTarget
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.read_views import EvalSuiteViewIdentity
from cayu.server import (
    DashboardConfig,
    EvalsConfig,
    EvaluationPromotionConfig,
    OpenAccess,
    ServerConfig,
    create_server,
)
from cayu.storage.evals_sqlite import SQLiteEvalStore


async def exercise_eval_authoring(flow: str = "all") -> None:
    import dashboard_behavior_live as fixture

    app, provider, _sessions, _tasks, directory = await fixture._seed_app()
    price_book = fixture._dashboard_price_book()
    judge_app = CayuApp(enable_logging=False)
    judge_app.register_provider(provider, default=True)
    judge_app.register_agent(AgentSpec(name="dashboard-contract-judge", model=fixture.MODEL_NAME))
    judge = ModelJudgeTarget(
        key="dashboard-quality-judge",
        label="Dashboard quality judge",
        app=judge_app,
        agent_name="dashboard-contract-judge",
        max_input_tokens=3_072,
        max_output_tokens=1_024,
        max_total_tokens=4_096,
        max_estimated_cost="0.01",
        price_book=price_book,
        allow_same_model=True,
    )
    store = SQLiteEvalStore(Path(directory.name) / "authoring-evals.sqlite")
    policy = EvaluationEvidencePolicySpec.create(
        include_tool_arguments=True,
        include_tool_results=True,
        include_artifact_text=True,
    )
    assets = os.environ.get("CAYU_EVAL_AUTHORING_ASSETS")
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(
                app,
                config=ServerConfig(
                    access=OpenAccess(trusted_local_development=True),
                    dashboard=DashboardConfig(directory=Path(assets) if assets else None),
                    evaluation_promotion=EvaluationPromotionConfig(
                        target_key="dashboard.regressions",
                        source_agent_name=fixture.AGENT_NAME,
                        application_release_id="dashboard-authoring-contract",
                        evidence_policy=policy,
                    ),
                    evals=EvalsConfig(
                        target=CorpusTarget(
                            key="dashboard.regressions",
                            app=app,
                            request_base=RunRequest(agent_name=fixture.AGENT_NAME, messages=[]),
                            application_release_id="dashboard-authoring-contract",
                            price_book=price_book,
                            model_judges=(judge,),
                            evidence_policy=policy,
                        ),
                        store=store,
                        # The scripted provider and read-only search tool have no external
                        # effects; each trial receives a fresh Cayu session/workspace.
                        execution_profile_policy=EvalExecutionProfilePolicyV1(
                            reset_strategy="application_managed",
                            isolation_revision="sha256:" + "a" * 64,
                            max_trials=3,
                            max_concurrency=2,
                        ),
                        poll_interval_seconds=0.02,
                    ),
                ),
            ),
            log_level="warning",
            lifespan="on",
        )
    )
    listener = fixture._loopback_listener()
    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await fixture._wait_for_server(server, task)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            expected_aborts: list[str] = []
            expected_console: list[str] = []
            expected_http: list[tuple[str, int]] = []

            def console(message):
                if message.type != "error":
                    return
                url = message.location.get("url", "")
                if (
                    re.search(r"/api/sessions/[^/]+/messages(?:\?|$)", url)
                    and "403" in message.text
                ):
                    return  # This fixture deliberately has no session message access policy.
                if any(
                    url == expected_url and str(status) in message.text
                    for expected_url, status in expected_http
                ):
                    return
                if url in expected_console and "net::ERR_CONNECTION_RESET" in message.text:
                    expected_console.remove(url)
                else:
                    errors.append(message.text)

            def failed(request):
                if is_superseded_browser_read_abort(
                    method=request.method,
                    path=urlsplit(request.url).path,
                    failure=request.failure,
                    mutation_id=request.headers.get("cayu-mutation-id"),
                ):
                    return
                if request.url in expected_aborts and request.failure in (
                    "net::ERR_CONNECTION_RESET",
                    "net::ERR_ABORTED",
                ):
                    expected_aborts.remove(request.url)
                else:
                    errors.append(f"{request.method} {request.url}: {request.failure}")

            page.on("console", console)
            page.on("requestfailed", failed)
            page.on(
                "response",
                lambda response: (
                    errors.append(f"HTTP {response.status}: {response.url}")
                    if response.status >= 400
                    and (response.url, response.status) not in expected_http
                    and not (
                        response.status == 403
                        and re.search(r"/api/sessions/[^/]+/messages(?:\?|$)", response.url)
                    )
                    else None
                ),
            )
            try:
                provider_requests = len(provider.requests)
                if flow in {"all", "settings"}:
                    from dashboard_eval_settings import exercise_eval_settings

                    await exercise_eval_settings(
                        page, base, provider, expected_aborts, expected_console
                    )
                    provider_requests = len(provider.requests)
                if flow in {"all", "checks", "keyboard"}:
                    await _exercise_all_checks(page, base, hold_refresh=flow in {"all", "keyboard"})
                if flow in {"all", "scenario"}:
                    await _exercise_scenario(page, base)
                if flow in {"all", "parent"}:
                    await _exercise_parent_replay(
                        page, base, fixture.PROMOTION_SESSION_ID, expected_aborts, expected_console
                    )
                assert len(provider.requests) == provider_requests, (
                    "authoring must not start candidate or judge work"
                )
                if flow in {"all", "calibration"}:
                    await _exercise_calibration(
                        page, base, provider, expected_aborts, expected_console, expected_http
                    )
                for rejection in (404, 409, 422):
                    if flow in {
                        "all",
                        "recovery-abandon-sequence",
                        f"calibration-abandon-{rejection}",
                    }:
                        await _exercise_calibration(
                            page,
                            base,
                            provider,
                            expected_aborts,
                            expected_console,
                            expected_http,
                            abandon_status=rejection,
                        )
                    if flow in {
                        "all",
                        "recovery-abandon-sequence",
                        f"scenario-abandon-{rejection}",
                    }:
                        from dashboard_eval_settings import _scenarios

                        await _scenarios(
                            page,
                            base,
                            provider,
                            expected_aborts,
                            expected_console,
                            expected_http=expected_http,
                            abandon_status=rejection,
                        )
                if flow in {"all", "navigation"}:
                    await _exercise_navigation_and_access(page, base)
                if flow in {"all", "late"}:
                    await _exercise_late_launch(page, base, provider, expected_aborts)
                if flow in {"all", "run"}:
                    await _exercise_reviewed_run(page, base, provider)
                if flow in {"all", "recovery-corpus"}:
                    from dashboard_eval_recovery import exercise_corpus_recovery

                    await exercise_corpus_recovery(
                        page, base, store, app, policy, expected_aborts, expected_console
                    )
                if flow in {"all", "recovery-captured"}:
                    from dashboard_eval_recovery import exercise_captured_recovery

                    await exercise_captured_recovery(page, base, expected_aborts, expected_console)
                for lifetime in (
                    "nested-status",
                    "direct-status",
                    "nested-error",
                    "direct-error",
                    "direct-authorization",
                    "nested-abandon",
                ):
                    if flow in {"all", f"recovery-captured-{lifetime}"}:
                        from dashboard_eval_recovery import exercise_captured_lifetime

                        await exercise_captured_lifetime(
                            page,
                            base,
                            _sessions,
                            expected_aborts,
                            expected_console,
                            expected_http,
                            lifetime,
                        )
                if flow in {"all", "recovery-scenario"}:
                    from dashboard_eval_settings import _scenarios

                    await _scenarios(
                        page,
                        base,
                        provider,
                        expected_aborts,
                        expected_console,
                        expected_http=expected_http,
                    )
                if flow in {"all", "recovery-authored"}:
                    from dashboard_eval_recovery import exercise_authored_partial_recovery

                    await exercise_authored_partial_recovery(page, base, store, app, expected_http)
                assert not expected_aborts and not expected_console, (
                    expected_aborts,
                    expected_console,
                )
                assert not errors, errors
            except BaseException:
                print("AUTHORING DIAGNOSTICS", errors, flush=True)
                print((await page.locator("body").inner_text())[:8000], flush=True)
                raise
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        finally:
            await store.close()
            directory.cleanup()


async def _exercise_all_checks(page, base: str, *, hold_refresh: bool = False) -> None:
    await page.goto(
        f"{base}/cayu/evals/suites/new?target=dashboard.regressions", wait_until="networkidle"
    )
    editor = page.get_by_test_id("eval-suite-editor")
    await expect(editor).to_be_visible(timeout=5000)
    await editor.get_by_test_id("authored-suite-name").fill("Every portable check")
    await editor.get_by_text("Suite details", exact=True).click()
    await editor.get_by_test_id("authored-suite-id").fill("browser-all-checks")
    await editor.get_by_role("tab", name="Checks (1)", exact=True).click()
    kinds = [
        "child_status",
        "final_output_equals",
        "final_output_contains",
        "tool_called",
        "tool_arguments_contain",
        "tool_result_contains",
        "tools_called_in_order",
        "process_event",
        "process_events_in_order",
        "workspace_file",
        "artifact",
        "memory_attribution",
        "max_tool_calls",
        "max_model_steps",
        "usage_recorded",
        "max_total_tokens",
        "max_estimated_cost",
        "model_judge",
        "structured_model_judge",
    ]
    for kind in kinds:
        await editor.get_by_label("Check type", exact=True).select_option(kind)
        await editor.get_by_role("button", name="Add check", exact=True).click()
        row = editor.get_by_test_id("authored-check").last
        await row.locator("summary").first.click()
        if kind == "final_output_equals":
            await row.get_by_label("Comparison", exact=True).select_option("json")
            await row.get_by_label("Expected output text", exact=True).fill(
                '{"enabled":true,"amount":1}'
            )
        if kind == "model_judge":
            await row.get_by_label("Evaluator key", exact=True).fill("dashboard-quality-judge")
        await row.locator("summary").first.click()
    await expect(editor.get_by_test_id("authored-check")).to_have_count(20)
    previews = []

    def observe_preview(request):
        if request.url.endswith("/suites/preview") and request.method == "POST":
            previews.append(request)

    page.on("request", observe_preview)
    equality_row = editor.get_by_test_id("authored-check").nth(2)
    await equality_row.locator("summary").first.click()
    await equality_row.get_by_label("Expected output text", exact=True).fill("{")
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-error")).to_contain_text("valid JSON")
    assert not previews, "invalid typed input must not be normalized into a server request"
    await equality_row.get_by_label("Expected output text", exact=True).fill(
        '{"enabled":true,"amount":1}'
    )
    await equality_row.locator("summary").first.click()
    judge_row = editor.get_by_test_id("authored-check").nth(18)
    await judge_row.locator("summary").first.click()
    await judge_row.get_by_label("Minimum passing score", exact=True).fill("")
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-error")).to_contain_text("threshold")
    assert not previews
    await judge_row.get_by_label("Minimum passing score", exact=True).fill("0.75")
    await judge_row.locator("summary").first.click()
    page.remove_listener("request", observe_preview)
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled(timeout=15000)

    async def delay_saved_catalog(route):
        await asyncio.sleep(2)
        await route.continue_()

    if hold_refresh:
        await page.route("**/api/evals/targets", delay_saved_catalog)
    async with page.expect_response(
        lambda response: (
            response.url.endswith("/api/evals/suites") and response.request.method == "POST"
        )
    ) as response_info:
        await editor.get_by_test_id("authored-suite-save").click()
    response = await response_info.value
    assert response.status == 201, await response.text()
    body = response.request.post_data_json
    assert body["parent_suite_revision"] is None
    saved = (await response.json())["suite"]
    assert len(saved["cases"][0]["assertions"]) == 20
    kinds_saved = {check["kind"] for check in saved["cases"][0]["assertions"]}
    assert kinds_saved == {"root_status", *kinds}
    equality = next(
        check for check in saved["cases"][0]["assertions"] if check["kind"] == "final_output_equals"
    )
    assert (
        equality["comparison"] == "json" and equality["expected"] == '{"enabled":true,"amount":1}'
    )
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    try:
        await _exercise_layout(page, editor)
    finally:
        if hold_refresh:
            await page.unroute("**/api/evals/targets", delay_saved_catalog)
    await page.goto(
        f"{base}/cayu/evals/suites/{_suite_identity(saved).id}/edit", wait_until="networkidle"
    )
    await expect(editor.get_by_test_id("authored-suite-name")).to_have_value("Every portable check")
    await editor.get_by_role("tab", name="Checks (20)", exact=True).click()
    await expect(editor.get_by_test_id("authored-check")).to_have_count(20)
    await editor.get_by_test_id("authored-suite-name").fill("Every portable check renamed")
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    async with page.expect_response(
        lambda response: (
            response.url.endswith("/api/evals/suites") and response.request.method == "POST"
        )
    ) as child_info:
        await editor.get_by_test_id("authored-suite-save").click()
    child_response = await child_info.value
    assert child_response.status == 201, await child_response.text()
    assert child_response.request.post_data_json["parent_suite_revision"] == saved["revision"]
    assert (await child_response.json())["suite"]["cases"] == saved["cases"]
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    print("AUTHORING ALL TWENTY CHECKS PASS", flush=True)


async def _exercise_layout(page, editor, check_count: int = 20):
    viewport = page.viewport_size
    dark = await page.evaluate("document.documentElement.classList.contains('dark')")
    try:
        for width, height in [(1440, 900), (1100, 800), (390, 844)]:
            await page.set_viewport_size({"width": width, "height": height})
            await page.emulate_media(color_scheme="dark" if width == 390 else "light")
            await page.evaluate(
                "dark => document.documentElement.classList.toggle('dark', dark)", width == 390
            )
            await expect(editor.get_by_test_id("authored-suite-save")).to_be_in_viewport()
            geometry = await page.evaluate(
                """() => ({
                    width: window.innerWidth,
                    content: document.documentElement.scrollWidth,
                    editor: document.querySelector('[data-testid="eval-suite-editor"]')
                        .getBoundingClientRect().toJSON(),
                    outside: Array.from(document.querySelectorAll('input, textarea, select, button'))
                        .filter(e => {
                            const box = e.getBoundingClientRect();
                            return box.width > 0 && (box.left < -1 || box.right > innerWidth + 1);
                        }).map(e => [e.tagName, e.getAttribute('aria-label'), e.id]),
                })"""
            )
            assert geometry["content"] <= geometry["width"] + 1, geometry
            assert geometry["editor"]["right"] <= geometry["width"] + 1, geometry
            assert not geometry["outside"], geometry
            checks = editor.get_by_role("tab", name=f"Checks ({check_count})", exact=True)
            # Save publication can finish before its awaited catalog refresh. An
            # inherited disabled fieldset makes focus() silently do nothing.
            await expect(checks).to_be_enabled()
            await checks.focus()
            await expect(checks).to_be_focused()
            await checks.press("ArrowLeft")
            await expect(editor.get_by_role("tab", name="Input", exact=True)).to_be_focused()
            await page.keyboard.press("Enter")
            await expect(editor.get_by_role("tabpanel", name="Input", exact=True)).to_contain_text(
                "Simple input"
            )
            await checks.click()
    finally:
        if viewport:
            await page.set_viewport_size(viewport)
        await page.emulate_media(color_scheme="light")
        await page.evaluate("dark => document.documentElement.classList.toggle('dark', dark)", dark)
    print("AUTHORING RESPONSIVE / KEYBOARD PASS", flush=True)


async def _exercise_scenario(page, base):
    await page.goto(
        f"{base}/cayu/evals/suites/new?target=dashboard.regressions", wait_until="networkidle"
    )
    editor = page.get_by_test_id("eval-suite-editor")
    await editor.get_by_test_id("authored-suite-name").fill("Embedded scenario")
    await editor.get_by_text("Suite details", exact=True).click()
    await editor.get_by_test_id("authored-suite-id").fill("browser-embedded-scenario")
    await editor.get_by_role("button", name="Multi-stage scenario", exact=True).click()
    scenario = editor.get_by_test_id("scenario-authoring")
    scenario_input = scenario.get_by_test_id("scenario-event-0").locator("textarea").first
    await scenario_input.fill("A retained scenario draft across case tabs")
    await expect(editor.get_by_test_id("authored-suite-preview")).to_be_disabled()
    await expect(editor.get_by_role("button", name="Add case", exact=True)).to_be_disabled()
    await editor.get_by_role("tab", name="Checks (1)", exact=True).click()
    await editor.get_by_role("tab", name="Input · unsaved", exact=True).click()
    await expect(scenario_input).to_have_value("A retained scenario draft across case tabs")
    await scenario.get_by_role("button", name="Check readiness", exact=True).click()
    await expect(scenario.get_by_role("button", name="Save scenario", exact=True)).to_be_enabled()
    started = asyncio.Event()
    release = asyncio.Event()

    async def pause_save(route, request):
        if request.method != "POST":
            await route.continue_()
            return
        started.set()
        await release.wait()
        await route.continue_()

    save_url = f"{base}/api/evals/scenarios"
    await page.route(save_url, pause_save)
    try:
        await scenario.get_by_role("button", name="Save scenario", exact=True).click()
        await asyncio.wait_for(started.wait(), 5)
        await expect(editor.get_by_test_id("authored-suite-name")).to_be_disabled()
        await expect(editor.get_by_test_id("authored-suite-preview")).to_be_disabled()
        await expect(editor.get_by_test_id("authored-suite-save")).to_be_disabled()
        release.set()
        await expect(
            editor.get_by_text(re.compile(r"Saved scenario .+ for case-1"))
        ).to_be_visible()
    finally:
        release.set()
        await page.unroute(save_url, pause_save)
    await expect(editor.get_by_test_id("authored-suite-scenario-lock")).not_to_be_visible()
    await expect(scenario_input).to_have_value("A retained scenario draft across case tabs")
    await scenario_input.fill("Changed but discarded")
    await expect(editor.get_by_test_id("authored-suite-preview")).to_be_disabled()
    await editor.get_by_role("button", name="Discard scenario edits", exact=True).click()
    await expect(scenario_input).to_have_value("A retained scenario draft across case tabs")
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    async with page.expect_response(
        lambda response: (
            response.url.endswith("/api/evals/suites") and response.request.method == "POST"
        )
    ) as saved_info:
        await editor.get_by_test_id("authored-suite-save").click()
    response = await saved_info.value
    assert response.status == 201, await response.text()
    suite = (await response.json())["suite"]
    assert suite["cases"][0]["stimulus"]["kind"] == "scenario"
    scenario_revision = suite["cases"][0]["stimulus"]["scenario_revision"]
    response = await page.request.get(f"{base}/api/evals/scenarios/{scenario_revision}")
    assert response.status == 200, await response.text()
    assert "A retained scenario draft across case tabs" in await response.text()
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    print("AUTHORING EMBEDDED SCENARIO OWNERSHIP PASS", flush=True)


async def _exercise_calibration(
    page, base, provider, expected_aborts, expected_console, expected_http, *, abandon_status=None
):
    before = len(provider.requests)
    judges_before = len(provider.judge_requests)
    await page.goto(
        f"{base}/cayu/evals/suites/new?target=dashboard.regressions", wait_until="networkidle"
    )
    editor = page.get_by_test_id("eval-suite-editor")
    await editor.get_by_test_id("authored-suite-name").fill("Calibration and memory checks")
    await editor.get_by_text("Suite details", exact=True).click()
    await editor.get_by_test_id("authored-suite-id").fill("browser-calibration-memory")
    await editor.get_by_role("button", name="Add case", exact=True).click()
    case_buttons = editor.get_by_test_id("authored-suite-cases").get_by_role("button")
    await case_buttons.first.click()
    await editor.get_by_role("tab", name="Checks (1)", exact=True).click()
    await editor.get_by_label("Check type", exact=True).select_option("structured_model_judge")
    await editor.get_by_role("button", name="Add check", exact=True).click()
    row = editor.get_by_test_id("authored-check").last
    await row.locator("summary").first.click()
    # Invalid/transient duplicate assertion IDs must not alias DOM labels or another draft.
    first_judge = editor.get_by_test_id("authored-check").nth(1)
    first_id = await first_judge.get_by_label("Assertion ID", exact=True).input_value()
    await editor.get_by_role("button", name="Add check", exact=True).click()
    second_judge = editor.get_by_test_id("authored-check").last
    await second_judge.locator("summary").first.click()
    await second_judge.get_by_label("Assertion ID", exact=True).fill(first_id)
    first_rubric = first_judge.get_by_label("Rubric ID", exact=True)
    second_rubric = second_judge.get_by_label("Rubric ID", exact=True)
    assert await first_rubric.get_attribute("id") != await second_rubric.get_attribute("id"), (
        "judge fields must use instance identity, not editable assertion IDs"
    )
    await second_judge.get_by_text("Rubric ID", exact=True).click()
    await expect(second_rubric).to_be_focused()
    await second_judge.get_by_role("button", name="Remove AI judge", exact=True).click()
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    await editor.get_by_test_id("authored-suite-save").click()
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    calibration = row.get_by_test_id("judge-calibration")
    await calibration.get_by_label("Evidence source ID", exact=True).fill("reviewed-fixed-output")
    output = calibration.get_by_label("Known candidate output", exact=True)
    await output.fill("dashboard authored evaluation output")
    await calibration.get_by_label("Repeated judge calls", exact=True).fill("2")
    await editor.get_by_role("tab", name="Input", exact=True).click()
    await editor.get_by_role("tab", name="Checks (2)", exact=True).click()
    await expect(output).to_have_value("dashboard authored evaluation output")
    # Calibration owns private draft evidence; a parent edit must not silently reset it.
    await expect(editor.get_by_test_id("authored-suite-name")).to_be_disabled()
    await expect(editor.get_by_role("button", name="Add case", exact=True)).to_be_disabled()
    await expect(case_buttons.nth(1)).to_be_disabled()
    await calibration.get_by_label(re.compile(r"^Human score for ")).first.fill("0.75")
    await calibration.get_by_role("button", name="Check calibration", exact=True).click()
    await expect(calibration.get_by_text("Calibration is ready", exact=True)).to_be_visible()
    assert len(provider.requests) == before
    lost = {}
    dispatched, release = asyncio.Event(), asyncio.Event()
    url = f"{base}/api/evals/judge-calibrations"

    async def lose_acknowledgement(route, request):
        if request.method != "POST" or lost:
            await route.continue_()
            return
        response = await route.fetch()
        assert response.status == 201, await response.text()
        lost["request"] = request.post_data_json
        lost["response"] = await response.json()
        dispatched.set()
        await release.wait()
        expected_aborts.append(url)
        expected_console.append(url)
        await route.abort("connectionreset")

    await page.route(url, lose_acknowledgement)
    try:
        await calibration.get_by_role("button", name="Run calibration", exact=True).click()
        await asyncio.wait_for(dispatched.wait(), 15)
        abandon = calibration.get_by_role("button", name="Abandon calibration recovery", exact=True)
        if abandon_status is not None and await abandon.count():
            await expect(abandon).to_be_disabled()
        await expect(case_buttons.nth(1)).to_be_disabled()
        await expect(editor.get_by_test_id("authored-suite-name")).to_be_disabled()
        # Catalog refresh must not discard the private paid calibration owner.
        from dashboard_eval_settings import _refresh_targets

        async def removed_target(route):
            response = await route.fetch()
            body = await response.json()
            body.update(items=[], default_target_key=None)
            await route.fulfill(response=response, json=body)

        await page.route("**/api/evals/targets", removed_target)
        await _refresh_targets(page)
        await expect(
            page.get_by_text("This specialist is not available.", exact=True)
        ).to_be_visible()
        await expect(calibration).to_be_visible()
        await expect(output).to_have_value("dashboard authored evaluation output")
        assert "dashboard authored evaluation output" not in await page.evaluate(
            "JSON.stringify([sessionStorage, localStorage])"
        )
        await page.get_by_role("link", name="Home", exact=True).first.click()
        dialog = page.get_by_role("dialog")
        await expect(dialog.get_by_role("button", name="Leave editor", exact=True)).to_be_disabled()
        await dialog.get_by_role("button", name="Keep editing", exact=True).click()
        release.set()
        await expect(calibration.get_by_role("alert")).to_be_visible()
        await expect(
            calibration.get_by_role("button", name="Discard calibration", exact=True)
        ).to_be_disabled()
        await expect(case_buttons.nth(1)).to_be_disabled()
        assert len(provider.requests) == before + 2
        assert len(provider.judge_requests) == judges_before + 2

        # A temporary server-side absence during recovery cannot prove the first
        # acknowledged-by-server operation never ran.
        rejected = []

        async def temporarily_unavailable(route):
            if (
                abandon_status is not None
                and route.request.post_data_json["run_id"] != lost["request"]["run_id"]
            ):
                await route.fallback()
                return
            rejected.append(route.request.post_data_json)
            await route.fulfill(
                status=abandon_status or 404,
                json={"detail": "Calibration target temporarily unavailable."},
            )

        expected_http.append((url, abandon_status or 404))
        await page.route(url, temporarily_unavailable)
        await calibration.get_by_role("button", name="Run calibration", exact=True).click()
        await expect(calibration.get_by_role("alert")).to_contain_text("temporarily unavailable")
        await expect(
            calibration.get_by_role("button", name="Discard calibration", exact=True)
        ).to_be_disabled()
        if abandon_status is not None:
            from dashboard_eval_recovery import confirm_recovery_abandonment

            await calibration.get_by_role("button", name="Run calibration", exact=True).click()
            await expect(calibration.get_by_role("alert")).to_contain_text(
                "temporarily unavailable"
            )
            assert rejected == [lost["request"], lost["request"]]
            assert len(provider.judge_requests) == judges_before + 2
            await confirm_recovery_abandonment(page, abandon, accept=False)
            await expect(output).to_be_disabled()
            await confirm_recovery_abandonment(page, abandon, accept=True)
            await expect(output).to_be_enabled()
            await expect(output).to_have_value("dashboard authored evaluation output")
            await expect(
                calibration.get_by_role("button", name="Run calibration", exact=True)
            ).to_be_disabled()
            await expect(
                calibration.get_by_role("button", name="Discard calibration", exact=True)
            ).to_be_enabled()
            await page.get_by_role("link", name="Home", exact=True).first.click()
            await expect(
                page.get_by_role("button", name="Leave editor", exact=True)
            ).to_be_enabled()
            await page.get_by_role("button", name="Keep editing", exact=True).click()
            await page.unroute("**/api/evals/targets", removed_target)
            await _refresh_targets(page)
            await calibration.get_by_role("button", name="Check calibration", exact=True).click()
            await expect(
                calibration.get_by_role("button", name="Run calibration", exact=True)
            ).to_be_enabled()
            async with page.expect_response(
                lambda response: response.url == url and response.request.method == "POST"
            ) as new_info:
                await calibration.get_by_role("button", name="Run calibration", exact=True).click()
            new_response = await new_info.value
            assert new_response.status == 201, await new_response.text()
            fresh_request = new_response.request.post_data_json
            assert fresh_request["run_id"] != lost["request"]["run_id"]
            assert {k: v for k, v in fresh_request.items() if k != "run_id"} == {
                k: v for k, v in lost["request"].items() if k != "run_id"
            }
            await expect(
                calibration.get_by_text(re.compile(r"Calibration complete"))
            ).to_be_visible()
            # Abandoning local observation does not delete the accepted old report.
            old = await page.request.post(url, data=lost["request"])
            assert old.status == 201 and await old.json() == lost["response"]
            assert len(provider.judge_requests) == judges_before + 4
            await page.unroute(url, temporarily_unavailable)
            print(f"CALIBRATION PERSISTENT {abandon_status} ABANDON / NEW INTENT PASS", flush=True)
            return
        await page.unroute(url, temporarily_unavailable)
        async with page.expect_response(
            lambda response: response.url == url and response.request.method == "POST"
        ) as info:
            await calibration.get_by_role("button", name="Run calibration", exact=True).click()
        replay = await info.value
        assert replay.status == 201, await replay.text()
        assert replay.request.post_data_json == lost["request"]
        assert await replay.json() == lost["response"]
        await expect(calibration.get_by_text(re.compile(r"Calibration complete"))).to_be_visible()
        await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
        await expect(editor.get_by_test_id("authored-suite-name")).to_be_enabled()
        await expect(case_buttons.nth(1)).to_be_enabled()
        await editor.get_by_test_id("authored-suite-save").click()
        await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
        await expect(calibration.get_by_text(re.compile(r"Calibration complete"))).to_be_visible()
        assert len(provider.requests) == before + 2
        await page.unroute("**/api/evals/targets", removed_target)
        await _refresh_targets(page)
        await expect(output).to_have_value("dashboard authored evaluation output")
    finally:
        release.set()
        await page.unroute(url, lose_acknowledgement)
    await output.fill("Changed evidence invalidates the paid-run preview")
    await expect(
        calibration.get_by_role("button", name="Run calibration", exact=True)
    ).to_be_disabled()
    await calibration.get_by_role("button", name="Discard calibration", exact=True).click()
    await expect(editor.get_by_test_id("authored-suite-name")).to_be_enabled()
    await expect(case_buttons.nth(1)).to_be_enabled()
    await case_buttons.filter(has_text="Case 2").click()
    await case_buttons.filter(has_text="Case 1").click()
    await editor.get_by_role("tab", name="Checks (2)", exact=True).click()
    await row.locator("summary").first.click()
    await expect(calibration.get_by_label("Known candidate output", exact=True)).to_have_value(
        "Paste a known candidate answer to judge."
    )
    await row.locator("summary").first.click()
    await editor.get_by_role("button", name="Require memory exposure", exact=True).click()
    await editor.get_by_role("button", name="Add reference-backed judge", exact=True).click()
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-error")).to_be_visible()
    memory = editor.get_by_test_id("authored-check").last
    await memory.locator("summary").first.click()
    await memory.get_by_label("Criterion 1 name", exact=True).fill("LongCriterion" * 9)
    await memory.get_by_label("Expected facts (one per line)", exact=True).fill(
        "The retained account currency is USD."
    )
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    async with page.expect_response(
        lambda response: (
            response.url.endswith("/api/evals/suites") and response.request.method == "POST"
        )
    ) as saved_info:
        await editor.get_by_test_id("authored-suite-save").click()
    saved = await (await saved_info.value).json()
    checks = next(case for case in saved["suite"]["cases"] if case["id"] == "case-1")["assertions"]
    assert sum(check["kind"] == "memory_attribution" for check in checks) == 1
    memory_check = next(
        check for check in checks if check.get("rubric", {}).get("id") == "memory-use"
    )
    assert memory_check["reference"]["expected_facts"] == ["The retained account currency is USD."]
    assert len(provider.requests) == before + 2
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    await _exercise_layout(page, editor, check_count=4)
    print("AUTHORING CALIBRATION RETRY / MEMORY PRESETS PASS", flush=True)


async def _exercise_reviewed_run(page, base, provider):
    before = len(provider.requests)
    await page.goto(
        f"{base}/cayu/evals/suites/new?target=dashboard.regressions", wait_until="networkidle"
    )
    editor = page.get_by_test_id("eval-suite-editor")
    await editor.get_by_test_id("authored-suite-name").fill("Reviewed run")
    await editor.get_by_text("Suite details", exact=True).click()
    await editor.get_by_test_id("authored-suite-id").fill("browser-reviewed-run")
    await editor.locator("summary").filter(has_text="Suite settings").click()
    await editor.get_by_label("Trials per case", exact=True).fill("3")
    await editor.get_by_label("Required passes", exact=True).fill("2")
    await editor.get_by_label("Maximum concurrency", exact=True).fill("2")
    await editor.get_by_label("Case input message 1", exact=True).fill(
        "Exercise one fresh control plane-authored evaluation"
    )
    await editor.get_by_role("button", name="Simple input", exact=True).click()
    await expect(editor.get_by_label("Case input message 1", exact=True)).to_have_value(
        "Exercise one fresh control plane-authored evaluation"
    )
    await editor.get_by_role("button", name="Duplicate", exact=True).click()
    case_name = await editor.get_by_test_id("authored-case-name").input_value()
    await editor.get_by_role(
        "checkbox", name=f"Select {case_name} for launch", exact=True
    ).uncheck()
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    async with page.expect_response(
        lambda response: (
            response.url.endswith("/api/evals/suites") and response.request.method == "POST"
        )
    ) as saved_info:
        await editor.get_by_test_id("authored-suite-save").click()
    saved = (await (await saved_info.value).json())["suite"]
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    assert len(provider.requests) == before
    await page.goto(
        f"{base}/cayu/evals/suites/{_suite_identity(saved).id}/edit", wait_until="networkidle"
    )
    await expect(editor.get_by_test_id("authored-suite-name")).to_have_value("Reviewed run")
    await editor.get_by_role(
        "checkbox", name=f"Select {case_name} for launch", exact=True
    ).uncheck()
    await editor.get_by_role("button", name="Run saved revision", exact=True).click()
    await editor.get_by_test_id("authored-suite-maximum-cost").fill("0.1")
    async with page.expect_response(
        lambda response: response.url.endswith("/runs/preview")
    ) as info:
        await editor.get_by_test_id("authored-suite-run-preview").click()
    preview_response = await info.value
    assert preview_response.status == 200, await preview_response.text()
    preview = await preview_response.json()
    assert preview["ready"] is True
    assert len(preview["selection"]["cases"]) == 1
    assert preview["exposure"]["candidate_trials"] == 3
    await expect(editor.get_by_test_id("authored-suite-launch")).to_be_enabled()
    await editor.get_by_test_id("authored-suite-maximum-cost").fill("0.2")
    await expect(editor.get_by_test_id("authored-suite-launch")).to_be_disabled()
    async with page.expect_response(
        lambda response: response.url.endswith("/runs/preview")
    ) as info:
        await editor.get_by_test_id("authored-suite-run-preview").click()
    reviewed = await (await info.value).json()
    assert reviewed["exposure"]["revision"] != preview["exposure"]["revision"]
    assert len(provider.requests) == before, "review must not launch candidate or judge work"
    await page.evaluate(
        "window.__savedRandomUUID = crypto.randomUUID; Object.defineProperty(crypto, 'randomUUID', {value: undefined, configurable: true})"
    )
    try:
        await editor.get_by_test_id("authored-suite-launch").click()
        await expect(editor.get_by_test_id("authored-suite-error")).to_contain_text(
            "Secure browser randomness is unavailable"
        )
        assert len(provider.requests) == before
    finally:
        await page.evaluate(
            "Object.defineProperty(crypto, 'randomUUID', {value: window.__savedRandomUUID, configurable: true}); delete window.__savedRandomUUID"
        )
    async with page.expect_response(
        lambda response: response.url.endswith("/runs") and response.request.method == "POST"
    ) as launched_info:
        await editor.get_by_test_id("authored-suite-launch").click()
    launched_response = await launched_info.value
    assert launched_response.status == 202, await launched_response.text()
    request = launched_response.request.post_data_json
    assert request["expected_exposure_revision"] == reviewed["exposure"]["revision"]
    assert len(request["case_ids"]) == 1
    assert request["cost_budget"] == {"max_estimated_cost": "0.2", "currency": "USD"}
    launched = await launched_response.json()
    assert len(launched["runs"]) == 1
    run_id = launched["runs"][0]["run"]["spec"]["run_id"]
    for _ in range(200):
        response = await page.request.get(f"{base}/api/evals/runs/{run_id}")
        assert response.status == 200, await response.text()
        run = await response.json()
        if run["status"] in {"completed", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.05)
    assert run["status"] == "completed", run
    assert len(provider.requests) > before
    print("AUTHORING SEPARATE REVIEWED RUN PASS", flush=True)


def _suite_identity(document):
    return EvalSuiteViewIdentity(
        source="authored", target_key=document["target_key"], revision=document["revision"]
    )


async def _source(page, base, document, case_id):
    response = await page.request.get(
        f"{base}/api/evals/suite-views/{_suite_identity(document).id}/cases/{case_id}/source"
    )
    assert response.status == 200, await response.text()
    return await response.json()


async def _exercise_parent_replay(page, base, session_id, expected_aborts, expected_console):
    request = {
        "mode": "case",
        "destination": {
            "kind": "new",
            "suite_id": "browser-captured-parent",
            "name": "Captured parent",
        },
        "case_id": "captured",
        "case_name": "Captured case",
    }
    addition_url = f"{base}/api/evals/sessions/{session_id}/additions"
    response = await page.request.post(f"{addition_url}/preview", data=request)
    assert response.status == 200, await response.text()
    plan = await response.json()
    response = await page.request.post(
        addition_url,
        headers={"Idempotency-Key": "browser-captured-parent"},
        data={
            **request,
            "expected_plan_revision": plan["revision"],
            "expected_source_revision": plan["source_revision"],
        },
    )
    assert response.status == 201, await response.text()
    response = await page.request.get(f"{base}/api/evals/suites/{plan['destination']['revision']}")
    assert response.status == 200, await response.text()
    parent = await response.json()
    assert await _source(page, base, parent, "captured") == {
        "state": "available",
        "session_id": session_id,
    }
    await page.goto(
        f"{base}/cayu/evals/suites/{_suite_identity(parent).id}/edit?case=captured",
        wait_until="networkidle",
    )
    editor = page.get_by_test_id("eval-suite-editor")
    await expect(editor.get_by_test_id("authored-case-name")).to_have_value("Captured case")
    await editor.get_by_test_id("authored-suite-name").fill("Captured child")
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    lost = {}
    save_url = f"{base}/api/evals/suites"

    async def lose_acknowledgement(route, request):
        if request.method != "POST" or lost:
            await route.continue_()
            return
        response = await route.fetch()
        assert response.status == 201, await response.text()
        lost["request"] = request.post_data_json
        lost["response"] = await response.json()
        expected_aborts.append(save_url)
        expected_console.append(save_url)
        await route.abort("connectionreset")

    await page.route(save_url, lose_acknowledgement)
    try:
        await editor.get_by_test_id("authored-suite-save").click()
        await expect(
            editor.get_by_role("button", name="Retry exact save", exact=True)
        ).to_be_visible()
        await expect(editor.get_by_test_id("authored-suite-name")).to_be_disabled()
        assert lost["request"]["parent_suite_revision"] == parent["revision"]
        async with page.expect_response(
            lambda response: response.url == save_url and response.request.method == "POST"
        ) as replay_info:
            await editor.get_by_role("button", name="Retry exact save", exact=True).click()
        replay = await replay_info.value
        assert replay.status == 201, await replay.text()
        assert replay.request.post_data_json == lost["request"]
        assert await replay.json() == lost["response"]
        await expect(editor.get_by_test_id("authored-suite-name")).to_be_enabled()
    finally:
        await page.unroute(save_url, lose_acknowledgement)
    child = lost["response"]["suite"]
    assert await _source(page, base, child, "captured") == {
        "state": "available",
        "session_id": session_id,
    }
    assert await _source(page, base, parent, "captured") == {
        "state": "available",
        "session_id": session_id,
    }
    await editor.get_by_test_id("authored-case-name").fill("Edited captured case")
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    async with page.expect_response(
        lambda response: response.url == save_url and response.request.method == "POST"
    ) as changed_info:
        await editor.get_by_test_id("authored-suite-save").click()
    changed = await changed_info.value
    assert changed.status == 201, await changed.text()
    assert changed.request.post_data_json["parent_suite_revision"] == child["revision"]
    grandchild = (await changed.json())["suite"]
    assert await _source(page, base, grandchild, "captured") == {
        "state": "unavailable",
        "session_id": None,
    }
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    await page.goto(
        f"{base}/cayu/evals/suites/{_suite_identity(grandchild).id}/edit", wait_until="networkidle"
    )
    await expect(editor.get_by_test_id("authored-case-name")).to_have_value("Edited captured case")
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    async with page.expect_response(
        lambda response: response.url == save_url and response.request.method == "POST"
    ) as noop_info:
        await editor.get_by_test_id("authored-suite-save").click()
    noop = await noop_info.value
    assert noop.status == 201, await noop.text()
    assert noop.request.post_data_json["parent_suite_revision"] == grandchild["revision"]
    assert (await noop.json())["suite"] == grandchild
    print("AUTHORING EXACT PARENT/RETRY/SOURCE PASS", flush=True)


async def _exercise_navigation_and_access(page, base):
    await page.goto(f"{base}/cayu/evals", wait_until="networkidle")
    await page.get_by_role("button", name="New suite", exact=True).click()
    editor = page.get_by_test_id("eval-suite-editor")
    await expect(editor).to_be_visible()
    await editor.get_by_test_id("authored-suite-name").fill("Navigation-owned draft")
    await page.get_by_role("link", name="Home", exact=True).first.click()
    await expect(page.get_by_role("dialog", name="Leave this editor?")).to_be_visible()
    await page.get_by_role("button", name="Keep editing", exact=True).click()
    await expect(editor.get_by_test_id("authored-suite-name")).to_have_value(
        "Navigation-owned draft"
    )
    # Browser history uses the same blocker, not only the visible Back link.
    await page.evaluate("history.back()")
    await page.get_by_role("button", name="Keep editing", exact=True).click()
    await expect(editor.get_by_test_id("authored-suite-name")).to_have_value(
        "Navigation-owned draft"
    )
    await editor.get_by_text("Suite details", exact=True).click()
    await editor.get_by_test_id("authored-suite-id").fill("navigation-owned")
    await editor.get_by_test_id("authored-suite-preview").click()
    async with page.expect_response(
        lambda r: r.url.endswith("/evals/suites") and r.request.method == "POST"
    ) as saved_response:
        await editor.get_by_test_id("authored-suite-save").click()
    saved = (await (await saved_response.value).json())["suite"]
    await expect(editor.get_by_text(re.compile(r"^Saved immutable suite"))).to_be_visible()
    await page.get_by_role("link", name="Evals /", exact=True).click()
    await page.get_by_role("link", name="Navigation-owned draft", exact=True).click()
    await page.get_by_role("link", name="Edit Case 1", exact=True).click()
    await expect(editor.get_by_test_id("authored-case-name")).to_have_value("Case 1")
    await expect(page).to_have_url(re.compile(r"/edit\?case=case-1$"))
    await page.get_by_role("link", name="Back to suite", exact=True).click()
    await page.get_by_role("button", name="Run suite", exact=True).click()
    await expect(editor.get_by_test_id("authored-suite-run-preview")).to_be_enabled()
    await page.get_by_role("link", name="Eval settings", exact=True).click()
    await page.get_by_role("link", name="Judges and calibration", exact=True).click()
    await page.get_by_role("link", name="Create a suite with judge checks", exact=True).click()
    await expect(editor).to_be_visible()
    # Resolve a saved suite by server selector; an authored-only editor never converts a corpus.
    edit_url = f"{base}/cayu/evals/suites/{_suite_identity(saved).id}/edit"

    async def role_transition(state):
        member = False
        held, release = asyncio.Event(), asyncio.Event()
        private_requests = []

        async def identity(route):
            response = await route.fetch()
            body = await response.json()
            if member:
                body.update(role="member", full_access=False)
            await route.fulfill(response=response, json=body)

        async def document(route):
            response = await route.fetch()
            if state == "late":
                held.set()
                await release.wait()
            await route.fulfill(response=response)

        def observe(request):
            if (
                member
                and "/api/evals/" in request.url
                and "/suite-views" not in request.url
                and "/run-views" not in request.url
            ):
                private_requests.append(request.url)

        pattern = "**/api/evals/suites/sha256*"
        await page.route("**/api/app", identity)
        await page.route(pattern, document)
        page.on("request", observe)
        try:
            await page.goto(edit_url, wait_until="domcontentloaded")
            if state == "late":
                await asyncio.wait_for(held.wait(), 10)
            else:
                await expect(editor.get_by_test_id("authored-suite-name")).to_have_value(
                    "Navigation-owned draft"
                )
                await editor.get_by_test_id("authored-suite-name").fill(
                    "Private draft must disappear"
                )
            member = True
            async with page.expect_response(
                lambda response: urlsplit(response.url).path == "/api/app"
            ):
                await page.evaluate("""() => {
                    Object.defineProperty(document, 'visibilityState', {configurable:true, value:'hidden'});
                    window.dispatchEvent(new Event('visibilitychange'));
                    Object.defineProperty(document, 'visibilityState', {configurable:true, value:'visible'});
                    window.dispatchEvent(new Event('visibilitychange'));
                }""")
            await expect(
                page.get_by_text("Suite authoring requires full access", exact=False)
            ).to_be_visible()
            release.set()
            await page.wait_for_timeout(150)
            await expect(editor).to_have_count(0)
            await expect(
                page.get_by_text("Private draft must disappear", exact=True)
            ).to_have_count(0)
            await expect(
                page.get_by_role("button", name="Retry loading suite", exact=True)
            ).to_have_count(0)
            private_requests.clear()
            await page.get_by_role("link", name="Evals", exact=True).first.click()
            await expect(page.get_by_role("button", name="New suite", exact=True)).to_have_count(0)
            assert not private_requests, private_requests
            await page.goto(edit_url, wait_until="networkidle")
            await expect(editor).to_have_count(0)
            assert not private_requests, private_requests
        finally:
            release.set()
            page.remove_listener("request", observe)
            await page.unroute("**/api/app", identity)
            await page.unroute(pattern, document)

    for state in ("cached", "late"):
        await role_transition(state)
    print("AUTHORING ROUTE / HISTORY / ROLE OWNERSHIP PASS", flush=True)


async def _exercise_late_launch(page, base, provider, expected_aborts):
    draft = {
        "schema_version": 3,
        "id": "late-launch",
        "name": "Delayed launch acknowledgement",
        "target_key": "dashboard.regressions",
        "cases": [
            {
                "id": "late",
                "name": "Late case",
                "stimulus": {
                    "kind": "simple_input",
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "text": "Exercise one fresh Control Plane-authored evaluation.",
                            }
                        ]
                    },
                },
                "assertions": [{"id": "ok", "kind": "root_status", "expected": "completed"}],
            }
        ],
    }
    import copy

    second = copy.deepcopy(draft["cases"][0])
    second.update(id="other", name="Other case")
    draft["cases"].append(second)
    preview = await page.request.post(f"{base}/api/evals/suites/preview", data={"draft": draft})
    assert preview.ok, await preview.text()
    suite = (await preview.json())["suite"]
    saved = await page.request.post(
        f"{base}/api/evals/suites",
        data={
            "suite": suite,
            "expected_suite_revision": suite["revision"],
            "parent_suite_revision": None,
        },
    )
    assert saved.ok, await saved.text()
    edit_url = f"{base}/cayu/evals/suites/{_suite_identity(suite).id}/edit?run=true"
    editor = page.get_by_test_id("eval-suite-editor")
    held, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests = []
    committed = {}
    previews = []
    page.on(
        "request",
        lambda request: (
            previews.append(request.url)
            if request.url.endswith("/runs/preview") and request.method == "POST"
            else None
        ),
    )

    async def changed_catalog(route):
        response = await route.fetch()
        payload = await response.json()
        for target in payload["items"]:
            target["cost_budget_available"] = False
            target["cost_budget_currencies"] = []
        await route.fulfill(response=response, json=payload)

    async def hold_ack(route):
        request = route.request
        requests.append((request.headers["idempotency-key"], request.post_data_json))
        response = await route.fetch()
        assert response.ok, await response.text()
        if len(requests) == 1:
            committed.update(await response.json())
            expected_aborts.append(request.url)
            held.set()
            await release.wait()
        await route.fulfill(response=response)
        finished.set()

    pattern = "**/api/evals/suites/*/runs"
    await page.route(pattern, hold_ack)
    try:
        await page.goto(edit_url, wait_until="networkidle")
        await editor.get_by_label("Select Other case for launch", exact=True).uncheck()
        await editor.get_by_test_id("authored-suite-maximum-cost").fill("0.01")
        await editor.get_by_test_id("authored-suite-run-preview").click()
        await expect(editor.get_by_test_id("authored-suite-launch")).to_be_enabled()
        await editor.get_by_test_id("authored-suite-launch").click()
        await asyncio.wait_for(held.wait(), 15)
        run_id = committed["runs"][0]["run"]["spec"]["run_id"]
        await page.get_by_role("link", name="Home", exact=True).first.click()
        await expect(page.get_by_role("heading", name="Home", exact=True)).to_be_visible()
        release.set()
        await asyncio.wait_for(finished.wait(), 10)
        await page.wait_for_timeout(150)
        assert page.url.endswith("/cayu/"), page.url
        async with asyncio.timeout(30):
            while True:
                response = await page.request.get(f"{base}/api/evals/runs/{run_id}")
                assert response.ok, await response.text()
                if (await response.json())["status"] == "completed":
                    break
                await asyncio.sleep(0.1)
        provider_calls = len(provider.requests)
        await page.route("**/api/evals/targets", changed_catalog)
        await page.goto(edit_url, wait_until="networkidle")
        await editor.get_by_role("button", name="Retry exact launch", exact=True).click()
        await expect(page).to_have_url(re.compile(r"/cayu/evals/runs/"), timeout=15000)
        assert len(requests) == 2 and requests[0] == requests[1], requests
        assert len(previews) == 1, "Recovery must not prepare current profiles/exposure again"
        assert len(provider.requests) == provider_calls
    finally:
        release.set()
        await page.unroute(pattern, hold_ack)
        await page.unroute("**/api/evals/targets", changed_catalog)
    print("AUTHORING LATE LAUNCH / EXACT REOPEN REPLAY PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        choices=[
            "all",
            "checks",
            "scenario",
            "parent",
            "calibration",
            "calibration-abandon-404",
            "calibration-abandon-409",
            "calibration-abandon-422",
            "scenario-abandon-404",
            "scenario-abandon-409",
            "scenario-abandon-422",
            "recovery-abandon-sequence",
            "run",
            "navigation",
            "late",
            "keyboard",
            "settings",
            "recovery-corpus",
            "recovery-captured",
            "recovery-captured-nested-status",
            "recovery-captured-direct-status",
            "recovery-captured-nested-error",
            "recovery-captured-direct-error",
            "recovery-captured-direct-authorization",
            "recovery-captured-nested-abandon",
            "recovery-authored",
            "recovery-scenario",
        ],
        default="all",
    )
    asyncio.run(exercise_eval_authoring(parser.parse_args().flow))
