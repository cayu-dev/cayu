"""Real-browser Settings coverage: shared owners, exact recovery and portable tools."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from playwright.async_api import expect


async def exercise_eval_settings(page, base: str, provider, expected_aborts, expected_console):
    before = len(provider.requests)
    await page.goto(f"{base}/cayu/evals/settings", wait_until="networkidle")
    tools = page.get_by_role("navigation", name="Eval settings tools", exact=True)
    await expect(tools).to_be_visible()
    names = (
        "Execution profiles",
        "Judges and calibration",
        "Production scenarios",
        "Memory reports",
        "Import / export",
    )
    for name in names:
        await expect(tools.get_by_role("link", name=name, exact=True)).to_be_visible()
    await tools.get_by_role("link", name="Judges and calibration", exact=True).click()
    await expect(page.get_by_role("heading", name="Judge calibration", exact=True)).to_be_visible()
    assert len(provider.requests) == before, "Opening Settings must not dispatch model work"
    await _calibration(page, base, provider, expected_aborts, expected_console)
    for change in ("missing", "ceilings", "pricing"):
        await _scenarios(page, base, provider, expected_aborts, expected_console, change)
    await _scenario_currency_selection(page, base, expected_aborts, expected_console)
    await _captured_scenario(page, base, provider, expected_aborts, expected_console)
    await page.goto(f"{base}/cayu/evals/settings", wait_until="networkidle")
    await _portability(page, base)
    await _memory(page, base)
    await _access_and_geometry(page, base, names)
    print("SETTINGS TOOLS / OWNERSHIP / DOWNLOADS / ACCESS PASS", flush=True)


async def _calibration(page, base, provider, expected_aborts, expected_console):
    await page.get_by_role("link", name="Create a suite with judge checks", exact=True).click()
    editor = page.get_by_test_id("eval-suite-editor")
    await editor.get_by_test_id("authored-suite-name").fill("Settings calibration")
    await editor.get_by_role("tab", name="Checks (1)", exact=True).click()
    await editor.get_by_label("Check type", exact=True).select_option("structured_model_judge")
    await editor.get_by_role("button", name="Add check", exact=True).click()
    await editor.get_by_test_id("authored-suite-preview").click()
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    async with page.expect_response(
        lambda r: r.url.endswith("/api/evals/suites") and r.request.method == "POST"
    ) as info:
        await editor.get_by_test_id("authored-suite-save").click()
    suite = (await (await info.value).json())["suite"]
    await expect(editor.get_by_test_id("authored-suite-save")).to_be_enabled()
    await page.get_by_role("link", name="Eval settings", exact=True).click()
    await page.get_by_role("link", name="Judges and calibration", exact=True).click()
    await page.get_by_label("Calibration suite", exact=True).select_option(suite["revision"])
    await page.get_by_label("Calibration case", exact=True).select_option("case-1")
    check = next(
        item for item in suite["cases"][0]["assertions"] if item["kind"] == "structured_model_judge"
    )
    await page.get_by_label("Calibration check", exact=True).select_option(check["id"])
    calibration = page.get_by_test_id("judge-calibration")
    await calibration.get_by_label("Known candidate output", exact=True).fill(
        "private-settings-evidence"
    )
    await calibration.get_by_label("Evidence source ID", exact=True).fill(
        "settings-reviewed-source"
    )
    await calibration.get_by_label("Repeated judge calls", exact=True).fill("1")
    await expect(page.get_by_label("Settings specialist", exact=True)).to_be_disabled()
    await expect(page.get_by_label("Calibration suite", exact=True)).to_be_disabled()
    await page.get_by_role("link", name="Memory reports", exact=True).click()
    await page.get_by_role("button", name="Keep working", exact=True).click()
    await expect(calibration.get_by_label("Known candidate output", exact=True)).to_have_value(
        "private-settings-evidence"
    )
    assert "private-settings-evidence" not in page.url
    assert "private-settings-evidence" not in await page.evaluate(
        "JSON.stringify([localStorage, sessionStorage])"
    )
    await calibration.get_by_role("button", name="Check calibration", exact=True).click()
    await expect(calibration.get_by_text("Calibration is ready", exact=True)).to_be_visible()
    before = len(provider.judge_requests)
    url = f"{base}/api/evals/judge-calibrations"
    lost, requests = {}, []
    dispatched, release = asyncio.Event(), asyncio.Event()

    async def lose_ack(route, request):
        requests.append(request.post_data_json)
        if lost:
            await route.continue_()
            return
        response = await route.fetch()
        assert response.status == 201, await response.text()
        lost["response"] = await response.json()
        dispatched.set()
        await release.wait()
        expected_aborts.append(url)
        expected_console.append(url)
        await route.abort("connectionreset")

    async def removed_target(route):
        response = await route.fetch()
        body = await response.json()
        body.update(items=[], default_target_key=None)
        await route.fulfill(response=response, json=body)

    await page.route(url, lose_ack)
    try:
        await calibration.get_by_role("button", name="Run calibration", exact=True).click()
        await asyncio.wait_for(dispatched.wait(), 15)
        await page.route("**/api/evals/targets", removed_target)
        await _refresh_targets(page)
        await expect(
            page.get_by_text("This specialist is not available.", exact=False)
        ).to_be_visible()
        await expect(calibration).to_be_visible()
        await page.get_by_role("link", name="Memory reports", exact=True).click()
        await expect(page.get_by_role("button", name="Leave tool", exact=True)).to_be_disabled()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        release.set()
        await expect(calibration.get_by_role("alert")).to_be_visible()
        await expect(page.get_by_label("Calibration check", exact=True)).to_be_disabled()
        await calibration.get_by_role("button", name="Run calibration", exact=True).click()
        await expect(calibration.get_by_text(re.compile("Calibration complete"))).to_be_visible()
        assert requests[0] == requests[1]
        assert requests[0]["definition"]["assertion"] == check
        assert len(provider.judge_requests) == before + 1
        await expect(page.get_by_label("Calibration suite", exact=True)).to_be_enabled()
    finally:
        release.set()
        await page.unroute(url, lose_ack)
        await page.unroute("**/api/evals/targets", removed_target)
    await _refresh_targets(page)
    await expect(page.get_by_label("Settings specialist", exact=True)).to_be_enabled()
    await calibration.get_by_label("Known candidate output", exact=True).fill("Edited again")
    await expect(page.get_by_label("Calibration suite", exact=True)).to_be_disabled()
    await calibration.get_by_role("button", name="Discard calibration", exact=True).click()
    await expect(page.get_by_label("Calibration suite", exact=True)).to_be_enabled()


async def _wait_for_scenario_runs(page, base, run_ids):
    # A settled launch response only proves admission. Do not let owned provider
    # work leak into the next browser flow's no-dispatch assertions.
    for run_id in run_ids:
        async with asyncio.timeout(30):
            while True:
                response = await page.request.get(f"{base}/api/evals/runs/{run_id}")
                assert response.status == 200, await response.text()
                run = await response.json()
                if run["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.05)
        assert run["status"] == "completed", run


async def _scenarios(
    page,
    base,
    provider,
    expected_aborts,
    expected_console,
    change="missing",
    *,
    expected_http=None,
    abandon_status=None,
):
    await page.goto(f"{base}/cayu/evals/settings?tool=scenarios", wait_until="networkidle")
    await page.get_by_role("button", name="New scenario", exact=True).click()
    editor = page.get_by_test_id("scenario-authoring")
    await editor.get_by_label("Scenario name", exact=True).fill("Settings scenario")
    await editor.get_by_label("Scenario ID", exact=True).fill("settings-scenario")
    text = editor.get_by_label("Input text", exact=True).first
    await text.fill("Exercise one fresh control plane-authored evaluation")
    await page.get_by_role("link", name="Judges and calibration", exact=True).click()
    await page.get_by_role("button", name="Keep working", exact=True).click()
    await expect(text).to_have_value("Exercise one fresh control plane-authored evaluation")
    await editor.get_by_role("spinbutton", name="Trials", exact=True).fill("2")
    await editor.get_by_placeholder("Optional", exact=True).fill("0.01")
    await editor.get_by_role("button", name="Check readiness", exact=True).click()
    await expect(editor.get_by_role("button", name="Save scenario", exact=True)).to_be_enabled()
    async with page.expect_response(
        lambda r: r.url.endswith("/api/evals/scenarios") and r.request.method == "POST"
    ) as info:
        await editor.get_by_role("button", name="Save scenario", exact=True).click()
    saved = (await (await info.value).json())["scenario"]
    await expect(editor.get_by_role("button", name="Run scenario", exact=True)).to_be_enabled()
    await expect(page.get_by_role("button", name="New scenario", exact=True)).to_be_enabled()
    url = f"{base}/api/evals/scenarios/{quote(saved['revision'], safe='')}/runs"
    lost, requests = {}, []
    admitted, release = asyncio.Event(), asyncio.Event()
    before = len(provider.requests)

    async def lose_ack(route, request):
        requests.append((request.post_data_json, request.headers.get("idempotency-key")))
        if expected_http is not None and len(requests) == 1:
            await route.fulfill(status=422, json={"detail": "Fixture rejected before admission."})
            return
        if expected_http is not None and (
            len(requests) == 3
            or (
                abandon_status is not None
                and len(requests) > 3
                and requests[-1][1] == requests[1][1]
            )
        ):
            await route.fulfill(
                status=abandon_status or 404,
                json={"detail": "Scenario target temporarily unavailable."},
            )
            return
        if lost:
            await route.continue_()
            return
        response = await route.fetch()
        assert response.status == 202, await response.text()
        lost["response"] = await response.json()
        admitted.set()
        if abandon_status is not None:
            await release.wait()
        expected_aborts.append(url)
        expected_console.append(url)
        await route.abort("connectionreset")

    async def changed_target(route):
        response = await route.fetch()
        body = await response.json()
        if change == "missing":
            body.update(items=[], default_target_key=None)
        else:
            for item in body["items"]:
                if change == "ceilings":
                    item["max_trials"] = 1
                elif change == "pricing":
                    item.update(cost_budget_available=False, cost_budget_currencies=[])
        await route.fulfill(response=response, json=body)

    await page.route(url, lose_ack)
    if expected_http is not None:
        expected_http.append((url, 422))
    try:
        if expected_http is not None:
            await editor.get_by_role("button", name="Run scenario", exact=True).click()
            await expect(editor.get_by_role("alert")).to_contain_text("before admission")
            await expect(editor.get_by_label("Scenario name", exact=True)).to_be_enabled()
            assert len(provider.requests) == before
        await editor.get_by_role("button", name="Run scenario", exact=True).click()
        if abandon_status is not None:
            await asyncio.wait_for(admitted.wait(), 15)
            abandon = editor.get_by_role("button", name="Abandon scenario recovery", exact=True)
            if await abandon.count():
                await expect(abandon).to_be_disabled()
            release.set()
        await expect(editor.get_by_role("alert")).to_be_visible()
        await page.route("**/api/evals/targets", changed_target)
        await _refresh_targets(page)
        await expect(editor.get_by_label("Scenario name", exact=True)).to_be_disabled()
        currency = editor.get_by_label("Scenario cost budget currency", exact=True)
        await expect(currency).to_be_disabled()
        await expect(currency).to_have_value("USD")
        if change in {"missing", "pricing"}:
            await expect(currency.locator("option:checked")).to_have_text("USD (unavailable)")
        await expect(page.get_by_role("button", name="New scenario", exact=True)).to_be_disabled()
        await page.get_by_role("link", name="Import / export", exact=True).click()
        await expect(page.get_by_role("button", name="Leave tool", exact=True)).to_be_disabled()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        if expected_http is not None:
            expected_http.append((url, abandon_status or 404))
            await editor.get_by_role("button", name="Run scenario", exact=True).click()
            await expect(editor.get_by_role("alert")).to_contain_text("temporarily unavailable")
            await expect(editor.get_by_label("Scenario name", exact=True)).to_be_disabled()
            await expect(
                page.get_by_role("button", name="New scenario", exact=True)
            ).to_be_disabled()
        if abandon_status is not None:
            from dashboard_eval_recovery import confirm_recovery_abandonment

            await editor.get_by_role("button", name="Run scenario", exact=True).click()
            await expect(editor.get_by_role("alert")).to_contain_text("temporarily unavailable")
            assert requests[1] == requests[2] == requests[3]
            await confirm_recovery_abandonment(page, abandon, accept=False)
            await expect(text).to_be_disabled()
            await confirm_recovery_abandonment(page, abandon, accept=True)
            # Fresh editing remains gated by the live catalog after the paid
            # recovery owner is released; restoring it must preserve the draft.
            await page.unroute("**/api/evals/targets", changed_target)
            await _refresh_targets(page)
            await expect(text).to_be_enabled()
            await expect(text).to_have_value("Exercise one fresh control plane-authored evaluation")
            await expect(
                editor.get_by_role("button", name="Run scenario", exact=True)
            ).to_be_disabled()
            await text.fill("A separate unsaved scenario edit")
            await page.get_by_role("link", name="Import / export", exact=True).click()
            await expect(page.get_by_role("button", name="Leave tool", exact=True)).to_be_enabled()
            await page.get_by_role("button", name="Keep working", exact=True).click()
            await text.fill("Exercise one fresh control plane-authored evaluation")
            await editor.get_by_role("button", name="Check readiness", exact=True).click()
            await expect(
                editor.get_by_role("button", name="Run scenario", exact=True)
            ).to_be_enabled()
            async with page.expect_response(
                lambda response: response.url == url and response.request.method == "POST"
            ) as new_info:
                await editor.get_by_role("button", name="Run scenario", exact=True).click()
            fresh = await new_info.value
            assert fresh.status == 202, await fresh.text()
            assert requests[-1][0] == requests[1][0]
            assert requests[-1][1] != requests[1][1]
            fresh_run_id = (await fresh.json())["spec"]["run_id"]
            old_run_id = lost["response"]["spec"]["run_id"]
            assert fresh_run_id != old_run_id
            await _wait_for_scenario_runs(page, base, (old_run_id, fresh_run_id))
            assert len(provider.requests) == before + 8
            await expect(
                editor.get_by_role("button", name="Run scenario", exact=True)
            ).to_be_enabled()
            print(f"SCENARIO PERSISTENT {abandon_status} ABANDON / NEW INTENT PASS", flush=True)
            return
        await expect(editor.get_by_role("button", name="Run scenario", exact=True)).to_be_enabled()
        await editor.get_by_role("button", name="Run scenario", exact=True).click()
        await expect(editor.get_by_role("alert")).not_to_be_visible()
        assert len(requests) == (4 if expected_http is not None else 2)
        recovery_requests = requests[1:] if expected_http is not None else requests
        assert all(request == recovery_requests[0] for request in recovery_requests)
        if expected_http is not None:
            assert requests[0][1] != requests[1][1]
        assert requests[0][1]
        await _wait_for_scenario_runs(page, base, (lost["response"]["spec"]["run_id"],))
        assert len(provider.requests) == before + 4
    finally:
        release.set()
        await page.unroute(url, lose_ack)
        await page.unroute("**/api/evals/targets", changed_target)
    await _refresh_targets(page)
    await expect(page.get_by_role("button", name="New scenario", exact=True)).to_be_enabled()
    async with page.expect_download() as download:
        await page.get_by_role(
            "button", name="Download scenario Settings scenario", exact=True
        ).click()
    exported = json.loads(Path(await (await download.value).path()).read_text())
    assert exported["revision"] == saved["revision"]
    await editor.get_by_label("Scenario name", exact=True).fill("Discard this")
    await page.get_by_role("button", name="Discard scenario edits", exact=True).click()
    await expect(editor).not_to_be_visible()
    await page.get_by_role("button", name=re.compile(r"^Settings scenario")).click()
    await expect(editor.get_by_label("Scenario name", exact=True)).to_have_value(
        "Settings scenario"
    )
    await _scenario_settings_drafts(page)


async def _scenario_currency_selection(page, base, expected_aborts, expected_console):
    """A retained budget cannot masquerade as a newly supported currency."""

    async def currencies(route):
        response = await route.fetch()
        body = await response.json()
        original = body["items"][0]
        body["items"].append(
            {
                **original,
                "target_key": "currency.eur",
                "label": "EUR profile",
                "cost_budget_available": True,
                "cost_budget_currencies": ["EUR"],
            }
        )
        await route.fulfill(response=response, json=body)

    await page.route("**/api/evals/targets", currencies)
    try:
        await page.goto(f"{base}/cayu/evals/settings?tool=scenarios", wait_until="networkidle")
        await page.get_by_role("button", name="New scenario", exact=True).click()
        editor = page.get_by_test_id("scenario-authoring")
        await editor.get_by_label("Input text", exact=True).first.fill("Currency selection")
        budget = editor.get_by_placeholder("Optional", exact=True)
        currency = editor.get_by_label("Scenario cost budget currency", exact=True)
        await budget.fill("0.01")
        await expect(currency).to_have_value("USD")
        await editor.get_by_label("Target profile", exact=False).select_option("currency.eur")
        await expect(currency).to_have_value("USD")
        await expect(currency.locator("option:checked")).to_have_text("USD (unavailable)")
        await expect(budget).to_have_value("0.01")
        await editor.get_by_role("button", name="Check readiness", exact=True).click()
        await expect(editor.get_by_role("alert")).to_contain_text("current target pricing")
        await currency.select_option("EUR")
        await expect(currency).to_have_value("EUR")
        await expect(budget).to_have_value("0.01")
        # This synthetic catalog exercises the public form-to-request boundary;
        # abort before dispatch because the fixture runtime has no EUR target.
        url = f"{base}/api/evals/scenarios/preview"
        dispatched = []

        async def capture(route, request):
            dispatched.append(request.post_data_json)
            expected_aborts.append(url)
            expected_console.append(url)
            await route.abort("connectionreset")

        await page.route(url, capture)
        try:
            await editor.get_by_role("button", name="Check readiness", exact=True).click()
            await expect(editor.get_by_role("alert")).not_to_contain_text("current target pricing")
            assert len(dispatched) == 1
            assert dispatched[0]["draft"]["target_key"] == "currency.eur"
            assert dispatched[0]["settings"]["cost_budget"] == {
                "currency": "EUR",
                "max_estimated_cost": "0.01",
            }
        finally:
            await page.unroute(url, capture)
        await page.get_by_role("button", name="Discard scenario edits", exact=True).click()
    finally:
        await page.unroute("**/api/evals/targets", currencies)
    await _refresh_targets(page)


async def _scenario_settings_drafts(page):
    editor = page.get_by_test_id("scenario-authoring")
    environment = editor.get_by_label("Scenario environment", exact=True)
    await environment.select_option("local")
    await expect(page.get_by_label("Settings specialist", exact=True)).to_be_disabled()
    await page.get_by_role("link", name="Import / export", exact=True).click()
    await page.get_by_role("button", name="Keep working", exact=True).click()
    await expect(environment).to_have_value("local")
    await environment.select_option("")
    fields = (
        ("Trials", "2"),
        ("Concurrency", "2"),
        ("Timeout seconds", "120"),
        ("Max steps", "10"),
        ("Max total tokens per trial", "9000"),
        ("Max tool calls per trial", "4"),
        ("Max run time per trial", "60"),
    )
    for name, value in fields:
        field = editor.get_by_role("spinbutton", name=name, exact=True)
        old = await field.input_value()
        await field.fill(value)
        await expect(page.get_by_label("Settings specialist", exact=True)).to_be_disabled()
        await expect(page.get_by_role("button", name="New scenario", exact=True)).to_be_disabled()
        await page.get_by_role("link", name="Import / export", exact=True).click()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        await expect(field).to_have_value(value)
        await field.fill(old)
        await expect(page.get_by_role("button", name="New scenario", exact=True)).to_be_enabled()
    await editor.get_by_placeholder("Optional", exact=True).fill("0.01")
    await expect(page.get_by_label("Settings specialist", exact=True)).to_be_disabled()
    await page.get_by_role("link", name="Import / export", exact=True).click()
    await page.get_by_role("button", name="Leave tool", exact=True).click()
    await page.get_by_role("link", name="Production scenarios", exact=True).click()
    await page.get_by_role("button", name=re.compile(r"^Settings scenario")).click()
    await expect(editor.get_by_placeholder("Optional", exact=True)).to_have_value("")


async def _captured_scenario(page, base, provider, expected_aborts, expected_console):
    import dashboard_behavior_live as fixture

    await page.goto(
        f"{base}/cayu/sessions/{fixture.PROMOTION_SESSION_ID}", wait_until="networkidle"
    )
    await page.get_by_role("tab", name="Trace", exact=True).click()
    await page.get_by_role("tab", name="Conversation", exact=True).click()
    origin = page.url
    await page.get_by_role("button", name="More actions", exact=True).click()
    await page.get_by_test_id("evaluate-session").click()
    addition = page.get_by_test_id("eval-addition-sheet")
    await addition.get_by_text("Captured evidence, export and fresh execution", exact=True).click()
    await addition.get_by_role(
        "button", name="Open captured evaluation details", exact=True
    ).click()
    sheet = page.get_by_test_id("promotion-sheet")
    editor = sheet.get_by_test_id("scenario-authoring")
    await expect(editor).to_be_visible()
    await expect(sheet.get_by_test_id("promotion-preview")).to_be_enabled()
    await editor.get_by_label("Scenario name", exact=True).fill("Captured recovery scenario")
    await sheet.get_by_test_id("promotion-preview").click()
    await expect(
        page.get_by_role("button", name="Discard scenario edits", exact=True)
    ).to_be_enabled()
    await page.get_by_role("button", name="Keep working", exact=True).click()
    await expect(editor.get_by_label("Scenario name", exact=True)).to_have_value(
        "Captured recovery scenario"
    )
    await editor.get_by_role("button", name="Check readiness", exact=True).click()
    await expect(editor.get_by_role("button", name="Save scenario", exact=True)).to_be_enabled()
    async with page.expect_response(
        lambda r: r.url.endswith("/api/evals/scenarios") and r.request.method == "POST"
    ) as info:
        await editor.get_by_role("button", name="Save scenario", exact=True).click()
    saved = (await (await info.value).json())["scenario"]
    url = f"{base}/api/evals/scenarios/{quote(saved['revision'], safe='')}/runs"
    requests, admitted = [], {}
    before = len(provider.requests)
    dispatched, release, ack_done = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def lose_ack(route, request):
        requests.append((request.post_data_json, request.headers.get("idempotency-key")))
        if admitted:
            await route.continue_()
            return
        response = await route.fetch()
        assert response.status == 202, await response.text()
        admitted.update(await response.json())
        dispatched.set()
        await release.wait()
        expected_aborts.append(url)
        expected_console.append(url)
        await route.abort("connectionreset")
        ack_done.set()

    await page.route(url, lose_ack)
    try:
        await editor.get_by_role("button", name="Run scenario", exact=True).click()
        await asyncio.wait_for(dispatched.wait(), 15)
        await expect(sheet.get_by_test_id("promotion-preview")).to_be_disabled()
        await sheet.get_by_role("button", name="Cancel", exact=True).click()
        await expect(
            page.get_by_role("button", name="Discard scenario edits", exact=True)
        ).to_be_disabled()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        await expect(sheet).to_be_visible()
        await sheet.get_by_role("button", name="Close", exact=True).click()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        await expect(sheet).to_be_visible()
        await sheet.get_by_role("button", name="Cancel", exact=True).press("Escape")
        await page.get_by_role("button", name="Keep working", exact=True).click()
        await expect(sheet).to_be_visible()
        await page.evaluate("history.back()")
        await expect(
            page.get_by_role("button", name="Discard scenario edits", exact=True)
        ).to_be_disabled()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        await expect(page).to_have_url(origin)
        release.set()
        await expect(editor.get_by_role("alert")).to_be_visible()
        await expect(sheet.get_by_test_id("promotion-preview")).to_be_disabled()
        await editor.get_by_role("button", name="Run scenario", exact=True).click()
        await expect(editor.get_by_role("alert")).not_to_be_visible()
        await expect(sheet.get_by_test_id("promotion-preview")).to_be_enabled()
        assert len(requests) == 2 and requests[0] == requests[1]
        for _ in range(100):
            response = await page.request.get(f"{base}/api/evals/runs/{admitted['spec']['run_id']}")
            if (await response.json())["status"] in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)
        assert len(provider.requests) == before + 1
        await editor.get_by_label("Scenario name", exact=True).fill("Explicit discard")
        await sheet.get_by_role("button", name="Cancel", exact=True).click()
        await page.get_by_role("button", name="Discard scenario edits", exact=True).click()
        await expect(sheet).not_to_be_visible()
    finally:
        release.set()
        if dispatched.is_set():
            await asyncio.wait_for(ack_done.wait(), 5)
        await page.unroute(url, lose_ack)
    print("SETTINGS CAPTURED SCENARIO OWNERSHIP PASS", flush=True)


async def _portability(page, base):
    import dashboard_behavior_live as fixture

    await page.get_by_role("link", name="Import / export", exact=True).click()
    preview = await page.request.post(
        f"{base}/api/evals/promotion/sessions/{fixture.PROMOTION_SESSION_ID}/preview", data={}
    )
    assert preview.status == 200, await preview.text()
    candidate = (await preview.json())["candidate"]
    exported = await page.request.post(
        f"{base}/api/evals/promotion/sessions/{fixture.PROMOTION_SESSION_ID}/export",
        data={"expected_candidate_revision": candidate["revision"], "candidate": candidate},
    )
    assert exported.status == 200, await exported.text()
    corpus = await exported.json()
    await page.get_by_label("Import corpus file", exact=True).set_input_files(
        {"name": "corpus.json", "mimeType": "application/json", "buffer": await exported.body()}
    )
    await expect(
        page.get_by_text("Corpus imported for dashboard.regressions.", exact=False)
    ).to_be_visible()
    async with page.expect_download() as download:
        label = "Download corpus " + corpus["revision"].split(":", 1)[1][:12] + "…"
        await page.get_by_role("button", name=label, exact=True).click()
    document = json.loads(Path(await (await download.value).path()).read_text())
    assert document == corpus


async def _memory(page, base):
    await page.get_by_role("link", name="Memory reports", exact=True).click()
    tool = page.get_by_test_id("memory-report-tool")
    await tool.get_by_label("Campaign request JSON", exact=True).set_input_files(
        {"name": "invalid.json", "mimeType": "application/json", "buffer": b"{}"}
    )
    await expect(tool.get_by_role("alert")).to_be_visible()
    await expect(tool.get_by_role("button", name="Build report", exact=True)).to_be_disabled()
    source = Path(__file__).parent / "fixtures/dashboard-memory-report-request.json"
    await tool.get_by_label("Campaign request JSON", exact=True).set_input_files(str(source))
    await expect(tool.get_by_role("button", name="Build report", exact=True)).to_be_enabled()
    async with page.expect_response(
        lambda r: r.url.endswith("/api/evals/memory-reports") and r.request.method == "POST"
    ) as info:
        await tool.get_by_role("button", name="Build report", exact=True).click()
    response = await info.value
    assert response.status == 200, await response.text()
    report = await response.json()
    assert report["selected_variant_id"] == "baseline"
    assert all(row["availability"] != "available" for row in report["rows"])
    await expect(tool.get_by_test_id("memory-report-summary")).to_be_visible()
    async with page.expect_download() as download:
        await tool.get_by_role("button", name="JSON", exact=True).click()
    assert json.loads(Path(await (await download.value).path()).read_text()) == report
    async with page.expect_download() as download:
        await tool.get_by_role("button", name="HTML", exact=True).click()
    assert "Complete trial matrix" in Path(await (await download.value).path()).read_text()
    await tool.get_by_role("button", name="Clear report", exact=True).click()
    await expect(tool.get_by_test_id("memory-report-summary")).not_to_be_visible()


async def _access_and_geometry(page, base, names):
    for name in names:
        await page.get_by_role("link", name=name, exact=True).click()
        await page.wait_for_load_state("networkidle")
        if name == "Production scenarios":
            await page.get_by_role("button", name=re.compile(r"^Settings scenario")).click()
            await expect(page.get_by_test_id("scenario-authoring")).to_be_visible()
        if name == "Judges and calibration":
            choices = page.get_by_label("Calibration suite", exact=True).locator("option")
            revision = await choices.nth(1).get_attribute("value")
            assert revision
            await page.get_by_label("Calibration suite", exact=True).select_option(revision)
            await page.get_by_label("Calibration case", exact=True).select_option("case-1")
            check = (
                await page.get_by_label("Calibration check", exact=True)
                .locator("option")
                .nth(1)
                .get_attribute("value")
            )
            assert check
            await page.get_by_label("Calibration check", exact=True).select_option(check)
            await expect(page.get_by_test_id("judge-calibration")).to_be_visible()
        for width in (320, 390, 900, 1440, 1920):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.wait_for_timeout(150)
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                name,
                width,
            )
            if os.environ.get("CAYU_EVAL_SETTINGS_SCREENSHOTS") and width in (390, 1440):
                output = Path(os.environ["CAYU_EVAL_SETTINGS_SCREENSHOTS"])
                output.mkdir(parents=True, exist_ok=True)
                await page.screenshot(
                    path=str(output / f"{names.index(name)}-{width}.png"), full_page=True
                )
    await page.set_viewport_size({"width": 1440, "height": 900})
    await page.reload(wait_until="networkidle")
    await expect(page.get_by_role("heading", name="Import / export", exact=True)).to_be_visible()
    private_reads = []

    async def member_identity(route):
        response = await route.fetch()
        body = await response.json()
        body.update(role="member", full_access=False)
        await route.fulfill(response=response, json=body)

    def observe(request):
        if "/api/evals/" in request.url:
            private_reads.append(request.url)

    browser = page.context.browser
    assert browser is not None
    member = await browser.new_page()
    await member.route("**/api/app", member_identity)
    member.on("request", observe)
    try:
        for tool in ("profiles", "judges", "scenarios", "memory", "portability"):
            await member.goto(f"{base}/cayu/evals/settings?tool={tool}", wait_until="networkidle")
            await expect(
                member.get_by_text("Execution configuration and advanced", exact=False)
            ).to_be_visible()
            await expect(
                member.get_by_role("navigation", name="Eval settings tools", exact=True)
            ).not_to_be_visible()
        assert not private_reads, private_reads
    finally:
        await member.close()


async def _refresh_targets(page):
    async with page.expect_response(lambda response: response.url.endswith("/api/evals/targets")):
        await page.evaluate("""() => {
            Object.defineProperty(document, 'visibilityState', {configurable:true, value:'hidden'});
            window.dispatchEvent(new Event('visibilitychange'));
            Object.defineProperty(document, 'visibilityState', {configurable:true, value:'visible'});
            window.dispatchEvent(new Event('visibilitychange'));
        }""")
