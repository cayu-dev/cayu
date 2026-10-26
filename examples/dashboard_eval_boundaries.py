"""Focused real-browser Evals editor, generated-target and storeless contracts."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import uvicorn
from playwright.async_api import async_playwright, expect

from cayu.evals import CorpusTarget
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
from cayu.server import DashboardConfig, EvalsConfig, OpenAccess, ServerConfig, create_server
from cayu.sessions import RunRequest
from cayu.storage.evals_sqlite import SQLiteEvalStore


async def exercise_eval_boundaries() -> None:
    selected = os.environ.get("CAYU_EVAL_BOUNDARY", "all")
    for mode in ("editor", "large", "suites", "storeless"):
        if (
            selected != "all"
            and ("editor" if selected in ("scroll", "reopen", "layout") else selected.split("-")[0])
            != mode
        ):
            continue
        # Each seeded runtime owns its invocation context as well as its stores.
        await asyncio.create_task(_exercise_eval_boundary(mode, selected))


async def _exercise_eval_boundary(mode: str, selected: str) -> None:
    import dashboard_behavior_live as fixture

    durable = mode != "storeless"
    app, _provider, _store, _tasks, directory = await fixture._seed_app()
    root = Path(directory.name)
    eval_store = SQLiteEvalStore(root / "evals.sqlite") if durable else None
    context = _create_project_control_plane_context(
        project_root=Path.cwd(),
        project_id="dashboard-editor-boundaries",
        eval_store=eval_store,
        store_backend="sqlite" if durable else None,
        store_source="project" if durable else None,
        access=ProjectControlPlaneAccess.TRUSTED_LOCAL_DEVELOPMENT,
        eval_price_book=fixture._dashboard_price_book(),
    )
    assets = os.environ.get("CAYU_EVAL_BOUNDARY_ASSETS")
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(
                app,
                config=ServerConfig(
                    access=OpenAccess(trusted_local_development=True),
                    dashboard=DashboardConfig(directory=Path(assets) if assets else None),
                    evals=(
                        EvalsConfig(
                            target=CorpusTarget(
                                key="large-output",
                                app=app,
                                request_base=RunRequest(agent_name=fixture.AGENT_NAME, messages=[]),
                                application_release_id="browser-boundary",
                            ),
                            store=eval_store,
                            execution_profile_policy=EvalExecutionProfilePolicyV1(
                                reset_strategy="application_managed",
                                isolation_revision="sha256:" + "a" * 64,
                                max_trials=3,
                                max_concurrency=3,
                            ),
                            poll_interval_seconds=0.02,
                        )
                        if mode in ("large", "suites") and eval_store is not None
                        else None
                    ),
                ),
                project_context=None if mode in ("large", "suites") else context,
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
            errors = []
            page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
            try:
                if not durable:
                    await page.goto(
                        f"{base}/cayu/sessions/{fixture.PROMOTION_SESSION_ID}",
                        wait_until="networkidle",
                    )
                    await page.set_viewport_size({"width": 390, "height": 700})
                    await expect(page.get_by_test_id("evaluate-session")).to_be_visible()
                    assert await page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    )
                    await page.get_by_test_id("evaluate-session").click()
                    sheet = page.get_by_test_id("promotion-sheet")
                    await expect(sheet).to_be_visible()
                    await expect(sheet.get_by_test_id("promotion-export")).to_be_enabled(
                        timeout=15000
                    )
                    await expect(sheet.get_by_test_id("promotion-save")).to_be_disabled()
                    await expect(sheet.get_by_test_id("promotion-launch")).to_be_disabled()
                    async with page.expect_download() as download:
                        await sheet.get_by_test_id("promotion-export").click()
                    assert (await download.value).suggested_filename.endswith(".json")
                    print("STORELESS PREVIEW EXPORT PASS", flush=True)
                    return
                if selected in ("all", "scroll"):
                    await page.goto(f"{base}/cayu/evals/settings", wait_until="networkidle")
                    await expect(page.get_by_role("heading", name="Advanced tools")).to_be_visible()
                    await page.locator("main details summary").first.click()
                    await _exercise_scroll_owner(page, require_overflow=True)
                    print("SCROLL OWNERSHIP PASS", flush=True)
                    if selected == "scroll":
                        return
                    await page.set_viewport_size({"width": 1440, "height": 900})
                catalog = await (await page.request.get(f"{base}/api/evals/targets")).json()
                if mode == "suites":
                    from dashboard_eval_suite_pages import exercise_large_suite_pages

                    await exercise_large_suite_pages(
                        page, base, catalog["items"][0]["target_key"], selected
                    )
                    return
                if mode == "large":
                    await _exercise_large_results(page, base, catalog["items"][0]["target_key"])
                    return
                a, b = [entry["target_key"] for entry in catalog["items"]][:2]
                if selected == "reopen":
                    draft: dict = {
                        "schema_version": 3,
                        "id": "reopen",
                        "name": "Specialist B suite",
                        "target_key": b,
                        "cases": [
                            {
                                "id": "case",
                                "name": "Case",
                                "stimulus": {
                                    "kind": "simple_input",
                                    "input": {
                                        "messages": [
                                            {
                                                "role": "user",
                                                "text": "Promote this captured dashboard run.",
                                            }
                                        ]
                                    },
                                },
                                "assertions": [
                                    {"id": "ok", "kind": "root_status", "expected": "completed"}
                                ],
                            }
                        ],
                    }
                    response = await page.request.post(
                        f"{base}/api/evals/suites/preview", data={"draft": draft}
                    )
                    assert response.ok, await response.text()
                    doc = (await response.json())["suite"]
                    response = await page.request.post(
                        f"{base}/api/evals/suites",
                        data={"suite": doc, "expected_suite_revision": doc["revision"]},
                    )
                    assert response.ok, await response.text()
                    await _exercise_reopen(page, base, doc, b)
                    return
                for path in ("/evals", "/evals/settings"):
                    await page.goto(f"{base}/cayu{path}", wait_until="networkidle")
                    await _exercise_scroll_owner(page)
                    picker = (
                        page.locator("main select").first
                        if path == "/evals"
                        else page.get_by_label("Settings specialist")
                    )
                    await picker.select_option(a)
                    await page.get_by_test_id("new-evaluation").click()
                    sheet = page.get_by_test_id("eval-suite-authoring-sheet")
                    if path == "/evals" and selected in ("all", "layout"):
                        from dashboard_eval_editor_layout import exercise_editor_layout

                        await exercise_editor_layout(page, sheet)
                    await sheet.get_by_test_id("authored-suite-name").fill(
                        "Retained specialist A draft"
                    )
                    await expect(sheet.get_by_test_id("authored-suite-preview")).to_be_enabled()
                    await page.keyboard.press("Escape")
                    await expect(sheet).not_to_be_visible()
                    await picker.select_option(b)
                    await page.get_by_test_id("new-evaluation").click()
                    await sheet.get_by_test_id("authored-suite-id").fill(
                        "boundary-" + path.rsplit("/", 1)[-1]
                    )
                    await sheet.get_by_test_id("authored-suite-name").fill("Specialist B suite")
                    await sheet.get_by_role(
                        "textbox", name="Case input message 1", exact=True
                    ).fill("Promote this captured dashboard run.")
                    async with page.expect_response(
                        lambda r: (
                            r.url.endswith("/evals/suites/preview") and r.request.method == "POST"
                        )
                    ) as preview:
                        await sheet.get_by_test_id("authored-suite-preview").click()
                    document = (await (await preview.value).json())["suite"]
                    assert document["target_key"] == b, (a, b, document["target_key"])
                    async with page.expect_response(
                        lambda r: r.url.endswith("/evals/suites") and r.request.method == "POST"
                    ) as saved:
                        await sheet.get_by_test_id("authored-suite-save").click()
                    saved_document = (await (await saved.value).json())["suite"]
                    assert saved_document["target_key"] == b
                    await expect(sheet.get_by_test_id("authored-suite-preview")).to_be_enabled()
                    await page.keyboard.press("Escape")
                    await expect(sheet).not_to_be_visible()
                    await picker.select_option(a)
                    await page.get_by_test_id("new-evaluation").click()
                    await expect(sheet.get_by_test_id("authored-suite-name")).to_have_value(
                        "Retained specialist A draft"
                    )
                    await expect(sheet.get_by_test_id("authored-suite-preview")).to_be_enabled()
                    await page.keyboard.press("Escape")
                    await expect(sheet).not_to_be_visible()
                    await picker.select_option(b)
                    await page.get_by_test_id("new-evaluation").click()
                    await sheet.get_by_test_id("authored-suite-run-preview").click()
                    await expect(
                        sheet.get_by_text("1 durable run ready", exact=True)
                    ).to_be_visible()
                    await sheet.get_by_test_id("authored-suite-launch").click()
                    await expect(page).to_have_url(re.compile(r"/cayu/evals/runs/"), timeout=20000)
                    run = await (
                        await page.request.get(
                            f"{base}/api/evals/runs/{fixture._selected_eval_run(page)}"
                        )
                    ).json()
                    assert run["spec"]["target_key"] == b
                    await fixture._open_eval_details(page)
                    await expect(page.get_by_test_id("eval-run-status-announcement")).to_have_text(
                        "Eval run status: completed.", timeout=20000
                    )
                    await expect(
                        page.locator('[data-slot="card-title"]').filter(has_text="Published result")
                    ).to_be_visible()
                    await _exercise_scroll_owner(page)
                await _exercise_reopen(page, base, saved_document, b)

            finally:
                assert not errors, errors
                await browser.close()
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)
        await context.close()
        directory.cleanup()


async def _exercise_scroll_owner(page, *, require_overflow=False):
    for width in (1440, 900, 390):
        await page.set_viewport_size({"width": width, "height": 700})
        header = page.locator("main header")
        home = page.get_by_role("link", name="Home", exact=True).first
        before = await header.bounding_box()
        sidebar_before = await home.bounding_box() if await home.is_visible() else None
        await page.mouse.move(width - 8, 550)
        await page.mouse.wheel(0, 12000)
        await page.wait_for_timeout(250)
        after = await header.bounding_box()
        assert before and after and abs(after["y"] - before["y"]) < 1, (
            page.url,
            width,
            before,
            after,
        )
        assert await page.evaluate("window.scrollY") == 0
        if sidebar_before:
            sidebar_after = await home.bounding_box()
            assert sidebar_after and abs(sidebar_after["y"] - sidebar_before["y"]) < 1
        scroll = None
        for _ in range(5):
            scroll = await page.locator("main").evaluate("""e => {
                const scroller = [...e.querySelectorAll('*')].find(x => getComputedStyle(x).overflowY === 'auto' && x.scrollHeight > x.clientHeight);
                return scroller ? {top: scroller.scrollTop, height: scroller.clientHeight, end: scroller.scrollHeight} : null;
            }""")
            if scroll is None or scroll["top"] + scroll["height"] >= scroll["end"] - 1:
                break
            await page.mouse.wheel(0, 12000)
            await page.wait_for_timeout(250)
        if require_overflow:
            assert scroll, (page.url, width)
        if scroll:
            assert scroll["top"] > 0 and scroll["top"] + scroll["height"] >= scroll["end"] - 1, (
                page.url,
                width,
                scroll,
            )
        await page.mouse.wheel(0, -12000)
        await page.wait_for_timeout(100)
    await page.set_viewport_size({"width": 1440, "height": 900})


async def _exercise_reopen(page, base, saved_document, b):
    from dashboard_behavior_live import _eval_view_selector

    selector = _eval_view_selector(
        "suite",
        {
            "source": "authored",
            "target_key": b,
            "revision": saved_document["revision"],
            "suite_id": None,
        },
    )
    await page.goto(f"{base}/cayu/evals/suites/{selector}", wait_until="networkidle")
    await _exercise_scroll_owner(page)
    await page.get_by_test_id("new-evaluation").first.click()
    sheet = page.get_by_test_id("eval-suite-authoring-sheet")
    await expect(sheet.get_by_test_id("authored-suite-name")).to_have_value("Specialist B suite")
    await sheet.get_by_test_id("authored-suite-name").fill("Preserved ordinary edit")
    await expect(sheet.get_by_test_id("authored-suite-preview")).to_be_enabled()
    await page.keyboard.press("Escape")
    await expect(sheet).not_to_be_visible()
    await page.get_by_test_id("new-evaluation").first.click()
    await expect(sheet.get_by_test_id("authored-suite-name")).to_have_value(
        "Preserved ordinary edit"
    )
    await sheet.get_by_test_id("authored-suite-preview").click()
    await expect(sheet.get_by_test_id("authored-suite-save")).to_be_enabled()
    await sheet.get_by_test_id("authored-suite-save").click()
    await expect(sheet.get_by_text(re.compile(r"Saved immutable suite .+\."))).to_be_visible()
    await expect(sheet.get_by_test_id("authored-suite-preview")).to_be_enabled()
    await page.keyboard.press("Escape")
    await expect(sheet).not_to_be_visible()
    await page.get_by_test_id("new-evaluation").first.click()
    await expect(sheet.get_by_test_id("authored-suite-name")).to_have_value(
        "Preserved ordinary edit"
    )
    await expect(sheet.get_by_test_id("authored-suite-run-preview")).to_be_enabled()
    # A failed initial fetch is retryable until the user chooses to edit the draft.
    await page.reload(wait_until="networkidle")
    initial_reads = []

    async def fail_initial_load(route):
        initial_reads.append(route.request.url)
        if len(initial_reads) == 1:
            await route.fulfill(
                status=503, json={"detail": "Initial suite temporarily unavailable"}
            )
        else:
            await route.continue_()

    pattern = "**/api/evals/suites/sha256*"
    await page.route(pattern, fail_initial_load)
    try:
        await page.get_by_test_id("new-evaluation").first.click()
        await expect(
            sheet.get_by_text("Initial suite temporarily unavailable", exact=False)
        ).to_be_visible()
        await sheet.get_by_test_id("authored-suite-name").fill("Draft after failed initial load")
        await page.keyboard.press("Escape")
        await expect(sheet).not_to_be_visible()
        await page.get_by_test_id("new-evaluation").first.click()
        await expect(sheet.get_by_test_id("authored-suite-name")).to_have_value(
            "Draft after failed initial load"
        )
        await page.wait_for_timeout(250)
        assert len(initial_reads) == 1
    finally:
        await page.unroute(pattern, fail_initial_load)
    print("EDITOR OWNERSHIP PASS", flush=True)


async def _exercise_large_results(page, base, target):
    from dashboard_behavior_live import _eval_view_selector

    draft = {
        "schema_version": 3,
        "id": "large-outputs",
        "target_key": target,
        "name": "Large retained outputs",
        "trial_request": {
            "trials": 3,
            "timeout_seconds": 30,
            "minimum_passed_trials": 3,
            "max_concurrency": 3,
        },
        "cases": [
            {
                "id": f"case-{index:02d}",
                "name": f"Large case {index:02d}",
                "stimulus": {
                    "kind": "simple_input",
                    "input": {
                        "messages": [{"role": "user", "text": "Dashboard large-output boundary"}]
                    },
                },
                "assertions": [{"id": "completed", "kind": "root_status", "expected": "completed"}],
            }
            for index in range(26)
        ],
    }
    response = await page.request.post(f"{base}/api/evals/suites/preview", data={"draft": draft})
    assert response.ok, await response.text()
    document = (await response.json())["suite"]
    response = await page.request.post(
        f"{base}/api/evals/suites",
        data={"suite": document, "expected_suite_revision": document["revision"]},
    )
    assert response.ok, await response.text()
    path = f"{base}/api/evals/suites/{document['revision']}/runs"
    response = await page.request.post(f"{path}/preview", data={})
    assert response.ok, await response.text()
    preview = await response.json()
    assert preview["ready"], preview
    response = await page.request.post(
        path,
        headers={"Idempotency-Key": "large-output-boundary"},
        data={
            "expected_exposure_revision": preview["exposure"]["revision"],
            "expected_execution_profiles": [
                {
                    "case_ids": item["case_ids"],
                    "execution_profile_revision": item["execution_profile_revision"],
                }
                for item in preview["launches"]
            ],
        },
    )
    assert response.status == 202, await response.text()
    run_id = (await response.json())["runs"][0]["run"]["spec"]["run_id"]
    async with asyncio.timeout(90):
        while True:
            run = await (await page.request.get(f"{base}/api/evals/runs/{run_id}")).json()
            if run["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.1)
    assert run["status"] == "completed", run
    selector = _eval_view_selector("run", run_id)
    response = await page.request.get(f"{base}/api/evals/run-views/{selector}?cases_limit=25")
    assert response.status == 413, await response.text()
    await page.goto(f"{base}/cayu/evals/runs/{selector}", wait_until="networkidle")
    await expect(page.get_by_text("Large case 00", exact=True)).to_be_visible(timeout=20000)
    await page.get_by_text("Large case 00", exact=True).click()
    await expect(page.get_by_role("tab", name="Trial 3 · passed", exact=True)).to_be_visible()
    await page.keyboard.press("Escape")
    await page.get_by_role("button", name="Next cases", exact=True).click()
    await expect(page.get_by_text("Large case 25", exact=True)).to_be_visible()
    await expect(page.get_by_text("Large case 00", exact=True)).not_to_be_visible()
    await page.get_by_role("button", name="First page", exact=True).click()
    await expect(page.get_by_text("Large case 00", exact=True)).to_be_visible(timeout=20000)
    print("LARGE RESULT PAGINATION PASS", flush=True)


if __name__ == "__main__":
    asyncio.run(exercise_eval_boundaries())
