"""Reported evidence traverses HTTP, durable history and the packaged dashboard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn
from playwright.async_api import async_playwright, expect

from cayu.evals import CorpusTarget
from cayu.evals.reported import EvalReport
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.server import DashboardConfig, EvalsConfig, OpenAccess, ServerConfig, create_server
from cayu.server.eval_report_auth import EvalReporterBearerAuth, EvalReportingAccess
from cayu.sessions import RunRequest
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _report():
    now = datetime.now(UTC).isoformat()
    return EvalReport.model_validate(
        {
            "report_id": "browser-report-0",
            "target_key": "reported-browser",
            "program_key": "ci.reviewer:build",
            "origin": {
                "kind": "ci",
                "repository": "example/reviewer",
                "branch": "feature/" + "long-branch-" * 18,
                "commit": "a" * 40,
                "pr_head_commit": "b" * 40,
                "pull_request": 42,
                "run_url": "https://github.com/example/reviewer/actions/runs/123",
            },
            "evidence": {
                "kind": "direct",
                "run": {
                    "suite_id": "CI reviewer checks",
                    "status": "passed",
                    "score": 1.0,
                    "started_at": now,
                    "completed_at": now,
                    "duration_ms": 0,
                    "cases": [
                        {
                            "case_id": f"report-case-{index:02d}",
                            "status": "passed",
                            "score": 1.0,
                            "trial_policy": EvalSuiteTrialPolicyV1.create().model_dump(mode="json"),
                            "trials": [
                                {
                                    "trial_number": 1,
                                    "status": "passed",
                                    "score": 1.0,
                                    "final_output": f"Retained report output {index}",
                                    "assertions": [
                                        {"name": "finished", "outcome": "passed", "score": 1.0}
                                    ],
                                }
                            ],
                        }
                        for index in range(26)
                    ],
                },
            },
        }
    )


async def exercise_reported_evals():
    import dashboard_behavior_live as fixture

    app, _provider, _store, _tasks, directory = await fixture._seed_app()
    store = SQLiteEvalStore(Path(directory.name) / "reports.sqlite")
    credential = "browser-report-private-test-token"
    auth = EvalReporterBearerAuth(
        [
            {
                "publisher_id": "ci-browser",
                "token_sha256": hashlib.sha256(credential.encode()).hexdigest(),
                "target_keys": ["reported-browser"],
            }
        ]
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(
                app,
                config=ServerConfig(
                    access=OpenAccess(trusted_local_development=True),
                    dashboard=DashboardConfig(),
                    evals=EvalsConfig(
                        target=CorpusTarget(
                            key="reported-browser",
                            app=app,
                            request_base=RunRequest(agent_name=fixture.AGENT_NAME, messages=[]),
                            application_release_id="report-browser",
                        ),
                        store=store,
                    ),
                    eval_reporting=EvalReportingAccess(auth),
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
        report = _report()
        async with httpx.AsyncClient(base_url=base) as client:
            receipts = []
            for index in range(21):
                payload = report.model_copy(
                    update={"report_id": f"browser-report-{index}"}
                ).model_dump(mode="json")
                response = await client.post(
                    "/api/evals/reports",
                    headers={"Authorization": f"Bearer {credential}"},
                    json=payload,
                )
                assert response.status_code == 200, response.text
                receipts.append(response.json())
            catalog = (await client.get("/api/evals/suite-views")).json()
            series = next(item for item in catalog["items"] if item["source"] == "reported")
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1440, "height": 900})
            errors = []
            mutations = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda request: (
                    mutations.append(request.url)
                    if request.method not in {"GET", "HEAD"} and "/api/evals/" in request.url
                    else None
                ),
            )
            try:
                await page.goto(f"{base}/cayu/evals", wait_until="networkidle")
                await expect(page.get_by_text("Reported series", exact=False)).to_be_visible()
                # Native authoring keeps its full-page owner beside reported history.
                await page.get_by_role("button", name="New suite", exact=True).click()
                await expect(page).to_have_url(re.compile(r"/cayu/evals/suites/new\?"))
                await expect(page.get_by_test_id("eval-suite-editor")).to_be_visible()
                await page.get_by_role("link", name="Evals /", exact=True).click()
                # A manually supplied reported selector never enters the native editor.
                await page.goto(
                    f"{base}/cayu/evals/suites/{series['id']}/edit", wait_until="networkidle"
                )
                await expect(
                    page.get_by_text("Reported series are read-only.", exact=False)
                ).to_be_visible()
                await expect(page.get_by_test_id("eval-suite-editor")).to_have_count(0)
                await page.get_by_role("link", name="Back to suite", exact=True).click()
                await expect(page.get_by_role("heading", name="Runs", exact=True)).to_be_visible()
                await expect(
                    page.get_by_role("button", name="Run suite", exact=True)
                ).to_have_count(0)
                await page.get_by_role("button", name="Next runs", exact=True).click()
                await expect(page.locator("tbody tr")).to_have_count(1)
                for width in (390, 800, 1440):
                    await page.set_viewport_size({"width": width, "height": 900})
                    await page.goto(
                        f"{base}/cayu/evals/runs/{receipts[-1]['run_key']}",
                        wait_until="networkidle",
                    )
                    await expect(
                        page.get_by_role("heading", name="CI reviewer checks", exact=True)
                    ).to_be_visible()
                    for label in ("Tested commit", "PR head commit", "Declared suite revision"):
                        await expect(page.get_by_text(label, exact=True)).to_be_visible()
                    await expect(page.get_by_text("Unknown", exact=True)).to_be_visible()
                    await expect(
                        page.get_by_role("button", name="Approve as baseline", exact=True)
                    ).to_have_count(0)
                    await expect(page.get_by_label("Compared with", exact=True)).to_have_count(0)
                    assert await page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth + 1"
                    )
                    case = page.get_by_role("button", name="report-case-00", exact=False)
                    await case.focus()
                    await page.keyboard.press("Enter")
                    await expect(
                        page.get_by_text("Retained report output 0", exact=True)
                    ).to_be_visible()
                    await page.keyboard.press("Escape")
                    await expect(case).to_be_focused()
                    await page.get_by_role("button", name="Next cases", exact=True).click()
                    await expect(
                        page.get_by_role("button", name="report-case-25", exact=False)
                    ).to_be_visible()
                    await page.get_by_role("button", name="First page", exact=True).click()
                    await page.get_by_label("Case filter", exact=True).select_option("failing")
                    await expect(
                        page.get_by_text("No retained cases match this filter.", exact=True)
                    ).to_be_visible()
                    async with page.expect_download() as downloaded:
                        await page.get_by_role("button", name="Download report", exact=True).click()
                    artifact = await downloaded.value
                    document = Path(await artifact.path()).read_text()
                    assert json.loads(document)["report_id"] == "browser-report-20"
                    assert credential not in document
                member_page = await browser.new_page(viewport={"width": 390, "height": 900})
                forbidden = []

                async def member_identity(route):
                    response = await route.fetch()
                    body = await response.json()
                    body.update(role="member", full_access=False)
                    await route.fulfill(response=response, json=body)

                member_page.on(
                    "request",
                    lambda request: (
                        forbidden.append(request.url)
                        if "/api/evals/" in request.url
                        and not any(
                            value in request.url
                            for value in ("/evals/suite-views", "/evals/run-views")
                        )
                        else None
                    ),
                )
                await member_page.route("**/api/app", member_identity)
                try:
                    await member_page.goto(
                        f"{base}/cayu/evals/suites/{series['id']}", wait_until="networkidle"
                    )
                    await expect(
                        member_page.get_by_role("heading", name="Runs", exact=True)
                    ).to_be_visible()
                    await member_page.goto(
                        f"{base}/cayu/evals/runs/{receipts[-1]['run_key']}",
                        wait_until="networkidle",
                    )
                    await expect(
                        member_page.get_by_role("heading", name="CI reviewer checks", exact=True)
                    ).to_be_visible()
                    await expect(
                        member_page.get_by_role("button", name="Download report", exact=True)
                    ).to_have_count(0)
                    await member_page.get_by_role(
                        "button", name="report-case-00", exact=False
                    ).click()
                    await expect(member_page.get_by_role("dialog")).to_be_visible()
                    await member_page.keyboard.press("Escape")
                    assert not forbidden, forbidden
                finally:
                    await member_page.close()
                # A legal indivisible case can exceed the entire run-view budget.
                # The ordinary route must still render and expose the exact download.
                oversized = report.model_dump(mode="json")
                oversized["report_id"] = "browser-large-report"
                large_case = oversized["evidence"]["run"]["cases"][0]
                large_case["trials"][0]["assertions"] = [
                    {
                        "name": f"large-{i}",
                        "outcome": "passed",
                        "score": 1.0,
                        "message": "x" * (64 << 10),
                    }
                    for i in range(129)
                ]
                async with httpx.AsyncClient(base_url=base) as client:
                    uploaded = await client.post(
                        "/api/evals/reports",
                        headers={"Authorization": f"Bearer {credential}"},
                        json=oversized,
                    )
                    assert uploaded.status_code == 200, uploaded.text
                await page.goto(
                    f"{base}/cayu/evals/runs/{uploaded.json()['run_key']}", wait_until="networkidle"
                )
                await page.get_by_role("button", name="report-case-00", exact=False).click()
                await expect(
                    page.get_by_text("This case is too large to display here.", exact=False)
                ).to_be_visible()
                await page.keyboard.press("Escape")
                async with page.expect_download() as downloaded:
                    await page.get_by_role("button", name="Download report", exact=True).click()
                assert json.loads(
                    Path(await (await downloaded.value).path()).read_text()
                ) == EvalReport.model_validate(oversized).model_dump(mode="json")
                assert not errors, errors
                assert not mutations, mutations
                print(
                    f"REPORTED EVALS BROWSER PASS: series {series['id']}; history/cases/filter/download; 390/800/1440",
                    flush=True,
                )
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)
        await store.close()
        directory.cleanup()


if __name__ == "__main__":
    asyncio.run(exercise_reported_evals())
