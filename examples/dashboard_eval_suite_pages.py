"""Large immutable suite reads through both suite and comparison UI entrances."""

import asyncio
import re
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from playwright.async_api import expect

from cayu.evals.corpus import (
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
)


async def exercise_large_suite_pages(page, base, target, selected):
    from dashboard_behavior_live import _eval_view_selector

    for kind in ("authored", "corpus"):
        if selected.startswith("suites-") and selected != f"suites-{kind}":
            continue
        text = "Promote this captured dashboard run. " + "x" * 50_000
        cases: list[dict] = [
            {
                "id": f"{kind}-case-{index:02d}",
                "name": f"Large {kind} case {index:02d}",
                "stimulus": {
                    "kind": "simple_input",
                    "input": {"messages": [{"role": "user", "text": text}]},
                },
                "assertions": [{"id": "completed", "kind": "root_status", "expected": "completed"}],
            }
            for index in range(26)
        ]
        suite_id = f"large-{kind}-inputs"
        name = f"Large {kind} input suite"
        if kind == "authored":
            preview = await page.request.post(
                f"{base}/api/evals/suites/preview",
                data={
                    "draft": {
                        "schema_version": 3,
                        "id": suite_id,
                        "name": name,
                        "target_key": target,
                        "cases": cases,
                    }
                },
            )
            assert preview.ok, await preview.text()
            suite = (await preview.json())["suite"]
            response = await page.request.post(
                f"{base}/api/evals/suites",
                data={"suite": suite, "expected_suite_revision": suite["revision"]},
            )
            assert response.ok, await response.text()
            revision = suite["revision"]
        else:
            # Portable diagnostic provenance is deliberately not an operational session link.
            source = EvaluationSourceIdentityV1(
                application_release_id="large-suite-browser-fixture",
                app_manifest_schema_version="1",
                app_manifest_fingerprint="0" * 64,
                evidence_revision="sha256:" + "0" * 64,
            )
            corpus = EvalCorpusDocument.create(
                target_key=target,
                evidence_policy=EvaluationEvidencePolicySpec.create(),
                suites=(EvalSuiteSpec.create(id=suite_id, name=name),),
                cases=tuple(
                    EvalCaseSpec.create(
                        id=case["id"],
                        name=case["name"],
                        suite_id=suite_id,
                        source=source,
                        input=RunInputSpec.model_validate(case["stimulus"]["input"]),
                        assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                    )
                    for case in cases
                ),
            )
            response = await page.request.post(
                f"{base}/api/evals/corpora", data=corpus.model_dump(mode="json")
            )
            assert response.status == 201, await response.text()
            revision = corpus.revision
        selector = _eval_view_selector(
            "suite",
            {
                "source": kind,
                "target_key": target,
                "revision": revision,
                "suite_id": suite_id if kind == "corpus" else None,
            },
        )
        endpoint = f"{base}/api/evals/suite-views/{selector}"
        oversized = await page.request.get(endpoint + "?cases_limit=25&runs_limit=20")
        assert oversized.status == 413, await oversized.text()
        url = f"{base}/cayu/evals/suites/{selector}"
        async with page.expect_response(
            lambda response, endpoint=endpoint: response.url.startswith(endpoint)
        ) as opened:
            await page.goto(url, wait_until="networkidle")
        opened_response = await opened.value
        assert opened_response.ok, (kind, opened_response.status, opened_response.url)
        await expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()
        for case in cases[:25]:
            await expect(page.get_by_text(case["name"], exact=True)).to_have_count(1)
        await page.get_by_role("button", name="Next cases", exact=True).click()
        await expect(page.get_by_text(cases[-1]["name"], exact=True)).to_be_visible()
        for case in cases[:25]:
            await expect(page.get_by_text(case["name"], exact=True)).to_have_count(0)
        await (
            page.get_by_role("navigation", name="Cases pages")
            .get_by_role("button", name="First page", exact=True)
            .click()
        )
        await expect(page.get_by_text(cases[0]["name"], exact=True)).to_be_visible()
        runs = []
        execution_suites = []
        for _ in range(2):
            await page.goto(url, wait_until="networkidle")
            await page.get_by_role("button", name="Run suite", exact=True).click()
            if kind == "authored":
                sheet = page.get_by_test_id("eval-suite-authoring-sheet")
                await expect(sheet.get_by_test_id("authored-suite-name")).to_have_value(name)
                await sheet.get_by_test_id("authored-suite-preview").click()
                await sheet.get_by_test_id("authored-suite-run-preview").click()
                await expect(sheet.get_by_text("1 durable run ready", exact=True)).to_be_visible()
                await sheet.get_by_test_id("authored-suite-launch").click()
            await expect(page).to_have_url(re.compile(r"/cayu/evals/runs/"), timeout=20000)
            run_selector = page.url.split("/runs/", 1)[1].split("?", 1)[0]
            runs.append(run_selector)
            async with asyncio.timeout(90):
                while True:
                    result = await page.request.get(
                        f"{base}/api/evals/run-views/{run_selector}?cases_limit=1&max_result_bytes=8388608"
                    )
                    assert result.ok, await result.text()
                    run = (await result.json())["run"]
                    status = run["status"]
                    if status in ("completed", "failed", "cancelled"):
                        assert status == "completed", (kind, status)
                        execution_suites.append(_eval_view_selector("suite", run["suite"]))
                        break
                    await asyncio.sleep(0.1)
        queries = []

        async def one_run_per_page(route, _request, *, queries=queries):
            # Exercise real server continuation with a smaller legal page, not synthetic rows.
            parts = urlsplit(route.request.url)
            query = parse_qs(parts.query)
            queries.append(query.copy())
            query["runs_limit"] = ["1"]
            await route.continue_(
                url=urlunsplit(parts._replace(query=urlencode(query, doseq=True)))
            )

        assert len(set(execution_suites)) == 1
        # Authored run comparison uses its actual compiled execution-group corpus.
        patterns = [
            "**/api/evals/suite-views/" + key + "?*"
            for key in dict.fromkeys([selector, *execution_suites])
        ]
        for pattern in patterns:
            await page.route(pattern, one_run_per_page)
        try:
            await page.goto(url, wait_until="networkidle")
            await expect(page.locator(f'table a[href$="/runs/{runs[1]}"]')).to_be_visible()
            await page.get_by_role("button", name="Next runs", exact=True).click()
            await expect(page.locator(f'table a[href$="/runs/{runs[0]}"]')).to_be_visible()
            await page.get_by_role("button", name="Next cases", exact=True).click()
            await expect(page.get_by_text(cases[-1]["name"], exact=True)).to_be_visible()
            await expect(page.locator(f'table a[href$="/runs/{runs[0]}"]')).to_be_visible()
            assert any(q.get("runs_cursor") and q.get("cases_cursor") for q in queries)
            await (
                page.get_by_role("navigation", name="Cases pages")
                .get_by_role("button", name="First page", exact=True)
                .click()
            )
            await expect(page.get_by_text(cases[0]["name"], exact=True)).to_be_visible()
            await expect(page.locator(f'table a[href$="/runs/{runs[0]}"]')).to_be_visible()
            await (
                page.get_by_role("navigation", name="Runs pages")
                .get_by_role("button", name="First page", exact=True)
                .click()
            )
            await expect(page.locator(f'table a[href$="/runs/{runs[1]}"]')).to_be_visible()
            await page.goto(f"{base}/cayu/evals/runs/{runs[1]}", wait_until="networkidle")
            await page.get_by_role("button", name="More runs", exact=True).click()
            await expect(page.locator(f'#compare-run option[value="{runs[0]}"]')).to_have_count(1)
            async with page.expect_response(
                lambda response, prior=runs[0]: (
                    parse_qs(urlsplit(response.url).query).get("compared_with") == [prior]
                )
            ) as compared:
                await page.get_by_label("Compared with", exact=True).select_option(runs[0])
            assert (await compared.value).ok, await (await compared.value).text()
            await expect(page).to_have_url(re.compile(r"[?&]compare="))
            await expect(page.get_by_role("button", name="Retry", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="First runs", exact=True).click()
            await expect(page.get_by_role("button", name="More runs", exact=True)).to_be_visible()
            assert all(q.get("max_result_bytes") == ["8388608"] for q in queries)
        finally:
            # This isolated fixture owns every route; drain handlers before reuse.
            await page.unroute_all(behavior="wait")
        print(f"LARGE {kind.upper()} SUITE PAGES AND COMPARISON PASS", flush=True)
