"""Focused real-browser regressions for exact Evals launch recovery."""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from urllib.parse import quote

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import expect

from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationSourceIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
)
from cayu.evals.read_views import EvalSuiteViewIdentity
from dashboard_eval_settings import _refresh_targets


async def confirm_recovery_abandonment(page, button, *, accept):
    confirmations = []

    async def confirm(dialog):
        confirmations.append(dialog.message)
        assert "does not cancel" in dialog.message
        assert "draft" in dialog.message
        if accept:
            await dialog.accept()
        else:
            await dialog.dismiss()

    await expect(button).to_be_enabled()
    page.once("dialog", confirm)
    await button.click()
    assert len(confirmations) == 1


async def _changed_catalog(route):
    response = await route.fetch()
    body = await response.json()
    for target in body["items"]:
        target["execution_profile"]["revision"] = "sha256:" + "e" * 64
        target["max_concurrency"] = 1
        target["max_timeout_seconds"] = 1
    await route.fulfill(response=response, json=body)


async def exercise_corpus_recovery(page, base, store, app, policy, aborts, console):
    suite = EvalSuiteSpec.create(id="corpus-recovery", name="Corpus recovery")
    case = EvalCaseSpec.create(
        id="recovery-case",
        suite_id=suite.id,
        name="Recovery case",
        source=EvaluationSourceIdentityV1(
            application_release_id="recovery-fixture",
            app_manifest_schema_version="1",
            app_manifest_fingerprint="a" * 64,
            evidence_revision="sha256:" + "b" * 64,
        ),
        input=RunInputSpec(
            messages=(
                CorpusUserMessageSpec(text="Exercise one fresh control plane-authored evaluation"),
            )
        ),
        assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
    )
    corpus = EvalCorpusDocument.create(
        target_key="dashboard.regressions",
        evidence_policy=policy,
        suites=(suite,),
        cases=(case,),
    )
    await store.save_corpus(corpus, redact_json=app.redact_json)
    identity = EvalSuiteViewIdentity(
        source="corpus", target_key=corpus.target_key, revision=corpus.revision, suite_id=suite.id
    )
    import dashboard_behavior_live as fixture

    selector = fixture._eval_view_selector("suite", identity.model_dump(mode="json"))
    await page.goto(f"{base}/cayu/evals/suites/{selector}", wait_until="networkidle")
    url = f"{base}/api/evals/runs"
    requests, accepted = [], []

    async def lose_ack(route, request):
        if request.method != "POST":
            await route.continue_()
            return
        requests.append((request.headers.get("idempotency-key"), request.post_data_json))
        response = await route.fetch()
        assert response.status == 202, await response.text()
        accepted.append((await response.json())["spec"]["run_id"])
        if len(requests) == 1:
            aborts.append(url)
            console.append(url)
            await route.abort("connectionreset")
        else:
            await route.fulfill(response=response)

    await page.route(url, lose_ack)
    try:
        await page.get_by_role("button", name="Run suite", exact=True).click()
        await expect(page.get_by_role("alert")).to_be_visible()
        await page.route("**/api/evals/targets", _changed_catalog)
        await _refresh_targets(page)
        # Public envelope is restored after navigation; fresh metadata must not rekey it.
        await page.reload(wait_until="networkidle")
        await expect(
            page.get_by_role("button", name="Retry exact launch", exact=True)
        ).to_be_enabled()
        await page.get_by_role("button", name="Retry exact launch", exact=True).click()
        await expect(page).to_have_url(re.compile("/cayu/evals/runs/"))
        assert len(requests) == 2 and requests[0] == requests[1]
        assert accepted[0] == accepted[1]
    finally:
        await page.unroute(url, lose_ack)
        await page.unroute("**/api/evals/targets", _changed_catalog)
    print("CORPUS EXACT RECOVERY AFTER METADATA CHANGE PASS", flush=True)


async def exercise_captured_recovery(page, base, aborts, console):
    import dashboard_behavior_live as fixture

    session = fixture.PROMOTION_SESSION_ID
    await page.goto(f"{base}/cayu/sessions/{session}", wait_until="networkidle")
    await page.get_by_role("button", name="More actions", exact=True).click()
    await page.get_by_test_id("evaluate-session").click()
    addition = page.get_by_test_id("eval-addition-sheet")
    await addition.get_by_text("Captured evidence, export and fresh execution", exact=True).click()
    await addition.get_by_role(
        "button", name="Open captured evaluation details", exact=True
    ).click()
    sheet = page.get_by_test_id("promotion-sheet")
    await expect(sheet.get_by_test_id("promotion-launch")).to_be_enabled()
    url = f"{base}/api/evals/sessions/{quote(session, safe='')}/evaluation/launch"
    requests, accepted = [], []

    async def lose_ack(route, request):
        requests.append((request.headers.get("idempotency-key"), request.post_data_json))
        response = await route.fetch()
        assert response.status == 202, await response.text()
        accepted.append((await response.json())["run"]["spec"]["run_id"])
        if len(requests) == 1:
            aborts.append(url)
            console.append(url)
            await route.abort("connectionreset")
        else:
            await route.fulfill(response=response)

    await page.route(url, lose_ack)
    try:
        await sheet.get_by_test_id("promotion-launch").click()
        await expect(sheet.get_by_role("alert")).to_be_visible()
        await page.route("**/api/evals/targets", _changed_catalog)
        # The captured editor deliberately caches targets for 15s. Cross that
        # freshness boundary before delivering the real focus/refetch signal.
        await page.wait_for_timeout(15_010)
        await _refresh_targets(page)
        await expect(
            sheet.get_by_test_id("promotion-fresh-execution").get_by_label("Trials", exact=True)
        ).to_be_disabled()
        await sheet.get_by_role("button", name="Close", exact=True).first.click()
        await expect(
            page.get_by_role("button", name="Abandon launch recovery", exact=True)
        ).to_be_visible()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        await expect(sheet.get_by_test_id("promotion-launch")).to_be_enabled()
        await sheet.get_by_test_id("promotion-launch").click()
        await expect(sheet.get_by_text(re.compile("Started current-app eval run"))).to_be_visible()
        assert len(requests) == 2 and requests[0] == requests[1]
        assert accepted[0] == accepted[1]
        assert "candidate" not in await page.evaluate(
            "JSON.stringify([sessionStorage, localStorage])"
        )
    finally:
        await page.unroute(url, lose_ack)
        await page.unroute("**/api/evals/targets", _changed_catalog)
    print("CAPTURED EXACT PRIVATE RECOVERY AFTER METADATA CHANGE PASS", flush=True)


async def exercise_captured_lifetime(page, base, sessions, aborts, console, expected_http, mode):
    import dashboard_behavior_live as fixture
    from cayu.sessions.records import SessionStatus

    direct = mode.startswith("direct")
    status_refresh = mode.endswith("status")
    session = fixture.PROMOTION_SESSION_ID
    detail_url = f"{base}/api/sessions/{quote(session, safe='')}"
    launch_url = f"{base}/api/evals/sessions/{quote(session, safe='')}/evaluation/launch"

    async def direct_contract(route):
        response = await route.fetch()
        body = await response.json()
        body["capabilities"]["surfaces"]["evals"]["mutate"] = {
            "enabled": False,
            "unavailable_reason": "not_configured",
        }
        await route.fulfill(response=response, json=body)

    async def refresh_detail():
        async with page.expect_response(lambda response: response.url == detail_url):
            await page.evaluate("""() => {
                Object.defineProperty(document, 'visibilityState', {configurable: true, value: 'hidden'});
                window.dispatchEvent(new Event('visibilitychange'));
                Object.defineProperty(document, 'visibilityState', {configurable: true, value: 'visible'});
                window.dispatchEvent(new Event('visibilitychange'));
            }""")

    async def failed_detail(route):
        await route.fulfill(
            status=403 if mode.endswith("authorization") else 503,
            json={"detail": "temporary fixture detail read failure"},
        )

    requests, accepted = [], []
    admitted, release = asyncio.Event(), asyncio.Event()

    async def launch(route, request):
        requests.append((request.headers.get("idempotency-key"), request.post_data_json))
        response = await route.fetch()
        assert response.status == 202, await response.text()
        accepted.append((await response.json())["run"]["spec"]["run_id"])
        if len(requests) == 1:
            aborts.append(launch_url)
            console.append(launch_url)
            admitted.set()
            if status_refresh:
                await release.wait()
            # A failed lifetime assertion may already have removed the owner
            # and aborted this intercepted request. Preserve that assertion.
            with suppress(PlaywrightError):
                await route.abort("connectionreset")
        else:
            await route.fulfill(response=response)

    if direct:
        await page.route("**/api/contract", direct_contract)
    await page.route(launch_url, launch)
    try:
        await page.goto(f"{base}/cayu/sessions/{session}", wait_until="networkidle")
        if direct:
            await page.get_by_test_id("evaluate-session").click()
        else:
            await page.get_by_role("button", name="More actions", exact=True).click()
            await page.get_by_test_id("evaluate-session").click()
            addition = page.get_by_test_id("eval-addition-sheet")
            await addition.get_by_text(
                "Captured evidence, export and fresh execution", exact=True
            ).click()
            await addition.get_by_role(
                "button", name="Open captured evaluation details", exact=True
            ).click()
        sheet = page.get_by_test_id("promotion-sheet")
        await expect(sheet.get_by_test_id("promotion-launch")).to_be_enabled()
        evidence = (
            await sheet.get_by_test_id("promotion-fresh-execution")
            .get_by_label("Trials", exact=True)
            .input_value()
        )
        await sheet.get_by_test_id("promotion-launch").click()
        await asyncio.wait_for(admitted.wait(), timeout=15)
        if mode.endswith("authorization"):
            await expect(sheet.get_by_test_id("promotion-error")).to_be_visible()
            expected_http.append((detail_url, 403))
            await page.route(detail_url, failed_detail)
            await refresh_detail()
            await expect(sheet).not_to_be_attached(timeout=15_000)
            assert len(requests) == len(accepted) == 1
            print("CAPTURED OWNER AUTHORIZATION LOSS REMOVES PRIVATE VIEW PASS", flush=True)
            return
        if mode.endswith("abandon"):
            await expect(sheet.get_by_test_id("promotion-error")).to_be_visible()
            await sheet.get_by_role("button", name="Close", exact=True).first.click()
            await expect(sheet).to_be_visible()
            await page.get_by_role("button", name="Abandon launch recovery", exact=True).click()
            await expect(sheet).not_to_be_attached()
            assert len(requests) == len(accepted) == 1
            print("CAPTURED OWNER EXPLICIT ABANDON DOES NOT CANCEL OR REDISPATCH PASS", flush=True)
            return
        if status_refresh:
            if not direct:
                # Focus also refetches the outer addition's read-only source preview;
                # the server correctly rejects that read while the source is active.
                expected_http.append(
                    (f"{base}/api/evals/sessions/{quote(session, safe='')}/evaluation/preview", 409)
                )
            # Real server admission precedes a real source update while observation is pending.
            await sessions.update_status(session, SessionStatus.RUNNING)
            await refresh_detail()
            await expect(sheet).to_be_visible()
            await expect(sheet.get_by_test_id("promotion-launch")).to_be_disabled()
            await expect(sheet.get_by_test_id("promotion-launch")).to_have_text("Starting...")
            release.set()
        else:
            await expect(sheet.get_by_test_id("promotion-error")).to_be_visible()
            expected_http.append((detail_url, 503))
            await page.route(detail_url, failed_detail)
            await refresh_detail()
            # Query retries must exhaust before its cached-data error path is exercised.
            await expect(
                page.get_by_text("temporary fixture detail read failure", exact=False).first
            ).to_be_attached(timeout=15_000)
        await expect(sheet).to_be_visible()
        await expect(sheet.get_by_test_id("promotion-launch")).to_have_text("Retry exact launch")
        assert (
            await sheet.get_by_test_id("promotion-fresh-execution")
            .get_by_label("Trials", exact=True)
            .input_value()
            == evidence
        )
        await expect(sheet.get_by_test_id("promotion-preview")).to_be_disabled()
        await sheet.get_by_role("button", name="Close", exact=True).first.click()
        await expect(
            page.get_by_role("button", name="Abandon launch recovery", exact=True)
        ).to_be_visible()
        await page.get_by_role("button", name="Keep working", exact=True).click()
        if status_refresh:
            await sessions.update_status(session, SessionStatus.COMPLETED)
        else:
            await page.unroute(detail_url, failed_detail)
        await refresh_detail()
        await sheet.get_by_test_id("promotion-launch").click()
        await expect(sheet.get_by_text(re.compile("Started current-app eval run"))).to_be_visible()
        assert len(requests) == 2 and requests[0] == requests[1]
        assert accepted[0] == accepted[1]
        assert "candidate" not in await page.evaluate(
            "JSON.stringify([sessionStorage, localStorage])"
        )
    finally:
        release.set()
        await sessions.update_status(session, SessionStatus.COMPLETED)
        await page.unroute(launch_url, launch)
        await page.unroute(detail_url, failed_detail)
        if direct:
            await page.unroute("**/api/contract", direct_contract)
    print(f"CAPTURED OWNER {mode.upper()} EXACT RECOVERY PASS", flush=True)


async def exercise_authored_partial_recovery(page, base, store, app, expected_http):
    import dashboard_behavior_live as fixture
    from cayu.evals.scenario import (
        EvalScenarioDocumentV2,
        ScenarioInitialInputEventV2,
        ScenarioInputV2,
        ScenarioTextPartV2,
        ScenarioUserMessageV2,
    )
    from cayu.evals.store import EvalRunQuery, EvalStorePublicationRejected
    from cayu.evals.suite_authoring import EvalSuiteDraftV3, compile_eval_suite_authoring_draft

    scenario = EvalScenarioDocumentV2.create(
        id="partial-scenario",
        target_key="dashboard.regressions",
        name="Partial scenario",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create(
                    (
                        ScenarioUserMessageV2.create(
                            (
                                ScenarioTextPartV2(
                                    text="Exercise one fresh control plane-authored evaluation"
                                ),
                            )
                        ),
                    )
                ),
            ),
        ),
    )
    await store.save_scenario(scenario, redact_json=app.redact_json)
    suite = compile_eval_suite_authoring_draft(
        EvalSuiteDraftV3.model_validate(
            {
                "id": "partial-suite",
                "target_key": scenario.target_key,
                "name": "Partial admission",
                "cases": [
                    {
                        "id": "simple",
                        "name": "Simple",
                        "stimulus": {
                            "kind": "simple_input",
                            "input": {
                                "messages": [
                                    {"text": "Exercise one fresh control plane-authored evaluation"}
                                ]
                            },
                        },
                        "assertions": [
                            {"id": "completed", "kind": "root_status", "expected": "completed"}
                        ],
                    },
                    {
                        "id": "scenario",
                        "name": "Scenario",
                        "stimulus": {
                            "kind": "scenario",
                            "scenario_id": scenario.id,
                            "scenario_revision": scenario.revision,
                        },
                        "assertions": [
                            {"id": "completed", "kind": "root_status", "expected": "completed"}
                        ],
                    },
                ],
            }
        )
    )
    await store.save_authored_suite(suite, redact_json=app.redact_json)
    selector = fixture._eval_view_selector(
        "suite",
        {
            "source": "authored",
            "target_key": suite.target_key,
            "revision": suite.revision,
            "suite_id": None,
        },
    )
    await page.goto(f"{base}/cayu/evals/suites/{selector}/edit?run=true", wait_until="networkidle")
    editor = page.get_by_test_id("eval-suite-editor")
    await expect(editor.get_by_test_id("authored-suite-run-preview")).to_be_enabled()
    await editor.get_by_test_id("authored-suite-run-preview").click()
    await expect(editor.get_by_test_id("authored-suite-launch")).to_be_enabled()
    existing_ids = {
        run.spec.run_id
        for run in (await store.list_runs(EvalRunQuery(target_key=suite.target_key))).items
    }
    original_admit = store.admit_run
    calls = 0
    requests = []
    url = f"{base}/api/evals/suites/{quote(suite.revision, safe='')}/runs"

    async def reject_second(request, *, redact_json):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise EvalStorePublicationRejected("partial admission fixture")
        return await original_admit(request, redact_json=redact_json)

    async def capture(route, request):
        requests.append((request.headers.get("idempotency-key"), request.post_data_json))
        await route.continue_()

    store.admit_run = reject_second
    await page.route(url, capture)
    expected_http.append((url, 422))
    try:
        await editor.get_by_test_id("authored-suite-launch").click()
        await expect(editor.get_by_test_id("authored-suite-error")).to_be_visible()
        runs = [
            run
            for run in (await store.list_runs(EvalRunQuery(target_key=suite.target_key))).items
            if run.spec.run_id not in existing_ids
        ]
        assert len(runs) == 1
        original_id = runs[0].spec.run_id
        await expect(
            editor.get_by_role("button", name="Retry exact launch", exact=True)
        ).to_be_enabled()
        await editor.get_by_role("button", name="Retry exact launch", exact=True).click()
        await expect(page).to_have_url(re.compile("/cayu/evals/runs/"))
        runs = [
            run
            for run in (await store.list_runs(EvalRunQuery(target_key=suite.target_key))).items
            if run.spec.run_id not in existing_ids
        ]
        assert len(runs) == 2 and original_id in {run.spec.run_id for run in runs}
        assert len(requests) == 2 and requests[0] == requests[1]
    finally:
        store.admit_run = original_admit
        await page.unroute(url, capture)
    print("AUTHORED PARTIAL 422 EXACT RECOVERY PASS", flush=True)
